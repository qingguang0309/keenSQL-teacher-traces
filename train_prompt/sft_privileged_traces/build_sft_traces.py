#!/usr/bin/env python3
"""Build EX-verified SFT traces with privileged gold-SQL teacher guidance.

The generation prompt is the original ReToolSQL v4 prompt plus a private
teacher appendix containing the gold SQL. Accepted SFT records deliberately
replace that prompt with the original, non-privileged v4 prompt and retain only
the generated assistant/tool trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PROJECT_ROOT = Path("/data/keenSQL")
TRAIN_PROMPT_DIR = PROJECT_ROOT / "train_prompt"
RETOOLSQL_DIR = TRAIN_PROMPT_DIR / "retoolsql_prompt"
DATABASE_TOOLS_DIR = PROJECT_ROOT / "database_tools"
ORIGINAL_V4_DIR = RETOOLSQL_DIR / "v4"
DATASET_DIR = PROJECT_ROOT / "dataset" / "all_dataset"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "dataset/sft_data/data/all_dataset_qwen35_9b_pass16_full_pcache_bt16384_wrong.json"
)
DEFAULT_DB_ROOT = DATASET_DIR / "databases"
DEFAULT_SCHEMA_CACHE = PROJECT_ROOT / "dataset/m_schema/cache"
HERE = Path(__file__).resolve().parent
# The canonical teacher prompt is versioned with the student prompt. The old
# profile-level addendum is kept only as a deprecation pointer.
DEFAULT_TEACHER_PROMPT = ORIGINAL_V4_DIR / "sft_teacher_prompt.txt"

for import_dir in (TRAIN_PROMPT_DIR, RETOOLSQL_DIR, DATABASE_TOOLS_DIR):
    sys.path.insert(0, str(import_dir))

import prompt_builder as PB  # noqa: E402
import sqlite_tools as st  # noqa: E402
import test_retoolsql_prompt as T  # noqa: E402
from sql_grading import execute_query, grade_results  # noqa: E402

TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.I | re.S)
FINAL_ANSWER_BLOCK = re.compile(r"<final_answer>.*?</final_answer>", re.I | re.S)
REFERENCE_MENTION = re.compile(
    r"\b(?:gold\s+sql|reference\s+sql|ground[- ]?truth\s+sql|"
    r"privileged\s+(?:sql|context|guidance)|provided\s+sql|hidden\s+sql|"
    r"teacher[- ]only|the\s+reference)\b",
    re.I,
)
STRING_LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'")
SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_THREAD_LOCAL = threading.local()

MIN_TEACHER_TOOL_CALLS = 2
PREFERRED_MAX_TEACHER_TOOL_CALLS = 4
MAX_TEACHER_TOOL_CALLS = 6
PREFERRED_MAX_TEACHER_ASSISTANT_TURNS = 5
MAX_TEACHER_ASSISTANT_TURNS = 7


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip malformed {path}:{line_number}: {exc}", flush=True)
                continue
            if isinstance(item, dict):
                yield item


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    errors: list[str] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            errors.append(f"index {index}: not an object")
            continue
        missing = [key for key in ("db_id", "question", "SQL") if not record.get(key)]
        if missing:
            errors.append(f"index {index}: missing/non-empty required fields {missing}")
    if errors:
        raise ValueError(f"invalid input ({len(errors)} records): {'; '.join(errors[:10])}")
    return payload


def select_records(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if n < 0:
        raise ValueError("--n must be zero (all) or positive")
    if n and n < len(indexed):
        indexed = random.Random(seed).sample(indexed, n)
        indexed.sort(key=lambda pair: pair[0])
    return indexed


def db_file_for(db_root: Path, db_id: str) -> Path | None:
    db_dir = db_root / db_id
    expected = db_dir / f"{db_id}.sqlite"
    if expected.is_file():
        return expected
    candidates = sorted(db_dir.glob("*.sqlite")) if db_dir.is_dir() else []
    return candidates[0] if len(candidates) == 1 else None


def sample_id(source_index: int, record: Mapping[str, Any]) -> str:
    identity = json.dumps(
        {
            "source_index": source_index,
            "question_id": record.get("question_id"),
            "db_id": record.get("db_id"),
            "question": record.get("question"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    prefix = str(record.get("question_id", source_index))
    return f"{prefix}-{sha256_text(identity)[:16]}"


def normalize_sql(sql: str) -> str:
    text = SQL_COMMENT.sub(" ", sql or "").lower()
    return re.sub(r"\s+", " ", text).strip().rstrip(";").strip()


def row_digest(rows: Sequence[Sequence[Any]] | None) -> str | None:
    if rows is None:
        return None
    serialized = json.dumps(rows, ensure_ascii=False, default=repr, separators=(",", ":"))
    return sha256_text(serialized)


def grade_final(
    db_file: Path, predicted_sql: str, gold_sql: str, method: str
) -> dict[str, Any]:
    try:
        gold_rows = execute_query(db_file, gold_sql)
    except Exception as exc:
        return {
            "ex": None,
            "reason": f"gold execution failed: {type(exc).__name__}: {exc}",
            "gold_row_count": None,
            "pred_row_count": None,
            "gold_rows_sha256": None,
            "pred_rows_sha256": None,
        }
    try:
        predicted_rows = execute_query(db_file, predicted_sql)
    except Exception as exc:
        return {
            "ex": False,
            "reason": f"prediction execution failed: {type(exc).__name__}: {exc}",
            "gold_row_count": len(gold_rows),
            "pred_row_count": None,
            "gold_rows_sha256": row_digest(gold_rows),
            "pred_rows_sha256": None,
        }
    try:
        matched = bool(grade_results(gold_rows, predicted_rows, method))
    except Exception as exc:
        return {
            "ex": None,
            "reason": f"grading failed: {type(exc).__name__}: {exc}",
            "gold_row_count": len(gold_rows),
            "pred_row_count": len(predicted_rows),
            "gold_rows_sha256": row_digest(gold_rows),
            "pred_rows_sha256": row_digest(predicted_rows),
        }
    return {
        "ex": matched,
        "reason": "match" if matched else "mismatch",
        "gold_row_count": len(gold_rows),
        "pred_row_count": len(predicted_rows),
        "gold_rows_sha256": row_digest(gold_rows),
        "pred_rows_sha256": row_digest(predicted_rows),
    }


def build_prompts(
    record: Mapping[str, Any],
    schema: str,
    teacher_template: str,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    gold_sql = str(record["SQL"]).strip()
    appendix = teacher_template.format(reference_sql=gold_sql)
    appendix_tokens = PB.estimate_tokens(appendix)
    base_budget = max_prompt_tokens - appendix_tokens
    if base_budget <= 0:
        raise PB.SchemaBudgetError(
            f"teacher appendix uses {appendix_tokens} tokens, exceeding budget"
        )
    built = PB.build_messages(
        engine="SQLite",
        db_details=schema,
        external_knowledge=str(record.get("evidence") or ""),
        question=str(record["question"]),
        fewshot=True,
        max_prompt_tokens=base_budget,
    )
    deployment_messages = copy.deepcopy(built["messages"])
    teacher_messages = copy.deepcopy(built["messages"])
    teacher_messages[-1]["content"] += "\n\n" + appendix
    meta = dict(built["meta"])
    meta.update(
        {
            "base_prompt_budget": base_budget,
            "teacher_appendix_tokens": appendix_tokens,
            "teacher_prompt_tokens": meta["prompt_tokens"] + appendix_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "prompt_messages": len(deployment_messages),
            "teacher_prompt_sha256": hashlib.sha256(
                teacher_template.encode("utf-8")
            ).hexdigest(),
        }
    )
    if meta["teacher_prompt_tokens"] > max_prompt_tokens:
        raise PB.SchemaBudgetError(
            f"teacher prompt is {meta['teacher_prompt_tokens']} tokens; budget is "
            f"{max_prompt_tokens}"
        )
    return deployment_messages, teacher_messages, meta


def get_client(base_url: str, api_key: str, timeout: float):
    key = (base_url, api_key, timeout)
    if getattr(_THREAD_LOCAL, "client_key", None) != key:
        from openai import OpenAI

        _THREAD_LOCAL.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=2,
        )
        _THREAD_LOCAL.client_key = key
    return _THREAD_LOCAL.client


def request_completion(
    *,
    messages: list[dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    seed: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    client = get_client(base_url, api_key, timeout)
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if chat_template_kwargs:
        request["extra_body"] = {"chat_template_kwargs": dict(chat_template_kwargs)}
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:  # noqa: BLE001 - hosted APIs may reject extra_body
        if "extra_body" not in request:
            raise
        text = str(exc)
        if "extra_body" not in text and "chat_template_kwargs" not in text:
            raise
        request.pop("extra_body", None)
        response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def run_episode(
    *,
    teacher_messages: list[dict[str, str]],
    db_file: Path,
    db_id: str,
    args: argparse.Namespace,
    request_seed: int,
) -> dict[str, Any]:
    messages = copy.deepcopy(teacher_messages)
    transcript: list[dict[str, str]] = []
    errors: list[str] = []
    warnings: list[str] = []
    executed_calls: list[tuple[str, str]] = []
    final: dict[str, Any] | None = None

    for _turn in range(args.max_turns):
        assistant = request_completion(
            messages=messages,
            base_url=args.model_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=request_seed,
            chat_template_kwargs=getattr(args, "chat_template_kwargs", None),
        )
        transcript.append({"role": "assistant", "content": assistant})
        try:
            parsed = T.parse_assistant_turn(
                assistant,
                require_final_scratch_pad=not args.lenient_scratchpad,
                require_tool_scratch_pad=not args.lenient_scratchpad,
                repair_unclosed_function=True,
            )
        except T.ProtocolError as exc:
            errors.append(f"protocol: {exc}")
            break
        warnings.extend(parsed.get("warnings", []))
        if parsed["action"] == "final_answer":
            final = parsed
            break
        if len(executed_calls) >= MAX_TEACHER_TOOL_CALLS:
            errors.append(
                f"tool-call budget ({MAX_TEACHER_TOOL_CALLS}) exceeded"
            )
            break
        try:
            observation = st.dispatch(
                str(db_file), parsed["name"], parsed["arguments"], db_id=db_id
            )
        except Exception as exc:
            errors.append(f"tool dispatch: {type(exc).__name__}: {exc}")
            break
        sql_text = str(parsed.get("arguments", {}).get("sql") or "").strip()
        executed_calls.append((parsed["name"], sql_text))
        transcript.append({"role": "observation", "content": observation})
        messages.append({"role": "assistant", "content": assistant})
        messages.append(
            {"role": "user", "content": f"<observation>\n{observation}\n</observation>"}
        )

    if final is None and not errors:
        errors.append(f"no final answer after {args.max_turns} turns")

    final_sql = str((final or {}).get("sql") or "").strip()
    executed_sqls = [
        sql for name, sql in executed_calls if name == "sqlite_query" and sql
    ]
    final_was_executed = bool(final_sql) and any(
        normalize_sql(sql) == normalize_sql(final_sql) for sql in executed_sqls
    )
    declared = (final or {}).get("verified_call")
    declared_matches = False
    verification_reason = ""
    if declared is None:
        verification_reason = "missing Verified call: k"
    elif not isinstance(declared, int) or declared < 1 or declared > len(executed_calls):
        verification_reason = f"verified index outside 1..{len(executed_calls)}"
    else:
        declared_tool, declared_sql = executed_calls[declared - 1]
        declared_matches = (
            declared_tool == "sqlite_query"
            and normalize_sql(declared_sql) == normalize_sql(final_sql)
        )
        if not declared_matches:
            verification_reason = "declared call is not the exact final sqlite_query"

    return {
        "transcript": transcript,
        "final": final,
        "final_sql": final_sql or None,
        "errors": errors,
        "warnings": warnings,
        "tool_calls": len(executed_calls),
        "executed_calls": [
            {"index": index, "tool": name, "sql": sql or None}
            for index, (name, sql) in enumerate(executed_calls, 1)
        ],
        "final_was_executed": final_was_executed,
        "verification": {
            "declared": declared,
            "matches": declared_matches,
            "reason": verification_reason,
        },
    }


def distinctive_gold_literals(gold_sql: str) -> list[str]:
    return sorted({value for value in STRING_LITERAL.findall(gold_sql) if len(value) >= 5})


def acceptance_audit(
    *,
    episode: Mapping[str, Any],
    grade: Mapping[str, Any],
    record: Mapping[str, Any],
    schema: str,
) -> dict[str, Any]:
    issues: list[str] = []
    if episode.get("errors"):
        issues.append("protocol_or_runtime_error")
    if not episode.get("final_sql"):
        issues.append("no_final_sql")
    if grade.get("ex") is not True:
        issues.append("ex_not_correct" if grade.get("ex") is False else "grading_infrastructure")
    if not episode.get("final_was_executed"):
        issues.append("final_sql_not_executed_exactly")
    verification = episode.get("verification") or {}
    if verification.get("declared") is None:
        issues.append("missing_verified_call")
    elif not verification.get("matches"):
        issues.append("verified_call_mismatch")
    tool_calls = int(episode.get("tool_calls", 0))
    if tool_calls < 1:
        issues.append("no_tool_call")
    if tool_calls < MIN_TEACHER_TOOL_CALLS:
        issues.append(f"tool_calls_below_required_{MIN_TEACHER_TOOL_CALLS}")
    if tool_calls > MAX_TEACHER_TOOL_CALLS:
        issues.append(f"tool_calls_above_allowed_{MAX_TEACHER_TOOL_CALLS}")

    assistant_text = "\n".join(
        str(message.get("content", ""))
        for message in episode.get("transcript", [])
        if message.get("role") == "assistant"
    )
    if re.search(r"<\s*/?\s*think\s*>", assistant_text, re.I):
        issues.append("contains_think_tags")
    if "PRIVILEGED TEACHER CONTEXT" in assistant_text or "TEACHER-ONLY" in assistant_text:
        issues.append("mentions_teacher_context")
    if "Gold SQL" in assistant_text or "Reference SQL" in assistant_text:
        issues.append("mentions_gold_sql_label")
    prose = TOOL_CALL_BLOCK.sub(" ", assistant_text)
    prose = FINAL_ANSWER_BLOCK.sub(" ", prose)
    if REFERENCE_MENTION.search(prose):
        issues.append("mentions_privileged_reference")

    observations = "\n".join(
        str(message.get("content", ""))
        for message in episode.get("transcript", [])
        if message.get("role") == "observation"
    )
    visible_context = "\n".join(
        [
            str(record.get("question") or ""),
            str(record.get("evidence") or ""),
            schema,
            observations,
        ]
    ).lower()
    leaked_literals = [
        literal
        for literal in distinctive_gold_literals(str(record["SQL"]))
        if literal.lower() in prose.lower() and literal.lower() not in visible_context
    ]
    if leaked_literals:
        issues.append("unobserved_gold_literal_in_reasoning")

    teacher_warnings: list[str] = []
    assistant_turns = sum(
        1 for item in episode.get("transcript", []) if item.get("role") == "assistant"
    )
    if assistant_turns > MAX_TEACHER_ASSISTANT_TURNS:
        issues.append(
            f"assistant_turns_above_allowed_{MAX_TEACHER_ASSISTANT_TURNS}"
        )
    if tool_calls > PREFERRED_MAX_TEACHER_TOOL_CALLS:
        teacher_warnings.append(
            f"tool_calls_above_preferred_{PREFERRED_MAX_TEACHER_TOOL_CALLS}"
        )
    if assistant_turns > PREFERRED_MAX_TEACHER_ASSISTANT_TURNS:
        teacher_warnings.append(
            "assistant_turns_above_preferred_"
            f"{PREFERRED_MAX_TEACHER_ASSISTANT_TURNS}"
        )

    return {
        "accepted": not issues,
        "issues": issues,
        "teacher_warnings": teacher_warnings,
        "leaked_literals": leaked_literals,
        "assistant_turns": assistant_turns,
        "tool_calls": tool_calls,
    }


def make_sft_record(
    *,
    deployment_messages: list[dict[str, str]],
    episode: Mapping[str, Any],
    sample_key: str,
    attempt_id: str,
    source_index: int,
    record: Mapping[str, Any],
    args: argparse.Namespace,
    grade: Mapping[str, Any],
    prompt_meta: Mapping[str, Any],
) -> dict[str, Any]:
    messages = copy.deepcopy(deployment_messages)
    for item in episode["transcript"]:
        if item["role"] == "assistant":
            messages.append({"role": "assistant", "content": item["content"]})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"<observation>\n{item['content']}\n</observation>",
                }
            )
    trajectory_hash = sha256_text(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return {
        "id": f"{attempt_id}-{trajectory_hash[:12]}",
        "messages": messages,
        "prompt_messages": int(prompt_meta["prompt_messages"]),
        "prompt_hash": prompt_meta.get("prompt_sha256"),
        "prompt_version": prompt_meta.get("prompt_version"),
        "teacher_prompt_sha256": prompt_meta.get("teacher_prompt_sha256"),
        "question_id": record.get("question_id"),
        "question": record.get("question"),
        "evidence": record.get("evidence"),
        "db_id": record["db_id"],
        "gold_sql": str(record["SQL"]).strip(),
        "grading_method": (record.get("grading_method") or "multiset").strip(),
        "data_variant": "hard_privileged",
        "metadata": {
            "sample_id": sample_key,
            "source_index": source_index,
            "question_id": record.get("question_id"),
            "db_id": record["db_id"],
            "model": args.model,
            "decoding": args.decoding,
            "attempt": int(attempt_id.rsplit("-a", 1)[1]),
            "grading_method": (record.get("grading_method") or "multiset").strip(),
            "ex": True,
            "ex_reason": grade["reason"],
            "tool_calls": episode["tool_calls"],
            "assistant_turns": sum(
                1 for item in episode["transcript"] if item["role"] == "assistant"
            ),
            "schema_tier": prompt_meta["schema_tier"],
            "teacher_prompt_tokens": prompt_meta["teacher_prompt_tokens"],
            "trajectory_sha256": trajectory_hash,
            "gold_sql_sha256": sha256_text(str(record["SQL"]).strip()),
            "teacher_context_in_messages": False,
            "loss_roles": ["assistant"],
        },
    }


def run_attempt(
    source_index: int,
    record: dict[str, Any],
    attempt: int,
    args: argparse.Namespace,
    teacher_template: str,
) -> dict[str, Any]:
    started = time.time()
    key = sample_id(source_index, record)
    attempt_id = f"{key}-a{attempt}"
    result: dict[str, Any] = {
        "attempt_id": attempt_id,
        "sample_id": key,
        "source_index": source_index,
        "question_id": record.get("question_id"),
        "db_id": record["db_id"],
        "model": args.model,
        "decoding": args.decoding,
        "attempt": attempt,
        "grading_method": (record.get("grading_method") or "multiset").strip(),
        "gold_sql_sha256": sha256_text(str(record["SQL"]).strip()),
    }
    try:
        db_file = db_file_for(args.db_root, str(record["db_id"]))
        if db_file is None:
            raise FileNotFoundError(f"database not found/ambiguous for {record['db_id']}")
        schema_file = args.schema_cache / f"{record['db_id']}.txt"
        if not schema_file.is_file():
            raise FileNotFoundError(f"schema cache missing: {schema_file}")
        schema = schema_file.read_text(encoding="utf-8")
        deployment_messages, teacher_messages, prompt_meta = build_prompts(
            record, schema, teacher_template, args.max_prompt_tokens
        )
        seed_material = f"{args.seed}:{key}:{attempt}"
        request_seed = int(sha256_text(seed_material)[:8], 16) % 2_147_483_647
        episode = run_episode(
            teacher_messages=teacher_messages,
            db_file=db_file,
            db_id=str(record["db_id"]),
            args=args,
            request_seed=request_seed,
        )
        if episode.get("final_sql"):
            grade = grade_final(
                db_file,
                str(episode["final_sql"]),
                str(record["SQL"]),
                result["grading_method"],
            )
        else:
            grade = {
                "ex": False,
                "reason": "no final SQL",
                "gold_row_count": None,
                "pred_row_count": None,
                "gold_rows_sha256": None,
                "pred_rows_sha256": None,
            }
        audit = acceptance_audit(
            episode=episode, grade=grade, record=record, schema=schema
        )
        result.update(
            {
                "request_seed": request_seed,
                "prompt_meta": prompt_meta,
                "final_sql": episode.get("final_sql"),
                "episode": episode,
                "grade": grade,
                "acceptance": audit,
                "accepted": audit["accepted"],
            }
        )
        if audit["accepted"]:
            result["sft_record"] = make_sft_record(
                deployment_messages=deployment_messages,
                episode=episode,
                sample_key=key,
                attempt_id=attempt_id,
                source_index=source_index,
                record=record,
                args=args,
                grade=grade,
                prompt_meta=prompt_meta,
            )
    except Exception as exc:
        result.update(
            {
                "accepted": False,
                "acceptance": {
                    "accepted": False,
                    "issues": ["attempt_exception"],
                    "teacher_warnings": [],
                },
                "exception": f"{type(exc).__name__}: {exc}",
            }
        )
    result["elapsed_s"] = round(time.time() - started, 3)
    return result


def materialize_run(output_dir: Path) -> dict[str, Any]:
    raw_path = output_dir / "raw_attempts.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(raw_path):
        if row.get("attempt_id"):
            latest[str(row["attempt_id"])] = row
    ordered = sorted(
        latest.values(),
        key=lambda row: (int(row.get("source_index", -1)), int(row.get("attempt", 0))),
    )
    accepted_rows = [row["sft_record"] for row in ordered if row.get("accepted")]
    rejected_rows = []
    for row in ordered:
        if row.get("accepted"):
            continue
        rejected_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"sft_record"}
            }
        )
    accepted_count = atomic_write_jsonl(output_dir / "accepted_sft.jsonl", accepted_rows)
    rejected_count = atomic_write_jsonl(output_dir / "rejected_attempts.jsonl", rejected_rows)
    issue_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for row in ordered:
        audit = row.get("acceptance") or {}
        issue_counts.update(audit.get("issues") or [])
        warning_counts.update(audit.get("teacher_warnings") or [])
    summary = {
        "attempts": len(ordered),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "acceptance_rate": round(accepted_count / len(ordered), 6) if ordered else 0.0,
        "ex_correct": sum(1 for row in ordered if (row.get("grade") or {}).get("ex") is True),
        "samples_with_accepted_trace": len(
            {row.get("sample_id") for row in ordered if row.get("accepted")}
        ),
        "issue_counts": dict(issue_counts),
        "teacher_warning_counts": dict(warning_counts),
        "files": {
            "raw_attempts": str(raw_path),
            "accepted_sft": str(output_dir / "accepted_sft.jsonl"),
            "rejected_attempts": str(output_dir / "rejected_attempts.jsonl"),
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def validate_assets(args: argparse.Namespace) -> str:
    required = [
        ORIGINAL_V4_DIR / "system_prompt.txt",
        ORIGINAL_V4_DIR / "user_template.txt",
        ORIGINAL_V4_DIR / "fewshot.json",
        args.teacher_prompt,
        args.input,
        args.db_root,
        args.schema_cache,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required paths: " + ", ".join(missing))
    expected_paths = {
        "system": ORIGINAL_V4_DIR / "system_prompt.txt",
        "fewshot": ORIGINAL_V4_DIR / "fewshot.json",
        "user": ORIGINAL_V4_DIR / "user_template.txt",
    }
    actual_paths = {
        "system": Path(PB.SYSTEM_PROMPT_PATH),
        "fewshot": Path(PB.FEWSHOT_PATH),
        "user": Path(PB.USER_TEMPLATE_PATH),
    }
    mismatched = [
        f"{name}: {actual_paths[name]} != {expected_paths[name]}"
        for name in expected_paths
        if actual_paths[name].resolve() != expected_paths[name].resolve()
    ]
    if mismatched:
        raise RuntimeError("prompt_builder is not pinned to v4: " + "; ".join(mismatched))
    teacher_template = args.teacher_prompt.read_text(encoding="utf-8").strip()
    try:
        teacher_template.format(reference_sql="SELECT 1")
    except Exception as exc:
        raise ValueError(f"invalid teacher prompt template: {exc}") from exc
    return teacher_template


def generate(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    attempts_per_sample = 1 if args.decoding == "greedy" else 8
    if args.temperature is None:
        args.temperature = 0.0 if args.decoding == "greedy" else 1.0
    if args.decoding == "greedy" and args.temperature != 0:
        raise ValueError("greedy decoding requires --temperature 0")
    if args.decoding == "pass8" and args.temperature <= 0:
        raise ValueError("pass8 requires --temperature > 0")
    args.api_key = args.api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
    teacher_template = validate_assets(args)
    records = load_records(args.input)
    selected = select_records(records, args.n, args.seed)
    if not selected:
        raise ValueError("no records selected")
    missing_db: list[str] = []
    missing_schema: list[str] = []
    for _, record in selected:
        db_id = str(record["db_id"])
        if db_file_for(args.db_root, db_id) is None:
            missing_db.append(db_id)
        if not (args.schema_cache / f"{db_id}.txt").is_file():
            missing_schema.append(db_id)
    if missing_db or missing_schema:
        raise FileNotFoundError(
            f"missing/ambiguous databases={sorted(set(missing_db))[:20]}; "
            f"missing schemas={sorted(set(missing_schema))[:20]}"
        )

    first_index, first_record = selected[0]
    first_schema = (args.schema_cache / f"{first_record['db_id']}.txt").read_text(
        encoding="utf-8"
    )
    if args.chat_template_kwargs is None:
        args.chat_template_kwargs = {"enable_thinking": bool(args.enable_thinking)}
    _, teacher_messages, first_meta = build_prompts(
        first_record, first_schema, teacher_template, args.max_prompt_tokens
    )
    print(
        f"input={args.input} selected={len(selected)} decoding={args.decoding} "
        f"attempts_per_sample={attempts_per_sample} total_attempts="
        f"{len(selected) * attempts_per_sample}",
        flush=True,
    )
    print(
        f"prompt_check sample={sample_id(first_index, first_record)} "
        f"messages={len(teacher_messages)} meta={json.dumps(first_meta, ensure_ascii=False)}",
        flush=True,
    )
    if args.dry_run:
        print("dry-run complete; no model request and no output files written", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "model_url": args.model_url,
        "model": args.model,
        "decoding": args.decoding,
        "temperature": args.temperature,
        "attempts_per_sample": attempts_per_sample,
        "seed": args.seed,
        "selected_records": len(selected),
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_turns": args.max_turns,
        "max_prompt_tokens": args.max_prompt_tokens,
        "lenient_scratchpad": args.lenient_scratchpad,
        "chat_template_kwargs": args.chat_template_kwargs,
        "teacher_prompt": str(args.teacher_prompt.resolve()),
        "teacher_prompt_sha256": sha256_file(args.teacher_prompt),
        "base_prompt_assets": {
            name: {
                "path": str((ORIGINAL_V4_DIR / filename).resolve()),
                "sha256": sha256_file(ORIGINAL_V4_DIR / filename),
            }
            for name, filename in {
                "system": "system_prompt.txt",
                "user_template": "user_template.txt",
                "fewshot": "fewshot.json",
            }.items()
        },
        "sft_prompt_contains_teacher_context": False,
    }
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise RuntimeError(
                f"resume config mismatch in {config_path}; choose a new --output-dir"
            )
    else:
        atomic_write_json(config_path, config)

    raw_path = args.output_dir / "raw_attempts.jsonl"
    completed_ids = {
        str(row["attempt_id"])
        for row in iter_jsonl(raw_path)
        if row.get("attempt_id")
    }
    jobs = [
        (source_index, record, attempt)
        for source_index, record in selected
        for attempt in range(1, attempts_per_sample + 1)
        if f"{sample_id(source_index, record)}-a{attempt}" not in completed_ids
    ]
    print(
        f"resume_completed={len(completed_ids)} pending={len(jobs)} "
        f"workers={args.workers} output={args.output_dir}",
        flush=True,
    )

    accepted_now = 0
    done_now = 0
    with raw_path.open("a", encoding="utf-8", buffering=1) as raw_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            job_iterator = iter(jobs)
            in_flight: dict[Any, tuple[int, dict[str, Any], int]] = {}

            def submit_one() -> bool:
                try:
                    source_index, record, attempt = next(job_iterator)
                except StopIteration:
                    return False
                future = pool.submit(
                    run_attempt,
                    source_index,
                    record,
                    attempt,
                    args,
                    teacher_template,
                )
                in_flight[future] = (source_index, record, attempt)
                return True

            for _ in range(min(len(jobs), args.workers * 2)):
                submit_one()
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    in_flight.pop(future)
                    row = future.result()
                    raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_handle.flush()
                    done_now += 1
                    accepted_now += int(bool(row.get("accepted")))
                    issues = (row.get("acceptance") or {}).get("issues") or []
                    print(
                        f"[{done_now}/{len(jobs)}] attempt={row['attempt_id']} "
                        f"accepted={row.get('accepted')} ex={(row.get('grade') or {}).get('ex')} "
                        f"issues={issues} elapsed={row.get('elapsed_s')}s",
                        flush=True,
                    )
                    submit_one()

    summary = materialize_run(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _merge_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def merge_quality_key(
    row: Mapping[str, Any], input_index: int, row_index: int
) -> tuple[int, ...]:
    """Return a deterministic key where lower means a better SFT trajectory.

    All inputs should already come from accepted_sft.jsonl, so EX correctness and
    the strict generation-time audit are prerequisites.  The merge nevertheless
    rechecks the invariants that are available in the exported record, then
    prefers concise, non-redundant trajectories in the teacher prompt's normal
    two-to-four-tool-call range.  Greedy is only a tie-breaker: a cleaner pass@8
    trajectory can still win.
    """

    messages = row.get("messages")
    if not isinstance(messages, list):
        messages = []
    prompt_messages = _merge_int(row.get("prompt_messages"), 0)
    if not 0 <= prompt_messages <= len(messages):
        prompt_messages = 0
    trajectory = messages[prompt_messages:]
    assistant_messages = [
        item
        for item in trajectory
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    observation_messages = [
        item
        for item in trajectory
        if isinstance(item, Mapping)
        and item.get("role") == "user"
        and "<observation>" in str(item.get("content", "")).lower()
    ]
    assistant_text = "\n".join(
        str(item.get("content", "")) for item in assistant_messages
    )
    tool_blocks = TOOL_CALL_BLOCK.findall(assistant_text)
    normalized_tool_blocks = [re.sub(r"\s+", " ", block).strip() for block in tool_blocks]
    repeated_tool_calls = len(normalized_tool_blocks) - len(set(normalized_tool_blocks))

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    tool_calls = _merge_int(metadata.get("tool_calls"), len(tool_blocks))
    assistant_turns = _merge_int(
        metadata.get("assistant_turns"), len(assistant_messages)
    )
    final_blocks = FINAL_ANSWER_BLOCK.findall(assistant_text)
    last_assistant = (
        str(assistant_messages[-1].get("content", "")) if assistant_messages else ""
    )
    prose = TOOL_CALL_BLOCK.sub(" ", assistant_text)
    prose = FINAL_ANSWER_BLOCK.sub(" ", prose)

    protocol_penalties = 0
    protocol_penalties += int(metadata.get("ex") is not True)
    protocol_penalties += int(metadata.get("teacher_context_in_messages") is not False)
    protocol_penalties += int(not MIN_TEACHER_TOOL_CALLS <= tool_calls <= MAX_TEACHER_TOOL_CALLS)
    protocol_penalties += int(not 1 <= assistant_turns <= MAX_TEACHER_ASSISTANT_TURNS)
    protocol_penalties += int(len(observation_messages) != tool_calls)
    protocol_penalties += int(len(final_blocks) != 1)
    protocol_penalties += int(not FINAL_ANSWER_BLOCK.search(last_assistant))
    protocol_penalties += int(bool(re.search(r"<\s*/?\s*think\s*>", assistant_text, re.I)))
    protocol_penalties += int(
        "TEACHER-ONLY" in assistant_text
        or "PRIVILEGED TEACHER CONTEXT" in assistant_text
        or bool(REFERENCE_MENTION.search(prose))
    )
    for tag in ("scratch_pad", "tool_call", "final_answer"):
        openings = len(re.findall(rf"<\s*{tag}\s*>", assistant_text, re.I))
        closings = len(re.findall(rf"<\s*/\s*{tag}\s*>", assistant_text, re.I))
        protocol_penalties += int(openings != closings)

    preferred_tool_penalty = max(0, tool_calls - PREFERRED_MAX_TEACHER_TOOL_CALLS)
    decoding_penalty = 0 if str(metadata.get("decoding", "")) == "greedy" else 1
    model_name = str(metadata.get("model", "")).lower()
    model_penalty = 0 if "3.6" in model_name else 1 if "3.5" in model_name else 2
    attempt = _merge_int(metadata.get("attempt"), 10**9)

    return (
        protocol_penalties,
        repeated_tool_calls,
        preferred_tool_penalty,
        tool_calls,
        assistant_turns,
        len(assistant_text),
        decoding_penalty,
        model_penalty,
        attempt,
        input_index,
        row_index,
    )


def merge_outputs(args: argparse.Namespace) -> int:
    missing_inputs = [str(path) for path in args.inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "cannot merge before every accepted SFT file exists: "
            + ", ".join(missing_inputs)
        )
    seen: set[str] = set()
    candidates_by_sample: dict[
        str, list[tuple[tuple[int, ...], dict[str, Any], str]]
    ] = {}
    selected_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    alternative_counts: dict[str, int] = {}
    total_input_records = 0
    for input_index, input_path in enumerate(args.inputs):
        source = str(input_path)
        candidates = duplicates = invalid = 0
        for row_index, row in enumerate(iter_jsonl(input_path), 1):
            total_input_records += 1
            messages = row.get("messages")
            metadata = row.get("metadata")
            sample_key = (
                str(metadata.get("sample_id", "")).strip()
                if isinstance(metadata, Mapping)
                else ""
            )
            if (
                not isinstance(messages, list)
                or not sample_key
                or not isinstance(metadata, Mapping)
                or metadata.get("ex") is not True
            ):
                invalid += 1
                continue
            trajectory_key = sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            if trajectory_key in seen:
                duplicates += 1
                continue
            seen.add(trajectory_key)
            quality = merge_quality_key(row, input_index, row_index)
            candidates_by_sample.setdefault(sample_key, []).append((quality, row, source))
            candidates += 1
        candidate_counts[source] = candidates
        duplicate_counts[source] = duplicates
        invalid_counts[source] = invalid

    merged: list[dict[str, Any]] = []
    for sample_key in sorted(candidates_by_sample):
        candidates = candidates_by_sample[sample_key]
        _quality, selected, selected_source = min(candidates, key=lambda item: item[0])
        merged.append(selected)
        selected_counts[selected_source] = selected_counts.get(selected_source, 0) + 1
        for _candidate_quality, _candidate, candidate_source in candidates:
            if _candidate is selected:
                continue
            alternative_counts[candidate_source] = (
                alternative_counts.get(candidate_source, 0) + 1
            )

    for input_path in args.inputs:
        source = str(input_path)
        selected_counts.setdefault(source, 0)
        alternative_counts.setdefault(source, 0)
    merged.sort(
        key=lambda row: (
            str((row.get("metadata") or {}).get("sample_id", "")),
            str((row.get("metadata") or {}).get("model", "")),
            str((row.get("metadata") or {}).get("decoding", "")),
            int((row.get("metadata") or {}).get("attempt", 0)),
        )
    )
    count = atomic_write_jsonl(args.out, merged)
    summary = {
        "selection_policy": "one_best_trajectory_per_sample_id",
        "quality_order": [
            "exported protocol/integrity checks",
            "no repeated identical tool call",
            "prefer 2-4 tool calls",
            "fewer tool calls and assistant turns",
            "shorter assistant trajectory",
            "greedy tie-breaker",
            "model/attempt/source deterministic tie-breakers",
        ],
        "total_input_records": total_input_records,
        "eligible_candidates": sum(candidate_counts.values()),
        "merged_records": count,
        "unique_samples": len(candidates_by_sample),
        "unique_trajectory_hashes": len(seen),
        "candidates_by_input": candidate_counts,
        "kept_by_input": selected_counts,
        "alternatives_dropped_by_input": alternative_counts,
        "duplicates_by_input": duplicate_counts,
        "invalid_by_input": invalid_counts,
        "output": str(args.out),
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate and EX-filter traces")
    generate_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    generate_parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    generate_parser.add_argument(
        "--schema-cache", type=Path, default=DEFAULT_SCHEMA_CACHE
    )
    generate_parser.add_argument(
        "--teacher-prompt", type=Path, default=DEFAULT_TEACHER_PROMPT
    )
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    generate_parser.add_argument("--model", default="Qwen3.5-9B")
    generate_parser.add_argument("--api-key", default=None)
    generate_parser.add_argument("--decoding", choices=("greedy", "pass8"), required=True)
    generate_parser.add_argument("--temperature", type=float, default=None)
    generate_parser.add_argument("--workers", type=int, default=16)
    generate_parser.add_argument("--max-tokens", type=int, default=8192)
    generate_parser.add_argument("--timeout", type=float, default=600.0)
    generate_parser.add_argument("--max-turns", type=int, default=7)
    generate_parser.add_argument("--max-prompt-tokens", type=int, default=16000)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--n", type=int, default=0, help="0 means all records")
    generate_parser.add_argument("--lenient-scratchpad", action="store_true")
    generate_parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="pass enable_thinking=True to the local vLLM chat template; default is false",
    )
    generate_parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help='extra chat_template_kwargs as JSON, e.g. {"enable_thinking": false}',
    )
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.set_defaults(func=generate)

    merge_parser = subparsers.add_parser(
        "merge", help="select one best accepted SFT trajectory per sample"
    )
    merge_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    merge_parser.add_argument("--out", type=Path, required=True)
    merge_parser.add_argument("--summary", type=Path, default=None)
    merge_parser.set_defaults(func=merge_outputs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
