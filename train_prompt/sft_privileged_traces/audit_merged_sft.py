#!/usr/bin/env python3
"""Re-audit merged privileged SFT trajectories without trusting export flags."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path("/data/keenSQL")
TRACE_DIR = PROJECT_ROOT / "train_prompt/sft_privileged_traces"
for path in (
    TRACE_DIR,
    PROJECT_ROOT / "train_prompt",
    PROJECT_ROOT / "train_prompt/retoolsql_prompt",
    PROJECT_ROOT / "database_tools",
):
    sys.path.insert(0, str(path))

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

import build_sft_traces as B  # noqa: E402


TAG_NAMES = (
    "scratch_pad",
    "tool_call",
    "function",
    "parameter",
    "relevant_tables",
    "relevant_columns",
    "final_answer",
)
QUOTED_FALLBACK = re.compile(r"'((?:''|[^'])*)'")
NUMERIC_FALLBACK = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def strip_one_semicolon(sql: str) -> str:
    text = (sql or "").strip()
    return text[:-1].rstrip() if text.endswith(";") else text


def canonical_payload(text: str) -> Any:
    stripped = (text or "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", stripped)


def unwrap_observation(text: str) -> str:
    match = re.fullmatch(r"\s*<observation>\s*(.*?)\s*</observation>\s*", text, re.S | re.I)
    return match.group(1) if match else text.strip()


def sql_literals(sql: str) -> tuple[list[str], list[str]]:
    strings: list[str] = []
    numbers: list[str] = []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        for literal in tree.find_all(exp.Literal):
            value = str(literal.this)
            if literal.is_string:
                strings.append(value.replace("''", "'"))
            elif literal.is_number:
                numbers.append(value)
    except Exception:
        strings.extend(value.replace("''", "'") for value in QUOTED_FALLBACK.findall(sql or ""))
        numbers.extend(NUMERIC_FALLBACK.findall(sql or ""))
    return sorted(set(strings)), sorted(set(numbers))


def text_has_literal(text: str, value: str, *, numeric: bool = False) -> bool:
    if numeric:
        return bool(re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", text))
    folded = text.casefold()
    candidates = {value.casefold(), value.strip("%_").casefold()}
    return any(candidate and candidate in folded for candidate in candidates)


def identifiers(sql: str) -> set[str]:
    found: set[str] = set()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        found.update(str(node.name).casefold() for node in tree.find_all(exp.Table) if node.name)
        found.update(str(node.name).casefold() for node in tree.find_all(exp.Column) if node.name)
    except Exception:
        found.update(token.casefold() for token in re.findall(r"\b[A-Za-z_]\w*\b", sql or ""))
    return found


def tool_footprint(name: str, arguments: Mapping[str, Any]) -> set[str]:
    values: list[str] = [name]
    for key, value in arguments.items():
        values.append(str(key))
        if isinstance(value, list):
            values.extend(map(str, value))
        else:
            values.append(str(value))
    text = " ".join(values)
    footprint = {token.casefold() for token in re.findall(r"\b[A-Za-z_]\w*\b", text)}
    sql = str(arguments.get("sql") or "")
    if sql:
        footprint.update(identifiers(sql))
    return footprint


def tag_balance_issues(text: str, turn: int) -> list[str]:
    issues: list[str] = []
    for tag in TAG_NAMES:
        openings = len(re.findall(rf"<\s*{tag}(?:\s|=|>)", text, re.I))
        closings = len(re.findall(rf"<\s*/\s*{tag}\s*>", text, re.I))
        if openings != closings:
            issues.append(f"unbalanced_{tag}_turn_{turn}")
    return issues


def audit_row(index: int, row: dict[str, Any], db_root: Path) -> tuple[int, dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    sample_id = str(metadata.get("sample_id") or "")
    db_id = str(row.get("db_id") or metadata.get("db_id") or "")
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    prompt_messages = row.get("prompt_messages")
    try:
        prompt_messages = int(prompt_messages)
    except (TypeError, ValueError):
        prompt_messages = -1

    if not sample_id:
        issues.append("missing_sample_id")
    if metadata.get("ex") is not True:
        issues.append("metadata_ex_not_true")
    if not messages or not 0 < prompt_messages <= len(messages):
        issues.append("invalid_prompt_message_boundary")
        return index, {
            "id": row.get("id"),
            "sample_id": sample_id,
            "hard_pass": False,
            "issues": issues,
            "warnings": warnings,
        }

    prompt = messages[:prompt_messages]
    trajectory = messages[prompt_messages:]
    initial_visible = "\n".join(
        [
            str(row.get("question") or ""),
            str(row.get("evidence") or ""),
            str(prompt[-1].get("content", "")) if isinstance(prompt[-1], dict) else "",
        ]
    )
    visible = initial_visible
    gold_sql = str(row.get("gold_sql") or "")
    gold_strings, gold_numbers = sql_literals(gold_sql)
    db_file = B.db_file_for(db_root, db_id)
    if db_file is None:
        issues.append("database_missing_or_ambiguous")

    calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    final_turn: int | None = None
    temporal_literal_hits: list[dict[str, Any]] = []
    numeric_literal_warnings: list[dict[str, Any]] = []
    parser_warnings: list[dict[str, Any]] = []
    replay_mismatches: list[int] = []
    i = 0
    while i < len(trajectory):
        item = trajectory[i]
        turn = i + 1
        if not isinstance(item, dict) or item.get("role") != "assistant":
            issues.append(f"expected_assistant_at_trajectory_message_{turn}")
            i += 1
            continue
        assistant = str(item.get("content", ""))
        issues.extend(tag_balance_issues(assistant, turn))
        if re.search(r"<\s*/?\s*think\s*>", assistant, re.I):
            issues.append(f"contains_think_tag_turn_{turn}")
        try:
            parsed = B.T.parse_assistant_turn(
                assistant,
                require_final_scratch_pad=True,
                require_tool_scratch_pad=True,
                repair_unclosed_function=False,
            )
        except Exception as exc:
            issues.append(f"strict_protocol_error_turn_{turn}")
            evidence.setdefault("protocol_errors", []).append(
                {"turn": turn, "error": f"{type(exc).__name__}: {exc}"}
            )
            i += 1
            continue

        if parsed.get("warnings"):
            parser_warnings.append({"turn": turn, "warnings": parsed["warnings"]})
            issues.append(f"parser_repair_or_warning_turn_{turn}")

        check_text = str(parsed.get("scratch_pad") or "")
        if parsed.get("action") == "tool_call":
            check_text += "\n" + json.dumps(parsed.get("arguments") or {}, ensure_ascii=False)
        elif parsed.get("action") == "final_answer":
            # The final SQL is also model-produced text.  A gold literal that first
            # appears here is still hidden-answer leakage unless it was in the
            # prompt or a prior observation.
            check_text += "\n" + str(parsed.get("sql") or "")
        for literal in gold_strings:
            core = literal.strip("%_")
            if len(core) < 2:
                continue
            if text_has_literal(check_text, literal) and not text_has_literal(visible, literal):
                temporal_literal_hits.append({"turn": turn, "literal": literal})
        numeric_check_text = re.sub(r"Verified call:\s*\d+", "", check_text, flags=re.I)
        for literal in gold_numbers:
            if len(literal.lstrip("+-")) < 2 or literal in {"10", "100"}:
                continue
            if text_has_literal(numeric_check_text, literal, numeric=True) and not text_has_literal(
                visible, literal, numeric=True
            ):
                numeric_literal_warnings.append({"turn": turn, "literal": literal})

        if parsed.get("action") == "final_answer":
            if final is not None:
                issues.append("multiple_final_answers")
            final = parsed
            final_turn = turn
            i += 1
            continue

        name = str(parsed.get("name") or "")
        arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
        if i + 1 >= len(trajectory):
            issues.append(f"missing_observation_after_call_{len(calls) + 1}")
            observation = ""
            i += 1
        else:
            observation_item = trajectory[i + 1]
            if not isinstance(observation_item, dict) or observation_item.get("role") != "user":
                issues.append(f"invalid_observation_after_call_{len(calls) + 1}")
                observation = ""
                i += 1
            else:
                observation = unwrap_observation(str(observation_item.get("content", "")))
                i += 2
        call = {
            "index": len(calls) + 1,
            "turn": turn,
            "name": name,
            "arguments": arguments,
            "sql": str(arguments.get("sql") or "").strip(),
            "observation": observation,
        }
        calls.append(call)
        if db_file is not None:
            try:
                replayed = B.st.dispatch(str(db_file), name, arguments, db_id=db_id)
                if canonical_payload(replayed) != canonical_payload(observation):
                    replay_mismatches.append(call["index"])
                    issues.append(f"observation_replay_mismatch_call_{call['index']}")
            except Exception as exc:
                issues.append(f"tool_replay_error_call_{call['index']}")
                evidence.setdefault("tool_replay_errors", []).append(
                    {"call": call["index"], "error": f"{type(exc).__name__}: {exc}"}
                )
        visible += "\n" + observation

    if final is None:
        issues.append("missing_final_answer")
        final_sql = ""
    else:
        final_sql = str(final.get("sql") or "").strip()
    if final_turn is not None and final_turn != len(trajectory):
        issues.append("final_answer_not_last_trajectory_message")
    if not 2 <= len(calls) <= 6:
        issues.append("tool_call_count_outside_2_to_6")
    if int(metadata.get("tool_calls", -1)) != len(calls):
        issues.append("metadata_tool_call_count_mismatch")

    call_keys = [
        (call["name"], json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True))
        for call in calls
    ]
    duplicates = len(call_keys) - len(set(call_keys))
    if duplicates:
        issues.append("repeated_identical_tool_call")

    raw_matches = [
        call["index"]
        for call in calls
        if call["name"] == "sqlite_query"
        and strip_one_semicolon(call["sql"]) == strip_one_semicolon(final_sql)
    ]
    normalized_matches = [
        call["index"]
        for call in calls
        if call["name"] == "sqlite_query"
        and B.normalize_sql(call["sql"]) == B.normalize_sql(final_sql)
    ]
    if not raw_matches:
        issues.append("final_sql_not_executed_text_identically")
    if not normalized_matches:
        issues.append("final_sql_not_executed_normalized")

    verified = final.get("verified_call") if final else None
    if not isinstance(verified, int) or not 1 <= verified <= len(calls):
        issues.append("missing_or_invalid_verified_call")
    else:
        verified_call = calls[verified - 1]
        if verified_call["name"] != "sqlite_query":
            issues.append("verified_call_not_sqlite_query")
        if verified not in raw_matches:
            issues.append("verified_call_not_text_identical_final")
        if verified not in normalized_matches:
            issues.append("verified_call_not_normalized_final")
        if verified != len(calls):
            warnings.append("verified_call_is_not_last_tool_call")

    if temporal_literal_hits:
        issues.append("temporally_unseen_gold_string_literal")
    if numeric_literal_warnings:
        warnings.append("temporally_unseen_gold_numeric_literal")

    assistant_text = "\n".join(
        str(item.get("content", ""))
        for item in trajectory
        if isinstance(item, dict) and item.get("role") == "assistant"
    )
    prose = B.TOOL_CALL_BLOCK.sub(" ", assistant_text)
    prose = B.FINAL_ANSWER_BLOCK.sub(" ", prose)
    if B.REFERENCE_MENTION.search(prose):
        issues.append("mentions_privileged_reference")

    replay_grade: dict[str, Any] | None = None
    if db_file is not None and final_sql and gold_sql:
        replay_grade = B.grade_final(
            db_file,
            final_sql,
            gold_sql,
            str(row.get("grading_method") or "multiset"),
        )
        if replay_grade.get("ex") is not True:
            issues.append(
                "replayed_ex_not_correct"
                if replay_grade.get("ex") is False
                else "replayed_grading_infrastructure"
            )

    final_ids = identifiers(final_sql)
    possibly_irrelevant: list[int] = []
    for call in calls:
        if call["index"] in normalized_matches:
            continue
        if final_ids and not (tool_footprint(call["name"], call["arguments"]) & final_ids):
            possibly_irrelevant.append(call["index"])
    if possibly_irrelevant:
        warnings.append("possibly_irrelevant_diagnostic_tool_call")

    evidence.update(
        {
            "model": metadata.get("model"),
            "decoding": metadata.get("decoding"),
            "attempt": metadata.get("attempt"),
            "db_id": db_id,
            "tool_calls": len(calls),
            "assistant_turns": sum(
                isinstance(item, dict) and item.get("role") == "assistant"
                for item in trajectory
            ),
            "raw_final_sql_matches": raw_matches,
            "normalized_final_sql_matches": normalized_matches,
            "verified_call": verified,
            "replay_grade": replay_grade,
            "temporal_gold_string_hits": temporal_literal_hits,
            "temporal_gold_numeric_warnings": numeric_literal_warnings,
            "parser_warnings": parser_warnings,
            "observation_replay_mismatches": replay_mismatches,
            "possibly_irrelevant_calls": possibly_irrelevant,
        }
    )
    unique_issues = list(dict.fromkeys(issues))
    unique_warnings = list(dict.fromkeys(warnings))
    return index, {
        "id": row.get("id"),
        "sample_id": sample_id,
        "hard_pass": not unique_issues,
        "issues": unique_issues,
        "warnings": unique_warnings,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, default=B.DEFAULT_DB_ROOT)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--validated-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_row, index, row, args.db_root): index
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"audited {completed}/{len(futures)}", flush=True)
    results.sort(key=lambda pair: pair[0])
    audits = [audit for _, audit in results]
    audit_count = write_jsonl(args.audit_out, audits)
    validated_rows = [row for row, audit in zip(rows, audits) if audit["hard_pass"]]
    validated_count = write_jsonl(args.validated_out, validated_rows)

    issue_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for audit in audits:
        issue_counts.update(audit["issues"])
        warning_counts.update(audit["warnings"])
    sample_ids = [str((row.get("metadata") or {}).get("sample_id") or "") for row in rows]
    summary = {
        "input": str(args.input),
        "input_records": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "audited_records": audit_count,
        "hard_pass_records": validated_count,
        "hard_fail_records": len(rows) - validated_count,
        "issue_counts": dict(issue_counts),
        "warning_counts": dict(warning_counts),
        "audit_output": str(args.audit_out),
        "validated_output": str(args.validated_out),
        "notes": [
            "All final and gold SQL statements were re-executed for EX.",
            "Every tool observation was replayed against the database.",
            "Protocol parsing disabled unclosed-function repair; parser warnings are hard failures.",
            "Gold string literal visibility is checked at each turn using only the current prompt and prior observations.",
            "Logic/usefulness warnings are deterministic heuristics, not a proof of semantic faithfulness.",
        ],
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
