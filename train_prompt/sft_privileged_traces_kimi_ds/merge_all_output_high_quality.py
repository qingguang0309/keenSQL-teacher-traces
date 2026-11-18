#!/usr/bin/env python3
"""Strictly audit and merge hosted-teacher privileged trajectories.

The source model outputs are never modified.  Selection is performed in two
stages: greedy/pass@8 within each model, then across models.  Model identity is
not part of the quality key.  Each stage writes a strict multi-tool output and
an inclusive output that adds one-tool fallbacks only for otherwise uncovered
sample IDs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq


PROJECT = Path("/data/keenSQL")
SCRIPT_ROOT = PROJECT / "train_prompt/sft_privileged_traces_kimi_ds"
UPSTREAM_ROOT = PROJECT / "train_prompt/sft_privileged_traces"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(1, str(UPSTREAM_ROOT))

import build_sft_traces as B  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_module("_strict_first_round_auditor", UPSTREAM_ROOT / "audit_merged_sft.py")

DEFAULT_MODEL_ROOT = (
    PROJECT
    / "dataset/sft_data/privileged_traces_sft_privileged_traces_kimi_ds/models"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "dataset/sft_data/privileged_traces_sft_privileged_traces_kimi_ds/all_output_merge"
)
DEFAULT_DB_ROOT = PROJECT / "dataset/all_dataset/databases"
DEFAULT_SCHEMA_CACHE = PROJECT / "dataset/m_schema/cache"
DEFAULT_FROZEN = PROJECT / "data/sft.parquet"

TAGS = (
    "scratch_pad",
    "tool_call",
    "function",
    "parameter",
    "relevant_tables",
    "relevant_columns",
    "final_answer",
)
ERROR_KINDS = {
    "timeout",
    "step_limit_exceeded",
    "execution_error",
    "validation_error",
    "infrastructure_error",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_parse(text: str) -> dict[str, Any] | None:
    try:
        parsed = B.T.parse_assistant_turn(
            text,
            require_final_scratch_pad=True,
            require_tool_scratch_pad=True,
            repair_unclosed_function=False,
        )
    except Exception:
        return None
    return parsed if not parsed.get("warnings") else None


def tag_counts(text: str) -> dict[str, tuple[int, int]]:
    return {
        tag: (
            len(re.findall(rf"<\s*{tag}(?:\s|=|>)", text, re.I)),
            len(re.findall(rf"<\s*/\s*{tag}\s*>", text, re.I)),
        )
        for tag in TAGS
    }


def balanced(text: str) -> bool:
    return all(opened == closed for opened, closed in tag_counts(text).values())


def repair_crossed_final_annotations(text: str) -> tuple[str, str | None]:
    counts = tag_counts(text)
    if counts["relevant_tables"] != (1, 0) or counts["relevant_columns"] != (0, 2):
        return text, None
    pattern = re.compile(
        r"(<\s*relevant_tables\s*>.*?)(<\s*/\s*relevant_columns\s*>)"
        r"(.*?)(<\s*/\s*relevant_columns\s*>)",
        re.I | re.S,
    )
    match = pattern.search(text)
    if not match:
        return text, None
    replacement = (
        match.group(1)
        + "</relevant_tables>\n<relevant_columns>"
        + match.group(3)
        + match.group(4)
    )
    return text[: match.start()] + replacement + text[match.end() :], "repair_crossed_relevant_annotation_tags"


def repair_related_tables_typo(text: str) -> tuple[str, str | None]:
    counts = tag_counts(text)
    if counts["relevant_tables"] != (1, 0):
        return text, None
    matches = list(re.finditer(r"<\s*/\s*related_tables\s*>", text, re.I))
    if len(matches) != 1:
        return text, None
    match = matches[0]
    return text[: match.start()] + "</relevant_tables>" + text[match.end() :], "repair_related_tables_closing_typo"


def repair_one_missing_parameter_close(text: str) -> tuple[str, str | None]:
    counts = tag_counts(text)
    unbalanced = {tag for tag, pair in counts.items() if pair[0] != pair[1]}
    if unbalanced != {"parameter"} or counts["parameter"][0] != counts["parameter"][1] + 1:
        return text, None
    if counts["tool_call"] != (1, 1) or counts["function"] != (1, 1):
        return text, None
    pattern = re.compile(
        r"(<\s*parameter\s*=\s*[^>]+>)(?P<body>(?:(?!<\s*/\s*parameter\s*>).)*?)"
        r"(?=<\s*parameter\s*=)",
        re.I | re.S,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return text, None
    match = matches[0]
    repaired = text[: match.end("body")] + "</parameter>\n" + text[match.end("body") :]
    return repaired, "insert_single_missing_parameter_close"


def observation_payload(text: str) -> Any:
    unwrapped = A.unwrap_observation(text)
    try:
        return json.loads(unwrapped)
    except json.JSONDecodeError:
        return None


def observation_error_kind(text: str) -> str | None:
    payload = observation_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return str(payload["error"].get("kind") or "unknown_error").strip().lower()
    return None


def successful_sqlite_observation(text: str) -> bool:
    payload = observation_payload(text)
    return (
        isinstance(payload, dict)
        and "error" not in payload
        and isinstance(payload.get("columns"), list)
        and isinstance(payload.get("rows"), list)
    )


def trajectory_details(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    try:
        start = int(row.get("prompt_messages"))
    except (TypeError, ValueError):
        return {"error": "invalid_prompt_messages"}
    calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    final_message_index: int | None = None
    assistant_turns = 0
    i = start
    while i < len(messages):
        item = messages[i]
        if not isinstance(item, dict) or item.get("role") != "assistant":
            return {"error": f"expected_assistant_at_{i}"}
        assistant_turns += 1
        parsed = strict_parse(str(item.get("content", "")))
        if parsed is None:
            return {"error": f"strict_parse_failed_at_{i}"}
        if parsed.get("action") == "final_answer":
            if final is not None:
                return {"error": "multiple_final_answers"}
            final = parsed
            final_message_index = i
            i += 1
            continue
        if i + 1 >= len(messages):
            return {"error": f"missing_observation_after_{i}"}
        observation_item = messages[i + 1]
        if not isinstance(observation_item, dict) or observation_item.get("role") != "user":
            return {"error": f"invalid_observation_after_{i}"}
        arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
        calls.append(
            {
                "index": len(calls) + 1,
                "name": str(parsed.get("name") or ""),
                "arguments": arguments,
                "sql": str(arguments.get("sql") or "").strip(),
                "observation": A.unwrap_observation(str(observation_item.get("content", ""))),
            }
        )
        i += 2
    return {
        "error": None,
        "calls": calls,
        "final": final,
        "final_message_index": final_message_index,
        "assistant_turns": assistant_turns,
        "final_is_last": final_message_index == len(messages) - 1,
    }


def refresh_identity(row: dict[str, Any], original_id: str, repairs: list[str]) -> None:
    serialized = json.dumps(
        row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    attempt_id = original_id.rsplit("-", 1)[0]
    row["id"] = f"{attempt_id}-{digest[:12]}"
    metadata = row.setdefault("metadata", {})
    metadata["trajectory_sha256"] = digest
    metadata["format_repairs"] = repairs
    metadata["original_record_id"] = original_id


def apply_mechanical_repairs(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row = copy.deepcopy(source)
    repairs: list[str] = []
    start = int(row["prompt_messages"])
    for message in row["messages"][start:]:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", ""))
        if balanced(text) and strict_parse(text) is not None:
            continue
        updated = text
        local_repairs: list[str] = []
        for repair in (
            repair_crossed_final_annotations,
            repair_related_tables_typo,
            repair_one_missing_parameter_close,
        ):
            updated, label = repair(updated)
            if label:
                local_repairs.append(label)
        if local_repairs and balanced(updated) and strict_parse(updated) is not None:
            message["content"] = updated
            repairs.extend(local_repairs)

    details = trajectory_details(row)
    if not details.get("error") and details.get("final") and details.get("calls"):
        calls = details["calls"]
        final = details["final"]
        final_sql = str(final.get("sql") or "").strip()
        last = calls[-1]
        can_repair_pointer = (
            last["name"] == "sqlite_query"
            and last["sql"] == final_sql
            and successful_sqlite_observation(last["observation"])
            and details.get("final_is_last")
        )
        if can_repair_pointer and final.get("verified_call") != len(calls):
            message_index = details["final_message_index"]
            message = row["messages"][message_index]
            text = str(message.get("content", ""))
            pointer_matches = list(re.finditer(r"\bVerified\s+call\s*:\s*\d+\b", text, re.I))
            label: str | None = None
            if len(pointer_matches) == 1:
                match = pointer_matches[0]
                text = text[: match.start()] + f"Verified call: {len(calls)}" + text[match.end() :]
                label = "repair_verified_call_to_last"
            elif not pointer_matches:
                scratch = list(
                    re.finditer(r"(<\s*scratch_pad\s*>)(.*?)(<\s*/\s*scratch_pad\s*>)", text, re.I | re.S)
                )
                if len(scratch) == 1:
                    match = scratch[0]
                    body = match.group(2).rstrip()
                    replacement = (
                        match.group(1)
                        + body
                        + f"\nVerified call: {len(calls)}\n"
                        + match.group(3)
                    )
                    text = text[: match.start()] + replacement + text[match.end() :]
                    label = "insert_verified_call_for_last"
            if label and strict_parse(text) is not None and balanced(text):
                message["content"] = text
                repairs.append(label)

    if repairs:
        refresh_identity(row, str(source.get("id") or ""), repairs)
    return row, repairs


def raw_episode_prefilter(raw: Mapping[str, Any]) -> tuple[bool, str]:
    grade = raw.get("grade") if isinstance(raw.get("grade"), dict) else {}
    if grade.get("ex") is not True:
        return False, "generation_ex_not_true"
    episode = raw.get("episode") if isinstance(raw.get("episode"), dict) else {}
    final_sql = str(episode.get("final_sql") or raw.get("final_sql") or "").strip()
    if not final_sql:
        return False, "missing_final_sql"
    tool_calls = int(episode.get("tool_calls") or 0)
    if not 1 <= tool_calls <= 8:
        return False, "tool_call_count_outside_1_to_8"
    executed = episode.get("executed_calls") if isinstance(episode.get("executed_calls"), list) else []
    if not executed:
        return False, "missing_executed_calls"
    last = executed[-1]
    if str(last.get("tool") or "") != "sqlite_query":
        return False, "last_tool_not_sqlite_query"
    if str(last.get("sql") or "").strip() != final_sql:
        return False, "last_sql_not_text_identical_final"
    observations = [
        str(item.get("content", ""))
        for item in episode.get("transcript") or []
        if isinstance(item, dict) and item.get("role") == "observation"
    ]
    if len(observations) != tool_calls:
        return False, "observation_count_mismatch"
    if any(observation_error_kind(text) for text in observations):
        return False, "trajectory_contains_tool_error"
    if not observations or not successful_sqlite_observation(observations[-1]):
        return False, "last_sqlite_observation_not_successful"
    issues = set((raw.get("acceptance") or {}).get("issues") or [])
    forbidden_generation_issues = {
        "mentions_privileged_reference",
        "mentions_gold_sql_label",
        "unobserved_gold_literal_in_reasoning",
        "grading_infrastructure",
        "no_tool_call",
        "ex_not_correct",
    }
    found = sorted(issues & forbidden_generation_issues)
    if found:
        return False, "generation_hard_issue:" + ",".join(found)
    return True, "eligible"


def custom_audit(index: int, row: dict[str, Any], db_root: Path) -> tuple[int, dict[str, Any]]:
    _, audit = A.audit_row(index, row, db_root)
    issues = list(audit.get("issues") or [])
    warnings = list(audit.get("warnings") or [])
    evidence = audit.get("evidence") if isinstance(audit.get("evidence"), dict) else {}
    tool_calls = int(evidence.get("tool_calls") or 0)

    if 1 <= tool_calls <= 8:
        issues = [issue for issue in issues if issue != "tool_call_count_outside_2_to_6"]
    if "repeated_identical_tool_call" in issues:
        issues = [issue for issue in issues if issue != "repeated_identical_tool_call"]
        warnings.append("repeated_identical_tool_call")
    if "verified_call_is_not_last_tool_call" in warnings:
        warnings = [warning for warning in warnings if warning != "verified_call_is_not_last_tool_call"]
        issues.append("verified_call_is_not_last_tool_call")
    if "temporally_unseen_gold_numeric_literal" in warnings:
        warnings = [warning for warning in warnings if warning != "temporally_unseen_gold_numeric_literal"]
        issues.append("temporally_unseen_gold_numeric_literal")

    details = trajectory_details(row)
    if details.get("error"):
        issues.append(str(details["error"]))
    else:
        calls = details["calls"]
        final = details.get("final") or {}
        final_sql = str(final.get("sql") or "").strip()
        if not 1 <= len(calls) <= 8:
            issues.append("tool_call_count_outside_1_to_8")
        error_kinds = [observation_error_kind(call["observation"]) for call in calls]
        error_kinds = [kind for kind in error_kinds if kind]
        if error_kinds:
            issues.append("trajectory_contains_failed_tool_call")
            evidence["tool_error_kinds"] = error_kinds
        if calls:
            last = calls[-1]
            if last["name"] != "sqlite_query":
                issues.append("last_tool_not_sqlite_query")
            if last["sql"] != final_sql:
                issues.append("last_sql_not_text_identical_final")
            if not successful_sqlite_observation(last["observation"]):
                issues.append("last_sqlite_observation_not_successful")
            if final.get("verified_call") != len(calls):
                issues.append("verified_call_not_last_tool_call")
        if not details.get("final_is_last"):
            issues.append("final_answer_not_last_trajectory_message")
        irrelevant = list(evidence.get("possibly_irrelevant_calls") or [])
        diagnostic_calls = max(0, len(calls) - 1)
        if diagnostic_calls and len(set(irrelevant)) >= diagnostic_calls:
            issues.append("all_diagnostic_tool_calls_irrelevant")

    audit["issues"] = list(dict.fromkeys(issues))
    audit["warnings"] = list(dict.fromkeys(warnings))
    audit["evidence"] = evidence
    audit["hard_pass"] = not audit["issues"]
    return index, audit


def quality_components(row: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    details = trajectory_details(row)
    calls = details.get("calls") or []
    call_keys = [
        (call["name"], json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True))
        for call in calls
    ]
    repeated = len(call_keys) - len(set(call_keys))
    evidence = audit.get("evidence") if isinstance(audit.get("evidence"), dict) else {}
    irrelevant = len(set(evidence.get("possibly_irrelevant_calls") or []))
    nonfinal = max(0, len(calls) - 1)
    useful_diagnostics = max(0, nonfinal - irrelevant)
    bm25_or_peek = sum(
        call["name"] in {"bm25_search_sqlite", "sqlite_peek"} for call in calls[:-1]
    )
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    repairs = list(metadata.get("format_repairs") or [])
    warnings = list(audit.get("warnings") or [])
    assistant_chars = sum(
        len(str(message.get("content", "")))
        for message in row.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    return {
        "repairs": len(repairs),
        "repair_labels": repairs,
        "irrelevant_calls": irrelevant,
        "repeated_calls": repeated,
        "warnings": len(warnings),
        "warning_labels": warnings,
        "useful_diagnostic_calls": useful_diagnostics,
        "bm25_or_peek_calls": bm25_or_peek,
        "tool_calls": len(calls),
        "assistant_turns": int(details.get("assistant_turns") or 0),
        "assistant_chars": assistant_chars,
        "decoding": str(metadata.get("decoding") or ""),
        "trajectory_sha256": str(metadata.get("trajectory_sha256") or ""),
    }


def quality_key(components: Mapping[str, Any]) -> tuple[Any, ...]:
    tool_calls = int(components["tool_calls"])
    preferred_excess = max(0, tool_calls - 4) if tool_calls > 1 else 0
    return (
        int(components["repairs"] > 0),
        int(components["repairs"]),
        int(components["irrelevant_calls"]),
        int(components["repeated_calls"]),
        int(components["warnings"]),
        -int(components["bm25_or_peek_calls"]),
        -int(components["useful_diagnostic_calls"]),
        preferred_excess,
        int(components["assistant_turns"]),
        int(components["assistant_chars"]),
        0 if components["decoding"] == "greedy" else 1,
        str(components["trajectory_sha256"]),
    )


def candidate_manifest(candidate: Mapping[str, Any], selected_stage: str) -> dict[str, Any]:
    row = candidate["row"]
    metadata = row.get("metadata") or {}
    return {
        "sample_id": metadata.get("sample_id"),
        "question_id": row.get("question_id"),
        "db_id": row.get("db_id"),
        "difficulty": candidate.get("difficulty"),
        "model": metadata.get("model"),
        "decoding": metadata.get("decoding"),
        "attempt": metadata.get("attempt"),
        "category": candidate["category"],
        "selected_stage": selected_stage,
        "quality": candidate["quality"],
        "record_id": row.get("id"),
    }


def select_best(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["sample_id"]].append(candidate)
    selected: list[dict[str, Any]] = []
    dropped = 0
    for sample_id in sorted(grouped):
        options = grouped[sample_id]
        selected.append(min(options, key=lambda item: item["quality_key"]))
        dropped += len(options) - 1
    return selected, dropped


def distribution(candidates: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(candidate.get(key) or "unknown") for candidate in candidates).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--schema-cache", type=Path, default=DEFAULT_SCHEMA_CACHE)
    parser.add_argument("--frozen-sft", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-source", type=int)
    args = parser.parse_args()

    raw_paths = sorted(args.model_root.glob("*/*/raw_attempts.jsonl"))
    if not raw_paths:
        raise FileNotFoundError(f"no raw attempts under {args.model_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    configs: dict[str, dict[str, Any]] = {}
    source_records: dict[str, list[dict[str, Any]]] = {}
    source_args: dict[str, SimpleNamespace] = {}
    teacher_templates: dict[str, str] = {}
    deployment_cache: dict[tuple[str, int], tuple[list[dict[str, str]], dict[str, Any]]] = {}
    prefilter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_totals: dict[str, Counter[str]] = defaultdict(Counter)
    prepared: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        source = f"{raw_path.parent.parent.name}/{raw_path.parent.name}"
        config = json.loads((raw_path.parent / "config.json").read_text(encoding="utf-8"))
        configs[source] = config
        input_path = str(config["input"])
        if input_path not in source_records:
            source_records[input_path] = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if input_path not in teacher_templates:
            teacher_templates[input_path] = Path(config["teacher_prompt"]).read_text(encoding="utf-8")
        source_args[source] = SimpleNamespace(model=config["model"], decoding=config["decoding"])
        rows = source_records[input_path]

        for row_number, raw in enumerate(iter_jsonl(raw_path), 1):
            if args.limit_per_source and row_number > args.limit_per_source:
                break
            source_totals[source]["attempts"] += 1
            eligible, reason = raw_episode_prefilter(raw)
            if not eligible:
                prefilter_counts[source][reason] += 1
                continue
            source_totals[source]["prefilter_eligible"] += 1
            source_index = int(raw["source_index"])
            record = rows[source_index]
            if raw.get("accepted") is True and isinstance(raw.get("sft_record"), dict):
                sft_row = copy.deepcopy(raw["sft_record"])
                source_totals[source]["used_existing_sft_record"] += 1
            else:
                cache_key = (input_path, source_index)
                if cache_key not in deployment_cache:
                    schema = (args.schema_cache / f"{record['db_id']}.txt").read_text(encoding="utf-8")
                    deployment, _teacher, rebuilt_meta = B.build_prompts(
                        record,
                        schema,
                        teacher_templates[input_path],
                        int(config["max_prompt_tokens"]),
                    )
                    deployment_cache[cache_key] = (deployment, rebuilt_meta)
                deployment, rebuilt_meta = deployment_cache[cache_key]
                prompt_meta = raw.get("prompt_meta") if isinstance(raw.get("prompt_meta"), dict) else rebuilt_meta
                sft_row = B.make_sft_record(
                    deployment_messages=deployment,
                    episode=raw["episode"],
                    sample_key=str(raw["sample_id"]),
                    attempt_id=str(raw["attempt_id"]),
                    source_index=source_index,
                    record=record,
                    args=source_args[source],
                    grade=raw["grade"],
                    prompt_meta=prompt_meta,
                )
                source_totals[source]["reconstructed_from_raw"] += 1
            repaired, repairs = apply_mechanical_repairs(sft_row)
            if repairs:
                source_totals[source]["mechanically_repaired"] += 1
                prefilter_counts[source].update(f"repair:{label}" for label in repairs)
            prepared.append(
                {
                    "source": source,
                    "row": repaired,
                    "raw_attempt_id": raw.get("attempt_id"),
                    "generation_accepted": raw.get("accepted") is True,
                    "difficulty": str(record.get("difficulty") or "unknown"),
                }
            )

    print(f"prepared {len(prepared)} candidates; strict replay audit starting", flush=True)
    audit_results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(custom_audit, index, item["row"], args.db_root): index
            for index, item in enumerate(prepared)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            audit_results.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"audited {completed}/{len(futures)}", flush=True)
    audit_results.sort(key=lambda pair: pair[0])
    audits = [audit for _, audit in audit_results]

    frozen_ids = {
        str(value)
        for value in pq.read_table(args.frozen_sft, columns=["question_id"])["question_id"].to_pylist()
    }
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    audit_rows: list[dict[str, Any]] = []
    for item, audit in zip(prepared, audits):
        row = item["row"]
        metadata = row.get("metadata") or {}
        details = trajectory_details(row)
        category = "single_tool" if len(details.get("calls") or []) == 1 else "multi_tool"
        audit_record = {
            "record_id": row.get("id"),
            "sample_id": metadata.get("sample_id"),
            "question_id": row.get("question_id"),
            "source": item["source"],
            "model": metadata.get("model"),
            "decoding": metadata.get("decoding"),
            "attempt": metadata.get("attempt"),
            "category": category,
            "generation_accepted": item["generation_accepted"],
            "hard_pass": audit.get("hard_pass") is True,
            "issues": audit.get("issues") or [],
            "warnings": audit.get("warnings") or [],
            "evidence": audit.get("evidence") or {},
        }
        audit_rows.append(audit_record)
        if audit.get("hard_pass") is not True:
            issue_counts.update(audit_record["issues"])
            rejected.append(audit_record)
            continue
        warning_counts.update(audit_record["warnings"])
        quality = quality_components(row, audit)
        valid.append(
            {
                "source": item["source"],
                "model": str(metadata.get("model") or ""),
                "decoding": str(metadata.get("decoding") or ""),
                "sample_id": str(metadata.get("sample_id") or ""),
                "question_id": str(row.get("question_id") or ""),
                "db_id": str(row.get("db_id") or ""),
                "difficulty": item["difficulty"],
                "category": category,
                "in_frozen_sft": str(row.get("question_id") or "") in frozen_ids,
                "row": row,
                "audit": audit,
                "quality": quality,
                "quality_key": quality_key(quality),
            }
        )

    audit_dir = args.output_root / "audit"
    atomic_jsonl(audit_dir / "candidate_audit.jsonl", audit_rows)
    atomic_jsonl(audit_dir / "rejected_candidates.jsonl", rejected)

    by_model_summary: dict[str, Any] = {}
    model_selected: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for model in sorted({candidate["model"] for candidate in valid}):
        model_dir = args.model_root / model / "merged"
        model_valid = [candidate for candidate in valid if candidate["model"] == model]
        category_pool_summary: dict[str, Any] = {}
        for category in ("multi_tool", "single_tool"):
            pool_candidates = [candidate for candidate in model_valid if candidate["category"] == category]
            selected, dropped = select_best(pool_candidates)
            model_selected[model][category] = selected
            category_pool_summary[category] = {
                "hard_pass_candidates": len(pool_candidates),
                "selected_unique_ids": len(selected),
                "alternatives_dropped": dropped,
                "selected_by_decoding": distribution(selected, "decoding"),
                "selected_difficulty": distribution(selected, "difficulty"),
                "selected_in_frozen_sft": sum(candidate["in_frozen_sft"] for candidate in selected),
                "selected_not_in_frozen_sft": sum(not candidate["in_frozen_sft"] for candidate in selected),
            }

        strict_selected = model_selected[model]["multi_tool"]
        strict_ids = {candidate["sample_id"] for candidate in strict_selected}
        single_fallback = [
            candidate for candidate in model_selected[model]["single_tool"]
            if candidate["sample_id"] not in strict_ids
        ]
        inclusive_selected = sorted(
            [*strict_selected, *single_fallback],
            key=lambda candidate: candidate["sample_id"],
        )
        strict_output = model_dir / "strict_high_quality.jsonl"
        inclusive_output = model_dir / "strict_high_quality_with_single_tool.jsonl"
        model_audit_dir = model_dir / "audit"
        strict_manifest = model_audit_dir / "strict_high_quality.selection.jsonl"
        inclusive_manifest = model_audit_dir / "strict_high_quality_with_single_tool.selection.jsonl"
        atomic_jsonl(strict_output, (candidate["row"] for candidate in strict_selected))
        atomic_jsonl(
            strict_manifest,
            (candidate_manifest(candidate, "within_model_strict") for candidate in strict_selected),
        )
        atomic_jsonl(inclusive_output, (candidate["row"] for candidate in inclusive_selected))
        atomic_jsonl(
            inclusive_manifest,
            (
                candidate_manifest(
                    candidate,
                    "within_model_strict" if candidate["sample_id"] in strict_ids else "single_tool_fallback",
                )
                for candidate in inclusive_selected
            ),
        )

        def model_output_summary(selected: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
            return {
                "selected_unique_ids": len(selected),
                "selected_by_decoding": distribution(selected, "decoding"),
                "selected_by_category": distribution(selected, "category"),
                "selected_difficulty": distribution(selected, "difficulty"),
                "selected_in_frozen_sft": sum(candidate["in_frozen_sft"] for candidate in selected),
                "selected_not_in_frozen_sft": sum(not candidate["in_frozen_sft"] for candidate in selected),
                "selected_with_repairs": sum(candidate["quality"]["repairs"] > 0 for candidate in selected),
                "output": str(output_path),
                "sha256": sha256_file(output_path),
            }

        model_summary = {
            "candidate_pools": category_pool_summary,
            "strict": {
                **model_output_summary(strict_selected, strict_output),
                "selection_manifest": str(strict_manifest),
            },
            "strict_with_single_tool": {
                **model_output_summary(inclusive_selected, inclusive_output),
                "single_tool_fallback_added": len(single_fallback),
                "selection_manifest": str(inclusive_manifest),
            },
            "policy": {
                "strict_records_preserved_in_inclusive_output": True,
                "single_tool_candidates_are_not_a_standalone_final_output": True,
            },
        }
        by_model_summary[model] = model_summary
        atomic_json(model_dir / "summary.json", model_summary)
        for legacy_name in (
            "high_quality_multi_tool.jsonl",
            "high_quality_multi_tool.selection.jsonl",
            "successful_single_tool.jsonl",
            "successful_single_tool.selection.jsonl",
        ):
            legacy_path = model_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

    legacy_by_model = args.output_root / "by_model"
    if legacy_by_model.exists():
        resolved_legacy = legacy_by_model.resolve()
        resolved_output = args.output_root.resolve()
        if resolved_legacy.parent != resolved_output or resolved_legacy.name != "by_model":
            raise RuntimeError(f"refusing to remove unexpected legacy path: {resolved_legacy}")
        shutil.rmtree(resolved_legacy)

    final_dir = args.output_root / "final"
    final_summary: dict[str, Any] = {}
    final_selected: dict[str, list[dict[str, Any]]] = {}
    category_pool_summary: dict[str, Any] = {}
    for category in ("multi_tool", "single_tool"):
        cross_model_candidates = [
            candidate
            for model in sorted(model_selected)
            for candidate in model_selected[model].get(category, [])
        ]
        selected, dropped = select_best(cross_model_candidates)
        final_selected[category] = selected
        category_pool_summary[category] = {
            "per_model_best_candidates": len(cross_model_candidates),
            "selected_unique_ids": len(selected),
            "alternatives_dropped": dropped,
            "selected_by_model": distribution(selected, "model"),
            "selected_by_decoding": distribution(selected, "decoding"),
            "selected_difficulty": distribution(selected, "difficulty"),
            "selected_in_frozen_sft": sum(candidate["in_frozen_sft"] for candidate in selected),
            "selected_not_in_frozen_sft": sum(not candidate["in_frozen_sft"] for candidate in selected),
            "selected_with_repairs": sum(candidate["quality"]["repairs"] > 0 for candidate in selected),
        }

    multi_ids = {candidate["sample_id"] for candidate in final_selected["multi_tool"]}
    single_ids = {candidate["sample_id"] for candidate in final_selected["single_tool"]}
    strict_selected = final_selected["multi_tool"]
    single_fallback = [
        candidate for candidate in final_selected["single_tool"]
        if candidate["sample_id"] not in multi_ids
    ]
    inclusive_selected = sorted(
        [*strict_selected, *single_fallback],
        key=lambda candidate: candidate["sample_id"],
    )

    strict_output = final_dir / "strict_high_quality.jsonl"
    inclusive_output = final_dir / "strict_high_quality_with_single_tool.jsonl"
    strict_manifest = audit_dir / "final_strict_high_quality.selection.jsonl"
    inclusive_manifest = audit_dir / "final_strict_high_quality_with_single_tool.selection.jsonl"
    atomic_jsonl(strict_output, (candidate["row"] for candidate in strict_selected))
    atomic_jsonl(
        strict_manifest,
        (candidate_manifest(candidate, "cross_model_strict") for candidate in strict_selected),
    )
    atomic_jsonl(inclusive_output, (candidate["row"] for candidate in inclusive_selected))
    atomic_jsonl(
        inclusive_manifest,
        (
            candidate_manifest(
                candidate,
                "cross_model_strict" if candidate["sample_id"] in multi_ids else "single_tool_fallback",
            )
            for candidate in inclusive_selected
        ),
    )

    def final_output_summary(selected: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
        return {
            "selected_unique_ids": len(selected),
            "selected_by_model": distribution(selected, "model"),
            "selected_by_decoding": distribution(selected, "decoding"),
            "selected_by_category": distribution(selected, "category"),
            "selected_difficulty": distribution(selected, "difficulty"),
            "selected_in_frozen_sft": sum(candidate["in_frozen_sft"] for candidate in selected),
            "selected_not_in_frozen_sft": sum(not candidate["in_frozen_sft"] for candidate in selected),
            "selected_with_repairs": sum(candidate["quality"]["repairs"] > 0 for candidate in selected),
            "output": str(output_path),
            "sha256": sha256_file(output_path),
        }

    final_summary["strict"] = {
        **final_output_summary(strict_selected, strict_output),
        "selection_manifest": str(strict_manifest),
    }
    final_summary["strict_with_single_tool"] = {
        **final_output_summary(inclusive_selected, inclusive_output),
        "single_tool_fallback_added": len(single_fallback),
        "selection_manifest": str(inclusive_manifest),
    }
    final_summary["cross_model_category_pools"] = category_pool_summary
    final_summary["category_relationship"] = {
        "overlap_ids": len(multi_ids & single_ids),
        "multi_only_ids": len(multi_ids - single_ids),
        "single_only_ids": len(single_ids - multi_ids),
        "union_ids": len(multi_ids | single_ids),
    }
    final_summary["policy"] = {
        "selection_stages": ["within_model_across_decoding", "across_models"],
        "model_priority": "none",
        "final_outputs": [
            "strict_multi_tool_only",
            "strict_multi_tool_plus_single_tool_for_ids_missing_from_strict",
        ],
        "strict_records_preserved_in_inclusive_output": True,
        "sql_repair": "never",
        "allowed_repairs": [
            "uniquely_determined_tag_closure",
            "verified_call_pointer_to_already_identical_successful_last_sqlite_call",
        ],
    }

    for legacy_name in (
        "high_quality_multi_tool.jsonl",
        "high_quality_multi_tool.selection.jsonl",
        "successful_single_tool.jsonl",
        "successful_single_tool.selection.jsonl",
    ):
        legacy_path = final_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    global_summary = {
        "model_root": str(args.model_root),
        "output_root": str(args.output_root),
        "sources": {
            source: {
                **dict(source_totals[source]),
                "prefilter_reasons": dict(prefilter_counts[source]),
            }
            for source in sorted(source_totals)
        },
        "prepared_candidates": len(prepared),
        "hard_pass_candidates": len(valid),
        "hard_fail_candidates": len(rejected),
        "hard_issue_counts": dict(issue_counts),
        "hard_pass_warning_counts": dict(warning_counts),
        "by_model": by_model_summary,
        "final": final_summary,
        "audit_files": {
            "candidate_audit": str(audit_dir / "candidate_audit.jsonl"),
            "rejected_candidates": str(audit_dir / "rejected_candidates.jsonl"),
        },
    }
    atomic_json(args.output_root / "summary.json", global_summary)
    atomic_json(final_dir / "summary.json", final_summary)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
