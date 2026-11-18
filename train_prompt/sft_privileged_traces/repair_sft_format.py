#!/usr/bin/env python3
"""Apply only mechanically provable formatting repairs to SFT candidates.

The source JSONL is never edited.  A candidate is changed only when a narrowly
defined malformed-tag pattern or a formatting-only final-SQL mismatch is found.
The repaired message list receives a new content hash and record id.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACE_DIR = Path("/data/keenSQL/train_prompt/sft_privileged_traces")
sys.path.insert(0, str(TRACE_DIR))

import build_sft_traces as B  # noqa: E402


TAGS = (
    "scratch_pad",
    "tool_call",
    "function",
    "parameter",
    "relevant_tables",
    "relevant_columns",
    "final_answer",
)


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


def repair_crossed_final_annotations(text: str) -> tuple[str, str | None]:
    """Repair the single observed qwen annotation crossing, and nothing broader."""
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
    repaired = text[: match.start()] + replacement + text[match.end() :]
    return repaired, "repair_crossed_relevant_annotation_tags"


def repair_related_tables_typo(text: str) -> tuple[str, str | None]:
    counts = tag_counts(text)
    if counts["relevant_tables"] != (1, 0):
        return text, None
    matches = list(re.finditer(r"<\s*/\s*related_tables\s*>", text, re.I))
    if len(matches) != 1:
        return text, None
    match = matches[0]
    repaired = text[: match.start()] + "</relevant_tables>" + text[match.end() :]
    return repaired, "repair_related_tables_closing_typo"


def repair_one_missing_parameter_close(text: str) -> tuple[str, str | None]:
    """Insert one close only for an otherwise-balanced single tool call."""
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


def parse_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    start = int(row["prompt_messages"])
    parsed_calls: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for message in row["messages"][start:]:
        if message.get("role") != "assistant":
            continue
        parsed = strict_parse(str(message.get("content", "")))
        if parsed is None:
            return [], None
        if parsed.get("action") == "tool_call":
            parsed_calls.append(parsed)
        elif parsed.get("action") == "final_answer":
            if final is not None:
                return [], None
            final = parsed
    return parsed_calls, final


def declared_verified_call(final: dict[str, Any]) -> int | None:
    text = str(final.get("scratch_pad") or "")
    match = re.search(r"\bVerified\s+call\s*:\s*(\d+)\b", text, re.I)
    return int(match.group(1)) if match else None


def replace_final_sql_with_executed_text(row: dict[str, Any]) -> str | None:
    calls, final = parse_trajectory(row)
    if final is None:
        return None
    final_sql = str(final.get("sql") or "").strip()
    verified = declared_verified_call(final)
    if verified is None or not 1 <= verified <= len(calls):
        return None
    call = calls[verified - 1]
    if call.get("name") != "sqlite_query":
        return None
    executed_sql = str((call.get("arguments") or {}).get("sql") or "").strip()
    if not executed_sql or B.normalize_sql(executed_sql) != B.normalize_sql(final_sql):
        return None
    if B.normalize_sql(executed_sql) == B.normalize_sql(final_sql) and executed_sql == final_sql:
        return None

    start = int(row["prompt_messages"])
    for message in row["messages"][start:]:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", ""))
        parsed = strict_parse(text)
        if parsed is None or parsed.get("action") != "final_answer":
            continue
        matches = list(re.finditer(r"(<\s*final_answer\s*>)(.*?)(<\s*/\s*final_answer\s*>)", text, re.I | re.S))
        if len(matches) != 1:
            return None
        match = matches[0]
        message["content"] = (
            text[: match.start()]
            + match.group(1)
            + "\n"
            + executed_sql
            + "\n"
            + match.group(3)
            + text[match.end() :]
        )
        return "restore_final_sql_from_declared_verified_call"
    return None


def refresh_identity(row: dict[str, Any], original_id: str, repairs: list[str]) -> None:
    serialized = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    attempt_id = original_id.rsplit("-", 1)[0]
    row["id"] = f"{attempt_id}-{digest[:12]}"
    metadata = row.setdefault("metadata", {})
    metadata["trajectory_sha256"] = digest
    metadata["format_repairs"] = repairs
    metadata["original_record_id"] = original_id


def repair_row(source: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row = copy.deepcopy(source)
    repairs: list[str] = []
    start = int(row["prompt_messages"])
    for message in row["messages"][start:]:
        if message.get("role") != "assistant":
            continue
        original = str(message.get("content", ""))
        if balanced(original) and strict_parse(original) is not None:
            continue
        candidates = (
            repair_crossed_final_annotations,
            repair_related_tables_typo,
            repair_one_missing_parameter_close,
        )
        repaired = original
        applied: list[str] = []
        for repair in candidates:
            updated, label = repair(repaired)
            if label:
                repaired = updated
                applied.append(label)
        if applied and balanced(repaired) and strict_parse(repaired) is not None:
            message["content"] = repaired
            repairs.extend(applied)

    sql_repair = replace_final_sql_with_executed_text(row)
    if sql_repair:
        repairs.append(sql_repair)
    if repairs:
        refresh_identity(row, str(source.get("id") or ""), repairs)
    return row, repairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_name(f".{args.output.name}.tmp")
    manifest_tmp = args.manifest.with_name(f".{args.manifest.name}.tmp")
    counts: Counter[str] = Counter()
    total = changed = 0
    with args.input.open(encoding="utf-8") as source, output_tmp.open("w", encoding="utf-8") as out, manifest_tmp.open("w", encoding="utf-8") as manifest:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            repaired, repairs = repair_row(row)
            out.write(json.dumps(repaired, ensure_ascii=False) + "\n")
            if repairs:
                changed += 1
                counts.update(repairs)
                manifest.write(json.dumps({
                    "source": str(args.input),
                    "line": line_number,
                    "original_id": row.get("id"),
                    "repaired_id": repaired.get("id"),
                    "repairs": repairs,
                }, ensure_ascii=False) + "\n")
    output_tmp.replace(args.output)
    manifest_tmp.replace(args.manifest)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "records": total,
        "changed_records": changed,
        "repairs": dict(counts),
        "manifest": str(args.manifest),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
