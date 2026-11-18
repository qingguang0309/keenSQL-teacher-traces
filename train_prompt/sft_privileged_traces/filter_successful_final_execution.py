#!/usr/bin/env python3
"""Keep EX-correct SFT rows whose exact-final sqlite_query returned successfully."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.I | re.S)
FINAL_BLOCK = re.compile(r"<final_answer>(.*?)</final_answer>", re.I | re.S)
FUNCTION_NAME = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.I)
SQL_PARAMETER = re.compile(
    r"<parameter\s*=\s*sql\s*>(.*?)</parameter>", re.I | re.S
)
VERIFIED_CALL = re.compile(r"\bVerified\s+call\s*:\s*(\d+)\b", re.I)
OBSERVATION = re.compile(r"^\s*<observation>\s*(.*?)\s*</observation>\s*$", re.I | re.S)
SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", SQL_COMMENT.sub(" ", sql or "")).strip().rstrip(";").strip().lower()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def evaluate(row: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("ex") is not True:
        return False, "metadata_ex_not_true", {}
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False, "messages_missing", {}
    try:
        prompt_messages = int(row.get("prompt_messages", 0))
    except (TypeError, ValueError):
        return False, "invalid_prompt_messages", {}
    if not 0 <= prompt_messages < len(messages):
        return False, "invalid_prompt_messages", {}
    trajectory = messages[prompt_messages:]
    assistant_positions = [
        index
        for index, message in enumerate(trajectory)
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if not assistant_positions:
        return False, "no_assistant_trajectory", {}
    final_index = assistant_positions[-1]
    final_text = str(trajectory[final_index].get("content") or "")
    final_matches = FINAL_BLOCK.findall(final_text)
    if len(final_matches) != 1:
        return False, "final_answer_not_unique", {}
    final_sql = final_matches[0].strip()
    verified_match = VERIFIED_CALL.search(final_text)
    if not verified_match:
        return False, "missing_verified_call", {}
    verified_call = int(verified_match.group(1))

    calls: list[tuple[int, str, str]] = []
    for index, message in enumerate(trajectory[:final_index]):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        blocks = TOOL_BLOCK.findall(content)
        if len(blocks) != 1:
            return False, "tool_call_not_unique_in_turn", {}
        function_match = FUNCTION_NAME.search(blocks[0])
        if not function_match:
            return False, "tool_function_missing", {}
        name = function_match.group(1).strip()
        sql_match = SQL_PARAMETER.search(blocks[0])
        sql = sql_match.group(1).strip() if sql_match else ""
        calls.append((index, name, sql))
    if not calls:
        return False, "no_tool_call", {}
    if verified_call != len(calls):
        return False, "verified_call_not_last", {"declared": verified_call, "actual": len(calls)}
    call_index, tool_name, called_sql = calls[-1]
    if tool_name != "sqlite_query":
        return False, "final_tool_not_sqlite_query", {"tool": tool_name}
    if normalize_sql(called_sql) != normalize_sql(final_sql):
        return False, "final_sql_not_exact_last_call", {}
    observation_index = call_index + 1
    if observation_index >= len(trajectory):
        return False, "final_observation_missing", {}
    observation_message = trajectory[observation_index]
    if not isinstance(observation_message, Mapping) or observation_message.get("role") != "user":
        return False, "final_observation_missing", {}
    observation_text = str(observation_message.get("content") or "")
    observation_match = OBSERVATION.fullmatch(observation_text)
    if not observation_match:
        return False, "final_observation_malformed", {}
    try:
        payload = json.loads(observation_match.group(1))
    except json.JSONDecodeError:
        return False, "final_observation_not_json", {}
    if not isinstance(payload, dict):
        return False, "final_observation_not_query_object", {}
    error = payload.get("error")
    if error is not None:
        kind = str(error.get("kind") or "unknown") if isinstance(error, Mapping) else "unknown"
        return False, f"final_observation_error:{kind}", {"error": error}
    if not isinstance(payload.get("columns"), list) or not isinstance(payload.get("rows"), list):
        return False, "final_observation_missing_result_rows", {}
    return True, "accepted", {
        "verified_call": verified_call,
        "row_count": payload.get("row_count"),
        "truncated": payload.get("truncated"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    total = 0
    for line_number, row in enumerate(iter_jsonl(args.input), 1):
        total += 1
        accepted, reason, details = evaluate(row)
        reasons[reason] += 1
        metadata = row.get("metadata") or {}
        if accepted:
            copied = dict(row)
            copied_metadata = dict(metadata)
            copied_metadata["final_tool_execution_succeeded"] = True
            copied["metadata"] = copied_metadata
            kept.append(copied)
        else:
            audit.append(
                {
                    "line": line_number,
                    "record_id": row.get("id"),
                    "sample_id": metadata.get("sample_id"),
                    "model": metadata.get("model"),
                    "decoding": metadata.get("decoding"),
                    "reason": reason,
                    "details": details,
                }
            )
    kept_count = atomic_jsonl(args.output, kept)
    rejected_count = atomic_jsonl(args.audit, audit)
    summary = {
        "policy": "EX=True plus successful exact-final sqlite_query observation; any final tool error is excluded",
        "input": str(args.input),
        "input_records": total,
        "kept": kept_count,
        "rejected": rejected_count,
        "reasons": dict(sorted(reasons.items())),
        "output": str(args.output),
        "audit": str(args.audit),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
