#!/usr/bin/env python3
"""Filter validated SFT trajectories by their actual number of tool calls."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.I | re.S)


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


def actual_tool_calls(row: Mapping[str, Any]) -> int:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages_missing")
    try:
        prompt_messages = int(row.get("prompt_messages", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_prompt_messages") from exc
    if not 0 <= prompt_messages < len(messages):
        raise ValueError("invalid_prompt_messages")
    count = 0
    for message in messages[prompt_messages:]:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            count += len(TOOL_BLOCK.findall(str(message.get("content") or "")))
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--min-tool-calls", type=int, default=2)
    args = parser.parse_args()
    if args.min_tool_calls < 1:
        raise ValueError("--min-tool-calls must be positive")

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    distribution: Counter[int] = Counter()
    metadata_mismatches = 0
    total = 0
    for line_number, row in enumerate(iter_jsonl(args.input), 1):
        total += 1
        try:
            calls = actual_tool_calls(row)
        except ValueError as exc:
            rejected.append(
                {
                    "line": line_number,
                    "record_id": row.get("id"),
                    "sample_id": (row.get("metadata") or {}).get("sample_id"),
                    "reason": str(exc),
                }
            )
            continue
        distribution[calls] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("tool_calls") != calls:
            metadata_mismatches += 1
        if calls < args.min_tool_calls:
            rejected.append(
                {
                    "line": line_number,
                    "record_id": row.get("id"),
                    "sample_id": metadata.get("sample_id"),
                    "model": metadata.get("model"),
                    "decoding": metadata.get("decoding"),
                    "reason": f"tool_calls_below_{args.min_tool_calls}",
                    "actual_tool_calls": calls,
                }
            )
            continue
        copied = dict(row)
        copied_metadata = dict(metadata)
        copied_metadata["minimum_tool_calls_variant"] = args.min_tool_calls
        copied["metadata"] = copied_metadata
        kept.append(copied)

    kept_count = atomic_jsonl(args.output, kept)
    rejected_count = atomic_jsonl(args.audit, rejected)
    summary = {
        "policy": f"retain only already validated trajectories with at least {args.min_tool_calls} actual tool calls",
        "input": str(args.input),
        "input_records": total,
        "kept": kept_count,
        "rejected": rejected_count,
        "tool_call_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "metadata_tool_call_mismatches": metadata_mismatches,
        "output": str(args.output),
        "audit": str(args.audit),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
