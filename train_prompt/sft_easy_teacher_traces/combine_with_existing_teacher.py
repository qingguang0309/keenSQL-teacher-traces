#!/usr/bin/env python3
"""Combine regenerated non-teacher traces with the frozen teacher pool."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

FINAL_SQL = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.I | re.S)
DEFAULT_MANIFEST = Path(
    "/data/keenSQL/dataset/sft_data/easy_teacher_traces/inputs/manifest.json"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
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


def final_sql(messages: Any, prompt_messages: int) -> str:
    if not isinstance(messages, list):
        return ""
    found = ""
    for item in messages[prompt_messages:]:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        match = FINAL_SQL.search(str(item.get("content") or ""))
        if match:
            found = match.group(1).strip()
    return found


def canonical_easy(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt_messages = int(row.get("prompt_messages") or 0)
    db_id = str(row.get("db_id") or metadata.get("db_id") or "")
    return {
        "messages": row.get("messages"),
        "prompt_messages": prompt_messages,
        "enable_thinking": False,
        "id": row.get("id"),
        "question_id": row.get("question_id") or metadata.get("question_id"),
        "question": row.get("question"),
        "source": row.get("source"),
        "db_id": db_id,
        "db_path": f"/data/keenSQL/dataset/all_dataset/databases/{db_id}/{db_id}.sqlite",
        "gold_sql": row.get("gold_sql"),
        "grading_method": row.get("grading_method") or metadata.get("grading_method"),
        "difficulty": row.get("difficulty"),
        "data_variant": "easy_teacher",
        "teacher_model": metadata.get("model"),
        "decoding": metadata.get("decoding"),
        "tool_calls": metadata.get("tool_calls"),
        "assistant_turns": metadata.get("assistant_turns"),
        "final_sql": final_sql(row.get("messages"), prompt_messages),
        "prompt_hash": row.get("prompt_hash"),
        "prompt_version": row.get("prompt_version"),
        "teacher_prompt_sha256": row.get("teacher_prompt_sha256"),
        "verification_mode": "strict",
        "repair_origin": None,
        "repair_action": None,
        "sft_split": row.get("sft_split") or metadata.get("sft_split"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--easy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    existing = list(iter_jsonl(args.existing))
    easy_raw = list(iter_jsonl(args.easy))
    easy = [canonical_easy(row) for row in easy_raw]
    existing_ids = [str(row.get("question_id") or "") for row in existing]
    easy_ids = [str(row.get("question_id") or "") for row in easy]
    if any(not value for value in existing_ids + easy_ids):
        raise ValueError("empty question_id in combine input")
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("duplicate question_id in existing teacher input")
    if len(easy_ids) != len(set(easy_ids)):
        raise ValueError("duplicate question_id in regenerated easy input")
    overlap = sorted(set(existing_ids) & set(easy_ids))
    if overlap:
        raise ValueError(
            f"existing/easy question_id overlap ({len(overlap)}); refusing silent replacement: {overlap[:20]}"
        )

    combined = existing + easy
    combined.sort(
        key=lambda row: (
            0 if str(row.get("sft_split")) == "train" else 1,
            str(row.get("question_id")),
        )
    )
    count = atomic_jsonl(args.out, combined)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pool_records = int(manifest["selected_records"])
    summary = {
        "existing_teacher_records": len(existing),
        "regeneration_pool_records": pool_records,
        "regenerated_records": len(easy),
        "combined_records": count,
        "unique_question_ids": len({str(row.get("question_id")) for row in combined}),
        "question_id_overlap": 0,
        "expected_max_if_all_regeneration_rows_pass": len(existing) + pool_records,
        "missing_regeneration_rows": max(0, pool_records - len(easy)),
        "split_counts": dict(Counter(str(row.get("sft_split")) for row in combined)),
        "data_variant_counts": dict(Counter(str(row.get("data_variant")) for row in combined)),
        "teacher_model_counts": dict(Counter(str(row.get("teacher_model")) for row in combined)),
        "decoding_counts": dict(Counter(str(row.get("decoding")) for row in combined)),
        "existing_input": str(args.existing),
        "easy_input": str(args.easy),
        "output": str(args.out),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
