#!/usr/bin/env python3
"""Build the second-round input from hard samples absent from the current merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def dataset_name(record: Mapping[str, Any]) -> str:
    for key in ("dataset", "data_source", "source", "dataset_name"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    question_id = str(record.get("question_id") or "")
    return question_id.split("_", 1)[0] if "_" in question_id else "unknown"


def distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    difficulty = Counter(str(row.get("difficulty") or "unknown") for row in records)
    datasets = Counter(dataset_name(row) for row in records)
    cross = Counter(
        (dataset_name(row), str(row.get("difficulty") or "unknown"))
        for row in records
    )
    return {
        "difficulty": dict(sorted(difficulty.items())),
        "dataset": dict(sorted(datasets.items())),
        "dataset_by_difficulty": {
            dataset: {
                level: cross[(dataset, level)]
                for level in sorted({key[1] for key in cross if key[0] == dataset})
            }
            for dataset in sorted({key[0] for key in cross})
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-input", type=Path, required=True)
    parser.add_argument("--base-merged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hard_rows = json.loads(args.hard_input.read_text(encoding="utf-8"))
    if not isinstance(hard_rows, list) or not all(isinstance(row, dict) for row in hard_rows):
        raise ValueError("hard input must be a JSON list of objects")

    merged_rows = list(iter_jsonl(args.base_merged))
    merged_indexes: set[int] = set()
    merged_ids: set[str] = set()
    for row_number, row in enumerate(merged_rows, 1):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"base merged row {row_number} has no metadata")
        source_index = metadata.get("source_index")
        sample_id = str(metadata.get("sample_id") or "")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"base merged row {row_number} has invalid source_index")
        if not 0 <= source_index < len(hard_rows):
            raise ValueError(f"base merged row {row_number} source_index is out of range")
        if source_index in merged_indexes or not sample_id or sample_id in merged_ids:
            raise ValueError(f"duplicate/missing base identity at row {row_number}")
        merged_indexes.add(source_index)
        merged_ids.add(sample_id)

    recovery_rows: list[dict[str, Any]] = []
    for source_index, record in enumerate(hard_rows):
        if source_index in merged_indexes:
            continue
        copied = dict(record)
        copied["__source_index"] = source_index
        recovery_rows.append(copied)

    if len(recovery_rows) + len(merged_indexes) != len(hard_rows):
        raise RuntimeError("partition count mismatch")
    atomic_json(args.out, recovery_rows)
    manifest = {
        "policy": "original hard samples whose source_index is absent from the current merged SFT",
        "hard_input": str(args.hard_input),
        "hard_input_sha256": sha256_file(args.hard_input),
        "base_merged": str(args.base_merged),
        "base_merged_sha256": sha256_file(args.base_merged),
        "hard_samples": len(hard_rows),
        "base_merged_samples": len(merged_indexes),
        "recovery_samples": len(recovery_rows),
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
        "recovery_distributions": distributions(recovery_rows),
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
