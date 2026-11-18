#!/usr/bin/env python3
"""Combine a validated recovery merge with the existing validated SFT merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def identity(row: Mapping[str, Any], source: Path, line_number: int) -> tuple[str, int]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}:{line_number} lacks metadata")
    sample_id = str(metadata.get("sample_id") or "").strip()
    source_index = metadata.get("source_index")
    if not sample_id or isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError(f"{source}:{line_number} has invalid sample identity")
    if metadata.get("ex") is not True:
        raise ValueError(f"{source}:{line_number} is not EX-correct")
    return sample_id, source_index


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_base_hash = str(input_manifest.get("base_merged_sha256") or "")
    actual_base_hash = sha256_file(args.base)
    if expected_base_hash != actual_base_hash:
        raise RuntimeError(
            "current base merge differs from the one used to construct the recovery input; "
            "run prepare again into a fresh recovery directory"
        )

    base_rows = list(iter_jsonl(args.base))
    recovery_rows = list(iter_jsonl(args.recovery))
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    combined: list[dict[str, Any]] = []
    for source, rows in ((args.base, base_rows), (args.recovery, recovery_rows)):
        for line_number, row in enumerate(rows, 1):
            sample_id, source_index = identity(row, source, line_number)
            if sample_id in seen_ids or source_index in seen_indexes:
                raise ValueError(
                    f"overlap/duplicate while combining {source}:{line_number}: "
                    f"sample_id={sample_id} source_index={source_index}"
                )
            seen_ids.add(sample_id)
            seen_indexes.add(source_index)
            combined.append(row)

    combined.sort(key=lambda row: int(row["metadata"]["source_index"]))
    atomic_jsonl(args.out, combined)
    summary = {
        "policy": "retain every existing merged sample and add one best EX-correct recovery trajectory per previously unmerged sample",
        "base": str(args.base),
        "base_sha256": actual_base_hash,
        "base_records": len(base_rows),
        "recovery": str(args.recovery),
        "recovery_sha256": sha256_file(args.recovery),
        "recovery_records": len(recovery_rows),
        "combined_records": len(combined),
        "unique_sample_ids": len(seen_ids),
        "unique_source_indexes": len(seen_indexes),
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
    }
    atomic_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
