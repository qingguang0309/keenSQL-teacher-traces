#!/usr/bin/env python3
"""Export the current non-easy teacher trajectories from SFT v2 Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

ROOT = Path("/data/keenSQL")
DEFAULT_TRAIN = ROOT / "data/sft_data/sft_v3_train.parquet"
DEFAULT_VAL = ROOT / "data/sft_data/sft_v3_val.parquet"
DEFAULT_OUTPUT = ROOT / "dataset/sft_data/easy_teacher_traces/existing_teacher"


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    all_variants: Counter[str] = Counter()
    for split, path in (("train", args.train), ("val", args.val)):
        for row in pq.read_table(path).to_pylist():
            all_variants.update([str(row.get("data_variant"))])
            if row.get("data_variant") == "easy":
                continue
            exported = dict(row)
            exported["sft_split"] = split
            rows.append(exported)
            split_rows[split].append(exported)

    question_ids = [str(row.get("question_id") or "") for row in rows]
    if any(not qid for qid in question_ids):
        raise ValueError("teacher trajectory with empty question_id")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("duplicate question_id in current teacher trajectories")
    unexpected = sorted(
        {str(row.get("data_variant")) for row in rows if row.get("data_variant") != "hard_privileged"}
    )
    if unexpected:
        raise ValueError(f"unexpected non-easy data variants: {unexpected}")

    rows.sort(key=lambda row: (str(row["sft_split"]), str(row["question_id"])))
    for values in split_rows.values():
        values.sort(key=lambda row: str(row["question_id"]))

    all_path = args.output_dir / "current_teacher_1781.jsonl"
    train_path = args.output_dir / "current_teacher_train.jsonl"
    val_path = args.output_dir / "current_teacher_val.jsonl"
    all_count = atomic_jsonl(all_path, rows)
    train_count = atomic_jsonl(train_path, split_rows["train"])
    val_count = atomic_jsonl(val_path, split_rows["val"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "current sft_v2 train+val rows where data_variant is not easy",
        "train_parquet": str(args.train),
        "val_parquet": str(args.val),
        "source_sha256": {
            "train": sha256_file(args.train),
            "val": sha256_file(args.val),
        },
        "records": all_count,
        "unique_question_ids": len(set(question_ids)),
        "split_counts": {"train": train_count, "val": val_count},
        "data_variant_counts": dict(Counter(str(row.get("data_variant")) for row in rows)),
        "teacher_model_counts": dict(Counter(str(row.get("teacher_model")) for row in rows)),
        "decoding_counts": dict(Counter(str(row.get("decoding")) for row in rows)),
        "source_counts": dict(Counter(str(row.get("source")) for row in rows)),
        "difficulty_counts": dict(Counter(str(row.get("difficulty")) for row in rows)),
        "verification_mode_counts": dict(
            Counter(str(row.get("verification_mode")) for row in rows)
        ),
        "all_output": str(all_path),
        "all_output_sha256": sha256_file(all_path),
        "train_output": str(train_path),
        "train_output_sha256": sha256_file(train_path),
        "val_output": str(val_path),
        "val_output_sha256": sha256_file(val_path),
        "full_current_variant_counts": dict(all_variants),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
