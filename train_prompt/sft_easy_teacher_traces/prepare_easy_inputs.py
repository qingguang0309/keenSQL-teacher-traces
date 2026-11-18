#!/usr/bin/env python3
"""Prepare every original sample not already covered by the frozen teacher pool.

The current 7,508 easy rows remain an immutable prefix. Remaining original
rows are appended, so per-chunk source indexes (and therefore resume IDs) for
already generated attempts do not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

ROOT = Path("/data/keenSQL")
DEFAULT_TRAIN = ROOT / "data/sft_data/sft_v3_train.parquet"
DEFAULT_VAL = ROOT / "data/sft_data/sft_v3_val.parquet"
DEFAULT_EASY = ROOT / "dataset/sft_data/easy/sft_easy.jsonl"
DEFAULT_ALL = ROOT / "dataset/all_dataset/train.json"
DEFAULT_EXISTING_TEACHER = (
    ROOT
    / "dataset/sft_data/easy_teacher_traces/existing_teacher/current_teacher_1781.jsonl"
)
DEFAULT_OUTPUT = ROOT / "dataset/sft_data/easy_teacher_traces/inputs"
DEFAULT_MODEL_ROOT = ROOT / "dataset/sft_data/easy_teacher_traces/models"
DEFAULT_EXCLUDED = DEFAULT_OUTPUT / "excluded_unfinished_458.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sql(sql: str) -> str:
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql or "", flags=re.S)
    return re.sub(r"\s+", " ", text).strip().rstrip(";").strip().casefold()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield row


def atomic_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )
        handle.write("\n")
    temporary.replace(path)


def load_current_easy(train: Path, val: Path) -> list[dict[str, Any]]:
    columns = [
        "id",
        "question_id",
        "question",
        "source",
        "db_id",
        "gold_sql",
        "grading_method",
        "difficulty",
        "data_variant",
        "teacher_model",
        "decoding",
    ]
    rows: list[dict[str, Any]] = []
    for split, path in (("train", train), ("val", val)):
        for row in pq.read_table(path, columns=columns).to_pylist():
            if row.get("data_variant") != "easy":
                continue
            row["sft_split"] = split
            rows.append(row)
    return rows


def unique_by_qid(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            raise ValueError(f"{label} row has empty question_id")
        if qid in result:
            raise ValueError(f"duplicate {label} question_id: {qid}")
        result[qid] = row
    return result


def source_for_question_id(qid: str) -> str:
    if qid.startswith("bird_"):
        return "bird_train"
    if qid.startswith("omni_"):
        return "omni"
    if qid.startswith("spider_"):
        return "spider"
    if qid.isdigit():
        return "bird_verified"
    raise ValueError(f"cannot infer source for question_id={qid!r}")


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


def validate_resume_rows(raw_path: Path, rows: list[dict[str, Any]]) -> int:
    if not raw_path.is_file():
        return 0
    checked = 0
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                attempt = json.loads(line)
            except json.JSONDecodeError:
                # A live worker may be in the middle of its final append. Only
                # an unterminated final line is safe to ignore.
                if not line.endswith("\n"):
                    continue
                raise ValueError(f"invalid JSON at {raw_path}:{line_number}")
            index = int(attempt.get("source_index", -1))
            if index < 0 or index >= len(rows):
                raise ValueError(f"resume source_index out of range at {raw_path}:{line_number}")
            row = rows[index]
            expected_sample = sample_id(index, row)
            if str(attempt.get("question_id")) != str(row.get("question_id")):
                raise ValueError(f"resume question_id mismatch at {raw_path}:{line_number}")
            if str(attempt.get("sample_id")) != expected_sample:
                raise ValueError(f"resume sample_id mismatch at {raw_path}:{line_number}")
            checked += 1
    return checked


def reconcile_run_configs(
    model_root: Path,
    output_dir: Path,
    chunks_by_name: Mapping[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    updated = 0
    validated_attempts = 0
    if not model_root.is_dir():
        return updated, validated_attempts
    for config_path in sorted(model_root.glob("*/*/chunk_*/config.json")):
        chunk = config_path.parent.name
        if chunk not in chunks_by_name:
            continue
        rows = chunks_by_name[chunk]
        configured = json.loads(config_path.read_text(encoding="utf-8"))
        input_path = Path(str(configured.get("input") or ""))
        expected_path = output_dir / "chunks" / f"{chunk}.json"
        if input_path.resolve() != expected_path.resolve():
            continue
        old_count = int(configured.get("selected_records", 0))
        if old_count > len(rows):
            raise ValueError(f"refusing to shrink configured input: {config_path}")
        validated_attempts += validate_resume_rows(
            config_path.parent / "raw_attempts.jsonl", rows
        )
        new_hash = sha256_file(expected_path)
        if configured.get("input_sha256") != new_hash or old_count != len(rows):
            configured["input_sha256"] = new_hash
            configured["selected_records"] = len(rows)
            atomic_json(config_path, configured)
            updated += 1
    return updated, validated_attempts


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunks < 1:
        raise ValueError("--chunks must be positive")

    # Once unfinished rows have deliberately been removed, a generic prepare
    # must not silently restore them or repartition compacted resume indexes.
    finalized_manifest = args.output_dir / "manifest.json"
    finalized_excluded = args.output_dir / DEFAULT_EXCLUDED.name
    if finalized_manifest.is_file() and finalized_excluded.is_file():
        existing = json.loads(finalized_manifest.read_text(encoding="utf-8"))
        all_input = Path(existing["all_input"])
        if sha256_file(all_input) != existing.get("all_input_sha256"):
            raise RuntimeError("finalized input hash mismatch; refusing to re-prepare")
        for entry in existing.get("chunk_records") or []:
            path = Path(entry["path"])
            if sha256_file(path) != entry.get("sha256"):
                raise RuntimeError(f"finalized chunk hash mismatch: {path}")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return existing

    current = load_current_easy(args.train, args.val)
    current_by_qid = unique_by_qid(current, "current easy")
    bad_models = sorted(
        {
            str(row.get("teacher_model"))
            for row in current
            if row.get("teacher_model") != "Qwen3.5-9B"
        }
    )
    bad_decoding = sorted(
        {
            str(row.get("decoding"))
            for row in current
            if row.get("decoding") != "pass16_temp1"
        }
    )
    if bad_models or bad_decoding:
        raise ValueError(
            "current easy subset is not the expected Qwen3.5 pass@16 pool: "
            f"models={bad_models} decoding={bad_decoding}"
        )

    easy_rows = list(iter_jsonl(args.easy_jsonl))
    easy_by_qid = unique_by_qid(easy_rows, "source easy")
    missing_easy = sorted(set(current_by_qid) - set(easy_by_qid))
    if missing_easy:
        raise ValueError(
            f"{len(missing_easy)} current easy IDs missing from source easy JSONL: "
            f"{missing_easy[:20]}"
        )

    current_records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for qid, current_row in current_by_qid.items():
        original = easy_by_qid[qid]
        checks = {
            "db_id": str(original.get("db_id")) == str(current_row.get("db_id")),
            "question": str(original.get("question")) == str(current_row.get("question")),
            "gold_sql": normalize_sql(str(original.get("gold_sql") or ""))
            == normalize_sql(str(current_row.get("gold_sql") or "")),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            mismatches.append(f"{qid}:{','.join(failed)}")
            continue
        current_records.append(
            {
                "question_id": qid,
                "question": original.get("question"),
                "evidence": original.get("evidence") or "",
                "db_id": original.get("db_id"),
                "SQL": original.get("gold_sql"),
                "grading_method": original.get("grading_method") or "multiset",
                "difficulty": original.get("difficulty"),
                "source": original.get("source"),
                "sft_split": current_row["sft_split"],
                "current_easy_id": current_row.get("id"),
                "source_easy_id": original.get("id"),
            }
        )
    if mismatches:
        raise ValueError(
            f"source/current field mismatch ({len(mismatches)}): {mismatches[:20]}"
        )
    current_records.sort(key=lambda row: (str(row["source"]), str(row["question_id"])))

    all_rows = json.loads(args.all_dataset.read_text(encoding="utf-8"))
    if not isinstance(all_rows, list):
        raise ValueError(f"expected JSON array: {args.all_dataset}")
    all_by_qid = unique_by_qid(all_rows, "all-dataset")
    teacher_rows = list(iter_jsonl(args.existing_teacher))
    teacher_by_qid = unique_by_qid(teacher_rows, "existing teacher")
    current_ids = {str(row["question_id"]) for row in current_records}
    teacher_ids = set(teacher_by_qid)
    all_ids = set(all_by_qid)
    if current_ids & teacher_ids:
        raise ValueError("current easy and existing teacher pools overlap")
    if not current_ids <= all_ids or not teacher_ids <= all_ids:
        raise ValueError("current easy or existing teacher IDs fall outside all-dataset")

    remaining_ids = all_ids - current_ids - teacher_ids
    remaining_records: list[dict[str, Any]] = []
    for original in all_rows:
        qid = str(original.get("question_id") or "").strip()
        if qid not in remaining_ids:
            continue
        source_easy = easy_by_qid.get(qid) or {}
        remaining_records.append(
            {
                "question_id": qid,
                "question": original.get("question"),
                "evidence": original.get("evidence") or "",
                "db_id": original.get("db_id"),
                "SQL": original.get("SQL"),
                "grading_method": original.get("grading_method") or "multiset",
                "difficulty": original.get("difficulty"),
                "source": source_for_question_id(qid),
                "sft_split": "unassigned",
                "current_easy_id": None,
                "source_easy_id": source_easy.get("id"),
                "input_variant": "remaining_unprocessed",
            }
        )

    records = current_records + remaining_records
    if len(records) + len(teacher_rows) != len(all_rows):
        raise ValueError(
            "pool accounting mismatch: "
            f"regeneration={len(records)} teacher={len(teacher_rows)} all={len(all_rows)}"
        )
    if len({str(row["question_id"]) for row in records}) != len(records):
        raise ValueError("duplicate question_id in regeneration pool")

    identity_digest = sha256_text(
        "\n".join(
            f"{row['question_id']}\t{row['db_id']}\t{normalize_sql(str(row['SQL']))}"
            for row in records
        )
    )
    source_hashes = {
        "train": sha256_file(args.train),
        "val": sha256_file(args.val),
        "easy_jsonl": sha256_file(args.easy_jsonl),
        "all_dataset": sha256_file(args.all_dataset),
        "existing_teacher": sha256_file(args.existing_teacher),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("chunks") == args.chunks
            and existing.get("selected_records") == len(records)
            and existing.get("identity_sha256") == identity_digest
            and existing.get("source_sha256") == source_hashes
        ):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing
        raise RuntimeError(
            f"existing manifest differs: {manifest_path}; pass --force for a "
            "validated append-only expansion"
        )

    # Old per-chunk rows must remain exact prefixes. This keeps source_index-
    # based sample IDs stable for every attempt already on disk.
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_full_path = Path(old_manifest["all_input"])
        old_records = json.loads(old_full_path.read_text(encoding="utf-8"))
        if old_records[: len(current_records)] != current_records:
            raise ValueError("existing 7,508-row current-easy prefix changed")
        for index in range(args.chunks):
            old_path = args.output_dir / "chunks" / f"chunk_{index:03d}.json"
            if old_path.is_file():
                old_chunk = json.loads(old_path.read_text(encoding="utf-8"))
                expected_prefix = current_records[index:: args.chunks]
                if old_chunk[: len(expected_prefix)] != expected_prefix:
                    raise ValueError(f"existing chunk prefix changed: {old_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "easy_current.json"
    atomic_json(all_path, records, compact=True)

    chunk_entries: list[dict[str, Any]] = []
    chunks_by_name: dict[str, list[dict[str, Any]]] = {}
    for index in range(args.chunks):
        chunk_name = f"chunk_{index:03d}"
        chunk_rows = records[index:: args.chunks]
        chunk_path = chunks_dir / f"{chunk_name}.json"
        atomic_json(chunk_path, chunk_rows, compact=True)
        chunks_by_name[chunk_name] = chunk_rows
        current_prefix_count = len(current_records[index:: args.chunks])
        chunk_entries.append(
            {
                "chunk": chunk_name,
                "path": str(chunk_path),
                "records": len(chunk_rows),
                "current_easy_prefix_records": current_prefix_count,
                "remaining_records": len(chunk_rows) - current_prefix_count,
                "sha256": sha256_file(chunk_path),
                "first_question_id": chunk_rows[0]["question_id"] if chunk_rows else None,
                "last_question_id": chunk_rows[-1]["question_id"] if chunk_rows else None,
            }
        )

    configs_updated, resume_attempts_validated = reconcile_run_configs(
        args.model_root, args.output_dir, chunks_by_name
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "all original rows excluding the frozen existing-teacher pool",
        "expected_origin": "Qwen3.5-9B pass16_temp1 filtering pool",
        "all_dataset": str(args.all_dataset),
        "all_dataset_records": len(all_rows),
        "existing_teacher": str(args.existing_teacher),
        "existing_teacher_records": len(teacher_rows),
        "current_easy_records": len(current_records),
        "remaining_unprocessed_records": len(remaining_records),
        "selected_records": len(records),
        "unique_question_ids": len({str(row["question_id"]) for row in records}),
        "identity_sha256": identity_digest,
        "train_parquet": str(args.train),
        "val_parquet": str(args.val),
        "source_easy_jsonl": str(args.easy_jsonl),
        "source_easy_records": len(easy_rows),
        "source_sha256": source_hashes,
        "all_input": str(all_path),
        "all_input_sha256": sha256_file(all_path),
        "append_only_resume": {
            "current_easy_prefix_preserved": True,
            "runtime_configs_updated": configs_updated,
            "existing_attempts_validated": resume_attempts_validated,
        },
        "chunks": args.chunks,
        "chunk_records": chunk_entries,
        "split_counts": dict(Counter(str(row["sft_split"]) for row in records)),
        "input_variant_counts": {
            "current_easy": len(current_records),
            "remaining_unprocessed": len(remaining_records),
        },
        "source_counts": dict(Counter(str(row["source"]) for row in records)),
        "difficulty_counts": dict(Counter(str(row["difficulty"]) for row in records)),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--easy-jsonl", type=Path, default=DEFAULT_EASY)
    parser.add_argument("--all-dataset", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--existing-teacher", type=Path, default=DEFAULT_EXISTING_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
