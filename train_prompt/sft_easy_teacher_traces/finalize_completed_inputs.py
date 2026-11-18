#!/usr/bin/env python3
"""Finalize the easy-teacher input pool to attempts already persisted on disk."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path("/data/keenSQL")
DATA_ROOT = ROOT / "dataset/sft_data/easy_teacher_traces"
INPUT_ROOT = DATA_ROOT / "inputs"
CHUNK_ROOT = INPUT_ROOT / "chunks"
MODEL_ROOT = DATA_ROOT / "models/qwen3.6-plus/greedy"
EXCLUDED_PATH = INPUT_ROOT / "excluded_unfinished_458.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            yield row


def normalize_sql(sql: str) -> str:
    import re

    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql or "", flags=re.S)
    return re.sub(r"\s+", " ", text).strip().rstrip(";").strip().casefold()


def latest_attempts(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        attempt_id = str(row.get("attempt_id") or "")
        if not attempt_id:
            raise ValueError(f"missing attempt_id in {path}")
        latest[attempt_id] = row
    return latest


def main() -> int:
    manifest_path = INPUT_ROOT / "manifest.json"
    all_path = INPUT_ROOT / "easy_current.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = json.loads(all_path.read_text(encoding="utf-8"))
    all_by_qid = {str(row["question_id"]): row for row in all_rows}
    if len(all_by_qid) != len(all_rows):
        raise ValueError("duplicate question_id in easy_current.json")

    kept_qids: set[str] = set()
    removed_rows: list[dict[str, Any]] = []
    chunks: dict[str, list[dict[str, Any]]] = {}
    chunk_entries: list[dict[str, Any]] = []
    attempts_checked = 0

    for chunk_path in sorted(CHUNK_ROOT.glob("chunk_*.json")):
        chunk = chunk_path.stem
        rows = json.loads(chunk_path.read_text(encoding="utf-8"))
        raw_path = MODEL_ROOT / chunk / "raw_attempts.jsonl"
        attempts = latest_attempts(raw_path)
        by_index: dict[int, dict[str, Any]] = {}
        for attempt in attempts.values():
            index = int(attempt.get("source_index", -1))
            if index < 0 or index >= len(rows):
                raise ValueError(f"source_index out of range in {raw_path}: {index}")
            if index in by_index:
                raise ValueError(f"multiple greedy attempts for source_index={index} in {raw_path}")
            expected_qid = str(rows[index].get("question_id"))
            if str(attempt.get("question_id")) != expected_qid:
                raise ValueError(f"question_id mismatch in {raw_path} index={index}")
            by_index[index] = attempt
            attempts_checked += 1

        kept: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            qid = str(row["question_id"])
            if index in by_index:
                compacted = copy.deepcopy(row)
                compacted["resume_source_index"] = index
                kept.append(compacted)
                if qid in kept_qids:
                    raise ValueError(f"question_id appears in multiple chunks: {qid}")
                kept_qids.add(qid)
            else:
                excluded = copy.deepcopy(row)
                excluded["exclusion"] = {
                    "reason": "generation_not_completed_before_provider_timeout_circuit_breaker",
                    "original_chunk": chunk,
                    "original_source_index": index,
                }
                removed_rows.append(excluded)

        chunks[chunk] = kept
        chunk_entries.append(
            {
                "chunk": chunk,
                "path": str(chunk_path),
                "records": len(kept),
                "current_easy_records": sum(
                    1 for row in kept if (row.get("input_variant") or "easy") == "easy"
                ),
                "remaining_unprocessed_records": sum(
                    1
                    for row in kept
                    if row.get("input_variant") == "remaining_unprocessed"
                ),
                "sha256": None,
                "first_question_id": kept[0]["question_id"] if kept else None,
                "last_question_id": kept[-1]["question_id"] if kept else None,
            }
        )

    kept_all = [row for row in all_rows if str(row["question_id"]) in kept_qids]
    removed_qids = {str(row["question_id"]) for row in removed_rows}
    if kept_qids & removed_qids:
        raise ValueError("kept and removed question IDs overlap")
    if kept_qids | removed_qids != set(all_by_qid):
        raise ValueError("kept and removed IDs do not reconstruct the previous input")
    if len(kept_all) != attempts_checked:
        raise ValueError("completed-attempt count does not match finalized input count")

    removed_by_chunk = Counter(row["exclusion"]["original_chunk"] for row in removed_rows)
    if removed_by_chunk != Counter({"chunk_000": 254, "chunk_003": 204}):
        raise ValueError(f"unexpected unfinished distribution: {dict(removed_by_chunk)}")
    if len(kept_all) != 9119 or len(removed_rows) != 458:
        raise ValueError(
            f"unexpected final counts: kept={len(kept_all)} removed={len(removed_rows)}"
        )

    # The excluded file is the recoverable record of what was removed.
    atomic_json(EXCLUDED_PATH, removed_rows)
    atomic_json(all_path, kept_all, compact=True)
    for entry in chunk_entries:
        chunk = entry["chunk"]
        path = Path(entry["path"])
        atomic_json(path, chunks[chunk], compact=True)
        entry["sha256"] = sha256_file(path)

        config_path = MODEL_ROOT / chunk / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["input_sha256"] = entry["sha256"]
            config["selected_records"] = entry["records"]
            atomic_json(config_path, config)

    variants = Counter(row.get("input_variant") or "easy" for row in kept_all)
    manifest.pop("append_only_resume", None)
    manifest.update(
        {
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": (
                "persisted greedy attempts after excluding the frozen existing-teacher "
                "pool and 458 unfinished inputs"
            ),
            "current_easy_records": variants["easy"],
            "remaining_unprocessed_records": variants["remaining_unprocessed"],
            "excluded_unfinished_records": len(removed_rows),
            "selected_records": len(kept_all),
            "unique_question_ids": len(kept_qids),
            "identity_sha256": sha256_text(
                "\n".join(
                    f"{row['question_id']}\t{row['db_id']}\t{normalize_sql(str(row['SQL']))}"
                    for row in kept_all
                )
            ),
            "all_input_sha256": sha256_file(all_path),
            "excluded_unfinished": str(EXCLUDED_PATH),
            "excluded_unfinished_sha256": sha256_file(EXCLUDED_PATH),
            "completed_only_finalization": {
                "raw_attempts_validated": attempts_checked,
                "removed_unfinished": len(removed_rows),
                "removed_by_chunk": dict(removed_by_chunk),
                "resume_source_index_preserved": True,
            },
            "chunk_records": chunk_entries,
            "split_counts": dict(Counter(str(row.get("sft_split")) for row in kept_all)),
            "input_variant_counts": dict(variants),
            "source_counts": dict(Counter(str(row.get("source")) for row in kept_all)),
            "difficulty_counts": dict(
                Counter(str(row.get("difficulty")) for row in kept_all)
            ),
            "excluded_counts": {
                "by_chunk": dict(removed_by_chunk),
                "by_source": dict(
                    Counter(str(row.get("source")) for row in removed_rows)
                ),
                "by_difficulty": dict(
                    Counter(str(row.get("difficulty")) for row in removed_rows)
                ),
            },
        }
    )
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "finalized_records": len(kept_all),
                "excluded_unfinished": len(removed_rows),
                "removed_by_chunk": dict(removed_by_chunk),
                "chunks": {name: len(rows) for name, rows in chunks.items()},
                "manifest": str(manifest_path),
                "excluded": str(EXCLUDED_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
