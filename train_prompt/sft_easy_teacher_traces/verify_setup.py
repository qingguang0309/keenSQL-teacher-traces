#!/usr/bin/env python3
"""Validate the prepared easy inputs, assets, databases, and chunk coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/keenSQL")
DEFAULT_INPUT_ROOT = ROOT / "dataset/sft_data/easy_teacher_traces/inputs"
DB_ROOT = ROOT / "dataset/all_dataset/databases"
SCHEMA_CACHE = ROOT / "dataset/m_schema/cache"
V4 = ROOT / "train_prompt/retoolsql_prompt/v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()
    manifest_path = args.input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_assets = [
        V4 / "system_prompt.txt",
        V4 / "fewshot.json",
        V4 / "user_template.txt",
        V4 / "sft_teacher_prompt.txt",
    ]
    missing_assets = [str(path) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError(f"missing v4 prompt assets: {missing_assets}")

    full_path = Path(manifest["all_input"])
    if sha256_file(full_path) != manifest["all_input_sha256"]:
        raise RuntimeError(f"input hash mismatch: {full_path}")
    full_rows = json.loads(full_path.read_text(encoding="utf-8"))
    full_ids = [str(row["question_id"]) for row in full_rows]
    if len(full_ids) != len(set(full_ids)):
        raise RuntimeError("duplicate question_id in full input")

    chunk_ids: list[str] = []
    chunk_rows = 0
    db_ids: set[str] = set()
    for entry in manifest["chunk_records"]:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"chunk hash mismatch: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != entry["records"]:
            raise RuntimeError(f"chunk count mismatch: {path}")
        chunk_rows += len(rows)
        chunk_ids.extend(str(row["question_id"]) for row in rows)
        db_ids.update(str(row["db_id"]) for row in rows)
    if chunk_rows != len(full_rows) or set(chunk_ids) != set(full_ids):
        raise RuntimeError("chunk union does not equal full easy input")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("question_id overlap between chunks")

    missing_db = []
    missing_schema = []
    for db_id in sorted(db_ids):
        expected = DB_ROOT / db_id / f"{db_id}.sqlite"
        candidates = list((DB_ROOT / db_id).glob("*.sqlite"))
        if not expected.is_file() and len(candidates) != 1:
            missing_db.append(db_id)
        if not (SCHEMA_CACHE / f"{db_id}.txt").is_file():
            missing_schema.append(db_id)
    if missing_db or missing_schema:
        raise FileNotFoundError(
            f"missing/ambiguous databases={missing_db[:20]} schemas={missing_schema[:20]}"
        )
    print(
        json.dumps(
            {
                "ok": True,
                "records": len(full_rows),
                "chunks": len(manifest["chunk_records"]),
                "databases": len(db_ids),
                "prompt_assets": {path.name: sha256_file(path) for path in required_assets},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
