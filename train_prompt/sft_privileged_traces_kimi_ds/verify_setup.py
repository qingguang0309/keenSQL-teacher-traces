#!/usr/bin/env python3
"""Validate the isolated Kimi/DeepSeek privileged-trace setup without API calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/data/keenSQL")
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "dataset/sft_data/privileged_traces/other_teacher_input/remaining_hard_1382.json"
DB_ROOT = ROOT / "dataset/all_dataset/databases"
SCHEMA_CACHE = ROOT / "dataset/m_schema/cache"
V4 = ROOT / "train_prompt/retoolsql_prompt/v4"
PROMPT = HERE / "sft_teacher_prompt_v1.txt"
EXPECTED_PROMPT_SHA256 = "431146a9ac3e73602ee1c0c69b79e92bca332d5b5aecd529d05e40abb9f68a67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required = [
    INPUT,
    DB_ROOT,
    SCHEMA_CACHE,
    V4 / "system_prompt.txt",
    V4 / "user_template.txt",
    V4 / "fewshot.json",
    PROMPT,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("missing required paths: " + ", ".join(missing))
prompt_hash = sha256(PROMPT)
if prompt_hash != EXPECTED_PROMPT_SHA256:
    raise RuntimeError(
        f"first-round teacher prompt hash mismatch: {prompt_hash}"
    )

records = json.loads(INPUT.read_text(encoding="utf-8"))
if not isinstance(records, list) or len(records) != 1382:
    raise ValueError(f"expected 1382 input rows, got {type(records).__name__}/{len(records)}")
required_fields = {"question_id", "db_id", "question", "SQL", "grading_method"}
bad = [i for i, row in enumerate(records) if not required_fields.issubset(row)]
if bad:
    raise ValueError(f"records missing required fields: {bad[:20]}")

db_ids = sorted({str(row["db_id"]) for row in records})
missing_db = []
missing_schema = []
for db_id in db_ids:
    db_dir = DB_ROOT / db_id
    expected = db_dir / f"{db_id}.sqlite"
    candidates = list(db_dir.glob("*.sqlite")) if db_dir.is_dir() else []
    if not expected.is_file() and len(candidates) != 1:
        missing_db.append(db_id)
    if not (SCHEMA_CACHE / f"{db_id}.txt").is_file():
        missing_schema.append(db_id)
if missing_db or missing_schema:
    raise FileNotFoundError(
        f"missing databases={missing_db}; missing schema cache={missing_schema}"
    )

summary = {
    "input": str(INPUT),
    "records": len(records),
    "unique_question_ids": len({str(row["question_id"]) for row in records}),
    "databases": len(db_ids),
    "difficulty": dict(Counter(str(row.get("difficulty")) for row in records)),
    "teacher_prompt": str(PROMPT),
    "teacher_prompt_sha256": prompt_hash,
    "missing_databases": missing_db,
    "missing_schema_cache": missing_schema,
    "ok": True,
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
