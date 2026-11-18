#!/usr/bin/env python3
"""Prepare BIRD dev (1,534 questions) as pipeline input for an EVALUATION run.

EVAL ONLY: traces generated from dev must never enter SFT or RL data. Every record carries
sft_split="dev_eval_only" and source="bird_dev"; release.py would also drop them through its
dev decontamination. The run measures how the full pipeline (generator v2 + L3 audit +
L3b) behaves on the benchmark split.

Source of truth is data/bird_dev_1534.json (the harness file used by eval_dev.py), whose
questions, db ids and gold SQL were checked against dataset/bird-dev/dev.json; this script
re-checks that every row matches dev.json index by index.

Databases and M-Schema come from the dev set, not the training set (the training set has a
different formula_1 database with the same name). Run the pipeline with
  SFT_DB_ROOT=/data/keenSQL/dataset/bird-dev/databases
  SFT_SCHEMA_CACHE=/data/keenSQL/data/mschema/bird_dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

HARNESS = C.ROOT / "data/bird_dev_1534.json"
DEV_DB_ROOT = C.ROOT / "dataset/bird-dev/databases"
DEV_SCHEMA = C.ROOT / "data/mschema/bird_dev"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--difficulties", default="simple,moderate,challenging",
                        help="comma list; questions of other difficulties are left out")
    args = parser.parse_args()
    keep = {d.strip() for d in args.difficulties.split(",") if d.strip()}
    harness = json.loads(HARNESS.read_text(encoding="utf-8"))
    dev = json.loads(C.DEV_JSON.read_text(encoding="utf-8"))
    if len(harness) != len(dev):
        raise SystemExit(f"harness {len(harness)} rows, dev.json {len(dev)} rows")
    records = []
    for index, (h, d) in enumerate(zip(harness, dev)):
        assert h["id"] == f"birddev_{index}", h["id"]
        for hk, dk in (("question", "question"), ("db_id", "db_id"), ("gold_sql", "SQL"), ("difficulty", "difficulty")):
            if str(h[hk]).strip() != str(d[dk]).strip():
                raise SystemExit(f"row {index}: {hk} differs from dev.json")
        if not (DEV_DB_ROOT / d["db_id"] / f"{d['db_id']}.sqlite").is_file():
            raise SystemExit(f"missing dev database {d['db_id']}")
        if not (DEV_SCHEMA / f"{d['db_id']}.txt").is_file():
            raise SystemExit(f"missing dev M-Schema {d['db_id']}")
        if d["difficulty"] not in keep:
            continue
        records.append({
            "question_id": h["id"],
            "question": d["question"],
            "evidence": d.get("evidence") or "",
            "db_id": d["db_id"],
            "SQL": d["SQL"],
            "grading_method": "set",  # BIRD official: set(fetchall) == set(fetchall)
            "difficulty": d["difficulty"],
            "source": "bird_dev",
            "sft_split": "dev_eval_only",
            "current_easy_id": h["id"],
            "source_easy_id": h["id"],
            "dev_index": index,
        })
    manifest = C.prepare_inputs(records, args.root, selection=f"BIRD dev 1534 [{','.join(sorted(keep))}], EVALUATION ONLY (never for training)",
                                parent_input=HARNESS, chunk_size=args.chunk_size)
    (args.root / "EVAL_ONLY_NOT_FOR_TRAINING.txt").write_text(
        "Traces under this directory come from BIRD dev. Never use them for SFT or RL.\n", encoding="utf-8")
    print(f"prepared {manifest['selected_records']} dev records in {manifest['chunks']} chunks under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
