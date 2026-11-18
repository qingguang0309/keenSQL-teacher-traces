#!/usr/bin/env python3
"""Re-grade every saved run with the exact grader and write the comparison table.

The grader used to special-case a single-cell numeric result with a 1% relative
tolerance. That shortcut is gone, so this script re-derives the EX of every saved
transcript from the stored ``gold_rows``/``pred_rows`` without paying for new
model calls. Output is the table consumed by the evaluation documents.

Usage::

    /data1/ReViSQL/.venv/bin/python retoolsql_prompt/regrade_exact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, "/data1/ReViSQL/local_rlvr")

from sql_grading import grade_results  # noqa: E402

RUNS = [
    "v4_qwen35_9b",
    "v4_qwen36plus",
    "v4abl_no_lenient_tag",
    "v4abl_no_verified_call",
    "v3opt_qwen35_9b",
    "v3opt_qwen36plus",
    "v3opt_qwen38max",
    "v3abl_ablate_verify_rule",
    "v3abl_ablate_shape_rules",
    "v3abl_ablate_peek_triggers",
    "v3abl_ablate_schema_filter_doc",
    "v3_final_qwen35_9b",
    "v3_final_qwen36plus",
    "v3_final_qwen38max",
    "v3_challenge_qwen35_9b",
    "v3_challenge_qwen36plus",
    "v3_challenge_qwen38max",
    "qwen35_9b_challenge_json",
    "qwen35_challenge_xml",
]


def exact_match(record: dict) -> bool:
    gold = record.get("gold_rows")
    predicted = record.get("pred_rows")
    if gold is None or predicted is None:
        # An episode that never produced final SQL cannot match.
        return bool(record.get("ex")) and gold is not None and predicted is not None
    return grade_results(gold, predicted, "multiset")


def main() -> int:
    rows = []
    for name in RUNS:
        path = ROOT / "runs" / f"{name}.json"
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8")).get("records") or []
        loose = sum(1 for record in records if record.get("ex"))
        exact = sum(1 for record in records if exact_match(record))
        false_positives = [
            record["db_id"]
            for record in records
            if record.get("ex") and not exact_match(record)
        ]
        rows.append((name, loose, exact, false_positives))

    width = max(len(name) for name, *_ in rows)
    lines = [
        f"{name:<{width}}  loose={loose:3d} exact={exact:3d} "
        f"false_pos={len(fps):2d} {fps}"
        for name, loose, exact, fps in rows
    ]
    text = "\n".join(lines) + "\n"
    out = ROOT / "runs" / "exact_vs_loose.txt"
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
