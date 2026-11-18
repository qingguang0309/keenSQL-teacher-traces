#!/usr/bin/env python3
"""Re-grade saved run_ex.py records with the current turn parser.

The transcripts are immutable: every assistant turn and observation is already
stored. Re-parsing them costs nothing and lets a parser fix (for example a
tolerated mismatched annotation tag) be applied uniformly to all models without
paying for another round of API calls.

Only episodes whose failure was a *terminal* formatting error can be recovered.
An episode truncated mid-answer has no complete <final_answer> and stays a
failure, which is the correct outcome: the model never produced a final SQL.

Usage::

    python rescore_runs.py runs/v3_challenge_qwen35_9b.json \
        runs/v3_challenge_qwen36plus.json runs/v3_challenge_qwen38max.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "database_tools"))
sys.path.insert(0, str(ROOT / "retoolsql_prompt"))
sys.path.insert(0, "/data1/ReViSQL/local_rlvr")

import test_retoolsql_prompt as T  # noqa: E402
from run_ex import db_file_for, grade_one  # noqa: E402


def replay(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse every assistant turn again; return the final answer if reachable."""

    final = None
    for message in record.get("transcript", []):
        if message.get("role") != "assistant":
            continue
        try:
            parsed = T.parse_assistant_turn(
                message["content"],
                require_final_scratch_pad=False,
                require_tool_scratch_pad=False,
            )
        except T.ProtocolError:
            continue
        if parsed["action"] == "final_answer":
            final = parsed
    return final


def rescore(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    config = payload.get("config") or {}

    ex = 0
    recovered = 0
    recovered_correct = 0
    details: List[Dict[str, Any]] = []

    for record in records:
        if record.get("ex"):
            ex += 1
            continue
        if record.get("protocol_ok"):
            continue  # genuine execution mismatch, nothing to recover
        final = replay(record)
        if not final or not final.get("sql"):
            continue
        db_file = db_file_for(record["db_id"])
        if not db_file:
            continue
        ok, reason, _pred, _gold = grade_one(
            db_file, final["sql"], record["gold_sql"], "multiset"
        )
        recovered += 1
        if ok:
            recovered_correct += 1
            ex += 1
        details.append(
            {
                "idx": record.get("idx"),
                "db_id": record["db_id"],
                "ex": bool(ok),
                "reason": reason,
                "sql": final["sql"][:200],
                "errors": record.get("errors", []),
            }
        )

    n = len(records)
    return {
        "file": path.name,
        "model": config.get("model", ""),
        "n": n,
        "ex_strict": sum(1 for r in records if r.get("ex")),
        "recovered_final_answers": recovered,
        "recovered_correct": recovered_correct,
        "ex_after_reparse": ex,
        "ex_after_reparse_rate": ex / n if n else 0.0,
        "recovered_details": details,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    results = [rescore(Path(f)) for f in args.files]

    print("| Model | N | EX (as run) | Final answers recovered | Of those correct "
          "| EX (after re-parse) |\n|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['model']} | {r['n']} | {r['ex_strict']}/{r['n']} "
            f"({r['ex_strict']/r['n']:.1%}) | {r['recovered_final_answers']} | "
            f"{r['recovered_correct']} | {r['ex_after_reparse']}/{r['n']} "
            f"({r['ex_after_reparse_rate']:.1%}) |"
        )

    for r in results:
        print(f"\n{r['file']}")
        for d in r["recovered_details"]:
            mark = "EX" if d["ex"] else "  "
            print(f"  [{mark}] #{d['idx']} {d['db_id']:<24} {d['reason'][:40]} "
                  f"| {d['errors'][:1]}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
