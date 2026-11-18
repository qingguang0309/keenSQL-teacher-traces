#!/usr/bin/env python3
"""Summarize one or more run_ex.py result files.

Reports, per run:
  * headline EX and protocol_ok
  * EX restricted to episodes where the protocol parsed cleanly
  * "recoverable" EX: episodes where the turn protocol failed but the transcript
    still contains a complete <final_answer> SQL block, graded the same way
  * tool-call distribution and failure reasons
  * a Markdown table for side-by-side model comparison

Usage::

    python analyze_runs.py runs/v3_challenge_qwen35_9b.json \
        runs/v3_challenge_qwen36plus.json runs/v3_challenge_qwen38max.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "database_tools"))
sys.path.insert(0, str(ROOT / "retoolsql_prompt"))
sys.path.insert(0, "/data1/ReViSQL/local_rlvr")

from run_ex import db_file_for, grade_one  # noqa: E402

_FINAL = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.S)
_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.I)


def _iter_final_blocks(transcript: List[Dict[str, str]]) -> List[str]:
    sqls: List[str] = []
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        for body in _FINAL.findall(message.get("content", "")):
            body = _FENCE.sub("", body.strip()).strip()
            if body:
                sqls.append(body)
    return sqls


def recover_sql(record: Dict[str, Any]) -> Optional[str]:
    """Return the last complete <final_answer> SQL from a failed episode."""

    if record.get("final_sql"):
        return record["final_sql"]
    blocks = _iter_final_blocks(record.get("transcript", []))
    return blocks[-1] if blocks else None


def analyze(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    config = payload.get("config") or {}

    n = len(records)
    ex = sum(1 for r in records if r.get("ex"))
    proto_ok = [r for r in records if r.get("protocol_ok")]
    proto_ex = sum(1 for r in proto_ok if r.get("ex"))

    recovered, recovered_ex = 0, 0
    recovered_details = []
    for r in records:
        if r.get("protocol_ok"):
            continue
        sql = recover_sql(r)
        if not sql:
            continue
        db_file = db_file_for(r["db_id"])
        if not db_file:
            continue
        ok, reason, _pred, _gold = grade_one(db_file, sql, r["gold_sql"], "multiset")
        recovered += 1
        if ok:
            recovered_ex += 1
        recovered_details.append(
            {
                "idx": r.get("idx"),
                "db_id": r["db_id"],
                "ex": bool(ok),
                "reason": reason,
                "errors": r.get("errors", []),
            }
        )

    tool_calls = [r.get("tool_calls", 0) for r in records]
    reasons: Dict[str, int] = {}
    for r in records:
        if r.get("ex"):
            continue
        key = (r.get("ex_reason") or "unknown").split(":")[0][:80]
        reasons[key] = reasons.get(key, 0) + 1

    error_kinds: Dict[str, int] = {}
    for r in records:
        for err in r.get("errors", []):
            error_kinds[err[:70]] = error_kinds.get(err[:70], 0) + 1

    return {
        "file": path.name,
        "model": config.get("model", ""),
        "prompt": config.get("system_prompt", ""),
        "n": n,
        "ex": ex,
        "ex_rate": ex / n if n else 0.0,
        "protocol_ok": len(proto_ok),
        "protocol_rate": len(proto_ok) / n if n else 0.0,
        "protocol_ex": proto_ex,
        "protocol_ex_rate": proto_ex / len(proto_ok) if proto_ok else 0.0,
        "recovered_sql": recovered,
        "recovered_ex": recovered_ex,
        "ex_with_recovery": ex + recovered_ex,
        "ex_with_recovery_rate": (ex + recovered_ex) / n if n else 0.0,
        "avg_tool_calls": round(sum(tool_calls) / n, 2) if n else 0.0,
        "max_tool_calls": max(tool_calls) if tool_calls else 0,
        "fail_reasons": reasons,
        "protocol_errors": error_kinds,
        "recovered_details": recovered_details,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    summaries = [analyze(Path(f)) for f in args.files]

    print("=" * 78)
    for s in summaries:
        print(f"{s['file']}  ({s['model']})")
        print(f"  questions          : {s['n']}")
        print(f"  EX                 : {s['ex']}/{s['n']} ({s['ex_rate']:.1%})")
        print(f"  protocol_ok        : {s['protocol_ok']}/{s['n']} ({s['protocol_rate']:.1%})")
        print(f"  EX on protocol-ok  : {s['protocol_ex']}/{s['protocol_ok']} "
              f"({s['protocol_ex_rate']:.1%})")
        print(f"  recovered SQL      : {s['recovered_sql']} graded, "
              f"{s['recovered_ex']} correct")
        print(f"  EX incl. recovery  : {s['ex_with_recovery']}/{s['n']} "
              f"({s['ex_with_recovery_rate']:.1%})")
        print(f"  tool calls         : avg {s['avg_tool_calls']}, max {s['max_tool_calls']}")
        print(f"  fail reasons       : {s['fail_reasons']}")
        print(f"  protocol errors    : {s['protocol_errors']}")
        print("-" * 78)

    header = (
        "| Model | Prompt | EX | Protocol OK | EX on protocol-ok | "
        "EX incl. recovery | Avg tools |\n"
        "|---|---|---|---|---|---|---|"
    )
    print(header)
    for s in summaries:
        print(
            f"| {s['model']} | v3 | {s['ex']}/{s['n']} ({s['ex_rate']:.1%}) | "
            f"{s['protocol_ok']}/{s['n']} ({s['protocol_rate']:.1%}) | "
            f"{s['protocol_ex']}/{s['protocol_ok']} ({s['protocol_ex_rate']:.1%}) | "
            f"{s['ex_with_recovery']}/{s['n']} ({s['ex_with_recovery_rate']:.1%}) | "
            f"{s['avg_tool_calls']} |"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
