#!/usr/bin/env python3
"""Adaptive resampling (borrowed from MiroFlow GAIA scripts_sol/adaptive_resample.py).

GAIA re-runs only low-confidence tasks k-1 more times instead of re-running everything.
Here the gate is the audit: a question is resampled when it has no SFT-eligible trace yet,
for a reason a new sample can fix.

Subcommands
  select       which questions to resample, with reasons
                 REGEN / DROP / DROP_FORMAT / PENDING_L3B   trace failed the audit
                 no_accepted_trace                          generation or strict audit rejected it
               never selected: questions that already have an SFT-eligible trace, and
               questions whose gold was judged wrong (EX cannot certify a new trace)
  prepare      write K sample roots <out>/s1..sK, each a launcher OUTPUT_ROOT with inputs/
               (use SFT_TEMPERATURE_OVERRIDE when the teacher's greedy decoding is
               deterministic, so the K samples differ)
  consistency  per question, compare the result digests of every attempt that produced a
               final SQL across roots; >= 2 distinct results marks the question unstable
               (instability report, GAIA G1); release.py carries the flag
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RESAMPLE_FINALS = {"REGEN", "DROP", "DROP_FORMAT", "PENDING_L3B"}


def question_of(item_id: str) -> str:
    """Audit ids are '<run>:<question_id>'; planted ones start with 'corrupt:'."""
    return item_id.split(":", 1)[1] if ":" in item_id else item_id


def load_decisions(audit_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit_dir in audit_dirs:
        for d in C.iter_jsonl(audit_dir / "final_decisions.jsonl"):
            if d.get("corruption"):
                continue
            d["_audit_dir"] = str(audit_dir)
            by_question[question_of(d["id"])].append(d)
    return by_question


def legacy_sft_eligible(d: dict[str, Any]) -> bool:
    """For final_decisions written before trace_pipeline_v1 (no sft_eligible field)."""
    if "sft_eligible" in d:
        return bool(d["sft_eligible"])
    return d.get("final") == "PASS"


def cmd_select(args) -> int:
    decisions = load_decisions(args.audit_dir)
    parent = json.loads(Path(args.parent_input).read_text(encoding="utf-8"))
    accepted_anywhere: set[str] = set()
    for root in args.gen_root:
        for path in C.raw_attempt_files(root):
            for row in C.iter_jsonl(path):
                if row.get("accepted"):
                    accepted_anywhere.add(str(row["question_id"]))
    selected, skipped = [], Counter()
    for record in parent:
        qid = str(record["question_id"])
        ds = decisions.get(qid, [])
        if any(legacy_sft_eligible(d) for d in ds):
            skipped["has_sft_trace"] += 1
            continue
        if any(d.get("final") == "DROP_GOLD" or d.get("rl_exclude_question") for d in ds):
            skipped["gold_judged_wrong"] += 1
            continue
        finals = sorted({d["final"] for d in ds if d.get("final") in RESAMPLE_FINALS})
        if finals:
            selected.append({"question_id": qid, "reasons": finals})
        elif not ds:
            reason = "no_accepted_trace" if qid not in accepted_anywhere else "accepted_but_not_audited"
            selected.append({"question_id": qid, "reasons": [reason]})
        else:
            skipped["other:" + ",".join(sorted({d.get('final', '?') for d in ds}))] += 1
    plan = {"created_at_utc": C.utc_now(), "parent_input": str(args.parent_input),
            "audit_dirs": [str(p) for p in args.audit_dir], "gen_roots": [str(p) for p in args.gen_root],
            "selected": len(selected),
            "by_reason": dict(Counter(r for s in selected for r in s["reasons"])),
            "skipped": dict(skipped), "questions": selected}
    C.write_json(args.out, plan)
    print(json.dumps({k: plan[k] for k in ("selected", "by_reason", "skipped")}, ensure_ascii=False))
    return 0


def cmd_prepare(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    wanted = {q["question_id"] for q in plan["questions"]}
    parent_path = Path(plan["parent_input"]) if not args.parent_input else Path(args.parent_input)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    records = [r for r in parent if str(r["question_id"]) in wanted]
    if len(records) != len(wanted):
        raise SystemExit(f"plan has {len(wanted)} questions, parent input has {len(records)} of them")
    roots = []
    for s in range(1, args.samples + 1):
        root = Path(args.out) / f"s{s}"
        C.prepare_inputs(records, root, selection=f"adaptive resample sample {s}/{args.samples}: "
                         f"{json.dumps(plan['by_reason'])}", parent_input=parent_path,
                         chunk_size=args.chunk_size)
        roots.append(str(root))
    C.write_json(Path(args.out) / "resample_plan.json", {**plan, "samples": args.samples, "roots": roots})
    print(f"prepared {args.samples} sample roots for {len(records)} questions under {args.out}")
    return 0


def consistency(roots: list[Path]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"attempts": 0, "with_final": 0, "ex": 0,
                                                            "accepted": 0, "digests": Counter()})
    for root in roots:
        for path in C.raw_attempt_files(root):
            for row in C.iter_jsonl(path):
                s = stats[str(row["question_id"])]
                s["attempts"] += 1
                s["accepted"] += bool(row.get("accepted"))
                grade = row.get("grade") or {}
                s["ex"] += bool(grade.get("ex"))
                digest = grade.get("pred_rows_sha256")
                if row.get("final_sql") and digest:
                    s["with_final"] += 1
                    s["digests"][digest] += 1
    out = {}
    for qid, s in stats.items():
        distinct = len(s["digests"])
        top = s["digests"].most_common(1)[0][1] if s["digests"] else 0
        out[qid] = {"attempts": s["attempts"], "with_final": s["with_final"], "ex": s["ex"],
                    "accepted": s["accepted"], "distinct_results": distinct,
                    "majority_share": round(top / s["with_final"], 3) if s["with_final"] else None,
                    "unstable": distinct >= 2}
    return out


def cmd_consistency(args) -> int:
    result = consistency(args.gen_root)
    rows = [{"question_id": q, **v} for q, v in sorted(result.items())]
    C.write_jsonl(args.out, rows)
    summary = {"questions": len(rows), "unstable": sum(r["unstable"] for r in rows),
               "with_any_ex": sum(r["ex"] > 0 for r in rows),
               "with_any_accepted": sum(r["accepted"] > 0 for r in rows)}
    C.write_json(Path(str(args.out) + ".summary.json"), summary)
    print(json.dumps(summary))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--audit-dir", type=Path, action="append", required=True)
    p.add_argument("--gen-root", type=Path, action="append", default=[])
    p.add_argument("--parent-input", type=Path, required=True, help="easy_current.json of the batch")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--parent-input", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=34)
    p = sub.add_parser("consistency")
    p.add_argument("--gen-root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"select": cmd_select, "prepare": cmd_prepare, "consistency": cmd_consistency}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
