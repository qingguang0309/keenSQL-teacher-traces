#!/usr/bin/env python3
"""Release: pick the best audited trace per question, decontaminate against BIRD dev, and
write the SFT file, the RL exclusion list and a manifest (borrowed from GAIA's best-of
selection over runs, plus our own reporting rule: count usable traces after the audit).

Inputs: one or more audit dirs (each: items.jsonl + items.jsonl.sources.json,
aggregate.jsonl, final_decisions.jsonl). Audit dirs written before trace_pipeline_v1 have
no sft_eligible field; it is derived with the same rule (trajectory PASS, not HTML-escaped,
and for DROP_GOLD questions the trajectory verdict of the judges or L3b).

Selection among a question's SFT-eligible traces, best first:
  1. accepted by both judges at L3a without escalation
  2. no ritual-call flag (T4) from any judge
  3. higher lowest judge confidence (missing counts as 80)
  4. at least one observation-driven step (2+ tool calls) — real exploration, not padding,
     since 2 already excludes T4
  5. shorter trajectory
Decontamination: a training question is dropped when its normalized text equals a BIRD dev
question, or its word-3-gram Jaccard with one is >= --decontam-threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

JUDGES = ("kimi", "deepseek")


def norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def grams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = norm_question(text).split()
    return {tuple(words[i: i + n]) for i in range(max(1, len(words) - n + 1))}


class DevIndex:
    def __init__(self, dev_path: Path) -> None:
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        self.exact = {norm_question(r["question"]): i for i, r in enumerate(dev)}
        self.grams = [grams(r["question"]) for r in dev]
        self.dbs = {r["db_id"] for r in dev}
        self.index: dict[tuple, list[int]] = defaultdict(list)
        for i, g in enumerate(self.grams):
            for gram in g:
                self.index[gram].append(i)
        self.dev = dev

    def match(self, question: str, threshold: float) -> dict[str, Any] | None:
        key = norm_question(question)
        if key in self.exact:
            return {"kind": "exact", "dev_index": self.exact[key], "score": 1.0}
        g = grams(question)
        hits = Counter(i for gram in g for i in self.index.get(gram, ()))
        best = None
        for i, shared in hits.most_common(5):
            score = shared / len(g | self.grams[i])
            if score >= threshold and (best is None or score > best["score"]):
                best = {"kind": "near", "dev_index": i, "score": round(score, 3)}
        return best


def derive(decision: dict[str, Any], agg: dict[str, Any], l3b: dict[str, Any] | None) -> dict[str, Any]:
    """Fill trajectory_verdict / sft_eligible / rl_exclude_question for legacy decisions."""
    if "sft_eligible" in decision:
        return decision
    final = decision["final"]
    escaped = bool(decision["rule_tags"].get("html_escaped_sql"))
    if final == "DROP_GOLD":
        if l3b and l3b.get("verdict_json"):
            trajectory = str(l3b["verdict_json"].get("verdict", "")).upper()
        else:
            verdicts = [(agg.get(n) or {}).get("policy_verdict") for n in JUDGES]
            trajectory = "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "NOT_PASS"
    elif final == "DROP_FORMAT":
        trajectory = "PASS"
    else:
        trajectory = final
    decision.update(trajectory_verdict=trajectory,
                    sft_eligible=trajectory == "PASS" and not escaped and not decision.get("corruption"),
                    rl_exclude_question=final == "DROP_GOLD", derived_legacy=True)
    return decision


def rank_key(c: dict[str, Any]) -> tuple:
    agg = c["agg"]
    codes = [code for n in JUDGES for code in ((agg.get(n) or {}).get("codes") or [])]
    confs = [(agg.get(n) or {}).get("confidence") for n in JUDGES]
    confs = [80 if x is None else x for x in confs]
    tags = c["decision"]["rule_tags"]
    return (
        c["decision"]["l3a_route"] == "PASS",
        not any(code.startswith("T4") for code in codes),
        min(confs) if confs else 80,
        int(tags.get("tool_calls") or 0) >= 2,
        -int(tags.get("observations_total_chars") or 0),
    )


def load_audit(audit_dir: Path) -> list[dict[str, Any]]:
    sources = json.loads((audit_dir / "items.jsonl.sources.json").read_text(encoding="utf-8"))
    agg = {r["id"]: r for r in C.iter_jsonl(audit_dir / "aggregate.jsonl")}
    l3b = {}
    if (audit_dir / "l3b.jsonl").is_file():
        l3b = {r["id"]: r for r in C.iter_jsonl(audit_dir / "l3b.jsonl") if r.get("verdict_json")}
    rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for run, path in sources.items():
        rows_by_run[run] = {str(r["question_id"]): r for r in C.iter_jsonl(path)}
    out = []
    for d in C.iter_jsonl(audit_dir / "final_decisions.jsonl"):
        if d.get("corruption"):
            continue
        run, qid = d["id"].split(":", 1)
        d = derive(d, agg.get(d["id"], {}), l3b.get(d["id"]))
        out.append({"audit_dir": str(audit_dir), "decision": d, "agg": agg.get(d["id"], {}),
                    "row": rows_by_run[run][qid], "question_id": qid})
    return out


def cmd_release(args) -> int:
    candidates = [c for audit_dir in args.audit_dir for c in load_audit(audit_dir)]
    unstable = {}
    if args.consistency:
        unstable = {r["question_id"]: r for r in C.iter_jsonl(args.consistency)}
    dev = C.DEV_JSON if args.dev_json is None else args.dev_json
    index = DevIndex(dev)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rl_exclude: dict[str, dict[str, Any]] = {}
    finals = Counter()
    for c in candidates:
        finals[c["decision"]["final"]] += 1
        if c["decision"].get("rl_exclude_question"):
            rl_exclude[c["question_id"]] = c
        if c["decision"]["sft_eligible"]:
            by_q[c["question_id"]].append(c)
    selected, contaminated = [], []
    for qid, cands in sorted(by_q.items()):
        cands.sort(key=rank_key, reverse=True)
        best = cands[: args.per_question]
        hit = index.match(best[0]["row"]["question"], args.decontam_threshold)
        if hit:
            contaminated.append({"question_id": qid, **hit,
                                 "dev_question": index.dev[hit["dev_index"]]["question"]})
            continue
        for rank, c in enumerate(best, 1):
            selected.append((c, rank, len(cands)))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft_release.jsonl"
    rows = []
    for c, rank, n in selected:
        row = dict(c["row"])
        d = c["decision"]
        agg = c["agg"]
        row["audit_l3"] = {
            "audit_dir": c["audit_dir"], "audit_id": d["id"], "final": d["final"],
            "trajectory_verdict": d["trajectory_verdict"], "l3a_route": d["l3a_route"],
            "gold_suspect": d["final"] == "DROP_GOLD", "exclude_from_rl": bool(d["rl_exclude_question"]),
            "judge_confidence": {n: (agg.get(n) or {}).get("confidence") for n in JUDGES},
            "candidates_for_question": n, "rank": rank,
            "unstable_across_samples": bool(unstable.get(c["question_id"], {}).get("unstable")),
        }
        rows.append(row)
    C.write_jsonl(sft_path, rows)
    (out_dir / "release_ids.txt").write_text("\n".join(r["audit_l3"]["audit_id"] for r in rows) + "\n")
    with (out_dir / "rl_exclude_questions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "db_id", "source", "reason", "question", "gold_sql"])
        for qid, c in sorted(rl_exclude.items()):
            r = c["row"]
            writer.writerow([qid, r.get("db_id"), r.get("source"), "gold_judged_wrong", r.get("question"), r.get("gold_sql")])
        for qid, info in sorted(unstable.items()):
            if info.get("unstable") and qid not in rl_exclude and args.rl_exclude_unstable:
                writer.writerow([qid, "", "", "unstable_across_samples", "", ""])
    C.write_jsonl(out_dir / "decontaminated.jsonl", contaminated)

    audits = []
    for audit_dir in args.audit_dir:
        entry: dict[str, Any] = {"dir": str(audit_dir),
                                 "final_decisions_md5": C.md5_file(audit_dir / "final_decisions.jsonl")}
        for name in ("report.json", "final_report.json"):
            path = audit_dir / name
            if path.is_file():
                rep = json.loads(path.read_text(encoding="utf-8"))
                if name == "report.json":
                    entry["judge_cost_usd"] = {n: rep.get(n, {}).get("cost_usd") for n in JUDGES}
                    entry["kappa_pass_vs_not"] = rep.get("kappa_pass_vs_not")
                    entry["recall_on_planted_errors"] = {
                        n: (rep.get("recall_on_planted_errors") or {}).get(n, {}).get("overall_flagged_rate")
                        for n in JUDGES}
                    entry["quote_valid_rate"] = {n: rep.get(n, {}).get("quote_valid_rate") for n in JUDGES}
                    entry["calibration_gate"] = rep.get("calibration_gate")
                else:
                    entry["l3b_cost_usd"] = rep.get("l3b_cost_usd")
                    entry["planted_escaped_rate"] = rep.get("planted_escaped_rate")
        audits.append(entry)
    generation = {}
    for root in args.gen_root:
        import scan_traces as S
        m = S.scan(root)
        generation[str(root)] = {k: m.get(k) for k in ("attempts", "accepted", "cost_usd", "telemetry_coverage")}
    manifest = {
        "created_at_utc": C.utc_now(),
        "policy": {"per_question": args.per_question, "decontam_threshold": args.decontam_threshold,
                   "gold_bad_trajectory_kept_for_sft": True, "rl_exclude_unstable": args.rl_exclude_unstable},
        "audits": audits,
        "generation": generation,
        "counts": {
            "real_traces_audited": len(candidates),
            "final_by_category": dict(finals),
            "sft_eligible_traces": sum(len(v) for v in by_q.values()),
            "questions_with_eligible_trace": len(by_q),
            "dropped_dev_contamination": len(contaminated),
            "released_traces": len(rows),
            "released_gold_bad_trajectories": sum(r["audit_l3"]["gold_suspect"] for r in rows),
            "rl_exclude_questions": len(rl_exclude),
            "train_questions_on_dev_databases": sum(1 for c in candidates if c["row"].get("db_id") in index.dbs),
        },
        "outputs": {"sft_release": str(sft_path), "sft_release_md5": C.md5_file(sft_path),
                    "rl_exclude": str(out_dir / "rl_exclude_questions.csv")},
    }
    C.write_json(out_dir / "release_manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"sft_release.jsonl md5 {manifest['outputs']['sft_release_md5']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-dir", type=Path, action="append", required=True)
    parser.add_argument("--gen-root", type=Path, action="append", default=[], help="for generation cost in the manifest")
    parser.add_argument("--consistency", type=Path, default=None, help="resample.py consistency output")
    parser.add_argument("--dev-json", type=Path, default=None)
    parser.add_argument("--decontam-threshold", type=float, default=0.8)
    parser.add_argument("--per-question", type=int, default=1)
    parser.add_argument("--rl-exclude-unstable", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return cmd_release(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
