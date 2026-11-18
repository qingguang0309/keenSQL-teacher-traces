#!/usr/bin/env python3
"""Score a BIRD-dev evaluation run of the pipeline with the official evaluator, and relate
the official result to the pipeline's own verdicts.

Inputs: the run root prepared by prepare_dev_eval.py (raw attempts under models/, NOT the
smoke subdirectory) and, when present, <root>/audit/final_decisions.jsonl.

Outputs under <root>/eval/:
  pred/predict_dev.json   official format, attempt 1's final SQL verbatim; a question without
                          a final SQL is submitted as an always-erroring statement (scores 0)
  gold/dev_gold.sql       dev.json gold with newlines flattened (tabs asserted absent)
  official_eval.txt       stdout of bird_evaluator/llm/src/evaluation.py (the number we report)
  per_question.csv        per question: difficulty, accepted, project EX, official EX (same
                          set comparison as the evaluator, 30 s timeout), audit final verdict
  summary.json            the tables used in the report

--base ROOT combines two runs: a question with no attempt under --root is taken from the
--base run (e.g. Opus for moderate/challenging under --root, GPT for simple from --base).
per_question.csv then names the run each question came from ("source").
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

DEV_DB = C.ROOT / "dataset/bird-dev/databases"
EVALUATOR = C.ROOT / "bird_evaluator/llm/src/evaluation.py"
NO_PREDICTION = "SELECT no_prediction FROM __no_prediction__"


def official_ex(db_id: str, pred: str, gold: str, timeout: float = 30.0) -> int:
    conn = sqlite3.connect(f"file:{DEV_DB / db_id / (db_id + '.sqlite')}?mode=ro", uri=True)
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 10000)
    try:
        p = conn.execute(pred).fetchall()
        start = time.time()
        g = conn.execute(gold).fetchall()
        return int(set(p) == set(g))
    except Exception:  # noqa: BLE001 - error/timeout scores 0, as in the evaluator
        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--base", type=Path, default=None,
                        help="run that supplies every question not attempted under --root")
    args = parser.parse_args()
    dev = json.loads(C.DEV_JSON.read_text(encoding="utf-8"))
    attempts = load_attempts(args.root)
    source = {qid: args.root.name for qid in attempts}
    if args.base:
        for qid, row in load_attempts(args.base).items():
            if qid not in attempts:
                attempts[qid] = row
                source[qid] = args.base.name
    missing = [i for i in range(len(dev)) if f"birddev_{i}" not in attempts]
    if missing:
        raise SystemExit(f"{len(missing)} dev questions have no attempt, e.g. {missing[:5]}")
    finals = {}
    fd = args.root / "audit/final_decisions.jsonl"
    if fd.is_file():
        for d in C.iter_jsonl(fd):
            if not d.get("corruption"):
                finals[d["id"].split(":", 1)[1]] = d
    out = args.root / "eval"
    (out / "pred").mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(parents=True, exist_ok=True)
    preds = {}
    for i, r in enumerate(dev):
        sql = str(attempts[f"birddev_{i}"].get("final_sql") or "").strip() or NO_PREDICTION
        preds[str(i)] = f"{sql}\t----- bird -----\t{r['db_id']}"
    (out / "pred/predict_dev.json").write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    gold_lines = []
    for r in dev:
        g = re.sub(r"\s*\n\s*", " ", r["SQL"].strip())
        assert "\t" not in g
        gold_lines.append(f"{g}\t{r['db_id']}")
    (out / "gold/dev_gold.sql").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(EVALUATOR), "--predicted_sql_path", str(out / "pred") + "/",
           "--ground_truth_path", str(out / "gold") + "/", "--data_mode", "dev",
           "--db_root_path", str(DEV_DB) + "/", "--num_cpus", str(args.num_cpus),
           "--meta_time_out", "30.0", "--diff_json_path", str(C.DEV_JSON)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out / "official_eval.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print(proc.stdout[-900:])

    def score(i: int) -> int:
        r = dev[i]
        return official_ex(r["db_id"], preds[str(i)].split("\t----- bird -----\t")[0], r["SQL"])

    with ThreadPoolExecutor(args.num_cpus) as pool:
        off = list(pool.map(score, range(len(dev))))
    rows = []
    for i, r in enumerate(dev):
        a = attempts[f"birddev_{i}"]
        d = finals.get(f"birddev_{i}")
        rows.append({
            "dev_index": i, "db_id": r["db_id"], "difficulty": r["difficulty"],
            "source": source[f"birddev_{i}"], "model": a.get("model"),
            "accepted": bool(a.get("accepted")), "project_ex": bool((a.get("grade") or {}).get("ex")),
            "official_ex": off[i], "has_final_sql": bool(a.get("final_sql")),
            "tool_calls": (a.get("episode") or {}).get("tool_calls"),
            "audit_final": d["final"] if d else ("NOT_AUDITED" if a.get("accepted") else "REJECTED_AT_GENERATION"),
            "trajectory_verdict": d.get("trajectory_verdict") if d else "",
            "cost_usd": (a.get("pipeline") or {}).get("cost_usd"),
            "prompt_budget": (a.get("prompt_meta") or {}).get("max_prompt_tokens"),
        })
    with (out / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_diff = defaultdict(Counter)
    for row in rows:
        for key in ("all", row["difficulty"]):
            c = by_diff[key]
            c["n"] += 1
            c["official_ex"] += row["official_ex"]
            c["project_ex"] += row["project_ex"]
            c["accepted"] += row["accepted"]
            c["audit_pass"] += row["audit_final"] == "PASS"
            c["sft_eligible_like"] += bool(finals.get(f"birddev_{row['dev_index']}", {}).get("sft_eligible"))
    pct = lambda a, b: round(100 * a / b, 2) if b else None
    summary = {
        "by_difficulty": {k: {"n": c["n"], "official_ex_pct": pct(c["official_ex"], c["n"]),
                              "project_ex_pct": pct(c["project_ex"], c["n"]),
                              "accepted_pct": pct(c["accepted"], c["n"]),
                              "audit_pass_pct": pct(c["audit_pass"], c["n"]),
                              "trajectory_ok_pct": pct(c["sft_eligible_like"], c["n"])}
                          for k, c in by_diff.items()},
        "audit_final_vs_official": {k: dict(v) for k, v in _cross(rows).items()},
        "generation_cost_usd": round(sum(float(r["cost_usd"] or 0) for r in rows), 3),
        "questions_by_source": dict(Counter(f"{r['source']} [{r['model']}] {r['difficulty']}" for r in rows)),
        # Questions whose prompt only fits above 16k tokens (the student harness budget): the
        # official number with them counted as wrong is the like-for-like comparison.
        "over_16k_prompt": {"n": sum(1 for r in rows if (r["prompt_budget"] or 16000) > 16000),
                            "official_correct": sum(r["official_ex"] for r in rows if (r["prompt_budget"] or 16000) > 16000)},
        "official_ex_pct_if_over_16k_counted_wrong": round(100 * sum(
            r["official_ex"] for r in rows if (r["prompt_budget"] or 16000) <= 16000) / len(rows), 2),
        "official_eval_tail": proc.stdout[-600:],
    }
    C.write_json(out / "summary.json", summary)
    print(json.dumps(summary["by_difficulty"], ensure_ascii=False, indent=1))
    return proc.returncode


def load_attempts(root: Path) -> dict:
    """Attempts of one run: the main root and its sub-batches under extra/ (never smoke/)."""
    attempts = {}
    for sub in [root] + sorted(p for p in (root / "extra").glob("*") if p.is_dir()):
        for path in C.raw_attempt_files(sub):
            for row in C.iter_jsonl(path):
                if row["question_id"] in attempts:
                    raise SystemExit(f"{row['question_id']} attempted twice under {root}")
                attempts[row["question_id"]] = row
    return attempts


def _cross(rows):
    table = defaultdict(Counter)
    for row in rows:
        table[row["audit_final"]]["official_correct" if row["official_ex"] else "official_wrong"] += 1
    return table


if __name__ == "__main__":
    raise SystemExit(main())
