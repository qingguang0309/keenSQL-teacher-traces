#!/usr/bin/env python3
"""Run a fixed list of questions with an explicit schema tier override.

Used to separate two explanations for a failure: the schema was compressed and
lost information, or the question is simply hard. Pass ``--tier 0`` to send the
full M-Schema and compare with the tier used in the original run.

Usage::

    python rerun_questions.py --questions runs/tier1_questions.json --tier 0 \
        --prompt retoolsql_prompt/v4/system_prompt.txt --out runs/tier1_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "database_tools"))
sys.path.insert(0, str(ROOT / "retoolsql_prompt"))
sys.path.insert(0, "/data1/ReViSQL/local_rlvr")

import prompt_builder as PB  # noqa: E402
import test_retoolsql_prompt as T  # noqa: E402
from ab_prompt import final_verification, shape_error, tool_mix  # noqa: E402
from run_ex import db_file_for, grade_one  # noqa: E402

SCHEMA_DIR = ROOT / "data_tools" / "output" / "bird_train"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--tier", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prompt", default=str(ROOT / "retoolsql_prompt" / "v4" / "system_prompt.txt"))
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=16000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    difficulty = json.load(open(
        "/data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/"
        "bird_train/download/train/bird_train_with_difficulty.json"
    ))
    by_question = {q["question"]: q for q in difficulty}

    policy = T._openai_policy(
        args.model_url, args.model, 0.0, args.max_tokens,
        api_key=args.api_key or os.environ.get("QWEN_API_KEY", "EMPTY"),
        timeout=args.timeout,
    )

    records: List[Dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        question = by_question.get(item["question"])
        if question is None:
            print(f"[{index}] missing question: {item['question'][:60]}")
            continue
        db_id = question["db_id"]
        db_file = str(db_file_for(db_id))
        evidence = question.get("evidence", "") or ""
        raw_schema = (SCHEMA_DIR / f"{db_id}.txt").read_text(encoding="utf-8")
        schema = raw_schema if args.tier == 0 else PB.compress_schema(raw_schema)
        tokens = PB.estimate_tokens(schema)
        if tokens > args.max_prompt_tokens - 6000:
            print(f"[{index}] {db_id}: tier {args.tier} schema is {tokens} tokens; "
                  f"skipping (would overflow)")
            continue
        result = T.run_episode(
            policy, db_file, db_id, question["question"], schema,
            max_turns=10, verbose=False, external_knowledge=evidence,
            fewshot=True, system_prompt=system_prompt,
        )
        final = result["final"]
        record: Dict[str, Any] = {
            "db_id": db_id, "question": question["question"],
            "gold_sql": question["SQL"], "tier": args.tier,
            "schema_tokens": tokens, "transcript": result["transcript"],
            "protocol_ok": final is not None and not result["errors"],
            "final_sql": (final or {}).get("sql"),
            "errors": result["errors"], "tool_calls": result["tool_calls"],
            "ex": False, "ex_reason": "no final SQL produced",
        }
        if record["final_sql"]:
            ex, reason, pred, gold = grade_one(
                db_file, record["final_sql"], question["SQL"], "multiset"
            )
            record.update({"ex": bool(ex), "ex_reason": reason,
                           "pred_rows": pred, "gold_rows": gold})
        record["verification"] = final_verification(record)
        record["shape_error"] = shape_error(record)
        records.append(record)
        print(f"[{index}/{len(items)}] {db_id:<16} tier={args.tier} EX={record['ex']} "
              f"verify={record['verification']} tools={record['tool_calls']}", flush=True)

    summary = {
        "tier": args.tier,
        "n": len(records),
        "ex": sum(1 for r in records if r["ex"]),
        "protocol_ok": sum(1 for r in records if r["protocol_ok"]),
        "verification": {},
    }
    for record in records:
        key = record["verification"]
        summary["verification"][key] = summary["verification"].get(key, 0) + 1
    Path(args.out).write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
