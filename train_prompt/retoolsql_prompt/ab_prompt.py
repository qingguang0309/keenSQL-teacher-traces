#!/usr/bin/env python3
"""A/B a prompt variant against a fixed question subset.

Runs the same questions through the real tool loop with a given system prompt,
then reports the metrics the training pipeline actually depends on:

  * EX (BIRD multiset)
  * protocol_ok
  * verification rate: the submitted SQL was executed verbatim in the episode,
    or an executed statement returned the same row set
  * output-shape errors (prediction returns more/fewer columns than gold)
  * tool mix

Usage::

    python ab_prompt.py --prompt retoolsql_prompt/v4/system_prompt.txt \
        --n 20 --out runs/ab_v4_9b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
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
from run_ex import db_file_for, grade_one  # noqa: E402
from sql_grading import execute_query  # noqa: E402

DIFFICULTY_JSON = Path(
    "/data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/"
    "bird_train/download/train/bird_train_with_difficulty.json"
)
SCHEMA_DIR = ROOT / "data_tools" / "output" / "bird_train"
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip().lower()


def final_verification(record: Dict[str, Any]) -> str:
    """Classify how the final SQL relates to what was executed."""

    final = record.get("final_sql")
    if not final:
        return "no_final"
    executed: List[str] = []
    for message in record.get("transcript", []):
        if message.get("role") != "assistant":
            continue
        for body in TOOL_CALL.findall(message["content"]):
            match = re.search(r"<parameter=sql>\s*(.*?)\s*</parameter>", body, re.S)
            if match:
                executed.append(match.group(1))
    final_norm = normalize_sql(final)
    if any(normalize_sql(sql) == final_norm for sql in executed):
        return "verbatim"
    db_file = db_file_for(record["db_id"])
    try:
        final_rows = frozenset(map(tuple, execute_query(db_file, final)))
    except Exception:
        return "never_executed"
    for sql in executed:
        try:
            if frozenset(map(tuple, execute_query(db_file, sql))) == final_rows:
                return "result_equivalent"
        except Exception:
            continue
    return "never_executed"


def shape_error(record: Dict[str, Any]) -> Optional[str]:
    if record.get("ex"):
        return None
    pred, gold = record.get("pred_rows"), record.get("gold_rows")
    if not pred or not gold:
        return None
    pred_cols, gold_cols = len(pred[0]), len(gold[0])
    if pred_cols > gold_cols:
        return "more_columns"
    if pred_cols < gold_cols:
        return "fewer_columns"
    return None


def tool_mix(record: Dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    for message in record.get("transcript", []):
        if message.get("role") != "assistant":
            continue
        for body in TOOL_CALL.findall(message["content"]):
            match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_]*)", body)
            if match:
                counts[match.group(1)] += 1
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=16000)
    parser.add_argument("--fewshot", action="store_true", default=True)
    parser.add_argument("--no-fewshot", dest="fewshot", action="store_false")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    prompt_path = Path(args.prompt)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    questions = [
        q for q in json.load(open(DIFFICULTY_JSON))
        if q.get("difficulty") == "challenging"
    ]
    usable = [
        q for q in questions
        if db_file_for(q["db_id"]) and (SCHEMA_DIR / f"{q['db_id']}.txt").is_file()
        and isinstance(q.get("SQL"), str) and q["SQL"].strip()
    ]
    random.seed(args.seed)
    sample = random.sample(usable, min(args.n, len(usable)))

    policy = T._openai_policy(
        args.model_url, args.model, args.temperature, args.max_tokens,
        api_key=args.api_key or os.environ.get("QWEN_API_KEY", "EMPTY"),
        timeout=args.timeout,
    )

    records = []
    verification = Counter()
    shapes = Counter()
    tools: Counter = Counter()
    for index, question in enumerate(sample, 1):
        db_id = question["db_id"]
        db_file = str(db_file_for(db_id))
        evidence = question.get("evidence", "") or ""
        schema = (SCHEMA_DIR / f"{db_id}.txt").read_text(encoding="utf-8")
        schema, tier, _tokens = PB.fit_schema(
            schema, question=question["question"], external_knowledge=evidence,
            budget_tokens=args.max_prompt_tokens - 6500,
        )
        try:
            result = T.run_episode(
                policy, db_file, db_id, question["question"], schema,
                max_turns=args.max_turns, verbose=False,
                require_final_scratch_pad=args.fewshot,
                require_tool_scratch_pad=args.fewshot,
                external_knowledge=evidence,
                fewshot=args.fewshot,
                system_prompt=system_prompt,
            )
        except Exception as exc:  # a single bad episode must not abort the A/B
            print(f"[{index}/{len(sample)}] {db_id:<24} ERROR {type(exc).__name__}: {str(exc)[:90]}",
                  flush=True)
            records.append({
                "db_id": db_id, "question": question["question"],
                "gold_sql": question["SQL"], "transcript": [],
                "protocol_ok": False, "final_sql": None,
                "errors": [f"runtime: {type(exc).__name__}"],
                "tool_calls": 0, "ex": False,
                "ex_reason": f"episode crashed: {type(exc).__name__}",
                "verification": "no_final", "shape_error": None,
                "schema_tier": tier,
            })
            verification["no_final"] += 1
            continue
        final = result["final"]
        record: Dict[str, Any] = {
            "db_id": db_id,
            "question": question["question"],
            "gold_sql": question["SQL"],
            "transcript": result["transcript"],
            "protocol_ok": final is not None and not result["errors"],
            "final_sql": (final or {}).get("sql"),
            "errors": result["errors"],
            "tool_calls": result["tool_calls"],
            "ex": False,
            "ex_reason": "no final SQL produced",
        }
        if record["final_sql"]:
            ex, reason, pred, gold = grade_one(
                db_file, record["final_sql"], question["SQL"], "multiset"
            )
            record.update({"ex": bool(ex), "ex_reason": reason,
                           "pred_rows": pred, "gold_rows": gold})
        record["verification"] = final_verification(record)
        record["shape_error"] = shape_error(record)
        verification[record["verification"]] += 1
        if record["shape_error"]:
            shapes[record["shape_error"]] += 1
        tools.update(tool_mix(record))
        records.append(record)
        print(
            f"[{index}/{len(sample)}] {db_id:<24} proto={'ok' if record['protocol_ok'] else 'FAIL'} "
            f"EX={record['ex']} verify={record['verification']} shape={record['shape_error']} "
            f"tools={record['tool_calls']}",
            flush=True,
        )

    n = len(records)
    ex = sum(1 for r in records if r["ex"])
    protocol = sum(1 for r in records if r["protocol_ok"])
    summary = {
        "prompt": str(prompt_path),
        "model": args.model,
        "n": n,
        "ex": ex,
        "ex_rate": ex / n if n else 0.0,
        "protocol_ok": protocol,
        "protocol_rate": protocol / n if n else 0.0,
        "verification": dict(verification),
        "shape_errors": dict(shapes),
        "tool_mix": dict(tools),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in records) / n, 2) if n else 0.0,
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("\n===== A/B summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
