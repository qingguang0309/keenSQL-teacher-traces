#!/usr/bin/env python3
"""Probe the SFT privileged-teacher prompt: protocol, verification, leakage.

Runs N challenging questions through ``build_teacher_messages`` with a real
model, then applies the SFT acceptance checks that the training pipeline would
use. The point is to find out whether the teacher addendum produces usable
trajectories before spending a full data-generation budget.

Usage::

    python probe_teacher.py --n 6 --model qwen3.6-plus --model-url "$QWEN_BASE_URL" \
        --api-key "$QWEN_API_KEY" --out runs/teacher_probe_plus.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

import prompt_builder as PB  # noqa: E402
import sqlite_tools as st  # noqa: E402
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
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(";").strip()


def _verify_declared(final: Optional[Dict[str, Any]], executed_calls: List[tuple]) -> Dict[str, Any]:
    """Resolve ``Verified call: k`` where k counts every executed tool call.

    ``executed_calls`` is a list of ``(tool_name, sql_text)`` in call order; the
    cited entry must be a ``sqlite_query`` whose SQL matches the final answer.
    """

    if not final or not final.get("sql"):
        return {"declared": None, "matches": False, "reason": "no_final_sql"}
    declared = final.get("verified_call")
    submitted = normalize_sql(final["sql"])
    if declared is None:
        exact = any(
            normalize_sql(sql) == submitted
            for tool, sql in executed_calls if tool == "sqlite_query" and sql
        )
        return {"declared": None, "matches": exact, "reason": "no_verified_call_line"}
    if declared < 1 or declared > len(executed_calls):
        return {
            "declared": declared,
            "matches": False,
            "reason": f"index_out_of_range (1..{len(executed_calls)})",
        }
    tool, sql = executed_calls[declared - 1]
    if tool != "sqlite_query":
        return {
            "declared": declared,
            "matches": False,
            "reason": f"cited_call_is_{tool}",
        }
    match = normalize_sql(sql) == submitted
    return {
        "declared": declared,
        "matches": match,
        "reason": "" if match else "cited_sql_differs_from_final",
    }


def gold_literals(sql: str) -> List[str]:
    """String literals long enough to be distinctive."""

    values = re.findall(r"'((?:[^'\\]|\\.)*)'", sql)
    unique = {v for v in values if len(v) >= 5}
    return sorted(unique)


def check_trace(record: Dict[str, Any], gold_sql: str) -> Dict[str, Any]:
    transcript = record.get("transcript", [])
    assistant_text = "\n".join(
        m["content"] for m in transcript if m.get("role") == "assistant"
    )
    # Leakage is only meaningful in the model's *prose*. Its own tool-call SQL
    # and final SQL may legitimately equal the gold SQL: that is the target we
    # want, not a leak.
    prose = TOOL_CALL.sub(" ", assistant_text)
    prose = re.sub(r"<final_answer>.*?</final_answer>", " ", prose, flags=re.S)
    issues: List[str] = []

    if not record.get("protocol_ok"):
        issues.append("protocol")
    if not record.get("final_sql"):
        issues.append("no_final_sql")

    # every final SQL must have been executed verbatim (or result-equivalently)
    db_file = db_file_for(record["db_id"])
    final = record.get("final_sql")
    executed = False
    executed_verbatim = False
    if final and db_file:
        final_norm = normalize_sql(final)
        for body in TOOL_CALL.findall(assistant_text):
            match = re.search(r"<parameter=sql>\s*(.*?)\s*</parameter>", body, re.S)
            if not match:
                continue
            candidate = match.group(1)
            if normalize_sql(candidate) == final_norm:
                executed = executed_verbatim = True
                break
        if not executed:
            try:
                final_rows = execute_query(db_file, final)
                for body in TOOL_CALL.findall(assistant_text):
                    match = re.search(r"<parameter=sql>\s*(.*?)\s*</parameter>", body, re.S)
                    if not match:
                        continue
                    try:
                        rows = execute_query(db_file, match.group(1))
                    except Exception:
                        continue
                    if frozenset(map(tuple, rows)) == frozenset(map(tuple, final_rows)):
                        executed = True
                        break
            except Exception:
                pass
    if not executed:
        issues.append("final_not_executed")

    # gold leakage in prose: normalized whole SQL or any distinctive literal
    gold_norm = normalize_sql(gold_sql)
    lowered = normalize_sql(prose)
    if gold_norm and gold_norm in lowered:
        issues.append("gold_sql_leak")
    leaked_literals = [
        literal for literal in gold_literals(gold_sql)
        if literal.lower() in prose.lower()
    ]
    # a literal also present in the question/evidence/schema is not a leak
    context = (record.get("question", "") + " " + record.get("evidence", "") + " "
               + record.get("schema", ""))
    context_lower = context.lower()
    leaked_literals = [x for x in leaked_literals if x.lower() not in context_lower]
    if leaked_literals:
        issues.append("gold_literal_leak")

    if re.search(r"\b(the reference|reference sql|gold sql|privileged)\b", prose, re.I):
        issues.append("mentions_reference")

    calls = len(TOOL_CALL.findall(assistant_text))
    turns = sum(1 for m in transcript if m.get("role") == "assistant")
    if calls < 1:
        issues.append("no_tool_call")
    if calls > 4:
        issues.append("too_many_calls")
    if turns > 6:
        issues.append("too_many_turns")

    # A trace is usable for SFT only when the submitted SQL is signed and the
    # signature points at an executed call whose text matches.
    verification = record.get("verification") or {}
    if verification.get("declared") is None:
        issues.append("unsigned_final")
    elif not verification.get("matches"):
        issues.append("verified_call_mismatch")

    return {
        "issues": issues,
        "accepted": not issues,
        "calls": calls,
        "turns": turns,
        "executed_verbatim": executed_verbatim,
        "leaked_literals": leaked_literals,
        "verified_call": verification.get("declared"),
        "ex": record.get("ex"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--questions-file", default=None,
                        help="JSON list of {question, db_id} to use instead of sampling")
    parser.add_argument("--questions-json", default=None,
                        help="full BIRD difficulty JSON to read questions from")
    parser.add_argument("--no-teacher", action="store_true",
                        help="use the plain deployment prompt (control condition)")
    parser.add_argument("--teacher-file", default=None,
                        help="override the teacher appendix file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    questions = [
        q for q in json.load(open(DIFFICULTY_JSON))
        if q.get("difficulty") == "challenging"
    ]
    usable = [
        q for q in questions
        if db_file_for(q["db_id"]) and (SCHEMA_DIR / f"{q['db_id']}.txt").is_file()
        and isinstance(q.get("SQL"), str) and q["SQL"].strip()
    ]
    if args.questions_file:
        sample = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        # accept either bare BIRD records or our trimmed {idx, db_id, question} form
    else:
        random.seed(args.seed)
        sample = random.sample(usable, min(args.n, len(usable)))

    policy = T._openai_policy(
        args.model_url, args.model, args.temperature, args.max_tokens,
        api_key=args.api_key, timeout=args.timeout,
    )

    records = []
    for index, question in enumerate(sample, 1):
        db_id = question["db_id"]
        db_file = str(db_file_for(db_id))
        schema = (SCHEMA_DIR / f"{db_id}.txt").read_text(encoding="utf-8")
        evidence = question.get("evidence", "") or ""
        built = PB.build_teacher_messages(
            engine="SQLite", db_details=schema, external_knowledge=evidence,
            question=question["question"], reference_sql=question["SQL"],
            fewshot=True, max_prompt_tokens=16000, allow_over_budget=True,
            teacher_file=args.teacher_file,
        ) if not args.no_teacher else PB.build_messages(
            engine="SQLite", db_details=schema, external_knowledge=evidence,
            question=question["question"],
            fewshot=True, max_prompt_tokens=16000, allow_over_budget=True,
        )
        messages = list(built["messages"])
        transcript: List[Dict[str, str]] = []
        final = None
        errors: List[str] = []
        tool_calls = 0

        for _turn in range(args.max_turns):
            assistant = policy(messages)
            transcript.append({"role": "assistant", "content": assistant})
            try:
                parsed = T.parse_assistant_turn(
                    assistant, require_final_scratch_pad=False,
                    require_tool_scratch_pad=False,
                )
            except T.ProtocolError as exc:
                errors.append(str(exc))
                break
            if parsed["action"] == "final_answer":
                final = parsed
                break
            if tool_calls >= T.MAX_TOOL_CALLS:
                errors.append("tool-call budget exceeded")
                break
            observation = st.dispatch(
                db_file, parsed["name"], parsed["arguments"], db_id=db_id
            )
            tool_calls += 1
            transcript.append({"role": "observation", "content": observation})
            messages.append({"role": "assistant", "content": assistant})
            messages.append(
                {"role": "user", "content": f"<observation>\n{observation}\n</observation>"}
            )

        record: Dict[str, Any] = {
            "db_id": db_id,
            "question": question["question"],
            "evidence": evidence,
            "gold_sql": question["SQL"],
            "schema": schema,
            "transcript": transcript,
            "protocol_ok": final is not None and not errors,
            "final_sql": (final or {}).get("sql"),
            "errors": errors,
            "tool_calls": tool_calls,
        }
        # Track the executed sqlite_query calls so the Verified call index can be
        # resolved the same way run_episode does.
        executed_calls = []
        for body in TOOL_CALL.findall("\n".join(
            m["content"] for m in transcript if m["role"] == "assistant"
        )):
            name_match = re.search(r"<function=([A-Za-z_]+)", body)
            if not name_match:
                continue
            sql_match = re.search(r"<parameter=sql>\s*(.*?)\s*</parameter>", body, re.S)
            executed_calls.append(
                (name_match.group(1), sql_match.group(1) if sql_match else "")
            )
        record["verification"] = _verify_declared(final, executed_calls)
        if final and final.get("sql"):
            ex, reason, pred, gold = grade_one(
                db_file, final["sql"], question["SQL"], "multiset"
            )
            record.update({"ex": bool(ex), "ex_reason": reason,
                           "pred_rows": pred, "gold_rows": gold})
        else:
            record["ex"] = False
            record["ex_reason"] = "no final SQL produced"

        record["acceptance"] = check_trace(record, question["SQL"])
        records.append(record)
        print(
            f"[{index}/{len(sample)}] {db_id:<24} accepted={record['acceptance']['accepted']} "
            f"EX={record['ex']} calls={tool_calls} "
            f"issues={record['acceptance']['issues']}",
            flush=True,
        )

    accepted = sum(1 for r in records if r["acceptance"]["accepted"])
    ex = sum(1 for r in records if r["ex"])
    summary = {
        "model": args.model,
        "n": len(records),
        "accepted": accepted,
        "ex": ex,
        "issue_counts": {},
    }
    for record in records:
        for issue in record["acceptance"]["issues"]:
            summary["issue_counts"][issue] = summary["issue_counts"].get(issue, 0) + 1

    Path(args.out).write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("\n===== teacher probe =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
