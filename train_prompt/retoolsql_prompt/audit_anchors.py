#!/usr/bin/env python3
"""Audit which prompt sections actually fired during a run.

Each section of the v3 system prompt makes a falsifiable promise. This script
turns the raw per-turn ``anchors`` recorded by ``run_episode`` into compliance
rates for those promises, so a drop in EX can be attributed to a section rather
than guessed at.

Sections audited:

  §2 schema reading      — on a pruned schema, did the final SQL stay inside the
                           shown tables?
  §3.2 peek triggers     — how often peek was used, and on columns the final SQL
                           actually mentions
  §3.3 bm25 usage        — bm25 calls and whether they targeted a column the
                           final SQL filters on
  §4 turn protocol       — scratch pad presence, one-action rule, tag closure,
                           strict annotation pairing
  §5 decision policy     — after an error/empty observation, was the next action
                           a diagnostic rather than an identical retry?
  §6 output shape        — column-count mismatches and helper-column additions
  §7 verification        — was the submitted SQL executed verbatim?

Usage::

    python audit_anchors.py runs/v3_new_9b.json --out runs/audit_9b.json
"""

from __future__ import annotations

import argparse
import json
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

from run_ex import db_file_for, grade_one  # noqa: E402
from sql_grading import execute_query  # noqa: E402

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TABLE_LINE = re.compile(r"^# Table:\s*(.+?)\s*\|", re.M)
OMITTED = re.compile(r"^# Omitted tables:\s*(.+)$", re.M)


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).strip().lower()


def executed_sqls(record: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for message in record.get("transcript", []):
        if message.get("role") != "assistant":
            continue
        for body in TOOL_CALL.findall(message["content"]):
            match = re.search(r"<parameter=sql>\s*(.*?)\s*</parameter>", body, re.S)
            if match:
                out.append(match.group(1))
    return out


def verification_of(record: Dict[str, Any]) -> str:
    final = record.get("final_sql")
    if not final:
        return "no_final"
    final_norm = normalize_sql(final)
    calls = executed_sqls(record)
    if any(normalize_sql(sql) == final_norm for sql in calls):
        return "verbatim"
    db_file = db_file_for(record["db_id"])
    try:
        final_rows = frozenset(map(tuple, execute_query(db_file, final)))
    except Exception:
        return "never_executed"
    for sql in calls:
        try:
            if frozenset(map(tuple, execute_query(db_file, sql))) == final_rows:
                return "result_equivalent"
        except Exception:
            continue
    return "never_executed"


def final_sql_column_count(record: Dict[str, Any]) -> Optional[int]:
    rows = record.get("pred_rows")
    if not rows:
        return None
    return len(rows[0])


def audit(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    n = len(records)

    tool_turns = scratch_ok = 0
    final_turns = 0
    closed_finals = 0
    strict_pairs = 0
    budget_exhausted = 0
    tool_mix: Counter = Counter()
    peek_used = 0
    peek_relevant = 0
    bm25_used = 0
    bm25_relevant = 0
    error_obs = 0
    error_routed = 0
    error_retried_same = 0
    empty_obs = 0
    empty_routed = 0
    shape_more = 0
    shape_fewer = 0
    tier2 = 0
    tier2_clean = 0
    verification: Counter = Counter()
    repair_turns = 0
    repair_episodes = 0
    repair_ex = 0
    verified_declared = 0
    verified_matches = 0
    verification_reasons: Counter = Counter()
    zero_aggregate_episodes = 0
    zero_aggregate_distrusted = 0
    budget_episodes = 0
    budget_last_sql_correct = 0

    for record in records:
        anchors = record.get("anchors") or []
        verification[verification_of(record)] += 1
        if "budget" in " ".join(record.get("errors", [])):
            budget_exhausted += 1
            budget_episodes += 1
            last_sql = ""
            for fact in anchors:
                if fact.get("executed_tool") == "sqlite_query" and fact.get("executed_sql"):
                    last_sql = fact["executed_sql"]
            if last_sql and db_file_for(record["db_id"]):
                ok, _reason, _pred, _gold = grade_one(
                    db_file_for(record["db_id"]), last_sql, record["gold_sql"], "multiset"
                )
                if ok:
                    budget_last_sql_correct += 1

        repairs_here = [
            fact for fact in anchors if fact.get("repairs")
        ]
        if repairs_here:
            repair_episodes += 1
            repair_turns += len(repairs_here)
            if record.get("ex"):
                repair_ex += 1

        declared = (record.get("verification") or {}).get("declared")
        matches = (record.get("verification") or {}).get("matches")
        reason = (record.get("verification") or {}).get("reason") or ""
        if declared is not None or reason:
            verification_reasons[reason or "ok"] += 1
        if declared is not None:
            verified_declared += 1
            if matches:
                verified_matches += 1

        # zero-valued aggregate treated as a fault: an executed aggregate query
        # returned [[0]] / [[0.0]] and the model kept investigating
        saw_zero_aggregate = False
        for index, fact in enumerate(anchors):
            observation = fact.get("observation_after", "")
            executed = fact.get("executed_sql", "")
            if (
                observation == "rows"
                and executed
                and re.search(r"\b(count|sum|avg|min|max)\s*\(", executed, re.I)
                and index + 1 < len(anchors)
                and anchors[index + 1].get("action") == "tool_call"
            ):
                saw_zero_aggregate = True
        if saw_zero_aggregate:
            zero_aggregate_episodes += 1
            if not record.get("ex"):
                zero_aggregate_distrusted += 1

        final_sql = record.get("final_sql") or ""
        final_norm = normalize_sql(final_sql)

        for index, fact in enumerate(anchors):
            action = fact.get("action")
            if action == "tool_call":
                tool_turns += 1
                if fact.get("has_scratch"):
                    scratch_ok += 1
                tool = fact.get("tool", "")
                tool_mix[tool] += 1
                if tool == "sqlite_peek":
                    peek_used += 1
                    columns = fact.get("peek_columns") or []
                    if isinstance(columns, str):
                        columns = [c.strip() for c in columns.split(",")]
                    if any(str(c).lower() in final_sql.lower() for c in columns):
                        peek_relevant += 1
                if tool == "bm25_search_sqlite":
                    bm25_used += 1
                    column = str(fact.get("bm25_column", "")).lower()
                    if column and column in final_sql.lower():
                        bm25_relevant += 1

                previous = fact.get("observation_kind", "")
                if previous == "error":
                    error_obs += 1
                    if tool in ("sqlite_peek", "bm25_search_sqlite") or (
                        fact.get("sql") and normalize_sql(fact["sql"]) != final_norm
                    ):
                        error_routed += 1
                    if any(
                        normalize_sql(fact.get("sql", "")) == normalize_sql(a.get("sql", ""))
                        for a in anchors[:index] if a.get("action") == "tool_call" and a.get("sql")
                    ):
                        error_retried_same += 1
                elif previous == "empty":
                    empty_obs += 1
                    if tool in ("sqlite_peek", "bm25_search_sqlite"):
                        empty_routed += 1
                    elif any(
                        normalize_sql(fact.get("sql", "")) != normalize_sql(a.get("sql", ""))
                        for a in anchors[:index] if a.get("action") == "tool_call" and a.get("sql")
                    ):
                        empty_routed += 1

            elif action == "final_answer":
                final_turns += 1
                if fact.get("final_tags_closed"):
                    closed_finals += 1
                if fact.get("annotation_pairing_strict"):
                    strict_pairs += 1

        # §2: pruned schemas
        if (record.get("schema_tier") or 0) >= 2:
            tier2 += 1
            omitted = record.get("omitted_tables") or []
            if not any(name.lower() in final_sql.lower() for name in omitted):
                tier2_clean += 1

        # §6: shape
        if not record.get("ex"):
            pred, gold = record.get("pred_rows"), record.get("gold_rows")
            if pred and gold:
                if len(pred[0]) > len(gold[0]):
                    shape_more += 1
                elif len(pred[0]) < len(gold[0]):
                    shape_fewer += 1

    def rate(num: int, den: int) -> str:
        return f"{num}/{den} ({num / den:.1%})" if den else f"0/0 (-)"

    return {
        "file": path.name,
        "n": n,
        "sections": {
            "section_4_turn_protocol": {
                "tool_turns_with_scratch": rate(scratch_ok, tool_turns),
                "final_turns_tags_closed": rate(closed_finals, final_turns),
                "final_turns_strict_annotation_pairing": rate(strict_pairs, final_turns),
            },
            "section_5_decision_policy": {
                "after_error_observation": error_obs,
                "routed_to_diagnostic_or_new_sql": rate(error_routed, error_obs),
                "identical_retry_after_error": error_retried_same,
                "after_empty_observation": empty_obs,
                "routed_to_diagnostic_or_new_sql_after_empty": rate(empty_routed, empty_obs),
            },
            "section_3_2_peek_trigger": {
                "peek_calls": peek_used,
                "peek_columns_used_by_final_sql": rate(peek_relevant, peek_used),
            },
            "section_3_3_bm25": {
                "bm25_calls": bm25_used,
                "bm25_column_used_by_final_sql": rate(bm25_relevant, bm25_used),
            },
            "section_2_schema_filter": {
                "pruned_episodes": tier2,
                "final_sql_stays_inside_shown_schema": rate(tier2_clean, tier2),
            },
            "section_6_output_shape": {
                "extra_columns": shape_more,
                "missing_columns": shape_fewer,
            },
            "section_7_verification": {
                "budget_exhausted": budget_exhausted,
                **{f"final_sql_{k}": v for k, v in verification.items()},
                "verbatim_rate": rate(
                    verification.get("verbatim", 0),
                    verification.get("verbatim", 0)
                    + verification.get("result_equivalent", 0)
                    + verification.get("never_executed", 0),
                ),
            },
            "section_7b_verified_call": {
                "episodes_declaring_index": verified_declared,
                "declared_index_matches_final_sql": rate(verified_matches, verified_declared),
                "reasons": dict(verification_reasons),
            },
            "section_4b_tool_call_repairs": {
                "episodes_with_any_repair": repair_episodes,
                "repaired_turns": repair_turns,
                "repaired_episodes_that_pass_EX": repair_ex,
            },
            "section_5b_budget_spirals": {
                "budget_episodes": budget_episodes,
                "last_executed_sql_correct": rate(budget_last_sql_correct, budget_episodes),
                "episodes_with_zero_valued_aggregate_then_more_probing": zero_aggregate_episodes,
                "of_those_failing": zero_aggregate_distrusted,
            },
            "tool_mix": dict(tool_mix),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    reports = [audit(Path(f)) for f in args.files]
    for report in reports:
        print("=" * 78)
        print(report["file"], f"(n={report['n']})")
        for section, values in report["sections"].items():
            print(f"  {section}")
            for key, value in values.items():
                print(f"    {key:52s} {value}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
