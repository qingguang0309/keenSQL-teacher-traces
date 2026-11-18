#!/usr/bin/env python3
"""L3b: tool-using adjudication of routed trajectories (轨迹审核Agent执行方案.md §3.3).

Takes the items that L3a routed to "L3B" (the two judges disagree on PASS vs
not-PASS, weak EX evidence, or one judge suspects the gold). One adjudicator
model sees the same rendered trajectory, both L3a verdicts, and may run up to
MAX_TOOL_CALLS read-only queries on the real database, through the same
sqlite_tools.dispatch the agent used. It ends by calling submit_verdict with
the L3a JSON schema plus `agrees_with`.

Subcommands
  run    adjudicate routed items (resumable, circuit breaker)
  final  merge L3a routes and L3b verdicts into one decision per item, plus report

trace_pipeline_v1 changes against train_prompt/trace_audit_l3/audit_l3b.py (GAIA "eliminate
a candidate only with explicit evidence against it"):
  - the adjudicator is told that trajectory observations are replayed facts, so a probe can
    never be used to call them log artifacts (the one planted error that escaped in the
    395-trace batch was dismissed that way);
  - overturning a judge's anchored non-PASS needs an `overturns` entry with a verbatim
    refutation quote from the trajectory or a probe; `final` verifies the quote and, when it
    is missing or not found, keeps the most severe anchored judge verdict instead;
  - findings carry verified quotes; probes keep up to OBS_CHARS of output so quotes match;
  - each decision gets trajectory_verdict, sft_eligible and rl_exclude_question (user
    decision 2026-11-12: a correct trajectory on a wrong-gold question stays in SFT, the
    question leaves RL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/data/keenSQL/database_tools")
import audit_l3 as L3  # noqa: E402
import sqlite_tools as st  # noqa: E402

ADJUDICATOR = "moonshotai/kimi-k3"
MAX_TOOL_CALLS = 6
MAX_TURNS = 10
OBS_CHARS = 3000
# Per-item spend caps (GAIA G3 runaway lesson): in the end-to-end check one item ran 7 turns and
# 62k completion tokens ($1.33). Past either cap the adjudicator must submit on its next turn.
MAX_COMPLETION_TOKENS_PER_ITEM = 30000
MAX_COST_PER_ITEM = 0.35

ADJUDICATION = """
# Your task in this round: adjudicate

Two independent judges already reviewed this trajectory. Their verdicts are in
`L3A_VERDICTS`. They disagree, or the trajectory has weak EX evidence (the gold
result is empty or a single 0/1/NULL, so a wrong query could match by chance),
or one of them suspects the gold SQL is wrong.

You can run read-only queries on the same database through the tools
`sqlite_query`, `sqlite_peek` and `bm25_search_sqlite` (at most {max_calls}
calls). Use them to test the disputed claims, for example:
- whether the final SQL would stay correct if the data were different (probe
  the rows a missing filter would remove, or what a different join returns);
- whether a stored literal really exists, and whether the agent could have seen
  it;
- whether the gold SQL answers the question (run a corrected version and
  compare).

Then call `submit_verdict` exactly once with the JSON object described above,
plus `"agrees_with": "kimi" | "deepseek" | "both" | "neither"`. Base the
verdict on what you verified. Cite tool results you ran yourself as
`probe#<n>` (n = 1 for your first query).

Rules for overturning a judge (they decide whether your verdict is accepted):
1. The trajectory's observations were replayed against the database and match
   exactly what the agent saw. They are facts about the trajectory, never "log
   artifacts" or "environment glitches". A probe shows what the database returns
   to you; it cannot show that the agent saw something else. Judge the agent's
   reasoning against the observations in the trajectory as written.
2. A judge's finding stands unless you refute it with explicit evidence. If you
   give PASS while a judge gave a non-PASS verdict, add one entry per refuted
   finding to `overturns`: {{"judge": "kimi"|"deepseek", "code": "<its code>",
   "refutation_cite": "<scratch#k | observation#k | call#k.sql | question |
   evidence | schema:... | probe#n>", "refutation_quote": "<verbatim span from
   that source that directly contradicts the finding>"}}. Quotes are checked
   mechanically; an overturn without a verifiable quote is not applied, and the
   judge's verdict is kept.
3. Every finding you submit needs a verbatim `quote` from its cited source,
   exactly as in the first round. Report `confidence` (0-100) as well.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "sqlite_query",
        "description": "Run one read-only SELECT/WITH on the trajectory's database.",
        "parameters": {"type": "object", "properties": {
            "sql": {"type": "string"},
            "max_return_rows": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "sqlite_peek",
        "description": "Profile columns of a table: types, nulls, distinct, top values, samples.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "where": {"type": "string"}},
            "required": ["table"]}}},
    {"type": "function", "function": {
        "name": "bm25_search_sqlite",
        "description": "Find stored values of table.column similar to a phrase.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "column": {"type": "string"},
            "query": {"type": "string"}, "top_k": {"type": "integer"},
            "where": {"type": "string"}},
            "required": ["table", "column", "query"]}}},
    {"type": "function", "function": {
        "name": "submit_verdict",
        "description": "Submit the final adjudication. Call exactly once, last.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "REGEN", "DROP"]},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "findings": {"type": "array", "items": {"type": "object"}},
            "decisions": {"type": "array", "items": {"type": "object"}},
            "gold_suspect": {"type": "boolean"},
            "gold_issue": {"type": "string"},
            "is_ambiguous": {"type": "boolean"},
            "agrees_with": {"type": "string", "enum": ["kimi", "deepseek", "both", "neither"]},
            "overturns": {"type": "array", "items": {"type": "object", "properties": {
                "judge": {"type": "string"}, "code": {"type": "string"},
                "refutation_cite": {"type": "string"}, "refutation_quote": {"type": "string"}}}},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "explanation": {"type": "string"}},
            "required": ["verdict", "reason_codes", "gold_suspect", "agrees_with", "confidence", "explanation"]}}},
]


def l3a_summary(row: dict[str, Any]) -> str:
    out = {}
    for name in L3.JUDGES:
        v = row.get(name) or {}
        out[name] = {
            "verdict": v.get("policy_verdict"),
            "reason_codes": v.get("codes"),
            "findings": v.get("findings", []),
            "rejected_findings": v.get("quote_problems", []),
            "gold_suspect": v.get("gold_suspect"),
            "gold_issue": v.get("gold_issue"),
            "confidence": v.get("confidence"),
            "explanation": v.get("explanation"),
        }
    return json.dumps(out, ensure_ascii=False, indent=2)


def adjudicate(client, item: dict[str, Any], agg_row: dict[str, Any], system: str, max_tokens: int) -> dict[str, Any]:
    db_file = L3.DB_ROOT / item["db_id"] / f"{item['db_id']}.sqlite"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": L3.render(item) + "\n\n## L3A_VERDICTS\n\n" + l3a_summary(agg_row)},
    ]
    record: dict[str, Any] = {"id": item["id"], "model": ADJUDICATOR, "probes": [], "usage": []}
    calls = 0
    for _turn in range(MAX_TURNS):
        spent_tokens = sum(int(u.get("completion_tokens") or 0) for u in record["usage"])
        spent_cost = sum(float(u.get("cost") or 0) for u in record["usage"])
        over_budget = spent_tokens >= MAX_COMPLETION_TOKENS_PER_ITEM or spent_cost >= MAX_COST_PER_ITEM
        if over_budget and not record.get("budget_forced"):
            record["budget_forced"] = {"completion_tokens": spent_tokens, "cost_usd": round(spent_cost, 4)}
            messages.append({"role": "user", "content": "Budget for this adjudication is used up. "
                             "Call submit_verdict now with the verdict your evidence so far supports."})
        must_submit = calls >= MAX_TOOL_CALLS or over_budget
        response = client.chat.completions.create(
            model=ADJUDICATOR,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit_verdict"}} if must_submit else "auto",
            parallel_tool_calls=False,
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"usage": {"include": True}},
        )
        record["usage"].append(response.usage.model_dump() if response.usage else {})
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        tool_calls = message.tool_calls or []
        if not tool_calls:
            # A plain-text reply: ask for the tool call once more.
            messages.append({"role": "user", "content": "Call submit_verdict now with your JSON verdict."})
            continue
        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                arguments, bad = {}, f"invalid JSON arguments: {exc}"
            else:
                bad = None
            if name == "submit_verdict" and bad is None:
                record["verdict_json"] = arguments
                record["tool_calls"] = calls
                return record
            if bad is None and name in {"sqlite_query", "sqlite_peek", "bm25_search_sqlite"} and calls >= MAX_TOOL_CALLS:
                # Every tool_call id needs a reply; calls past the budget are not run.
                observation = json.dumps({"error": f"tool budget of {MAX_TOOL_CALLS} calls used; call submit_verdict"})
            elif bad is None and name in {"sqlite_query", "sqlite_peek", "bm25_search_sqlite"}:
                calls += 1
                if name == "sqlite_query":
                    arguments["max_return_rows"] = min(int(arguments.get("max_return_rows") or 20), 50)
                observation = st.dispatch(str(db_file), name, arguments, db_id=item["db_id"])[:OBS_CHARS]
                record["probes"].append({"n": calls, "tool": name, "arguments": arguments,
                                         "observation": observation})
            else:
                observation = json.dumps({"error": bad or f"unknown tool {name}"})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
    record["verdict_json"] = None
    record["tool_calls"] = calls
    return record


def cmd_run(args) -> int:
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=args.timeout, max_retries=2)
    system = L3.PROMPT_PATH.read_text(encoding="utf-8") + ADJUDICATION.format(max_calls=MAX_TOOL_CALLS)
    system = system.replace("Reply with one JSON object and nothing else:",
                            "Your verdict object (passed to submit_verdict) has this shape:")
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8")}
    agg = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.l3a_dir) / "aggregate.jsonl", encoding="utf-8")}
    routed = [i for i, row in agg.items() if row.get("route") == "L3B"]
    out = Path(args.out)
    done = set()
    if out.is_file():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if json.loads(l).get("verdict_json")}
    todo = [i for i in routed if i not in done]
    print(f"[l3b] {len(routed)} routed, {len(done)} done, {len(todo)} to run", flush=True)
    breaker, lock, cost, finished = L3.CircuitBreaker(args.breaker), threading.Lock(), 0.0, 0
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(adjudicate, client, items[i], agg[i], system, args.max_tokens): i for i in todo}
        for future in as_completed(futures):
            if breaker.open:
                continue
            try:
                record = future.result()
                breaker.record(True)
            except Exception as exc:  # noqa: BLE001
                breaker.record(False)
                print(f"[l3b] API error on {futures[future]}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                if breaker.open:
                    print("[l3b] circuit breaker open; cancelling the rest", flush=True)
                    for other in futures:
                        other.cancel()
                continue
            cost += sum(float(u.get("cost") or 0) for u in record["usage"])
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
            if finished % 10 == 0:
                print(f"[l3b] {finished}/{len(todo)} done, cost so far ${cost:.3f}", flush=True)
    print(f"[l3b] finished {finished}/{len(todo)}; cost ${cost:.3f}", flush=True)
    return 2 if breaker.open else 0


def cmd_final(args) -> int:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8")}
    agg = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.l3a_dir) / "aggregate.jsonl", encoding="utf-8")}
    l3b = {}
    if Path(args.l3b).is_file():
        for line in open(args.l3b, encoding="utf-8"):
            row = json.loads(line)
            if row.get("verdict_json"):
                l3b[row["id"]] = row
    decisions, cost = [], 0.0
    for item_id, row in agg.items():
        route = row["route"]
        item = items[item_id]
        decision = {"id": item_id, "l3a_route": route, "route_reasons": row.get("route_reasons", []),
                    "corruption": row.get("corruption"), "rule_tags": row["rule_tags"]}
        judges = {n: row.get(n) for n in L3.JUDGES if row.get(n) and row[n].get("ok")}
        if route == "L3B":
            record = l3b.get(item_id)
            if record is None:
                decision["final"] = trajectory = "PENDING_L3B"
            else:
                cost += sum(float(u.get("cost") or 0) for u in record["usage"])
                verdict = record["verdict_json"]
                probes = {f"probe#{p['n']}": p.get("observation", "") for p in record.get("probes", [])}
                v = L3.normalize(verdict, item, quote_policy=args.quote_policy, extra_sources=probes)
                trajectory = v["supported_policy"] if args.quote_policy == "enforce" else v["policy_verdict"]
                # An overturn of an anchored judge finding needs a verified refutation quote.
                kept = []
                if trajectory == "PASS" and args.quote_policy == "enforce":
                    overturned = {}
                    for entry in verdict.get("overturns") or []:
                        if not isinstance(entry, dict):
                            continue
                        probe = {"cite": entry.get("refutation_cite", ""), "quote": entry.get("refutation_quote", "")}
                        if L3.check_quote(probe, item, probes) is None and str(probe["cite"]).strip() not in ("", "none"):
                            overturned.setdefault(entry.get("judge"), set()).add(entry.get("code"))
                    for name, jv in judges.items():
                        if jv.get("supported_policy", jv["policy_verdict"]) == "PASS":
                            continue
                        anchored_codes = {f["code"] for f in jv.get("findings", [])
                                          if f.get("code") in L3.CODES and L3.CODES[f["code"]] != "PASS"
                                          and L3.check_quote(f, item) is None}
                        if not anchored_codes <= overturned.get(name, set()):
                            kept.append((name, jv.get("supported_policy", jv["policy_verdict"]), sorted(anchored_codes - overturned.get(name, set()))))
                    if kept:
                        trajectory = max((k[1] for k in kept), key=L3.SEVERITY.get)
                decision["l3b"] = {**v, "agrees_with": verdict.get("agrees_with"),
                                   "probes": len(record.get("probes", [])),
                                   "overturns": verdict.get("overturns") or [],
                                   "unrefuted_judge_findings": kept}
                decision["final"] = "DROP_GOLD" if v["gold_suspect"] else trajectory
        else:
            decision["final"] = route
            if route == "DROP_GOLD":
                trajectory = max((jv.get("supported_policy", jv["policy_verdict"]) if args.quote_policy == "enforce" else jv["policy_verdict"]
                                  for jv in judges.values()), key=L3.SEVERITY.get, default="PENDING_L3B")
            else:
                trajectory = route
        escaped = bool(row["rule_tags"].get("html_escaped_sql"))
        # Rule-level drop agreed earlier: HTML-escaped SQL teaches a format slip.
        if decision["final"] == "PASS" and escaped:
            decision["final"] = "DROP_FORMAT"
        decision["trajectory_verdict"] = trajectory
        decision["sft_eligible"] = bool(trajectory == "PASS" and not escaped and not row.get("corruption"))
        decision["rl_exclude_question"] = decision["final"] == "DROP_GOLD"
        decisions.append(decision)
    out = Path(args.out_dir)
    with (out / "final_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for d in decisions:
            handle.write(json.dumps(d, ensure_ascii=False) + "\n")
    real = [d for d in decisions if not d["corruption"]]
    planted = [d for d in decisions if d["corruption"]]
    report = {
        "real_traces": len(real),
        "final_real": dict(Counter(d["final"] for d in real)),
        "l3a_route_real": dict(Counter(d["l3a_route"] for d in real)),
        "planted_errors": len(planted),
        "planted_final": dict(Counter(d["final"] for d in planted)),
        "planted_escaped_rate": round(sum(d["final"] == "PASS" for d in planted) / max(1, len(planted)), 3),
        "sft_eligible_real": sum(d["sft_eligible"] for d in real),
        "sft_eligible_gold_bad": sum(d["sft_eligible"] and d["final"] == "DROP_GOLD" for d in real),
        "rl_exclude_questions": sum(d["rl_exclude_question"] for d in real),
        "l3b_overturns_rejected": sum(bool(d.get("l3b", {}).get("unrefuted_judge_findings")) for d in decisions),
        "l3b_budget_forced": sum(bool(r.get("budget_forced")) for r in l3b.values()),
        "l3b_max_item_cost_usd": round(max((sum(float(u.get("cost") or 0) for u in r["usage"])
                                            for r in l3b.values()), default=0.0), 3),
        "l3b_adjudicated": len(l3b),
        "l3b_agrees_with": dict(Counter(d["l3b"]["agrees_with"] for d in decisions if d.get("l3b"))),
        "l3b_cost_usd": round(cost, 3),
    }
    (out / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--items", required=True)
    p.add_argument("--l3a-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--breaker", type=int, default=5)
    p = sub.add_parser("final")
    p.add_argument("--items", required=True)
    p.add_argument("--l3a-dir", required=True)
    p.add_argument("--l3b", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quote-policy", choices=("enforce", "report", "off"), default="enforce")
    args = parser.parse_args()
    return {"run": cmd_run, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
