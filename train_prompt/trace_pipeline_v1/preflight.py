#!/usr/bin/env python3
"""G0 preflight: check every model and resource a run will use, the way the run uses it.

Borrowed from MiroFlow GAIA scripts_sol/preflight_api.py (lesson E35: separate key probes
were green while the pipeline sent the wrong key). Our own reasons: OpenRouter dropped
`stop` for GPT-5.6 Sol silently (13.4% fabricated turns), deepseek judged at 24k tokens
hit finish_reason=length, and OpenAI models return 403 from the cluster's region.

Checks (each is PASS, WARN or FAIL; any FAIL exits 1 unless waived with --waive NAME=reason):
  families        the teacher's family differs from every judge and the adjudicator
  models_listed   every model is listed by OpenRouter /models
  teacher_params  `stop`/`temperature` support (missing -> WARN: the client clip and
                  temperature override carry the behavior)
  judge_params    response_format support (missing -> WARN: the parser has a fallback)
  adjud_params    tools + tool_choice (missing -> FAIL: L3b cannot run)
  teacher_live    one real turn through build_traces_v2 (same client, clip, entity repair,
                  parser): the reply must end at </tool_call>, contain no fabricated
                  observation, have no &gt;/&lt; left, and parse
  judge_live      one json_object request per judge
  adjud_live      one forced tool call
  inputs          every input record has its database and M-Schema cache file
  balance         OpenRouter limit_remaining >= floor + budget

The live probes cost well under $0.05 together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

DEFAULT_JUDGES = {"kimi": "moonshotai/kimi-k3", "deepseek": "deepseek/deepseek-v4-pro"}
DEFAULT_ADJUDICATOR = "moonshotai/kimi-k3"

TEACHER_PROBE = (
    "<scratch_pad>\nCount the rows of table t whose value exceeds 5.\n</scratch_pad>\n"
    "<tool_call>\n<function=sqlite_query>\n<parameter=sql>\n"
    "SELECT COUNT(*) FROM t WHERE value &gt; 5\n</parameter>\n</function>\n</tool_call>\n"
    "<observation>\nFAKE_OBSERVATION_7731\n</observation>"
)


class Report:
    def __init__(self, waivers: dict[str, str]) -> None:
        self.checks: list[dict[str, Any]] = []
        self.waivers = waivers

    def add(self, name: str, status: str, detail: Any) -> None:
        waived = status == "FAIL" and name in self.waivers
        self.checks.append({"check": name, "status": "WAIVED" if waived else status,
                            "detail": detail, **({"waiver": self.waivers[name]} if waived else {})})
        print(f"[preflight] {name:<15} {('WAIVED' if waived else status):<6} "
              f"{json.dumps(detail, ensure_ascii=False)[:300]}", flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def check_families(rep: Report, teacher: str, judges: dict[str, str], adjudicator: str) -> None:
    tf = C.family(teacher)
    clash = {name: m for name, m in {**judges, "adjudicator": adjudicator}.items() if C.family(m) == tf}
    rep.add("families", "FAIL" if clash else "PASS",
            {"teacher_family": tf, "same_family_reviewers": clash})


def check_params(rep: Report, teacher: str, judges: dict[str, str], adjudicator: str) -> None:
    try:
        models = C.openrouter_models()
    except Exception as exc:  # noqa: BLE001
        rep.add("models_listed", "FAIL", f"GET /models failed: {exc}")
        return
    wanted = {teacher, adjudicator, *judges.values()}
    missing = sorted(m for m in wanted if "/" in m and m not in models)
    rep.add("models_listed", "FAIL" if missing else "PASS", {"missing": missing})
    params = lambda m: set((models.get(m) or {}).get("supported_parameters") or [])
    if teacher in models:
        tp = params(teacher)
        notes = []
        if "stop" not in tp:
            notes.append("no `stop`: provider ignores it, client-side clip at </tool_call> is required (active)")
        if "temperature" not in tp:
            notes.append("no `temperature`: greedy == provider default sampling; resampling still differs")
        if "max_tokens" not in tp and "max_completion_tokens" not in tp:
            rep.add("teacher_params", "FAIL", "no max_tokens support")
        else:
            rep.add("teacher_params", "WARN" if notes else "PASS", notes or sorted(tp & {"stop", "temperature", "max_tokens"}))
    for name, model in judges.items():
        if model in models:
            jp = params(model)
            ok = bool(jp & {"response_format", "structured_outputs"})
            rep.add(f"judge_params:{name}", "PASS" if ok else "WARN",
                    "json mode supported" if ok else "no response_format: relying on JSON parse fallback")
    if adjudicator in models:
        ap = params(adjudicator)
        need = {"tools", "tool_choice"} - ap
        rep.add("adjud_params", "FAIL" if need else "PASS", {"missing": sorted(need)})


def check_teacher_live(rep: Report, teacher: str, key: str) -> None:
    os.environ.setdefault("SFT_API_PROVIDER", "openrouter")
    os.environ.setdefault("SFT_THINKING_MODE", "omit")
    try:
        import build_traces_v2 as G  # noqa: WPS433 - heavy import, only for this probe
    except Exception as exc:  # noqa: BLE001
        rep.add("teacher_live", "FAIL", f"cannot import build_traces_v2: {type(exc).__name__}: {exc}")
        return
    messages = [
        {"role": "system", "content": "You are a copy machine. Reply with the user's text exactly, character for character, and nothing else."},
        {"role": "user", "content": TEACHER_PROBE},
    ]
    G._TLS.tele = G.new_telemetry()
    started = time.time()
    try:
        text = G.core.request_completion(
            messages=messages, base_url=C.OPENROUTER, api_key=key, model=teacher,
            temperature=0.0, max_tokens=400, timeout=120, seed=0, chat_template_kwargs=None)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = " (region block: run on aws-asia)" if "403" in msg or "region" in msg.lower() else ""
        rep.add("teacher_live", "FAIL", f"{type(exc).__name__}: {msg[:200]}{hint}")
        return
    tele = G._TLS.tele
    G._TLS.tele = None
    problems = []
    if not text.rstrip().endswith("</tool_call>"):
        problems.append("reply does not end at </tool_call>")
    if "FAKE_OBSERVATION" in text:
        problems.append("fabricated observation survived the clip")
    if "&gt;" in text or "&lt;" in text:
        problems.append("HTML entity survived the repair")
    error = G.protocol_error(text)
    if error:
        problems.append(f"parser: {error}")
    detail = {"elapsed_s": round(time.time() - started, 1), "clipped_turns": tele["clipped_turns"],
              "closed_at_stop": tele["closed_at_stop"], "entity_repairs": tele["entity_repairs"],
              "resampled_turns": tele["resampled_turns"], "finish_reasons": tele["finish_reasons"],
              "cost_usd": round(tele["cost_usd"], 5),
              "stop_honored": tele["clipped_turns"] == 0}
    if problems:
        detail["problems"] = problems
        detail["reply_tail"] = text[-200:]
    rep.add("teacher_live", "FAIL" if problems else "PASS", detail)


def _client(key: str):
    from openai import OpenAI
    return OpenAI(base_url=C.OPENROUTER, api_key=key, timeout=120, max_retries=1)


def check_judges_live(rep: Report, judges: dict[str, str], key: str) -> None:
    client = _client(key)
    for name, model in judges.items():
        try:
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=2000,
                response_format={"type": "json_object"}, extra_body={"usage": {"include": True}},
                messages=[{"role": "system", "content": "Reply with one JSON object only."},
                          {"role": "user", "content": 'Return {"ok": true, "verdict": "PASS"}.'}])
            content = response.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}")
            parsed = json.loads(content[start: end + 1])
            ok = parsed.get("ok") is True
            rep.add(f"judge_live:{name}", "PASS" if ok else "FAIL",
                    {"finish_reason": response.choices[0].finish_reason,
                     "cost_usd": (response.usage.model_dump() if response.usage else {}).get("cost")})
        except Exception as exc:  # noqa: BLE001
            rep.add(f"judge_live:{name}", "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")


def check_adjudicator_live(rep: Report, model: str, key: str) -> None:
    tool = {"type": "function", "function": {
        "name": "submit_verdict", "description": "Submit the verdict.",
        "parameters": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["PASS", "DROP"]}},
                       "required": ["verdict"]}}}
    try:
        response = _client(key).chat.completions.create(
            model=model, temperature=0, max_tokens=2000, tools=[tool], parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
            extra_body={"usage": {"include": True}},
            messages=[{"role": "user", "content": "Submit the verdict PASS."}])
        calls = response.choices[0].message.tool_calls or []
        ok = bool(calls) and calls[0].function.name == "submit_verdict"
        rep.add("adjud_live", "PASS" if ok else "FAIL",
                {"tool_calls": len(calls), "cost_usd": (response.usage.model_dump() if response.usage else {}).get("cost")})
    except Exception as exc:  # noqa: BLE001
        rep.add("adjud_live", "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")


def check_inputs(rep: Report, input_path: Path) -> None:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    missing_db, missing_schema = set(), set()
    for record in records:
        db = str(record["db_id"])
        if not (C.DB_ROOT / db / f"{db}.sqlite").is_file() and len(list((C.DB_ROOT / db).glob("*.sqlite"))) != 1:
            missing_db.add(db)
        if not (C.SCHEMA_CACHE / f"{db}.txt").is_file():
            missing_schema.add(db)
    bad = bool(missing_db or missing_schema)
    rep.add("inputs", "FAIL" if bad else "PASS",
            {"records": len(records), "dbs": len({r['db_id'] for r in records}),
             "missing_db": sorted(missing_db)[:20], "missing_schema": sorted(missing_schema)[:20]})


def check_balance(rep: Report, key: str, floor: float, budget: float) -> None:
    try:
        bal = C.openrouter_balance(key)
    except Exception as exc:  # noqa: BLE001
        rep.add("balance", "FAIL", f"GET /key failed: {exc}")
        return
    remaining = bal["limit_remaining"]
    if remaining is None:
        rep.add("balance", "WARN", {"note": "key has no limit; cannot enforce floor", **bal})
        return
    need = floor + budget
    rep.add("balance", "PASS" if remaining >= need else "FAIL",
            {"limit_remaining": round(remaining, 2), "floor": floor, "budget": budget, "needed": need})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, help="OpenRouter model id, e.g. openai/gpt-5.6-sol")
    parser.add_argument("--judges", default=",".join(f"{k}={v}" for k, v in DEFAULT_JUDGES.items()))
    parser.add_argument("--adjudicator", default=DEFAULT_ADJUDICATOR)
    parser.add_argument("--input", type=Path, default=None, help="easy_current.json of the run")
    parser.add_argument("--floor", type=float, default=15.0)
    parser.add_argument("--budget", type=float, default=0.0, help="expected spend of the run in USD")
    parser.add_argument("--no-live", action="store_true", help="skip the paid live probes")
    parser.add_argument("--waive", action="append", default=[], help="CHECK=reason")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    waivers = dict(w.split("=", 1) for w in args.waive)
    judges = dict(item.split("=", 1) for item in args.judges.split(",") if item)
    key = C.openrouter_key()
    rep = Report(waivers)
    check_families(rep, args.teacher, judges, args.adjudicator)
    check_params(rep, args.teacher, judges, args.adjudicator)
    if args.input:
        check_inputs(rep, args.input)
    check_balance(rep, key, args.floor, args.budget)
    if not args.no_live:
        check_teacher_live(rep, args.teacher, key)
        check_judges_live(rep, judges, key)
        check_adjudicator_live(rep, args.adjudicator, key)
    result = {"created_at_utc": C.utc_now(), "teacher": args.teacher, "judges": judges,
              "adjudicator": args.adjudicator, "passed": not rep.failed, "checks": rep.checks}
    if args.out:
        C.write_json(args.out, result)
    print(f"[preflight] {'PASSED' if not rep.failed else 'FAILED'}", flush=True)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
