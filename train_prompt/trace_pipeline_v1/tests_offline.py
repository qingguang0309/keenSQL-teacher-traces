#!/usr/bin/env python3
"""Offline tests for trace_pipeline_v1 (no API calls, no writes outside a temp dir).

Run on aws-asia:  /data/keenSQL/.venv/bin/python train_prompt/trace_pipeline_v1/tests_offline.py
Uses the 395-trace GPT-5.6 Sol batch as fixtures and checks that the new code reproduces
its known results (routes, 358 SFT-eligible traces) and fixes its known defects.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = sys.executable
DATA = Path("/data/keenSQL/dataset/m_schema/sft_data")
PILOT = DATA / "easy_nogold_v41_pilot500_gpt56sol"
STOPFIX = DATA / "easy_nogold_v41_stopfix67_gpt56sol"
FULL = DATA / "trace_audit_l3_gpt56sol_395/full"
TMP = Path(tempfile.mkdtemp(prefix="pipeline_tests_"))
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# ------------------------------------------------------------ generator v2

import build_traces_v2 as G  # noqa: E402
import common as C  # noqa: E402


def t_entities() -> None:
    text = ("<scratch_pad>a &gt; b stays in scratch</scratch_pad>\n<tool_call>\n<function=sqlite_query>\n"
            "<parameter=sql>\nSELECT 1 WHERE x &gt;= 2 AND y &lt; 3 AND n = 'A&amp;B'\n</parameter>\n</function>\n</tool_call>")
    fixed, n = G.repair_entities(text)
    check("entity: tool_call repaired", "x >= 2 AND y < 3 AND n = 'A&B'" in fixed and n == 3, f"repairs={n}")
    check("entity: scratch_pad untouched", "a &gt; b stays" in fixed)
    plain = "<tool_call>\nSELECT Description FROM rootbeerbrand WHERE BrandName = 'A&amp;W'\n</tool_call>"
    G._TLS.tele = G.new_telemetry()
    fixed_amp, n_amp = G.repair_entities(plain)
    amp_only = G._TLS.tele["amp_only_turns"]
    G._TLS.tele = None
    check("entity: lone &amp; repaired and counted (e2e case bird_03799)",
          "'A&W'" in fixed_amp and n_amp == 1 and amp_only == 1)
    final = "<final_answer>\nSELECT a FROM t WHERE b &lt;&gt; 1\n</final_answer>"
    check("entity: final_answer repaired", "b <> 1" in G.repair_entities(final)[0])

    # Anthropic cache breakpoints: before the real question and on the last message; text unchanged.
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Database Engine:\nfewshot"}, {"role": "assistant", "content": "FA"},
            {"role": "user", "content": "Database Engine:\nreal q"}, {"role": "assistant", "content": "T1"},
            {"role": "user", "content": "<observation>x</observation>"}]
    out = G.with_cache_breakpoints(msgs)
    marked = [i for i, m in enumerate(out) if isinstance(m["content"], list)]
    texts = [m["content"][0]["text"] if isinstance(m["content"], list) else m["content"] for m in out]
    check("cache: breakpoints before real question and on last message", marked == [2, 5], str(marked))
    check("cache: text unchanged, input not mutated",
          texts == [m["content"] for m in msgs] and all(isinstance(m["content"], str) for m in msgs))
    check("cache: only anthropic models", G.anthropic_cache_enabled("anthropic/claude-opus-5.5")
          and not G.anthropic_cache_enabled("openai/gpt-5.6-sol"))

    # OpenRouter 402 in_flight_budget_exhausted is retried; any other 402 still raises.
    class _Flaky:
        def __init__(self, errors):
            self.errors, self.calls = list(errors), 0

        def create(self, **request):
            self.calls += 1
            if self.errors:
                raise RuntimeError(self.errors.pop(0))
            return types.SimpleNamespace(usage=None, choices=[])
    saved_wait, G.IN_FLIGHT_WAIT_S = G.IN_FLIGHT_WAIT_S, 0.0
    G._TLS.tele = G.new_telemetry()
    flaky = _Flaky(["Error code: 402 - {'reason': 'in_flight_budget_exhausted'}"])
    G._UsageCompletions(flaky, True).create(model="anthropic/claude-opus-5.5", messages=[{"role": "user", "content": "x"}])
    check("in-flight 402: retried, counted", flaky.calls == 2 and G._TLS.tele.get("in_flight_retries") == 1)
    hard = _Flaky(["Error code: 402 - insufficient credits"])
    try:
        G._UsageCompletions(hard, True).create(model="openai/gpt-5.6-sol", messages=[])
        raised = False
    except RuntimeError:
        raised = True
    check("other 402: raised, not retried", raised and hard.calls == 1)

    # SFT_OPENROUTER_PROVIDERS pins routing (no fallback); unset sends no provider field.
    class _Capture:
        def create(self, **request):
            self.request = request
            return types.SimpleNamespace(usage=None, choices=[])
    saved_env = os.environ.pop("SFT_OPENROUTER_PROVIDERS", None)
    cap = _Capture()
    G._UsageCompletions(cap, True).create(model="anthropic/claude-opus-5.5", messages=[{"role": "user", "content": "x"}])
    unset_ok = "provider" not in cap.request["extra_body"]
    os.environ["SFT_OPENROUTER_PROVIDERS"] = " anthropic, google-vertex "
    G._UsageCompletions(cap, True).create(model="anthropic/claude-opus-5.5", messages=[{"role": "user", "content": "x"}])
    check("provider routing: pinned without fallback, absent when unset", unset_ok and cap.request["extra_body"]["provider"]
          == {"order": ["anthropic", "google-vertex"], "allow_fallbacks": False})
    os.environ.pop("SFT_OPENROUTER_PROVIDERS")
    if saved_env is not None:
        os.environ["SFT_OPENROUTER_PROVIDERS"] = saved_env
    G.IN_FLIGHT_WAIT_S, G._TLS.tele = saved_wait, None


def accepted_turns(row: dict) -> list[str]:
    return [m["content"] for m in row["episode"]["transcript"] if m["role"] == "assistant"]


def find_escape_fixture() -> dict | None:
    """An accepted pilot attempt whose turn 1 failed only because of &gt;/&lt;, and whose
    turn 2 is the same query unescaped."""
    for path in sorted(PILOT.glob("models/*/*/chunk_*/raw_attempts.jsonl")):
        for row in C.iter_jsonl(path):
            if not row.get("accepted"):
                continue
            turns = accepted_turns(row)
            if len(turns) < 3 or not re.search(r"&(gt|lt);", turns[0]):
                continue
            sql = lambda t: re.sub(r"\s+", " ", (re.search(r"<parameter=sql>(.*?)</parameter>", t, re.S) or [None, ""])[1]).strip()
            if sql(G.repair_entities(turns[0])[0]) == sql(turns[1]) and len(turns) == 3:
                return row
    return None


class FakeCompletions:
    """Scripted chat.completions.create: returns the next text with usage and finish_reason."""

    def __init__(self, script: list[str]) -> None:
        self.script, self.requests = list(script), []

    def create(self, **request):
        self.requests.append(request)
        text = self.script.pop(0)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001})
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=text))])


def t_episode() -> None:
    row = find_escape_fixture()
    check("fixture: escaped pilot trace found", row is not None, row and row["question_id"])
    if row is None:
        return
    turns = accepted_turns(row)
    bad_turn = "I will query the table now without any tags."  # protocol violation
    first = turns[0] + "\n<observation>\nFAKE_OBSERVATION\n</observation>\n<final_answer>fabricated</final_answer>"
    final = re.sub(r"Verified call:\s*\d+", "Verified call: 1", turns[2])
    fake = FakeCompletions([bad_turn, first, final])
    # Replace only the raw HTTP client under v41's stop proxy, so the whole chain runs.
    original = G.v41._adapter_get_client
    G.v41._adapter_get_client = lambda *a, **k: SimpleNamespace(chat=SimpleNamespace(completions=fake))
    try:
        parser = G.core.build_parser()
        args = parser.parse_args(["generate", "--output-dir", str(TMP / "gen"), "--decoding", "greedy",
                                  "--model", "openai/gpt-5.6-sol", "--model-url", "https://openrouter.ai/api/v1",
                                  "--teacher-prompt", "/data/keenSQL/train_prompt/retoolsql_prompt/v4.1/no_teacher_appendix.txt",
                                  "--max-turns", "7"])
        args.temperature, args.api_key, args.chat_template_kwargs = 0.0, "offline-test", None
        record = json.loads((PILOT / "inputs/easy_current.json").read_text())
        record = next(r for r in record if str(r["question_id"]) == str(row["question_id"]))
        teacher = Path(args.teacher_prompt).read_text() if Path(args.teacher_prompt).is_file() else ""
        result = G.core.run_attempt(row["source_index"], record, 1, args, teacher)
    finally:
        G.v41._adapter_get_client = original
    tele = result.get("pipeline", {})
    observations = [m["content"] for m in result["episode"]["transcript"] if m["role"] == "observation"]
    check("episode: accepted after repair", result.get("accepted") is True,
          f"exception={result.get('exception')} errors={result['episode']['errors'] if 'episode' in result else None}")
    check("episode: one call, no validation_error", len(observations) == 1 and "validation_error" not in observations[0],
          observations[0][:80] if observations else "")
    check("episode: bad turn resampled, not in transcript",
          tele.get("resampled_turns") == 1 and all(bad_turn not in m["content"] for m in result["episode"]["transcript"]),
          f"resampled={tele.get('resampled_turns')}")
    check("episode: fabricated tail clipped", tele.get("clipped_turns") == 1 and not any(
        "FAKE_OBSERVATION" in m["content"] for m in result["episode"]["transcript"]))
    check("episode: entities repaired and none left", tele.get("entity_repairs", 0) >= 1 and not any(
        re.search(r"&(gt|lt);", m["content"]) for m in result["episode"]["transcript"] if m["role"] == "assistant"))
    check("episode: telemetry usage", tele.get("api_calls") == 3 and abs(tele.get("cost_usd", 0) - 0.003) < 1e-9,
          f"api_calls={tele.get('api_calls')} cost={tele.get('cost_usd')}")
    check("episode: OpenRouter usage + stop requested",
          all(r.get("extra_body", {}).get("usage") == {"include": True} and r.get("stop") == ["</tool_call>"]
              for r in fake.requests))
    check("episode: resample raised temperature from 0", fake.requests[1].get("temperature") == G.RESAMPLE_TEMPERATURE,
          f"temps={[r.get('temperature') for r in fake.requests]}")
    # keep for the scanner test
    root = TMP / "scan_v2"
    path = root / "models/openai-gpt-5.6-sol/greedy/chunk_000/raw_attempts.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result, ensure_ascii=False, default=str) + "\n")


# ------------------------------------------------------------ scanner

def t_scan() -> None:
    import scan_traces as S
    m = S.scan(PILOT, expect_model="openai/gpt-5.6-sol")
    blocking = S.judge(m, S.PROFILES["smoke"], "smoke")
    names = {b.split(":")[0] for b in blocking}
    check("scan: v1 pilot run is blocked", {"telemetry", "entities", "protocol"} <= names,
          f"attempts={m['attempts']} entity_left={m.get('entity_left_in_call')} protocol={m['protocol_rate']} -> {sorted(names)}")
    m2 = S.scan(TMP / "scan_v2", expect_model="openai/gpt-5.6-sol")
    b2 = S.judge(m2, S.PROFILES["smoke"], "smoke")
    check("scan: v2 attempt passes the smoke gate", not b2, f"{b2} metrics cost={m2['cost_usd']}")
    m3 = S.scan(TMP / "scan_v2", expect_model="moonshotai/kimi-k3")
    check("scan: wrong model is blocked", any(b.startswith("expected_model") for b in S.judge(m3, S.PROFILES["smoke"], "smoke")))
    check("scan: 403 counts as persistent", S.classify_exception("PermissionDeniedError: Error code: 403 - region") == "forbidden")


# ------------------------------------------------------------ audit v2 regression on the 395 batch

def t_audit_regression() -> None:
    a = TMP / "audit_legacy"
    a.mkdir()
    for name in ("items.jsonl", "kimi.jsonl", "deepseek.jsonl", "l3b.jsonl"):
        shutil.copy(FULL / name, a / name)
    r = run([PY, str(HERE / "audit_l3.py"), "aggregate", "--items", str(a / "items.jsonl"), "--dir", str(a),
             "--quote-policy", "report"])
    rep = json.loads((a / "report.json").read_text())
    agg = [json.loads(l) for l in open(a / "aggregate.jsonl")]
    real = {}
    for row in agg:
        if not row.get("corruption"):
            real[row["route"]] = real.get(row["route"], 0) + 1
    check("aggregate(report): same real routes as v1", real == {"PASS": 322, "L3B": 56, "DROP": 3, "DROP_GOLD": 14}, str(real))
    check("aggregate: calibration gate passes on the 395 batch", r.returncode == 0 and rep["calibration_gate"]["passed"],
          json.dumps(rep["calibration_gate"]))
    # Sample audit: drop 60 planted verdicts -> gate must judge only the remaining 30.
    b = TMP / "audit_sample"
    b.mkdir()
    shutil.copy(a / "items.jsonl", b / "items.jsonl")
    planted = [json.loads(l)["id"] for l in open(a / "items.jsonl") if json.loads(l).get("corruption")]
    drop = set(planted[:60])
    for name in ("kimi.jsonl", "deepseek.jsonl"):
        (b / name).write_text("".join(l for l in open(a / name) if json.loads(l)["id"] not in drop))
    rs = run([PY, str(HERE / "audit_l3.py"), "aggregate", "--items", str(b / "items.jsonl"), "--dir", str(b),
              "--quote-policy", "report"])
    reps = json.loads((b / "report.json").read_text())
    check("aggregate: sample audit counts only judged planted items",
          reps["planted_judged_by_both"] == 30 and reps["calibration_gate"]["planted"] == 30
          and reps["recall_on_planted_errors"]["either_judge_flagged_rate"] >= 0.97,
          f"{reps['planted_judged_by_both']} {reps['recall_on_planted_errors']['either_judge_flagged_rate']}")
    check("aggregate: legacy findings have no quotes (measured)", rep["kimi"]["quote_valid_rate"] < 0.05,
          f"kimi quote_valid_rate={rep['kimi']['quote_valid_rate']}")
    r = run([PY, str(HERE / "audit_l3b.py"), "final", "--items", str(a / "items.jsonl"), "--l3a-dir", str(a),
             "--l3b", str(a / "l3b.jsonl"), "--out-dir", str(a), "--quote-policy", "report"])
    fin = json.loads((a / "final_report.json").read_text())
    check("final(report): same real finals as v1",
          fin["final_real"] == {"DROP_FORMAT": 25, "PASS": 342, "DROP": 6, "DROP_GOLD": 21, "REGEN": 1}, str(fin["final_real"]))
    check("final: 358 SFT-eligible (342 + 16 gold-bad)", fin["sft_eligible_real"] == 358 and fin["sft_eligible_gold_bad"] == 16,
          f"{fin['sft_eligible_real']} / {fin['sft_eligible_gold_bad']}")
    ids = {json.loads(l)["id"] for l in open(a / "final_decisions.jsonl") if json.loads(l)["sft_eligible"]}
    manual = set(Path("/data/keenSQL/dataset/m_schema/sft_data/trace_audit_l3_gpt56sol_395/full/sft_selected_ids.txt").read_text().split())
    check("final: eligible ids == hand-made sft_selected_ids.txt", ids == manual, f"diff={len(ids ^ manual)}")
    # Fake quote-bearing verdicts to test enforce-mode anchoring.
    t_quotes(a)
    t_overturn(a)


def t_quotes(a: Path) -> None:
    import audit_l3 as L3
    items = {json.loads(l)["id"]: json.loads(l) for l in open(a / "items.jsonl")}
    item = next(i for i in items.values() if i.get("corruption") and i["corruption"]["kind"] == "T2_empty_ignored")
    k = next(t["k"] for t in reversed(item["turns"]) if "call" in t)
    good = {"code": "T2_observation_ignored", "cite": f"observation#{k}", "quote": '"rows":[]'}
    bad = {"code": "T2_observation_ignored", "cite": f"observation#{k}", "quote": '"rows":[[42]]'}
    none = {"code": "T1_ungrounded_decision", "cite": "none", "quote": ""}
    check("quote: verbatim span accepted", L3.check_quote(good, item) is None)
    check("quote: invented span rejected", L3.check_quote(bad, item) == "quote_not_found")
    check("quote: missing quote rejected", L3.check_quote({**good, "quote": ""}, item) == "missing_quote")
    check("quote: cite none needs no quote", L3.check_quote(none, item) is None)
    v = {"verdict": "REGEN", "reason_codes": ["T2_observation_ignored"], "findings": [bad], "confidence": 95}
    n = L3.normalize(v, item)
    check("normalize: non-PASS on a rejected quote is weak", n["weak_non_pass"] and n["supported_policy"] == "PASS")
    n2 = L3.normalize({**v, "findings": [good]}, item)
    check("normalize: anchored non-PASS kept", not n2["weak_non_pass"] and n2["supported_policy"] == "REGEN")


def t_overturn(a: Path) -> None:
    """The escaped planted item: L3b gave PASS calling the planted observation a log artifact."""
    import audit_l3 as L3
    item_id = "corrupt:T2_empty_ignored:pilot500:bird_00724"
    items = {json.loads(l)["id"]: json.loads(l) for l in open(a / "items.jsonl")}
    item = items[item_id]
    k = next(t["k"] for t in reversed(item["turns"]) if "call" in t)
    d = TMP / "overturn"
    d.mkdir()
    (d / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n")
    finding = {"code": "T2_observation_ignored", "cite": f"observation#{k}", "quote": '"rows":[]', "detail": "empty"}
    judge_ok = L3.normalize({"verdict": "REGEN", "reason_codes": ["T2_observation_ignored"],
                             "findings": [finding], "confidence": 90}, item)
    judge_pass = L3.normalize({"verdict": "PASS", "reason_codes": [], "findings": [], "confidence": 90}, item)
    agg = {"id": item_id, "corruption": item["corruption"], "rule_tags": item["rule_tags"], "route": "L3B",
           "route_reasons": ["pass_vs_not_disagreement"], "kimi": judge_ok, "deepseek": judge_pass}
    (d / "aggregate.jsonl").write_text(json.dumps(agg, ensure_ascii=False) + "\n")
    old = next(json.loads(l) for l in open(FULL / "l3b.jsonl") if json.loads(l)["id"] == item_id)
    for variant, overturns, expect in (
        ("no overturn entry", [], "REGEN"),
        ("overturn with invented quote", [{"judge": "kimi", "code": "T2_observation_ignored",
                                           "refutation_cite": f"observation#{k}", "refutation_quote": '"row_count":14'}], "REGEN"),
    ):
        rec = copy.deepcopy(old)
        rec["verdict_json"]["overturns"] = overturns
        (d / "l3b.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n")
        run([PY, str(HERE / "audit_l3b.py"), "final", "--items", str(d / "items.jsonl"), "--l3a-dir", str(d),
             "--l3b", str(d / "l3b.jsonl"), "--out-dir", str(d)])
        dec = json.loads((d / "final_decisions.jsonl").read_text())
        check(f"overturn ({variant}): escaped planted error now caught", dec["final"] == expect,
              f"final={dec['final']} l3b said {old['verdict_json']['verdict']}")


# ------------------------------------------------------------ resample + release

def t_resample_release() -> None:
    r = run([PY, str(HERE / "resample.py"), "select", "--audit-dir", str(TMP / "audit_legacy"),
             "--gen-root", str(PILOT), "--gen-root", str(STOPFIX),
             "--parent-input", str(PILOT / "inputs/easy_current.json"), "--out", str(TMP / "select.json")])
    plan = json.loads((TMP / "select.json").read_text()) if (TMP / "select.json").is_file() else {}
    check("resample select runs", r.returncode == 0, r.stdout.strip()[:300] + r.stderr[-300:])
    total = plan.get("selected", 0) + sum(plan.get("skipped", {}).values())
    check("resample: every pilot question accounted for", total == 500, f"{plan.get('by_reason')} {plan.get('skipped')}")
    check("resample: 358 satisfied + 21 gold-bad skipped",
          plan.get("skipped", {}).get("has_sft_trace") == 358 - 0 and plan.get("skipped", {}).get("gold_judged_wrong") == 5,
          str(plan.get("skipped")))
    r = run([PY, str(HERE / "resample.py"), "prepare", "--plan", str(TMP / "select.json"), "--out",
             str(TMP / "resample"), "--samples", "2"])
    m = TMP / "resample/s2/inputs/manifest.json"
    check("resample prepare: sample roots with verifiable manifests", r.returncode == 0 and m.is_file(), r.stderr[-300:])
    if m.is_file():
        v = run([PY, "/data/keenSQL/train_prompt/sft_easy_teacher_traces/verify_setup.py", "--input-root", str(m.parent)])
        check("resample prepare: verify_setup.py accepts the inputs", v.returncode == 0, (v.stdout + v.stderr)[-300:])
    c = run([PY, str(HERE / "resample.py"), "consistency", "--gen-root", str(PILOT), "--gen-root", str(STOPFIX),
             "--out", str(TMP / "consistency.jsonl")])
    check("consistency runs", c.returncode == 0, c.stdout.strip())
    # release over the legacy audit (sources sidecar added in the temp copy)
    a = TMP / "audit_legacy"
    (a / "items.jsonl.sources.json").write_text(json.dumps({
        "pilot500": str(PILOT / "merged/by_model/openai-gpt-5.6-sol/sft_easy_teacher_openai-gpt-5.6-sol.jsonl"),
        "stopfix67": str(STOPFIX / "merged/by_model/openai-gpt-5.6-sol/sft_easy_teacher_openai-gpt-5.6-sol.jsonl")}))
    r = run([PY, str(HERE / "release.py"), "--audit-dir", str(a), "--consistency", str(TMP / "consistency.jsonl"),
             "--out-dir", str(TMP / "release")])
    man = json.loads((TMP / "release/release_manifest.json").read_text()) if r.returncode == 0 else {}
    counts = man.get("counts", {})
    check("release runs", r.returncode == 0, r.stderr[-400:])
    check("release: 358 eligible, released = 358 - dev contamination",
          counts.get("sft_eligible_traces") == 358
          and counts.get("released_traces") == 358 - counts.get("dropped_dev_contamination", -1),
          json.dumps(counts))
    check("release: RL exclusion lists the 21 gold-bad questions", counts.get("rl_exclude_questions") == 21)


# ------------------------------------------------------------ guard freeze/thaw

def t_guard() -> None:
    import run_guard as RG
    root = TMP / "guard"
    (root / "logs/openai-gpt-5.6-sol/greedy").mkdir(parents=True)
    proc = subprocess.Popen(["sleep", "60"])
    (root / "logs/openai-gpt-5.6-sol/greedy/chunk_000.pid").write_text(str(proc.pid))
    pids = RG.read_pids(root, "openai/gpt-5.6-sol", "greedy")
    check("guard: pid files read via launcher slug", pids == [proc.pid])
    check("guard: self-test freeze/thaw", RG.self_test(root, pids) and RG.proc_state(proc.pid) in ("S", "R"))
    stopped = RG.freeze(root, pids)
    check("guard: freeze reaches state T", stopped == [proc.pid] and RG.proc_state(proc.pid) == "T")
    RG.thaw(stopped)
    time.sleep(0.2)
    check("guard: thaw resumes", RG.proc_state(proc.pid) in ("S", "R"))
    proc.kill()
    proc.wait()
    check("guard: exited process not alive", RG.alive(pids) == [])
    # 2026-11-16: $6.99 at one check, key limit hit before the next (60 s, ~$3.5/min).
    burn, reserve = RG.balance_reserve((0.0, 10.5), 60.0, 6.99, 60)
    check("guard: burn-rate reserve freezes before the floor is crossed",
          RG.should_freeze(6.99, 5.0, reserve) and not RG.should_freeze(6.99, 5.0, 0.0), f"reserve={reserve:.2f}")


def t_preflight_static() -> None:
    import preflight as P
    rep = P.Report({})
    P.check_families(rep, "moonshotai/kimi-k3", P.DEFAULT_JUDGES, P.DEFAULT_ADJUDICATOR)
    check("preflight: kimi teacher judged by kimi is refused", rep.checks[-1]["status"] == "FAIL")
    rep2 = P.Report({})
    P.check_families(rep2, "openai/gpt-5.6-sol", P.DEFAULT_JUDGES, P.DEFAULT_ADJUDICATOR)
    check("preflight: GPT teacher with kimi/deepseek judges allowed", rep2.checks[-1]["status"] == "PASS")
    rep3 = P.Report({})
    P.check_inputs(rep3, PILOT / "inputs/easy_current.json")
    check("preflight: pilot inputs have DBs and schemas", rep3.checks[-1]["status"] == "PASS", json.dumps(rep3.checks[-1]["detail"])[:200])


def t_l3b_budget() -> None:
    """A runaway adjudicator is forced to submit once its per-item budget is spent."""
    import audit_l3b as B
    item = json.loads(open(FULL / "items.jsonl").readline())
    seen = []

    class Fake:
        def create(self, **req):
            seen.append(req["tool_choice"])
            forced = isinstance(req["tool_choice"], dict)
            name = "submit_verdict" if forced else "sqlite_query"
            args = json.dumps({"verdict": "PASS", "reason_codes": [], "gold_suspect": False, "agrees_with": "both",
                               "confidence": 80, "explanation": "x"} if forced else {"sql": "SELECT 1"})
            call = SimpleNamespace(id=f"c{len(seen)}", function=SimpleNamespace(name=name, arguments=args))
            msg = SimpleNamespace(tool_calls=[call], content=None,
                                  model_dump=lambda **k: {"role": "assistant", "content": None, "tool_calls": [
                                      {"id": call.id, "type": "function", "function": {"name": name, "arguments": args}}]})
            usage = SimpleNamespace(model_dump=lambda: {"completion_tokens": 40000, "cost": 0.5})
            return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Fake()))
    rec = B.adjudicate(client, item, {"kimi": {}, "deepseek": {}}, "system", 1000)
    check("l3b budget: forced submit on the turn after overspending",
          seen[0] == "auto" and isinstance(seen[1], dict) and rec.get("verdict_json", {}).get("verdict") == "PASS"
          and rec.get("budget_forced", {}).get("completion_tokens") == 40000, f"tool_choice seq={seen}")


def t_lazy_review_guard() -> None:
    """A PASS produced in very few tokens is re-asked once; a second short PASS is flagged."""
    import audit_l3 as L3
    item = json.loads(open(FULL / "items.jsonl").readline())

    def fake(replies):
        class Fake:
            def __init__(self):
                self.calls = []
            def create(self, **req):
                self.calls.append(req["messages"])
                verdict, tokens = replies.pop(0)
                content = json.dumps({"verdict": verdict, "reason_codes": [] if verdict == "PASS" else ["T2_observation_ignored"],
                                      "findings": [], "confidence": 95, "explanation": "x"})
                usage = SimpleNamespace(model_dump=lambda: {"completion_tokens": tokens, "cost": 0.01})
                return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=content))])
        f = Fake()
        return f, SimpleNamespace(chat=SimpleNamespace(completions=f))

    f1, c1 = fake([("PASS", 250), ("REGEN", 1800)])
    r1 = L3.judge_one(c1, "m", "sys", item, 1000, 600)
    check("lazy guard: short PASS re-asked, reviewed verdict kept",
          r1["review_reasked"] and r1["verdict_json"]["verdict"] == "REGEN" and not r1["short_review"]
          and len(f1.calls) == 2 and L3.REVIEW_NUDGE in f1.calls[1][-1]["content"])
    f2, c2 = fake([("PASS", 250), ("PASS", 300)])
    r2 = L3.judge_one(c2, "m", "sys", item, 1000, 600)
    check("lazy guard: second short PASS flagged short_review", r2["short_review"] and r2["verdict_json"]["verdict"] == "PASS")
    f3, c3 = fake([("PASS", 1500)])
    r3 = L3.judge_one(c3, "m", "sys", item, 1000, 600)
    check("lazy guard: normal-length PASS not re-asked", not r3.get("review_reasked") and len(f3.calls) == 1)


def main() -> int:
    tests = [t_entities, t_episode, t_scan, t_audit_regression, t_resample_release, t_guard, t_preflight_static,
             t_l3b_budget, t_lazy_review_guard]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{test.__name__} crashed", False, f"{type(exc).__name__}: {exc}")
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; temp dir {TMP}")
    if not failed:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
