#!/usr/bin/env python3
"""L3a trajectory audit with two LLM judges (see 轨迹审核Agent执行方案.md §3.2, §3.4).

Subcommands
  build      SFT jsonl files -> audit items (parsed turns, gold result, rule tags)
  corrupt    clean items -> planted-error items for judge recall calibration
  judge      items -> one judge's verdicts (OpenRouter, resumable, circuit breaker)
  aggregate  verdicts -> evidence check, two-judge agreement, routing, report

The judge prompt is l3a_judge_prompt.md next to this file. The API key is read
from OPENROUTER_API_KEY only and is never written anywhere.

trace_pipeline_v1 changes against train_prompt/trace_audit_l3/audit_l3.py (GAIA borrowings):
  - findings carry a verbatim `quote`; aggregate checks it against the cited source and
    discards findings whose quote is missing or not found ("verbatim evidence" rule);
    a non-PASS verdict resting only on discarded findings is routed to L3b, not accepted;
  - judges report `confidence`; a PASS/PASS item with either confidence below
    --min-pass-confidence goes to L3b (confidence-gated escalation), and the report shows
    how confidence relates to correctness on planted errors;
  - corrupt plants a fixed share of the batch (--rate), and aggregate refuses to finish
    (exit 3) when a judge's planted-error recall is under --min-recall: an uncalibrated
    batch is not trusted;
  - build writes <out>.sources.json (run name -> SFT file) for release.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "l3a_judge_prompt.md"
DB_ROOT = Path(os.environ.get("SFT_DB_ROOT") or "/data/keenSQL/dataset/all_dataset/databases")
JUDGES = {
    "kimi": "moonshotai/kimi-k3",
    "deepseek": "deepseek/deepseek-v4-pro",
}
CODES = {
    "S1_final_semantics_wrong": "DROP",
    "S2_output_shape": "REGEN",
    "T1_ungrounded_decision": "REGEN",
    "T2_observation_ignored": "REGEN",
    "T3_reasoning_action_mismatch": "REGEN",
    "T4_ritual_calls_minor": "PASS",
    "T4_ritual_calls_severe": "REGEN",
    "T5_leak_wording": "DROP",
}
SEVERITY = {"PASS": 0, "REGEN": 1, "DROP": 2}
GOLD_ROWS_SHOWN = 10


# ---------------------------------------------------------------- build

def _tag(name: str, text: str) -> str | None:
    match = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return match.group(1).strip() if match else None


def _strip_observation(text: str) -> str:
    text = text.strip()
    if text.startswith("<observation>"):
        text = text[len("<observation>"):]
    if text.endswith("</observation>"):
        text = text[: -len("</observation>")]
    return text.strip()


def _run(db_id: str, sql: str) -> list[tuple]:
    path = DB_ROOT / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    started = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - started > 60 else 0, 10000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def parse_trajectory(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traj = row["messages"][row["prompt_messages"]:]
    turns: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    k = 0
    for message in traj:
        if message["role"] == "assistant":
            k += 1
            text = message["content"]
            turn = {"k": k, "scratch": _tag("scratch_pad", text) or ""}
            call = _tag("tool_call", text)
            if call is not None:
                name = re.search(r"<function=(\w+)>", call)
                turn["call_name"] = name.group(1) if name else "?"
                turn["call"] = call
            else:
                final = {
                    "k": k,
                    "scratch": turn["scratch"],
                    "sql": _tag("final_answer", text) or "",
                    "tables": _tag("relevant_tables", text) or "",
                    "columns": _tag("relevant_columns", text) or "",
                }
            turns.append(turn)
        else:
            turns[-1]["observation"] = _strip_observation(message["content"])
    return turns, final


def rule_tags(turns, final, gold_rows, row) -> dict[str, Any]:
    calls = [t for t in turns if "call" in t]
    observations = [t.get("observation", "") for t in calls]
    anomaly = [
        t["k"] for t in calls
        if '"error"' in t.get("observation", "").lower()
        or '"rows":[]' in t.get("observation", "").replace(" ", "")
    ]
    weak = (not gold_rows) or (
        len(gold_rows) == 1 and len(gold_rows[0]) == 1 and gold_rows[0][0] in (0, 1, None)
    )
    norm = lambda s: re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()
    return {
        "tool_calls": len(calls),
        "single_call": len(calls) == 1,
        "diagnostic_tools_used": sorted({t["call_name"] for t in calls} - {"sqlite_query"}),
        "anomalous_observations": anomaly,
        "html_escaped_sql": any(re.search(r"&(gt|lt|amp);", t["call"]) for t in calls),
        "weak_ex_evidence": weak,
        "final_equals_gold_text": norm(final.get("sql", "")) == norm(row["gold_sql"]),
        "observations_total_chars": sum(len(o) for o in observations),
    }


def cmd_build(args) -> int:
    items = []
    sources = {}
    for spec in args.inputs:
        run, path = spec.split("=", 1)
        sources[run] = str(Path(path).resolve())
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            turns, final = parse_trajectory(row)
            gold_rows = _run(row["db_id"], row["gold_sql"])
            items.append(
                {
                    "id": f"{run}:{row['question_id']}",
                    "run": run,
                    "question_id": row["question_id"],
                    "db_id": row["db_id"],
                    "source": row.get("source"),
                    "difficulty": row.get("difficulty"),
                    "grading_method": row.get("grading_method"),
                    "student_prompt": row["messages"][row["prompt_messages"] - 1]["content"],
                    "turns": turns,
                    "final": final,
                    "gold_sql": row["gold_sql"],
                    "gold_result": {
                        "row_count": len(gold_rows),
                        "rows": [list(r) for r in gold_rows[:GOLD_ROWS_SHOWN]],
                    },
                    "rule_tags": rule_tags(turns, final, gold_rows, row),
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    Path(str(out) + ".sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    tags = Counter()
    for item in items:
        for key, value in item["rule_tags"].items():
            if value is True or (isinstance(value, list) and value):
                tags[key] += 1
    print(f"wrote {len(items)} items to {out}; tag counts {dict(tags)}")
    return 0


# ---------------------------------------------------------------- corrupt

LEAK_SENTENCES = [
    "This matches the reference SQL for this question, so I keep the same filter.",
    "Following the expected answer, I use the same tables as the gold query.",
    "The answer key uses this join, so I adopt it directly.",
]


def _row_count(observation: str) -> int | None:
    try:
        return int(json.loads(observation).get("row_count"))
    except Exception:  # noqa: BLE001
        return None


def corrupt_item(item: dict[str, Any], kind: str, rng: random.Random) -> dict[str, Any] | None:
    """Plant one process error; the final SQL and its EX=True status stay as they were."""

    bad = copy.deepcopy(item)
    calls = [t for t in bad["turns"] if "call" in t]
    final_turn = next((t for t in bad["turns"] if t["k"] == bad["final"].get("k")), None)
    if not calls or final_turn is None:
        return None
    last_call = calls[-1]
    if kind == "T2_empty_ignored":
        # The last call returned nothing, yet the agent finalizes on it unchanged.
        try:
            obs = json.loads(last_call.get("observation", ""))
        except Exception:  # noqa: BLE001
            return None
        if "error" in obs or not obs.get("rows"):
            return None
        last_call["observation"] = json.dumps(
            {"columns": obs.get("columns", []), "rows": [], "row_count": 0,
             "truncated": False, "output_truncated": False},
            separators=(",", ":"),
        )
        expect = {"T2_observation_ignored", "T3_reasoning_action_mismatch"}
    elif kind == "T5_leak":
        sentence = rng.choice(LEAK_SENTENCES)
        target = bad["turns"][0]
        target["scratch"] = target["scratch"].rstrip() + " " + sentence
        expect = {"T5_leak_wording"}
    elif kind == "T3_false_claim":
        actual = _row_count(last_call.get("observation", ""))
        if actual is None:
            return None
        claimed = actual + 3 if actual < 3 else max(actual - 2, 0) if actual > 5 else actual + 7
        sentence = f"call#{last_call['k']} returned exactly {claimed} rows, which confirms the result."
        final_turn["scratch"] = sentence + " " + final_turn["scratch"]
        bad["final"]["scratch"] = final_turn["scratch"]
        expect = {"T3_reasoning_action_mismatch", "T2_observation_ignored"}
    else:
        raise ValueError(kind)
    # Keep the hints honest: a planted empty result must show up as an anomaly.
    bad["rule_tags"]["anomalous_observations"] = [
        t["k"] for t in calls
        if '"error"' in t.get("observation", "").lower()
        or '"rows":[]' in t.get("observation", "").replace(" ", "")
    ]
    bad["id"] = f"corrupt:{kind}:{item['id']}"
    bad["corruption"] = {"kind": kind, "expected_codes": sorted(expect), "base_id": item["id"]}
    return bad


def cmd_corrupt(args) -> int:
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    rng = random.Random(args.seed)
    out_rows = []
    kinds = ("T2_empty_ignored", "T5_leak", "T3_false_claim")
    if args.rate:
        # A fixed share of the batch, at least 10 per kind (30 planted) so the recall gate has
        # statistical meaning; small batches get as many distinct (base, kind) pairs as exist.
        args.per_kind = max(10, -(-int(round(args.rate * len(items))) // len(kinds)))
    for kind in kinds:
        pool = items[:]
        rng.shuffle(pool)
        made = 0
        for item in pool:
            bad = corrupt_item(item, kind, rng)
            if bad is not None:
                out_rows.append(bad)
                made += 1
            if made == args.per_kind:
                break
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} corrupted items: {Counter(r['corruption']['kind'] for r in out_rows)}")
    return 0


# ---------------------------------------------------------------- judge

def render(item: dict[str, Any]) -> str:
    parts = ["## STUDENT_PROMPT (exactly what the agent saw: SCHEMA, EVIDENCE, QUESTION)", "",
             item["student_prompt"].strip(), "", "## TRAJECTORY", ""]
    for turn in item["turns"]:
        parts.append(f"### turn {turn['k']}")
        parts.append(f"scratch#{turn['k']}:\n{turn['scratch']}")
        if "call" in turn:
            parts.append(f"call#{turn['k']} ({turn['call_name']}):\n{turn['call']}")
            parts.append(f"observation#{turn['k']}:\n{turn.get('observation', '(none)')}")
        else:
            final = item["final"]
            parts.append(f"FINAL ANSWER (turn {turn['k']}):")
            parts.append(f"relevant_tables: {final['tables']}")
            parts.append(f"relevant_columns: {final['columns']}")
            parts.append(f"final_sql:\n{final['sql']}")
        parts.append("")
    gold = item["gold_result"]
    shown = json.dumps(gold["rows"], ensure_ascii=False, default=str)
    parts += [
        "## GOLD (never shown to the agent; comparison only)", "",
        f"gold_sql:\n{item['gold_sql']}", "",
        f"gold_result: row_count={gold['row_count']}; first rows: {shown}",
        f"grading_method: {item.get('grading_method')}", "",
        "## RULE_TAGS (automatic hints, not verdicts)", "",
        json.dumps(item["rule_tags"], ensure_ascii=False),
    ]
    return "\n".join(parts)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start: end + 1])


class CircuitBreaker:
    def __init__(self, limit: int) -> None:
        self.limit, self.streak, self.lock, self.open = limit, 0, threading.Lock(), False

    def record(self, ok: bool) -> None:
        with self.lock:
            self.streak = 0 if ok else self.streak + 1
            if self.streak >= self.limit:
                self.open = True


REVIEW_NUDGE = (
    "Your verdict above was produced without a visible review, so it cannot be trusted yet. "
    "Work through the checks for this trajectory one by one before answering again: "
    "(3) for every observation, say whether it shows an anomaly (0 rows, error, NULLs, 0, "
    "duplicates, truncation) and how the agent reacted; (4) compare every factual claim in each "
    "scratch_pad with the observation it refers to; (6) read every scratch_pad sentence for "
    "wording that implies access to a reference answer; (1) and (7) compare the final SQL and "
    "the gold SQL with the question. Then reply with the JSON object."
)


def judge_one(client, model: str, system: str, item: dict[str, Any], max_tokens: int,
              min_review_tokens: int = 0) -> dict[str, Any]:
    record = _judge_once(client, model, system, item, max_tokens)
    verdict = record.get("verdict_json") or {}
    tokens = int((record["attempts"][-1].get("usage") or {}).get("completion_tokens") or 0)
    # Lazy-review guard (GAIA "summary collapse"): kimi-k3 sometimes stamps a confident PASS in
    # ~200-500 tokens without reviewing; 7 of its 30 planted-error misses in the resample batch
    # were such replies. Re-ask once with the checklist; a second short PASS goes to L3b.
    if min_review_tokens and str(verdict.get("verdict", "")).upper() == "PASS" and tokens < min_review_tokens:
        first = record
        messages_extra = [{"role": "assistant", "content": first.get("raw", "")},
                          {"role": "user", "content": REVIEW_NUDGE}]
        second = _judge_once(client, model, system, item, max_tokens, extra=messages_extra)
        second["attempts"] = first["attempts"] + [dict(a, review_reask=True) for a in second["attempts"]]
        second["first_verdict"] = verdict
        second["review_reasked"] = True
        if second.get("verdict_json") is None:
            second["verdict_json"], second["raw"] = first.get("verdict_json"), first.get("raw")
        v2 = second.get("verdict_json") or {}
        t2 = int((second["attempts"][-1].get("usage") or {}).get("completion_tokens") or 0)
        second["short_review"] = str(v2.get("verdict", "")).upper() == "PASS" and t2 < min_review_tokens
        return second
    return record


def _judge_once(client, model: str, system: str, item: dict[str, Any], max_tokens: int,
                extra: list[dict[str, str]] | None = None) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": render(item)}] + (extra or [])
    record: dict[str, Any] = {"id": item["id"], "model": model, "attempts": []}
    for attempt in (1, 2):
        started = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"usage": {"include": True}},
        )
        usage = response.usage.model_dump() if response.usage else {}
        content = response.choices[0].message.content or ""
        info = {
            "attempt": attempt,
            "elapsed_s": round(time.time() - started, 1),
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
        }
        record["attempts"].append(info)
        try:
            record["verdict_json"] = parse_json(content)
            record["raw"] = content
            return record
        except Exception as exc:  # noqa: BLE001 - retried once, then recorded
            info["parse_error"] = f"{type(exc).__name__}: {exc}"
            info["raw_tail"] = content[-500:]
    record["verdict_json"] = None
    return record


def cmd_judge(args) -> int:
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    model = JUDGES[args.judge]
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    timeout=args.timeout, max_retries=2)
    system = PROMPT_PATH.read_text(encoding="utf-8")
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    if args.ids:
        wanted = set(Path(args.ids).read_text().split())
        items = [item for item in items if item["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    done = set()
    if out.is_file():
        for line in out.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("verdict_json") is not None:
                done.add(row["id"])
    todo = [item for item in items if item["id"] not in done]
    print(f"[{args.judge}] {model}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    breaker, lock = CircuitBreaker(args.breaker), threading.Lock()
    cost, finished = 0.0, 0
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = {}
        for item in todo:
            futures[pool.submit(judge_one, client, model, system, item, args.max_tokens,
                                args.min_review_tokens)] = item
        for future in as_completed(futures):
            item = futures[future]
            if breaker.open:
                continue
            try:
                record = future.result()
                breaker.record(True)
            except Exception as exc:  # noqa: BLE001 - provider failure
                breaker.record(False)
                print(f"[{args.judge}] API error on {item['id']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                if breaker.open:
                    print(f"[{args.judge}] circuit breaker open; cancelling the rest", flush=True)
                    for other in futures:
                        other.cancel()
                continue
            for info in record["attempts"]:
                cost += float((info.get("usage") or {}).get("cost") or 0.0)
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
            if finished % 10 == 0:
                print(f"[{args.judge}] {finished}/{len(todo)} done, cost so far ${cost:.3f}", flush=True)
    print(f"[{args.judge}] finished {finished}/{len(todo)}; cost ${cost:.3f}", flush=True)
    return 2 if breaker.open else 0


# ---------------------------------------------------------------- aggregate

CITE_ONE = re.compile(r"^(question|evidence|none|schema:.+|call#(\d+)\.sql|observation#(\d+)|scratch#(\d+))$")
FK_SECTION = {"foreign_keys", "foreign keys", "fk", "fks", "relations"}


def _check_schema_ref(ref: str, prompt: str) -> str | None:
    ref = ref.strip().strip("`").strip()
    if ref.lower() in FK_SECTION or ref.lower().startswith(("foreign key", "fk ")):
        return None if "[Foreign Keys]" in prompt else "foreign keys cited but none in schema"
    # "a.b references c.d" -> check both sides
    if " references " in ref:
        for side in ref.split(" references "):
            reason = _check_schema_ref(side, prompt)
            if reason:
                return reason
        return None
    tables = [m.group(1).strip() for m in re.finditer(r"^# Table: (.+?)(?:\s*\||$)", prompt, re.M)]
    lowered = {t.lower(): t for t in tables}
    if ref.lower() in lowered:
        return None
    # longest table name that prefixes "table.column"
    for name in sorted(tables, key=len, reverse=True):
        if ref.lower().startswith(name.lower() + "."):
            column = ref[len(name) + 1:].strip().strip("`")
            return None if column and column.lower() in prompt.lower() else f"schema:{ref}: column not in schema"
    return f"schema:{ref}: table not in schema"


def check_cite(cite: str, item: dict[str, Any]) -> str | None:
    """Return None when every source in the citation exists, else the first reason."""

    calls = {t["k"] for t in item["turns"] if "call" in t}
    turns = {t["k"] for t in item["turns"]}
    prompt = item["student_prompt"]
    parts = [p for p in re.split(r"\s*[;,]\s*(?=question|evidence|none|schema:|call#|observation#|scratch#)",
                                  (cite or "").strip().strip("`")) if p]
    if not parts:
        return "empty cite"
    for part in parts:
        part = re.sub(r"\s*\([^)]*\)\s*$", "", part.strip()).strip().strip("`")
        match = CITE_ONE.match(part)
        if not match:
            return f"unparseable cite {part!r}"
        if match.group(2) and int(match.group(2)) not in calls:
            return f"{part}: no such call"
        if match.group(3) and int(match.group(3)) not in calls:
            return f"{part}: no such observation"
        if match.group(4) and int(match.group(4)) not in turns:
            return f"{part}: no such turn"
        if part.startswith("schema:"):
            reason = _check_schema_ref(part[len("schema:"):], prompt)
            if reason:
                return reason
        if part == "evidence":
            text = prompt.split("External Knowledge:", 1)[-1].split("Question:", 1)[0]
            if not text.strip():
                return "evidence cited but empty"
    return None


CITE_SPLIT = re.compile(r"\s*[;,]\s*(?=question|evidence|none|schema:|call#|observation#|scratch#|probe#)")


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def source_texts(cite: str, item: dict[str, Any], extra: dict[str, str] | None = None) -> list[str]:
    """Texts a citation points at; `extra` maps further refs (e.g. probe#1) to text."""

    turns = {t["k"]: t for t in item["turns"]}
    final = item.get("final") or {}
    texts: list[str] = []
    for part in CITE_SPLIT.split((cite or "").strip().strip("`")):
        part = re.sub(r"\s*\([^)]*\)\s*$", "", part.strip()).strip().strip("`")
        if part in ("question", "evidence") or part.startswith("schema:"):
            texts.append(item["student_prompt"])
        elif extra and part in extra:
            texts.append(extra[part])
        else:
            match = re.match(r"(call|observation|scratch)#(\d+)", part)
            if not match or int(match.group(2)) not in turns:
                continue
            turn = turns[int(match.group(2))]
            if match.group(1) == "call":
                texts.append(turn.get("call", ""))
            elif match.group(1) == "observation":
                observation = turn.get("observation", "")
                texts.append(observation)
                try:  # also the decoded form, so non-ASCII quotes match \\u escapes
                    texts.append(json.dumps(json.loads(observation), ensure_ascii=False))
                except Exception:  # noqa: BLE001
                    pass
            else:
                texts.append(turn.get("scratch", ""))
                if final.get("k") == turn["k"]:
                    texts.extend([final.get("sql", ""), final.get("tables", ""), final.get("columns", "")])
    return texts


def check_quote(entry: dict[str, Any], item: dict[str, Any], extra: dict[str, str] | None = None) -> str | None:
    """None when the finding's quote is found verbatim (whitespace/case-insensitive) in its cited source."""

    cite = str(entry.get("cite", "")).strip()
    if cite in ("", "none"):
        return None
    quote = entry.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return "missing_quote"
    texts = source_texts(cite, item, extra)
    if not texts:
        return "unresolvable_cite"
    hay = _norm_text("\n".join(texts))
    core_quote = quote.strip().strip("\"'`“”‘’").strip()
    fragments = [f for f in re.split(r"\.\.\.|…", core_quote) if len(_norm_text(f)) >= 3]
    if not fragments:
        return "missing_quote"
    for fragment in fragments:
        if _norm_text(fragment).strip("\"'`“”") not in hay:
            return "quote_not_found"
    return None


def normalize(verdict: dict[str, Any] | None, item: dict[str, Any], *, quote_policy: str = "enforce",
              extra_sources: dict[str, str] | None = None) -> dict[str, Any]:
    if verdict is None:
        return {"ok": False, "policy_verdict": None, "problems": ["no parseable JSON"]}
    problems = []
    codes = [c for c in verdict.get("reason_codes") or [] if isinstance(c, str)]
    unknown = [c for c in codes if c not in CODES]
    if unknown:
        problems.append(f"unknown codes {unknown}")
    known = [c for c in codes if c in CODES]
    policy = max((CODES[c] for c in known), key=SEVERITY.get, default="PASS")
    # Verbatim-evidence rule: a code counts only if some finding for it has a verified quote.
    quote_problems: list[str] = []
    supported: set[str] = set()
    for finding in verdict.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        reason = check_quote(finding, item, extra_sources) if quote_policy != "off" else None
        if reason:
            quote_problems.append(f"{finding.get('code')}: {reason}")
        elif finding.get("code") in CODES:
            supported.add(finding["code"])
    if quote_policy == "enforce":
        supported_policy = max((CODES[c] for c in known if c in supported), key=SEVERITY.get, default="PASS")
    else:
        supported_policy = policy
    confidence = verdict.get("confidence")
    try:
        confidence = int(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    stated = str(verdict.get("verdict") or "").upper()
    if stated not in SEVERITY:
        problems.append(f"bad verdict {stated!r}")
    elif stated != policy:
        problems.append(f"stated {stated} but codes imply {policy}")
    bad_cites = []
    for entry in (verdict.get("findings") or []) + (verdict.get("decisions") or []):
        if isinstance(entry, dict):
            reason = check_cite(str(entry.get("cite", "")), item)
            if reason:
                bad_cites.append(reason)
            if entry.get("grounded") is True and str(entry.get("cite", "")).strip() == "none":
                bad_cites.append("grounded=true but cite=none")
    n_entries = len((verdict.get("findings") or []) + (verdict.get("decisions") or []))
    return {
        "ok": True,
        "stated_verdict": stated,
        "policy_verdict": policy,
        "supported_policy": supported_policy,
        "weak_non_pass": policy != "PASS" and SEVERITY[supported_policy] < SEVERITY[policy],
        "quote_problems": quote_problems,
        "findings_n": len([f for f in verdict.get("findings") or [] if isinstance(f, dict)]),
        "findings": [{k: f.get(k) for k in ("code", "cite", "quote", "detail")}
                     for f in (verdict.get("findings") or [])[:12] if isinstance(f, dict)],
        "confidence": confidence,
        "codes": known,
        "gold_suspect": bool(verdict.get("gold_suspect")),
        "is_ambiguous": bool(verdict.get("is_ambiguous")),
        "bad_cites": bad_cites,
        "cite_entries": n_entries,
        "problems": problems,
        "explanation": verdict.get("explanation", ""),
        "gold_issue": verdict.get("gold_issue", ""),
    }


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({x for pair in pairs for x in pair})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[label] * cb[label] for label in labels) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def load_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("verdict_json") is not None or row["id"] not in rows:
                rows[row["id"]] = row
    return rows


def cmd_aggregate(args) -> int:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8")}
    judged = {name: load_verdicts(Path(args.dir) / f"{name}.jsonl") for name in JUDGES}
    rows, pairs3, pairs2 = [], [], []
    cost = defaultdict(float)
    tokens = defaultdict(lambda: Counter())
    for item_id, item in items.items():
        row = {"id": item_id, "corruption": item.get("corruption"), "rule_tags": item["rule_tags"]}
        for name in JUDGES:
            record = judged[name].get(item_id)
            if record is None:
                row[name] = None
                continue
            for info in record.get("attempts", []):
                usage = info.get("usage") or {}
                cost[name] += float(usage.get("cost") or 0.0)
                tokens[name]["prompt"] += int(usage.get("prompt_tokens") or 0)
                tokens[name]["completion"] += int(usage.get("completion_tokens") or 0)
            row[name] = normalize(record.get("verdict_json"), item, quote_policy=args.quote_policy)
            row[name]["review_reasked"] = bool(record.get("review_reasked"))
            row[name]["short_review"] = bool(record.get("short_review"))
        k, d = row.get("kimi"), row.get("deepseek")
        if k and d and k["ok"] and d["ok"]:
            pairs3.append((k["policy_verdict"], d["policy_verdict"]))
            pairs2.append((k["policy_verdict"] == "PASS", d["policy_verdict"] == "PASS"))
            verdicts = {k["policy_verdict"], d["policy_verdict"]}
            reasons = []
            if verdicts == {"PASS"}:
                final = "PASS"
                confidences = [v["confidence"] for v in (k, d) if v.get("confidence") is not None]
                if confidences and min(confidences) < args.min_pass_confidence:
                    final, reasons = "L3B", reasons + ["low_confidence_pass"]
            elif "PASS" not in verdicts:
                # Only non-PASS verdicts anchored in verified quotes are accepted as they are.
                anchored = [v["supported_policy"] for v in (k, d) if v["supported_policy"] != "PASS"]
                if anchored:
                    final = max(anchored, key=SEVERITY.get)
                else:
                    final, reasons = "L3B", reasons + ["unanchored_non_pass"]
            else:
                final, reasons = "L3B", reasons + ["pass_vs_not_disagreement"]
            if final == "PASS" and any(v.get("short_review") for v in (k, d)):
                final, reasons = "L3B", reasons + ["short_review_pass"]
            if final == "PASS" and item["rule_tags"]["weak_ex_evidence"]:
                final, reasons = "L3B", reasons + ["weak_ex_evidence"]
            row["route_reasons"] = reasons
            # Decision 4: a question whose gold is judged wrong is removed.
            suspects = k["gold_suspect"] + d["gold_suspect"]
            if suspects == 2:
                final = "DROP_GOLD"
            elif suspects == 1 and final == "PASS":
                final = "L3B"
                row["route_reasons"].append("one_gold_suspect")
            row["route"] = final
        elif any(row.get(n) and row[n]["ok"] for n in JUDGES):
            # One judge never produced a usable verdict: let the adjudicator decide.
            row["route"] = "L3B"
        else:
            row["route"] = "INCOMPLETE"
        rows.append(row)

    out_dir = Path(args.dir)
    with (out_dir / "aggregate.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {"items": len(rows)}
    for name in JUDGES:
        valid = [r[name] for r in rows if r.get(name) and r[name]["ok"]]
        report[name] = {
            "judged": sum(1 for r in rows if r.get(name)),
            "parse_ok": len(valid),
            "verdicts": dict(Counter(v["policy_verdict"] for v in valid)),
            "codes": dict(Counter(c for v in valid for c in v["codes"])),
            "gold_suspect": sum(v["gold_suspect"] for v in valid),
            "verdict_inconsistent_with_codes": sum(1 for v in valid if any("imply" in p for p in v["problems"])),
            "bad_cite_rate": round(
                sum(len(v["bad_cites"]) for v in valid) / max(1, sum(v["cite_entries"] for v in valid)), 4
            ),
            "cost_usd": round(cost[name], 4),
            "tokens": dict(tokens[name]),
            "findings": sum(v.get("findings_n", 0) for v in valid),
            "findings_quote_rejected": sum(len(v.get("quote_problems", [])) for v in valid),
            "quote_valid_rate": round(
                1 - sum(len(v.get("quote_problems", [])) for v in valid)
                / max(1, sum(v.get("findings_n", 0) for v in valid)), 4),
            "weak_non_pass": sum(bool(v.get("weak_non_pass")) for v in valid),
            "confidence_reported": sum(v.get("confidence") is not None for v in valid),
            "review_reasked": sum(bool(v.get("review_reasked")) for v in valid),
            "short_review_after_reask": sum(bool(v.get("short_review")) for v in valid),
        }
    report["kappa_3class"] = kappa(pairs3)
    report["kappa_pass_vs_not"] = kappa([(str(a), str(b)) for a, b in pairs2])
    report["agreement_pass_vs_not"] = (
        round(sum(a == b for a, b in pairs2) / len(pairs2), 4) if pairs2 else None
    )
    report["routes"] = dict(Counter(r["route"] for r in rows))

    # Only planted items both judges actually reviewed count toward calibration. A sample audit
    # (judge --ids) leaves the other planted items unjudged; counting them as "missed" made the
    # 2026-11-16 dev sample fail with either=60/150 although every judged item was caught.
    corrupted = [r for r in rows if r.get("corruption") and all(r.get(n) and r[n].get("ok") for n in JUDGES)]
    report["planted_total"] = sum(1 for r in rows if r.get("corruption"))
    report["planted_judged_by_both"] = len(corrupted)
    if corrupted:
        recall = {}
        for name in JUDGES:
            by_kind = defaultdict(lambda: Counter())
            for r in corrupted:
                v = r.get(name)
                if not v or not v["ok"]:
                    continue
                kind = r["corruption"]["kind"]
                by_kind[kind]["n"] += 1
                by_kind[kind]["flagged"] += v["policy_verdict"] != "PASS"
                by_kind[kind]["right_code"] += bool(set(v["codes"]) & set(r["corruption"]["expected_codes"]))
            recall[name] = {
                kind: {"n": c["n"], "flagged_rate": round(c["flagged"] / c["n"], 3),
                       "right_code_rate": round(c["right_code"] / c["n"], 3)}
                for kind, c in by_kind.items() if c["n"]
            }
            total = sum(c["n"] for c in by_kind.values())
            recall[name]["overall_flagged_rate"] = round(
                sum(c["flagged"] for c in by_kind.values()) / total, 3) if total else None
        either = [r for r in corrupted if any(
            r.get(n) and r[n]["ok"] and r[n]["policy_verdict"] != "PASS" for n in JUDGES)]
        recall["either_judge_flagged_rate"] = round(len(either) / len(corrupted), 3)
        report["recall_on_planted_errors"] = recall

    # Confidence calibration: planted items -> share flagged (correct), real items -> agreement.
    def bucket(c: int | None) -> str:
        return "missing" if c is None else "<70" if c < 70 else "70-89" if c < 90 else ">=90"

    calibration: dict[str, Any] = {}
    for name in JUDGES:
        other = [n for n in JUDGES if n != name][0]
        planted, real = defaultdict(Counter), defaultdict(Counter)
        for r in rows:
            v, o = r.get(name), r.get(other)
            if not v or not v["ok"]:
                continue
            b = bucket(v.get("confidence"))
            if r.get("corruption"):
                planted[b]["n"] += 1
                planted[b]["flagged"] += v["policy_verdict"] != "PASS"
            elif o and o["ok"]:
                real[b]["n"] += 1
                real[b]["agree_pass_vs_not"] += (v["policy_verdict"] == "PASS") == (o["policy_verdict"] == "PASS")
        calibration[name] = {
            "planted": {b: {"n": c["n"], "flagged_rate": round(c["flagged"] / c["n"], 3)} for b, c in planted.items()},
            "real": {b: {"n": c["n"], "agreement": round(c["agree_pass_vs_not"] / c["n"], 3)} for b, c in real.items()},
        }
    report["confidence_calibration"] = calibration
    report["route_reasons"] = dict(Counter(x for r in rows for x in r.get("route_reasons", [])))

    # Calibration gate: a batch whose judges miss planted errors is not trusted.
    # Size-aware: with n planted items a 0.90 bar is only meaningful from ~30 on (at n=15 a
    # judge whose true recall is 0.95 fails ~17% of batches). Below --gate-min-planted the batch
    # is "underpowered": per-judge recall must still reach --min-recall-underpowered. The
    # either-judge rate decides escapes and keeps its bar at every size.
    n_planted = len(corrupted)
    powered = n_planted >= args.gate_min_planted
    bar = args.min_recall if powered else args.min_recall_underpowered
    gate: dict[str, Any] = {"planted": n_planted, "powered": powered, "per_judge_bar": bar,
                            "min_either": args.min_either}
    rec = report.get("recall_on_planted_errors")
    if not rec:
        gate.update(passed=False, reason="no planted errors in this batch")
    else:
        low = {n: rec[n].get("overall_flagged_rate") for n in JUDGES
               if (rec[n].get("overall_flagged_rate") or 0) < bar}
        either_ok = rec["either_judge_flagged_rate"] >= args.min_either
        below_target = {n: rec[n].get("overall_flagged_rate") for n in JUDGES
                        if (rec[n].get("overall_flagged_rate") or 0) < args.min_recall}
        gate.update(passed=not low and either_ok, below_bar=low, below_target_recall=below_target,
                    either_judge_flagged_rate=rec["either_judge_flagged_rate"])
    report["calibration_gate"] = gate

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        print(f"[aggregate] CALIBRATION GATE FAILED: {gate}", flush=True)
        return 0 if args.allow_uncalibrated else 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--inputs", nargs="+", required=True, help="run=path/to/sft.jsonl")
    p.add_argument("--out", required=True)
    p = sub.add_parser("corrupt")
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-kind", type=int, default=30)
    p.add_argument("--rate", type=float, default=0.0,
                   help="plant this share of the batch (split over 3 kinds, >=5 each); overrides --per-kind")
    p.add_argument("--seed", type=int, default=20261112)
    p = sub.add_parser("judge")
    p.add_argument("--judge", choices=sorted(JUDGES), required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=None, help="file with item ids to run")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=24000)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--breaker", type=int, default=5)
    p.add_argument("--min-review-tokens", type=int, default=600,
                   help="re-ask a PASS shorter than this once (0 disables the lazy-review guard)")
    p = sub.add_parser("aggregate")
    p.add_argument("--items", required=True)
    p.add_argument("--dir", required=True, help="directory holding kimi.jsonl and deepseek.jsonl")
    p.add_argument("--quote-policy", choices=("enforce", "report", "off"), default="enforce",
                   help="enforce: unanchored findings do not count; report: only measured (legacy verdicts)")
    p.add_argument("--min-pass-confidence", type=int, default=70)
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--min-recall-underpowered", type=float, default=0.80)
    p.add_argument("--gate-min-planted", type=int, default=30)
    p.add_argument("--min-either", type=float, default=0.97)
    p.add_argument("--allow-uncalibrated", action="store_true")
    args = parser.parse_args()
    return {"build": cmd_build, "corrupt": cmd_corrupt, "judge": cmd_judge, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
