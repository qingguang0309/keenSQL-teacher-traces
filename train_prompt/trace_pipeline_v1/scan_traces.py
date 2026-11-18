#!/usr/bin/env python3
"""Scan a generation output root for anomalies (smoke gate and in-run tripwire).

Borrowed from MiroFlow GAIA scripts_sol/scan_trace_errors.py + check_smoke.py: tool and API
failures hide inside per-attempt records and never reach the console, so a run can look
healthy while every call fails. This reads <root>/models/*/*/chunk_*/raw_attempts.jsonl.

Blocking rules (thresholds are flags; `--profile smoke` is strict, `--profile guard` is the
in-run tripwire and waits for --min-attempts before judging rates):
  telemetry       every attempt carries the v2 "pipeline" object (else the v1 generator ran)
  expected_model  every attempt used --expect-model
  api_persistent  any auth / payment / region (401, 402, 403) failure
  api_transient   rate-limit, 5xx, timeout, connection failures > --max-api-error-rate
  protocol        attempts ending in a protocol error > --max-protocol-rate
  entities        any &gt;/&lt;/&amp; left inside an executed tool call (repair layer failed)
  validation      validation_error observations per tool call > --max-validation-rate
  length          finish_reason=length per API call > --max-length-rate
  cost            mean cost per attempt > --max-cost-per-attempt
  accept          acceptance rate < --min-accept (smoke profile only)
Exit 0 = pass, 1 = blocking anomaly, 2 = nothing to scan.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

PROFILES = {
    "smoke": {"min_attempts": 1, "max_api_error_rate": 0.0, "max_protocol_rate": 0.10,
              "max_validation_rate": 0.05, "max_length_rate": 0.05, "max_cost_per_attempt": 0.30,
              "min_accept": 0.30},
    "guard": {"min_attempts": 20, "max_api_error_rate": 0.10, "max_protocol_rate": 0.10,
              "max_validation_rate": 0.08, "max_length_rate": 0.08, "max_cost_per_attempt": 0.40,
              "min_accept": 0.0},
}
ENTITY_IN_CALL = re.compile(r"<tool_call>(?:(?!</tool_call>).)*&(?:gt|lt|amp);", re.S)


def classify_exception(text: str) -> str:
    t = text.lower()
    if "401" in t or "authentication" in t or "invalid api key" in t:
        return "auth"
    if "402" in t or "insufficient" in t or "credits" in t:
        return "payment"
    if "403" in t or "region" in t or "permissiondenied" in t:
        return "forbidden"
    if "429" in t or "rate limit" in t or "ratelimit" in t:
        return "rate"
    if re.search(r"\b5\d\d\b", t) or "internalserver" in t or "bad gateway" in t:
        return "server"
    if "timeout" in t or "timed out" in t:
        return "timeout"
    if "connection" in t:
        return "connection"
    return "other"


PERSISTENT = {"auth", "payment", "forbidden"}
API_CATEGORIES = PERSISTENT | {"rate", "server", "timeout", "connection"}


def scan(root: Path, *, expect_model: str | None = None) -> dict[str, Any]:
    files = C.raw_attempt_files(root)
    m: dict[str, Any] = Counter()
    exceptions: Counter = Counter()
    finish: Counter = Counter()
    models: Counter = Counter()
    samples: dict[str, str] = {}
    cost = 0.0
    for path in files:
        for row in C.iter_jsonl(path):
            m["attempts"] += 1
            m["accepted"] += bool(row.get("accepted"))
            models[str(row.get("model"))] += 1
            if row.get("exception"):
                category = classify_exception(row["exception"])
                exceptions[category] += 1
                samples.setdefault(f"exception:{category}", row["exception"][:200])
            tele = row.get("pipeline")
            if tele:
                m["with_telemetry"] += 1
                cost += float(tele.get("cost_usd") or 0.0)
                m["api_calls"] += int(tele.get("api_calls") or 0)
                for key in ("clipped_turns", "entity_repairs", "entity_repaired_turns", "amp_only_turns",
                            "resampled_turns", "unrecovered_protocol_turns"):
                    m[key] += int(tele.get(key) or 0)
                finish.update(tele.get("finish_reasons") or {})
            episode = row.get("episode") or {}
            errors = episode.get("errors") or []
            if any(str(e).startswith("protocol") for e in errors):
                m["protocol_error_attempts"] += 1
                samples.setdefault("protocol", str(errors[0])[:200])
            if any("budget" in str(e) for e in errors):
                m["tool_budget_exceeded"] += 1
            for message in episode.get("transcript") or []:
                content = str(message.get("content") or "")
                if message.get("role") == "observation":
                    m["observations"] += 1
                    if '"error"' in content[:40]:
                        m["error_observations"] += 1
                        if "validation_error" in content[:80]:
                            m["validation_error_observations"] += 1
                            samples.setdefault("validation_error", content[:200])
                elif message.get("role") == "assistant" and ENTITY_IN_CALL.search(content):
                    m["entity_left_in_call"] += 1
    attempts = m["attempts"]
    api_errors = sum(exceptions[c] for c in API_CATEGORIES)
    metrics = {
        "files": len(files),
        **{k: int(v) for k, v in m.items()},
        "accept_rate": round(m["accepted"] / attempts, 4) if attempts else None,
        "telemetry_coverage": round(m["with_telemetry"] / attempts, 4) if attempts else None,
        "exceptions": dict(exceptions),
        "api_error_rate": round(api_errors / attempts, 4) if attempts else None,
        "protocol_rate": round(m["protocol_error_attempts"] / attempts, 4) if attempts else None,
        "validation_rate": round(m["validation_error_observations"] / m["observations"], 4) if m["observations"] else 0.0,
        "finish_reasons": dict(finish),
        "length_rate": round(finish.get("length", 0) / max(1, sum(finish.values())), 4),
        "cost_usd": round(cost, 4),
        "cost_per_attempt": round(cost / m["with_telemetry"], 5) if m["with_telemetry"] else None,
        "cost_per_accepted": round(cost / m["accepted"], 5) if m["accepted"] else None,
        "models": dict(models),
        "samples": samples,
    }
    if expect_model:
        metrics["expected_model"] = expect_model
    return metrics


def judge(metrics: dict[str, Any], limits: dict[str, Any], profile: str) -> list[str]:
    blocking = []
    attempts = metrics.get("attempts", 0)
    exc = metrics.get("exceptions", {})
    if any(exc.get(c) for c in PERSISTENT):
        blocking.append(f"api_persistent: {{{', '.join(f'{c}={exc[c]}' for c in PERSISTENT if exc.get(c))}}}")
    if metrics.get("expected_model"):
        others = {k: v for k, v in metrics["models"].items() if k != metrics["expected_model"]}
        if others:
            blocking.append(f"expected_model: other models in attempts {others}")
    if metrics.get("entity_left_in_call"):
        blocking.append(f"entities: {metrics['entity_left_in_call']} executed calls still contain &gt;/&lt;/&amp;")
    if attempts and metrics["telemetry_coverage"] < 1.0:
        blocking.append(f"telemetry: coverage {metrics['telemetry_coverage']} (v1 generator ran?)")
    if attempts < limits["min_attempts"]:
        return blocking
    checks = [
        ("api_transient", metrics["api_error_rate"], limits["max_api_error_rate"], ">"),
        ("protocol", metrics["protocol_rate"], limits["max_protocol_rate"], ">"),
        ("validation", metrics["validation_rate"], limits["max_validation_rate"], ">"),
        ("length", metrics["length_rate"], limits["max_length_rate"], ">"),
        ("cost", metrics["cost_per_attempt"] or 0.0, limits["max_cost_per_attempt"], ">"),
    ]
    if profile == "smoke":
        checks.append(("accept", metrics["accept_rate"] or 0.0, limits["min_accept"], "<"))
    for name, value, limit, op in checks:
        if (op == ">" and value > limit) or (op == "<" and value < limit):
            blocking.append(f"{name}: {value} {op} limit {limit}")
    return blocking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="generation OUTPUT_ROOT")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--expect-model", default=None)
    parser.add_argument("--out", type=Path, default=None)
    for key, value in PROFILES["smoke"].items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=None)
    args = parser.parse_args()
    limits = dict(PROFILES[args.profile])
    for key in limits:
        override = getattr(args, key)
        if override is not None:
            limits[key] = override
    metrics = scan(args.root, expect_model=args.expect_model)
    if not metrics["attempts"]:
        print(f"[scan] no attempts under {args.root}")
        return 2
    blocking = judge(metrics, limits, args.profile)
    result = {"created_at_utc": C.utc_now(), "root": str(args.root), "profile": args.profile,
              "limits": limits, "metrics": metrics, "blocking": blocking, "passed": not blocking}
    if args.out:
        C.write_json(args.out, result)
    print(json.dumps({k: metrics[k] for k in ("attempts", "accepted", "accept_rate", "telemetry_coverage",
                                              "api_error_rate", "protocol_rate", "validation_rate",
                                              "length_rate", "entity_repairs", "resampled_turns",
                                              "clipped_turns", "cost_usd", "cost_per_accepted")
                      if k in metrics}, ensure_ascii=False))
    for line in blocking:
        print(f"[scan] BLOCK {line}")
    print(f"[scan] {'PASS' if not blocking else 'FAIL'} ({args.profile})")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
