#!/usr/bin/env python3
"""Teacher-trace generator v2: the v4.1 no-gold generator plus a deterministic output layer.

The v4.1 generator (sft_easy_nogold_v41/build_nogold_traces.py) is imported unchanged
and keeps all of its behavior: no gold in the prompt, v4.1 prompt assets, `stop` sent
and every turn clipped at its first </tool_call>. This file adds three things, taken
from the MiroFlow GAIA project (deterministic output layer, reject-and-resample
instead of in-context nudges, telemetry that a gate can read):

1. Entity repair. Inside <tool_call> and <final_answer> blocks, `&lt;`, `&gt;` and
   `&amp;` are turned back into `<`, `>` and `&` before the turn is parsed or
   executed (see repair_entities for the `&amp;` choice). In the GPT-5.6 Sol pilot
   29 of 395 traces (7.3%) sent `&gt;`, hit validation_error and then fixed it;
   those turns now execute as intended.
2. Turn-level resampling. A turn that the protocol parser rejects is discarded and
   requested again from the same context, at most SFT_TURN_RESAMPLES times
   (default 2). The rejected text never enters the transcript, so the trace holds
   no correction messages that the student could not reproduce at inference time.
   If the resample request would repeat a temperature-0 call, it is sent at
   SFT_RESAMPLE_TEMPERATURE (default 0.7) so it can differ.
3. Telemetry. Each raw attempt gets a "pipeline" object: API calls, tokens and cost
   (OpenRouter usage accounting), finish reasons, clipped turns, entity repairs,
   resampled turns with the parser errors, and unrecovered protocol turns.
   scan_traces.py and run_guard.py read it.

SFT_TEMPERATURE_OVERRIDE, when set, replaces the request temperature for every turn
(used by adaptive resampling to draw real samples from a teacher whose greedy
decoding is deterministic).

The command line is the generator's own (`generate`, `merge`); launch it through
launch_jobs_v2.sh, which differs from the v4.1 launcher only in SCRIPT.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

V41_PATH = Path("/data/keenSQL/train_prompt/sft_easy_nogold_v41/build_nogold_traces.py")
PIPELINE_VERSION = "trace_pipeline_v1"

_spec = importlib.util.spec_from_file_location("_nogold_v41_for_v2", V41_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot import {V41_PATH}")
v41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v41)
core = v41.core
easy = v41.easy
T = core.T

_TLS = threading.local()


def new_telemetry() -> dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "finish_reasons": {},
        "clipped_turns": 0,
        "closed_at_stop": 0,
        "entity_repairs": 0,
        "entity_repaired_turns": 0,
        "amp_only_turns": 0,
        "resampled_turns": 0,
        "resample_errors": [],
        "unrecovered_protocol_turns": 0,
        "temperature_override": os.environ.get("SFT_TEMPERATURE_OVERRIDE") or None,
    }


def telemetry() -> dict[str, Any]:
    tele = getattr(_TLS, "tele", None)
    if tele is None:
        tele = _TLS.tele = new_telemetry()
    return tele


# --- usage accounting at the HTTP client -------------------------------------


# OpenRouter answers 402 "in_flight_budget_exhausted" when the worst-case cost of all in-flight
# requests exceeds the credits left (seen 2026-11-16 with 21 parallel Opus 5.5 requests and ~$32
# left). It is not a spend failure: the request is retried after a wait instead of reaching the
# generator's provider circuit breaker. A real out-of-credits 402 has another reason and still raises.
IN_FLIGHT_RETRIES = int(os.environ.get("SFT_IN_FLIGHT_RETRIES", "8"))
IN_FLIGHT_WAIT_S = float(os.environ.get("SFT_IN_FLIGHT_WAIT_S", "30"))


def is_in_flight_limit(exc: BaseException) -> bool:
    return "in_flight_budget_exhausted" in str(exc)


# SFT_OPENROUTER_PROVIDERS="anthropic,google-vertex" pins OpenRouter routing to those provider
# slugs, in that order, without fallback to others. Seen 2026-11-18: OpenRouter routed Opus 5.5 to
# "Claude Platform on AWS" / Amazon Bedrock, which refuse every v4.1 prompt (finish_reason
# content_filter, native refusal, 0 completion tokens, prompt still billed), while Anthropic and
# Google Vertex answer the same request normally. Unset: OpenRouter's default routing.
def provider_routing() -> dict[str, Any] | None:
    order = [p.strip() for p in os.environ.get("SFT_OPENROUTER_PROVIDERS", "").split(",") if p.strip()]
    return {"order": order, "allow_fallbacks": False} if order else None


def anthropic_cache_enabled(model: str) -> bool:
    """Anthropic models on OpenRouter cache only at explicit cache_control breakpoints
    (OpenAI models cache automatically). SFT_ANTHROPIC_CACHE=0 turns this off."""
    return model.lower().startswith("anthropic/") and os.environ.get("SFT_ANTHROPIC_CACHE", "1") != "0"


def with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` with two Anthropic cache breakpoints; the text is unchanged.

    1. The message before the real question (system prompt + few-shot, identical for every
       question), so every question reads that prefix from cache.
    2. The last message, so the next turn of the same episode reads the whole conversation
       so far from cache.
    The real question is the last user message that starts with "Database Engine:" (the
    few-shot examples start the same way, so the last one is the real one).
    """
    out = [dict(m) for m in messages]
    marks = {len(out) - 1}
    starts = [i for i, m in enumerate(out) if m.get("role") == "user"
              and isinstance(m.get("content"), str) and m["content"].startswith("Database Engine:")]
    if starts and starts[-1] >= 1:
        marks.add(starts[-1] - 1)
    for i in marks:
        content = out[i].get("content")
        if isinstance(content, str) and content:
            out[i]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    return out


class _UsageCompletions:
    """Wraps client.chat.completions: asks OpenRouter for usage and records it."""

    def __init__(self, inner: Any, openrouter: bool) -> None:
        self._inner = inner
        self._openrouter = openrouter

    def create(self, **request: Any) -> Any:
        if self._openrouter:
            extra = dict(request.get("extra_body") or {})
            extra.setdefault("usage", {"include": True})
            routing = provider_routing()
            if routing:
                extra.setdefault("provider", routing)
            request["extra_body"] = extra
            if anthropic_cache_enabled(str(request.get("model") or "")):
                request["messages"] = with_cache_breakpoints(request["messages"])
        tele = telemetry()
        for retry in range(IN_FLIGHT_RETRIES + 1):
            try:
                response = self._inner.create(**request)
                break
            except Exception as exc:  # noqa: BLE001 - only the in-flight 402 is retried
                if not is_in_flight_limit(exc) or retry == IN_FLIGHT_RETRIES:
                    raise
                tele["in_flight_retries"] = tele.get("in_flight_retries", 0) + 1
                wait = IN_FLIGHT_WAIT_S * (1 + random.random())
                print(f"[pipeline-inflight] OpenRouter in-flight budget exhausted; retry {retry + 1} in {wait:.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        tele["api_calls"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            tele["prompt_tokens"] += int(data.get("prompt_tokens") or 0)
            tele["completion_tokens"] += int(data.get("completion_tokens") or 0)
            tele["cost_usd"] += float(data.get("cost") or 0.0)
            details = data.get("prompt_tokens_details") or {}
            tele["cache_read_tokens"] = tele.get("cache_read_tokens", 0) + int(details.get("cached_tokens") or 0)
            tele["cache_write_tokens"] = tele.get("cache_write_tokens", 0) + int(details.get("cache_write_tokens") or 0)
        choices = getattr(response, "choices", None) or []
        finish = str(getattr(choices[0], "finish_reason", None)) if choices else "none"
        tele["finish_reasons"][finish] = tele["finish_reasons"].get(finish, 0) + 1
        return response


_v41_get_client = easy.adapter.get_client  # v41's stop-adding client


def get_client_v2(base_url: str, api_key: str, timeout: float) -> Any:
    inner = _v41_get_client(base_url, api_key, timeout)
    completions = _UsageCompletions(inner.chat.completions, "openrouter.ai" in str(base_url))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


easy.adapter.get_client = get_client_v2


# --- count clipping per attempt ----------------------------------------------

_v41_clip = v41.clip_at_first_tool_call


def clip_and_count(text: str, finish_reason: str | None) -> str:
    kept = _v41_clip(text, finish_reason)
    tele = telemetry()
    if len(kept) < len(text) and text[len(kept):].strip():
        tele["clipped_turns"] += 1
    elif len(kept) > len(text):
        tele["closed_at_stop"] += 1
    return kept


v41.clip_at_first_tool_call = clip_and_count


# --- entity repair -----------------------------------------------------------

_BLOCK = re.compile(r"(<(tool_call|final_answer)>)(.*?)(</\2>)", re.S)
_ENTITY = re.compile(r"&(?:lt|gt|amp);")


def repair_entities(text: str) -> tuple[str, int]:
    """Undo HTML escaping inside tool_call/final_answer blocks; return (text, repairs).

    `&amp;` is repaired even when it is the only entity: in the end-to-end check the
    teacher wrote `BrandName = 'A&amp;W'` for the stored value 'A&W'. Should a database
    really store a literal `&amp;`, the repaired query fails EX and the trace is rejected,
    which is the safe direction. Blocks whose only entity is `&amp;` are counted
    separately in telemetry (amp_only_turns) so the choice stays visible.
    """

    repairs = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal repairs
        body = match.group(3)
        found = _ENTITY.findall(body)
        if not found:
            return match.group(0)
        if all(e == "&amp;" for e in found):
            telemetry()["amp_only_turns"] += 1
        body = body.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        repairs += len(found)
        return match.group(1) + body + match.group(4)

    return _BLOCK.sub(fix, text), repairs


# --- validate, resample ------------------------------------------------------

MAX_TURN_RESAMPLES = int(os.environ.get("SFT_TURN_RESAMPLES", "2"))
RESAMPLE_TEMPERATURE = float(os.environ.get("SFT_RESAMPLE_TEMPERATURE", "0.7"))


def protocol_error(text: str) -> str | None:
    """The same parser and flags run_episode uses (the launcher never sets --lenient-scratchpad)."""

    try:
        T.parse_assistant_turn(
            text,
            require_final_scratch_pad=True,
            require_tool_scratch_pad=True,
            repair_unclosed_function=True,
        )
    except T.ProtocolError as exc:
        return str(exc)[:200]
    return None


_v41_request_completion = core.request_completion  # v41 clip wrapper


def request_completion(*args: Any, **kwargs: Any) -> str:
    override = os.environ.get("SFT_TEMPERATURE_OVERRIDE")
    if override:
        kwargs["temperature"] = float(override)
    tele = telemetry()
    text = ""
    for round_index in range(MAX_TURN_RESAMPLES + 1):
        if round_index and float(kwargs.get("temperature") or 0.0) == 0.0:
            kwargs["temperature"] = RESAMPLE_TEMPERATURE
        text = _v41_request_completion(*args, **kwargs)
        text, repairs = repair_entities(text)
        if repairs:
            tele["entity_repairs"] += repairs
            tele["entity_repaired_turns"] += 1
            print(f"[pipeline-entity] repaired {repairs} entities", file=sys.stderr, flush=True)
        error = protocol_error(text)
        if error is None:
            return text
        if round_index < MAX_TURN_RESAMPLES:
            tele["resampled_turns"] += 1
            tele["resample_errors"].append(error[:160])
            print(f"[pipeline-resample] {error[:160]}", file=sys.stderr, flush=True)
    tele["unrecovered_protocol_turns"] += 1
    return text


core.request_completion = request_completion


# --- attach telemetry to each attempt ----------------------------------------

_previous_run_attempt = core.run_attempt


def run_attempt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _TLS.tele = new_telemetry()
    try:
        result = _previous_run_attempt(*args, **kwargs)
    finally:
        tele = _TLS.tele
        _TLS.tele = None
    tele["cost_usd"] = round(tele["cost_usd"], 6)
    result["pipeline"] = tele
    return result


core.run_attempt = run_attempt


if __name__ == "__main__":
    raise SystemExit(core.main())
