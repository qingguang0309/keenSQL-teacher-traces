#!/usr/bin/env python3
"""Kimi/DeepSeek/Qwen adapter for the frozen first-round trace builder."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
from pathlib import Path
from typing import Any, Mapping

UPSTREAM_DIR = Path("/data/keenSQL/train_prompt/sft_privileged_traces")
UPSTREAM = UPSTREAM_DIR / "build_sft_traces.py"
EXPECTED_SHA256 = "c70af902f7bd5c4989f41ec20bc374f3c7d03dea6a26468eed436011fc358169"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not UPSTREAM.is_file():
    raise FileNotFoundError(f"missing first-round builder: {UPSTREAM}")
actual_hash = file_sha256(UPSTREAM)
if actual_hash != EXPECTED_SHA256 and os.environ.get("ALLOW_UPSTREAM_DRIFT") != "1":
    raise RuntimeError(
        "first-round builder changed; refusing silent behavior drift: "
        f"expected={EXPECTED_SHA256} actual={actual_hash}. "
        "Review the change, then set ALLOW_UPSTREAM_DRIFT=1 only if intentional."
    )

spec = importlib.util.spec_from_file_location("_sft_first_round_core", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

_CLIENTS = threading.local()


def get_client(base_url: str, api_key: str, timeout: float):
    key = (base_url, api_key, timeout)
    if getattr(_CLIENTS, "key", None) != key:
        from openai import OpenAI

        _CLIENTS.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=int(os.environ.get("SFT_MAX_RETRIES", "4")),
        )
        _CLIENTS.key = key
    return _CLIENTS.client


def request_completion(
    *,
    messages: list[dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    seed: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Call official OpenAI-compatible APIs without local-vLLM-only fields.

    Hosted APIs do not consistently accept `seed` or `chat_template_kwargs`, so
    neither is sent. DeepSeek defaults to non-thinking mode in the launcher so
    temperature=0 and temperature=1 retain their greedy/pass@8 meanings. Kimi
    K2.6 non-thinking mode has a provider-fixed temperature of 0.6; pass@8 is
    therefore implemented as eight independent requests at that fixed value.
    """

    del seed, chat_template_kwargs
    client = get_client(base_url, api_key, timeout)
    provider = os.environ.get("SFT_API_PROVIDER", "").strip().lower()
    thinking_mode = os.environ.get("SFT_THINKING_MODE", "omit").strip().lower()
    if thinking_mode not in {"omit", "enabled", "disabled"}:
        raise ValueError("SFT_THINKING_MODE must be omit, enabled, or disabled")

    effective_temperature = temperature
    if (
        provider == "kimi"
        and model.strip().lower() == "kimi-k2.6"
        and thinking_mode == "disabled"
    ):
        effective_temperature = 0.6

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": effective_temperature,
        "max_tokens": max_tokens,
    }
    reasoning_effort = os.environ.get("SFT_REASONING_EFFORT", "").strip()
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    if thinking_mode != "omit":
        if provider == "qwen":
            request["extra_body"] = {"enable_thinking": thinking_mode == "enabled"}
        else:
            request["extra_body"] = {"thinking": {"type": thinking_mode}}

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        # Kimi model families differ on the preferred output-limit field. Retry
        # only when the rejected parameter is explicitly named by the server.
        text = str(exc).lower()
        if "max_tokens" not in text and "max_completion_tokens" not in text:
            raise
        request["max_completion_tokens"] = request.pop("max_tokens")
        response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    return content or ""


core.get_client = get_client
core.request_completion = request_completion

_first_round_run_attempt = core.run_attempt


def run_attempt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Stop the job after provider failures, except permanent content rejects.

    The OpenAI client has already exhausted its configured retries when this
    happens. Raising here prevents thousands of authentication, endpoint,
    quota, or outage failures from being materialized as completed attempts.
    The failing attempt remains pending and will be retried on resume. Qwen
    ``data_inspection_failed`` responses are different: retrying the same
    immutable prompt cannot fix a permanent provider-side content rejection.
    Record that single attempt as rejected so the job can continue and resume
    logic will not repeatedly stop on it.
    """

    row = _first_round_run_attempt(*args, **kwargs)
    issues = (row.get("acceptance") or {}).get("issues") or []
    if "attempt_exception" in issues:
        provider = os.environ.get("SFT_API_PROVIDER", "").strip().lower()
        exception_text = str(row.get("exception") or "").lower()
        qwen_content_rejected = provider == "qwen" and (
            "data_inspection_failed" in exception_text
            or "input text data may contain inappropriate content" in exception_text
        )
        if qwen_content_rejected:
            acceptance = dict(row.get("acceptance") or {})
            acceptance["issues"] = [
                issue for issue in issues if issue != "attempt_exception"
            ] + ["provider_content_rejected"]
            acceptance["accepted"] = False
            row["acceptance"] = acceptance
            row["accepted"] = False
            row["exception_disposition"] = "recorded_and_skipped"
            return row
        raise RuntimeError(
            "provider circuit breaker: "
            f"attempt={row.get('attempt_id')} exception={row.get('exception')}"
        )
    return row


core.run_attempt = run_attempt

# The strict auditor imports this module as `build_sft_traces`; re-export the
# first-round helpers so its behavior remains identical to the original audit.
for _name in dir(core):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(core, _name)


if __name__ == "__main__":
    raise SystemExit(core.main())
