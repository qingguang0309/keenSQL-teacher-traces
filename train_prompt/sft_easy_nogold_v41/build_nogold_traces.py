#!/usr/bin/env python3
"""No-gold easy-band adapter for the frozen first-round trace builder.

Same generator, API adapter and acceptance rules as the easy-teacher run, with
four changes:

1. the teacher sees only the student prompt: no reference SQL is appended;
2. every prompt asset comes from retoolsql_prompt/v4.1 instead of v4;
3. a single tool call is enough, because an unprivileged first draft that
   executes correctly is the answer (system prompt section 7);
4. each teacher turn ends at its first </tool_call>: the rest of a response is
   an invented observation, never real data (see "one tool call per turn").

Gold SQL is still used after generation, for EX grading only. Accepted rows are
written with data_variant="easy", which sft_train/import_traces.py treats as an
unprivileged trace and therefore exempts from the gold-leak check.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

UPSTREAM = Path(
    "/data/keenSQL/train_prompt/sft_easy_teacher_traces/build_easy_teacher_traces.py"
)
PROMPT_DIR = Path("/data/keenSQL/train_prompt/retoolsql_prompt/v4.1")
MIN_TOOL_CALLS = 1
# Gold strings shorter than this can legitimately occur in a schema or few-shot.
MIN_GOLD_CHARS_FOR_LEAK_CHECK = 25

spec = importlib.util.spec_from_file_location("_easy_nogold_upstream", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
easy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(easy)
core = easy.adapter.core
PB = core.PB


def pin_prompt_dir(directory: Path) -> None:
    """Point prompt_builder at another prompt directory.

    prompt_builder binds its asset paths as default arguments when the module is
    defined, so rebinding the module constants alone would not reach them.
    """

    paths = {
        "SYSTEM_PROMPT_PATH": directory / "system_prompt.txt",
        "FEWSHOT_PATH": directory / "fewshot.json",
        "USER_TEMPLATE_PATH": directory / "user_template.txt",
        "TEACHER_PROMPT_PATH": directory / "sft_teacher_prompt.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing prompt assets: " + ", ".join(missing))
    PB.V4_DIR = PB.DEFAULT_PROMPT_DIR = directory
    for name, path in paths.items():
        setattr(PB, name, path)
    PB.load_system_prompt.__defaults__ = (paths["SYSTEM_PROMPT_PATH"],)
    PB.load_user_template.__defaults__ = (paths["USER_TEMPLATE_PATH"],)
    PB.load_teacher_prompt.__defaults__ = (paths["TEACHER_PROMPT_PATH"],)
    PB.teacher_prompt_sha256.__defaults__ = (paths["TEACHER_PROMPT_PATH"],)
    PB.load_fewshot.__defaults__ = (paths["FEWSHOT_PATH"],)
    PB.fewshot_messages.__defaults__ = (paths["FEWSHOT_PATH"],)
    # validate_assets() and the run config read the base prompt from here.
    core.ORIGINAL_V4_DIR = directory


def build_nogold_prompts(
    record: dict[str, Any],
    schema: str,
    teacher_template: str,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Return the student prompt twice: the teacher gets no appendix."""

    built = PB.build_messages(
        engine="SQLite",
        db_details=schema,
        external_knowledge=str(record.get("evidence") or ""),
        question=str(record["question"]),
        fewshot=True,
        max_prompt_tokens=max_prompt_tokens,
    )
    deployment_messages = copy.deepcopy(built["messages"])
    teacher_messages = copy.deepcopy(built["messages"])
    gold_sql = str(record.get("SQL") or "").strip()
    if len(gold_sql) >= MIN_GOLD_CHARS_FOR_LEAK_CHECK and any(
        gold_sql in message["content"] for message in teacher_messages
    ):
        raise RuntimeError(
            f"gold SQL appears in the no-gold prompt for {record.get('question_id')}"
        )
    meta = dict(built["meta"])
    meta.update(
        {
            "base_prompt_budget": max_prompt_tokens,
            "teacher_appendix_tokens": 0,
            "teacher_prompt_tokens": meta["prompt_tokens"],
            "max_prompt_tokens": max_prompt_tokens,
            "prompt_messages": len(deployment_messages),
            "teacher_prompt_sha256": "nogold",
            "teacher_mode": "nogold",
        }
    )
    return deployment_messages, teacher_messages, meta


_easy_make_sft_record = core.make_sft_record


def make_nogold_sft_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = _easy_make_sft_record(*args, **kwargs)
    row["data_variant"] = "easy"
    metadata = row.setdefault("metadata", {})
    metadata["teacher_mode"] = "nogold"
    metadata["prompt_dir"] = str(PROMPT_DIR)
    metadata["teacher_context_in_messages"] = False
    return row


# --- one tool call per turn -------------------------------------------------
# In the 500-question GPT-5.6 Sol pilot, 67 of 500 attempts (13.4%) wrote a
# tool call, then invented its observation (and often a final answer) in the
# same response, so the parser rejected the turn. Everything after the first
# </tool_call> is the model's guess at what the database would return, never
# real data, so it is cut off before parsing and the real observation follows.
# `stop` is also sent so providers that honor it skip generating that tail;
# OpenRouter drops it silently for models that do not support it (GPT-5.6 Sol
# lists no `stop`), so the client-side cut is what guarantees the behavior.
# Set SFT_NOGOLD_SEND_STOP=0 for an API that rejects the `stop` field outright.

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
_STOP_STATE = threading.local()
STOP_STATS = {"clipped_turns": 0, "clipped_chars": 0, "closed_at_stop": 0}
_STATS_LOCK = threading.Lock()


def send_stop_enabled() -> bool:
    return os.environ.get("SFT_NOGOLD_SEND_STOP", "1").strip() != "0"


class _StopCompletions:
    """Proxy for client.chat.completions that adds `stop` and keeps finish_reason."""

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, **request: Any) -> Any:
        if send_stop_enabled():
            request.setdefault("stop", [TOOL_CALL_END])
        response = self._completions.create(**request)
        choices = getattr(response, "choices", None) or []
        _STOP_STATE.finish_reason = (
            getattr(choices[0], "finish_reason", None) if choices else None
        )
        return response


class _StopClient:
    def __init__(self, client: Any) -> None:
        self.chat = SimpleNamespace(completions=_StopCompletions(client.chat.completions))


def clip_at_first_tool_call(text: str, finish_reason: str | None) -> str:
    """Keep the turn up to and including its first </tool_call>.

    A provider that honored `stop` returns the call without its closing tag
    (finish_reason "stop"); the tag is restored so the turn matches the
    few-shot format. An unclosed call cut by max_tokens is left alone, so it
    still fails the way a truncated response always has.
    """

    end = text.find(TOOL_CALL_END)
    if end >= 0:
        kept = text[: end + len(TOOL_CALL_END)]
        dropped = text[len(kept):]
        if dropped.strip():
            with _STATS_LOCK:
                STOP_STATS["clipped_turns"] += 1
                STOP_STATS["clipped_chars"] += len(dropped)
            print(
                f"[nogold-stop] dropped {len(dropped)} chars after the first "
                f"{TOOL_CALL_END}",
                file=sys.stderr,
                flush=True,
            )
        return kept
    if finish_reason == "stop" and TOOL_CALL_START in text:
        with _STATS_LOCK:
            STOP_STATS["closed_at_stop"] += 1
        return text + ("" if text.endswith("\n") else "\n") + TOOL_CALL_END
    return text


_adapter_get_client = easy.adapter.get_client
_adapter_request_completion = core.request_completion


def get_stop_client(*args: Any, **kwargs: Any) -> _StopClient:
    return _StopClient(_adapter_get_client(*args, **kwargs))


def request_completion(*args: Any, **kwargs: Any) -> str:
    _STOP_STATE.finish_reason = None
    text = _adapter_request_completion(*args, **kwargs)
    return clip_at_first_tool_call(text, getattr(_STOP_STATE, "finish_reason", None))


# The adapter's request_completion looks get_client up in its own module.
easy.adapter.get_client = get_stop_client
core.request_completion = request_completion

pin_prompt_dir(PROMPT_DIR)
core.build_prompts = build_nogold_prompts
core.make_sft_record = make_nogold_sft_record
core.MIN_TEACHER_TOOL_CALLS = MIN_TOOL_CALLS


if __name__ == "__main__":
    raise SystemExit(core.main())
