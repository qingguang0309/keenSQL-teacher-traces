#!/usr/bin/env python3
"""Recovery-round wrapper around build_sft_traces.py.

It preserves original source indexes embedded in the filtered input and raises
the protocol budget to eight tool calls plus one final-answer turn.  The base
generator, grader, prompt builder, and exported SFT format remain unchanged.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Mapping

import build_sft_traces as base


_base_acceptance_audit = base.acceptance_audit
_base_request_completion = base.request_completion
_base_run_attempt = base.run_attempt
_base_dispatch = base.st.dispatch
_PROTOCOL_TAGS = (
    "scratch_pad",
    "tool_call",
    "function",
    "parameter",
    "relevant_tables",
    "relevant_columns",
    "final_answer",
)


def _tag_spans(text: str, tag: str, closing: bool = False) -> list[re.Match[str]]:
    slash = r"/\s*" if closing else ""
    suffix = r"\s*>" if closing else r"(?:\s|=|>)"
    return list(re.finditer(rf"<\s*{slash}{tag}{suffix}", text, re.I))


def _insert_single_missing_close(
    text: str, tag: str, boundary_patterns: tuple[str, ...]
) -> tuple[str, str | None]:
    openings = _tag_spans(text, tag)
    closings = _tag_spans(text, tag, closing=True)
    if len(openings) != len(closings) + 1:
        return text, None

    tokens = sorted(
        [(match.start(), 1, match) for match in openings]
        + [(match.start(), -1, match) for match in closings]
    )
    stack: list[re.Match[str]] = []
    for _position, kind, match in tokens:
        if kind == 1:
            stack.append(match)
        elif stack:
            stack.pop()
    if len(stack) != 1:
        return text, None

    start = stack[0].end()
    boundaries: list[int] = []
    for pattern in boundary_patterns:
        match = re.search(pattern, text[start:], re.I | re.S)
        if match:
            boundaries.append(start + match.start())
    if not boundaries:
        return text, None
    position = min(boundaries)
    repaired = text[:position].rstrip() + f"\n</{tag}>\n" + text[position:].lstrip()
    return repaired, f"insert_missing_{tag}_close"


def repair_completion_tags(text: str) -> tuple[str, list[str]]:
    """Repair only unambiguous closing-tag defects without changing content."""
    repaired = text
    labels: list[str] = []

    if len(_tag_spans(repaired, "relevant_tables")) == 1 and not _tag_spans(
        repaired, "relevant_tables", closing=True
    ):
        typo = list(re.finditer(r"<\s*/\s*related_tables\s*>", repaired, re.I))
        if len(typo) == 1:
            match = typo[0]
            repaired = repaired[: match.start()] + "</relevant_tables>" + repaired[match.end() :]
            labels.append("repair_related_tables_closing_typo")

    repairs = (
        ("final_answer", (r"\Z",)),
        ("tool_call", (r"\Z",)),
        ("scratch_pad", (r"<\s*(?:tool_call|final_answer)(?:\s|>)",)),
        ("function", (r"<\s*/\s*tool_call\s*>",)),
        (
            "parameter",
            (
                r"<\s*parameter(?:\s|=|>)",
                r"<\s*/\s*function\s*>",
                r"<\s*/\s*tool_call\s*>",
            ),
        ),
        (
            "relevant_tables",
            (r"<\s*relevant_columns(?:\s|=|>)", r"<\s*/\s*final_answer\s*>"),
        ),
        ("relevant_columns", (r"<\s*/\s*final_answer\s*>",)),
    )
    for tag, boundaries in repairs:
        repaired, label = _insert_single_missing_close(repaired, tag, boundaries)
        if label:
            labels.append(label)
    return repaired, labels


def request_completion(*args: Any, **kwargs: Any) -> str:
    raw = _base_request_completion(*args, **kwargs)
    repaired, labels = repair_completion_tags(raw)
    if labels:
        history = getattr(base._THREAD_LOCAL, "recovery_format_repairs", None)
        if isinstance(history, list):
            history.extend(labels)
    return repaired


def dispatch(*args: Any, **kwargs: Any) -> str:
    """Mark resource-budget failures as non-semantic so the model can finalize."""
    observation = _base_dispatch(*args, **kwargs)
    try:
        payload = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return observation
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return observation
    kind = str(payload["error"].get("kind") or "")
    if kind not in {"step_limit_exceeded", "timeout"}:
        return observation
    payload["recovery_guidance"] = (
        "This is an environment execution-budget failure, not evidence that the "
        "SQL semantics are wrong. Do not retry or reformulate an equivalent SQL "
        "solely for this error. If this call attempted the intended final SQL, "
        "return the exact same SQL in <final_answer> and cite this call now; the "
        "external EX grader will determine correctness."
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def run_attempt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    base._THREAD_LOCAL.recovery_format_repairs = []
    result = _base_run_attempt(*args, **kwargs)
    repairs = list(getattr(base._THREAD_LOCAL, "recovery_format_repairs", []))
    if repairs:
        result["format_repairs"] = repairs
        sft_record = result.get("sft_record")
        if isinstance(sft_record, dict):
            metadata = sft_record.setdefault("metadata", {})
            metadata["format_repairs"] = repairs
    return result


def acceptance_audit(
    *,
    episode: Mapping[str, Any],
    grade: Mapping[str, Any],
    record: Mapping[str, Any],
    schema: str,
) -> dict[str, Any]:
    """Keep the base audit and require the final exact query to be the last call."""
    audit = _base_acceptance_audit(
        episode=episode,
        grade=grade,
        record=record,
        schema=schema,
    )
    issues = list(audit.get("issues") or [])
    calls = episode.get("executed_calls")
    final_sql = str(episode.get("final_sql") or "")
    last_call = calls[-1] if isinstance(calls, list) and calls else {}
    last_is_exact_final = (
        isinstance(last_call, Mapping)
        and last_call.get("tool") == "sqlite_query"
        and base.normalize_sql(str(last_call.get("sql") or ""))
        == base.normalize_sql(final_sql)
    )
    if not last_is_exact_final:
        issues.append("final_sql_not_last_tool_call")
    verification = episode.get("verification") or {}
    if calls and verification.get("declared") != len(calls):
        issues.append("verified_call_is_not_last_tool_call")
    assistant_text = "\n".join(
        str(message.get("content") or "")
        for message in episode.get("transcript", [])
        if message.get("role") == "assistant"
    )
    if any(
        len(_tag_spans(assistant_text, tag))
        != len(_tag_spans(assistant_text, tag, closing=True))
        for tag in _PROTOCOL_TAGS
    ):
        issues.append("unbalanced_protocol_tags_after_repair")
    audit["issues"] = list(dict.fromkeys(issues))
    audit["accepted"] = not audit["issues"]
    return audit


def select_records(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    """Select records while retaining their indexes in the original hard set."""
    if n < 0:
        raise ValueError("--n must be zero (all) or positive")
    indexed: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for filtered_index, record in enumerate(records):
        raw_index = record.get("__source_index", filtered_index)
        if isinstance(raw_index, bool):
            raise ValueError(f"record {filtered_index}: invalid __source_index")
        try:
            source_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {filtered_index}: invalid __source_index={raw_index!r}"
            ) from exc
        if source_index < 0 or source_index in seen:
            raise ValueError(
                f"record {filtered_index}: duplicate/negative __source_index={source_index}"
            )
        seen.add(source_index)
        indexed.append((source_index, record))
    if n and n < len(indexed):
        indexed = random.Random(seed).sample(indexed, n)
        indexed.sort(key=lambda pair: pair[0])
    return indexed


base.select_records = select_records
base.acceptance_audit = acceptance_audit
base.request_completion = request_completion
base.run_attempt = run_attempt
base.st.dispatch = dispatch
base.MIN_TEACHER_TOOL_CALLS = 1
base.PREFERRED_MAX_TEACHER_TOOL_CALLS = 4
base.MAX_TEACHER_TOOL_CALLS = 8
base.PREFERRED_MAX_TEACHER_ASSISTANT_TURNS = 7
base.MAX_TEACHER_ASSISTANT_TURNS = 9


if __name__ == "__main__":
    raise SystemExit(base.main())
