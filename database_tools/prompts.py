#!/usr/bin/env python3
"""ReToolSQL v4 prompt and turn-protocol helpers for the SQLite tools.

The canonical prompt lives in ``train_prompt/retoolsql_prompt/v4`` and is the
single source of truth shared by SFT, RL, and inference:

  fixed prefix   system prompt + two static few-shot episodes
  user message   engine, M-Schema, external knowledge, question

This module does not restate that prompt. It loads it, exposes the tool section
(``RETOOLSQL_TOOL_PROMPT``) for consumers that need only the tool contract, and
assembles the exact message list a chat template consumes. The tool declarations
come from ``sqlite_tools.TOOL_SCHEMAS``, so the prompt, the schemas, and the
execution layer cannot drift apart.

The v4 system prompt defines the turn protocol used by the harness: every
assistant turn carries a bounded ``<scratch_pad>`` followed by exactly one
``<tool_call>`` or the ``<final_answer>`` block, and the final-answer scratch pad
names the executed ``sqlite_query`` call through ``Verified call: <k>``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    from sqlite_tools import TOOL_NAMES, TOOL_SCHEMAS
except ImportError:  # pragma: no cover - package-style import fallback
    from .sqlite_tools import TOOL_NAMES, TOOL_SCHEMAS  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

SCRATCH_PAD_TAG = "scratch_pad"
TOOL_CALL_TAG = "tool_call"
FINAL_ANSWER_TAG = "final_answer"
RELEVANT_TABLES_TAG = "relevant_tables"
RELEVANT_COLUMNS_TAG = "relevant_columns"

PROTOCOL_TAGS = (
    SCRATCH_PAD_TAG,
    TOOL_CALL_TAG,
    FINAL_ANSWER_TAG,
    RELEVANT_TABLES_TAG,
    RELEVANT_COLUMNS_TAG,
)

# Qwen-style textual tool call, the v4 protocol.
TEXT_TOOL_CALL_FORMAT = (
    "<tool_call>\n<function=sqlite_query>\n<parameter=sql>\nSELECT ...\n"
    "</parameter>\n</function>\n</tool_call>"
)

_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "train_prompt" / "retoolsql_prompt"
# v4 is the only maintained prompt; the old v3 snapshot was removed.
PROMPT_DIR = _PROMPT_ROOT / "v4"
if not (PROMPT_DIR / "system_prompt.txt").is_file():
    raise FileNotFoundError(
        f"canonical prompt assets are missing: {PROMPT_DIR / 'system_prompt.txt'}"
    )
PROMPT_VERSION = PROMPT_DIR.name
PROMPT_ROOT = _PROMPT_ROOT
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt.txt"
USER_TEMPLATE_FILE = PROMPT_DIR / "user_template.txt"
FEWSHOT_FILE = PROMPT_DIR / "fewshot.json"

# Standalone fallback used only when the prompt files are absent. It keeps the
# v4 contract: tool signatures and limits, the closure self-check, and the
# mechanically checkable ``Verified call: <k>`` line. The repository layout
# always prefers the full v4 files.
_EMBEDDED_SYSTEM_PROMPT = """# ReToolSQL v4 — Tool-Using Text-to-SQL Agent

## 1. Role
You are a Text-to-SQL agent that answers one question over one SQLite database.
Your deliverable is a single complete SELECT or WITH statement whose result
answers the question. You may call read-only database tools across at most 8
turns to verify and ground that statement.

## 2. Tools
Tools take no database identifier and are read-only and deterministic. Call at
most one per turn.

### sqlite_query(sql, max_return_rows=200)
Runs exactly one read-only SELECT or WITH statement. max_return_rows is in
[1, 10000]. Success: {"columns", "rows", "row_count", "truncated",
"output_truncated"}. Failure: {"error": {"kind", "message"}}. Use it to test
drafts, count rows, and re-verify repairs.

### sqlite_peek(table, columns=["..."], limit=5, where="...")
Profiles up to 100 columns over the first 20,000 rows: type, nulls, distinct,
min/max, top_values, samples, character classes, prefixes, plus _scan. Call it
only on a concrete signal: fraction-vs-percent scale, coded values, a date-like
observed_pattern, a join key with mismatched types or match_rate below 100%, or
an implausible result magnitude.

### bm25_search_sqlite(table, column, query, top_k=10, where="...")
Searches distinct stored values with BM25 plus a case-insensitive regex
fallback. Returns at most 2 x top_k {"value", "bm25", "regex"} entries. Use it
when a WHERE literal may differ in storage: paraphrase, abbreviation, plural,
punctuation, diacritics, or casing.

### Error kinds
validation_error, execution_error, timeout, step_limit_exceeded,
infrastructure_error. Treat every error as a recoverable observation and use its
message as repair guidance.

## 3. Turn protocol
Every assistant turn is a bounded <scratch_pad> followed by exactly one action:
<scratch_pad>
Intent: ... Expected output columns: ... Candidate SQL: ...
</scratch_pad>
then either one <tool_call>...</tool_call> or the final-answer block. Never mix
both and never emit <observation>.

Tool calls use:
<tool_call>
<function=sqlite_query>
<parameter=sql>
SELECT ...
</parameter>
</function>
</tool_call>
Exactly one <function=...>...</function> block; one <parameter=name>...</parameter>
per argument; omit unneeded optional arguments.
Closure self-check before sending: <function=X> needs </function>, and
<tool_call> needs </tool_call>.

Final answer:
<scratch_pad>
Intent: ...
Expected output columns: ...
Verified call: <k>
</scratch_pad>
<relevant_tables>
table_a, table_b
</relevant_tables>
<relevant_columns>
table_a.col1, table_b.col2
</relevant_columns>
<final_answer>
SELECT ...
</final_answer>
Verified call k is the 1-based index of the tool call in this episode, counting
every sqlite_query, sqlite_peek, and bm25_search_sqlite call in order. The cited
call must be a sqlite_query whose SQL is exactly the text you submit.

## 4. Hard constraints
Never fabricate tables, columns, stored values, tool observations, or execution
results. <final_answer> contains SQL only. <relevant_tables> lists base tables,
not CTE names. <relevant_columns> lists every referenced column as
table.column. No markdown fences and no prose after the final answer.
"""


def _allow_embedded() -> bool:
    return os.environ.get("KEENSQL_ALLOW_EMBEDDED_PROMPT") == "1"


def _load_text(path: Path, fallback: str) -> str:
    """Load a prompt asset, failing loudly instead of silently changing the prompt.

    SFT, RLVR, and evaluation must all see the same v4 prefix. A missing asset
    used to fall back to a shortened embedded prompt, which silently changed
    the training distribution. The fallback now requires an explicit opt-in.
    """

    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if _allow_embedded():
            logger.warning("prompt asset %s is unavailable (%s); using embedded fallback", path, exc)
            return fallback
        raise FileNotFoundError(
            f"required prompt asset is missing: {path}. "
            "Set KEENSQL_ALLOW_EMBEDDED_PROMPT=1 only for isolated unit tests."
        ) from exc


def _load_fewshot_examples(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if _allow_embedded():
            logger.warning("few-shot file %s is unavailable (%s); using no examples", path, exc)
            return []
        raise FileNotFoundError(
            f"required few-shot file is missing or invalid: {path}. "
            "Set KEENSQL_ALLOW_EMBEDDED_PROMPT=1 only for isolated unit tests."
        ) from exc
    examples = payload.get("examples", [])
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"few-shot file has no examples: {path}")
    return examples


RETOOLSQL_SYSTEM_PROMPT = _load_text(SYSTEM_PROMPT_FILE, _EMBEDDED_SYSTEM_PROMPT)
FEWSHOT_EXAMPLES = _load_fewshot_examples(FEWSHOT_FILE)


def _extract_section(text: str, *headings: str) -> str:
    """Return the text under the first matching heading, up to the next ``## ``.

    Section numbers move between prompt revisions, so callers pass candidate
    headings and the first one present wins.
    """

    for heading in headings:
        begin = text.find(heading)
        if begin < 0:
            continue
        begin += len(heading)
        finish = text.find("\n## ", begin)
        return text[begin:finish].strip() if finish >= 0 else text[begin:].strip()
    return ""


# The exact tool contract from the v4 system prompt, for consumers that render
# tool instructions separately from the rest of the system prompt.
RETOOLSQL_TOOL_PROMPT = _extract_section(
    RETOOLSQL_SYSTEM_PROMPT, "## 3. Tools", "## 2. Tools"
)

# The four user fields mirror the ReToolSQL task format; external knowledge may
# be an empty string.
_USER_TEMPLATE_FALLBACK = (
    "Database Engine:\n{engine}\n\n"
    "Database Schema:\n{db_details}\n\n"
    "External Knowledge:\n{external_knowledge}\n\n"
    "Question:\n{question}"
)
USER_MESSAGE_TEMPLATE = _load_text(USER_TEMPLATE_FILE, _USER_TEMPLATE_FALLBACK)


def _validate_fields(fields: Mapping[str, Any]) -> None:
    for name, value in fields.items():
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")


def build_user_message(
    *,
    engine: str,
    db_details: str,
    external_knowledge: str,
    question: str,
) -> Dict[str, str]:
    """Build the four-field user message (engine, schema, knowledge, question)."""

    fields: Mapping[str, Any] = {
        "engine": engine,
        "db_details": db_details,
        "external_knowledge": external_knowledge,
        "question": question,
    }
    _validate_fields(fields)
    return {"role": "user", "content": USER_MESSAGE_TEMPLATE.format(**fields)}


def build_user_prompt(
    engine: str, db_details: str, external_knowledge: str, question: str
) -> str:
    """Return only the four-field user message content."""

    return build_user_message(
        engine=engine,
        db_details=db_details,
        external_knowledge=external_knowledge,
        question=question,
    )["content"]


def fewshot_messages() -> List[Dict[str, str]]:
    """Flatten the configured few-shot episodes into one chat sequence."""

    messages: List[Dict[str, str]] = []
    for example in FEWSHOT_EXAMPLES:
        for message in example.get("messages", []):
            role = message.get("role")
            if role not in ("user", "assistant"):
                raise ValueError(f"few-shot role must be user/assistant, got {role!r}")
            messages.append({"role": role, "content": message["content"]})
    return messages


def build_messages(
    *,
    engine: str,
    db_details: str,
    external_knowledge: str,
    question: str,
    fewshot: bool = True,
) -> List[Dict[str, str]]:
    """Return the complete fixed prefix plus the user message.

    The layout matches ``retoolsql_prompt.prompt_builder.build_messages``:
    system prompt, the two static few-shot episodes, then the four-field user
    message. Pass ``fewshot=False`` only for ablations or contract checks.
    """

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": RETOOLSQL_SYSTEM_PROMPT}
    ]
    if fewshot:
        messages.extend(fewshot_messages())
    messages.append(
        build_user_message(
            engine=engine,
            db_details=db_details,
            external_knowledge=external_knowledge,
            question=question,
        )
    )
    return messages


def text_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """Render the v4 Qwen-style tool call as text."""

    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool {name!r}; expected one of {', '.join(TOOL_NAMES)}")
    parts = [f"<{TOOL_CALL_TAG}>", f"<function={name}>"]
    for key, value in arguments.items():
        if isinstance(value, (list, tuple)):
            rendered = json.dumps(list(value), ensure_ascii=False)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        parts.append(f"<parameter={key}>")
        parts.append(rendered)
        parts.append("</parameter>")
    parts.append("</function>")
    parts.append(f"</{TOOL_CALL_TAG}>")
    return "\n".join(parts)


def describe_tools() -> List[Dict[str, Any]]:
    """Flat tool descriptions derived from the execution schemas."""

    return [
        {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "parameters": schema["function"]["parameters"],
        }
        for schema in TOOL_SCHEMAS
    ]


__all__ = [
    "FINAL_ANSWER_TAG",
    "FEWSHOT_EXAMPLES",
    "FEWSHOT_FILE",
    "PROMPT_DIR",
    "PROMPT_ROOT",
    "PROMPT_VERSION",
    "PROTOCOL_TAGS",
    "RELEVANT_COLUMNS_TAG",
    "RELEVANT_TABLES_TAG",
    "RETOOLSQL_SYSTEM_PROMPT",
    "RETOOLSQL_TOOL_PROMPT",
    "SCRATCH_PAD_TAG",
    "SYSTEM_PROMPT_FILE",
    "TEXT_TOOL_CALL_FORMAT",
    "TOOL_CALL_TAG",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "USER_MESSAGE_TEMPLATE",
    "USER_TEMPLATE_FILE",
    "build_messages",
    "build_user_message",
    "build_user_prompt",
    "describe_tools",
    "fewshot_messages",
    "text_tool_call",
]
