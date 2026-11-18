#!/usr/bin/env python3
"""Assemble the ReToolSQL v4 prompt.

The v4 prompt has four layers:

  Layer 0  fixed prefix: system prompt + two static few-shot episodes
  Layer 1  user message: engine, M-Schema, external knowledge, question
  Layer 2  runtime turns: one tool call or the final answer per assistant turn
  Layer 3  training-only: the privileged SFT teacher appendix

This module builds layers 0-1 and the layer-3 variant from files in
``retoolsql_prompt/v4`` so that SFT, RL, and inference share exactly the same
prefix. It also implements the deterministic schema budget ladder described in
the v4 design: use the M-Schema as-is, then compress ``[Observed]`` fields, then
prune tables by lexical match plus foreign-key closure.

CLI usage::

    python prompt_builder.py --schema-file schema.txt \
        --question "How many schools opened after 2020?" --evidence "" \
        --max-prompt-tokens 16000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

V4_DIR = Path(__file__).resolve().parent / "v4"
# v4 is the only maintained prompt. The old v3 snapshot was removed; failing
# loudly is safer than silently building a prompt from a stale directory.
if not (V4_DIR / "system_prompt.txt").is_file():
    raise FileNotFoundError(
        f"canonical prompt assets are missing: {V4_DIR / 'system_prompt.txt'}"
    )
DEFAULT_PROMPT_DIR = V4_DIR
SYSTEM_PROMPT_PATH = DEFAULT_PROMPT_DIR / "system_prompt.txt"
FEWSHOT_PATH = DEFAULT_PROMPT_DIR / "fewshot.json"
USER_TEMPLATE_PATH = DEFAULT_PROMPT_DIR / "user_template.txt"
TEACHER_PROMPT_PATH = DEFAULT_PROMPT_DIR / "sft_teacher_prompt.txt"

# Calibrated from the v3 design: the whole prompt must fit the 32k training
# window while leaving room for 8 turns of assistant text and observations.
DEFAULT_MAX_PROMPT_TOKENS = 16_000
MIN_SCHEMA_TOKENS = 800
SAFETY_TOKENS = 512
MIN_TABLES_AFTER_PRUNING = 2
MAX_TABLES_AFTER_PRUNING = 8

CHARS_PER_TOKEN = 3.5  # conservative English/SQL estimate when no tokenizer exists


class ToolCallParseError(ValueError):
    """A ``<tool_call>`` body could not be parsed or violates the tool schema."""


class SchemaBudgetError(ValueError):
    """Even the pruned schema does not fit the requested prompt budget."""


# --------------------------------------------------------------------------- #
# File loading
# --------------------------------------------------------------------------- #

def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_user_template(path: Path = USER_TEMPLATE_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_teacher_prompt(path: Path = TEACHER_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def teacher_prompt_sha256(path: Path = TEACHER_PROMPT_PATH) -> str:
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest()


def load_fewshot(path: Path = FEWSHOT_PATH) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("examples", [])


def fewshot_messages(path: Path = FEWSHOT_PATH) -> List[Dict[str, str]]:
    """Flatten the configured few-shot episodes into one chat sequence."""

    messages: List[Dict[str, str]] = []
    for example in load_fewshot(path):
        for message in example["messages"]:
            role = message["role"]
            if role not in ("user", "assistant"):
                raise ValueError(f"few-shot role must be user/assistant, got {role!r}")
            messages.append({"role": role, "content": message["content"]})
    return messages


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #

_encoder = None
_encoder_ready = False


def _get_encoder():
    global _encoder, _encoder_ready
    if _encoder_ready:
        return _encoder
    _encoder_ready = True
    try:  # optional dependency; the design never requires it
        import tiktoken  # type: ignore

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = None
    return _encoder


def estimate_tokens(text: str, *, use_tiktoken: bool = True) -> int:
    """Approximate token count; deterministic and dependency-free by default."""

    if use_tiktoken:
        encoder = _get_encoder()
        if encoder is not None:
            return len(encoder.encode(text))
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


# --------------------------------------------------------------------------- #
# Tool-call parsing (Qwen XML and legacy JSON bodies)
# --------------------------------------------------------------------------- #

_FUNCTION_BLOCK = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</function>", re.S
)
_PARAMETER_BLOCK = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</parameter>", re.S
)


def _coerce_parameter(name: str, key: str, raw: str, expected: Any) -> Any:
    text = raw.strip()
    if expected is int:
        try:
            return int(text)
        except ValueError as exc:
            raise ToolCallParseError(
                f"argument {key!r} of {name} must be an integer, got {text!r}"
            ) from exc
    if expected is str:
        return text
    # tuple of accepted types, currently (list, tuple, str) for sqlite_peek.columns
    if text.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"argument {key!r} of {name} must be a list, got invalid JSON {text!r}"
            ) from exc
        if not isinstance(value, list):
            raise ToolCallParseError(
                f"argument {key!r} of {name} must be a JSON array"
            )
        return value
    return text


def parse_tool_call_body(
    body: str,
    tool_signatures: Mapping[str, Tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    repair_unclosed_function: bool = True,
    repairs: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Parse a ``<tool_call>`` body and return ``(name, arguments)``.

    Both the Qwen XML form and the legacy v1 JSON body are accepted so old prompt
    versions and archived runs remain replayable.

    ``repair_unclosed_function`` enables one narrow, deterministic repair: a body
    with exactly one ``<function=NAME>`` opener, no ``</function>`` closer, all
    parameters closed, and no duplicated ``</tool_call>`` is treated as if the
    closer were present. This shape accounted for every observed qwen3.6-plus
    format failure. Anything ambiguous -- several openers, a duplicated closer,
    an unclosed parameter, unknown names -- still raises, so a real protocol
    violation is never hidden.
    """

    text = body.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"<tool_call> body is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ToolCallParseError("<tool_call> body must be a JSON object")
        if set(payload) != {"name", "arguments"}:
            raise ToolCallParseError(
                "<tool_call> body must have exactly the keys 'name' and 'arguments'"
            )
        name = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(name, str) or name not in tool_signatures:
            raise ToolCallParseError(f"unknown tool {name!r}")
        if not isinstance(arguments, dict):
            raise ToolCallParseError(f"'arguments' of {name} must be a JSON object")
    else:
        if not text.startswith("<function="):
            raise ToolCallParseError(
                "tool call body must be Qwen XML <function=...> tags "
                "or a legacy JSON object"
            )
        functions = _FUNCTION_BLOCK.findall(text)
        if len(functions) == 1 and repairs is not None:
            match = _FUNCTION_BLOCK.search(text)
            if match is not None:
                outside = (text[: match.start()] + text[match.end():]).strip()
                if outside:
                    repairs.append("trailing_text_ignored")
        if len(functions) != 1:
            if repair_unclosed_function:
                repaired = _repair_single_unclosed_function(text, tool_signatures)
                if repaired is not None:
                    name, inner = repaired
                    if repairs is not None:
                        repairs.append("inserted </function>")
                    # fall through to argument parsing below
                    return _arguments_from_inner(
                        name, inner, tool_signatures, repairs
                    )
            raise ToolCallParseError(
                "<tool_call> must contain exactly one <function=...>...</function> block"
            )
        name, inner = functions[0]
        if name not in tool_signatures:
            raise ToolCallParseError(f"unknown tool {name!r}")
        return _arguments_from_inner(name, inner, tool_signatures, repairs)

    required, optional = tool_signatures[name]
    missing = sorted(set(required) - set(arguments))
    if missing:
        raise ToolCallParseError(f"missing required argument(s) {missing} for {name}")
    for key, value in arguments.items():
        expected = required.get(key, optional.get(key))
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ToolCallParseError(f"argument {key!r} of {name} must be an integer")
        if expected is str and not isinstance(value, str):
            raise ToolCallParseError(f"argument {key!r} of {name} must be a string")
        if isinstance(expected, tuple) and not isinstance(value, (list, tuple, str)):
            raise ToolCallParseError(
                f"argument {key!r} of {name} must be a list or a string"
            )
    return name, arguments


_FUNCTION_OPEN = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*>")


def _repair_single_unclosed_function(
    text: str,
    tool_signatures: Mapping[str, Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Optional[Tuple[str, str]]:
    """Return ``(name, inner)`` when the only defect is one missing closer."""

    if text.count("</function>") != 0:
        return None
    if text.count("</tool_call>") > 0:
        # a duplicated closer is a different, ambiguous defect
        return None
    openers = _FUNCTION_OPEN.findall(text)
    if len(openers) != 1:
        return None
    name = openers[0]
    if name not in tool_signatures:
        return None
    match = _FUNCTION_OPEN.search(text)
    assert match is not None
    inner = text[match.end():].strip()
    parameter_opens = len(re.findall(r"<parameter=[A-Za-z_][A-Za-z0-9_]*\s*>", inner))
    parameter_closes = inner.count("</parameter>")
    if parameter_opens != parameter_closes:
        return None
    if not inner:
        return None
    return name, inner


def _arguments_from_inner(
    name: str,
    inner: str,
    tool_signatures: Mapping[str, Tuple[Mapping[str, Any], Mapping[str, Any]]],
    repairs: Optional[List[str]],
) -> Tuple[str, Dict[str, Any]]:
    required, optional = tool_signatures[name]
    arguments: Dict[str, Any] = {}
    for key, raw in _PARAMETER_BLOCK.findall(inner):
        if key in arguments:
            raise ToolCallParseError(f"duplicate parameter {key!r} for {name}")
        if key not in required and key not in optional:
            raise ToolCallParseError(f"unknown argument {key!r} for {name}")
        arguments[key] = _coerce_parameter(
            name, key, raw, required.get(key, optional.get(key))
        )
    leftover = _PARAMETER_BLOCK.sub("", inner).strip()
    if leftover:
        raise ToolCallParseError(
            f"unexpected text inside <function={name}>: {leftover!r}"
        )
    missing = sorted(set(required) - set(arguments))
    if missing:
        raise ToolCallParseError(f"missing required argument(s) {missing} for {name}")
    return name, arguments


# --------------------------------------------------------------------------- #
# M-Schema budget ladder
# --------------------------------------------------------------------------- #

# Storage-class counts, non-null counts, and averages are low-signal for SQL
# authoring. Min/Max are kept: they are the only scale evidence for
# high-cardinality numeric columns (percent vs fraction, HHMM, year ranges).
_OBSERVED_DROP_KEYS = ("Storage", "NonNull", "Avg", "Unit")

# Presentation bounds for the compact tier. They are calibrated from all 326
# checked-in M-Schemas: normal literals have p99=64 and p99.9=257 (max 1333), so
# 256 keeps essentially every useful literal while still capping a stored
# image (``0x…`` hex text), XML document, or base64 payload. 1400 covers the
# longest observed line after that cap (1256: a five-value Domain list) and the
# longest description line (1066), so the compact tier no longer cuts ordinary
# raw M-Schema lines.
MAX_LITERAL_CHARS = 256
MAX_LINE_CHARS = 1400
MAX_EXAMPLES = 3
MAX_DOMAIN = 12
MAX_TOP = 8

_TABLE_HEADER = re.compile(r"^# Table:\s*(.*?)\s*\|")
_COLUMN_LINE = re.compile(r"^\s{2}(.+?):.*normalized=")
_FK_LINE = re.compile(r"^(.+?)\.([^.\s]+)\s*=\s*(.+?)\.([^.\s]+)\s*$")

_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'")
_NAMED_LIST = re.compile(r"^(Examples|Domain|Top)=\[(.*)\]$")

_STOPWORDS = frozenset(
    """
    a an and are as at be been by can could describe did do does for from had has
    have how i if in into is it its list many me month most name names not number
    of on or over per please show that the their there these this those to total
    was were what when where which who whom whose why with year years
    """.split()
)


def compress_schema(schema_text: str) -> str:
    """Compact the M-Schema for the prompt, preserving its layout.

    Two reductions, applied per physical line so the M-Schema grammar the system
    prompt teaches (a header block, one ``# Table:`` line, a ``[`` opener, one
    two-space-indented line per column) survives unchanged:

    * ``[Observed]`` keeps the decision-bearing fields only (role, nulls,
      distinct, domain/top, examples, patterns); storage-class counts, non-null
      counts, averages, and units are dropped.
    * Very long literals and over-long lines are truncated. A single base64 or
      XML payload can otherwise consume more context than the whole schema.

    Multi-line literals are joined only *within* their own continuation lines:
    a line that starts a new schema element is never merged with the previous
    one, which is what keeps the format readable and parseable.
    """

    out: List[str] = []
    for line in _join_continuations(schema_text):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("[Observed]"):
            body = stripped[len("[Observed]") :].strip()
            kept: List[str] = []
            for segment in body.split(";"):
                segment = segment.strip()
                if not segment:
                    continue
                key = segment.split("=", 1)[0].strip()
                if key in _OBSERVED_DROP_KEYS:
                    continue
                kept.append(_cap_segment(segment))
            line = f"{indent}[Observed] " + "; ".join(kept)
        elif stripped.startswith("[Description]"):
            line = _cap_line(line)

        if len(line) > MAX_LINE_CHARS:
            line = _cap_line(line)
        out.append(line)
    return "\n".join(out)


def _join_continuations(schema_text: str) -> List[str]:
    """Join physical lines that continue a literal, leaving structure intact."""

    lines: List[str] = []
    carry = ""
    for raw in schema_text.splitlines():
        stripped = raw.strip()
        starts_element = bool(_STRUCTURAL_START.match(raw))
        if carry:
            merged = f"{carry} {stripped}".strip()
            if starts_element:
                # The previous value ended without an explicit marker; keep the
                # accumulated text as its own line and start a fresh element.
                lines.append(carry)
                carry = ""
            else:
                carry = merged
                continue
        if stripped and _needs_continuation(stripped):
            carry = stripped
            continue
        lines.append(raw)
    if carry:
        lines.append(carry)
    return lines


_STRUCTURAL_START = re.compile(
    r"^(?:【|# Table:|\[|\s*\[(?:Table Summary|Description|Observed|Quality|"
    r"Foreign Keys|Relation)\]|\s{2}\S.*:.*normalized=|"
    r"\S.*\.\S+\s*=\s*\S+\.\S+\s*$)"
)


def _needs_continuation(line: str) -> bool:
    """True when a line has an unterminated quoted literal."""

    quotes = 0
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'":
            quotes += 1
    return quotes % 2 == 1


def _logical_lines(schema_text: str) -> List[str]:
    """Compatibility wrapper: join continuations without reflowing structure."""

    return _join_continuations(schema_text)


def _cap_segment(segment: str) -> str:
    """Cap a ``Key=[...]`` segment; ``Key=value`` segments pass through."""

    match = _NAMED_LIST.match(segment)
    if not match:
        return _cap_literals(segment)
    key, body = match.group(1), match.group(2)
    items = _split_list_items(body)
    if key == "Examples":
        items = items[:MAX_EXAMPLES]
    elif key == "Domain":
        items = items[:MAX_DOMAIN]
    elif key == "Top":
        items = items[:MAX_TOP]
    items = [_cap_literals(item) for item in items]
    return f"{key}=[" + ", ".join(items) + "]"


def _split_list_items(body: str) -> List[str]:
    """Split a bracketed value list on top-level commas (quotes respected)."""

    items: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escaped = False
    for char in body:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == ",":
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return [item for item in items if item]


def _cap_literals(text: str) -> str:
    def shorten(match: "re.Match[str]") -> str:
        value = match.group(0)
        if len(value) <= MAX_LITERAL_CHARS + 2:
            return value
        return value[: MAX_LITERAL_CHARS - 1] + "…'"

    return _QUOTED.sub(shorten, text)


def _cap_line(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return _cap_literals(line)
    # Reserve the two characters of the ellipsis marker so the result never
    # exceeds MAX_LINE_CHARS (the old caller appended it after a full-width cap).
    return _cap_literals(line[: MAX_LINE_CHARS - 2]).rstrip() + " …"


def _word_set(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _split_schema(schema_text: str) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    """Split a rendered M-Schema into preamble, table blocks, and FK lines."""

    preamble: List[str] = []
    tables: List[Tuple[str, List[str]]] = []
    fk_lines: List[str] = []
    current_name: Optional[str] = None
    current_lines: List[str] = []
    in_tables = False

    for line in schema_text.splitlines():
        if line.startswith("# Table:"):
            if current_name is not None:
                tables.append((current_name, current_lines))
            match = _TABLE_HEADER.match(line)
            current_name = match.group(1).strip() if match else line[8:].strip()
            current_lines = [line]
            in_tables = True
            continue
        if line.strip() == "[Foreign Keys]":
            if current_name is not None:
                tables.append((current_name, current_lines))
                current_name, current_lines = None, []
            in_tables = False
            fk_lines.append(line)
            continue
        if in_tables and current_name is not None:
            current_lines.append(line)
        elif fk_lines:
            fk_lines.append(line)
        else:
            preamble.append(line)

    if current_name is not None:
        tables.append((current_name, current_lines))
    return preamble, tables, fk_lines


def _table_score(block: List[str], query_words: set) -> int:
    if not query_words:
        return 0
    name = _TABLE_HEADER.match(block[0]).group(1) if _TABLE_HEADER.match(block[0]) else ""
    name_words = _word_set(name.replace("_", " "))
    column_words: set = set()
    description_words: set = set()
    for line in block[1:]:
        column = _COLUMN_LINE.match(line)
        if column:
            column_words |= _word_set(column.group(1).replace("_", " "))
        if "[Description]" in line:
            description_words |= _word_set(line)
        if "[Observed]" in line:
            description_words |= _word_set(line)
    return (
        3 * len(query_words & name_words)
        + 2 * len(query_words & column_words)
        + len(query_words & description_words)
    )


def prune_schema(
    schema_text: str,
    question: str,
    external_knowledge: str = "",
    *,
    min_tables: int = MIN_TABLES_AFTER_PRUNING,
    max_tables: int = MAX_TABLES_AFTER_PRUNING,
) -> str:
    """Keep lexically relevant tables plus the foreign-key parents they join to.

    Expansion is deliberately shallow: a seed table brings along the parent
    tables its declared foreign keys point at, because those are the tables a
    join needs, but neighbors of neighbors are not pulled in. The total is capped
    at ``max_tables`` so a densely connected schema cannot defeat the budget.

    The output stays a valid M-Schema excerpt: the original preamble is kept,
    a ``# Schema Filter`` line states which tables were omitted, and only FK
    groups whose two endpoints survived are retained.
    """

    preamble, tables, fk_lines = _split_schema(schema_text)
    if len(tables) <= min_tables:
        return schema_text

    query_words = _word_set(f"{question} {external_knowledge}")
    scored = [(_table_score(block, query_words), name, block) for name, block in tables]
    ranked = sorted(
        range(len(scored)), key=lambda i: (-scored[i][0], scored[i][1].lower())
    )

    selected: Dict[str, int] = {}
    for index in ranked:
        if scored[index][0] <= 0 or len(selected) >= max_tables:
            break
        selected[scored[index][1]] = index
    for index in ranked:
        if len(selected) >= min_tables:
            break
        selected.setdefault(scored[index][1], index)

    # Parse FK groups as (child, parent, lines) before deciding what to keep.
    fk_groups: List[Tuple[str, str, List[str]]] = []
    current: Optional[Tuple[str, str, List[str]]] = None
    for line in fk_lines:
        if line.strip() == "[Foreign Keys]":
            continue
        match = _FK_LINE.match(line)
        if match:
            current = (match.group(1).strip(), match.group(3).strip(), [line])
            fk_groups.append(current)
        elif current is not None:
            current[2].append(line)

    # One shallow hop: add the parent table of each seed's declared FK.
    index_by_name = {name: index for index, (_, name, _) in enumerate(scored)}
    for child, parent, _lines in fk_groups:
        if len(selected) >= max_tables:
            break
        if child in selected and parent in index_by_name and parent not in selected:
            selected[parent] = index_by_name[parent]

    selected_names = {name for _, name, _ in scored if name in selected}
    kept_blocks = [block for _, name, block in scored if name in selected]
    omitted = [name for _, name, _ in scored if name not in selected_names]

    out: List[str] = list(preamble)
    if out and out[-1].strip():
        out.append("")
    out.append(
        f"# Schema Filter: showing {len(kept_blocks)} of {len(tables)} tables "
        f"(question-matched tables plus the foreign-key parents they join to)."
    )
    if omitted:
        out.append("# Omitted tables: " + ", ".join(omitted))
    out.append("")
    for block in kept_blocks:
        out.extend(block)
        out.append("")

    kept_groups = [
        lines for child, parent, lines in fk_groups
        if child in selected_names and parent in selected_names
    ]
    if kept_groups:
        out.append("[Foreign Keys]")
        for lines in kept_groups:
            out.extend(lines)
    return "\n".join(out).strip()


def fit_schema(
    schema_text: str,
    *,
    question: str,
    external_knowledge: str,
    budget_tokens: int,
    count: Callable[[str], int] = estimate_tokens,
) -> Tuple[str, int, int]:
    """Return ``(schema, tier, tokens)`` using the cheapest tier that fits.

    Tier 0: the M-Schema unchanged.
    Tier 1: ``[Observed]`` compressed.
    Tier 2: additionally pruned to relevant tables plus FK closure.
    Tier 3: pruning was not enough; the caller decides whether to filter or warn.
    """

    original_tokens = count(schema_text)
    if original_tokens <= budget_tokens:
        return schema_text, 0, original_tokens

    compressed = compress_schema(schema_text)
    compressed_tokens = count(compressed)
    if compressed_tokens <= budget_tokens:
        return compressed, 1, compressed_tokens

    # Tighten the table cap until the excerpt fits. The lower the cap, the less
    # context the model gets, so this always tries the most generous cap first.
    pruned = compressed
    pruned_tokens = compressed_tokens
    for cap in (MAX_TABLES_AFTER_PRUNING, 6, 4, 3, 2):
        candidate = prune_schema(
            compressed, question, external_knowledge, max_tables=cap
        )
        candidate_tokens = count(candidate)
        if candidate_tokens < pruned_tokens:
            pruned, pruned_tokens = candidate, candidate_tokens
        if candidate_tokens <= budget_tokens:
            return candidate, 2, candidate_tokens

    tier = 2 if pruned_tokens <= budget_tokens else 3
    return pruned, tier, pruned_tokens


# --------------------------------------------------------------------------- #
# Message assembly
# --------------------------------------------------------------------------- #

def render_user_message(
    *,
    engine: str,
    db_details: str,
    external_knowledge: str,
    question: str,
    template: Optional[str] = None,
) -> str:
    template = template if template is not None else load_user_template()
    return template.format(
        engine=engine,
        db_details=db_details,
        external_knowledge=external_knowledge,
        question=question,
    )


def build_messages(
    *,
    engine: str,
    db_details: str,
    external_knowledge: str,
    question: str,
    fewshot: bool = True,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    count: Optional[Callable[[str], int]] = None,
    system_prompt: Optional[str] = None,
    allow_over_budget: bool = False,
) -> Dict[str, Any]:
    """Assemble the v4 prompt and return messages plus budget metadata.

    Raises ``SchemaBudgetError`` when the pruned schema still exceeds the
    budget and ``allow_over_budget`` is False. The training pipeline should
    skip such questions rather than truncate silently, matching the paper's
    "filter, do not truncate" rule.
    """

    counter = count or estimate_tokens
    system = system_prompt if system_prompt is not None else load_system_prompt()
    shots = fewshot_messages() if fewshot else []
    template = load_user_template()

    prefix_tokens = counter(system) + sum(counter(m["content"]) for m in shots)
    skeleton = render_user_message(
        engine="", db_details="", external_knowledge="", question="", template=template
    )
    overhead = prefix_tokens + counter(skeleton) + SAFETY_TOKENS
    remaining = max_prompt_tokens - overhead
    if remaining < MIN_SCHEMA_TOKENS:
        raise SchemaBudgetError(
            f"fixed prompt prefix leaves only {remaining} tokens for the schema; "
            f"need at least {MIN_SCHEMA_TOKENS}. Lower --fewshot or raise the budget."
        )

    schema, tier, schema_tokens = fit_schema(
        db_details,
        question=question,
        external_knowledge=external_knowledge,
        budget_tokens=remaining,
        count=counter,
    )

    user_content = render_user_message(
        engine=engine,
        db_details=schema,
        external_knowledge=external_knowledge,
        question=question,
        template=template,
    )
    messages = [
        {"role": "system", "content": system},
        *shots,
        {"role": "user", "content": user_content},
    ]
    total_tokens = prefix_tokens + counter(user_content)
    if total_tokens > max_prompt_tokens and not allow_over_budget:
        raise SchemaBudgetError(
            f"prompt is {total_tokens} tokens after schema tier {tier}; "
            f"budget is {max_prompt_tokens}"
        )

    return {
        "messages": messages,
        "meta": {
            "prompt_version": DEFAULT_PROMPT_DIR.name,
            "prompt_sha256": system_prompt_sha256(system, load_fewshot(), template),
            "fewshot": fewshot,
            "system_tokens": counter(system),
            "fewshot_tokens": sum(counter(m["content"]) for m in shots),
            "schema_tokens": schema_tokens,
            "schema_tier": tier,
            "prompt_tokens": total_tokens,
            "max_prompt_tokens": max_prompt_tokens,
            "schema_chars": len(schema),
            "original_schema_chars": len(db_details),
        },
    }


def system_prompt_sha256(
    system: str,
    fewshot_examples: Sequence[Any],
    user_template: str,
) -> str:
    """Stable hash of the student-visible prompt assets, not the per-question data.

    The payload must match ``common.prompts.PROMPT_HASH`` so a trace's recorded
    hash can be compared against the assets currently on disk.
    """

    payload = {
        "version": DEFAULT_PROMPT_DIR.name,
        "system": system,
        "fewshot": list(fewshot_examples),
        "user_template": user_template,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_teacher_messages(
    *,
    engine: str,
    db_details: str,
    external_knowledge: str,
    question: str,
    reference_sql: str,
    teacher_file: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build the SFT privileged-teacher variant of the same prompt."""

    built = build_messages(
        engine=engine,
        db_details=db_details,
        external_knowledge=external_knowledge,
        question=question,
        **kwargs,
    )
    template = (
        Path(teacher_file).read_text(encoding="utf-8").strip()
        if teacher_file
        else load_teacher_prompt()
    )
    appendix = template.format(reference_sql=reference_sql.strip())
    built["messages"][-1]["content"] += "\n\n" + appendix
    built["meta"]["teacher"] = True
    built["meta"]["teacher_file"] = str(teacher_file or TEACHER_PROMPT_PATH)
    built["meta"]["teacher_prompt_sha256"] = hashlib.sha256(
        template.encode("utf-8")
    ).hexdigest()
    built["meta"]["prompt_tokens"] += estimate_tokens(appendix)
    return built


def prepare_schema(
    schema_text: str,
    *,
    question: str,
    external_knowledge: str = "",
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    fewshot: bool = True,
    count: Optional[Callable[[str], int]] = None,
) -> Tuple[str, int, int, int, List[str]]:
    """Fit a schema using the same overhead accounting as :func:`build_messages`.

    Returns ``(schema, tier, schema_tokens, prompt_tokens, omitted_tables)``.
    Use this instead of calling :func:`fit_schema` with a hand-picked reserve: a
    fixed reserve either over-compresses schemas that would have fit, or
    under-reserves and overflows. ``omitted_tables`` is empty unless pruning ran,
    and lets the anchor audit check whether the model stayed inside the shown
    schema.
    """

    counter = count or estimate_tokens
    system = load_system_prompt()
    shots = fewshot_messages() if fewshot else []
    skeleton = render_user_message(
        engine="", db_details="", external_knowledge="", question=""
    )
    overhead = (
        counter(system)
        + sum(counter(m["content"]) for m in shots)
        + counter(skeleton)
        + SAFETY_TOKENS
    )
    remaining = max_prompt_tokens - overhead
    if remaining < MIN_SCHEMA_TOKENS:
        raise SchemaBudgetError(
            f"fixed prompt prefix leaves only {remaining} tokens for the schema; "
            f"need at least {MIN_SCHEMA_TOKENS}"
        )
    schema, tier, schema_tokens = fit_schema(
        schema_text,
        question=question,
        external_knowledge=external_knowledge,
        budget_tokens=remaining,
        count=counter,
    )
    user_with_schema = render_user_message(
        engine="SQLite",
        db_details=schema,
        external_knowledge=external_knowledge,
        question=question,
    )
    prompt_tokens = (
        overhead - SAFETY_TOKENS + counter(user_with_schema) - counter(skeleton)
    )
    omitted: List[str] = []
    if tier >= 2:
        match = re.search(r"^# Omitted tables:\s*(.+)$", schema, re.M)
        if match:
            omitted = [name.strip() for name in match.group(1).split(",") if name.strip()]
    return schema, tier, schema_tokens, prompt_tokens, omitted


def iter_message_text(messages: Iterable[Mapping[str, str]]) -> Iterable[Tuple[str, str]]:
    for message in messages:
        yield message["role"], message["content"]


def render_messages(messages: Sequence[Mapping[str, str]]) -> str:
    blocks = []
    for role, content in iter_message_text(messages):
        blocks.append(f"<<<{role}>>>\n{content}")
    return "\n\n".join(blocks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--engine", default="SQLite")
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--no-fewshot", action="store_true")
    parser.add_argument("--teacher", default=None, help="reference SQL for the SFT teacher variant")
    parser.add_argument("--json", action="store_true", help="print the full message list as JSON")
    parser.add_argument("--meta", action="store_true", help="print only budget metadata")
    args = parser.parse_args(argv)

    schema = Path(args.schema_file).read_text(encoding="utf-8")
    builder = build_teacher_messages if args.teacher else build_messages
    extra = {"reference_sql": args.teacher} if args.teacher else {}
    built = builder(
        engine=args.engine,
        db_details=schema,
        external_knowledge=args.evidence,
        question=args.question,
        fewshot=not args.no_fewshot,
        max_prompt_tokens=args.max_prompt_tokens,
        **extra,
    )

    if args.json:
        print(json.dumps(built, ensure_ascii=False, indent=2))
    elif args.meta:
        print(json.dumps(built["meta"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(built["meta"], ensure_ascii=False, indent=2))
        print()
        print(render_messages(built["messages"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
