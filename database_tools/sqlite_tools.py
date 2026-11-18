#!/usr/bin/env python3
"""ReToolSQL verification-first SQLite tool suite.

This module implements the three deterministic, read-only database tools of
the ReToolSQL action space (arXiv:2608.27796v1, Section 3.3 "Action Space: A
Verification-First Tool Suite" and Appendix C "Tool Definitions"):

    sqlite_query(db_file, sql, max_return_rows=200)
        Sandboxed executor restricted to SELECT/WITH statements with a busy
        timeout, a VM-step limit, and a returned-row cap.  Reports the result
        columns and rows or a structured execution error.

    sqlite_peek(db_file, table, columns=None, limit=5, where=None)
        Column profiler returning declared type, null/distinct counts,
        value ranges, character-class summaries, top values with frequencies,
        representative samples, and value prefixes.

    bm25_search_sqlite(db_file, table, column, query, top_k=10, where=None)
        Hybrid lexical search over stored cell values that combines BM25
        ranking with a case-insensitive regex/substring fallback.

The training environment binds exactly one SQLite file per question, so the
model-visible arguments never contain a database identifier.  ``db_id``
remains available as an optional keyword argument purely for environment-side
consistency checks and logging.

Each tool returns a Python payload: a ``dict`` for ``sqlite_query`` and
``sqlite_peek``, a ``list`` for ``bm25_search_sqlite``, and a shared error
payload (``{"error": {"kind": ..., "message": ...}}``) on failure.  Use
``dispatch`` to turn a tool name plus JSON arguments into the compact JSON
text that a training environment places in the tool observation message.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:  # sqlglot is the preferred static validator; the fallback is conservative.
    import sqlglot
    from sqlglot import exp

    _HAS_SQLGLOT = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    _HAS_SQLGLOT = False


# --------------------------------------------------------------------------- #
# Execution limits
#
# The paper (Appendix C, Table 7) fixes the VM-step budget and the 5 s busy
# timeout; the row, peek-scan, and value-list defaults are calibrated here from
# the databases this project trains on. See TOOL_DEFAULTS_CALIBRATION.md.
# --------------------------------------------------------------------------- #

BUSY_TIMEOUT_SECONDS = 5.0
VM_STEP_LIMIT = 15_000_000
VM_STEP_INTERVAL = 1_000

# 2,928 tables over 395 databases: 38.9% hold <= 100 rows and 44.0% <= 200;
# 90.5% of gold answers have <= 200 rows. Replaying 5,821 recorded agent SQL
# statements put every mid-size result in the 101-200 band (5 in 201-300, none
# in 301-500; the rest exceed 500), so 200 is the elbow: row truncation falls
# 8.45% -> 7.44% while the 4,000-character budget cuts 4.29% -> 6.18%.
DEFAULT_MAX_RETURN_ROWS = 200

# 20,000 rows is the p95 table size (94.9% of tables fit entirely) and halves
# the share of peek calls that profile a truncated sample (18.7% -> 9.0% by
# distinct call). A three-column peek stays under ~0.4 s; profiling all columns
# of a 40-column table takes ~1.8 s.
PEEK_SCAN_LIMIT = 20_000

# 96% of the recorded peek calls pass limit=5 explicitly (the few-shot example
# uses 5), and the 4,000-character budget trims value lists to 5/3/1 anyway.
DEFAULT_PEEK_LIMIT = 5

# 84% of BM25 calls return one entry and 98.4% of the literals the agent later
# uses sit at rank 0. Ten keeps the two longest useful lists (a rank-17 regex
# hit was needed by an EX-true episode) without inflating the common case.
DEFAULT_TOP_K = 10

# Implementation supplements (documented in README.md).
DEFAULT_WALL_TIMEOUT = float(os.environ.get("DATABASE_TOOLS_SQL_TIMEOUT", "30.0"))
DEFAULT_MAX_OUTPUT_CHARS = int(os.environ.get("DATABASE_TOOLS_MAX_OUTPUT_CHARS", "4000"))
MAX_RETURN_ROWS_CAP = 10_000
MAX_PEEK_LIMIT = 500
MAX_PEEK_COLUMNS = 100
MAX_TOP_K = 100
BM25_CANDIDATE_LIMIT = 50_000
BM25_K1 = 1.2
BM25_B = 0.75
CELL_MAX_CHARS = 200
PREFIX_LENGTH = 2
MAX_SQL_CHARS = 100_000
BLOB_PREVIEW_BYTES = 32

ERROR_KINDS = (
    "validation_error",
    "execution_error",
    "timeout",
    "step_limit_exceeded",
    "infrastructure_error",
)


# --------------------------------------------------------------------------- #
# Errors and the shared error payload
# --------------------------------------------------------------------------- #


class ToolError(RuntimeError):
    """Base class for every structured tool failure."""

    kind = "execution_error"


class ToolValidationError(ToolError, ValueError):
    """The request itself is malformed or violates the read-only contract."""

    kind = "validation_error"


class ToolExecutionError(ToolError):
    """SQLite refused or could not complete the statement."""

    kind = "execution_error"


class ToolTimeoutError(ToolExecutionError):
    """The wall-clock safety net stopped the statement."""

    kind = "timeout"


class ToolStepLimitError(ToolExecutionError):
    """The SQLite VM-step budget stopped the statement."""

    kind = "step_limit_exceeded"


class ToolInfrastructureError(ToolError):
    """The database file or execution harness could not be used."""

    kind = "infrastructure_error"


MAX_ERROR_MESSAGE_CHARS = 1_000

_CONTROL_TOKEN_LITERALS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|tts_pad|>",
    "<tool_response>",
    "</tool_response>",
)


def _neutralize_control_tokens(text: str) -> str:
    """Stop database content from injecting real chat-template control tokens.

    A cell value can literally contain ``<|im_end|>`` or ``<|im_start|>``. The
    rollout tokenizer treats those as special tokens, which would terminate the
    assistant turn or start a new one in the middle of an observation. Replacing
    the angle brackets keeps the value inspectable without making it syntax.
    """

    if not text:
        return text
    for literal in _CONTROL_TOKEN_LITERALS:
        if literal in text:
            text = text.replace(literal, literal.replace("<", "‹").replace(">", "›"))
    return text


def fit_observation(payload: Any, budget: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Serialize a tool payload and guarantee the result fits ``budget`` chars.

    The per-tool fitters only trim ``rows``/profile lists. Error messages and
    column names bypassed them entirely: a 50,000-character SQLite error or a
    90,000-character column alias produced a 25,000-token observation, blowing
    through both the response and context budgets. This is the final gate that
    every tool result passes through, including errors.
    """

    text = _neutralize_control_tokens(json_text(payload))
    if budget is None or len(text) <= budget:
        return text

    marker = {
        "output_truncated": True,
        "original_chars": len(text),
        "note": "observation exceeded the tool output budget",
    }
    skeleton = _neutralize_control_tokens(json_text(marker))
    if len(skeleton) >= budget:
        return skeleton

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _neutralize_control_tokens(json_text({**marker, "preview": text[:middle]}))
        if len(candidate) <= budget:
            low = middle
        else:
            high = middle - 1
    return _neutralize_control_tokens(json_text({**marker, "preview": text[:low]}))


def error_payload(kind: str, message: str) -> Dict[str, Any]:
    """Build the shared structured error returned by every tool.

    Long SQLite messages are truncated here so a verbose or attacker-controlled
    error cannot dominate the observation.
    """

    if kind not in ERROR_KINDS:
        kind = "execution_error"
    text = str(message)
    if len(text) > MAX_ERROR_MESSAGE_CHARS:
        text = text[:MAX_ERROR_MESSAGE_CHARS] + f"…[truncated {len(text) - MAX_ERROR_MESSAGE_CHARS} chars]"
    return {"error": {"kind": kind, "message": text}}


def _run_tool(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except ToolError as exc:
        return error_payload(exc.kind, str(exc))


# --------------------------------------------------------------------------- #
# Read-only sandbox
# --------------------------------------------------------------------------- #


def _constant(name: str) -> Optional[int]:
    value = getattr(sqlite3, name, None)
    return value if isinstance(value, int) else None


_DENIED_ACTIONS = set()
for _name in (
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_VTABLE",
    "SQLITE_DELETE",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_TEMP_TRIGGER",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_VTABLE",
    "SQLITE_INSERT",
    "SQLITE_LOAD_EXTENSION",
    "SQLITE_REINDEX",
    "SQLITE_UPDATE",
    "SQLITE_VACUUM",
    "SQLITE_ALTER_TABLE",
    "SQLITE_ATTACH",
    "SQLITE_DETACH",
):
    _action = _constant(_name)
    if _action is not None:
        _DENIED_ACTIONS.add(_action)

# Metadata pragmas are required to inspect a schema; every other pragma is
# denied, including journal_mode and writable_schema changes.
_SAFE_METADATA_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "table_list",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
    "database_list",
}
_DANGEROUS_FUNCTIONS = {
    "eval",
    "fts3_tokenizer",
    "load_extension",
    "readfile",
    "writefile",
}


def readonly_authorizer(
    action: int,
    first: Optional[str],
    second: Optional[str],
    database: Optional[str],
    source: Optional[str],
) -> int:
    """SQLite authorizer callback enforcing the read-only policy."""

    del database, source
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    pragma = _constant("SQLITE_PRAGMA")
    if pragma is not None and action == pragma:
        name = (first or "").strip().lower()
        return sqlite3.SQLITE_OK if name in _SAFE_METADATA_PRAGMAS else sqlite3.SQLITE_DENY
    function = _constant("SQLITE_FUNCTION")
    if function is not None and action == function:
        name = (second or first or "").strip().lower()
        if name in _DANGEROUS_FUNCTIONS:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _resolve_db_file(db_file: Union[str, Path]) -> Path:
    path = Path(db_file).expanduser()
    if not path.is_file():
        raise ToolInfrastructureError(f"SQLite database not found: {path}")
    return path.resolve()


def _open_readonly(db_file: Union[str, Path]) -> Tuple[sqlite3.Connection, Path]:
    path = _resolve_db_file(db_file)
    uri = f"{path.as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_SECONDS)
    except (OSError, sqlite3.Error) as exc:
        raise ToolInfrastructureError(f"could not open SQLite database {path}: {exc}") from exc
    try:
        # This runs before the authorizer because it is itself a pragma while
        # subsequent user pragmas are subject to the read-only policy.
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(readonly_authorizer)
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        connection.close()
        raise ToolInfrastructureError(f"could not configure SQLite database {path}: {exc}") from exc
    return connection, path


def _check_db_identity(db_file: Union[str, Path], db_id: Optional[str]) -> None:
    """Optional environment-side guard: db_id must name the bound database."""

    if db_id is None:
        return
    if not isinstance(db_id, str) or not db_id:
        raise ToolValidationError("db_id must be a non-empty string when provided")
    path = Path(db_file)
    if path.stem != db_id and path.parent.name != db_id:
        raise ToolInfrastructureError(
            f"db_id {db_id!r} does not match the bound database {path}"
        )


# --------------------------------------------------------------------------- #
# Static statement validation
# --------------------------------------------------------------------------- #

_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "AlterTable",
    "Command",
    "Pragma",
    "Transaction",
    "Commit",
    "Rollback",
    "Attach",
    "Detach",
    "Set",
    "Use",
    "Grant",
    "Into",
    "Copy",
    "TruncateTable",
    "Fetch",
    "Describe",
    "Explain",
    "Cache",
    "Uncache",
    "LoadData",
    "Values",
    "Show",
    "Kill",
    "Analyze",
    "Vacuum",
)

if _HAS_SQLGLOT:  # pragma: no branch - import-time selection
    _FORBIDDEN_NODES = tuple(
        node
        for node in (getattr(exp, name, None) for name in _FORBIDDEN_NODE_NAMES)
        if node is not None
    )
    _SEMICOLON_NODE = getattr(exp, "Semicolon", None)
else:  # pragma: no cover - exercised only in minimal environments
    _FORBIDDEN_NODES = ()
    _SEMICOLON_NODE = None

_CONSERVATIVE_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|"
    r"REINDEX|TRIGGER|LOAD_EXTENSION|READFILE|WRITEFILE)\b",
    re.IGNORECASE,
)


def _validate_with_sqlglot(sql: str) -> None:
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:  # sqlglot.errors.ParseError and subclasses
        raise ToolValidationError(f"could not parse SQL: {exc}") from exc
    statements = [
        expression
        for expression in parsed
        if expression is not None
        and not (_SEMICOLON_NODE is not None and isinstance(expression, _SEMICOLON_NODE))
    ]
    if len(statements) != 1:
        raise ToolValidationError(
            f"sql must contain exactly one statement; found {len(statements)}"
        )
    expression = statements[0]
    if not isinstance(expression, exp.Query):
        raise ToolValidationError(
            "only read-only SELECT or WITH statements are allowed; "
            f"found {type(expression).__name__}"
        )
    for node in expression.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise ToolValidationError(
                "only read-only SELECT or WITH statements are allowed; "
                f"found a {type(node).__name__} construct"
            )


def _validate_conservative(sql: str) -> None:
    body = sql.strip()
    while body.endswith(";"):
        body = body[:-1].strip()
    if ";" in body:
        raise ToolValidationError("sql must contain exactly one statement")
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", body)
    keyword = match.group(1).upper() if match else ""
    if keyword not in {"SELECT", "WITH"}:
        raise ToolValidationError(
            "only read-only SELECT or WITH statements are allowed"
        )
    if _CONSERVATIVE_FORBIDDEN.search(body):
        raise ToolValidationError(
            "sql contains a statement or function that is not allowed in the "
            "read-only sandbox"
        )


def validate_readonly_sql(sql: Any) -> None:
    """Reject anything that is not a single read-only SELECT/WITH statement."""

    if not isinstance(sql, str) or not sql.strip():
        raise ToolValidationError("sql must be a non-empty string")
    if len(sql) > MAX_SQL_CHARS:
        raise ToolValidationError(
            f"sql exceeds the {MAX_SQL_CHARS}-character limit"
        )
    if _HAS_SQLGLOT:
        _validate_with_sqlglot(sql)
    else:  # pragma: no cover - exercised only in minimal environments
        _validate_conservative(sql)


# --------------------------------------------------------------------------- #
# Execution with the SQLite VM-step budget and the wall-clock safety net
# --------------------------------------------------------------------------- #


class _QueryBudget:
    """Progress-handler callback counting SQLite VM steps and wall clock."""

    __slots__ = ("step_limit", "steps", "deadline", "stopped")

    def __init__(self, step_limit: int, deadline: Optional[float]) -> None:
        self.step_limit = int(step_limit)
        self.steps = 0
        self.deadline = deadline
        self.stopped: Optional[str] = None

    def __call__(self) -> int:
        self.steps += VM_STEP_INTERVAL
        if self.step_limit and self.steps >= self.step_limit:
            self.stopped = "steps"
            return 1
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stopped = "timeout"
            return 1
        return 0


def _raise_translated(
    exc: sqlite3.Error, budget: _QueryBudget, timeout: Optional[float]
) -> None:
    message = str(exc)
    if budget.stopped == "steps":
        raise ToolStepLimitError(
            f"query exceeded the SQLite VM-step limit ({budget.step_limit:,} steps)"
        ) from exc
    if budget.stopped == "timeout":
        raise ToolTimeoutError(
            f"query exceeded the wall-clock timeout ({timeout:g}s)"
        ) from exc
    lowered = message.lower()
    if "not authorized" in lowered:
        raise ToolValidationError(
            f"statement rejected by the read-only sandbox: {message}"
        ) from exc
    if "interrupted" in lowered:
        raise ToolExecutionError(f"query was interrupted: {message}") from exc
    raise ToolExecutionError(f"SQL execution failed: {message}") from exc


def _run_statement(
    connection: sqlite3.Connection,
    sql: str,
    *,
    max_rows: Optional[int],
    timeout: Optional[float],
    step_limit: int,
) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """Execute one validated statement and fetch at most ``max_rows`` rows."""

    deadline = None if timeout is None else time.monotonic() + float(timeout)
    budget = _QueryBudget(step_limit, deadline)
    connection.set_progress_handler(budget, VM_STEP_INTERVAL)
    try:
        cursor = connection.execute(sql)
        columns = [str(description[0]) for description in (cursor.description or [])]
        rows = cursor.fetchmany(max_rows) if max_rows is not None else cursor.fetchall()
        return columns, [tuple(row) for row in rows]
    except sqlite3.OperationalError as exc:
        _raise_translated(exc, budget, timeout)
    except sqlite3.Error as exc:
        raise ToolExecutionError(f"SQL execution failed: {exc}") from exc
    finally:
        try:
            connection.set_progress_handler(None, 0)
        except Exception:  # pragma: no cover - defensive
            pass
    raise AssertionError("unreachable")  # pragma: no cover


def _fetch_scan(
    connection: sqlite3.Connection,
    sql: str,
    limit: int,
    *,
    timeout: Optional[float],
    step_limit: int,
) -> List[Tuple[Any, ...]]:
    """Fetch up to ``limit`` rows from a pre-validated single-column SELECT."""

    _columns, rows = _run_statement(
        connection,
        sql,
        max_rows=limit,
        timeout=timeout,
        step_limit=step_limit,
    )
    return rows


# --------------------------------------------------------------------------- #
# Value serialization
# --------------------------------------------------------------------------- #


def _bounded_text(text: str, limit: int = CELL_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def serialize_cell(value: Any) -> Any:
    """Convert one SQLite cell into a bounded, JSON-safe value."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        preview = "0x" + raw[:BLOB_PREVIEW_BYTES].hex().upper()
        if len(raw) > BLOB_PREVIEW_BYTES:
            preview += f" <BLOB_TRUNCATED original_length={len(raw)}>"
        return _bounded_text(preview)
    return _bounded_text(str(value))


def _hash_key(value: Any) -> Tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("blob", bytes(value))
    return ("text", str(value))


def _display_text(value: Any) -> str:
    """Full (untruncated) display text used for matching and tie-breaks."""

    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "0x" + bytes(value).hex().upper()
    return str(value)


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Observation budget (implementation supplement)
#
# ReTool bounds every interpreter feedback so one tool call cannot flood the
# context; these fitters apply the analogous bound to the serialized JSON of
# each tool result.  Degradation is deterministic and never changes the set of
# JSON keys a caller can rely on.
# --------------------------------------------------------------------------- #


def _validate_output_budget(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError("max_output_chars must be an integer or None")
    if value < 1:
        raise ToolValidationError("max_output_chars must be a positive integer or None")
    return value


def _fit_query_budget(payload: Dict[str, Any], budget: Optional[int]) -> Dict[str, Any]:
    """Keep the leading rows that fit the JSON observation budget."""

    if budget is None or len(json_text(payload)) <= budget:
        return payload
    rows = payload["rows"]
    if not rows:
        return payload

    def render(count: int) -> Dict[str, Any]:
        return {
            **payload,
            "rows": rows[:count],
            "row_count": count,
            "truncated": payload["truncated"] or count < len(rows),
            "output_truncated": count < len(rows),
        }

    low, high = 0, len(rows)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json_text(render(middle))) <= budget:
            low = middle
        else:
            high = middle - 1
    return render(low)


_PEEK_LIST_RUNGS = (5, 3, 1, 0)


def _trim_peek_lists(payload: Dict[str, Any], keep: int) -> Dict[str, Any]:
    """Limit top_values/samples/prefixes to ``keep`` entries per column."""

    trimmed: Dict[str, Any] = {}
    for name, stats in payload.items():
        if name == "_scan":
            trimmed[name] = dict(stats)
            continue
        entry = dict(stats)
        for key in ("top_values", "samples", "prefixes"):
            if key in entry:
                entry[key] = list(entry[key])[:keep]
        trimmed[name] = entry
    return trimmed


def _drop_peek_detail_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the samples/prefixes keys from every column profile."""

    dropped: Dict[str, Any] = {}
    for name, stats in payload.items():
        if name == "_scan":
            dropped[name] = dict(stats)
            continue
        entry = dict(stats)
        entry.pop("samples", None)
        entry.pop("prefixes", None)
        dropped[name] = entry
    return dropped


def _mark_peek_output_truncated(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = {name: value for name, value in payload.items() if name != "_scan"}
    scan = dict(payload.get("_scan", {}))
    scan["output_truncated"] = True
    result["_scan"] = scan
    return result


def _trim_peek_columns(payload: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Drop trailing columns; at least the first column is kept."""

    names = [name for name in payload if name != "_scan"]
    scan = dict(payload.get("_scan", {}))

    def render(count: int) -> Dict[str, Any]:
        candidate = {name: payload[name] for name in names[:count]}
        scan_candidate = dict(scan)
        if count < len(names):
            scan_candidate["columns_truncated"] = True
        candidate["_scan"] = scan_candidate
        return candidate

    low, high = 1, len(names)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json_text(render(middle))) <= budget:
            low = middle
        else:
            high = middle - 1
    return render(low)


def _fit_peek_budget(payload: Dict[str, Any], budget: Optional[int]) -> Dict[str, Any]:
    """Walk the deterministic peek degradation ladder."""

    if budget is None or len(json_text(payload)) <= budget:
        return payload
    for keep in _PEEK_LIST_RUNGS:
        candidate = _mark_peek_output_truncated(_trim_peek_lists(payload, keep))
        if len(json_text(candidate)) <= budget:
            return candidate
    candidate = _mark_peek_output_truncated(
        _drop_peek_detail_keys(_trim_peek_lists(payload, 0))
    )
    if len(json_text(candidate)) <= budget:
        return candidate
    return _trim_peek_columns(candidate, budget)


def _fit_bm25_budget(
    entries: List[Dict[str, Any]], budget: Optional[int]
) -> List[Dict[str, Any]]:
    """Keep the leading entries and mark the cut with a trailing marker entry."""

    if budget is None or len(json_text(entries)) <= budget:
        return entries
    total = len(entries)

    def render(count: int) -> List[Dict[str, Any]]:
        return entries[:count] + [
            {"output_truncated": True, "kept": count, "total": total}
        ]

    low, high = 0, total
    while low < high:
        middle = (low + high + 1) // 2
        if len(json_text(render(middle))) <= budget:
            low = middle
        else:
            high = middle - 1
    return render(low)


# --------------------------------------------------------------------------- #
# sqlite_query
# --------------------------------------------------------------------------- #


def _validate_max_return_rows(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError("max_return_rows must be an integer")
    if not 1 <= value <= MAX_RETURN_ROWS_CAP:
        raise ToolValidationError(
            f"max_return_rows must be between 1 and {MAX_RETURN_ROWS_CAP}"
        )
    return value


def _sqlite_query_impl(
    db_file: Union[str, Path],
    sql: str,
    max_return_rows: int = DEFAULT_MAX_RETURN_ROWS,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> Dict[str, Any]:
    _check_db_identity(db_file, db_id)
    max_rows = _validate_max_return_rows(max_return_rows)
    budget = _validate_output_budget(max_output_chars)
    validate_readonly_sql(sql)

    connection, _path = _open_readonly(db_file)
    try:
        columns, rows = _run_statement(
            connection,
            sql,
            max_rows=max_rows + 1,
            timeout=timeout,
            step_limit=step_limit,
        )
    finally:
        connection.close()

    truncated = len(rows) > max_rows
    visible = rows[:max_rows]
    payload = {
        "columns": columns,
        "rows": [[serialize_cell(value) for value in row] for row in visible],
        "row_count": len(visible),
        "truncated": truncated,
        "output_truncated": False,
    }
    return _fit_query_budget(payload, budget)


def sqlite_query(
    db_file: Union[str, Path],
    sql: str,
    max_return_rows: int = DEFAULT_MAX_RETURN_ROWS,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> Dict[str, Any]:
    """Execute one read-only SELECT/WITH statement (paper tool ``sqlite_query``)."""

    return _run_tool(
        _sqlite_query_impl,
        db_file,
        sql,
        max_return_rows,
        db_id=db_id,
        timeout=timeout,
        step_limit=step_limit,
        max_output_chars=max_output_chars,
    )


# --------------------------------------------------------------------------- #
# sqlite_peek
# --------------------------------------------------------------------------- #


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier using SQL-standard double quotes."""

    if not isinstance(identifier, str) or not identifier:
        raise ToolValidationError("identifier must be a non-empty string")
    return '"' + identifier.replace('"', '""') + '"'


def _validate_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ToolValidationError(f"{name} must be between {low} and {high}")
    return value


def _validate_where(where: Any) -> Optional[str]:
    if where is None:
        return None
    if not isinstance(where, str) or not where.strip():
        raise ToolValidationError("where must be a non-empty string or omitted")
    return where.strip()


def _parse_columns(columns: Any) -> Optional[List[str]]:
    if columns is None:
        return None
    if isinstance(columns, str):
        parts = [part.strip() for part in columns.split(",")]
    elif isinstance(columns, (list, tuple)):
        parts = []
        for item in columns:
            if not isinstance(item, str):
                raise ToolValidationError("columns must contain only strings")
            parts.append(item.strip())
    else:
        raise ToolValidationError(
            "columns must be a list of strings or a comma-separated string"
        )
    parts = [part for part in parts if part]
    if not parts:
        raise ToolValidationError("columns must not be empty")
    if len(parts) > MAX_PEEK_COLUMNS:
        raise ToolValidationError(
            f"at most {MAX_PEEK_COLUMNS} columns can be profiled in one call"
        )
    return parts


def _list_tables(connection: sqlite3.Connection) -> List[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _resolve_table(connection: sqlite3.Connection, table: Any) -> str:
    if not isinstance(table, str) or not table.strip():
        raise ToolValidationError("table must be a non-empty string")
    requested = table.strip()
    names = _list_tables(connection)
    if requested in names:
        return requested
    lowered = {name.lower(): name for name in names}
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    preview = ", ".join(names[:20]) if names else "(none)"
    raise ToolValidationError(f"unknown table {requested!r}; available tables: {preview}")


def _table_columns(
    connection: sqlite3.Connection, table: str
) -> List[Tuple[str, str]]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    if not rows:
        raise ToolValidationError(f"unknown table: {table!r}")
    return [(str(row[1]), str(row[2] or "")) for row in rows]


def _resolve_column(requested: str, available: Sequence[Tuple[str, str]]) -> Optional[str]:
    for name, _declared in available:
        if name == requested:
            return name
    lowered = {name.lower(): name for name, _declared in available}
    return lowered.get(requested.lower())


_PERCENT_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?\s*%\s*$")
_DATE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?)$"
)
_ALPHA_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")


def _classify_string(value: str) -> str:
    text = value.strip()
    if not text:
        return "other"
    if _PERCENT_RE.match(text):
        return "percent"
    if _DATE_RE.match(text):
        return "date_like"
    if _to_float(text) is not None:
        return "numeric"
    if _ALPHA_RE.match(text):
        return "alpha"
    return "other"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _profile_column(
    values: Sequence[Any], declared_type: str, limit: int
) -> Dict[str, Any]:
    nulls = 0
    non_null: List[Any] = []
    for value in values:
        if value is None:
            nulls += 1
        else:
            non_null.append(value)

    keyed: Dict[Tuple[str, Any], List[Any]] = {}
    for value in non_null:
        key = _hash_key(value)
        entry = keyed.get(key)
        if entry is None:
            keyed[key] = [serialize_cell(value), 1]
        else:
            entry[1] += 1
    top_values = [
        {"value": display, "count": count}
        for display, count in sorted(
            keyed.values(), key=lambda entry: (-entry[1], str(entry[0]))
        )[:limit]
    ]

    samples: List[Any] = []
    seen = set()
    for value in non_null:
        key = _hash_key(value)
        if key in seen:
            continue
        seen.add(key)
        samples.append(serialize_cell(value))
        if len(samples) >= limit:
            break

    if non_null and all(_is_number(value) for value in non_null):
        minimum: Any = min(non_null)
        maximum: Any = max(non_null)
    elif non_null:
        minimum = min(str(value) for value in non_null)
        maximum = max(str(value) for value in non_null)
    else:
        minimum = maximum = None

    chars = {"numeric": 0, "date_like": 0, "percent": 0, "alpha": 0, "other": 0}
    prefix_counts: Dict[str, int] = {}
    for value in non_null:
        if not isinstance(value, str):
            continue
        chars[_classify_string(value)] += 1
        if len(value) >= PREFIX_LENGTH:
            prefix = value[:PREFIX_LENGTH]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    prefixes = [
        {"prefix": prefix, "count": count}
        for prefix, count in sorted(
            prefix_counts.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]

    return {
        "type": declared_type,
        "nulls": nulls,
        "distinct": len(keyed),
        "min": serialize_cell(minimum) if minimum is not None else None,
        "max": serialize_cell(maximum) if maximum is not None else None,
        "top_values": top_values,
        "samples": samples,
        "chars": chars,
        "prefixes": prefixes,
    }


def _sqlite_peek_impl(
    db_file: Union[str, Path],
    table: str,
    columns: Union[None, str, Sequence[str]] = None,
    limit: int = DEFAULT_PEEK_LIMIT,
    where: Optional[str] = None,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> Dict[str, Any]:
    _check_db_identity(db_file, db_id)
    requested_columns = _parse_columns(columns)
    limit_value = _validate_int(limit, "limit", 1, MAX_PEEK_LIMIT)
    where_value = _validate_where(where)
    budget = _validate_output_budget(max_output_chars)

    connection, _path = _open_readonly(db_file)
    try:
        resolved_table = _resolve_table(connection, table)
        available = _table_columns(connection, resolved_table)
        columns_truncated = False
        if requested_columns is None:
            selected = [name for name, _declared in available][:MAX_PEEK_COLUMNS]
            columns_truncated = len(available) > MAX_PEEK_COLUMNS
        else:
            selected = []
            for requested in requested_columns:
                match = _resolve_column(requested, available)
                if match is None:
                    names = ", ".join(name for name, _declared in available[:20])
                    raise ToolValidationError(
                        f"unknown column {requested!r} in table {resolved_table!r}; "
                        f"available columns: {names}"
                    )
                if match not in selected:
                    selected.append(match)
        if not selected:
            raise ToolValidationError(
                f"table {resolved_table!r} has no columns to profile"
            )

        projection = ", ".join(quote_identifier(name) for name in selected)
        sql = f"SELECT {projection} FROM {quote_identifier(resolved_table)}"
        if where_value is not None:
            sql += f" WHERE {where_value}"
        sql += f" LIMIT {PEEK_SCAN_LIMIT}"
        validate_readonly_sql(sql)

        rows = _fetch_scan(
            connection,
            sql,
            PEEK_SCAN_LIMIT,
            timeout=timeout,
            step_limit=step_limit,
        )
    finally:
        connection.close()

    declared_types = {name: declared for name, declared in available}
    payload: Dict[str, Any] = {}
    for index, name in enumerate(selected):
        values = [row[index] for row in rows]
        payload[name] = _profile_column(values, declared_types[name], limit_value)
    payload["_scan"] = {
        "rows_scanned": len(rows),
        "limit": PEEK_SCAN_LIMIT,
        "truncated": len(rows) >= PEEK_SCAN_LIMIT,
        "columns_truncated": columns_truncated,
        "output_truncated": False,
    }
    return _fit_peek_budget(payload, budget)


def sqlite_peek(
    db_file: Union[str, Path],
    table: str,
    columns: Union[None, str, Sequence[str]] = None,
    limit: int = DEFAULT_PEEK_LIMIT,
    where: Optional[str] = None,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> Dict[str, Any]:
    """Profile one table's columns (paper tool ``sqlite_peek``)."""

    return _run_tool(
        _sqlite_peek_impl,
        db_file,
        table,
        columns,
        limit,
        where,
        db_id=db_id,
        timeout=timeout,
        step_limit=step_limit,
        max_output_chars=max_output_chars,
    )


# --------------------------------------------------------------------------- #
# bm25_search_sqlite
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Conservative screens against pathological backtracking in model-supplied
# patterns.  A screened pattern is matched literally instead of as a regex.
_REGEX_MAX_CHARS = 256
_REGEX_MAX_UNBOUNDED = 4
_NESTED_QUANTIFIER_RE = re.compile(
    r"\([^()]*(?:[*+]|\{\d*,\})[^()]*\)\s*(?:[*+]|\{\d*,\})"
)
_ALT_QUANTIFIER_RE = re.compile(r"\([^()]*\|[^()]*\)\s*(?:[*+]|\{\d*,\})")
_UNBOUNDED_QUANTIFIER_RE = re.compile(r"(?<!\\)(?:[*+]|\{\d*,\})")


def _regex_is_safe(pattern: str) -> bool:
    if len(pattern) > _REGEX_MAX_CHARS:
        return False
    if _NESTED_QUANTIFIER_RE.search(pattern):
        return False
    if _ALT_QUANTIFIER_RE.search(pattern):
        return False
    if len(_UNBOUNDED_QUANTIFIER_RE.findall(pattern)) > _REGEX_MAX_UNBOUNDED:
        return False
    return True


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(texts: Sequence[str], query: str) -> List[float]:
    """Okapi BM25 (k1=1.2, b=0.75) over the candidate cell values."""

    tokenized = [_tokenize(text) for text in texts]
    query_tokens = list(dict.fromkeys(_tokenize(query)))
    count = len(tokenized)
    scores = [0.0] * count
    if not query_tokens or count == 0:
        return scores

    average_length = sum(len(doc) for doc in tokenized) / count or 1.0
    document_frequency: Dict[str, int] = {}
    for doc in tokenized:
        for term in set(doc):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    for index, doc in enumerate(tokenized):
        if not doc:
            continue
        length = len(doc)
        term_frequency: Dict[str, int] = {}
        for term in doc:
            term_frequency[term] = term_frequency.get(term, 0) + 1
        total = 0.0
        for term in query_tokens:
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + BM25_K1 * (
                1 - BM25_B + BM25_B * length / average_length
            )
            total += idf * (frequency * (BM25_K1 + 1)) / denominator
        scores[index] = total
    return scores


def _build_matcher(query: str) -> "re.Pattern[str]":
    """Compile the case-insensitive fallback matcher.

    Invalid patterns and patterns screened as backtracking hazards are treated
    as literal text, so a model-supplied pattern can never stall a rollout.
    """

    if _regex_is_safe(query):
        try:
            return re.compile(query, re.IGNORECASE)
        except re.error:
            pass
    return re.compile(re.escape(query), re.IGNORECASE)


def _bm25_search_impl(
    db_file: Union[str, Path],
    table: str,
    column: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: Optional[str] = None,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> List[Dict[str, Any]]:
    _check_db_identity(db_file, db_id)
    if not isinstance(query, str) or not query.strip():
        raise ToolValidationError("query must be a non-empty string")
    top_k_value = _validate_int(top_k, "top_k", 1, MAX_TOP_K)
    where_value = _validate_where(where)
    budget = _validate_output_budget(max_output_chars)

    connection, _path = _open_readonly(db_file)
    try:
        resolved_table = _resolve_table(connection, table)
        available = _table_columns(connection, resolved_table)
        if not isinstance(column, str) or not column.strip():
            raise ToolValidationError("column must be a non-empty string")
        resolved_column = _resolve_column(column.strip(), available)
        if resolved_column is None:
            names = ", ".join(name for name, _declared in available[:20])
            raise ToolValidationError(
                f"unknown column {column!r} in table {resolved_table!r}; "
                f"available columns: {names}"
            )

        sql = (
            f"SELECT DISTINCT {quote_identifier(resolved_column)} "
            f"FROM {quote_identifier(resolved_table)}"
        )
        if where_value is not None:
            sql += f" WHERE {where_value}"
        sql += f" LIMIT {BM25_CANDIDATE_LIMIT + 1}"
        validate_readonly_sql(sql)

        rows = _fetch_scan(
            connection,
            sql,
            BM25_CANDIDATE_LIMIT + 1,
            timeout=timeout,
            step_limit=step_limit,
        )
    finally:
        connection.close()

    values = [row[0] for row in rows[:BM25_CANDIDATE_LIMIT] if row[0] is not None]
    if not values:
        return []
    match_texts = [_display_text(value) for value in values]
    display_values = [serialize_cell(value) for value in values]

    scores = _bm25_scores(match_texts, query)
    ranked = sorted(
        range(len(values)), key=lambda index: (-scores[index], match_texts[index])
    )
    bm25_hits = [index for index in ranked if scores[index] > 0][:top_k_value]

    # Regex / substring fallback over the same candidate set, bounded by the
    # wall-clock budget so a pathological pattern cannot stall a rollout.
    matcher = _build_matcher(query)
    regex_hits = set()
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    for index, text in enumerate(match_texts):
        if index % 256 == 0 and deadline is not None and time.monotonic() > deadline:
            break
        if matcher.search(text[:512]):
            regex_hits.add(index)

    entries: List[Dict[str, Any]] = []
    chosen = set()
    for index in bm25_hits:
        entry: Dict[str, Any] = {
            "value": display_values[index],
            "bm25": round(scores[index], 3),
        }
        if index in regex_hits:
            entry["regex"] = True
        entries.append(entry)
        chosen.add(index)

    regex_only = [
        index for index in sorted(regex_hits - chosen, key=lambda i: (len(match_texts[i]), match_texts[i]))
    ][:top_k_value]
    for index in regex_only:
        entries.append({"value": display_values[index], "regex": True})
    return _fit_bm25_budget(entries, budget)


def bm25_search_sqlite(
    db_file: Union[str, Path],
    table: str,
    column: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: Optional[str] = None,
    *,
    db_id: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
    step_limit: int = VM_STEP_LIMIT,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Search stored cell values with BM25 plus regex fallback (paper tool)."""

    return _run_tool(
        _bm25_search_impl,
        db_file,
        table,
        column,
        query,
        top_k,
        where,
        db_id=db_id,
        timeout=timeout,
        step_limit=step_limit,
        max_output_chars=max_output_chars,
    )


# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function format) and dispatch
# --------------------------------------------------------------------------- #

TOOL_NAMES = ("sqlite_query", "sqlite_peek", "bm25_search_sqlite")

SQLITE_QUERY_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sqlite_query",
        "description": (
            "Execute exactly one read-only SELECT or WITH statement against the "
            "bound SQLite database (SQL at most {sql_chars:,} characters). Success "
            'returns {{"columns", "rows", "row_count", "truncated", '
            '"output_truncated"}}; truncated means more rows existed than '
            "max_return_rows, output_truncated means the "
            "{output_chars:,}-character observation budget removed rows. Failure "
            'returns {{"error": {{"kind", "message"}}}}. Use it to test a draft, '
            "inspect values, count rows, and re-verify a repair; probe narrowly "
            "instead of raising max_return_rows. Write statements are rejected."
        ).format(
            sql_chars=MAX_SQL_CHARS,
            output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single read-only SELECT or WITH statement.",
                },
                "max_return_rows": {
                    "type": "integer",
                    "default": DEFAULT_MAX_RETURN_ROWS,
                    "description": (
                        "Maximum number of rows returned "
                        f"(1-{MAX_RETURN_ROWS_CAP}, default "
                        f"{DEFAULT_MAX_RETURN_ROWS}). truncated is set when more "
                        "rows were available; output_truncated when the "
                        "observation budget trimmed rows."
                    ),
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}

SQLITE_PEEK_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sqlite_peek",
        "description": (
            "Profile up to {columns:,} columns of one table over the first "
            "{rows:,} rows: declared type, null and distinct counts, min/max, "
            "top_values, samples, character classes, and prefixes, plus a _scan "
            "entry with rows_scanned, limit, truncated, columns_truncated, and "
            "output_truncated. Call it only on a concrete signal named in the "
            "scratch pad: a fraction-vs-percent scale, coded values, a date-like "
            "observed_pattern that disagrees with the predicate, a join key "
            "whose two sides have different normalized types or match_rate below "
            "100%, or an implausible result magnitude."
        ).format(columns=MAX_PEEK_COLUMNS, rows=PEEK_SCAN_LIMIT),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table (or view) name in the current database.",
                },
                "columns": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": (
                        "Columns to profile as a list of names or a comma-separated "
                        "string. Omit to profile every column."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": DEFAULT_PEEK_LIMIT,
                    "description": (
                        "Maximum number of top values, samples, and prefixes per "
                        f"column (1-{MAX_PEEK_LIMIT}, default "
                        f"{DEFAULT_PEEK_LIMIT})."
                    ),
                },
                "where": {
                    "type": "string",
                    "description": (
                        "Optional read-only filter fragment applied before profiling, "
                        "for example \"state = 'CA'\"."
                    ),
                },
            },
            "required": ["table"],
            "additionalProperties": False,
        },
    },
}

BM25_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bm25_search_sqlite",
        "description": (
            "Search the distinct non-NULL stored values of one column with Okapi "
            "BM25 (k1=1.2, b=0.75, lowercase [a-z0-9]+ tokens) plus a "
            "case-insensitive regex/substring fallback. Returns at most 2 x top_k "
            'entries: {"value", "bm25"} hits with "regex": true when also '
            'regex-matched, then regex-only {"value", "regex": true} hits; no '
            "match returns []. Use it when a WHERE literal may differ in storage: "
            "paraphrase, abbreviation, plural, punctuation, diacritics, or "
            "casing. Pass a short keyword, not the whole question. When the "
            "observation budget cuts the list, a trailing "
            '{"output_truncated": true, "kept": N, "total": M} entry marks the cut.'
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table (or view) name in the current database.",
                },
                "column": {
                    "type": "string",
                    "description": "Column whose stored values should be searched.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Literal text or a regular expression to find in the stored "
                        "values. An invalid regular expression is treated as literal "
                        "text."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "default": DEFAULT_TOP_K,
                    "description": (
                        "Maximum number of BM25 hits (and, separately, regex-fallback "
                        f"hits): 1-{MAX_TOP_K}, default {DEFAULT_TOP_K}."
                    ),
                },
                "where": {
                    "type": "string",
                    "description": (
                        "Optional read-only filter fragment applied before searching, "
                        "for example \"state = 'CA'\"."
                    ),
                },
            },
            "required": ["table", "column", "query"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    SQLITE_QUERY_TOOL_SCHEMA,
    SQLITE_PEEK_TOOL_SCHEMA,
    BM25_SEARCH_TOOL_SCHEMA,
]

_TOOL_SIGNATURES: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    "sqlite_query": ({"sql": str}, {"max_return_rows": int}),
    "sqlite_peek": ({"table": str}, {"columns": (list, tuple, str), "limit": int, "where": str}),
    "bm25_search_sqlite": (
        {"table": str, "column": str, "query": str},
        {"top_k": int, "where": str},
    ),
}


def json_text(payload: Any) -> str:
    """Compact, non-ASCII-escaped JSON text for a tool observation."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _coerce_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(f"tool arguments must be valid JSON: {exc}") from exc
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("tool arguments must be a JSON object")
    return dict(arguments)


def _validate_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    required, optional = _TOOL_SIGNATURES[name]
    allowed = set(required) | set(optional)
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolValidationError(
            f"unknown argument(s) for {name}: {', '.join(unknown)}; "
            f"allowed arguments: {', '.join(sorted(allowed))}"
        )
    cleaned: Dict[str, Any] = {}
    for key, value in arguments.items():
        if value is None:
            continue  # an explicit null is treated as an omitted optional
        expected = required.get(key) or optional[key]
        if expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolValidationError(f"{name} argument {key!r} must be an integer")
        elif expected is str:
            if not isinstance(value, str):
                raise ToolValidationError(f"{name} argument {key!r} must be a string")
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(item, str) for item in value):
                raise ToolValidationError(
                    f"{name} argument {key!r} must contain only strings"
                )
        cleaned[key] = value
    missing = [key for key in required if key not in cleaned]
    if missing:
        raise ToolValidationError(
            f"{name} requires the argument(s): {', '.join(missing)}"
        )
    return cleaned


def dispatch(
    db_file: Union[str, Path],
    name: str,
    arguments: Any,
    *,
    db_id: Optional[str] = None,
    max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Run one tool call and return the JSON observation text.

    ``name`` and ``arguments`` come straight from a model tool call; every
    validation failure is reported through the shared error payload instead of
    raising, so a training environment can feed the text back verbatim.
    """

    try:
        if not isinstance(name, str) or name not in _TOOL_SIGNATURES:
            raise ToolValidationError(
                f"unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}"
            )
        cleaned = _validate_arguments(name, _coerce_arguments(arguments))
        if name == "sqlite_query":
            payload = sqlite_query(
                db_file, db_id=db_id, max_output_chars=max_output_chars, **cleaned
            )
        elif name == "sqlite_peek":
            payload = sqlite_peek(
                db_file, db_id=db_id, max_output_chars=max_output_chars, **cleaned
            )
        else:
            payload = bm25_search_sqlite(
                db_file, db_id=db_id, max_output_chars=max_output_chars, **cleaned
            )
        return fit_observation(payload, max_output_chars)
    except ToolError as exc:
        return fit_observation(error_payload(exc.kind, str(exc)), max_output_chars)
    except Exception as exc:  # never let an unexpected failure break a rollout
        return fit_observation(
            error_payload(
                "infrastructure_error",
                f"unexpected tool failure: {type(exc).__name__}: {exc}",
            ),
            max_output_chars,
        )


class SQLiteToolbox:
    """One question's bound tool set (the environment-side convenience wrapper)."""

    def __init__(
        self,
        db_file: Union[str, Path],
        db_id: Optional[str] = None,
        *,
        timeout: Optional[float] = DEFAULT_WALL_TIMEOUT,
        step_limit: int = VM_STEP_LIMIT,
        max_output_chars: Optional[int] = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.db_file = db_file
        self.db_id = db_id
        self.timeout = timeout
        self.step_limit = step_limit
        self.max_output_chars = _validate_output_budget(max_output_chars)
        _check_db_identity(db_file, db_id)

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def query(
        self, sql: str, max_return_rows: int = DEFAULT_MAX_RETURN_ROWS
    ) -> Dict[str, Any]:
        return sqlite_query(
            self.db_file,
            sql,
            max_return_rows,
            db_id=self.db_id,
            timeout=self.timeout,
            step_limit=self.step_limit,
            max_output_chars=self.max_output_chars,
        )

    def peek(
        self,
        table: str,
        columns: Union[None, str, Sequence[str]] = None,
        limit: int = DEFAULT_PEEK_LIMIT,
        where: Optional[str] = None,
    ) -> Dict[str, Any]:
        return sqlite_peek(
            self.db_file,
            table,
            columns,
            limit,
            where,
            db_id=self.db_id,
            timeout=self.timeout,
            step_limit=self.step_limit,
            max_output_chars=self.max_output_chars,
        )

    def bm25_search(
        self,
        table: str,
        column: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: Optional[str] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        return bm25_search_sqlite(
            self.db_file,
            table,
            column,
            query,
            top_k,
            where,
            db_id=self.db_id,
            timeout=self.timeout,
            step_limit=self.step_limit,
            max_output_chars=self.max_output_chars,
        )

    def dispatch(self, name: str, arguments: Any) -> str:
        return dispatch(
            self.db_file,
            name,
            arguments,
            db_id=self.db_id,
            max_output_chars=self.max_output_chars,
        )


__all__ = [
    "BM25_CANDIDATE_LIMIT",
    "BM25_SEARCH_TOOL_SCHEMA",
    "BUSY_TIMEOUT_SECONDS",
    "CELL_MAX_CHARS",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_RETURN_ROWS",
    "DEFAULT_PEEK_LIMIT",
    "DEFAULT_TOP_K",
    "DEFAULT_WALL_TIMEOUT",
    "ERROR_KINDS",
    "MAX_PEEK_COLUMNS",
    "PEEK_SCAN_LIMIT",
    "SQLITE_PEEK_TOOL_SCHEMA",
    "SQLITE_QUERY_TOOL_SCHEMA",
    "SQLiteToolbox",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "ToolError",
    "ToolExecutionError",
    "ToolInfrastructureError",
    "ToolStepLimitError",
    "ToolTimeoutError",
    "ToolValidationError",
    "VM_STEP_LIMIT",
    "bm25_search_sqlite",
    "dispatch",
    "error_payload",
    "json_text",
    "quote_identifier",
    "readonly_authorizer",
    "serialize_cell",
    "sqlite_peek",
    "sqlite_query",
    "validate_readonly_sql",
]
