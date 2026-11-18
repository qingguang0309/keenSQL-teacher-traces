"""Read-only SQLite execution and BIRD-compatible result grading."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# The base image ships Python 3.10.12 with sqlite3 3.37.2, which predates
# RIGHT/FULL OUTER JOIN (added in SQLite 3.39). A few BIRD gold queries use
# that syntax, and modern SQLite also plans some of the slow queries far
# faster. Prefer the vendored pysqlite3 (bundles SQLite 3.51.1) when present,
# falling back to the stdlib module otherwise.
_VENDOR_SQLITE = Path(__file__).resolve().parents[1] / "vendor" / "pysqlite3"
try:
    if _VENDOR_SQLITE.is_dir():
        sys.path.insert(0, str(_VENDOR_SQLITE))
        import pysqlite3 as _pysqlite3

        sys.modules["sqlite3"] = _pysqlite3
    import sqlite3  # noqa: E402
except Exception:
    import sqlite3  # noqa: E402

# Runtime cap for one read-only query. A pathological rollout must not stall
# training indefinitely. REVISQL_SQL_TIMEOUT overrides this for offline builds.
DEFAULT_SQL_TIMEOUT = float(os.environ.get("REVISQL_SQL_TIMEOUT", "30.0"))
TOOL_MAX_ROWS = 50
TOOL_MAX_CELL_CHARS = 200


class SQLExecutionError(RuntimeError):
    """A candidate SQL statement could not be executed."""


class SQLTimeoutError(SQLExecutionError):
    """SQLite interrupted a statement after the progress-handler deadline."""


class SQLInfrastructureError(RuntimeError):
    """The database or execution infrastructure could not be used."""


class UnknownGradingMethod(ValueError):
    """The dataset requested a grading method this implementation does not know."""


def _constant(name: str) -> int | None:
    value = getattr(sqlite3, name, None)
    return value if isinstance(value, int) else None


# These actions can mutate the database or load code. immutable=1 and
# query_only=ON are defense in depth; the authorizer makes the policy explicit.
_DENIED_ACTIONS = {
    value
    for name in (
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
    )
    if (value := _constant(name)) is not None
}
for _name in ("SQLITE_ATTACH", "SQLITE_DETACH"):
    if (_value := _constant(_name)) is not None:
        _DENIED_ACTIONS.add(_value)

# Metadata pragmas are needed to inspect a schema. All other pragmas are
# denied, including journal_mode, writable_schema, and trusted_schema changes.
_SAFE_METADATA_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "table_list",
    "index_list",
    "index_info",
    "index_xinfo",
    "foreign_key_list",
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
    first: str | None,
    second: str | None,
    database: str | None,
    source: str | None,
) -> int:
    """SQLite authorizer callback used for every user and schema connection."""
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


def readonly_uri(db_file: str | Path) -> str:
    """Return the immutable, read-only URI used for SQLite connections."""
    path = Path(db_file).expanduser().resolve()
    return f"{path.as_uri()}?mode=ro&immutable=1"


def _open_readonly(db_file: str | Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(readonly_uri(db_file), uri=True, timeout=5.0)
    except (OSError, sqlite3.Error) as exc:
        raise SQLInfrastructureError(f"could not open SQLite database: {exc}") from exc
    try:
        # Set this before installing the authorizer because it is itself a
        # PRAGMA, while subsequent user PRAGMAs are subject to the policy.
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(readonly_authorizer)
    except (OSError, sqlite3.Error) as exc:
        connection.close()
        raise SQLInfrastructureError(f"could not configure SQLite database: {exc}") from exc
    return connection


def execute_query(
    db_file: str | Path,
    sql: str,
    timeout: float | None = DEFAULT_SQL_TIMEOUT,
    *,
    max_rows: int | None = None,
) -> list[tuple[Any, ...]]:
    """Execute one read-only SQL statement and return rows.

    ``max_rows`` is only an optional fetch limit for display/tool callers. The
    default is unbounded because final result grading must see every row.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SQLExecutionError("SQL must be a non-empty string")

    connection = _open_readonly(db_file)
    if timeout is not None:
        started = time.monotonic()

        def watchdog() -> int:
            return int(time.monotonic() - started >= timeout)

        connection.set_progress_handler(watchdog, 10_000)
    try:
        cursor = connection.execute(sql)
        if max_rows is None:
            rows = cursor.fetchall()
        else:
            if max_rows < 0:
                raise ValueError("max_rows must be non-negative or None")
            rows = cursor.fetchmany(max_rows)
        return [tuple(row) for row in rows]
    except sqlite3.OperationalError as exc:
        if timeout is not None and "interrupted" in str(exc).lower():
            raise SQLTimeoutError(f"SQL query timed out after {timeout:g}s") from exc
        raise SQLExecutionError(f"SQL execution failed: {exc}") from exc
    except sqlite3.Error as exc:
        raise SQLExecutionError(f"SQL execution failed: {exc}") from exc
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def run_sql(
    db_file: str | Path,
    sql: str,
    timeout: float | None = DEFAULT_SQL_TIMEOUT,
    *,
    max_rows: int | None = None,
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    """Compatibility wrapper returning ``(rows, error_text)`` for candidates."""
    try:
        return execute_query(db_file, sql, timeout, max_rows=max_rows), None
    except SQLInfrastructureError:
        raise
    except SQLExecutionError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _display_cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text[:TOOL_MAX_CELL_CHARS]


def format_tool_text(
    rows: Sequence[Sequence[Any]],
    *,
    truncated: bool = False,
) -> str:
    """Render at most 50 rows and at most 200 characters per cell."""
    visible = rows[:TOOL_MAX_ROWS]
    if not visible:
        text = "(empty result)"
    else:
        text = "\n".join("\t".join(_display_cell(value) for value in row) for row in visible)
    if truncated or len(rows) > TOOL_MAX_ROWS:
        text = "Result truncated to 50 rows.\n" + text
    return text


def execute_tool_query(
    db_file: str | Path,
    sql: str,
    timeout: float | None = DEFAULT_SQL_TIMEOUT,
) -> str:
    """Execute a tool query and return the bounded textual observation."""
    connection_rows = execute_query(db_file, sql, timeout, max_rows=TOOL_MAX_ROWS + 1)
    return format_tool_text(connection_rows, truncated=len(connection_rows) > TOOL_MAX_ROWS)


# Names used by the local launcher and by older callers.
_run_sql = run_sql
execute_sql = execute_query
execute_sql_query = execute_tool_query


def _nonempty(rows: Iterable[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [
        tuple(row)
        for row in rows
        if any(value is not None and str(value).strip() != "" for value in row)
    ]


def _normalized_row(row: Sequence[Any]) -> tuple[str, ...]:
    # BIRD/Tinker semantics intentionally ignore SELECT column order inside a
    # row while list semantics preserve result-row order. Values are compared by
    # their string form, exactly as the reference grader does.
    return tuple(sorted(str(value) for value in row))


def _parse_method(method: str) -> tuple[str, str | None, int | None]:
    if not isinstance(method, str):
        raise UnknownGradingMethod(f"Unknown grading method: {method!r}")
    parts = [part.strip() for part in method.split(",")]
    name = parts[0].lower()
    if name in {"multiset", "set", "list"} and len(parts) == 1:
        return name, None, None
    if name == "subset":
        if len(parts) == 1:
            return name, None, None
        if len(parts) == 3 and parts[1] in {"=", ">="}:
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise UnknownGradingMethod(f"Unknown grading method: {method!r}") from exc
            if count < 0:
                raise UnknownGradingMethod(f"Unknown grading method: {method!r}")
            return name, parts[1], count
    raise UnknownGradingMethod(f"Unknown grading method: {method!r}")


def grade_results(
    gold_rows: Iterable[Sequence[Any]],
    predicted_rows: Iterable[Sequence[Any]],
    method: str = "multiset",
) -> bool:
    """Grade results with the exact BIRD/Tinker comparison behavior.

    Delegates to :func:`common.sql_exec.grade_rows` when the project package is
    importable, so the SFT importer, the RL reward, and this screening harness
    share one implementation. Comparison is exact: no numeric tolerance.
    """

    try:
        from common.sql_exec import grade_rows as _grade_rows
    except ImportError:  # pragma: no cover - standalone script use
        _PROJECT_ROOT = Path(__file__).resolve().parents[1]
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            from common.sql_exec import grade_rows as _grade_rows
        except ImportError:
            _grade_rows = None
    if _grade_rows is not None:
        return _grade_rows(gold_rows, predicted_rows, grading_method=method)

    gold = _nonempty(gold_rows)
    predicted = _nonempty(predicted_rows)
    name, count_operator, count = _parse_method(method)

    if bool(gold) != bool(predicted):
        return False
    if gold and len(gold[0]) != len(predicted[0]):
        return False

    normalized_gold = [_normalized_row(row) for row in gold]
    normalized_predicted = [_normalized_row(row) for row in predicted]
    if name == "multiset":
        return Counter(normalized_gold) == Counter(normalized_predicted)
    if name == "set":
        return set(normalized_gold) == set(normalized_predicted)
    if name == "list":
        return normalized_gold == normalized_predicted

    # The BIRD subset grader deduplicates candidate rows before checking both
    # the optional row count and membership in the gold result.
    unique_predicted = set(normalized_predicted)
    if count_operator == "=" and len(unique_predicted) != count:
        return False
    if count_operator == ">=" and len(unique_predicted) < count:
        return False
    return unique_predicted.issubset(set(normalized_gold))


def grade(gold_rows: Iterable[Sequence[Any]], predicted_rows: Iterable[Sequence[Any]], grading_method: str = "multiset") -> tuple[bool, dict[str, str]]:
    """Compatibility form of the repository grader, with hard-fail methods."""
    matched = grade_results(gold_rows, predicted_rows, grading_method)
    return matched, {"message": "Results match" if matched else "Results do not match"}


__all__ = [
    "DEFAULT_SQL_TIMEOUT",
    "SQLExecutionError",
    "SQLTimeoutError",
    "SQLInfrastructureError",
    "UnknownGradingMethod",
    "readonly_uri",
    "readonly_authorizer",
    "execute_query",
    "execute_sql",
    "execute_tool_query",
    "execute_sql_query",
    "run_sql",
    "_run_sql",
    "format_tool_text",
    "grade_results",
    "grade",
]
