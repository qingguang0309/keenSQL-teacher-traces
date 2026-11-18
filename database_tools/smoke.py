#!/usr/bin/env python3
"""Smoke-run the three ReToolSQL database tools against a real SQLite database.

Example:
    /data1/ReViSQL/.venv/bin/python smoke.py \
        --db /data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/\
bird_train/download/train/databases/airline/airline.sqlite --db-id airline

Prints the exact JSON text a training environment would place in the tool
observation message for one sqlite_query, one sqlite_peek, and one
bm25_search_sqlite call.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlite_tools import TOOL_SCHEMAS, dispatch  # noqa: E402


def _pretty(text):
    return json.dumps(json.loads(text), ensure_ascii=False, indent=2)


def _error(payload):
    return isinstance(payload, dict) and "error" in payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to a SQLite database file")
    parser.add_argument("--db-id", default=None, help="optional database identifier")
    parser.add_argument("--table", default=None, help="table to profile (default: first table)")
    parser.add_argument("--column", default=None, help="column to search (default: first text-like column)")
    parser.add_argument("--show-prompt", action="store_true", help="also print the system prompt")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.is_file():
        parser.error(f"database not found: {db_path}")

    print(f"# database : {db_path}")
    print(f"# tools    : {', '.join(schema['function']['name'] for schema in TOOL_SCHEMAS)}")
    failed = False

    # 1) sqlite_query -------------------------------------------------------- #
    list_tables = dispatch(
        db_path,
        "sqlite_query",
        {
            "sql": "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name",
            "max_return_rows": 20,
        },
        db_id=args.db_id,
    )
    print("\n## sqlite_query: SELECT name FROM sqlite_master ...")
    print(_pretty(list_tables))
    listing = json.loads(list_tables)
    if _error(listing):
        print("!! sqlite_query failed", file=sys.stderr)
        return 1
    tables = [row[0] for row in listing["rows"]]
    if not tables:
        print("!! database has no user tables", file=sys.stderr)
        return 1

    table = args.table or tables[0]
    count = dispatch(
        db_path,
        "sqlite_query",
        {"sql": f'SELECT COUNT(*) AS row_count FROM "{table}"'},
        db_id=args.db_id,
    )
    print(f'\n## sqlite_query: SELECT COUNT(*) FROM "{table}"')
    print(_pretty(count))
    failed = failed or _error(json.loads(count))

    # 2) sqlite_peek --------------------------------------------------------- #
    peek_text = dispatch(
        db_path,
        "sqlite_peek",
        {"table": table, "limit": 5},
        db_id=args.db_id,
    )
    print(f'\n## sqlite_peek: table "{table}" (limit=5)')
    print(_pretty(peek_text))
    peek = json.loads(peek_text)
    if _error(peek):
        print("!! sqlite_peek failed", file=sys.stderr)
        return 1

    # 3) bm25_search_sqlite -------------------------------------------------- #
    search_column = args.column
    search_value = None
    for name, stats in peek.items():
        if name == "_scan" or not isinstance(stats, dict):
            continue
        if search_column and name != search_column:
            continue
        for sample in stats.get("samples", []):
            if isinstance(sample, str) and re.search(r"[A-Za-z0-9]", sample):
                token = re.search(r"[A-Za-z0-9]+", sample).group(0)
                search_value = token
                search_column = name
                break
        if search_value:
            break
    if search_column is None:
        search_column = next(name for name in peek if name != "_scan")
    if search_value is None:
        search_value = search_column

    search_text = dispatch(
        db_path,
        "bm25_search_sqlite",
        {"table": table, "column": search_column, "query": search_value, "top_k": 5},
        db_id=args.db_id,
    )
    print(
        f'\n## bm25_search_sqlite: table "{table}", column "{search_column}", '
        f'query "{search_value}" (top_k=5)'
    )
    print(_pretty(search_text))
    failed = failed or _error(json.loads(search_text))

    if args.show_prompt:
        from prompts import RETOOLSQL_SYSTEM_PROMPT

        print("\n## system prompt")
        print(RETOOLSQL_SYSTEM_PROMPT)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
