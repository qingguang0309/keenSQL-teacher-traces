#!/usr/bin/env python3
"""Print the complete ReToolSQL v4 prompt set plus one worked trajectory.

Six sections, in the order a training environment consumes them:

1. tool declarations — the ``tools=`` argument of the chat template
2. system prompt     — ``prompts.RETOOLSQL_SYSTEM_PROMPT`` (v4)
3. few-shot episodes — two static assistant/user sequences
4. user message      — the four-field message (engine / schema / knowledge / question)
5. trajectory        — representative assistant turns, every tool observation real
6. error contract    — real structured-error observations

The assistant scratch pads and tool calls are representative (a real rollout is
produced by the model); every observation is the exact text
``sqlite_tools.dispatch`` returned for that call against the BIRD ``airline``
database, so the row counts, scores and error messages shown are real.

Example:
    /data1/ReViSQL/.venv/bin/python demo_prompt_trajectory.py
    /data1/ReViSQL/.venv/bin/python demo_prompt_trajectory.py --bird-evidence
    /data1/ReViSQL/.venv/bin/python demo_prompt_trajectory.py --dump-json /tmp/prompt.json
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompts import (  # noqa: E402
    FEWSHOT_EXAMPLES,
    RETOOLSQL_SYSTEM_PROMPT,
    SCRATCH_PAD_TAG,
    TOOL_SCHEMAS,
    build_messages,
    text_tool_call,
)
from sqlite_tools import dispatch  # noqa: E402

DEFAULT_DB = Path(
    "/data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/bird_train/download/train"
    "/databases/airline/airline.sqlite"
)
DEFAULT_SCHEMA = Path("/data1/keensql/data_tools/output/bird_train/airline.txt")

EXAMPLE_QUESTION = (
    "Among the flights on 2018/8/1, how many of them were scheduled to depart "
    "from John F. Kennedy International in New York?"
)

# The BIRD instance's own ``evidence`` field, which names both target literals.
# Kept out of the default demo so the tools stay load-bearing; pass
# --bird-evidence to see the instance exactly as BIRD ships it.
BIRD_EVIDENCE = (
    "on 2018/8/1 refers to FL_DATE = '2018/8/1'; depart from refers to ORIGIN; "
    "John F. Kennedy International in New York refers to "
    "Description = 'New York, NY: John F. Kennedy International';"
)

TRAJECTORY = [
    {
        "scratch_pad": (
            "Intent: count flights on 2018/8/1 scheduled to depart from John F. "
            "Kennedy International. Expected output columns: one integer count. "
            "Candidate SQL: join Airlines to Airports on ORIGIN = Code and filter "
            "both literals."
        ),
        "call": {
            "name": "sqlite_query",
            "arguments": {
                "sql": (
                    "SELECT COUNT(T1.Code) FROM Airports AS T1 "
                    "INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN "
                    "WHERE T2.FL_DATE = '2018-08-01' "
                    "AND T1.Description = 'John F. Kennedy International'"
                )
            },
        },
    },
    {
        "scratch_pad": (
            "Intent: the draft returned 0 over a table with 701,352 flights, so at "
            "least one predicate matches nothing. Expected output columns: the "
            "FL_DATE profile. Probe the stored date format before touching the SQL."
        ),
        "call": {
            "name": "sqlite_peek",
            "arguments": {"table": "Airlines", "columns": ["FL_DATE"], "limit": 3},
        },
    },
    {
        "scratch_pad": (
            "Intent: FL_DATE is stored as 'YYYY/M/D' text ('2018/8/1'), so the ISO "
            "literal is dead. The description literal is still a guess; list the "
            "stored spellings instead of assuming the bare airport name. Expected "
            "output columns: distinct stored descriptions."
        ),
        "call": {
            "name": "sqlite_query",
            "arguments": {
                "sql": (
                    "SELECT DISTINCT Description FROM Airports "
                    "WHERE Description LIKE '%Kennedy%'"
                )
            },
        },
    },
    {
        "scratch_pad": (
            "Intent: two stored spellings exist and neither equals the draft "
            "literal, so string equality cannot choose. Rank the candidates "
            "against the question's wording. Expected output columns: candidate "
            "values with BM25 scores."
        ),
        "call": {
            "name": "bm25_search_sqlite",
            "arguments": {
                "table": "Airports",
                "column": "Description",
                "query": "John F. Kennedy International New York",
                "top_k": 3,
            },
        },
    },
    {
        "scratch_pad": (
            "Intent: BM25 ranks 'New York, NY: John F. Kennedy International' first, "
            "so that is the exact stored literal. Re-run the repaired query. "
            "Expected output columns: one integer count."
        ),
        "call": {
            "name": "sqlite_query",
            "arguments": {
                "sql": (
                    "SELECT COUNT(T1.Code) FROM Airports AS T1 "
                    "INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN "
                    "WHERE T2.FL_DATE = '2018/8/1' "
                    "AND T1.Description = 'New York, NY: John F. Kennedy International'"
                )
            },
        },
    },
]

# Both predicates are now backed by observed stored values, and the join follows
# the declared Airlines.ORIGIN = Airports.Code foreign key (match_rate 100.00%).
# The submitted SQL is the fifth tool call in this episode and already ran.
FINAL_SCRATCH_PAD = (
    "Intent: finalize the repaired count. Expected output columns: one integer "
    "count. Verified call: 5"
)

FINAL_SQL = (
    "SELECT COUNT(T1.Code) FROM Airports AS T1 "
    "INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN "
    "WHERE T2.FL_DATE = '2018/8/1' "
    "AND T1.Description = 'New York, NY: John F. Kennedy International'"
)

ERROR_CASES = [
    (
        "write statement (sandbox authorizer)",
        "sqlite_query",
        {"sql": "DELETE FROM Airlines"},
        None,
    ),
    (
        "bad column (execution)",
        "sqlite_query",
        {"sql": "SELECT COUNT(nope) FROM Airlines"},
        None,
    ),
    (
        "unknown table (validation)",
        "sqlite_peek",
        {"table": "NoSuchTable"},
        None,
    ),
    (
        "db_id not bound to this file (infrastructure)",
        "sqlite_query",
        {"sql": "SELECT 1"},
        "not_airline",
    ),
    (
        "more rows than the cap (truncation is not an error)",
        "sqlite_query",
        {"sql": "SELECT TAIL_NUM FROM Airlines WHERE TAIL_NUM IS NOT NULL", "max_return_rows": 2},
        None,
    ),
]


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def render_call(call):
    return text_tool_call(call["name"], call["arguments"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database file")
    parser.add_argument("--db-id", default="airline", help="database identifier")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="M-schema .txt file")
    parser.add_argument("--question", default=EXAMPLE_QUESTION)
    parser.add_argument(
        "--bird-evidence",
        action="store_true",
        help="use the BIRD evidence field as external knowledge (default: empty)",
    )
    parser.add_argument(
        "--dump-json", default=None, help="also write {tools, messages} JSON to this path"
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    schema_path = Path(args.schema)
    for label, path in (("database", db_path), ("schema", schema_path)):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")

    db_details = schema_path.read_text(encoding="utf-8")
    external_knowledge = BIRD_EVIDENCE if args.bird_evidence else ""
    messages = build_messages(
        engine="SQLite",
        db_details=db_details,
        external_knowledge=external_knowledge,
        question=args.question,
        fewshot=True,
    )

    # 1) tool declarations --------------------------------------------------- #
    banner("1. TOOL DECLARATIONS  (chat template: tools=TOOL_SCHEMAS)")
    print(json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))

    # 2) system prompt ------------------------------------------------------- #
    banner('2. SYSTEM PROMPT  (messages[0], role="system")')
    print(RETOOLSQL_SYSTEM_PROMPT)

    # 3) few-shot episodes --------------------------------------------------- #
    banner("3. FEW-SHOT EPISODES  (static prefix, 2 episodes)")
    for example in FEWSHOT_EXAMPLES:
        print()
        print(f"# {example.get('name', '(unnamed)')}: {example.get('demonstrates', '')}")
        for message in example.get("messages", []):
            print(f"---- {message['role']} ----")
            print(message["content"])

    # 4) user message -------------------------------------------------------- #
    banner('4. USER MESSAGE  (messages[-1], role="user")')
    if not external_knowledge:
        print(
            "# external_knowledge is empty on purpose: it keeps bm25_search_sqlite "
            "load-bearing.\n"
            "# Re-run with --bird-evidence to see this BIRD instance as shipped "
            "(its evidence\n"
            "# field names both target literals outright).\n"
        )
    print(messages[-1]["content"])

    # 5) trajectory ---------------------------------------------------------- #
    banner("5. MULTI-TURN TRAJECTORY  (assistant turns representative, observations real)")
    for index, turn in enumerate(TRAJECTORY, start=1):
        print()
        print(f"---- assistant turn {index} ----")
        print(f"<{SCRATCH_PAD_TAG}>")
        print(turn["scratch_pad"])
        print(f"</{SCRATCH_PAD_TAG}>")
        print(render_call(turn["call"]))
        print()
        print(f'---- observation {index} (role="tool", exact dispatch return) ----')
        print(dispatch(db_path, turn["call"]["name"], turn["call"]["arguments"], db_id=args.db_id))

    print()
    print(f"---- assistant turn {len(TRAJECTORY) + 1} (final) ----")
    print(f"<{SCRATCH_PAD_TAG}>")
    print(FINAL_SCRATCH_PAD)
    print(f"</{SCRATCH_PAD_TAG}>")
    print("<relevant_tables>")
    print("Airlines, Airports")
    print("</relevant_tables>")
    print("<relevant_columns>")
    print("Airlines.FL_DATE, Airlines.ORIGIN, Airports.Code, Airports.Description")
    print("</relevant_columns>")
    print("<final_answer>")
    print(FINAL_SQL)
    print("</final_answer>")

    # 6) error contract ------------------------------------------------------ #
    banner('6. ERROR CONTRACT  (every failure is an observation, never an exception)')
    for label, tool, tool_args, db_id in ERROR_CASES:
        payload = dispatch(db_path, tool, tool_args, db_id=db_id or args.db_id)
        print()
        print(f"# {label}")
        print(f"# call: {render_call({'name': tool, 'arguments': tool_args})}")
        print(payload)

    if args.dump_json:
        Path(args.dump_json).write_text(
            json.dumps(
                {"tools": TOOL_SCHEMAS, "messages": messages},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        banner(f"wrote {{tools, messages}} to {args.dump_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
