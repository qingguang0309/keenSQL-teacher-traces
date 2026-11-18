#!/usr/bin/env python3
"""Test harness for the ReToolSQL prompt + database_tools contract.

Three levels, cheapest first:

  L1  contract  -- static consistency between retoolsql_prompt/retoolsql_prompt.txt, the tool
                   schemas (TOOL_SCHEMAS), and the documented limits. No model.
  L2  protocol  -- drive a real episode through the parser + agent loop +
                   real tool dispatch, using a *scripted* assistant (no model).
                   Verifies format parsing, error-driven repair, the 8-call
                   budget, and final-answer annotations.
  L3  model     -- same loop, but the assistant turns come from an
                   OpenAI-compatible chat endpoint (e.g. vLLM serving
                   Qwen3.5-9B). Skipped unless --model-url is given and
                   reachable.

Usage:
    PY=/data/keenSQL/.venv/bin/python
    $PY /data/keenSQL/train_prompt/test_retoolsql_prompt.py --db <db.sqlite> --db-id <id>
    # L3:
    $PY /data/keenSQL/train_prompt/test_retoolsql_prompt.py --db <db.sqlite> --model-url http://127.0.0.1:8000/v1 \
        --model Qwen3.5-9B --question "How many flights were cancelled?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_DIR = HERE / "retoolsql_prompt"
PROMPT_FILE = PROMPT_DIR / "v4" / "system_prompt.txt"
LEGACY_PROMPT_FILE = PROMPT_DIR / "retoolsql_prompt.txt"
TOOLS_DIR = HERE.parent / "database_tools"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(PROMPT_DIR))

import prompt_builder as PB  # noqa: E402
import sqlite_tools as st  # noqa: E402

MAX_TOOL_CALLS = 8  # prompt section 5.1


def load_system_prompt(path: Path = PROMPT_FILE) -> str:
    """The canonical v4 system prompt under test."""
    return path.read_text(encoding="utf-8").strip()


def build_messages(*, engine, db_details, external_knowledge, question, system_prompt=None,
                   fewshot=False):
    """Assemble [system, (few-shot), user] for the agent loop."""
    if system_prompt is None:
        system_prompt = load_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    if fewshot:
        messages.extend(PB.fewshot_messages())
    messages.append(
        {
            "role": "user",
            "content": PB.render_user_message(
                engine=engine,
                db_details=db_details,
                external_knowledge=external_knowledge,
                question=question,
            ),
        }
    )
    return messages


# --------------------------------------------------------------------------- #
# L1: static contract checks
# --------------------------------------------------------------------------- #

def check_contract() -> list[str]:
    """Return a list of failure strings ([] means all good)."""
    failures: list[str] = []
    text = PROMPT_FILE.read_text(encoding="utf-8")
    flat = " ".join(text.split())

    # v4/system_prompt.txt is the only copy; the legacy root duplicate was
    # removed during the prompt cleanup, so there is nothing to drift-check.

    # every tool name and parameter must appear; no db_file/db_id leak
    for name in st.TOOL_NAMES:
        if name not in text:
            failures.append(f"prompt does not mention tool {name!r}")
    for schema in st.TOOL_SCHEMAS:
        fn = schema["function"]
        for param in fn["parameters"]["properties"]:
            if param not in text:
                failures.append(f"prompt does not mention parameter {param!r} of {fn['name']}")
    # db_id/db_file may be mentioned in prose (the prompt explicitly says tools do
    # not take db_id); what must never happen is them being exposed as a
    # model-visible tool parameter in the schemas or in an example call.
    if "db_id" in json.dumps(st.TOOL_SCHEMAS) or "db_file" in json.dumps(st.TOOL_SCHEMAS):
        failures.append("TOOL_SCHEMAS exposes db_id/db_file")

    # documented limits must match module constants (line wrapping is allowed)
    for needle, why in [
        ("100,000", "MAX_SQL_CHARS=100000"),
        ("[1, 10000]", "max_return_rows cap"),
        ("[1, 500]", "peek limit range"),
        ("[1, 100]", "top_k range"),
        ("20,000 rows", "PEEK_SCAN_LIMIT"),
        ("100 columns", "MAX_PEEK_COLUMNS"),
        ("k1 = 1.2", "BM25 k1"),
        ("b = 0.75", "BM25 b"),
        ("2 × top_k", "result cap"),
    ]:
        if needle not in flat:
            failures.append(f"prompt missing documented value {needle!r} ({why})")

    # all five error kinds must be documented
    for kind in st.ERROR_KINDS:
        if kind not in text:
            failures.append(f"prompt does not document error kind {kind!r}")

    # required behavioural phrases
    for phrase in (
        "one per turn",
        "verification",
        "single complete statement",
        "never use placeholders, ellipses, or omitted fragments",
        "repair guidance",
    ):
        if " ".join(phrase.split()) not in flat:
            failures.append(f"prompt missing required phrase {phrase!r}")

    # every concrete <tool_call> example must parse with schema-legal arguments
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        block = block.strip()
        # Prose such as "one tool call inside <tool_call>...</tool_call>" also
        # matches the regex; keep only bodies that start like a real call.
        if not block.startswith(("<function=", "{")):
            continue
        if block == "..." or "<tool name>" in block:
            continue
        try:
            name, args = _parse_tool_call_body(block)
        except ProtocolError as exc:
            failures.append(f"example tool_call is invalid: {exc}: {block!r}")
            continue
        req, opt = st._TOOL_SIGNATURES[name]
        allowed = set(req) | set(opt)
        if set(args) - allowed:
            failures.append(f"example {name} has unknown args {set(args) - allowed}")
        if set(req) - set(args):
            failures.append(f"example {name} missing required args {set(req) - set(args)}")

    # the protocol tags the parser depends on must all be present
    for tag in ("scratch_pad", "tool_call", "relevant_tables", "relevant_columns", "final_answer"):
        if f"<{tag}>" not in text:
            failures.append(f"prompt missing protocol tag <{tag}>")

    return failures


# --------------------------------------------------------------------------- #
# L2/L3: assistant-turn parser and agent loop
# --------------------------------------------------------------------------- #

class ProtocolError(Exception):
    """The assistant turn violated the required turn protocol."""


_TAG = lambda name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.S)


def _block(text: str, name: str) -> str | None:
    """Return one annotation block, tolerating a mismatched closing tag.

    Models occasionally close `<relevant_tables>` with `</relevant_columns>`.
    The content is still unambiguous because the next opening tag bounds it, so
    accept any closing tag or the next known opening tag as the terminator.
    """
    strict = _TAG(name).search(text)
    if strict:
        return strict.group(1)
    opened = re.search(rf"<{name}>", text)
    if not opened:
        return None
    start = opened.end()
    ends = [
        match.start()
        for pattern in (
            rf"</{name}>",
            r"</relevant_tables>",
            r"</relevant_columns>",
            r"</final_answer>",
            r"<relevant_tables>",
            r"<relevant_columns>",
            r"<final_answer>",
        )
        for match in re.finditer(pattern, text[start:])
    ]
    if not ends:
        return text[start:]
    return text[start : start + min(ends)]


_ANNOTATION_END = re.compile(
    r"</relevant_tables>|</relevant_columns>|</final_answer>"
    r"|<relevant_tables>|<relevant_columns>|<final_answer>"
)


def _annotation_blocks(text: str) -> tuple[str | None, str | None, list[str]]:
    """Parse the two annotation blocks, recovering from malformed tags.

    Two shapes appear in real runs and are accepted as recoverable:

    1. the tables block is closed with ``</relevant_columns>``;
    2. the ``<relevant_columns>`` opening tag is missing, leaving its content
       between two closing tags.

    Both are unambiguous from the tag sequence, so they are parsed with a
    warning rather than discarding an otherwise complete final answer.
    """

    warnings: list[str] = []
    strict_tables = _TAG("relevant_tables").search(text)
    strict_columns = _TAG("relevant_columns").search(text)
    if strict_tables and strict_columns:
        return strict_tables.group(1), strict_columns.group(1), warnings

    open_tables = re.search(r"<relevant_tables>", text)
    if not open_tables:
        return None, None, warnings

    start = open_tables.end()
    end = _ANNOTATION_END.search(text, start)
    if not end:
        return None, None, warnings
    tables = text[start : end.start()]

    rest = end.end()
    open_columns = re.search(r"<relevant_columns>", text[rest:])
    columns_start = rest + open_columns.end() if open_columns else rest
    columns_end = _ANNOTATION_END.search(text, columns_start)
    if not columns_end:
        return None, None, warnings
    columns = text[columns_start : columns_end.start()]

    warnings.append("recovered annotation blocks from malformed tags")
    return tables, columns, warnings


def _validate_arguments(name: str, arguments: dict) -> dict:
    """Check argument names and JSON types against the tool signature."""
    required, optional = st._TOOL_SIGNATURES.get(name, ({}, {}))
    allowed = set(required) | set(optional)
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise ProtocolError(f"unknown argument(s) {extra} for {name}")
    missing = sorted(set(required) - set(arguments))
    if missing:
        raise ProtocolError(f"missing required argument(s) {missing} for {name}")
    for key, value in arguments.items():
        expected = required.get(key, optional.get(key))
        if expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProtocolError(f"argument {key!r} of {name} must be an integer")
        elif expected is str:
            if not isinstance(value, str):
                raise ProtocolError(f"argument {key!r} of {name} must be a string")
        elif isinstance(expected, tuple):
            if isinstance(value, list):
                if not all(isinstance(item, str) for item in value):
                    raise ProtocolError(
                        f"argument {key!r} of {name} must be a list of strings"
                    )
            elif isinstance(value, str):
                pass
            else:
                raise ProtocolError(
                    f"argument {key!r} of {name} must be a list or a string"
                )
    return arguments


def _parse_tool_call_body(body: str, repairs: list[str] | None = None,
                          repair_unclosed_function: bool = True) -> tuple[str, dict]:
    """Parse a tool-call body; accepts Qwen XML and legacy v1 JSON."""

    try:
        return PB.parse_tool_call_body(
            body, st._TOOL_SIGNATURES, repairs=repairs,
            repair_unclosed_function=repair_unclosed_function,
        )
    except PB.ToolCallParseError as exc:
        raise ProtocolError(str(exc)) from exc


_VERIFIED_CALL = re.compile(r"Verified call:\s*(\d+)", re.I)


def parse_assistant_turn(text: str, *, require_final_scratch_pad: bool = True,
                         require_tool_scratch_pad: bool = True,
                         repair_unclosed_function: bool = True) -> dict:
    """Parse one assistant turn per section 3 of the prompt.

    Returns one of:
      {"action": "tool_call", "name": ..., "arguments": {...}, "scratch_pad": ...}
      {"action": "final_answer", "tables": [...], "columns": [...], "sql": ...,
       "scratch_pad": ..., "warnings": [...]}
    Raises ProtocolError on any violation.

    ``require_*_scratch_pad=False`` downgrades a missing <scratch_pad> on that
    kind of turn from a hard error to a recorded warning. Reasoning-API models
    sometimes keep the plan in their hidden reasoning channel and emit only the
    action; that should not abort an otherwise valid episode.
    """
    if "<observation>" in text:
        raise ProtocolError("assistant must not emit <observation> (the environment does)")

    warnings: list[str] = []
    m_scratch = _TAG("scratch_pad").search(text)
    is_final = _TAG("final_answer").search(text) is not None
    allow_missing = (
        (is_final and not require_final_scratch_pad)
        or (not is_final and not require_tool_scratch_pad)
    )
    if not m_scratch and not allow_missing:
        raise ProtocolError("turn is missing a bounded <scratch_pad>")
    if not m_scratch and allow_missing:
        warnings.append(
            ("final-answer" if is_final else "tool-call") + " turn omitted <scratch_pad>"
        )
    scratch = m_scratch.group(1) if m_scratch else ""

    tool_blocks = _TAG("tool_call").findall(text)
    has_final = _TAG("final_answer").search(text) is not None

    if tool_blocks and has_final:
        raise ProtocolError("turn must not contain both a tool call and a final answer")
    if len(tool_blocks) > 1:
        raise ProtocolError(f"turn emitted {len(tool_blocks)} tool calls; exactly one per turn")

    if tool_blocks:
        repairs: list[str] = []
        name, arguments = _parse_tool_call_body(
            tool_blocks[0], repairs, repair_unclosed_function
        )
        warnings.extend(f"tool-call repair: {item}" for item in repairs)
        return {
            "action": "tool_call",
            "name": name,
            "arguments": arguments,
            "scratch_pad": scratch,
            "warnings": warnings,
            "repairs": repairs,
        }

    if has_final:
        tables_block, columns_block, annotation_warnings = _annotation_blocks(text)
        m_sql = _TAG("final_answer").search(text)
        if tables_block is None or columns_block is None or m_sql is None:
            raise ProtocolError(
                "final-answer turn needs <relevant_tables>, <relevant_columns>, <final_answer>"
            )
        warnings.extend(annotation_warnings)
        split = lambda block: [x.strip() for x in re.split(r"[,\n]", block) if x.strip()]
        verified = _VERIFIED_CALL.search(scratch)
        verified_call = int(verified.group(1)) if verified else None
        if verified_call is None:
            warnings.append("final-answer turn omitted 'Verified call: k'")
        return {
            "action": "final_answer",
            "tables": split(tables_block),
            "columns": split(columns_block),
            "sql": m_sql.group(1).strip(),
            "scratch_pad": scratch,
            "verified_call": verified_call,
            "warnings": warnings,
        }

    raise ProtocolError("turn has neither a tool call nor a final answer")


def _turn_facts(parsed: dict, assistant: str, observation: str = "") -> dict:
    """Raw per-turn facts used to audit whether each prompt section worked.

    These are deliberately mechanical: they record what the model emitted, not
    whether it was right. An offline aggregator turns them into section-level
    compliance rates.
    """

    scratch = parsed.get("scratch_pad", "") or ""
    strict_tables = _TAG("relevant_tables").search(assistant) is not None
    strict_columns = _TAG("relevant_columns").search(assistant) is not None
    facts = {
        "action": parsed["action"],
        "has_scratch": bool(scratch.strip()),
        "scratch_words": len(scratch.split()),
        "has_tool_call": "<tool_call>" in assistant,
        "final_tags_closed": (
            assistant.count("<final_answer>") == assistant.count("</final_answer>")
        ),
        "annotation_pairing_strict": strict_tables and strict_columns,
        "observation_kind": "",
        "next_tool": "",
    }
    if "error" in observation:
        facts["observation_kind"] = "error"
    elif '"row_count":0' in observation:
        facts["observation_kind"] = "empty"
    elif observation:
        facts["observation_kind"] = "rows"
    facts["repairs"] = parsed.get("repairs", [])
    facts["verified_call"] = parsed.get("verified_call")
    if parsed["action"] == "tool_call":
        facts["tool"] = parsed["name"]
        arguments = parsed.get("arguments", {})
        facts["sql"] = (arguments.get("sql") or "").strip()
        facts["peek_columns"] = arguments.get("columns")
        facts["bm25_column"] = arguments.get("column", "")
    else:
        facts["tool"] = ""
        facts["sql"] = parsed.get("sql", "").strip()
        facts["tables"] = parsed.get("tables", [])
        facts["columns"] = parsed.get("columns", [])
    return facts


def run_episode(policy, db_file, db_id, question, schema, max_turns=10, verbose=True,
                require_final_scratch_pad=True, external_knowledge="", fewshot=False,
                require_tool_scratch_pad=True, system_prompt=None,
                repair_unclosed_function=True, prompt_messages=None):
    """Drive the loop: policy(messages) -> assistant text -> parse -> dispatch.

    ``prompt_messages`` overrides the rendered prefix with a literal message list.
    The RFT screening run uses it to replay the exact prefix stored in the SFT
    corpus: the SFT builder pruned large schemas to ``16000 - teacher_appendix``
    tokens per question, so re-rendering at a flat 16,000 would show the policy a
    different schema (8 of 22 tables instead of 6) than training ever used.

    Returns {"transcript": [...], "final": {...}|None, "tool_calls": int,
             "errors": [...], "warnings": [...], "anchors": [...]}.
    """
    if prompt_messages is not None:
        messages = [
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in prompt_messages
        ]
    else:
        messages = build_messages(
            engine="SQLite", db_details=schema, external_knowledge=external_knowledge,
            question=question, fewshot=fewshot, system_prompt=system_prompt,
        )
    transcript, tool_calls, final, errors, warnings = [], 0, None, [], []
    anchors: list[dict] = []
    pending_observation = ""

    for _turn in range(max_turns):
        assistant = policy(messages)
        transcript.append({"role": "assistant", "content": assistant})
        if verbose:
            print("\n--- assistant ---")
            print(assistant)

        try:
            parsed = parse_assistant_turn(
                assistant, require_final_scratch_pad=require_final_scratch_pad,
                require_tool_scratch_pad=require_tool_scratch_pad,
                repair_unclosed_function=repair_unclosed_function,
            )
        except ProtocolError as exc:
            errors.append(str(exc))
            fact = {
                "action": "protocol_error",
                "has_scratch": "<scratch_pad>" in assistant,
                "scratch_words": 0,
                "has_tool_call": "<tool_call>" in assistant,
                "final_tags_closed": (
                    assistant.count("<final_answer>") == assistant.count("</final_answer>")
                ),
                "annotation_pairing_strict": False,
                "observation_kind": _observation_kind(pending_observation),
                "next_tool": "",
                "tool": "",
                "sql": "",
                "error": str(exc),
            }
            anchors.append(fact)
            if verbose:
                print(f"!! PROTOCOL ERROR: {exc}")
            break

        facts = _turn_facts(parsed, assistant, pending_observation)
        anchors.append(facts)
        pending_observation = ""
        warnings.extend(parsed.get("warnings", []))

        if parsed["action"] == "final_answer":
            final = parsed
            break

        if tool_calls >= MAX_TOOL_CALLS:
            errors.append(f"tool-call budget ({MAX_TOOL_CALLS}) exceeded")
            if verbose:
                print("!! tool-call budget exceeded")
            break

        observation = st.dispatch(db_file, parsed["name"], parsed["arguments"], db_id=db_id)
        tool_calls += 1
        anchors[-1]["executed_sql"] = (parsed["arguments"].get("sql") or "").strip()
        anchors[-1]["executed_tool"] = parsed["name"]
        anchors[-1]["observation_after"] = _observation_kind(observation)
        transcript.append({"role": "observation", "content": observation})
        pending_observation = observation
        if verbose:
            print(f"--- observation ({parsed['name']}) ---")
            print(observation)
        # the environment appends the observation; feed it back as the next turn input
        messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": f"<observation>\n{observation}\n</observation>"})

    verification = {"declared": None, "matches": None, "reason": ""}
    if final is not None:
        verification = _check_verified_call(final, anchors)

    return {"transcript": transcript, "final": final, "tool_calls": tool_calls,
            "errors": errors, "warnings": warnings, "anchors": anchors,
            "verification": verification}


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).strip().lower()


def _check_verified_call(final: dict, anchors: list) -> dict:
    """Check the declared ``Verified call: k`` against the recorded calls.

    ``k`` counts every executed tool call in order, which is how the models
    actually index them; the cited call must itself be a ``sqlite_query``.
    """

    declared = final.get("verified_call")
    executed = [
        (fact.get("executed_tool", ""), fact.get("executed_sql", ""))
        for fact in anchors
        if fact.get("executed_tool")
    ]
    query_sqls = [sql for tool, sql in executed if tool == "sqlite_query" and sql]
    submitted = _normalize_sql(final.get("sql", ""))
    if declared is None:
        exact = any(_normalize_sql(sql) == submitted for sql in query_sqls)
        return {
            "declared": None,
            "matches": exact,
            "reason": "no_verified_call_line",
        }
    if declared < 1 or declared > len(executed):
        return {
            "declared": declared,
            "matches": False,
            "reason": f"index_out_of_range (1..{len(executed)})",
        }
    tool, sql = executed[declared - 1]
    if tool != "sqlite_query":
        return {
            "declared": declared,
            "matches": False,
            "reason": f"cited_call_is_{tool}",
        }
    match = _normalize_sql(sql) == submitted
    return {
        "declared": declared,
        "matches": match,
        "reason": "" if match else "cited_sql_differs_from_final",
    }


def _observation_kind(observation: str) -> str:
    if not observation:
        return ""
    if "error" in observation:
        return "error"
    if '"row_count":0' in observation:
        return "empty"
    return "rows"


# --------------------------------------------------------------------------- #
# Scripted policies (no model required)
# --------------------------------------------------------------------------- #

def scripted_repair_policy():
    """draft -> execution error -> repaired draft -> final answer (v3 XML calls)."""
    turns = iter([
        # 1: wrong table name on purpose -> execution_error
        """<scratch_pad>
Intent: count cancelled flights.
Expected output columns: cancelled_flights.
Candidate SQL: SELECT COUNT(*) AS cancelled_flights FROM airline WHERE CANCELLED = 1
</scratch_pad>
<tool_call>
<function=sqlite_query>
<parameter=sql>
SELECT COUNT(*) AS cancelled_flights FROM airline WHERE CANCELLED = 1
</parameter>
</function>
</tool_call>""",
        # 2: repaired after reading the "no such table: airline" observation
        """<scratch_pad>
Intent: the previous call failed with "no such table: airline"; the schema table is named Airlines. Fix the table name and re-run.
Expected output columns: cancelled_flights.
Candidate SQL: SELECT COUNT(*) AS cancelled_flights FROM Airlines WHERE CANCELLED = 1
</scratch_pad>
<tool_call>
<function=sqlite_query>
<parameter=sql>
SELECT COUNT(*) AS cancelled_flights FROM Airlines WHERE CANCELLED = 1
</parameter>
</function>
</tool_call>""",
        # 3: final answer
        """<scratch_pad>
Intent: finalize; verified a non-empty count.
Expected output columns: cancelled_flights.
Final SQL: SELECT COUNT(*) AS cancelled_flights FROM Airlines WHERE CANCELLED = 1
</scratch_pad>
<relevant_tables>
Airlines
</relevant_tables>
<relevant_columns>
Airlines.CANCELLED
</relevant_columns>
<final_answer>
SELECT COUNT(*) AS cancelled_flights FROM Airlines WHERE CANCELLED = 1
</final_answer>""",
    ])
    return lambda messages: next(turns)


def scripted_violation_policy(bad_turn):
    return lambda messages: bad_turn


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _all_tables(db_file):
    listing = st.sqlite_query(
        db_file,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    )
    return [row[0] for row in listing.get("rows", [])]


def _schema_from_output(schema_dir, db_id):
    """Load the model-visible M-schema .txt produced by data_tools/generate_mschema.py.

    Returns the text, or None when no matching file exists.

    Read verbatim, deliberately: the SFT/RLVR prompt builders insert this text
    into the ``db_details`` slot of the user template without normalising it, so
    a ``.strip()`` here made every screening/evaluation prompt differ from the
    training prompt by one trailing newline (and one token). This path is now
    only the *fallback* for questions without a stored SFT prefix; the v2 screen
    replays the stored prefix, so ``tools/check_prompt_consistency.py`` checks
    stored-prefix rendering against the rollout rendering rather than this path.
    """
    path = Path(schema_dir) / f"{db_id}.txt"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _schema_from_db(db_file, db_id):
    """Fallback: derive a CREATE-TABLE-style schema by peeking the live database."""
    lines = []
    for table in _all_tables(db_file):
        peek = st.sqlite_peek(db_file, table, limit=1)
        if "error" in peek:
            continue
        cols = [
            f'  "{name}" {stats["type"]}'
            for name, stats in peek.items()
            if name != "_scan"
        ]
        lines.append(f"CREATE TABLE \"{table}\" (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(lines) if lines else f"-- {db_id}: no tables found"


def _schema_for(db_file, db_id, schema_dir=None):
    """Return the schema text for the prompt.

    Prefers the data_tools M-schema .txt (``schema_dir/<db_id>.txt``) when a
    schema_dir is given; otherwise falls back to deriving it from the database.
    """
    if schema_dir:
        text = _schema_from_output(schema_dir, db_id)
        if text is None:
            raise FileNotFoundError(
                f"no M-schema for db_id {db_id!r} under {schema_dir}"
            )
        return text
    return _schema_from_db(db_file, db_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--db-id", default=None)
    ap.add_argument("--table", default="Airlines")
    ap.add_argument("--question", default="How many flights were cancelled?")
    ap.add_argument(
        "--schema-dir",
        default="/data/keenSQL/dataset/m_schema/output/bird_train",
        help="directory of data_tools M-schema .txt files (<schema-dir>/<db_id>.txt); "
        "pass an empty string to derive the schema from the database instead",
    )
    ap.add_argument("--model-url", default=None, help="OpenAI-compatible /v1 base url for L3")
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="treat a missing <scratch_pad> on the final-answer turn as a warning "
        "instead of a hard protocol error (tool-call turns still require it)",
    )
    args = ap.parse_args(argv)

    db_file = str(Path(args.db).resolve())
    print("=" * 72)
    print(f"L1  contract checks  ({PROMPT_FILE.name})")
    fails = check_contract()
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print(f"L1 result: {len(fails)} failure(s)")
    else:
        print("  all contract checks passed")

    print("=" * 72)
    print("L2  protocol + loop (scripted assistant, real tool dispatch)")
    schema_dir = args.schema_dir or None
    db_id = args.db_id or Path(db_file).stem
    schema = _schema_for(db_file, db_id, schema_dir=schema_dir)
    src = f"M-schema {Path(schema_dir) / (db_id + '.txt')}" if schema_dir else "derived from db"
    sys_prompt = load_system_prompt()
    print(f"  system prompt: {len(sys_prompt)} chars (source: {PROMPT_FILE.name}); "
          f"schema {len(schema)} chars (source: {src})")
    result = run_episode(
        scripted_repair_policy(), db_file, args.db_id, args.question, schema
    )
    ok = result["final"] is not None and not result["errors"]
    print(f"  tool calls used: {result['tool_calls']}/{MAX_TOOL_CALLS}")
    if result["final"]:
        print(f"  final tables   : {result['final']['tables']}")
        print(f"  final columns  : {result['final']['columns']}")
        print(f"  final SQL      : {result['final']['sql']}")
    print(f"L2 result: {'PASS' if ok else 'FAIL'}" + (f" errors={result['errors']}" if result['errors'] else ""))

    print("=" * 72)
    print("L2b protocol-violation detection")
    viol = {
        "no scratch_pad": '<tool_call><function=sqlite_query><parameter=sql>SELECT 1</parameter></function></tool_call>',
        "two calls": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_query><parameter=sql>SELECT 1</parameter></function></tool_call>\n<tool_call><function=sqlite_query><parameter=sql>SELECT 2</parameter></function></tool_call>',
        "call+final": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_query><parameter=sql>SELECT 1</parameter></function></tool_call>\n<final_answer>SELECT 1</final_answer>',
        "old name{} syntax": "<scratch_pad>x</scratch_pad>\n<tool_call>sqlite_query{sql: SELECT 1}</tool_call>",
        "two function blocks": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_query><parameter=sql>SELECT 1</parameter></function><function=sqlite_query><parameter=sql>SELECT 2</parameter></function></tool_call>',
        "unknown argument": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_query><parameter=sql>SELECT 1</parameter><parameter=db_id>x</parameter></function></tool_call>',
        "wrong argument type": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_peek><parameter=table>T</parameter><parameter=limit>ten</parameter></function></tool_call>',
        "missing required argument": '<scratch_pad>x</scratch_pad>\n<tool_call><function=sqlite_peek><parameter=limit>10</parameter></function></tool_call>',
        "legacy JSON body still accepted": '<scratch_pad>x</scratch_pad>\n<tool_call>{"name": "sqlite_query", "arguments": {"sql": "SELECT 1"}}</tool_call>',
    }
    for label, turn in viol.items():
        if label == "legacy JSON body still accepted":
            try:
                parsed = parse_assistant_turn(turn)
                print(f"  ok    {label}: parsed as {parsed['name']}")
            except ProtocolError as exc:
                print(f"  FAIL  {label}: rejected ({exc})")
            continue
        try:
            parse_assistant_turn(turn)
            print(f"  FAIL  {label}: not detected")
        except ProtocolError as exc:
            print(f"  ok    {label}: {exc}")

    print("=" * 72)
    if args.model_url:
        print(f"L3  real model via {args.model_url} (model={args.model})")
        print(f"    strict={'no (--lenient)' if args.lenient else 'yes'}")
        policy = _openai_policy(args.model_url, args.model, args.temperature, args.max_tokens)
        try:
            result = run_episode(
                policy, db_file, args.db_id, args.question, schema,
                require_final_scratch_pad=not args.lenient,
            )
            print(f"L3 result: final={'yes' if result['final'] else 'no'} "
                  f"tool_calls={result['tool_calls']} errors={result['errors']}")
            if result["final"]:
                print(f"  final SQL : {result['final']['sql']}")
        except Exception as exc:  # pragma: no cover
            print(f"L3 FAILED to reach model: {exc}")
    else:
        print("L3  skipped (pass --model-url to run against a served model)")
        print("    to serve Qwen3.5-9B:")
        print("      /data/keenSQL/.venv/bin/python -m vllm.entrypoints.openai.api_server \\")
        print("        --model /data/keenSQL/models/Qwen3.5-9B --port 8000 --max-model-len 131072")
    print("=" * 72)
    return 0 if not fails and ok else 1


def _openai_policy(base_url, model, temperature, max_tokens=2048, api_key="EMPTY",
                   timeout=180.0, max_retries=2, chat_template_kwargs=None):
    """Chat-completions policy with an explicit chat-template contract.

    ``chat_template_kwargs`` defaults to ``{"enable_thinking": False}``. Without
    it the stock Qwen3.5 template takes its ``else`` branch and opens
    ``<|im_start|>assistant\\n<think>\\n`` on *every* turn, i.e. the evaluated
    model sees an unclosed reasoning block that the SFT and RFT distributions
    never contain. That silently made screening/band and BIRD-dev numbers
    incomparable with training, so the flag is now always sent.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key,
                    timeout=timeout, max_retries=max_retries)
    if chat_template_kwargs is None:
        chat_template_kwargs = {"enable_thinking": False}

    def policy(messages):
        request = dict(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        if chat_template_kwargs:
            request["extra_body"] = {"chat_template_kwargs": dict(chat_template_kwargs)}
        try:
            resp = client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - older servers ignore/reject the kwarg
            text = str(exc)
            if "extra_body" not in request or (
                "extra_body" not in text and "chat_template_kwargs" not in text
            ):
                raise
            request.pop("extra_body", None)
            resp = client.chat.completions.create(**request)
        return resp.choices[0].message.content or ""

    return policy


if __name__ == "__main__":
    raise SystemExit(main())
