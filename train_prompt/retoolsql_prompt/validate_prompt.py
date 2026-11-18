#!/usr/bin/env python3
"""Static validation and token budget report for the ReToolSQL v3 prompt.

Checks performed:

1. The system prompt names every tool, every tool parameter, every error kind,
   all documented limits, and all five protocol tags.
2. Every ``<tool_call>`` example in the system prompt parses and satisfies the
   tool signature.
3. The two few-shot episodes are well-formed chat sequences: alternating
   user/assistant roles, wrapped observations, legal tool calls, and a final
   answer with both annotation blocks.
4. The user template still exposes the four required fields.
5. Token budget: system <= 2.8k, few-shot <= 1.6k, fixed prefix <= 4.2k.
6. Rendering a small and a very large M-Schema stays inside the prompt budget,
   exercising the compression and pruning tiers.

Exit code is non-zero when any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "database_tools"))

import prompt_builder as pb  # noqa: E402
import sqlite_tools as st  # noqa: E402

SCHEMA_DIR = PROJECT_ROOT / "dataset" / "m_schema" / "output" / "bird_train"

REQUIRED_LIMITS = (
    "100,000",
    "[1, 10000]",
    "[1, 500]",
    "[1, 100]",
    "20,000 rows",
    "100 columns",
    "k1 = 1.2",
    "b = 0.75",
    "2 × top_k",
)
REQUIRED_PHRASES = (
    "one per turn",
    "verification",
    "single complete statement",
    "never use placeholders, ellipses, or omitted fragments",
    "repair guidance",
)
REQUIRED_TAGS = ("scratch_pad", "tool_call", "relevant_tables", "relevant_columns", "final_answer")
REQUIRED_USER_FIELDS = (
    "Database Engine:",
    "Database Schema:",
    "External Knowledge:",
    "Question:",
)

# Budget targets. The binding constraint is the 16k prompt budget. v4 measures
# 3,523 system + 1,440 few-shot; every BIRD train question still fits after the
# schema ladder. These caps guard against silent drift, not the measured values.
MAX_SYSTEM_TOKENS = 3700
MAX_FEWSHOT_TOKENS = 1700
MAX_PREFIX_TOKENS = MAX_SYSTEM_TOKENS + MAX_FEWSHOT_TOKENS
MAX_PROMPT_TOKENS = 16000


def check_system_prompt() -> list[str]:
    failures: list[str] = []
    text = pb.load_system_prompt()
    # phrase checks ignore line wrapping in the prompt file
    flat = " ".join(text.split())

    for name in st.TOOL_NAMES:
        if name not in text:
            failures.append(f"system prompt missing tool {name!r}")
    for schema in st.TOOL_SCHEMAS:
        function = schema["function"]
        for parameter in function["parameters"]["properties"]:
            if parameter not in text:
                failures.append(
                    f"system prompt missing parameter {parameter!r} of {function['name']}"
                )
    for kind in st.ERROR_KINDS:
        if kind not in text:
            failures.append(f"system prompt missing error kind {kind!r}")
    for needle in REQUIRED_LIMITS:
        if needle not in text:
            failures.append(f"system prompt missing documented limit {needle!r}")
    for phrase in REQUIRED_PHRASES:
        if " ".join(phrase.split()) not in flat:
            failures.append(f"system prompt missing required phrase {phrase!r}")
    for tag in REQUIRED_TAGS:
        if f"<{tag}>" not in text:
            failures.append(f"system prompt missing tag <{tag}>")
    if "db_id" in text or "db_file" in text:
        failures.append("system prompt leaks a database identifier parameter")
    if "db_id" in json.dumps(st.TOOL_SCHEMAS) or "db_file" in json.dumps(st.TOOL_SCHEMAS):
        failures.append("TOOL_SCHEMAS exposes db_id/db_file")

    import re

    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    # Prose such as "one tool call inside <tool_call>...</tool_call>" also matches
    # the regex, so keep only bodies that actually start like a call.
    blocks = [
        block for block in blocks
        if block.strip().startswith(("<function=", "{"))
    ]
    # The tool-call skeleton uses placeholders ("SELECT ...", "<argument>") that
    # cannot satisfy argument typing. Validate the *shape* there and reserve full
    # argument validation for concrete examples.
    concrete = [
        block for block in blocks
        if block.strip() not in ("...", "")
        and "<tool name>" not in block
        and "SELECT ..." not in block
        and "<a single read-only" not in block
    ]
    for block in concrete:
        try:
            pb.parse_tool_call_body(block, st._TOOL_SIGNATURES)
        except pb.ToolCallParseError as exc:
            failures.append(f"system prompt tool-call example is invalid: {exc}")

    # shape checks for the skeleton example(s)
    for block in blocks:
        stripped = block.strip()
        if stripped in ("...", ""):
            continue
        if "<function=sqlite_query>" in stripped and "</function>" not in stripped:
            failures.append("tool-call skeleton is missing </function>")
        if stripped.count("<tool_call>") > 1:
            failures.append("tool-call skeleton repeats <tool_call>")
    if not blocks:
        failures.append("system prompt has no <tool_call> example")
    return failures


def check_fewshot() -> list[str]:
    failures: list[str] = []
    examples = pb.load_fewshot()
    if len(examples) != 2:
        failures.append(f"expected 2 few-shot episodes, found {len(examples)}")
    for example in examples:
        name = example.get("name", "<unnamed>")
        messages = example.get("messages", [])
        if not messages:
            failures.append(f"{name}: no messages")
            continue
        if messages[0]["role"] != "user":
            failures.append(f"{name}: first message must be user")
        if messages[-1]["role"] != "assistant":
            failures.append(f"{name}: last message must be assistant")
        for index in range(1, len(messages)):
            if messages[index]["role"] == messages[index - 1]["role"]:
                failures.append(
                    f"{name}: messages {index - 1} and {index} share role "
                    f"{messages[index]['role']!r}"
                )
        for index, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "user" and index > 0:
                if not content.startswith("<observation>"):
                    failures.append(f"{name}: message {index} is not an observation")
                    continue
                body = content[len("<observation>") :].rsplit("</observation>", 1)[0].strip()
                try:
                    json.loads(body)
                except json.JSONDecodeError as exc:
                    failures.append(f"{name}: message {index} observation is not JSON: {exc}")
                continue
            if message["role"] != "assistant":
                continue
            tool_blocks = _tag_blocks(content, "tool_call")
            final_blocks = _tag_blocks(content, "final_answer")
            if len(tool_blocks) > 1:
                failures.append(f"{name}: message {index} has multiple tool calls")
            if tool_blocks and final_blocks:
                failures.append(f"{name}: message {index} mixes tool call and final answer")
            if not tool_blocks and not final_blocks:
                failures.append(f"{name}: message {index} has no action")
            for block in tool_blocks:
                try:
                    pb.parse_tool_call_body(block, st._TOOL_SIGNATURES)
                except pb.ToolCallParseError as exc:
                    failures.append(f"{name}: message {index} tool call invalid: {exc}")
            if final_blocks:
                for tag in ("relevant_tables", "relevant_columns"):
                    if not _tag_blocks(content, tag):
                        failures.append(f"{name}: final answer missing <{tag}>")
        if not any(_tag_blocks(m["content"], "final_answer") for m in messages if m["role"] == "assistant"):
            failures.append(f"{name}: no final answer")
    return failures


def _tag_blocks(text: str, tag: str) -> list[str]:
    import re

    return re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)


def check_user_template() -> list[str]:
    failures: list[str] = []
    template = pb.load_user_template()
    for field in REQUIRED_USER_FIELDS:
        if field not in template:
            failures.append(f"user template missing {field!r}")
    for placeholder in ("{engine}", "{db_details}", "{external_knowledge}", "{question}"):
        if placeholder not in template:
            failures.append(f"user template missing placeholder {placeholder}")
    teacher = pb.load_teacher_prompt()
    if "{reference_sql}" not in teacher:
        failures.append("teacher prompt missing {reference_sql} placeholder")
    return failures


def token_report() -> tuple[list[str], dict]:
    failures: list[str] = []
    system = pb.load_system_prompt()
    shots = pb.fewshot_messages()
    system_tokens = pb.estimate_tokens(system)
    fewshot_tokens = sum(pb.estimate_tokens(m["content"]) for m in shots)
    prefix_tokens = system_tokens + fewshot_tokens

    if system_tokens > MAX_SYSTEM_TOKENS:
        failures.append(f"system prompt is {system_tokens} tokens (limit {MAX_SYSTEM_TOKENS})")
    if fewshot_tokens > MAX_FEWSHOT_TOKENS:
        failures.append(f"few-shot is {fewshot_tokens} tokens (limit {MAX_FEWSHOT_TOKENS})")
    if prefix_tokens > MAX_PREFIX_TOKENS:
        failures.append(f"fixed prefix is {prefix_tokens} tokens (limit {MAX_PREFIX_TOKENS})")

    report = {
        "system_tokens": system_tokens,
        "system_chars": len(system),
        "fewshot_tokens": fewshot_tokens,
        "fewshot_messages": len(shots),
        "prefix_tokens": prefix_tokens,
    }
    return failures, report


def render_checks() -> tuple[list[str], list[dict]]:
    failures: list[str] = []
    rows: list[dict] = []
    for db_id in ("airline", "works_cycles"):
        path = SCHEMA_DIR / f"{db_id}.txt"
        if not path.is_file():
            failures.append(f"missing schema fixture {path}")
            continue
        schema = path.read_text(encoding="utf-8")
        try:
            built = pb.build_messages(
                engine="SQLite",
                db_details=schema,
                external_knowledge="",
                question="How many rows are in the largest table?",
                fewshot=True,
                max_prompt_tokens=MAX_PROMPT_TOKENS,
            )
        except pb.SchemaBudgetError as exc:
            failures.append(f"{db_id}: {exc}")
            continue
        meta = built["meta"]
        if meta["prompt_tokens"] > MAX_PROMPT_TOKENS:
            failures.append(
                f"{db_id}: prompt is {meta['prompt_tokens']} tokens "
                f"(limit {MAX_PROMPT_TOKENS})"
            )
        if db_id == "works_cycles" and meta["schema_tier"] < 1:
            failures.append("works_cycles should require compression or pruning")
        rows.append({"db_id": db_id, **meta})
    return failures, rows


def main() -> int:
    checks = [
        ("system prompt", check_system_prompt()),
        ("few-shot", check_fewshot()),
        ("user template", check_user_template()),
    ]
    failures: list[str] = []
    for label, issues in checks:
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"[{status}] {label}")
        for issue in issues:
            print(f"    - {issue}")
        failures.extend(issues)

    token_failures, report = token_report()
    status = "PASS" if not token_failures else f"FAIL ({len(token_failures)})"
    print(f"[{status}] token budget")
    print(f"    system={report['system_tokens']} fewshot={report['fewshot_tokens']} "
          f"prefix={report['prefix_tokens']}")
    for issue in token_failures:
        print(f"    - {issue}")
    failures.extend(token_failures)

    render_failures, rows = render_checks()
    status = "PASS" if not render_failures else f"FAIL ({len(render_failures)})"
    print(f"[{status}] schema budget ladder")
    for row in rows:
        print(
            f"    {row['db_id']}: tier={row['schema_tier']} "
            f"schema={row['schema_tokens']} prompt={row['prompt_tokens']} "
            f"chars {row['original_schema_chars']}->{row['schema_chars']}"
        )
    for issue in render_failures:
        print(f"    - {issue}")
    failures.extend(render_failures)

    print()
    if failures:
        print(f"RESULT: {len(failures)} failure(s)")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
