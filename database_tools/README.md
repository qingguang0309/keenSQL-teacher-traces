# database_tools — ReToolSQL verification-first SQLite tools

Read-only, deterministic database tools used as the action space of the
Text-to-SQL agent, implemented from ReToolSQL:

> *ReToolSQL: Agentic Reinforcement Learning for Robust Text-to-SQL*
> (arXiv:2608.27796v1, Section 3.3 "Action Space: A Verification-First Tool
> Suite", Appendix C "Tool Definitions" Table 7, Appendix A observation
> examples). Definitions retrieved 2026-10-15.

An open-source survey found no reusable implementation of this tool suite
(no official code release; no equivalent of `sqlite_peek` /
`bm25_search_sqlite` exists in other Text-to-SQL agent repositories), so the
three tools are implemented here from the paper contract. The module is
self-contained: it imports only the Python standard library, plus `sqlglot`
for static statement validation when available (a conservative fallback
keyword check is used otherwise).

## Files

| File | Purpose |
| --- | --- |
| `sqlite_tools.py` | Read-only sandbox, the three tools, OpenAI-format tool schemas (descriptions aligned with the v4 system prompt), `dispatch`, `SQLiteToolbox` |
| `prompts.py` | Loads the v4 system prompt / user template / few-shot set, exposes the tool section, and assembles the exact message list |
| `smoke.py` | CLI smoke run of all three tools against a real database |
| `demo_prompt_trajectory.py` | Prints the complete v4 prompt set (tool declarations, system prompt, few-shot, four-field user message) plus one worked multi-turn trajectory |
| `tests/test_sqlite_tools.py` | Tool, sandbox, contract, and dispatch tests |
| `tests/test_prompts.py` | Prompt / schema consistency tests |

## Tool contracts (paper Table 7 mapping)

All three tools are read-only and deterministic. The model-visible arguments
never contain a database identifier: the environment binds one SQLite file per
question (`db_file`). The Python functions additionally accept an optional
`db_id=` keyword that the environment can use to assert the binding matches; a
mismatch is reported as an `infrastructure_error`.

| Tool | Model-visible arguments | Limits | Returns |
| --- | --- | --- | --- |
| `sqlite_query` | `sql`, `max_return_rows=200` | SELECT/WITH only; 15M VM steps; 5s busy timeout; row cap | `{"columns": [...], "rows": [[...]], "row_count": N, "truncated": bool, "output_truncated": bool}` or the structured error |
| `sqlite_peek` | `table`, `columns` (list or `"a,b"`), `limit=5`, `where` | profile scan limited to 20,000 rows | per column: `type`, `nulls`, `distinct`, `min`, `max`, `top_values`, `samples`, `chars`, `prefixes`, plus a `_scan` meta key |
| `bm25_search_sqlite` | `table`, `column`, `query`, `top_k=10`, `where` | read-only; case-insensitive by default | up to `2 x top_k` entries: `{"value": ..., "bm25": score}` for BM25 hits (with `"regex": true` when also regex-matched) followed by regex/substring-only hits `{"value": ..., "regex": true}`; a trailing marker entry when the observation budget cuts the list |

Details:

- `sqlite_query` counts `row_count` as the number of returned rows;
  `truncated` is `true` when more rows were available, and
  `output_truncated` is `true` when the observation budget trimmed further
  rows. Cells are JSON-safe: `NULL` stays `null`, BLOBs become a `0x...`
  preview, non-finite floats become strings, and text is bounded to 200
  characters (ellipsis suffix).
- `sqlite_peek` profiles at most the first 20,000 rows (SQL `LIMIT 20000`).
  `_scan` reports `rows_scanned`, `limit`, `truncated`, `columns_truncated`,
  and `output_truncated`. `min`/`max` are numeric when every scanned value of
  the column is a number, otherwise lexical. `chars` counts `percent`,
  `date_like`, `numeric`, `alpha`, and `other` classes over the string values.
  `top_values`/`samples`/`prefixes` are capped by `limit`; prefixes are the
  first two characters of string values.
- `bm25_search_sqlite` searches the distinct non-NULL stored values with Okapi
  BM25 (`k1=1.2`, `b=0.75`, lowercased `[a-z0-9]+` tokens). The fallback is a
  case-insensitive regular expression; if the query does not compile as a
  regular expression it is matched literally (so `**` finds `a**`). Ties and
  regex-only hits are ordered by value length, then value, so results are
  deterministic. If the observation budget cuts the list, the last entry is a
  marker `{"output_truncated": true, "kept": N, "total": M}`.

## Error contract

Every failure is returned as (never raised from the public API):

```json
{"error": {"kind": "validation_error", "message": "..."}}
```

`kind` is one of `validation_error`, `execution_error`, `timeout`,
`step_limit_exceeded`, `infrastructure_error`. `dispatch` always returns this
payload as compact JSON text, so a training environment can place it verbatim
into the tool observation message.

## Sandbox

- Connections use `file:...?mode=ro&immutable=1` plus `PRAGMA query_only=ON`.
- A SQLite authorizer denies every write action (CREATE/INSERT/UPDATE/DELETE/
  DROP/ALTER/ATTACH/DETACH/VACUUM/REINDEX/...), denies all pragmas except the
  metadata whitelist (`table_info`, `table_xlist`, `index_*`,
  `foreign_key_list`, `database_list`), and denies `eval`, `readfile`,
  `writefile`, `load_extension`, `fts3_tokenizer`.
- Statements are statically validated before execution: exactly one statement,
  and the AST must be a read-only `SELECT`/`WITH` query (no `VALUES`,
  `EXPLAIN`, `PRAGMA`, or DML/DDL nodes anywhere).
- A progress handler enforces the paper's 15,000,000 VM-step limit and aborts
  runaway recursive queries with `step_limit_exceeded`.
- The busy timeout is 5 seconds (paper) and one statement per call.

## Implementation supplements (not in the paper)

Everything below is an engineering safety bound, not a paper requirement; all
are configurable through module constants or keyword arguments:

- **Observation budget.** Each tool result is bounded to
  `DEFAULT_MAX_OUTPUT_CHARS` (4000, `DATABASE_TOOLS_MAX_OUTPUT_CHARS`
  overrides; `max_output_chars=None` disables). This mirrors ReTool's practice
  of truncating every interpreter feedback so one call cannot flood the
  context; the per-cell bound alone cannot cap the total. Degradation is
  deterministic and only removes the least important content:
  - `sqlite_query`: keeps the longest prefix of rows that fits (binary search)
    and sets `row_count`, `truncated`, `output_truncated`;
  - `sqlite_peek`: trims `top_values`/`samples`/`prefixes` to 5, then 3, then
    1, then drops the `prefixes`/`samples` keys entirely (each column profile
    keeps its remaining key set), and only then drops trailing columns
    (`_scan.columns_truncated`); `_scan.output_truncated` marks every cut;
  - `bm25_search_sqlite`: keeps the leading entries and appends an
    `output_truncated` marker with `kept`/`total`.
  The default observation budget is calibrated on real BIRD databases: on
  `airline`, a 100-row/24-column `sqlite_query` shrinks from ~15.1 KB to
  ~3.9 KB (23 rows kept) and a 24-column `sqlite_peek` from ~8.2 KB to
  ~4.0 KB; a `bm25_search_sqlite` result usually stays below the budget.
- Wall-clock safety net per statement: `DEFAULT_WALL_TIMEOUT` (30 s,
  `DATABASE_TOOLS_SQL_TIMEOUT` overrides) reports `timeout`.
- `max_return_rows` is accepted in `1..10000`; `limit` in `1..500`;
  `top_k` in `1..100`.
- At most 100 columns are profiled per `sqlite_peek` call
  (`_scan.columns_truncated` flags an implicit truncation).
- `bm25_search_sqlite` scans at most 50,000 distinct candidate values, and the
  regex phase is bounded by the wall-clock budget. Patterns longer than 256
  characters, nested/alternation quantifiers (`(a+)+b`, `(SD|CA)+`), or more
  than four unbounded quantifiers are matched as literal text so a
  model-supplied pattern cannot stall a rollout. The regex phase is not a
  general regex sandbox; hazardous patterns degrade to literal matching.
- Single cell values are bounded to 200 characters in observations, and the
  regex phase scans at most the first 512 characters of each candidate value.

## Prompt and turn protocol (`prompts.py`)

The canonical prompt is `../retoolsql_prompt/v4/system_prompt.txt`; `prompts.py`
loads it rather than restating it, so there is exactly one maintained copy. The
module exposes:

| Name | Content |
| --- | --- |
| `RETOOLSQL_SYSTEM_PROMPT` | The full v4 system prompt (8 sections: role, schema reading, tools, turn protocol, decision policy, SQL rules, budget, hard constraints) |
| `RETOOLSQL_TOOL_PROMPT` | Just the `## 3. Tools` section, for callers that render the tool contract separately |
| `FEWSHOT_EXAMPLES` / `fewshot_messages()` | The two static v4 episodes from `v4/fewshot.json`, flattened for the chat template |
| `USER_MESSAGE_TEMPLATE` / `build_user_message()` | The four-field template (`Database Engine`, `Database Schema`, `External Knowledge`, `Question`) |
| `build_messages()` | The complete fixed prefix: system prompt → two few-shot episodes → user message |
| `text_tool_call()` | The Qwen XML rendering used in the few-shot and demo turns |

The tool declarations come from `sqlite_tools.TOOL_SCHEMAS`, and their
descriptions mirror v4 §3: call triggers, argument ranges, and the exact success
and error payloads. The prompt, the schemas, and the execution layer therefore
cannot drift apart.

Tool calls use the Qwen XML grammar:

```text
<tool_call>
<function=sqlite_query>
<parameter=sql>
SELECT ...
</parameter>
</function>
</tool_call>
```

v4 adds two mechanically checkable requirements that the earlier prompt lacked:

- **Closure self-check.** Before sending, `<function=X>` must have `</function>`
  and `<tool_call>` must have `</tool_call>`, with no tag repeated. A single
  missing `</function>` accounted for every observed API format failure.
- **`Verified call: <k>`.** The final scratch pad names the 1-based index of the
  executed `sqlite_query` call whose SQL is submitted, counting every
  `sqlite_query`, `sqlite_peek`, and `bm25_search_sqlite` call in order. The
  harness can reject unverified submissions mechanically instead of inferring
  them afterwards.

See `../runs/v4_evaluation.md` for the measured effects of each v4 change,
`../runs/v4_context_examples.md` for three complete worked contexts and the
version history, and `../retoolsql_prompt/v4/README_ABLATIONS.md` for the
ablation variants.

## Usage

```python
import sys
sys.path.insert(0, "/data1/keensql/database_tools")

from sqlite_tools import SQLiteToolbox, TOOL_SCHEMAS, dispatch
from prompts import RETOOLSQL_TOOL_PROMPT, build_messages

messages = build_messages(
    engine="SQLite",
    db_details=open("schema.sql").read(),
    external_knowledge="",
    question="How many schools opened after 2020?",
)  # [system, few-shot user/assistant x2, user]
# or render only the tool contract:
# system_content = RETOOLSQL_TOOL_PROMPT
# Pass TOOL_SCHEMAS to the chat template (tools=...), then for each model call:
observation = dispatch(db_file, tool_name, tool_arguments_as_dict_or_json, db_id=db_id)
# `observation` is the exact JSON text for the tool-role message.
```

Run the smoke check against a real database:

```bash
/data1/ReViSQL/.venv/bin/python /data1/keensql/database_tools/smoke.py \
  --db /data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/bird_train/download/train/databases/airline/airline.sqlite \
  --db-id airline
```

See the whole prompt set with one worked trajectory:

```bash
/data1/ReViSQL/.venv/bin/python /data1/keensql/database_tools/demo_prompt_trajectory.py
# --bird-evidence   use the BIRD evidence field as external knowledge
# --dump-json FILE  also write {tools, messages} for the chat template
```

Every tool observation this prints is the exact text `dispatch` returned against
the BIRD `airline` database, so the row counts, BM25 scores and error messages
are real; only the assistant scratch pads and tool calls are representative.

## Tests

```bash
/data1/ReViSQL/.venv/bin/python -m pytest /data1/keensql/database_tools/tests -q
# or: /data1/ReViSQL/.venv/bin/python -m unittest discover -s /data1/keensql/database_tools/tests -v
```

The suite covers statement validation, the sandbox authorizer, truncation and
serialization bounds, the VM-step and wall-clock budgets, profiler statistics
and the 20,000-row scan cap, BM25 ranking plus regex fallback, the dispatch
error contract, prompt/schema consistency, and an optional smoke test against
the BIRD `airline` database (skipped when the fixture is absent).

## Future training-environment integration

The module is intentionally independent of any training framework:

1. Bind one database per question: `toolbox = SQLiteToolbox(db_file, db_id)`.
2. Render the chat prefix with `tools=TOOL_SCHEMAS` (native tool calling) or
   with the textual `<tool_call>` fallback in the system prompt.
3. On each assistant turn, parse the single tool call, then call
   `toolbox.dispatch(name, arguments)`; append the returned JSON text as the
   tool observation message.
4. Treat an `"error"` key as a recoverable observation — the model is expected
   to repair its SQL from it, which is the behavior the paper trains.
