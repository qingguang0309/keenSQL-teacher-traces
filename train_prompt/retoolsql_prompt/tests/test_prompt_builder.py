import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train_prompt" / "retoolsql_prompt"))
sys.path.insert(0, str(ROOT / "database_tools"))

import prompt_builder as pb  # noqa: E402
import sqlite_tools as st  # noqa: E402

SCHEMA_DIR = ROOT / "dataset" / "m_schema" / "output" / "bird_train"

TOY_SCHEMA = """【DB_ID】 toy
【Dialect】 SQLite
【Database Summary】
  Tables: 2 | Rows: 10 | Columns: 4

# Table: orders | Rows: 8 | Columns: 2
  [Table Summary] PK=order_id; declared_FKs=1; empty=false
[
  order_id:INTEGER normalized=Integer, PK, Not Null
    [Observed] Role=identifier; Storage={integer:8}; Nulls=0; NonNull=8; Distinct=8; Examples=[1, 2]
  customer_id:INTEGER normalized=Integer, Nullable
    [Observed] Role=identifier; Storage={integer:8}; Nulls=0; NonNull=8; Distinct=3; Examples=[10, 11]
]

# Table: customers | Rows: 3 | Columns: 2
  [Table Summary] PK=customer_id; declared_FKs=0; empty=false
[
  customer_id:INTEGER normalized=Integer, PK, Not Null
    [Observed] Role=identifier; Storage={integer:3}; Nulls=0; NonNull=3; Distinct=3; Examples=[10, 11]
  name:TEXT normalized=Text, Not Null
    [Observed] Storage={text:3}; Nulls=0; NonNull=3; Distinct=3; Examples=['Ada', 'Grace']
]

[Foreign Keys]
orders.customer_id = customers.customer_id
  [Relation] active=true; declared_parent_unique=yes; child_non_null=8; matched=8; orphan=0; match_rate=100.00%
"""


class ParseTests(unittest.TestCase):
    def test_xml_sqlite_query(self):
        body = (
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter>"
            "<parameter=max_return_rows>5</parameter></function>"
        )
        name, args = pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)
        self.assertEqual(name, "sqlite_query")
        self.assertEqual(args, {"sql": "SELECT 1", "max_return_rows": 5})

    def test_xml_list_argument(self):
        body = (
            '<function=sqlite_peek><parameter=table>T</parameter>'
            '<parameter=columns>["a", "b"]</parameter></function>'
        )
        _, args = pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)
        self.assertEqual(args["columns"], ["a", "b"])

    def test_xml_string_column_fallback(self):
        body = (
            '<function=sqlite_peek><parameter=table>T</parameter>'
            "<parameter=columns>a,b</parameter></function>"
        )
        _, args = pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)
        self.assertEqual(args["columns"], "a,b")

    def test_legacy_json_still_parses(self):
        body = '{"name": "sqlite_query", "arguments": {"sql": "SELECT 1"}}'
        name, args = pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)
        self.assertEqual(name, "sqlite_query")
        self.assertEqual(args, {"sql": "SELECT 1"})

    def test_unknown_argument_rejected(self):
        body = (
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter>"
            "<parameter=db_id>x</parameter></function>"
        )
        with self.assertRaises(pb.ToolCallParseError):
            pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)

    def test_missing_required_rejected(self):
        body = "<function=sqlite_peek><parameter=limit>10</parameter></function>"
        with self.assertRaises(pb.ToolCallParseError):
            pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)

    def test_two_function_blocks_rejected(self):
        body = (
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter></function>"
            "<function=sqlite_query><parameter=sql>SELECT 2</parameter></function>"
        )
        with self.assertRaises(pb.ToolCallParseError):
            pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)

    def test_bad_integer_rejected(self):
        body = (
            "<function=sqlite_peek><parameter=table>T</parameter>"
            "<parameter=limit>ten</parameter></function>"
        )
        with self.assertRaises(pb.ToolCallParseError):
            pb.parse_tool_call_body(body, st._TOOL_SIGNATURES)


class CompressTests(unittest.TestCase):
    def test_drops_storage_and_averages(self):
        compact = pb.compress_schema(TOY_SCHEMA)
        self.assertNotIn("Storage=", compact)
        self.assertNotIn("NonNull=", compact)
        self.assertIn("Role=identifier", compact)
        self.assertIn("Examples=[1, 2]", compact)

    def test_keeps_structure(self):
        compact = pb.compress_schema(TOY_SCHEMA)
        for marker in ("# Table: orders", "[Table Summary]", "[Foreign Keys]", "match_rate=100.00%"):
            self.assertIn(marker, compact)

    def test_caps_long_literal(self):
        long_value = "x" * 500
        schema = (
            "【DB_ID】 t\n# Table: t | Rows: 1 | Columns: 1\n[\n"
            "  blob_col:TEXT normalized=Text, Nullable\n"
            f"    [Observed] Nulls=0; Distinct=1; Examples=['{long_value}']\n]\n"
        )
        compact = pb.compress_schema(schema)
        self.assertLess(len(compact), 1000)
        self.assertIn("…", compact)

    def test_keeps_literals_below_the_calibrated_cap(self):
        value = "x" * 250
        schema = (
            "【DB_ID】 t\n# Table: t | Rows: 1 | Columns: 1\n[\n"
            "  text_col:TEXT normalized=Text, Nullable\n"
            f"    [Observed] Nulls=0; Distinct=3; Examples=['{value}', "
            f"'{value}', '{value}']\n]\n"
        )
        compact = pb.compress_schema(schema)
        self.assertNotIn("…", compact)
        self.assertLessEqual(len(compact), 1000)

    def test_line_cap_never_exceeds_the_budget(self):
        description = "d" * (pb.MAX_LINE_CHARS + 100)
        schema = (
            "【DB_ID】 t\n# Table: t | Rows: 1 | Columns: 1\n[\n"
            "  text_col:TEXT normalized=Text, Nullable\n"
            f"    [Description] {description}\n]\n"
        )
        compact = pb.compress_schema(schema)
        longest = max(len(line) for line in compact.splitlines())
        self.assertLessEqual(longest, pb.MAX_LINE_CHARS)
        self.assertTrue(next(line for line in compact.splitlines()
                            if "[Description]" in line).endswith(" …"))


class PruneTests(unittest.TestCase):
    def test_keeps_relevant_table_and_fk_closure(self):
        schema = TOY_SCHEMA + (
            "\n# Table: noise | Rows: 2 | Columns: 1\n"
            "  [Table Summary] PK=id; declared_FKs=0; empty=false\n[\n"
            "  id:INTEGER normalized=Integer, PK, Not Null\n"
            "    [Observed] Role=identifier; Nulls=0; Distinct=2; Examples=[1, 2]\n]\n"
        )
        pruned = pb.prune_schema(schema, "How many orders per customer?", "")
        self.assertIn("# Table: orders", pruned)
        self.assertIn("# Table: customers", pruned)
        self.assertIn("Omitted tables: noise", pruned)
        self.assertIn("orders.customer_id = customers.customer_id", pruned)

    def test_small_schema_unchanged(self):
        self.assertEqual(pb.prune_schema(TOY_SCHEMA, "orders", ""), TOY_SCHEMA)


class BudgetTests(unittest.TestCase):
    def test_small_schema_uses_tier_zero(self):
        built = pb.build_messages(
            engine="SQLite",
            db_details=TOY_SCHEMA,
            external_knowledge="",
            question="How many orders?",
            fewshot=False,
        )
        self.assertEqual(built["meta"]["schema_tier"], 0)
        self.assertEqual([m["role"] for m in built["messages"]], ["system", "user"])

    def test_fewshot_added_after_system(self):
        built = pb.build_messages(
            engine="SQLite",
            db_details=TOY_SCHEMA,
            external_knowledge="",
            question="How many orders?",
            fewshot=True,
        )
        roles = [m["role"] for m in built["messages"]]
        self.assertEqual(roles[:2], ["system", "user"])
        self.assertEqual(roles[-1], "user")
        self.assertIn("<observation>", built["messages"][3]["content"])

    def test_evidence_reaches_user_message(self):
        built = pb.build_messages(
            engine="SQLite",
            db_details=TOY_SCHEMA,
            external_knowledge="orders means purchases",
            question="How many orders?",
            fewshot=False,
        )
        self.assertIn("orders means purchases", built["messages"][-1]["content"])

    def test_oversized_schema_is_compressed_or_pruned(self):
        path = SCHEMA_DIR / "works_cycles.txt"
        if not path.is_file():
            self.skipTest("works_cycles schema fixture not available")
        built = pb.build_messages(
            engine="SQLite",
            db_details=path.read_text(encoding="utf-8"),
            external_knowledge="",
            question="Which film has the longest title?",
            max_prompt_tokens=16000,
        )
        self.assertGreaterEqual(built["meta"]["schema_tier"], 1)
        self.assertLessEqual(built["meta"]["prompt_tokens"], 16000)

    def test_budget_error_when_prefix_alone_too_large(self):
        with self.assertRaises(pb.SchemaBudgetError):
            pb.build_messages(
                engine="SQLite",
                db_details=TOY_SCHEMA,
                external_knowledge="",
                question="q",
                max_prompt_tokens=1000,
            )

    def test_teacher_appendix_contains_reference_sql(self):
        built = pb.build_teacher_messages(
            engine="SQLite",
            db_details=TOY_SCHEMA,
            external_knowledge="",
            question="How many orders?",
            reference_sql="SELECT COUNT(*) FROM orders",
            fewshot=False,
        )
        self.assertIn("SELECT COUNT(*) FROM orders", built["messages"][-1]["content"])
        self.assertIn("TEACHER-ONLY", built["messages"][-1]["content"])
        self.assertTrue(built["meta"]["teacher"])


class FewshotTests(unittest.TestCase):
    def test_two_examples_alternate_and_end_with_final_answer(self):
        examples = pb.load_fewshot()
        self.assertEqual(len(examples), 2)
        for example in examples:
            messages = example["messages"]
            for index in range(1, len(messages)):
                self.assertNotEqual(messages[index]["role"], messages[index - 1]["role"])
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertIn("<final_answer>", messages[-1]["content"])

    def test_every_tool_call_in_fewshot_parses(self):
        for example in pb.load_fewshot():
            for message in example["messages"]:
                if message["role"] != "assistant":
                    continue
                for block in _blocks(message["content"], "tool_call"):
                    pb.parse_tool_call_body(block, st._TOOL_SIGNATURES)


def _blocks(text, tag):
    import re

    return re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)


if __name__ == "__main__":
    unittest.main()
