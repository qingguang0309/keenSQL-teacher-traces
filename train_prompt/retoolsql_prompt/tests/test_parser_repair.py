import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retoolsql_prompt"))
sys.path.insert(0, str(ROOT / "database_tools"))

import prompt_builder as pb  # noqa: E402
import sqlite_tools as st  # noqa: E402

SIG = st._TOOL_SIGNATURES


class RepairTests(unittest.TestCase):
    def parse(self, body, **kwargs):
        repairs = []
        name, args = pb.parse_tool_call_body(body, SIG, repairs=repairs, **kwargs)
        return name, args, repairs

    def test_complete_call_needs_no_repair(self):
        name, args, repairs = self.parse(
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter></function>"
        )
        self.assertEqual(name, "sqlite_query")
        self.assertEqual(args, {"sql": "SELECT 1"})
        self.assertEqual(repairs, [])

    def test_missing_function_closer_is_repaired(self):
        name, args, repairs = self.parse(
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter>"
        )
        self.assertEqual(name, "sqlite_query")
        self.assertEqual(args, {"sql": "SELECT 1"})
        self.assertEqual(repairs, ["inserted </function>"])

    def test_repair_works_for_multi_parameter_tools(self):
        name, args, repairs = self.parse(
            "<function=sqlite_peek><parameter=table>T</parameter>"
            '<parameter=columns>["a", "b"]</parameter><parameter=limit>5</parameter>'
        )
        self.assertEqual(name, "sqlite_peek")
        self.assertEqual(args["columns"], ["a", "b"])
        self.assertEqual(args["limit"], 5)
        self.assertEqual(repairs, ["inserted </function>"])

    def test_duplicate_tool_call_closer_is_not_repaired(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse(
                "<function=sqlite_query><parameter=sql>SELECT 1</parameter></tool_call>"
            )

    def test_unclosed_parameter_is_not_repaired(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse("<function=sqlite_query><parameter=sql>SELECT 1")

    def test_two_function_blocks_are_not_repaired(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse(
                "<function=sqlite_query><parameter=sql>A</parameter></function>"
                "<function=sqlite_query><parameter=sql>B</parameter></function>"
            )

    def test_unknown_tool_is_not_repaired(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse("<function=not_a_tool><parameter=sql>SELECT 1</parameter>")

    def test_unknown_argument_is_not_repaired(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse(
                "<function=sqlite_query><parameter=sql>SELECT 1</parameter>"
                "<parameter=db_id>x</parameter>"
            )

    def test_repair_can_be_disabled(self):
        with self.assertRaises(pb.ToolCallParseError):
            self.parse(
                "<function=sqlite_query><parameter=sql>SELECT 1</parameter>",
                repair_unclosed_function=False,
            )

    def test_extra_text_after_closer_is_flagged_not_silently_dropped(self):
        name, args, repairs = self.parse(
            "<function=sqlite_query><parameter=sql>SELECT 1</parameter></function>"
            " trailing noise"
        )
        self.assertEqual(name, "sqlite_query")
        self.assertIn("trailing_text_ignored", repairs)


if __name__ == "__main__":
    unittest.main()
