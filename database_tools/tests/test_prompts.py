import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import prompts as p  # noqa: E402
import sqlite_tools as st  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_prompt_is_the_v4_source(self):
        self.assertEqual(p.PROMPT_VERSION, "v4")
        self.assertEqual(p.SYSTEM_PROMPT_FILE.name, "system_prompt.txt")
        self.assertEqual(p.SYSTEM_PROMPT_FILE.parent.name, "v4")
        self.assertTrue(p.RETOOLSQL_SYSTEM_PROMPT.startswith("# ReToolSQL v4"))

    def test_system_prompt_names_every_tool_and_tag(self):
        for name in st.TOOL_NAMES:
            self.assertIn(name, p.RETOOLSQL_SYSTEM_PROMPT)
        for tag in p.PROTOCOL_TAGS:
            self.assertIn(f"<{tag}>", p.RETOOLSQL_SYSTEM_PROMPT)
        self.assertIn("one per turn", p.RETOOLSQL_SYSTEM_PROMPT)
        self.assertIn("verification", p.RETOOLSQL_SYSTEM_PROMPT.lower())

    def test_system_prompt_requires_complete_sql_and_error_repair(self):
        prompt = " ".join(p.RETOOLSQL_SYSTEM_PROMPT.split())
        self.assertIn("single complete statement", prompt)
        self.assertIn("never use placeholders, ellipses, or omitted fragments", prompt)
        self.assertIn("repair guidance", prompt)

    def test_system_prompt_teaches_the_mechanical_contracts(self):
        prompt = " ".join(p.RETOOLSQL_SYSTEM_PROMPT.split())
        # The two v4 additions that the harness checks mechanically.
        self.assertIn("Closure self-check before sending", prompt)
        self.assertIn("Verified call: <k>", prompt)
        self.assertIn("counting every `sqlite_query`, `sqlite_peek`, and", prompt)
        # Every tool signature must appear with its exact name.
        for signature in (
            "sqlite_query(sql, max_return_rows=200)",
            'sqlite_peek(table, columns=["..."], limit=5, where="...")',
            'bm25_search_sqlite(table, column, query, top_k=10, where="...")',
        ):
            self.assertIn(signature, prompt)

    def test_tool_prompt_section_is_extracted_from_the_system_prompt(self):
        section = p.RETOOLSQL_TOOL_PROMPT
        self.assertTrue(section.startswith("Tools take no database identifier"))
        for name in st.TOOL_NAMES:
            self.assertIn(f"`{name}`", section)
        self.assertNotIn("## 4. Turn protocol", section)

    def test_build_messages_has_system_and_four_field_user_message(self):
        messages = p.build_messages(
            engine="SQLite",
            db_details="CREATE TABLE t (x INTEGER);",
            external_knowledge="",
            question="How many rows?",
            fewshot=False,
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], p.RETOOLSQL_SYSTEM_PROMPT)
        content = messages[1]["content"]
        for label in (
            "Database Engine:",
            "Database Schema:",
            "External Knowledge:",
            "Question:",
        ):
            self.assertIn(label, content)
        self.assertIn("How many rows?", content)

    def test_build_messages_includes_the_two_fewshot_episodes_by_default(self):
        messages = p.build_messages(
            engine="SQLite",
            db_details="schema",
            external_knowledge="",
            question="q",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        expected = 1 + sum(len(ex["messages"]) for ex in p.FEWSHOT_EXAMPLES) + 1
        self.assertEqual(len(messages), expected)
        self.assertEqual(len(p.FEWSHOT_EXAMPLES), 2)
        # Each episode ends with a final answer that cites an executed call.
        for example in p.FEWSHOT_EXAMPLES:
            final = example["messages"][-1]["content"]
            self.assertIn("Verified call:", final)
            self.assertIn("<final_answer>", final)

    def test_fewshot_observations_use_the_harness_wrapper(self):
        for example in p.FEWSHOT_EXAMPLES:
            messages = example["messages"]
            self.assertEqual(messages[0]["role"], "user")
            self.assertNotIn("<observation>", messages[0]["content"])
            for message in messages[2:]:
                if message["role"] == "user":
                    self.assertTrue(message["content"].startswith("<observation>"))
                    self.assertTrue(message["content"].endswith("</observation>"))

    def test_build_messages_requires_string_fields(self):
        with self.assertRaises(TypeError):
            p.build_messages(
                engine="SQLite",
                db_details=None,
                external_knowledge="",
                question="q",
            )

    def test_describe_tools_covers_the_execution_schemas(self):
        described = p.describe_tools()
        self.assertEqual([entry["name"] for entry in described], list(st.TOOL_NAMES))
        for entry, schema in zip(described, st.TOOL_SCHEMAS):
            self.assertEqual(entry["description"], schema["function"]["description"])
            self.assertEqual(entry["parameters"], schema["function"]["parameters"])

    def test_tool_descriptions_match_the_v4_contract(self):
        descriptions = {
            schema["function"]["name"]: " ".join(schema["function"]["description"].split())
            for schema in st.TOOL_SCHEMAS
        }
        self.assertIn("exactly one read-only SELECT or WITH", descriptions["sqlite_query"])
        self.assertIn('"output_truncated"', descriptions["sqlite_query"])
        self.assertIn("first 20,000 rows", descriptions["sqlite_peek"])
        self.assertIn("concrete signal", descriptions["sqlite_peek"])
        self.assertIn("2 x top_k", descriptions["bm25_search_sqlite"])
        self.assertIn("short keyword", descriptions["bm25_search_sqlite"])

    def test_no_database_identifier_leaks_to_the_model(self):
        self.assertNotIn("db_id", p.RETOOLSQL_SYSTEM_PROMPT)
        self.assertNotIn("db_id", json.dumps(st.TOOL_SCHEMAS))
        self.assertNotIn("db_file", json.dumps(st.TOOL_SCHEMAS))
        self.assertNotIn("db_file", p.RETOOLSQL_SYSTEM_PROMPT)

    def test_text_tool_call_round_trip(self):
        rendered = p.text_tool_call("sqlite_query", {"sql": "SELECT 1", "max_return_rows": 5})
        self.assertTrue(rendered.startswith(f"<{p.TOOL_CALL_TAG}>"))
        self.assertTrue(rendered.endswith(f"</{p.TOOL_CALL_TAG}>"))
        self.assertIn("<function=sqlite_query>", rendered)
        self.assertIn("<parameter=sql>\nSELECT 1\n</parameter>", rendered)
        self.assertIn("<parameter=max_return_rows>\n5\n</parameter>", rendered)
        with self.assertRaises(ValueError):
            p.text_tool_call("nope", {})

    def test_text_tool_call_renders_lists_as_json_arrays(self):
        rendered = p.text_tool_call(
            "sqlite_peek", {"table": "T", "columns": ["a", "b"], "limit": 3}
        )
        self.assertIn('<parameter=columns>\n["a", "b"]\n</parameter>', rendered)

    def test_fallback_format_constant_is_a_qwen_tool_call(self):
        body = p.TEXT_TOOL_CALL_FORMAT
        self.assertIn("<tool_call>", body)
        self.assertIn("<function=sqlite_query>", body)
        self.assertIn("<parameter=sql>", body)
        self.assertNotIn('{"name"', body)

    def test_prompt_declares_every_tool_parameter(self):
        for schema in st.TOOL_SCHEMAS:
            function = schema["function"]
            for parameter in function["parameters"]["properties"]:
                self.assertIn(parameter, p.RETOOLSQL_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
