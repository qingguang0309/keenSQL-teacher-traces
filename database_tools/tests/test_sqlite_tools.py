import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import sqlite_tools as st  # noqa: E402

AIRLINE_DB = Path(
    "/data1/XiYan-SQL/XiYan-SQLTraining/data/data_warehouse/"
    "bird_train/download/train/databases/airline/airline.sqlite"
)

SCHEMA_SCRIPT = """
CREATE TABLE school (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    "Enrollment (K-12)" REAL,
    rate TEXT,
    opened TEXT,
    payload BLOB
);
INSERT INTO school (id, name, "Enrollment (K-12)", rate, opened, payload) VALUES
    (1, 'Alameda', 1200.0, '5%', '2023-09-01', x'0001ff'),
    (2, 'Alameda County', 800.5, '0.75', '2023/09/02', NULL),
    (3, 'Berkeley', NULL, '12%', '2024-01-15', NULL),
    (4, 'Albany', 340.0, '33.5%', '2022-08-30', NULL),
    (5, 'Oakland', 6800.0, 'NULL', '2021-12-31', NULL);

CREATE TABLE dup (v TEXT);
INSERT INTO dup (v) VALUES ('ab'), ('ab'), ('cd'), ('x');

CREATE TABLE cities (city TEXT);
INSERT INTO cities (city) VALUES
    ('Alameda'), ('Alameda County'), ('Albany'),
    ('Berkeley'), ('San Francisco'), ('San Jose');

CREATE TABLE odds (v TEXT);
INSERT INTO odds (v) VALUES ('a**'), ('b'), ('c');

CREATE TABLE risky (v TEXT);
INSERT INTO risky (v) VALUES ('(a+)+b'), ('aaaaab'), ('a*a*a*a*a*b');

CREATE TABLE xprefix (v TEXT);
INSERT INTO xprefix (v) VALUES
    ('x001'), ('x002'), ('x003'), ('x004'), ('x005'),
    ('x006'), ('x007'), ('x008'), ('x009'), ('x010'), ('other');

CREATE TABLE big (value TEXT, n INTEGER);
CREATE TABLE empty_table (a INTEGER, b TEXT);

CREATE TABLE dates (d TEXT);
INSERT INTO dates (d) VALUES
    ('2023-01-02'), ('01/02/2023'), ('2023/01/02'), ('1/2/23'),
    ('2023-01-02 10:30'), ('10:30:00'), ('n/a 5'), ('20230102');

CREATE TABLE wide (id INTEGER, note TEXT);
CREATE TABLE wide_words (w TEXT);
"""


def _create_database(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    connection.executescript(SCHEMA_SCRIPT)
    connection.executemany(
        "INSERT INTO big (value, n) VALUES (?, ?)",
        [(f"v{index:05d}", index) for index in range(st.PEEK_SCAN_LIMIT + 1_000)],
    )
    connection.executemany(
        "INSERT INTO wide (id, note) VALUES (?, ?)",
        [(index, f"row{index:03d} " + "n" * 180) for index in range(40)],
    )
    connection.executemany(
        "INSERT INTO wide_words (w) VALUES (?)",
        [(f"word {index:02d} " + "w" * 60,) for index in range(12)],
    )
    connection.commit()
    connection.close()


class ToolTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="database_tools_test_"))
        cls.db_dir = cls.tmpdir / "school"
        cls.db_dir.mkdir()
        cls.db = cls.db_dir / "school.sqlite"
        _create_database(cls.db)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


class TestSQLiteQuery(ToolTestCase):
    def test_select_returns_columns_rows_and_count(self):
        payload = st.sqlite_query(self.db, "SELECT id, name FROM school ORDER BY id")
        self.assertEqual(payload["columns"], ["id", "name"])
        self.assertEqual(payload["row_count"], 5)
        self.assertEqual(payload["rows"][0], [1, "Alameda"])
        self.assertFalse(payload["truncated"])

    def test_empty_result_shape(self):
        payload = st.sqlite_query(
            self.db, 'SELECT "Enrollment (K-12)" FROM school WHERE 1 = 0'
        )
        self.assertEqual(
            set(payload),
            {"columns", "rows", "row_count", "truncated", "output_truncated"},
        )
        self.assertEqual(payload["columns"], ["Enrollment (K-12)"])
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["row_count"], 0)
        self.assertFalse(payload["truncated"])
        self.assertFalse(payload["output_truncated"])

    def test_max_return_rows_truncation_and_boundary(self):
        truncated = st.sqlite_query(self.db, "SELECT id FROM school", max_return_rows=2)
        self.assertEqual(truncated["row_count"], 2)
        self.assertTrue(truncated["truncated"])
        self.assertFalse(truncated["output_truncated"])
        exact = st.sqlite_query(self.db, "SELECT id FROM school", max_return_rows=5)
        self.assertEqual(exact["row_count"], 5)
        self.assertFalse(exact["truncated"])
        self.assertFalse(exact["output_truncated"])

    def test_invalid_max_return_rows(self):
        for value in (0, -1, 10001, "10", True, 1.5):
            payload = st.sqlite_query(self.db, "SELECT 1", max_return_rows=value)
            self.assertEqual(payload["error"]["kind"], "validation_error", value)

    def test_write_and_unsafe_statements_are_rejected(self):
        statements = [
            "INSERT INTO school (id, name) VALUES (9, 'x')",
            "UPDATE school SET name = 'x'",
            "DELETE FROM school",
            "DROP TABLE school",
            "CREATE TABLE injected (x)",
            "ALTER TABLE school ADD COLUMN extra TEXT",
            "ATTACH DATABASE '/tmp/evil.db' AS evil",
            "PRAGMA journal_mode=DELETE",
            "VACUUM",
            "SELECT 1; SELECT 2",
            "EXPLAIN SELECT 1",
            "VALUES (1)",
        ]
        for sql in statements:
            payload = st.sqlite_query(self.db, sql)
            self.assertIn("error", payload, sql)
            self.assertEqual(payload["error"]["kind"], "validation_error", sql)

    def test_dangerous_functions_are_blocked_by_the_sandbox(self):
        payload = st.sqlite_query(self.db, "SELECT readfile('/etc/passwd')")
        self.assertIn("error", payload)

    def test_authorizer_blocks_non_metadata_pragmas(self):
        connection, _path = st._open_readonly(self.db)
        try:
            with self.assertRaises(sqlite3.Error):
                connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()

    def test_cte_and_union_are_allowed(self):
        payload = st.sqlite_query(
            self.db,
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t UNION ALL SELECT 2",
        )
        self.assertNotIn("error", payload)
        self.assertEqual(payload["rows"], [[1], [2]])

    def test_runtime_sql_errors_are_execution_errors(self):
        missing = st.sqlite_query(self.db, "SELECT * FROM missing_table")
        self.assertEqual(missing["error"]["kind"], "execution_error")
        bad_column = st.sqlite_query(
            self.db, "SELECT * FROM school WHERE missing_column = 1"
        )
        self.assertEqual(bad_column["error"]["kind"], "execution_error")

    def test_malformed_sql_is_caught_by_static_validation(self):
        payload = st.sqlite_query(self.db, "SELECT FROM")
        self.assertEqual(payload["error"]["kind"], "validation_error")

    def test_step_limit_stops_a_runaway_query(self):
        payload = st.sqlite_query(
            self.db,
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
            "SELECT sum(x) FROM c",
            step_limit=200_000,
        )
        self.assertEqual(payload["error"]["kind"], "step_limit_exceeded")

    def test_wall_timeout_stops_a_runaway_query(self):
        payload = st.sqlite_query(
            self.db,
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
            "SELECT sum(x) FROM c",
            timeout=0.0,
        )
        self.assertEqual(payload["error"]["kind"], "timeout")

    def test_cell_serialization(self):
        payload = st.sqlite_query(
            self.db, "SELECT payload, NULL AS missing_value, 1.5 AS ratio FROM school WHERE id = 1"
        )
        row = payload["rows"][0]
        self.assertTrue(row[0].startswith("0x0001FF"))
        self.assertIsNone(row[1])
        self.assertEqual(row[2], 1.5)

    def test_long_cells_are_bounded(self):
        payload = st.sqlite_query(
            self.db, "SELECT substr(hex(zeroblob(160)), 1, 320) AS big"
        )
        cell = payload["rows"][0][0]
        self.assertLessEqual(len(cell), st.CELL_MAX_CHARS)
        self.assertTrue(cell.endswith("\u2026"))

    def test_db_id_consistency_guard(self):
        ok = st.sqlite_query(self.db, "SELECT 1", db_id="school")
        self.assertNotIn("error", ok)
        bad = st.sqlite_query(self.db, "SELECT 1", db_id="other")
        self.assertEqual(bad["error"]["kind"], "infrastructure_error")

    def test_empty_sql_is_a_validation_error(self):
        for value in ("", "   ", None, 5):
            payload = st.sqlite_query(self.db, value)
            self.assertEqual(payload["error"]["kind"], "validation_error")


class TestOutputBudget(ToolTestCase):
    def test_query_budget_trims_tail_rows_and_flags(self):
        full = st.sqlite_query(
            self.db, "SELECT id, note FROM wide", max_return_rows=100, max_output_chars=None
        )
        self.assertEqual(len(full["rows"]), 40)
        self.assertFalse(full["truncated"])
        self.assertFalse(full["output_truncated"])
        budget = len(st.json_text(full)) // 3
        fitted = st.sqlite_query(
            self.db, "SELECT id, note FROM wide", max_output_chars=budget
        )
        self.assertTrue(fitted["truncated"])
        self.assertTrue(fitted["output_truncated"])
        self.assertLessEqual(len(st.json_text(fitted)), budget)
        kept = len(fitted["rows"])
        self.assertTrue(0 < kept < len(full["rows"]))
        self.assertEqual(fitted["rows"], full["rows"][:kept])
        self.assertEqual(fitted["row_count"], kept)
        candidate = {**fitted, "rows": full["rows"][: kept + 1], "row_count": kept + 1}
        self.assertGreater(len(st.json_text(candidate)), budget)

    def test_query_default_budget_applies_and_none_disables(self):
        unbounded = st.sqlite_query(
            self.db, "SELECT id, note FROM wide", max_output_chars=None
        )
        self.assertEqual(len(unbounded["rows"]), 40)
        if len(st.json_text(unbounded)) <= st.DEFAULT_MAX_OUTPUT_CHARS:
            self.skipTest("fixture is smaller than the default observation budget")
        defaulted = st.sqlite_query(self.db, "SELECT id, note FROM wide")
        self.assertTrue(defaulted["output_truncated"])
        self.assertLessEqual(
            len(st.json_text(defaulted)), st.DEFAULT_MAX_OUTPUT_CHARS
        )

    def test_output_truncation_is_distinct_from_the_row_cap(self):
        capped = st.sqlite_query(
            self.db, "SELECT id FROM school", max_return_rows=3, max_output_chars=None
        )
        self.assertTrue(capped["truncated"])
        self.assertFalse(capped["output_truncated"])
        budget_cut = st.sqlite_query(
            self.db, "SELECT id, note FROM wide", max_output_chars=400
        )
        self.assertTrue(budget_cut["output_truncated"])
        self.assertLessEqual(len(st.json_text(budget_cut)), 400)

    def test_output_budget_validation(self):
        calls = (
            lambda value: st.sqlite_query(self.db, "SELECT 1", max_output_chars=value),
            lambda value: st.sqlite_peek(self.db, "school", max_output_chars=value),
            lambda value: st.bm25_search_sqlite(
                self.db, "cities", "city", "x", max_output_chars=value
            ),
        )
        for value in (0, -1, True, "4000", 1.5):
            for call in calls:
                payload = call(value)
                self.assertEqual(payload["error"]["kind"], "validation_error", value)


class TestSQLitePeek(ToolTestCase):
    def test_numeric_profile(self):
        payload = st.sqlite_peek(self.db, "school", ["Enrollment (K-12)"], limit=5)
        stats = payload["Enrollment (K-12)"]
        self.assertEqual(stats["type"], "REAL")
        self.assertEqual(stats["nulls"], 1)
        self.assertEqual(stats["distinct"], 4)
        self.assertEqual(stats["min"], 340.0)
        self.assertEqual(stats["max"], 6800.0)
        self.assertIn("_scan", payload)
        self.assertEqual(payload["_scan"]["rows_scanned"], 5)
        self.assertFalse(payload["_scan"]["truncated"])
        self.assertFalse(payload["_scan"]["output_truncated"])

    def test_character_class_summary(self):
        payload = st.sqlite_peek(self.db, "school", ["rate", "opened"], limit=5)
        self.assertEqual(
            payload["rate"]["chars"],
            {"numeric": 1, "date_like": 0, "percent": 3, "alpha": 1, "other": 0},
        )
        self.assertEqual(payload["rate"]["min"], "0.75")
        self.assertEqual(payload["rate"]["max"], "NULL")
        self.assertEqual(payload["opened"]["chars"]["date_like"], 5)

    def test_mixed_date_formats_are_recognized(self):
        payload = st.sqlite_peek(self.db, "dates", ["d"], limit=5)
        stats = payload["d"]
        self.assertEqual(stats["chars"]["date_like"], 6)
        self.assertEqual(stats["chars"]["numeric"], 1)
        self.assertEqual(stats["chars"]["other"], 1)

    def test_top_values_samples_and_prefixes(self):
        payload = st.sqlite_peek(self.db, "dup", ["v"], limit=2)
        stats = payload["v"]
        self.assertEqual(
            stats["top_values"],
            [{"value": "ab", "count": 2}, {"value": "cd", "count": 1}],
        )
        self.assertEqual(stats["prefixes"], [{"prefix": "ab", "count": 2}, {"prefix": "cd", "count": 1}])
        self.assertEqual(stats["samples"], ["ab", "cd"])
        self.assertEqual(stats["distinct"], 3)

    def test_default_columns_include_every_column(self):
        payload = st.sqlite_peek(self.db, "school", limit=3)
        self.assertEqual(
            set(payload),
            {"id", "name", "Enrollment (K-12)", "rate", "opened", "payload", "_scan"},
        )

    def test_columns_string_form_and_order(self):
        payload = st.sqlite_peek(self.db, "school", "name, id", limit=3)
        self.assertEqual([key for key in payload if key != "_scan"], ["name", "id"])

    def test_where_filter(self):
        payload = st.sqlite_peek(self.db, "school", ["name"], limit=5, where="id > 3")
        self.assertEqual(payload["_scan"]["rows_scanned"], 2)
        self.assertEqual(payload["name"]["distinct"], 2)

    def test_scan_is_limited_to_the_peek_scan_cap(self):
        payload = st.sqlite_peek(self.db, "big", ["n"], limit=5)
        self.assertEqual(payload["_scan"]["rows_scanned"], st.PEEK_SCAN_LIMIT)
        self.assertTrue(payload["_scan"]["truncated"])
        self.assertEqual(payload["n"]["distinct"], st.PEEK_SCAN_LIMIT)
        self.assertEqual(payload["n"]["max"], st.PEEK_SCAN_LIMIT - 1)

    def test_empty_table_profile(self):
        payload = st.sqlite_peek(self.db, "empty_table", limit=5)
        self.assertEqual(payload["a"]["type"], "INTEGER")
        self.assertEqual(payload["a"]["nulls"], 0)
        self.assertEqual(payload["a"]["distinct"], 0)
        self.assertIsNone(payload["a"]["min"])
        self.assertEqual(payload["a"]["top_values"], [])
        self.assertEqual(payload["_scan"]["rows_scanned"], 0)

    def test_unknown_table_and_column(self):
        self.assertEqual(
            st.sqlite_peek(self.db, "nope")["error"]["kind"], "validation_error"
        )
        self.assertEqual(
            st.sqlite_peek(self.db, "school", ["nope"])["error"]["kind"],
            "validation_error",
        )

    def test_invalid_arguments(self):
        self.assertEqual(
            st.sqlite_peek(self.db, "school", limit=0)["error"]["kind"],
            "validation_error",
        )
        self.assertEqual(
            st.sqlite_peek(self.db, "school", limit=st.MAX_PEEK_LIMIT + 1)["error"]["kind"],
            "validation_error",
        )
        self.assertEqual(
            st.sqlite_peek(self.db, "school", "a,,b")["error"]["kind"], "validation_error"
        )
        self.assertEqual(
            st.sqlite_peek(self.db, "school", where="")["error"]["kind"],
            "validation_error",
        )
        self.assertEqual(
            st.sqlite_peek(self.db, "school", where="1=1; DROP TABLE school")["error"]["kind"],
            "validation_error",
        )


class TestPeekOutputBudget(ToolTestCase):
    @staticmethod
    def _rung_lengths(payload):
        lengths = {
            keep: len(st.json_text(st._trim_peek_lists(payload, keep)))
            for keep in (5, 3, 1, 0)
        }
        dropped = st._drop_peek_detail_keys(st._trim_peek_lists(payload, 0))
        lengths["drop"] = len(st.json_text(dropped))
        return lengths

    def test_peek_budget_trims_lists_at_the_first_fitting_rung(self):
        base = st.sqlite_peek(
            self.db, "big", ["value", "n"], limit=10, max_output_chars=None
        )
        lengths = self._rung_lengths(base)
        self.assertGreater(lengths[5], lengths[3] + 1)
        budget = lengths[3]
        fitted = st.sqlite_peek(
            self.db, "big", ["value", "n"], limit=10, max_output_chars=budget
        )
        self.assertTrue(fitted["_scan"]["output_truncated"])
        self.assertFalse(fitted["_scan"]["columns_truncated"])
        self.assertEqual(set(fitted), set(base))
        self.assertLessEqual(len(st.json_text(fitted)), budget)
        for name in ("value", "n"):
            for key in ("top_values", "samples", "prefixes"):
                self.assertLessEqual(len(fitted[name][key]), 3)
            self.assertEqual(len(fitted[name]["top_values"]), 3)
            self.assertEqual(fitted[name]["distinct"], base[name]["distinct"])
            self.assertEqual(fitted[name]["min"], base[name]["min"])

    def test_peek_budget_drops_detail_keys_then_columns(self):
        base = st.sqlite_peek(self.db, "school", max_output_chars=None)
        lengths = self._rung_lengths(base)
        key_trimmed = st.sqlite_peek(
            self.db, "school", max_output_chars=lengths["drop"]
        )
        for name, stats in key_trimmed.items():
            if name == "_scan":
                continue
            self.assertNotIn("samples", stats)
            self.assertNotIn("prefixes", stats)
            self.assertEqual(stats["top_values"], [])
        self.assertFalse(key_trimmed["_scan"]["columns_truncated"])
        self.assertTrue(key_trimmed["_scan"]["output_truncated"])
        self.assertLessEqual(len(st.json_text(key_trimmed)), lengths["drop"])

        budget = lengths["drop"] // 2
        column_trimmed = st.sqlite_peek(self.db, "school", max_output_chars=budget)
        names = [name for name in column_trimmed if name != "_scan"]
        self.assertGreaterEqual(len(names), 1)
        self.assertLess(len(names), len([name for name in base if name != "_scan"]))
        self.assertTrue(column_trimmed["_scan"]["columns_truncated"])
        self.assertTrue(column_trimmed["_scan"]["output_truncated"])
        self.assertLessEqual(len(st.json_text(column_trimmed)), budget)


class TestBM25Search(ToolTestCase):
    def test_exact_match_ranks_first(self):
        results = st.bm25_search_sqlite(self.db, "cities", "city", "Alameda", top_k=2)
        self.assertEqual([entry["value"] for entry in results], ["Alameda", "Alameda County"])
        for entry in results:
            self.assertIsInstance(entry["bm25"], float)
            self.assertEqual(entry["bm25"], round(entry["bm25"], 3))
            self.assertTrue(entry.get("regex"))

    def test_case_insensitive(self):
        results = st.bm25_search_sqlite(self.db, "cities", "city", "alameda", top_k=1)
        self.assertEqual(results[0]["value"], "Alameda")

    def test_regex_fallback_orders_short_values_first(self):
        anchored = st.bm25_search_sqlite(self.db, "cities", "city", "^Ala", top_k=5)
        self.assertEqual(
            [entry["value"] for entry in anchored],
            ["Alameda", "Alameda County"],
        )
        for entry in anchored:
            self.assertTrue(entry["regex"])
            self.assertNotIn("bm25", entry)
        wider = st.bm25_search_sqlite(self.db, "cities", "city", "^Al", top_k=5)
        self.assertEqual(
            [entry["value"] for entry in wider],
            ["Albany", "Alameda", "Alameda County"],
        )

    def test_invalid_regex_is_treated_as_literal(self):
        results = st.bm25_search_sqlite(self.db, "odds", "v", "**")
        self.assertEqual(results, [{"value": "a**", "regex": True}])
        literal = st.bm25_search_sqlite(self.db, "odds", "v", "a**")
        self.assertEqual(literal[0]["value"], "a**")
        self.assertTrue(literal[0]["regex"])

    def test_backtracking_hazards_fall_back_to_literal_matching(self):
        results = st.bm25_search_sqlite(self.db, "risky", "v", "(a+)+b")
        values = [entry["value"] for entry in results]
        self.assertIn("(a+)+b", values)
        self.assertNotIn("aaaaab", values)
        hammered = st.bm25_search_sqlite(self.db, "risky", "v", "a*a*a*a*a*b")
        self.assertIn("a*a*a*a*a*b", [entry["value"] for entry in hammered])
        self.assertNotIn("aaaaab", [entry["value"] for entry in hammered])
        self.assertFalse(st._regex_is_safe("(a+)+b"))
        self.assertFalse(st._regex_is_safe("a*a*a*a*a*b"))
        self.assertFalse(st._regex_is_safe("(SD|CA)+"))
        self.assertFalse(st._regex_is_safe("x" * 257))
        self.assertTrue(st._regex_is_safe("^Ala"))
        self.assertTrue(st._regex_is_safe("San.*"))
        self.assertTrue(st._regex_is_safe(r"^\d{4}-\d{2}-\d{2}$"))

    def test_regex_only_hits_respect_top_k(self):
        results = st.bm25_search_sqlite(self.db, "xprefix", "v", "^x", top_k=4)
        self.assertEqual(len(results), 4)
        for entry in results:
            self.assertTrue(entry["regex"])

    def test_where_filter(self):
        results = st.bm25_search_sqlite(
            self.db, "cities", "city", "San", where="city LIKE 'San%'"
        )
        self.assertEqual(
            sorted(entry["value"] for entry in results),
            ["San Francisco", "San Jose"],
        )

    def test_no_match_returns_empty_list(self):
        self.assertEqual(st.bm25_search_sqlite(self.db, "cities", "city", "zzzz"), [])

    def test_unknown_table_and_column(self):
        self.assertEqual(
            st.bm25_search_sqlite(self.db, "nope", "v", "x")["error"]["kind"],
            "validation_error",
        )
        self.assertEqual(
            st.bm25_search_sqlite(self.db, "cities", "nope", "x")["error"]["kind"],
            "validation_error",
        )

    def test_invalid_arguments(self):
        self.assertEqual(
            st.bm25_search_sqlite(self.db, "cities", "city", "x", top_k=0)["error"]["kind"],
            "validation_error",
        )
        self.assertEqual(
            st.bm25_search_sqlite(self.db, "cities", "city", "")["error"]["kind"],
            "validation_error",
        )

    def test_output_budget_keeps_leading_entries_and_appends_marker(self):
        full = st.bm25_search_sqlite(
            self.db, "wide_words", "w", "word", top_k=12, max_output_chars=None
        )
        self.assertEqual(len(full), 12)
        budget = len(st.json_text(full)) // 3
        fitted = st.bm25_search_sqlite(
            self.db, "wide_words", "w", "word", top_k=12, max_output_chars=budget
        )
        marker = fitted[-1]
        self.assertTrue(marker["output_truncated"])
        self.assertEqual(marker["total"], 12)
        kept = marker["kept"]
        self.assertEqual(len(fitted), kept + 1)
        self.assertTrue(0 < kept < 12)
        self.assertEqual(fitted[:kept], full[:kept])
        self.assertTrue(all("value" in entry for entry in fitted[:-1]))
        self.assertLessEqual(len(st.json_text(fitted)), budget)
        candidate = full[: kept + 1] + [
            {"output_truncated": True, "kept": kept + 1, "total": 12}
        ]
        self.assertGreater(len(st.json_text(candidate)), budget)


class TestDispatch(ToolTestCase):
    def test_every_tool_is_callable_through_dispatch(self):
        query = json.loads(
            st.dispatch(self.db, "sqlite_query", {"sql": "SELECT 1 AS one"})
        )
        self.assertEqual(query["rows"], [[1]])
        peek = json.loads(st.dispatch(self.db, "sqlite_peek", {"table": "cities"}))
        self.assertIn("city", peek)
        search = json.loads(
            st.dispatch(
                self.db,
                "bm25_search_sqlite",
                {"table": "cities", "column": "city", "query": "Alameda", "top_k": 1},
            )
        )
        self.assertEqual(search[0]["value"], "Alameda")

    def test_dispatch_matches_the_direct_payload_text(self):
        arguments = {"sql": "SELECT id, name FROM school ORDER BY id", "max_return_rows": 3}
        self.assertEqual(
            st.dispatch(self.db, "sqlite_query", arguments),
            st.json_text(st.sqlite_query(self.db, arguments["sql"], 3)),
        )

    def test_arguments_may_arrive_as_a_json_string(self):
        payload = json.loads(
            st.dispatch(self.db, "sqlite_query", '{"sql": "SELECT 1 AS one"}')
        )
        self.assertEqual(payload["rows"], [[1]])

    def test_unknown_tool_and_bad_arguments(self):
        cases = [
            ("nope", {}),
            ("sqlite_query", {}),
            ("sqlite_query", {"sql": "SELECT 1", "foo": 1}),
            ("sqlite_query", {"sql": 1}),
            ("sqlite_peek", {"table": "cities", "limit": True}),
            ("sqlite_peek", {"table": "cities", "columns": [1, 2]}),
            ("bm25_search_sqlite", {"table": "cities", "column": "city"}),
        ]
        for name, arguments in cases:
            payload = json.loads(st.dispatch(self.db, name, arguments))
            self.assertEqual(payload["error"]["kind"], "validation_error", (name, arguments))
            self.assertTrue(payload["error"]["message"])

    def test_schemas_match_the_paper_signatures(self):
        self.assertEqual(
            [schema["function"]["name"] for schema in st.TOOL_SCHEMAS],
            list(st.TOOL_NAMES),
        )
        expected = {
            "sqlite_query": {"sql", "max_return_rows"},
            "sqlite_peek": {"table", "columns", "limit", "where"},
            "bm25_search_sqlite": {"table", "column", "query", "top_k", "where"},
        }
        for schema in st.TOOL_SCHEMAS:
            function = schema["function"]
            name = function["name"]
            self.assertEqual(schema["type"], "function")
            self.assertFalse(function["parameters"]["additionalProperties"])
            self.assertEqual(set(function["parameters"]["properties"]), expected[name])
            self.assertNotIn("db_id", function["parameters"]["properties"])

    def test_toolbox_round_trip(self):
        toolbox = st.SQLiteToolbox(self.db, "school")
        self.assertEqual(toolbox.query("SELECT 1 AS one")["rows"], [[1]])
        self.assertIn("city", toolbox.peek("cities"))
        self.assertEqual(
            json.loads(
                toolbox.dispatch(
                    "bm25_search_sqlite",
                    {"table": "cities", "column": "city", "query": "Berkeley"},
                )
            )[0]["value"],
            "Berkeley",
        )
        with self.assertRaises(st.ToolInfrastructureError):
            st.SQLiteToolbox(self.db, "wrong")

    def test_error_payload_contract(self):
        payload = json.loads(st.dispatch(self.db, "nope", {}))
        self.assertEqual(set(payload), {"error"})
        self.assertEqual(set(payload["error"]), {"kind", "message"})
        self.assertIn(payload["error"]["kind"], st.ERROR_KINDS)


class TestAirlineSmoke(unittest.TestCase):
    """Optional end-to-end check against a real BIRD database."""

    @unittest.skipUnless(AIRLINE_DB.is_file(), "airline.sqlite fixture is missing")
    def test_airline_query_peek_and_search(self):
        query = st.sqlite_query(AIRLINE_DB, "SELECT COUNT(*) AS n FROM airlines")
        self.assertNotIn("error", query)
        self.assertEqual(query["row_count"], 1)
        peek = st.sqlite_peek(AIRLINE_DB, "airlines", limit=3)
        self.assertNotIn("error", peek)
        self.assertIn("_scan", peek)
        self.assertLessEqual(len(st.json_text(peek)), st.DEFAULT_MAX_OUTPUT_CHARS)
        column = next(name for name in peek if name != "_scan")
        probe = st.sqlite_query(
            AIRLINE_DB,
            f'SELECT "{column}" FROM airlines WHERE "{column}" IS NOT NULL LIMIT 1',
        )
        self.assertNotIn("error", probe)
        self.assertTrue(probe["rows"], "airlines has no non-NULL values")
        sample = str(probe["rows"][0][0])
        results = st.bm25_search_sqlite(AIRLINE_DB, "airlines", column, sample)
        self.assertNotIn("error", results)
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
