import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.sql_exec import grade_rows  # noqa: E402


class ExactGradingTests(unittest.TestCase):
    """The shared grader must be exact: no numeric tolerance, no row folding."""

    def test_identical_single_number(self):
        self.assertTrue(grade_rows([[2]], [[2]], grading_method="multiset"))

    def test_int_and_float_have_different_string_forms(self):
        # The reference BIRD/Tinker grader compares str(value), so 2 and 2.0
        # are not interchangeable: only exact textual equality counts.
        self.assertFalse(grade_rows([[2]], [[2.0]], grading_method="multiset"))
        self.assertFalse(grade_rows([[7]], [[7.0]], grading_method="multiset"))

    def test_close_but_different_integers_are_not_equal(self):
        # This pair used to pass through a 1% relative tolerance.
        self.assertFalse(grade_rows([[11243778]], [[11204735]], grading_method="multiset"))

    def test_half_unit_difference_is_not_equal(self):
        self.assertFalse(grade_rows([[245]], [[245.5]], grading_method="multiset"))

    def test_float_rounding_dust_is_not_equal(self):
        self.assertFalse(
            grade_rows(
                [[0.2494107955975562]],
                [[0.24941079559755625]],
                grading_method="multiset",
            )
        )

    def test_column_order_within_row_is_ignored(self):
        self.assertTrue(grade_rows([["a", "b"]], [["b", "a"]], grading_method="multiset"))

    def test_row_order_is_ignored_for_multiset(self):
        self.assertTrue(grade_rows([[1], [2]], [[2], [1]], grading_method="multiset"))

    def test_multiset_multiplicity_matters(self):
        self.assertFalse(grade_rows([[1], [1]], [[1]], grading_method="multiset"))

    def test_different_row_counts_are_not_equal(self):
        self.assertFalse(grade_rows([[1]], [[1], [2]], grading_method="multiset"))

    def test_one_percent_off_integer_is_rejected(self):
        # 100 vs 101 is exactly 1% apart; it must not be treated as a match.
        self.assertFalse(grade_rows([[100]], [[101]], grading_method="multiset"))
        self.assertFalse(grade_rows([[100]], [[100.9]], grading_method="multiset"))

    def test_list_preserves_row_order(self):
        self.assertFalse(grade_rows([[1], [2]], [[2], [1]], grading_method="list"))
        self.assertTrue(grade_rows([[1], [2]], [[1], [2]], grading_method="list"))

    def test_subset_directions(self):
        self.assertTrue(grade_rows([[1], [2]], [[1]], grading_method="subset"))
        self.assertFalse(grade_rows([[1]], [[1], [2]], grading_method="subset"))


if __name__ == "__main__":
    unittest.main()
