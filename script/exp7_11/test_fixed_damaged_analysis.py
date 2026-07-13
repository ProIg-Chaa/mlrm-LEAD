#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_fixed_damaged_mechanisms.py")
SPEC = importlib.util.spec_from_file_location("analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class ExtractionTests(unittest.TestCase):
    def test_last_answer_wins(self):
        text = "Answer: A\nAfter checking, final answer is (C)."
        self.assertEqual(analysis.extract_prediction(text), "C")

    def test_option_text_fallback(self):
        self.assertEqual(analysis.extract_prediction("The result is blue", "(A) red\n(B) blue"), "B")

    def test_markdown_answer(self):
        self.assertEqual(analysis.extract_prediction("**Answer:** (D) leather"), "D")

    def test_parenthesized_options(self):
        self.assertEqual(analysis.parse_options("(A) red\n(B) blue"), {"A": "red", "B": "blue"})

    def test_answer_before_think_close(self):
        self.assertEqual(analysis.extract_prediction("<think>Therefore:\n\n(B) blue\n</think>"), "B")

    def test_failed_extraction(self):
        self.assertIsNone(analysis.extract_prediction("I cannot determine this."))

    def test_answer_reversal(self):
        self.assertEqual(analysis.explicit_answers("Answer: A. Final answer: B"), ["A", "B"])

    def test_mcnemar(self):
        self.assertAlmostEqual(analysis.mcnemar_exact(1, 1), 1.0)
        self.assertLess(analysis.mcnemar_exact(10, 0), 0.01)

    def test_groups_are_exclusive(self):
        expected = {
            (False, True): "fixed", (True, False): "damaged",
            (True, True): "both_correct", (False, False): "both_wrong",
        }
        for pair, group in expected.items():
            self.assertEqual(analysis.subgroup({"baseline_correct": pair[0], "method_correct": pair[1]}), group)


if __name__ == "__main__":
    unittest.main()
