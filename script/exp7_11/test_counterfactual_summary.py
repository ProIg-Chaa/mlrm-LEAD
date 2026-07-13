#!/usr/bin/env python3
import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_counterfactual_replay.py")
SPEC = importlib.util.spec_from_file_location("replay_summary", MODULE_PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ReplaySummaryTests(unittest.TestCase):
    def test_continuation_distance(self):
        metrics = summary.continuation_distances([1, 2, 3, 4], [1, 2, 9, 4], 1)
        self.assertEqual(metrics["token_edit_distance_8"], 1)
        self.assertEqual(metrics["token_edit_distance_8_normalized"], 0.5)

    def test_identical_topk_has_zero_divergence(self):
        topk = [{"token_id": 1, "prob": 0.7}, {"token_id": 2, "prob": 0.2}]
        result = summary.approximate_divergence(topk, topk)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["js_divergence"], 0.0)
        self.assertAlmostEqual(result["kl_actual_to_branch"], 0.0)

    def test_different_topk_has_positive_divergence(self):
        left = [{"token_id": 1, "prob": 0.8}]
        right = [{"token_id": 2, "prob": 0.8}]
        result = summary.approximate_divergence(left, right)
        self.assertGreater(result["js_divergence"], 0.0)
        self.assertTrue(math.isfinite(result["kl_actual_to_branch"]))

    def test_mmvp_gold_normalization(self):
        self.assertEqual(summary.normalize_gold("(a) Open"), "A")
        self.assertEqual(summary.normalize_gold("B"), "B")


if __name__ == "__main__":
    unittest.main()
