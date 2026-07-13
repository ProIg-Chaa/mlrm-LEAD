#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("analyze_soft_confidence_correctness.py")
SPEC = importlib.util.spec_from_file_location("confidence_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class ConfidenceAnalysisTests(unittest.TestCase):
    def test_auc_detects_confident_errors(self):
        self.assertEqual(analysis.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_auc_ties(self):
        self.assertEqual(analysis.auc([0.5, 0.5], [0, 1]), 0.5)

    def test_high_confidence_group(self):
        rows = [
            {"mean_raw_conf": 0.96, "correct": False, "failed_extraction": False},
            {"mean_raw_conf": 0.94, "correct": False, "failed_extraction": True},
            {"mean_raw_conf": 0.92, "correct": True, "failed_extraction": False},
            {"mean_raw_conf": 0.50, "correct": True, "failed_extraction": False},
        ]
        group = analysis.threshold_group(rows, "mean_raw_conf", 0.90)
        self.assertEqual(group["count"], 3)
        self.assertEqual(group["wrong"], 2)
        self.assertAlmostEqual(group["accuracy"], 1 / 3)
        self.assertEqual(group["semantic_wrong"], 1)
        self.assertEqual(group["semantic_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
