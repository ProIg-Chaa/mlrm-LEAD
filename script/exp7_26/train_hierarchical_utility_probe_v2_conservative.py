#!/usr/bin/env python3
"""Conservative entry point for Probe V2.

This wrapper keeps the main implementation unchanged while pre-registering two
deployment constraints during calibration:
1. the actionability threshold cannot be below the calibration median;
2. selected policy coverage cannot exceed 50 percent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_hierarchical_utility_probe_v2 as v2  # noqa: E402


MAX_COVERAGE = 0.50


def conservative_thresholds(rows, predictions, rho):
    q, _, _, utility = predictions
    q_values = np.sort(q)
    u_values = np.sort(utility)
    q_indices = np.rint(np.linspace(0, len(q_values) - 1, 17)).astype(int)
    u_indices = np.rint(np.linspace(0, len(u_values) - 1, 33)).astype(int)
    median_q = float(np.median(q_values))
    q_candidates = sorted(
        {
            float(value)
            for value in q_values[q_indices].tolist()
            if value >= median_q
        }
        | {float(q_values[-1] + 1e-6)}
    )
    u_candidates = sorted(
        set(u_values[u_indices].tolist() + [float(u_values[-1] + 1e-6)])
    )
    best_q = q_candidates[-1]
    best_u = u_candidates[-1]
    best = v2.simulate(rows, predictions, best_q, best_u)
    best_key = (0.0, best["net"], -best["interventions"], best_q, best_u)
    for q_threshold in q_candidates:
        for utility_threshold in u_candidates:
            result = v2.simulate(
                rows, predictions, q_threshold, utility_threshold
            )
            if result["coverage"] > MAX_COVERAGE:
                continue
            objective = result["fixed"] - rho * result["damaged"]
            key = (
                objective,
                result["net"],
                -result["interventions"],
                q_threshold,
                utility_threshold,
            )
            if key > best_key:
                best_key = key
                best = result
                best_q = q_threshold
                best_u = utility_threshold
    best["selection_objective"] = best_key[0]
    best["max_calibration_coverage"] = MAX_COVERAGE
    best["actionability_threshold_floor"] = "calibration_median"
    return best_q, best_u, best


v2.choose_thresholds = conservative_thresholds


if __name__ == "__main__":
    v2.main()
