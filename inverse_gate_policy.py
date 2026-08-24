#!/usr/bin/env python3
"""Locked inverse visual gate shared by regression and new experiments."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple


HORIZONS = (4, 8)
BRANCHES = ("baseline", "action")
SWAPS = ("swap1", "swap2", "swap3")


def choice_vector(probe: Dict[str, Any]) -> List[float]:
    probs = probe["choice_probs"]
    values = [float(probs.get(choice, 0.0)) for choice in "ABCDE"]
    total = sum(values)
    return [value / total for value in values] if total > 0 else values


def support(true_probs: List[float], control_probs: List[float]) -> Tuple[float, str]:
    index = max(range(len(true_probs)), key=true_probs.__getitem__)
    eps = 1e-12
    score = math.log(true_probs[index] + eps) - math.log(control_probs[index] + eps)
    return score, "ABCDE"[index]


def derive_features(probes: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for horizon in HORIZONS:
        for branch in BRANCHES:
            true_probs = choice_vector(probes[f"h{horizon}_{branch}_true"])
            for control in ("mask",) + SWAPS:
                control_probs = choice_vector(
                    probes[f"h{horizon}_{branch}_{control}"]
                )
                score, choice = support(true_probs, control_probs)
                output[f"{branch}_{control}_support_h{horizon}"] = score
                output[f"{branch}_probe_choice_h{horizon}"] = choice
        for control in ("mask",) + SWAPS:
            output[f"{control}_gain_h{horizon}"] = (
                output[f"action_{control}_support_h{horizon}"]
                - output[f"baseline_{control}_support_h{horizon}"]
            )
    for control in ("mask",) + SWAPS:
        output[f"{control}_transient"] = (
            output[f"{control}_gain_h4"] - output[f"{control}_gain_h8"]
        )
    output["median_swap_transient"] = statistics.median(
        output[f"{control}_transient"] for control in SWAPS
    )
    output["reject_action"] = bool(
        output["median_swap_transient"] >= 0.0
        and output["mask_transient"] >= 0.0
    )
    return output


def legacy_feature_decision(
    primary: Dict[str, Any], alt1: Dict[str, Any], alt2: Dict[str, Any]
) -> Dict[str, Any]:
    swap_transients = [
        float(row["visual_gain_h4"]) - float(row["visual_gain_h8"])
        for row in (primary, alt1, alt2)
    ]
    mask_transient = (
        float(primary["mask_gain_h4"]) - float(primary["mask_gain_h8"])
    )
    median_swap = statistics.median(swap_transients)
    return {
        "swap_transients": swap_transients,
        "mask_transient": mask_transient,
        "median_swap_transient": median_swap,
        "reject_action": bool(median_swap >= 0.0 and mask_transient >= 0.0),
    }
