#!/usr/bin/env python3
"""Final strict entry point for the TALR formal ablation."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_formal_ablation_20260722_v2 as wrapped  # noqa: E402


base = wrapped.base
previous_method_errors = base.method_config_errors


def method_config_errors(config: dict, method_key: str) -> list[str]:
    errors = previous_method_errors(config, method_key)
    checks = {
        "lead_transition_dynamic_entropy_window": int(
            config.get("lead_transition_dynamic_entropy_window", 0) or 0
        ) == 0,
        "lead_transition_semantic_adaptive": not bool(
            config.get("lead_transition_semantic_adaptive", False)
        ),
        "lead_transition_norm_match": not bool(
            config.get("lead_transition_norm_match", False)
        ),
        "lead_transition_anchor": str(
            config.get("lead_transition_anchor", "end_thinking")
        ) == "end_thinking",
        "lead_transition_source": str(
            config.get("lead_transition_source", "soft")
        ) == "soft",
        "lead_transition_beta0": abs(
            float(config.get("lead_transition_beta0", 0.7)) - 0.7
        ) <= 1e-9,
    }
    errors.extend(key for key, valid in checks.items() if not valid)
    return sorted(set(errors))


base.method_config_errors = method_config_errors


if __name__ == "__main__":
    raise SystemExit(base.main())
