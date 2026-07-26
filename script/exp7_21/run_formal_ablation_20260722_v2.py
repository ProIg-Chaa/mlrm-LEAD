#!/usr/bin/env python3
"""Strict wrapper for the formal ablation runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_formal_ablation_20260722 as base  # noqa: E402


original_method_errors = base.method_config_errors
original_audit_candidate = base.audit_candidate


def method_config_errors(config: dict, method_key: str) -> list[str]:
    errors = original_method_errors(config, method_key)
    for key in (
        "lead_force_normal",
        "lead_disable_to_normal_transition",
        "lead_disable_step0_linebreak_mix",
        "lead_initial_transition_hard_boundary_only",
        "lead_early_visual_anchor",
        "lead_force_initial_transition_step1",
    ):
        if bool(config.get(key, False)):
            errors.append(key)
    if int(config.get("lead_initial_transition_delay_steps", 0) or 0) != 0:
        errors.append("lead_initial_transition_delay_steps")
    if method_key in {"full_lead", "initial_soft_only", "initial_transition"}:
        if float(config.get("lead_soft_quota_ratio", 0.0) or 0.0) != 0.0:
            errors.append("lead_soft_quota_ratio")
        if bool(config.get("lead_format_cooldown", False)):
            errors.append("lead_format_cooldown")
        if bool(config.get("lead_soft_veto_on_diffuse", False)):
            errors.append("lead_soft_veto_on_diffuse")
    return sorted(set(errors))


def audit_candidate(run_dir, model, dataset, method, targets):
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"run_dir": str(run_dir), "eligible": False, "reasons": [str(exc)]}
    reasons = base.common_config_errors(config, model)
    reasons.extend(method_config_errors(config, method))
    if reasons:
        return {
            "run_dir": str(run_dir),
            "eligible": False,
            "reasons": sorted(set(reasons)),
        }
    return original_audit_candidate(run_dir, model, dataset, method, targets)


base.method_config_errors = method_config_errors
base.audit_candidate = audit_candidate


if __name__ == "__main__":
    raise SystemExit(base.main())
