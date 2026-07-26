#!/usr/bin/env python3
"""Production entry point with strict config prefiltering."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_formal_ablation_20260722_v3 as strict  # noqa: E402


base = strict.base
raw_audit_candidate = strict.wrapped.original_audit_candidate


def audit_candidate(run_dir, model, dataset, method, targets):
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"run_dir": str(run_dir), "eligible": False, "reasons": [str(exc)]}
    reasons = base.common_config_errors(config, model)
    reasons.extend(base.method_config_errors(config, method))
    if reasons:
        return {
            "run_dir": str(run_dir),
            "eligible": False,
            "reasons": sorted(set(reasons)),
        }
    return raw_audit_candidate(run_dir, model, dataset, method, targets)


base.audit_candidate = audit_candidate


if __name__ == "__main__":
    raise SystemExit(base.main())
