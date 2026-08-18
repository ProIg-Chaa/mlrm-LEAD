#!/usr/bin/env python3
"""Evaluate a frozen utility probe on an external matched Atlas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir))
    import train_intervention_utility_probe as probe  # noqa: PLC0415

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    raw_rows = probe.read_jsonl(args.atlas)
    rows, feature_names, stats = probe.prepare_rows(raw_rows)
    expected_names = list(artifact["feature_names"])
    if feature_names != expected_names:
        missing = sorted(set(expected_names) - set(feature_names))
        extra = sorted(set(feature_names) - set(expected_names))
        raise RuntimeError(
            f"Feature mismatch: missing={missing}, extra={extra}"
        )

    model = probe.FittedHeads(
        fix=probe.BinaryProbe(len(feature_names), artifact["model_kind"]),
        damage=probe.BinaryProbe(len(feature_names), artifact["model_kind"]),
        mean=np.asarray(artifact["mean"], dtype=np.float32),
        std=np.asarray(artifact["std"], dtype=np.float32),
    )
    model.fix.load_state_dict(artifact["fix_state_dict"])
    model.damage.load_state_dict(artifact["damage_state_dict"])
    model.fix.eval()
    model.damage.eval()
    p_fix, p_damage, scores = probe.predict_scores(
        model, rows, float(artifact["rho"])
    )
    policy = probe.simulate_policy(
        rows, scores, float(artifact["threshold"])
    )
    event = probe.event_metrics(rows, p_fix, p_damage)
    result = {
        "artifact": str(args.artifact),
        "atlas": str(args.atlas),
        "model_kind": artifact["model_kind"],
        "rho": artifact["rho"],
        "threshold": artifact["threshold"],
        "dataset": stats,
        "hard_baseline": probe.hard_baseline(rows),
        "event_metrics": event,
        "external_policy": policy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
