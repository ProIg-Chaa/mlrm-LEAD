#!/usr/bin/env python3
"""Evaluate saved hierarchical Probe V2 artifacts without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model(v2, artifact: Path, action_names: list[str], utility_names: list[str]):
    state = torch.load(artifact, map_location="cpu", weights_only=False)
    if state["action_features"] != action_names:
        raise RuntimeError(f"Action feature mismatch for {artifact}")
    if state["utility_features"] != utility_names:
        raise RuntimeError(f"Utility feature mismatch for {artifact}")
    kind = state["kind"]
    action = v2.BinaryProbe(len(action_names), kind)
    fix = v2.BinaryProbe(len(utility_names), kind)
    damage = v2.BinaryProbe(len(utility_names), kind)
    action.load_state_dict(state["action_state_dict"])
    fix.load_state_dict(state["fix_state_dict"])
    damage.load_state_dict(state["damage_state_dict"])
    action.eval()
    fix.eval()
    damage.eval()
    model = v2.FittedV2(
        action=action,
        fix=fix,
        damage=damage,
        action_mean=np.asarray(state["action_mean"]),
        action_std=np.asarray(state["action_std"]),
        utility_mean=np.asarray(state["utility_mean"]),
        utility_std=np.asarray(state["utility_std"]),
    )
    return model, state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--linear-artifact", type=Path, required=True)
    parser.add_argument("--mlp-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir))
    import train_hierarchical_utility_probe_v2 as v2  # noqa: PLC0415

    rows, action_names, utility_names, data_stats = v2.prepare_rows(
        read_jsonl(args.atlas)
    )
    results = {}
    for kind, artifact in {
        "linear": args.linear_artifact,
        "mlp": args.mlp_artifact,
    }.items():
        model, state = load_model(v2, artifact, action_names, utility_names)
        predictions = v2.predict(model, rows, float(state["rho"]))
        results[kind] = {
            "artifact": str(artifact.resolve()),
            "seed": state["seed"],
            "rho": float(state["rho"]),
            "action_threshold": float(state["action_threshold"]),
            "utility_threshold": float(state["utility_threshold"]),
            "hard": v2.hard_baseline(rows),
            "event_metrics": v2.event_metrics(rows, predictions),
            "policy": v2.simulate(
                rows,
                predictions,
                float(state["action_threshold"]),
                float(state["utility_threshold"]),
            ),
        }

    summary = {
        "evaluation": "exact_frozen_artifact",
        "external_data": data_stats,
        "external": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hierarchical_probe_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Hierarchical Probe V2 External Expansion",
        "",
        f"- Independent samples: {data_stats['independent_samples']}",
        f"- Eligible rows: {data_stats['eligible_rows']}",
        "- Evaluation: exact saved artifacts; no fitting or threshold selection.",
        "",
        "| Model | Hard acc | Probe acc | Fixed | Damaged | Net | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, value in results.items():
        policy = value["policy"]
        lines.append(
            f"| {kind} | {value['hard']['accuracy']:.4f} | "
            f"{policy['accuracy']:.4f} | {policy['fixed']} | "
            f"{policy['damaged']} | {policy['net']} | "
            f"{policy['coverage']:.4f} |"
        )
    lines += [
        "",
        "These samples were excluded from all Probe V2 fitting and calibration.",
    ]
    (args.output_dir / "hierarchical_probe_v2_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
