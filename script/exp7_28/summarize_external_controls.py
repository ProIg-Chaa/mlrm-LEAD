#!/usr/bin/env python3
"""Evaluate matched-coverage entropy/random controls and the oracle ceiling."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(groups: dict[str, list], chosen: dict[str, object]) -> dict:
    correct = fixed = damaged = 0
    by_dataset: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "fixed": 0, "damaged": 0}
    )
    for group, rows in groups.items():
        base = bool(rows[0].base_correct)
        row = chosen.get(group)
        final = bool(row.treatment_correct) if row is not None else base
        dataset = rows[0].dataset
        correct += int(final)
        fixed += int(not base and final)
        damaged += int(base and not final)
        stats = by_dataset[dataset]
        stats["n"] += 1
        stats["correct"] += int(final)
        stats["fixed"] += int(not base and final)
        stats["damaged"] += int(base and not final)
    total = len(groups)
    output_by_dataset = {}
    for dataset, stats in sorted(by_dataset.items()):
        output_by_dataset[dataset] = {
            **stats,
            "accuracy": stats["correct"] / stats["n"],
            "net": stats["fixed"] - stats["damaged"],
        }
    return {
        "samples": total,
        "accuracy": correct / total,
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "coverage": len(chosen) / total,
        "by_dataset": output_by_dataset,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--random-repeats", type=int, default=1000)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_dir))
    import train_hierarchical_utility_probe_v2 as v2  # noqa: PLC0415

    rows, _, utility_names, stats = v2.prepare_rows(read_jsonl(args.atlas))
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row.group].append(row)

    summary = json.loads(args.v2_summary.read_text(encoding="utf-8"))
    policy = summary["external"][args.model]["policy"]
    target_count = int(policy["interventions"])

    entropy_index = utility_names.index("log1p_raw_entropy")
    entropy_candidates = []
    for group, candidates in groups.items():
        best = max(candidates, key=lambda row: (row.x_utility[entropy_index], -row.event_step))
        entropy_candidates.append((float(best.x_utility[entropy_index]), group, best))
    entropy_candidates.sort(reverse=True, key=lambda item: item[0])
    entropy_chosen = {
        group: row for _, group, row in entropy_candidates[:target_count]
    }

    oracle_chosen = {}
    for group, candidates in groups.items():
        fixes = [
            row
            for row in candidates
            if not row.base_correct and row.treatment_correct
        ]
        if fixes:
            oracle_chosen[group] = min(fixes, key=lambda row: row.event_step)

    random_results = []
    group_names = sorted(groups)
    rng = random.Random(20260728)
    for _ in range(args.random_repeats):
        selected = rng.sample(group_names, min(target_count, len(group_names)))
        chosen = {
            group: rng.choice(groups[group])
            for group in selected
        }
        random_results.append(evaluate(groups, chosen))

    result = {
        "data": stats,
        "probe": {
            key: policy[key]
            for key in (
                "samples",
                "accuracy",
                "fixed",
                "damaged",
                "net",
                "coverage",
                "by_dataset",
            )
        },
        "entropy_matched_coverage": evaluate(groups, entropy_chosen),
        "oracle_any_action": evaluate(groups, oracle_chosen),
        "random_matched_coverage": {
            "repeats": args.random_repeats,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in random_results])),
            "accuracy_std": float(np.std([row["accuracy"] for row in random_results])),
            "net_mean": float(np.mean([row["net"] for row in random_results])),
            "net_std": float(np.std([row["net"] for row in random_results])),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_control_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Probe V2 External Controls",
        "",
        "| Policy | Accuracy | Fixed | Damaged | Net | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("probe", "entropy_matched_coverage", "oracle_any_action"):
        value = result[name]
        lines.append(
            f"| {name} | {value['accuracy']:.4f} | {value['fixed']} | "
            f"{value['damaged']} | {value['net']} | {value['coverage']:.4f} |"
        )
    random_value = result["random_matched_coverage"]
    lines += [
        "",
        "## Matched-Coverage Random",
        "",
        f"- Accuracy: {random_value['accuracy_mean']:.4f} +/- "
        f"{random_value['accuracy_std']:.4f}",
        f"- Net: {random_value['net_mean']:.2f} +/- "
        f"{random_value['net_std']:.2f}",
        "",
        "The oracle is a diagnostic ceiling and is not deployable.",
    ]
    (args.output_dir / "external_control_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
