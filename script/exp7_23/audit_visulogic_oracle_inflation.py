#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVENT_TYPES = (
    "fixed_1",
    "fixed_2",
    "fixed_4",
    "fixed_8",
    "fixed_16",
    "fixed_32",
    "entropy_top1",
    "random_control",
)
TREATMENTS = ("contracted_soft_l095", "pure_soft_l100")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="visulogic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()

    rows = [
        row
        for row in load_jsonl(args.atlas)
        if row.get("dataset") == args.dataset
    ]
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["original_id"])
        sample = by_sample.setdefault(
            sample_id,
            {
                "base_correct": bool(row["base_correct"]),
                "base_pred": row.get("base_pred"),
                "gold": row.get("gold"),
                "actions": {},
            },
        )
        key = (str(row["event_type"]), str(row["treatment"]))
        sample["actions"][key] = {
            "correct": bool(row["treatment_correct"]),
            "pred": row.get("treatment_pred"),
            "failed": bool(row.get("treatment_failed_extraction")),
            "prefix_match": bool(row.get("prefix_match")),
        }

    total = len(by_sample)
    base_correct = sum(sample["base_correct"] for sample in by_sample.values())

    single_policy = {}
    for event_type, treatment in itertools.product(EVENT_TYPES, TREATMENTS):
        correct = sum(
            sample["actions"][(event_type, treatment)]["correct"]
            for sample in by_sample.values()
        )
        single_policy[f"{event_type}/{treatment}"] = {
            "correct": correct,
            "accuracy": accuracy(correct, total),
            "delta_correct": correct - base_correct,
        }

    event_two_action_oracle = {}
    for event_type in EVENT_TYPES:
        correct = sum(
            sample["base_correct"]
            or any(
                sample["actions"][(event_type, treatment)]["correct"]
                for treatment in TREATMENTS
            )
            for sample in by_sample.values()
        )
        event_two_action_oracle[event_type] = {
            "correct": correct,
            "accuracy": accuracy(correct, total),
            "delta_correct": correct - base_correct,
        }

    treatment_oracle = {}
    for treatment in TREATMENTS:
        correct = sum(
            sample["base_correct"]
            or any(
                sample["actions"][(event_type, treatment)]["correct"]
                for event_type in EVENT_TYPES
            )
            for sample in by_sample.values()
        )
        treatment_oracle[treatment] = {
            "correct": correct,
            "accuracy": accuracy(correct, total),
            "delta_correct": correct - base_correct,
        }

    all_action_keys = list(itertools.product(EVENT_TYPES, TREATMENTS))
    joint_correct = sum(
        sample["base_correct"]
        or any(sample["actions"][key]["correct"] for key in all_action_keys)
        for sample in by_sample.values()
    )

    rng = random.Random(args.seed)
    best_of_k = {}
    for k in (1, 2, 4, 8, 12, 16):
        scores = []
        for _ in range(args.trials):
            chosen = rng.sample(all_action_keys, k)
            correct = sum(
                sample["base_correct"]
                or any(sample["actions"][key]["correct"] for key in chosen)
                for sample in by_sample.values()
            )
            scores.append(accuracy(correct, total))
        ordered = sorted(scores)
        best_of_k[str(k)] = {
            "mean_accuracy": statistics.fmean(scores),
            "p05_accuracy": ordered[int(0.05 * (len(ordered) - 1))],
            "p95_accuracy": ordered[int(0.95 * (len(ordered) - 1))],
            "independent_random_guess_null": (
                base_correct
                + (total - base_correct) * (1.0 - (3.0 / 4.0) ** k)
            )
            / total,
        }

    unique_prediction_counts = []
    prediction_set_contains_gold = 0
    prediction_sets = Counter()
    for sample in by_sample.values():
        predictions = {
            prediction
            for prediction in (
                [sample["base_pred"]]
                + [
                    sample["actions"][key]["pred"]
                    for key in all_action_keys
                ]
            )
            if prediction is not None
        }
        unique_prediction_counts.append(len(predictions))
        prediction_set_contains_gold += int(sample["gold"] in predictions)
        prediction_sets["".join(sorted(predictions)) or "NONE"] += 1

    integrity = {
        "rows": len(rows),
        "samples": total,
        "expected_rows": total * len(all_action_keys),
        "all_prefix_match": all(bool(row.get("prefix_match")) for row in rows),
        "runtime_errors": sum(
            bool(row.get("treatment_runtime_error")) for row in rows
        ),
        "treatment_failed_extraction": sum(
            bool(row.get("treatment_failed_extraction")) for row in rows
        ),
        "gold_distribution": dict(
            Counter(str(sample["gold"]) for sample in by_sample.values())
        ),
        "base_prediction_distribution": dict(
            Counter(str(sample["base_pred"]) for sample in by_sample.values())
        ),
    }

    summary = {
        "dataset": args.dataset,
        "base": {
            "correct": base_correct,
            "samples": total,
            "accuracy": accuracy(base_correct, total),
        },
        "joint_oracle": {
            "correct": joint_correct,
            "accuracy": accuracy(joint_correct, total),
            "delta_correct": joint_correct - base_correct,
        },
        "single_policy": single_policy,
        "event_two_action_oracle": event_two_action_oracle,
        "treatment_oracle": treatment_oracle,
        "best_of_k": best_of_k,
        "prediction_coverage": {
            "mean_unique_predictions": statistics.fmean(unique_prediction_counts),
            "median_unique_predictions": statistics.median(
                unique_prediction_counts
            ),
            "max_unique_predictions": max(unique_prediction_counts),
            "contains_gold": prediction_set_contains_gold,
            "contains_gold_accuracy": accuracy(
                prediction_set_contains_gold, total
            ),
            "prediction_set_distribution": dict(prediction_sets),
        },
        "integrity": integrity,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visulogic_oracle_inflation_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# VisuLogic Oracle Inflation Audit",
        "",
        f"- Base: {base_correct}/{total} = {100*accuracy(base_correct,total):.2f}%",
        f"- Joint oracle: {joint_correct}/{total} = {100*accuracy(joint_correct,total):.2f}%",
        "",
        "## Single deployable policies",
        "",
        "| Policy | Correct | Accuracy | Delta correct |",
        "|---|---:|---:|---:|",
    ]
    for key, value in single_policy.items():
        lines.append(
            f"| {key} | {value['correct']} | "
            f"{100*value['accuracy']:.2f}% | {value['delta_correct']:+d} |"
        )
    lines += [
        "",
        "## Two-action oracle at one event type",
        "",
        "| Event | Correct | Accuracy | Delta correct |",
        "|---|---:|---:|---:|",
    ]
    for key, value in event_two_action_oracle.items():
        lines.append(
            f"| {key} | {value['correct']} | "
            f"{100*value['accuracy']:.2f}% | {value['delta_correct']:+d} |"
        )
    lines += [
        "",
        "## Best-of-k inflation",
        "",
        "| k | Realized mean oracle | 5%-95% | Independent random-guess null |",
        "|---:|---:|---:|---:|",
    ]
    for key, value in best_of_k.items():
        lines.append(
            f"| {key} | {100*value['mean_accuracy']:.2f}% | "
            f"{100*value['p05_accuracy']:.2f}% - "
            f"{100*value['p95_accuracy']:.2f}% | "
            f"{100*value['independent_random_guess_null']:.2f}% |"
        )
    (args.output_dir / "visulogic_oracle_inflation_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
