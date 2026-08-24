#!/usr/bin/env python3
"""Full-data threshold and aggregation sweep for the inverse visual gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


DATASETS = ("mmvp", "vstar", "realworldqa", "visulogic")
QUANTILES = (0.20, 0.35, 0.50, 0.65, 0.80)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Empty quantile input")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate(values: Sequence[float], name: str) -> float:
    if name == "min":
        return min(values)
    if name == "median":
        return statistics.median(values)
    if name == "mean":
        return sum(values) / len(values)
    if name == "max":
        return max(values)
    raise ValueError(name)


def unique_thresholds(values: Sequence[float]) -> List[Tuple[str, float]]:
    candidates = [("zero", 0.0)] + [("q%.2f" % q, quantile(values, q)) for q in QUANTILES]
    output = []
    seen = set()
    for label, value in candidates:
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            output.append((label, value))
    return output


def choose(row: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, bool]:
    if not row["probe_eligible"]:
        return False, bool(row["action_correct"])
    features = row["gate_features"]
    swap_values = [float(features["swap%d_transient" % index]) for index in (1, 2, 3)]
    swap_stat = aggregate(swap_values, config["aggregation"])
    mask_value = float(features["mask_transient"])
    reject = bool(swap_stat >= config["swap_threshold"] and mask_value >= config["mask_threshold"])
    correct = bool(row["baseline_correct"] if reject else row["action_correct"])
    return reject, correct


def evaluate(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    datasets = {}
    total_baseline = total_action = total_gated = total_rejected = 0
    for dataset in DATASETS:
        items = grouped[dataset]
        decisions = [(row, *choose(row, config)) for row in items]
        total = len(items)
        baseline_correct = sum(bool(row["baseline_correct"]) for row in items)
        action_correct = sum(bool(row["action_correct"]) for row in items)
        gated_correct = sum(correct for _, _, correct in decisions)
        rejected = sum(reject for _, reject, _ in decisions)
        fixed = [row for row in items if not row["baseline_correct"] and row["action_correct"]]
        damaged = [row for row in items if row["baseline_correct"] and not row["action_correct"]]
        fixed_rejected = sum(choose(row, config)[0] for row in fixed)
        damaged_rejected = sum(choose(row, config)[0] for row in damaged)
        report = {
            "total": total,
            "talr_correct": baseline_correct,
            "ungated_correct": action_correct,
            "gated_correct": gated_correct,
            "talr_accuracy": baseline_correct / total,
            "ungated_accuracy": action_correct / total,
            "gated_accuracy": gated_correct / total,
            "net_vs_talr": gated_correct - baseline_correct,
            "net_vs_ungated": gated_correct - action_correct,
            "delta_vs_talr_pp": 100.0 * (gated_correct - baseline_correct) / total,
            "delta_vs_ungated_pp": 100.0 * (gated_correct - action_correct) / total,
            "rejected": rejected,
            "coverage": rejected / total,
            "fixed": len(fixed),
            "damaged": len(damaged),
            "fixed_retention": (1.0 - fixed_rejected / len(fixed)) if fixed else None,
            "damage_rejection": damaged_rejected / len(damaged) if damaged else None,
        }
        if dataset == "mmvp":
            pair_ids = sorted({int(row["pair_id"]) for row in items})
            by_id = {str(row["id"]): choose(row, config)[1] for row in items}
            pair_correct = 0
            for pair_id in pair_ids:
                pair_rows = [row for row in items if int(row["pair_id"]) == pair_id]
                pair_correct += int(len(pair_rows) == 2 and all(by_id[str(row["id"])] for row in pair_rows))
            report.update({"pair_correct": pair_correct, "pair_total": len(pair_ids), "pair_accuracy": pair_correct / len(pair_ids)})
        datasets[dataset] = report
        total_baseline += baseline_correct
        total_action += action_correct
        total_gated += gated_correct
        total_rejected += rejected
    deltas = [datasets[name]["delta_vs_talr_pp"] for name in DATASETS]
    return {
        "config": config,
        "datasets": datasets,
        "total": len(rows),
        "talr_correct": total_baseline,
        "ungated_correct": total_action,
        "gated_correct": total_gated,
        "net_vs_talr": total_gated - total_baseline,
        "net_vs_ungated": total_gated - total_action,
        "rejected": total_rejected,
        "coverage": total_rejected / len(rows),
        "nonnegative_vs_talr": sum(value >= 0.0 for value in deltas),
        "worst_delta_vs_talr_pp": min(deltas),
        "macro_delta_vs_talr_pp": sum(deltas) / len(deltas),
    }


def rank_key(result: Dict[str, Any], selected_datasets: Iterable[str] = DATASETS) -> Tuple[Any, ...]:
    names = tuple(selected_datasets)
    net_talr = sum(result["datasets"][name]["net_vs_talr"] for name in names)
    net_action = sum(result["datasets"][name]["net_vs_ungated"] for name in names)
    deltas = [result["datasets"][name]["delta_vs_talr_pp"] for name in names]
    return (net_talr, net_action, min(deltas), sum(deltas) / len(deltas), -result["coverage"])


def compact(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": result["config"],
        "net_vs_talr": result["net_vs_talr"],
        "net_vs_ungated": result["net_vs_ungated"],
        "coverage": result["coverage"],
        "nonnegative_vs_talr": result["nonnegative_vs_talr"],
        "worst_delta_vs_talr_pp": result["worst_delta_vs_talr_pp"],
        "macro_delta_vs_talr_pp": result["macro_delta_vs_talr_pp"],
        "datasets": result["datasets"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.gated_results)
    if len(rows) != 1691:
        raise RuntimeError("Expected 1691 full rows, got %d" % len(rows))
    eligible = [row for row in rows if row["probe_eligible"]]
    if any(row.get("error_type") for row in rows):
        raise RuntimeError("Runtime errors present")

    mask_values = [float(row["gate_features"]["mask_transient"]) for row in eligible]
    configs = []
    for aggregation in ("min", "median", "mean", "max"):
        swap_values = [aggregate([float(row["gate_features"]["swap%d_transient" % i]) for i in (1, 2, 3)], aggregation) for row in eligible]
        for swap_label, swap_threshold in unique_thresholds(swap_values):
            for mask_label, mask_threshold in unique_thresholds(mask_values):
                configs.append({
                    "aggregation": aggregation,
                    "swap_threshold_label": swap_label,
                    "swap_threshold": swap_threshold,
                    "mask_threshold_label": mask_label,
                    "mask_threshold": mask_threshold,
                })
    locked = {"aggregation": "median", "swap_threshold_label": "zero", "swap_threshold": 0.0, "mask_threshold_label": "zero", "mask_threshold": 0.0}
    if locked not in configs:
        configs.append(locked)
    results = [evaluate(rows, config) for config in configs]
    eligible_results = [result for result in results if result["net_vs_talr"] > 0 and result["nonnegative_vs_talr"] >= 3 and result["worst_delta_vs_talr_pp"] >= -1.0]
    selected = max(eligible_results or results, key=rank_key)
    locked_result = next(result for result in results if result["config"] == locked)

    per_dataset_best = {}
    for dataset in DATASETS:
        per_dataset_best[dataset] = compact(max(results, key=lambda result: (result["datasets"][dataset]["gated_correct"], result["datasets"][dataset].get("pair_correct", -1), -result["datasets"][dataset]["coverage"])))

    lodo = {}
    for heldout in DATASETS:
        train = tuple(name for name in DATASETS if name != heldout)
        chosen = max(results, key=lambda result: rank_key(result, train))
        lodo[heldout] = {"selected_config": chosen["config"], "train_datasets": train, "heldout": chosen["datasets"][heldout]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gate_full_sweep_all_configs.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=rank_key, reverse=True):
            handle.write(json.dumps(compact(result), ensure_ascii=False) + "\n")
    artifact = {
        "status": "full-data development sweep; not an untouched test estimate",
        "action": "TALR true visual residual strength=1.0 duration=4",
        "rows": len(rows), "eligible_actions": len(eligible), "config_count": len(results),
        "search_space": {"aggregation": ["min", "median", "mean", "max"], "thresholds": ["zero"] + ["q%.2f" % q for q in QUANTILES], "logic": "swap statistic >= threshold AND mask transient >= threshold"},
        "selection_constraints": {"net_vs_talr": ">0", "nonnegative_datasets": ">=3/4", "worst_drop_vs_talr": ">=-1pp"},
        "locked_original": compact(locked_result),
        "selected_global": compact(selected),
        "top10": [compact(result) for result in sorted(results, key=rank_key, reverse=True)[:10]],
        "per_dataset_best_ceiling": per_dataset_best,
        "leave_one_dataset_out": lodo,
    }
    (args.output_dir / "gate_full_tuning_summary.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Inverse Gate Full Tuning", "", "This is a full-data development sweep over the already generated 1.0 x 4 residual action. It is not an untouched test estimate.", "", "## Locked versus selected", "", "| Policy | Aggregation | Swap threshold | Mask threshold | Net vs TALR | Net vs ungated | Coverage | Worst dataset |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in (("Locked", locked_result), ("Selected", selected)):
        cfg = result["config"]
        lines.append("| %s | %s | %.6g | %.6g | %+d | %+d | %.1f%% | %+.2f pp |" % (name, cfg["aggregation"], cfg["swap_threshold"], cfg["mask_threshold"], result["net_vs_talr"], result["net_vs_ungated"], 100*result["coverage"], result["worst_delta_vs_talr_pp"]))
    lines.extend(["", "## Selected full-data result", "", "| Dataset | TALR | Ungated | Gated | Delta vs TALR | Delta vs ungated | Coverage |", "|---|---:|---:|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        report = selected["datasets"][dataset]
        lines.append("| %s | %.2f%% | %.2f%% | %.2f%% | %+.2f | %+.2f | %.1f%% |" % (dataset, 100*report["talr_accuracy"], 100*report["ungated_accuracy"], 100*report["gated_accuracy"], report["delta_vs_talr_pp"], report["delta_vs_ungated_pp"], 100*report["coverage"]))
    lines.extend(["", "The selected policy must be validated on another model or held-out benchmark before it can replace the locked gate in the method."])
    (args.output_dir / "gate_full_tuning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(results), "locked": compact(locked_result), "selected": compact(selected)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
