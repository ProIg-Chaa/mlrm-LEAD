#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TREATMENTS = ("contracted_soft_l095", "pure_soft_l100")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    lmean = statistics.fmean(left)
    rmean = statistics.fmean(right)
    numerator = sum((x - lmean) * (y - rmean) for x, y in zip(left, right))
    lden = math.sqrt(sum((x - lmean) ** 2 for x in left))
    rden = math.sqrt(sum((y - rmean) ** 2 for y in right))
    if lden == 0.0 or rden == 0.0:
        return None
    return numerator / (lden * rden)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def utility_counter(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        utility = row.get("utility_acc")
        counts["events"] += 1
        counts["prefix_match"] += int(bool(row.get("prefix_match")))
        counts["fixed"] += int(utility == 1)
        counts["damaged"] += int(utility == -1)
        counts["unchanged"] += int(utility == 0)
        counts["answer_changed"] += int(bool(row.get("answer_changed")))
        counts["failed"] += int(bool(row.get("treatment_failed_extraction")))
        counts["runtime_error"] += int(bool(row.get("treatment_runtime_error")))
    result = dict(counts)
    result["net"] = counts["fixed"] - counts["damaged"]
    result["nonzero_utility"] = counts["fixed"] + counts["damaged"]
    result["nonzero_rate"] = (
        result["nonzero_utility"] / counts["events"] if counts["events"] else 0.0
    )
    return result


def group_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return {
        "/".join(map(str, key)): utility_counter(group)
        for key, group in sorted(grouped.items())
    }


def entropy_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["treatment"]))].append(row)
    for (dataset, treatment), group in sorted(grouped.items()):
        usable = [
            row
            for row in group
            if row.get("utility_acc") is not None
            and row.get("online_pre_intervention_features", {}).get("raw_entropy")
            is not None
        ]
        entropies = [
            float(row["online_pre_intervention_features"]["raw_entropy"])
            for row in usable
        ]
        utilities = [float(row["utility_acc"]) for row in usable]
        ordered = sorted(
            usable,
            key=lambda row: float(
                row["online_pre_intervention_features"]["raw_entropy"]
            ),
        )
        quintiles = {}
        for index in range(5):
            start = len(ordered) * index // 5
            stop = len(ordered) * (index + 1) // 5
            bucket = ordered[start:stop]
            stats = utility_counter(bucket)
            bucket_entropies = [
                float(row["online_pre_intervention_features"]["raw_entropy"])
                for row in bucket
            ]
            stats["entropy_min"] = min(bucket_entropies) if bucket_entropies else None
            stats["entropy_max"] = max(bucket_entropies) if bucket_entropies else None
            stats["entropy_mean"] = (
                statistics.fmean(bucket_entropies) if bucket_entropies else None
            )
            quintiles[f"Q{index + 1}"] = stats
        output[f"{dataset}/{treatment}"] = {
            "events": len(usable),
            "spearman_entropy_utility": spearman(entropies, utilities),
            "quintiles": quintiles,
        }
    return output


def action_pair_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        events[str(row["event_id"])][str(row["treatment"])] = row
    totals = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    for pair in events.values():
        if set(pair) != set(TREATMENTS):
            totals["incomplete_pairs"] += 1
            continue
        contracted = pair["contracted_soft_l095"]
        pure = pair["pure_soft_l100"]
        cu = contracted.get("utility_acc")
        pu = pure.get("utility_acc")
        dataset = str(contracted["dataset"])
        target = by_dataset[dataset]
        totals["pairs"] += 1
        target["pairs"] += 1
        if cu is not None and pu is not None:
            if cu > pu:
                totals["contracted_better"] += 1
                target["contracted_better"] += 1
            elif pu > cu:
                totals["pure_better"] += 1
                target["pure_better"] += 1
            else:
                totals["utility_tie"] += 1
                target["utility_tie"] += 1
        same_pred = contracted.get("treatment_pred") == pure.get("treatment_pred")
        totals["prediction_agreement"] += int(same_pred)
        target["prediction_agreement"] += int(same_pred)
    return {
        "overall": dict(totals),
        "by_dataset": {
            key: dict(value) for key, value in sorted(by_dataset.items())
        },
    }


def oracle_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["original_id"]))
        state = samples.setdefault(
            key,
            {
                "base_correct": bool(row.get("base_correct")),
                "contracted_correct": False,
                "pure_correct": False,
            },
        )
        if row["treatment"] == "contracted_soft_l095":
            state["contracted_correct"] |= bool(row.get("treatment_correct"))
        elif row["treatment"] == "pure_soft_l100":
            state["pure_correct"] |= bool(row.get("treatment_correct"))
    grouped: dict[str, Counter] = defaultdict(Counter)
    for (dataset, _), state in samples.items():
        counter = grouped[dataset]
        counter["samples"] += 1
        counter["base_correct"] += int(state["base_correct"])
        counter["contracted_oracle_correct"] += int(
            state["base_correct"] or state["contracted_correct"]
        )
        counter["pure_oracle_correct"] += int(
            state["base_correct"] or state["pure_correct"]
        )
        counter["joint_oracle_correct"] += int(
            state["base_correct"]
            or state["contracted_correct"]
            or state["pure_correct"]
        )
    output = {}
    for dataset, counter in sorted(grouped.items()):
        stats = dict(counter)
        total = counter["samples"]
        stats["base_accuracy"] = counter["base_correct"] / total
        stats["joint_oracle_accuracy"] = counter["joint_oracle_correct"] / total
        stats["joint_oracle_gain"] = (
            counter["joint_oracle_correct"] - counter["base_correct"]
        ) / total
        output[dataset] = stats
    return output


def decision_gates(analysis: dict[str, Any]) -> dict[str, Any]:
    oracle = analysis["oracle"]
    event_totals = analysis["by_dataset_treatment"]
    all_prefix_match = all(
        value["prefix_match"] == value["events"]
        for value in event_totals.values()
    )
    no_errors = all(
        value.get("failed", 0) == 0 and value.get("runtime_error", 0) == 0
        for value in event_totals.values()
    )
    positive_oracle_datasets = sum(
        stats["joint_oracle_gain"] > 0 for stats in oracle.values()
    )
    minimum_oracle_gain = min(
        stats["joint_oracle_gain"] for stats in oracle.values()
    )
    pairs = analysis["action_pairs"]["overall"]
    neither_action_dominates = (
        pairs.get("contracted_better", 0) > 0
        and pairs.get("pure_better", 0) > 0
    )
    return {
        "atlas_valid": all_prefix_match and no_errors,
        "all_prefix_match": all_prefix_match,
        "no_failed_or_runtime_errors": no_errors,
        "positive_oracle_datasets": positive_oracle_datasets,
        "minimum_joint_oracle_gain": minimum_oracle_gain,
        "proceed_to_utility_probe": (
            all_prefix_match
            and no_errors
            and positive_oracle_datasets >= 3
            and minimum_oracle_gain >= 0.03
        ),
        "proceed_to_action_strength_modeling": neither_action_dominates,
        "realworldqa_requires_conservative_policy": (
            event_totals["realworldqa/contracted_soft_l095"]["net"] < 0
            and event_totals["realworldqa/pure_soft_l100"]["net"] < 0
        ),
    }


def fmt_percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Intervention Atlas V0B Decision Analysis",
        "",
        "## Decision Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in analysis["decision_gates"].items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "## Oracle Upper Bound",
        "",
        "| Dataset | Samples | Base | Joint oracle | Gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, value in analysis["oracle"].items():
        lines.append(
            f"| {dataset} | {value['samples']} | "
            f"{fmt_percent(value['base_accuracy'])} | "
            f"{fmt_percent(value['joint_oracle_accuracy'])} | "
            f"{fmt_percent(value['joint_oracle_gain'])} |"
        )
    lines += [
        "",
        "## Event-level Utility",
        "",
        "| Dataset / treatment | Events | Fixed | Damaged | Net | Changed | Failed | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in analysis["by_dataset_treatment"].items():
        lines.append(
            f"| {key} | {value['events']} | {value['fixed']} | "
            f"{value['damaged']} | {value['net']:+d} | "
            f"{value['answer_changed']} | {value.get('failed', 0)} | "
            f"{value.get('runtime_error', 0)} |"
        )
    lines += [
        "",
        "## Entropy vs Utility",
        "",
        "| Dataset / treatment | Events | Spearman rho | Q1 net | Q2 net | Q3 net | Q4 net | Q5 net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in analysis["entropy"].items():
        rho = value["spearman_entropy_utility"]
        rho_text = "-" if rho is None else f"{rho:.4f}"
        nets = [value["quintiles"][f"Q{i}"]["net"] for i in range(1, 6)]
        lines.append(
            f"| {key} | {value['events']} | {rho_text} | "
            + " | ".join(f"{net:+d}" for net in nets)
            + " |"
        )
    lines += [
        "",
        "## Action Pair Comparison",
        "",
        "```json",
        json.dumps(analysis["action_pairs"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Registered Next Step",
        "",
    ]
    gates = analysis["decision_gates"]
    if gates["proceed_to_utility_probe"]:
        lines.append(
            "Proceed to a leakage-controlled utility probe with sample-grouped "
            "splits and leave-one-dataset-out evaluation."
        )
    else:
        lines.append(
            "Do not train a utility probe yet. Revisit treatment actions or "
            "counterfactual label validity."
        )
    if gates["proceed_to_action_strength_modeling"]:
        lines.append(
            "Because neither contracted nor pure soft dominates, retain action "
            "strength as a decision variable for trajectory trust-region tests."
        )
    if gates["realworldqa_requires_conservative_policy"]:
        lines.append(
            "RealWorldQA requires an abstaining/conservative policy: most "
            "unselected interventions are harmful despite a positive oracle gap."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.atlas)
    analysis = {
        "rows": len(rows),
        "unique_events": len({row["event_id"] for row in rows}),
        "by_dataset_treatment": group_summary(
            rows, ("dataset", "treatment")
        ),
        "by_event_type": group_summary(
            rows, ("dataset", "treatment", "event_type")
        ),
        "entropy": entropy_analysis(rows),
        "action_pairs": action_pair_analysis(rows),
        "oracle": oracle_analysis(rows),
    }
    analysis["decision_gates"] = decision_gates(analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "atlas_decision_analysis.json"
    md_path = args.output_dir / "atlas_decision_analysis.md"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, analysis)
    print(json.dumps(analysis["decision_gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
