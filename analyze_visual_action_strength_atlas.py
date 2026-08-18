#!/usr/bin/env python3
"""Analyze the complete visual action position/strength atlas."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DATASETS = ("vstar", "mmvp", "realworldqa", "visulogic")
STRENGTHS = ("lambda_025", "lambda_050", "lambda_075", "lambda_095", "lambda_100")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def answer_region(text: str) -> str:
    markers = list(re.finditer(r"(?:final\s+)?answer\s*[:.]", text or "", re.I))
    return text[markers[-1].start() :] if markers else (text or "")[-1500:]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def parse_options(raw: str) -> dict[str, str]:
    raw = raw or ""
    starts = list(re.finditer(r"(?:^|\n|Options:\s*)\s*\(?([A-Da-d])\)?[.):]\s*", raw))
    parsed: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(raw)
        parsed[match.group(1).upper()] = raw[match.end() : end].strip()
    return parsed


def extract_prediction(row: dict) -> str | None:
    region = answer_region(row.get("model_answer") or "")
    patterns = (
        r"\\boxed\{\s*([A-Da-d])\s*\}",
        r"(?:the\s+)?(?:correct\s+)?answer\s+(?:is\s*)?[:\s]+\(?([A-Da-d])\)?",
        r"final\s+(?:answer|choice)\s*(?:is)?\s*[:\s]+\(?([A-Da-d])\)?",
        r"answer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\*\*\s*([A-Da-d])(?:\.|\)|\s|$)",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?[.)]?\s*$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, region, re.I | re.M))
        if matches:
            return matches[-1].group(1).upper()

    norm_region = normalize(region)
    contained = []
    for letter, option in parse_options(row.get("options") or "").items():
        label = normalize(option)
        if label and label in norm_region:
            contained.append((len(label), letter))
    if contained:
        return max(contained)[1]
    letters = re.findall(r"\b([A-D])\b", region[-200:])
    return letters[-1] if letters else None


def enrich(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for original in rows:
        row = dict(original)
        row["pred"] = extract_prediction(row)
        gold = str(row.get("gold") or row.get("answer") or "").strip().upper()
        match = re.search(r"[A-D]", gold)
        row["gold_letter"] = match.group(0) if match else gold
        row["correct"] = row["pred"] == row["gold_letter"]
        result[str(row["event_id"])] = row
    return result


def exact_mcnemar(fixed: int, damaged: int) -> float:
    n = fixed + damaged
    if n == 0:
        return 1.0
    def combination(total: int, selected: int) -> int:
        selected = min(selected, total - selected)
        value = 1
        for index in range(1, selected + 1):
            value = value * (total - selected + index) // index
        return value

    tail = sum(
        combination(n, k) for k in range(0, min(fixed, damaged) + 1)
    ) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare(items: list[dict], baseline: dict[str, dict]) -> dict:
    paired = [(item, baseline[item["event_id"]]) for item in items if item["event_id"] in baseline]
    correct = sum(item["correct"] for item, _ in paired)
    fixed = sum((not hard["correct"]) and item["correct"] for item, hard in paired)
    damaged = sum(hard["correct"] and (not item["correct"]) for item, hard in paired)
    changed = sum(item["pred"] != hard["pred"] for item, hard in paired)
    failed = sum(item["pred"] is None for item, _ in paired)
    return {
        "n": len(paired),
        "correct": correct,
        "accuracy": correct / len(paired) if paired else 0.0,
        "failed_extraction": failed,
        "changed_vs_hard": changed,
        "fixed_vs_hard": fixed,
        "damaged_vs_hard": damaged,
        "net_vs_hard": fixed - damaged,
        "mcnemar_exact_p": exact_mcnemar(fixed, damaged),
    }


def paired_branch(target: dict[str, dict], control: dict[str, dict]) -> dict:
    ids = sorted(set(target) & set(control))
    target_only = sum(target[key]["correct"] and not control[key]["correct"] for key in ids)
    control_only = sum(control[key]["correct"] and not target[key]["correct"] for key in ids)
    return {
        "n": len(ids),
        "target_only_correct": target_only,
        "control_only_correct": control_only,
        "net": target_only - control_only,
        "mcnemar_exact_p": exact_mcnemar(target_only, control_only),
    }


def load_result(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return enrich(read_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strengths",
        nargs="+",
        choices=STRENGTHS,
        default=list(STRENGTHS),
        help="Analyze a complete subset of strengths while longer runs are still in progress.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Analyze only datasets whose requested strengths are complete.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {"datasets": {}, "gate": {}}
    all_gate_rows = []
    for dataset in args.datasets:
        baseline = load_result(args.root / "hard" / dataset / "results.jsonl")
        dataset_report = {"hard_events": len(baseline), "strengths": {}}
        for strength in args.strengths:
            result_path = args.root / strength / dataset / "results.jsonl"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing results for requested strength: {result_path}")
            rows = read_jsonl(result_path)
            by_branch: dict[str, list[dict]] = defaultdict(list)
            for raw in rows:
                enriched_row = enrich([raw])[str(raw["event_id"])]
                by_branch[str(raw["branch"])].append(enriched_row)
            branch_maps = {
                branch: {row["event_id"]: row for row in items}
                for branch, items in by_branch.items()
            }
            strength_report = {"branches": {}, "by_event_type": {}, "paired_controls": {}}
            for branch, items in sorted(by_branch.items()):
                strength_report["branches"][branch] = compare(items, baseline)
                grouped: dict[str, list[dict]] = defaultdict(list)
                for item in items:
                    grouped[str(item["event_type"])].append(item)
                strength_report["by_event_type"][branch] = {
                    key: compare(value, baseline) for key, value in sorted(grouped.items())
                }

            comparisons = (
                ("true_mask_residual", "random_residual"),
                ("true_mask_residual", "shuffled_mask_residual"),
                ("true_mask_residual", "reverse_mask_residual"),
                ("true_dataset_noise_residual", "random_residual"),
                ("true_dataset_noise_residual", "reverse_dataset_noise_residual"),
                ("true_swap_residual", "random_residual"),
            )
            for target, control in comparisons:
                if target in branch_maps and control in branch_maps:
                    key = f"{target}_vs_{control}"
                    value = paired_branch(branch_maps[target], branch_maps[control])
                    strength_report["paired_controls"][key] = value
                    all_gate_rows.append({
                        "dataset": dataset,
                        "strength": strength,
                        "comparison": key,
                        **value,
                    })

            # Select one checkpoint per sample using visual residual magnitude only.
            metadata_path = args.root / strength / dataset / "vector_metadata.jsonl"
            metadata = read_jsonl(metadata_path)
            for source_name, metric_name, branch_name in (
                ("mask", "residual_norm_before_matching", "true_mask_residual"),
                (
                    "dataset_noise",
                    "dataset_noise_residual_norm_before_matching",
                    "true_dataset_noise_residual",
                ),
            ):
                selected: dict[str, dict] = {}
                for item in metadata:
                    value = item.get(metric_name)
                    if value is None:
                        continue
                    sample_id = str(item["receiver_original_id"])
                    if sample_id not in selected or value > selected[sample_id][metric_name]:
                        selected[sample_id] = item
                selected_ids = {str(item["event_id"]) for item in selected.values()}
                selected_rows = [
                    row for row in by_branch.get(branch_name, [])
                    if row["event_id"] in selected_ids
                ]
                strength_report.setdefault("geometry_selected", {})[source_name] = {
                    "selection_metric": metric_name,
                    **compare(selected_rows, baseline),
                }

            dataset_report["strengths"][strength] = strength_report
        report["datasets"][dataset] = dataset_report

    # A conservative gate: the true direction must improve over hard and beat
    # every pre-registered generic/content-destroyed/sign-reversed control.
    strict_candidates = []
    requirements = {
        "true_mask_residual": (
            "true_mask_residual_vs_random_residual",
            "true_mask_residual_vs_shuffled_mask_residual",
            "true_mask_residual_vs_reverse_mask_residual",
        ),
        "true_dataset_noise_residual": (
            "true_dataset_noise_residual_vs_random_residual",
            "true_dataset_noise_residual_vs_reverse_dataset_noise_residual",
        ),
    }
    for dataset, dataset_report in report["datasets"].items():
        for strength, strength_report in dataset_report["strengths"].items():
            for branch, required_comparisons in requirements.items():
                branch_stats = strength_report["branches"].get(branch)
                controls = strength_report["paired_controls"]
                if not branch_stats or not all(key in controls for key in required_comparisons):
                    continue
                if branch_stats["net_vs_hard"] <= 0:
                    continue
                if not all(controls[key]["net"] > 0 for key in required_comparisons):
                    continue
                strict_candidates.append({
                    "dataset": dataset,
                    "strength": strength,
                    "branch": branch,
                    "net_vs_hard": branch_stats["net_vs_hard"],
                    "paired_controls": {
                        key: controls[key] for key in required_comparisons
                    },
                })
    report["gate"] = {
        "status": "pass_candidate" if strict_candidates else "not_supported",
        "strict_candidates": strict_candidates,
        "positive_individual_comparisons": [
            row for row in all_gate_rows if row["net"] > 0
        ],
        "rule": (
            "A visual action is only a candidate when it has positive correctness net against "
            "hard and simultaneously beats every pre-registered generic, content-destroyed, "
            "and sign-reversed control. Final claims also require cross-dataset replication, "
            "sample-clustered uncertainty, and readable-case audit."
        ),
    }

    json_path = args.output_dir / "visual_action_strength_atlas_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Visual Action Position-Strength Atlas",
        "",
        "Each row is paired against the hard continuation under the same forced prefix.",
        "",
        "| Dataset | Strength | Branch | Acc | Fixed | Damaged | Net | Changed | Failed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_report in report["datasets"].items():
        for strength, strength_report in dataset_report["strengths"].items():
            for branch, stats in strength_report["branches"].items():
                lines.append(
                    f"| {dataset} | {strength.replace('lambda_', '0.')} | {branch} | "
                    f"{stats['accuracy']:.2%} | {stats['fixed_vs_hard']} | "
                    f"{stats['damaged_vs_hard']} | {stats['net_vs_hard']:+d} | "
                    f"{stats['changed_vs_hard']} | {stats['failed_extraction']} |"
                )
    lines.extend(["", "## Gate", "", f"Status: **{report['gate']['status']}**", ""])
    (args.output_dir / "visual_action_strength_atlas_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
