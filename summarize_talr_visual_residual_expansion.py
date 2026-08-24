#!/usr/bin/env python3
"""Merge shards and evaluate TALR visual-residual probe results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "script"))
from evaluate_specialized_results import evaluate
from evaluate_realworldqa_mcq import evaluate as evaluate_realworldqa


BRANCHES = ("talr", "talr_true_residual", "talr_random_residual")

MCQ_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}"),
    re.compile(r"[Ff]inal\s+(?:answer|choice)\s*(?:is)?\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?"),
    re.compile(r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?"),
    re.compile(r"\*\*([A-Da-d])\*\*"),
    re.compile(r"(?:^|\n)\s*([A-Da-d])\s*$"),
]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare(target, reference):
    ids = sorted(set(target) & set(reference), key=int)
    fixed = [key for key in ids if target[key]["specialized_is_correct"] and not reference[key]["specialized_is_correct"]]
    damaged = [key for key in ids if reference[key]["specialized_is_correct"] and not target[key]["specialized_is_correct"]]
    changed = [key for key in ids if target[key]["specialized_pred"] != reference[key]["specialized_pred"]]
    return {
        "n": len(ids),
        "fixed": len(fixed),
        "damaged": len(damaged),
        "net": len(fixed) - len(damaged),
        "changed_predictions": len(changed),
        "fixed_ids": fixed,
        "damaged_ids": damaged,
    }


def extract_last_mcq(text):
    if not text:
        return None
    markers = list(re.finditer(r"answer\s*[:.]", text, re.I))
    region = text[markers[-1].start():] if markers else text[-1500:]
    hits = []
    for pattern in MCQ_PATTERNS:
        hits.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(region))
    if hits:
        return sorted(hits)[-1][1]
    letters = re.findall(r"\b([A-D])\b", region[-200:])
    return letters[-1].upper() if letters else None


def evaluate_mcq(dataset_rows, result_rows):
    by_id = {int(row["id"]): row for row in dataset_rows}
    total = correct = failed = 0
    subtopics = defaultdict(lambda: {"correct": 0, "total": 0})
    enriched = []
    for row in result_rows:
        sample = by_id[int(row["id"])]
        pred = extract_last_mcq(row.get("model_answer") or "")
        gold = str(sample.get("answer") or "").strip().upper()
        ok = pred is not None and pred == gold
        total += 1
        correct += int(ok)
        failed += int(pred is None)
        subtopic = sample.get("subtopic", "unknown")
        subtopics[subtopic]["total"] += 1
        subtopics[subtopic]["correct"] += int(ok)
        item = dict(row)
        item.update({
            "specialized_gold": gold,
            "specialized_pred": pred,
            "specialized_match_method": "corrected_last_answer",
            "specialized_is_correct": ok,
        })
        enriched.append(item)
    report = {
        "mode": "corrected_mcq",
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failed_extraction": failed,
        "by_subtopic": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(subtopics.items())
        },
    }
    return report, enriched


def evaluate_realworldqa_standardized(dataset_rows, result_rows):
    report, enriched = evaluate_realworldqa(dataset_rows, result_rows)
    standardized = []
    for row in enriched:
        item = dict(row)
        item["specialized_gold"] = item.get("realworldqa_gold")
        item["specialized_pred"] = item.get("realworldqa_pred")
        item["specialized_match_method"] = item.get("realworldqa_match_method")
        item["specialized_is_correct"] = item.get("realworldqa_is_correct", False)
        standardized.append(item)
    return report, standardized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        choices=("mmvp", "vstar", "realworldqa", "visulogic"),
        required=True,
    )
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    config_path = args.root / "shard_0" / "config.json"
    if config_path.is_file():
        selection = json.loads(config_path.read_text(encoding="utf-8")).get("selection", "unknown")
    elif args.root.name == "full_merged":
        selection = "full_merged"
    else:
        selection = args.root.name
    merged_dir = args.root / "merged"
    all_rows = []
    for shard in range(args.num_shards):
        all_rows.extend(read_jsonl(args.root / f"shard_{shard}" / "results.jsonl"))
    unique = {}
    for row in all_rows:
        key = (int(row["id"]), str(row["branch"]))
        if key in unique:
            raise RuntimeError(f"Duplicate result: {key}")
        unique[key] = row
    branch_ids = {
        branch: {sample_id for sample_id, name in unique if name == branch}
        for branch in BRANCHES
    }
    if not branch_ids["talr"] or any(ids != branch_ids["talr"] for ids in branch_ids.values()):
        raise RuntimeError(f"Incomplete or mismatched branches: {branch_ids}")
    dataset = read_jsonl(args.dataset)
    reports = {}
    enriched_maps = {}
    for branch in BRANCHES:
        rows = [row for (_, name), row in unique.items() if name == branch]
        rows.sort(key=lambda row: int(row["id"]))
        if args.dataset_name == "mmvp":
            report, enriched = evaluate(dataset, rows, "mmvp")
        elif args.dataset_name == "realworldqa":
            report, enriched = evaluate_realworldqa_standardized(dataset, rows)
        else:
            report, enriched = evaluate_mcq(dataset, rows)
        branch_dir = merged_dir / branch
        write_jsonl(branch_dir / "results.jsonl", rows)
        write_jsonl(branch_dir / "specialized_results.jsonl", enriched)
        (branch_dir / "specialized_eval_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports[branch] = report
        enriched_maps[branch] = {str(row["id"]): row for row in enriched}

    comparisons = {
        "true_residual_vs_talr": compare(
            enriched_maps["talr_true_residual"], enriched_maps["talr"]
        ),
        "random_residual_vs_talr": compare(
            enriched_maps["talr_random_residual"], enriched_maps["talr"]
        ),
        "true_residual_vs_random": compare(
            enriched_maps["talr_true_residual"], enriched_maps["talr_random_residual"]
        ),
    }
    true_rows = [row for (_, name), row in unique.items() if name == "talr_true_residual"]
    applied = [row for row in true_rows if row.get("injection_applied")]
    summary = {
        "reports": reports,
        "comparisons": comparisons,
        "injection": {
            "applied_samples": len(applied),
            "total_samples": len(true_rows),
            "mean_refinement_step": (
                sum(int(row["refinement_step"]) for row in applied) / len(applied)
                if applied else None
            ),
            "mean_target_soft_hard_norm": (
                sum(float(row["target_soft_hard_norm"]) for row in applied) / len(applied)
                if applied else None
            ),
        },
    }
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# TALR + Visual Residual: {args.dataset_name.upper()} {selection}",
        "",
        "Generation settings and residual strength are frozen. Selection is recorded in the heading.",
        "",
        "| Method | Sample Acc | Pair Acc | Failed |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "talr": "Frozen TALR",
        "talr_true_residual": "TALR + true visual residual",
        "talr_random_residual": "TALR + random residual",
    }
    for branch in BRANCHES:
        report = reports[branch]
        pair_acc = report.get("pair_accuracy")
        pair_text = f"{pair_acc:.2%}" if pair_acc is not None else "-"
        lines.append(
            f"| {labels[branch]} | {report['accuracy']:.2%} | "
            f"{pair_text} | {report['failed_extraction']} |"
        )
    lines.extend([
        "",
        "| Comparison | Fixed | Damaged | Net | Changed |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in comparisons.items():
        lines.append(
            f"| {name} | {row['fixed']} | {row['damaged']} | "
            f"{row['net']:+d} | {row['changed_predictions']} |"
        )
    lines.extend([
        "",
        f"Residual injection applied to {len(applied)}/{len(true_rows)} samples.",
        "",
    ])
    (merged_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
