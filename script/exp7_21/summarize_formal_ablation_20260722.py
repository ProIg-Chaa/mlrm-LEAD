#!/usr/bin/env python3
"""Re-evaluate and summarize the frozen TALR formal ablation matrix."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260722_talr_formal_ablation"
)
sys.path.insert(0, str(REPO / "script/exp7_17"))
from talr_analysis_common import score_row  # noqa: E402


DATASETS = {
    "vstar": REPO / "data/vstar.jsonl",
    "mmvp": REPO / "data/mmvp.jsonl",
    "realworldqa": REPO / "data/realworldqa_fixed_mcq_random200_seed42.jsonl",
    "visulogic": Path(
        "/root/autodl-tmp/gushuo/outputs/experiments/"
        "20260718_talr_worst_cell_tuning/subsets/visulogic300.jsonl"
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_ci(values: list[int], repeats: int = 4000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(42)
    n = len(values)
    samples = []
    for _ in range(repeats):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return [samples[int(0.025 * repeats)], samples[int(0.975 * repeats)]]


def mcnemar_exact(fixed: int, damaged: int) -> float:
    discordant = fixed + damaged
    if discordant == 0:
        return 1.0
    lower = min(fixed, damaged)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def evaluate(run_dir: Path, dataset: str, label: str) -> tuple[dict, list[dict]]:
    eval_dir = ROOT / "unified_eval" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "mmvp":
        report = eval_dir / "specialized_eval_report.json"
        rows_path = eval_dir / "specialized_results.jsonl"
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_specialized_results.py",
                "--dataset", str(DATASETS[dataset]),
                "--results", str(run_dir / "results.jsonl"),
                "--mode", "mmvp",
                "--output_json", str(report),
                "--output_results_jsonl", str(rows_path),
            ], cwd=REPO, check=True,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        rows = load_jsonl(rows_path)
        scored = [
            {
                "id": str(row["id"]),
                "pred": row.get("specialized_pred"),
                "gold": row.get("specialized_gold"),
                "correct": bool(row.get("specialized_is_correct")),
            }
            for row in rows
        ]
        return data, scored
    if dataset == "realworldqa":
        report = eval_dir / "realworldqa_mcq_eval.json"
        rows_path = eval_dir / "specialized_results.jsonl"
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_realworldqa_mcq.py",
                "--dataset", str(DATASETS[dataset]),
                "--results", str(run_dir / "results.jsonl"),
                "--output_json", str(report),
                "--output_results_jsonl", str(rows_path),
            ], cwd=REPO, check=True,
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        rows = load_jsonl(rows_path)
        scored = [
            {
                "id": str(row["id"]),
                "pred": row.get("realworldqa_pred"),
                "gold": row.get("realworldqa_gold"),
                "correct": bool(row.get("realworldqa_is_correct")),
            }
            for row in rows
        ]
        return data, scored

    rows = load_jsonl(run_dir / "results.jsonl")
    scored_raw = [score_row(row) for row in rows]
    scored = [
        {
            "id": str(row["id"]),
            "pred": item.get("pred"),
            "gold": item.get("gold"),
            "correct": bool(item.get("correct")),
        }
        for row, item in zip(rows, scored_raw)
    ]
    report = {
        "mode": "corrected_last_answer",
        "accuracy": sum(item["correct"] for item in scored) / len(scored),
        "correct": sum(item["correct"] for item in scored),
        "total": len(scored),
        "failed_extraction": sum(item["pred"] is None for item in scored),
    }
    (eval_dir / "corrected_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(eval_dir / "corrected_results.jsonl", scored)
    return report, scored


def trace_metrics(run_dir: Path) -> dict:
    results = load_jsonl(run_dir / "results.jsonl")
    traces = load_jsonl(run_dir / "token_entropy.jsonl")
    lengths = [int(row.get("output_tokens") or 0) for row in results]
    summaries = [row.get("entropy_summary") or {} for row in traces]
    active = [int(item.get("lead_refinement_active_count", 0) or 0) for item in summaries]
    positions: list[int] = []
    for item in summaries:
        value = item.get("lead_refinement_active_positions") or item.get("refinement_positions") or []
        if isinstance(value, list):
            positions.extend(int(pos) for pos in value)
    return {
        "runtime_errors": sum(bool(row.get("error_type")) for row in results),
        "avg_tokens": sum(lengths) / len(lengths),
        "long_ge_256": sum(length >= 256 for length in lengths),
        "maxed_1024": sum(length >= 1024 for length in lengths),
        "refinement_candidates": sum(int(item.get("lead_refinement_candidate_count", 0) or 0) for item in summaries),
        "refinement_active": sum(active),
        "refinement_active_per_sample": sum(active) / len(results),
        "refinement_positions": positions,
        "soft_ratio_mean": sum(float(item.get("soft_ratio", 0.0) or 0.0) for item in summaries) / len(results),
        "format_cooldown_active": sum(int(item.get("format_cooldown_active_steps", 0) or 0) for item in summaries),
    }


def pairwise(left: list[dict], right: list[dict]) -> dict:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    if list(left_by_id) != list(right_by_id):
        raise RuntimeError("Pairwise ID mismatch")
    fixed: list[str] = []
    damaged: list[str] = []
    both_correct: list[str] = []
    both_wrong: list[str] = []
    prediction_agreement = 0
    for item_id, a in left_by_id.items():
        b = right_by_id[item_id]
        if a["pred"] == b["pred"]:
            prediction_agreement += 1
        if not a["correct"] and b["correct"]:
            fixed.append(item_id)
        elif a["correct"] and not b["correct"]:
            damaged.append(item_id)
        elif a["correct"] and b["correct"]:
            both_correct.append(item_id)
        else:
            both_wrong.append(item_id)
    return {
        "fixed": fixed,
        "damaged": damaged,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net": len(fixed) - len(damaged),
        "mcnemar_exact_p": mcnemar_exact(len(fixed), len(damaged)),
        "prediction_agreement": prediction_agreement / len(left),
    }


def main() -> int:
    manifest = json.loads((ROOT / "ablation_reuse_manifest.json").read_text(encoding="utf-8"))
    metrics: dict[str, dict] = {}
    predictions: dict[str, list[dict]] = {}
    for label, run_dir_text in manifest["selected"].items():
        model, dataset, method = label.split("/")
        run_dir = Path(run_dir_text)
        report, scored = evaluate(run_dir, dataset, label)
        ci = bootstrap_ci([int(row["correct"]) for row in scored])
        metrics[label] = {
            "model": model,
            "dataset": dataset,
            "method": method,
            "run_dir": str(run_dir),
            "accuracy": report["accuracy"],
            "correct": report["correct"],
            "total": report["total"],
            "failed_extraction": report.get("failed_extraction", 0),
            "pair_accuracy": report.get("pair_accuracy"),
            "bootstrap_95ci": ci,
            **trace_metrics(run_dir),
        }
        predictions[label] = scored

    comparisons: dict[str, dict] = {}
    for label in metrics:
        model, dataset, method = label.split("/")
        for baseline in ("cot", "full_lead", "initial_transition", "w8k2_l100"):
            base_label = f"{model}/{dataset}/{baseline}"
            if method == baseline or base_label not in predictions:
                continue
            key = f"{base_label} -> {label}"
            comparisons[key] = pairwise(predictions[base_label], predictions[label])

    payload = {"metrics": metrics, "comparisons": comparisons}
    (ROOT / "formal_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / "pairwise_fixed_damaged.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    methods = (
        "cot", "full_lead", "initial_soft_only", "initial_transition",
        "w8k2_l100", "w8k2_l095", "w8k2_l095_format2",
    )
    lines = [
        "# TALR Formal Ablation Summary", "",
        "Frozen main method: W8K2-T1.25-L0.95-NoGuard.", "",
        "| Model | Dataset | Method | Accuracy | MMVP pair | Failed | Runtime | Avg tokens | Refine/sample |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("r1_rl", "vision_r1", "openvl"):
        for dataset in DATASETS:
            for method in methods:
                item = metrics.get(f"{model}/{dataset}/{method}")
                if not item:
                    continue
                pair = "-" if item["pair_accuracy"] is None else f"{100 * item['pair_accuracy']:.2f}%"
                lines.append(
                    f"| {model} | {dataset} | {method} | {100 * item['accuracy']:.2f}% | "
                    f"{pair} | {item['failed_extraction']} | {item['runtime_errors']} | "
                    f"{item['avg_tokens']:.1f} | {item['refinement_active_per_sample']:.2f} |"
                )
    (ROOT / "formal_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sensitivity = [
        "# Window/Cap Sensitivity", "",
        "| Dataset | Method | Accuracy | Refine/sample | Avg tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ("vstar", "mmvp"):
        for method in ("initial_transition", "w8k1_l095", "w8k2_l095", "w16k2_l095"):
            item = metrics[f"r1_rl/{dataset}/{method}"]
            sensitivity.append(
                f"| {dataset} | {method} | {100 * item['accuracy']:.2f}% | "
                f"{item['refinement_active_per_sample']:.2f} | {item['avg_tokens']:.1f} |"
            )
    (ROOT / "window_cap_sensitivity.md").write_text("\n".join(sensitivity) + "\n", encoding="utf-8")

    cross = [
        "# Cross-model Contraction Validation", "",
        "| Model | Dataset | L1.00 | L0.95 | Delta | Prediction agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in ("vision_r1", "openvl"):
        for dataset in ("vstar", "mmvp"):
            a_label = f"{model}/{dataset}/w8k2_l100"
            b_label = f"{model}/{dataset}/w8k2_l095"
            a, b = metrics[a_label], metrics[b_label]
            comparison = comparisons[f"{a_label} -> {b_label}"]
            cross.append(
                f"| {model} | {dataset} | {100 * a['accuracy']:.2f}% | "
                f"{100 * b['accuracy']:.2f}% | {100 * (b['accuracy'] - a['accuracy']):+.2f} pp | "
                f"{100 * comparison['prediction_agreement']:.2f}% |"
            )
    (ROOT / "cross_model_contraction_validation.md").write_text("\n".join(cross) + "\n", encoding="utf-8")
    print(f"Wrote formal summaries under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
