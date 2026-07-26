#!/usr/bin/env python3
"""Run the frozen TALR T1.25 configuration on the stable expansion suite."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp7_17"))

import run_talr_worst_tuning_queue as queue
from talr_analysis_common import score_row


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260722_talr_l095_main_expansion"
)
CONFIG_NAME = "talr_w8k2_t125_l095_noguard"
CONFIG = (8, 2, 1.25, 0.95)
DATASETS = {
    "mmk12_biology": REPO / "data/mmk12_biology.jsonl",
    "pope_adversarial": REPO / "data/pope_adversarial_balanced500_seed42.jsonl",
    "pope_popular": REPO / "data/pope_popular_balanced500_seed42.jsonl",
    "pope_random": REPO / "data/pope_random_balanced500_seed42.jsonl",
}
MODEL_ORDER = ('r1_rl',)


def audit_datasets() -> dict:
    audit = {}
    for name, path in DATASETS.items():
        rows = queue.load_jsonl(path)
        missing_images = [
            str(row.get("id"))
            for row in rows
            if not row.get("image") or not Path(row["image"]).exists()
        ]
        empty_answers = [
            str(row.get("id"))
            for row in rows
            if not str(row.get("answer") or "").strip()
        ]
        audit[name] = {
            "rows": len(rows),
            "missing_images": len(missing_images),
            "empty_answers": len(empty_answers),
            "first_missing_ids": missing_images[:10],
        }
        if missing_images or empty_answers:
            raise RuntimeError(f"Dataset audit failed for {name}: {audit[name]}")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def run_dir(model: str, dataset: str) -> Path:
    return ROOT / "runs" / model / dataset / f"{CONFIG_NAME}__none"


def run_lane(model: str) -> None:
    for dataset, path in DATASETS.items():
        queue.run_one(
            model,
            dataset,
            path,
            CONFIG_NAME,
            CONFIG,
            "none",
            "runs",
        )


def pope_metrics(path: Path) -> dict:
    rows = queue.load_jsonl(path / "results.jsonl")
    tp = tn = fp = fn = failed = 0
    for row in rows:
        scored = score_row(row)
        gold = scored["gold"]
        pred = scored["pred"]
        if pred not in {"A", "B"}:
            failed += 1
        if gold == "A" and pred == "A":
            tp += 1
        elif gold == "B" and pred == "B":
            tn += 1
        elif gold == "B" and pred == "A":
            fp += 1
        elif gold == "A":
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "failed": failed,
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / len(rows),
    }


def summarize(audit: dict) -> None:
    payload = {
        "config": {
            "name": CONFIG_NAME,
            "window": 8,
            "soft_cap": 2,
            "entropy_threshold": 1.25,
            "guard": "none",
        },
        "audit": audit,
        "models": {},
    }
    for model in MODEL_ORDER:
        model_results = {}
        for dataset in DATASETS:
            path = run_dir(model, dataset)
            item = queue.metrics(path)
            if dataset.startswith("pope_"):
                item["pope"] = pope_metrics(path)
                item["accuracy"] = item["pope"]["accuracy"]
                item["failed"] = item["pope"]["failed"]
            model_results[dataset] = item
        payload["models"][model] = model_results
    (ROOT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TALR T1.25 Full Expansion",
        "",
        "Frozen configuration: W8K2-T1.25-L0.95-NoGuard.",
        "",
        "| Model | Dataset | Accuracy | Failed | Avg tokens | "
        "Refinement events |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, datasets in payload["models"].items():
        for dataset, item in datasets.items():
            lines.append(
                f"| {model} | {dataset} | {100 * item['accuracy']:.2f}% | "
                f"{item['failed']} | {item['avg_tokens']:.1f} | "
                f"{item['refinement_active']} |"
            )
    (ROOT / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    queue.ROOT = ROOT
    queue.LOG = ROOT / "queue.log"
    audit = audit_datasets()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_lane, model)
            for model in MODEL_ORDER
        ]
        for future in futures:
            future.result()
    summarize(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
