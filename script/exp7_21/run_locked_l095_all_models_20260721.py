#!/usr/bin/env python3
"""Run the frozen W8K2-T1.25-L0.95-NoGuard configuration externally."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
sys.path.insert(0, str(REPO / "script/exp7_17"))

import run_talr_worst_tuning_queue as queue


ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260721_locked_l095_all_models"
)
CONFIG_NAME = "talr_w8k2_t125_l095_noguard"
CONFIG = (8, 2, 1.25, 0.95)
MODELS = {
    "vision_r1": Path("/dev/shm/wangzixu_models/Vision-R1-7B"),
    "openvlthinker": Path(
        "/root/autodl-tmp/gushuo/models/OpenVLThinker-7B"
    ),
}
DATASETS = {
    "vstar": REPO / "data/vstar.jsonl",
    "realworldqa": (
        REPO / "data/realworldqa_fixed_mcq_random200_seed42.jsonl"
    ),
    "mmvp": REPO / "data/mmvp.jsonl",
    "visulogic": Path(
        "/root/autodl-tmp/gushuo/outputs/experiments/"
        "20260718_talr_worst_cell_tuning/subsets/visulogic300.jsonl"
    ),
}
LANES = (
    ("vstar", "realworldqa"),
    ("mmvp", "visulogic"),
)


def audit() -> dict:
    report = {"config": CONFIG_NAME, "models": {}, "datasets": {}}
    for key, path in MODELS.items():
        config = path / "config.json"
        if not config.exists():
            raise FileNotFoundError(f"Incomplete model {key}: {config}")
        report["models"][key] = str(path)
    for key, path in DATASETS.items():
        rows = queue.load_jsonl(path)
        missing = [str(row.get("id")) for row in rows if not Path(row["image"]).exists()]
        if missing:
            raise RuntimeError(f"Missing images in {key}: {missing[:10]}")
        report["datasets"][key] = {"path": str(path), "rows": len(rows)}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_lane(model_key: str, names: tuple[str, ...]) -> None:
    for dataset_name in names:
        queue.run_one(
            model_key,
            dataset_name,
            DATASETS[dataset_name],
            CONFIG_NAME,
            CONFIG,
            "none",
            "runs",
        )


def summarize(manifest: dict) -> None:
    payload = {"manifest": manifest, "results": {}}
    lines = [
        "# Locked L0.95 External-Model Results",
        "",
        "Frozen configuration: W8K2-T1.25-L0.95-NoGuard.",
        "",
        "| Model | Dataset | Accuracy | Failed | Avg tokens | Refinements |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model_key in MODELS:
        payload["results"][model_key] = {}
        for dataset_name in DATASETS:
            run_dir = (
                ROOT / "runs" / model_key / dataset_name /
                f"{CONFIG_NAME}__none"
            )
            item = queue.metrics(run_dir)
            if dataset_name == "mmvp":
                item["specialized"] = json.loads(
                    (run_dir / "specialized_eval_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                item["accuracy"] = item["specialized"]["accuracy"]
            elif dataset_name == "realworldqa":
                item["specialized"] = json.loads(
                    (run_dir / "realworldqa_mcq_eval.json").read_text(
                        encoding="utf-8"
                    )
                )
                item["accuracy"] = item["specialized"]["accuracy"]
            payload["results"][model_key][dataset_name] = item
            lines.append(
                f"| {model_key} | {dataset_name} | "
                f"{100 * item['accuracy']:.2f}% | {item['failed']} | "
                f"{item['avg_tokens']:.1f} | {item['refinement_active']} |"
            )
    (ROOT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    queue.ROOT = ROOT
    queue.LOG = ROOT / "queue.log"
    queue.MODELS = MODELS
    manifest = audit()
    for model_key in MODELS:
        queue.log(f"MODEL START {model_key}")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_lane, model_key, lane) for lane in LANES]
            for future in futures:
                future.result()
        queue.log(f"MODEL DONE {model_key}")
    summarize(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
