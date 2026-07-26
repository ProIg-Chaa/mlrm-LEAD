#!/usr/bin/env python3
"""Run the frozen L0.95 configuration sequentially for OpenVLThinker."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
sys.path.insert(0, str(REPO / "script/exp7_17"))
import run_talr_worst_tuning_queue as queue


ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260721_locked_l095_all_models"
)
MODEL_KEY = "openvlthinker"
MODEL = Path("/root/autodl-tmp/gushuo/models/OpenVLThinker-7B")
CONFIG_NAME = "talr_w8k2_t125_l095_noguard"
CONFIG = (8, 2, 1.25, 0.95)
DATASETS = {
    "vstar": REPO / "data/vstar.jsonl",
    "mmvp": REPO / "data/mmvp.jsonl",
    "realworldqa": REPO / "data/realworldqa_fixed_mcq_random200_seed42.jsonl",
    "visulogic": Path(
        "/root/autodl-tmp/gushuo/outputs/experiments/"
        "20260718_talr_worst_cell_tuning/subsets/visulogic300.jsonl"
    ),
}


def main() -> int:
    queue.ROOT = ROOT
    queue.LOG = ROOT / "openvl_sequential.log"
    queue.MODELS = {MODEL_KEY: MODEL}
    for name, dataset in DATASETS.items():
        queue.run_one(
            MODEL_KEY,
            name,
            dataset,
            CONFIG_NAME,
            CONFIG,
            "none",
            "runs",
        )
        run_dir = (
            ROOT / "runs" / MODEL_KEY / name /
            f"{CONFIG_NAME}__none"
        )
        if name in {"mmvp", "realworldqa"}:
            queue.run_specialized(name, dataset, run_dir)
    summary = {}
    for name in DATASETS:
        run_dir = (
            ROOT / "runs" / MODEL_KEY / name /
            f"{CONFIG_NAME}__none"
        )
        summary[name] = queue.metrics(run_dir)
        if name == "mmvp":
            summary[name]["specialized"] = json.loads(
                (run_dir / "specialized_eval_report.json").read_text(encoding="utf-8")
            )
        elif name == "realworldqa":
            summary[name]["specialized"] = json.loads(
                (run_dir / "realworldqa_mcq_eval.json").read_text(encoding="utf-8")
            )
    (ROOT / "openvl_sequential_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
