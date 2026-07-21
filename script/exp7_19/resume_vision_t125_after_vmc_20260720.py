#!/usr/bin/env python3
"""Resume the Vision-R1 frozen TALR expansion after VMCBench auditing."""

import sys
from pathlib import Path

REPO = Path("/root/gushuo/proj/mlrm-LEAD")
sys.path.insert(0, str(REPO / "script/exp7_19"))

import run_t125_full_expansion as runner


REMAINING = (
    "mmk12_math",
    "mmk12_physics",
    "mmk12_chemistry",
    "mmk12_biology",
    "pope_adversarial",
    "pope_popular",
    "pope_random",
)


def main() -> int:
    runner.queue.ROOT = runner.ROOT
    runner.queue.LOG = runner.ROOT / "queue.log"
    runner.audit_datasets()
    for dataset in REMAINING:
        runner.queue.run_one(
            "vision_r1",
            dataset,
            runner.DATASETS[dataset],
            runner.CONFIG_NAME,
            runner.CONFIG,
            "none",
            "runs",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
