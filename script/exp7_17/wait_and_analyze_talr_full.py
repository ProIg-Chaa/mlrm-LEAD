#!/usr/bin/env python3
"""Wait for final_summary.json, then run CPU-only post-full analysis once."""

import subprocess
import time
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260718_talr_worst_cell_tuning"
)
REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
ANALYZER = REPO / "script/exp7_17/analyze_talr_post_full.py"


def main():
    final_summary = ROOT / "final_summary.json"
    output = ROOT / "post_full_decision.json"
    while not final_summary.exists():
        time.sleep(60)
    if output.exists():
        return
    subprocess.run([str(PYTHON), str(ANALYZER)], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
