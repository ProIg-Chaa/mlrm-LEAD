#!/usr/bin/env python3
"""Resume the Vision-R1 TALR expansion lane after the dual-process OOM."""

import sys
from pathlib import Path

REPO = Path("/root/gushuo/proj/mlrm-LEAD")
sys.path.insert(0, str(REPO / "script/exp7_19"))

import run_t125_full_expansion as runner


def main() -> int:
    runner.queue.ROOT = runner.ROOT
    runner.queue.LOG = runner.ROOT / "queue.log"
    audit = runner.audit_datasets()
    runner.run_lane("vision_r1")
    runner.summarize(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
