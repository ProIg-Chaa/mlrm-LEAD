#!/usr/bin/env python3
"""Run the MMVP initializer control beside VStar, then resume the coordinator."""

from __future__ import annotations

import os
import signal

import run_talr_worst_tuning_queue as queue


def main() -> int:
    coordinator_pid = int(os.environ["TALR_COORDINATOR_PID"])
    try:
        queue.run_one(
            "r1_rl",
            "mmvp",
            queue.FULL_DATASETS["mmvp"],
            "initial_transition",
            "initial_transition",
            "none",
            "phase_f_initializer_control",
        )
    finally:
        try:
            os.kill(coordinator_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
