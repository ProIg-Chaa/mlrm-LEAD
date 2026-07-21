#!/usr/bin/env python3
"""Run the pre-registered R1 T=1.25 full VStar/MMVP A/B challenger."""

import run_talr_worst_tuning_queue as queue


def main():
    name = "r_w8k2_t125_l100"
    config = queue.R_CONFIGS[name]
    for dataset in ("vstar", "mmvp"):
        queue.run_one(
            "r1_rl",
            dataset,
            queue.FULL_DATASETS[dataset],
            name,
            config,
            "none",
            "phase_d_t125_ab",
        )


if __name__ == "__main__":
    main()
