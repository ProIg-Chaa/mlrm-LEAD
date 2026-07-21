#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_talr_optimization import (
    DATASETS,
    build_jobs,
    common_command,
    dataset_path_for,
    method_args,
    run_jobs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--r1-model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    window = int(locked["refinement_window"])
    cap = int(locked["refinement_soft_cap"])
    selected_guard = locked["selected_guard"]
    dataset_paths = {
        key: dataset_path_for(root, args.output_root, key)
        for key in DATASETS
    }

    methods = [
        ("initializer", ["--lead_initial_transition_only"]),
        (
            "initializer_refiner",
            method_args("answer_lock", window, cap),
        ),
        (
            "initializer_refiner_format",
            method_args("answer_format", window, cap),
        ),
        (
            "initializer_refiner_full_guard",
            method_args("full_guard", window, cap),
        ),
        (
            "initializer_guard_no_refiner",
            [
                "--lead_initial_transition_only",
                "--lead_format_cooldown",
                "--format_cooldown_steps",
                "2",
                "--format_cooldown_min_step",
                "2",
                "--lead_soft_veto_on_diffuse",
                "--lead_veto_min_step",
                "0",
                "--lead_veto_require_repeat_degen",
                "--lead_guard_candidate_only",
            ],
        ),
    ]
    if selected_guard != "full_guard":
        methods.append(
            (
                "selected_talr",
                method_args(selected_guard, window, cap),
            )
        )

    for model_key, model in (
        ("r1_onevision_7b_rl", args.r1_model),
        ("vision_r1_7b", args.vision_model),
    ):
        jobs = []
        for method_name, extra_args in methods:
            for dataset, dataset_path in dataset_paths.items():
                run_dir = (
                    args.output_root
                    / model_key
                    / dataset
                    / method_name
                )
                command = [
                    *common_command(args, dataset_path, run_dir, model),
                    *extra_args,
                ]
                jobs.append(
                    (
                        command,
                        run_dir,
                        DATASETS[dataset][1],
                    )
                )
        if args.dry_run:
            print(
                json.dumps(
                    [
                        {
                            "command": command,
                            "run_dir": str(run_dir),
                            "expected_rows": expected,
                        }
                        for command, run_dir, expected in jobs
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue
        run_jobs(root, jobs, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
