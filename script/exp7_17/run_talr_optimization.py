#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from talr_analysis_common import load_jsonl, score_row, write_json, write_jsonl


DATASETS = {
    "vstar": ("vstar.jsonl", 191),
    "realworldqa_fixed200": (
        "realworldqa_fixed_mcq_random200_seed42.jsonl",
        200,
    ),
    "mmvp": ("mmvp.jsonl", 300),
    "visulogic300": ("visulogic.jsonl", 300),
}


def make_subset(source: Path, output: Path, count: int, seed: int = 42) -> None:
    if output.exists() and len(load_jsonl(output)) == count:
        return
    buckets = defaultdict(list)
    for row in load_jsonl(source):
        buckets[str(row.get("subtopic") or row.get("answer") or "unknown")].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count and buckets:
        for key in sorted(list(buckets)):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
            if not buckets[key]:
                del buckets[key]
    write_jsonl(output, selected)


def dataset_path_for(
    root: Path,
    output_root: Path,
    dataset: str,
) -> Path:
    source = root / "data" / DATASETS[dataset][0]
    if dataset != "visulogic300":
        return source
    subset = output_root / "eval_subsets" / "visulogic_first300.jsonl"
    if not subset.exists() or len(load_jsonl(subset)) != 300:
        write_jsonl(subset, load_jsonl(source)[:300])
    return subset


def common_command(args, dataset: Path, output_dir: Path, model: Path) -> list[str]:
    return [
        str(args.python),
        "main.py",
        "--model_name",
        str(model),
        "--dataset",
        str(dataset),
        "--output_dir",
        str(output_dir),
        "--method",
        "lead",
        "--alpha",
        "0.4",
        "--max_switch_count",
        "5",
        "--window_size",
        "128",
        "--cot_prompt_mode",
        "orign",
        "--no-do_sample",
        "--temperature",
        "0.6",
        "--top_p",
        "0.95",
        "--top_k",
        "20",
        "--seed",
        "42",
        "--max_new_tokens",
        "1024",
        "--device",
        "cuda",
        "--save_token_entropy",
        "--save_full_token_entropy",
        "--trace_topk",
        "0",
    ]


def method_args(name: str, window: int = 0, cap: int = 0) -> list[str]:
    if name == "full_lead":
        return []
    if name == "initial_transition":
        return ["--lead_initial_transition_only"]
    base = [
        "--lead_initial_transition_with_refinement",
        "--lead_refinement_window",
        str(window),
        "--lead_refinement_soft_cap",
        str(cap),
        "--lead_guard_candidate_only",
    ]
    if name == "guard_none":
        return [*base, "--lead_disable_answer_zone_lock"]
    if name == "answer_lock":
        return base
    if name == "answer_format":
        return [
            *base,
            "--lead_format_cooldown",
            "--format_cooldown_steps",
            "2",
            "--format_cooldown_min_step",
            "2",
        ]
    if name == "full_guard":
        return [
            *base,
            "--lead_format_cooldown",
            "--format_cooldown_steps",
            "2",
            "--format_cooldown_min_step",
            "2",
            "--lead_soft_veto_on_diffuse",
            "--lead_veto_min_step",
            "0",
            "--lead_veto_require_repeat_degen",
            "--lead_veto_repeat_ngram",
            "3",
            "--lead_veto_recent_repeat_window",
            "64",
            "--lead_veto_recent_repeat_tau",
            "0.35",
        ]
    if name.startswith("w"):
        return base
    raise ValueError(f"Unknown method name: {name}")


def complete(run_dir: Path, expected: int) -> bool:
    return (
        len(load_jsonl(run_dir / "results.jsonl")) == expected
        and (run_dir / "eval_report.json").exists()
        and (run_dir / "config.json").exists()
        and (run_dir / "token_entropy.jsonl").exists()
    )


def run_one(root: Path, command: list[str], run_dir: Path, expected: int) -> None:
    if complete(run_dir, expected):
        print(f"[SKIP] {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = run_dir / "runner.log"
    print(f"[START] {run_dir}")
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if result.returncode != 0 or not complete(run_dir, expected):
        raise RuntimeError(f"Run failed or incomplete: {run_dir}")
    print(f"[DONE] {run_dir}")


def run_jobs(root: Path, jobs: list[tuple], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_one, root, command, run_dir, expected)
            for command, run_dir, expected in jobs
        ]
        for future in as_completed(futures):
            future.result()


def summarize(run_dir: Path) -> dict:
    rows = load_jsonl(run_dir / "results.jsonl")
    scored = [score_row(row) for row in rows]
    valid = [item for item in scored if not item["runtime_error"]]
    lengths = [
        int(row.get("output_tokens") or 0)
        for row in rows
        if not row.get("error_type")
    ]
    return {
        "run_dir": str(run_dir),
        "samples": len(rows),
        "accuracy": mean(item["correct"] for item in valid) if valid else None,
        "failed": sum(item["failed_extraction"] for item in valid),
        "runtime_errors": sum(item["runtime_error"] for item in scored),
        "avg_tokens": mean(lengths) if lengths else None,
        "long": sum(value >= 256 for value in lengths),
        "maxed": sum(value >= 1024 for value in lengths),
    }


def rank_methods(
    method_dirs: dict[str, dict[str, Path]],
    baseline_dirs: dict[str, Path],
) -> tuple[list[str], dict]:
    details = {}
    for name, datasets in method_dirs.items():
        deltas = []
        penalties = []
        per_dataset = {}
        for dataset, run_dir in datasets.items():
            candidate = summarize(run_dir)
            baseline = summarize(baseline_dirs[dataset])
            delta = candidate["accuracy"] - baseline["accuracy"]
            failed_extra = max(0, candidate["failed"] - baseline["failed"])
            maxed_extra = max(0, candidate["maxed"] - baseline["maxed"])
            penalty = failed_extra / candidate["samples"] + 0.25 * (
                maxed_extra / candidate["samples"]
            )
            deltas.append(delta)
            penalties.append(penalty)
            per_dataset[dataset] = {
                "candidate": candidate,
                "full_lead": baseline,
                "delta": delta,
                "penalty": penalty,
            }
        worst = min(deltas)
        objective = mean(deltas) - mean(penalties) - max(0.0, -worst - 0.005)
        details[name] = {
            "objective": objective,
            "mean_delta": mean(deltas),
            "worst_delta": worst,
            "datasets": per_dataset,
        }
    ranking = sorted(
        details,
        key=lambda name: (
            details[name]["objective"],
            details[name]["mean_delta"],
            details[name]["worst_delta"],
        ),
        reverse=True,
    )
    return ranking, details


def build_jobs(
    args,
    stage_dir: Path,
    dataset_paths: dict[str, Path],
    methods: list[tuple[str, int, int]],
    model: Path,
) -> tuple[list[tuple], dict[str, dict[str, Path]]]:
    jobs = []
    method_dirs = defaultdict(dict)
    for name, window, cap in methods:
        for dataset, path in dataset_paths.items():
            expected = len(load_jsonl(path))
            run_dir = stage_dir / dataset / name
            command = [
                *common_command(args, path, run_dir, model),
                *method_args(name, window, cap),
            ]
            jobs.append((command, run_dir, expected))
            method_dirs[name][dataset] = run_dir
    return jobs, dict(method_dirs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase",
        choices=["optimize", "validate", "all"],
        default="all",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(args.python),
            "-m",
            "py_compile",
            "main.py",
            "lead/inference.py",
            "lead/generation_utils.py",
            "script/exp7_17/analyze_talr_components.py",
        ],
        cwd=root,
        check=True,
    )

    dev_subset_dir = output_root / "dev_subsets"
    dev_subset_dir.mkdir(parents=True, exist_ok=True)
    dev_subsets = {}
    for dataset in ("vstar", "realworldqa_fixed200"):
        source = root / "data" / DATASETS[dataset][0]
        subset = dev_subset_dir / f"{dataset}_stratified64_seed42.jsonl"
        make_subset(source, subset, 64)
        dev_subsets[dataset] = subset

    locked_path = output_root / "locked_talr_config.json"
    if args.phase in {"optimize", "all"}:
        stage1_methods = [
            ("full_lead", 0, 0),
            ("initial_transition", 0, 0),
            *[
                (f"w{window}_k{cap}", window, cap)
                for window in (8, 16, 32)
                for cap in (1, 2)
            ],
        ]
        jobs, dirs = build_jobs(
            args,
            output_root / "stage1_subset64",
            dev_subsets,
            stage1_methods,
            args.model,
        )
        if args.dry_run:
            write_json(
                output_root / "stage1_dry_run.json",
                [
                    {
                        "command": command,
                        "run_dir": str(run_dir),
                        "expected_rows": expected,
                    }
                    for command, run_dir, expected in jobs
                ],
            )
            print(f"[DRY RUN] Wrote {output_root / 'stage1_dry_run.json'}")
            return 0
        run_jobs(root, jobs, args.workers)
        baseline_dirs = dirs["full_lead"]
        candidate_dirs = {
            key: value for key, value in dirs.items() if key.startswith("w")
        }
        ranking, stage1_details = rank_methods(candidate_dirs, baseline_dirs)
        write_json(
            output_root / "stage1_ranking.json",
            {"ranking": ranking, "details": stage1_details},
        )

        full_dev = {
            key: dataset_path_for(root, output_root, key)
            for key in ("vstar", "realworldqa_fixed200")
        }
        top3_specs = []
        for name in ranking[:3]:
            window, cap = name.removeprefix("w").split("_k")
            top3_specs.append((name, int(window), int(cap)))
        stage2_methods = [("full_lead", 0, 0), *top3_specs]
        jobs, dirs = build_jobs(
            args,
            output_root / "stage2_full_dev",
            full_dev,
            stage2_methods,
            args.model,
        )
        run_jobs(root, jobs, args.workers)
        ranking2, stage2_details = rank_methods(
            {key: value for key, value in dirs.items() if key != "full_lead"},
            dirs["full_lead"],
        )
        best_refiner = ranking2[0]
        window, cap = best_refiner.removeprefix("w").split("_k")
        window, cap = int(window), int(cap)

        guard_methods = [
            ("guard_none", window, cap),
            ("answer_lock", window, cap),
            ("answer_format", window, cap),
            ("full_guard", window, cap),
        ]
        jobs, guard_dirs = build_jobs(
            args,
            output_root / "stage3_guard_full_dev",
            full_dev,
            guard_methods,
            args.model,
        )
        run_jobs(root, jobs, args.workers)
        guard_ranking, guard_details = rank_methods(
            guard_dirs, dirs["full_lead"]
        )
        selected_guard = guard_ranking[0]
        locked = {
            "refinement_window": window,
            "refinement_soft_cap": cap,
            "selected_guard": selected_guard,
            "method_args": method_args(selected_guard, window, cap),
            "selection_protocol": {
                "development_model": Path(args.model).name,
                "development_datasets": ["vstar", "realworldqa_fixed200"],
                "validation_locked": True,
            },
            "stage1": {"ranking": ranking, "details": stage1_details},
            "stage2": {"ranking": ranking2, "details": stage2_details},
            "guard_selection": {
                "ranking": guard_ranking,
                "details": guard_details,
            },
        }
        write_json(locked_path, locked)
        print(f"[LOCKED] {locked_path}: {selected_guard}, W={window}, K={cap}")

    if args.phase in {"validate", "all"}:
        if not locked_path.exists():
            raise FileNotFoundError(f"Missing locked config: {locked_path}")
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        window = int(locked["refinement_window"])
        cap = int(locked["refinement_soft_cap"])
        guard = locked["selected_guard"]
        validation = [
            (args.model, "r1_rl", ["mmvp", "visulogic300"]),
        ]
        if args.vision_model:
            validation.append(
                (
                    args.vision_model,
                    "vision_r1",
                    [
                        "vstar",
                        "realworldqa_fixed200",
                        "mmvp",
                        "visulogic300",
                    ],
                )
            )
        validation_dirs = {}
        validation_references = {}
        for model, model_key, datasets in validation:
            dataset_paths = {
                key: dataset_path_for(root, output_root, key)
                for key in datasets
            }
            jobs, dirs = build_jobs(
                args,
                output_root / "locked_validation" / model_key,
                dataset_paths,
                [("full_lead", 0, 0), (guard, window, cap)],
                model,
            )
            run_jobs(root, jobs, args.workers)
            validation_dirs[model_key] = {
                dataset: str(run_dir)
                for dataset, run_dir in dirs[guard].items()
            }
            validation_references[model_key] = {
                dataset: str(run_dir)
                for dataset, run_dir in dirs["full_lead"].items()
            }
        write_json(
            output_root / "locked_validation_runs.json",
            {
                "talr": validation_dirs,
                "full_lead": validation_references,
            },
        )
        if args.reference_manifest and args.reference_manifest.exists():
            summary_dir = output_root / "locked_summary"
            subprocess.run(
                [
                    str(args.python),
                    "script/exp7_17/summarize_talr_optimization.py",
                    "--optimization-root",
                    str(output_root),
                    "--reference-manifest",
                    str(args.reference_manifest),
                    "--output-dir",
                    str(summary_dir),
                ],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    str(args.python),
                    "script/exp7_17/analyze_talr_components.py",
                    "--manifest",
                    str(summary_dir / "locked_comparison_manifest.json"),
                    "--output-dir",
                    str(summary_dir / "component_diagnosis"),
                    "--selected-per-group",
                    "20",
                ],
                cwd=root,
                check=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
