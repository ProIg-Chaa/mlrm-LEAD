#!/usr/bin/env python3
"""Full dynamic-handoff validation for the frozen W8K2 TALR refiner."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
MODEL = Path("/dev/shm/wangzixu_models/R1-Onevision-7B-RL")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260720_talr_dynamic_transition_full"
)
DATASETS = {
    "vstar": (REPO / "data/vstar.jsonl", 191),
    "mmvp": (REPO / "data/mmvp.jsonl", 300),
}
VARIANTS = {
    "semantic_adaptive_tau080": [
        "--lead_transition_semantic_adaptive",
        "--lead_transition_semantic_entropy_threshold", "0.80",
        "--lead_transition_semantic_max_extra_steps", "1",
    ],
    "rolling_w2_r050_max4": [
        "--lead_transition_dynamic_entropy_window", "2",
        "--lead_transition_dynamic_entropy_ratio", "0.50",
        "--lead_transition_dynamic_min_history", "2",
        "--lead_transition_dynamic_max_step", "4",
    ],
}


def log(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def complete(run_dir: Path, expected: int) -> bool:
    rows = load_jsonl(run_dir / "results.jsonl")
    if len(rows) != expected or not (run_dir / "eval_report.json").exists():
        return False
    return not any(row.get("error_type") or row.get("error") for row in rows)


def run_one(dataset: str, variant: str) -> None:
    dataset_path, expected = DATASETS[dataset]
    run_dir = ROOT / "runs" / dataset / variant
    if complete(run_dir, expected):
        log(f"SKIP {dataset}/{variant}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), "main.py",
        "--model_name", str(MODEL),
        "--dataset", str(dataset_path),
        "--method", "lead",
        "--alpha", "0.4",
        "--max_switch_count", "5",
        "--window_size", "128",
        "--cot_prompt_mode", "orign",
        "--no-do_sample",
        "--temperature", "0.6",
        "--top_p", "0.95",
        "--top_k", "20",
        "--seed", "42",
        "--max_new_tokens", "1024",
        "--device", "cuda",
        "--save_token_entropy",
        "--save_full_token_entropy",
        "--trace_topk", "0",
        "--output_dir", str(run_dir),
        "--lead_initial_transition_with_refinement",
        "--lead_refinement_window", "8",
        "--lead_refinement_soft_cap", "2",
        "--lead_refinement_entropy_threshold", "1.25",
        "--lead_refinement_soft_mix_lambda", "1.0",
        "--lead_guard_candidate_only",
        "--lead_disable_answer_zone_lock",
        *VARIANTS[variant],
    ]
    log(f"START {dataset}/{variant}")
    environment = os.environ.copy()
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with (run_dir / "runner.log").open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if dataset == "mmvp":
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_specialized_results.py",
                "--dataset", str(dataset_path),
                "--results", str(run_dir / "results.jsonl"),
                "--output_json", str(run_dir / "specialized_eval_report.json"),
                "--output_results_jsonl",
                str(run_dir / "specialized_eval_rows.jsonl"),
            ],
            cwd=REPO,
            check=True,
        )
    if not complete(run_dir, expected):
        raise RuntimeError(f"Incomplete run: {run_dir}")
    log(f"DONE {dataset}/{variant}")


def run_lane(dataset: str) -> None:
    for variant in VARIANTS:
        run_one(dataset, variant)


def summarize() -> None:
    summary = {}
    for dataset, (_, expected) in DATASETS.items():
        summary[dataset] = {}
        for variant in VARIANTS:
            run_dir = ROOT / "runs" / dataset / variant
            rows = load_jsonl(run_dir / "results.jsonl")
            report_path = (
                run_dir / "specialized_eval_report.json"
                if dataset == "mmvp"
                else run_dir / "eval_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary[dataset][variant] = {
                "rows": len(rows),
                "expected": expected,
                "runtime_errors": sum(
                    bool(row.get("error_type") or row.get("error"))
                    for row in rows
                ),
                "report": report,
            }
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    subprocess.run(
        [
            str(PYTHON), "-m", "py_compile",
            "main.py", "lead/inference.py", "lead/generation_utils.py",
        ],
        cwd=REPO,
        check=True,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_lane, dataset) for dataset in DATASETS]
        for future in futures:
            future.result()
    summarize()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
