#!/usr/bin/env python3
"""Repair only runtime-error rows and create an immutable merged run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict) -> str:
    return str(row.get("id"))


def stable_hash(rows: list[dict]) -> str:
    payload = [
        {
            "id": row_id(row),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "options": row.get("options"),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def method_args(config: dict) -> list[str]:
    args: list[str] = []
    flags = (
        "lead_initial_soft_only",
        "lead_initial_transition_only",
        "lead_initial_transition_with_refinement",
        "lead_guard_candidate_only",
        "lead_disable_answer_zone_lock",
        "lead_format_cooldown",
        "lead_soft_veto_on_diffuse",
    )
    for key in flags:
        if config.get(key):
            args.append("--" + key)
    values = (
        "lead_refinement_window",
        "lead_refinement_soft_cap",
        "lead_refinement_entropy_threshold",
        "lead_refinement_soft_mix_lambda",
        "format_cooldown_steps",
        "format_cooldown_min_step",
    )
    for key in values:
        if key in config:
            args.extend(["--" + key, str(config[key])])
    return args


def run_specialized(dataset: Path, run_dir: Path) -> None:
    name = dataset.name.casefold()
    if "mmvp" in name:
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_specialized_results.py",
                "--dataset", str(dataset),
                "--results", str(run_dir / "results.jsonl"),
                "--mode", "mmvp",
                "--output_json", str(run_dir / "specialized_eval_report.json"),
                "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
            ],
            cwd=REPO,
            check=True,
        )
    elif "realworldqa" in name:
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_realworldqa_mcq.py",
                "--dataset", str(dataset),
                "--results", str(run_dir / "results.jsonl"),
                "--output_json", str(run_dir / "realworldqa_mcq_eval.json"),
                "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
            ],
            cwd=REPO,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-name", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_run.resolve()
    source_rows = load_jsonl(source / "results.jsonl")
    dataset_rows = load_jsonl(args.dataset)
    target_by_id = {row_id(row): row for row in dataset_rows}
    error_ids = [row_id(row) for row in source_rows if row.get("error_type")]
    missing = [item for item in error_ids if item not in target_by_id]
    if missing:
        raise RuntimeError(f"Runtime-error IDs absent from dataset: {missing}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "repair_manifest.json"
    if not error_ids:
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "no_repair_needed",
                    "source_run": str(source),
                    "runtime_error_ids": [],
                    "rows": len(source_rows),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"No runtime errors: {source}")
        return 0

    subset = output_root / "runtime_error_subset.jsonl"
    write_jsonl(subset, [target_by_id[item] for item in error_ids])
    repair_run = output_root / "selected_id_repair"
    merged_run = output_root / "repaired_merged"
    if repair_run.exists() or merged_run.exists():
        raise FileExistsError("Repair output already exists; preserve it and choose a new root")
    repair_run.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    command = [
        str(PYTHON), "main.py",
        "--model_name", str(args.model_name),
        "--dataset", str(subset),
        "--method", str(config["method"]),
        "--cot_prompt_mode", str(config.get("cot_prompt_mode", "orign")),
        "--no-do_sample",
        "--temperature", str(config.get("temperature", 0.6)),
        "--top_p", str(config.get("top_p", 0.95)),
        "--top_k", str(config.get("top_k", 20)),
        "--seed", str(config.get("seed", 42)),
        "--max_new_tokens", str(config.get("max_new_tokens", 1024)),
        "--device", "cuda",
        "--save_token_entropy",
        "--trace_topk", "0",
        "--output_dir", str(repair_run),
    ]
    if str(config["method"]) == "lead":
        command.extend(
            [
                "--alpha", str(config.get("alpha", 0.4)),
                "--max_switch_count", str(config.get("max_switch_count", 5)),
                "--window_size", str(config.get("window_size", 128)),
            ]
        )
        command.extend(method_args(config))
    with (repair_run / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)

    repaired_rows = load_jsonl(repair_run / "results.jsonl")
    repaired_by_id = {row_id(row): row for row in repaired_rows}
    if set(repaired_by_id) != set(error_ids):
        raise RuntimeError("Selected-ID repair returned an unexpected ID set")
    if any(row.get("error_type") for row in repaired_rows):
        raise RuntimeError("Selected-ID repair still contains runtime errors")

    merged_run.mkdir(parents=True)
    merged_rows = [repaired_by_id.get(row_id(row), row) for row in source_rows]
    if len(merged_rows) != len(source_rows):
        raise AssertionError("Merged row count changed")
    if len({row_id(row) for row in merged_rows}) != len(merged_rows):
        raise AssertionError("Merged results contain duplicate IDs")
    write_jsonl(merged_run / "results.jsonl", merged_rows)

    source_trace = load_jsonl(source / "token_entropy.jsonl")
    repaired_trace = load_jsonl(repair_run / "token_entropy.jsonl")
    repaired_trace_by_id = {row_id(row): row for row in repaired_trace}
    merged_trace = [repaired_trace_by_id.get(row_id(row), row) for row in source_trace]
    write_jsonl(merged_run / "token_entropy.jsonl", merged_trace)
    shutil.copy2(repair_run / "config.json", merged_run / "config.json")
    run_specialized(args.dataset, merged_run)

    unchanged_before = [row for row in source_rows if row_id(row) not in set(error_ids)]
    unchanged_after = [row for row in merged_rows if row_id(row) not in set(error_ids)]
    if unchanged_before != unchanged_after:
        raise AssertionError("A non-error row changed during merge")
    manifest = {
        "status": "repaired",
        "source_run": str(source),
        "repair_run": str(repair_run),
        "merged_run": str(merged_run),
        "dataset": str(args.dataset.resolve()),
        "runtime_error_ids": error_ids,
        "source_rows": len(source_rows),
        "merged_rows": len(merged_rows),
        "dataset_hash": stable_hash(dataset_rows),
        "non_error_rows_unchanged": True,
        "runtime_errors_after_merge": 0,
        "command": command,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Repaired {len(error_ids)} IDs into {merged_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
