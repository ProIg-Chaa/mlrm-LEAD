#!/usr/bin/env python3
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from talr_analysis_common import score_row

REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260718_talr_worst_cell_tuning"
)
SUBSETS = ROOT / "subsets"
LOG = ROOT / "queue.log"
MAX_WORKERS = max(1, int(os.environ.get("TALR_MAX_WORKERS", "2")))

MODELS = {
    "r1_rl": Path("/dev/shm/wangzixu_models/R1-Onevision-7B-RL"),
    "vision_r1": Path("/dev/shm/wangzixu_models/Vision-R1-7B"),
}
FULL_DATASETS = {
    "vstar": REPO / "data/vstar.jsonl",
    "realworldqa": REPO / "data/realworldqa_fixed_mcq_random200_seed42.jsonl",
    "mmvp": REPO / "data/mmvp.jsonl",
    "visulogic": SUBSETS / "visulogic300.jsonl",
}


def log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} | {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_subsets():
    SUBSETS.mkdir(parents=True, exist_ok=True)
    existing = Path(
        "/root/autodl-tmp/gushuo/outputs/experiments/"
        "20260717_talr_diagnosis_optimization/dev_subsets"
    )
    shutil.copy2(
        existing / "vstar_stratified64_seed42.jsonl",
        SUBSETS / "vstar64.jsonl",
    )
    shutil.copy2(
        existing / "realworldqa_fixed200_stratified64_seed42.jsonl",
        SUBSETS / "realworldqa64.jsonl",
    )

    mmvp = load_jsonl(FULL_DATASETS["mmvp"])
    pair_indices = list(range(len(mmvp) // 2))
    random.Random(42).shuffle(pair_indices)
    selected_pairs = set(pair_indices[:32])
    mmvp64 = [
        row for index, row in enumerate(mmvp) if index // 2 in selected_pairs
    ]
    write_jsonl(SUBSETS / "mmvp64_pairs.jsonl", mmvp64)

    visulogic = load_jsonl(REPO / "data/visulogic.jsonl")[:300]
    write_jsonl(SUBSETS / "visulogic300.jsonl", visulogic)
    random.Random(42).shuffle(visulogic)
    write_jsonl(SUBSETS / "visulogic64.jsonl", visulogic[:64])
    write_jsonl(SUBSETS / "vstar2.jsonl", load_jsonl(FULL_DATASETS["vstar"])[:2])


SCREEN_DATASETS = {
    "vstar": SUBSETS / "vstar64.jsonl",
    "realworldqa": SUBSETS / "realworldqa64.jsonl",
    "mmvp": SUBSETS / "mmvp64_pairs.jsonl",
    "visulogic": SUBSETS / "visulogic64.jsonl",
}

R_CONFIGS = {
    "r_base_w8k2_t100": (8, 2, 1.00, 1.00),
    "r_w8k1_t100": (8, 1, 1.00, 1.00),
    "r_w6k1_t075": (6, 1, 0.75, 1.00),
    "r_w4k1_t050": (4, 1, 0.50, 1.00),
    "r_w8k2_t125_l100": (8, 2, 1.25, 1.00),
    "r_w8k2_t150_l100": (8, 2, 1.50, 1.00),
    "r_w8k2_t100_l075": (8, 2, 1.00, 0.75),
    "r_w8k2_t100_l050": (8, 2, 1.00, 0.50),
}
V_CONFIGS = {
    "v_base_w8k2_t100": (8, 2, 1.00, 1.00),
    "v_w16k1_t075": (16, 1, 0.75, 1.00),
    "v_w16k1_t050": (16, 1, 0.50, 1.00),
    "v_w16k2_t075": (16, 2, 0.75, 1.00),
    "v_w16k2_t050": (16, 2, 0.50, 1.00),
    "v_w12k2_t050": (12, 2, 0.50, 1.00),
    "v_w16k2_t025": (16, 2, 0.25, 1.00),
}


def count_lines(path):
    if not Path(path).exists():
        return 0
    with Path(path).open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def complete(run_dir, expected):
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    if not (
        count_lines(results_path) == expected
        and (run_dir / "eval_report.json").exists()
        and (run_dir / "config.json").exists()
        and (run_dir / "token_entropy.jsonl").exists()
    ):
        return False
    return not any(row.get("error_type") for row in load_jsonl(results_path))


def method_args(config, guard):
    if config == "initial_transition":
        return ["--lead_initial_transition_only"]
    window, cap, threshold, mix_lambda = config
    args = [
        "--lead_initial_transition_with_refinement",
        "--lead_refinement_window", str(window),
        "--lead_refinement_soft_cap", str(cap),
        "--lead_refinement_entropy_threshold", str(threshold),
        "--lead_refinement_soft_mix_lambda", str(mix_lambda),
        "--lead_guard_candidate_only",
    ]
    if guard == "none":
        args.append("--lead_disable_answer_zone_lock")
    elif guard == "answer_format2":
        args.extend([
            "--lead_format_cooldown",
            "--format_cooldown_steps", "2",
            "--format_cooldown_min_step", "2",
        ])
    return args


def run_specialized(dataset_name, dataset_path, run_dir):
    if dataset_name == "mmvp":
        command = [
            str(PYTHON), "script/evaluate_specialized_results.py",
            "--dataset", str(dataset_path),
            "--results", str(run_dir / "results.jsonl"),
            "--mode", "mmvp",
            "--output_json", str(run_dir / "specialized_eval_report.json"),
            "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
        ]
        subprocess.run(command, cwd=REPO, check=True)
    elif dataset_name == "realworldqa":
        command = [
            str(PYTHON), "script/evaluate_realworldqa_mcq.py",
            "--dataset", str(dataset_path),
            "--results", str(run_dir / "results.jsonl"),
            "--output_json", str(run_dir / "realworldqa_mcq_eval.json"),
            "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
        ]
        subprocess.run(command, cwd=REPO, check=True)


def run_one(model_key, dataset_name, dataset_path, config_name, config, guard, phase):
    rows = load_jsonl(dataset_path)
    expected = len(rows)
    run_dir = ROOT / phase / model_key / dataset_name / f"{config_name}__{guard}"
    if complete(run_dir, expected):
        log(f"SKIP complete {run_dir}")
        return
    if run_dir.exists():
        backup = run_dir.with_name(run_dir.name + f".incomplete.{int(time.time())}")
        run_dir.rename(backup)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(PYTHON), "main.py",
        "--model_name", str(MODELS[model_key]),
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
        "--trace_topk", "0",
        "--output_dir", str(run_dir),
        *method_args(config, guard),
    ]
    log(f"START {model_key}/{dataset_name}/{config_name}/{guard}")
    with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Run failed ({result.returncode}): {run_dir}")
    if not complete(run_dir, expected):
        raise RuntimeError(f"Incomplete outputs: {run_dir}")
    run_specialized(dataset_name, dataset_path, run_dir)
    log(f"DONE {model_key}/{dataset_name}/{config_name}/{guard}")


def metrics(run_dir):
    rows = load_jsonl(Path(run_dir) / "results.jsonl")
    scored = [score_row(row) for row in rows]
    traces = load_jsonl(Path(run_dir) / "token_entropy.jsonl")
    lengths = [
        int(item["output_tokens"])
        for item in traces
        if item.get("output_tokens") is not None
    ]
    summaries = [item.get("entropy_summary", {}) for item in traces]
    return {
        "samples": len(rows),
        "accuracy": sum(bool(item["correct"]) for item in scored) / len(scored),
        "failed": sum(item.get("pred") is None for item in scored),
        "runtime_errors": sum(
            bool(row.get("error_type")) for row in rows
        ),
        "avg_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "long": sum(length >= 256 for length in lengths),
        "maxed": sum(length >= 1024 for length in lengths),
        "refinement_candidates": sum(
            item.get("lead_refinement_candidate_count", 0) for item in summaries
        ),
        "refinement_active": sum(
            item.get("lead_refinement_active_count", 0) for item in summaries
        ),
        "format_active": sum(
            item.get("format_cooldown_active_steps", 0) for item in summaries
        ),
        "veto": sum(item.get("lead_soft_veto_count", 0) for item in summaries),
    }


def select_config(model_key, config_names, datasets, phase):
    table = {}
    for name, config in config_names.items():
        per_dataset = {}
        for dataset_name in datasets:
            run_dir = ROOT / phase / model_key / dataset_name / f"{name}__none"
            per_dataset[dataset_name] = metrics(run_dir)
        if model_key == "r1_rl":
            objective = (
                2.0 * per_dataset["mmvp"]["accuracy"]
                + per_dataset["vstar"]["accuracy"]
            ) / 3.0
        else:
            objective = sum(
                item["accuracy"] for item in per_dataset.values()
            ) / len(per_dataset)
            objective -= 0.002 * sum(
                item["maxed"] for item in per_dataset.values()
            )
        table[name] = {
            "config": config,
            "objective": objective,
            "datasets": per_dataset,
        }
    selected = max(table, key=lambda name: table[name]["objective"])
    return selected, table


def select_guard(model_key, config_name, config, datasets):
    table = {}
    for guard in ["none", "answer_lock", "answer_format2"]:
        per_dataset = {}
        for dataset_name in datasets:
            run_dir = (
                ROOT / "phase_c_guard" / model_key / dataset_name
                / f"{config_name}__{guard}"
            )
            per_dataset[dataset_name] = metrics(run_dir)
        accuracy = sum(x["accuracy"] for x in per_dataset.values()) / len(per_dataset)
        stability = sum(x["long"] + 3 * x["maxed"] for x in per_dataset.values())
        table[guard] = {
            "objective": accuracy - 0.0005 * stability,
            "accuracy": accuracy,
            "stability": stability,
            "datasets": per_dataset,
        }
    selected = max(
        table,
        key=lambda name: (table[name]["objective"], name == "none"),
    )
    return selected, table


def wait_for_dynamic_runs():
    dynamic = Path(
        "/root/autodl-tmp/gushuo/outputs/experiments/"
        "20260718_dynamic_transition_two_full"
    )
    targets = [
        dynamic / "vstar_semantic_adaptive_tau080/eval_report.json",
        dynamic / "mmvp_semantic_adaptive_tau080/eval_report.json",
    ]
    while not all(path.exists() for path in targets):
        process = subprocess.run(
            ["pgrep", "-f", "20260718_dynamic_transition_two_full"],
            stdout=subprocess.DEVNULL,
        )
        if process.returncode != 0:
            log("Dynamic transition process ended before both reports appeared")
            break
        log("WAIT dynamic transition full runs")
        time.sleep(60)


def run_screening():
    def r_lane():
        for name, config in R_CONFIGS.items():
            for dataset_name in ["mmvp", "vstar"]:
                run_one(
                    "r1_rl", dataset_name, SCREEN_DATASETS[dataset_name],
                    name, config, "none", "phase_ab_screen",
                )
        run_one(
            "r1_rl", "mmvp", SCREEN_DATASETS["mmvp"],
            "initial_transition", "initial_transition", "none", "phase_ab_screen",
        )

    def v_lane():
        for name, config in V_CONFIGS.items():
            for dataset_name in ["vstar", "realworldqa", "mmvp", "visulogic"]:
                run_one(
                    "vision_r1", dataset_name, SCREEN_DATASETS[dataset_name],
                    name, config, "none", "phase_ab_screen",
                )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(r_lane), executor.submit(v_lane)]
        for future in futures:
            future.result()


def run_guard_phase(selected):
    def lane(model_key, datasets):
        name, config = selected[model_key]
        for guard in ["none", "answer_lock", "answer_format2"]:
            for dataset_name in datasets:
                run_one(
                    model_key, dataset_name, SCREEN_DATASETS[dataset_name],
                    name, config, guard, "phase_c_guard",
                )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(lane, "r1_rl", ["mmvp"]),
            executor.submit(lane, "vision_r1", ["realworldqa"]),
        ]
        for future in futures:
            future.result()
    # VisuLogic has much longer generations; run it alone to avoid dual-model OOM.
    lane("r1_rl", ["visulogic"])
    lane("vision_r1", ["visulogic"])


def run_full(selected, guards):
    def lane(model_key, dataset_names):
        name, config = selected[model_key]
        guard = guards[model_key]
        for dataset_name in dataset_names:
            dataset_path = FULL_DATASETS[dataset_name]
            run_one(
                model_key, dataset_name, dataset_path,
                name, config, guard, "phase_d_full",
            )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        regular = ["vstar", "realworldqa", "mmvp"]
        futures = [executor.submit(lane, key, regular) for key in MODELS]
        for future in futures:
            future.result()
    lane("r1_rl", ["visulogic"])
    lane("vision_r1", ["visulogic"])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prepare_subsets()
    subprocess.run(
        [str(PYTHON), "-m", "py_compile", "main.py",
         "lead/inference.py", "lead/generation_utils.py"],
        cwd=REPO,
        check=True,
    )
    wait_for_dynamic_runs()
    run_screening()

    r_selected, r_table = select_config(
        "r1_rl", R_CONFIGS, ["mmvp", "vstar"], "phase_ab_screen"
    )
    v_selected, v_table = select_config(
        "vision_r1", V_CONFIGS,
        ["vstar", "realworldqa", "mmvp", "visulogic"],
        "phase_ab_screen",
    )
    selected = {
        "r1_rl": (r_selected, R_CONFIGS[r_selected]),
        "vision_r1": (v_selected, V_CONFIGS[v_selected]),
    }
    selection = {
        "selected_configs": selected,
        "r1_table": r_table,
        "vision_table": v_table,
    }
    (ROOT / "phase_ab_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"SELECT configs: {selected}")

    run_guard_phase(selected)
    r_guard, r_guard_table = select_guard(
        "r1_rl", r_selected, R_CONFIGS[r_selected], ["mmvp", "visulogic"]
    )
    v_guard, v_guard_table = select_guard(
        "vision_r1", v_selected, V_CONFIGS[v_selected],
        ["realworldqa", "visulogic"],
    )
    guards = {"r1_rl": r_guard, "vision_r1": v_guard}
    guard_selection = {
        "selected_guards": guards,
        "r1_table": r_guard_table,
        "vision_table": v_guard_table,
    }
    (ROOT / "phase_c_guard_selection.json").write_text(
        json.dumps(guard_selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"SELECT guards: {guards}")

    run_full(selected, guards)
    summary = {
        "selected_configs": selected,
        "selected_guards": guards,
        "full_results": {},
    }
    for model_key in MODELS:
        name, _ = selected[model_key]
        guard = guards[model_key]
        summary["full_results"][model_key] = {}
        for dataset_name in FULL_DATASETS:
            run_dir = (
                ROOT / "phase_d_full" / model_key / dataset_name
                / f"{name}__{guard}"
            )
            summary["full_results"][model_key][dataset_name] = metrics(run_dir)
    (ROOT / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("ALL TALR tuning and full validation runs completed")


if __name__ == "__main__":
    main()
