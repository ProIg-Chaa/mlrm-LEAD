#!/usr/bin/env python3
"""Audit historical TALR runs and execute only missing formal ablations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260722_talr_formal_ablation"
)
LOG = ROOT / "queue.log"
VISULOGIC300 = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260718_talr_worst_cell_tuning/subsets/visulogic300.jsonl"
)

MODELS = {
    "r1_rl": Path("/dev/shm/wangzixu_models/R1-Onevision-7B-RL"),
    "vision_r1": Path("/dev/shm/wangzixu_models/Vision-R1-7B"),
    "openvl": Path("/root/autodl-tmp/gushuo/models/OpenVLThinker-7B"),
}
MODEL_BASENAMES = {
    "r1_rl": "R1-Onevision-7B-RL",
    "vision_r1": "Vision-R1-7B",
    "openvl": "OpenVLThinker-7B",
}
DATASETS = {
    "vstar": REPO / "data/vstar.jsonl",
    "mmvp": REPO / "data/mmvp.jsonl",
    "realworldqa": REPO / "data/realworldqa_fixed_mcq_random200_seed42.jsonl",
    "visulogic": VISULOGIC300,
}
HISTORY_ROOTS = (
    Path("/root/autodl-tmp/gushuo/outputs/experiments"),
    Path("/root/gushuo/outputs/experiments"),
    Path("/root/gushuo/migrated_results"),
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    method: str = "lead"
    extra: tuple[str, ...] = ()


METHODS = {
    "cot": MethodSpec("cot", method="cot_greedy"),
    "full_lead": MethodSpec("full_lead"),
    "initial_soft_only": MethodSpec(
        "initial_soft_only", extra=("--lead_initial_soft_only",)
    ),
    "initial_transition": MethodSpec(
        "initial_transition", extra=("--lead_initial_transition_only",)
    ),
    "w8k2_l100": MethodSpec(
        "w8k2_l100",
        extra=(
            "--lead_initial_transition_with_refinement",
            "--lead_refinement_window", "8",
            "--lead_refinement_soft_cap", "2",
            "--lead_refinement_entropy_threshold", "1.25",
            "--lead_refinement_soft_mix_lambda", "1.0",
            "--lead_guard_candidate_only",
            "--lead_disable_answer_zone_lock",
        ),
    ),
    "w8k2_l095": MethodSpec(
        "w8k2_l095",
        extra=(
            "--lead_initial_transition_with_refinement",
            "--lead_refinement_window", "8",
            "--lead_refinement_soft_cap", "2",
            "--lead_refinement_entropy_threshold", "1.25",
            "--lead_refinement_soft_mix_lambda", "0.95",
            "--lead_guard_candidate_only",
            "--lead_disable_answer_zone_lock",
        ),
    ),
    "w8k2_l095_format2": MethodSpec(
        "w8k2_l095_format2",
        extra=(
            "--lead_initial_transition_with_refinement",
            "--lead_refinement_window", "8",
            "--lead_refinement_soft_cap", "2",
            "--lead_refinement_entropy_threshold", "1.25",
            "--lead_refinement_soft_mix_lambda", "0.95",
            "--lead_disable_answer_zone_lock",
            "--lead_format_cooldown",
            "--format_cooldown_steps", "2",
            "--format_cooldown_min_step", "2",
        ),
    ),
    "w8k1_l095": MethodSpec(
        "w8k1_l095",
        extra=(
            "--lead_initial_transition_with_refinement",
            "--lead_refinement_window", "8",
            "--lead_refinement_soft_cap", "1",
            "--lead_refinement_entropy_threshold", "1.25",
            "--lead_refinement_soft_mix_lambda", "0.95",
            "--lead_guard_candidate_only",
            "--lead_disable_answer_zone_lock",
        ),
    ),
    "w16k2_l095": MethodSpec(
        "w16k2_l095",
        extra=(
            "--lead_initial_transition_with_refinement",
            "--lead_refinement_window", "16",
            "--lead_refinement_soft_cap", "2",
            "--lead_refinement_entropy_threshold", "1.25",
            "--lead_refinement_soft_mix_lambda", "0.95",
            "--lead_guard_candidate_only",
            "--lead_disable_answer_zone_lock",
        ),
    ),
}


def desired_cells() -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    main_methods = (
        "cot", "full_lead", "initial_soft_only", "initial_transition",
        "w8k2_l100", "w8k2_l095", "w8k2_l095_format2",
    )
    for dataset in DATASETS:
        cells.extend(("r1_rl", dataset, method) for method in main_methods)
    for dataset in ("vstar", "mmvp"):
        cells.extend(
            ("r1_rl", dataset, method)
            for method in ("w8k1_l095", "w16k2_l095")
        )
    for model in ("vision_r1", "openvl"):
        for dataset in ("vstar", "mmvp"):
            cells.extend(
                (model, dataset, method)
                for method in (
                    "cot", "full_lead", "initial_transition",
                    "w8k2_l100", "w8k2_l095",
                )
            )
    return cells


def log(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(row: dict) -> str:
    return str(row.get("id"))


def dataset_signature(rows: list[dict]) -> str:
    payload = [
        {
            "id": row_id(row),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "options": row.get("options"),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def semantic_dataset_match(results: list[dict], target: list[dict]) -> tuple[bool, str]:
    if len(results) != len(target):
        return False, f"row_count:{len(results)}!={len(target)}"
    if [row_id(row) for row in results] != [row_id(row) for row in target]:
        return False, "id_order_mismatch"
    target_by_id = {row_id(row): row for row in target}
    for result in results:
        reference = target_by_id[row_id(result)]
        for key in ("question", "answer", "options"):
            if key in result and result.get(key) != reference.get(key):
                return False, f"content_mismatch:{row_id(result)}:{key}"
    return True, "matched_ids_and_content"


def value(config: dict, key: str, default=None):
    return config.get(key, default)


def common_config_errors(config: dict, model: str) -> list[str]:
    errors = []
    if Path(str(config.get("model_name", ""))).name != MODEL_BASENAMES[model]:
        errors.append("checkpoint")
    expected = {
        "cot_prompt_mode": "orign",
        "do_sample": False,
        "seed": 42,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    for key, wanted in expected.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            if actual is None or abs(float(actual) - wanted) > 1e-9:
                errors.append(key)
        elif actual != wanted:
            errors.append(key)
    return errors


def method_config_errors(config: dict, method_key: str) -> list[str]:
    errors: list[str] = []
    if method_key == "cot":
        if config.get("method") != "cot_greedy":
            errors.append("method")
        return errors
    if config.get("method") != "lead":
        errors.append("method")
    for key, wanted in {"alpha": 0.4, "max_switch_count": 5, "window_size": 128}.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            if actual is None or abs(float(actual) - wanted) > 1e-9:
                errors.append(key)
        elif actual != wanted:
            errors.append(key)

    variant_flags = {
        "lead_initial_soft_only": method_key == "initial_soft_only",
        "lead_initial_transition_only": method_key == "initial_transition",
        "lead_initial_transition_with_refinement": method_key.startswith("w"),
    }
    for key, wanted in variant_flags.items():
        if bool(value(config, key, False)) != wanted:
            errors.append(key)
    if method_key == "full_lead" and any(variant_flags.values()):
        errors.append("full_lead_variant")
    if method_key.startswith("w"):
        expected = {
            "lead_refinement_window": 16 if method_key == "w16k2_l095" else 8,
            "lead_refinement_soft_cap": 1 if method_key == "w8k1_l095" else 2,
            "lead_refinement_entropy_threshold": 1.25,
            "lead_refinement_soft_mix_lambda": 1.0 if method_key == "w8k2_l100" else 0.95,
            "lead_disable_answer_zone_lock": True,
            "lead_format_cooldown": method_key == "w8k2_l095_format2",
        }
        for key, wanted in expected.items():
            actual = value(config, key, False if isinstance(wanted, bool) else None)
            if isinstance(wanted, float):
                if actual is None or abs(float(actual) - wanted) > 1e-9:
                    errors.append(key)
            elif actual != wanted:
                errors.append(key)
        if method_key == "w8k2_l095_format2":
            if int(value(config, "format_cooldown_steps", 0)) != 2:
                errors.append("format_cooldown_steps")
            if int(value(config, "format_cooldown_min_step", 0)) != 2:
                errors.append("format_cooldown_min_step")
        elif bool(value(config, "lead_format_cooldown", False)):
            errors.append("unexpected_format_guard")
    return errors


def discover_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen = set()
    for root in HISTORY_ROOTS:
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            run_dir = config_path.parent.resolve()
            if run_dir not in seen and (run_dir / "results.jsonl").exists():
                seen.add(run_dir)
                candidates.append(run_dir)
    return candidates


def audit_candidate(
    run_dir: Path,
    model: str,
    dataset: str,
    method: str,
    targets: dict[str, list[dict]],
) -> dict:
    reasons: list[str] = []
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        rows = load_jsonl(run_dir / "results.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        return {"run_dir": str(run_dir), "eligible": False, "reasons": [str(exc)]}
    reasons.extend(common_config_errors(config, model))
    reasons.extend(method_config_errors(config, method))
    matched, detail = semantic_dataset_match(rows, targets[dataset])
    if not matched:
        reasons.append(detail)
    runtime_errors = sum(bool(row.get("error_type")) for row in rows)
    if runtime_errors:
        reasons.append(f"runtime_errors:{runtime_errors}")
    for required in ("eval_report.json", "token_entropy.jsonl"):
        if not (run_dir / required).exists():
            reasons.append(f"missing:{required}")
    return {
        "run_dir": str(run_dir),
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "rows": len(rows),
        "runtime_errors": runtime_errors,
        "mtime": (run_dir / "results.jsonl").stat().st_mtime,
        "processor_binding": "checkpoint-coupled",
        "semantic_equivalence_notes": (
            ["lead_guard_candidate_only is inert when all guards are disabled"]
            if method in {"w8k2_l100", "w8k2_l095", "w8k1_l095", "w16k2_l095"}
            else []
        ),
    }


def output_dir(model: str, dataset: str, method: str) -> Path:
    return ROOT / "new_runs" / model / dataset / method


def run_specialized(dataset: str, dataset_path: Path, run_dir: Path) -> None:
    if dataset == "mmvp":
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_specialized_results.py",
                "--dataset", str(dataset_path),
                "--results", str(run_dir / "results.jsonl"),
                "--mode", "mmvp",
                "--output_json", str(run_dir / "specialized_eval_report.json"),
                "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
            ], cwd=REPO, check=True,
        )
    elif dataset == "realworldqa":
        subprocess.run(
            [
                str(PYTHON), "script/evaluate_realworldqa_mcq.py",
                "--dataset", str(dataset_path),
                "--results", str(run_dir / "results.jsonl"),
                "--output_json", str(run_dir / "realworldqa_mcq_eval.json"),
                "--output_results_jsonl", str(run_dir / "specialized_results.jsonl"),
            ], cwd=REPO, check=True,
        )


def run_cell(model: str, dataset: str, method_key: str) -> Path:
    method = METHODS[method_key]
    run_dir = output_dir(model, dataset, method_key)
    expected = len(load_jsonl(DATASETS[dataset]))
    if (run_dir / "results.jsonl").exists():
        rows = load_jsonl(run_dir / "results.jsonl")
        if len(rows) == expected and not any(row.get("error_type") for row in rows):
            log(f"SKIP complete {model}/{dataset}/{method_key}")
            return run_dir
        backup = run_dir.with_name(run_dir.name + f".incomplete.{int(time.time())}")
        run_dir.rename(backup)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), "main.py",
        "--model_name", str(MODELS[model]),
        "--dataset", str(DATASETS[dataset]),
        "--method", method.method,
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
    ]
    if method.method == "lead":
        command.extend(
            ["--alpha", "0.4", "--max_switch_count", "5", "--window_size", "128"]
        )
    command.extend(method.extra)
    log(f"START {model}/{dataset}/{method_key}")
    with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Run failed ({result.returncode}): {run_dir}")
    rows = load_jsonl(run_dir / "results.jsonl")
    if len(rows) != expected or any(row.get("error_type") for row in rows):
        raise RuntimeError(f"Incomplete or runtime-error result: {run_dir}")
    run_specialized(dataset, DATASETS[dataset], run_dir)
    log(f"DONE {model}/{dataset}/{method_key}")
    return run_dir


def build_manifest() -> dict:
    targets = {key: load_jsonl(path) for key, path in DATASETS.items()}
    candidates = discover_candidates()
    selected: dict[str, str] = {}
    audit: dict[str, dict] = {}
    cells = desired_cells()
    for model, dataset, method in cells:
        label = f"{model}/{dataset}/{method}"
        matches = [
            audit_candidate(path, model, dataset, method, targets)
            for path in candidates
        ]
        eligible = [item for item in matches if item["eligible"]]
        eligible.sort(key=lambda item: item["mtime"], reverse=True)
        if eligible:
            selected[label] = eligible[0]["run_dir"]
        audit[label] = {
            "selected": selected.get(label),
            "eligible_candidates": eligible,
            "checked_candidates": len(matches),
        }
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "dataset_signatures": {
            key: dataset_signature(rows) for key, rows in targets.items()
        },
        "cells": ["/".join(cell) for cell in cells],
        "selected": selected,
        "audit": audit,
        "missing": ["/".join(cell) for cell in cells if "/".join(cell) not in selected],
        "reuse_policy": {
            "exact_generation": "checkpoint basename, prompt, greedy, seed, decoding and method parameters",
            "dataset": "ordered IDs plus question/answer/options",
            "processor": "loaded from the same named checkpoint; historical config has no separate processor field",
            "runtime_errors_allowed": 0,
            "candidate_only_exception": "semantically inert for NoGuard runs",
        },
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "ablation_reuse_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for key, path in MODELS.items():
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Incomplete model {key}: {path}")
    subprocess.run(
        [str(PYTHON), "-m", "py_compile", "main.py", "lead/inference.py", "lead/generation_utils.py"],
        cwd=REPO,
        check=True,
    )
    manifest = build_manifest()
    selected = dict(manifest["selected"])
    for label in manifest["cells"]:
        if label in selected:
            log(f"REUSE {label} -> {selected[label]}")
            continue
        model, dataset, method = label.split("/")
        run_dir = run_cell(model, dataset, method)
        selected[label] = str(run_dir)
        manifest["selected"] = selected
        manifest["missing"] = [item for item in manifest["cells"] if item not in selected]
        (ROOT / "ablation_reuse_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    log("RUN MATRIX COMPLETE; starting unified evaluation")
    subprocess.run(
        [str(PYTHON), "script/exp7_21/summarize_formal_ablation_20260722.py"],
        cwd=REPO,
        check=True,
    )
    log("FORMAL ABLATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
