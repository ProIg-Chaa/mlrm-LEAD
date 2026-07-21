#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from talr_analysis_common import load_jsonl, write_json


EXPECTED = {
    "vstar": 191,
    "mmvp": 300,
    "realworldqa_fixed200": 200,
    "visulogic300": 300,
}


def dataset_key(config: dict, path: Path) -> str | None:
    name = Path(str(config.get("dataset") or "")).name.casefold()
    text = f"{name} {path}".casefold()
    if "realworldqa_fixed" in text:
        return "realworldqa_fixed200"
    if "visulogic" in text:
        return "visulogic300"
    if "mmvp" in text:
        return "mmvp"
    if "vstar" in text:
        return "vstar"
    return None


def model_key(config: dict, path: Path) -> str:
    value = Path(str(config.get("model_name") or "")).name.casefold()
    text = f"{value} {path}".casefold()
    if "vision-r1" in text or "vision_r1" in text:
        return "vision_r1_7b"
    if "r1-onevision-7b-rl" in text or "r1_onevision_7b_rl" in text:
        return "r1_onevision_7b_rl"
    return value or "unknown"


def method_key(config: dict, path: Path) -> str | None:
    text = str(path).casefold()
    method = str(config.get("method") or "").casefold()
    if method == "cot_greedy" or "cot_orign_greedy" in text:
        return "cot"
    if (
        config.get("lead_initial_transition_only")
        and not config.get("lead_disable_to_normal_transition")
        and not config.get("lead_disable_step0_linebreak_mix")
        and not config.get("lead_initial_transition_hard_boundary_only")
        and not config.get("lead_early_visual_anchor")
        and int(config.get("lead_initial_transition_delay_steps") or 0) == 0
    ):
        return "initial_transition"
    if config.get("lead_initial_transition_with_refinement"):
        return "true_talr"
    if (
        "transition_preserving_quota05_guard_min2" in text
        or "legacy_talr" in text
        or (
            float(config.get("lead_soft_quota_ratio") or 0.0) > 0
            and config.get("lead_format_cooldown")
        )
    ):
        return "legacy_talr"
    if (
        method == "lead"
        and not any(
        config.get(key)
        for key in (
            "lead_initial_soft_only",
            "lead_initial_transition_only",
            "lead_initial_transition_with_refinement",
        )
        )
        and not config.get("lead_force_normal")
        and float(config.get("lead_soft_quota_ratio") or 0.0) == 0.0
        and not config.get("lead_format_cooldown")
        and not config.get("lead_soft_veto_on_diffuse")
        and not config.get("lead_disable_to_normal_transition")
        and not config.get("lead_disable_step0_linebreak_mix")
        and not config.get("lead_initial_transition_hard_boundary_only")
        and not config.get("lead_early_visual_anchor")
        and int(config.get("lead_initial_transition_delay_steps") or 0) == 0
    ):
        return "full_lead"
    return None


def canonical_config(config: dict) -> tuple[bool, list[str]]:
    errors = []
    if bool(config.get("do_sample", True)):
        errors.append("do_sample")
    if int(config.get("seed", -1)) != 42:
        errors.append("seed")
    if int(config.get("max_new_tokens", -1)) != 1024:
        errors.append("max_new_tokens")
    if str(config.get("cot_prompt_mode", "orign")) != "orign":
        errors.append("cot_prompt_mode")
    if str(config.get("method", "")).startswith("lead"):
        if abs(float(config.get("alpha", -1)) - 0.4) > 1e-9:
            errors.append("alpha")
        if int(config.get("window_size", -1)) != 128:
            errors.append("window_size")
        if int(config.get("max_switch_count", -1)) != 5:
            errors.append("max_switch_count")
    return not errors, errors


def candidate_score(run_dir: Path, dataset: str, compatible: bool) -> tuple:
    rows = load_jsonl(run_dir / "results.jsonl")
    expected = EXPECTED[dataset]
    runtime_errors = sum(bool(row.get("error_type")) for row in rows)
    complete = (
        len(rows) == expected
        and runtime_errors == 0
        and (run_dir / "eval_report.json").exists()
        and (run_dir / "token_entropy.jsonl").exists()
    )
    return (
        int(compatible),
        int(complete),
        len(rows),
        (run_dir / "results.jsonl").stat().st_mtime,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for root in args.roots:
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            run_dir = config_path.parent
            if not (run_dir / "results.jsonl").exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            dataset = dataset_key(config, run_dir)
            method = method_key(config, run_dir)
            if dataset is None or method is None:
                continue
            model = model_key(config, run_dir)
            compatible, config_errors = canonical_config(config)
            score = candidate_score(run_dir, dataset, compatible)
            grouped[(model, dataset, method)].append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "score": score,
                    "canonical_compatible": compatible,
                    "complete": bool(score[1]),
                    "config_errors": config_errors,
                    "config": config,
                }
            )

    selected = {}
    candidates = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item["score"], reverse=True)
        label = "/".join(key)
        if values[0]["canonical_compatible"] and values[0]["complete"]:
            selected[label] = values[0]["run_dir"]
        candidates[label] = values

    comparisons = []
    for model in sorted({key[0] for key in grouped}):
        for dataset in EXPECTED:
            runs = {}
            for method in (
                "cot",
                "full_lead",
                "initial_transition",
                "legacy_talr",
                "true_talr",
            ):
                label = f"{model}/{dataset}/{method}"
                if label in selected:
                    runs[method] = selected[label]
            if {"full_lead", "initial_transition"} & set(runs) and {
                "legacy_talr",
                "true_talr",
            } & set(runs):
                comparisons.append(
                    {"model": model, "dataset": dataset, "runs": runs}
                )

    write_json(
        args.output,
        {
            "comparisons": comparisons,
            "selected": selected,
            "all_candidates": candidates,
        },
    )
    print(
        f"Wrote {args.output}: {len(comparisons)} comparison cells, "
        f"{len(selected)} selected runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
