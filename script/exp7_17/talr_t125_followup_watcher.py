#!/usr/bin/env python3
"""Analyze the R1 T=1.25 full A/B and launch exactly one follow-up branch."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import run_talr_worst_tuning_queue as queue
from talr_analysis_common import paired_groups


NAME = "r_w8k2_t125_l100"
CONFIG = queue.R_CONFIGS[NAME]
BASE_NAME = "r_base_w8k2_t100"
ROOT = queue.ROOT
OUTPUT_JSON = ROOT / "t125_followup_decision.json"
OUTPUT_MD = ROOT / "t125_followup_decision.md"


def run_dir(phase: str, dataset: str, name: str) -> Path:
    return ROOT / phase / "r1_rl" / dataset / f"{name}__none"


def expected(dataset: str) -> int:
    return len(queue.load_jsonl(queue.FULL_DATASETS[dataset]))


def finished(phase: str, dataset: str, name: str) -> bool:
    return queue.complete(run_dir(phase, dataset, name), expected(dataset))


def t125_runner_alive() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "run_r1_t125_full_ab.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def pairwise(reference_dir: Path, method_dir: Path) -> dict:
    reference = {
        str(row.get("id")): row
        for row in queue.load_jsonl(reference_dir / "results.jsonl")
    }
    method = {
        str(row.get("id")): row
        for row in queue.load_jsonl(method_dir / "results.jsonl")
    }
    groups = paired_groups(reference, method)
    fixed = len(groups.get("fixed", []))
    damaged = len(groups.get("damaged", []))
    return {
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
        "both_correct": len(groups.get("both_correct", [])),
        "both_wrong": len(groups.get("both_wrong", [])),
    }


def write_decision(payload: dict) -> None:
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# R1 T1.25 Full A/B Follow-up",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mean accuracy delta vs T1.0: {100 * payload['mean_delta']:+.2f} pp",
        f"- Minimum cell delta: {100 * payload['min_delta']:+.2f} pp",
        f"- Refinement events: {payload['t125_active']} vs "
        f"{payload['t100_active']}",
        f"- Next branch: `{payload['next_branch']}`",
        "",
        "| Dataset | T1.0 | T1.25 | Delta | Fixed | Damaged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in payload["datasets"].items():
        lines.append(
            f"| {dataset} | {100 * item['t100']['accuracy']:.2f}% | "
            f"{100 * item['t125']['accuracy']:.2f}% | "
            f"{100 * item['delta']:+.2f} pp | "
            f"{item['pairwise']['fixed']} | {item['pairwise']['damaged']} |"
        )
    if payload.get("followup_results"):
        lines.extend(["", "## Follow-up Results", ""])
        for dataset, item in payload["followup_results"].items():
            lines.append(
                f"- {dataset}: {100 * item['accuracy']:.2f}% "
                f"({item['failed']} failed, {item['refinement_active']} "
                "refinement events)"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    datasets = ("vstar", "mmvp")
    missing_rounds = 0
    while not all(
        finished("phase_d_t125_ab", dataset, NAME) for dataset in datasets
    ):
        if t125_runner_alive():
            missing_rounds = 0
        else:
            missing_rounds += 1
            if missing_rounds >= 3:
                raise RuntimeError(
                    "T1.25 runner stopped before both A/B runs completed"
                )
        time.sleep(30)

    # Let the foreground runner release CUDA before the follow-up starts.
    while t125_runner_alive():
        time.sleep(5)

    cells = {}
    deltas = []
    t125_active = 0
    t100_active = 0
    t125_failed = 0
    t100_failed = 0
    t125_maxed = 0
    t100_maxed = 0
    for dataset in datasets:
        baseline_dir = run_dir("phase_d_full", dataset, BASE_NAME)
        challenger_dir = run_dir("phase_d_t125_ab", dataset, NAME)
        baseline = queue.metrics(baseline_dir)
        challenger = queue.metrics(challenger_dir)
        delta = challenger["accuracy"] - baseline["accuracy"]
        deltas.append(delta)
        t125_active += challenger["refinement_active"]
        t100_active += baseline["refinement_active"]
        t125_failed += challenger["failed"]
        t100_failed += baseline["failed"]
        t125_maxed += challenger["maxed"]
        t100_maxed += baseline["maxed"]
        cells[dataset] = {
            "t100": baseline,
            "t125": challenger,
            "delta": delta,
            "pairwise": pairwise(baseline_dir, challenger_dir),
        }

    mean_delta = sum(deltas) / len(deltas)
    min_delta = min(deltas)
    performance_ok = mean_delta >= -1e-12 and min_delta >= -0.006
    efficiency_ok = t125_active <= t100_active
    stability_ok = (
        t125_failed <= t100_failed + 1
        and t125_maxed <= t100_maxed + 1
    )
    accepted = performance_ok and efficiency_ok and stability_ok

    payload = {
        "decision": "accept_t125" if accepted else "retain_t100",
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "performance_ok": performance_ok,
        "efficiency_ok": efficiency_ok,
        "stability_ok": stability_ok,
        "t125_active": t125_active,
        "t100_active": t100_active,
        "datasets": cells,
        "next_branch": (
            "extend_t125_realworldqa_visulogic"
            if accepted
            else "run_initial_transition_vstar_mmvp"
        ),
    }
    write_decision(payload)

    followup = {}
    if accepted:
        for dataset in ("realworldqa", "visulogic"):
            queue.run_one(
                "r1_rl",
                dataset,
                queue.FULL_DATASETS[dataset],
                NAME,
                CONFIG,
                "none",
                "phase_d_t125_extension",
            )
            followup[dataset] = queue.metrics(
                run_dir("phase_d_t125_extension", dataset, NAME)
            )
    else:
        for dataset in datasets:
            queue.run_one(
                "r1_rl",
                dataset,
                queue.FULL_DATASETS[dataset],
                "initial_transition",
                "initial_transition",
                "none",
                "phase_f_initializer_control",
            )
            followup[dataset] = queue.metrics(
                run_dir(
                    "phase_f_initializer_control",
                    dataset,
                    "initial_transition",
                )
            )

    payload["followup_results"] = followup
    payload["followup_complete"] = True
    write_decision(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
