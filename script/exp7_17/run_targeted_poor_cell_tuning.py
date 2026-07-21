#!/usr/bin/env python3
"""Run the pre-registered targeted tuning matrix for three weak cells."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_talr_worst_tuning_queue as queue


ROOT = queue.ROOT
READY_FILE = ROOT / "initializer_refiner_control_summary.json"
OUTPUT_JSON = ROOT / "targeted_poor_cells_summary.json"
OUTPUT_MD = ROOT / "targeted_poor_cells_summary.md"
VISION_SOURCE = Path("/root/autodl-tmp/gushuo/models/Vision-R1-7B")
VISION_RAM = queue.MODELS["vision_r1"]

R_MMVP_CONFIGS = {
    "r_mmvp_w8k2_t100_l050": (8, 2, 1.00, 0.50),
    "r_mmvp_w8k2_t125_l050": (8, 2, 1.25, 0.50),
}
R_RWQA_CONFIGS = {
    "initial_transition": "initial_transition",
    "r_rwqa_w4k1_t100": (4, 1, 1.00, 1.00),
    "r_rwqa_w8k1_t100": (8, 1, 1.00, 1.00),
    "r_rwqa_w8k2_t100": (8, 2, 1.00, 1.00),
}
V_VISU_CONFIGS = {
    "v_visu_w16k1_t075": (16, 1, 0.75, 1.00),
    "v_visu_w16k2_t075": (16, 2, 0.75, 1.00),
}


def wait_for_initializer_controls() -> None:
    missing_rounds = 0
    while not READY_FILE.exists():
        result = subprocess.run(
            [
                "pgrep",
                "-f",
                "run_r1_initializer_controls_and_analyze.py|"
                "run_initializer_mmvp_parallel.py",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            missing_rounds = 0
        else:
            missing_rounds += 1
            if missing_rounds >= 3:
                raise RuntimeError(
                    "Initializer controls stopped before the summary was written"
                )
        time.sleep(30)


def prepare_vision_model() -> None:
    if (VISION_RAM / "config.json").exists():
        return
    if not (VISION_SOURCE / "config.json").exists():
        raise RuntimeError(f"Missing Vision-R1 model: {VISION_SOURCE}")
    VISION_RAM.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                f"{VISION_SOURCE}/",
                f"{VISION_RAM}/",
            ],
            check=True,
        )
    else:
        shutil.copytree(VISION_SOURCE, VISION_RAM, dirs_exist_ok=True)


def run_dir(
    phase: str,
    model: str,
    dataset: str,
    config_name: str,
) -> Path:
    return ROOT / phase / model / dataset / f"{config_name}__none"


def reported_metrics(path: Path, dataset: str) -> dict:
    result = queue.metrics(path)
    if dataset == "mmvp":
        report = json.loads(
            (path / "specialized_eval_report.json").read_text(encoding="utf-8")
        )
        result["accuracy"] = report["accuracy"]
        result["pair_accuracy"] = report["pair_accuracy"]
        result["failed"] = report["failed_extraction"]
    elif dataset == "realworldqa":
        report = json.loads(
            (path / "realworldqa_mcq_eval.json").read_text(encoding="utf-8")
        )
        result["accuracy"] = report["accuracy"]
        result["failed"] = report["failed_extraction"]
    return result


def run_r1_lane() -> dict:
    rwqa_screen = {}
    for name, config in R_RWQA_CONFIGS.items():
        queue.run_one(
            "r1_rl",
            "realworldqa",
            queue.SCREEN_DATASETS["realworldqa"],
            name,
            config,
            "none",
            "phase_g_targeted_screen",
        )
        path = run_dir(
            "phase_g_targeted_screen",
            "r1_rl",
            "realworldqa",
            name,
        )
        rwqa_screen[name] = reported_metrics(path, "realworldqa")

    selected_rwqa = max(
        rwqa_screen,
        key=lambda name: (
            rwqa_screen[name]["accuracy"],
            -rwqa_screen[name]["failed"],
            -rwqa_screen[name]["refinement_active"],
            -rwqa_screen[name]["avg_tokens"],
        ),
    )
    selected_config = R_RWQA_CONFIGS[selected_rwqa]
    queue.run_one(
        "r1_rl",
        "realworldqa",
        queue.FULL_DATASETS["realworldqa"],
        selected_rwqa,
        selected_config,
        "none",
        "phase_g_targeted_full",
    )
    rwqa_full = reported_metrics(
        run_dir(
            "phase_g_targeted_full",
            "r1_rl",
            "realworldqa",
            selected_rwqa,
        ),
        "realworldqa",
    )

    mmvp = {}
    for name, config in R_MMVP_CONFIGS.items():
        queue.run_one(
            "r1_rl",
            "mmvp",
            queue.FULL_DATASETS["mmvp"],
            name,
            config,
            "none",
            "phase_g_targeted_full",
        )
        mmvp[name] = reported_metrics(
            run_dir(
                "phase_g_targeted_full",
                "r1_rl",
                "mmvp",
                name,
            ),
            "mmvp",
        )
    return {
        "realworldqa_screen": rwqa_screen,
        "realworldqa_selected": selected_rwqa,
        "realworldqa_full": rwqa_full,
        "mmvp_full": mmvp,
    }


def run_vision_lane() -> dict:
    visu = {}
    for name, config in V_VISU_CONFIGS.items():
        queue.run_one(
            "vision_r1",
            "visulogic",
            queue.FULL_DATASETS["visulogic"],
            name,
            config,
            "none",
            "phase_g_targeted_full",
        )
        visu[name] = reported_metrics(
            run_dir(
                "phase_g_targeted_full",
                "vision_r1",
                "visulogic",
                name,
            ),
            "visulogic",
        )
    return {"visulogic_full": visu}


def write_summary(payload: dict) -> None:
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    r1 = payload["r1"]
    vision = payload["vision"]
    lines = [
        "# Targeted Poor-Cell Tuning",
        "",
        f"- RealWorldQA selected on dev64: "
        f"`{r1['realworldqa_selected']}`",
        f"- RealWorldQA full accuracy: "
        f"{100 * r1['realworldqa_full']['accuracy']:.2f}%",
        "",
        "## R1 MMVP",
        "",
        "| Configuration | Sample accuracy | Pair accuracy | "
        "Refinement events |",
        "|---|---:|---:|---:|",
    ]
    for name, item in r1["mmvp_full"].items():
        lines.append(
            f"| {name} | {100 * item['accuracy']:.2f}% | "
            f"{100 * item['pair_accuracy']:.2f}% | "
            f"{item['refinement_active']} |"
        )
    lines.extend(
        [
            "",
            "## Vision-R1 VisuLogic",
            "",
            "| Configuration | Accuracy | Long | Maxed | "
            "Refinement events |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in vision["visulogic_full"].items():
        lines.append(
            f"| {name} | {100 * item['accuracy']:.2f}% | "
            f"{item['long']} | {item['maxed']} | "
            f"{item['refinement_active']} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    wait_for_initializer_controls()
    prepare_vision_model()
    with ThreadPoolExecutor(max_workers=2) as executor:
        r1_future = executor.submit(run_r1_lane)
        vision_future = executor.submit(run_vision_lane)
        payload = {
            "r1": r1_future.result(),
            "vision": vision_future.result(),
        }
    write_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
