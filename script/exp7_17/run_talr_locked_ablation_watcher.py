#!/usr/bin/env python3
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import run_talr_worst_tuning_queue as queue


MAX_WORKERS = max(1, int(os.environ.get("TALR_MAX_WORKERS", "2")))


def main():
    final_summary_path = queue.ROOT / "final_summary.json"
    while not final_summary_path.exists():
        queue.log("ABLATION WAIT for locked full-validation summary")
        time.sleep(120)

    summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    selected = {
        model: (
            values[0],
            tuple(values[1]),
        )
        for model, values in summary["selected_configs"].items()
    }
    selected_guards = summary["selected_guards"]

    def lane(model_key, dataset_names):
        config_name, config = selected[model_key]
        for dataset_name in dataset_names:
            dataset_path = queue.FULL_DATASETS[dataset_name]
            queue.run_one(
                model_key,
                dataset_name,
                dataset_path,
                "initializer",
                "initial_transition",
                "none",
                "phase_e_ablation",
            )
            queue.run_one(
                model_key,
                dataset_name,
                dataset_path,
                config_name,
                config,
                "none",
                "phase_e_ablation",
            )
            queue.run_one(
                model_key,
                dataset_name,
                dataset_path,
                config_name,
                config,
                "answer_format2",
                "phase_e_ablation",
            )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        regular = ["vstar", "realworldqa", "mmvp"]
        futures = [
            executor.submit(lane, model, regular) for model in queue.MODELS
        ]
        for future in futures:
            future.result()
    lane("r1_rl", ["visulogic"])
    lane("vision_r1", ["visulogic"])

    output = {
        "selected_configs": selected,
        "selected_guards": selected_guards,
        "ablation_methods": [
            "initializer",
            "initializer_plus_refiner",
            "initializer_plus_refiner_plus_answer_format2",
            "selected_full_talr_from_phase_d",
        ],
        "status": "complete",
    }
    (queue.ROOT / "phase_e_ablation_manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    queue.log("ALL LOCKED TALR ABLATION RUNS COMPLETED")


if __name__ == "__main__":
    main()
