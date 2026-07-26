#!/usr/bin/env python3
"""Run selected-ID repairs needed before the formal ablation audit."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO = Path("/root/gushuo/proj/mlrm-LEAD")
PYTHON = Path("/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python")
ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260722_talr_formal_ablation"
)
VISION_ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260714_vision_r1_compact_matrix/vision_r1_7b/vstar"
)
LOCKED_ROOT = Path(
    "/root/autodl-tmp/gushuo/outputs/experiments/"
    "20260721_locked_l095_all_models/runs"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def error_ids(path: Path) -> list[str]:
    return [str(row.get("id")) for row in load_jsonl(path) if row.get("error_type")]


def selected_repair(name: str, source: Path) -> dict:
    output = ROOT / "repairs" / name
    manifest = output / "repair_manifest.json"
    if not manifest.exists():
        subprocess.run(
            [
                str(PYTHON),
                "script/exp7_21/repair_runtime_error_samples_20260722.py",
                "--source-run", str(source),
                "--dataset", str(REPO / "data/vstar.jsonl"),
                "--model-name", "/dev/shm/wangzixu_models/Vision-R1-7B",
                "--output-root", str(output),
            ],
            cwd=REPO,
            check=True,
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data["status"] == "repaired":
        merged = Path(data["merged_run"])
        subprocess.run(
            [str(PYTHON), "script/exp7_21/recompute_single_corrected_eval.py", str(merged)],
            cwd=REPO,
            check=True,
        )
        shutil.copy2(merged / "corrected_eval_report.json", merged / "eval_report.json")
    return data


def adopted_full_repair(name: str, invalid: Path, valid: Path) -> dict:
    invalid_rows = invalid / "results.jsonl"
    valid_rows = valid / "results.jsonl"
    if not valid_rows.exists():
        raise FileNotFoundError(valid_rows)
    valid_data = load_jsonl(valid_rows)
    if any(row.get("error_type") for row in valid_data):
        raise RuntimeError(f"Adopted full repair still has runtime errors: {valid}")
    return {
        "name": name,
        "status": "full_single_process_repair_adopted",
        "invalid_run": str(invalid),
        "valid_run": str(valid),
        "replaced_error_ids": error_ids(invalid_rows) if invalid_rows.exists() else [],
        "valid_rows": len(valid_data),
        "runtime_errors_after_repair": 0,
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    entries = [
        selected_repair("vision_vstar_cot", VISION_ROOT / "cot_orign_greedy"),
        selected_repair("vision_vstar_full_lead", VISION_ROOT / "lead"),
        adopted_full_repair(
            "vision_vstar_l095",
            LOCKED_ROOT / "vision_r1/vstar/talr_w8k2_t125_l095_noguard__none.oom4_invalid_20260721_2147",
            LOCKED_ROOT / "vision_r1/vstar/talr_w8k2_t125_l095_noguard__none",
        ),
        adopted_full_repair(
            "openvl_vstar_l095",
            LOCKED_ROOT / "openvlthinker/vstar/talr_w8k2_t125_l095_noguard__none.incomplete.1784643851",
            LOCKED_ROOT / "openvlthinker/vstar/talr_w8k2_t125_l095_noguard__none",
        ),
    ]
    (ROOT / "repair_manifest.json").write_text(
        json.dumps({"repairs": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(entries)} repair records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
