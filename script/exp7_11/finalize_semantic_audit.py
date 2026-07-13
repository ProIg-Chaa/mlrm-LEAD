#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path("output/experiments/20260711_fixed_damaged_mechanism_analysis/fixed_damaged_mechanism_analysis")

LABELS = {
    ("mmvp", "105"): "camera_viewpoint",
    ("mmvp", "112"): "object_presence",
    ("mmvp", "124"): "object_presence",
    ("mmvp", "160"): "clock_reading",
    ("mmvp", "161"): "clock_reading",
    ("mmvp", "19"): "object_part_visibility",
    ("mmvp", "104"): "camera_viewpoint",
    ("mmvp", "106"): "object_part_visibility",
    ("mmvp", "107"): "object_part_visibility",
    ("realworldqa_fixed200", "132"): "physical_safety_affordance",
    ("realworldqa_fixed200", "145"): "visual_counting",
    ("realworldqa_fixed200", "194"): "spatial_relation",
    ("realworldqa_fixed200", "217"): "attribute_comparison",
    ("realworldqa_fixed200", "105"): "spatial_relation",
    ("realworldqa_fixed200", "144"): "size_comparison",
    ("realworldqa_fixed200", "146"): "distance_estimation",
    ("realworldqa_fixed200", "182"): "spatial_relation",
    ("realworldqa_fixed200", "313"): "navigation_direction",
    ("visulogic300", "101"): "3d_spatial_reasoning",
    ("visulogic300", "108"): "3d_spatial_reasoning",
    ("visulogic300", "130"): "3d_spatial_reasoning",
    ("visulogic300", "111"): "visual_pattern_logic",
    ("visulogic300", "1"): "visual_pattern_logic",
    ("visulogic300", "155"): "visual_pattern_logic",
    ("visulogic300", "110"): "visual_pattern_logic",
    ("visulogic300", "103"): "visual_pattern_logic",
    ("vstar", "105"): "visual_attribute",
    ("vstar", "113"): "visual_attribute",
    ("vstar", "115"): "spatial_relation",
    ("vstar", "116"): "spatial_relation",
    ("vstar", "117"): "spatial_relation",
    ("vstar", "126"): "spatial_relation",
    ("vstar", "134"): "spatial_relation",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_mode(row):
    base, method = row["baseline_features"], row["method_features"]
    if row.get("extraction_only_flip"):
        return "extraction_only_flip"
    if method.get("answer_reversal"):
        return "answer_reversal"
    repeat_growth = method.get("repeat_ngram3_ratio", 0.0) - base.get("repeat_ngram3_ratio", 0.0)
    severe_length_growth = method.get("output_tokens", 0) >= 256 and method.get("output_tokens", 0) >= 2 * max(1, base.get("output_tokens", 0))
    if repeat_growth >= 0.15 or severe_length_growth or method.get("maxed_1024"):
        return "generation_degeneration"
    if row.get("method_failed_extraction"):
        return "unresolved_generation_or_extraction_failure"
    return "semantic_answer_trajectory_flip"


def main():
    selected = load_jsonl(ROOT / "selected_rows.jsonl")
    representatives = load_jsonl(ROOT / "representative_cases_for_audit.jsonl")
    rows = {(row["dataset"], row["method"], row["group"], str(row["id"])): row for row in selected}
    output = []
    for item in representatives:
        key = (item["dataset"], item["method"], item["group"], str(item["id"]))
        row = rows[key]
        output.append({
            **item,
            "semantic_label": LABELS.get(
                (item["dataset"], str(item["id"])),
                item.get("suggested_semantic_label", "visual_reasoning_other"),
            ),
            "objective_flip_type": failure_mode(row),
            "cot_output_tokens": row["baseline_features"]["output_tokens"],
            "method_output_tokens": row["method_features"]["output_tokens"],
            "cot_repeat_ngram3_ratio": row["baseline_features"]["repeat_ngram3_ratio"],
            "method_repeat_ngram3_ratio": row["method_features"]["repeat_ngram3_ratio"],
            "first_token_divergence": row.get("first_token_divergence"),
            "audit_status": "manually_reviewed_question_level",
            "audit_scope": "task semantics and objective trace/output signals; image-level visual correctness follows the benchmark gold label",
        })
    path = ROOT / "semantic_audit_labels.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
