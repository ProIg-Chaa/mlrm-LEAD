#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
SOURCE=/root/autodl-tmp/gushuo/outputs/experiments/20260717_talr_diagnosis_optimization
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260718_w8_cap_smoke

mkdir -p "$OUT"

# Do not compete with the final locked-validation worker.
while kill -0 3400 2>/dev/null; do
    sleep 30
done

run_one() {
    local dataset_name=$1
    local dataset_path=$2
    local cap=$3
    local output_dir="$OUT/$dataset_name/w8_k$cap"
    mkdir -p "$output_dir"
    if [[ -s "$output_dir/results.jsonl" && -s "$output_dir/eval_report.json" ]]; then
        return
    fi
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=0 "$PY" main.py \
        --model_name "$MODEL" \
        --dataset "$dataset_path" \
        --output_dir "$output_dir" \
        --method lead \
        --alpha 0.4 \
        --max_switch_count 5 \
        --window_size 128 \
        --cot_prompt_mode orign \
        --no-do_sample \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 20 \
        --seed 42 \
        --max_new_tokens 1024 \
        --device cuda \
        --save_token_entropy \
        --save_full_token_entropy \
        --trace_topk 0 \
        --lead_initial_transition_with_refinement \
        --lead_refinement_window 8 \
        --lead_refinement_soft_cap "$cap" \
        --lead_guard_candidate_only \
        --lead_disable_answer_zone_lock
}

lane_vstar() {
    for cap in 4 8; do
        run_one \
            vstar64 \
            "$SOURCE/dev_subsets/vstar_stratified64_seed42.jsonl" \
            "$cap"
    done
}

lane_realworldqa() {
    for cap in 4 8; do
        run_one \
            realworldqa64 \
            "$SOURCE/dev_subsets/realworldqa_fixed200_stratified64_seed42.jsonl" \
            "$cap"
    done
}

lane_vstar >"$OUT/vstar_lane.log" 2>&1 &
pid_a=$!
lane_realworldqa >"$OUT/realworldqa_lane.log" 2>&1 &
pid_b=$!
wait "$pid_a" "$pid_b"

"$PY" - <<'PY' >"$OUT/quick_summary.tsv"
import json
from pathlib import Path

root = Path("/root/autodl-tmp/gushuo/outputs/experiments/20260718_w8_cap_smoke")
print("dataset\tmethod\tcorrect\ttotal\taccuracy")
for result_path in sorted(root.glob("*/*/eval_report.json")):
    report = json.loads(result_path.read_text(encoding="utf-8"))
    total = int(report.get("total", report.get("total_samples", 0)))
    correct = int(report.get("correct", report.get("correct_samples", 0)))
    accuracy = report.get("accuracy", 0.0)
    print(
        f"{result_path.parent.parent.name}\t{result_path.parent.name}\t"
        f"{correct}\t{total}\t{accuracy}"
    )
PY
