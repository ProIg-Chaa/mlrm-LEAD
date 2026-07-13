#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
PYTHON=/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python
STATE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/gpu09_combo_state
BASE=${ROOT}/output/experiments/20260711_fixed_damaged_mechanism_analysis/transition_preserving_combo

cd "${ROOT}"
while [[ ! -f ${STATE}/lane0.done || ! -f ${STATE}/lane1.done ]]; do
  echo "[WAIT] $(date '+%F %T') combo lanes"
  sleep 30
done

for name in quota05_guard_min0 transition_preserving_quota05_guard_min2; do
  dir=${BASE}/realworldqa_fixed200/${name}
  ${PYTHON} script/evaluate_realworldqa_mcq.py \
    --dataset data/realworldqa_fixed_mcq_random200_seed42.jsonl \
    --results "${dir}/results.jsonl" \
    --output_json "${dir}/realworldqa_mcq_eval.json" \
    --output_results_jsonl "${dir}/realworldqa_mcq_eval_results.jsonl"
done

bash script/exp7_11/run_fixed_damaged_cpu_0711.sh
${PYTHON} script/exp7_11/extract_representative_cases.py >/dev/null
${PYTHON} script/exp7_11/finalize_semantic_audit.py
${PYTHON} script/exp7_11/build_fixed_damaged_report.py
touch "${STATE}/finalized.done"
