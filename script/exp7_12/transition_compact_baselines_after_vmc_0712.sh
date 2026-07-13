#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD
OUT="${ROOT}/output/experiments/20260712_uniform_multimodel_full_matrix"
BASE="${OUT}/uniform_multimodel_full_matrix"
RUNNER="${ROOT}/script/exp7_12/run_compact_talr_matrix_0712.sh"

wait_file() {
  local path="$1" label="$2"
  while [[ ! -f "${path}" ]]; do
    echo "[WAIT] $(date '+%F %T') ${label}"
    sleep 15
  done
  echo "[READY] $(date '+%F %T') ${label}"
}

transition_r1() {
  wait_file "${BASE}/r1_onevision_7b/vmcbench_dev/cot_orign_greedy/eval_report.json" "R1 VMCBench COT"
  tmux kill-session -t uniform_r1_0712 2>/dev/null || true
  tmux kill-session -t compact_r1_baseline_0712 2>/dev/null || true
  tmux new-session -d -s compact_r1_baseline_0712 \
    "srun --jobid=26623 --overlap --gres=gpu:1 --ntasks=1 env GPU=0 MODEL_NAME=R1-Onevision-7B MODEL_KEY=r1_onevision_7b GROUP=baseline bash ${RUNNER} 2>&1 | tee ${OUT}/compact_r1_baseline_gpu09_0.log"
  echo "[LAUNCHED] compact_r1_baseline_0712"
}

transition_vision() {
  wait_file "${BASE}/vision_r1_7b/vmcbench_dev/cot_orign_greedy/eval_report.json" "Vision VMCBench COT"
  ssh gpu09 "tmux kill-session -t uniform_vision_0712 2>/dev/null || true"
  ssh gpu09 "tmux kill-session -t compact_vision_baseline_0712 2>/dev/null || true"
  ssh gpu09 "tmux new-session -d -s compact_vision_baseline_0712 'env GPU=1 MODEL_NAME=Vision-R1-7B MODEL_KEY=vision_r1_7b GROUP=baseline bash ${RUNNER} 2>&1 | tee ${OUT}/compact_vision_baseline_gpu09_1.log'"
  echo "[LAUNCHED] compact_vision_baseline_0712"
}

transition_r1 &
pid_r1=$!
transition_vision &
pid_vision=$!
wait "${pid_r1}" "${pid_vision}"
echo "[ALL-TRANSITIONED] $(date '+%F %T')"
