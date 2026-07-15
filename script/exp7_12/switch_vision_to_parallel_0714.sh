#!/usr/bin/env bash
set -euo pipefail

VBASE=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b
RBASE=/root/autodl-tmp/gushuo/outputs/experiments/20260712_uniform_multimodel_full_matrix/uniform_multimodel_full_matrix/r1_onevision_7b_rl
SCRIPT=/root/gushuo/proj/mlrm-LEAD/script/exp7_12/run_newgpu_vision_parallel_worker_0714.sh
LOGROOT=/root/autodl-tmp/gushuo/outputs/logs

complete() {
  local dir=$1 expected=$2
  [[ -f "$dir/results.jsonl" && -f "$dir/eval_report.json" &&
     "$(wc -l < "$dir/results.jsonl")" -eq "$expected" ]]
}

while ! complete "$VBASE/mmk12_physics/cot_orign_greedy" 500; do sleep 20; done
while ! complete "$RBASE/mmk12_physics/transition_preserving_quota05_guard_min2" 500; do sleep 20; done

tmux kill-session -t vision_r1_matrix_0714 2>/dev/null || true
tmux kill-session -t vision_parallel_a_0714 2>/dev/null || true
tmux kill-session -t vision_parallel_b_0714 2>/dev/null || true
sleep 3

tmux new-session -d -s vision_parallel_a_0714 \
  "bash $SCRIPT worker_a >> $LOGROOT/vision_parallel_a_0714.log 2>&1"
tmux new-session -d -s vision_parallel_b_0714 \
  "bash $SCRIPT worker_b >> $LOGROOT/vision_parallel_b_0714.log 2>&1"

echo "[LAUNCHED] $(date '+%F %T') two Vision-R1 workers" >> "$LOGROOT/vision_parallel_switch_0714.log"
