#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b/mmvp/lead
RUNNER=/root/gushuo/proj/mlrm-LEAD/script/exp7_15/run_eava_priority_0715.sh
LOG=/root/autodl-tmp/gushuo/outputs/logs/eava_boundary_watcher_0715.log

echo "[$(date '+%F %T')] waiting for Vision-R1 MMVP LEAD boundary" >> "$LOG"
while true; do
  rows=0
  [[ -f "$BASE/results.jsonl" ]] && rows=$(wc -l < "$BASE/results.jsonl")
  if [[ "$rows" -eq 300 && -f "$BASE/eval_report.json" ]]; then
    break
  fi
  sleep 30
done

echo "[$(date '+%F %T')] boundary reached; stopping worker B queue" >> "$LOG"
tmux kill-session -t compact_remaining_worker_b_0715 2>/dev/null || true
sleep 5
tmux new-session -d -s eava_priority_0715 "bash '$RUNNER' 2>&1 | tee -a /root/autodl-tmp/gushuo/outputs/logs/eava_priority_0715.log"
echo "[$(date '+%F %T')] EAVA priority session launched" >> "$LOG"
