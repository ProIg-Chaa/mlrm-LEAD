#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
STAGE=/root/gushuo/transition_logit_stage_0717
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
CUBE_LOG=/root/gushuo/outputs/experiments/20260717_transition_cube/worker.log
SOURCE=/root/gushuo/outputs/experiments/20260716_transition_causal_newgpu/transition_causal_decomposition
OUT=/root/gushuo/outputs/experiments/20260717_same_prefix_logit_probe

while [[ ! -f "$CUBE_LOG" ]] || ! grep -q '\[ALL DONE\]' "$CUBE_LOG"; do
  if [[ -f "$CUBE_LOG" ]] && grep -q '\[FAILED\]' "$CUBE_LOG"; then
    echo "[FAILED] Transition Cube failed; logit probe not started" >&2
    exit 1
  fi
  sleep 60
done

cp "$STAGE/lead/generation_utils.py" "$ROOT/lead/generation_utils.py"
cp "$STAGE/script/exp7_17/run_same_prefix_logit_probe_20260717.py" "$ROOT/script/exp7_17/"
cd "$ROOT"
"$PYTHON" -m py_compile lead/generation_utils.py script/exp7_17/run_same_prefix_logit_probe_20260717.py
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" script/exp7_17/run_same_prefix_logit_probe_20260717.py \
  --root "$ROOT" --model "$MODEL" --dataset "$ROOT/data/vstar.jsonl" \
  --cot-trace "$SOURCE/vstar/cot_orign_greedy/token_entropy_full.jsonl" \
  --output-dir "$OUT/vstar" --limit 80 --device cuda

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" script/exp7_17/run_same_prefix_logit_probe_20260717.py \
  --root "$ROOT" --model "$MODEL" --dataset "$ROOT/data/mmvp.jsonl" \
  --cot-trace "$SOURCE/mmvp/cot_orign_greedy/token_entropy_full.jsonl" \
  --output-dir "$OUT/mmvp" --limit 100 --device cuda

echo "[ALL DONE] $(date '+%F %T') same-prefix logit probe"
