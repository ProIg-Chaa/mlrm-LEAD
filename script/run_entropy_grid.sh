#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL}"
DATASET="${DATASET:-data/physunibench.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/experiments/entropy_grid}"
LIMIT="${LIMIT:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GPU_INDEX="${GPU_INDEX:-1}"
ALPHAS="${ALPHAS:-0.4 0.6}"
MAX_SWITCH_COUNTS="${MAX_SWITCH_COUNTS:-3 5}"

for alpha in ${ALPHAS}; do
  for switch_count in ${MAX_SWITCH_COUNTS}; do
    echo "Running alpha=${alpha}, max_switch_count=${switch_count}"
    micromamba run -n mlrm-lead python script/run_method_comparison.py \
      --model_name "${MODEL_NAME}" \
      --dataset "${DATASET}" \
      --output_root "${OUTPUT_ROOT}" \
      --limit "${LIMIT}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --gpu_index "${GPU_INDEX}" \
      --methods lead \
      --alpha "${alpha}" \
      --max_switch_count "${switch_count}" \
      --save_token_entropy
  done
done
