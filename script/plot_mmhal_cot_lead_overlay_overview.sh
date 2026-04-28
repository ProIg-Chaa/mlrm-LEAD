#!/usr/bin/env bash
set -euo pipefail

cd /share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD

RUN_ROOT="${1:-}"
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$(ls -td output/experiments/*/mmhal_balanced_entropy_curves_* 2>/dev/null | head -n 1)"
fi

if [[ -z "$RUN_ROOT" || ! -d "$RUN_ROOT" ]]; then
  echo "Could not find MMHal balanced entropy run directory." >&2
  exit 1
fi

DATASET_PATH="data/mmhal_bench_balanced_2pertype.jsonl"
OUTPUT_PATH="$RUN_ROOT/mmhal_cot_lead_raw_entropy_overlay_overview.png"

python3 script/plot_mmhal_cot_lead_overlay_overview.py \
  --cot "$RUN_ROOT/cot/token_entropy_full.jsonl" \
  --lead "$RUN_ROOT/lead/token_entropy_full.jsonl" \
  --dataset_jsonl "$DATASET_PATH" \
  --metric raw_entropy \
  --output "$OUTPUT_PATH"

echo "$OUTPUT_PATH"
