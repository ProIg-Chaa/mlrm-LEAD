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

python3 script/plot_token_entropy_curve.py \
  "$RUN_ROOT/cot/token_entropy_full.jsonl" \
  --metric raw_entropy \
  --dataset_jsonl "$DATASET_PATH" \
  --overview_per_type

python3 script/plot_token_entropy_curve.py \
  "$RUN_ROOT/lead/token_entropy_full.jsonl" \
  --metric raw_entropy \
  --dataset_jsonl "$DATASET_PATH" \
  --overview_per_type

echo "Generated overview figures under:"
echo "  $RUN_ROOT/cot/"
echo "  $RUN_ROOT/lead/"
