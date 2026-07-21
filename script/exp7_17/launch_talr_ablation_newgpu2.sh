#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/gushuo/proj/mlrm-LEAD}"
PYTHON="${PYTHON:-/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python}"
R1_MODEL="${R1_MODEL:-/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL}"
VISION_MODEL="${VISION_MODEL:-/root/autodl-tmp/gushuo/models/Vision-R1-7B}"
BASE="${BASE:-/root/autodl-tmp/gushuo/outputs/experiments/20260717_talr_diagnosis_optimization}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/gushuo/outputs/experiments/20260717_talr_locked_ablation}"

cd "$ROOT"
mkdir -p "$OUTPUT"
"$PYTHON" script/exp7_17/run_talr_ablation.py \
  --root "$ROOT" \
  --python "$PYTHON" \
  --r1-model "$R1_MODEL" \
  --vision-model "$VISION_MODEL" \
  --locked-config "$BASE/locked_talr_config.json" \
  --output-root "$OUTPUT" \
  --workers 2 \
  2>&1 | tee -a "$OUTPUT/ablation_runner.log"
