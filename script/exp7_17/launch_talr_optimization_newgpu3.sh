#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/gushuo/proj/mlrm-LEAD}"
PYTHON="${PYTHON:-/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python}"
R1_MODEL="${R1_MODEL:-/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL}"
VISION_MODEL="${VISION_MODEL:-/root/autodl-tmp/gushuo/models/Vision-R1-7B}"
STAMP="${STAMP:-20260717_talr_diagnosis_optimization}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/gushuo/outputs/experiments/${STAMP}}"

cd "$ROOT"
mkdir -p "$OUTPUT"

"$PYTHON" script/exp7_17/discover_talr_runs.py \
  --roots \
    /root/gushuo/migrated_results \
    /root/autodl-tmp/gushuo/outputs/experiments \
    /root/gushuo/outputs/experiments \
  --output "$OUTPUT/historical_run_manifest.json"

"$PYTHON" script/exp7_17/analyze_talr_components.py \
  --manifest "$OUTPUT/historical_run_manifest.json" \
  --output-dir "$OUTPUT/historical_diagnosis" \
  --selected-per-group 20

"$PYTHON" script/exp7_17/run_talr_optimization.py \
  --root "$ROOT" \
  --python "$PYTHON" \
  --model "$R1_MODEL" \
  --vision-model "$VISION_MODEL" \
  --output-root "$OUTPUT" \
  --reference-manifest "$OUTPUT/historical_run_manifest.json" \
  --workers 2 \
  --phase all \
  2>&1 | tee -a "$OUTPUT/optimization_runner.log"
