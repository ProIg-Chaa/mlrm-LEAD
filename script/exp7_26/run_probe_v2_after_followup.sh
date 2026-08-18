#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup
REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
SCRIPT="$REPO/script/exp7_26/train_hierarchical_utility_probe_v2_conservative.py"
OLD=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_extended/analysis/combined_eight_dataset_rows.jsonl
FRESH="$ROOT/analysis/fresh_followup_rows.jsonl"
COMBINED="$ROOT/analysis/combined_eight_plus_followup_rows.jsonl"

while [[ ! -f "$ROOT/PROBE_FOLLOWUP_COMPLETE" ]]; do
  sleep 120
done

cat \
  "$ROOT/merged/contracted_soft_l090.jsonl" \
  "$ROOT/merged/contracted_soft_l095.jsonl" \
  "$ROOT/merged/pure_soft_l100.jsonl" >"$FRESH"

"$PY" "$SCRIPT" \
  --train-atlas "$OLD" \
  --external-atlas "$FRESH" \
  --output-dir "$ROOT/probe_v2_frozen_external" \
  --folds 5 \
  --seeds 11 22 42 \
  --models linear mlp \
  --rho 1.5

"$PY" "$SCRIPT" \
  --train-atlas "$COMBINED" \
  --output-dir "$ROOT/probe_v2_combined_retrained" \
  --folds 5 \
  --seeds 11 22 42 \
  --models linear mlp \
  --rho 1.5

date >"$ROOT/PROBE_V2_COMPLETE"
