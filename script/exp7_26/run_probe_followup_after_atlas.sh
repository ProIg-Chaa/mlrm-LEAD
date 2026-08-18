#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup
REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
LABELER="$REPO/script/exp7_26/label_atlas_strength_extension.py"
TRAINER_DIR="$REPO/script/exp7_26"
OLD_ATLAS=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_extended/analysis/combined_eight_dataset_rows.jsonl
MERGED="$ROOT/analysis/combined_eight_plus_followup_rows.jsonl"

mkdir -p "$ROOT/merged" "$ROOT/analysis"
while [[ $(find "$ROOT/runs" -name SHARD_COMPLETE | wc -l) -lt 4 ]]; do
  sleep 300
done

for spec in \
  "contracted_soft_l090 contracted_soft_l090" \
  "contracted_soft_l095 contracted_soft_l095" \
  "pure_soft_l100 pure_soft_l100"
do
  read -r run_name treatment_name <<<"$spec"
  "$PY" "$LABELER" \
    --repo "$REPO" \
    --atlas-root "$ROOT" \
    --runs-subdir runs \
    --run-dir-name "$run_name" \
    --treatment-name "$treatment_name" \
    --output "$ROOT/merged/$treatment_name.jsonl"
done

cat "$OLD_ATLAS" \
  "$ROOT/merged/contracted_soft_l090.jsonl" \
  "$ROOT/merged/contracted_soft_l095.jsonl" \
  "$ROOT/merged/pure_soft_l100.jsonl" >"$MERGED"

"$PY" "$REPO/script/exp7_26/train_intervention_utility_probe.py" \
  --atlas "$MERGED" \
  --output-dir "$ROOT/probe_retrained" \
  --folds 5 --seeds 11 22 42 --models linear mlp

"$PY" "$REPO/script/exp7_26/analyze_probe_data_and_labels.py" \
  --trainer-dir "$TRAINER_DIR" \
  --atlas "$MERGED" \
  --output-dir "$ROOT/probe_data_label_analysis" \
  --folds 5 --seeds 11 22 42 \
  --fractions 0.25 0.5 0.75 1.0

date >"$ROOT/PROBE_FOLLOWUP_COMPLETE"
