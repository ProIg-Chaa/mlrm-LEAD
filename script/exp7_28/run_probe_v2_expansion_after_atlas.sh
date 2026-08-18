#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/gushuo/outputs/experiments/20260728_probe_v2_external_expansion
REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
LABELER="$REPO/script/exp7_26/label_atlas_strength_extension.py"
FROZEN_DIR=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup/probe_v2_combined_retrained

mkdir -p "$ROOT/merged" "$ROOT/analysis"
while [[ $(find "$ROOT/runs" -name SHARD_COMPLETE 2>/dev/null | wc -l) -lt 4 ]]; do
  sleep 180
done

for treatment in contracted_soft_l090 contracted_soft_l095 pure_soft_l100; do
  "$PY" "$LABELER" \
    --repo "$REPO" \
    --atlas-root "$ROOT" \
    --runs-subdir runs \
    --run-dir-name "$treatment" \
    --treatment-name "$treatment" \
    --output "$ROOT/merged/$treatment.jsonl"
done

cat \
  "$ROOT/merged/contracted_soft_l090.jsonl" \
  "$ROOT/merged/contracted_soft_l095.jsonl" \
  "$ROOT/merged/pure_soft_l100.jsonl" \
  >"$ROOT/analysis/external_expansion_rows.jsonl"

"$PY" "$REPO/script/exp7_28/evaluate_frozen_hierarchical_probe_v2.py" \
  --trainer-dir "$REPO/script/exp7_26" \
  --atlas "$ROOT/analysis/external_expansion_rows.jsonl" \
  --linear-artifact "$FROZEN_DIR/hierarchical_probe_v2_linear.pt" \
  --mlp-artifact "$FROZEN_DIR/hierarchical_probe_v2_mlp.pt" \
  --output-dir "$ROOT/probe_v2_frozen_external" \
  >"$ROOT/frozen_evaluation.log"

sha256sum \
  "$FROZEN_DIR/hierarchical_probe_v2_linear.pt" \
  "$FROZEN_DIR/hierarchical_probe_v2_mlp.pt" \
  >"$ROOT/frozen_artifact_hashes.txt"
"$PY" "$REPO/script/exp7_28/summarize_external_controls.py" \
  --trainer-dir "$REPO/script/exp7_26" \
  --atlas "$ROOT/analysis/external_expansion_rows.jsonl" \
  --v2-summary "$ROOT/probe_v2_frozen_external/hierarchical_probe_v2_summary.json" \
  --output-dir "$ROOT/external_controls" \
  --model mlp \
  --random-repeats 1000

date >"$ROOT/PROBE_V2_EXPANSION_COMPLETE"
