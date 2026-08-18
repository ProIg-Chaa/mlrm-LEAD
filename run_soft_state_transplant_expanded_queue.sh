#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
ATLAS=/root/autodl-tmp/gushuo/outputs/experiments/20260730_intervention_atlas_full
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260801_soft_state_transplant_expanded
MANIFEST="$OUT/outcome_agnostic_1024_events.jsonl"

mkdir -p "$OUT"
cd "$ROOT"
"$PYTHON" -m py_compile \
  lead/generation_utils.py \
  run_soft_state_transplant.py \
  prepare_soft_state_transplant_expanded.py

if [[ ! -s "$MANIFEST" ]]; then
  "$PYTHON" prepare_soft_state_transplant_expanded.py \
    --atlas "$ATLAS/merged/intervention_atlas_v0b.jsonl" \
    --roots \
      /root/autodl-tmp/gushuo/outputs/experiments/20260724_intervention_atlas_v0b/newgpu3 \
      "$ATLAS/runs" \
      "$ATLAS/reused_partial" \
    --output "$MANIFEST" \
    --samples-per-dataset 32 \
    --seed 420731
fi

printf '[%s] expanded smoke: all new branch families\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" --model "$MODEL" \
  --output-dir "$OUT/smoke2_core" --limit 2 \
  --branches swapped_image,masked_image,random,dataset_noise --trace-topk 0
for sigma in 10 30 60; do
  "$PYTHON" run_soft_state_transplant.py \
    --manifest "$MANIFEST" --model "$MODEL" \
    --output-dir "$OUT/smoke2_gaussian_sigma${sigma}" --limit 2 \
    --branches generic_noise --generic-noise-sigma "$sigma" --trace-topk 0
done

printf '[%s] expanded core: swap/mask/random/dataset-specific\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" --model "$MODEL" \
  --output-dir "$OUT/core_1024" \
  --branches swapped_image,masked_image,random,dataset_noise --trace-topk 0

for sigma in 10 30 60; do
  printf '[%s] gaussian dose sigma=%s\n' "$(date -Is)" "$sigma"
  "$PYTHON" run_soft_state_transplant.py \
    --manifest "$MANIFEST" --model "$MODEL" \
    --output-dir "$OUT/gaussian_sigma${sigma}_1024" \
    --branches generic_noise --generic-noise-sigma "$sigma" --trace-topk 0
done

cat > "$OUT/experiment_design.json" <<'EOF'
{
  "unique_samples": 128,
  "events": 1024,
  "event_types": ["fixed_1", "fixed_2", "fixed_4", "fixed_8", "fixed_16", "fixed_32", "entropy_top1", "random_control"],
  "new_branches": ["swapped_image", "masked_image", "random", "dataset_noise", "gaussian_sigma10", "gaussian_sigma30", "gaussian_sigma60"],
  "historical_reuse": ["hard", "true_image_contracted_l095"],
  "selection": "outcome-agnostic; 32 samples per dataset; all eight checkpoints paired within sample"
}
EOF
touch "$OUT/EXPANDED_QUEUE_COMPLETE"
printf '[%s] expanded transplant queue complete\n' "$(date -Is)"
