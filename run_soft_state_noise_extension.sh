#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260731_soft_state_transplant
MANIFEST="$BASE/soft_state_transplant_manifest.jsonl"
OUT="$BASE/noise_extension40"

cd "$ROOT"
"$PYTHON" -m py_compile lead/generation_utils.py run_soft_state_transplant.py

printf '[%s] starting dataset-noise smoke\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --output-dir "$BASE/noise_smoke2" \
  --limit 2 \
  --branches generic_noise,dataset_noise \
  --trace-topk 20

printf '[%s] smoke passed; starting 40-event noise extension\n' "$(date -Is)"
"$PYTHON" run_soft_state_transplant.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --output-dir "$OUT" \
  --branches generic_noise,dataset_noise \
  --trace-topk 20

cat > "$OUT/noise_design.json" <<'EOF'
{
  "generic_noise": "Gaussian RGB pixel noise with sigma=30 for every dataset",
  "vstar": "4x4 spatial patch shuffle",
  "mmvp": "paired contrast image from the official MMVP pair",
  "realworldqa": "Fourier phase scramble with restored per-channel mean/std",
  "visulogic": "3x3 logic-cell shuffle",
  "causal_constraint": "Only the event-step soft vector comes from the corrupted image; receiver prefix, KV cache, and all subsequent decoding use the true image."
}
EOF
touch "$OUT/NOISE_EXTENSION_COMPLETE"
printf '[%s] noise extension complete\n' "$(date -Is)"
