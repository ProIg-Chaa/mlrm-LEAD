#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260801_soft_state_transplant_expanded
MANIFEST="$OUT/outcome_agnostic_1024_events.jsonl"

printf '[%s] strength extension waiting for expanded noise queue\n' "$(date -Is)"
while [[ ! -e "$OUT/EXPANDED_QUEUE_COMPLETE" ]]; do
  sleep 120
done

cd "$ROOT"
"$PYTHON" -m py_compile run_soft_state_transplant.py
for lambda in 0.80 1.00; do
  tag=${lambda/./}
  printf '[%s] source x strength lambda=%s\n' "$(date -Is)" "$lambda"
  "$PYTHON" run_soft_state_transplant.py \
    --manifest "$MANIFEST" --model "$MODEL" \
    --output-dir "$OUT/source_strength_lambda${tag}_1024" \
    --branches true_image,swapped_image \
    --mix-lambda "$lambda" --trace-topk 0
done

cat > "$OUT/source_strength_design.json" <<'EOF'
{
  "factors": {
    "image_source": ["true_image", "swapped_image"],
    "mix_lambda": [0.80, 0.95, 1.00]
  },
  "lambda_095_reuse": {
    "true_image": "historical Atlas contracted_soft_l095",
    "swapped_image": "core_1024"
  },
  "purpose": "Separate image-source specificity from intervention strength and test their interaction."
}
EOF
touch "$OUT/STRENGTH_EXTENSION_COMPLETE"
printf '[%s] strength extension complete\n' "$(date -Is)"
