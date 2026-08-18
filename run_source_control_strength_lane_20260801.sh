#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD-transplant
PY=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260801_soft_state_transplant_expanded
MANIFEST="$BASE/outcome_agnostic_1024_events.jsonl"

mkdir -p "$BASE/source_control_strength"

cat >"$BASE/source_control_strength/experiment_manifest.json" <<'EOF'
{
  "experiment_id": "A0_SOURCE_CONTROL_STRENGTH_080_100",
  "research_question": "RQ1",
  "hypothesis": "H1",
  "model": "R1-Onevision-7B-RL",
  "manifest": "outcome_agnostic_1024_events.jsonl",
  "baseline": "hard route at the same matched event",
  "changed_variable": "soft-state source and intervention strength",
  "sources": ["masked_image", "random", "dataset_noise"],
  "mix_lambdas": [0.80, 1.00],
  "primary_metric": "true-image visual specificity relative to controls",
  "success_gate": "true-image pooled fixed-damaged exceeds controls or visual-specific margin contrast is positive",
  "stop_rule": "no source-specific signal at lambda 0.80, 0.95, and 1.00",
  "seed": 42
}
EOF

cd "$ROOT"
"$PY" -m py_compile run_soft_state_transplant.py

for lambda in 0.80 1.00; do
  tag=${lambda/./}
  out="$BASE/source_control_strength/lambda${tag}_1024"
  if [[ -s "$out/RUN_COMPLETE" ]]; then
    printf '[%s] skip complete lambda=%s\n' "$(date -Is)" "$lambda"
    continue
  fi
  printf '[%s] start source controls lambda=%s\n' "$(date -Is)" "$lambda"
  "$PY" run_soft_state_transplant.py \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --output-dir "$out" \
    --branches masked_image,random,dataset_noise \
    --mix-lambda "$lambda" \
    --trace-topk 0
  printf '[%s] complete source controls lambda=%s\n' "$(date -Is)" "$lambda"
done

touch "$BASE/source_control_strength/QUEUE_COMPLETE"

