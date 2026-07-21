#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
RL_MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
VISION_MODEL=/dev/shm/wangzixu_models/Vision-R1-7B
RL_BASE=/root/gushuo/migrated_results/rl_compact_matrix_migration_20260713/reusable_results/r1_onevision_7b_rl
VISION_BASE=/root/autodl-tmp/gushuo/outputs/experiments/20260714_vision_r1_compact_matrix/vision_r1_7b
TRUE_ROOT=/root/gushuo/outputs/experiments/20260716_talr_dual_line/true_talr_core_runs
SUMMARY=/root/gushuo/outputs/experiments/20260716_talr_dual_line/main_summary_true_talr
COMMON=(--cot_prompt_mode orign --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --max_new_tokens 1024 --device cuda --no-do_sample --save_token_entropy --trace_topk 0)
INITIAL=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_only)
TALR=(--method lead --alpha 0.4 --max_switch_count 5 --window_size 128 --lead_initial_transition_with_refinement --lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 --format_cooldown_min_step 2 --lead_soft_veto_on_diffuse --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 64 --lead_veto_recent_repeat_tau 0.35)

cd "$ROOT"
"$PYTHON" -m py_compile main.py lead/inference.py lead/generation_utils.py \
  script/exp7_16/summarize_talr_core_main.py
mkdir -p "$TRUE_ROOT" "$SUMMARY"

assert_complete() {
  local out=$1 expected=$2
  if [[ ! -f "$out/results.jsonl" || ! -f "$out/eval_report.json" ]]; then
    return 1
  fi
  if [[ "$(wc -l < "$out/results.jsonl")" -ne "$expected" ]]; then
    return 1
  fi
  if grep -q '"error_type": "' "$out/results.jsonl"; then
    return 1
  fi
  return 0
}

run_main() {
  local model=$1 dataset=$2 expected=$3 out=$4; shift 4
  if assert_complete "$out" "$expected"; then
    echo "[SKIP] complete $out"
    return
  fi
  if [[ -e "$out" ]]; then
    mv "$out" "${out}.superseded.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py --model_name "$model" --dataset "$dataset" --output_dir "$out" \
    "${COMMON[@]}" "$@"
  assert_complete "$out" "$expected"
}

run_eval() {
  local dataset_key=$1 out=$2
  if [[ "$dataset_key" == "mmvp" ]]; then
    "$PYTHON" script/evaluate_specialized_results.py --dataset "$ROOT/data/mmvp.jsonl" \
      --results "$out/results.jsonl" --output_json "$out/specialized_eval_report.json" \
      --output_results_jsonl "$out/specialized_eval_rows.jsonl"
  elif [[ "$dataset_key" == "realworldqa_fixed200" ]]; then
    "$PYTHON" script/evaluate_realworldqa_mcq.py --dataset "$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl" \
      --results "$out/results.jsonl" --output_json "$out/realworldqa_mcq_eval.json" \
      --output_results_jsonl "$out/realworldqa_mcq_rows.jsonl"
  fi
}

# The original Vision VStar initial-transition and quota+guard runs are
# resource-invalid: 176/191 rows are CUDA OOM. Re-run transition alone first.
VISION_IT="$VISION_BASE/vstar/initial_transition_only"
if ! assert_complete "$VISION_IT" 191 && [[ -e "$VISION_IT" ]]; then
  mv "$VISION_IT" "${VISION_IT}.oom_invalid.$(date +%Y%m%d_%H%M%S)"
fi
run_main "$VISION_MODEL" "$ROOT/data/vstar.jsonl" 191 "$VISION_IT" "${INITIAL[@]}"

# Smoke the new composition before spending the full two-model matrix. The
# trace must show the explicit transition-with-refinement configuration.
SMOKE="$TRUE_ROOT/smoke/r1_vstar_talr"
run_main "$RL_MODEL" "$ROOT/data/vstar.jsonl" 2 "$SMOKE" --limit 2 "${TALR[@]}"
grep -q '"lead_initial_transition_with_refinement": true' "$SMOKE/config.json"

for spec in \
  "r1_onevision_7b_rl|$RL_MODEL|$RL_BASE" \
  "vision_r1_7b|$VISION_MODEL|$VISION_BASE"; do
  IFS='|' read -r key model base <<< "$spec"
  for data_spec in \
    "vstar|$ROOT/data/vstar.jsonl|191" \
    "mmvp|$ROOT/data/mmvp.jsonl|300" \
    "realworldqa_fixed200|$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl|200" \
    "visulogic300|$ROOT/data/visulogic.jsonl|300"; do
    IFS='|' read -r dataset_key dataset expected <<< "$data_spec"
    out="$TRUE_ROOT/$key/$dataset_key/talr_early_quota05_guard_min2"
    limit_args=()
    if [[ "$dataset_key" == "visulogic300" ]]; then
      limit_args=(--limit 300)
    fi
    run_main "$model" "$dataset" "$expected" "$out" "${limit_args[@]}" "${TALR[@]}"
    run_eval "$dataset_key" "$out"
  done
done

"$PYTHON" script/exp7_16/summarize_talr_core_main.py \
  --rl-root "$RL_BASE" --vision-root "$VISION_BASE" \
  --rl-talr-root "$TRUE_ROOT/r1_onevision_7b_rl" \
  --vision-talr-root "$TRUE_ROOT/vision_r1_7b" --output-dir "$SUMMARY"
echo "[ALL DONE] $(date '+%F %T') true TALR repair"
