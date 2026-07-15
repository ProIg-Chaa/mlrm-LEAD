#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gushuo/proj/mlrm-LEAD
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/dev/shm/wangzixu_models/R1-Onevision-7B-RL
STAMP=20260715_early_actual_visual_anchor
BASE=/root/autodl-tmp/gushuo/outputs/experiments/$STAMP/early_actual_visual_anchor_realworld_hard54
HIST=/root/gushuo/migrated_results/rl_compact_matrix_migration_20260713/reusable_results/r1_onevision_7b_rl/realworldqa_fixed200
FULL=$ROOT/data/realworldqa_fixed_mcq_random200_seed42.jsonl
HARD=$ROOT/data/realworldqa_hard_wrong54_eava.jsonl
CONTROL=$ROOT/data/realworldqa_cot_correct_control20_eava.jsonl

COMMON=(
  --method lead --cot_prompt_mode orign --alpha 0.4 --max_switch_count 5
  --window_size 128 --lead_initial_transition_only --no-do_sample
  --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42
  --max_new_tokens 1024 --device cuda --save_token_entropy
  --save_full_token_entropy --trace_topk 20
)
ANCHOR=(
  --lead_early_visual_anchor --lead_early_visual_anchor_top_m 8
  --lead_early_visual_anchor_temperature 0.10
)

cd "$ROOT"
mkdir -p "$BASE"
"$PYTHON" -m py_compile \
  main.py lead/inference.py lead/generation_utils.py \
  script/exp7_15/build_realworld_eava_subsets_0715.py \
  script/exp7_15/summarize_eava_priority_0715.py
"$PYTHON" -m pytest -q tests/test_early_visual_anchor.py

"$PYTHON" script/exp7_15/build_realworld_eava_subsets_0715.py \
  --dataset "$FULL" \
  --cot-results "$HIST/cot_orign_greedy/results.jsonl" \
  --initial-results "$HIST/initial_transition_only/results.jsonl" \
  --talr-results "$HIST/talr/results.jsonl" \
  --hard-output "$HARD" --control-output "$CONTROL" \
  --manifest-output "$BASE/subset_manifest.json"

run_one() {
  local out=$1 dataset=$2 expected=$3 limit=$4
  shift 4
  if [[ -f "$out/results.jsonl" && -f "$out/eval_report.json" &&
        "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]; then
    echo "[SKIP] $out"
    return
  fi
  if [[ -d "$out" ]]; then
    case "$out" in "$BASE"/*) rm -rf -- "$out" ;; *) exit 3 ;; esac
  fi
  mkdir -p "$out"
  local limit_args=()
  [[ "$limit" != none ]] && limit_args=(--limit "$limit")
  echo "[START] $(date '+%F %T') $(basename "$out") expected=$expected"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py "${COMMON[@]}" --model_name "$MODEL" \
    --dataset "$dataset" --output_dir "$out" "${limit_args[@]}" "$@"
  [[ -f "$out/eval_report.json" && "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]]
  echo "[DONE] $(date '+%F %T') $(basename "$out")"
}

# Two-sample token-level regression: lambda=0 must preserve Initial Transition.
run_one "$BASE/smoke2_initial" "$HARD" 2 2
run_one "$BASE/smoke2_visual_lambda0" "$HARD" 2 2 \
  "${ANCHOR[@]}" --lead_early_visual_anchor_source visual_hidden \
  --lead_early_visual_anchor_lambda 0.0
"$PYTHON" - "$BASE/smoke2_initial/results.jsonl" "$BASE/smoke2_visual_lambda0/results.jsonl" <<'PY'
import json, sys
def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
left, right = load(sys.argv[1]), load(sys.argv[2])
assert [x["id"] for x in left] == [x["id"] for x in right]
assert [x["model_answer"] for x in left] == [x["model_answer"] for x in right]
print("[PASS] lambda=0 exactly reproduces Initial Transition on smoke2")
PY

# Smoke both controls before the formal hard54 runs.
run_one "$BASE/smoke2_static" "$HARD" 2 2 \
  "${ANCHOR[@]}" --lead_early_visual_anchor_source image_pad \
  --lead_early_visual_anchor_lambda 0.10
run_one "$BASE/smoke2_actual_visual" "$HARD" 2 2 \
  "${ANCHOR[@]}" --lead_early_visual_anchor_source visual_hidden \
  --lead_early_visual_anchor_lambda 0.10

run_one "$BASE/hard54_static" "$HARD" 54 none \
  "${ANCHOR[@]}" --lead_early_visual_anchor_source image_pad \
  --lead_early_visual_anchor_lambda 0.10
run_one "$BASE/hard54_actual_visual" "$HARD" 54 none \
  "${ANCHOR[@]}" --lead_early_visual_anchor_source visual_hidden \
  --lead_early_visual_anchor_lambda 0.10

"$PYTHON" script/exp7_15/summarize_eava_priority_0715.py \
  --stage hard --dataset "$HARD" \
  --initial-results "$HIST/initial_transition_only/results.jsonl" \
  --static-results "$BASE/hard54_static/results.jsonl" \
  --actual-results "$BASE/hard54_actual_visual/results.jsonl" \
  --output-dir "$BASE"

if "$PYTHON" - "$BASE/hard_summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["hard_gate_passed"] else 1)
PY
then
  echo "[GATE] hard54 passed; running COT-correct control20"
  run_one "$BASE/control20_actual_visual" "$CONTROL" 20 none \
    "${ANCHOR[@]}" --lead_early_visual_anchor_source visual_hidden \
    --lead_early_visual_anchor_lambda 0.10
  "$PYTHON" script/exp7_15/summarize_eava_priority_0715.py \
    --stage control --dataset "$CONTROL" \
    --initial-results "$HIST/cot_orign_greedy/results.jsonl" \
    --actual-results "$BASE/control20_actual_visual/results.jsonl" \
    --output-dir "$BASE"
  if "$PYTHON" - "$BASE/control_summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["control_gate_passed"] else 1)
PY
  then
    echo "[GATE] control20 passed; running fixed200"
    run_one "$BASE/full200_actual_visual" "$FULL" 200 none \
      "${ANCHOR[@]}" --lead_early_visual_anchor_source visual_hidden \
      --lead_early_visual_anchor_lambda 0.10
    "$PYTHON" script/exp7_15/summarize_eava_priority_0715.py \
      --stage full --dataset "$FULL" \
      --initial-results "$HIST/initial_transition_only/results.jsonl" \
      --actual-results "$BASE/full200_actual_visual/results.jsonl" \
      --output-dir "$BASE"
  else
    echo "[GATE] control20 failed; fixed200 intentionally skipped"
  fi
else
  echo "[GATE] hard54 failed; control20 and fixed200 intentionally skipped"
fi

echo "[ALL DONE] $(date '+%F %T') EAVA priority experiment"
