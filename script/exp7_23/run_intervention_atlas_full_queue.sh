#!/usr/bin/env bash
set -euo pipefail

REPO=/root/gushuo/proj/mlrm-LEAD-atlas-v0b
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
MODEL=/root/autodl-tmp/gushuo/models/R1-Onevision-7B-RL
OUT=/root/autodl-tmp/gushuo/outputs/experiments/20260730_intervention_atlas_full
SELECTION="$OUT/selection"
RUNS="$OUT/runs"
REUSED="$OUT/reused_partial"
LOG="$OUT/logs"
V0B=/root/autodl-tmp/gushuo/outputs/experiments/20260724_intervention_atlas_v0b
FOLLOW=/root/autodl-tmp/gushuo/outputs/experiments/20260727_intervention_atlas_followup

cd "$REPO"
mkdir -p "$LOG" "$RUNS" "$REUSED"

while true; do
  active_sessions="$(tmux list-sessions -F '#S' 2>/dev/null || true)"
  if ! grep -Fxq 'handoff_followup_resume' <<< "$active_sessions" &&
     ! grep -Fxq 'corrected_handoff_externalization' <<< "$active_sessions"; then
    break
  fi
  echo "[$(date '+%F %T')] waiting for handoff queues" >> "$LOG/watcher.log"
  sleep 60
done

"$PYTHON" -m py_compile \
  main.py lead/inference.py lead/generation_utils.py \
  script/exp7_23/prepare_intervention_atlas_full.py \
  script/exp7_23/prepare_atlas_missing_adaptive_events.py \
  script/exp7_23/run_intervention_atlas_v0b_shard.py \
  script/exp7_23/summarize_intervention_atlas_v0b.py \
  script/exp7_23/merge_intervention_atlas_v0b.py

if [[ ! -f "$SELECTION/selection_manifest.json" ]]; then
  "$PYTHON" script/exp7_23/prepare_intervention_atlas_full.py \
    --data-root "$REPO/data" \
    --image-root /root/autodl-tmp/gushuo/datasets/mlrm-LEAD \
    --full-reuse "$V0B/selection/selected_all.jsonl" \
    --partial-reuse "$FOLLOW/selection/selected_all.jsonl" \
    --output-dir "$SELECTION" \
    --num-shards 12 \
    > "$LOG/prepare_full.log" 2>&1
fi

link_force() {
  local target=$1 link=$2
  if [[ -L "$link" ]]; then
    return
  fi
  if [[ -e "$link" ]]; then
    echo "Refusing to replace non-symlink: $link" >&2
    return 1
  fi
  ln -s "$target" "$link"
}

# Preserve the six fixed-step results already completed for the extra 64
# VisuLogic samples and label them in the unified V0B schema.
FIXED_REUSE="$REUSED/shard_visulogic_fixed"
mkdir -p "$FIXED_REUSE"
link_force "$FOLLOW/runs/shard_0/hard_baseline" "$FIXED_REUSE/hard_baseline"
link_force "$FOLLOW/runs/shard_0/contracted_soft_l095" "$FIXED_REUSE/contracted_soft_l095"
link_force "$FOLLOW/runs/shard_0/pure_soft_l100" "$FIXED_REUSE/pure_soft_l100"
link_force "$FOLLOW/runs/shard_0/event_manifest.jsonl" "$FIXED_REUSE/event_manifest.jsonl"
"$PYTHON" script/exp7_23/summarize_intervention_atlas_v0b.py \
  --shard-dir "$FIXED_REUSE" > "$LOG/label_reused_visulogic_fixed.log" 2>&1
printf 'reused_from=%s\nsamples=64\nevents=384\n' \
  "$FOLLOW/runs/shard_0" > "$FIXED_REUSE/SHARD_COMPLETE"

complete() {
  local out=$1 expected=$2
  [[ -f "$out/results.jsonl" ]] &&
    [[ -f "$out/eval_report.json" ]] &&
    [[ -f "$out/config.json" ]] &&
    [[ -f "$out/token_entropy_full.jsonl" ]] &&
    [[ "$(wc -l < "$out/results.jsonl")" -eq "$expected" ]] &&
    ! grep -q '"error_type": "[^"]' "$out/results.jsonl"
}

run_treatment() {
  local name=$1 kind=$2 mix=$3 supplement=$4 expected=$5
  local out="$supplement/$name"
  if complete "$out" "$expected"; then
    echo "[SKIP] supplement/$name"
    return
  fi
  if [[ -e "$out" ]]; then
    mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" main.py \
      --model_name "$MODEL" \
      --dataset "$supplement/event_dataset.jsonl" \
      --output_dir "$out" \
      --method cot_greedy \
      --cot_prompt_mode orign \
      --no-do_sample \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --seed 42 \
      --max_new_tokens 1024 \
      --device cuda \
      --save_token_entropy \
      --save_full_token_entropy \
      --trace_topk 0 \
      --trace_route_override_manifest "$supplement/event_override_manifest.json" \
      --trace_route_override_kind "$kind" \
      --trace_route_override_mix_lambda "$mix"
  complete "$out" "$expected"
}

run_supplement() {
  local supplement="$REUSED/shard_visulogic_adaptive"
  mkdir -p "$supplement"
  link_force "$FOLLOW/runs/shard_0/hard_baseline" "$supplement/hard_baseline"
  if [[ ! -f "$supplement/event_manifest.jsonl" ]]; then
    "$PYTHON" script/exp7_23/prepare_atlas_missing_adaptive_events.py \
      --samples "$FOLLOW/selection/selected_all.jsonl" \
      --baseline-trace "$FOLLOW/runs/shard_0/hard_baseline/token_entropy_full.jsonl" \
      --output-dir "$supplement"
  fi
  local expected
  expected="$(wc -l < "$supplement/event_dataset.jsonl")"
  run_treatment contracted_soft_l095 contracted_soft 0.95 "$supplement" "$expected"
  run_treatment pure_soft_l100 raw_soft 1.0 "$supplement" "$expected"
  "$PYTHON" script/exp7_23/summarize_intervention_atlas_v0b.py \
    --shard-dir "$supplement"
  printf 'reused_baseline_from=%s\nsamples=64\nevents=%s\n' \
    "$FOLLOW/runs/shard_0" "$expected" > "$supplement/SHARD_COMPLETE"
}

run_shard() {
  local index=$1
  "$PYTHON" script/exp7_23/run_intervention_atlas_v0b_shard.py \
    --repo "$REPO" \
    --python "$PYTHON" \
    --model "$MODEL" \
    --shard "$SELECTION/shards/shard_${index}.jsonl" \
    --output-dir "$RUNS/shard_${index}" \
    --shard-index "$index"
}

jobs=(supplement)
for index in $(seq 0 11); do
  jobs+=("shard:$index")
done

lane() {
  local parity=$1 index=0
  for job in "${jobs[@]}"; do
    if (( index % 2 == parity )); then
      echo "[$(date '+%F %T')] START $job"
      if [[ "$job" == supplement ]]; then
        run_supplement
      else
        run_shard "${job#shard:}"
      fi
      echo "[$(date '+%F %T')] DONE $job"
    fi
    index=$((index + 1))
  done
}

lane 0 > "$LOG/lane0.log" 2>&1 &
p0=$!
lane 1 > "$LOG/lane1.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"

"$PYTHON" script/exp7_23/merge_intervention_atlas_v0b.py \
  --roots "$V0B/newgpu3" "$RUNS" "$REUSED" \
  --output-dir "$OUT/merged"

cat > "$OUT/queue_complete.json" <<EOF
{"completed_at":"$(date --iso-8601=seconds)","status":"complete","expected_samples":991}
EOF
echo "[$(date '+%F %T')] ALL FULL ATLAS RUNS DONE" >> "$LOG/watcher.log"
