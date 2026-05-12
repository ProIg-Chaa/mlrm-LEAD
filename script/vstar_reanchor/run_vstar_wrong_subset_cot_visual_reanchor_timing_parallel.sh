#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar_wrong_subset_from_cot_visual_attn_rerun.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT}/output/experiments/${STAMP}/vstar_wrong_subset_cot_visual_reanchor_timing_parallel"

mkdir -p "${RUN_ROOT}"

launch_run() {
  local name="$1"
  local gpu="$2"
  local min_step="$3"
  local max_step="$4"
  local run_dir="${RUN_ROOT}/${name}"
  mkdir -p "${run_dir}"

  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \
  --model_name "${MODEL}" \
  --dataset "${DATASET}" \
  --output_dir "${run_dir}" \
  --method cot_visual_reanchor \
  --max_new_tokens 1024 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --device cuda \
  --no-do_sample \
  --save_token_entropy \
  --save_full_token_entropy \
  --save_visual_attn_summary \
  --visual_attn_summary_last_k 4 \
  --reanchor_entropy_threshold 1.0 \
  --reanchor_visual_attn_threshold 0.12 \
  --reanchor_lambda 0.15 \
  --reanchor_top_m 4 \
  --reanchor_attn_last_k 4 \
  --reanchor_max_trigger_count 1 \
  --reanchor_cooldown 32 \
  --reanchor_min_step ${min_step} \
  --reanchor_max_step ${max_step}
EOF
  chmod +x "${run_dir}/run_command.sh"
  nohup bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 &
  echo $! > "${run_dir}/pid.txt"
  echo "${name}: RUN_DIR=${run_dir} PID=$(cat "${run_dir}/pid.txt") GPU=${gpu}"
}

# early: step <= 10
launch_run "early_step_le_10_gpu0" 0 0 10

# mid: 11 <= step <= 30
launch_run "mid_step_11_30_gpu1" 1 11 30

echo
echo "RUN_ROOT=${RUN_ROOT}"
echo "Queued separately after one GPU frees:"
echo "  late_step_ge_31 (min_step=31, max_step=1000000)"
