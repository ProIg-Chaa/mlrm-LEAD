#!/usr/bin/env bash
set -euo pipefail

LEAD_ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
ORIGN_ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-orign"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
VSTAR_DATASET="${LEAD_ROOT}/data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${LEAD_ROOT}/output/experiments/${STAMP}/vstar_original_cot_and_full_early_parallel"

ORIGN_RUN_DIR="${RUN_ROOT}/orign_cot_full_gpu0"
EARLY_RUN_DIR="${RUN_ROOT}/lead_full_dynamic_early_gpu1"

mkdir -p "${ORIGN_RUN_DIR}" "${EARLY_RUN_DIR}"

cat > "${ORIGN_RUN_DIR}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ORIGN_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \
  --model_name "${MODEL}" \
  --dataset "${VSTAR_DATASET}" \
  --output_dir "${ORIGN_RUN_DIR}" \
  --method cot \
  --max_new_tokens 1024 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42 \
  --device cuda
EOF
chmod +x "${ORIGN_RUN_DIR}/run_command.sh"

cat > "${EARLY_RUN_DIR}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${LEAD_ROOT}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \
  --model_name "${MODEL}" \
  --dataset "${VSTAR_DATASET}" \
  --output_dir "${EARLY_RUN_DIR}" \
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
  --reanchor_min_step 0 \
  --reanchor_max_step 10 \
  --reanchor_anchor_mode dynamic \
  --reanchor_trigger_mode absolute
EOF
chmod +x "${EARLY_RUN_DIR}/run_command.sh"

setsid bash "${ORIGN_RUN_DIR}/run_command.sh" > "${ORIGN_RUN_DIR}/nohup.log" 2>&1 < /dev/null &
echo $! > "${ORIGN_RUN_DIR}/pid.txt"

setsid bash "${EARLY_RUN_DIR}/run_command.sh" > "${EARLY_RUN_DIR}/nohup.log" 2>&1 < /dev/null &
echo $! > "${EARLY_RUN_DIR}/pid.txt"

echo "Started VStar original COT + full dynamic early:"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  ORIGN_COT_RUN_DIR=${ORIGN_RUN_DIR}"
echo "  ORIGN_COT_PID=$(cat "${ORIGN_RUN_DIR}/pid.txt")"
echo "  EARLY_FULL_RUN_DIR=${EARLY_RUN_DIR}"
echo "  EARLY_FULL_PID=$(cat "${EARLY_RUN_DIR}/pid.txt")"
