#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
SOURCE_RUN="${ROOT}/output/experiments/20260513_210625/vstar_cot_orign_aligned_full_gpu0"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/output/experiments/${STAMP}/vstar_cot_orign_aligned_sidecar_attention_gpu1"

mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" script/vstar_reanchor/replay_cot_attention_sidecar.py \
  --model_name "${MODEL}" \
  --results "${SOURCE_RUN}/results.jsonl" \
  --output_dir "${RUN_DIR}" \
  --device cuda \
  --cot_prompt_mode orign \
  --attn_last_k 4
EOF
chmod +x "${RUN_DIR}/run_command.sh"

setsid bash "${RUN_DIR}/run_command.sh" > "${RUN_DIR}/nohup.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pid.txt"

echo "Started VStar sidecar attention replay:"
echo "  SOURCE_RUN=${SOURCE_RUN}"
echo "  RUN_DIR=${RUN_DIR}"
echo "  PID=$(cat "${RUN_DIR}/pid.txt")"
