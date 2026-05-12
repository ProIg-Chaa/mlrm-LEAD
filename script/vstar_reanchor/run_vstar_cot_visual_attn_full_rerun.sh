#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/output/experiments/${STAMP}/vstar_cot_visual_attn_full_rerun_gpu0"

mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \
  --model_name "${MODEL}" \
  --dataset data/vstar.jsonl \
  --output_dir "${RUN_DIR}" \
  --method cot \
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
  --visual_attn_summary_last_k 4
EOF
chmod +x "${RUN_DIR}/run_command.sh"

nohup bash "${RUN_DIR}/run_command.sh" > "${RUN_DIR}/nohup.log" 2>&1 &
echo $! > "${RUN_DIR}/pid.txt"

echo "Started VStar COT visual-attention rerun:"
echo "  RUN_DIR=${RUN_DIR}"
echo "  PID=$(cat "${RUN_DIR}/pid.txt")"
echo
echo "After completion, analyze with:"
echo "  ${PYTHON_BIN} ${ROOT}/script/analyze_cot_visual_attention_vs_entropy.py \\"
echo "    --trace ${RUN_DIR}/token_entropy_full.jsonl \\"
echo "    --results ${RUN_DIR}/results.jsonl \\"
echo "    --evaluator ${ROOT}/lead/evaluator.py \\"
echo "    --results_format default \\"
echo "    --reasoning_only \\"
echo "    --exclude_nonword \\"
echo "    --output_json ${RUN_DIR}/result/vstar_cot_visual_attn_vs_entropy_reasoning.json"
