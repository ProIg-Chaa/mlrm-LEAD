#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/exp1_vstar_spike_type_parallel"

mkdir -p "${BASE_DIR}"

write_run() {
  local name="$1"
  local gpu="$2"
  local method="$3"
  local extra_args="$4"
  local run_dir="${BASE_DIR}/${name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset data/vstar.jsonl \\
  --output_dir "${run_dir}" \\
  --method ${method} \\
  --cot_prompt_mode orign \\
  --max_new_tokens 1024 \\
  --temperature 0.6 \\
  --top_p 0.95 \\
  --top_k 20 \\
  --seed 42 \\
  --device cuda \\
  --no-do_sample \\
  --save_token_entropy \\
  --save_full_token_entropy \\
  --trace_topk 20 \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "cot" 0 "cot" ""
write_run "lead" 1 "lead" "--alpha 0.4 --max_switch_count 5 --window_size 128"
write_run "pure_soft" 0 "pure_soft" ""

cat > "${BASE_DIR}/analyze_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" script/exp5_16/analyze_spike_types.py \\
  --run cot "${BASE_DIR}/cot_gpu0" \\
  --run lead "${BASE_DIR}/lead_gpu1" \\
  --run pure_soft "${BASE_DIR}/pure_soft_gpu0" \\
  --output "${ROOT}/result/exp1_vstar_spike_type_analysis_${STAMP}.md"
EOF
chmod +x "${BASE_DIR}/analyze_after_done.sh"

for run_dir in "${BASE_DIR}"/cot_gpu0 "${BASE_DIR}"/lead_gpu1 "${BASE_DIR}"/pure_soft_gpu0; do
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  echo $! > "${run_dir}/pid.txt"
done

echo "Started exp1 VStar spike-type runs:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  cot       PID=$(cat "${BASE_DIR}/cot_gpu0/pid.txt")       GPU=0"
echo "  lead      PID=$(cat "${BASE_DIR}/lead_gpu1/pid.txt")      GPU=1"
echo "  pure_soft PID=$(cat "${BASE_DIR}/pure_soft_gpu0/pid.txt") GPU=0"
echo "Analyze after all finish:"
echo "  bash ${BASE_DIR}/analyze_after_done.sh"
