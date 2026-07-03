#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL="${MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/tune_transition_delay_quota_format}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-4096}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
FORMAT_ARGS="--lead_format_cooldown --format_cooldown_steps 2"

dataset_path() {
  case "$1" in
    vstar) echo "${ROOT}/data/vstar.jsonl" ;;
    mmvp) echo "${ROOT}/data/mmvp.jsonl" ;;
    visulogic300) echo "${ROOT}/data/visulogic.jsonl" ;;
    realworldqa_fixed200) echo "${ROOT}/data/realworldqa_fixed_mcq_random200_seed42.jsonl" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

limit_arg() {
  case "$1" in
    visulogic300) echo "--limit 300" ;;
    *) echo "" ;;
  esac
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk 'NR==1 {print $1}')"
    if [[ "${used}" -lt "${GPU_MEM_LIMIT_MB}" ]]; then
      break
    fi
    echo "GPU${gpu} busy: ${used} MiB used; waiting..."
    sleep 60
  done
}

write_run() {
  local phase="$1"
  local dataset="$2"
  local run_name="$3"
  local method="$4"
  local gpu="$5"
  local extra_args="$6"
  local run_dir="${BASE_DIR}/${phase}/${dataset}/${run_name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "$(dataset_path "${dataset}")" \\
  --output_dir "${run_dir}" \\
  --method "${method}" \\
  ${COMMON_ARGS} \\
  ${LEAD_ARGS} \\
  $(limit_arg "${dataset}") \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

append_queue() {
  local gpu="$1"
  local run_script="$2"
  {
    echo "wait_for_gpu \"${gpu}\""
    echo "bash \"${run_script}\""
  } >> "${BASE_DIR}/queue_gpu${gpu}.body"
}

rm -f "${BASE_DIR}/queue_gpu${GPU0}.body" "${BASE_DIR}/queue_gpu${GPU1}.body"

# Direction 1: refine early transition timing where prior matrix lacked step1/step2.
for dataset in mmvp realworldqa_fixed200; do
  write_run phase1_transition_delay_refine "${dataset}" transition_step1 lead "${GPU0}" "--lead_initial_transition_only --lead_initial_transition_delay_steps 1"
  write_run phase1_transition_delay_refine "${dataset}" transition_step2 lead "${GPU1}" "--lead_initial_transition_only --lead_initial_transition_delay_steps 2"
done
write_run phase1_transition_delay_refine visulogic300 transition_step1 lead "${GPU0}" "--lead_initial_transition_only --lead_initial_transition_delay_steps 1"
write_run phase1_transition_delay_refine visulogic300 transition_step2 lead "${GPU0}" "--lead_initial_transition_only --lead_initial_transition_delay_steps 2"

# Direction 2: quota ratio and quota+format2 sweep. Keep VisuLogic out for now:
# quota was weak there and it is the long-run dataset that triggered kills.
for dataset in vstar mmvp realworldqa_fixed200; do
  for ratio in 0.02 0.03 0.05 0.08; do
    tag="${ratio/./}"
    write_run phase2_quota_format_sweep "${dataset}" "quota${tag}" lead "${GPU0}" "--lead_soft_quota_ratio ${ratio}"
    write_run phase2_quota_format_sweep "${dataset}" "quota${tag}_format2" lead "${GPU1}" "--lead_soft_quota_ratio ${ratio} ${FORMAT_ARGS}"
  done
done

# Ordered queues. Only GPU0 runs VisuLogic transition refinements, and they are near
# the front so they do not overlap with another VisuLogic run.
append_queue "${GPU0}" "${BASE_DIR}/phase1_transition_delay_refine/mmvp/transition_step1_gpu${GPU0}/run_command.sh"
append_queue "${GPU0}" "${BASE_DIR}/phase1_transition_delay_refine/realworldqa_fixed200/transition_step1_gpu${GPU0}/run_command.sh"
append_queue "${GPU0}" "${BASE_DIR}/phase1_transition_delay_refine/visulogic300/transition_step1_gpu${GPU0}/run_command.sh"
append_queue "${GPU0}" "${BASE_DIR}/phase1_transition_delay_refine/visulogic300/transition_step2_gpu${GPU0}/run_command.sh"
for dataset in vstar mmvp realworldqa_fixed200; do
  for ratio in 0.02 0.03 0.05 0.08; do
    tag="${ratio/./}"
    append_queue "${GPU0}" "${BASE_DIR}/phase2_quota_format_sweep/${dataset}/quota${tag}_gpu${GPU0}/run_command.sh"
  done
done

append_queue "${GPU1}" "${BASE_DIR}/phase1_transition_delay_refine/mmvp/transition_step2_gpu${GPU1}/run_command.sh"
append_queue "${GPU1}" "${BASE_DIR}/phase1_transition_delay_refine/realworldqa_fixed200/transition_step2_gpu${GPU1}/run_command.sh"
for dataset in vstar mmvp realworldqa_fixed200; do
  for ratio in 0.02 0.03 0.05 0.08; do
    tag="${ratio/./}"
    append_queue "${GPU1}" "${BASE_DIR}/phase2_quota_format_sweep/${dataset}/quota${tag}_format2_gpu${GPU1}/run_command.sh"
  done
done

for gpu in "${GPU0}" "${GPU1}"; do
  cat > "${BASE_DIR}/queue_gpu${gpu}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB}"
$(cat "${BASE_DIR}/queue_gpu${gpu}.body")
EOF
  chmod +x "${BASE_DIR}/queue_gpu${gpu}.sh"
done

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" script/exp5_27/summarize_rerun_early_path_dependence.py --base_dir "${BASE_DIR}" --root "${ROOT}"
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

"${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py script/exp5_27/summarize_rerun_early_path_dependence.py

if [[ "${START}" == "1" ]]; then
  setsid bash "${BASE_DIR}/queue_gpu${GPU0}.sh" > "${BASE_DIR}/queue_gpu${GPU0}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU0}.pid"
  setsid bash "${BASE_DIR}/queue_gpu${GPU1}.sh" > "${BASE_DIR}/queue_gpu${GPU1}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU1}.pid"
else
  echo "START=0, queues were generated but not launched."
fi

echo "BASE_DIR=${BASE_DIR}"
for gpu in "${GPU0}" "${GPU1}"; do
  if [[ -f "${BASE_DIR}/queue_gpu${gpu}.pid" ]]; then
    echo "GPU${gpu} queue PID=$(cat "${BASE_DIR}/queue_gpu${gpu}.pid")"
  fi
  echo "GPU${gpu} queue log: ${BASE_DIR}/queue_gpu${gpu}.log"
done
echo "Compare after done: bash ${BASE_DIR}/compare_after_done.sh"
