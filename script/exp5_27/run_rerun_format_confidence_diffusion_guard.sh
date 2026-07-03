#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL="${MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/rerun_format_confidence_diffusion_guard}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-4096}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
FORMAT_ARGS="--lead_format_cooldown --format_cooldown_steps 2"
VETO_ARGS="--lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35"
QUOTA05_ARGS="--lead_soft_quota_ratio 0.05"
PURE_COLLAPSE_ARGS="--pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"
PURE_FORMAT_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2"

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
  printf "%s\t%s\n" "${gpu}" "${run_dir}/run_command.sh" >> "${BASE_DIR}/queue_manifest.tsv"
}

append_queue() {
  local gpu="$1"
  local run_script="$2"
  {
    echo "wait_for_gpu \"${gpu}\""
    echo "bash \"${run_script}\""
  } >> "${BASE_DIR}/queue_gpu${gpu}.body"
}

rm -f "${BASE_DIR}/queue_manifest.tsv" "${BASE_DIR}/queue_gpu${GPU0}.body" "${BASE_DIR}/queue_gpu${GPU1}.body"

for dataset in vstar mmvp visulogic300 realworldqa_fixed200; do
  write_run phase1_cross_dataset_guard "${dataset}" cot_orign_greedy cot_greedy "${GPU0}" ""
  write_run phase1_cross_dataset_guard "${dataset}" lead_force_normal lead "${GPU1}" "--lead_force_normal"
  write_run phase1_cross_dataset_guard "${dataset}" lead lead "${GPU0}" ""
  write_run phase1_cross_dataset_guard "${dataset}" initial_transition_only lead "${GPU1}" "--lead_initial_transition_only"
  write_run phase1_cross_dataset_guard "${dataset}" lead_format2 lead "${GPU0}" "${FORMAT_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" lead_diffuse_veto lead "${GPU1}" "${VETO_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" lead_guard lead "${GPU0}" "${FORMAT_ARGS} ${VETO_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" quota05 lead "${GPU1}" "${QUOTA05_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" quota05_format2 lead "${GPU0}" "${QUOTA05_ARGS} ${FORMAT_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" quota05_diffuse_veto lead "${GPU1}" "${QUOTA05_ARGS} ${VETO_ARGS}"
  write_run phase1_cross_dataset_guard "${dataset}" quota05_guard lead "${GPU0}" "${QUOTA05_ARGS} ${FORMAT_ARGS} ${VETO_ARGS}"
done

write_run phase2_vstar_puresoft_guard vstar pure_soft pure_soft "${GPU1}" ""
write_run phase2_vstar_puresoft_guard vstar pure_soft_format2 pure_soft "${GPU0}" "${PURE_FORMAT_ARGS}"
write_run phase2_vstar_puresoft_guard vstar pure_soft_diffuse_collapse pure_soft "${GPU1}" "${PURE_COLLAPSE_ARGS}"
write_run phase2_vstar_puresoft_guard vstar pure_soft_guard pure_soft "${GPU0}" "${PURE_FORMAT_ARGS} ${PURE_COLLAPSE_ARGS}"

while IFS=$'\t' read -r gpu script; do
  append_queue "${gpu}" "${script}"
done < "${BASE_DIR}/queue_manifest.tsv"

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
