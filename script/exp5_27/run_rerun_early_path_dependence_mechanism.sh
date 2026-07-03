#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python}"
MODEL="${MODEL:-/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${ROOT}/output/experiments/${STAMP}/rerun_early_path_dependence_mechanism}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-4096}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
GUARD_ARGS="--lead_soft_quota_ratio 0.05 --lead_format_cooldown --format_cooldown_steps 2 --lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35"

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

compile_check() {
  cd "${ROOT}"
  "${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py
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
compile_check() { "${PYTHON_BIN}" -m py_compile main.py lead/inference.py lead/generation_utils.py; }
compile_check
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

rm -f "${BASE_DIR}/queue_gpu${GPU0}.body" "${BASE_DIR}/queue_gpu${GPU1}.body" "${BASE_DIR}/queue_manifest.tsv"

# Phase 1: clean VStar component controls.
write_run phase1_vstar_mechanism vstar cot_orign_greedy cot_greedy "${GPU0}" ""
write_run phase1_vstar_mechanism vstar lead_force_normal lead "${GPU1}" "--lead_force_normal"
write_run phase1_vstar_mechanism vstar lead lead "${GPU0}" ""
write_run phase1_vstar_mechanism vstar initial_soft_only lead "${GPU1}" "--lead_initial_soft_only"
write_run phase1_vstar_mechanism vstar initial_transition_only lead "${GPU0}" "--lead_initial_transition_only"
write_run phase1_vstar_mechanism vstar initial_transition_no_to_normal lead "${GPU1}" "--lead_initial_transition_only --lead_disable_to_normal_transition"
write_run phase1_vstar_mechanism vstar initial_transition_no_linebreak lead "${GPU0}" "--lead_initial_transition_only --lead_disable_step0_linebreak_mix"
write_run phase1_vstar_mechanism vstar initial_transition_no_anchor lead "${GPU1}" "--lead_initial_transition_only --lead_disable_simple_visual_anchor"
write_run phase1_vstar_mechanism vstar initial_transition_no_linebreak_no_to_normal lead "${GPU0}" "--lead_initial_transition_only --lead_disable_step0_linebreak_mix --lead_disable_to_normal_transition"

# Phase 2: VStar full timing curve.
for step in 0 1 2 4 8 16 32; do
  gpu="${GPU0}"
  if [[ "${step}" == "1" || "${step}" == "4" || "${step}" == "16" ]]; then
    gpu="${GPU1}"
  fi
  write_run phase2_timing_curve vstar "transition_step${step}" lead "${gpu}" "--lead_initial_transition_only --lead_initial_transition_delay_steps ${step}"
done

# Phase 2 cross-dataset projection for the most informative delays.
for dataset in mmvp visulogic300 realworldqa_fixed200; do
  for step in 0 4 16 32; do
    gpu="${GPU0}"
    if [[ "${step}" == "4" || "${step}" == "32" ]]; then
      gpu="${GPU1}"
    fi
    write_run phase2_timing_curve_cross "${dataset}" "transition_step${step}" lead "${gpu}" "--lead_initial_transition_only --lead_initial_transition_delay_steps ${step}"
  done
done

# Phase 3: cross-dataset minimal confirmation.
for dataset in vstar mmvp visulogic300 realworldqa_fixed200; do
  write_run phase3_cross_dataset_minimal "${dataset}" cot_orign_greedy cot_greedy "${GPU0}" ""
  write_run phase3_cross_dataset_minimal "${dataset}" lead_force_normal lead "${GPU1}" "--lead_force_normal"
  write_run phase3_cross_dataset_minimal "${dataset}" lead lead "${GPU0}" ""
  write_run phase3_cross_dataset_minimal "${dataset}" initial_soft_only lead "${GPU1}" "--lead_initial_soft_only"
  write_run phase3_cross_dataset_minimal "${dataset}" initial_transition_only lead "${GPU0}" "--lead_initial_transition_only"
  write_run phase3_cross_dataset_minimal "${dataset}" initial_transition_no_to_normal lead "${GPU1}" "--lead_initial_transition_only --lead_disable_to_normal_transition"
  write_run phase3_cross_dataset_minimal "${dataset}" initial_transition_no_anchor lead "${GPU0}" "--lead_initial_transition_only --lead_disable_simple_visual_anchor"
  write_run phase3_cross_dataset_minimal "${dataset}" quota05_guard lead "${GPU1}" "${GUARD_ARGS}"
done

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
"${PYTHON_BIN}" script/exp5_27/analyze_early_token_divergence.py --base_dir "${BASE_DIR}" --root "${ROOT}"
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

compile_check

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
