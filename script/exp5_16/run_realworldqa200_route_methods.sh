#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/realworldqa_mcq_random200_seed42.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/realworldqa200_route_methods"
GPU="${GPU:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy"
TRACE_ARGS="--save_full_token_entropy --trace_topk 20"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

write_run() {
  local run_name="$1"
  local method="$2"
  local extra_args="$3"
  local run_dir="${BASE_DIR}/${run_name}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${DATASET}" \\
  --output_dir "${run_dir}" \\
  --method "${method}" \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "cot_gpu${GPU}" "cot" ""
write_run "lead_gpu${GPU}" "lead" ""
write_run "pure_soft_gpu${GPU}" "pure_soft" "${TRACE_ARGS}"
write_run "bestcombo_gpu${GPU}" "pure_soft" "${TRACE_ARGS} ${BEST_COMBO_ARGS}"
write_run "router_midbias003_gpu${GPU}" "pure_soft" "${TRACE_ARGS} ${BEST_COMBO_ARGS} --pure_soft_image_pad_bias --image_pad_bias_lambda 0.03 --image_pad_bias_min_step 129 --image_pad_bias_max_step 512"

cat > "${BASE_DIR}/queue_gpu${GPU}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/cot_gpu${GPU}/run_command.sh"
bash "${BASE_DIR}/lead_gpu${GPU}/run_command.sh"
bash "${BASE_DIR}/pure_soft_gpu${GPU}/run_command.sh"
bash "${BASE_DIR}/bestcombo_gpu${GPU}/run_command.sh"
bash "${BASE_DIR}/router_midbias003_gpu${GPU}/run_command.sh"
EOF
chmod +x "${BASE_DIR}/queue_gpu${GPU}.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
BASE="${BASE_DIR}"
DATASET="${DATASET}"
for run in cot_gpu${GPU} lead_gpu${GPU} pure_soft_gpu${GPU} bestcombo_gpu${GPU} router_midbias003_gpu${GPU}; do
  if [[ ! -f "\${BASE}/\${run}/results.jsonl" ]]; then
    echo "\${run}: MISSING"
    continue
  fi
  echo "== \${run} =="
  "${PYTHON_BIN}" script/evaluate_realworldqa_mcq.py \\
    --dataset "\${DATASET}" \\
    --results "\${BASE}/\${run}/results.jsonl" \\
    --output_json "\${BASE}/\${run}/realworldqa_mcq_eval.json" \\
    --output_results_jsonl "\${BASE}/\${run}/realworldqa_mcq_enriched.jsonl"
done
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
BASE = Path("${BASE_DIR}")
runs = ["cot_gpu${GPU}", "lead_gpu${GPU}", "pure_soft_gpu${GPU}", "bestcombo_gpu${GPU}", "router_midbias003_gpu${GPU}"]
summary = {}
for run in runs:
    p = BASE / run / "realworldqa_mcq_eval.json"
    if not p.exists():
        continue
    summary[run] = json.loads(p.read_text(encoding="utf-8"))
print("== compact ==")
for run, rep in summary.items():
    print(f"{run}: {rep['correct']}/{rep['total']}={rep['accuracy']*100:.2f}% failed={rep['failed_extraction']}")
(BASE / "realworldqa200_route_methods_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

cat > "${BASE_DIR}/route_summary_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
BASE="${BASE_DIR}"
BASELINE="\${BASE}/pure_soft_gpu${GPU}"
for run in pure_soft_gpu${GPU} bestcombo_gpu${GPU} router_midbias003_gpu${GPU}; do
  if [[ ! -f "\${BASE}/\${run}/token_entropy_full.jsonl" ]]; then
    echo "missing trace: \${run}" >&2
    continue
  fi
  extra=()
  if [[ "\${run}" != "pure_soft_gpu${GPU}" ]]; then
    extra=(--baseline_run_dir "\${BASELINE}")
  fi
  "${PYTHON_BIN}" script/exp5_16/analyze_route_summary.py \\
    --run_dir "\${BASE}/\${run}" \\
    "\${extra[@]}" \\
    --output "\${BASE}/\${run}/route_summary.md" \\
    --output_json "\${BASE}/\${run}/route_summary.json"
done
EOF
chmod +x "${BASE_DIR}/route_summary_after_done.sh"

setsid bash "${BASE_DIR}/queue_gpu${GPU}.sh" > "${BASE_DIR}/queue_gpu${GPU}.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu${GPU}.pid"

echo "BASE_DIR=${BASE_DIR}"
echo "DATASET=${DATASET}"
echo "GPU=${GPU}"
echo "queue PID=$(cat "${BASE_DIR}/queue_gpu${GPU}.pid")"
echo "Runs: cot, lead, pure_soft, bestcombo, router_midbias003"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
echo "Route summaries after done:"
echo "  bash ${BASE_DIR}/route_summary_after_done.sh"
