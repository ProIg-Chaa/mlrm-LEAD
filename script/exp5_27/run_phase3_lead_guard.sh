#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/phase3_lead_guard"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
FORMAT_ARGS="--lead_format_cooldown --format_cooldown_steps 2"
VETO_ARGS="--lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35"

dataset_path() {
  case "$1" in
    vstar) echo "${ROOT}/data/vstar.jsonl" ;;
    mmvp) echo "${ROOT}/data/mmvp.jsonl" ;;
    realworldqa_fixed) echo "${ROOT}/data/realworldqa_fixed_mcq_random200_seed42.jsonl" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

write_run() {
  local dataset="$1"
  local run_name="$2"
  local gpu="$3"
  local extra_args="$4"
  local ds_path
  ds_path="$(dataset_path "${dataset}")"
  local run_dir="${BASE_DIR}/${dataset}/${run_name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${ds_path}" \\
  --output_dir "${run_dir}" \\
  --method lead \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

for dataset in vstar mmvp realworldqa_fixed; do
  write_run "${dataset}" "lead" "${GPU0}" ""
  write_run "${dataset}" "lead_format2" "${GPU1}" "${FORMAT_ARGS}"
  write_run "${dataset}" "lead_veto_late64_repeat" "${GPU0}" "${VETO_ARGS}"
  write_run "${dataset}" "lead_guard" "${GPU1}" "${FORMAT_ARGS} ${VETO_ARGS}"
done

cat > "${BASE_DIR}/queue_gpu${GPU0}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/vstar/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/vstar/lead_veto_late64_repeat_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/mmvp/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/mmvp/lead_veto_late64_repeat_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/lead_veto_late64_repeat_gpu${GPU0}/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu${GPU1}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/vstar/lead_format2_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/vstar/lead_guard_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/mmvp/lead_format2_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/mmvp/lead_guard_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/lead_format2_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/lead_guard_gpu${GPU1}/run_command.sh"
EOF
chmod +x "${BASE_DIR}/queue_gpu${GPU0}.sh" "${BASE_DIR}/queue_gpu${GPU1}.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
BASE="${BASE_DIR}"
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

base = Path("${BASE_DIR}")
datasets = ["vstar", "mmvp", "realworldqa_fixed"]
runs = ["lead", "lead_format2", "lead_veto_late64_repeat", "lead_guard"]
gpu_for = {
    "lead": "${GPU0}",
    "lead_veto_late64_repeat": "${GPU0}",
    "lead_format2": "${GPU1}",
    "lead_guard": "${GPU1}",
}

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def length_stats(run_dir):
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open("r", encoding="utf-8") if line.strip()]
    lens = [int(row.get("output_tokens") or 0) for row in rows]
    return {
        "avg_len": sum(lens) / len(lens) if lens else 0.0,
        "long_ge_256": sum(x >= 256 for x in lens),
        "maxed": sum(x >= 1024 for x in lens),
    }

summary = {}
for dataset in datasets:
    summary[dataset] = {}
    print(f"## {dataset}")
    for run in runs:
        run_dir = base / dataset / f"{run}_gpu{gpu_for[run]}"
        result_path = run_dir / "results.jsonl"
        if not result_path.exists():
            print(f"{run}: MISSING")
            continue
        eval_report = load_json(run_dir / "eval_report.json") if (run_dir / "eval_report.json").exists() else {}
        stats = length_stats(run_dir)
        rep = {
            "accuracy": eval_report.get("accuracy"),
            "correct": eval_report.get("correct"),
            "total": eval_report.get("total"),
            **stats,
        }
        if dataset == "realworldqa_fixed":
            # Prefer fixed RealWorldQA MCQ evaluator for the final count.
            pass
        summary[dataset][run] = rep
        acc = rep["accuracy"]
        acc_s = "NA" if acc is None else f"{acc * 100:.2f}%"
        print(
            f"{run}: acc={acc_s} correct={rep.get('correct')}/{rep.get('total')} "
            f"avg_len={rep['avg_len']:.1f} long>=256={rep['long_ge_256']} maxed={rep['maxed']}"
        )
    print()

(base / "phase3_lead_guard_summary_builtin.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
print(f"saved {base / 'phase3_lead_guard_summary_builtin.json'}")
PY

for run in lead lead_format2 lead_veto_late64_repeat lead_guard; do
  case "\${run}" in
    lead|lead_veto_late64_repeat) gpu="${GPU0}" ;;
    *) gpu="${GPU1}" ;;
  esac
  run_dir="\${BASE}/realworldqa_fixed/\${run}_gpu\${gpu}"
  if [[ -f "\${run_dir}/results.jsonl" ]]; then
    echo "== realworldqa_fixed \${run} =="
    "${PYTHON_BIN}" script/evaluate_realworldqa_mcq.py \\
      --dataset data/realworldqa_fixed_mcq_random200_seed42.jsonl \\
      --results "\${run_dir}/results.jsonl" \\
      --output_json "\${run_dir}/realworldqa_mcq_eval.json" \\
      --output_results_jsonl "\${run_dir}/realworldqa_mcq_eval_rows.jsonl"
  fi
done

for run in lead lead_format2 lead_veto_late64_repeat lead_guard; do
  case "\${run}" in
    lead|lead_veto_late64_repeat) gpu="${GPU0}" ;;
    *) gpu="${GPU1}" ;;
  esac
  run_dir="\${BASE}/mmvp/\${run}_gpu\${gpu}"
  if [[ -f "\${run_dir}/results.jsonl" ]]; then
    "${PYTHON_BIN}" script/evaluate_mmvp_official.py \\
      --results "\${run_dir}/results.jsonl" \\
      --answer_file "\${run_dir}/mmvp_official_answers.jsonl" \\
      --judge_output "\${run_dir}/mmvp_official_judged.jsonl" \\
      --report_json "\${run_dir}/mmvp_official_convert_only.json" \\
      --convert_only >/dev/null
  fi
done
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

cat > "${BASE_DIR}/route_summary_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
BASE="${BASE_DIR}"
for dataset in vstar mmvp realworldqa_fixed; do
  baseline="\${BASE}/\${dataset}/lead_gpu${GPU0}"
  for spec in "lead:${GPU0}" "lead_format2:${GPU1}" "lead_veto_late64_repeat:${GPU0}" "lead_guard:${GPU1}"; do
    run="\${spec%%:*}"
    gpu="\${spec##*:}"
    run_dir="\${BASE}/\${dataset}/\${run}_gpu\${gpu}"
    if [[ ! -f "\${run_dir}/token_entropy_full.jsonl" ]]; then
      echo "missing trace: \${dataset}/\${run}" >&2
      continue
    fi
    extra=()
    if [[ "\${run}" != "lead" && -f "\${baseline}/results.jsonl" ]]; then
      extra=(--baseline_run_dir "\${baseline}")
    fi
    "${PYTHON_BIN}" script/exp5_16/analyze_route_summary.py \\
      --run_dir "\${run_dir}" \\
      "\${extra[@]}" \\
      --output "\${run_dir}/route_summary.md" \\
      --output_json "\${run_dir}/route_summary.json"
  done
done
EOF
chmod +x "${BASE_DIR}/route_summary_after_done.sh"

if [[ "${START}" == "1" ]]; then
  setsid bash "${BASE_DIR}/queue_gpu${GPU0}.sh" > "${BASE_DIR}/queue_gpu${GPU0}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU0}.pid"
  setsid bash "${BASE_DIR}/queue_gpu${GPU1}.sh" > "${BASE_DIR}/queue_gpu${GPU1}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/queue_gpu${GPU1}.pid"
else
  echo "START=0, queues were generated but not launched."
fi

echo "BASE_DIR=${BASE_DIR}"
if [[ -f "${BASE_DIR}/queue_gpu${GPU0}.pid" ]]; then
  echo "GPU${GPU0} queue PID=$(cat "${BASE_DIR}/queue_gpu${GPU0}.pid")"
fi
if [[ -f "${BASE_DIR}/queue_gpu${GPU1}.pid" ]]; then
  echo "GPU${GPU1} queue PID=$(cat "${BASE_DIR}/queue_gpu${GPU1}.pid")"
fi
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
echo "Route summaries after done:"
echo "  bash ${BASE_DIR}/route_summary_after_done.sh"
