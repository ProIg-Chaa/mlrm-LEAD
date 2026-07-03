#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/cross_dataset_lead_transition_quota"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
GUARD_ARGS="--lead_format_cooldown --format_cooldown_steps 2 --lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35"

dataset_path() {
  case "$1" in
    realworldqa_fixed) echo "${ROOT}/data/realworldqa_fixed_mcq_random200_seed42.jsonl" ;;
    visulogic300) echo "${ROOT}/data/visulogic.jsonl" ;;
    mmvp) echo "${ROOT}/data/mmvp.jsonl" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

limit_arg() {
  case "$1" in
    visulogic300) echo "--limit 300" ;;
    *) echo "" ;;
  esac
}

write_run() {
  local dataset="$1"
  local run_name="$2"
  local gpu="$3"
  local extra_args="$4"
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
  --dataset "$(dataset_path "${dataset}")" \\
  --output_dir "${run_dir}" \\
  --method lead \\
  ${COMMON_ARGS} \\
  ${LEAD_ARGS} \\
  $(limit_arg "${dataset}") \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

for dataset in realworldqa_fixed visulogic300 mmvp; do
  write_run "${dataset}" "lead" "${GPU0}" ""
  write_run "${dataset}" "initial_transition_only" "${GPU1}" "--lead_initial_transition_only"
  write_run "${dataset}" "quota20" "${GPU0}" "--lead_soft_quota_ratio 0.20"
  write_run "${dataset}" "quota05_guard" "${GPU1}" "--lead_soft_quota_ratio 0.05 ${GUARD_ARGS}"
done

cat > "${BASE_DIR}/queue_gpu${GPU0}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/realworldqa_fixed/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/quota20_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/visulogic300/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/visulogic300/quota20_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/mmvp/lead_gpu${GPU0}/run_command.sh"
bash "${BASE_DIR}/mmvp/quota20_gpu${GPU0}/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu${GPU1}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/realworldqa_fixed/initial_transition_only_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/realworldqa_fixed/quota05_guard_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/visulogic300/initial_transition_only_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/visulogic300/quota05_guard_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/mmvp/initial_transition_only_gpu${GPU1}/run_command.sh"
bash "${BASE_DIR}/mmvp/quota05_guard_gpu${GPU1}/run_command.sh"
EOF
chmod +x "${BASE_DIR}/queue_gpu${GPU0}.sh" "${BASE_DIR}/queue_gpu${GPU1}.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"

BASE="${BASE_DIR}"

for run in lead initial_transition_only quota20 quota05_guard; do
  case "\${run}" in
    lead|quota20) gpu="${GPU0}" ;;
    *) gpu="${GPU1}" ;;
  esac
  rw_dir="\${BASE}/realworldqa_fixed/\${run}_gpu\${gpu}"
  if [[ -f "\${rw_dir}/results.jsonl" ]]; then
    "${PYTHON_BIN}" script/evaluate_realworldqa_mcq.py \\
      --dataset data/realworldqa_fixed_mcq_random200_seed42.jsonl \\
      --results "\${rw_dir}/results.jsonl" \\
      --output_json "\${rw_dir}/realworldqa_mcq_eval.json" \\
      --output_results_jsonl "\${rw_dir}/realworldqa_mcq_eval_rows.jsonl" >/dev/null
  fi
  mmvp_dir="\${BASE}/mmvp/\${run}_gpu\${gpu}"
  if [[ -f "\${mmvp_dir}/results.jsonl" ]]; then
    "${PYTHON_BIN}" script/evaluate_specialized_results.py \\
      --dataset data/mmvp.jsonl \\
      --results "\${mmvp_dir}/results.jsonl" \\
      --output_json "\${mmvp_dir}/specialized_eval_report.json" \\
      --output_results_jsonl "\${mmvp_dir}/specialized_eval_rows.jsonl" >/dev/null
  fi
done

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

base = Path("${BASE_DIR}")
runs = ["lead", "initial_transition_only", "quota20", "quota05_guard"]
gpu_for = {
    "lead": "${GPU0}",
    "quota20": "${GPU0}",
    "initial_transition_only": "${GPU1}",
    "quota05_guard": "${GPU1}",
}
datasets = ["realworldqa_fixed", "visulogic300", "mmvp"]

def load_report(dataset, run, run_dir):
    if dataset == "realworldqa_fixed" and (run_dir / "realworldqa_mcq_eval.json").exists():
        return json.loads((run_dir / "realworldqa_mcq_eval.json").read_text(encoding="utf-8")), "realworldqa_mcq"
    if dataset == "mmvp" and (run_dir / "specialized_eval_report.json").exists():
        return json.loads((run_dir / "specialized_eval_report.json").read_text(encoding="utf-8")), "specialized_mmvp"
    return json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8")), "default"

summary = {}
for dataset in datasets:
    print(f"## {dataset}")
    summary[dataset] = {}
    for run in runs:
        run_dir = base / dataset / f"{run}_gpu{gpu_for[run]}"
        if not (run_dir / "results.jsonl").exists() or not (run_dir / "eval_report.json").exists():
            print(f"{run}: MISSING")
            continue
        report, report_type = load_report(dataset, run, run_dir)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (run_dir / "results.jsonl").open("r", encoding="utf-8") if line.strip()]
        lens = [int(row.get("output_tokens") or 0) for row in rows]
        token_rows = []
        token_path = run_dir / "token_entropy.jsonl"
        if token_path.exists():
            token_rows = [json.loads(line) for line in token_path.open("r", encoding="utf-8") if line.strip()]
        soft_ratios = []
        for row in token_rows:
            es = row.get("entropy_summary") or {}
            if isinstance(es.get("soft_ratio"), (int, float)):
                soft_ratios.append(float(es["soft_ratio"]))
        item = {
            "correct": report.get("correct"),
            "total": report.get("total"),
            "accuracy": report.get("accuracy"),
            "failed_extraction": report.get("failed_extraction"),
            "pair_accuracy": report.get("pair_accuracy"),
            "pair_correct": report.get("pair_correct"),
            "pair_total": report.get("pair_total"),
            "report_type": report_type,
            "avg_len": sum(lens) / len(lens) if lens else 0.0,
            "maxed": sum(x >= 1024 for x in lens),
            "mean_soft_ratio": sum(soft_ratios) / len(soft_ratios) if soft_ratios else None,
            "lead_initial_transition_only": cfg.get("lead_initial_transition_only"),
            "lead_soft_quota_ratio": cfg.get("lead_soft_quota_ratio"),
            "guard": bool(cfg.get("lead_format_cooldown") or cfg.get("lead_soft_veto_on_diffuse")),
        }
        summary[dataset][run] = item
        pair = ""
        if item["pair_accuracy"] is not None:
            pair = f" pair={item['pair_correct']}/{item['pair_total']}={item['pair_accuracy']*100:.2f}%"
        print(
            f"{run}: {item['correct']}/{item['total']}={item['accuracy']*100:.2f}%{pair} "
            f"type={report_type} avg_len={item['avg_len']:.1f} maxed={item['maxed']} "
            f"soft={None if item['mean_soft_ratio'] is None else item['mean_soft_ratio']*100:.2f}%"
        )
    print()

(base / "cross_dataset_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

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
