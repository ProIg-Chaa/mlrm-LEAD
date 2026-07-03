#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/vstar_lead_soft_quota_sweep"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
START="${START:-1}"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --trace_topk 20"
LEAD_ARGS="--cot_prompt_mode orign --alpha 0.4 --max_switch_count 5 --window_size 128"
GUARD_ARGS="--lead_format_cooldown --format_cooldown_steps 2 --lead_soft_veto_on_diffuse --lead_veto_entropy_window 16 --lead_veto_entropy_alpha 2.0 --lead_veto_min_history 4 --lead_veto_min_entropy 1.0 --lead_veto_low_conf_tau 0.20 --lead_veto_low_margin_tau 0.05 --lead_veto_min_step 64 --lead_veto_require_repeat_degen --lead_veto_repeat_ngram 3 --lead_veto_recent_repeat_window 32 --lead_veto_recent_repeat_tau 0.35"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | awk 'NR==1 {print $1}')"
    if [[ "${used}" -lt 4096 ]]; then
      break
    fi
    echo "GPU${gpu} busy: ${used} MiB used; waiting..."
    sleep 60
  done
}

write_run() {
  local run_name="$1"
  local gpu="$2"
  local ratio="$3"
  local extra_args="$4"
  local run_dir="${BASE_DIR}/${run_name}_gpu${gpu}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${DATASET}" \\
  --output_dir "${run_dir}" \\
  --method lead \\
  ${COMMON_ARGS} \\
  ${LEAD_ARGS} \\
  --lead_soft_quota_ratio "${ratio}" \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "quota05" "${GPU0}" "0.05" ""
write_run "quota10" "${GPU0}" "0.10" ""
write_run "quota20" "${GPU0}" "0.20" ""
write_run "quota05_guard" "${GPU1}" "0.05" "${GUARD_ARGS}"
write_run "quota10_guard" "${GPU1}" "0.10" "${GUARD_ARGS}"
write_run "quota20_guard" "${GPU1}" "0.20" "${GUARD_ARGS}"

cat > "${BASE_DIR}/queue_gpu${GPU0}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/quota05_gpu${GPU0}/run_command.sh"
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/quota10_gpu${GPU0}/run_command.sh"
wait_for_gpu "${GPU0}"
bash "${BASE_DIR}/quota20_gpu${GPU0}/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu${GPU1}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(declare -f wait_for_gpu)
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/quota05_guard_gpu${GPU1}/run_command.sh"
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/quota10_guard_gpu${GPU1}/run_command.sh"
wait_for_gpu "${GPU1}"
bash "${BASE_DIR}/quota20_guard_gpu${GPU1}/run_command.sh"
EOF
chmod +x "${BASE_DIR}/queue_gpu${GPU0}.sh" "${BASE_DIR}/queue_gpu${GPU1}.sh"

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

base = Path("${BASE_DIR}")
runs = [
    ("quota05", "${GPU0}"),
    ("quota10", "${GPU0}"),
    ("quota20", "${GPU0}"),
    ("quota05_guard", "${GPU1}"),
    ("quota10_guard", "${GPU1}"),
    ("quota20_guard", "${GPU1}"),
]

print("# VStar LEAD Soft Quota Sweep")
summary = {}
for run, gpu in runs:
    run_dir = base / f"{run}_gpu{gpu}"
    if not (run_dir / "results.jsonl").exists():
        print(f"{run}: MISSING")
        continue
    report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open("r", encoding="utf-8") if line.strip()]
    lens = [int(row.get("output_tokens") or 0) for row in rows]
    token_rows = []
    token_path = run_dir / "token_entropy.jsonl"
    if token_path.exists():
        token_rows = [json.loads(line) for line in token_path.open("r", encoding="utf-8") if line.strip()]
    soft_ratios = []
    quota_ratios = []
    for row in token_rows:
        summary_row = row.get("entropy_summary") or {}
        if isinstance(summary_row.get("soft_ratio"), (int, float)):
            soft_ratios.append(float(summary_row["soft_ratio"]))
        tokens = row.get("tokens") or []
        if tokens:
            quota_count = sum(1 for token in tokens if token.get("route_action") == "lead_soft_quota")
            quota_ratios.append(quota_count / max(1, len(tokens)))
    item = {
        "accuracy": report.get("accuracy"),
        "correct": report.get("correct"),
        "total": report.get("total"),
        "avg_len": sum(lens) / len(lens) if lens else 0.0,
        "long_ge_256": sum(x >= 256 for x in lens),
        "maxed": sum(x >= 1024 for x in lens),
        "lead_soft_quota_ratio": cfg.get("lead_soft_quota_ratio"),
        "guard": bool(cfg.get("lead_format_cooldown") or cfg.get("lead_soft_veto_on_diffuse")),
        "mean_soft_ratio": sum(soft_ratios) / len(soft_ratios) if soft_ratios else None,
        "mean_quota_route_ratio": sum(quota_ratios) / len(quota_ratios) if quota_ratios else None,
    }
    summary[run] = item
    print(
        f"{run}: {item['correct']}/{item['total']} = {item['accuracy'] * 100:.2f}% | "
        f"quota={item['lead_soft_quota_ratio']} guard={item['guard']} "
        f"avg_len={item['avg_len']:.1f} long>=256={item['long_ge_256']} maxed={item['maxed']} "
        f"mean_soft_ratio={item['mean_soft_ratio']}"
    )

(base / "soft_quota_summary.json").write_text(
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
