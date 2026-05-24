#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
REF_CROSS="${ROOT}/output/experiments/20260520_231938/cross_dataset_base_lead_bestcombo"
REF_VSTAR_BEST="${ROOT}/output/experiments/20260520_113540/pure_soft_cooldown2_late64_repeat_gate_vstar_full/cooldown2_late64_repeat_gate_gpu0"
REF_VSTAR_LEAD="${ROOT}/output/experiments/20260516_183300/exp1_vstar_spike_type_parallel/lead_gpu1"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/bestcombo_image_pad_bias_entropy_gate"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"
BIAS_ARGS="--pure_soft_image_pad_bias --image_pad_bias_lambda 0.05"

wait_for_gpu() {
  local gpu="$1"
  local max_used_mb="${2:-20000}"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    if [[ "${used}" -le "${max_used_mb}" ]]; then
      echo "$(date '+%F %T') GPU${gpu} ready: used=${used}MiB <= ${max_used_mb}MiB"
      break
    fi
    echo "$(date '+%F %T') waiting GPU${gpu}: used=${used}MiB > ${max_used_mb}MiB"
    sleep 120
  done
}

write_run_command() {
  local run_name="$1"
  local gpu="$2"
  local dataset="$3"
  local limit_arg="$4"
  local entropy_min="$5"
  local run_dir="${BASE_DIR}/${run_name}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${dataset}" \\
  --output_dir "${run_dir}" \\
  --method pure_soft \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${limit_arg} \\
  ${BEST_COMBO_ARGS} \\
  ${BIAS_ARGS} \\
  --image_pad_bias_entropy_min "${entropy_min}"
EOF
  chmod +x "${run_dir}/run_command.sh"
}

for tag in e10 e15 e20; do
  case "${tag}" in
    e10) ent="1.0" ;;
    e15) ent="1.5" ;;
    e20) ent="2.0" ;;
  esac
  if [[ "${tag}" == "e20" ]]; then
    vm_gpu=1
  else
    vm_gpu=0
  fi
  write_run_command "vstar_${tag}_gpu${vm_gpu}" "${vm_gpu}" "data/vstar.jsonl" "" "${ent}"
  write_run_command "mmvp_${tag}_gpu${vm_gpu}" "${vm_gpu}" "data/mmvp.jsonl" "" "${ent}"
  write_run_command "visulogic300_${tag}_gpu1" 1 "data/visulogic.jsonl" "--limit 300" "${ent}"
done

cat > "${BASE_DIR}/queue_gpu0_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
wait_for_gpu() {
  local gpu="\$1"
  local max_used_mb="\${2:-20000}"
  while true; do
    local used
    used="\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "\${gpu}" | tr -d ' ')"
    if [[ "\${used}" -le "\${max_used_mb}" ]]; then
      echo "\$(date '+%F %T') GPU\${gpu} ready: used=\${used}MiB <= \${max_used_mb}MiB"
      break
    fi
    echo "\$(date '+%F %T') waiting GPU\${gpu}: used=\${used}MiB > \${max_used_mb}MiB"
    sleep 120
  done
}
wait_for_gpu 0 20000
bash "${BASE_DIR}/vstar_e10_gpu0/run_command.sh"
bash "${BASE_DIR}/mmvp_e10_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu0_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
wait_for_gpu() {
  local gpu="\$1"
  local max_used_mb="\${2:-20000}"
  while true; do
    local used
    used="\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "\${gpu}" | tr -d ' ')"
    if [[ "\${used}" -le "\${max_used_mb}" ]]; then
      echo "\$(date '+%F %T') GPU\${gpu} ready: used=\${used}MiB <= \${max_used_mb}MiB"
      break
    fi
    echo "\$(date '+%F %T') waiting GPU\${gpu}: used=\${used}MiB > \${max_used_mb}MiB"
    sleep 120
  done
}
wait_for_gpu 0 20000
bash "${BASE_DIR}/vstar_e15_gpu0/run_command.sh"
bash "${BASE_DIR}/mmvp_e15_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
wait_for_gpu() {
  local gpu="\$1"
  local max_used_mb="\${2:-20000}"
  while true; do
    local used
    used="\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "\${gpu}" | tr -d ' ')"
    if [[ "\${used}" -le "\${max_used_mb}" ]]; then
      echo "\$(date '+%F %T') GPU\${gpu} ready: used=\${used}MiB <= \${max_used_mb}MiB"
      break
    fi
    echo "\$(date '+%F %T') waiting GPU\${gpu}: used=\${used}MiB > \${max_used_mb}MiB"
    sleep 120
  done
}
wait_for_gpu 1 20000
bash "${BASE_DIR}/vstar_e20_gpu1/run_command.sh"
bash "${BASE_DIR}/mmvp_e20_gpu1/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
wait_for_gpu() {
  local gpu="\$1"
  local max_used_mb="\${2:-20000}"
  while true; do
    local used
    used="\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "\${gpu}" | tr -d ' ')"
    if [[ "\${used}" -le "\${max_used_mb}" ]]; then
      echo "\$(date '+%F %T') GPU\${gpu} ready: used=\${used}MiB <= \${max_used_mb}MiB"
      break
    fi
    echo "\$(date '+%F %T') waiting GPU\${gpu}: used=\${used}MiB > \${max_used_mb}MiB"
    sleep 120
  done
}
wait_for_gpu 1 20000
bash "${BASE_DIR}/visulogic300_e10_gpu1/run_command.sh"
bash "${BASE_DIR}/visulogic300_e15_gpu1/run_command.sh"
bash "${BASE_DIR}/visulogic300_e20_gpu1/run_command.sh"
EOF
chmod +x "${BASE_DIR}"/queue_gpu*.sh

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import sys
from pathlib import Path

from lead.evaluator import evaluate_dataset, extract_mcq_answer

ROOT = Path("${ROOT}")
BASE = Path("${BASE_DIR}")
REF_CROSS = Path("${REF_CROSS}")
REF_VSTAR_BEST = Path("${REF_VSTAR_BEST}")
REF_VSTAR_LEAD = Path("${REF_VSTAR_LEAD}")
sys.path.insert(0, str(ROOT / "script"))
from evaluate_specialized_results import load_jsonl, evaluate as specialized_evaluate

def eval_mmvp(run_dir):
    report, _ = specialized_evaluate(
        load_jsonl(ROOT / "data/mmvp.jsonl"),
        load_jsonl(run_dir / "results.jsonl"),
        "mmvp",
    )
    return {
        "correct": report["correct"],
        "total": report["total"],
        "accuracy": report["accuracy"],
        "pair_correct": report["pair_correct"],
        "pair_total": report["pair_total"],
        "pair_accuracy": report["pair_accuracy"],
        "failed_extraction": report["failed_extraction"],
    }

def eval_mcq(run_dir):
    rows = load_jsonl(run_dir / "results.jsonl")
    report = evaluate_dataset(rows)
    failed_real = sum(1 for row in rows if extract_mcq_answer(row.get("model_answer", "")) is None)
    return {
        "correct": report["correct"],
        "total": report["total"],
        "accuracy": report["accuracy"],
        "failed_extraction": report["failed_extraction"],
        "failed_extraction_real": failed_real,
    }

def route_stats(run_dir):
    p = run_dir / "token_entropy_full.jsonl"
    if not p.exists():
        return {}
    total_tokens = format_active = collapsed = image_bias = maxed = rows = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            tokens = row.get("tokens") or []
            total_tokens += len(tokens)
            maxed += int(len(tokens) >= 1024)
            for token in tokens:
                format_active += int(bool(token.get("format_cooldown_active")))
                collapsed += int(bool(token.get("collapse_on_diffuse")))
                image_bias += int(bool(token.get("image_pad_bias_active")))
    denom = max(1, rows)
    return {
        "avg_tokens": total_tokens / denom,
        "avg_format_active": format_active / denom,
        "avg_collapsed": collapsed / denom,
        "avg_image_pad_bias_active": image_bias / denom,
        "maxed": maxed,
    }

runs = {
    "vstar": {
        "eval": eval_mcq,
        "items": {
            "lead_ref": REF_VSTAR_LEAD,
            "bestcombo_ref": REF_VSTAR_BEST,
            "entropy10": BASE / "vstar_e10_gpu0",
            "entropy15": BASE / "vstar_e15_gpu0",
            "entropy20": BASE / "vstar_e20_gpu1",
        },
    },
    "mmvp": {
        "eval": eval_mmvp,
        "items": {
            "lead_ref": REF_CROSS / "mmvp_lead_gpu1",
            "bestcombo_ref": REF_CROSS / "mmvp_bestcombo_gpu1",
            "entropy10": BASE / "mmvp_e10_gpu0",
            "entropy15": BASE / "mmvp_e15_gpu0",
            "entropy20": BASE / "mmvp_e20_gpu1",
        },
    },
    "visulogic300": {
        "eval": eval_mcq,
        "items": {
            "lead_ref": REF_CROSS / "visulogic300_lead_gpu0",
            "bestcombo_ref": REF_CROSS / "visulogic300_bestcombo_gpu1",
            "entropy10": BASE / "visulogic300_e10_gpu1",
            "entropy15": BASE / "visulogic300_e15_gpu1",
            "entropy20": BASE / "visulogic300_e20_gpu1",
        },
    },
}

summary = {}
for dataset, cfg in runs.items():
    print(f"== {dataset} ==")
    summary[dataset] = {}
    for name, run_dir in cfg["items"].items():
        if not (run_dir / "results.jsonl").exists():
            print(f"{name}: MISSING {run_dir}")
            continue
        report = cfg["eval"](run_dir)
        report.update(route_stats(run_dir))
        summary[dataset][name] = report
        if dataset == "mmvp":
            print(
                f"{name}: sample {report['correct']}/{report['total']}={report['accuracy']:.4f}; "
                f"pair {report['pair_correct']}/{report['pair_total']}={report['pair_accuracy']:.4f}; "
                f"bias_active={report.get('avg_image_pad_bias_active', 0):.1f}; "
                f"fmt_active={report.get('avg_format_active', 0):.1f}; maxed={report.get('maxed', 0)}"
            )
        else:
            print(
                f"{name}: {report['correct']}/{report['total']}={report['accuracy']:.4f}; "
                f"bias_active={report.get('avg_image_pad_bias_active', 0):.1f}; "
                f"fmt_active={report.get('avg_format_active', 0):.1f}; "
                f"maxed={report.get('maxed', 0)}; failed_real={report.get('failed_extraction_real', 0)}"
            )
    print()

out = BASE / "bestcombo_image_pad_bias_entropy_gate_summary.json"
with out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"saved {out}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

if [[ "${1:-}" == "--no-launch" ]]; then
  echo "BASE_DIR=${BASE_DIR}"
  exit 0
fi

setsid bash "${BASE_DIR}/queue_gpu0_a.sh" > "${BASE_DIR}/queue_gpu0_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_a.pid"
setsid bash "${BASE_DIR}/queue_gpu0_b.sh" > "${BASE_DIR}/queue_gpu0_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_b.pid"
setsid bash "${BASE_DIR}/queue_gpu1_a.sh" > "${BASE_DIR}/queue_gpu1_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_a.pid"
setsid bash "${BASE_DIR}/queue_gpu1_b.sh" > "${BASE_DIR}/queue_gpu1_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_b.pid"

echo "BASE_DIR=${BASE_DIR}"
echo "queue_gpu0_a PID=$(cat "${BASE_DIR}/queue_gpu0_a.pid"): entropy1.0 vstar -> mmvp"
echo "queue_gpu0_b PID=$(cat "${BASE_DIR}/queue_gpu0_b.pid"): entropy1.5 vstar -> mmvp"
echo "queue_gpu1_a PID=$(cat "${BASE_DIR}/queue_gpu1_a.pid"): entropy2.0 vstar -> mmvp"
echo "queue_gpu1_b PID=$(cat "${BASE_DIR}/queue_gpu1_b.pid"): entropy1.0/1.5/2.0 visulogic"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
