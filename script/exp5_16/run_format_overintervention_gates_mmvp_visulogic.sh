#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
REF_BASE="${ROOT}/output/experiments/20260520_231938/cross_dataset_base_lead_bestcombo"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/format_overintervention_gates_mmvp_visulogic"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

write_run_command() {
  local run_name="$1"
  local gpu="$2"
  local dataset="$3"
  local limit_arg="$4"
  local gate_args="$5"
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
  ${gate_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

# Gate semantics: OR across provided conditions. A format token starts cooldown only when
# raw entropy is high enough, or top1/margin show instability.
write_run_command "mmvp_gate_entropy10_gpu0" 0 "data/mmvp.jsonl" "" "--format_cooldown_entropy_min 1.0"
write_run_command "mmvp_gate_top080_margin040_gpu0" 0 "data/mmvp.jsonl" "" "--format_cooldown_top1_max 0.80 --format_cooldown_margin_max 0.40"
write_run_command "mmvp_gate_strict_gpu1" 1 "data/mmvp.jsonl" "" "--format_cooldown_entropy_min 1.5 --format_cooldown_top1_max 0.60 --format_cooldown_margin_max 0.25"

write_run_command "visulogic300_gate_entropy10_gpu1" 1 "data/visulogic.jsonl" "--limit 300" "--format_cooldown_entropy_min 1.0"
write_run_command "visulogic300_gate_top080_margin040_gpu0" 0 "data/visulogic.jsonl" "--limit 300" "--format_cooldown_top1_max 0.80 --format_cooldown_margin_max 0.40"
write_run_command "visulogic300_gate_strict_gpu1" 1 "data/visulogic.jsonl" "--limit 300" "--format_cooldown_entropy_min 1.5 --format_cooldown_top1_max 0.60 --format_cooldown_margin_max 0.25"

cat > "${BASE_DIR}/queue_gpu0_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_gate_entropy10_gpu0/run_command.sh"
bash "${BASE_DIR}/visulogic300_gate_top080_margin040_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu0_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_gate_top080_margin040_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_gate_strict_gpu1/run_command.sh"
bash "${BASE_DIR}/visulogic300_gate_strict_gpu1/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/visulogic300_gate_entropy10_gpu1/run_command.sh"
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
REF_BASE = Path("${REF_BASE}")
sys.path.insert(0, str(ROOT / "script"))
from evaluate_specialized_results import load_jsonl, evaluate as specialized_evaluate

def eval_mmvp(run_dir):
    dataset_rows = load_jsonl(ROOT / "data/mmvp.jsonl")
    result_rows = load_jsonl(run_dir / "results.jsonl")
    report, _ = specialized_evaluate(dataset_rows, result_rows, "mmvp")
    return {
        "correct": report["correct"],
        "total": report["total"],
        "accuracy": report["accuracy"],
        "pair_correct": report["pair_correct"],
        "pair_total": report["pair_total"],
        "pair_accuracy": report["pair_accuracy"],
        "failed_extraction": report["failed_extraction"],
    }

def eval_visulogic(run_dir):
    rows = load_jsonl(run_dir / "results.jsonl")
    report = evaluate_dataset(rows)
    failed_real = 0
    for row in rows:
        if extract_mcq_answer(row.get("model_answer", "")) is None:
            failed_real += 1
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
    total_tokens = 0
    format_active = 0
    format_tokens = 0
    collapsed = 0
    maxed = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = row.get("tokens") or []
            total_tokens += len(tokens)
            maxed += int(len(tokens) >= 1024)
            for token in tokens:
                format_active += int(bool(token.get("format_cooldown_active")))
                format_tokens += int(bool(token.get("format_token")))
                collapsed += int(bool(token.get("collapse_on_diffuse")))
    return {
        "avg_tokens": total_tokens / 300,
        "avg_format_active": format_active / 300,
        "avg_format_tokens": format_tokens / 300,
        "avg_collapsed": collapsed / 300,
        "maxed": maxed,
    }

runs = {
    "mmvp": {
        "pure_soft_ref": REF_BASE / "mmvp_pure_soft_gpu1",
        "lead_ref": REF_BASE / "mmvp_lead_gpu1",
        "bestcombo_ref": REF_BASE / "mmvp_bestcombo_gpu1",
        "gate_entropy10": BASE / "mmvp_gate_entropy10_gpu0",
        "gate_top080_margin040": BASE / "mmvp_gate_top080_margin040_gpu0",
        "gate_strict": BASE / "mmvp_gate_strict_gpu1",
    },
    "visulogic300": {
        "pure_soft_ref": REF_BASE / "visulogic300_pure_soft_gpu0",
        "lead_ref": REF_BASE / "visulogic300_lead_gpu0",
        "bestcombo_ref": REF_BASE / "visulogic300_bestcombo_gpu1",
        "gate_entropy10": BASE / "visulogic300_gate_entropy10_gpu1",
        "gate_top080_margin040": BASE / "visulogic300_gate_top080_margin040_gpu0",
        "gate_strict": BASE / "visulogic300_gate_strict_gpu1",
    },
}

summary = {}
for dataset, items in runs.items():
    print(f"== {dataset} ==")
    summary[dataset] = {}
    for name, run_dir in items.items():
        if not (run_dir / "results.jsonl").exists():
            print(f"{name}: MISSING {run_dir}")
            continue
        report = eval_mmvp(run_dir) if dataset == "mmvp" else eval_visulogic(run_dir)
        report.update(route_stats(run_dir))
        summary[dataset][name] = report
        if dataset == "mmvp":
            print(
                f"{name}: sample {report['correct']}/{report['total']}={report['accuracy']:.4f}; "
                f"pair {report['pair_correct']}/{report['pair_total']}={report['pair_accuracy']:.4f}; "
                f"fmt_active={report.get('avg_format_active', 0):.1f}; "
                f"fmt_token={report.get('avg_format_tokens', 0):.1f}; "
                f"maxed={report.get('maxed', 0)}; failed={report['failed_extraction']}"
            )
        else:
            print(
                f"{name}: {report['correct']}/{report['total']}={report['accuracy']:.4f}; "
                f"fmt_active={report.get('avg_format_active', 0):.1f}; "
                f"fmt_token={report.get('avg_format_tokens', 0):.1f}; "
                f"maxed={report.get('maxed', 0)}; failed_real={report['failed_extraction_real']}"
            )
    print()

out = BASE / "format_overintervention_summary.json"
with out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"saved {out}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

for q in queue_gpu0_a queue_gpu0_b queue_gpu1_a queue_gpu1_b; do
  setsid bash "${BASE_DIR}/${q}.sh" > "${BASE_DIR}/${q}.log" 2>&1 < /dev/null &
  echo $! > "${BASE_DIR}/${q}.pid"
done

echo "BASE_DIR=${BASE_DIR}"
echo "queue_gpu0_a PID=$(cat "${BASE_DIR}/queue_gpu0_a.pid"): mmvp entropy10 -> visulogic top080_margin040"
echo "queue_gpu0_b PID=$(cat "${BASE_DIR}/queue_gpu0_b.pid"): mmvp top080_margin040"
echo "queue_gpu1_a PID=$(cat "${BASE_DIR}/queue_gpu1_a.pid"): mmvp strict -> visulogic strict"
echo "queue_gpu1_b PID=$(cat "${BASE_DIR}/queue_gpu1_b.pid"): visulogic entropy10"
echo "Compare after all queues finish:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
