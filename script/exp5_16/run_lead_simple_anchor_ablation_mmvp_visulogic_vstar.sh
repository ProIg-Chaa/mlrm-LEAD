#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
REF_BASE="${ROOT}/output/experiments/20260520_231938/cross_dataset_base_lead_bestcombo"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/lead_simple_anchor_ablation_mmvp_visulogic_vstar"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
LEAD_ARGS="--alpha 0.4 --max_switch_count 5 --window_size 128"

write_run_command() {
  local run_name="$1"
  local gpu="$2"
  local dataset="$3"
  local limit_arg="$4"
  local extra_args="$5"
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
  --method lead \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${LEAD_ARGS} \\
  ${limit_arg} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

# Main ablation: same generate_lead path, only remove the original <|image_pad|> simple anchor.
write_run_command "mmvp_lead_no_simple_anchor_gpu0" 0 "data/mmvp.jsonl" "" "--lead_disable_simple_visual_anchor"
write_run_command "visulogic300_lead_no_simple_anchor_gpu1" 1 "data/visulogic.jsonl" "--limit 300" "--lead_disable_simple_visual_anchor"
write_run_command "vstar_lead_no_simple_anchor_gpu0" 0 "data/vstar.jsonl" "" "--lead_disable_simple_visual_anchor"

cat > "${BASE_DIR}/queue_gpu0.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_lead_no_simple_anchor_gpu0/run_command.sh"
bash "${BASE_DIR}/vstar_lead_no_simple_anchor_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/visulogic300_lead_no_simple_anchor_gpu1/run_command.sh"
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
    p = run_dir / "token_entropy.jsonl"
    if not p.exists():
        return {}
    total_tokens = 0
    maxed = 0
    soft_tokens = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n = row.get("output_tokens") or 0
            total_tokens += n
            maxed += int(n >= 1024)
            summary = row.get("entropy_summary") or {}
            soft_tokens += summary.get("soft_token_count") or 0
    return {
        "avg_tokens": total_tokens / 300,
        "maxed": maxed,
        "avg_soft_tokens": soft_tokens / 300,
    }

items = {
    "mmvp": {
        "eval": eval_mmvp,
        "runs": {
            "lead_ref_simple_anchor": REF_BASE / "mmvp_lead_gpu1",
            "lead_no_simple_anchor": BASE / "mmvp_lead_no_simple_anchor_gpu0",
        },
    },
    "visulogic300": {
        "eval": eval_mcq,
        "runs": {
            "lead_ref_simple_anchor": REF_BASE / "visulogic300_lead_gpu0",
            "lead_no_simple_anchor": BASE / "visulogic300_lead_no_simple_anchor_gpu1",
        },
    },
    "vstar": {
        "eval": eval_mcq,
        "runs": {
            "lead_no_simple_anchor": BASE / "vstar_lead_no_simple_anchor_gpu0",
        },
    },
}

summary = {}
for dataset, cfg in items.items():
    print(f"== {dataset} ==")
    summary[dataset] = {}
    for name, run_dir in cfg["runs"].items():
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
                f"avg_tokens={report.get('avg_tokens', 0):.1f}; maxed={report.get('maxed', 0)}"
            )
        else:
            print(
                f"{name}: {report['correct']}/{report['total']}={report['accuracy']:.4f}; "
                f"avg_tokens={report.get('avg_tokens', 0):.1f}; maxed={report.get('maxed', 0)}; "
                f"failed_real={report.get('failed_extraction_real', 0)}"
            )
    print()

out = BASE / "lead_simple_anchor_ablation_summary.json"
with out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"saved {out}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

setsid bash "${BASE_DIR}/queue_gpu0.sh" > "${BASE_DIR}/queue_gpu0.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0.pid"
setsid bash "${BASE_DIR}/queue_gpu1.sh" > "${BASE_DIR}/queue_gpu1.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1.pid"

echo "BASE_DIR=${BASE_DIR}"
echo "queue_gpu0 PID=$(cat "${BASE_DIR}/queue_gpu0.pid"): mmvp lead_no_simple_anchor -> vstar lead_no_simple_anchor"
echo "queue_gpu1 PID=$(cat "${BASE_DIR}/queue_gpu1.pid"): visulogic300 lead_no_simple_anchor"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
