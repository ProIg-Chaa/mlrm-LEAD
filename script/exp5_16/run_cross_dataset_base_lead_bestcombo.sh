#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/cross_dataset_base_lead_bestcombo"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
LEAD_ARGS="--alpha 0.4 --max_switch_count 5 --window_size 128"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

write_run_command() {
  local run_name="$1"
  local gpu="$2"
  local dataset="$3"
  local method="$4"
  local limit_arg="$5"
  local extra_args="$6"
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
  --method "${method}" \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${limit_arg} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run_command "phys300_pure_soft_gpu0" 0 "data/physunibench_uniform300.jsonl" "pure_soft" "" ""
write_run_command "phys300_bestcombo_gpu0" 0 "data/physunibench_uniform300.jsonl" "pure_soft" "" "${BEST_COMBO_ARGS}"
write_run_command "phys300_lead_gpu0" 0 "data/physunibench_uniform300.jsonl" "lead" "" "${LEAD_ARGS}"

write_run_command "mmvp_pure_soft_gpu1" 1 "data/mmvp.jsonl" "pure_soft" "" ""
write_run_command "mmvp_bestcombo_gpu1" 1 "data/mmvp.jsonl" "pure_soft" "" "${BEST_COMBO_ARGS}"
write_run_command "mmvp_lead_gpu1" 1 "data/mmvp.jsonl" "lead" "" "${LEAD_ARGS}"

write_run_command "visulogic300_pure_soft_gpu0" 0 "data/visulogic.jsonl" "pure_soft" "--limit 300" ""
write_run_command "visulogic300_bestcombo_gpu1" 1 "data/visulogic.jsonl" "pure_soft" "--limit 300" "${BEST_COMBO_ARGS}"
write_run_command "visulogic300_lead_gpu0" 0 "data/visulogic.jsonl" "lead" "--limit 300" "${LEAD_ARGS}"

cat > "${BASE_DIR}/queue_gpu0_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/phys300_pure_soft_gpu0/run_command.sh"
bash "${BASE_DIR}/phys300_bestcombo_gpu0/run_command.sh"
bash "${BASE_DIR}/visulogic300_pure_soft_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu0_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/phys300_lead_gpu0/run_command.sh"
bash "${BASE_DIR}/visulogic300_lead_gpu0/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_pure_soft_gpu1/run_command.sh"
bash "${BASE_DIR}/mmvp_bestcombo_gpu1/run_command.sh"
EOF
cat > "${BASE_DIR}/queue_gpu1_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mmvp_lead_gpu1/run_command.sh"
bash "${BASE_DIR}/visulogic300_bestcombo_gpu1/run_command.sh"
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

from lead.evaluator import evaluate_dataset

ROOT = Path("${ROOT}")
BASE = Path("${BASE_DIR}")
sys.path.insert(0, str(ROOT / "script"))
from evaluate_specialized_results import load_jsonl, evaluate as specialized_evaluate

runs = {
    "phys300": {
        "dataset": ROOT / "data/physunibench_uniform300.jsonl",
        "mode": "physunibench",
        "items": {
            "pure_soft": BASE / "phys300_pure_soft_gpu0",
            "lead": BASE / "phys300_lead_gpu0",
            "bestcombo": BASE / "phys300_bestcombo_gpu0",
        },
    },
    "mmvp": {
        "dataset": ROOT / "data/mmvp.jsonl",
        "mode": "mmvp",
        "items": {
            "pure_soft": BASE / "mmvp_pure_soft_gpu1",
            "lead": BASE / "mmvp_lead_gpu1",
            "bestcombo": BASE / "mmvp_bestcombo_gpu1",
        },
    },
    "visulogic300": {
        "dataset": ROOT / "data/visulogic.jsonl",
        "mode": "mcq",
        "items": {
            "pure_soft": BASE / "visulogic300_pure_soft_gpu0",
            "lead": BASE / "visulogic300_lead_gpu0",
            "bestcombo": BASE / "visulogic300_bestcombo_gpu1",
        },
    },
}

summary = {}
for dataset_name, cfg in runs.items():
    print(f"== {dataset_name} ==")
    dataset_rows = load_jsonl(cfg["dataset"])
    if dataset_name == "visulogic300":
        dataset_rows = dataset_rows[:300]
    summary[dataset_name] = {}
    for method, run_dir in cfg["items"].items():
        result_path = run_dir / "results.jsonl"
        if not result_path.exists():
            print(f"{method}: MISSING {result_path}")
            continue
        result_rows = load_jsonl(result_path)
        if cfg["mode"] in {"physunibench", "mmvp"}:
            report, _ = specialized_evaluate(dataset_rows, result_rows, cfg["mode"])
        else:
            report = evaluate_dataset(result_rows)
        row = {
            "accuracy": report.get("accuracy", 0.0),
            "correct": report.get("correct", 0),
            "total": report.get("total", len(result_rows)),
            "failed_extraction": report.get("failed_extraction", 0),
        }
        if cfg["mode"] == "mmvp":
            row.update({
                "pair_accuracy": report.get("pair_accuracy", 0.0),
                "pair_correct": report.get("pair_correct", 0),
                "pair_total": report.get("pair_total", 0),
            })
            print(
                f"{method}: sample {row['correct']}/{row['total']} = {row['accuracy']:.4f}; "
                f"pair {row['pair_correct']}/{row['pair_total']} = {row['pair_accuracy']:.4f}; "
                f"failed={row['failed_extraction']}"
            )
        else:
            print(
                f"{method}: {row['correct']}/{row['total']} = {row['accuracy']:.4f}; "
                f"failed={row['failed_extraction']}"
            )
        summary[dataset_name][method] = row
    print()

out = BASE / "cross_dataset_summary.json"
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
echo "queue_gpu0_a PID=$(cat "${BASE_DIR}/queue_gpu0_a.pid"): phys pure_soft -> phys bestcombo -> visulogic pure_soft"
echo "queue_gpu0_b PID=$(cat "${BASE_DIR}/queue_gpu0_b.pid"): phys lead -> visulogic lead"
echo "queue_gpu1_a PID=$(cat "${BASE_DIR}/queue_gpu1_a.pid"): mmvp pure_soft -> mmvp bestcombo"
echo "queue_gpu1_b PID=$(cat "${BASE_DIR}/queue_gpu1_b.pid"): mmvp lead -> visulogic bestcombo"
echo "Compare after all queues finish:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
