#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar_damage_bestcombo_to_early_bias005_20260523.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/vstar_damage_image_pad_lambda_sweep"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --trace_topk 20"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

if [[ ! -f "${DATASET}" ]]; then
  echo "Missing damage dataset: ${DATASET}" >&2
  exit 1
fi

write_no_bias() {
  local run_name="no_bias_bestcombo_gpu0"
  local run_dir="${BASE_DIR}/${run_name}"
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" main.py \\
  --model_name "${MODEL}" \\
  --dataset "${DATASET}" \\
  --output_dir "${run_dir}" \\
  --method pure_soft \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${BEST_COMBO_ARGS}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_bias_run() {
  local phase="$1"
  local lambda="$2"
  local gpu="$3"
  local min_step="$4"
  local max_step="$5"
  local lambda_tag="${lambda/./}"
  local run_name="${phase}_lambda${lambda_tag}_gpu${gpu}"
  local run_dir="${BASE_DIR}/${run_name}"
  mkdir -p "${run_dir}"

  local phase_args=""
  if [[ "${min_step}" != "none" ]]; then
    phase_args="${phase_args} --image_pad_bias_min_step ${min_step}"
  fi
  if [[ "${max_step}" != "none" ]]; then
    phase_args="${phase_args} --image_pad_bias_max_step ${max_step}"
  fi

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
  --method pure_soft \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${BEST_COMBO_ARGS} \\
  --pure_soft_image_pad_bias \\
  --image_pad_bias_lambda "${lambda}" \\
  ${phase_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_no_bias

for lambda in 0.01 0.02 0.03 0.05; do
  write_bias_run "full" "${lambda}" 0 "none" "none"
  write_bias_run "early" "${lambda}" 0 0 128
  write_bias_run "mid" "${lambda}" 1 129 512
  write_bias_run "late" "${lambda}" 1 513 "none"
done

cat > "${BASE_DIR}/queue_gpu0_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/no_bias_bestcombo_gpu0/run_command.sh"
bash "${BASE_DIR}/full_lambda001_gpu0/run_command.sh"
bash "${BASE_DIR}/full_lambda002_gpu0/run_command.sh"
bash "${BASE_DIR}/full_lambda003_gpu0/run_command.sh"
bash "${BASE_DIR}/full_lambda005_gpu0/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu0_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/early_lambda001_gpu0/run_command.sh"
bash "${BASE_DIR}/early_lambda002_gpu0/run_command.sh"
bash "${BASE_DIR}/early_lambda003_gpu0/run_command.sh"
bash "${BASE_DIR}/early_lambda005_gpu0/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu1_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/mid_lambda001_gpu1/run_command.sh"
bash "${BASE_DIR}/mid_lambda002_gpu1/run_command.sh"
bash "${BASE_DIR}/mid_lambda003_gpu1/run_command.sh"
bash "${BASE_DIR}/mid_lambda005_gpu1/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu1_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/late_lambda001_gpu1/run_command.sh"
bash "${BASE_DIR}/late_lambda002_gpu1/run_command.sh"
bash "${BASE_DIR}/late_lambda003_gpu1/run_command.sh"
bash "${BASE_DIR}/late_lambda005_gpu1/run_command.sh"
EOF
chmod +x "${BASE_DIR}"/queue_gpu*.sh

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
import re
from pathlib import Path

BASE = Path("${BASE_DIR}")
DATASET = Path("${DATASET}")

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_mcq(text):
    if not text:
        return None
    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*[:\s]+\(?([A-Da-d])\)?",
        r"\\boxed\{([A-Da-d])\}",
        r"\*\*([A-Da-d])\*\*",
        r"(?:^|\n)\s*([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    last_letters = re.findall(r"\b([A-D])\b", text[-200:])
    return last_letters[-1].upper() if last_letters else None

def eval_run(run_dir):
    rows = load_jsonl(run_dir / "results.jsonl")
    total = correct = failed = maxed = 0
    tokens = []
    correct_ids = []
    wrong_ids = []
    for row in rows:
        total += 1
        pred = extract_mcq(row.get("model_answer") or "")
        gold = (row.get("answer") or "").strip().upper()
        ok = pred is not None and pred == gold
        correct += int(ok)
        failed += int(pred is None)
        out_tokens = int(row.get("output_tokens") or 0)
        tokens.append(out_tokens)
        maxed += int(out_tokens >= 1024)
        (correct_ids if ok else wrong_ids).append(int(row["id"]))
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "failed_extraction": failed,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "maxed": maxed,
        "correct_ids": correct_ids,
        "wrong_ids": wrong_ids,
    }

runs = {"no_bias": BASE / "no_bias_bestcombo_gpu0"}
for phase in ["full", "early", "mid", "late"]:
    for tag in ["001", "002", "003", "005"]:
        gpu = 0 if phase in {"full", "early"} else 1
        runs[f"{phase}_lambda{tag}"] = BASE / f"{phase}_lambda{tag}_gpu{gpu}"

summary = {}
print(f"damage dataset: {DATASET}")
print(f"total samples: {len(load_jsonl(DATASET))}")
for name, run_dir in runs.items():
    if not (run_dir / "results.jsonl").exists():
        print(f"{name:16s} MISSING {run_dir}")
        continue
    report = eval_run(run_dir)
    summary[name] = report
    print(
        f"{name:16s} {report['correct']:2d}/{report['total']}="
        f"{report['accuracy']*100:5.2f}% "
        f"avg_tok={report['avg_tokens']:6.1f} maxed={report['maxed']} "
        f"failed={report['failed_extraction']}"
    )

out = BASE / "damage_lambda_sweep_summary.json"
with out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"saved {out}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

setsid bash "${BASE_DIR}/queue_gpu0_a.sh" > "${BASE_DIR}/queue_gpu0_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_a.pid"
setsid bash "${BASE_DIR}/queue_gpu0_b.sh" > "${BASE_DIR}/queue_gpu0_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_b.pid"
setsid bash "${BASE_DIR}/queue_gpu1_a.sh" > "${BASE_DIR}/queue_gpu1_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_a.pid"
setsid bash "${BASE_DIR}/queue_gpu1_b.sh" > "${BASE_DIR}/queue_gpu1_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_b.pid"

echo "BASE_DIR=${BASE_DIR}"
echo "DATASET=${DATASET}"
echo "queue_gpu0_a PID=$(cat "${BASE_DIR}/queue_gpu0_a.pid"): no_bias + full lambdas"
echo "queue_gpu0_b PID=$(cat "${BASE_DIR}/queue_gpu0_b.pid"): early lambdas"
echo "queue_gpu1_a PID=$(cat "${BASE_DIR}/queue_gpu1_a.pid"): mid lambdas"
echo "queue_gpu1_b PID=$(cat "${BASE_DIR}/queue_gpu1_b.pid"): late lambdas"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
