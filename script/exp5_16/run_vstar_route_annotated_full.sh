#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="${ROOT}/data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/vstar_route_annotated_full"

mkdir -p "${BASE_DIR}"

COMMON_ARGS="--max_new_tokens 1024 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 42 --device cuda --no-do_sample --save_token_entropy --save_full_token_entropy --trace_topk 20"
BEST_COMBO_ARGS="--pure_soft_format_cooldown --format_cooldown_steps 2 --pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05 --collapse_min_step 64 --collapse_require_repeat_degen --collapse_repeat_ngram 3 --collapse_recent_repeat_window 32 --collapse_recent_repeat_tau 0.35"

write_run() {
  local run_name="$1"
  local gpu="$2"
  local extra_args="$3"
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
  --dataset "${DATASET}" \\
  --output_dir "${run_dir}" \\
  --method pure_soft \\
  --cot_prompt_mode orign \\
  ${COMMON_ARGS} \\
  ${BEST_COMBO_ARGS} \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "bestcombo_route_annotated_gpu0" 0 ""
write_run "router_v0_midbias002_gpu0" 0 "--pure_soft_image_pad_bias --image_pad_bias_lambda 0.02 --image_pad_bias_min_step 129 --image_pad_bias_max_step 512"
write_run "router_v0_midbias003_gpu1" 1 "--pure_soft_image_pad_bias --image_pad_bias_lambda 0.03 --image_pad_bias_min_step 129 --image_pad_bias_max_step 512"
write_run "router_v0_midbias002_answerzone_gpu1" 1 "--pure_soft_image_pad_bias --image_pad_bias_lambda 0.02 --image_pad_bias_min_step 129 --image_pad_bias_max_step 512 --pure_soft_answer_zone_discrete"

cat > "${BASE_DIR}/queue_gpu0_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/bestcombo_route_annotated_gpu0/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu0_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/router_v0_midbias002_gpu0/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu1_a.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/router_v0_midbias003_gpu1/run_command.sh"
EOF

cat > "${BASE_DIR}/queue_gpu1_b.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash "${BASE_DIR}/router_v0_midbias002_answerzone_gpu1/run_command.sh"
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
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    last = re.findall(r"\b([A-D])\b", text[-200:])
    return last[-1].upper() if last else None

def eval_run(run_dir):
    rows = load_jsonl(run_dir / "results.jsonl")
    total = correct = failed = maxed = 0
    direct = [0, 0]
    relative = [0, 0]
    lengths = []
    for row in rows:
        total += 1
        pred = extract_mcq(row.get("model_answer") or "")
        gold = (row.get("answer") or "").strip().upper()
        ok = pred is not None and pred == gold
        correct += int(ok)
        failed += int(pred is None)
        out_tokens = int(row.get("output_tokens") or 0)
        lengths.append(out_tokens)
        maxed += int(out_tokens >= 1024)
        sub = row.get("subtopic")
        if sub == "direct_attributes":
            direct[1] += 1
            direct[0] += int(ok)
        elif sub == "relative_position":
            relative[1] += 1
            relative[0] += int(ok)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "direct": direct,
        "relative": relative,
        "failed_extraction": failed,
        "avg_tokens": sum(lengths) / len(lengths) if lengths else 0,
        "maxed": maxed,
    }

runs = [
    "bestcombo_route_annotated_gpu0",
    "router_v0_midbias002_gpu0",
    "router_v0_midbias003_gpu1",
    "router_v0_midbias002_answerzone_gpu1",
]
summary = {}
for name in runs:
    run_dir = BASE / name
    if not (run_dir / "results.jsonl").exists():
        print(f"{name}: MISSING")
        continue
    rep = eval_run(run_dir)
    summary[name] = rep
    d = rep["direct"]
    r = rep["relative"]
    print(
        f"{name}: {rep['correct']}/{rep['total']}={rep['accuracy']*100:.2f}% "
        f"direct={d[0]}/{d[1]} relative={r[0]}/{r[1]} "
        f"avg_tok={rep['avg_tokens']:.1f} maxed={rep['maxed']} failed={rep['failed_extraction']}"
    )

out = BASE / "vstar_route_annotated_full_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"saved {out}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

cat > "${BASE_DIR}/route_summary_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
BASE="${BASE_DIR}"
BASELINE="\${BASE}/bestcombo_route_annotated_gpu0"
for run in bestcombo_route_annotated_gpu0 router_v0_midbias002_gpu0 router_v0_midbias003_gpu1 router_v0_midbias002_answerzone_gpu1; do
  if [[ ! -f "\${BASE}/\${run}/token_entropy_full.jsonl" ]]; then
    echo "missing trace: \${run}" >&2
    continue
  fi
  extra=()
  if [[ "\${run}" != "bestcombo_route_annotated_gpu0" ]]; then
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

setsid bash "${BASE_DIR}/queue_gpu0_a.sh" > "${BASE_DIR}/queue_gpu0_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_a.pid"
setsid bash "${BASE_DIR}/queue_gpu0_b.sh" > "${BASE_DIR}/queue_gpu0_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu0_b.pid"
setsid bash "${BASE_DIR}/queue_gpu1_a.sh" > "${BASE_DIR}/queue_gpu1_a.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_a.pid"
setsid bash "${BASE_DIR}/queue_gpu1_b.sh" > "${BASE_DIR}/queue_gpu1_b.log" 2>&1 < /dev/null &
echo $! > "${BASE_DIR}/queue_gpu1_b.pid"

echo "BASE_DIR=${BASE_DIR}"
echo "queue_gpu0_a PID=$(cat "${BASE_DIR}/queue_gpu0_a.pid"): bestcombo route annotated"
echo "queue_gpu0_b PID=$(cat "${BASE_DIR}/queue_gpu0_b.pid"): router v0 mid bias lambda=0.02"
echo "queue_gpu1_a PID=$(cat "${BASE_DIR}/queue_gpu1_a.pid"): router v0 mid bias lambda=0.03"
echo "queue_gpu1_b PID=$(cat "${BASE_DIR}/queue_gpu1_b.pid"): router v0 mid bias lambda=0.02 + answer-zone"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
echo "Route summaries after done:"
echo "  bash ${BASE_DIR}/route_summary_after_done.sh"
