#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar_exp1_wrong_union.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_collapse_wrong_union_parallel"

mkdir -p "${BASE_DIR}"

write_run() {
  local name="$1"
  local gpu="$2"
  local extra_args="$3"
  local run_dir="${BASE_DIR}/${name}_gpu${gpu}"
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
  --max_new_tokens 1024 \\
  --temperature 0.6 \\
  --top_p 0.95 \\
  --top_k 20 \\
  --seed 42 \\
  --device cuda \\
  --no-do_sample \\
  --save_token_entropy \\
  --save_full_token_entropy \\
  --trace_topk 20 \\
  ${extra_args}
EOF
  chmod +x "${run_dir}/run_command.sh"
}

write_run "pure_soft_baseline" 0 ""
write_run "pure_soft_collapse_diffuse" 1 "--pure_soft_collapse_on_diffuse --collapse_entropy_window 16 --collapse_entropy_alpha 2.0 --collapse_min_history 4 --collapse_min_entropy 1.0 --collapse_low_conf_tau 0.20 --collapse_low_margin_tau 0.05"

for run_dir in "${BASE_DIR}"/pure_soft_baseline_gpu0 "${BASE_DIR}"/pure_soft_collapse_diffuse_gpu1; do
  setsid bash "${run_dir}/run_command.sh" > "${run_dir}/nohup.log" 2>&1 < /dev/null &
  echo $! > "${run_dir}/pid.txt"
done

cat > "${BASE_DIR}/compare_after_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
from lead.evaluator import evaluate_single

base = Path("${BASE_DIR}")
for name in ["pure_soft_baseline_gpu0", "pure_soft_collapse_diffuse_gpu1"]:
    run = base / name
    rows = []
    with (run / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ok, ex = evaluate_single(r.get("model_answer"), r.get("answer", ""))
            rows.append((r.get("id"), ok, ex, r.get("output_tokens") or 0))
    traces = {}
    collapse_count = 0
    with (run / "token_entropy_full.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = r.get("tokens") or []
            collapse_count += sum(1 for t in toks if t.get("collapse_on_diffuse"))
            traces[r.get("id")] = len(toks)
    correct = sum(ok for _, ok, _, _ in rows)
    lengths = sorted(v for *_, v in rows)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(rows)} = {correct/len(rows):.4f}")
    print(f"output length mean={sum(lengths)/len(lengths):.2f} median={lengths[len(lengths)//2]} p90={lengths[int(0.9*(len(lengths)-1))]} max={lengths[-1]}")
    print(f"collapse_count: {collapse_count}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

echo "Started pure-soft collapse wrong-union experiment:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  baseline PID=$(cat "${BASE_DIR}/pure_soft_baseline_gpu0/pid.txt") GPU=0"
echo "  collapse PID=$(cat "${BASE_DIR}/pure_soft_collapse_diffuse_gpu1/pid.txt") GPU=1"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
