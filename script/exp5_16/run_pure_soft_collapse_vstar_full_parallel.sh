#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
PYTHON_BIN="/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/python"
MODEL="/share/home/wangzixu/liudinghao/gushuo/models/R1-Onevision-7B-RL"
DATASET="data/vstar.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="${ROOT}/output/experiments/${STAMP}/pure_soft_collapse_vstar_full_parallel"

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
import statistics as st
from pathlib import Path
from lead.evaluator import evaluate_single

base = Path("${BASE_DIR}")
runs = {
    "baseline": base / "pure_soft_baseline_gpu0",
    "collapse": base / "pure_soft_collapse_diffuse_gpu1",
}
rows = {}
for name, run in runs.items():
    cur = {}
    with (run / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ok, ex = evaluate_single(r.get("model_answer"), r.get("answer", ""))
            cur[r.get("id")] = {
                "ok": ok,
                "ex": ex,
                "len": r.get("output_tokens") or 0,
                "answer": r.get("model_answer"),
            }
    rows[name] = cur

ids = sorted(rows["baseline"])
for name in ["baseline", "collapse"]:
    vals = sorted(rows[name][i]["len"] for i in ids)
    correct = sum(rows[name][i]["ok"] for i in ids)
    print(f"== {name} ==")
    print(f"accuracy: {correct}/{len(ids)} = {correct/len(ids):.4f}")
    print(f"length mean={sum(vals)/len(vals):.2f} median={st.median(vals)} p90={vals[int(0.9*(len(vals)-1))]} max={vals[-1]}")
    print(f"long>=256: {sum(v >= 256 for v in vals)}")
    print(f"maxed1024: {sum(v >= 1024 for v in vals)}")

fixed = [i for i in ids if not rows["baseline"][i]["ok"] and rows["collapse"][i]["ok"]]
damaged = [i for i in ids if rows["baseline"][i]["ok"] and not rows["collapse"][i]["ok"]]
changed = [i for i in ids if rows["baseline"][i]["answer"] != rows["collapse"][i]["answer"]]
print("== delta ==")
print(f"changed_outputs: {len(changed)}/{len(ids)}")
print(f"fixed: {len(fixed)} {fixed[:50]}")
print(f"damaged: {len(damaged)} {damaged[:50]}")

collapse_count = 0
samples_with_collapse = 0
per_sample = []
with (runs["collapse"] / "token_entropy_full.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        count = sum(1 for t in (r.get("tokens") or []) if t.get("collapse_on_diffuse"))
        collapse_count += count
        samples_with_collapse += int(count > 0)
        per_sample.append((r.get("id"), count))
counts = sorted(c for _, c in per_sample)
print("== collapse ==")
print(f"collapse_count: {collapse_count}")
print(f"samples_with_collapse: {samples_with_collapse}/{len(per_sample)}")
print(f"mean={sum(counts)/len(counts):.2f} median={st.median(counts)} p90={counts[int(0.9*(len(counts)-1))]} max={counts[-1]}")
print(f"top_samples: {sorted(per_sample, key=lambda x: x[1], reverse=True)[:20]}")
PY
EOF
chmod +x "${BASE_DIR}/compare_after_done.sh"

echo "Started full VStar pure-soft collapse experiment:"
echo "  BASE_DIR=${BASE_DIR}"
echo "  baseline PID=$(cat "${BASE_DIR}/pure_soft_baseline_gpu0/pid.txt") GPU=0"
echo "  collapse PID=$(cat "${BASE_DIR}/pure_soft_collapse_diffuse_gpu1/pid.txt") GPU=1"
echo "Compare after done:"
echo "  bash ${BASE_DIR}/compare_after_done.sh"
