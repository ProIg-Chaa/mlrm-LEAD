#!/usr/bin/env bash
set -euo pipefail

# Download model weights and datasets used by the local mlrm-LEAD experiments.
# This script is intentionally non-destructive: it creates directories and
# downloads missing/cached files, but never removes existing assets.

PROJECT_DIR="/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD"
ASSET_ROOT="/share/home/wangzixu/liudinghao/gushuo"
DATASET_ROOT="$ASSET_ROOT/datasets"
SOURCE_ROOT="$DATASET_ROOT/sources"
MLRM_DATA_ROOT="$DATASET_ROOT/mlrm-LEAD"
MODEL_ROOT="$ASSET_ROOT/models"

ENV_NAME="${ENV_NAME:-mlrm-lead}"
HF_BIN="${HF_BIN:-/share/home/wangzixu/.local/share/mamba/envs/${ENV_NAME}/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/wangzixu/.local/share/mamba/envs/${ENV_NAME}/bin/python}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:17991}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$MODEL_ROOT" "$SOURCE_ROOT" "$MLRM_DATA_ROOT" "$PROJECT_DIR/data"

echo "Project: $PROJECT_DIR"
echo "Dataset root: $DATASET_ROOT"
echo "Model root: $MODEL_ROOT"
echo "Proxy: $PROXY_URL"
echo

download_hf_repo() {
  local repo_id="$1"
  local repo_type="$2"
  local local_dir="$3"

  mkdir -p "$local_dir"
  echo "==> Downloading ${repo_type}: ${repo_id}"
  echo "    -> ${local_dir}"
  "$HF_BIN" download "$repo_id" \
    --repo-type "$repo_type" \
    --local-dir "$local_dir" \
    --max-workers "$HF_MAX_WORKERS"
  echo
}

download_mmhal_with_curl() {
  local repo_id="Shengcao1006/MMHal-Bench"
  local local_dir="$SOURCE_ROOT/Shengcao1006__MMHal-Bench"

  mkdir -p "$local_dir"
  echo "==> Downloading dataset with curl fallback: ${repo_id}"
  echo "    -> ${local_dir}"

  "$PYTHON_BIN" - <<'PY'
import os
import subprocess
from pathlib import Path
from urllib.parse import quote
from huggingface_hub import HfApi

repo = "Shengcao1006/MMHal-Bench"
local_dir = Path("/share/home/wangzixu/liudinghao/gushuo/datasets/sources/Shengcao1006__MMHal-Bench")
local_dir.mkdir(parents=True, exist_ok=True)

files = HfApi().list_repo_files(repo, repo_type="dataset")
base = "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/"
env = os.environ.copy()

for index, name in enumerate(files, 1):
    target = local_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[{index}/{len(files)}] skip {name}")
        continue

    tmp = Path(str(target) + ".part")
    url = base + quote(name)
    print(f"[{index}/{len(files)}] download {name}")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "600",
            "-o",
            str(tmp),
            url,
        ],
        check=True,
        env=env,
    )
    tmp.replace(target)

local_files = {
    str(p.relative_to(local_dir))
    for p in local_dir.rglob("*")
    if p.is_file()
    and ".cache/" not in str(p.relative_to(local_dir))
    and not str(p).endswith(".part")
}
missing = sorted(set(files) - local_files)
print(f"repo_files={len(files)} local_repo_files={len(local_files)} missing={len(missing)}")
if missing:
    for name in missing:
        print(f"MISSING {name}")
    raise SystemExit(1)
PY
  echo
}

echo "==> Model weights"
download_hf_repo \
  "Fancy-MLLM/R1-Onevision-7B-RL" \
  "model" \
  "$MODEL_ROOT/R1-Onevision-7B-RL"

echo "==> Dataset source repositories"
download_hf_repo "PrismaX/PhysUniBench" "dataset" "$SOURCE_ROOT/PrismaX__PhysUniBench"
download_hf_repo "xai-org/RealworldQA" "dataset" "$SOURCE_ROOT/xai-org__RealworldQA"
download_hf_repo "AI4Math/MathVista" "dataset" "$SOURCE_ROOT/AI4Math__MathVista"
download_hf_repo "MMVP/MMVP" "dataset" "$SOURCE_ROOT/MMVP__MMVP"
download_hf_repo "MathLLMs/MathVision" "dataset" "$SOURCE_ROOT/MathLLMs__MathVision"
download_hf_repo "VisuLogic/VisuLogic" "dataset" "$SOURCE_ROOT/VisuLogic__VisuLogic"
download_hf_repo "craigwu/vstar_bench" "dataset" "$SOURCE_ROOT/craigwu__vstar_bench"

# MMHal-Bench is stored through Hugging Face's LFS/Xet path. On this cluster,
# the standard snapshot downloader can stall, so use a plain curl fallback.
download_mmhal_with_curl

echo "==> Sync prepared JSONL files into project data/"
if [[ -d "$MLRM_DATA_ROOT/jsonl" ]]; then
  find "$MLRM_DATA_ROOT/jsonl" -maxdepth 1 -type f -name "*.jsonl" -print0 |
    while IFS= read -r -d '' file; do
      target="$PROJECT_DIR/data/$(basename "$file")"
      if [[ -f "$target" ]]; then
        echo "skip $(basename "$target")"
      else
        cp "$file" "$target"
        echo "copied $(basename "$target")"
      fi
    done
else
  echo "warning: prepared JSONL directory not found: $MLRM_DATA_ROOT/jsonl"
fi

echo
echo "==> Prepare MMHal-Bench JSONL"
cd "$PROJECT_DIR"
python3 script/prepare_mmhal_bench_jsonl.py

echo
echo "==> Final checks"
echo "Model dir:"
du -sh "$MODEL_ROOT/R1-Onevision-7B-RL" || true
echo
echo "Dataset sources:"
du -sh "$SOURCE_ROOT"/* 2>/dev/null || true
echo
echo "Project data files:"
find "$PROJECT_DIR/data" -maxdepth 1 -type f -name "*.jsonl" -printf "%f\n" | sort

echo
echo "Done."
