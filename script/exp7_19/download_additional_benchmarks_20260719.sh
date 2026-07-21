#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/gushuo/datasets/sources
PYTHON=/root/autodl-tmp/gushuo/envs/mlrm-lead/bin/python
export http_proxy=http://127.0.0.1:8886
export https_proxy=http://127.0.0.1:8886
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

mkdir -p "$ROOT"

sync_repo() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" -c http.proxy="$http_proxy" pull --ff-only
  elif [[ -e "$dest" ]]; then
    echo "SKIP non-git destination: $dest"
  else
    git -c http.proxy="$http_proxy" clone --depth 1 "$url" "$dest"
  fi
}

sync_repo \
  https://github.com/chenllliang/MMEvalPro.git \
  "$ROOT/chenllliang__MMEvalPro"
sync_repo \
  https://github.com/ZrrSkywalker/MathVerse.git \
  "$ROOT/ZrrSkywalker__MathVerse"
sync_repo \
  https://github.com/gzcch/Bingo.git \
  "$ROOT/gzcch__Bingo"
sync_repo \
  https://github.com/lupantech/InterGPS.git \
  "$ROOT/lupantech__InterGPS"

"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

targets = [
    ("AI4Math/MathVerse", "/root/autodl-tmp/gushuo/datasets/sources/AI4Math__MathVerse"),
    ("AI4Math/MathVista", "/root/autodl-tmp/gushuo/datasets/sources/AI4Math__MathVista"),
    ("MathLLMs/MathVision", "/root/autodl-tmp/gushuo/datasets/sources/MathLLMs__MathVision"),
]
for repo_id, local_dir in targets:
    print(f"START {repo_id} -> {local_dir}", flush=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=1,
            resume_download=True,
        )
        print(f"DONE {repo_id}", flush=True)
    except Exception as exc:
        print(f"FAILED {repo_id}: {exc!r}", flush=True)
PY

date '+DOWNLOAD QUEUE FINISHED %F %T %Z'
