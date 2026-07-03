#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/home/wangzixu/liudinghao/gushuo/proj/mlrm-LEAD}"
SOURCE_ROOT="${SOURCE_ROOT:-/share/home/wangzixu/liudinghao/gushuo/datasets/sources}"
HF_BIN="${HF_BIN:-/share/home/wangzixu/.local/share/mamba/envs/mlrm-lead/bin/hf}"
DOWNLOAD_TIER="${DOWNLOAD_TIER:-tier1}"
DRY_RUN="${DRY_RUN:-0}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_XET
export HF_HUB_DOWNLOAD_TIMEOUT
export HF_HUB_ETAG_TIMEOUT

if [[ -n "${PROXY_URL:-}" ]]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

download_hf_dataset() {
  local repo_id="$1"
  local local_dir="$2"
  run_cmd mkdir -p "${local_dir}"
  run_cmd "${HF_BIN}" download "${repo_id}" --repo-type dataset --local-dir "${local_dir}" --max-workers "${HF_MAX_WORKERS}"
}

clone_repo() {
  local repo_url="$1"
  local local_dir="$2"
  if [[ -d "${local_dir}/.git" ]]; then
    echo "[skip] ${repo_url} already cloned at ${local_dir}"
    return 0
  fi
  run_cmd mkdir -p "$(dirname "${local_dir}")"
  run_cmd git clone --depth 1 "${repo_url}" "${local_dir}"
}

echo "[info] project=${PROJECT_DIR}"
echo "[info] source_root=${SOURCE_ROOT}"
echo "[info] tier=${DOWNLOAD_TIER}"
echo "[info] proxy=${PROXY_URL:-<environment/default>}"
echo "[info] hf_max_workers=${HF_MAX_WORKERS}"
echo "[info] hf_download_timeout=${HF_HUB_DOWNLOAD_TIMEOUT}"

case "${DOWNLOAD_TIER}" in
  tier1|all)
    download_hf_dataset "suyc21/VMCBench" "${SOURCE_ROOT}/suyc21__VMCBench"
    download_hf_dataset "lmms-lab/POPE" "${SOURCE_ROOT}/lmms-lab__POPE"
    download_hf_dataset "FanqingM/MMK12" "${SOURCE_ROOT}/FanqingM__MMK12"
    download_hf_dataset "MathLLMs/MathVision" "${SOURCE_ROOT}/MathLLMs__MathVision"
    ;;&
  tier2|all)
    clone_repo "https://github.com/chenllliang/MMEvalPro.git" "${SOURCE_ROOT}/chenllliang__MMEvalPro"
    clone_repo "https://github.com/ZrrSkywalker/MathVerse.git" "${SOURCE_ROOT}/ZrrSkywalker__MathVerse"
    clone_repo "https://github.com/gzcch/Bingo.git" "${SOURCE_ROOT}/gzcch__Bingo"
    echo "[note] Geometry3K needs a separate format audit before download; no single stable VLM JSONL source is assumed here."
    ;;
esac

echo "[done] download step finished; run script/exp7_02/audit_origin_lead_new_datasets.py next."
