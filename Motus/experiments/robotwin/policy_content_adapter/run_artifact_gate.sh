#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/motus/bin/python}"
OUT="${OUT:-${ROOT}/outputs/policy_content_adapter/motus_v1}"
BASE="${BASE:-${ROOT}/pretrained_models/Motus_robotwin2}"
WAN="${WAN:-/mnt/cpfs-E/baoshifeng/Motus/pretrained_models/Wan2.2-TI2V-5B}"
VLM="${VLM:-/mnt/cpfs-E/baoshifeng/Motus/pretrained_models/Qwen3-VL-2B-Instruct}"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"
CACHE_ROOT="${CACHE_ROOT:-/root/motus_policy_artifacts/motus_v1}"
AUDIT_ROOT="${AUDIT_ROOT:-${CACHE_ROOT}/audits}"

LINEAGE="${AUDIT_ROOT}/motus_robotwin2_lineage.json"
IMPLEMENTATION="${AUDIT_ROOT}/implementation_audit.json"
STRICT="${AUDIT_ROOT}/strict_load_audit.json"
TEXT_CACHE="${CACHE_ROOT}/task_text_cache"
ZERO_GATE="${AUDIT_ROOT}/zero_gate_audit.json"
PAIRED="${OUT}/paired_observation_manifest.json"
OFFICIAL="${OUT}/official_three_task_manifest_v4.json"
TOKEN_CACHE="${CACHE_ROOT}/layer16_token_cache"

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/motus-artifact-gate-pycache}"
export DS_IGNORE_CUDA_DETECTION=1
mkdir -p "${AUDIT_ROOT}/logs"

while pgrep -f '/root/anaconda3/bin/hf download motus-robotics/Motus_robotwin2' >/dev/null 2>&1 \
  || pgrep -f 'curl .*Motus_robotwin2/resolve/main/mp_rank_00_model_states.pt' >/dev/null 2>&1; do
  size="$(find "${BASE}/.cache/huggingface/download" -name '*.incomplete' -printf '%s\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  echo "[$(date -u +%FT%TZ)] WAIT checkpoint download incomplete_bytes=${size:-0}"
  sleep "${HEARTBEAT_SECONDS}"
done

if [[ ! -f "${BASE}/mp_rank_00_model_states.pt" ]]; then
  echo "checkpoint download exited without ${BASE}/mp_rank_00_model_states.pt" >&2
  exit 1
fi

while pgrep -f 'pv2_actiondit_full5ep_c1_matched_v1.*train' >/dev/null 2>&1; do
  echo "[$(date -u +%FT%TZ)] WAIT FastWAM matched C1 still owns GPUs"
  sleep "${HEARTBEAT_SECONDS}"
done

while true; do
  free_mib="$(nvidia-smi --id "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  echo "[$(date -u +%FT%TZ)] WAIT GPU${GPU_ID} free_mib=${free_mib:-unknown} threshold=${MIN_FREE_MIB}"
  sleep "${HEARTBEAT_SECONDS}"
done

if [[ ! -f "${LINEAGE}" ]]; then
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.lineage build \
    --repo-root "${ROOT}" --checkpoint-dir "${BASE}" \
    --wan-dir "${WAN}" --vlm-dir "${VLM}" --output "${LINEAGE}" \
    > "${AUDIT_ROOT}/logs/lineage.log" 2>&1
fi

if [[ ! -f "${IMPLEMENTATION}" ]]; then
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.source_audit build \
    --repo-root "${ROOT}" --output "${IMPLEMENTATION}" \
    > "${AUDIT_ROOT}/logs/implementation_audit.log" 2>&1
fi

if [[ ! -f "${STRICT}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.strict_load \
    --lineage "${LINEAGE}" --output "${STRICT}" --local-cuda-index 0 \
    > "${AUDIT_ROOT}/logs/strict_load.log" 2>&1
fi

if [[ ! -d "${TEXT_CACHE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.task_text_cache build \
    --output "${TEXT_CACHE}" --motus-repo-root "${ROOT}" \
    --wan-dir "${WAN}" --device cuda:0 \
    > "${AUDIT_ROOT}/logs/task_text_cache.log" 2>&1
fi

if [[ ! -f "${ZERO_GATE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.zero_gate_audit \
    --lineage "${LINEAGE}" --official-manifest "${OFFICIAL}" \
    --task-text-cache "${TEXT_CACHE}" --output "${ZERO_GATE}" \
    --local-cuda-index 0 \
    > "${AUDIT_ROOT}/logs/zero_gate.log" 2>&1
fi

if [[ ! -d "${TOKEN_CACHE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.extract_token_cache \
    --lineage "${LINEAGE}" --strict-load-audit "${STRICT}" \
    --paired-manifest "${PAIRED}" --task-text-cache "${TEXT_CACHE}" \
    --output "${TOKEN_CACHE}" --local-cuda-index 0 \
    --groups-per-batch 1 --capture-layer 16 --heartbeat-groups 20 \
    > "${AUDIT_ROOT}/logs/layer16_cache.log" 2>&1
fi

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.lineage validate \
  --manifest "${LINEAGE}" > "${AUDIT_ROOT}/logs/final_lineage_validate.log" 2>&1
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.task_text_cache validate \
  --cache-dir "${TEXT_CACHE}" > "${AUDIT_ROOT}/logs/final_text_validate.log" 2>&1
"${PYTHON_BIN}" - "${TOKEN_CACHE}" > "${AUDIT_ROOT}/logs/final_token_validate.log" 2>&1 <<'PY'
import json, sys
from experiments.robotwin.policy_content_adapter.token_cache import validate_token_cache
print(json.dumps(validate_token_cache(sys.argv[1], verify_shards=True), sort_keys=True))
PY

echo "[$(date -u +%FT%TZ)] PASS Motus artifact gate"
