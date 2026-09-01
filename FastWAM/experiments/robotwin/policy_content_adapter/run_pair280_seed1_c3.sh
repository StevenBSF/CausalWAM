#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/torchrun}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/fastwam_policy_artifacts/pair280_layer16_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PHASE="${PHASE:-prepare}"
RESUME="${RESUME:-NO}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"
MIN_FREE_ARTIFACT_BYTES="${MIN_FREE_ARTIFACT_BYTES:-350000000000}"

case "${PHASE}" in
  prepare|smoke|train|audit) ;;
  *) echo "PHASE must be prepare, smoke, train, or audit" >&2; exit 2 ;;
esac
case "${RESUME}" in
  YES|NO) ;;
  *) echo "RESUME must be YES or NO" >&2; exit 2 ;;
esac
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 8 || "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight distinct GPUs" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${FASTWAM_ROOT}"

run_root="${ARTIFACT_ROOT}/seed1_c3_pair280_posttraining_v1"
status_file="${run_root}/pair280.status"
mkdir -p "${run_root}/logs" "${run_root}/audits"

prepare() {
  if [[ ! -f "${run_root}/materialization.json" ]]; then
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
      materialize > "${run_root}/logs/materialize.log" 2>&1
  fi
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
    validate-config --config "${run_root}/configs/seed1_c3_smoke.yaml" > /dev/null
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
    validate-config --config "${run_root}/configs/seed1_c3_formal.yaml" > /dev/null
}

gpu_preflight() {
  local gpu_id free_mib
  for gpu_id in "${gpus[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
    if [[ "${free_mib}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
      echo "GPU ${gpu_id} has ${free_mib} MiB free; requires ${MIN_FREE_GPU_MIB}" >&2
      exit 2
    fi
  done
}

disk_preflight() {
  local free_bytes
  free_bytes="$(df -B1 --output=avail "${ARTIFACT_ROOT}" | tail -n 1 | tr -d ' ')"
  if [[ ! "${free_bytes}" =~ ^[0-9]+$ || "${free_bytes}" -lt "${MIN_FREE_ARTIFACT_BYTES}" ]]; then
    echo "Artifact filesystem has ${free_bytes:-unknown} bytes free; requires ${MIN_FREE_ARTIFACT_BYTES}" >&2
    exit 2
  fi
}

train_config() {
  local config="$1"
  local log="$2"
  local -a resume_args=()
  if [[ "${RESUME}" == "YES" ]]; then
    resume_args=(--resume latest)
  fi
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
    -m experiments.robotwin.policy_content_adapter.pair280_run \
    train --config "${config}" "${resume_args[@]}" > "${log}" 2>&1
}

case "${PHASE}" in
  prepare)
    prepare
    printf 'PREPARED gpu_started=false utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    ;;
  smoke)
    [[ "${CONFIRM_GPU_WORK}" == "YES" ]] || { echo "smoke requires CONFIRM_GPU_WORK=YES" >&2; exit 2; }
    [[ "${RESUME}" == "NO" ]] || { echo "smoke cannot resume" >&2; exit 2; }
    prepare
    disk_preflight
    gpu_preflight
    [[ ! -e "${run_root}/smoke" ]] || { echo "refusing to overwrite smoke output" >&2; exit 2; }
    printf 'RUNNING phase=smoke utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    train_config "${run_root}/configs/seed1_c3_smoke.yaml" "${run_root}/logs/smoke.log"
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
      audit-run --run-root "${run_root}/smoke" --smoke \
      --output "${run_root}/audits/smoke.json" > "${run_root}/logs/smoke_audit.log" 2>&1
    printf 'DONE phase=smoke audit=PASS utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    ;;
  train)
    [[ "${CONFIRM_GPU_WORK}" == "YES" ]] || { echo "training requires CONFIRM_GPU_WORK=YES" >&2; exit 2; }
    prepare
    [[ -f "${run_root}/audits/smoke.json" ]] || { echo "Pair-280 smoke audit is required" >&2; exit 2; }
    disk_preflight
    gpu_preflight
    if [[ "${RESUME}" == "NO" ]]; then
      [[ ! -e "${run_root}/formal" ]] || { echo "refusing to overwrite formal output" >&2; exit 2; }
    else
      [[ -d "${run_root}/formal" ]] || { echo "formal resume output is missing" >&2; exit 2; }
    fi
    printf 'RUNNING phase=formal resume=%s utc=%s\n' "${RESUME}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    train_config "${run_root}/configs/seed1_c3_formal.yaml" "${run_root}/logs/formal.log"
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
      audit-run --run-root "${run_root}/formal" \
      --output "${run_root}/audits/formal.json" > "${run_root}/logs/formal_audit.log" 2>&1
    printf 'DONE phase=formal audit=PASS utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    ;;
  audit)
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_run \
      audit-run --run-root "${run_root}/formal"
    ;;
esac
