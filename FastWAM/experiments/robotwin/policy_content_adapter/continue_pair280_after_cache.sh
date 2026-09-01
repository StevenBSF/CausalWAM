#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CACHE_PIPELINE_PID="${CACHE_PIPELINE_PID:?CACHE_PIPELINE_PID is required}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/fastwam_policy_artifacts/pair280_layer16_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
status_file="${ARTIFACT_ROOT}/automatic_continuation.status"

if [[ ! -r "/proc/${CACHE_PIPELINE_PID}/cmdline" ]]; then
  echo "cache pipeline PID is not alive: ${CACHE_PIPELINE_PID}" >&2
  exit 2
fi
cmdline="$(tr '\0' ' ' < "/proc/${CACHE_PIPELINE_PID}/cmdline")"
if [[ "${cmdline}" != *"run_pair280_cache_extraction.sh"* ]]; then
  echo "CACHE_PIPELINE_PID does not identify Pair-280 extraction" >&2
  exit 2
fi

printf 'WAITING cache_pipeline_pid=%s utc=%s\n' \
  "${CACHE_PIPELINE_PID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
tail --pid="${CACHE_PIPELINE_PID}" -f /dev/null

[[ -f "${ARTIFACT_ROOT}/cache_manifest.json" ]] || {
  printf 'BLOCKED cache_manifest_missing=true utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit 1
}
[[ -f "${ARTIFACT_ROOT}/cache_audit.json" ]] || {
  printf 'BLOCKED cache_audit_missing=true utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit 1
}

cd "${FASTWAM_ROOT}"
printf 'STARTING phase=prepare utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
PHASE=prepare GPU_IDS="${GPU_IDS}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash experiments/robotwin/policy_content_adapter/run_pair280_seed1_c3.sh

printf 'STARTING phase=smoke utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
PHASE=smoke CONFIRM_GPU_WORK=YES GPU_IDS="${GPU_IDS}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash experiments/robotwin/policy_content_adapter/run_pair280_seed1_c3.sh

printf 'STARTING phase=formal utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
PHASE=train CONFIRM_GPU_WORK=YES RESUME=NO GPU_IDS="${GPU_IDS}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash experiments/robotwin/policy_content_adapter/run_pair280_seed1_c3.sh
printf 'DONE formal_audit=PASS utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"

