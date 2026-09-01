#!/usr/bin/env bash
set -euo pipefail

# Event-driven handoff: wait for the four-run expansion parent to exit, require
# its strict posttrain audit, then start the already-authorized seed59 matrix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1}"
TRAINING_RUNNER_PID="${TRAINING_RUNNER_PID:?TRAINING_RUNNER_PID is required}"
GPU_IDS="${GPU_IDS:-0,1,3,4,5,6}"

case "${TRAINING_RUNNER_PID}" in
  ''|*[!0-9]*) echo "TRAINING_RUNNER_PID must be an integer" >&2; exit 2 ;;
esac
if [[ "${TRAINING_RUNNER_PID}" -le 1 ]]; then
  echo "TRAINING_RUNNER_PID must be greater than one" >&2
  exit 2
fi
pid_cmdline="/proc/${TRAINING_RUNNER_PID}/cmdline"
if [[ ! -r "${pid_cmdline}" ]]; then
  echo "Training runner PID is not live: ${TRAINING_RUNNER_PID}" >&2
  exit 2
fi
cmdline="$(tr '\0' ' ' < "${pid_cmdline}")"
if [[ "${cmdline}" != *"run_pv2_actiondit_followup_expansion.sh"* ]]; then
  echo "PID is not the audited P-v2 expansion runner" >&2
  exit 2
fi

cd "${FASTWAM_ROOT}"
printf 'WAITING training_runner_pid=%s mode=event_driven utc=%s\n' \
  "${TRAINING_RUNNER_PID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail --pid="${TRAINING_RUNNER_PID}" -f /dev/null

posttrain="${OUTPUT_ROOT}/expansion_posttrain_audit.json"
if [[ ! -f "${posttrain}" ]]; then
  echo "Expansion runner exited without strict posttrain audit" >&2
  exit 1
fi
printf 'EXPANSION_COMPLETE starting_seed59=true utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env PHASE=all CONFIRM_GPU_WORK=YES RUN_TESTS=1 GPU_IDS="${GPU_IDS}" \
  bash experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup_confirmatory.sh
printf 'CONFIRMATORY_COMPLETE terminal_deliverables=true utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
