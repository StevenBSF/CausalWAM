#!/usr/bin/env bash
set -euo pipefail

# Event-driven, no-result-peeking handoff from the long online rollout to the
# immutable pilot gate.  This script never starts seeds 2/3 or seed59 work.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1}"
PILOT_RUNNER_PID="${PILOT_RUNNER_PID:?PILOT_RUNNER_PID is required}"

case "${PILOT_RUNNER_PID}" in
  ''|*[!0-9]*) echo "PILOT_RUNNER_PID must be a positive integer" >&2; exit 2 ;;
esac
if [[ "${PILOT_RUNNER_PID}" -le 1 ]]; then
  echo "PILOT_RUNNER_PID must be greater than 1" >&2
  exit 2
fi

export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

pid_cmdline="/proc/${PILOT_RUNNER_PID}/cmdline"
if [[ ! -r "${pid_cmdline}" ]]; then
  echo "Pilot runner PID is not live: ${PILOT_RUNNER_PID}" >&2
  exit 2
fi
cmdline="$(tr '\0' ' ' < "${pid_cmdline}")"
if [[ "${cmdline}" != *"run_pv2_actiondit_followup.sh"* ]]; then
  echo "PID ${PILOT_RUNNER_PID} is not the P-v2 follow-up runner" >&2
  exit 2
fi

printf 'WAITING runner_pid=%s mode=event_driven_no_result_polling utc=%s\n' \
  "${PILOT_RUNNER_PID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# GNU tail uses --pid to terminate after the observed process exits.  No
# result/log file is read while the policy rollouts are active.
tail --pid="${PILOT_RUNNER_PID}" -f /dev/null

c1_manifest="${OUTPUT_ROOT}/pilot_rollouts_100ep_seed53_v1/c1/completed_rollouts.json"
c3_manifest="${OUTPUT_ROOT}/pilot_rollouts_100ep_seed53_v1/c3/completed_rollouts.json"
if [[ ! -f "${c1_manifest}" || ! -f "${c3_manifest}" ]]; then
  printf 'ROLLOUT_INCOMPLETE c1_manifest=%s c3_manifest=%s utc=%s\n' \
    "$([[ -f "${c1_manifest}" ]] && echo PRESENT || echo MISSING)" \
    "$([[ -f "${c3_manifest}" ]] && echo PRESENT || echo MISSING)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  exit 1
fi

printf 'ROLLOUT_COMPLETE starting_locked_gate=true utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
env PHASE=pilot_gate RUN_TESTS=0 \
  bash experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup.sh
printf 'GATE_COMPLETE expansion_not_started=true utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
