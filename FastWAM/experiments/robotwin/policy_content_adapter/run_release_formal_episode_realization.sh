#!/usr/bin/env bash
set -euo pipefail

# Policy-independent realization of the checkpoint-bound final candidate bank.
# The default audit phase is CPU-only.  The realize phase uses six GPUs only to
# create the six exact expert-valid (seed, instruction) lists; it never loads a
# policy checkpoint and never executes a learned policy action.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
PHASE="${PHASE:-audit}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
CANDIDATE_BANK="${CANDIDATE_BANK:-${FORMAL_OUTPUT_ROOT}/manifests/final_test_seed_bank.json}"
FORMAL_LOCK="${FORMAL_LOCK:-${FORMAL_OUTPUT_ROOT}/manifests/formal_protocol_lock.json}"
REALIZATION_ROOT="${REALIZATION_ROOT:-${FORMAL_OUTPUT_ROOT}/manifests/final_test_exact_realization_v1}"
REALIZATION_BANK="${REALIZATION_BANK:-${REALIZATION_ROOT}/realization_bank.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTWAM_ROOT}/third_party/RoboTwin}"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6}"
CONFIRM_EXPERT_REALIZATION="${CONFIRM_EXPERT_REALIZATION:-NO}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${PHASE}" in
  audit|realize|finalize|all) ;;
  *) echo "PHASE must be audit, realize, finalize, or all" >&2; exit 2 ;;
esac

export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

audit_inputs() {
  if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pytest -q \
      experiments/robotwin/policy_content_adapter/tests/test_formal_episode_selector.py
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.formal_episode_protocol \
    audit-inputs \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --candidate-bank "${CANDIDATE_BANK}" \
    --formal-lock "${FORMAL_LOCK}"
}

cell_path() {
  local task="$1"
  local task_config="$2"
  echo "${REALIZATION_ROOT}/cells/${task}/${task_config}.json"
}

realize_one() {
  local task="$1"
  local task_config="$2"
  local gpu="$3"
  local output
  output="$(cell_path "${task}" "${task_config}")"
  local log="${output%.json}.attempt_${RUN_STAMP}.log"
  local status="${output%.json}.status"
  if [[ -f "${output}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.formal_episode_protocol \
      audit --cell "${output}" > /dev/null
    echo "SKIP audited exact realization cell: ${task}/${task_config}"
    return 0
  fi
  if [[ -e "${log}" ]]; then
    echo "Refusing to overwrite realization attempt log: ${log}" >&2
    return 2
  fi
  mkdir -p "$(dirname "${output}")"
  printf 'RUNNING task=%s task_config=%s gpu=%s expert_only=true utc=%s\n' \
    "${task}" "${task_config}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${status}"
  if "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.formal_episode_protocol \
      realize-cell \
      --robotwin-root "${ROBOTWIN_ROOT}" \
      --candidate-bank "${CANDIDATE_BANK}" \
      --formal-lock "${FORMAL_LOCK}" \
      --task "${task}" \
      --task-config "${task_config}" \
      --gpu-id "${gpu}" \
      --output "${output}" > "${log}" 2>&1; then
    printf 'DONE task=%s task_config=%s gpu=%s episodes=100 expert_only=true utc=%s\n' \
      "${task}" "${task_config}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "${status}"
  else
    local code=$?
    printf 'FAILED task=%s task_config=%s gpu=%s exit_code=%s utc=%s\n' \
      "${task}" "${task_config}" "${gpu}" "${code}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    return "${code}"
  fi
}

finalize_bank() {
  local cells=(
    "$(cell_path place_a2b_left demo_clean)"
    "$(cell_path place_a2b_left demo_randomized)"
    "$(cell_path open_microwave demo_clean)"
    "$(cell_path open_microwave demo_randomized)"
    "$(cell_path move_stapler_pad demo_clean)"
    "$(cell_path move_stapler_pad demo_randomized)"
  )
  if [[ -f "${REALIZATION_BANK}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.formal_episode_protocol \
      audit --bank "${REALIZATION_BANK}"
    return 0
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.formal_episode_protocol \
    finalize \
    --candidate-bank "${CANDIDATE_BANK}" \
    --formal-lock "${FORMAL_LOCK}" \
    --cell "${cells[0]}" \
    --cell "${cells[1]}" \
    --cell "${cells[2]}" \
    --cell "${cells[3]}" \
    --cell "${cells[4]}" \
    --cell "${cells[5]}" \
    --output "${REALIZATION_BANK}"
}

realize_all() {
  if [[ "${CONFIRM_EXPERT_REALIZATION}" != "YES" ]]; then
    echo "GPU expert-only realization requires CONFIRM_EXPERT_REALIZATION=YES" >&2
    return 2
  fi
  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  if [[ "${#gpu_array[@]}" -ne 6 ]]; then
    echo "GPU_IDS must contain exactly six devices" >&2
    return 2
  fi
  declare -A seen=()
  local gpu
  for gpu in "${gpu_array[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -n "${seen[${gpu}]:-}" ]]; then
      echo "GPU_IDS must be unique non-negative integers" >&2
      return 2
    fi
    seen["${gpu}"]=1
  done
  audit_inputs
  mkdir -p "${REALIZATION_ROOT}/cells"
  local tasks=(
    place_a2b_left place_a2b_left
    open_microwave open_microwave
    move_stapler_pad move_stapler_pad
  )
  local configs=(
    demo_clean demo_randomized
    demo_clean demo_randomized
    demo_clean demo_randomized
  )
  local pids=()
  local index
  for index in 0 1 2 3 4 5; do
    realize_one "${tasks[${index}]}" "${configs[${index}]}" \
      "${gpu_array[${index}]}" &
    pids+=("$!")
  done
  local failures=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  if [[ "${failures}" -ne 0 ]]; then
    echo "Exact expert realization failed in ${failures} cell(s); successful cells are preserved." >&2
    return 1
  fi
  finalize_bank
}

case "${PHASE}" in
  audit) audit_inputs ;;
  realize) realize_all ;;
  finalize) finalize_bank ;;
  all) realize_all ;;
esac

echo "Formal exact episode realization phase=${PHASE} complete: ${REALIZATION_BANK}"
