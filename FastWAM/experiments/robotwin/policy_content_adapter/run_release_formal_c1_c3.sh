#!/usr/bin/env bash
set -euo pipefail

# Formal release-base C1/C3 Stage-2 runner.  The safe default is CPU-only
# materialization.  GPU training requires both PHASE=train/all and an explicit
# confirmation.  Online rollout and C0 evaluation are intentionally absent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
PHASE="${PHASE:-prepare}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
CONFIRM_FORMAL_TRAINING="${CONFIRM_FORMAL_TRAINING:-NO}"
RUN_TESTS="${RUN_TESTS:-1}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"

case "${PHASE}" in
  prepare|train|audit|all) ;;
  *) echo "PHASE must be prepare, train, audit, or all" >&2; exit 2 ;;
esac

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${FASTWAM_ROOT}"

prepare_formal() {
  if [[ -e "${FORMAL_OUTPUT_ROOT}" ]]; then
    echo "Refusing to reuse formal output root: ${FORMAL_OUTPUT_ROOT}" >&2
    return 2
  fi
  if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pytest -q \
      experiments/robotwin/policy_content_adapter/tests/test_release_formal_c1c3.py \
      experiments/robotwin/policy_content_adapter/tests/test_configs.py \
      experiments/robotwin/policy_content_adapter/tests/test_p_mode_selection.py \
      experiments/robotwin/policy_content_adapter/tests/test_train_protocol.py
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.materialize_release_formal_c1c3 \
    --output-root "${FORMAL_OUTPUT_ROOT}" \
    > "${FORMAL_OUTPUT_ROOT}.materialize.log" 2>&1
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_c1c3_audit \
    --materialization-manifest "${FORMAL_OUTPUT_ROOT}/materialization_manifest.json" \
    --stage prelaunch \
    > "${FORMAL_OUTPUT_ROOT}/prelaunch_audit.log"
  printf 'PREPARED utc=%s gpu_training_started=false output=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FORMAL_OUTPUT_ROOT}" \
    > "${FORMAL_OUTPUT_ROOT}/formal_c1_c3.status"
}

run_one() {
  local seed="$1"
  local short="$2"
  local gpu="$3"
  local config="${FORMAL_OUTPUT_ROOT}/configs/seed_${seed}/${short}.yaml"
  local run_root="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}"
  local log="${FORMAL_OUTPUT_ROOT}/logs/seed_${seed}_${short}.log"
  local status="${FORMAL_OUTPUT_ROOT}/logs/seed_${seed}_${short}.status"
  if [[ ! -f "${config}" ]]; then
    echo "Missing formal config: ${config}" >&2
    return 2
  fi
  if [[ -e "${run_root}" || -e "${log}" || -e "${status}" ]]; then
    echo "Refusing to overwrite seed=${seed} control=${short} artifacts" >&2
    return 2
  fi
  printf 'RUNNING seed=%s control=%s gpu=%s utc=%s\n' \
    "${seed}" "${short}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.train \
      --config "${config}" > "${log}" 2>&1; then
    printf 'DONE seed=%s control=%s gpu=%s utc=%s\n' \
      "${seed}" "${short}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  else
    local code=$?
    printf 'FAILED seed=%s control=%s gpu=%s exit_code=%s utc=%s\n' \
      "${seed}" "${short}" "${gpu}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    return "${code}"
  fi
}

run_seed_pair() {
  local seed="$1"
  local gpu="$2"
  run_one "${seed}" c1 "${gpu}"
  run_one "${seed}" c3 "${gpu}"
}

train_formal() {
  if [[ "${CONFIRM_FORMAL_TRAINING}" != "YES" ]]; then
    echo "Formal GPU training requires CONFIRM_FORMAL_TRAINING=YES" >&2
    return 2
  fi
  if [[ ! -f "${FORMAL_OUTPUT_ROOT}/materialization_manifest.json" ]]; then
    echo "Formal materialization is missing: ${FORMAL_OUTPUT_ROOT}" >&2
    return 2
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_c1c3_audit \
    --materialization-manifest "${FORMAL_OUTPUT_ROOT}/materialization_manifest.json" \
    --stage prelaunch > "${FORMAL_OUTPUT_ROOT}/prelaunch_train_audit.log"

  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  if [[ "${#gpu_array[@]}" -ne 1 && "${#gpu_array[@]}" -ne 3 && "${#gpu_array[@]}" -ne 6 ]]; then
    echo "GPU_IDS must contain exactly 1, 3, or 6 unique devices" >&2
    return 2
  fi
  declare -A seen_gpu=()
  for gpu in "${gpu_array[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -n "${seen_gpu[${gpu}]:-}" ]]; then
      echo "GPU_IDS must contain unique non-negative integers" >&2
      return 2
    fi
    seen_gpu["${gpu}"]=1
  done
  mkdir -p "${FORMAL_OUTPUT_ROOT}/logs"
  printf 'RUNNING formal_training=true jobs=6 gpu_ids=%s utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${FORMAL_OUTPUT_ROOT}/formal_c1_c3.status"

  local failures=0
  local pids=()
  if [[ "${#gpu_array[@]}" -eq 6 ]]; then
    local index=0
    for seed in 1 2 3; do
      for short in c1 c3; do
        run_one "${seed}" "${short}" "${gpu_array[${index}]}" &
        pids+=("$!")
        index=$((index + 1))
      done
    done
  elif [[ "${#gpu_array[@]}" -eq 3 ]]; then
    for seed in 1 2 3; do
      run_seed_pair "${seed}" "${gpu_array[$((seed - 1))]}" &
      pids+=("$!")
    done
  else
    run_seed_pair 1 "${gpu_array[0]}" & pids+=("$!")
    wait "${pids[0]}" || failures=$((failures + 1))
    pids=()
    if [[ "${failures}" -eq 0 ]]; then
      run_seed_pair 2 "${gpu_array[0]}" & pids+=("$!")
      wait "${pids[0]}" || failures=$((failures + 1))
      pids=()
    fi
    if [[ "${failures}" -eq 0 ]]; then
      run_seed_pair 3 "${gpu_array[0]}" & pids+=("$!")
    fi
  fi
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  if [[ "${failures}" -ne 0 ]]; then
    printf 'FAILED failed_workers=%s utc=%s\n' \
      "${failures}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "${FORMAL_OUTPUT_ROOT}/formal_c1_c3.status"
    return 1
  fi

  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_c1c3_audit \
    --materialization-manifest "${FORMAL_OUTPUT_ROOT}/materialization_manifest.json" \
    --stage posttrain \
    --output-json "${FORMAL_OUTPUT_ROOT}/strict_posttrain_pair_audit.json" \
    > "${FORMAL_OUTPUT_ROOT}/posttrain_pair_audit.log"
  printf 'DONE formal_training=true online_rollout_started=false utc=%s output=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${FORMAL_OUTPUT_ROOT}" \
    > "${FORMAL_OUTPUT_ROOT}/formal_c1_c3.status"
}

audit_formal() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_c1c3_audit \
    --materialization-manifest "${FORMAL_OUTPUT_ROOT}/materialization_manifest.json" \
    --stage posttrain \
    --output-json "${FORMAL_OUTPUT_ROOT}/strict_posttrain_pair_audit.json"
}

case "${PHASE}" in
  prepare) prepare_formal ;;
  train) train_formal ;;
  audit) audit_formal ;;
  all) prepare_formal; train_formal ;;
esac

echo "Release formal C1/C3 phase=${PHASE} complete: ${FORMAL_OUTPUT_ROOT}"
echo "This runner did not start online rollout or C0 evaluation."
