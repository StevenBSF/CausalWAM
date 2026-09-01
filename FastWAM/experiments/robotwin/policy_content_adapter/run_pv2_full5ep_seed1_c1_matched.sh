#!/usr/bin/env bash
set -euo pipefail

# Run the matched five-epoch P-v2 C1 control for the completed seed-1 C3 run.
# A fresh invocation first executes the immutable three-step C1/C3 pair smoke;
# only a passing smoke audit is allowed to unlock the 18,215-step C1 run.

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_c1_matched_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-NO}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"
BASE_RUNNER="${FASTWAM_ROOT}/experiments/robotwin/policy_content_adapter/run_pv2_actiondit_full5ep.sh"
MASTER_STATUS="${OUTPUT_ROOT}/matched_c1.status"

case "${RESUME}" in
  YES|NO) ;;
  *) echo "RESUME must be YES or NO" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${FASTWAM_ROOT}"

if [[ "${RESUME}" == "NO" ]]; then
  if [[ ! -f "${OUTPUT_ROOT}/smoke/smoke_audit.json" ]]; then
    printf 'RUNNING stage=paired_smoke seed=1 gpu_ids=%s utc=%s\n' \
      "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MASTER_STATUS}"
    env OUTPUT_ROOT="${OUTPUT_ROOT}" GPU_IDS="${GPU_IDS}" \
      PHASE=smoke CONTROL_ONLY=pair RESUME=NO CONFIRM_GPU_WORK=YES \
      MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB}" \
      bash "${BASE_RUNNER}"
  fi
  printf 'RUNNING stage=full5ep seed=1 control=c1 steps=18215 global_batch=128 gpu_ids=%s utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MASTER_STATUS}"
  env OUTPUT_ROOT="${OUTPUT_ROOT}" GPU_IDS="${GPU_IDS}" \
    PHASE=train_seed TRAINING_SEED=1 CONTROL_ONLY=c1 RESUME=NO \
    CONFIRM_GPU_WORK=YES MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB}" \
    bash "${BASE_RUNNER}"
else
  [[ -f "${OUTPUT_ROOT}/smoke/smoke_audit.json" ]] || {
    echo "matched C1 resume requires a passing pair smoke" >&2
    exit 2
  }
  [[ -d "${OUTPUT_ROOT}/runs/seed_1/c1" ]] || {
    echo "matched C1 resume target does not exist" >&2
    exit 2
  }
  printf 'RUNNING stage=full5ep_resume seed=1 control=c1 steps=18215 global_batch=128 gpu_ids=%s utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MASTER_STATUS}"
  env OUTPUT_ROOT="${OUTPUT_ROOT}" GPU_IDS="${GPU_IDS}" \
    PHASE=train_seed TRAINING_SEED=1 CONTROL_ONLY=c1 RESUME=YES \
    CONFIRM_GPU_WORK=YES MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB}" \
    bash "${BASE_RUNNER}"
fi

printf 'DONE stage=full5ep seed=1 control=c1 steps=18215 action_gate=PASS utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MASTER_STATUS}"
