#!/usr/bin/env bash
set -euo pipefail

# Conditional seed-2/3 P-v2 training.  This runner cannot materialize or train
# unless the immutable 100-episode seed-53 pilot decision is PASS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1}"
GPU_IDS="${GPU_IDS:-0,1,4,5}"
PHASE="${PHASE:-prepare}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"

case "${PHASE}" in
  prepare|train|audit|all) ;;
  *) echo "PHASE must be prepare, train, audit, or all" >&2; exit 2 ;;
esac

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "GPU_IDS must provide exactly four physical GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU_IDS must be distinct" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

status_file="${OUTPUT_ROOT}/pv2_expansion.status"

record_failure() {
  local code=$?
  printf 'FAILED phase=%s exit_code=%s utc=%s\n' \
    "${PHASE}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit "${code}"
}
trap record_failure ERR

prepare_expansion() {
  if [[ ! -f "${OUTPUT_ROOT}/expansion_materialization_audit.json" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_expansion \
      materialize --experiment-root "${OUTPUT_ROOT}" \
      > "${OUTPUT_ROOT}/expansion_materialize.log" 2>&1
  fi
  for seed in 2 3; do
    for short in c1 c3; do
      "${PYTHON_BIN}" -m \
        experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_expansion \
        validate-config --config "${OUTPUT_ROOT}/configs/seed_${seed}/${short}.yaml" \
        > /dev/null
    done
  done
  printf 'PREPARED seeds=2,3 gpu_training_started=false seed59_rollout_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

gpu_preflight() {
  local gpu_id="$1"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu_id}" > /dev/null
}

train_one() {
  local seed="$1"
  local short="$2"
  local gpu_id="$3"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_expansion \
    train --config "${OUTPUT_ROOT}/configs/seed_${seed}/${short}.yaml" \
    > "${OUTPUT_ROOT}/logs/seed${seed}_${short}_train.log" 2>&1
}

compact_action_gate() {
  local seed="$1"
  local short="$2"
  local gpu_id="$3"
  local run_root="${OUTPUT_ROOT}/runs/seed_${seed}/${short}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${run_root}/checkpoint.pt" \
    --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${run_root}/pre_online_action_gate.json" \
    > "${OUTPUT_ROOT}/logs/seed${seed}_${short}_action_gate.log" 2>&1
}

train_expansion() {
  if [[ "${CONFIRM_GPU_WORK}" != "YES" ]]; then
    echo "Seed2/3 GPU training requires CONFIRM_GPU_WORK=YES" >&2
    return 2
  fi
  prepare_expansion
  mkdir -p "${OUTPUT_ROOT}/logs"
  local rows=("2 c1 ${gpus[0]}" "2 c3 ${gpus[1]}" "3 c1 ${gpus[2]}" "3 c3 ${gpus[3]}")
  local pids=()
  for row in "${rows[@]}"; do
    read -r seed short gpu_id <<< "${row}"
    if [[ -e "${OUTPUT_ROOT}/runs/seed_${seed}/${short}" ]]; then
      echo "Refusing to overwrite seed${seed}/${short} run" >&2
      return 2
    fi
    gpu_preflight "${gpu_id}"
  done
  printf 'RUNNING stage=seed2_seed3_train gpu_ids=%s utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  for row in "${rows[@]}"; do
    read -r seed short gpu_id <<< "${row}"
    train_one "${seed}" "${short}" "${gpu_id}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one seed2/3 training worker failed" >&2
    return 1
  fi

  pids=()
  for row in "${rows[@]}"; do
    read -r seed short gpu_id <<< "${row}"
    compact_action_gate "${seed}" "${short}" "${gpu_id}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one seed2/3 action gate failed" >&2
    return 1
  fi
  audit_expansion
  printf 'DONE stage=seed2_seed3_train posttrain_audit=PASS seed59_rollout_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

audit_expansion() {
  if [[ -e "${OUTPUT_ROOT}/expansion_posttrain_audit.json" ]]; then
    echo "Refusing to overwrite expansion posttrain audit" >&2
    return 2
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_expansion \
    audit-posttrain --experiment-root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/expansion_posttrain_audit.json" \
    > "${OUTPUT_ROOT}/expansion_posttrain_audit.log" 2>&1
}

case "${PHASE}" in
  prepare) prepare_expansion ;;
  train) train_expansion ;;
  audit) audit_expansion ;;
  all) train_expansion ;;
esac

echo "P-v2 seed2/3 expansion phase=${PHASE}: ${OUTPUT_ROOT}"
