#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU, five-official-epoch P-v2 C1/C3 runner. Preferred micro-batch is
# 16/GPU (global 128). This runner never silently falls back after OOM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/torchrun}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PHASE="${PHASE:-prepare}"
TRAINING_SEED="${TRAINING_SEED:-1}"
CONTROL_ONLY="${CONTROL_ONLY:-pair}"
RESUME="${RESUME:-NO}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"

case "${PHASE}" in
  prepare|smoke|train_seed|audit_seed) ;;
  *) echo "PHASE must be prepare, smoke, train_seed, or audit_seed" >&2; exit 2 ;;
esac
case "${TRAINING_SEED}" in
  1|2|3) ;;
  *) echo "TRAINING_SEED must be 1, 2, or 3" >&2; exit 2 ;;
esac
case "${CONTROL_ONLY}" in
  pair|c1|c3) ;;
  *) echo "CONTROL_ONLY must be pair, c1, or c3" >&2; exit 2 ;;
esac
case "${RESUME}" in
  YES|NO) ;;
  *) echo "RESUME must be YES or NO" >&2; exit 2 ;;
esac
if [[ "${PHASE}" == "smoke" && "${RESUME}" == "YES" ]]; then
  echo "the pair smoke cannot be resumed" >&2
  exit 2
fi
if [[ "${PHASE}" == "smoke" && "${CONTROL_ONLY}" != "pair" ]]; then
  echo "smoke must run the complete C1/C3 pair" >&2
  exit 2
fi
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 8 || "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight distinct physical GPUs" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

status_file="${OUTPUT_ROOT}/full5ep.status"

record_failure() {
  local code=$?
  if [[ -d "${OUTPUT_ROOT}" ]]; then
    printf 'FAILED phase=%s seed=%s exit_code=%s utc=%s\n' \
      "${PHASE}" "${TRAINING_SEED}" "${code}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  fi
  exit "${code}"
}
trap record_failure ERR

prepare_protocol() {
  if [[ ! -f "${OUTPUT_ROOT}/materialization_audit.json" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
      materialize --output-root "${OUTPUT_ROOT}" \
      > "${OUTPUT_ROOT}.materialize.log" 2>&1
  fi
  for short in c1 c3; do
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
      validate-config --config "${OUTPUT_ROOT}/smoke/configs/${short}.yaml" \
      > /dev/null
  done
  for seed in 1 2 3; do
    for short in c1 c3; do
      "${PYTHON_BIN}" -m \
        experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
        validate-config --config "${OUTPUT_ROOT}/configs/seed_${seed}/${short}.yaml" \
        > /dev/null
    done
  done
  printf 'PREPARED epochs=5 preferred_global_batch=128 preferred_steps=18215 gpu_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

gpu_preflight_all() {
  local gpu_id free_mib
  for gpu_id in "${gpus[@]}"; do
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
      preflight --gpu-id "${gpu_id}" > /dev/null
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
    if [[ "${free_mib}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
      echo "GPU ${gpu_id} has ${free_mib} MiB free; requires ${MIN_FREE_GPU_MIB}" >&2
      return 2
    fi
  done
}

distributed_train() {
  local config="$1"
  local log="$2"
  local -a resume_args=()
  if [[ "${RESUME}" == "YES" ]]; then
    resume_args=(--resume latest)
    "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
      -m experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
      train --config "${config}" "${resume_args[@]}" >> "${log}" 2>&1
    return 0
  fi
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
    -m experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
    train --config "${config}" "${resume_args[@]}" > "${log}" 2>&1
}

compact_action_gate() {
  local run_root="$1"
  local log="$2"
  CUDA_VISIBLE_DEVICES="${gpus[0]}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${run_root}/checkpoint.pt" \
    --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda --mixed-precision bf16 --action-horizon 32 \
    --replan-steps 1 --num-inference-steps 1 --seed 0 \
    --output-json "${run_root}/pre_online_action_gate.json" > "${log}" 2>&1
}

run_pair() {
  local seed="$1"
  local smoke="$2"
  local config_root run_root log_prefix audit_path
  local -a controls=(c1 c3)
  if [[ "${CONTROL_ONLY}" != "pair" ]]; then
    controls=("${CONTROL_ONLY}")
  fi
  if [[ "${smoke}" == "YES" ]]; then
    config_root="${OUTPUT_ROOT}/smoke/configs"
    run_root="${OUTPUT_ROOT}/smoke/seed_1"
    log_prefix="${OUTPUT_ROOT}/smoke"
    audit_path="${OUTPUT_ROOT}/smoke/smoke_audit.json"
  else
    config_root="${OUTPUT_ROOT}/configs/seed_${seed}"
    run_root="${OUTPUT_ROOT}/runs/seed_${seed}"
    log_prefix="${OUTPUT_ROOT}/logs/seed${seed}"
    audit_path="${OUTPUT_ROOT}/audits/seed${seed}_posttrain.json"
  fi
  mkdir -p "$(dirname "${log_prefix}")" "$(dirname "${audit_path}")"
  for short in "${controls[@]}"; do
    if [[ "${RESUME}" == "NO" && -e "${run_root}/${short}" ]]; then
      echo "Refusing to overwrite ${run_root}/${short}" >&2
      return 2
    fi
    if [[ "${RESUME}" == "YES" && ! -d "${run_root}/${short}" ]]; then
      echo "Resume target does not exist: ${run_root}/${short}" >&2
      return 2
    fi
    printf 'RUNNING stage=%s seed=%s control=%s global_batch=128 world_size=8 utc=%s\n' \
      "$([[ "${smoke}" == "YES" ]] && echo smoke || echo full5ep)" \
      "${seed}" "${short}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    distributed_train "${config_root}/${short}.yaml" "${log_prefix}_${short}_train.log"
    compact_action_gate "${run_root}/${short}" "${log_prefix}_${short}_action_gate.log"
  done
  if [[ "${CONTROL_ONLY}" != "pair" ]]; then
    printf 'DONE stage=%s seed=%s control=%s action_gate=PASS pair_audit=DEFERRED utc=%s\n' \
      "$([[ "${smoke}" == "YES" ]] && echo smoke || echo full5ep)" \
      "${seed}" "${CONTROL_ONLY}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "${status_file}"
    return 0
  fi
  local smoke_flag=()
  if [[ "${smoke}" == "YES" ]]; then smoke_flag=(--smoke); fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
    audit-pair --output-root "${OUTPUT_ROOT}" --seed "${seed}" \
    "${smoke_flag[@]}" --output "${audit_path}" \
    > "${audit_path%.json}.log" 2>&1
  printf 'DONE stage=%s seed=%s pair_audit=PASS utc=%s\n' \
    "$([[ "${smoke}" == "YES" ]] && echo smoke || echo full5ep)" \
    "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

case "${PHASE}" in
  prepare)
    prepare_protocol
    ;;
  smoke)
    if [[ "${CONFIRM_GPU_WORK}" != "YES" ]]; then
      echo "8-GPU smoke requires CONFIRM_GPU_WORK=YES" >&2
      exit 2
    fi
    prepare_protocol
    gpu_preflight_all
    run_pair 1 YES
    ;;
  train_seed)
    if [[ "${CONFIRM_GPU_WORK}" != "YES" ]]; then
      echo "8-GPU full training requires CONFIRM_GPU_WORK=YES" >&2
      exit 2
    fi
    prepare_protocol
    if [[ ! -f "${OUTPUT_ROOT}/smoke/smoke_audit.json" ]]; then
      echo "Preferred global-batch-128 smoke audit is required" >&2
      exit 2
    fi
    gpu_preflight_all
    run_pair "${TRAINING_SEED}" NO
    ;;
  audit_seed)
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
      audit-pair --output-root "${OUTPUT_ROOT}" --seed "${TRAINING_SEED}" \
      --output "${OUTPUT_ROOT}/audits/seed${TRAINING_SEED}_posttrain.json"
    ;;
esac

echo "P-v2 full5ep phase=${PHASE}: ${OUTPUT_ROOT}"
