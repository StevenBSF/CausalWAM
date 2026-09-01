#!/usr/bin/env bash
set -euo pipefail

MOTUS_ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-/root/anaconda3/envs/motusdata/bin/python}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM/third_party/RoboTwin}"
WAN_DIR="${WAN_DIR:-/mnt/cpfs-E/baoshifeng/Motus/pretrained_models/Wan2.2-TI2V-5B}"
VLM_DIR="${VLM_DIR:-/mnt/cpfs-E/baoshifeng/Motus/pretrained_models/Qwen3-VL-2B-Instruct}"
BASE_DIR="${BASE_DIR:-${MOTUS_ROOT}/pretrained_models/Motus_robotwin2}"
: "${SETTINGS:?SETTINGS is required}" "${CHECKPOINT:?CHECKPOINT is required}" "${TRAINING_SUMMARY:?TRAINING_SUMMARY is required}"
: "${CONTROL:?CONTROL is required}" "${TRAINING_SEED:?TRAINING_SEED is required}" "${TASK:?TASK is required}"
: "${DOMAIN:?DOMAIN is required}" "${GPU_ID:?GPU_ID is required}" "${CELL_ROOT:?CELL_ROOT is required}"

export PYTHONPATH="${MOTUS_ROOT}:${PYTHONPATH:-}"
COMPLETED="${CELL_ROOT}/completed_rollout.json"
if [[ -f "${COMPLETED}" ]]; then
  "${PY}" -m experiments.robotwin.policy_content_adapter.rollout audit --cell "${COMPLETED}"
  exit 0
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEMPT="${CELL_ROOT}/attempt_${STAMP}"
mkdir -p "${ATTEMPT}/episodes"
PLAN="${ATTEMPT}/plan.json"
"${PY}" -m experiments.robotwin.policy_content_adapter.rollout plan --settings "${SETTINGS}" --checkpoint "${CHECKPOINT}" --training-summary "${TRAINING_SUMMARY}" --control "${CONTROL}" --training-seed "${TRAINING_SEED}" --task "${TASK}" --domain "${DOMAIN}" --cell-output-dir "${ATTEMPT}" --output "${PLAN}"
BINDING="${ATTEMPT}/gpu_binding.json"
"${PY}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime --gpu-id "${GPU_ID}" --output "${BINDING}"
if [[ "${DOMAIN}" == clean ]]; then TASK_CONFIG=demo_clean; SUFFIX=clean; else TASK_CONFIG=demo_randomized; SUFFIX=random; fi
LOG="${ATTEMPT}/worker.log"
set +e
"${PY}" -m experiments.robotwin.policy_content_adapter.launch_pinned_eval --binding "${BINDING}" --robotwin-root "${ROBOTWIN_ROOT}" --motus-root "${MOTUS_ROOT}" -- --config "${MOTUS_ROOT}/inference/robotwin/Motus/deploy_policy.yml" --overrides --task_name "${TASK}" --task_config "${TASK_CONFIG}" --ckpt_setting "${BASE_DIR}" --adapter_checkpoint_path "${CHECKPOINT}" --seed 42 --policy_name Motus --log_dir "${ATTEMPT}" --wan_path "${WAN_DIR}" --vlm_path "${VLM_DIR}" --eval_num_episodes 100 --eval_output_dir "${ATTEMPT}/episodes" --instruction_type unseen >"${LOG}" 2>&1
CODE=$?
set -e
if [[ ${CODE} -ne 0 ]]; then
  printf 'FAILED exit_code=%s\n' "${CODE}" >"${ATTEMPT}/status"
  exit "${CODE}"
fi
RESULT="${ATTEMPT}/episodes/_result_${SUFFIX}.txt"
"${PY}" -m experiments.robotwin.policy_content_adapter.rollout finalize --plan "${PLAN}" --result "${RESULT}" --log "${LOG}" --output "${COMPLETED}"
printf 'DONE\n' >"${ATTEMPT}/status"
