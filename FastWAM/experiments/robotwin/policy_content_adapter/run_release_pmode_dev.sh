#!/usr/bin/env bash
set -euo pipefail

# Release-base P-v1/P-v2 dev selection pipeline.  The safe default is CPU-only
# preparation.  GPU work starts only with an explicit PHASE=train/rollout/all.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1}"
PHASE="${PHASE:-prepare}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-0,0}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-1}"
PARALLEL_ROLLOUT="${PARALLEL_ROLLOUT:-0}"

case "${PHASE}" in
  prepare|train|rollout|select|all) ;;
  *) echo "PHASE must be prepare, train, rollout, select, or all" >&2; exit 2 ;;
esac

IFS=',' read -r -a train_gpus <<< "${TRAIN_GPU_IDS}"
IFS=',' read -r -a rollout_gpus <<< "${ROLLOUT_GPU_IDS}"
if [[ "${#train_gpus[@]}" -ne 2 || "${#rollout_gpus[@]}" -ne 2 ]]; then
  echo "TRAIN_GPU_IDS and ROLLOUT_GPU_IDS must each contain exactly two ids" >&2
  exit 2
fi
if [[ "${PARALLEL_ROLLOUT}" == "1" && "${rollout_gpus[0]}" == "${rollout_gpus[1]}" ]]; then
  echo "Parallel rollout requires two distinct explicitly PCI-pinned GPUs" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

manifest="${OUTPUT_ROOT}/materialization_manifest.json"
p_v1_config="${OUTPUT_ROOT}/configs/p_v1_dev_pilot.yaml"
p_v2_config="${OUTPUT_ROOT}/configs/p_v2_dev_pilot.yaml"
status_file="${OUTPUT_ROOT}/p_mode_dev.status"

prepare() {
  if [[ ! -e "${OUTPUT_ROOT}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.materialize_release_pmode_dev \
      --output-root "${OUTPUT_ROOT}" \
      --training-seed 42 \
      --max-steps 100 \
      --official-batch-size 1 \
      --paired-groups-per-batch 2 \
      --world-size 1 \
      --gradient-accumulation-steps 1 \
      --simulator-seed 23
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_pmode_dev_audit \
    --materialization-manifest "${manifest}" \
    --stage materialization
}

run_train_one() {
  local regime="$1"
  local gpu_id="$2"
  local config="$3"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.train \
    --config "${config}" \
    > "${OUTPUT_ROOT}/${regime}_train.log" 2>&1
}

train_pair() {
  prepare
  if [[ -e "${OUTPUT_ROOT}/runs/p_v1" || -e "${OUTPUT_ROOT}/runs/p_v2" ]]; then
    echo "Refusing to overwrite an existing P-mode dev training run" >&2
    return 2
  fi
  printf 'RUNNING stage=train utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  if [[ "${PARALLEL_TRAIN}" == "1" ]]; then
    run_train_one p_v1 "${train_gpus[0]}" "${p_v1_config}" &
    local p1_pid=$!
    run_train_one p_v2 "${train_gpus[1]}" "${p_v2_config}" &
    local p2_pid=$!
    local failed=0
    wait "${p1_pid}" || failed=1
    wait "${p2_pid}" || failed=1
    if [[ "${failed}" -ne 0 ]]; then
      echo "P-mode dev training failed; inspect ${OUTPUT_ROOT}/p_v*_train.log" >&2
      return 1
    fi
  else
    run_train_one p_v1 "${train_gpus[0]}" "${p_v1_config}"
    run_train_one p_v2 "${train_gpus[1]}" "${p_v2_config}"
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_pmode_dev_audit \
    --materialization-manifest "${manifest}" \
    --stage posttrain \
    > "${OUTPUT_ROOT}/posttrain_pair_audit.json"
  printf 'DONE stage=train utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
}

compact_action_gate() {
  local regime="$1"
  local gpu_id="$2"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${OUTPUT_ROOT}/runs/${regime}/checkpoint.pt" \
    --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${OUTPUT_ROOT}/runs/${regime}/pre_online_action_gate.json"
}

runtime_preflight() {
  local gpu_id="$1"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu_id}"
}

run_rollout_one() {
  local regime="$1"
  local gpu_id="$2"
  local simulator_seed="$3"
  runtime_preflight "${gpu_id}"
  compact_action_gate "${regime}" "${gpu_id}"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${OUTPUT_ROOT}/runs/${regime}/checkpoint.pt" \
    "gpu_id=${gpu_id}" \
    "seed=${simulator_seed}" \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    'EVALUATION.task_name=[place_a2b_left,open_microwave,move_stapler_pad]' \
    EVALUATION.task_config=both \
    EVALUATION.eval_num_episodes=20 \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/rollouts/${regime}" \
    > "${OUTPUT_ROOT}/${regime}_rollout.log" 2>&1
}

rollout_pair() {
  if [[ -e "${OUTPUT_ROOT}/rollouts/p_v1" || -e "${OUTPUT_ROOT}/rollouts/p_v2" ]]; then
    echo "Refusing to overwrite existing P-mode dev rollout results" >&2
    return 2
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_pmode_dev_audit \
    --materialization-manifest "${manifest}" \
    --stage posttrain \
    > "${OUTPUT_ROOT}/posttrain_pair_audit.json"
  local simulator_seed
  simulator_seed="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulator_seed"])' "${OUTPUT_ROOT}/manifests/dev_selection_seed_bank.json")"
  printf 'RUNNING stage=rollout utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  if [[ "${PARALLEL_ROLLOUT}" == "1" ]]; then
    run_rollout_one p_v1 "${rollout_gpus[0]}" "${simulator_seed}" &
    local p1_pid=$!
    run_rollout_one p_v2 "${rollout_gpus[1]}" "${simulator_seed}" &
    local p2_pid=$!
    local failed=0
    wait "${p1_pid}" || failed=1
    wait "${p2_pid}" || failed=1
    if [[ "${failed}" -ne 0 ]]; then
      echo "P-mode online rollout failed; inspect ${OUTPUT_ROOT}/p_v*_rollout.log" >&2
      return 1
    fi
  else
    run_rollout_one p_v1 "${rollout_gpus[0]}" "${simulator_seed}"
    run_rollout_one p_v2 "${rollout_gpus[1]}" "${simulator_seed}"
  fi
  printf 'DONE stage=rollout utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
}

select_mode() {
  printf 'RUNNING stage=selection utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.p_mode_selection select \
    --p-v1-manifest "${OUTPUT_ROOT}/rollouts/p_v1/completed_rollouts.json" \
    --p-v2-manifest "${OUTPUT_ROOT}/rollouts/p_v2/completed_rollouts.json" \
    --output "${OUTPUT_ROOT}/p_mode_selection.json"
  printf 'DONE stage=selection utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
}

case "${PHASE}" in
  prepare) prepare ;;
  train) train_pair ;;
  rollout) rollout_pair ;;
  select) select_mode ;;
  all)
    trap 'code=$?; printf "FAILED exit_code=%s utc=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"; exit "${code}"' ERR
    train_pair
    rollout_pair
    select_mode
    ;;
esac

echo "P-mode dev phase ${PHASE} completed: ${OUTPUT_ROOT}"
