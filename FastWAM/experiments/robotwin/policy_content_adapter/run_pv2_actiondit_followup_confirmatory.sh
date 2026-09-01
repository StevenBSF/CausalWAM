#!/usr/bin/env bash
set -euo pipefail

# Six-checkpoint seed59 confirmation: 3 training seeds x C1/C3, each running
# 3 tasks x 2 official domains x 100 episodes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${OUTPUT_ROOT}/confirmatory_rollouts_seed59_v1}"
GPU_IDS="${GPU_IDS:-0,1,3,4,5,6}"
PHASE="${PHASE:-prepare}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"
RUN_TESTS="${RUN_TESTS:-1}"

case "${PHASE}" in
  prepare|rollout|aggregate|all) ;;
  *) echo "PHASE must be prepare, rollout, aggregate, or all" >&2; exit 2 ;;
esac
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 6 ]]; then
  echo "GPU_IDS must provide exactly six physical GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 6 ]]; then
  echo "GPU_IDS must be distinct" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

amendment="${OUTPUT_ROOT}/manifests/confirmatory_seed59_amendment_v1.json"
status_file="${OUTPUT_ROOT}/pv2_confirmatory.status"

record_failure() {
  local code=$?
  printf 'FAILED phase=%s exit_code=%s utc=%s\n' \
    "${PHASE}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit "${code}"
}
trap record_failure ERR

prepare_confirmatory() {
  if [[ ! -f "${OUTPUT_ROOT}/expansion_posttrain_audit.json" ]]; then
    echo "Seed2/3 expansion posttrain audit is required" >&2
    return 2
  fi
  if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pytest -q \
      experiments/robotwin/policy_content_adapter/tests \
      > "${OUTPUT_ROOT}/confirmatory_cpu_tests.log" 2>&1
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_final \
      --experiment-root "${OUTPUT_ROOT}" --record-cpu-tests \
      > "${OUTPUT_ROOT}/confirmatory_cpu_test_audit.log" 2>&1
  fi
  if [[ ! -f "${amendment}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_confirmatory \
      materialize --experiment-root "${OUTPUT_ROOT}" \
      > "${OUTPUT_ROOT}/confirmatory_amendment_materialize.log" 2>&1
  else
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_confirmatory \
      validate --amendment "${amendment}" > /dev/null
  fi
  printf 'PREPARED seed59=true checkpoints=6 episodes_per_cell=100 rollout_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

gpu_preflight() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "$1" > /dev/null
}

rollout_one() {
  local seed="$1"
  local short="$2"
  local gpu_id="$3"
  local output="${ROLLOUT_ROOT}/seed_${seed}/${short}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_pv2_confirmatory \
    "ckpt=${OUTPUT_ROOT}/runs/seed_${seed}/${short}/checkpoint.pt" \
    "gpu_id=${gpu_id}" \
    seed=59 \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    'EVALUATION.task_name=[place_a2b_left,open_microwave,move_stapler_pad]' \
    EVALUATION.task_config=both \
    EVALUATION.eval_num_episodes=100 \
    "+EVALUATION.pv2_followup_eval_amendment=${amendment}" \
    "EVALUATION.output_dir=${output}" \
    > "${OUTPUT_ROOT}/logs/seed${seed}_${short}_confirmatory_rollout.log" 2>&1
}

run_rollouts() {
  if [[ "${CONFIRM_GPU_WORK}" != "YES" ]]; then
    echo "Confirmatory GPU rollout requires CONFIRM_GPU_WORK=YES" >&2
    return 2
  fi
  prepare_confirmatory
  if [[ -e "${ROLLOUT_ROOT}" ]]; then
    echo "Refusing to overwrite confirmatory rollout root" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/logs"
  local rows=(
    "1 c1 ${gpus[0]}" "1 c3 ${gpus[1]}"
    "2 c1 ${gpus[2]}" "2 c3 ${gpus[3]}"
    "3 c1 ${gpus[4]}" "3 c3 ${gpus[5]}"
  )
  for row in "${rows[@]}"; do
    read -r seed short gpu_id <<< "${row}"
    gpu_preflight "${gpu_id}"
  done
  printf 'RUNNING seed59=true checkpoints=6 total_episodes=3600 gpu_ids=%s utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  local pids=()
  for row in "${rows[@]}"; do
    read -r seed short gpu_id <<< "${row}"
    rollout_one "${seed}" "${short}" "${gpu_id}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one confirmatory rollout failed" >&2
    return 1
  fi
  aggregate_confirmatory
}

aggregate_confirmatory() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_final \
    --experiment-root "${OUTPUT_ROOT}" --write \
    > "${OUTPUT_ROOT}/confirmatory_aggregate.log" 2>&1
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.package_pv2_actiondit_followup \
    --experiment-root "${OUTPUT_ROOT}" \
    > "${OUTPUT_ROOT}/package_core_code.log" 2>&1
  printf 'DONE seed59=true checkpoints=6 records=36 total_episodes=3600 aggregate=PASS utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

case "${PHASE}" in
  prepare) prepare_confirmatory ;;
  rollout) run_rollouts ;;
  aggregate) aggregate_confirmatory ;;
  all) run_rollouts ;;
esac

echo "P-v2 seed59 confirmatory phase=${PHASE}: ${OUTPUT_ROOT}"
