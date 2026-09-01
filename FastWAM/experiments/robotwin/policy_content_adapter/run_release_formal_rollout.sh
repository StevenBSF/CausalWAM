#!/usr/bin/env bash
set -euo pipefail

# Strict release-base C1/C3 final evaluation.
#
# The checkpoint-bound final-test seed bank is only a candidate pool.  This
# runner therefore refuses to prepare until the policy-independent exact
# realization bank exists.  Formal evaluation is six sequential task/domain
# waves; each wave evaluates all six C1/C3 checkpoints in parallel on six
# physical GPUs.  Every worker executes exactly one fixed 100-episode list.
# Failed attempts are preserved and a retry creates a new attempt directory;
# an already completed cell is re-audited and skipped, never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROFILE="${PROFILE:-author_stock}"
if [[ "${PROFILE}" == "author_stock" ]]; then
  exec /bin/bash "${SCRIPT_DIR}/run_release_formal_stock_rollout.sh" "$@"
fi
if [[ "${PROFILE}" != "exact" ]]; then
  echo "PROFILE must be author_stock (default) or exact" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
PHASE="${PHASE:-prepare}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
REALIZATION_BANK="${REALIZATION_BANK:-${FORMAL_OUTPUT_ROOT}/manifests/final_test_exact_realization_v1/realization_bank.json}"
ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-${FORMAL_OUTPUT_ROOT}/online_rollouts_final_test_v1}"
PLAN_PATH="${PLAN_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/formal_rollout_plan_v1.json}"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"
CONFIRM_FORMAL_ROLLOUT="${CONFIRM_FORMAL_ROLLOUT:-NO}"
RUN_TESTS="${RUN_TESTS:-1}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTWAM_ROOT}/third_party/RoboTwin}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${PHASE}" in
  prepare|rollout|aggregate|all) ;;
  *) echo "PHASE must be prepare, rollout, aggregate, or all" >&2; exit 2 ;;
esac
if [[ "${RUN_TESTS}" != "0" && "${RUN_TESTS}" != "1" ]]; then
  echo "RUN_TESTS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${MIN_FREE_GPU_MIB}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GPU_MIB must be a non-negative integer" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${FASTWAM_ROOT}"

prepare_rollout() {
  if [[ -e "${PLAN_PATH}" ]]; then
    echo "Refusing to overwrite formal rollout plan: ${PLAN_PATH}" >&2
    return 2
  fi
  if [[ -e "${ROLLOUT_OUTPUT_ROOT}" ]]; then
    echo "Refusing to reuse rollout root during immutable plan creation: ${ROLLOUT_OUTPUT_ROOT}" >&2
    return 2
  fi
  if [[ ! -f "${REALIZATION_BANK}" ]]; then
    echo "BLOCKED: exact policy-independent realization bank is missing: ${REALIZATION_BANK}" >&2
    echo "Do not run policy rollout against the candidate seed bank directly." >&2
    return 2
  fi
  if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pytest -q \
      experiments/robotwin/policy_content_adapter/tests/test_formal_episode_selector.py \
      experiments/robotwin/policy_content_adapter/tests/test_formal_episode_replay.py \
      experiments/robotwin/policy_content_adapter/tests/test_release_formal_rollout.py \
      experiments/robotwin/policy_content_adapter/tests/test_rollout.py \
      experiments/robotwin/policy_content_adapter/tests/test_evaluation_protocol.py \
      experiments/robotwin/policy_content_adapter/tests/test_robotwin_gpu_runtime.py
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_rollout \
    prepare \
    --formal-root "${FORMAL_OUTPUT_ROOT}" \
    --rollout-root "${ROLLOUT_OUTPUT_ROOT}" \
    --realization-bank "${REALIZATION_BANK}" \
    --gpu-ids "${GPU_IDS}" \
    --output-plan "${PLAN_PATH}"
  echo "CPU-only exact formal rollout preparation PASS: ${PLAN_PATH}"
}

audit_plan_for_runtime() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_rollout \
    audit-plan --plan "${PLAN_PATH}" --allow-existing-rollout-root > /dev/null
  local planned_gpus
  planned_gpus="$("${PYTHON_BIN}" -c \
    'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1],encoding="utf-8"))["parallelism"]["physical_gpu_ids"])))' \
    "${PLAN_PATH}")"
  if [[ "${planned_gpus}" != "${GPU_IDS}" ]]; then
    echo "GPU_IDS differs from immutable rollout plan: ${GPU_IDS} != ${planned_gpus}" >&2
    return 2
  fi
}

preflight_all_gpus() {
  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  mkdir -p "${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}"
  for gpu in "${gpu_array[@]}"; do
    local report="${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}/gpu_${gpu}.json"
    if [[ -e "${report}" ]]; then
      echo "Refusing to overwrite GPU preflight report: ${report}" >&2
      return 2
    fi
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
      preflight --gpu-id "${gpu}" > "${report}"
    "${PYTHON_BIN}" -c \
      'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); minimum=int(sys.argv[2]); free=int(p["memory_free_mib_at_preflight"]); assert free >= minimum, "GPU {} free VRAM {} MiB < required {} MiB".format(p["physical_gpu_index"],free,minimum)' \
      "${report}" "${MIN_FREE_GPU_MIB}"
  done
}

completed_manifest_for_cell() {
  local cell_root="$1"
  find "${cell_root}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/attempt_*/completed_rollouts.json' -print 2>/dev/null | sort
}

run_one_cell() {
  local seed="$1"
  local short="$2"
  local gpu="$3"
  local task="$4"
  local task_config="$5"
  local domain="$6"
  local checkpoint="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/checkpoint.pt"
  local dataset_stats="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/dataset_stats.json"
  local cell_root="${ROLLOUT_OUTPUT_ROOT}/cells/${task}/${domain}/seed_${seed}/${short}"
  local existing=()
  mapfile -t existing < <(completed_manifest_for_cell "${cell_root}")
  if [[ "${#existing[@]}" -gt 1 ]]; then
    echo "Cell has multiple completed attempts and is invalid: ${cell_root}" >&2
    return 2
  fi
  if [[ "${#existing[@]}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_formal_rollout \
      audit-cell --plan "${PLAN_PATH}" --manifest "${existing[0]}" > /dev/null
    echo "SKIP audited completed cell seed=${seed} control=${short} task=${task} domain=${domain}"
    return 0
  fi

  local attempt="${cell_root}/attempt_${RUN_STAMP}_pid${BASHPID}"
  local outer_log="${attempt}.worker.log"
  local status="${attempt}.status"
  if [[ -e "${attempt}" || -e "${outer_log}" || -e "${status}" ]]; then
    echo "Refusing to overwrite append-only attempt: ${attempt}" >&2
    return 2
  fi
  mkdir -p "${cell_root}"
  printf 'RUNNING seed=%s control=%s gpu=%s task=%s domain=%s episodes=100 utc=%s\n' \
    "${seed}" "${short}" "${gpu}" "${task}" "${domain}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  if "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.eval_robotwin_single \
      "ckpt=${checkpoint}" \
      "gpu_id=${gpu}" \
      'seed=47' \
      'mixed_precision=bf16' \
      "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
      "EVALUATION.task_name=${task}" \
      "EVALUATION.task_config=${task_config}" \
      'EVALUATION.eval_num_episodes=100' \
      "EVALUATION.output_dir=${attempt}" \
      "EVALUATION.dataset_stats_path=${dataset_stats}" \
      "+EVALUATION.formal_episode_realization_bank=${REALIZATION_BANK}" \
      'EVALUATION.instruction_type=unseen' \
      'EVALUATION.action_horizon=null' \
      'EVALUATION.replan_steps=24' \
      'EVALUATION.num_inference_steps=10' \
      'EVALUATION.sigma_shift=null' \
      'EVALUATION.text_cfg_scale=1.0' \
      'EVALUATION.rand_device=cpu' \
      'EVALUATION.tiled=false' \
      'EVALUATION.timing_enabled=false' \
      'EVALUATION.skip_get_obs_within_replan=true' \
      > "${outer_log}" 2>&1; then
    local manifest="${attempt}/completed_rollouts.json"
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_formal_rollout \
      audit-cell --plan "${PLAN_PATH}" --manifest "${manifest}" > /dev/null
    printf 'DONE seed=%s control=%s gpu=%s task=%s domain=%s episodes=100 utc=%s manifest=%s\n' \
      "${seed}" "${short}" "${gpu}" "${task}" "${domain}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${manifest}" > "${status}"
  else
    local code=$?
    printf 'FAILED seed=%s control=%s gpu=%s task=%s domain=%s exit_code=%s utc=%s\n' \
      "${seed}" "${short}" "${gpu}" "${task}" "${domain}" "${code}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    return "${code}"
  fi
}

run_wave() {
  local task="$1"
  local task_config="$2"
  local domain="$3"
  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  local seeds=(1 1 2 2 3 3)
  local shorts=(c1 c3 c1 c3 c1 c3)
  local pids=()
  local index
  for index in 0 1 2 3 4 5; do
    run_one_cell \
      "${seeds[${index}]}" "${shorts[${index}]}" "${gpu_array[${index}]}" \
      "${task}" "${task_config}" "${domain}" &
    pids+=("$!")
  done
  local failures=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  if [[ "${failures}" -ne 0 ]]; then
    echo "Wave ${task}/${domain} failed in ${failures} candidate cell(s)" >&2
    return 1
  fi
}

aggregate_rollout() {
  audit_plan_for_runtime
  local log="${ROLLOUT_OUTPUT_ROOT}/aggregate_${RUN_STAMP}.log"
  if [[ -e "${log}" ]]; then
    echo "Refusing to overwrite aggregate log: ${log}" >&2
    return 2
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_rollout \
    aggregate --plan "${PLAN_PATH}" > "${log}"
}

run_rollout() {
  if [[ "${CONFIRM_FORMAL_ROLLOUT}" != "YES" ]]; then
    echo "Formal final-test rollout requires CONFIRM_FORMAL_ROLLOUT=YES" >&2
    return 2
  fi
  if [[ ! -f "${PLAN_PATH}" ]]; then
    echo "Formal rollout plan is missing; run PHASE=prepare first: ${PLAN_PATH}" >&2
    return 2
  fi
  audit_plan_for_runtime
  mkdir -p "${ROLLOUT_OUTPUT_ROOT}"
  preflight_all_gpus
  local main_status="${ROLLOUT_OUTPUT_ROOT}/formal_rollout_${RUN_STAMP}.status"
  if [[ -e "${main_status}" ]]; then
    echo "Refusing to overwrite rollout status: ${main_status}" >&2
    return 2
  fi
  printf 'RUNNING waves=6 cells=36 gpu_ids=%s exact_replay=true utc=%s\n' \
    "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"

  local wave_spec
  for wave_spec in \
    'place_a2b_left demo_clean clean' \
    'place_a2b_left demo_randomized official_random' \
    'open_microwave demo_clean clean' \
    'open_microwave demo_randomized official_random' \
    'move_stapler_pad demo_clean clean' \
    'move_stapler_pad demo_randomized official_random'; do
    read -r wave_task wave_config wave_domain <<< "${wave_spec}"
    if ! run_wave "${wave_task}" "${wave_config}" "${wave_domain}"; then
      printf 'FAILED stage=wave task=%s domain=%s exact_replay=true utc=%s\n' \
        "${wave_task}" "${wave_domain}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${main_status}"
      return 1
    fi
  done

  printf 'RUNNING stage=aggregate cells=36 utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"
  if ! aggregate_rollout; then
    printf 'FAILED stage=aggregate cells=36 exact_replay=true utc=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"
    return 1
  fi
  printf 'DONE waves=6 cells=36 exact_replay=true aggregate=PASS utc=%s output=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ROLLOUT_OUTPUT_ROOT}" > "${main_status}"
}

case "${PHASE}" in
  prepare) prepare_rollout ;;
  rollout) run_rollout ;;
  aggregate) aggregate_rollout ;;
  all) prepare_rollout; run_rollout ;;
esac

echo "Release formal C1/C3 rollout phase=${PHASE} complete: ${ROLLOUT_OUTPUT_ROOT}"
