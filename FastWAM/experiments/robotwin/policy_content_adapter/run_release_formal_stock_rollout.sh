#!/usr/bin/env bash
set -euo pipefail

# Main final-evaluation profile: author-stock RoboTwin seed-42 protocol.
# Each checkpoint is one wave; its six task/domain cells run on six GPUs in
# parallel.  Stock expert filtering is independent in every process, so only
# the starting seed is shared and episode-level pairing is explicitly denied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
PHASE="${PHASE:-prepare}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
AMENDMENT_PATH="${AMENDMENT_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_unpaired_v1.json}"
ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-${FORMAL_OUTPUT_ROOT}/online_rollouts_author_stock_seed42_v1}"
PLAN_PATH="${PLAN_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json}"
ASSET_REPAIR_CONTINUATION_PATH="${ASSET_REPAIR_CONTINUATION_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_asset_repair_continuation_v1.json}"
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

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

audit_asset_repair_continuation() {
  [[ -f "${ASSET_REPAIR_CONTINUATION_PATH}" ]] || {
    echo "Missing immutable asset-repair continuation: ${ASSET_REPAIR_CONTINUATION_PATH}" >&2
    return 2
  }
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.asset_repair_selection_confirmation \
    validate-continuation \
    --path "${ASSET_REPAIR_CONTINUATION_PATH}" \
    --plan "${PLAN_PATH}" \
    --amendment "${AMENDMENT_PATH}" \
    --rollout-root "${ROLLOUT_OUTPUT_ROOT}" > /dev/null
}

prepare_stock() {
  if [[ -e "${PLAN_PATH}" ]]; then
    echo "Refusing to overwrite author-stock rollout plan: ${PLAN_PATH}" >&2
    return 2
  fi
  if [[ -e "${ROLLOUT_OUTPUT_ROOT}" ]]; then
    echo "Refusing to reuse rollout root during plan creation: ${ROLLOUT_OUTPUT_ROOT}" >&2
    return 2
  fi
  if [[ "${RUN_TESTS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pytest -q \
      experiments/robotwin/policy_content_adapter/tests/test_release_stock_eval_protocol.py \
      experiments/robotwin/policy_content_adapter/tests/test_release_formal_stock_rollout.py \
      experiments/robotwin/policy_content_adapter/tests/test_asset_repair_selection_confirmation.py \
      experiments/robotwin/policy_content_adapter/tests/test_rollout.py \
      experiments/robotwin/policy_content_adapter/tests/test_evaluation_protocol.py \
      experiments/robotwin/policy_content_adapter/tests/test_robotwin_gpu_runtime.py
  fi
  if [[ ! -f "${AMENDMENT_PATH}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_stock_eval_protocol \
      materialize \
      --formal-root "${FORMAL_OUTPUT_ROOT}" \
      --robotwin-root "${ROBOTWIN_ROOT}" \
      --output "${AMENDMENT_PATH}"
  else
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_stock_eval_protocol \
      validate --path "${AMENDMENT_PATH}" > /dev/null
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    prepare \
    --formal-root "${FORMAL_OUTPUT_ROOT}" \
    --rollout-root "${ROLLOUT_OUTPUT_ROOT}" \
    --amendment "${AMENDMENT_PATH}" \
    --gpu-ids "${GPU_IDS}" \
    --output-plan "${PLAN_PATH}"
  audit_asset_repair_continuation
}

audit_plan_for_runtime() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    audit-plan --plan "${PLAN_PATH}" --allow-existing-rollout-root > /dev/null
  audit_asset_repair_continuation
  local planned_gpus
  planned_gpus="$("${PYTHON_BIN}" -c \
    'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1],encoding="utf-8"))["parallelism"]["physical_gpu_ids"])))' \
    "${PLAN_PATH}")"
  if [[ "${planned_gpus}" != "${GPU_IDS}" ]]; then
    echo "GPU_IDS differs from immutable stock plan: ${GPU_IDS} != ${planned_gpus}" >&2
    return 2
  fi
}

preflight_all_gpus() {
  IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
  mkdir -p "${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}"
  local gpu
  for gpu in "${gpu_array[@]}"; do
    local report="${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}/gpu_${gpu}.json"
    [[ ! -e "${report}" ]] || { echo "Refusing to overwrite ${report}" >&2; return 2; }
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
      preflight --gpu-id "${gpu}" > "${report}"
    "${PYTHON_BIN}" -c \
      'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); free=int(p["memory_free_mib_at_preflight"]); need=int(sys.argv[2]); assert free>=need, "GPU {}: {} MiB free < {} MiB".format(p["physical_gpu_index"],free,need)' \
      "${report}" "${MIN_FREE_GPU_MIB}"
  done
}

run_deployment_gate_one() {
  local seed="$1" short="$2" gpu="$3"
  local checkpoint="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/checkpoint.pt"
  local stats="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/dataset_stats.json"
  local gate_root="${ROLLOUT_OUTPUT_ROOT}/deployment_gates"
  local output="${gate_root}/seed_${seed}_${short}.json"
  local log="${gate_root}/seed_${seed}_${short}.log"
  if [[ -f "${output}" ]]; then
    "${PYTHON_BIN}" -c \
      'import json,pathlib,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); expected=str(pathlib.Path(sys.argv[2]).resolve()); assert p.get("status")=="PASS"; assert str(pathlib.Path(p.get("checkpoint","")).resolve())==expected; tasks=p.get("tasks"); assert isinstance(tasks,list) and len(tasks)==3; assert all(row.get("action_finite") is True and row.get("action_shape")==[14] for row in tasks)' \
      "${output}" "${checkpoint}"
    echo "SKIP audited deployment gate seed=${seed}/${short}"
    return 0
  fi
  [[ ! -e "${log}" ]] || {
    echo "Deployment gate log exists without PASS artifact seed=${seed}/${short}" >&2
    return 2
  }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${checkpoint}" \
    --dataset-stats "${stats}" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${output}" > "${log}" 2>&1
}

run_all_deployment_gates() {
  mkdir -p "${ROLLOUT_OUTPUT_ROOT}/deployment_gates"
  IFS=',' read -r -a gpus <<< "${GPU_IDS}"
  local seeds=(1 1 2 2 3 3)
  local shorts=(c1 c3 c1 c3 c1 c3)
  local pids=() index failures=0 pid
  for index in 0 1 2 3 4 5; do
    run_deployment_gate_one \
      "${seeds[${index}]}" "${shorts[${index}]}" "${gpus[${index}]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  [[ "${failures}" -eq 0 ]] || {
    echo "Formal checkpoint deployment gate failed in ${failures} worker(s)" >&2
    return 1
  }
}

completed_for_cell() {
  local cell_root="$1"
  find "${cell_root}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/attempt_*/completed_rollouts.json' -print 2>/dev/null | sort
}

run_one_cell() {
  local seed="$1" short="$2" gpu="$3" task="$4" task_config="$5" domain="$6"
  local checkpoint="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/checkpoint.pt"
  local stats="${FORMAL_OUTPUT_ROOT}/runs/seed_${seed}/${short}/dataset_stats.json"
  local cell_root="${ROLLOUT_OUTPUT_ROOT}/cells/seed_${seed}/${short}/${task}/${domain}"
  local completed=()
  mapfile -t completed < <(completed_for_cell "${cell_root}")
  if [[ "${#completed[@]}" -gt 1 ]]; then
    echo "Multiple completed attempts exist: ${cell_root}" >&2
    return 2
  fi
  if [[ "${#completed[@]}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
      audit-cell --plan "${PLAN_PATH}" --manifest "${completed[0]}" > /dev/null
    echo "SKIP audited stock cell seed=${seed} ${short} ${task}/${domain}"
    return 0
  fi
  local attempt="${cell_root}/attempt_${RUN_STAMP}_pid${BASHPID}"
  local log="${attempt}.worker.log" status="${attempt}.status"
  [[ ! -e "${attempt}" && ! -e "${log}" && ! -e "${status}" ]] || {
    echo "Refusing to overwrite stock attempt: ${attempt}" >&2; return 2;
  }
  mkdir -p "${cell_root}"
  printf 'RUNNING profile=author_stock_seed42_unpaired_v1 seed=%s control=%s gpu=%s task=%s domain=%s utc=%s\n' \
    "${seed}" "${short}" "${gpu}" "${task}" "${domain}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  if "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.eval_robotwin_single \
      "ckpt=${checkpoint}" \
      "gpu_id=${gpu}" \
      'seed=42' \
      'mixed_precision=bf16' \
      "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
      "EVALUATION.task_name=${task}" \
      "EVALUATION.task_config=${task_config}" \
      'EVALUATION.eval_num_episodes=100' \
      "EVALUATION.output_dir=${attempt}" \
      "EVALUATION.dataset_stats_path=${stats}" \
      "+EVALUATION.stock_protocol_amendment=${AMENDMENT_PATH}" \
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
      > "${log}" 2>&1; then
    local manifest="${attempt}/completed_rollouts.json"
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
      audit-cell --plan "${PLAN_PATH}" --manifest "${manifest}" > /dev/null
    printf 'DONE profile=author_stock_seed42_unpaired_v1 seed=%s control=%s task=%s domain=%s episode_pairing=not_claimed utc=%s\n' \
      "${seed}" "${short}" "${task}" "${domain}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  else
    local code=$?
    printf 'FAILED seed=%s control=%s task=%s domain=%s exit_code=%s utc=%s\n' \
      "${seed}" "${short}" "${task}" "${domain}" "${code}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    return "${code}"
  fi
}

run_checkpoint_wave() {
  local seed="$1" short="$2"
  IFS=',' read -r -a gpus <<< "${GPU_IDS}"
  local tasks=(place_a2b_left place_a2b_left open_microwave open_microwave move_stapler_pad move_stapler_pad)
  local configs=(demo_clean demo_randomized demo_clean demo_randomized demo_clean demo_randomized)
  local domains=(clean official_random clean official_random clean official_random)
  local pids=() index
  for index in 0 1 2 3 4 5; do
    run_one_cell "${seed}" "${short}" "${gpus[${index}]}" \
      "${tasks[${index}]}" "${configs[${index}]}" "${domains[${index}]}" &
    pids+=("$!")
  done
  local failures=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  [[ "${failures}" -eq 0 ]] || {
    echo "Checkpoint wave seed=${seed}/${short} failed in ${failures} cells" >&2
    return 1
  }
}

aggregate_stock() {
  audit_plan_for_runtime
  local log="${ROLLOUT_OUTPUT_ROOT}/aggregate_${RUN_STAMP}.log"
  [[ ! -e "${log}" ]] || { echo "Refusing to overwrite ${log}" >&2; return 2; }
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    aggregate --plan "${PLAN_PATH}" > "${log}"
}

run_stock() {
  [[ "${CONFIRM_FORMAL_ROLLOUT}" == "YES" ]] || {
    echo "Author-stock formal rollout requires CONFIRM_FORMAL_ROLLOUT=YES" >&2
    return 2
  }
  [[ -f "${PLAN_PATH}" ]] || { echo "Run PHASE=prepare first: ${PLAN_PATH}" >&2; return 2; }
  audit_plan_for_runtime
  mkdir -p "${ROLLOUT_OUTPUT_ROOT}"
  preflight_all_gpus
  run_all_deployment_gates
  local main_status="${ROLLOUT_OUTPUT_ROOT}/formal_stock_${RUN_STAMP}.status"
  [[ ! -e "${main_status}" ]] || { echo "Refusing to overwrite ${main_status}" >&2; return 2; }
  printf 'RUNNING profile=author_stock_seed42_unpaired_v1 checkpoint_waves=6 cells=36 episode_pairing=not_claimed utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"
  local spec wave_seed wave_short
  for spec in '1 c1' '1 c3' '2 c1' '2 c3' '3 c1' '3 c3'; do
    read -r wave_seed wave_short <<< "${spec}"
    if ! run_checkpoint_wave "${wave_seed}" "${wave_short}"; then
      printf 'FAILED stage=checkpoint_wave seed=%s control=%s episode_pairing=not_claimed utc=%s\n' \
        "${wave_seed}" "${wave_short}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"
      return 1
    fi
  done
  if ! aggregate_stock; then
    printf 'FAILED stage=aggregate episode_pairing=not_claimed utc=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${main_status}"
    return 1
  fi
  printf 'DONE profile=author_stock_seed42_unpaired_v1 cells=36 aggregate=PASS episode_pairing=not_claimed utc=%s output=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ROLLOUT_OUTPUT_ROOT}" > "${main_status}"
}

case "${PHASE}" in
  prepare) prepare_stock ;;
  rollout) run_stock ;;
  aggregate) aggregate_stock ;;
  all) prepare_stock; run_stock ;;
esac

echo "Author-stock formal rollout phase=${PHASE}: ${ROLLOUT_OUTPUT_ROOT}"
