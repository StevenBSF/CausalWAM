#!/usr/bin/env bash
set -euo pipefail

# No-stop Seed-3 acceleration.  The original runner continues Seed 1/2 while
# these twelve planned Seed-3 cells run in their original append-only roots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
PLAN_PATH="${PLAN_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json}"
CONTINUATION_PATH="${CONTINUATION_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_asset_repair_continuation_v1.json}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${FORMAL_OUTPUT_ROOT}/online_rollouts_author_stock_seed42_v1}"
STOCK_AMENDMENT="${STOCK_AMENDMENT:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_unpaired_v1.json}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed3_parallel_schedule_v1.json}"
COMPLETION_PATH="${COMPLETION_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed3_parallel_completion_v1.json}"
MODE="${MODE:-check}"
CONFIRM_SEED3_PARALLEL="${CONFIRM_SEED3_PARALLEL:-NO}"
HELPER_TIMEOUT_SECONDS="${HELPER_TIMEOUT_SECONDS:-240}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-14400}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-10}"
DYNAMIC_MIN_FREE_GPU_MIB="${DYNAMIC_MIN_FREE_GPU_MIB:-30000}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${MODE}" in check|run) ;; *) echo "MODE must be check or run" >&2; exit 2 ;; esac
for name in HELPER_TIMEOUT_SECONDS CELL_TIMEOUT_SECONDS START_STAGGER_SECONDS DYNAMIC_MIN_FREE_GPU_MIB; do
  value="${!name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be positive" >&2; exit 2; }
done
command -v timeout >/dev/null
command -v setsid >/dev/null
command -v nvidia-smi >/dev/null

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

TMP_DIR="$(mktemp -d /tmp/fastwam-seed3-parallel.XXXXXX)"
ROWS_PATH="${TMP_DIR}/rows.tsv"
GPU_REPORTS=()
WORKER_PIDS=()
declare -A PID_ACTIVE=() PID_GPU=() PID_CELL=() PID_ATTEMPT=() PID_STATUS=()

terminate_workers() {
  local pid deadline any_alive
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${WORKER_PIDS[@]}"; do
      if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]] && kill -0 "${pid}" 2>/dev/null; then any_alive=1; fi
    done
    (( any_alive == 0 )) && break
    sleep 0.2
  done
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      fi
      wait "${pid}" 2>/dev/null || true
      PID_ACTIVE[${pid}]=0
    fi
  done
}

cleanup() {
  local code=$?
  trap - EXIT HUP INT TERM
  terminate_workers
  if [[ -d "${TMP_DIR}" && "${TMP_DIR}" == /tmp/fastwam-seed3-parallel.* ]]; then rm -rf -- "${TMP_DIR}"; fi
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[[ ! -e "${SCHEDULE_PATH}" && ! -e "${COMPLETION_PATH}" ]] || {
  echo "Refusing to overwrite Seed-3 schedule/completion" >&2; exit 2;
}
if [[ "${MODE}" == run && "${CONFIRM_SEED3_PARALLEL}" != YES ]]; then
  echo "MODE=run requires CONFIRM_SEED3_PARALLEL=YES" >&2; exit 2
fi

if [[ "${MODE}" == run ]]; then
  GPU_REPORT_ROOT="${ROLLOUT_ROOT}/gpu_preflight/${RUN_STAMP}_seed3_parallel"
  [[ ! -e "${GPU_REPORT_ROOT}" ]] || { echo "GPU report root exists" >&2; exit 2; }
  mkdir -p "${GPU_REPORT_ROOT}"
  schedule_output="${SCHEDULE_PATH}"
else
  GPU_REPORT_ROOT="${TMP_DIR}/gpu_preflight"
  mkdir "${GPU_REPORT_ROOT}"
  schedule_output="${TMP_DIR}/schedule.json"
fi

for gpu in 0 1 2 3 4 5 6 7; do
  report="${GPU_REPORT_ROOT}/gpu_${gpu}.json"
  timeout --foreground --signal=TERM --kill-after=10s "${HELPER_TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${report}"
  minimum=30000
  if [[ "${gpu}" == 0 || "${gpu}" == 1 || "${gpu}" == 5 || "${gpu}" == 6 || "${gpu}" == 7 ]]; then minimum=60000; fi
  "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["physical_gpu_index"]==int(sys.argv[2]); assert int(p["memory_free_mib_at_preflight"])>=int(sys.argv[3])' \
    "${report}" "${gpu}" "${minimum}"
  GPU_REPORTS+=("${report}")
done

materialize=(
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_seed3_parallel
  materialize --plan "${PLAN_PATH}" --continuation "${CONTINUATION_PATH}"
  --output "${schedule_output}"
)
for report in "${GPU_REPORTS[@]}"; do materialize+=(--gpu-preflight-report "${report}"); done
timeout --foreground --signal=TERM --kill-after=10s "${HELPER_TIMEOUT_SECONDS}s" \
  "${materialize[@]}" >/dev/null
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_seed3_parallel \
  emit --path "${schedule_output}" > "${ROWS_PATH}"

if [[ "${MODE}" == check ]]; then
  echo "PASS: Seed-3 12-cell parallel schedule and eight GPUs validate; no rollout started."
  exit 0
fi

mapfile -t ROWS < "${ROWS_PATH}"
[[ "${#ROWS[@]}" -eq 12 ]] || { echo "Expected 12 Seed-3 cells" >&2; exit 2; }
JOURNAL="${ROLLOUT_ROOT}/seed3_parallel_${RUN_STAMP}.journal"
MAIN_STATUS="${ROLLOUT_ROOT}/seed3_parallel_${RUN_STAMP}.status"
[[ ! -e "${JOURNAL}" && ! -e "${MAIN_STATUS}" ]] || { echo "Run record exists" >&2; exit 2; }
printf 'RUNNING cells=12 schedule=%s utc=%s\n' "${SCHEDULE_PATH}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MAIN_STATUS}"

free_mib() {
  local value
  value="$(timeout --foreground --signal=TERM --kill-after=2s 15s \
    nvidia-smi --id "$1" --query-gpu=memory.free --format=csv,noheader,nounits)" || return 1
  value="${value//[[:space:]]/}"
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r cell gpu checkpoint stats task config domain root extra <<< "${row}"
  [[ -z "${extra:-}" && ! -e "${root}" ]] || { echo "Seed-3 cell root appeared: ${root}" >&2; exit 2; }
  free="$(free_mib "${gpu}")" || { echo "Cannot read GPU ${gpu} memory" >&2; exit 1; }
  (( free >= DYNAMIC_MIN_FREE_GPU_MIB )) || { echo "GPU ${gpu} has only ${free} MiB free" >&2; exit 1; }
  mkdir -p "${root}"
  attempt="${root}/attempt_${RUN_STAMP}_seed3parallel_pid${BASHPID}"
  log="${attempt}.worker.log"
  status="${attempt}.status"
  mkdir "${attempt}"
  printf 'RUNNING cell=%s actual_gpu=%s free_mib=%s schedule=%s utc=%s\n' \
    "${cell}" "${gpu}" "${free}" "${SCHEDULE_PATH}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  printf 'LAUNCH cell=%s gpu=%s free_mib=%s utc=%s\n' "${cell}" "${gpu}" "${free}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}"
  setsid timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${checkpoint}" "gpu_id=${gpu}" seed=42 mixed_precision=bf16 \
    "EVALUATION.robotwin_root=${FASTWAM_ROOT}/third_party/RoboTwin" \
    "EVALUATION.task_name=${task}" "EVALUATION.task_config=${config}" \
    EVALUATION.eval_num_episodes=100 "EVALUATION.output_dir=${attempt}" \
    "EVALUATION.dataset_stats_path=${stats}" \
    "+EVALUATION.stock_protocol_amendment=${STOCK_AMENDMENT}" \
    EVALUATION.instruction_type=unseen EVALUATION.action_horizon=null \
    EVALUATION.replan_steps=24 EVALUATION.num_inference_steps=10 \
    EVALUATION.sigma_shift=null EVALUATION.text_cfg_scale=1.0 \
    EVALUATION.rand_device=cpu EVALUATION.tiled=false \
    EVALUATION.timing_enabled=false EVALUATION.skip_get_obs_within_replan=true \
    > "${log}" 2>&1 &
  pid=$!
  WORKER_PIDS+=("${pid}")
  PID_ACTIVE[${pid}]=1; PID_GPU[${pid}]="${gpu}"; PID_CELL[${pid}]="${cell}"
  PID_ATTEMPT[${pid}]="${attempt}"; PID_STATUS[${pid}]="${status}"
  sleep "${START_STAGGER_SECONDS}"
done

remaining=12
while (( remaining > 0 )); do
  active=()
  for pid in "${WORKER_PIDS[@]}"; do [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]] && active+=("${pid}"); done
  [[ "${#active[@]}" -gt 0 ]] || { echo "No active worker with ${remaining} remaining" >&2; exit 2; }
  finished=""
  if wait -n -p finished "${active[@]}"; then code=0; else code=$?; fi
  [[ -n "${finished}" && "${PID_ACTIVE[${finished}]:-0}" == 1 ]] || { echo "Unknown finished worker" >&2; exit 2; }
  PID_ACTIVE[${finished}]=0
  cell="${PID_CELL[${finished}]}"; gpu="${PID_GPU[${finished}]}"; attempt="${PID_ATTEMPT[${finished}]}"; status="${PID_STATUS[${finished}]}"
  if (( code != 0 )); then
    printf 'FAILED cell=%s gpu=%s exit=%s utc=%s\n' "${cell}" "${gpu}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    exit "${code}"
  fi
  manifest="${attempt}/completed_rollouts.json"; audit="${attempt}.audit.json"
  [[ -f "${manifest}" && ! -e "${audit}" ]] || { echo "Missing manifest for cell ${cell}" >&2; exit 2; }
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    audit-cell --plan "${PLAN_PATH}" --manifest "${manifest}" > "${audit}"
  printf 'DONE cell=%s gpu=%s audit=%s utc=%s\n' "${cell}" "${gpu}" "${audit}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  printf 'DONE cell=%s gpu=%s utc=%s\n' "${cell}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}"
  remaining=$((remaining - 1))
done

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_seed3_parallel \
  complete --schedule "${SCHEDULE_PATH}" --output "${COMPLETION_PATH}" >/dev/null
printf 'DONE cells=12 completion=%s utc=%s\n' "${COMPLETION_PATH}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MAIN_STATUS}"
echo "PASS: Seed-3 twelve-cell acceleration completed and audited."
