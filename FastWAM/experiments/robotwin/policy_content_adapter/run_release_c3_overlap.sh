#!/usr/bin/env bash
set -euo pipefail

# One-shot seed-1/C3 scheduling overlap.  MODE=check is read-only with respect
# to formal experiment artifacts.  MODE=run requires an explicit confirmation,
# stops only PID 3759159 (never its process group), writes the immutable sidecar
# after the post-STOP proof, and resumes that exact process from the EXIT trap.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
PLAN_PATH="${PLAN_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json}"
ASSET_REPAIR_CONTINUATION_PATH="${ASSET_REPAIR_CONTINUATION_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_asset_repair_continuation_v1.json}"
ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-${FORMAL_OUTPUT_ROOT}/online_rollouts_author_stock_seed42_v1}"
SCHEDULING_AMENDMENT_PATH="${SCHEDULING_AMENDMENT_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed1_c3_overlap_schedule_v1.json}"
MODE="${MODE:-check}"
CONFIRM_C3_OVERLAP="${CONFIRM_C3_OVERLAP:-NO}"
HELPER_TIMEOUT_SECONDS="${HELPER_TIMEOUT_SECONDS:-180}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-7200}"
MIN_FREE_GPU_MIB=60000
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAIN_RUNNER_PID=3759159

case "${MODE}" in
  check|run) ;;
  *) echo "MODE must be check or run" >&2; exit 2 ;;
esac
[[ "${MAIN_RUNNER_PID}" =~ ^[1-9][0-9]*$ ]] && (( MAIN_RUNNER_PID > 1 )) || {
  echo "Refusing non-positive/system main-runner PID" >&2
  exit 2
}
[[ "${HELPER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "HELPER_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}
[[ "${CELL_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "CELL_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}
command -v timeout >/dev/null
command -v setsid >/dev/null

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

TMP_DIR="$(mktemp -d /tmp/fastwam-c3-overlap.XXXXXX)"
PREFLIGHT_PATH="${TMP_DIR}/live_preflight.json"
CELL_SPEC_PATH="${TMP_DIR}/cells.tsv"
GPU_REPORT_PATHS=()
EXPECTED_PARENT_START=""
EXPECTED_PARENT_CMDLINE_SHA256=""
PARENT_STOPPED=0
WORKER_PIDS=()
WORKER_ACTIVE=()

helper() {
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_c3_overlap "$@"
}

stock_audit_cell() {
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    audit-cell "$@"
}

terminate_and_reap_owned_workers() {
  local index pid deadline any_alive
  for index in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[${index}]}"
    if [[ "${WORKER_ACTIVE[${index}]:-0}" == 1 && "${pid}" =~ ^[1-9][0-9]*$ ]]; then
      # Every evaluator was launched with setsid, so this group contains only
      # the helper-owned evaluator and its RoboTwin descendants.
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM -- "${pid}" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    any_alive=0
    for index in "${!WORKER_PIDS[@]}"; do
      pid="${WORKER_PIDS[${index}]}"
      if [[ "${WORKER_ACTIVE[${index}]:-0}" == 1 ]] && kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    (( any_alive == 0 )) && break
    sleep 0.2
  done
  for index in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[${index}]}"
    if [[ "${WORKER_ACTIVE[${index}]:-0}" == 1 ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL -- "${pid}" 2>/dev/null || true
      fi
      wait "${pid}" 2>/dev/null || true
      WORKER_ACTIVE[${index}]=0
    fi
  done
}

cleanup() {
  local original_status=$? cleanup_status=0 safe_to_continue=0 current_start current_cmdline_sha
  trap - EXIT HUP INT TERM
  # Ordering is intentional and safety-critical: no parent CONT is possible
  # until all helper-owned evaluators have been terminated and reaped.
  terminate_and_reap_owned_workers
  if (( PARENT_STOPPED == 1 )); then
    # The tmp preflight is self-hashed and validates only /proc identity here;
    # it deliberately avoids CPFS/checkpoint rehashing on the liveness path.
    if helper validate-parent --path "${PREFLIGHT_PATH}" >/dev/null; then
      safe_to_continue=1
    else
      current_start="$(awk '{print $22}' "/proc/${MAIN_RUNNER_PID}/stat" 2>/dev/null || true)"
      current_cmdline_sha="$(sha256sum "/proc/${MAIN_RUNNER_PID}/cmdline" 2>/dev/null | awk '{print $1}')"
      if [[ -n "${EXPECTED_PARENT_START}" \
            && "${current_start}" == "${EXPECTED_PARENT_START}" \
            && "${current_cmdline_sha}" == "${EXPECTED_PARENT_CMDLINE_SHA256}" ]]; then
        echo "WARNING: Python CONT audit failed; exact shell-captured PID identity still matches" >&2
        safe_to_continue=1
        cleanup_status=1
      fi
    fi
    if (( safe_to_continue == 1 )); then
      # Positive PID only.  Never use a negative PID or process-group signal
      # for the stock runner: its two C1/Open children must remain independent.
      kill -CONT -- "${MAIN_RUNNER_PID}" || cleanup_status=1
    else
      echo "FATAL: exact main-runner identity could not be proven before SIGCONT" >&2
      cleanup_status=1
    fi
    PARENT_STOPPED=0
  fi
  if [[ -d "${TMP_DIR}" && "${TMP_DIR}" == /tmp/fastwam-c3-overlap.* ]]; then
    rm -rf -- "${TMP_DIR}"
  fi
  (( original_status == 0 && cleanup_status != 0 )) && original_status=${cleanup_status}
  exit "${original_status}"
}

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[[ ! -e "${SCHEDULING_AMENDMENT_PATH}" ]] || {
  echo "Refusing to reuse create-only scheduling amendment: ${SCHEDULING_AMENDMENT_PATH}" >&2
  exit 2
}

# First live proof: exact PID/start-time/cmdline, exactly two direct stock-shell
# children, and exactly one C1/Open evaluator under each child (cells 2 and 3).
helper preflight \
  --plan "${PLAN_PATH}" \
  --continuation "${ASSET_REPAIR_CONTINUATION_PATH}" \
  --output "${PREFLIGHT_PATH}" >/dev/null
EXPECTED_PARENT_START="$(awk '{print $22}' "/proc/${MAIN_RUNNER_PID}/stat")"
EXPECTED_PARENT_CMDLINE_SHA256="$(sha256sum "/proc/${MAIN_RUNNER_PID}/cmdline" | awk '{print $1}')"

if [[ "${MODE}" == run && "${CONFIRM_C3_OVERLAP}" != YES ]]; then
  echo "MODE=run requires CONFIRM_C3_OVERLAP=YES" >&2
  exit 2
fi

# GPU preflight is complete before STOP.  MODE=check keeps reports in /tmp;
# MODE=run writes a create-only experiment audit directory and the immutable
# scheduling sidecar binds all four report identities.
if [[ "${MODE}" == run ]]; then
  GPU_REPORT_ROOT="${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}_c3_overlap"
  [[ ! -e "${GPU_REPORT_ROOT}" ]] || { echo "Refusing to overwrite ${GPU_REPORT_ROOT}" >&2; exit 2; }
  mkdir -p "${GPU_REPORT_ROOT}"
else
  GPU_REPORT_ROOT="${TMP_DIR}/gpu_preflight"
  mkdir "${GPU_REPORT_ROOT}"
fi
for gpu in 0 1 5 6; do
  report="${GPU_REPORT_ROOT}/gpu_${gpu}.json"
  [[ ! -e "${report}" ]] || { echo "Refusing to overwrite ${report}" >&2; exit 2; }
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${report}"
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["physical_gpu_index"]==int(sys.argv[2]); assert int(p["memory_free_mib_at_preflight"])>=int(sys.argv[3])' \
    "${report}" "${gpu}" "${MIN_FREE_GPU_MIB}"
  GPU_REPORT_PATHS+=("${report}")
done

if [[ "${MODE}" == check ]]; then
  echo "PASS: C3 overlap window is currently valid; no signal sent and no formal artifact written."
  exit 0
fi

# This is deliberately a positive PID signal.  The direct C1/Open workers on
# GPUs 2 and 4 continue running while only their waiting parent is stopped.
kill -STOP -- "${MAIN_RUNNER_PID}"
PARENT_STOPPED=1

# Bound the signal-delivery wait.  The Python post-STOP proof below remains the
# authority and rechecks state T, the full identity, both children, no live C3,
# and all four still-absent target roots before creating the sidecar.
for _ in $(seq 1 50); do
  parent_state="$(/bin/ps -o stat= -p "${MAIN_RUNNER_PID}" 2>/dev/null | awk '{print substr($1,1,1)}')"
  [[ "${parent_state}" == T ]] && break
  sleep 0.1
done
[[ "${parent_state:-}" == T ]] || {
  echo "Main runner did not reach stopped state within 5 seconds" >&2
  exit 1
}

# Create-only and intentionally after STOP: this closes the validation-to-STOP
# race.  A changed parent, completed C1/Open child, active seed1/C3 evaluator,
# or newly-created target root aborts before any helper evaluator starts.
helper materialize-after-stop \
  --preflight "${PREFLIGHT_PATH}" \
  --plan "${PLAN_PATH}" \
  --continuation "${ASSET_REPAIR_CONTINUATION_PATH}" \
  --gpu-preflight-report "${GPU_REPORT_PATHS[0]}" \
  --gpu-preflight-report "${GPU_REPORT_PATHS[1]}" \
  --gpu-preflight-report "${GPU_REPORT_PATHS[2]}" \
  --gpu-preflight-report "${GPU_REPORT_PATHS[3]}" \
  --output "${SCHEDULING_AMENDMENT_PATH}" >/dev/null
helper validate-parent --path "${SCHEDULING_AMENDMENT_PATH}" --require-stopped >/dev/null
helper emit-cells --path "${SCHEDULING_AMENDMENT_PATH}" > "${CELL_SPEC_PATH}"

mapfile -t CELL_ROWS < "${CELL_SPEC_PATH}"
[[ "${#CELL_ROWS[@]}" -eq 4 ]] || { echo "Scheduling amendment did not emit four cells" >&2; exit 2; }
expected_indices=(6 7 10 11)
expected_gpus=(0 1 5 6)

ATTEMPTS=()
STATUSES=()
for index in 0 1 2 3; do
  IFS=$'\t' read -r cell_index gpu checkpoint stats task task_config domain cell_root stock_amendment extra \
    <<< "${CELL_ROWS[${index}]}"
  [[ -z "${extra:-}" ]] || { echo "Unexpected field in cell specification" >&2; exit 2; }
  [[ "${cell_index}" == "${expected_indices[${index}]}" ]] || { echo "Cell index order differs" >&2; exit 2; }
  [[ "${gpu}" == "${expected_gpus[${index}]}" ]] || { echo "Cell GPU differs" >&2; exit 2; }
  [[ ! -e "${cell_root}" ]] || { echo "Target cell root appeared before launch: ${cell_root}" >&2; exit 2; }
  mkdir -p "${cell_root}"
  attempt="${cell_root}/attempt_${RUN_STAMP}_overlap_pid${BASHPID}"
  log="${attempt}.worker.log"
  status="${attempt}.status"
  [[ ! -e "${attempt}" && ! -e "${log}" && ! -e "${status}" ]] || {
    echo "Refusing to overwrite overlap attempt: ${attempt}" >&2
    exit 2
  }
  mkdir "${attempt}"
  printf 'RUNNING schedule=%s cell_index=%s gpu=%s utc=%s\n' \
    "${SCHEDULING_AMENDMENT_PATH}" "${cell_index}" "${gpu}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  ATTEMPTS[${index}]="${attempt}"
  STATUSES[${index}]="${status}"

  # These arguments are byte-for-byte the author-stock evaluator settings in
  # run_release_formal_stock_rollout.sh; paths and GPUs come from cells
  # 6,7,10,11 of the revalidated immutable plan.
  setsid timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    'seed=42' \
    'mixed_precision=bf16' \
    "EVALUATION.robotwin_root=${FASTWAM_ROOT}/third_party/RoboTwin" \
    "EVALUATION.task_name=${task}" \
    "EVALUATION.task_config=${task_config}" \
    'EVALUATION.eval_num_episodes=100' \
    "EVALUATION.output_dir=${attempt}" \
    "EVALUATION.dataset_stats_path=${stats}" \
    "+EVALUATION.stock_protocol_amendment=${stock_amendment}" \
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
    > "${log}" 2>&1 &
  WORKER_PIDS[${index}]=$!
  WORKER_ACTIVE[${index}]=1
done

for index in 0 1 2 3; do
  pid="${WORKER_PIDS[${index}]}"
  attempt="${ATTEMPTS[${index}]}"
  if wait "${pid}"; then
    WORKER_ACTIVE[${index}]=0
  else
    code=$?
    WORKER_ACTIVE[${index}]=0
    printf 'FAILED cell_index=%s exit_code=%s utc=%s\n' \
      "${expected_indices[${index}]}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "${STATUSES[${index}]}"
    exit "${code}"
  fi
  manifest="${attempt}/completed_rollouts.json"
  audit="${attempt}.audit.json"
  [[ -f "${manifest}" && ! -e "${audit}" ]] || {
    echo "Completed manifest missing or audit path exists: ${attempt}" >&2
    exit 2
  }
  stock_audit_cell --plan "${PLAN_PATH}" --manifest "${manifest}" > "${audit}"
  printf 'DONE cell_index=%s audit=%s utc=%s\n' \
    "${expected_indices[${index}]}" "${audit}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUSES[${index}]}"
done

echo "PASS: audited seed-1/C3 overlap cells 6,7,10,11; EXIT trap will resume PID ${MAIN_RUNNER_PID}."
