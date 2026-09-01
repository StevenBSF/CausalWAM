#!/usr/bin/env bash
set -euo pipefail

# Packed, one-shot execution of immutable stock cells 12..35.  MODE=check is
# signal-free and writes no formal experiment artifact.  MODE=run is gated,
# stops only PID 3759159, and relies on the EXIT trap for worker cleanup/reap
# before a positive-PID SIGCONT on every success/failure/signal path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
PLAN_PATH="${PLAN_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json}"
ASSET_REPAIR_CONTINUATION_PATH="${ASSET_REPAIR_CONTINUATION_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_asset_repair_continuation_v1.json}"
ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-${FORMAL_OUTPUT_ROOT}/online_rollouts_author_stock_seed42_v1}"
SCHEDULING_AMENDMENT_PATH="${SCHEDULING_AMENDMENT_PATH:-${FORMAL_OUTPUT_ROOT}/manifests/author_stock_seed23_packed_schedule_v1.json}"
MODE="${MODE:-check}"
CONFIRM_SEED23_PACKED="${CONFIRM_SEED23_PACKED:-NO}"
HELPER_TIMEOUT_SECONDS="${HELPER_TIMEOUT_SECONDS:-240}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-7200}"
DYNAMIC_MIN_FREE_GPU_MIB="${DYNAMIC_MIN_FREE_GPU_MIB:-30000}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-10}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAIN_RUNNER_PID=3759159

case "${MODE}" in
  check|run) ;;
  *) echo "MODE must be check or run" >&2; exit 2 ;;
esac
for value_name in HELPER_TIMEOUT_SECONDS CELL_TIMEOUT_SECONDS DYNAMIC_MIN_FREE_GPU_MIB START_STAGGER_SECONDS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "${value_name} must be a positive integer" >&2; exit 2; }
done
[[ "${MAIN_RUNNER_PID}" =~ ^[1-9][0-9]*$ ]] && (( MAIN_RUNNER_PID > 1 )) || {
  echo "Refusing non-positive/system main-runner PID" >&2; exit 2;
}
command -v timeout >/dev/null
command -v setsid >/dev/null
command -v nvidia-smi >/dev/null
[[ "${PYTHON_BIN}" == /root/anaconda3/envs/fastwam-robotwin-bw/bin/python ]] || {
  echo "Packed formal helper requires the stock Python executable" >&2; exit 2;
}

export DIFFSYNTH_MODEL_BASE_PATH="/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

TMP_DIR="$(mktemp -d /tmp/fastwam-seed23-packed.XXXXXX)"
PREFLIGHT_PATH="${TMP_DIR}/live_preflight.json"
PAIR_SPEC_PATH="${TMP_DIR}/pairs.tsv"
GPU_REPORT_PATHS=()
PARENT_STOPPED=0
EXPECTED_PARENT_START=""
EXPECTED_PARENT_CMDLINE_SHA256=""
MAIN_STATUS=""
JOURNAL=""
declare -a WORKER_PIDS=()
declare -A PID_ACTIVE=()
declare -A PID_GPU=()
declare -A PID_CELL=()
declare -A PID_ATTEMPT=()
declare -A PID_STATUS=()
declare -A GPU_RUNNING=([0]=0 [1]=0 [2]=0 [3]=0 [4]=0 [5]=0 [6]=0 [7]=0)
declare -A GPU_CAP=([0]=2 [1]=2 [2]=1 [3]=1 [4]=1 [5]=2 [6]=2 [7]=2)

helper() {
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_seed23_packed "$@"
}

stock_audit_cell() {
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    audit-cell "$@"
}

terminate_and_reap_owned_workers() {
  local pid deadline any_alive
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 && "${pid}" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM -- "${pid}" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${WORKER_PIDS[@]}"; do
      if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]] && kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    (( any_alive == 0 )) && break
    sleep 0.2
  done
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL -- "${pid}" 2>/dev/null || true
      fi
      wait "${pid}" 2>/dev/null || true
      PID_ACTIVE[${pid}]=0
    fi
  done
}

cleanup() {
  local original_status=$? cleanup_status=0 safe_to_continue=0 current_start current_cmdline_sha
  trap - EXIT HUP INT TERM
  terminate_and_reap_owned_workers
  if (( original_status != 0 )) && [[ -n "${MAIN_STATUS}" && -f "${MAIN_STATUS}" ]]; then
    printf 'FAILED partial_recovery=true exit_code=%s utc=%s\n' \
      "${original_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MAIN_STATUS}" || cleanup_status=1
    if [[ -n "${JOURNAL}" && -f "${JOURNAL}" ]]; then
      printf 'ABORT partial_recovery=true exit_code=%s utc=%s\n' \
        "${original_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}" || cleanup_status=1
    fi
  fi
  if (( PARENT_STOPPED == 1 )); then
    if helper validate-parent --path "${PREFLIGHT_PATH}" >/dev/null; then
      safe_to_continue=1
    else
      current_start="$(awk '{print $22}' "/proc/${MAIN_RUNNER_PID}/stat" 2>/dev/null || true)"
      current_cmdline_sha="$(sha256sum "/proc/${MAIN_RUNNER_PID}/cmdline" 2>/dev/null | awk '{print $1}')"
      if [[ -n "${EXPECTED_PARENT_START}" \
            && "${current_start}" == "${EXPECTED_PARENT_START}" \
            && "${current_cmdline_sha}" == "${EXPECTED_PARENT_CMDLINE_SHA256}" ]]; then
        echo "WARNING: Python CONT audit failed; exact shell-captured runner identity matches" >&2
        safe_to_continue=1
        cleanup_status=1
      fi
    fi
    if (( safe_to_continue == 1 )); then
      /bin/kill -CONT "${MAIN_RUNNER_PID}" || cleanup_status=1
    else
      echo "FATAL: exact main-runner identity could not be proven before SIGCONT" >&2
      cleanup_status=1
    fi
    PARENT_STOPPED=0
  fi
  if [[ -d "${TMP_DIR}" && "${TMP_DIR}" == /tmp/fastwam-seed23-packed.* ]]; then
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
if [[ "${MODE}" == run && "${CONFIRM_SEED23_PACKED}" != YES ]]; then
  echo "MODE=run requires CONFIRM_SEED23_PACKED=YES" >&2
  exit 2
fi

# Eight exact GPU/runtime reports are generated before touching the parent.
# Run-mode reports are permanent create-only evidence bound by the sidecar;
# check-mode reports remain temporary.
if [[ "${MODE}" == run ]]; then
  GPU_REPORT_ROOT="${ROLLOUT_OUTPUT_ROOT}/gpu_preflight/${RUN_STAMP}_seed23_packed"
  [[ ! -e "${GPU_REPORT_ROOT}" ]] || { echo "Refusing to overwrite ${GPU_REPORT_ROOT}" >&2; exit 2; }
  mkdir -p "${GPU_REPORT_ROOT}"
else
  GPU_REPORT_ROOT="${TMP_DIR}/gpu_preflight"
  mkdir "${GPU_REPORT_ROOT}"
fi
for gpu in 0 1 2 3 4 5 6 7; do
  report="${GPU_REPORT_ROOT}/gpu_${gpu}.json"
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${report}"
  if [[ "${gpu}" == 2 || "${gpu}" == 3 || "${gpu}" == 4 ]]; then minimum=30000; else minimum=60000; fi
  timeout --foreground --signal=TERM --kill-after=10s \
    "${HELPER_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["physical_gpu_index"]==int(sys.argv[2]); assert int(p["memory_free_mib_at_preflight"])>=int(sys.argv[3])' \
    "${report}" "${gpu}" "${minimum}"
  GPU_REPORT_PATHS+=("${report}")
done

# This expensive immutable audit is intentionally last before STOP so its live
# snapshot cannot go stale behind GPU preflights.
helper preflight \
  --plan "${PLAN_PATH}" \
  --continuation "${ASSET_REPAIR_CONTINUATION_PATH}" \
  --output "${PREFLIGHT_PATH}" >/dev/null
EXPECTED_PARENT_START="$(awk '{print $22}' "/proc/${MAIN_RUNNER_PID}/stat")"
EXPECTED_PARENT_CMDLINE_SHA256="$(sha256sum "/proc/${MAIN_RUNNER_PID}/cmdline" | awk '{print $1}')"

if [[ "${MODE}" == check ]]; then
  echo "PASS: seed2/3 packed window and eight GPU reports validate; no signal or formal artifact written."
  exit 0
fi

/bin/kill -STOP "${MAIN_RUNNER_PID}"
PARENT_STOPPED=1
for _ in $(seq 1 50); do
  parent_state="$("${PYTHON_BIN}" -c \
    'import pathlib,sys; text=pathlib.Path(sys.argv[1]).read_text(); print(text[text.rfind(")")+2:].split()[0])' \
    "/proc/${MAIN_RUNNER_PID}/stat" 2>/dev/null || true)"
  [[ "${parent_state}" == T ]] && break
  sleep 0.1
done
[[ "${parent_state:-}" == T ]] || { echo "Main runner did not stop within five seconds" >&2; exit 1; }

# Post-STOP authority: same exact parent and both direct seed1/C3 Open process
# trees, no seed2/3 evaluator, and all target roots still absent.
materialize_args=(
  materialize-after-stop
  --preflight "${PREFLIGHT_PATH}"
  --plan "${PLAN_PATH}"
  --continuation "${ASSET_REPAIR_CONTINUATION_PATH}"
  --output "${SCHEDULING_AMENDMENT_PATH}"
)
for report in "${GPU_REPORT_PATHS[@]}"; do
  materialize_args+=(--gpu-preflight-report "${report}")
done
helper "${materialize_args[@]}" >/dev/null
helper validate-parent --path "${SCHEDULING_AMENDMENT_PATH}" --require-stopped >/dev/null
helper emit-pairs --path "${SCHEDULING_AMENDMENT_PATH}" > "${PAIR_SPEC_PATH}"

mapfile -t CELL_ROWS < "${PAIR_SPEC_PATH}"
[[ "${#CELL_ROWS[@]}" -eq 24 ]] || { echo "Packed amendment did not emit 24 cells" >&2; exit 2; }
declare -a ROW_PAIR ROW_CELL ROW_PLANNED_GPU ROW_CHECKPOINT ROW_STATS ROW_TASK ROW_CONFIG ROW_DOMAIN ROW_ROOT ROW_AMENDMENT
for index in "${!CELL_ROWS[@]}"; do
  IFS=$'\t' read -r pair_index cell_index planned_gpu checkpoint stats task task_config domain cell_root stock_amendment extra \
    <<< "${CELL_ROWS[${index}]}"
  [[ -z "${extra:-}" ]] || { echo "Unexpected packed cell field" >&2; exit 2; }
  [[ "${cell_index}" =~ ^(1[2-9]|2[0-9]|3[0-5])$ ]] || { echo "Cell index outside 12..35" >&2; exit 2; }
  ROW_PAIR[${index}]="${pair_index}"
  ROW_CELL[${index}]="${cell_index}"
  ROW_PLANNED_GPU[${index}]="${planned_gpu}"
  ROW_CHECKPOINT[${index}]="${checkpoint}"
  ROW_STATS[${index}]="${stats}"
  ROW_TASK[${index}]="${task}"
  ROW_CONFIG[${index}]="${task_config}"
  ROW_DOMAIN[${index}]="${domain}"
  ROW_ROOT[${index}]="${cell_root}"
  ROW_AMENDMENT[${index}]="${stock_amendment}"
done

JOURNAL="${ROLLOUT_OUTPUT_ROOT}/seed23_packed_${RUN_STAMP}.journal"
MAIN_STATUS="${ROLLOUT_OUTPUT_ROOT}/seed23_packed_${RUN_STAMP}.status"
[[ ! -e "${JOURNAL}" && ! -e "${MAIN_STATUS}" ]] || { echo "Packed run record exists" >&2; exit 2; }
printf 'schedule=%s cells=12..35 gpu_total_eval_cap=2 gpu3_helper_cap=1 utc=%s\n' \
  "${SCHEDULING_AMENDMENT_PATH}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${JOURNAL}"
printf 'RUNNING cells=24 utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MAIN_STATUS}"

dynamic_free_mib() {
  local gpu="$1" value
  value="$(timeout --foreground --signal=TERM --kill-after=2s 15s \
    nvidia-smi --id "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)" || return 1
  value="${value//[[:space:]]/}"
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

launch_cell() {
  local row_index="$1" actual_gpu="$2" free cell_index cell_root attempt log status pid
  (( GPU_RUNNING[${actual_gpu}] < GPU_CAP[${actual_gpu}] )) || return 75
  free="$(dynamic_free_mib "${actual_gpu}")" || return 75
  (( free >= DYNAMIC_MIN_FREE_GPU_MIB )) || return 75
  cell_index="${ROW_CELL[${row_index}]}"
  cell_root="${ROW_ROOT[${row_index}]}"
  [[ ! -e "${cell_root}" ]] || { echo "Target root appeared before cell ${cell_index}: ${cell_root}" >&2; return 2; }
  mkdir -p "${cell_root}"
  attempt="${cell_root}/attempt_${RUN_STAMP}_packed_pid${BASHPID}"
  log="${attempt}.worker.log"
  status="${attempt}.status"
  [[ ! -e "${attempt}" && ! -e "${log}" && ! -e "${status}" ]] || return 2
  mkdir "${attempt}"
  printf 'RUNNING cell_index=%s pair_index=%s planned_gpu=%s actual_gpu=%s free_mib_before_launch=%s schedule=%s utc=%s\n' \
    "${cell_index}" "${ROW_PAIR[${row_index}]}" "${ROW_PLANNED_GPU[${row_index}]}" \
    "${actual_gpu}" "${free}" "${SCHEDULING_AMENDMENT_PATH}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  printf 'LAUNCH cell_index=%s pair_index=%s planned_gpu=%s actual_gpu=%s free_mib=%s utc=%s\n' \
    "${cell_index}" "${ROW_PAIR[${row_index}]}" "${ROW_PLANNED_GPU[${row_index}]}" \
    "${actual_gpu}" "${free}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}"
  setsid timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${ROW_CHECKPOINT[${row_index}]}" \
    "gpu_id=${actual_gpu}" \
    'seed=42' \
    'mixed_precision=bf16' \
    "EVALUATION.robotwin_root=${FASTWAM_ROOT}/third_party/RoboTwin" \
    "EVALUATION.task_name=${ROW_TASK[${row_index}]}" \
    "EVALUATION.task_config=${ROW_CONFIG[${row_index}]}" \
    'EVALUATION.eval_num_episodes=100' \
    "EVALUATION.output_dir=${attempt}" \
    "EVALUATION.dataset_stats_path=${ROW_STATS[${row_index}]}" \
    "+EVALUATION.stock_protocol_amendment=${ROW_AMENDMENT[${row_index}]}" \
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
  pid=$!
  WORKER_PIDS+=("${pid}")
  PID_ACTIVE[${pid}]=1
  PID_GPU[${pid}]="${actual_gpu}"
  PID_CELL[${pid}]="${cell_index}"
  PID_ATTEMPT[${pid}]="${attempt}"
  PID_STATUS[${pid}]="${status}"
  GPU_RUNNING[${actual_gpu}]=$((GPU_RUNNING[${actual_gpu}] + 1))
  sleep "${START_STAGGER_SECONDS}"
}

wait_and_audit_one() {
  local active=() pid finished code gpu cell attempt status manifest audit
  for pid in "${WORKER_PIDS[@]}"; do
    [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]] && active+=("${pid}")
  done
  [[ "${#active[@]}" -gt 0 ]] || return 75
  if wait -n -p finished "${active[@]}"; then code=0; else code=$?; fi
  [[ -n "${finished:-}" && "${PID_ACTIVE[${finished}]:-0}" == 1 ]] || {
    echo "wait -n returned an unknown packed worker" >&2; return 2;
  }
  PID_ACTIVE[${finished}]=0
  gpu="${PID_GPU[${finished}]}"
  cell="${PID_CELL[${finished}]}"
  attempt="${PID_ATTEMPT[${finished}]}"
  status="${PID_STATUS[${finished}]}"
  GPU_RUNNING[${gpu}]=$((GPU_RUNNING[${gpu}] - 1))
  if (( code != 0 )); then
    printf 'FAILED cell_index=%s actual_gpu=%s exit_code=%s utc=%s\n' \
      "${cell}" "${gpu}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
    printf 'FAILED cell_index=%s actual_gpu=%s exit_code=%s utc=%s\n' \
      "${cell}" "${gpu}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}"
    return "${code}"
  fi
  manifest="${attempt}/completed_rollouts.json"
  audit="${attempt}.audit.json"
  [[ -f "${manifest}" && ! -e "${audit}" ]] || return 2
  stock_audit_cell --plan "${PLAN_PATH}" --manifest "${manifest}" > "${audit}"
  printf 'DONE cell_index=%s actual_gpu=%s audit=%s utc=%s\n' \
    "${cell}" "${gpu}" "${audit}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  printf 'DONE cell_index=%s actual_gpu=%s audit=%s utc=%s\n' \
    "${cell}" "${gpu}" "${audit}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${JOURNAL}"
}

# The first eight emitted rows are the four C1/C3 Open pairs.  Spread exactly
# one Open cell to each GPU before filling any Place/Move slot.
for row_index in 0 1 2 3 4 5 6 7; do
  if ! launch_cell "${row_index}" "${row_index}"; then
    echo "Failed to launch required Open cell ${ROW_CELL[${row_index}]} on GPU ${row_index}" >&2
    exit 1
  fi
done

next_row=8
gpu_order=(0 1 5 6 7 2 3 4)
while (( next_row < 24 )); do
  launched=0
  for gpu in "${gpu_order[@]}"; do
    (( next_row < 24 )) || break
    if (( GPU_RUNNING[${gpu}] < GPU_CAP[${gpu}] )); then
      if launch_cell "${next_row}" "${gpu}"; then
        next_row=$((next_row + 1))
        launched=1
      else
        code=$?
        (( code == 75 )) || exit "${code}"
      fi
    fi
  done
  if (( launched == 0 )); then
    wait_and_audit_one || exit $?
  fi
done

while :; do
  active_count=0
  for pid in "${WORKER_PIDS[@]}"; do
    [[ "${PID_ACTIVE[${pid}]:-0}" == 1 ]] && active_count=$((active_count + 1))
  done
  (( active_count > 0 )) || break
  wait_and_audit_one || exit $?
done

printf 'DONE cells=24 audits=24 utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MAIN_STATUS}"
echo "PASS: packed seed2/3 cells 12..35 audited; EXIT trap will resume PID ${MAIN_RUNNER_PID}."
