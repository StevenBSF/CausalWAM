#!/usr/bin/env bash
# Run the three-task paired-background collection behind a strict pilot gate.
#
# PHASE=all  (default): collect 1 content per task, strictly validate all three,
#                       then resume each dataset to 50 and validate all three.
# PHASE=full:            resume an already pilot-gated run directly to 50.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROBOTWIN_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ROBOTWIN_PY="${ROBOTWIN_PY:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
RUN_PHASE="${PHASE:-all}"
PILOT_MAX_ATTEMPTS="${ROBOTWIN_PILOT_MAX_ATTEMPTS:-100}"
FULL_MAX_ATTEMPTS="${ROBOTWIN_FULL_MAX_ATTEMPTS:-5000}"
NVIDIA_VULKAN_ICD="${ROBOTWIN_NVIDIA_VULKAN_ICD:-/etc/vulkan/icd.d/nvidia_icd.json}"
NVIDIA_EGL_VENDOR="${ROBOTWIN_NVIDIA_EGL_VENDOR:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

readonly SCRIPT_DIR ROBOTWIN_ROOT ROBOTWIN_PY RUN_PHASE
readonly PILOT_MAX_ATTEMPTS FULL_MAX_ATTEMPTS NVIDIA_VULKAN_ICD NVIDIA_EGL_VENDOR
readonly COLLECTOR="$SCRIPT_DIR/collect_paired_random_background.py"
readonly VALIDATOR="$SCRIPT_DIR/validate_paired_random_background.py"
readonly -a TASKS=(place_a2b_left open_microwave move_stapler_pad)
readonly -a GPUS=(0 1 2)

case "$RUN_PHASE" in
  all | full) ;;
  *)
    printf 'PHASE must be "all" or "full"; got %q\n' "$RUN_PHASE" >&2
    exit 2
    ;;
esac

if [[ ! "$PILOT_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROBOTWIN_PILOT_MAX_ATTEMPTS must be a positive integer; got %q\n' \
    "$PILOT_MAX_ATTEMPTS" >&2
  exit 2
fi
if [[ ! "$FULL_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROBOTWIN_FULL_MAX_ATTEMPTS must be a positive integer; got %q\n' \
    "$FULL_MAX_ATTEMPTS" >&2
  exit 2
fi
if [[ ! -x "$ROBOTWIN_PY" ]]; then
  printf 'Python interpreter is not executable: %s\n' "$ROBOTWIN_PY" >&2
  exit 2
fi
if [[ ! -f "$COLLECTOR" || ! -f "$VALIDATOR" ]]; then
  printf 'Collector or validator is missing under %s\n' "$SCRIPT_DIR" >&2
  exit 2
fi

readonly LOG_PARENT="${ROBOTWIN_GATE_LOG_ROOT:-$ROBOTWIN_ROOT/logs/paired_random_background_gate}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly LOG_DIR="$LOG_PARENT/$RUN_ID"
readonly GATE_LOG="$LOG_DIR/gate.log"
readonly MPL_BASE="/tmp/robotwin-paired-gate-$RUN_ID"
readonly TORCH_EXTENSIONS_BASE="/tmp/robotwin-torch-extensions-$RUN_ID"
readonly CUROBO_SOURCE="$ROBOTWIN_ROOT/envs/curobo/src"
readonly GATE_PYTHONPATH="$CUROBO_SOURCE:$SCRIPT_DIR:$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
readonly GATE_PATH="$(dirname -- "$ROBOTWIN_PY"):$PATH"
readonly GATE_CUDA_HOME="$(dirname -- "$(dirname -- "$ROBOTWIN_PY")")"
readonly GPU_INVENTORY="$LOG_DIR/gpu_inventory.csv"
readonly VULKAN_SUMMARY="$LOG_DIR/vulkan_summary.log"

mkdir -p -- "$LOG_PARENT"
# RUN_ID contains the shell PID. Refuse an unlikely collision instead of
# overwriting an earlier run's logs.
mkdir -- "$LOG_DIR"
touch -- "$GATE_LOG"

declare -a ACTIVE_PIDS=()

log_message() {
  local timestamp line
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf -v line '[%s] %s' "$timestamp" "$*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$GATE_LOG"
}

cleanup_children() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  ACTIVE_PIDS=()
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  if ((${#ACTIVE_PIDS[@]} > 0)); then
    cleanup_children
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

show_failure_tail() {
  local log_path=$1
  printf '%s\n' "Last 100 lines of $log_path:" >&2
  tail -n 100 -- "$log_path" >&2 || true
}

run_collect_phase() {
  local phase=$1
  local requested_contents=$2
  local max_attempts=$3
  local failed=0
  local index task gpu process_id exit_code log_path rc_path mpl_dir output_root torch_dir
  local -a process_ids=()
  local -a log_paths=()
  local -a rc_paths=()

  ACTIVE_PIDS=()
  log_message "Starting $phase collection for $requested_contents content(s) per task."

  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    output_root="$ROBOTWIN_ROOT/data/$task/paired_random_background"
    log_path="$LOG_DIR/$phase.collect.$task.log"
    rc_path="$LOG_DIR/$phase.collect.$task.rc"
    mpl_dir="$MPL_BASE/$phase/collect/$task"
    torch_dir="$TORCH_EXTENSIONS_BASE/$phase/$task"
    mkdir -p -- "$mpl_dir" "$torch_dir"

    env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      ROBOTWIN_PHYSICAL_GPU_INDEX="$gpu" \
      CUDA_HOME="$GATE_CUDA_HOME" \
      MPLCONFIGDIR="$mpl_dir" \
      PATH="$GATE_PATH" \
      PYTHONPATH="$GATE_PYTHONPATH" \
      PYTHONUNBUFFERED=1 \
      TORCH_EXTENSIONS_DIR="$torch_dir" \
      VK_DRIVER_FILES="$NVIDIA_VULKAN_ICD" \
      VK_ICD_FILENAMES="$NVIDIA_VULKAN_ICD" \
      __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR" \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      "$ROBOTWIN_PY" "$COLLECTOR" \
      --task "$task" \
      --num-contents "$requested_contents" \
      --output-root "$output_root" \
      --start-seed 0 \
      --max-attempts "$max_attempts" \
      >"$log_path" 2>&1 &

    process_id=$!
    process_ids[$index]="$process_id"
    log_paths[$index]="$log_path"
    rc_paths[$index]="$rc_path"
    ACTIVE_PIDS[$index]="$process_id"
    log_message "$phase collector task=$task physical_gpu=$gpu pid=$process_id log=$log_path"
  done

  # Wait for every task so all real child exit codes and logs are preserved.
  # The next phase is gated on the aggregate result below.
  for index in "${!TASKS[@]}"; do
    if wait "${process_ids[$index]}"; then
      exit_code=0
    else
      exit_code=$?
      failed=1
    fi
    ACTIVE_PIDS[$index]=""
    if ! printf '%s\n' "$exit_code" >"${rc_paths[$index]}"; then
      failed=1
      log_message "Could not save exit code for $phase collector ${TASKS[$index]}."
    fi
    log_message "$phase collector task=${TASKS[$index]} exit_code=$exit_code"
    if ((exit_code != 0)); then
      show_failure_tail "${log_paths[$index]}"
    fi
  done

  ACTIVE_PIDS=()
  return "$failed"
}

run_validate_phase() {
  local phase=$1
  local expected_contents=$2
  local failed=0
  local index task gpu process_id exit_code log_path rc_path mpl_dir output_root
  local -a process_ids=()
  local -a log_paths=()
  local -a rc_paths=()
  local -a validator_args=()

  ACTIVE_PIDS=()
  log_message "Starting strict $phase validation for $expected_contents content(s) per task."

  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    output_root="$ROBOTWIN_ROOT/data/$task/paired_random_background"
    log_path="$LOG_DIR/$phase.validate.$task.log"
    rc_path="$LOG_DIR/$phase.validate.$task.rc"
    mpl_dir="$MPL_BASE/$phase/validate/$task"
    mkdir -p -- "$mpl_dir"

    validator_args=(
      --root "$output_root"
      --task "$task"
      --expected-contents "$expected_contents"
    )
    if [[ "$phase" == pilot ]]; then
      # Keep pilot evidence separate from the canonical full-dataset outputs.
      validator_args+=(
        --report "$output_root/validation_report_pilot.json"
        --manifest "$output_root/valid_variants_pilot.jsonl"
        --split-manifest-dir "$output_root/split_manifests_pilot"
      )
    fi

    env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      ROBOTWIN_PHYSICAL_GPU_INDEX="$gpu" \
      MPLCONFIGDIR="$mpl_dir" \
      PYTHONPATH="$GATE_PYTHONPATH" \
      PYTHONUNBUFFERED=1 \
      "$ROBOTWIN_PY" "$VALIDATOR" "${validator_args[@]}" \
      >"$log_path" 2>&1 &

    process_id=$!
    process_ids[$index]="$process_id"
    log_paths[$index]="$log_path"
    rc_paths[$index]="$rc_path"
    ACTIVE_PIDS[$index]="$process_id"
    log_message "$phase validator task=$task physical_gpu=$gpu pid=$process_id log=$log_path"
  done

  for index in "${!TASKS[@]}"; do
    if wait "${process_ids[$index]}"; then
      exit_code=0
    else
      exit_code=$?
      failed=1
    fi
    ACTIVE_PIDS[$index]=""
    if ! printf '%s\n' "$exit_code" >"${rc_paths[$index]}"; then
      failed=1
      log_message "Could not save exit code for $phase validator ${TASKS[$index]}."
    fi
    log_message "$phase validator task=${TASKS[$index]} exit_code=$exit_code"
    if ((exit_code != 0)); then
      show_failure_tail "${log_paths[$index]}"
    else
      tail -n 1 -- "${log_paths[$index]}" || true
    fi
  done

  ACTIVE_PIDS=()
  return "$failed"
}

cd -- "$ROBOTWIN_ROOT"
log_message "RoboTwin root: $ROBOTWIN_ROOT"
log_message "Python: $ROBOTWIN_PY"
log_message "Mode: PHASE=$RUN_PHASE; logs: $LOG_DIR"
if [[ ! -r "$NVIDIA_VULKAN_ICD" ]]; then
  log_message "NVIDIA Vulkan ICD is unreadable: $NVIDIA_VULKAN_ICD; no collection was started."
  exit 1
fi
if [[ ! -r "$NVIDIA_EGL_VENDOR" ]]; then
  log_message "NVIDIA EGL vendor manifest is unreadable: $NVIDIA_EGL_VENDOR; no collection was started."
  exit 1
fi
if ! command -v vulkaninfo >/dev/null 2>&1; then
  log_message "vulkaninfo is unavailable; no collection was started."
  exit 1
fi
if ! env \
  VK_DRIVER_FILES="$NVIDIA_VULKAN_ICD" \
  VK_ICD_FILENAMES="$NVIDIA_VULKAN_ICD" \
  __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR" \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  vulkaninfo --summary >"$VULKAN_SUMMARY" 2>&1; then
  log_message "NVIDIA Vulkan preflight failed; no collection was started."
  show_failure_tail "$VULKAN_SUMMARY"
  exit 1
fi
if ! grep -Eiq 'deviceName[[:space:]]*=.*NVIDIA|GPU[0-9]+:.*NVIDIA' \
  "$VULKAN_SUMMARY"; then
  log_message "Vulkan did not enumerate an NVIDIA rendering device; no collection was started."
  show_failure_tail "$VULKAN_SUMMARY"
  exit 1
fi
log_message "NVIDIA Vulkan preflight: $VULKAN_SUMMARY"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log_message "nvidia-smi is unavailable; no collection was started."
  exit 1
fi
if ! nvidia-smi \
  --query-gpu=index,pci.bus_id,name,driver_version \
  --format=csv,noheader >"$GPU_INVENTORY"; then
  log_message "nvidia-smi preflight failed; no collection was started."
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if ! nvidia-smi --id "$gpu" --query-gpu=pci.bus_id --format=csv,noheader,nounits \
    >/dev/null; then
    log_message "Physical GPU $gpu is unavailable; no collection was started."
    exit 1
  fi
done
log_message "GPU inventory: $GPU_INVENTORY"

if [[ "$RUN_PHASE" == all ]]; then
  if ! run_collect_phase pilot 1 "$PILOT_MAX_ATTEMPTS"; then
    log_message "Pilot collection failed; full collection was NOT started."
    exit 1
  fi
  if ! run_validate_phase pilot 1; then
    log_message "Pilot strict validation failed; full collection was NOT started."
    exit 1
  fi
  log_message "All three pilot datasets passed strict validation; opening the full gate."
else
  log_message "PHASE=full explicitly skips the completed pilot gate and resumes toward 50."
fi

if ! run_collect_phase full 50 "$FULL_MAX_ATTEMPTS"; then
  log_message "Full collection failed or is incomplete; rerun with PHASE=full to resume."
  exit 1
fi
if ! run_validate_phase full 50; then
  log_message "Full strict validation failed. No dataset content was deleted or overwritten by this script."
  exit 1
fi

log_message "SUCCESS: every task is strictly valid with 50 Clean + 150 Random variants."
log_message "Canonical reports and manifests are under each task's paired_random_background root."
