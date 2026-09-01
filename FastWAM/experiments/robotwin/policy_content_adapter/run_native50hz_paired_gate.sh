#!/usr/bin/env bash
set -euo pipefail

# Policy Protocol v2 gate.
#
# Default: PHASE=pilot collects 1 physical trajectory x C/R1/R2/R3 per task.
# Full:    PHASE=full requires immutable pilot PASS reports, then resumes each
#          raw task root to 50 physical trajectories (30/10/10 split).
# Both:    PHASE=all runs pilot, validates/exports it, and only then runs full.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
PHASE="${PHASE:-pilot}"
GPU_IDS="${GPU_IDS:-0,1,2}"
RAW_BASE="${RAW_BASE:-${FASTWAM_ROOT}/third_party/RoboTwin/data/policy_native50hz_three_task_rgb640x480_v1}"
OUTPUT_BASE="${OUTPUT_BASE:-${FASTWAM_ROOT}/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1}"
NVIDIA_VULKAN_ICD="${ROBOTWIN_NVIDIA_VULKAN_ICD:-/etc/vulkan/icd.d/nvidia_icd.json}"
NVIDIA_EGL_VENDOR="${ROBOTWIN_NVIDIA_EGL_VENDOR:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
POLICY_PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
TASKS=(place_a2b_left open_microwave move_stapler_pad)
if [[ ${#GPUS[@]} -lt ${#TASKS[@]} ]]; then
  echo "GPU_IDS must provide at least three comma-separated physical GPU IDs" >&2
  exit 2
fi
if [[ "${PHASE}" != pilot && "${PHASE}" != full && "${PHASE}" != all ]]; then
  echo "PHASE must be pilot, full, or all" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${RAW_BASE}/audits/pilot" "${RAW_BASE}/audits/full" "${OUTPUT_BASE}/logs"

preflight() {
  local vulkan_log="${OUTPUT_BASE}/logs/vulkan_preflight.log"
  local gpu_log="${OUTPUT_BASE}/logs/gpu_inventory.csv"
  for executable in vulkaninfo nvidia-smi ffmpeg ffprobe; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
      echo "Preflight failed: ${executable} is unavailable; no collection was started." >&2
      exit 3
    fi
  done
  if [[ ! -r "${NVIDIA_VULKAN_ICD}" || ! -r "${NVIDIA_EGL_VENDOR}" ]]; then
    echo "Preflight failed: NVIDIA Vulkan/EGL manifests are unreadable; no collection was started." >&2
    exit 3
  fi
  if ! env \
    VK_DRIVER_FILES="${NVIDIA_VULKAN_ICD}" \
    VK_ICD_FILENAMES="${NVIDIA_VULKAN_ICD}" \
    __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR}" \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    vulkaninfo --summary >"${vulkan_log}" 2>&1; then
    echo "Preflight failed: NVIDIA Vulkan cannot create an instance; see ${vulkan_log}." >&2
    exit 3
  fi
  if ! grep -Eiq 'deviceName[[:space:]]*=.*NVIDIA|GPU[0-9]+:.*NVIDIA' "${vulkan_log}"; then
    echo "Preflight failed: Vulkan did not enumerate NVIDIA; see ${vulkan_log}." >&2
    exit 3
  fi
  nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version \
    --format=csv,noheader >"${gpu_log}"
  for gpu in "${GPUS[@]}"; do
    if ! nvidia-smi --id "${gpu}" --query-gpu=pci.bus_id \
      --format=csv,noheader,nounits >/dev/null; then
      echo "Preflight failed: physical GPU ${gpu} is unavailable; no collection was started." >&2
      exit 3
    fi
  done
  if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep 'libsvtav1' >/dev/null; then
    echo "Preflight failed: ffmpeg lacks libsvtav1 required by the audited export." >&2
    exit 3
  fi
  if ! env PYTHONPATH="${POLICY_PYTHONPATH}" "${PYTHON_BIN}" -c \
    'from experiments.robotwin.policy_content_adapter.native50hz_paired import validate_collection_config; x = validate_collection_config(); assert x["camera_type"] == "Large_D435" and x["image_shape_hwc"] == [480, 640, 3]'; then
    echo "Preflight failed: native camera config is not audited Large_D435 640x480." >&2
    exit 3
  fi
  echo "Preflight PASS: Vulkan, GPUs, native Large_D435 640x480, ffmpeg/libsvtav1 and ffprobe are available."
}

require_pilot_pass() {
  for task in "${TASKS[@]}"; do
    report="${RAW_BASE}/audits/pilot/${task}.json"
    if [[ ! -f "${report}" ]]; then
      echo "Full collection blocked: missing pilot report ${report}" >&2
      exit 3
    fi
    "${PYTHON_BIN}" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"] == "PASS" and x["content_count"] == 1 and x["scene_episode_count"] == 4' "${report}"
  done
  pilot_export="${OUTPUT_BASE}/pilot_lerobot_v21"
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.validate_native50hz_paired \
    lerobot --root "${pilot_export}" --expected-contents 1 \
    --report "${RAW_BASE}/audits/pilot/lerobot.json"
}

run_collection_phase() {
  local phase="$1"
  local contents max_attempts audit_dir export_root
  if [[ "${phase}" == pilot ]]; then
    contents=1
    max_attempts=100
    audit_dir="${RAW_BASE}/audits/pilot"
    export_root="${OUTPUT_BASE}/pilot_lerobot_v21"
  else
    require_pilot_pass
    contents=50
    max_attempts=1000
    audit_dir="${RAW_BASE}/audits/full"
    export_root="${OUTPUT_BASE}/full_lerobot_v21"
  fi

  pids=()
  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPUS[$index]}"
    raw_root="${RAW_BASE}/${task}/raw"
    log_path="${OUTPUT_BASE}/logs/${phase}_${task}.log"
    echo "[${phase}] start task=${task} gpu=${gpu} raw=${raw_root}"
    env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      ROBOTWIN_PHYSICAL_GPU_INDEX="${gpu}" \
      VK_DRIVER_FILES="${NVIDIA_VULKAN_ICD}" \
      VK_ICD_FILENAMES="${NVIDIA_VULKAN_ICD}" \
      __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR}" \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      PYTHONPATH="${POLICY_PYTHONPATH}" \
      PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" \
      -m experiments.robotwin.policy_content_adapter.collect_native50hz_paired \
      --task "${task}" --num-contents "${contents}" \
      --max-attempts "${max_attempts}" --output-root "${raw_root}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "[${phase}] collection failed: ${TASKS[$index]} (see log)" >&2
      failed=1
    fi
  done
  if [[ ${failed} -ne 0 ]]; then
    exit 4
  fi

  for task in "${TASKS[@]}"; do
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.validate_native50hz_paired \
      raw --task "${task}" --root "${RAW_BASE}/${task}/raw" \
      --expected-contents "${contents}" --report "${audit_dir}/${task}.json"
  done

  if [[ -e "${export_root}" ]]; then
    echo "[${phase}] existing export found; validating without overwriting: ${export_root}"
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.validate_native50hz_paired \
      lerobot --root "${export_root}" --expected-contents "${contents}" \
      --report "${audit_dir}/lerobot.json"
  else
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.export_native50hz_paired \
      --place-a2b-left-root "${RAW_BASE}/place_a2b_left/raw" \
      --open-microwave-root "${RAW_BASE}/open_microwave/raw" \
      --move-stapler-pad-root "${RAW_BASE}/move_stapler_pad/raw" \
      --expected-contents "${contents}" --output-root "${export_root}"
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.validate_native50hz_paired \
      lerobot --root "${export_root}" --expected-contents "${contents}" \
      --report "${audit_dir}/lerobot.json"
  fi
  echo "[${phase}] PASS: ${export_root}"
}

cd "${FASTWAM_ROOT}"
preflight
if [[ "${PHASE}" == pilot ]]; then
  run_collection_phase pilot
elif [[ "${PHASE}" == full ]]; then
  run_collection_phase full
else
  run_collection_phase pilot
  run_collection_phase full
fi
