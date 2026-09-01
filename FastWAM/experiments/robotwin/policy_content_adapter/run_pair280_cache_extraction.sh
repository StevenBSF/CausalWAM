#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/fastwam_policy_artifacts/pair280_layer16_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-50000}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 8 || "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight distinct GPUs" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PAIR280_CPU_THREADS_PER_WORKER="${PAIR280_CPU_THREADS_PER_WORKER:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
cd "${FASTWAM_ROOT}"

input_audit="${OUTPUT_ROOT}/cache_input_audit.json"
state_bank="${OUTPUT_ROOT}/pair280_state_bank.json"
paired_text_cache="${OUTPUT_ROOT}/paired_text_cache"
paired_root="${FASTWAM_ROOT}/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21"
paired_manifest="${paired_root}/meta/policy_native_action_manifest.json"
paired_audit="${paired_root}/meta/policy_native_action_audit.json"
status_file="${OUTPUT_ROOT}/extraction.status"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/workers" "${OUTPUT_ROOT}/shards"

if [[ -f "${OUTPUT_ROOT}/cache_manifest.json" ]]; then
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_protocol \
    verify-cache --manifest "${OUTPUT_ROOT}/cache_manifest.json" --verify-shard-hashes
  printf 'DONE cache_manifest_verified=true utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit 0
fi

for gpu_id in "${gpus[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
  if [[ "${free_mib}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
    echo "GPU ${gpu_id} has ${free_mib} MiB free; requires ${MIN_FREE_GPU_MIB}" >&2
    exit 2
  fi
done

printf 'RUNNING workers=8 groups=25200 views=100800 utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
pids=()
for worker_index in 0 1 2 3 4 5 6 7; do
  gpu_id="${gpus[${worker_index}]}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.extract_pair280_cache \
    extract-worker \
    --input-audit "${input_audit}" \
    --output-root "${OUTPUT_ROOT}" \
    --paired-root "${paired_root}" \
    --paired-manifest "${paired_manifest}" \
    --paired-audit "${paired_audit}" \
    --state-bank "${state_bank}" \
    --paired-text-cache "${paired_text_cache}" \
    --model-base-path "${MODEL_BASE}" \
    --worker-index "${worker_index}" --worker-count 8 --device cuda \
    > "${OUTPUT_ROOT}/logs/extract_worker_${worker_index}.log" 2>&1 &
  pids+=("$!")
  sleep 3
done

failed=0
for worker_index in 0 1 2 3 4 5 6 7; do
  if ! wait "${pids[${worker_index}]}"; then
    echo "Pair-280 extraction worker ${worker_index} failed" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'FAILED worker_failure=true utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  exit 1
fi

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.extract_pair280_cache \
  merge --input-audit "${input_audit}" --output-root "${OUTPUT_ROOT}" \
  > "${OUTPUT_ROOT}/logs/merge.log" 2>&1
printf 'DONE workers=8 cache_manifest_verified=true utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
echo "Pair-280 cache complete: ${OUTPUT_ROOT}/cache_manifest.json"
