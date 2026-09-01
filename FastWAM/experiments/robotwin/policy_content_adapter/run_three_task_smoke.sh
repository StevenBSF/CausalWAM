#!/usr/bin/env bash
set -euo pipefail

# Sequential, single-GPU P-v1/P-v2 smoke.  This script intentionally stops
# after three training steps plus one real no-SAPIEN action per task; it never
# starts controls or formal long training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_ID="${GPU_ID:-0}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
DATASET_STATS="${DATASET_STATS:-${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${FASTWAM_ROOT}"

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.config_audit \
  --config-dir experiments/robotwin/policy_content_adapter/configs

"${PYTHON_BIN}" -m pytest -q \
  experiments/robotwin/policy_content_adapter/tests

for regime in p_v1 p_v2; do
  run_dir="${SMOKE_OUTPUT_ROOT}/${regime}_smoke"
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing smoke directory: ${run_dir}" >&2
    exit 2
  fi

  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.train \
    --config "experiments/robotwin/policy_content_adapter/configs/${regime}_smoke.yaml" \
    --output-dir "${run_dir}"

  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --dataset-stats "${DATASET_STATS}" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${run_dir}/rollout_load_execute.json"

  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.smoke_audit \
    --run-dir "${run_dir}" \
    --regime "${regime}" \
    --output-json "${run_dir}/strict_smoke_audit.json"
done

echo "P-v1/P-v2 smoke complete: ${SMOKE_OUTPUT_ROOT}"
echo "Formal long training was not started."
