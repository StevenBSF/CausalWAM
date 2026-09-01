#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1_retry2}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-NO}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"

case "${RESUME}" in
  YES|NO) ;;
  *) echo "RESUME must be YES or NO" >&2; exit 2 ;;
esac
IFS=',' read -r -a gpus <<< "${GPU_IDS}"
if [[ "${#gpus[@]}" -ne 8 || "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight distinct physical GPUs" >&2
  exit 2
fi

config="${OUTPUT_ROOT}/configs/seed_1/c3.yaml"
run_root="${OUTPUT_ROOT}/runs/seed_1/c3"
amendment="${OUTPUT_ROOT}/manifests/step6803_resume_amendment_v3.json"
mkdir -p "${OUTPUT_ROOT}/logs"
if [[ "${RESUME}" == "NO" && -e "${run_root}" ]]; then
  echo "Refusing to overwrite formal C3 target: ${run_root}" >&2
  exit 2
fi
if [[ "${RESUME}" == "YES" && ! -d "${run_root}" ]]; then
  echo "Resume target does not exist: ${run_root}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

if [[ "${RESUME}" == "YES" ]]; then
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.full5ep_resume_amendment \
    verify --amendment "${amendment}"
else
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
    validate-config --config "${config}"
fi
for gpu in "${gpus[@]}"; do
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${OUTPUT_ROOT}/logs/formal_preflight_gpu${gpu}.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  if [[ "${free_mib}" -lt "${MIN_FREE_GPU_MIB}" ]]; then
    echo "GPU ${gpu} has ${free_mib} MiB free; requires ${MIN_FREE_GPU_MIB}" >&2
    exit 2
  fi
done

resume_args=()
if [[ "${RESUME}" == "YES" ]]; then
  resume_args=(--resume latest)
fi
if [[ "${RESUME}" == "YES" ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
    -m experiments.robotwin.policy_content_adapter.full5ep_resume_amendment \
    resume --config "${config}" --amendment "${amendment}" "${resume_args[@]}" \
    > "${OUTPUT_ROOT}/logs/seed1_c3_resume_from6000.log" 2>&1
else
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
    -m experiments.robotwin.policy_content_adapter.pv2_actiondit_full5ep \
    train --config "${config}" \
    > "${OUTPUT_ROOT}/logs/seed1_c3_train.log" 2>&1
fi

CUDA_VISIBLE_DEVICES="${gpus[0]}" "${PYTHON_BIN}" -m \
  experiments.robotwin.policy_content_adapter.rollout_policy \
  --checkpoint "${run_root}/checkpoint.pt" \
  --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
  --model-base-path "${MODEL_BASE}" --device cuda --mixed-precision bf16 \
  --action-horizon 32 --replan-steps 1 --num-inference-steps 1 --seed 0 \
  --output-json "${run_root}/pre_online_action_gate.json" \
  > "${OUTPUT_ROOT}/logs/seed1_c3_action_gate.log" 2>&1

echo "PASS: seed1/C3 full5ep training and compact action gate completed."
