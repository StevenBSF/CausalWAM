#!/usr/bin/env bash
set -euo pipefail

# Sequential one-GPU, three-step C1/C3 engineering gate.  It does not select a
# P-mode, run simulator success evaluation, or start formal training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_ID="${GPU_ID:-0}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
REGIME="${REGIME:-p_v1}"
SEED="${SEED:-42}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/c1_c3_engineering_smoke_${RUN_STAMP}}"
RUN_TESTS="${RUN_TESTS:-1}"
RESUME_AFTER_C1_TRAIN="${RESUME_AFTER_C1_TRAIN:-0}"

if [[ "${GPU_ID}" == *,* ]]; then
  echo "Engineering smoke is locked to one visible GPU; got GPU_ID=${GPU_ID}" >&2
  exit 2
fi
if [[ "${RESUME_AFTER_C1_TRAIN}" != "0" && "${RESUME_AFTER_C1_TRAIN}" != "1" ]]; then
  echo "RESUME_AFTER_C1_TRAIN must be 0 or 1" >&2
  exit 2
fi
if [[ -e "${SMOKE_OUTPUT_ROOT}" && "${RESUME_AFTER_C1_TRAIN}" != "1" ]]; then
  echo "Refusing to reuse smoke output root: ${SMOKE_OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ ! -e "${SMOKE_OUTPUT_ROOT}" && "${RESUME_AFTER_C1_TRAIN}" == "1" ]]; then
  echo "C1-train resume requires an existing smoke output root: ${SMOKE_OUTPUT_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${FASTWAM_ROOT}"

if [[ "${RUN_TESTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pytest -q \
    experiments/robotwin/policy_content_adapter/tests/test_release_engineering_smoke.py \
    experiments/robotwin/policy_content_adapter/tests/test_configs.py \
    experiments/robotwin/policy_content_adapter/tests/test_train_protocol.py
fi

if [[ "${RESUME_AFTER_C1_TRAIN}" == "0" ]]; then
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.materialize_release_engineering_smoke \
    --output-root "${SMOKE_OUTPUT_ROOT}" \
    --regime "${REGIME}" \
    --seed "${SEED}" \
    --steps 3
fi

status_file="${SMOKE_OUTPUT_ROOT}/engineering_smoke.status"
trap 'code=$?; printf "FAILED exit_code=%s utc=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"; exit "${code}"' ERR
printf 'RUNNING stage=c1 utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"

c1_config="${SMOKE_OUTPUT_ROOT}/configs/c1_engineering_smoke.yaml"
c3_config="${SMOKE_OUTPUT_ROOT}/configs/c3_engineering_smoke.yaml"
c1_run="${SMOKE_OUTPUT_ROOT}/runs/c1"
c3_run="${SMOKE_OUTPUT_ROOT}/runs/c3"

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.config_audit \
  --config "${c1_config}" --config "${c3_config}" --require-ready

deploy_control() {
  local control="$1"
  local run_dir="$2"
  printf 'RUNNING stage=%s_deploy utc=%s\n' "${control}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${run_dir}/rollout_load_execute.json"
}

controls=(c1 c3)
if [[ "${RESUME_AFTER_C1_TRAIN}" == "1" ]]; then
  printf 'RUNNING stage=c1_train_resume_audit utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.release_engineering_smoke_resume \
    --output-root "${SMOKE_OUTPUT_ROOT}" \
    --output-json "${SMOKE_OUTPUT_ROOT}/c1_train_resume_audit.json"
  deploy_control c1 "${c1_run}"
  controls=(c3)
fi

for control in "${controls[@]}"; do
  config_var="${control}_config"
  run_var="${control}_run"
  config_path="${!config_var}"
  run_dir="${!run_var}"
  printf 'RUNNING stage=%s_train utc=%s\n' "${control}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.train \
    --config "${config_path}"

  deploy_control "${control}" "${run_dir}"
done

printf 'RUNNING stage=pair_audit utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
"${PYTHON_BIN}" -m \
  experiments.robotwin.policy_content_adapter.release_engineering_smoke_audit \
  --materialization-manifest "${SMOKE_OUTPUT_ROOT}/materialization_manifest.json" \
  --c1-run-dir "${c1_run}" \
  --c3-run-dir "${c3_run}" \
  --output-json "${SMOKE_OUTPUT_ROOT}/strict_pair_audit.json"

printf 'DONE stage=complete utc=%s output=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SMOKE_OUTPUT_ROOT}" > "${status_file}"
echo "Release-base C1/C3 engineering smoke PASS: ${SMOKE_OUTPUT_ROOT}"
echo "Formal training and online simulator success evaluation were not started."
