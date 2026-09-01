#!/usr/bin/env bash
set -euo pipefail

# Fixed author-release C0 engineering deployment gate.
#
# This runs one episode for each of 3 tasks x {Clean, official Random}.  It is
# intentionally bound to a development_analysis seed bank and cannot emit
# formal evaluation records or open the final-test bank.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_ID="${GPU_ID:-0}"
SIMULATOR_SEED="${SIMULATOR_SEED:-29}"
EPISODES_PER_CELL=1
RUN_TESTS="${RUN_TESTS:-1}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/c0_dev_gate_${RUN_STAMP}}"

MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
OFFICIAL_DATASET_ROOT="${OFFICIAL_DATASET_ROOT:-/mnt/cpfs-E/baoshifeng/FastWAM/data/robotwin2.0/robotwin2.0}"
OFFICIAL_MANIFEST="${OFFICIAL_MANIFEST:-${SCRIPT_DIR}/configs/official_three_task_manifest.json}"
LINEAGE_MANIFEST="${LINEAGE_MANIFEST:-${SCRIPT_DIR}/configs/author_release_base_manifest.json}"
OFFICIAL_TEXT_CACHE="${OFFICIAL_TEXT_CACHE:-${FASTWAM_ROOT}/outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache}"
OFFICIAL_TEXT_CACHE_BINDING="${OFFICIAL_TEXT_CACHE_BINDING:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/official_text_cache_binding_manifest.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTWAM_ROOT}/third_party/RoboTwin}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one numeric physical GPU index; got ${GPU_ID}" >&2
  exit 2
fi
if [[ ! "${SIMULATOR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "SIMULATOR_SEED must be a non-negative integer; got ${SIMULATOR_SEED}" >&2
  exit 2
fi
if [[ "${RUN_TESTS}" != "0" && "${RUN_TESTS}" != "1" ]]; then
  echo "RUN_TESTS must be 0 or 1" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse C0 dev-gate output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi
for required in \
  "${PYTHON_BIN}" \
  "${BASE_CHECKPOINT}" \
  "${DATASET_STATS}" \
  "${OFFICIAL_MANIFEST}" \
  "${LINEAGE_MANIFEST}" \
  "${OFFICIAL_TEXT_CACHE_BINDING}" \
  "${ROBOTWIN_ROOT}/script/eval_policy.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required C0 dev-gate artifact is missing: ${required}" >&2
    exit 2
  fi
done
for required_dir in "${MODEL_BASE}" "${OFFICIAL_DATASET_ROOT}" "${OFFICIAL_TEXT_CACHE}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required C0 dev-gate directory is missing: ${required_dir}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/evaluation"
STATUS_FILE="${OUTPUT_ROOT}/c0_dev_gate.status"
trap 'code=$?; printf "FAILED exit_code=%s utc=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"; exit "${code}"' ERR

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${FASTWAM_ROOT}"

if [[ "${RUN_TESTS}" == "1" ]]; then
  printf 'RUNNING stage=cpu_tests utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
  "${PYTHON_BIN}" -m pytest -q \
    experiments/robotwin/policy_content_adapter/tests/test_c0_eval_transport.py \
    experiments/robotwin/policy_content_adapter/tests/test_rollout.py \
    experiments/robotwin/policy_content_adapter/tests/test_robotwin_gpu_runtime.py
fi

printf 'RUNNING stage=gpu_runtime_preflight utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
  preflight --gpu-id "${GPU_ID}" \
  > "${OUTPUT_ROOT}/gpu_runtime_preflight.json"

SEED_BANK="${OUTPUT_ROOT}/manifests/c0_development_analysis_seed_bank.json"
printf 'RUNNING stage=seed_bank utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.p_mode_selection \
  seed-bank \
  --purpose development_analysis \
  --simulator-seed "${SIMULATOR_SEED}" \
  --episodes-per-cell "${EPISODES_PER_CELL}" \
  --evaluator-source "${ROBOTWIN_ROOT}/script/eval_policy.py" \
  --output "${SEED_BANK}"
SEED_BANK_ID="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["simulator_seed_bank_id"])' "${SEED_BANK}")"

IDENTITY_AUDIT="${OUTPUT_ROOT}/c0_runtime_identity_audit.json"
TRANSPORT_CHECKPOINT="${OUTPUT_ROOT}/c0_author_release_dev_transport.pt"
printf 'RUNNING stage=bit_exact_and_transport utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.c0_eval_transport \
  --run-identity-audit \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --dataset-stats "${DATASET_STATS}" \
  --dataset-root "${OFFICIAL_DATASET_ROOT}" \
  --model-base-path "${MODEL_BASE}" \
  --official-manifest "${OFFICIAL_MANIFEST}" \
  --base-lineage-manifest "${LINEAGE_MANIFEST}" \
  --text-cache-dir "${OFFICIAL_TEXT_CACHE}" \
  --text-cache-binding-manifest "${OFFICIAL_TEXT_CACHE_BINDING}" \
  --identity-audit "${IDENTITY_AUDIT}" \
  --output "${TRANSPORT_CHECKPOINT}" \
  --rollout-protocol-id three_task_policy_online_v2 \
  --simulator-seed-bank-id "${SEED_BANK_ID}" \
  --simulator-seed-bank-manifest "${SEED_BANK}" \
  --episodes-per-task "${EPISODES_PER_CELL}" \
  --transport-seed 0 \
  --evaluation-stage deployment_gate \
  --device cuda \
  --model-dtype bf16

printf 'RUNNING stage=six_cell_online_deployment utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.eval_robotwin_single \
  "ckpt=${TRANSPORT_CHECKPOINT}" \
  "gpu_id=${GPU_ID}" \
  "seed=${SIMULATOR_SEED}" \
  "EVALUATION.robotwin_root=${ROBOTWIN_ROOT}" \
  'EVALUATION.task_name=[place_a2b_left,open_microwave,move_stapler_pad]' \
  'EVALUATION.task_config=both' \
  "EVALUATION.eval_num_episodes=${EPISODES_PER_CELL}" \
  "EVALUATION.output_dir=${OUTPUT_ROOT}/evaluation" \
  "EVALUATION.dataset_stats_path=${DATASET_STATS}"

printf 'RUNNING stage=final_audit utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.c0_dev_gate_audit \
  --identity-audit "${IDENTITY_AUDIT}" \
  --transport-checkpoint "${TRANSPORT_CHECKPOINT}" \
  --simulator-seed-bank-manifest "${SEED_BANK}" \
  --completed-rollouts "${OUTPUT_ROOT}/evaluation/completed_rollouts.json" \
  --output "${OUTPUT_ROOT}/strict_c0_dev_gate_audit.json"

printf 'DONE stage=complete utc=%s output=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${OUTPUT_ROOT}" > "${STATUS_FILE}"
echo "Author-release C0 development deployment gate PASS: ${OUTPUT_ROOT}"
echo "This six-episode development gate is not a formal Success Rate result."
