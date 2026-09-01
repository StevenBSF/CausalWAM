#!/usr/bin/env bash
set -euo pipefail

export DS_IGNORE_CUDA_DETECTION=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/motus/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/anaconda3/envs/motus/bin/torchrun}"
CONFIG_DIR="${CONFIG_DIR:?set CONFIG_DIR to the materialized formal M1/M3 directory}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-8}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT}/experiments/robotwin/policy_content_adapter/configs/deepspeed_zero1.json}"
CONTROLS="${CONTROLS:-m1 m3}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.config_audit \
  --m1 "${CONFIG_DIR}/m1.yaml" --m3 "${CONFIG_DIR}/m3.yaml"

for control in ${CONTROLS}; do
  config="${CONFIG_DIR}/${control}.yaml"
  output="$(${PYTHON_BIN} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${config}")"
  if [[ -f "${output}/training_summary.json" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP complete ${control} output=${output}"
    continue
  fi
  args=(
    --standalone --nproc_per_node="${EXPECTED_WORLD_SIZE}"
    -m experiments.robotwin.policy_content_adapter.train
    --config "${config}"
    --deepspeed "${DEEPSPEED_CONFIG}"
  )
  if [[ -d "${output}" ]]; then
    latest="$(find "${output}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${latest}" ]]; then
      echo "existing incomplete output has no resumable checkpoint: ${output}" >&2
      exit 1
    fi
    args+=(--resume "${latest}")
    echo "[$(date -u +%FT%TZ)] RESUME ${control} checkpoint=${latest}"
  else
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.config_audit \
      --m1 "${config}" --require-runnable
    echo "[$(date -u +%FT%TZ)] START ${control}"
  fi
  "${TORCHRUN_BIN}" "${args[@]}"
  echo "[$(date -u +%FT%TZ)] DONE ${control}"
done

M1_OUTPUT="$(${PYTHON_BIN} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${CONFIG_DIR}/m1.yaml")"
M3_OUTPUT="$(${PYTHON_BIN} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${CONFIG_DIR}/m3.yaml")"
if [[ -f "${M1_OUTPUT}/training_summary.json" && -f "${M3_OUTPUT}/training_summary.json" ]]; then
  if [[ ! -f "${CONFIG_DIR}/strict_training_pair_audit.json" ]]; then
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.training_audit \
      --m1-dir "${M1_OUTPUT}" --m3-dir "${M3_OUTPUT}" \
      --output "${CONFIG_DIR}/strict_training_pair_audit.json"
  fi
  echo "[$(date -u +%FT%TZ)] PASS formal M1/M3 pair"
fi
