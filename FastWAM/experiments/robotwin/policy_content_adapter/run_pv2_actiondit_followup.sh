#!/usr/bin/env bash
set -euo pipefail

# P-v2 ActionDiT post-hoc mechanism pilot.  Safe default: CPU-only prepare.
# No seeds 2/3 or confirmatory seed59 work is started before pilot_gate PASS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1}"
PHASE="${PHASE:-prepare}"
SMOKE_GPU_ID="${SMOKE_GPU_ID:-0}"
PILOT_TRAIN_GPU_IDS="${PILOT_TRAIN_GPU_IDS:-0,1}"
PILOT_ROLLOUT_GPU_IDS="${PILOT_ROLLOUT_GPU_IDS:-0,1}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"
RUN_TESTS="${RUN_TESTS:-1}"

case "${PHASE}" in
  prepare|smoke|pilot_train|pilot_rollout|pilot_gate|pilot_all) ;;
  *) echo "PHASE must be prepare, smoke, pilot_train, pilot_rollout, pilot_gate, or pilot_all" >&2; exit 2 ;;
esac

IFS=',' read -r -a train_gpus <<< "${PILOT_TRAIN_GPU_IDS}"
IFS=',' read -r -a rollout_gpus <<< "${PILOT_ROLLOUT_GPU_IDS}"
if [[ "${#train_gpus[@]}" -ne 2 || "${#rollout_gpus[@]}" -ne 2 ]]; then
  echo "PILOT_TRAIN_GPU_IDS and PILOT_ROLLOUT_GPU_IDS require exactly two ids" >&2
  exit 2
fi
if [[ "${train_gpus[0]}" == "${train_gpus[1]}" || "${rollout_gpus[0]}" == "${rollout_gpus[1]}" ]]; then
  echo "P-v2 paired jobs require two distinct GPUs" >&2
  exit 2
fi

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

manifest="${OUTPUT_ROOT}/materialization_manifest.json"
eval100_amendment="${OUTPUT_ROOT}/manifests/eval100_user_amendment_v1.json"
eval100_rollout_root="${OUTPUT_ROOT}/pilot_rollouts_100ep_seed53_v1"
status_file="${OUTPUT_ROOT}/pv2_followup.status"

record_failure() {
  local code=$?
  if [[ -d "${OUTPUT_ROOT}" ]]; then
    printf 'FAILED phase=%s exit_code=%s utc=%s\n' \
      "${PHASE}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  fi
  exit "${code}"
}
trap record_failure ERR

require_gpu_confirmation() {
  if [[ "${CONFIRM_GPU_WORK}" != "YES" ]]; then
    echo "GPU work requires CONFIRM_GPU_WORK=YES" >&2
    return 2
  fi
}

gpu_preflight() {
  local gpu_id="$1"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu_id}" >/dev/null
}

prepare() {
  if [[ ! -e "${OUTPUT_ROOT}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.materialize_pv2_actiondit_followup \
      --output-root "${OUTPUT_ROOT}" \
      > "${OUTPUT_ROOT}.materialize.log" 2>&1
  fi
  if [[ ! -e "${OUTPUT_ROOT}/implementation_protocol_audit.json" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
      --materialization-manifest "${manifest}" \
      --stage materialization \
      --output-json "${OUTPUT_ROOT}/implementation_protocol_audit.json" \
      >> "${OUTPUT_ROOT}/implementation_protocol_audit.log"
  else
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
      --materialization-manifest "${manifest}" \
      --stage materialization >/dev/null
  fi
  if [[ ! -e "${eval100_amendment}" ]]; then
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_followup_eval100_amendment \
      materialize --experiment-root "${OUTPUT_ROOT}" \
      > "${OUTPUT_ROOT}/eval100_amendment_materialize.log" 2>&1
  else
    "${PYTHON_BIN}" -m \
      experiments.robotwin.policy_content_adapter.pv2_followup_eval100_amendment \
      validate --amendment "${eval100_amendment}" >/dev/null
  fi
  printf 'PREPARED gpu_training_started=false online_rollout_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

train_one() {
  local config="$1"
  local gpu_id="$2"
  local log="$3"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.train \
    --config "${config}" > "${log}" 2>&1
}

compact_action_gate() {
  local checkpoint="$1"
  local output_json="$2"
  local gpu_id="$3"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.rollout_policy \
    --checkpoint "${checkpoint}" \
    --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "${MODEL_BASE}" \
    --device cuda \
    --mixed-precision bf16 \
    --action-horizon 32 \
    --replan-steps 1 \
    --num-inference-steps 1 \
    --seed 0 \
    --output-json "${output_json}"
}

smoke() {
  require_gpu_confirmation
  prepare
  gpu_preflight "${SMOKE_GPU_ID}"
  local c1_config="${OUTPUT_ROOT}/smoke/configs/c1.yaml"
  local c3_config="${OUTPUT_ROOT}/smoke/configs/c3.yaml"
  local c1_root="${OUTPUT_ROOT}/smoke/runs/c1"
  local c3_root="${OUTPUT_ROOT}/smoke/runs/c3"
  if [[ -e "${c1_root}" || -e "${c3_root}" ]]; then
    echo "Refusing to overwrite P-v2 smoke outputs" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/smoke/logs"
  printf 'RUNNING stage=smoke gpu=%s utc=%s\n' \
    "${SMOKE_GPU_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  train_one "${c1_config}" "${SMOKE_GPU_ID}" "${OUTPUT_ROOT}/smoke/logs/c1_train.log"
  compact_action_gate \
    "${c1_root}/checkpoint.pt" "${c1_root}/pre_online_action_gate.json" "${SMOKE_GPU_ID}"
  train_one "${c3_config}" "${SMOKE_GPU_ID}" "${OUTPUT_ROOT}/smoke/logs/c3_train.log"
  compact_action_gate \
    "${c3_root}/checkpoint.pt" "${c3_root}/pre_online_action_gate.json" "${SMOKE_GPU_ID}"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
    --materialization-manifest "${manifest}" \
    --stage smoke \
    --output-json "${OUTPUT_ROOT}/smoke/strict_smoke_audit.json" \
    >> "${OUTPUT_ROOT}/smoke/strict_smoke_audit.log"
  printf 'DONE stage=smoke gpu=%s utc=%s\n' \
    "${SMOKE_GPU_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

pilot_train() {
  require_gpu_confirmation
  prepare
  if [[ ! -f "${OUTPUT_ROOT}/smoke/strict_smoke_audit.json" ]]; then
    echo "Strict P-v2 smoke audit is required before pilot training" >&2
    return 2
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
    --materialization-manifest "${manifest}" --stage smoke >/dev/null
  gpu_preflight "${train_gpus[0]}"
  gpu_preflight "${train_gpus[1]}"
  local c1_root="${OUTPUT_ROOT}/runs/seed_1/c1"
  local c3_root="${OUTPUT_ROOT}/runs/seed_1/c3"
  if [[ -e "${c1_root}" || -e "${c3_root}" ]]; then
    echo "Refusing to overwrite seed1 P-v2 pilot training" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/logs"
  printf 'RUNNING stage=pilot_train gpu_ids=%s utc=%s\n' \
    "${PILOT_TRAIN_GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  train_one "${OUTPUT_ROOT}/configs/seed_1/c1.yaml" "${train_gpus[0]}" \
    "${OUTPUT_ROOT}/logs/seed1_c1_train.log" &
  local c1_pid=$!
  train_one "${OUTPUT_ROOT}/configs/seed_1/c3.yaml" "${train_gpus[1]}" \
    "${OUTPUT_ROOT}/logs/seed1_c3_train.log" &
  local c3_pid=$!
  local failed=0
  wait "${c1_pid}" || failed=1
  wait "${c3_pid}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    printf 'FAILED stage=pilot_train utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    return 1
  fi
  compact_action_gate \
    "${c1_root}/checkpoint.pt" "${c1_root}/pre_online_action_gate.json" "${train_gpus[0]}" &
  c1_pid=$!
  compact_action_gate \
    "${c3_root}/checkpoint.pt" "${c3_root}/pre_online_action_gate.json" "${train_gpus[1]}" &
  c3_pid=$!
  failed=0
  wait "${c1_pid}" || failed=1
  wait "${c3_pid}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    printf 'FAILED stage=pilot_deployment_gate utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    return 1
  fi
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
    --materialization-manifest "${manifest}" \
    --stage pilot_posttrain \
    --output-json "${OUTPUT_ROOT}/pilot_posttrain_audit.json" \
    >> "${OUTPUT_ROOT}/pilot_posttrain_audit.log"
  printf 'DONE stage=pilot_train online_rollout_started=false utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

rollout_one() {
  local short="$1"
  local gpu_id="$2"
  gpu_preflight "${gpu_id}"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${OUTPUT_ROOT}/runs/seed_1/${short}/checkpoint.pt" \
    "gpu_id=${gpu_id}" \
    "seed=53" \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    'EVALUATION.task_name=[place_a2b_left,open_microwave,move_stapler_pad]' \
    EVALUATION.task_config=both \
    EVALUATION.eval_num_episodes=100 \
    "+EVALUATION.pv2_followup_eval_amendment=${eval100_amendment}" \
    "EVALUATION.output_dir=${eval100_rollout_root}/${short}" \
    > "${OUTPUT_ROOT}/logs/seed1_${short}_rollout_100ep.log" 2>&1
}

pilot_rollout() {
  require_gpu_confirmation
  prepare
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
    --materialization-manifest "${manifest}" --stage pilot_posttrain >/dev/null
  if [[ -e "${eval100_rollout_root}" ]]; then
    echo "Refusing to overwrite seed1 100-episode pilot rollouts" >&2
    return 2
  fi
  printf 'RUNNING stage=pilot_rollout simulator_seed=53 episodes_per_task_domain=100 gpu_ids=%s utc=%s\n' \
    "${PILOT_ROLLOUT_GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  rollout_one c1 "${rollout_gpus[0]}" & local c1_pid=$!
  rollout_one c3 "${rollout_gpus[1]}" & local c3_pid=$!
  local failed=0
  wait "${c1_pid}" || failed=1
  wait "${c3_pid}" || failed=1
  if [[ "${failed}" -ne 0 ]]; then
    printf 'FAILED stage=pilot_rollout utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
    return 1
  fi
  printf 'DONE stage=pilot_rollout utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
}

pilot_gate() {
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_audit \
    --materialization-manifest "${manifest}" \
    --stage pilot_gate \
    --evaluation-amendment "${eval100_amendment}" \
    --c1-rollout-manifest "${eval100_rollout_root}/c1/completed_rollouts.json" \
    --c3-rollout-manifest "${eval100_rollout_root}/c3/completed_rollouts.json" \
    --output-json "${OUTPUT_ROOT}/pilot_decision.json" \
    >> "${OUTPUT_ROOT}/pilot_decision.log"
  local decision
  decision="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_action"])' "${OUTPUT_ROOT}/pilot_decision.json")"
  "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.pv2_actiondit_followup_report \
    --materialization-manifest "${manifest}" \
    --pilot-decision "${OUTPUT_ROOT}/pilot_decision.json" \
    --output-dir "${OUTPUT_ROOT}/pilot_report" \
    >> "${OUTPUT_ROOT}/pilot_report.log"
  printf 'DONE stage=pilot_gate decision=%s utc=%s\n' \
    "${decision}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
  echo "Pilot decision: ${decision}"
}

case "${PHASE}" in
  prepare)
    if [[ "${RUN_TESTS}" == "1" ]]; then
      "${PYTHON_BIN}" -m pytest -q \
        experiments/robotwin/policy_content_adapter/tests/test_pv2_actiondit_followup.py \
        experiments/robotwin/policy_content_adapter/tests/test_configs.py \
        experiments/robotwin/policy_content_adapter/tests/test_train_protocol.py
    fi
    prepare
    ;;
  smoke) smoke ;;
  pilot_train) pilot_train ;;
  pilot_rollout) pilot_rollout ;;
  pilot_gate) pilot_gate ;;
  pilot_all)
    smoke
    pilot_train
    pilot_rollout
    pilot_gate
    ;;
esac

echo "P-v2 ActionDiT follow-up phase=${PHASE} complete: ${OUTPUT_ROOT}"
