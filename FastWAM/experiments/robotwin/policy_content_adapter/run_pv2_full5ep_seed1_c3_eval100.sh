#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
ROOT="${ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1_retry2}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_ID="${GPU_ID:-0}"
PHASE="${PHASE:-all}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"

case "${PHASE}" in
  prepare|rollout|summarize|all) ;;
  *) echo "PHASE must be prepare, rollout, summarize, or all" >&2; exit 2 ;;
esac

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

amendment="${ROOT}/manifests/seed1_c3_dev_seed53_eval100_v2.json"
rollout_root="${ROOT}/online_rollouts_dev_seed53_v1/c3"
status_file="${ROOT}/seed1_c3_eval100.status"

prepare() {
  if [[ ! -f "${amendment}" ]]; then
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_full5ep_eval100 \
      materialize --root "${ROOT}" --output "${amendment}"
  else
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_full5ep_eval100 \
      validate --amendment "${amendment}"
  fi
}

rollout() {
  [[ "${CONFIRM_GPU_WORK}" == "YES" ]] || {
    echo "GPU evaluation requires CONFIRM_GPU_WORK=YES" >&2
    return 2
  }
  prepare
  [[ ! -e "${rollout_root}" ]] || {
    echo "Refusing to overwrite rollout root: ${rollout_root}" >&2
    return 2
  }
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${GPU_ID}" > "${ROOT}/logs/seed1_c3_eval100_gpu${GPU_ID}_preflight.log"
  printf 'RUNNING checkpoint_step=18215 seed=53 episodes=600 gpu=%s utc=%s\n' \
    "${GPU_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_pv2_full5ep \
    "ckpt=${ROOT}/runs/seed_1/c3/checkpoint.pt" \
    "gpu_id=${GPU_ID}" seed=53 \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    'EVALUATION.task_name=[place_a2b_left,open_microwave,move_stapler_pad]' \
    EVALUATION.task_config=both EVALUATION.eval_num_episodes=100 \
    "+EVALUATION.pv2_followup_eval_amendment=${amendment}" \
    "EVALUATION.output_dir=${rollout_root}" \
    > "${ROOT}/logs/seed1_c3_full5ep_eval100.log" 2>&1
  summarize
}

summarize() {
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_full5ep_eval100 \
    summarize --amendment "${amendment}" --rollout-root "${rollout_root}" \
    > "${ROOT}/logs/seed1_c3_full5ep_eval100_summary.log" 2>&1
  printf 'DONE checkpoint_step=18215 seed=53 episodes=600 summary=PASS utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
}

case "${PHASE}" in
  prepare) prepare ;;
  rollout|all) rollout ;;
  summarize) summarize ;;
esac
