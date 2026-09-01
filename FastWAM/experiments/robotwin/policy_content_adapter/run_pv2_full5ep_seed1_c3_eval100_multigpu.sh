#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
ROOT="${ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1_retry2}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
[[ "${#gpus[@]}" -eq 6 ]] || { echo "GPU_IDS must contain six GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -eq 6 ]] || {
  echo "GPU_IDS must be distinct" >&2; exit 2;
}
[[ "${CONFIRM_GPU_WORK}" == "YES" ]] || {
  echo "Six-GPU evaluation requires CONFIRM_GPU_WORK=YES" >&2; exit 2;
}

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

amendment="${ROOT}/manifests/seed1_c3_dev_seed53_eval100_v3.json"
rollout_root="${ROOT}/online_rollouts_dev_seed53_multigpu_v1/c3"
status_file="${ROOT}/seed1_c3_eval100_multigpu.status"
[[ -f "${amendment}" ]] || { echo "Missing amendment: ${amendment}" >&2; exit 2; }
[[ ! -e "${rollout_root}" ]] || { echo "Refusing to overwrite ${rollout_root}" >&2; exit 2; }

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_full5ep_eval100 \
  validate --amendment "${amendment}" > /dev/null

tasks=(place_a2b_left place_a2b_left open_microwave open_microwave move_stapler_pad move_stapler_pad)
domains=(clean official_random clean official_random clean official_random)
configs=(demo_clean demo_randomized demo_clean demo_randomized demo_clean demo_randomized)
mkdir -p "${ROOT}/logs"
for gpu in "${gpus[@]}"; do
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${ROOT}/logs/seed1_c3_multigpu_preflight_gpu${gpu}.log"
done
printf 'RUNNING checkpoint_step=18215 seed=53 cells=6 episodes=600 gpu_ids=%s utc=%s\n' \
  "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"

pids=()
for index in 0 1 2 3 4 5; do
  task="${tasks[$index]}"
  domain="${domains[$index]}"
  task_config="${configs[$index]}"
  gpu="${gpus[$index]}"
  output="${rollout_root}/cells/${task}/${domain}"
  log="${ROOT}/logs/seed1_c3_multigpu_${task}_${domain}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_pv2_full5ep \
    "ckpt=${ROOT}/runs/seed_1/c3/checkpoint.pt" "gpu_id=${gpu}" seed=53 \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    "EVALUATION.task_name=${task}" "EVALUATION.task_config=${task_config}" \
    EVALUATION.eval_num_episodes=100 \
    "+EVALUATION.pv2_followup_eval_amendment=${amendment}" \
    "EVALUATION.output_dir=${output}" > "${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]] || { echo "At least one evaluation shard failed" >&2; exit 1; }

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pv2_full5ep_eval100 \
  summarize-shards --amendment "${amendment}" --rollout-root "${rollout_root}" \
  > "${ROOT}/logs/seed1_c3_multigpu_summary.log" 2>&1
printf 'DONE checkpoint_step=18215 seed=53 cells=6 episodes=600 summary=PASS utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
