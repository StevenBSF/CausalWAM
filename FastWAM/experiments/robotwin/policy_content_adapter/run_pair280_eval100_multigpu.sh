#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
PAIR_ROOT="${PAIR_ROOT:-/root/fastwam_policy_artifacts/pair280_layer16_v1/seed1_c3_pair280_posttraining_v1}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,4,5,6,7}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
[[ "${#gpus[@]}" -eq 6 ]] || { echo "GPU_IDS must contain six GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -eq 6 ]] || {
  echo "GPU_IDS must be distinct" >&2; exit 2;
}
[[ "${CONFIRM_GPU_WORK}" == "YES" ]] || {
  echo "Pair-280 six-GPU evaluation requires CONFIRM_GPU_WORK=YES" >&2; exit 2;
}

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

eval_root="${PAIR_ROOT}/evaluation_seed53_100ep_v2"
amendment="${eval_root}/manifests/pair280_seed1_c3_seed53_eval100_v2.json"
rollout_root="${eval_root}/online_rollouts/c3"
status_file="${eval_root}/pair280_eval100_multigpu.status"
logs="${eval_root}/logs"

if [[ ! -f "${amendment}" ]]; then
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_eval100 \
    materialize --output "${amendment}"
fi
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_eval100 \
  validate --amendment "${amendment}" > /dev/null
[[ ! -e "${rollout_root}" ]] || {
  echo "Refusing to overwrite Pair-280 rollout root: ${rollout_root}" >&2
  exit 2
}

mkdir -p "${logs}"
for gpu in "${gpus[@]}"; do
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${logs}/gpu${gpu}_preflight.log"
done

printf 'RUNNING profile=pair280_seed53_eval100 checkpoint_step=18215 cells=6 episodes=600 gpu_ids=%s utc=%s\n' \
  "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"

tasks=(place_a2b_left place_a2b_left open_microwave open_microwave move_stapler_pad move_stapler_pad)
domains=(clean official_random clean official_random clean official_random)
configs=(demo_clean demo_randomized demo_clean demo_randomized demo_clean demo_randomized)
pids=()

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM HUP

for index in 0 1 2 3 4 5; do
  task="${tasks[$index]}"
  domain="${domains[$index]}"
  task_config="${configs[$index]}"
  gpu="${gpus[$index]}"
  output="${rollout_root}/cells/${task}/${domain}"
  log="${logs}/${task}_${domain}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_pair280 \
    "ckpt=${PAIR_ROOT}/formal/checkpoint.pt" "gpu_id=${gpu}" seed=53 \
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
trap - INT TERM HUP
if [[ "${failed}" -ne 0 ]]; then
  printf 'FAILED profile=pair280_seed53_eval100 utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
  echo "At least one Pair-280 evaluation cell failed" >&2
  exit 1
fi

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.pair280_eval100 \
  summarize-shards --amendment "${amendment}" --rollout-root "${rollout_root}" \
  > "${logs}/summary.log" 2>&1
printf 'DONE profile=pair280_seed53_eval100 cells=6 episodes=600 summary=PASS utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
