#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/c0_author_release_seed53_eval100_multigpu_v1}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,3,5,6,7}"
CONFIRM_GPU_WORK="${CONFIRM_GPU_WORK:-NO}"

IFS=',' read -r -a gpus <<< "${GPU_IDS}"
[[ "${#gpus[@]}" -eq 6 ]] || { echo "GPU_IDS must contain six GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -eq 6 ]] || {
  echo "GPU_IDS must be distinct" >&2; exit 2;
}
[[ "${CONFIRM_GPU_WORK}" == "YES" ]] || {
  echo "C0 six-GPU evaluation requires CONFIRM_GPU_WORK=YES" >&2; exit 2;
}
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to reuse ${OUTPUT_ROOT}" >&2; exit 2; }

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/logs"
status_file="${OUTPUT_ROOT}/c0_seed53_eval100.status"
trap 'code=$?; printf "FAILED exit_code=%s utc=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"; exit "${code}"' ERR

seed_bank="${OUTPUT_ROOT}/manifests/c0_seed53_100ep_bank.json"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.p_mode_selection \
  seed-bank --purpose development_analysis --simulator-seed 53 \
  --episodes-per-cell 100 \
  --evaluator-source "${FASTWAM_ROOT}/third_party/RoboTwin/script/eval_policy.py" \
  --output "${seed_bank}" > "${OUTPUT_ROOT}/logs/seed_bank.log"
seed_bank_id="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulator_seed_bank_id"])' "${seed_bank}")"

transport="${OUTPUT_ROOT}/c0_author_release_seed53_transport.pt"
"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.c0_eval_transport \
  --base-checkpoint "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384.pt" \
  --dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
  --model-base-path "${MODEL_BASE}" \
  --official-manifest "${FASTWAM_ROOT}/experiments/robotwin/policy_content_adapter/configs/official_three_task_manifest.json" \
  --base-lineage-manifest "${FASTWAM_ROOT}/experiments/robotwin/policy_content_adapter/configs/author_release_base_manifest.json" \
  --identity-audit "${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/c0_dev_gate_v1/c0_runtime_identity_audit.json" \
  --output "${transport}" --rollout-protocol-id three_task_policy_online_v2 \
  --simulator-seed-bank-id "${seed_bank_id}" \
  --simulator-seed-bank-manifest "${seed_bank}" \
  --episodes-per-task 100 --transport-seed 0 --evaluation-stage deployment_gate \
  > "${OUTPUT_ROOT}/logs/transport.log"

for gpu in "${gpus[@]}"; do
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${OUTPUT_ROOT}/logs/preflight_gpu${gpu}.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${free_mib}" -ge 30000 ]] || { echo "GPU ${gpu} has insufficient free memory" >&2; exit 2; }
done

tasks=(place_a2b_left place_a2b_left open_microwave open_microwave move_stapler_pad move_stapler_pad)
domains=(clean official_random clean official_random clean official_random)
configs=(demo_clean demo_randomized demo_clean demo_randomized demo_clean demo_randomized)
printf 'RUNNING control=C0 seed=53 cells=6 episodes=600 gpu_ids=%s utc=%s\n' \
  "${GPU_IDS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
pids=()
for index in 0 1 2 3 4 5; do
  task="${tasks[$index]}"; domain="${domains[$index]}"; task_config="${configs[$index]}"; gpu="${gpus[$index]}"
  output="${OUTPUT_ROOT}/cells/${task}/${domain}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m \
    experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${transport}" "gpu_id=${gpu}" seed=53 \
    "EVALUATION.dataset_stats_path=${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    "EVALUATION.task_name=${task}" "EVALUATION.task_config=${task_config}" \
    EVALUATION.eval_num_episodes=100 "EVALUATION.output_dir=${output}" \
    > "${OUTPUT_ROOT}/logs/${task}_${domain}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" -eq 0 ]] || { echo "At least one C0 evaluation shard failed" >&2; exit 1; }

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.c0_seed53_eval100 \
  --checkpoint "${transport}" --rollout-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.json" > "${OUTPUT_ROOT}/logs/summary.log"
printf 'DONE control=C0 seed=53 cells=6 episodes=600 summary=PASS utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
