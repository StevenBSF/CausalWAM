#!/usr/bin/env bash
set -euo pipefail

# Recover the four short Seed-2/C1 cells whose first launch encountered a
# third-model GPU OOM while the packed Seed-3 jobs were still resident.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${FORMAL_ROOT}/online_rollouts_author_stock_seed42_v1}"
PLAN="${PLAN:-${FORMAL_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json}"
STOCK_AMENDMENT="${STOCK_AMENDMENT:-${FORMAL_ROOT}/manifests/author_stock_seed42_unpaired_v1.json}"
CHECKPOINT="${FORMAL_ROOT}/runs/seed_2/c1/checkpoint.pt"
STATS="${FORMAL_ROOT}/runs/seed_2/c1/dataset_stats.json"
CONFIRM_RECOVERY="${CONFIRM_RECOVERY:-NO}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-7200}"
MIN_FREE_MIB="${MIN_FREE_MIB:-35000}"

[[ "${CONFIRM_RECOVERY}" == YES ]] || { echo "CONFIRM_RECOVERY=YES required" >&2; exit 2; }
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

tasks=(place_a2b_left place_a2b_left move_stapler_pad move_stapler_pad)
configs=(demo_clean demo_randomized demo_clean demo_randomized)
domains=(clean official_random clean official_random)
gpus=(6 7 2 4)
cells=(12 13 16 17)
WORKERS=()
declare -A ACTIVE=() ATTEMPT=() STATUS=() CELL=() GPU=()

cleanup() {
  local code=$? pid
  trap - EXIT HUP INT TERM
  for pid in "${WORKERS[@]}"; do
    if [[ "${ACTIVE[${pid}]:-0}" == 1 ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${WORKERS[@]}"; do
    if [[ "${ACTIVE[${pid}]:-0}" == 1 ]]; then
      if kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pid}" 2>/dev/null || true; fi
      wait "${pid}" 2>/dev/null || true
      ACTIVE[${pid}]=0
    fi
  done
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
  audit-plan --plan "${PLAN}" --allow-existing-rollout-root >/dev/null

RECOVERY_ROOT="${ROLLOUT_ROOT}/seed2_c1_oom_recovery_${RUN_STAMP}"
[[ ! -e "${RECOVERY_ROOT}" ]] || { echo "Recovery root exists" >&2; exit 2; }
mkdir -p "${RECOVERY_ROOT}/gpu_preflight"

for index in 0 1 2 3; do
  task="${tasks[${index}]}"; domain="${domains[${index}]}"; gpu="${gpus[${index}]}"
  cell_root="${ROLLOUT_ROOT}/cells/seed_2/c1/${task}/${domain}"
  [[ -d "${cell_root}" ]] || { echo "Missing failed cell root ${cell_root}" >&2; exit 2; }
  mapfile -t completed < <(find "${cell_root}" -mindepth 2 -maxdepth 2 -type f -path '*/attempt_*/completed_rollouts.json' -print)
  [[ "${#completed[@]}" -eq 0 ]] || { echo "Cell already completed: ${cell_root}" >&2; exit 2; }
  failed_log="$(find "${cell_root}" -maxdepth 1 -type f -name 'attempt_*.worker.log' -print | sort | tail -1)"
  [[ -f "${failed_log}" ]] && grep -q 'OutOfMemoryError' "${failed_log}" || {
    echo "Cell lacks audited OOM evidence: ${cell_root}" >&2; exit 2;
  }
  report="${RECOVERY_ROOT}/gpu_preflight/gpu_${gpu}.json"
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.robotwin_gpu_runtime \
    preflight --gpu-id "${gpu}" > "${report}"
  "${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["physical_gpu_index"]==int(sys.argv[2]); assert int(p["memory_free_mib_at_preflight"])>=int(sys.argv[3])' \
    "${report}" "${gpu}" "${MIN_FREE_MIB}"
done

printf 'RUNNING cells=12,13,16,17 cause=third_model_oom utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RECOVERY_ROOT}/status"

for index in 0 1 2 3; do
  task="${tasks[${index}]}"; config="${configs[${index}]}"; domain="${domains[${index}]}"; gpu="${gpus[${index}]}"
  cell_root="${ROLLOUT_ROOT}/cells/seed_2/c1/${task}/${domain}"
  attempt="${cell_root}/attempt_${RUN_STAMP}_oomrecovery_pid${BASHPID}"
  log="${attempt}.worker.log"; status="${attempt}.status"
  [[ ! -e "${attempt}" && ! -e "${log}" && ! -e "${status}" ]] || { echo "Attempt exists" >&2; exit 2; }
  mkdir "${attempt}"
  printf 'RUNNING recovery=%s cell=%s actual_gpu=%s utc=%s\n' "${RECOVERY_ROOT}" "${cells[${index}]}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status}"
  setsid timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}s" \
    "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.eval_robotwin_single \
    "ckpt=${CHECKPOINT}" "gpu_id=${gpu}" seed=42 mixed_precision=bf16 \
    "EVALUATION.robotwin_root=${FASTWAM_ROOT}/third_party/RoboTwin" \
    "EVALUATION.task_name=${task}" "EVALUATION.task_config=${config}" \
    EVALUATION.eval_num_episodes=100 "EVALUATION.output_dir=${attempt}" \
    "EVALUATION.dataset_stats_path=${STATS}" \
    "+EVALUATION.stock_protocol_amendment=${STOCK_AMENDMENT}" \
    EVALUATION.instruction_type=unseen EVALUATION.action_horizon=null \
    EVALUATION.replan_steps=24 EVALUATION.num_inference_steps=10 \
    EVALUATION.sigma_shift=null EVALUATION.text_cfg_scale=1.0 \
    EVALUATION.rand_device=cpu EVALUATION.tiled=false \
    EVALUATION.timing_enabled=false EVALUATION.skip_get_obs_within_replan=true \
    > "${log}" 2>&1 &
  pid=$!; WORKERS+=("${pid}"); ACTIVE[${pid}]=1; ATTEMPT[${pid}]="${attempt}"
  STATUS[${pid}]="${status}"; CELL[${pid}]="${cells[${index}]}"; GPU[${pid}]="${gpu}"
  sleep 10
done

for pid in "${WORKERS[@]}"; do
  if wait "${pid}"; then ACTIVE[${pid}]=0; else
    code=$?; ACTIVE[${pid}]=0
    printf 'FAILED cell=%s gpu=%s exit=%s utc=%s\n' "${CELL[${pid}]}" "${GPU[${pid}]}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS[${pid}]}"
    exit "${code}"
  fi
  manifest="${ATTEMPT[${pid}]}/completed_rollouts.json"; audit="${ATTEMPT[${pid}]}.audit.json"
  "${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
    audit-cell --plan "${PLAN}" --manifest "${manifest}" > "${audit}"
  printf 'DONE cell=%s gpu=%s audit=%s utc=%s\n' "${CELL[${pid}]}" "${GPU[${pid}]}" "${audit}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS[${pid}]}"
done

printf 'DONE cells=4 audits=4 utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RECOVERY_ROOT}/status"
echo "PASS: recovered Seed-2/C1 cells 12,13,16,17."
