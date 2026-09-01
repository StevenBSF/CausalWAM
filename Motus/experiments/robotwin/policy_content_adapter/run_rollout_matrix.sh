#!/usr/bin/env bash
set -euo pipefail
ROOT="${MOTUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-/root/anaconda3/envs/motusdata/bin/python}"
OUT="${OUT:-${ROOT}/outputs/policy_content_adapter/motus_v1/formal_rollout}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/outputs/policy_content_adapter/motus_v1/formal_runs}"
LINEAGE="${LINEAGE:-${ROOT}/outputs/policy_content_adapter/motus_v1/motus_robotwin2_lineage.json}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM/third_party/RoboTwin}"
PHASE="${PHASE:-prepare}"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6}"
SETTINGS="${OUT}/rollout_settings.json"
export MOTUS_ROOT="${ROOT}" PY ROBOTWIN_ROOT PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

prepare() {
  mkdir -p "${OUT}"
  if [[ ! -f "${SETTINGS}" ]]; then
    "${PY}" -m experiments.robotwin.policy_content_adapter.rollout settings --lineage "${LINEAGE}" --robotwin-root "${ROBOTWIN_ROOT}" --motus-root "${ROOT}" --simulator-seed 42 --output "${SETTINGS}"
  fi
  for seed in 1 2 3; do
    for short in m1 m3; do
      test -f "${RUNS_ROOT}/seed_${seed}/${short}/deployment_checkpoint.pt"
      test -f "${RUNS_ROOT}/seed_${seed}/${short}/training_summary.json"
    done
  done
  echo "PREPARE PASS settings=${SETTINGS}"
}

run_wave() {
  local seed="$1" short="$2" control checkpoint summary
  if [[ "${short}" == m1 ]]; then control=m1_architecture_action_control; else control=m3_ours; fi
  checkpoint="${RUNS_ROOT}/seed_${seed}/${short}/deployment_checkpoint.pt"
  summary="${RUNS_ROOT}/seed_${seed}/${short}/training_summary.json"
  IFS=',' read -ra gpus <<<"${GPU_IDS}"
  [[ ${#gpus[@]} -eq 6 ]] || { echo "exactly six GPU ids are required" >&2; exit 2; }
  local specs=("place_a2b_left clean" "place_a2b_left official_random" "open_microwave clean" "open_microwave official_random" "move_stapler_pad clean" "move_stapler_pad official_random")
  local pids=() index task domain cell
  for index in 0 1 2 3 4 5; do
    read -r task domain <<<"${specs[$index]}"
    cell="${OUT}/cells/${task}/${domain}/seed_${seed}/${short}"
    SETTINGS="${SETTINGS}" CHECKPOINT="${checkpoint}" TRAINING_SUMMARY="${summary}" CONTROL="${control}" TRAINING_SEED="${seed}" TASK="${task}" DOMAIN="${domain}" GPU_ID="${gpus[$index]}" CELL_ROOT="${cell}" bash "${ROOT}/experiments/robotwin/policy_content_adapter/run_rollout_cell.sh" >"${cell}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ ${failed} -eq 0 ]] || { echo "wave failed seed=${seed} control=${short}" >&2; exit 1; }
}

rollout() { for seed in 1 2 3; do run_wave "${seed}" m1; run_wave "${seed}" m3; done; }
aggregate() {
  mapfile -t cells < <(find "${OUT}/cells" -name completed_rollout.json -type f | sort)
  [[ ${#cells[@]} -eq 36 ]] || { echo "expected 36 completed cells, found ${#cells[@]}" >&2; exit 2; }
  "${PY}" -m experiments.robotwin.policy_content_adapter.evaluation --cells "${cells[@]}" --seeds 1,2,3 --output "${OUT}/summary.json"
}
case "${PHASE}" in prepare) prepare;; rollout) prepare; rollout;; aggregate) aggregate;; all) prepare; rollout; aggregate;; *) echo "invalid PHASE" >&2; exit 2;; esac
