#!/usr/bin/env bash
set -euo pipefail

# Wait without episode polling for the current failed wave, the four-cell OOM
# recovery, and the Seed-3 acceleration.  Revalidate the expected completed
# matrix, then exec the stock runner so only Seed-2/C3 remains.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${FORMAL_ROOT}/online_rollouts_author_stock_seed42_v1}"
PLAN="${FORMAL_ROOT}/manifests/author_stock_seed42_rollout_plan_v1.json"
SEED3_COMPLETION="${FORMAL_ROOT}/manifests/author_stock_seed3_parallel_completion_v1.json"
OLD_MAIN_PID="${OLD_MAIN_PID:-3759159}"
SEED3_HELPER_PID="${SEED3_HELPER_PID:-4030543}"
OOM_HELPER_PID="${OOM_HELPER_PID:-68315}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"
CONFIRM_SUPERVISOR="${CONFIRM_SUPERVISOR:-NO}"

[[ "${CONFIRM_SUPERVISOR}" == YES ]] || { echo "CONFIRM_SUPERVISOR=YES required" >&2; exit 2; }
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${FASTWAM_ROOT}"

wait_exact_pid() {
  local pid="$1" expected="$2" actual
  if [[ -e "/proc/${pid}/cmdline" ]]; then
    actual="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${actual}" == "${expected}" ]] || { echo "PID ${pid} identity differs: ${actual}" >&2; return 2; }
    timeout --signal=TERM "${WAIT_TIMEOUT_SECONDS}s" tail --pid="${pid}" -f /dev/null
  fi
}

wait_exact_pid "${OLD_MAIN_PID}" \
  "bash experiments/robotwin/policy_content_adapter/run_release_formal_stock_rollout.sh "
wait_exact_pid "${OOM_HELPER_PID}" \
  "bash experiments/robotwin/policy_content_adapter/run_release_seed2_c1_oom_recovery.sh "
wait_exact_pid "${SEED3_HELPER_PID}" \
  "bash experiments/robotwin/policy_content_adapter/run_release_seed3_parallel.sh "

recovery_status="$(find "${ROLLOUT_ROOT}" -maxdepth 1 -type d -name 'seed2_c1_oom_recovery_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
[[ -n "${recovery_status}" && -f "${recovery_status}/status" ]] || { echo "OOM recovery status missing" >&2; exit 2; }
grep -q '^DONE cells=4 audits=4 ' "${recovery_status}/status" || { echo "OOM recovery not complete" >&2; exit 2; }

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_seed3_parallel \
  validate --path "${FORMAL_ROOT}/manifests/author_stock_seed3_parallel_schedule_v1.json" >/dev/null
[[ -f "${SEED3_COMPLETION}" ]] || { echo "Seed-3 completion missing" >&2; exit 2; }

"${PYTHON_BIN}" -c \
  'import glob,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files=glob.glob(str(root/"cells/**/completed_rollouts.json"),recursive=True)
keys=[]
for name in files:
 p=pathlib.Path(name); q=p.parts; i=q.index("cells"); x=json.load(open(name)); r=x["runs"][0]; keys.append((q[i+1],q[i+2],r["task"],r["domain"]))
required={("seed_1",c,t,d) for c in ("c1","c3") for t in ("place_a2b_left","open_microwave","move_stapler_pad") for d in ("clean","official_random")}
required|={("seed_2","c1",t,d) for t in ("place_a2b_left","open_microwave","move_stapler_pad") for d in ("clean","official_random")}
required|={("seed_3",c,t,d) for c in ("c1","c3") for t in ("place_a2b_left","open_microwave","move_stapler_pad") for d in ("clean","official_random")}
assert len(keys)==30 and set(keys)==required, (len(keys), sorted(required-set(keys)), sorted(set(keys)-required))' \
  "${ROLLOUT_ROOT}"

"${PYTHON_BIN}" -m experiments.robotwin.policy_content_adapter.release_formal_stock_rollout \
  audit-plan --plan "${PLAN}" --allow-existing-rollout-root >/dev/null

exec env PHASE=rollout CONFIRM_FORMAL_ROLLOUT=YES GPU_IDS=0,1,2,4,5,6 RUN_TESTS=0 \
  bash experiments/robotwin/policy_content_adapter/run_release_formal_stock_rollout.sh
