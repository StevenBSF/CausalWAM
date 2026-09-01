#!/usr/bin/env bash
set -euo pipefail

# Background dependency gate: wait for the already-running C0 development
# process, prove its strict non-formal audit passed, and only then start the
# locked P-v1/P-v2 dev-selection pipeline.  This script never polls GPUs or
# interprets diagnostic success rates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}"
WAIT_PID="${WAIT_PID:?WAIT_PID must name the running C0 wrapper process}"
C0_ROOT="${C0_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/c0_dev_gate_v1}"
P_ROOT="${P_ROOT:-${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1}"
STATUS_FILE="${P_ROOT}/post_c0_chain.status"

if [[ ! "${WAIT_PID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAIT_PID must be a positive integer; got ${WAIT_PID}" >&2
  exit 2
fi
if [[ ! -d "${P_ROOT}" ]]; then
  echo "P-mode materialization root is missing: ${P_ROOT}" >&2
  exit 2
fi
if [[ -e "${P_ROOT}/p_mode_selection.json" ]]; then
  echo "Refusing to reuse an already-selected P-mode root: ${P_ROOT}" >&2
  exit 2
fi

trap 'code=$?; printf "FAILED exit_code=%s utc=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"; exit "${code}"' ERR
printf 'WAITING stage=c0_dev_gate pid=%s utc=%s\n' \
  "${WAIT_PID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"

# GNU tail uses inotify for the file and only checks the named PID for exit;
# no log or GPU polling is performed.
/usr/bin/tail --pid="${WAIT_PID}" -f /dev/null

printf 'VERIFYING stage=c0_strict_audit utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
"${PYTHON_BIN}" - "${C0_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
status = (root / "c0_dev_gate.status").read_text(encoding="utf-8").strip()
if not status.startswith("DONE stage=complete "):
    raise SystemExit(f"C0 wrapper did not complete successfully: {status!r}")
audit_path = root / "strict_c0_dev_gate_audit.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
required = {
    "status": "PASS",
    "kind": "policy_c0_author_release_dev_deployment_gate",
    "scientific_result": False,
    "formal_test_bank_opened": False,
    "formal_evaluation_records_emitted": False,
    "total_rollout_episodes": 6,
}
for key, expected in required.items():
    if audit.get(key) != expected:
        raise SystemExit(
            f"C0 strict audit field {key!r} differs: "
            f"{audit.get(key)!r} != {expected!r}"
        )
PY

printf 'RUNNING stage=p_mode_dev utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
env \
  PHASE=all \
  OUTPUT_ROOT="${P_ROOT}" \
  TRAIN_GPU_IDS=0,1 \
  PARALLEL_TRAIN=1 \
  ROLLOUT_GPU_IDS=0,1 \
  PARALLEL_ROLLOUT=1 \
  bash "${SCRIPT_DIR}/run_release_pmode_dev.sh"

printf 'DONE stage=complete utc=%s output=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${P_ROOT}" > "${STATUS_FILE}"
