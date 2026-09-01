#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SMOKE_CONFIG_DIR="${SMOKE_CONFIG_DIR:?set SMOKE_CONFIG_DIR}"
FORMAL_CONFIG_DIR="${FORMAL_CONFIG_DIR:?set FORMAL_CONFIG_DIR}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

cd "${ROOT}"

echo "[$(date -u +%FT%TZ)] START M-P2 author-batch8 telemetry smoke"
CONFIG_DIR="${SMOKE_CONFIG_DIR}" GPU_IDS="${GPU_IDS}" \
  bash experiments/robotwin/policy_content_adapter/run_smoke_pair.sh
echo "[$(date -u +%FT%TZ)] PASS M-P2 author-batch8 telemetry smoke"

echo "[$(date -u +%FT%TZ)] START M-P2 seed1 formal five-epoch M1/M3"
CONFIG_DIR="${FORMAL_CONFIG_DIR}" GPU_IDS="${GPU_IDS}" \
  bash experiments/robotwin/policy_content_adapter/run_formal_pair.sh
echo "[$(date -u +%FT%TZ)] PASS M-P2 seed1 formal five-epoch M1/M3"
