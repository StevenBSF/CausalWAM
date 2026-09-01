#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}
GPU_ID=${GPU_ID:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/e0_e1/smoke}
MODEL_BASE=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"

cd -- "$REPO_ROOT"
mkdir -p -- "$OUTPUT_DIR/cache" "$OUTPUT_DIR/selection_metrics" \
  "$OUTPUT_DIR/test_metrics" "$OUTPUT_DIR/e1" "$OUTPUT_DIR/comparison"

run_extract() {
  local split=$1
  local content_id=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    -m experiments.robotwin.e0_e1.extract \
    --tasks place_a2b_left \
    --split "$split" \
    --states-per-trajectory 2 \
    --allow-incomplete \
    --content-ids "$content_id" \
    --layers 8,16,24 \
    --device cuda \
    --output "$OUTPUT_DIR/cache/$split.pt" \
    "$@"
}

echo "[smoke] extracting train representations on physical GPU $GPU_ID"
run_extract train 0 --verify-native-prefill
echo "[smoke] extracting validation representations"
run_extract val 30
echo "[smoke] extracting held-out test representations"
run_extract test 40

for layer in 8 16 24; do
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
    --cache "$OUTPUT_DIR/cache/val.pt" \
    --layer "$layer" \
    --experiment E0-RawBackbone \
    --output-dir "$OUTPUT_DIR/selection_metrics"
done

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$OUTPUT_DIR/cache/val.pt" \
  --layer 16 \
  --experiment E1-InitHead \
  --seed 0 \
  --output-dir "$OUTPUT_DIR/selection_metrics"

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.train_e1 \
  --train-cache "$OUTPUT_DIR/cache/train.pt" \
  --val-cache "$OUTPUT_DIR/cache/val.pt" \
  --layer 16 \
  --steps 1 \
  --groups-per-batch 2 \
  --val-every 1 \
  --seed 0 \
  --device cpu \
  --output-dir "$OUTPUT_DIR/e1"

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$OUTPUT_DIR/cache/test.pt" \
  --layer 16 \
  --experiment E0-RawBackbone \
  --output-dir "$OUTPUT_DIR/test_metrics"

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$OUTPUT_DIR/cache/test.pt" \
  --layer 16 \
  --experiment E1-InitHead \
  --seed 0 \
  --output-dir "$OUTPUT_DIR/test_metrics"

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$OUTPUT_DIR/cache/test.pt" \
  --layer 16 \
  --experiment E1-TrainedHead \
  --head-checkpoint "$OUTPUT_DIR/e1/e1_content_head.pt" \
  --device cpu \
  --output-dir "$OUTPUT_DIR/test_metrics"

"$PYTHON_BIN" -m experiments.robotwin.e0_e1.compare \
  --metrics \
    "$OUTPUT_DIR/test_metrics/e0_rawbackbone_layer_16.json" \
    "$OUTPUT_DIR/test_metrics/e1_inithead_layer_16.json" \
    "$OUTPUT_DIR/test_metrics/e1_trainedhead_layer_16.json" \
  --min-state-retention 0.90 \
  --output-dir "$OUTPUT_DIR/comparison"

echo "[smoke] complete: $OUTPUT_DIR/comparison/summary.md"
