#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}
GPU_ID=${GPU_ID:-0}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/e0_e1/full}
MODEL_BASE=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}
TASKS=${TASKS:-place_a2b_left,open_microwave,move_stapler_pad}
LAYERS=${LAYERS:-8,16,24}
STATES_PER_TRAJECTORY=${STATES_PER_TRAJECTORY:-8}
TRAIN_STEPS=${TRAIN_STEPS:-1000}
GROUPS_PER_BATCH=${GROUPS_PER_BATCH:-8}
VAL_EVERY=${VAL_EVERY:-50}
SEED=${SEED:-0}
TEMPERATURE=${TEMPERATURE:-0.07}
MIN_TEMPORAL_GAP=${MIN_TEMPORAL_GAP:-8}
MIN_STATE_DISTANCE=${MIN_STATE_DISTANCE:-1e-5}
MIN_STATE_RETENTION=${MIN_STATE_RETENTION:-0.90}
RESUME=${RESUME:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export PYTHONUNBUFFERED=1

LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
mkdir -p -- "$RUN_DIR/cache" "$RUN_DIR/selection_metrics" \
  "$RUN_DIR/test_metrics" "$RUN_DIR/e1" "$RUN_DIR/comparison" \
  "$LOG_DIR" "$STATUS_DIR"

LOCK_FILE="$RUN_DIR/.run.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Another E0/E1 runner already holds %s\n' "$LOCK_FILE" >&2
  exit 1
fi

RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$RUN_LOG") 2>&1

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

on_error() {
  local exit_code=$?
  trap - ERR
  log "FAILED exit_code=$exit_code line=${BASH_LINENO[0]} command=$BASH_COMMAND"
  printf '%s\n' "OPERATIONAL_ERROR" >"$STATUS_DIR/state.txt"
  printf '%s\n' "$exit_code" >"$STATUS_DIR/OPERATIONAL_ERROR"
  log "Resume after fixing the cause with the same command (RESUME=1)."
  exit "$exit_code"
}
trap on_error ERR

on_interrupt() {
  trap - ERR INT TERM HUP
  printf '%s\n' "INTERRUPTED" >"$STATUS_DIR/state.txt"
  log "INTERRUPTED; rerun the same command to resume from the last validated stage"
  exit 130
}
trap on_interrupt INT TERM HUP

run_stage() {
  local stage=$1
  local validator=$2
  shift
  shift
  local marker="$STATUS_DIR/$stage.done"
  if [[ "$RESUME" == 1 && -f "$marker" ]]; then
    if "$validator"; then
      log "SKIP validated completed stage=$stage"
      return 0
    fi
    log "Stale/invalid completion marker for stage=$stage; refusing unsafe reuse"
    return 1
  fi
  log "START stage=$stage"
  "$@"
  "$validator"
  printf '%s\n' "$(timestamp)" >"$marker"
  log "DONE stage=$stage"
}

validate_settings() {
  [[ -x "$PYTHON_BIN" ]] || { log "Python is not executable: $PYTHON_BIN"; return 1; }
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { log "GPU_ID must be one non-negative integer"; return 1; }
  [[ "$RESUME" == 0 || "$RESUME" == 1 ]] || { log "RESUME must be 0 or 1"; return 1; }
  [[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || return 1
  [[ "$STATES_PER_TRAJECTORY" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$GROUPS_PER_BATCH" =~ ^[2-9][0-9]*$ ]] || return 1
  [[ "$VAL_EVERY" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$LAYERS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || return 1
  [[ -d "$MODEL_BASE" ]] || { log "Model base is missing: $MODEL_BASE"; return 1; }
  local checkpoint stats available_kib required_kib
  checkpoint="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt"
  stats="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
  [[ -r "$checkpoint" ]] || { log "Checkpoint is missing: $checkpoint"; return 1; }
  [[ -r "$stats" ]] || { log "Dataset stats are missing: $stats"; return 1; }
  available_kib=$(df -Pk "$RUN_DIR" | awk 'NR==2 {print $4}')
  required_kib=$((MIN_FREE_GIB * 1024 * 1024))
  ((available_kib >= required_kib)) || {
    log "Insufficient free disk: need ${MIN_FREE_GIB}GiB under $RUN_DIR"
    return 1
  }
}

write_and_check_config() {
  local commit dirty checkpoint stats
  commit=$(git rev-parse HEAD)
  # The working tree was already dirty before this experiment was added.
  # Persist a stable digest instead of embedding multiline shell text.
  dirty=$(git status --short | sha256sum | awk '{print $1}')
  checkpoint="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt"
  stats="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks init-config \
    --path "$RUN_DIR/run_config.json" \
    --repo-root "$REPO_ROOT" --git-commit "$commit" --git-dirty "$dirty" \
    --python "$PYTHON_BIN" --gpu-id "$GPU_ID" --model-base "$MODEL_BASE" \
    --checkpoint "$checkpoint" --dataset-stats "$stats" \
    --tasks "$TASKS" --layers "$LAYERS" \
    --states-per-trajectory "$STATES_PER_TRAJECTORY" \
    --train-steps "$TRAIN_STEPS" --groups-per-batch "$GROUPS_PER_BATCH" \
    --val-every "$VAL_EVERY" --seed "$SEED" --temperature "$TEMPERATURE" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --min-state-retention "$MIN_STATE_RETENTION"
}

extract_split() {
  local split=$1
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    -m experiments.robotwin.e0_e1.extract \
    --tasks "$TASKS" \
    --split "$split" \
    --states-per-trajectory "$STATES_PER_TRAJECTORY" \
    --layers "$LAYERS" \
    --device cuda \
    --output "$RUN_DIR/cache/$split.pt"
}

validate_cache_split() {
  local split=$1
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-cache \
    --cache "$RUN_DIR/cache/$split.pt" --split "$split" --tasks "$TASKS" \
    --layers "$LAYERS" --states-per-trajectory "$STATES_PER_TRAJECTORY"
}

validate_train_cache() { validate_cache_split train; }
validate_val_cache() { validate_cache_split val; }
validate_test_cache() { validate_cache_split test; }

evaluate_e0_validation() {
  local layer
  IFS=',' read -r -a layer_values <<<"$LAYERS"
  for layer in "${layer_values[@]}"; do
    "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
      --cache "$RUN_DIR/cache/val.pt" \
      --layer "$layer" \
      --experiment E0-RawBackbone \
      --min-temporal-gap "$MIN_TEMPORAL_GAP" \
      --min-state-distance "$MIN_STATE_DISTANCE" \
      --output-dir "$RUN_DIR/selection_metrics"
  done
}

validate_e0_selection_metrics() {
  local -a metric_paths=()
  local layer padded_layer
  IFS=',' read -r -a layer_values <<<"$LAYERS"
  for layer in "${layer_values[@]}"; do
    printf -v padded_layer '%02d' "$layer"
    metric_paths+=("$RUN_DIR/selection_metrics/e0_rawbackbone_layer_${padded_layer}.json")
  done
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-e0-metrics \
    --cache "$RUN_DIR/cache/val.pt" --tasks "$TASKS" \
    --layers "$LAYERS" --metrics "${metric_paths[@]}" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE"
}

select_layer() {
  local -a metric_paths=()
  local layer
  IFS=',' read -r -a layer_values <<<"$LAYERS"
  for layer in "${layer_values[@]}"; do
    printf -v padded_layer '%02d' "$layer"
    metric_paths+=("$RUN_DIR/selection_metrics/e0_rawbackbone_layer_${padded_layer}.json")
  done
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.select_layer \
    --metrics "${metric_paths[@]}" \
    --output-dir "$RUN_DIR/layer_selection"
}

validate_layer_selection() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-selection \
    --selection "$RUN_DIR/layer_selection/selection.json" \
    --selected-layer "$RUN_DIR/layer_selection/selected_layer.txt" \
    --cache "$RUN_DIR/cache/val.pt" --tasks "$TASKS" --layers "$LAYERS"
}

selected_layer() {
  local value
  value=$(tr -d '[:space:]' <"$RUN_DIR/layer_selection/selected_layer.txt")
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    log "Invalid selected layer: $value"
    return 1
  }
  printf '%s' "$value"
}

validate_init_validation() {
  local layer padded_layer
  layer=$(selected_layer)
  printf -v padded_layer '%02d' "$layer"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-metric \
    --metric "$RUN_DIR/selection_metrics/e1_inithead_layer_${padded_layer}.json" \
    --cache "$RUN_DIR/cache/val.pt" --split val --tasks "$TASKS" \
    --layer "$layer" --experiment E1-InitHead --seed "$SEED" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE"
}

evaluate_init_validation() {
  local layer
  layer=$(selected_layer)
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
    --cache "$RUN_DIR/cache/val.pt" \
    --layer "$layer" \
    --experiment E1-InitHead \
    --seed "$SEED" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --output-dir "$RUN_DIR/selection_metrics"
}

validate_e1_training() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-training \
    --checkpoint "$RUN_DIR/e1/e1_content_head.pt" \
    --log "$RUN_DIR/e1/train_log.json" \
    --train-cache "$RUN_DIR/cache/train.pt" --val-cache "$RUN_DIR/cache/val.pt" \
    --layer "$(selected_layer)" --steps "$TRAIN_STEPS" --seed "$SEED" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE"
}

train_e1() {
  local layer
  layer=$(selected_layer)
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    -m experiments.robotwin.e0_e1.train_e1 \
    --train-cache "$RUN_DIR/cache/train.pt" \
    --val-cache "$RUN_DIR/cache/val.pt" \
    --layer "$layer" \
    --steps "$TRAIN_STEPS" \
    --groups-per-batch "$GROUPS_PER_BATCH" \
    --temperature "$TEMPERATURE" \
    --val-every "$VAL_EVERY" \
    --seed "$SEED" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --device cuda \
    --output-dir "$RUN_DIR/e1"
}

validate_test_controls() {
  local layer padded_layer
  layer=$(selected_layer)
  printf -v padded_layer '%02d' "$layer"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks validate-test-metrics \
    --cache "$RUN_DIR/cache/test.pt" --tasks "$TASKS" --layer "$layer" \
    --seed "$SEED" \
    --e0 "$RUN_DIR/test_metrics/e0_rawbackbone_layer_${padded_layer}.json" \
    --init "$RUN_DIR/test_metrics/e1_inithead_layer_${padded_layer}.json" \
    --trained "$RUN_DIR/test_metrics/e1_trainedhead_layer_${padded_layer}.json"
}

evaluate_test_controls() {
  local layer
  layer=$(selected_layer)
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
    --cache "$RUN_DIR/cache/test.pt" --layer "$layer" \
    --experiment E0-RawBackbone \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --output-dir "$RUN_DIR/test_metrics"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate \
    --cache "$RUN_DIR/cache/test.pt" --layer "$layer" \
    --experiment E1-InitHead --seed "$SEED" \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --output-dir "$RUN_DIR/test_metrics"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    -m experiments.robotwin.e0_e1.evaluate \
    --cache "$RUN_DIR/cache/test.pt" --layer "$layer" \
    --experiment E1-TrainedHead \
    --head-checkpoint "$RUN_DIR/e1/e1_content_head.pt" \
    --seed "$SEED" --device cuda \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --output-dir "$RUN_DIR/test_metrics"
}

compare_final() {
  local layer padded_layer compare_exit=0
  layer=$(selected_layer)
  printf -v padded_layer '%02d' "$layer"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.compare \
    --metrics \
      "$RUN_DIR/test_metrics/e0_rawbackbone_layer_${padded_layer}.json" \
      "$RUN_DIR/test_metrics/e1_inithead_layer_${padded_layer}.json" \
      "$RUN_DIR/test_metrics/e1_trainedhead_layer_${padded_layer}.json" \
    --min-state-retention "$MIN_STATE_RETENTION" \
    --require-success \
    --output-dir "$RUN_DIR/comparison" || compare_exit=$?
  if ((compare_exit != 0)); then
    if "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks \
      validate-comparison --comparison "$RUN_DIR/comparison/comparison.json" \
      --allow-scientific-fail; then
      printf '%s\n' "$(timestamp)" >"$STATUS_DIR/SCIENTIFIC_FAIL"
      printf '%s\n' "SCIENTIFIC_FAIL" >"$STATUS_DIR/state.txt"
      log "Scientific success gate FAILED; reports preserved in $RUN_DIR/comparison"
      trap - ERR
      exit 2
    fi
    return "$compare_exit"
  fi
}

validate_final_comparison() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.runner_checks \
    validate-comparison --comparison "$RUN_DIR/comparison/comparison.json"
}

clear_terminal_status() {
  local marker
  for marker in OPERATIONAL_ERROR SCIENTIFIC_FAIL SUCCESS; do
    if [[ -e "$STATUS_DIR/$marker" ]]; then
      mv -- "$STATUS_DIR/$marker" \
        "$STATUS_DIR/${marker}.previous.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  done
}

cd -- "$REPO_ROOT"
validate_settings
write_and_check_config
log "RUN_DIR=$RUN_DIR GPU_ID=$GPU_ID TASKS=$TASKS LAYERS=$LAYERS"
log "states/trajectory=$STATES_PER_TRAJECTORY train_steps=$TRAIN_STEPS resume=$RESUME"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  log "PREFLIGHT_OK; no model or GPU job was started"
  exit 0
fi
clear_terminal_status
printf '%s\n' "RUNNING" >"$STATUS_DIR/state.txt"

run_stage extract_train validate_train_cache extract_split train
run_stage extract_val validate_val_cache extract_split val
run_stage evaluate_e0_validation validate_e0_selection_metrics evaluate_e0_validation
run_stage select_layer validate_layer_selection select_layer
log "Selected E1 layer=$(selected_layer)"
run_stage evaluate_init_validation validate_init_validation evaluate_init_validation
run_stage train_e1 validate_e1_training train_e1
run_stage extract_test validate_test_cache extract_split test
run_stage evaluate_test_controls validate_test_controls evaluate_test_controls
run_stage compare_final validate_final_comparison compare_final

printf '%s\n' "$(timestamp)" >"$STATUS_DIR/SUCCESS"
printf '%s\n' "SUCCESS" >"$STATUS_DIR/state.txt"
log "SUCCESS summary=$RUN_DIR/comparison/summary.md"
