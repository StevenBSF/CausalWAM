#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM"
PYTHON_BIN="/root/anaconda3/envs/fastwam-robotwin-bw/bin/python"
MODEL_BASE="/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints"
LINEAGE="$REPO_ROOT/experiments/robotwin/policy_content_adapter/configs/author_release_base_manifest.json"
LINEAGE_SHA="d90e6d545c04c28e9e73b6b8a9356ec5e9320be4be6f6b7e3b69237a3f38cefc"
BINDING="$REPO_ROOT/outputs/policy_content_adapter/release_base_v1/paired_binding_manifest.json"
BINDING_SHA="ab2904a01636fdb6fd80798a65580cc58c07451fc30ebcd5a527161c56025835"
CHECKPOINT="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt"
DATASET_STATS="$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
OFFICIAL_MANIFEST="$REPO_ROOT/experiments/robotwin/policy_content_adapter/configs/official_three_task_manifest.json"
PAIRED_ROOT="$REPO_ROOT/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21"
PAIRED_MANIFEST="$PAIRED_ROOT/meta/policy_native_action_manifest.json"
PAIRED_AUDIT="$PAIRED_ROOT/meta/policy_native_action_audit.json"
STATE_BANK="$PAIRED_ROOT/meta/policy_paired_state_bank.json"
OUTPUT_ROOT="$REPO_ROOT/outputs/policy_content_adapter/release_base_v1"
TEXT_CACHE="$OUTPUT_ROOT/paired_text_cache"
TEXT_AUDIT="$TEXT_CACHE/release_paired_text_cache.audit.json"
POLICY_CACHE="$OUTPUT_ROOT/policy_release50tasks_native50hz_four_scene_v1.pt"
STATUS_FILE="$OUTPUT_ROOT/release_cache_pipeline.status"
LOG_FILE="$OUTPUT_ROOT/logs/release_cache_pipeline.log"

mkdir -p "$OUTPUT_ROOT/logs"
exec >> "$LOG_FILE" 2>&1
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD="True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

stage="preflight"
on_error() {
  local exit_code=$?
  printf 'FAILED stage=%s exit_code=%s utc=%s\n' "$stage" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"
  exit "$exit_code"
}
trap on_error ERR

printf 'RUNNING stage=%s utc=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"
test -x "$PYTHON_BIN"
test -f "$LINEAGE"
test -f "$BINDING"
test -f "$CHECKPOINT"
test -f "$DATASET_STATS"
test -f "$OFFICIAL_MANIFEST"
test -d "$PAIRED_ROOT"
test -f "$PAIRED_MANIFEST"
test -f "$PAIRED_AUDIT"
test -f "$STATE_BANK"

stage="paired_text_cache"
printf 'RUNNING stage=%s utc=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"
if [[ -f "$TEXT_AUDIT" ]]; then
  printf '[%s] Reusing completed paired text cache: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TEXT_CACHE"
else
  if [[ -d "$TEXT_CACHE" ]] && find "$TEXT_CACHE" -mindepth 1 -print -quit | grep -q .; then
    printf 'Refusing partial paired text cache without audit: %s\n' "$TEXT_CACHE" >&2
    false
  fi
  "$PYTHON_BIN" -m experiments.robotwin.policy_content_adapter.prepare_release_paired_text_cache \
    --cache-dir "$TEXT_CACHE" \
    --base-lineage-manifest "$LINEAGE" \
    --base-lineage-sha256 "$LINEAGE_SHA" \
    --release-paired-binding "$BINDING" \
    --release-paired-binding-sha256 "$BINDING_SHA" \
    --checkpoint "$CHECKPOINT" \
    --dataset-stats "$DATASET_STATS" \
    --official-manifest "$OFFICIAL_MANIFEST" \
    --model-base-path "$MODEL_BASE" \
    --device cuda \
    --prepare-release-paired-text-cache
fi
test -f "$TEXT_AUDIT"

stage="layer16_cache"
printf 'RUNNING stage=%s utc=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_FILE"
test ! -e "$POLICY_CACHE"
"$PYTHON_BIN" -m experiments.robotwin.policy_content_adapter.extract_policy_cache \
  --base-lineage-manifest "$LINEAGE" \
  --base-lineage-sha256 "$LINEAGE_SHA" \
  --release-paired-binding "$BINDING" \
  --release-paired-binding-sha256 "$BINDING_SHA" \
  --checkpoint "$CHECKPOINT" \
  --dataset-stats "$DATASET_STATS" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --paired-root "$PAIRED_ROOT" \
  --paired-manifest "$PAIRED_MANIFEST" \
  --paired-audit "$PAIRED_AUDIT" \
  --state-bank "$STATE_BANK" \
  --text-cache-dir "$TEXT_CACHE" \
  --model-base-path "$MODEL_BASE" \
  --output "$POLICY_CACHE" \
  --device cuda \
  --states-per-trajectory 8 \
  --layer 16 \
  --extract-policy-cache

test -f "$POLICY_CACHE"
test -f "$POLICY_CACHE.audit.json"
stage="complete"
printf 'DONE stage=%s utc=%s cache=%s audit=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$POLICY_CACHE" "$POLICY_CACHE.audit.json" > "$STATUS_FILE"
printf '[%s] Release paired text + Layer-16 cache pipeline PASS\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
