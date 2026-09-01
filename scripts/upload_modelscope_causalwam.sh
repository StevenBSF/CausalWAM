#!/usr/bin/env bash
set -euo pipefail

# Upload the CausalWAM runtime/training artifacts to ModelScope without using
# the host Clash proxy.  This script is intentionally resumable: ModelScope's
# upload cache is enabled and a local create-only marker is written only after
# each remote upload command exits successfully.

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"
export LC_ALL=C

REPO_ID="${REPO_ID:-StevenHZ/CausalWAM}"
REVISION="${REVISION:-master}"
PHASE="${PHASE:-preflight}"
CONFIRM_UPLOAD="${CONFIRM_UPLOAD:-NO}"
MAX_WORKERS="${MAX_WORKERS:-4}"
PART_BYTES="${PART_BYTES:-8G}"
KEEP_PACKAGES="${KEEP_PACKAGES:-NO}"

CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-/mnt/cpfs-E/baoshifeng/CausalWAM}"
FASTWAM_ROOT="${FASTWAM_ROOT:-${CAUSALWAM_ROOT}/FastWAM}"
MODEL_BASE="${MODEL_BASE:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/mnt/cpfs-E/baoshifeng/FastWAM/data/robotwin2.0/robotwin2.0}"
PAIR_ROOT="${PAIR_ROOT:-/root/fastwam_policy_artifacts/pair280_layer16_v1}"
PAIR_RUN_ROOT="${PAIR_RUN_ROOT:-${PAIR_ROOT}/seed1_c3_pair280_posttraining_v1}"
STATE_ROOT="${STATE_ROOT:-/root/fastwam_policy_artifacts/modelscope_causalwam_upload_v1}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${STATE_ROOT}/packages}"
DONE_ROOT="${STATE_ROOT}/done"
EVENT_LOG="${STATE_ROOT}/events.log"
SDK_VENV="${SDK_VENV:-${STATE_ROOT}/modelscope_venv}"

mkdir -p "${STATE_ROOT}" "${PACKAGE_ROOT}" "${DONE_ROOT}"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
event() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${EVENT_LOG}" >&2
}
die() {
  event "ERROR $*"
  exit 2
}
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

case "${PHASE}" in
  preflight|docs|models|final|pair_cache|official_text_cache|train_data|assets|provenance|resume|train|eval|all) ;;
  *) die "PHASE must be preflight, docs, models, final, pair_cache, official_text_cache, train_data, assets, provenance, resume, train, eval, or all" ;;
esac
case "${CONFIRM_UPLOAD}" in YES|NO) ;; *) die "CONFIRM_UPLOAD must be YES or NO" ;; esac
case "${KEEP_PACKAGES}" in YES|NO) ;; *) die "KEEP_PACKAGES must be YES or NO" ;; esac

ensure_modelscope() {
  if [[ -n "${MODELSCOPE_BIN:-}" ]]; then
    [[ -x "${MODELSCOPE_BIN}" ]] || die "MODELSCOPE_BIN is not executable: ${MODELSCOPE_BIN}"
    MS="${MODELSCOPE_BIN}"
  elif [[ -x "${SDK_VENV}/bin/modelscope" ]]; then
    MS="${SDK_VENV}/bin/modelscope"
  elif command -v modelscope >/dev/null 2>&1; then
    MS="$(command -v modelscope)"
  elif [[ -x /tmp/causalwam-modelscope-venv/bin/modelscope ]]; then
    MS=/tmp/causalwam-modelscope-venv/bin/modelscope
  else
    event "Installing ModelScope SDK 1.39.1 into ${SDK_VENV} with all proxy variables unset"
    /root/anaconda3/bin/python -m venv "${SDK_VENV}"
    "${SDK_VENV}/bin/pip" install --disable-pip-version-check 'modelscope==1.39.1'
    MS="${SDK_VENV}/bin/modelscope"
  fi
  event "ModelScope CLI: $(${MS} --version 2>/dev/null || true)"
}

preflight() {
  ensure_modelscope
  for name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; do
    [[ -z "${!name:-}" ]] || die "proxy variable unexpectedly remains set: ${name}"
  done
  event "Proxy contract PASS: upload process has no HTTP/HTTPS/ALL proxy"
  if ! "${MS}" whoami; then
    die "ModelScope login missing. Run: env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy ${MS} login"
  fi
  "${MS}" info "${REPO_ID}" --repo-type dataset >/dev/null
  event "Repository access PASS: ${REPO_ID} revision=${REVISION}"
}

marker_for() {
  local key="$1"
  key="${key//\//__}"
  printf '%s/%s.done' "${DONE_ROOT}" "${key}"
}

upload_file() {
  local key="$1" source="$2" target="$3" marker
  marker="$(marker_for "${key}")"
  if [[ -f "${marker}" ]]; then
    event "SKIP completed ${key}"
    return
  fi
  require_file "${source}"
  event "UPLOAD file ${key}: ${source} -> ${target}"
  "${MS}" upload "${REPO_ID}" "${source}" "${target}" \
    --repo-type dataset --revision "${REVISION}" \
    --commit-message "CausalWAM ${key}"
  printf 'status=PASS utc=%s source=%s target=%s\n' "$(timestamp)" "${source}" "${target}" > "${marker}"
}

upload_folder() {
  local key="$1" source="$2" target="$3" marker
  marker="$(marker_for "${key}")"
  if [[ -f "${marker}" ]]; then
    event "SKIP completed ${key}"
    return
  fi
  require_dir "${source}"
  event "UPLOAD folder ${key}: ${source} -> ${target}"
  "${MS}" upload "${REPO_ID}" "${source}" "${target}" \
    --repo-type dataset --revision "${REVISION}" \
    --max-workers "${MAX_WORKERS}" --use-cache \
    --exclude '.ms_upload_cache' '.ms_upload_cache/**' \
    --commit-message "CausalWAM ${key}"
  printf 'status=PASS utc=%s source=%s target=%s\n' "$(timestamp)" "${source}" "${target}" > "${marker}"
}

upload_folder_filtered() {
  local key="$1" source="$2" target="$3" marker
  marker="$(marker_for "${key}")"
  if [[ -f "${marker}" ]]; then
    event "SKIP completed ${key}"
    return
  fi
  require_dir "${source}"
  event "UPLOAD filtered folder ${key}: ${source} -> ${target}"
  "${MS}" upload "${REPO_ID}" "${source}" "${target}" \
    --repo-type dataset --revision "${REVISION}" \
    --max-workers "${MAX_WORKERS}" --use-cache \
    --exclude 'checkpoint.pt' 'checkpoints/**' '*.log' '.ms_upload_cache' '.ms_upload_cache/**' \
    --commit-message "CausalWAM ${key}"
  printf 'status=PASS utc=%s source=%s target=%s\n' "$(timestamp)" "${source}" "${target}" > "${marker}"
}

archive_incomplete_package() {
  local dir="$1"
  if [[ -d "${dir}" && ! -f "${dir}/.package_complete" ]]; then
    mv "${dir}" "${dir}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

finish_package() {
  local dir="$1" description="$2"
  (
    cd "${dir}"
    sha256sum ./*.tar.part-* > SHA256SUMS
    printf '%s\n' "${description}" > CONTENTS.txt
    printf 'status=PASS utc=%s part_bytes=%s\n' "$(timestamp)" "${PART_BYTES}" > .package_complete
  )
}

make_path_archive() {
  local key="$1" base="$2" description="$3"
  shift 3
  local dir="${PACKAGE_ROOT}/${key}"
  if [[ -f "${dir}/.package_complete" ]]; then
    event "REUSE package ${key}"
    printf '%s' "${dir}"
    return
  fi
  archive_incomplete_package "${dir}"
  mkdir -p "${dir}"
  event "PACKAGE ${key} with deterministic tar split ${PART_BYTES}"
  tar --sort=name --format=posix \
    --pax-option=delete=atime,delete=ctime \
    --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -C "${base}" -cf - "$@" \
    | split --bytes="${PART_BYTES}" --numeric-suffixes=0 --suffix-length=3 \
      - "${dir}/${key}.tar.part-"
  finish_package "${dir}" "${description}"
  printf '%s' "${dir}"
}

build_official_subset_list() {
  local list="${STATE_ROOT}/official_three_task_subset.files"
  if [[ -f "${list}" ]]; then
    printf '%s' "${list}"
    return
  fi
  require_dir "${OFFICIAL_ROOT}/meta"
  {
    find "${OFFICIAL_ROOT}/meta" -maxdepth 1 -type f -printf 'meta/%f\n'
    local start end ep chunk camera
    for bounds in '11000 11549' '9350 9899' '8250 8799'; do
      read -r start end <<< "${bounds}"
      for ((ep=start; ep<=end; ep++)); do
        printf -v chunk '%03d' "$((ep / 1000))"
        printf 'data/chunk-%s/episode_%06d.parquet\n' "${chunk}" "${ep}"
        for camera in observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist; do
          printf 'videos/chunk-%s/%s/episode_%06d.mp4\n' "${chunk}" "${camera}" "${ep}"
        done
      done
    done
  } | sort -u > "${list}"
  local count missing=0 rel
  count="$(wc -l < "${list}")"
  [[ "${count}" -eq 6604 ]] || die "official subset list has ${count} paths; expected 6604"
  while IFS= read -r rel; do
    [[ -f "${OFFICIAL_ROOT}/${rel}" ]] || { event "MISSING official subset file ${rel}"; missing=$((missing + 1)); }
  done < "${list}"
  [[ "${missing}" -eq 0 ]] || die "official subset has ${missing} missing files"
  printf '%s' "${list}"
}

make_official_subset_archive() {
  local key=official-three-task-subset dir="${PACKAGE_ROOT}/official-three-task-subset" list
  if [[ -f "${dir}/.package_complete" ]]; then
    event "REUSE package ${key}"
    printf '%s' "${dir}"
    return
  fi
  list="$(build_official_subset_list)"
  archive_incomplete_package "${dir}"
  mkdir -p "${dir}"
  event "PACKAGE ${key}: 1650 episodes, 6604 files"
  tar --sort=name --format=posix \
    --pax-option=delete=atime,delete=ctime \
    --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -C "${OFFICIAL_ROOT}" -T "${list}" -cf - \
    | split --bytes="${PART_BYTES}" --numeric-suffixes=0 --suffix-length=3 \
      - "${dir}/${key}.tar.part-"
  cp "${list}" "${dir}/FILES.txt"
  finish_package "${dir}" 'Original full metadata plus the exact 1650 official three-task episode parquet/video files.'
  printf '%s' "${dir}"
}

upload_archive() {
  local key="$1" package_dir="$2" marker
  marker="$(marker_for "package-${key}")"
  if [[ -f "${marker}" ]]; then
    event "SKIP completed package ${key}"
    return
  fi
  require_file "${package_dir}/.package_complete"
  upload_folder "package-${key}" "${package_dir}" "packages/${key}"
  if [[ "${KEEP_PACKAGES}" == "NO" ]]; then
    find "${package_dir}" -maxdepth 1 -type f -name '*.tar.part-*' -delete
    event "Removed local generated tar parts for ${key}; source artifacts were untouched"
  fi
}

phase_docs() {
  upload_file docs-readme "${CAUSALWAM_ROOT}/docs/MODELSCOPE_DATASET_README.md" README.md
  upload_file docs-upload-manifest "${CAUSALWAM_ROOT}/docs/modelscope_upload_manifest_20260901.json" docs/modelscope_upload_manifest_20260901.json
  upload_file docs-h100-handoff "${CAUSALWAM_ROOT}/docs/H100_HANDOFF_20260901.md" docs/H100_HANDOFF_20260901.md
}

phase_models() {
  upload_file model-fastwam-release "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384.pt" artifacts/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
  upload_file model-dataset-stats "${MODEL_BASE}/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" artifacts/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
  upload_file model-t5 "${MODEL_BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" artifacts/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
  upload_file model-vae "${MODEL_BASE}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" artifacts/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
  upload_folder model-tokenizer "${MODEL_BASE}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" artifacts/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
}

phase_final() {
  upload_file pair280-final-checkpoint "${PAIR_RUN_ROOT}/formal/checkpoint.pt" artifacts/pair280_layer16_v1/seed1_c3_pair280_posttraining_v1/formal/checkpoint.pt
  upload_file pair280-final-completion "${PAIR_RUN_ROOT}/audits/formal_completion.json" artifacts/pair280_layer16_v1/seed1_c3_pair280_posttraining_v1/audits/formal_completion.json
  upload_folder_filtered pair280-final-sidecars "${PAIR_RUN_ROOT}/formal" artifacts/pair280_layer16_v1/seed1_c3_pair280_posttraining_v1/formal
}

phase_pair_cache() {
  upload_folder pair280-shards "${PAIR_ROOT}/shards" artifacts/pair280_layer16_v1/shards
  upload_folder pair280-text-cache "${PAIR_ROOT}/paired_text_cache" artifacts/pair280_layer16_v1/paired_text_cache
  local file
  for file in cache_manifest.json cache_input_audit.json cache_audit.json pair280_state_bank.json pair280_protocol.json pair280_release_binding.json; do
    upload_file "pair280-${file}" "${PAIR_ROOT}/${file}" "artifacts/pair280_layer16_v1/${file}"
  done
}

phase_official_text_cache() {
  local package
  package="$(make_path_archive official-text-cache "${CAUSALWAM_ROOT}" \
    '68,704 official prompt embeddings plus inventory and completion audit.' \
    FastWAM/outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache \
    FastWAM/outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache.audit.json \
    FastWAM/outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache.inventory.json)"
  upload_archive official-text-cache "${package}"
}

phase_train_data() {
  local package
  package="$(make_official_subset_archive)"
  upload_archive official-three-task-subset "${package}"
  package="$(make_path_archive native50hz-paired "${CAUSALWAM_ROOT}" \
    'Native 50Hz three-task paired C/R1/R2/R3 LeRobot v2.1 dataset and audits.' \
    FastWAM/outputs/policy_content_adapter/native50hz_three_task_rgb640x480_v1/full_lerobot_v21)"
  upload_archive native50hz-paired "${package}"
}

phase_assets() {
  local package
  package="$(make_path_archive robotwin-assets "${CAUSALWAM_ROOT}" \
    'RoboTwin runtime assets after the 2026-08-20 missing/corrupt asset repair audit.' \
    FastWAM/third_party/RoboTwin/assets)"
  upload_archive robotwin-assets "${package}"
}

phase_provenance() {
  upload_file provenance-official-cache-binding "${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/official_text_cache_binding_manifest.json" artifacts/FastWAM/outputs/policy_content_adapter/release_base_v1/official_text_cache_binding_manifest.json
  upload_file provenance-pmode-selection "${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1_retry1/p_mode_selection.json" artifacts/FastWAM/outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1_retry1/p_mode_selection.json
  upload_folder provenance-pv2-manifests "${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1/manifests" artifacts/FastWAM/outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1/manifests
  upload_file provenance-asset-repair "${FASTWAM_ROOT}/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1/asset_repair_author_tree_20260820/asset_repair_audit.json" artifacts/FastWAM/outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1_retry1/asset_repair_author_tree_20260820/asset_repair_audit.json
}

phase_resume() {
  upload_folder pair280-resume-step18215 "${PAIR_RUN_ROOT}/formal/checkpoints/state/step_00018215" artifacts/pair280_layer16_v1/seed1_c3_pair280_posttraining_v1/formal/checkpoints/state/step_00018215
}

preflight
if [[ "${PHASE}" == "preflight" ]]; then
  event "PREFLIGHT PASS; no upload started"
  exit 0
fi
[[ "${CONFIRM_UPLOAD}" == "YES" ]] || die "upload phase requires CONFIRM_UPLOAD=YES"

case "${PHASE}" in
  docs) phase_docs ;;
  models) phase_models ;;
  final) phase_final ;;
  pair_cache) phase_pair_cache ;;
  official_text_cache) phase_official_text_cache ;;
  train_data) phase_train_data ;;
  assets) phase_assets ;;
  provenance) phase_provenance ;;
  resume) phase_resume ;;
  train)
    phase_docs; phase_models; phase_pair_cache; phase_official_text_cache; phase_train_data; phase_provenance
    ;;
  eval)
    phase_docs; phase_models; phase_final; phase_assets; phase_provenance
    ;;
  all)
    phase_docs; phase_models; phase_final; phase_pair_cache; phase_official_text_cache; phase_train_data; phase_assets; phase_provenance
    ;;
esac

event "SUCCESS phase=${PHASE} repo=${REPO_ID} revision=${REVISION}"
