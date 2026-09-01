#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:-}
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  printf 'Usage: %s {smoke|full}\n' "$0" >&2
  exit 64
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/fastwam-robotwin-bw/bin/python}
GPU_ID=${GPU_ID:-0}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/e2_e3/$MODE}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/third_party/RoboTwin/data}
MODEL_BASE=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints}
if [[ -z "${TASKS+x}" ]]; then
  if [[ "$MODE" == smoke ]]; then
    TASKS=place_a2b_left
  else
    TASKS=place_a2b_left,open_microwave,move_stapler_pad
  fi
fi
LAYERS=8,16,24
SEED=${SEED:-0}
TEMPERATURE=0.07
MIN_TEMPORAL_GAP=${MIN_TEMPORAL_GAP:-8}
MIN_STATE_DISTANCE=${MIN_STATE_DISTANCE:-1e-5}
RESUME=${RESUME:-1}
SMOKE_THEN_FULL=${SMOKE_THEN_FULL:-1}
CANONICAL_SMOKE_DIR="$REPO_ROOT/outputs/e2_e3/smoke"
CANONICAL_SMOKE_PROOF="$CANONICAL_SMOKE_DIR/canonical_smoke_proof.json"

if [[ "$MODE" == smoke ]]; then
  STATES_PER_TRAJECTORY=${STATES_PER_TRAJECTORY:-2}
  TRAIN_STEPS=${TRAIN_STEPS:-1}
  GROUPS_PER_BATCH=${GROUPS_PER_BATCH:-2}
  VAL_EVERY=${VAL_EVERY:-1}
  ALLOW_ARGS=(--allow-incomplete)
  TRAIN_IDS=(--content-ids 0)
  VAL_IDS=(--content-ids 30)
  TEST_IDS=(--content-ids 40)
else
  STATES_PER_TRAJECTORY=${STATES_PER_TRAJECTORY:-8}
  TRAIN_STEPS=${TRAIN_STEPS:-1000}
  GROUPS_PER_BATCH=${GROUPS_PER_BATCH:-8}
  VAL_EVERY=${VAL_EVERY:-50}
  ALLOW_ARGS=()
  TRAIN_IDS=()
  VAL_IDS=()
  TEST_IDS=()
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export PYTHONUNBUFFERED=1

CACHE_DIR="$RUN_DIR/cache"
SELECTION_METRICS="$RUN_DIR/selection_metrics"
SELECTION_DIR="$RUN_DIR/layer_selection"
TEST_METRICS="$RUN_DIR/test_metrics"
STATUS_DIR="$RUN_DIR/status"
LOG_DIR="$RUN_DIR/logs"
mkdir -p -- "$CACHE_DIR" "$SELECTION_METRICS" "$SELECTION_DIR" \
  "$RUN_DIR/e2" "$RUN_DIR/e3" "$TEST_METRICS" "$RUN_DIR/comparison" \
  "$STATUS_DIR" "$LOG_DIR"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
on_error() {
  local code=$?
  trap - ERR
  printf '%s\n' OPERATIONAL_ERROR >"$STATUS_DIR/state.txt"
  log "FAILED exit_code=$code line=${BASH_LINENO[0]} command=$BASH_COMMAND"
  exit "$code"
}

run_stage() {
  local stage=$1
  shift
  local marker="$STATUS_DIR/$stage.done"
  if [[ "$RESUME" == 1 && -f "$marker" ]]; then
    if validate_stage "$stage"; then
      log "SKIP validated stage=$stage"
      return 0
    fi
    log "completion marker for stage=$stage is stale or invalid"
    return 1
  fi
  log "START stage=$stage"
  "$@"
  validate_stage "$stage"
  printf '%s\n' "$(timestamp)" >"$marker"
  log "DONE stage=$stage"
}

validate_cache() {
  local path=$1
  local split=$2
  local mode=$3
  "$PYTHON_BIN" - "$path" "$split" "$mode" "$MODE" "$TASKS" \
    "$STATES_PER_TRAJECTORY" <<'PY'
import hashlib,sys
from experiments.robotwin.e0_e1.cache import load_cache
p=load_cache(sys.argv[1]); split=sys.argv[2]; mode=sys.argv[3]
run_mode=sys.argv[4]; tasks=tuple(sys.argv[5].split(',')); states=int(sys.argv[6])
assert p['provenance']['protocol']=='r3_holdout_v1'
assert p['provenance']['split']==split
assert p['provenance']['proprio_mode']==mode
assert tuple(p['provenance']['tasks'])==tasks
assert {r['task'] for r in p['records']}==set(tasks)
expected=('clean','style_00_seed_0','style_01_seed_1') if split!='test' else ('clean','style_02_seed_2')
assert tuple(p['variant_names'])==expected
assert {r['split'] for r in p['records']}=={split}
assert {r['variant'] for r in p['records']}==set(expected)
if split in ('train','val'):
    assert 'style_02_seed_2' not in {r['variant'] for r in p['records']}
    for key in ('manifest_jsonl','manifest_csv'):
        assert 'style_02_seed_2' not in open(p['provenance'][key], encoding='utf-8').read()
else:
    assert p['provenance']['decision_lock_created_before_test'] is True
expected_ids={
    'train': ({0} if run_mode=='smoke' else set(range(0,30))),
    'val': ({30} if run_mode=='smoke' else set(range(30,40))),
    'test': ({40} if run_mode=='smoke' else set(range(40,50))),
}[split]
groups={}
for r in p['records']:
    groups.setdefault((r['task'],int(r['content_id'])),set()).add(r['physical_state_id'])
for task in tasks:
    assert {content for record_task,content in groups if record_task==task}==expected_ids
    assert all(len(groups[(task,content)])==states for content in expected_ids)
assert len(p['records'])==len(tasks)*len(expected_ids)*states*len(expected)
assert len(p['physical_states'])==len(tasks)*len(expected_ids)*states
if run_mode=='smoke':
    assert p['provenance']['allow_incomplete'] is True
    assert p['provenance']['content_ids']==sorted(expected_ids)
else:
    assert p['provenance']['allow_incomplete'] is False
    assert p['provenance']['content_ids'] is None
if split in ('train','val'):
    assert set(p['tokens_by_layer'])=={'8','16','24'}
manifest=open(p['provenance']['manifest_jsonl'],'rb').read()
assert hashlib.sha256(manifest).hexdigest()==p['provenance']['source_manifest_sha256']
PY
}

validate_metric_file() {
  local path=$1
  local split=$2
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$split" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['schema_version']==2
assert p['protocol']=='r3_holdout_v1' and p['evaluation_split']==sys.argv[2]
assert p['metrics']
PY
}

validate_training() {
  local experiment=$1
  local checkpoint="$RUN_DIR/${experiment,,}/${experiment,,}_best_content_head.pt"
  "$PYTHON_BIN" - "$checkpoint" "$experiment" "$(selected_layer)" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
assert p['schema_version']==2 and p['experiment']==sys.argv[2]
assert p['checkpoint_kind']=='best_val' and p['step']==p['best_step']
assert p['layer']==int(sys.argv[3]) and p['best_metric']['r3_used'] is False
PY
  [[ -s "$RUN_DIR/${experiment,,}/train_log.csv" ]]
  [[ -s "$RUN_DIR/${experiment,,}/training_curves.svg" ]]
}

validate_e2_e3_equivalence() {
  "$PYTHON_BIN" - "$RUN_DIR/e2/e2_best_content_head.pt" \
    "$RUN_DIR/e3/e3_best_content_head.pt" <<'PY'
import sys,torch
a=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
b=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
assert a['controlled_training_config_sha256']==b['controlled_training_config_sha256']
assert a['initial_head_sha256']==b['initial_head_sha256']
for split in ('train','val'):
    assert a[f'{split}_scientific_cache_contract']==b[f'{split}_scientific_cache_contract']
assert a['proprio_mode']=='observed'
assert b['proprio_mode']=='constant_zero_normalized'
PY
}

write_run_config() {
  "$PYTHON_BIN" - "$RUN_DIR/run_config.json" "$MODE" "$GPU_ID" "$TASKS" \
    "$LAYERS" "$STATES_PER_TRAJECTORY" "$TRAIN_STEPS" "$GROUPS_PER_BATCH" \
    "$VAL_EVERY" "$SEED" "$MIN_TEMPORAL_GAP" "$MIN_STATE_DISTANCE" \
    "$DATA_ROOT" "$MODEL_BASE" "$SCRIPT_DIR" "$CANONICAL_SMOKE_PROOF" <<'PY'
import hashlib,json,os,sys,tempfile
from pathlib import Path
from experiments.robotwin.e0_e1.decision_lock_e2e3 import strong_file_identity
(output,mode,gpu,tasks,layers,states,steps,groups,val_every,seed,gap,distance,
 data_root,model_base,script_dir,smoke_proof)=sys.argv[1:]
script_dir=Path(script_dir).resolve()
code_names=(
    'audit_e2e3.py','backbone.py','cache.py','compare_e2e3.py','data.py',
    'decision_lock_e2e3.py','evaluate_e2e3.py','extract.py','head.py',
    'io_utils.py','metrics.py','negatives.py','prompts.py','select_layer_e2.py',
    'smoke_proof_e2e3.py','train_e2e3.py',
    'run_e2_e3.sh',
)
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
model_base=Path(model_base).resolve()
checkpoint=model_base/'fastwam_release/robotwin_uncond_3cam_384.pt'
stats=model_base/'fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json'
value={
    'schema_version':1,'protocol':'r3_holdout_v1','mode':mode,
    'gpu_id':int(gpu),'tasks':tasks.split(','),'layers':[int(x) for x in layers.split(',')],
    'states_per_trajectory':int(states),'train_steps':int(steps),
    'groups_per_batch':int(groups),'val_every':int(val_every),'seed':int(seed),
    'temperature':0.07,'min_temporal_gap':int(gap),'min_state_distance':float(distance),
    'data_root':str(Path(data_root).resolve()),'model_base':str(model_base),
    'checkpoint':{'path':str(checkpoint),'size_bytes':checkpoint.stat().st_size,
                  'mtime_ns':checkpoint.stat().st_mtime_ns,'sha256':digest(checkpoint)},
    'dataset_stats':{'path':str(stats),'size_bytes':stats.stat().st_size,
                     'mtime_ns':stats.stat().st_mtime_ns,'sha256':digest(stats)},
    'canonical_smoke_proof':(
        strong_file_identity(smoke_proof) if mode=='full' else None
    ),
    'experiment_code_sha256':{name:digest(script_dir/name) for name in code_names},
}
destination=Path(output).resolve()
if destination.exists():
    previous=json.loads(destination.read_text())
    if previous != value:
        raise SystemExit('resume configuration/code differs from immutable run_config.json')
else:
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.run_config.',suffix='.tmp',dir=destination.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,destination)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
PY
}

validate_stage() {
  local stage=$1
  local padded
  case "$stage" in
    e2_val) validate_cache "$CACHE_DIR/e2_val.pt" val observed ;;
    e2_layer_metrics)
      validate_metric_file "$SELECTION_METRICS/e2_rawbackbone_layer_08.json" val
      validate_metric_file "$SELECTION_METRICS/e2_rawbackbone_layer_16.json" val
      validate_metric_file "$SELECTION_METRICS/e2_rawbackbone_layer_24.json" val ;;
    e2_layer_selection)
      [[ -s "$SELECTION_DIR/selection.json" ]] && selected_layer >/dev/null ;;
    e2_train) validate_cache "$CACHE_DIR/e2_train.pt" train observed ;;
    e2_head) validate_training E2 ;;
    e3_train) validate_cache "$CACHE_DIR/e3_train.pt" train constant_zero_normalized ;;
    e3_val) validate_cache "$CACHE_DIR/e3_val.pt" val constant_zero_normalized ;;
    e3_head) validate_training E3 ;;
    decision_lock)
      "$PYTHON_BIN" - "$RUN_DIR/decision_lock.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['schema_version']==1
assert p['protocol']=='r3_holdout_v1' and p['r3_access_before_lock'] is False
assert set(p['checkpoints'])=={'E2','E3'}
PY
      ;;
    e2_test) validate_cache "$CACHE_DIR/e2_test.pt" test observed ;;
    e3_test) validate_cache "$CACHE_DIR/e3_test.pt" test constant_zero_normalized ;;
    e2_controls)
      printf -v padded '%02d' "$(selected_layer)"
      validate_metric_file "$TEST_METRICS/e2_rawbackbone_layer_${padded}.json" test
      validate_metric_file "$TEST_METRICS/e2_inithead_layer_${padded}.json" test
      validate_metric_file "$TEST_METRICS/e2_trainedhead_layer_${padded}.json" test ;;
    e3_controls)
      printf -v padded '%02d' "$(selected_layer)"
      validate_metric_file "$TEST_METRICS/e3_noproprio_rawbackbone_layer_${padded}.json" test
      validate_metric_file "$TEST_METRICS/e3_noproprio_inithead_layer_${padded}.json" test
      validate_metric_file "$TEST_METRICS/e3_noproprio_trainedhead_layer_${padded}.json" test ;;
    comparison) [[ -s "$RUN_DIR/comparison/comparison.json" && -s "$RUN_DIR/comparison/summary.md" ]] ;;
    final_audit)
      [[ -s "$RUN_DIR/protocol_audit.json" && -s "$RUN_DIR/deliverables.json" ]] && \
      "$PYTHON_BIN" -m experiments.robotwin.e0_e1.audit_e2e3 \
        --run-dir "$RUN_DIR"
      ;;
    *) return 1 ;;
  esac
}

extract_seen() {
  local experiment=$1
  local mode=$2
  local split=$3
  local layer_values=$4
  local -a id_args=()
  case "$split" in
    train) id_args=("${TRAIN_IDS[@]}") ;;
    val) id_args=("${VAL_IDS[@]}") ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.robotwin.e0_e1.extract \
    --data-root "$DATA_ROOT" --tasks "$TASKS" --split "$split" \
    --checkpoint "$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt" \
    --dataset-stats "$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "$MODEL_BASE" \
    --states-per-trajectory "$STATES_PER_TRAJECTORY" --layers "$layer_values" \
    --protocol r3_holdout_v1 --proprio-mode "$mode" --device cuda \
    "${ALLOW_ARGS[@]}" "${id_args[@]}" \
    --output "$CACHE_DIR/${experiment,,}_${split}.pt"
}

evaluate_e2_layers() {
  local layer
  for layer in 8 16 24; do
    "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate_e2e3 \
      --cache "$CACHE_DIR/e2_val.pt" --layer "$layer" \
      --experiment E2-RawBackbone --output-dir "$SELECTION_METRICS" \
      --min-temporal-gap "$MIN_TEMPORAL_GAP" \
      --min-state-distance "$MIN_STATE_DISTANCE"
  done
}

select_layer() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.select_layer_e2 \
    --metrics "$SELECTION_METRICS/e2_rawbackbone_layer_08.json" \
      "$SELECTION_METRICS/e2_rawbackbone_layer_16.json" \
      "$SELECTION_METRICS/e2_rawbackbone_layer_24.json" \
    --output-dir "$SELECTION_DIR"
}

selected_layer() {
  local layer
  layer=$(tr -d '[:space:]' <"$SELECTION_DIR/selected_layer.txt")
  [[ "$layer" == 8 || "$layer" == 16 || "$layer" == 24 ]] || return 1
  printf '%s' "$layer"
}

validate_selection_strict() {
  "$PYTHON_BIN" - "$RUN_DIR" "$TASKS" "$CACHE_DIR/e2_val.pt" <<'PY'
import sys
from pathlib import Path
from experiments.robotwin.e0_e1.audit_e2e3 import _validate_selection
from experiments.robotwin.e0_e1.decision_lock_e2e3 import strong_file_identity
root=Path(sys.argv[1]).resolve()
_validate_selection(
    root,
    e2_val_identity=strong_file_identity(sys.argv[3]),
    tasks=tuple(sys.argv[2].split(',')),
    memo={},
)
PY
}

train_head() {
  local experiment=$1
  local mode=$2
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.robotwin.e0_e1.train_e2e3 \
    --experiment "$experiment" --proprio-mode "$mode" \
    --train-cache "$CACHE_DIR/${experiment,,}_train.pt" \
    --val-cache "$CACHE_DIR/${experiment,,}_val.pt" \
    --layer "$(selected_layer)" --steps "$TRAIN_STEPS" \
    --groups-per-batch "$GROUPS_PER_BATCH" --val-every "$VAL_EVERY" \
    --temperature "$TEMPERATURE" --seed "$SEED" --device cuda \
    --min-temporal-gap "$MIN_TEMPORAL_GAP" \
    --min-state-distance "$MIN_STATE_DISTANCE" \
    --output-dir "$RUN_DIR/${experiment,,}"
}

create_lock() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.decision_lock_e2e3 \
    --selection "$SELECTION_DIR/selection.json" \
    --e2-checkpoint "$RUN_DIR/e2/e2_best_content_head.pt" \
    --e3-checkpoint "$RUN_DIR/e3/e3_best_content_head.pt" \
    --e2-test-output "$CACHE_DIR/e2_test.pt" \
    --e3-test-output "$CACHE_DIR/e3_test.pt" \
    --output "$RUN_DIR/decision_lock.json"
}

extract_test() {
  local experiment=$1
  local mode=$2
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.robotwin.e0_e1.extract \
    --data-root "$DATA_ROOT" --tasks "$TASKS" --split test \
    --checkpoint "$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt" \
    --dataset-stats "$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
    --model-base-path "$MODEL_BASE" \
    --states-per-trajectory "$STATES_PER_TRAJECTORY" --layers "$(selected_layer)" \
    --protocol r3_holdout_v1 --proprio-mode "$mode" \
    --decision-lock "$RUN_DIR/decision_lock.json" --device cuda \
    "${ALLOW_ARGS[@]}" "${TEST_IDS[@]}" \
    --output "$CACHE_DIR/${experiment,,}_test.pt"
}

evaluate_controls() {
  local experiment=$1
  local cache="$CACHE_DIR/${experiment,,}_test.pt"
  local checkpoint="$RUN_DIR/${experiment,,}/${experiment,,}_best_content_head.pt"
  local prefix
  if [[ "$experiment" == E2 ]]; then
    prefix=E2
  else
    prefix=E3-NoProprio
  fi
  local control
  for control in RawBackbone InitHead TrainedHead; do
    "$PYTHON_BIN" -m experiments.robotwin.e0_e1.evaluate_e2e3 \
      --cache "$cache" --layer "$(selected_layer)" \
      --experiment "$prefix-$control" --head-checkpoint "$checkpoint" \
      --decision-lock "$RUN_DIR/decision_lock.json" --seed "$SEED" \
      --device cpu --min-temporal-gap "$MIN_TEMPORAL_GAP" \
      --min-state-distance "$MIN_STATE_DISTANCE" --output-dir "$TEST_METRICS"
  done
}

compare_results() {
  local padded
  printf -v padded '%02d' "$(selected_layer)"
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.compare_e2e3 \
    --metrics \
      "$TEST_METRICS/e2_rawbackbone_layer_${padded}.json" \
      "$TEST_METRICS/e2_inithead_layer_${padded}.json" \
      "$TEST_METRICS/e2_trainedhead_layer_${padded}.json" \
      "$TEST_METRICS/e3_noproprio_rawbackbone_layer_${padded}.json" \
      "$TEST_METRICS/e3_noproprio_inithead_layer_${padded}.json" \
      "$TEST_METRICS/e3_noproprio_trainedhead_layer_${padded}.json" \
    --e1-metric "$REPO_ROOT/outputs/e0_e1/full/test_metrics/e1_trainedhead_layer_16.json" \
    --output-dir "$RUN_DIR/comparison"
}

run_final_audit() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.audit_e2e3 --run-dir "$RUN_DIR"
}

validate_completed_run_read_only() {
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.audit_e2e3 \
    --run-dir "$RUN_DIR" --require-success-marker --read-only
}

cd -- "$REPO_ROOT"
[[ -x "$PYTHON_BIN" ]] || { log "missing Python: $PYTHON_BIN"; exit 1; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { log "GPU_ID must be one integer"; exit 1; }
[[ -d "$DATA_ROOT" ]] || { log "missing data root: $DATA_ROOT"; exit 1; }
[[ -r "$MODEL_BASE/fastwam_release/robotwin_uncond_3cam_384.pt" ]] || {
  log "missing FastWAM checkpoint"; exit 1;
}
exec 9>"$RUN_DIR/.run.lock"
if ! flock -n 9; then
  printf 'Another E2/E3 runner holds %s/.run.lock\n' "$RUN_DIR" >&2
  exit 1
fi
if [[ -e "$STATUS_DIR/SUCCESS" ]]; then
  if [[ ! -s "$STATUS_DIR/SUCCESS" || ! -s "$STATUS_DIR/final_audit.done" || \
        ! -s "$STATUS_DIR/state.txt" || \
        "$(tr -d '[:space:]' <"$STATUS_DIR/state.txt")" != SUCCESS ]]; then
    log "inconsistent completed-run markers; refusing to modify immutable run"
    exit 1
  fi
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.audit_e2e3 \
    --run-dir "$RUN_DIR" --require-success-marker --read-only
  log "SUCCESS already complete and strictly revalidated; no artifacts were changed"
  exit 0
fi
RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$RUN_LOG") 2>&1
trap on_error ERR
trap 'printf "%s\n" INTERRUPTED >"$STATUS_DIR/state.txt"; exit 130' INT TERM HUP
if [[ "$MODE" == full && "$SMOKE_THEN_FULL" == 1 ]]; then
  SMOKE_THEN_FULL=0 RUN_DIR="$CANONICAL_SMOKE_DIR" GPU_ID="$GPU_ID" \
    PYTHON_BIN="$PYTHON_BIN" DATA_ROOT="$DATA_ROOT" \
    DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE" \
    bash "$SCRIPT_DIR/run_e2_e3.sh" smoke
fi
if [[ "$MODE" == full ]]; then
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.audit_e2e3 \
    --run-dir "$CANONICAL_SMOKE_DIR" --require-success-marker --read-only
  "$PYTHON_BIN" -m experiments.robotwin.e0_e1.smoke_proof_e2e3 \
    --smoke-run-dir "$CANONICAL_SMOKE_DIR" \
    --output "$CANONICAL_SMOKE_PROOF"
fi
write_run_config
if [[ -e "$RUN_DIR/decision_lock.json" && ! -f "$STATUS_DIR/decision_lock.done" ]]; then
  log "orphan decision lock exists; refusing to overwrite immutable lock"
  exit 1
fi

printf '%s\n' RUNNING >"$STATUS_DIR/state.txt"
log "mode=$MODE run_dir=$RUN_DIR gpu=$GPU_ID tasks=$TASKS"
run_stage e2_val extract_seen E2 observed val "$LAYERS"
run_stage e2_layer_metrics evaluate_e2_layers
run_stage e2_layer_selection select_layer
validate_selection_strict
log "E2 selected layer=$(selected_layer); E3 is locked to the same layer"
run_stage e2_train extract_seen E2 observed train "$LAYERS"
run_stage e2_head train_head E2 observed
run_stage e3_train extract_seen E3 constant_zero_normalized train "$LAYERS"
run_stage e3_val extract_seen E3 constant_zero_normalized val "$LAYERS"
run_stage e3_head train_head E3 constant_zero_normalized
validate_e2_e3_equivalence
validate_selection_strict
run_stage decision_lock create_lock
run_stage e2_test extract_test E2 observed
run_stage e3_test extract_test E3 constant_zero_normalized
run_stage e2_controls evaluate_controls E2
run_stage e3_controls evaluate_controls E3
run_stage comparison compare_results
run_stage final_audit run_final_audit

printf '%s\n' SUCCESS >"$STATUS_DIR/state.txt"
printf '%s\n' "$(timestamp)" >"$STATUS_DIR/SUCCESS"
log "SUCCESS summary=$RUN_DIR/comparison/summary.md"
