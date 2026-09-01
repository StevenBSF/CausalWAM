# Motus Policy Content Adapter

This port keeps the FastWAM method unchanged. M1 and M3 share the author
Motus_robotwin2 base, initialization, two data streams, step RNG, optimizer and
steps. M1 has `lambda_contrastive=0`; M3 has `0.1`.

## Data and method

- `outputs/policy_content_adapter/motus_v1/paired_observation_manifest.json`:
  720 physical states x C/R1/R2/R3 = 2,880 observations.
- `outputs/policy_content_adapter/motus_v1/official_three_task_manifest_v4.json`:
  1,650 episodes (50 Clean + 500 Random/task), 16,500 virtual samples/epoch.
  Native 50 Hz uses stride 5 to match Motus 10 Hz, 16 actions, 8 video frames
  and 1.6 s. No interpolation is used. State and future targets both come from
  raw LeRobot `action`, matching Motus `joint_action/vector -> qpos.pt`.

Layer-16 observation tokens enter an 8-query, 384-D Content Head. GCA injects
them after the Action Expert input encoder. The observation branch is frozen
and stops at Layer 16. Official batches use action flow matching; paired
batches use contrastive loss. M-P1 freezes Action Expert; M-P2 tunes it at
the configured learning rate.

## Formal five-epoch M-P2 profile

The formal M-P2 profile mirrors the author Motus optimizer and loader
settings: per-device batch 8, BF16, AdamW (`betas=[0.9,0.95]`, weight decay
0.01), gradient clip 0.5, the author linear scheduler (200 warmup steps,
5,000,000-step cycle, `f_max=0.99`, `f_min=0.4`), and LR `5e-5` for both the
new Head/GCA and the unfrozen Action Expert.  On eight GPUs the effective
global batch is 64.

The three-task dataset has 16,500 virtual samples.  Reproducing Motus's
`DistributedSampler` plus rank-local `drop_last=True` gives 257 optimizer
steps per epoch, so five epochs are exactly 1,285 steps and process 82,240
samples (4.984 effective full-dataset exposures).  Formal checkpoints are
saved every 257 steps so every completed epoch can be resumed exactly; this
is a reliability-only increase over the author's 5,000-step save interval.

## Artifact gate and smoke

CPFS is full, so generated audits/caches use rootfs by default:

```bash
export PYTHONPATH="$PWD"
CACHE_ROOT=/root/motus_policy_artifacts/motus_v1 \
bash experiments/robotwin/policy_content_adapter/run_artifact_gate.sh
```

The gate waits for GPU 0, then runs lineage, implementation audit, strict base
load, three text embeddings, zero-gate identity, and 720x4 Layer-16 extraction.
It never starts training. Once it passes:

```bash
OUT="$PWD/outputs/policy_content_adapter/motus_v1"
CACHE=/root/motus_policy_artifacts/motus_v1
PY=/root/anaconda3/envs/motus/bin/python
$PY -m experiments.robotwin.policy_content_adapter.materialize \
 --m1-template experiments/robotwin/policy_content_adapter/configs/m1_m_p1_smoke.yaml \
 --m3-template experiments/robotwin/policy_content_adapter/configs/m3_m_p1_smoke.yaml \
 --output-dir "$OUT/materialized/mp1_seed1_smoke" \
 --run-output-root "$OUT/runs/mp1_seed1_smoke" \
 --base-lineage "$CACHE/audits/motus_robotwin2_lineage.json" \
 --implementation-audit "$CACHE/audits/implementation_audit.json" \
 --strict-load-audit "$CACHE/audits/strict_load_audit.json" \
 --zero-gate-audit "$CACHE/audits/zero_gate_audit.json" \
 --official-manifest "$OUT/official_three_task_manifest_v4.json" \
 --paired-manifest "$OUT/paired_observation_manifest.json" \
 --token-cache "$CACHE/layer16_token_cache" \
 --task-text-cache "$CACHE/task_text_cache" \
 --regime m_p1 --training-seed 1 --world-size 8 \
 --per-device-batch 1 --paired-groups-per-device 2 \
 --gradient-accumulation-steps 1 --max-steps 3 --checkpoint-interval 2
CONFIG_DIR="$OUT/materialized/mp1_seed1_smoke" GPU_IDS=0,1,2,3,4,5,6,7 \
bash experiments/robotwin/policy_content_adapter/run_smoke_pair.sh
```

## Online evaluation

Formal outputs belong at `formal_runs/seed_{1,2,3}/{m1,m3}`. Prepare is CPU
only; `PHASE=all` later runs six GPUs and aggregates all 36 cells:

```bash
PHASE=prepare RUNS_ROOT="$OUT/formal_runs" \
bash experiments/robotwin/policy_content_adapter/run_rollout_matrix.sh
PHASE=all RUNS_ROOT="$OUT/formal_runs" GPU_IDS=0,1,2,4,5,6 \
bash experiments/robotwin/policy_content_adapter/run_rollout_matrix.sh
```

Every task/domain cell contains 100 RoboTwin episodes. Evaluation uses author
seed 42 and stock expert filtering. M1/M3 share the start seed, but exact
episode pairing is not claimed.
