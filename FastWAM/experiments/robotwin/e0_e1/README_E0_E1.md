# E0/E1 FastWAM representation validation

This is an offline, minimal experiment. It never changes the FastWAM policy,
action expert, action loss, or rollout path.

## Questions

- E0 asks whether the frozen, released FastWAM current-observation video
  representation changes too much when only the background texture changes.
- E1 asks whether a small trainable content head can reduce that sensitivity
  while preserving discrimination between different physical states.

The code intentionally stops after these questions. It does not implement
LOVO, SIVA, a router, a teacher, a momentum encoder, or a memory bank.

## Required data contract

Each physical trajectory must have exactly:

```text
content_XXXXXX/
  COMPLETE.json
  clean/
  style_00_seed_0/       # R1
  style_01_seed_1/       # R2
  style_02_seed_2/       # R3
```

The reader fails closed unless metadata, initial state, action trace, state
trace, every non-RGB HDF5 leaf, camera matrices, frame map, robot state, and
object/task state prove exact correspondence. The sample key is
`(task, content_id, saved_frame_idx)`; `trace_idx` is stored separately because
multiple saved frames may legitimately map to the same physics step.

The fixed physical-trajectory split is content IDs 0–29 train, 30–39 val, and
40–49 test. Clean/R1/R2/R3 never cross splits. Formal mode requires all 50
content IDs plus the canonical final validator report/manifests. `--allow-incomplete`
and `--content-ids` are smoke-only controls.

## Representation

Input preprocessing exactly follows RoboTwin deployment: head RGB is resized
to 256×320, left/right wrist RGB to 128×160, then composed into one 384×320
uint8 current observation. It follows deployment's exact operation order by
casting to the model dtype before applying `2/255-1`. Current 14-D proprio is
normalized with the released stats. A deterministic task-level prompt is held
fixed across paired renderings.

The extractor runs current image → deterministic VAE mean → video `pre_dit` at
timestep zero → the exact video-prefill block computations. It never creates a
future video, future ground truth, noisy future token, action denoising step, or
rollout. Candidate layers are one-based video blocks 8, 16, and 24; every layer
uses float32 mean pooling over its 120 spatial tokens followed by L2
normalization.

## Environment

Run from the FastWAM repository root. The explicit `PYTHONPATH` is mandatory:
the installed editable package on this machine points at a sibling checkout.

```bash
cd /mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM
export PYTHONPATH="$PWD/src:$PWD"
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/cpfs-E/baoshifeng/FastWAM/checkpoints
PY=/root/anaconda3/envs/fastwam-robotwin-bw/bin/python
RUN=outputs/e0_e1/full
TASKS=place_a2b_left,open_microwave,move_stapler_pad
```

Do not start the formal commands until the three paired roots contain their
canonical `validation_report.json`, `valid_variants.jsonl`, and
`split_manifests/` outputs.

For an unattended, strictly ordered full run, use the checked runner. It
serializes extraction on one GPU, selects the layer from E0 validation only,
trains E1, then extracts and evaluates test once. It preserves a scientific
FAIL report with exit code 2. The default run needs at least 20 GiB free space.

```bash
GPU_ID=0 bash experiments/robotwin/e0_e1/run_full_e0_e1.sh
```

It is safe to rerun the same command after an interruption: `RESUME=1` is the
default and reuses only completed stages whose artifacts pass strict validation.
Extraction and training resume at stage boundaries, not within a split or an
optimizer step. To intentionally start a different configuration, use a new
directory, for example `RUN_DIR="$PWD/outputs/e0_e1/full_seed1" SEED=1 ...`.

The combined log is written to `outputs/e0_e1/full/logs/`. A successful run
creates `status/SUCCESS`; scientific criteria not being met creates
`status/SCIENTIFIC_FAIL` while retaining the comparison reports; operational
errors create `status/OPERATIONAL_ERROR`. During training, diagnostics are
printed at step 1 and every `VAL_EVERY` steps (50 by default). To check setup
without loading a model or touching the GPU, run the same command with
`PREFLIGHT_ONLY=1`.

## E0

Extract the three split caches. Set `CUDA_VISIBLE_DEVICES` to a GPU confirmed
free by `nvidia-smi`; the `0` below is only an example. Do not share a GPU with
any active RoboTwin collection process.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks "$TASKS" --split train --states-per-trajectory 8 \
  --layers 8,16,24 --device cuda --output "$RUN/cache/train.pt"

CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks "$TASKS" --split val --states-per-trajectory 8 \
  --layers 8,16,24 --device cuda --output "$RUN/cache/val.pt"

CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks "$TASKS" --split test --states-per-trajectory 8 \
  --layers 8,16,24 --device cuda --output "$RUN/cache/test.pt"
```

Evaluate every candidate on validation trajectories and select one E1 layer
using ratio, retrieval, and interface—not a single absolute cosine value. Do
not inspect the test cache while choosing the layer:

```bash
for layer in 8 16 24; do
  "$PY" -m experiments.robotwin.e0_e1.evaluate \
    --cache "$RUN/cache/val.pt" --layer "$layer" \
    --experiment E0-RawBackbone --output-dir "$RUN/selection_metrics"
done
```

## E1

Assume E0 selected layer 16 below; replace it with the evidence-based choice.
The head is `Linear(3072,384)` + 8 queries + one standard 8-head cross-attention
layer + query mean + `384→384→384` SiLU MLP + L2. Only its 2,070,144 parameters
are trainable. Training consumes detached, cached backbone tokens, so no FastWAM
parameter can receive a gradient.

Before any optimizer step, record the deterministic random-head control on the
selected validation layer:

```bash
"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$RUN/cache/val.pt" --layer 16 \
  --experiment E1-InitHead --seed 0 --output-dir "$RUN/selection_metrics"
```

Start with 1,000 steps; after completion inspect train/val loss,
positive/negative cosine,
embedding norm, state spread, and gradient norm before extending to 3k–5k:

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.train_e1 \
  --train-cache "$RUN/cache/train.pt" --val-cache "$RUN/cache/val.pt" \
  --layer 16 --steps 1000 --groups-per-batch 8 --temperature 0.07 \
  --device cuda --output-dir "$RUN/e1"
```

Only after the layer and training procedure are fixed, evaluate all three
controls once on the held-out test trajectories. The deterministic InitHead
hash is checked against the initialization recorded by training:

```bash
"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$RUN/cache/test.pt" --layer 16 \
  --experiment E0-RawBackbone --output-dir "$RUN/test_metrics"

"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$RUN/cache/test.pt" --layer 16 \
  --experiment E1-InitHead --seed 0 --output-dir "$RUN/test_metrics"

CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache "$RUN/cache/test.pt" --layer 16 \
  --experiment E1-TrainedHead --head-checkpoint "$RUN/e1/e1_content_head.pt" \
  --device cuda --output-dir "$RUN/test_metrics"
```

Compare all three controls:

```bash
"$PY" -m experiments.robotwin.e0_e1.compare \
  --metrics \
    "$RUN/test_metrics/e0_rawbackbone_layer_16.json" \
    "$RUN/test_metrics/e1_inithead_layer_16.json" \
    "$RUN/test_metrics/e1_trainedhead_layer_16.json" \
  --min-state-retention 0.90 --require-success \
  --output-dir "$RUN/comparison"
```

The comparison refuses validation metrics, mismatched test caches, negative
filters, or an `E1-InitHead` that is not the exact random initialization used
by training.
With `--require-success`, it exits nonzero unless trained-vs-init style distance
decreases, state distance retains at least 90%, ratio increases, and Retrieval@1
increases for every task and the macro average.

## Metric interpretation

- Style distance: mean `1-cos(clean, Rk)` for the same physical state. Lower is
  better; R1/R2/R3 are also reported separately.
- State distance: `1-cos(clean_i, clean_j)` for physically distinct states in
  the same task. Same-trajectory far timesteps are preferred and named
  robot/object state filters reject overly close negatives. It must not collapse.
- State/style ratio: state distance divided by style distance plus epsilon.
  Higher is better.
- Retrieval@1: each R1/R2/R3 query retrieves the exact synchronized clean state
  from the same-task clean gallery. Higher is better.

E1 is promising only when its trained head improves over `E1-InitHead`, lowers
style sensitivity, retains state distance, and improves the ratio/retrieval.

## Outputs

- `cache/*.pt`: frozen tokens, normalized raw pooling, identities, named states,
  and full extraction provenance.
- `cache/manifests/paired_frames_{train,val,test}.{jsonl,csv}`: sampled
  synchronized physical timestep manifests.
- `selection_metrics/*_layer_*.json|csv`: validation-only candidate-layer
  metrics; each JSON records `evaluation_split: val`.
- `test_metrics/*_layer_*.json|csv`: final per-task and macro-average E0/E1
  metrics; each JSON records `evaluation_split: test`.
- `e1/e1_content_head.pt`: head-only checkpoint.
- `e1/train_log.json|csv`: loss/similarity/norm/gradient diagnostics.
- `comparison/comparison.json|csv` and `comparison/summary.md`: final comparison.

## Complete smoke example

While full collection is still active, use one already-published Place
trajectory from each fixed split. A valid metric smoke needs at least two
physical states, hence `--states-per-trajectory 2`. Only extraction needs the
FastWAM GPU; evaluation and this one-step head smoke run on CPU. The native
prefill check is enabled once to prove that the instrumented hidden-state path
matches the unmodified model output.

The complete sequence below is also available as a one-command runner:

```bash
GPU_ID=0 bash experiments/robotwin/e0_e1/run_place_smoke.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks place_a2b_left --split train --states-per-trajectory 2 \
  --allow-incomplete --content-ids 0 --layers 8,16,24 --device cuda \
  --verify-native-prefill \
  --output outputs/e0_e1/smoke/cache/train.pt

CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks place_a2b_left --split val --states-per-trajectory 2 \
  --allow-incomplete --content-ids 30 --layers 8,16,24 --device cuda \
  --output outputs/e0_e1/smoke/cache/val.pt

CUDA_VISIBLE_DEVICES=0 "$PY" -m experiments.robotwin.e0_e1.extract \
  --tasks place_a2b_left --split test --states-per-trajectory 2 \
  --allow-incomplete --content-ids 40 --layers 8,16,24 --device cuda \
  --output outputs/e0_e1/smoke/cache/test.pt

for layer in 8 16 24; do
  "$PY" -m experiments.robotwin.e0_e1.evaluate \
    --cache outputs/e0_e1/smoke/cache/val.pt --layer "$layer" \
    --experiment E0-RawBackbone \
    --output-dir outputs/e0_e1/smoke/selection_metrics
done

"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache outputs/e0_e1/smoke/cache/val.pt --layer 16 \
  --experiment E1-InitHead --seed 0 \
  --output-dir outputs/e0_e1/smoke/selection_metrics

"$PY" -m experiments.robotwin.e0_e1.train_e1 \
  --train-cache outputs/e0_e1/smoke/cache/train.pt \
  --val-cache outputs/e0_e1/smoke/cache/val.pt --layer 16 \
  --steps 1 --groups-per-batch 2 --val-every 1 --seed 0 --device cpu \
  --output-dir outputs/e0_e1/smoke/e1

"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache outputs/e0_e1/smoke/cache/test.pt --layer 16 \
  --experiment E0-RawBackbone \
  --output-dir outputs/e0_e1/smoke/test_metrics

"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache outputs/e0_e1/smoke/cache/test.pt --layer 16 \
  --experiment E1-InitHead --seed 0 \
  --output-dir outputs/e0_e1/smoke/test_metrics

"$PY" -m experiments.robotwin.e0_e1.evaluate \
  --cache outputs/e0_e1/smoke/cache/test.pt --layer 16 \
  --experiment E1-TrainedHead \
  --head-checkpoint outputs/e0_e1/smoke/e1/e1_content_head.pt \
  --device cpu --output-dir outputs/e0_e1/smoke/test_metrics

"$PY" -m experiments.robotwin.e0_e1.compare \
  --metrics \
    outputs/e0_e1/smoke/test_metrics/e0_rawbackbone_layer_16.json \
    outputs/e0_e1/smoke/test_metrics/e1_inithead_layer_16.json \
    outputs/e0_e1/smoke/test_metrics/e1_trainedhead_layer_16.json \
  --min-state-retention 0.90 --output-dir outputs/e0_e1/smoke/comparison
```

The smoke comparison deliberately omits `--require-success`: one optimizer
step checks wiring and gradients, not the scientific success criterion.

## E0/E1/E2/E3 definitions

- **E0** diagnoses the frozen raw FastWAM current-observation video
  representation. It trains nothing.
- **E1** validates the fixed-capacity contrastive content head when all three
  random-background styles R1/R2/R3 are seen during training.
- **E2** tests unseen-style generalization. Training, validation, layer
  selection and best-checkpoint selection use only Clean/R1/R2. R3 is opened
  only on test trajectories 40--49, after the layer and both checkpoints have
  been frozen in a decision-lock artifact.
- **E3** repeats E2 with the same layer, head initialization, head capacity,
  loss, optimizer, steps, splits and checkpoint rule. Its only intervention is
  `constant_zero_normalized`: the released processor first normalizes the real
  14-D state for audit, then an exact all-zero tensor of the same shape/dtype is
  passed through the original proprio encoder and one-token append path. The
  proprio token and attention structure therefore remain intact, but no
  sample-specific proprio value reaches the video representation.

E2/E3 use protocol `r3_holdout_v1`. The data layer materializes exactly
`Clean/R1/R2` for train/val and exactly `Clean/R3` for test. Cache schema v2
records its active variants and fails closed on an R3 train/val record. E3 has
an independent cache identity and requires one identical effective-proprio
hash across all physical states.

The content head and objective remain exactly the E1 definitions: 384 hidden
dimensions, eight learnable content queries, one eight-head cross-attention
layer, 384-D normalized output, multi-positive SupCon temperature 0.07, and the
same physical-state negative filter. FastWAM, ActionDiT, native flow-matching
losses and rollout are untouched.

Run the required wiring smoke first:

```bash
GPU_ID=0 bash experiments/robotwin/e0_e1/run_e2_e3.sh smoke
```

Then run the formal three-task experiment (the runner is resumable):

```bash
GPU_ID=0 bash experiments/robotwin/e0_e1/run_e2_e3.sh full
```

`full` automatically runs and requires the canonical strict smoke first. Setting
`SMOKE_THEN_FULL=0` skips only execution: the runner still read-only re-audits
`outputs/e2_e3/smoke`, creates or verifies its immutable
`canonical_smoke_proof.json`, and binds that proof's strong identity into the
full `run_config.json`. Thus a formal success is impossible without a current,
strictly audited canonical smoke. Re-running a completed directory performs a
read-only terminal re-audit and exits; inconsistent or stale success/state/audit
markers fail closed without deleting or rewriting the run.

The formal order is enforced: E2 validation extraction at layers 8/16/24,
seen-style raw evaluation and layer selection; E2/E3 train/validation
extraction and best-validation head training; immutable decision lock; only
then E2/E3 Clean/R3 test extraction and all six Raw/Init/Trained controls.
Outputs live under `outputs/e2_e3/{smoke,full}` and include manifests, separate
observed/no-proprio caches, selection JSON/CSV/Markdown, both best/final head
checkpoints, train/validation curves in CSV/JSON plus `training_curves.svg`,
decision lock, per-control metric JSON/CSV, and the final E1/E2/E3 comparison
JSON/CSV/Markdown. The terminal audit writes `protocol_audit.json` and
`deliverables.json` before `status/SUCCESS` is allowed.
