# Strict random-background pairing audit

This document records the Stage-1 audit performed before changing the copied
RoboTwin tree.  The original tree at
`/mnt/cpfs-E/baoshifeng/FastWAM` was read only.  All implementation and data
generation are confined to
`/mnt/cpfs-E/baoshifeng/CausalWAM/FastWAM`.

## Audited baseline

- FastWAM commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Nested RoboTwin commit: `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`
- The source trees were already dirty before this work.  In particular,
  RoboTwin had local changes in `README.md`, `assets/_download.py`,
  `envs/_base_task.py`, `script/_install.sh`, and `script/eval_policy.py`.
  Those pre-existing changes were copied, not created by this work.
- Baseline SHA-256 of the audited source `envs/_base_task.py`:
  `a4fa65537d2ba34206b89980b031e0bdc543561e01b0b1a0ae735617d577d2e7`.
- No existing `grab_roller` seed list, planned trajectory pickle, HDF5 episode,
  or reusable clean episode was found.

## Collection and replay flow

The normal entry point is `collect_data.sh`, which invokes
`script/collect_data.py`.

1. With `use_seed: false`, `collect_data.py` tries integer seeds in order,
   calls `setup_demo(..., need_plan=True)`, runs `play_once()`, and accepts a
   seed only when both `plan_success` and `check_success()` are true.
2. `Base_Task.save_traj_data()` saves deep copies of `left_joint_path` and
   `right_joint_path` to `_traj_data/episodeN.pkl`.
3. The collection pass recreates the scene with the successful seed, sets
   `need_plan=False`, loads those two path lists, and calls `play_once()`.
   The arm-motion methods then consume the saved planner result dictionaries
   instead of invoking the planner again.
4. Per-frame pickle observations are sorted and merged into native RoboTwin
   HDF5 and MP4 files by `envs/utils/pkl2hdf5.py`.

This replay mechanism is suitable for strict pairing, but the stock collector
cannot directly create cross-style pairs: trajectory files are scoped to the
configuration output directory, and a changed randomization configuration
recreates a different scene before loading the clean joint paths.

The stock collector also merges/publishes HDF5 before its final success
assertion.  A failed replay can therefore leave an HDF5 that a resumed run
mistakes for a completed episode.  The paired collector stages a complete
content group and publishes it only after success and strict validation.

## Domain-randomization audit

`Base_Task._init_task_env_()` reads the following controls from
`domain_randomization`:

- `random_background`
- `cluttered_table`
- `clean_background_rate`
- `random_head_camera_dis`
- `random_table_height`
- `random_light`
- `crazy_random_light_rate`
- `random_embodiment` (read but otherwise marked TODO)

The stock `demo_randomized.yml` is not usable for this experiment: besides
background textures it enables clutter, random lighting, and random table
height.  The paired configuration fixes the embodiment to `aloha-agilex`,
sets clean-background rate to zero for style variants, and disables every
non-background control.

The implementation of `random_background` selects independent texture IDs
from `assets/background_texture/seen` during collection (`unseen` during eval)
and applies them to the wall and tabletop.  Inspection of
`envs/utils/create_actor.py` shows that the texture changes only the render
material: in addition to the selected base-color texture, the textured wall
and tabletop use white base color, metallic `0.1`, and roughness `0.3`.
Wall/table collision shapes, dimensions, poses, physical material, and the
table legs remain unchanged.  The ground is also unchanged.  The copied assets
contain 10,000 seen and 1,000 unseen textures.

Other controls are not appearance-only:

- clutter adds physical actors and consumes many global random draws;
- table-height randomization changes table and task-object geometry;
- head-camera randomization changes camera position;
- random light changes light colors and can change them on every rendered
  frame in the "crazy" mode.

All of these are forbidden in the paired configuration.

## RNG finding

Stock pairing by `seed` is invalid.

`Base_Task._init_task_env_()` calls `np.random.seed(seed)`.  Before task actors
are loaded, stock background sampling consumes two global
`np.random.randint()` calls for wall/table IDs and two global
`np.random.rand()` calls for the clean-background gates.  Camera construction
also consumes global NumPy draws even when its displacement range is zero.
`grab_roller.load_actors()` later uses that same global stream to select roller
model 0 or 2 and to sample its position and rotation.  Consequently, enabling
background randomization shifts roller identity/pose and planning inputs even
when the nominal episode seed is unchanged.

A direct reproduction with content seed 42 produced different roller model,
position, and rotation between clean and stock random-background control flow.
Loading the clean joint paths afterward does not repair the already-different
scene.

The minimal paired intervention derives explicit wall/table IDs from a private
style RNG and passes those IDs into scene construction.  It does not seed or
consume global NumPy state.  The existing `seed` remains the content seed for
task, object, camera, robot, and planner behavior.  When the new explicit
texture override is absent, legacy clean and randomized behavior is unchanged.

## Native data gaps and strict evidence

Native HDF5 `joint_action/*` values come from joint drive targets; despite the
configuration label `qpos`, they are not physical articulation qpos.  Native
HDF5 does contain sampled end-effector poses, but it does not contain:

- raw physical articulation qpos/qvel;
- the complete per-physics-step commanded action trace;
- initial roller pose and model identity as numeric state;
- source trajectory hashes/path-consumption proof;
- a replay success flag.

The paired path therefore adds sidecar action/state traces and frame-to-state
indices, stores raw physical qpos/qvel in HDF5 when requested, and records
initial state, content/style seeds, actual texture files/hashes, trajectory
hash, path consumption, planner-call count, and final success.  The validator
compares exact bytes first.  Any floating mismatch remains invalid and is
reported with its maximum and mean absolute numerical difference; no hidden
tolerance is used.

## Intended finite run

- Task: `grab_roller` only
- Accepted content trajectories: 20
- Background styles per content: 3 (`style_seed` 0, 1, and 2)
- Accepted random-background variants: exactly 60
- Clean references: one replay of the same saved plan for each content
- No FastWAM model, loss, training code, or dataset loader changes
- No training is started after validation

Development first requests one accepted content in the same transactional
output root.  After that group passes the strict validator, the resumable
collector raises the requested total to 20; the validated pilot is content 0
of that final set rather than an extra episode.  The final manifest therefore
still contains exactly 20-by-3 random-background variants.
