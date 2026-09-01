"""Materialize the immutable release-base formal C1/C3 training matrix.

This is a CPU-only operation.  It resolves six configs (C1/C3 for Stage-2
seeds 1/2/3), writes a pre-final matrix audit, creates the formal protocol
lock, creates a disjoint final-test seed bank, and only then emits executable
config copies.  It never imports a renderer and never starts GPU training.

The immutable ``prelock_configs`` files are intentionally retained: the
formal lock stores their exact file identities.  Executable copies live under
``configs`` and differ only in the cycle-breaking lock/final-bank pointers and
execution flags excluded by ``formal_config_protocol_projection``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .config_audit import (
    AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    AUTHOR_RELEASE_CHECKPOINT_SHA256,
    AUTHOR_RELEASE_DATASET_STATS_SHA256,
    AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
    FORMAL_RELEASE_RECIPE_AMENDMENT_SHA256,
    load_config,
    validate_c1_c3_pair,
    validate_config_structure,
    validate_execution_ready,
)
from .materialize_release_engineering_smoke import (
    CONFIG_DIR,
    DEFAULT_BINDING,
    DEFAULT_CACHE,
    DEFAULT_CACHE_AUDIT,
    DEFAULT_LINEAGE,
    DEFAULT_OFFICIAL_TEXT_BINDING,
    DEFAULT_OFFICIAL_TEXT_CACHE,
    DEFAULT_TEXT_CACHE,
    _file_sha256,
    _load_json,
    _write_new_json,
    _write_new_yaml,
)
from .materialize_release_pmode_dev import DEFAULT_EVALUATOR_SOURCE
from .model import (
    GatedCrossAttentionAdapter,
    PolicyContentHead,
    artifact_identity,
    module_state_sha256,
)
from .p_mode_selection import (
    canonical_sha256,
    formal_config_protocol_projection,
    seed_bank_identity_payload,
    validate_seed_bank_descriptor,
    validate_selection_manifest_payload,
    write_formal_protocol_lock_manifest,
    write_seed_bank_manifest,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import verify_binding as verify_official_binding
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import PROJECT_ROOT


FORMAL_SEEDS = (1, 2, 3)
CONTROLS = ("c1_architecture_only", "c3_ours")
DEFAULT_MAX_STEPS = 1800
DEFAULT_OFFICIAL_BATCH_SIZE = 1
DEFAULT_PAIRED_GROUPS_PER_BATCH = 2
DEFAULT_WORLD_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_NUM_WORKERS = 4
DEFAULT_FINAL_SIMULATOR_SEED = 47
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/formal_c1_c3_release_v1"
).resolve()
DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "p_mode_dev_v1_retry1/p_mode_selection.json"
).resolve()
DEFAULT_DEV_SEED_BANK = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "p_mode_dev_v1_retry1/manifests/dev_selection_seed_bank.json"
).resolve()
DEFAULT_RECIPE_AMENDMENT = (
    CONFIG_DIR / "formal_release_stage2_recipe_amendment_20260819.json"
).resolve()


class FormalC1C3MaterializationError(ValueError):
    """The six-run formal matrix cannot be proven immutable and fair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalC1C3MaterializationError(message)


def _stable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"artifact does not exist: {resolved}")
    before = resolved.stat()
    digest = _file_sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"artifact changed while hashing: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _expected_initialization(seed: int) -> dict[str, Any]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = PolicyContentHead()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        adapter = GatedCrossAttentionAdapter()
    result = {
        "seed": int(seed),
        "source_fp32_content_head_sha256": module_state_sha256(head),
        "source_fp32_adapter_sha256": module_state_sha256(adapter),
        "adapter_gate_raw": float(adapter.gate.detach().item()),
    }
    _require(result["adapter_gate_raw"] == 0.0, "formal GCA gate is not exact zero")
    return result


def _seed_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only per-seed names, paths, and seeded initialization integers."""

    value = copy.deepcopy(dict(config))
    value.pop("experiment_id", None)
    value.pop("output_dir", None)
    value["training"].pop("seed", None)
    value["policy"].pop("head_init_seed", None)
    value["policy"].pop("adapter_init_seed", None)
    return value


def validate_formal_matrix_configs(
    configs: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    _require(tuple(sorted(configs)) == FORMAL_SEEDS, "formal seeds must be exactly 1,2,3")
    rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        pair = configs[seed]
        _require(set(pair) == set(CONTROLS), f"seed {seed} does not contain exactly C1/C3")
        c1 = pair["c1_architecture_only"]
        c3 = pair["c3_ours"]
        for control, config in pair.items():
            validate_config_structure(config)
            _require(config.get("formal") is True and config.get("stage") == "formal", f"{control}/{seed} is not formal")
            _require(config["training"]["seed"] == seed, f"{control}/{seed} seed differs")
            _require(config["policy"]["regime"] == "p_v1", f"{control}/{seed} is not locked P-v1")
            _require(config["policy"]["freeze"]["action_dit"] is True, f"{control}/{seed} does not freeze ActionDiT")
            _require(config["training"]["max_steps"] == DEFAULT_MAX_STEPS, f"{control}/{seed} max_steps differs")
            _require(config["training"]["world_size"] == 1, f"{control}/{seed} world_size differs")
        fairness = validate_c1_c3_pair(c1, c3)
        rows.append(
            {
                "training_seed": seed,
                "status": "PASS",
                "fairness": fairness,
                "c1_protocol_projection_sha256": canonical_sha256(
                    formal_config_protocol_projection(c1)
                ),
                "c3_protocol_projection_sha256": canonical_sha256(
                    formal_config_protocol_projection(c3)
                ),
                "expected_initialization": _expected_initialization(seed),
            }
        )
    for control in CONTROLS:
        reference = _seed_projection(configs[FORMAL_SEEDS[0]][control])
        for seed in FORMAL_SEEDS[1:]:
            _require(
                _seed_projection(configs[seed][control]) == reference,
                f"{control} differs across seeds outside seeded initialization/output fields",
            )
    return {
        "kind": "policy_release_formal_c1_c3_matrix_audit",
        "schema_version": 1,
        "status": "PASS",
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": list(FORMAL_SEEDS),
        "only_c1_c3_intervention": (
            "lambda_contrastive 0.0->0.1 and the two corresponding "
            "paired-contrastive gradient switches"
        ),
        "rows": rows,
    }


def _resolved_prelock_config(
    *,
    template: Mapping[str, Any],
    control: str,
    seed: int,
    output_root: Path,
    selection_path: Path,
    selection_sha256: str,
    binding_path: Path,
    binding_sha256: str,
    paired_text_cache: Path,
    paired_text_cache_sha256: str,
    paired_cache: Path,
    paired_cache_sha256: str,
    official_text_cache: Path,
    official_text_binding: Path,
    official_text_binding_sha256: str,
    amendment_path: Path,
    lock_path: Path,
    final_seed_bank_path: Path,
) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.create(dict(template)), resolve=True)
    _require(isinstance(value, dict), f"{control} template root must be a mapping")
    short = "c1" if control == "c1_architecture_only" else "c3"
    value["experiment_id"] = f"formal_{short}_author_release_p_v1_seed{seed}_v1"
    value["output_dir"] = str((output_root / f"runs/seed_{seed}/{short}").resolve())
    value["execution"] = {
        "runner": "policy_content_adapter",
        "runnable": False,
        "fail_closed": True,
        "long_formal_training": True,
        "blocked_reason": "Awaiting immutable formal lock and disjoint final-test seed bank.",
    }
    value["formal_recipe_amendment_manifest"] = str(amendment_path)
    value["formal_protocol_lock_manifest"] = str(lock_path)
    value["p_mode_selection_manifest"] = str(selection_path)
    value["release_paired_binding_manifest"] = str(binding_path)
    value["artifacts"].update(
        {
            "base_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
            "dataset_stats_sha256": AUTHOR_RELEASE_DATASET_STATS_SHA256,
            "official_task_manifest_sha256": AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
            "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
            "release_paired_binding_manifest_sha256": binding_sha256,
            "formal_recipe_amendment_manifest_sha256": FORMAL_RELEASE_RECIPE_AMENDMENT_SHA256,
            "formal_protocol_lock_manifest_sha256": "__REQUIRED_FORMAL_PROTOCOL_LOCK_MANIFEST_SHA256__",
            "head_init_sha256": None,
            "paired_text_cache_sha256": paired_text_cache_sha256,
            "paired_cache_sha256": paired_cache_sha256,
            "p_mode_selection_manifest_sha256": selection_sha256,
            "simulator_seed_bank_manifest_sha256": "__REQUIRED_FINAL_TEST_SEED_BANK_MANIFEST_SHA256__",
            "official_text_cache_binding_manifest_sha256": official_text_binding_sha256,
        }
    )
    value["policy"].update(
        {
            "regime": "p_v1",
            "head_init_mode": "random",
            "head_init_seed": seed,
            "adapter_init_seed": seed,
            "head_init": None,
        }
    )
    value["policy"]["freeze"]["action_dit"] = True
    value["official"].update(
        {
            "sampling_mode": "all_frames",
            "text_cache_dir": str(official_text_cache),
            "text_cache_binding_manifest": str(official_text_binding),
            "on_the_fly_text_smoke": False,
            "domain_verified": True,
        }
    )
    value["paired"].update(
        {
            "text_cache_dir": str(paired_text_cache),
            "cache": str(paired_cache),
            "contrastive_supervision": control == "c3_ours",
        }
    )
    value["supervision"]["paired_contrastive"] = control == "c3_ours"
    value["loss"]["lambda_contrastive"] = 0.1 if control == "c3_ours" else 0.0
    value["training"].update(
        {
            "seed": seed,
            "max_steps": DEFAULT_MAX_STEPS,
            "official_batch_size": DEFAULT_OFFICIAL_BATCH_SIZE,
            "paired_groups_per_batch": DEFAULT_PAIRED_GROUPS_PER_BATCH,
            "world_size": DEFAULT_WORLD_SIZE,
            "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
            "effective_official_global_batch": DEFAULT_OFFICIAL_BATCH_SIZE,
            "effective_paired_groups_per_step": DEFAULT_PAIRED_GROUPS_PER_BATCH,
            "num_workers": DEFAULT_NUM_WORKERS,
            "save_optimizer": True,
        }
    )
    value["evaluation"].update(
        {
            "simulator_seed_bank_manifest": str(final_seed_bank_path),
            "simulator_seed_bank_id": "__REQUIRED_FINAL_TEST_SEED_BANK_ID__",
            "simulator_seed_bank_purpose": "final_test",
            "episodes_per_task": 100,
        }
    )
    validate_config_structure(value)
    projection = formal_config_protocol_projection(value)
    encoded = json.dumps(projection, sort_keys=True)
    _require(
        "__REQUIRED_" not in encoded and "__SELECT_" not in encoded,
        f"{control}/{seed} has a non-cycle placeholder in its prelock projection",
    )
    return value


def _finalize_config(
    prelock: Mapping[str, Any],
    *,
    lock_sha256: str,
    final_seed_bank_sha256: str,
    final_seed_bank_id: str,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(prelock))
    value["execution"] = {
        "runner": "policy_content_adapter",
        "runnable": True,
        "fail_closed": False,
        "long_formal_training": True,
    }
    value["artifacts"]["formal_protocol_lock_manifest_sha256"] = lock_sha256
    value["artifacts"]["simulator_seed_bank_manifest_sha256"] = final_seed_bank_sha256
    value["evaluation"]["simulator_seed_bank_id"] = final_seed_bank_id
    return value


def materialize(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    selection_manifest: str | Path = DEFAULT_SELECTION,
    dev_seed_bank_manifest: str | Path = DEFAULT_DEV_SEED_BANK,
    final_simulator_seed: int = DEFAULT_FINAL_SIMULATOR_SEED,
    lineage_manifest: str | Path = DEFAULT_LINEAGE,
    release_paired_binding_manifest: str | Path = DEFAULT_BINDING,
    paired_text_cache: str | Path = DEFAULT_TEXT_CACHE,
    paired_cache: str | Path = DEFAULT_CACHE,
    paired_cache_audit: str | Path = DEFAULT_CACHE_AUDIT,
    evaluator_source: str | Path = DEFAULT_EVALUATOR_SOURCE,
    official_text_cache: str | Path = DEFAULT_OFFICIAL_TEXT_CACHE,
    official_text_cache_binding: str | Path = DEFAULT_OFFICIAL_TEXT_BINDING,
    recipe_amendment_manifest: str | Path = DEFAULT_RECIPE_AMENDMENT,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    _require(not destination.exists(), f"refusing to reuse formal output root: {destination}")
    selection_path = Path(selection_manifest).expanduser().resolve()
    dev_bank_path = Path(dev_seed_bank_manifest).expanduser().resolve()
    lineage_path = Path(lineage_manifest).expanduser().resolve()
    binding_path = Path(release_paired_binding_manifest).expanduser().resolve()
    text_cache_path = Path(paired_text_cache).expanduser().resolve()
    cache_path = Path(paired_cache).expanduser().resolve()
    cache_audit_path = Path(paired_cache_audit).expanduser().resolve()
    evaluator_path = Path(evaluator_source).expanduser().resolve()
    official_text_path = Path(official_text_cache).expanduser().resolve()
    official_binding_path = Path(official_text_cache_binding).expanduser().resolve()
    amendment_path = Path(recipe_amendment_manifest).expanduser().resolve()

    amendment_identity = _stable_file_identity(amendment_path)
    _require(
        amendment_identity["sha256"] == FORMAL_RELEASE_RECIPE_AMENDMENT_SHA256,
        "formal recipe amendment differs from the reviewed disclosure",
    )
    amendment = _load_json(amendment_path, "formal recipe amendment")
    _require(
        amendment.get("kind") == "policy_release_stage2_recipe_amendment"
        and amendment.get("schema_version") == 2
        and amendment.get("status") == "PASS"
        and amendment.get("locked_recipe", {}).get("max_steps") == DEFAULT_MAX_STEPS,
        "formal recipe amendment contract differs",
    )

    selection_identity = _stable_file_identity(selection_path)
    selection = validate_selection_manifest_payload(
        _load_json(selection_path, "P-mode selection manifest")
    )
    _require(selection["winner"] == "p_v1", "formal materializer requires selected P-v1")
    dev_bank_identity = _stable_file_identity(dev_bank_path)
    dev_bank = validate_seed_bank_descriptor(
        _load_json(dev_bank_path, "dev-selection seed bank"),
        expected_purpose="dev_selection",
    )
    _require(
        seed_bank_identity_payload(dev_bank)
        == seed_bank_identity_payload(selection["dev_seed_bank"]),
        "dev seed-bank file differs from the selected P-mode evidence",
    )

    c1_template = load_config(CONFIG_DIR / "formal_c1_architecture_only.yaml")
    c3_template = load_config(CONFIG_DIR / "formal_c3_ours.yaml")
    checkpoint = Path(c1_template["base_checkpoint"]).expanduser().resolve()
    stats = Path(c1_template["official"]["dataset_stats"]).expanduser().resolve()
    official_manifest = Path(
        c1_template["official"]["canonical_task_manifest"]
    ).expanduser().resolve()
    lineage = verify_author_release_lineage(
        lineage_path,
        checkpoint_path=checkpoint,
        dataset_stats_path=stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    )
    binding = verify_release_paired_binding(
        binding_path,
        expected_sha256=c1_template["artifacts"][
            "release_paired_binding_manifest_sha256"
        ],
    )
    paired_text = verify_release_paired_text_cache(
        text_cache_path,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_release_paired_binding_sha256=binding[
            "binding_manifest_identity"
        ]["sha256"],
    )
    cache_identity = artifact_identity(cache_path)
    cache_audit = _load_json(cache_audit_path, "Layer-16 cache audit")
    _require(cache_audit.get("status") == "PASS", "Layer-16 cache audit is not PASS")
    _require(cache_audit.get("cache", {}).get("sha256") == cache_identity["sha256"], "Layer-16 cache differs from audit")
    _require(cache_audit.get("layer16_shape") == [2880, 120, 3072], "Layer-16 cache shape changed")
    official_binding_sha = _file_sha256(official_binding_path)
    official_binding = verify_official_binding(
        official_binding_path,
        expected_sha256=official_binding_sha,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_cache_dir=official_text_path,
    )

    lock_path = (destination / "manifests/formal_protocol_lock.json").resolve()
    final_bank_path = (destination / "manifests/final_test_seed_bank.json").resolve()
    matrix_path = (destination / "manifests/formal_matrix_audit.json").resolve()
    prelock_configs: dict[int, dict[str, dict[str, Any]]] = {}
    prelock_paths: dict[int, dict[str, Path]] = {}
    for seed in FORMAL_SEEDS:
        prelock_configs[seed] = {}
        prelock_paths[seed] = {}
        for control, template in (
            ("c1_architecture_only", c1_template),
            ("c3_ours", c3_template),
        ):
            value = _resolved_prelock_config(
                template=template,
                control=control,
                seed=seed,
                output_root=destination,
                selection_path=selection_path,
                selection_sha256=selection_identity["sha256"],
                binding_path=binding_path,
                binding_sha256=binding["binding_manifest_identity"]["sha256"],
                paired_text_cache=text_cache_path,
                paired_text_cache_sha256=paired_text["directory_identity"]["sha256"],
                paired_cache=cache_path,
                paired_cache_sha256=cache_identity["sha256"],
                official_text_cache=official_text_path,
                official_text_binding=official_binding_path,
                official_text_binding_sha256=official_binding_sha,
                amendment_path=amendment_path,
                lock_path=lock_path,
                final_seed_bank_path=final_bank_path,
            )
            short = "c1" if control == "c1_architecture_only" else "c3"
            path = (destination / f"prelock_configs/seed_{seed}/{short}.yaml").resolve()
            prelock_configs[seed][control] = value
            prelock_paths[seed][control] = path

    matrix_audit = validate_formal_matrix_configs(prelock_configs)
    matrix_audit["recipe_amendment"] = amendment_identity
    matrix_audit["p_mode_selection_manifest"] = selection_identity
    matrix_audit["exposure_semantics"] = amendment["exposure_accounting"]
    for seed in FORMAL_SEEDS:
        for control in CONTROLS:
            _write_new_yaml(prelock_paths[seed][control], prelock_configs[seed][control])
    _write_new_json(matrix_path, matrix_audit)

    lock = write_formal_protocol_lock_manifest(
        base_lineage_manifest=lineage_path,
        p_mode_selection_manifest=selection_path,
        formal_matrix_audit=matrix_path,
        c1_configs=[prelock_paths[seed]["c1_architecture_only"] for seed in FORMAL_SEEDS],
        c3_configs=[prelock_paths[seed]["c3_ours"] for seed in FORMAL_SEEDS],
        output=lock_path,
    )
    lock_identity = _stable_file_identity(lock_path)
    final_bank = write_seed_bank_manifest(
        simulator_seed=int(final_simulator_seed),
        episodes_per_cell=100,
        evaluator_source=evaluator_path,
        purpose="final_test",
        output=final_bank_path,
        disjoint_from_dev_manifest=dev_bank_path,
        p_mode_selection_manifest=selection_path,
        formal_protocol_lock_manifest=lock_path,
    )
    final_bank_identity = _stable_file_identity(final_bank_path)

    final_configs: dict[int, dict[str, dict[str, Any]]] = {}
    config_identities: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in FORMAL_SEEDS:
        final_configs[seed] = {}
        config_identities[str(seed)] = {}
        for control in CONTROLS:
            short = "c1" if control == "c1_architecture_only" else "c3"
            value = _finalize_config(
                prelock_configs[seed][control],
                lock_sha256=lock_identity["sha256"],
                final_seed_bank_sha256=final_bank_identity["sha256"],
                final_seed_bank_id=final_bank["simulator_seed_bank_id"],
            )
            path = (destination / f"configs/seed_{seed}/{short}.yaml").resolve()
            _write_new_yaml(path, value)
            emitted = load_config(path)
            validate_execution_ready(emitted)
            final_configs[seed][control] = emitted
            config_identities[str(seed)][short] = _stable_file_identity(path)
        validate_c1_c3_pair(
            final_configs[seed]["c1_architecture_only"],
            final_configs[seed]["c3_ours"],
        )
        for control in CONTROLS:
            _require(
                canonical_sha256(
                    formal_config_protocol_projection(final_configs[seed][control])
                )
                == canonical_sha256(
                    formal_config_protocol_projection(prelock_configs[seed][control])
                ),
                f"seed {seed}/{control} executable config differs from locked projection",
            )

    manifest = {
        "kind": "policy_release_formal_c1_c3_materialization",
        "schema_version": 1,
        "status": "PASS",
        "gpu_training_started": False,
        "online_rollout_started": False,
        "c0_evaluation_requested": False,
        "selected_policy_regime": "p_v1",
        "stage2_training_seeds": list(FORMAL_SEEDS),
        "recipe": amendment["locked_recipe"],
        "exposure_accounting": amendment["exposure_accounting"],
        "configs": config_identities,
        "prelock_configs": {
            str(seed): {
                ("c1" if control == "c1_architecture_only" else "c3"): _stable_file_identity(path)
                for control, path in prelock_paths[seed].items()
            }
            for seed in FORMAL_SEEDS
        },
        "artifacts": {
            "base_lineage_manifest": lineage["manifest_identity"],
            "release_paired_binding_manifest": binding["binding_manifest_identity"],
            "recipe_amendment_manifest": amendment_identity,
            "p_mode_selection_manifest": selection_identity,
            "dev_seed_bank_manifest": dev_bank_identity,
            "formal_matrix_audit": _stable_file_identity(matrix_path),
            "formal_protocol_lock": lock_identity,
            "final_test_seed_bank": final_bank_identity,
            "final_test_seed_bank_id": final_bank["simulator_seed_bank_id"],
            "paired_text_cache_sha256": paired_text["directory_identity"]["sha256"],
            "paired_cache_sha256": cache_identity["sha256"],
            "paired_cache_audit_sha256": _file_sha256(cache_audit_path),
            "official_text_cache_binding_manifest_sha256": official_binding_sha,
            "official_text_cache_aggregate_payload_sha256": official_binding["cache"]["aggregate_payload_sha256"],
        },
        "matrix_audit": matrix_audit,
        "formal_lock_status": lock["status"],
        "launch_contract": {
            "jobs": 6,
            "one_gpu_per_job": True,
            "distributed_world_size_per_job": 1,
            "recommended_parallelism": "six independent single-GPU jobs when six GPUs are free",
            "no_output_overwrite": True,
            "posttrain_pair_audit_required_before_rollout": True,
        },
    }
    manifest_path = (destination / "materialization_manifest.json").resolve()
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_SELECTION))
    parser.add_argument("--dev-seed-bank-manifest", default=str(DEFAULT_DEV_SEED_BANK))
    parser.add_argument("--final-simulator-seed", type=int, default=DEFAULT_FINAL_SIMULATOR_SEED)
    parser.add_argument("--lineage-manifest", default=str(DEFAULT_LINEAGE))
    parser.add_argument("--release-paired-binding-manifest", default=str(DEFAULT_BINDING))
    parser.add_argument("--paired-text-cache", default=str(DEFAULT_TEXT_CACHE))
    parser.add_argument("--paired-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--paired-cache-audit", default=str(DEFAULT_CACHE_AUDIT))
    parser.add_argument("--evaluator-source", default=str(DEFAULT_EVALUATOR_SOURCE))
    parser.add_argument("--official-text-cache", default=str(DEFAULT_OFFICIAL_TEXT_CACHE))
    parser.add_argument("--official-text-cache-binding", default=str(DEFAULT_OFFICIAL_TEXT_BINDING))
    parser.add_argument("--recipe-amendment-manifest", default=str(DEFAULT_RECIPE_AMENDMENT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_root=args.output_root,
        selection_manifest=args.selection_manifest,
        dev_seed_bank_manifest=args.dev_seed_bank_manifest,
        final_simulator_seed=args.final_simulator_seed,
        lineage_manifest=args.lineage_manifest,
        release_paired_binding_manifest=args.release_paired_binding_manifest,
        paired_text_cache=args.paired_text_cache,
        paired_cache=args.paired_cache,
        paired_cache_audit=args.paired_cache_audit,
        evaluator_source=args.evaluator_source,
        official_text_cache=args.official_text_cache,
        official_text_cache_binding=args.official_text_cache_binding,
        recipe_amendment_manifest=args.recipe_amendment_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_OUTPUT_ROOT",
    "FormalC1C3MaterializationError",
    "materialize",
    "validate_formal_matrix_configs",
]
