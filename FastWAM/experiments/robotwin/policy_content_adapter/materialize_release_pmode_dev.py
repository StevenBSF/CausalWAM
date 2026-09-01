"""Materialize the locked release-base P-v1/P-v2 online-dev pilot pair.

The two candidates are C1/action-only runs (``lambda_contrastive=0``).  They
share the author release, initialized Head/GCA, input streams, optimizer
recipe, training seed, and one immutable ``dev_selection`` simulator bank.
Only the policy regime and the corresponding ActionDiT freeze switch differ.

This module is CPU-only.  It writes configs and provenance but never starts
training, a renderer, or the P-mode selector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config_audit import (
    AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    AUTHOR_RELEASE_CHECKPOINT_SHA256,
    AUTHOR_RELEASE_DATASET_STATS_SHA256,
    AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
    load_config,
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
    _write_new_bytes,
    _write_new_json,
    _write_new_yaml,
)
from .model import artifact_identity
from .p_mode_selection import (
    DEV_EPISODES_PER_CELL,
    build_seed_bank_descriptor,
    canonical_sha256,
)
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import verify_binding as verify_official_binding
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import PROJECT_ROOT


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/policy_content_adapter/release_base_v1/p_mode_dev_v1"
).resolve()
DEFAULT_EVALUATOR_SOURCE = (
    PROJECT_ROOT / "third_party/RoboTwin/script/eval_policy.py"
).resolve()
DEFAULT_TRAINING_SEED = 42
DEFAULT_MAX_STEPS = 100
DEFAULT_OFFICIAL_BATCH_SIZE = 1
DEFAULT_PAIRED_GROUPS_PER_BATCH = 2
DEFAULT_WORLD_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_SIMULATOR_SEED = 23


class PModeDevMaterializationError(ValueError):
    """The P-v1/P-v2 dev pair cannot be proven execution-ready and fair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PModeDevMaterializationError(message)


def _pair_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the experiment variable from one dev candidate."""

    value = copy.deepcopy(dict(config))
    for key in ("experiment_id", "control", "output_dir"):
        value.pop(key, None)
    policy = value.get("policy")
    _require(isinstance(policy, dict), "dev config policy must be a mapping")
    policy.pop("regime", None)
    freeze = policy.get("freeze")
    _require(isinstance(freeze, dict), "dev config policy.freeze must be a mapping")
    freeze.pop("action_dit", None)
    return value


def validate_pmode_dev_pair(
    p_v1: Mapping[str, Any], p_v2: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the candidates differ only by P-v1 versus P-v2."""

    for candidate, regime, frozen in (
        (p_v1, "p_v1", True),
        (p_v2, "p_v2", False),
    ):
        _require(candidate.get("control") == regime, f"{regime} control differs")
        _require(candidate.get("stage") == "dev_pilot", f"{regime} is not dev_pilot")
        _require(candidate.get("formal") is False, f"{regime} must be non-formal")
        _require(
            candidate.get("selection_role") == "c1_lambda0",
            f"{regime} is not the C1/lambda-zero selector candidate",
        )
        _require(
            float(candidate["loss"]["lambda_contrastive"]) == 0.0,
            f"{regime} must set lambda_contrastive=0",
        )
        _require(
            candidate["paired"]["contrastive_supervision"] is False
            and candidate["supervision"]["paired_contrastive"] is False,
            f"{regime} leaked contrastive gradients",
        )
        _require(
            candidate["policy"]["regime"] == regime,
            f"{regime} policy label differs",
        )
        _require(
            candidate["policy"]["freeze"]["action_dit"] is frozen,
            f"{regime} ActionDiT freeze contract differs",
        )
        _require(
            candidate.get("p_mode_selection_manifest") is None
            and candidate["artifacts"]["p_mode_selection_manifest_sha256"] is None,
            f"{regime} must predate mode selection",
        )
        _require(
            candidate["evaluation"]["simulator_seed_bank_purpose"]
            == "dev_selection",
            f"{regime} does not use a dev_selection bank",
        )
        _require(
            candidate["evaluation"]["episodes_per_task"]
            == DEV_EPISODES_PER_CELL,
            f"{regime} dev episode count differs",
        )
    _require(
        _pair_projection(p_v1) == _pair_projection(p_v2),
        "P-v1/P-v2 differ outside regime and ActionDiT freeze",
    )
    shared = {
        "training_seed": int(p_v1["training"]["seed"]),
        "max_steps": int(p_v1["training"]["max_steps"]),
        "official_batch_size": int(p_v1["training"]["official_batch_size"]),
        "paired_groups_per_batch": int(
            p_v1["training"]["paired_groups_per_batch"]
        ),
        "world_size": int(p_v1["training"]["world_size"]),
        "gradient_accumulation_steps": int(
            p_v1["training"]["gradient_accumulation_steps"]
        ),
        "simulator_seed_bank_id": p_v1["evaluation"][
            "simulator_seed_bank_id"
        ],
        "simulator_seed_bank_manifest_sha256": p_v1["artifacts"][
            "simulator_seed_bank_manifest_sha256"
        ],
        "episodes_per_task_domain": DEV_EPISODES_PER_CELL,
        "lambda_contrastive": 0.0,
    }
    return {
        "status": "PASS",
        "only_candidate_difference": "policy_regime_and_action_dit_freeze",
        "shared_recipe_sha256": canonical_sha256(_pair_projection(p_v1)),
        "shared": shared,
    }


def build_resolved_dev_pair(
    *,
    p_v1_template: Mapping[str, Any],
    p_v2_template: Mapping[str, Any],
    output_root: Path,
    training_seed: int,
    max_steps: int,
    official_batch_size: int,
    paired_groups_per_batch: int,
    world_size: int,
    gradient_accumulation_steps: int,
    release_paired_binding_manifest: Path,
    release_paired_binding_sha256: str,
    paired_text_cache: Path,
    paired_text_cache_sha256: str,
    paired_cache: Path,
    paired_cache_sha256: str,
    official_text_cache: Path,
    official_text_cache_binding_manifest: Path,
    official_text_cache_binding_manifest_sha256: str,
    seed_bank_manifest: Path,
    seed_bank_manifest_sha256: str,
    seed_bank_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(training_seed >= 0, "training_seed must be non-negative")
    _require(max_steps >= 3, "dev pilot needs at least three optimizer steps")
    for name, value in (
        ("official_batch_size", official_batch_size),
        ("paired_groups_per_batch", paired_groups_per_batch),
        ("world_size", world_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        _require(int(value) > 0, f"{name} must be positive")

    effective_official = (
        official_batch_size * world_size * gradient_accumulation_steps
    )
    effective_paired = (
        paired_groups_per_batch * world_size * gradient_accumulation_steps
    )
    resolved: list[dict[str, Any]] = []
    for source, regime in (
        (p_v1_template, "p_v1"),
        (p_v2_template, "p_v2"),
    ):
        value = OmegaConf.to_container(OmegaConf.create(dict(source)), resolve=True)
        _require(isinstance(value, dict), f"{regime} template root is not a mapping")
        value["experiment_id"] = f"{regime}_author_release_c1_lambda0_dev_v1"
        value["stage"] = "dev_pilot"
        value["formal"] = False
        value["control"] = regime
        value["selection_role"] = "c1_lambda0"
        value["output_dir"] = str((output_root / "runs" / regime).resolve())
        value["execution"] = {
            "runner": "policy_content_adapter",
            "runnable": True,
            "fail_closed": False,
            "long_formal_training": False,
        }
        value["p_mode_selection_manifest"] = None
        value["release_paired_binding_manifest"] = str(
            release_paired_binding_manifest.resolve()
        )
        value["artifacts"].update(
            {
                "base_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
                "dataset_stats_sha256": AUTHOR_RELEASE_DATASET_STATS_SHA256,
                "official_task_manifest_sha256": AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
                "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
                "release_paired_binding_manifest_sha256": (
                    release_paired_binding_sha256
                ),
                "head_init_sha256": None,
                "paired_text_cache_sha256": paired_text_cache_sha256,
                "paired_cache_sha256": paired_cache_sha256,
                "p_mode_selection_manifest_sha256": None,
                "simulator_seed_bank_manifest_sha256": seed_bank_manifest_sha256,
                "official_text_cache_binding_manifest_sha256": (
                    official_text_cache_binding_manifest_sha256
                ),
            }
        )
        value["policy"].update(
            {
                "regime": regime,
                "head_init_mode": "random",
                "head_init_seed": training_seed,
                "adapter_init_seed": training_seed,
                "head_init": None,
            }
        )
        value["policy"]["freeze"]["action_dit"] = regime == "p_v1"
        value["official"].update(
            {
                "sampling_mode": "all_frames",
                "text_cache_dir": str(official_text_cache.resolve()),
                "text_cache_binding_manifest": str(
                    official_text_cache_binding_manifest.resolve()
                ),
                "on_the_fly_text_smoke": False,
                "domain_verified": True,
            }
        )
        value["paired"].update(
            {
                "text_cache_dir": str(paired_text_cache.resolve()),
                "cache": str(paired_cache.resolve()),
                "contrastive_supervision": False,
            }
        )
        value["supervision"]["paired_contrastive"] = False
        value["loss"]["lambda_contrastive"] = 0.0
        value["training"].update(
            {
                "seed": training_seed,
                "max_steps": max_steps,
                "official_batch_size": official_batch_size,
                "paired_groups_per_batch": paired_groups_per_batch,
                "world_size": world_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_official_global_batch": effective_official,
                "effective_paired_groups_per_step": effective_paired,
                "num_workers": 0,
                "save_optimizer": False,
            }
        )
        value["evaluation"].update(
            {
                "simulator_seed_bank_manifest": str(seed_bank_manifest.resolve()),
                "simulator_seed_bank_id": seed_bank_id,
                "simulator_seed_bank_purpose": "dev_selection",
                "episodes_per_task": DEV_EPISODES_PER_CELL,
            }
        )
        resolved.append(value)
    p_v1, p_v2 = resolved
    validate_pmode_dev_pair(p_v1, p_v2)
    return p_v1, p_v2


def materialize(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    training_seed: int = DEFAULT_TRAINING_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
    official_batch_size: int = DEFAULT_OFFICIAL_BATCH_SIZE,
    paired_groups_per_batch: int = DEFAULT_PAIRED_GROUPS_PER_BATCH,
    world_size: int = DEFAULT_WORLD_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    simulator_seed: int = DEFAULT_SIMULATOR_SEED,
    lineage_manifest: str | Path = DEFAULT_LINEAGE,
    release_paired_binding_manifest: str | Path = DEFAULT_BINDING,
    paired_text_cache: str | Path = DEFAULT_TEXT_CACHE,
    paired_cache: str | Path = DEFAULT_CACHE,
    paired_cache_audit: str | Path = DEFAULT_CACHE_AUDIT,
    evaluator_source: str | Path = DEFAULT_EVALUATOR_SOURCE,
    official_text_cache: str | Path = DEFAULT_OFFICIAL_TEXT_CACHE,
    official_text_cache_binding: str | Path = DEFAULT_OFFICIAL_TEXT_BINDING,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    _require(not destination.exists(), f"refusing to reuse output root: {destination}")
    lineage_path = Path(lineage_manifest).expanduser().resolve()
    binding_path = Path(release_paired_binding_manifest).expanduser().resolve()
    text_cache_path = Path(paired_text_cache).expanduser().resolve()
    cache_path = Path(paired_cache).expanduser().resolve()
    cache_audit_path = Path(paired_cache_audit).expanduser().resolve()
    evaluator_path = Path(evaluator_source).expanduser().resolve()
    official_text_path = Path(official_text_cache).expanduser().resolve()
    official_binding_path = Path(official_text_cache_binding).expanduser().resolve()

    p_v1_template = load_config(CONFIG_DIR / "p_v1_dev_pilot.yaml")
    p_v2_template = load_config(CONFIG_DIR / "p_v2_dev_pilot.yaml")
    checkpoint = Path(p_v1_template["base_checkpoint"]).expanduser().resolve()
    stats = Path(p_v1_template["official"]["dataset_stats"]).expanduser().resolve()
    official_manifest = Path(
        p_v1_template["official"]["canonical_task_manifest"]
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
        expected_sha256=p_v1_template["artifacts"][
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
    _require(
        cache_audit.get("cache", {}).get("sha256") == cache_identity["sha256"],
        "Layer-16 cache differs from its audit",
    )
    _require(
        cache_audit.get("layer16_shape") == [2880, 120, 3072],
        "Layer-16 cache shape changed",
    )
    official_binding_sha = _file_sha256(official_binding_path)
    official_binding = verify_official_binding(
        official_binding_path,
        expected_sha256=official_binding_sha,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_cache_dir=official_text_path,
    )
    seed_bank = build_seed_bank_descriptor(
        simulator_seed=simulator_seed,
        episodes_per_cell=DEV_EPISODES_PER_CELL,
        evaluator_source=evaluator_path,
        purpose="dev_selection",
    )
    seed_bank_path = (destination / "manifests/dev_selection_seed_bank.json").resolve()
    seed_bank_bytes = (
        json.dumps(seed_bank, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    seed_bank_sha = hashlib.sha256(seed_bank_bytes).hexdigest()

    p_v1, p_v2 = build_resolved_dev_pair(
        p_v1_template=p_v1_template,
        p_v2_template=p_v2_template,
        output_root=destination,
        training_seed=training_seed,
        max_steps=max_steps,
        official_batch_size=official_batch_size,
        paired_groups_per_batch=paired_groups_per_batch,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        release_paired_binding_manifest=binding_path,
        release_paired_binding_sha256=binding["binding_manifest_identity"]["sha256"],
        paired_text_cache=text_cache_path,
        paired_text_cache_sha256=paired_text["directory_identity"]["sha256"],
        paired_cache=cache_path,
        paired_cache_sha256=cache_identity["sha256"],
        official_text_cache=official_text_path,
        official_text_cache_binding_manifest=official_binding_path,
        official_text_cache_binding_manifest_sha256=official_binding_sha,
        seed_bank_manifest=seed_bank_path,
        seed_bank_manifest_sha256=seed_bank_sha,
        seed_bank_id=seed_bank["simulator_seed_bank_id"],
    )
    _write_new_bytes(seed_bank_path, seed_bank_bytes)
    p_v1_path = (destination / "configs/p_v1_dev_pilot.yaml").resolve()
    p_v2_path = (destination / "configs/p_v2_dev_pilot.yaml").resolve()
    _write_new_yaml(p_v1_path, p_v1)
    _write_new_yaml(p_v2_path, p_v2)
    emitted_p_v1 = load_config(p_v1_path)
    emitted_p_v2 = load_config(p_v2_path)
    validate_execution_ready(emitted_p_v1)
    validate_execution_ready(emitted_p_v2)
    fairness = validate_pmode_dev_pair(emitted_p_v1, emitted_p_v2)

    manifest = {
        "schema_version": 1,
        "kind": "policy_release_pmode_dev_materialization",
        "status": "PASS",
        "scientific_results_present": False,
        "gpu_training_started": False,
        "online_rollout_started": False,
        "selection_started": False,
        "selection_rule_locked_elsewhere": "policy_p_mode_selection_rule_v1",
        "fairness": fairness,
        "configs": {
            "p_v1": {"path": str(p_v1_path), "sha256": _file_sha256(p_v1_path)},
            "p_v2": {"path": str(p_v2_path), "sha256": _file_sha256(p_v2_path)},
        },
        "artifacts": {
            "base_lineage_manifest_sha256": lineage["manifest_identity"]["sha256"],
            "release_paired_binding_manifest_sha256": binding[
                "binding_manifest_identity"
            ]["sha256"],
            "paired_text_cache_sha256": paired_text["directory_identity"]["sha256"],
            "paired_cache_sha256": cache_identity["sha256"],
            "paired_cache_audit_sha256": _file_sha256(cache_audit_path),
            "official_text_cache_binding_manifest_sha256": official_binding_sha,
            "official_text_cache_aggregate_payload_sha256": official_binding["cache"][
                "aggregate_payload_sha256"
            ],
            "simulator_seed_bank_manifest_sha256": seed_bank_sha,
            "simulator_seed_bank_id": seed_bank["simulator_seed_bank_id"],
            "evaluator_source_sha256": seed_bank["evaluator_source_sha256"],
        },
        "resource_plan": {
            "training": "two independent single-GPU jobs; GPU0/GPU1 may run concurrently",
            "online_rollout": (
                "single-GPU sequential by default; concurrency is forbidden unless "
                "each CUDA device is explicitly PCI-matched to its Vulkan renderer"
            ),
            "episodes_total": 2 * 3 * 2 * DEV_EPISODES_PER_CELL,
        },
    }
    manifest_path = (destination / "materialization_manifest.json").resolve()
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--official-batch-size", type=int, default=DEFAULT_OFFICIAL_BATCH_SIZE
    )
    parser.add_argument(
        "--paired-groups-per-batch",
        type=int,
        default=DEFAULT_PAIRED_GROUPS_PER_BATCH,
    )
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--simulator-seed", type=int, default=DEFAULT_SIMULATOR_SEED)
    parser.add_argument("--lineage-manifest", default=str(DEFAULT_LINEAGE))
    parser.add_argument(
        "--release-paired-binding-manifest", default=str(DEFAULT_BINDING)
    )
    parser.add_argument("--paired-text-cache", default=str(DEFAULT_TEXT_CACHE))
    parser.add_argument("--paired-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--paired-cache-audit", default=str(DEFAULT_CACHE_AUDIT))
    parser.add_argument("--evaluator-source", default=str(DEFAULT_EVALUATOR_SOURCE))
    parser.add_argument("--official-text-cache", default=str(DEFAULT_OFFICIAL_TEXT_CACHE))
    parser.add_argument(
        "--official-text-cache-binding", default=str(DEFAULT_OFFICIAL_TEXT_BINDING)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_root=args.output_root,
        training_seed=args.training_seed,
        max_steps=args.max_steps,
        official_batch_size=args.official_batch_size,
        paired_groups_per_batch=args.paired_groups_per_batch,
        world_size=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        simulator_seed=args.simulator_seed,
        lineage_manifest=args.lineage_manifest,
        release_paired_binding_manifest=args.release_paired_binding_manifest,
        paired_text_cache=args.paired_text_cache,
        paired_cache=args.paired_cache,
        paired_cache_audit=args.paired_cache_audit,
        evaluator_source=args.evaluator_source,
        official_text_cache=args.official_text_cache,
        official_text_cache_binding=args.official_text_cache_binding,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TRAINING_SEED",
    "PModeDevMaterializationError",
    "build_resolved_dev_pair",
    "materialize",
    "validate_pmode_dev_pair",
]
