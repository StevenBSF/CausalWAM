"""Materialize an execution-ready, matched C1/C3 release-base GPU smoke.

This gate is deliberately non-formal and pre-P-mode-selection.  It pins one
provisional runtime regime only to prove that both C1 and C3 can load the exact
author release, consume the exact same official/paired stream sequence, update
and save.  It cannot be used as P-mode selection or scientific rollout evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    validate_c1_c3_pair,
    validate_execution_ready,
)
from .model import artifact_identity
from .p_mode_selection import build_seed_bank_descriptor
from .prepare_release_paired_text_cache import verify_release_paired_text_cache
from .release_lineage import verify_author_release_lineage
from .release_official_text_cache_binding import (
    build_binding as build_official_text_cache_binding,
    verify_binding as verify_official_text_cache_binding,
    write_binding as write_official_text_cache_binding,
)
from .release_paired_binding import verify_release_paired_binding
from .runtime_utils import PROJECT_ROOT


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_RELEASE_ROOT = (
    PROJECT_ROOT / "outputs/policy_content_adapter/release_base_v1"
).resolve()
DEFAULT_LINEAGE = (CONFIG_DIR / "author_release_base_manifest.json").resolve()
DEFAULT_BINDING = (DEFAULT_RELEASE_ROOT / "paired_binding_manifest.json").resolve()
DEFAULT_TEXT_CACHE = (DEFAULT_RELEASE_ROOT / "paired_text_cache").resolve()
DEFAULT_CACHE = (
    DEFAULT_RELEASE_ROOT / "policy_release50tasks_native50hz_four_scene_v1.pt"
).resolve()
DEFAULT_CACHE_AUDIT = Path(f"{DEFAULT_CACHE}.audit.json").resolve()
DEFAULT_EVALUATOR_SOURCE = (
    Path(__file__).resolve().parent / "eval_robotwin_single.py"
).resolve()
DEFAULT_OFFICIAL_TEXT_CACHE = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/stage1_artifacts/full550_three_task_text_cache"
).resolve()
DEFAULT_OFFICIAL_TEXT_CACHE_AUDIT = Path(
    f"{DEFAULT_OFFICIAL_TEXT_CACHE}.audit.json"
).resolve()
DEFAULT_OFFICIAL_TEXT_STRICT_EVIDENCE = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/stage1_clean_random_base_seed1/stage1_protocol_audit.json"
).resolve()
DEFAULT_OFFICIAL_TEXT_BINDING = (
    DEFAULT_RELEASE_ROOT / "official_text_cache_binding_manifest.json"
).resolve()


class EngineeringSmokeMaterializationError(ValueError):
    """The real release artifacts cannot prove an executable smoke pair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringSmokeMaterializationError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EngineeringSmokeMaterializationError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _write_new_bytes(path: Path, payload: bytes) -> None:
    """Atomically create one immutable materialization artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EngineeringSmokeMaterializationError(
                f"refusing to overwrite materialization artifact: {path}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_new_yaml(path: Path, value: Mapping[str, Any]) -> None:
    rendered = OmegaConf.to_yaml(OmegaConf.create(dict(value)), resolve=True)
    _write_new_bytes(path, rendered.encode("utf-8"))


def build_resolved_pair(
    *,
    c1_template: Mapping[str, Any],
    c3_template: Mapping[str, Any],
    output_root: Path,
    regime: str,
    seed: int,
    steps: int,
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
    """Resolve the two templates while preserving the one-treatment contract."""

    normalized_regime = str(regime).replace("-", "_")
    _require(normalized_regime in {"p_v1", "p_v2"}, "regime must be p_v1 or p_v2")
    _require(isinstance(seed, int) and seed >= 0, "seed must be non-negative")
    _require(3 <= int(steps) <= 3, "three-task engineering smoke is locked to 3 steps")

    configs: list[dict[str, Any]] = []
    for source in (c1_template, c3_template):
        config = OmegaConf.to_container(OmegaConf.create(dict(source)), resolve=True)
        _require(isinstance(config, dict), "template root must be a mapping")
        control = str(config["control"])
        short_name = "c1" if control == "c1_architecture_only" else "c3"
        config["experiment_id"] = (
            f"{short_name}_author_release_{normalized_regime}_engineering_smoke_v1"
        )
        config["stage"] = "smoke"
        config["formal"] = False
        config["output_dir"] = str((output_root / "runs" / short_name).resolve())
        config["execution"] = {
            "runner": "policy_content_adapter",
            "runnable": True,
            "fail_closed": False,
            "long_formal_training": False,
        }
        config["p_mode_selection_manifest"] = None
        artifacts = config["artifacts"]
        artifacts.update(
            {
                "base_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
                "dataset_stats_sha256": AUTHOR_RELEASE_DATASET_STATS_SHA256,
                "official_task_manifest_sha256": AUTHOR_RELEASE_OFFICIAL_MANIFEST_SHA256,
                "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
                "release_paired_binding_manifest_sha256": release_paired_binding_sha256,
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
        config["release_paired_binding_manifest"] = str(
            release_paired_binding_manifest.resolve()
        )
        policy = config["policy"]
        policy["regime"] = normalized_regime
        policy["head_init_mode"] = "random"
        policy["head_init_seed"] = int(seed)
        policy["adapter_init_seed"] = int(seed)
        policy["head_init"] = None
        policy["freeze"]["action_dit"] = normalized_regime == "p_v1"
        official = config["official"]
        official["sampling_mode"] = "episode_anchor"
        official["text_cache_dir"] = str(official_text_cache.resolve())
        official["text_cache_binding_manifest"] = str(
            official_text_cache_binding_manifest.resolve()
        )
        official["on_the_fly_text_smoke"] = False
        official["domain_verified"] = True
        paired = config["paired"]
        paired["text_cache_dir"] = str(paired_text_cache.resolve())
        paired["cache"] = str(paired_cache.resolve())
        training = config["training"]
        training.update(
            {
                "seed": int(seed),
                "max_steps": int(steps),
                "official_batch_size": 1,
                "paired_groups_per_batch": 2,
                "world_size": 1,
                "gradient_accumulation_steps": 1,
                "effective_official_global_batch": 1,
                "effective_paired_groups_per_step": 2,
                "num_workers": 0,
                "save_optimizer": False,
            }
        )
        evaluation = config["evaluation"]
        evaluation.update(
            {
                "simulator_seed_bank_manifest": str(seed_bank_manifest.resolve()),
                "simulator_seed_bank_id": str(seed_bank_id),
                "simulator_seed_bank_purpose": "engineering_smoke",
                "episodes_per_task": 1,
            }
        )
        configs.append(config)

    c1, c3 = configs
    validate_c1_c3_pair(c1, c3)
    return c1, c3


def materialize(
    *,
    output_root: str | Path,
    regime: str = "p_v1",
    seed: int = 42,
    steps: int = 3,
    simulator_seed: int = 17,
    lineage_manifest: str | Path = DEFAULT_LINEAGE,
    release_paired_binding_manifest: str | Path = DEFAULT_BINDING,
    paired_text_cache: str | Path = DEFAULT_TEXT_CACHE,
    paired_cache: str | Path = DEFAULT_CACHE,
    paired_cache_audit: str | Path = DEFAULT_CACHE_AUDIT,
    evaluator_source: str | Path = DEFAULT_EVALUATOR_SOURCE,
    official_text_cache: str | Path = DEFAULT_OFFICIAL_TEXT_CACHE,
    official_text_cache_audit: str | Path = DEFAULT_OFFICIAL_TEXT_CACHE_AUDIT,
    official_text_strict_evidence: str | Path = DEFAULT_OFFICIAL_TEXT_STRICT_EVIDENCE,
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
    official_text_cache_path = Path(official_text_cache).expanduser().resolve()
    official_text_cache_audit_path = Path(official_text_cache_audit).expanduser().resolve()
    official_text_strict_evidence_path = Path(
        official_text_strict_evidence
    ).expanduser().resolve()
    official_text_binding_path = Path(official_text_cache_binding).expanduser().resolve()

    c1_template = load_config(CONFIG_DIR / "c1_architecture_only.yaml")
    c3_template = load_config(CONFIG_DIR / "c3_ours.yaml")
    checkpoint = Path(c1_template["base_checkpoint"]).expanduser().resolve()
    dataset_stats = Path(c1_template["official"]["dataset_stats"]).expanduser().resolve()
    official_manifest = Path(
        c1_template["official"]["canonical_task_manifest"]
    ).expanduser().resolve()
    lineage = verify_author_release_lineage(
        lineage_path,
        checkpoint_path=checkpoint,
        dataset_stats_path=dataset_stats,
        official_manifest_path=official_manifest,
        expected_manifest_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
    )
    if not official_text_binding_path.exists():
        official_text_binding = build_official_text_cache_binding(
            cache_dir=official_text_cache_path,
            completion_audit=official_text_cache_audit_path,
            strict_payload_evidence=official_text_strict_evidence_path,
            base_lineage_manifest=lineage_path,
            base_checkpoint=checkpoint,
            dataset_stats=dataset_stats,
            official_manifest=official_manifest,
            expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        )
        write_official_text_cache_binding(
            official_text_binding_path, official_text_binding
        )
    official_text_binding_sha = _file_sha256(official_text_binding_path)
    official_text_binding = verify_official_text_cache_binding(
        official_text_binding_path,
        expected_sha256=official_text_binding_sha,
        expected_base_lineage_sha256=AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        expected_cache_dir=official_text_cache_path,
    )
    binding = verify_release_paired_binding(
        binding_path,
        expected_sha256=str(
            c1_template["artifacts"]["release_paired_binding_manifest_sha256"]
        ),
    )
    text_cache_audit = verify_release_paired_text_cache(
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
        "Layer-16 cache bytes differ from its audit",
    )
    expected_cache_fields = {
        "backbone_checkpoint_sha256": AUTHOR_RELEASE_CHECKPOINT_SHA256,
        "base_lineage_manifest_sha256": AUTHOR_RELEASE_BASE_MANIFEST_SHA256,
        "release_paired_binding_manifest_sha256": binding[
            "binding_manifest_identity"
        ]["sha256"],
        "paired_action_manifest_sha256": c1_template["artifacts"][
            "paired_action_manifest_sha256"
        ],
        "paired_state_bank_sha256": c1_template["artifacts"][
            "paired_state_bank_sha256"
        ],
    }
    for key, expected in expected_cache_fields.items():
        _require(cache_audit.get(key) == expected, f"Layer-16 cache audit {key} differs")
    _require(
        cache_audit.get("layer16_shape") == [2880, 120, 3072],
        "Layer-16 cache shape is not [2880,120,3072]",
    )

    seed_bank = build_seed_bank_descriptor(
        simulator_seed=int(simulator_seed),
        episodes_per_cell=1,
        evaluator_source=evaluator_path,
        purpose="engineering_smoke",
    )
    seed_bank_path = (destination / "manifests/engineering_smoke_seed_bank.json").resolve()
    seed_bank_bytes = (
        json.dumps(seed_bank, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    seed_bank_sha = hashlib.sha256(seed_bank_bytes).hexdigest()

    c1, c3 = build_resolved_pair(
        c1_template=c1_template,
        c3_template=c3_template,
        output_root=destination,
        regime=regime,
        seed=int(seed),
        steps=int(steps),
        release_paired_binding_manifest=binding_path,
        release_paired_binding_sha256=binding["binding_manifest_identity"]["sha256"],
        paired_text_cache=text_cache_path,
        paired_text_cache_sha256=text_cache_audit["directory_identity"]["sha256"],
        paired_cache=cache_path,
        paired_cache_sha256=cache_identity["sha256"],
        official_text_cache=official_text_cache_path,
        official_text_cache_binding_manifest=official_text_binding_path,
        official_text_cache_binding_manifest_sha256=official_text_binding_sha,
        seed_bank_manifest=seed_bank_path,
        seed_bank_manifest_sha256=seed_bank_sha,
        seed_bank_id=seed_bank["simulator_seed_bank_id"],
    )
    _write_new_bytes(seed_bank_path, seed_bank_bytes)
    c1_path = (destination / "configs/c1_engineering_smoke.yaml").resolve()
    c3_path = (destination / "configs/c3_engineering_smoke.yaml").resolve()
    _write_new_yaml(c1_path, c1)
    _write_new_yaml(c3_path, c3)

    # Re-read the emitted bytes: these exact files, not only in-memory objects,
    # must pass execution-ready and one-treatment fairness audits.
    emitted_c1 = load_config(c1_path)
    emitted_c3 = load_config(c3_path)
    validate_execution_ready(emitted_c1)
    validate_execution_ready(emitted_c3)
    fairness = validate_c1_c3_pair(emitted_c1, emitted_c3)
    manifest = {
        "schema_version": 1,
        "kind": "policy_release_c1_c3_engineering_smoke_materialization",
        "status": "PASS",
        "scientific_result": False,
        "p_mode_selection_evidence": False,
        "formal_training_auto_started": False,
        "regime": str(regime).replace("-", "_"),
        "training_seed": int(seed),
        "optimizer_steps_per_control": int(steps),
        "only_permitted_cross_control_difference": (
            "lambda_contrastive_0.0_vs_0.1_and_its_gradient_switches"
        ),
        "fairness": fairness,
        "configs": {
            "c1": {"path": str(c1_path), "sha256": _file_sha256(c1_path)},
            "c3": {"path": str(c3_path), "sha256": _file_sha256(c3_path)},
        },
        "artifacts": {
            "base_lineage_manifest_sha256": lineage["manifest_identity"]["sha256"],
            "release_paired_binding_manifest_sha256": binding[
                "binding_manifest_identity"
            ]["sha256"],
            "paired_text_cache_sha256": text_cache_audit["directory_identity"][
                "sha256"
            ],
            "paired_cache_sha256": cache_identity["sha256"],
            "paired_cache_audit_sha256": _file_sha256(cache_audit_path),
            "simulator_seed_bank_manifest_sha256": seed_bank_sha,
            "official_text_cache_binding_manifest_sha256": (
                official_text_binding_sha
            ),
            "official_text_cache_aggregate_payload_sha256": (
                official_text_binding["cache"]["aggregate_payload_sha256"]
            ),
        },
    }
    manifest_path = (destination / "materialization_manifest.json").resolve()
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--regime", choices=("p_v1", "p_v2"), default="p_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--simulator-seed", type=int, default=17)
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
        "--official-text-cache-audit", default=str(DEFAULT_OFFICIAL_TEXT_CACHE_AUDIT)
    )
    parser.add_argument(
        "--official-text-strict-evidence",
        default=str(DEFAULT_OFFICIAL_TEXT_STRICT_EVIDENCE),
    )
    parser.add_argument(
        "--official-text-cache-binding", default=str(DEFAULT_OFFICIAL_TEXT_BINDING)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_root=args.output_root,
        regime=args.regime,
        seed=args.seed,
        steps=args.steps,
        simulator_seed=args.simulator_seed,
        lineage_manifest=args.lineage_manifest,
        release_paired_binding_manifest=args.release_paired_binding_manifest,
        paired_text_cache=args.paired_text_cache,
        paired_cache=args.paired_cache,
        paired_cache_audit=args.paired_cache_audit,
        evaluator_source=args.evaluator_source,
        official_text_cache=args.official_text_cache,
        official_text_cache_audit=args.official_text_cache_audit,
        official_text_strict_evidence=args.official_text_strict_evidence,
        official_text_cache_binding=args.official_text_cache_binding,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EngineeringSmokeMaterializationError",
    "build_resolved_pair",
    "materialize",
]
