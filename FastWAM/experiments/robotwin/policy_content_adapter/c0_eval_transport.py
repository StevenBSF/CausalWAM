#!/usr/bin/env python3
"""Package the fixed author-release C0 for the common online evaluator.

C0 is the native ``B_release`` model and receives no Stage-2 training.  The shared
RoboTwin evaluator loads compact Policy checkpoints, so this utility creates a
transport-only overlay whose GCA gate is exactly zero.  The overlay is not a
C0 architecture change: it carries no ActionDiT weights, and its residual is
identically the native release action path.  Every base/runtime artifact is
content-addressed so the evaluator can fail closed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from .config_audit import TASKS
from .losses import zero_init_policy_identity_audit
from .model import (
    GatedCrossAttentionAdapter,
    PolicyContentConditioner,
    PolicyContentHead,
    artifact_identity,
    module_state_sha256,
    save_policy_checkpoint,
    install_policy_content_adapter,
)
from .native50hz_paired import atomic_write_json
from .official_data import OfficialThreeTaskDataset
from .p_mode_selection import validate_seed_bank_descriptor
from .release_lineage import ReleaseLineageError, verify_author_release_lineage
from .release_official_text_cache_binding import (
    ReleaseOfficialTextCacheBindingError,
    verify_binding as verify_official_text_cache_binding,
)
from .runtime_utils import (
    audit_local_fastwam_source,
    dtype_from_name,
    instantiate_official_dataset,
    instantiate_release_model,
)


class C0TransportError(RuntimeError):
    """The transport checkpoint cannot prove exact native C0 identity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C0TransportError(message)


def _resolved_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is not a file: {resolved}")
    return resolved


def _resolved_dir(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _non_placeholder(value: str, label: str) -> str:
    text = str(value).strip()
    _require(bool(text) and not text.startswith("__"), f"{label} is unresolved")
    return text


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _runtime_artifacts(
    *,
    dataset_stats: Path,
    model_base_path: Path,
    official_manifest: Path,
    base_lineage_manifest: Path,
    simulator_seed_bank_manifest: Path,
    final_lock_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates = {
        "dataset_stats": dataset_stats,
        "official_manifest": official_manifest,
        "base_lineage_manifest": base_lineage_manifest,
        "simulator_seed_bank_manifest": simulator_seed_bank_manifest,
        "vae": model_base_path
        / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
        "text_encoder": model_base_path
        / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
        "tokenizer": model_base_path / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    }
    for name, path in (final_lock_artifacts or {}).items():
        candidates[name] = path
    identities: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        identity = artifact_identity(path)
        identity["required_for_rollout"] = name in {
            "dataset_stats",
            "vae",
            "text_encoder",
            "tokenizer",
            "simulator_seed_bank_manifest",
            "formal_protocol_lock_manifest",
            "p_mode_selection_manifest",
        }
        identity["verification_status"] = "PASS"
        identities[name] = identity
    return identities


def _new_transport_conditioner(seed: int) -> PolicyContentConditioner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        conditioner = PolicyContentConditioner(
            head=PolicyContentHead(),
            adapter=GatedCrossAttentionAdapter(),
            enabled=True,
            content_layer=16,
        )
    _require(
        float(conditioner.adapter.gate.detach().item()) == 0.0,
        "C0 transport gate is not exact zero",
    )
    return conditioner


@contextlib.contextmanager
def _c0_dataset_initialization_work_dir(output_path: str | Path) -> Iterator[Path]:
    """Scope the native dataset's stats mirror away from ambiguous ``./runs``."""

    from fastwam.utils import misc

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_work_dir = getattr(misc, "_WORK_DIR", None)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.dataset-init-",
            dir=output.parent,
        ) as temporary:
            staging = Path(temporary).resolve()
            misc.register_work_dir(staging)
            _require(staging.is_dir(), "C0 dataset-init work directory was not created")
            yield staging
    finally:
        # Upstream ``register_work_dir(None)`` is invalid; restore its exact
        # prior sentinel without creating the default ./runs directory.
        misc._WORK_DIR = previous_work_dir  # noqa: SLF001


def create_c0_runtime_identity_audit(
    *,
    base_checkpoint: str | Path,
    dataset_stats: str | Path,
    dataset_root: str | Path,
    model_base_path: str | Path,
    official_manifest: str | Path,
    base_lineage_manifest: str | Path,
    text_cache_dir: str | Path,
    text_cache_binding_manifest: str | Path,
    output: str | Path,
    transport_seed: int = 0,
    device: str = "cuda",
    model_dtype: str = "bf16",
) -> dict[str, Any]:
    """Run the real B_release prefill/action path and prove transport bit-exact."""

    _require(isinstance(transport_seed, int) and transport_seed >= 0, "transport_seed must be nonnegative")
    _require(str(device).startswith("cuda"), "formal C0 identity audit requires CUDA")
    base = _resolved_file(base_checkpoint, "author release checkpoint")
    stats = _resolved_file(dataset_stats, "author release dataset stats")
    official_root = _resolved_dir(dataset_root, "official dataset root")
    model_base = _resolved_dir(model_base_path, "model base")
    official = _resolved_file(official_manifest, "official manifest")
    lineage_path = _resolved_file(base_lineage_manifest, "author release lineage")
    try:
        lineage = verify_author_release_lineage(
            lineage_path,
            checkpoint_path=base,
            dataset_stats_path=stats,
            official_manifest_path=official,
        )
    except ReleaseLineageError as exc:
        raise C0TransportError(f"invalid author release lineage: {exc}") from exc
    text_cache = _resolved_dir(text_cache_dir, "text cache")
    text_cache_binding_path = _resolved_file(
        text_cache_binding_manifest, "official text-cache binding manifest"
    )
    text_cache_binding_identity = artifact_identity(text_cache_binding_path)
    try:
        text_cache_binding = verify_official_text_cache_binding(
            text_cache_binding_path,
            expected_sha256=text_cache_binding_identity["sha256"],
            expected_base_lineage_sha256=lineage["manifest_identity"]["sha256"],
            expected_cache_dir=text_cache,
        )
    except ReleaseOfficialTextCacheBindingError as exc:
        raise C0TransportError(
            f"invalid official text-cache binding: {exc}"
        ) from exc
    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite C0 identity audit: {destination}")

    model, native_cfg, _ = instantiate_release_model(
        base,
        device=device,
        dtype=dtype_from_name(model_dtype),
        load_text_encoder=False,
        model_base_path=model_base,
        compute_checkpoint_sha256=False,
    )
    conditioner = _new_transport_conditioner(transport_seed)
    source_head_sha256 = module_state_sha256(conditioner.head)
    source_adapter_sha256 = module_state_sha256(conditioner.adapter)
    runtime = install_policy_content_adapter(
        model,
        head=conditioner.head,
        adapter=conditioner.adapter,
        enabled=True,
        content_layer=16,
        patch_video_prefill=True,
    )
    with _c0_dataset_initialization_work_dir(destination):
        native_dataset = instantiate_official_dataset(
            native_cfg,
            dataset_root=official_root,
            dataset_stats_path=stats,
            text_cache_dir=text_cache,
            manifest_path=official,
            episode_selection_mode="full_550_per_task",
        )
    official_dataset = OfficialThreeTaskDataset(
        native_dataset,
        dataset_root=official_root,
        manifest_path=official,
        sampling_mode="episode_anchor",
    )
    sample = default_collate([official_dataset[0]])
    identity = zero_init_policy_identity_audit(model, runtime, sample)
    fastwam_source = audit_local_fastwam_source()
    report = {
        **identity,
        "kind": "policy_c0_zero_gate_runtime_identity",
        "base_checkpoint_sha256": artifact_identity(base)["sha256"],
        "dataset_stats_sha256": artifact_identity(stats)["sha256"],
        "official_manifest_sha256": artifact_identity(official)["sha256"],
        "base_lineage_manifest_sha256": lineage["manifest_identity"]["sha256"],
        "base_kind": "author_release",
        "official_dataset_root": str(official_root),
        "transport_seed": transport_seed,
        # The transport checkpoint stores the seeded FP32 source state.  The
        # runtime installer then casts that same state to the release dtype; both
        # identities are recorded so packaging can reproduce the source while
        # the audit still proves the actually executed modules.
        "transport_head_sha256": source_head_sha256,
        "transport_adapter_sha256": source_adapter_sha256,
        "installed_transport_head_sha256": module_state_sha256(conditioner.head),
        "installed_transport_adapter_sha256": module_state_sha256(conditioner.adapter),
        "installed_model_dtype": str(dtype_from_name(model_dtype)),
        "official_selection": official_dataset.audit_report,
        "runtime_artifacts": {
            "vae": artifact_identity(
                model_base
                / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
            ),
            # The strict cache gate already tensor-reloaded and aggregate-hashed
            # all 68,704 payloads (~72 GiB).  Re-hashing that directory here
            # would add hours without creating independent evidence.  Bind the
            # immutable audit instead; RobotVideoDataset still validates each
            # payload consumed by this real-path identity check.
            "text_cache_binding_manifest": text_cache_binding_identity,
            "text_cache": {
                "kind": "audited_directory_binding",
                "path": str(text_cache),
                "file_count": text_cache_binding["cache"]["file_count"],
                "size_bytes": text_cache_binding["cache"]["total_size_bytes"],
                "aggregate_payload_sha256": text_cache_binding["cache"][
                    "aggregate_payload_sha256"
                ],
                "directory_bytes_rehashed_for_c0": False,
                "runtime_validation": text_cache_binding["cache"][
                    "runtime_validation"
                ],
            },
        },
        "fastwam_source": fastwam_source,
        "fastwam_source_sha256": _canonical_sha256(fastwam_source),
    }
    atomic_write_json(destination, report)
    return report


def build_c0_eval_transport(
    *,
    base_checkpoint: str | Path,
    dataset_stats: str | Path,
    model_base_path: str | Path,
    official_manifest: str | Path,
    base_lineage_manifest: str | Path,
    identity_audit: str | Path,
    output: str | Path,
    rollout_protocol_id: str,
    simulator_seed_bank_id: str,
    simulator_seed_bank_manifest: str | Path,
    episodes_per_task: int = 100,
    transport_seed: int = 0,
    evaluation_stage: str = "deployment_gate",
    source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one no-overwrite, exact-zero-gate C0 evaluation transport."""

    _require(isinstance(transport_seed, int) and transport_seed >= 0, "transport_seed must be nonnegative")
    _require(isinstance(episodes_per_task, int) and episodes_per_task > 0, "episodes_per_task must be positive")
    _require(
        evaluation_stage in {"deployment_gate", "formal_test"},
        "evaluation_stage must be deployment_gate or formal_test",
    )
    expected_bank_purpose = (
        "final_test" if evaluation_stage == "formal_test" else "development_analysis"
    )
    protocol_id = _non_placeholder(rollout_protocol_id, "rollout_protocol_id")
    seed_bank_id = _non_placeholder(simulator_seed_bank_id, "simulator_seed_bank_id")
    seed_bank_path = _resolved_file(
        simulator_seed_bank_manifest, "simulator seed-bank manifest"
    )
    try:
        seed_bank = validate_seed_bank_descriptor(
            json.loads(seed_bank_path.read_text(encoding="utf-8")),
            expected_purpose=expected_bank_purpose,
        )
    except Exception as exc:
        raise C0TransportError(f"invalid final-test seed-bank manifest: {exc}") from exc
    _require(
        seed_bank["simulator_seed_bank_id"] == seed_bank_id,
        "simulator seed-bank id differs from manifest",
    )
    _require(
        seed_bank["episodes_per_cell"] == episodes_per_task,
        "simulator seed-bank episodes differ",
    )
    base = _resolved_file(base_checkpoint, "author release checkpoint")
    stats = _resolved_file(dataset_stats, "author release dataset stats")
    model_base = _resolved_dir(model_base_path, "model base")
    official = _resolved_file(official_manifest, "official manifest")
    lineage_path = _resolved_file(base_lineage_manifest, "author release lineage")
    identity_audit_path = _resolved_file(identity_audit, "C0 runtime identity audit")
    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite C0 transport: {destination}")

    base_identity = artifact_identity(base)
    stats_identity = artifact_identity(stats)
    official_identity = artifact_identity(official)
    try:
        lineage = verify_author_release_lineage(
            lineage_path,
            checkpoint_path=base,
            dataset_stats_path=stats,
            official_manifest_path=official,
        )
    except ReleaseLineageError as exc:
        raise C0TransportError(f"invalid author release lineage: {exc}") from exc
    lineage_identity = lineage["manifest_identity"]

    conditioner = _new_transport_conditioner(transport_seed)
    identity_payload = json.loads(identity_audit_path.read_text(encoding="utf-8"))
    _require(isinstance(identity_payload, Mapping), "C0 identity audit root must be an object")
    _require(
        identity_payload.get("status") == "PASS"
        and identity_payload.get("kind") == "policy_c0_zero_gate_runtime_identity"
        and identity_payload.get("native_prefill_kv_bit_exact") is True
        and identity_payload.get("action_output_bit_exact") is True
        and float(identity_payload.get("max_abs_error", -1.0)) == 0.0
        and float(identity_payload.get("max_rel_error", -1.0)) == 0.0
        and float(identity_payload.get("gate_raw", -1.0)) == 0.0,
        "C0 runtime identity audit did not prove exact native behavior",
    )
    _require(identity_payload.get("base_checkpoint_sha256") == base_identity["sha256"], "C0 identity audit binds a different author release")
    _require(identity_payload.get("dataset_stats_sha256") == stats_identity["sha256"], "C0 identity audit binds different stats")
    _require(identity_payload.get("official_manifest_sha256") == official_identity["sha256"], "C0 identity audit binds a different official manifest")
    _require(
        identity_payload.get("base_lineage_manifest_sha256")
        == lineage_identity["sha256"],
        "C0 identity audit binds a different author release lineage",
    )
    _require(identity_payload.get("transport_seed") == transport_seed, "C0 identity audit transport seed differs")
    _require(identity_payload.get("transport_head_sha256") == module_state_sha256(conditioner.head), "C0 identity audit Head differs")
    _require(identity_payload.get("transport_adapter_sha256") == module_state_sha256(conditioner.adapter), "C0 identity audit adapter differs")
    for field in (
        "installed_transport_head_sha256",
        "installed_transport_adapter_sha256",
    ):
        installed_digest = str(identity_payload.get(field, ""))
        _require(
            len(installed_digest) == 64
            and all(character in "0123456789abcdef" for character in installed_digest),
            f"C0 identity audit lacks valid {field}",
        )
    final_lock_artifacts: dict[str, Path] = {}
    if evaluation_stage == "formal_test":
        lock_ancestry = seed_bank["lock_ancestry"]
        for name in ("p_mode_selection_manifest", "formal_protocol_lock_manifest"):
            declaration = lock_ancestry[name]
            path = _resolved_file(declaration["path"], f"final-test {name}")
            actual = artifact_identity(path)
            _require(
                actual["sha256"] == declaration["sha256"]
                and actual["size_bytes"] == declaration["size_bytes"],
                f"final-test {name} changed after seed-bank lock",
            )
            final_lock_artifacts[name] = path
    identities = _runtime_artifacts(
        dataset_stats=stats,
        model_base_path=model_base,
        official_manifest=official,
        base_lineage_manifest=lineage_path,
        simulator_seed_bank_manifest=seed_bank_path,
        final_lock_artifacts=final_lock_artifacts,
    )
    identities["c0_runtime_identity_audit"] = artifact_identity(identity_audit_path)
    identities["c0_runtime_identity_audit"]["required_for_rollout"] = False
    identities["c0_runtime_identity_audit"]["verification_status"] = "PASS"
    resolved_source_audit = dict(source_audit or audit_local_fastwam_source())
    _require(resolved_source_audit.get("status") == "PASS", "FastWAM source audit is not PASS")
    source_files = resolved_source_audit.get("files")
    _require(
        resolved_source_audit.get("scope") == "all_python_files_under_src_fastwam"
        and isinstance(source_files, Mapping)
        and bool(source_files)
        and resolved_source_audit.get("file_count") == len(source_files),
        "FastWAM source audit is incomplete",
    )
    _require(
        identity_payload.get("fastwam_source_sha256")
        == _canonical_sha256(resolved_source_audit),
        "C0 runtime identity audit used different FastWAM source",
    )
    run_config = {
        "schema_version": 3,
        "kind": "policy_c0_eval_transport",
        "stage": "control",
        "formal": evaluation_stage == "formal_test",
        "control": "c0_original",
        "tasks": list(TASKS),
        "base_checkpoint": str(base),
        "base_lineage_manifest": str(lineage_path),
        "model_base_path": str(model_base),
        "policy": {
            "method_modules_active": False,
            "transport_modules_installed": True,
            "transport_only_zero_gate": True,
            "transport_seed": transport_seed,
        },
        "training": {"seed": None, "stage2_steps": 0},
        "official": {
            "dataset_stats": str(stats),
            "canonical_task_manifest": str(official),
            "selection_mode": "full_550_per_task",
        },
        "evaluation": {
            "tasks": list(TASKS),
            "required_domains": ["clean", "official_random"],
            "rollout_protocol_id": protocol_id,
            "simulator_seed_bank_id": seed_bank_id,
            "simulator_seed_bank_purpose": expected_bank_purpose,
            "simulator_seed_bank_manifest": str(seed_bank_path),
            "episodes_per_task": episodes_per_task,
        },
        "runtime_provenance": {
            "fastwam_source": resolved_source_audit,
            "fastwam_source_sha256": _canonical_sha256(resolved_source_audit),
        },
        "c0_semantics": {
            "stage2_training": False,
            "head_gca_present_only_for_common_evaluator_transport": True,
            "head_gca_effect_on_action": "none_exact_zero_gate",
            "transport_residual": "Xa + tanh(0) * GCA(Xa,Zc) == Xa",
            "action_expert_overlay": False,
            "base_lineage_manifest_sha256": identities[
                "base_lineage_manifest"
            ]["sha256"],
            "runtime_identity_audit_sha256": identities[
                "c0_runtime_identity_audit"
            ]["sha256"],
        },
        "artifacts": {
            "base_checkpoint_sha256": base_identity["sha256"],
            "dataset_stats_sha256": stats_identity["sha256"],
            "base_lineage_manifest_sha256": lineage_identity["sha256"],
            "simulator_seed_bank_manifest_sha256": identities[
                "simulator_seed_bank_manifest"
            ]["sha256"],
            "p_mode_selection_manifest_sha256": None,
            "formal_protocol_lock_manifest_sha256": (
                identities["formal_protocol_lock_manifest"]["sha256"]
                if evaluation_stage == "formal_test"
                else None
            ),
        },
    }
    saved = save_policy_checkpoint(
        destination,
        model=nn.Module(),
        conditioner=conditioner,
        base_checkpoint=base,
        regime="p_v1",
        step=0,
        run_config=run_config,
        include_base_sha256=True,
        verified_base_identity=base_identity,
        artifact_identities=identities,
    )
    checkpoint_identity = artifact_identity(saved)
    return {
        "status": "PASS",
        "control": "c0_original",
        "transport_only": True,
        "stage2_training": False,
        "exact_zero_gate": True,
        "action_expert_overlay": False,
        "training_seed": None,
        "base_kind": "author_release",
        "base_lineage_manifest": lineage_identity,
        "evaluation_stage": evaluation_stage,
        "base_checkpoint": base_identity,
        "dataset_stats": stats_identity,
        "transport_checkpoint": checkpoint_identity,
        "transport_head_sha256": module_state_sha256(conditioner.head),
        "transport_adapter_sha256": module_state_sha256(conditioner.adapter),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--model-base-path", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--base-lineage-manifest", required=True)
    parser.add_argument("--identity-audit", required=True)
    parser.add_argument("--run-identity-audit", action="store_true")
    parser.add_argument("--text-cache-dir")
    parser.add_argument("--text-cache-binding-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rollout-protocol-id", required=True)
    parser.add_argument("--simulator-seed-bank-id", required=True)
    parser.add_argument("--simulator-seed-bank-manifest", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--transport-seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-stage",
        choices=("deployment_gate", "formal_test"),
        default="deployment_gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_identity_audit:
        if not args.text_cache_dir:
            raise C0TransportError("--run-identity-audit requires --text-cache-dir")
        if not args.dataset_root:
            raise C0TransportError("--run-identity-audit requires --dataset-root")
        if not args.text_cache_binding_manifest:
            raise C0TransportError(
                "--run-identity-audit requires --text-cache-binding-manifest"
            )
        create_c0_runtime_identity_audit(
            base_checkpoint=args.base_checkpoint,
            dataset_stats=args.dataset_stats,
            dataset_root=args.dataset_root,
            model_base_path=args.model_base_path,
            official_manifest=args.official_manifest,
            base_lineage_manifest=args.base_lineage_manifest,
            text_cache_dir=args.text_cache_dir,
            text_cache_binding_manifest=args.text_cache_binding_manifest,
            output=args.identity_audit,
            transport_seed=args.transport_seed,
            device=args.device,
            model_dtype=args.model_dtype,
        )
    report = build_c0_eval_transport(
        base_checkpoint=args.base_checkpoint,
        dataset_stats=args.dataset_stats,
        model_base_path=args.model_base_path,
        official_manifest=args.official_manifest,
        base_lineage_manifest=args.base_lineage_manifest,
        identity_audit=args.identity_audit,
        output=args.output,
        rollout_protocol_id=args.rollout_protocol_id,
        simulator_seed_bank_id=args.simulator_seed_bank_id,
        simulator_seed_bank_manifest=args.simulator_seed_bank_manifest,
        episodes_per_task=args.episodes_per_task,
        transport_seed=args.transport_seed,
        evaluation_stage=args.evaluation_stage,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C0TransportError",
    "build_c0_eval_transport",
    "create_c0_runtime_identity_audit",
]
