"""Materialize and validate the P-v2 seed-53 100-episode amendment.

The P-v2 mechanism-study checkpoints were trained against an immutable pilot
configuration that declared 20 online episodes per task/domain.  Before any
complete pilot result was produced, the user required the online evaluation to
match the native FastWAM RoboTwin episode count: 100 episodes for every task
under Clean and Official Random.  This CPU-only module records that transparent
post-materialization amendment without changing either trained checkpoint.

The abandoned 20-episode output tree is bound as invalid evidence.  Its result
files are hashed but never parsed, and it must contain no completed-rollouts
manifest.  The replacement seed bank keeps simulator seed 53 and the same
candidate stream; only the episode count and derived bank identity change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .p_mode_selection import (
    build_seed_bank_descriptor,
    validate_seed_bank_descriptor,
)
from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_followup_eval100_amendment"
SCHEMA_VERSION = 1
PROFILE = "pv2_actiondit_seed53_fastwam_100ep_v1"
SIMULATOR_SEED = 53
ORIGINAL_EPISODES_PER_CELL = 20
RUNTIME_EPISODES_PER_CELL = 100
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
CONTROLS = {
    "c1": "c1_architecture_only",
    "c3": "c3_ours",
}
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/"
    "pv2_actiondit_followup_v1"
).resolve()
DEFAULT_AMENDMENT_RELATIVE = Path("manifests/eval100_user_amendment_v1.json")
DEFAULT_RUNTIME_BANK_RELATIVE = Path("manifests/dev_seed53_100ep_bank_v1.json")
DEFAULT_INVALID_PARTIAL_RELATIVE = Path("pilot_rollouts")
EXPECTED_PARTIAL_RESULTS = (
    "seed_1/c1/place_a2b_left/_result_clean.txt",
    "seed_1/c3/place_a2b_left/_result_clean.txt",
)


class Pv2Eval100AmendmentError(ValueError):
    """The evaluation amendment cannot be proven from immutable inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2Eval100AmendmentError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file is missing: {resolved}")
    before = resolved.stat()
    digest = _file_sha256(resolved)
    after = resolved.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {resolved}",
    )
    return {
        "kind": "file",
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Pv2Eval100AmendmentError(
            f"cannot parse {label}: {resolved}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{label} root must be an object")
    return payload, resolved


def _verify_identity(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).expanduser()
    _require(path.is_absolute(), f"{label} path must be absolute")
    actual = _file_identity(path)
    fields = ("path", "size_bytes", "sha256")
    if value.get("kind") is not None:
        fields = ("kind", *fields)
    for field in fields:
        expected = (
            str(Path(str(value.get(field, ""))).expanduser().resolve())
            if field == "path"
            else value.get(field)
        )
        _require(actual[field] == expected, f"{label} {field} changed")
    return actual


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Pv2Eval100AmendmentError(
                f"refusing to overwrite immutable artifact: {destination}"
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return destination


def _same_candidate_stream(
    original: Mapping[str, Any], replacement: Mapping[str, Any]
) -> None:
    old = dict(original)
    new = dict(replacement)
    for field in ("episodes_per_cell", "simulator_seed_bank_id"):
        old.pop(field, None)
        new.pop(field, None)
    _require(
        old == new,
        "replacement seed bank changed more than episode count and derived identity",
    )


def _checkpoint_rows(
    posttrain: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(posttrain.get("status") == "PASS", "pilot posttrain audit is not PASS")
    _require(
        posttrain.get("stage") == "pilot_posttrain"
        and posttrain.get("steps_per_control") == 1800,
        "pilot posttrain stage/step contract differs",
    )
    runs = posttrain.get("runs")
    _require(isinstance(runs, Mapping), "pilot posttrain audit lacks runs")
    rows: list[dict[str, Any]] = []
    for short, control in CONTROLS.items():
        run = runs.get(short)
        _require(isinstance(run, Mapping), f"posttrain audit lacks {short}")
        checkpoint = run.get("checkpoint")
        _require(isinstance(checkpoint, Mapping), f"{short} checkpoint identity missing")
        actual = _file_identity(str(checkpoint.get("path", "")))
        for field in ("path", "size_bytes", "sha256"):
            _require(
                actual[field] == checkpoint.get(field),
                f"{short} checkpoint {field} differs from posttrain audit",
            )
        rows.append(
            {
                "short": short,
                "control": control,
                "training_seed": 1,
                "checkpoint_step": 1800,
                **actual,
            }
        )
    return rows


def _invalid_partial_evidence(root: Path) -> dict[str, Any]:
    partial_root = (root / DEFAULT_INVALID_PARTIAL_RELATIVE).resolve()
    _require(partial_root.is_dir(), f"aborted partial root is missing: {partial_root}")
    completed = sorted(partial_root.rglob("completed_rollouts.json"))
    _require(
        not completed,
        "aborted 20-episode tree unexpectedly contains a completed-rollouts manifest",
    )
    result_paths = sorted(partial_root.rglob("_result_*.txt"))
    relatives = tuple(str(path.relative_to(partial_root)) for path in result_paths)
    _require(
        relatives == EXPECTED_PARTIAL_RESULTS,
        "aborted 20-episode result-file inventory differs from the witnessed pair",
    )
    return {
        "root": str(partial_root),
        "status": "INVALID_ABORTED_NOT_USED",
        "reason": "user_required_fastwam_aligned_100_episodes_before_gate",
        "declared_episode_count": ORIGINAL_EPISODES_PER_CELL,
        "completed_rollout_manifest_count": 0,
        "partial_result_file_count": len(result_paths),
        "partial_result_files": [_file_identity(path) for path in result_paths],
        "result_values_parsed": False,
        "result_values_used_for_decision": False,
        "eligible_for_pilot_gate": False,
        "eligible_for_reporting": False,
    }


def _runtime_source_identities(project_root: Path) -> dict[str, Any]:
    robotwin = project_root / "third_party/RoboTwin"
    module_root = project_root / "experiments/robotwin/policy_content_adapter"
    return {
        "author_eval_policy": _file_identity(robotwin / "script/eval_policy.py"),
        "pinned_eval_policy": _file_identity(module_root / "pinned_eval_policy.py"),
        "eval_robotwin_single": _file_identity(module_root / "eval_robotwin_single.py"),
        "amendment_implementation": _file_identity(Path(__file__)),
        "sim_robotwin_config": _file_identity(project_root / "configs/sim_robotwin.yaml"),
        "train_defaults_config": _file_identity(project_root / "configs/train.yaml"),
    }


def materialize_eval100_amendment(
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    output: str | Path | None = None,
    runtime_seed_bank_output: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], Path]:
    """Create the replacement bank and immutable user-requested amendment."""

    root = Path(experiment_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else (root / DEFAULT_AMENDMENT_RELATIVE).resolve()
    )
    bank_destination = (
        Path(runtime_seed_bank_output).expanduser().resolve()
        if runtime_seed_bank_output is not None
        else (root / DEFAULT_RUNTIME_BANK_RELATIVE).resolve()
    )
    _require(not destination.exists(), f"amendment already exists: {destination}")
    _require(not bank_destination.exists(), f"runtime seed bank already exists: {bank_destination}")

    materialization, materialization_path = _load_json(
        root / "materialization_manifest.json", "materialization manifest"
    )
    _require(
        materialization.get("kind") == "policy_pv2_actiondit_followup_materialization"
        and materialization.get("schema_version") == 1
        and materialization.get("status") == "PASS",
        "materialization kind/version/status differs",
    )
    protocol_identity = _verify_identity(
        materialization.get("protocol"), "mechanism protocol"
    )
    protocol, _ = _load_json(protocol_identity["path"], "mechanism protocol")
    pilot_gate = protocol.get("pilot_gate")
    _require(isinstance(pilot_gate, Mapping), "mechanism protocol lacks pilot gate")
    _require(
        pilot_gate.get("simulator_seed") == SIMULATOR_SEED
        and pilot_gate.get("episodes_per_task_domain") == ORIGINAL_EPISODES_PER_CELL
        and pilot_gate.get("official_random_macro_delta_min") == 0.03
        and pilot_gate.get("clean_macro_delta_min") == -0.03,
        "original pilot gate differs from the witnessed 20-episode contract",
    )
    original_bank_identity = _verify_identity(
        materialization.get("pilot_seed_bank"), "original seed53 pilot bank"
    )
    original_raw, _ = _load_json(original_bank_identity["path"], "original pilot bank")
    try:
        original_bank = validate_seed_bank_descriptor(
            original_raw, expected_purpose="dev_selection"
        )
    except ValueError as exc:
        raise Pv2Eval100AmendmentError(f"invalid original pilot bank: {exc}") from exc
    _require(
        original_bank["simulator_seed"] == SIMULATOR_SEED
        and original_bank["episodes_per_cell"] == ORIGINAL_EPISODES_PER_CELL,
        "original pilot bank is not seed53/20 episodes",
    )

    evaluator_source = project / "third_party/RoboTwin/script/eval_policy.py"
    replacement_bank = build_seed_bank_descriptor(
        simulator_seed=SIMULATOR_SEED,
        episodes_per_cell=RUNTIME_EPISODES_PER_CELL,
        evaluator_source=evaluator_source,
        purpose="dev_selection",
        disjoint_from=original_bank.get("disjoint_from", ()),
        lock_ancestry=original_bank.get("lock_ancestry", {}),
    )
    replacement_bank = validate_seed_bank_descriptor(
        replacement_bank, expected_purpose="dev_selection"
    )
    _same_candidate_stream(original_bank, replacement_bank)

    posttrain, posttrain_path = _load_json(
        root / "pilot_posttrain_audit.json", "pilot posttrain audit"
    )
    checkpoints = _checkpoint_rows(posttrain)
    partial = _invalid_partial_evidence(root)
    sources = _runtime_source_identities(project)

    _write_new_json(bank_destination, replacement_bank)
    replacement_bank_identity = _file_identity(bank_destination)
    core = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "profile": PROFILE,
        "decision": {
            "requested_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "requested_by": "user",
            "change": "episodes_per_task_domain_20_to_100",
            "reason": "align_online_evaluation_episode_count_with_original_fastwam_robotwin",
            "post_materialization": True,
            "before_any_complete_pilot_rollout": True,
            "result_driven": False,
            "training_checkpoints_changed": False,
            "training_recipe_changed": False,
            "gate_thresholds_changed": False,
        },
        "study": {
            "role": "post_hoc_actiondit_mechanism",
            "primary_pv1_remains_authoritative": True,
            "training_seed": 1,
            "controls": CONTROLS,
            "tasks": list(TASKS),
            "domains": list(DOMAINS),
        },
        "materialization": _file_identity(materialization_path),
        "mechanism_protocol": protocol_identity,
        "pilot_posttrain_audit": _file_identity(posttrain_path),
        "checkpoints": checkpoints,
        "original_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": ORIGINAL_EPISODES_PER_CELL,
            "seed_bank": original_bank_identity,
            "seed_bank_id": original_bank["simulator_seed_bank_id"],
            "status": "SUPERSEDED_BEFORE_COMPLETE_RESULT",
        },
        "runtime_evaluation": {
            "simulator_seed": SIMULATOR_SEED,
            "episodes_per_task_domain": RUNTIME_EPISODES_PER_CELL,
            "episodes_per_checkpoint": (
                len(TASKS) * len(DOMAINS) * RUNTIME_EPISODES_PER_CELL
            ),
            "episodes_both_controls": (
                len(CONTROLS)
                * len(TASKS)
                * len(DOMAINS)
                * RUNTIME_EPISODES_PER_CELL
            ),
            "seed_bank": replacement_bank_identity,
            "seed_bank_id": replacement_bank["simulator_seed_bank_id"],
            "seed_bank_purpose": "dev_selection",
            "episode_selection": "author_stock_candidate_scan_and_per_checkpoint_expert_filtering",
            "episode_pairing": "not_claimed",
            "shared_starting_seed_only": True,
            "per_checkpoint_expert_filtering": True,
            "required_task_matrix": [
                {"task": task, "domain": domain, "episodes": RUNTIME_EPISODES_PER_CELL}
                for task in TASKS
                for domain in DOMAINS
            ],
        },
        "gate": {
            "official_random_macro_delta_min": 0.03,
            "clean_macro_delta_min": -0.03,
            "both_required": True,
            "unchanged_from_original_mechanism_protocol": True,
        },
        "invalid_aborted_20_episode_artifacts": partial,
        "runtime_sources": sources,
        "claim_boundary": {
            "aligned_with_original_fastwam_episode_count": True,
            "aligned_episode_count_per_task_domain": RUNTIME_EPISODES_PER_CELL,
            "exact_episode_pairing_claimed": False,
            "same_simulator_seed_start_for_c1_c3": True,
            "old_partial_values_used": False,
        },
    }
    payload = {
        **core,
        "amendment_id": "pv2-eval100-amendment-v1:" + _canonical_sha256(core),
    }
    amendment_path = _write_new_json(destination, payload)
    validated, resolved = validate_eval100_amendment(amendment_path)
    return validated, resolved


def validate_eval100_amendment(
    path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Recompute every external identity and validate the narrow amendment."""

    payload, resolved = _load_json(path, "P-v2 eval100 amendment")
    _require(
        payload.get("kind") == KIND
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "PASS"
        and payload.get("profile") == PROFILE,
        "P-v2 eval100 amendment kind/version/status/profile differs",
    )
    core = dict(payload)
    amendment_id = core.pop("amendment_id", None)
    _require(
        amendment_id == "pv2-eval100-amendment-v1:" + _canonical_sha256(core),
        "P-v2 eval100 amendment id differs",
    )
    decision = payload.get("decision")
    _require(isinstance(decision, Mapping), "amendment decision missing")
    expected_decision = {
        "change": "episodes_per_task_domain_20_to_100",
        "reason": "align_online_evaluation_episode_count_with_original_fastwam_robotwin",
        "post_materialization": True,
        "before_any_complete_pilot_rollout": True,
        "result_driven": False,
        "training_checkpoints_changed": False,
        "training_recipe_changed": False,
        "gate_thresholds_changed": False,
    }
    for field, expected in expected_decision.items():
        _require(decision.get(field) == expected, f"amendment decision changed: {field}")

    materialization_identity = _verify_identity(
        payload.get("materialization"), "materialization manifest"
    )
    materialization, _ = _load_json(
        materialization_identity["path"], "materialization manifest"
    )
    _require(materialization.get("status") == "PASS", "materialization is not PASS")
    protocol_identity = _verify_identity(
        payload.get("mechanism_protocol"), "mechanism protocol"
    )
    materialized_protocol = materialization.get("protocol")
    _require(
        isinstance(materialized_protocol, Mapping)
        and all(
            materialized_protocol.get(field) == protocol_identity[field]
            for field in ("path", "size_bytes", "sha256")
        ),
        "amendment mechanism protocol differs from materialization",
    )
    protocol, _ = _load_json(protocol_identity["path"], "mechanism protocol")
    pilot_gate = protocol.get("pilot_gate")
    _require(isinstance(pilot_gate, Mapping), "mechanism protocol pilot gate missing")
    _require(
        pilot_gate.get("simulator_seed") == SIMULATOR_SEED
        and pilot_gate.get("episodes_per_task_domain") == ORIGINAL_EPISODES_PER_CELL
        and pilot_gate.get("official_random_macro_delta_min") == 0.03
        and pilot_gate.get("clean_macro_delta_min") == -0.03,
        "original mechanism pilot contract changed",
    )
    posttrain_identity = _verify_identity(
        payload.get("pilot_posttrain_audit"), "pilot posttrain audit"
    )
    posttrain, _ = _load_json(posttrain_identity["path"], "pilot posttrain audit")
    expected_rows = _checkpoint_rows(posttrain)
    _require(payload.get("checkpoints") == expected_rows, "amendment checkpoints changed")

    original = payload.get("original_evaluation")
    runtime = payload.get("runtime_evaluation")
    _require(isinstance(original, Mapping), "original evaluation block missing")
    _require(isinstance(runtime, Mapping), "runtime evaluation block missing")
    original_bank_identity = _verify_identity(
        original.get("seed_bank"), "original 20-episode bank"
    )
    runtime_bank_identity = _verify_identity(
        runtime.get("seed_bank"), "runtime 100-episode bank"
    )
    original_raw, _ = _load_json(original_bank_identity["path"], "original bank")
    runtime_raw, _ = _load_json(runtime_bank_identity["path"], "runtime bank")
    try:
        original_bank = validate_seed_bank_descriptor(
            original_raw, expected_purpose="dev_selection"
        )
        runtime_bank = validate_seed_bank_descriptor(
            runtime_raw, expected_purpose="dev_selection"
        )
    except ValueError as exc:
        raise Pv2Eval100AmendmentError(f"invalid seed bank: {exc}") from exc
    _same_candidate_stream(original_bank, runtime_bank)
    _require(
        original_bank["simulator_seed"] == SIMULATOR_SEED
        and original_bank["episodes_per_cell"] == ORIGINAL_EPISODES_PER_CELL
        and original.get("seed_bank_id") == original_bank["simulator_seed_bank_id"],
        "original evaluation seed/episode/bank identity differs",
    )
    _require(
        runtime_bank["simulator_seed"] == SIMULATOR_SEED
        and runtime_bank["episodes_per_cell"] == RUNTIME_EPISODES_PER_CELL
        and runtime.get("seed_bank_id") == runtime_bank["simulator_seed_bank_id"]
        and runtime.get("episodes_per_task_domain") == RUNTIME_EPISODES_PER_CELL
        and runtime.get("episodes_per_checkpoint") == 600
        and runtime.get("episodes_both_controls") == 1200,
        "runtime evaluation is not the required 3-task x 2-domain x 100 contract",
    )
    expected_matrix = [
        {"task": task, "domain": domain, "episodes": RUNTIME_EPISODES_PER_CELL}
        for task in TASKS
        for domain in DOMAINS
    ]
    _require(
        runtime.get("required_task_matrix") == expected_matrix,
        "runtime task/domain matrix differs",
    )
    for field, expected in {
        "episode_pairing": "not_claimed",
        "shared_starting_seed_only": True,
        "per_checkpoint_expert_filtering": True,
        "seed_bank_purpose": "dev_selection",
    }.items():
        _require(runtime.get(field) == expected, f"runtime evaluation differs: {field}")

    gate = payload.get("gate")
    _require(isinstance(gate, Mapping), "amendment gate block missing")
    _require(
        gate.get("official_random_macro_delta_min") == 0.03
        and gate.get("clean_macro_delta_min") == -0.03
        and gate.get("both_required") is True
        and gate.get("unchanged_from_original_mechanism_protocol") is True,
        "pilot gate thresholds changed",
    )
    partial = payload.get("invalid_aborted_20_episode_artifacts")
    _require(isinstance(partial, Mapping), "invalid partial evidence missing")
    partial_root = Path(str(partial.get("root", ""))).expanduser().resolve()
    actual_partial = _invalid_partial_evidence(partial_root.parent)
    _require(dict(partial) == actual_partial, "invalid partial evidence changed")
    _require(
        partial.get("result_values_parsed") is False
        and partial.get("result_values_used_for_decision") is False,
        "aborted 20-episode result values must remain unused",
    )

    sources = payload.get("runtime_sources")
    _require(isinstance(sources, Mapping), "runtime source identities missing")
    for name, identity in sources.items():
        _verify_identity(identity, f"runtime source {name}")
    study = payload.get("study")
    _require(isinstance(study, Mapping), "study block missing")
    _require(
        study.get("training_seed") == 1
        and study.get("controls") == CONTROLS
        and study.get("tasks") == list(TASKS)
        and study.get("domains") == list(DOMAINS),
        "study matrix differs",
    )
    return payload, resolved


def matching_checkpoint_row(
    amendment: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    control: str,
    training_seed: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Return and revalidate the one checkpoint row authorized for evaluation."""

    actual = _file_identity(checkpoint_path)
    matches = [
        row
        for row in amendment.get("checkpoints", [])
        if row.get("control") == control
        and row.get("training_seed") == training_seed
        and row.get("checkpoint_step") == checkpoint_step
    ]
    _require(len(matches) == 1, "eval100 amendment lacks one exact checkpoint row")
    row = matches[0]
    for field in ("path", "size_bytes", "sha256"):
        _require(row.get(field) == actual[field], f"eval100 checkpoint {field} differs")
    return dict(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    materialize.add_argument("--output")
    materialize.add_argument("--runtime-seed-bank-output")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--amendment", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        payload, path = materialize_eval100_amendment(
            experiment_root=args.experiment_root,
            output=args.output,
            runtime_seed_bank_output=args.runtime_seed_bank_output,
        )
    else:
        payload, path = validate_eval100_amendment(args.amendment)
    print(
        json.dumps(
            {
                "status": "PASS",
                "amendment_id": payload["amendment_id"],
                "path": str(path),
                "sha256": _file_sha256(path),
                "runtime_episodes_per_task_domain": RUNTIME_EPISODES_PER_CELL,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOMAINS",
    "KIND",
    "PROFILE",
    "RUNTIME_EPISODES_PER_CELL",
    "SIMULATOR_SEED",
    "TASKS",
    "Pv2Eval100AmendmentError",
    "matching_checkpoint_row",
    "materialize_eval100_amendment",
    "validate_eval100_amendment",
]
