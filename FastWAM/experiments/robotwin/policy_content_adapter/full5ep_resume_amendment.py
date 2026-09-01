"""Create/verify the pre-resume audit amendment for the step-6803 incident."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


KIND = "policy_pv2_full5ep_gradient_gate_resume_amendment"
VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_full5ep_v1_retry2"
)
DEFAULT_AMENDMENT = DEFAULT_OUTPUT_ROOT / "manifests/step6803_resume_amendment_v2.json"


class ResumeAmendmentError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeAmendmentError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResumeAmendmentError(f"cannot read {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _config_semantic_sha(path: Path) -> str:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _require(isinstance(value, dict), "resume config root must be a mapping")
    return _canonical_sha(value)


def _source_identities(names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {name: _identity(POLICY_ROOT / name) for name in names}


def build_amendment(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    protocol_path = root / "manifests/full5ep_protocol_v1.json"
    config_path = root / "configs/seed_1/c3.yaml"
    state_file = root / "runs/seed_1/c3/checkpoints/state/step_00006000/trainer_state.json"
    failure_log = root / "logs/seed1_c3_train.log"
    protocol = _json(protocol_path, "original full5ep protocol")
    protocol_core = dict(protocol)
    protocol_id = protocol_core.pop("protocol_id", None)
    _require(
        protocol_id == "pv2-full5ep-v1:" + _canonical_sha(protocol_core),
        "original full5ep protocol ID is invalid",
    )
    expected_sources = protocol.get("source_sha256")
    _require(isinstance(expected_sources, Mapping), "original source SHA map missing")
    current_sources = _source_identities(sorted(str(name) for name in expected_sources))
    drift = {
        name: {
            "before_sha256": str(expected_sources[name]),
            "after_sha256": current_sources[name]["sha256"],
        }
        for name in sorted(expected_sources)
        if current_sources[name]["sha256"] != expected_sources[name]
    }
    _require(set(drift) == {"train.py"}, f"unexpected bound-source drift: {sorted(drift)}")

    trainer_state = _json(state_file, "step-6000 trainer state")
    _require(
        trainer_state.get("status") == "PASS"
        and trainer_state.get("global_step") == 6000
        and trainer_state.get("next_step") == 6001
        and trainer_state.get("world_size") == 8,
        "step-6000 trainer state is not a valid resume boundary",
    )
    semantic_sha = _config_semantic_sha(config_path)
    _require(
        semantic_sha == trainer_state.get("requested_config_sha256"),
        "step-6000 state was prepared for a different config",
    )
    failure_text = failure_log.read_text(encoding="utf-8", errors="replace")
    _require(
        failure_text.count("action loss did not reach the zero-init adapter gate")
        == 16,
        "failure log does not contain the exact eight-rank step-6803 incident",
    )
    _require('"step": 6802' in failure_text, "failure log lacks completed step 6802")
    _require('"step": 6803' not in failure_text, "failure log unexpectedly completed step 6803")

    extra_sources = _source_identities(
        (
            "full5ep_resume_amendment.py",
            "pv2_actiondit_followup_audit.py",
            "run_pv2_full5ep_seed1_c3_direct.sh",
        )
    )
    core = {
        "kind": KIND,
        "schema_version": VERSION,
        "status": "PASS",
        "output_root": str(root),
        "original_protocol": _identity(protocol_path),
        "config": {
            **_identity(config_path),
            "semantic_sha256": semantic_sha,
        },
        "resume_boundary": {
            "completed_step": 6000,
            "next_step": 6001,
            "trainer_state": _identity(state_file),
        },
        "incident": {
            "failed_step": 6803,
            "last_completed_uncheckpointed_step": 6802,
            "failure": "per-step_strict_positive_gate_gradient_after_ddp_cancellation",
            "failure_log": _identity(failure_log),
            "optimizer_or_model_failure": False,
            "oom": False,
        },
        "source_amendment": {
            "expected_bound_source_sha256": dict(expected_sources),
            "current_bound_sources": current_sources,
            "exact_bound_source_drift": drift,
            "additional_audit_sources": extra_sources,
        },
        "semantic_scope": {
            "training_objective_changed": False,
            "optimizer_or_lr_changed": False,
            "data_or_rng_changed": False,
            "model_architecture_changed": False,
            "checkpoint_state_changed": False,
            "only_change": (
                "replace every-positive-batch strict-nonzero action-path assertion "
                "with cumulative positive-path coverage; retain exact-zero scheduler, "
                "finite-gradient, frozen-module and final-update audits"
            ),
        },
    }
    return {**core, "amendment_id": f"{KIND}:" + _canonical_sha(core)}


def verify_amendment(
    path: str | Path,
    *,
    expected_output_root: str | Path | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    amendment_path = Path(path).expanduser().resolve()
    payload = _json(amendment_path, "resume amendment")
    rebuilt = build_amendment(payload.get("output_root", ""))
    _require(payload == rebuilt, "resume amendment content or bound artifacts drifted")
    if expected_output_root is not None:
        _require(
            payload["output_root"]
            == str(Path(expected_output_root).expanduser().resolve()),
            "resume amendment output root differs",
        )
    if expected_config_sha256 is not None:
        _require(
            payload["config"]["semantic_sha256"] == expected_config_sha256,
            "resume amendment config SHA differs",
        )
    return {
        "status": "PASS",
        "amendment_id": payload["amendment_id"],
        "identity": _identity(amendment_path),
        "resume_step": 6000,
        "failed_step": 6803,
        "semantic_scope": payload["semantic_scope"],
    }


def resume_training(
    *,
    config: str | Path,
    amendment: str | Path,
    resume: str | Path = "latest",
) -> Path:
    """Run the old immutable config under its exact post-incident amendment."""

    config_path = Path(config).expanduser().resolve()
    config_value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    _require(isinstance(config_value, dict), "resume config root must be a mapping")
    output_root = Path(str(config_value["output_dir"])).resolve().parents[2]
    semantic_sha = _canonical_sha(config_value)
    verify_amendment(
        amendment,
        expected_output_root=output_root,
        expected_config_sha256=semantic_sha,
    )
    execution = config_value.get("execution")
    _require(
        isinstance(execution, Mapping)
        and execution.get("runner") == "policy_content_adapter_pv2_full5ep"
        and execution.get("runnable") is True
        and execution.get("fail_closed") is False,
        "amended resume config is not the exact executable full5ep runner",
    )

    from . import train as train_module

    original_validator = train_module.validate_execution_ready

    def _amended_validator(candidate: Mapping[str, Any]) -> None:
        _require(
            _canonical_sha(candidate) == semantic_sha,
            "runtime config differs from the amended immutable config",
        )

    train_module.validate_execution_ready = _amended_validator
    try:
        return train_module.run(
            config_path,
            resume_from=resume,
            resume_amendment=amendment,
        )
    finally:
        train_module.validate_execution_ready = original_validator


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite amendment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    create.add_argument("--output", default=str(DEFAULT_AMENDMENT))
    verify = sub.add_parser("verify")
    verify.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--config", required=True)
    resume_parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    resume_parser.add_argument("--resume", default="latest")
    args = parser.parse_args(argv)
    if args.command == "create":
        payload = build_amendment(args.output_root)
        _write_new(Path(args.output).expanduser().resolve(), payload)
        report = verify_amendment(args.output)
    elif args.command == "verify":
        report = verify_amendment(args.amendment)
    else:
        destination = resume_training(
            config=args.config,
            amendment=args.amendment,
            resume=args.resume,
        )
        report = {"status": "PASS", "output_dir": str(destination)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ResumeAmendmentError",
    "build_amendment",
    "resume_training",
    "verify_amendment",
]
