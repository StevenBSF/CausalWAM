"""Immutable author Motus_robotwin2 checkpoint lineage."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paired_data import canonical_json_sha256, sha256_file
from .protocol import MOTUS_ACTION_CHUNK, MOTUS_ACTION_DIM, PROTOCOL_ID


LINEAGE_SCHEMA = "motus_robotwin2_author_release_lineage"
LINEAGE_VERSION = 1
CHECKPOINT_NAME = "mp_rank_00_model_states.pt"


class LineageError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LineageError(message)


def _expert_contract(section: Mapping[str, Any], *, include_vlm: bool) -> dict[str, Any]:
    """Normalize YAML/JSON scalar spelling without weakening architecture checks."""

    result = {
        "hidden_size": int(section["hidden_size"]),
        "ffn_dim_multiplier": int(section["ffn_dim_multiplier"]),
        "norm_eps": float(section["norm_eps"]),
    }
    if include_vlm:
        vlm = section["vlm"]
        result["vlm"] = {
            "input_dim": int(vlm["input_dim"]),
            "projector_type": str(vlm["projector_type"]),
        }
    return result


def _file_identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"lineage file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    _require(len(revision) == 40, "Motus git revision is invalid")
    return revision


def probe_lineage_inputs(
    *,
    repo_root: str | Path,
    checkpoint_dir: str | Path,
    wan_dir: str | Path,
    vlm_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    checkpoint = Path(checkpoint_dir).resolve() / CHECKPOINT_NAME
    wan = Path(wan_dir).resolve()
    vlm = Path(vlm_dir).resolve()
    required = [
        checkpoint,
        Path(checkpoint_dir).resolve() / "config.json",
        repo / "configs" / "robotwin.yaml",
        repo / "inference" / "robotwin" / "Motus" / "utils" / "robotwin.yml",
        repo / "inference" / "robotwin" / "Motus" / "utils" / "stat.json",
        repo / "models" / "motus.py",
        repo / "models" / "action_expert.py",
        repo / "models" / "und_expert.py",
        repo / "models" / "wan_model.py",
        wan / "config.json",
        wan / "Wan2.2_VAE.pth",
        vlm / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "status": "PASS" if not missing else "BLOCKED",
        "missing": missing,
        "required_count": len(required),
        "present_count": len(required) - len(missing),
    }


def build_author_release_lineage(
    *,
    repo_root: str | Path,
    checkpoint_dir: str | Path,
    wan_dir: str | Path,
    vlm_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    checkpoint_root = Path(checkpoint_dir).resolve()
    wan = Path(wan_dir).resolve()
    vlm = Path(vlm_dir).resolve()
    probe = probe_lineage_inputs(
        repo_root=repo,
        checkpoint_dir=checkpoint_root,
        wan_dir=wan,
        vlm_dir=vlm,
    )
    _require(probe["status"] == "PASS", f"lineage inputs are missing: {probe['missing']}")
    train_config_path = repo / "configs" / "robotwin.yaml"
    inference_config_path = repo / "inference" / "robotwin" / "Motus" / "utils" / "robotwin.yml"
    checkpoint_config_path = checkpoint_root / "config.json"
    stats_path = repo / "inference" / "robotwin" / "Motus" / "utils" / "stat.json"
    train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    inference_config = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
    checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
    for name, config in (("training", train_config), ("inference", inference_config)):
        common = config.get("common", {})
        _require(common.get("action_dim") == MOTUS_ACTION_DIM, f"{name} action dim changed")
        _require(common.get("state_dim") == MOTUS_ACTION_DIM, f"{name} state dim changed")
        _require(common.get("num_video_frames") == 8, f"{name} video frame count changed")
        _require(common.get("video_action_freq_ratio") == 2, f"{name} action/video ratio changed")
        _require(common.get("num_video_frames") * common.get("video_action_freq_ratio") == MOTUS_ACTION_CHUNK, f"{name} action chunk changed")
        _require((common.get("video_height"), common.get("video_width")) == (384, 320), f"{name} image size changed")
    checkpoint_common = checkpoint_config.get("common", {})
    for field in (
        "action_dim",
        "state_dim",
        "num_video_frames",
        "video_height",
        "video_width",
        "global_downsample_rate",
        "video_action_freq_ratio",
    ):
        _require(
            checkpoint_common.get(field)
            == inference_config["common"].get(field),
            f"checkpoint/inference config differs at common.{field}",
        )
    for section in ("action_expert", "und_expert"):
        _require(
            _expert_contract(
                checkpoint_config[section], include_vlm=section == "und_expert"
            )
            == _expert_contract(
                inference_config["model"][section], include_vlm=section == "und_expert"
            ),
            f"checkpoint/inference config differs at {section}",
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))["robotwin2"]
    _require(len(stats["min"]) == len(stats["max"]) == MOTUS_ACTION_DIM, "normalization shape changed")

    source_paths = [
        repo / "models" / "motus.py",
        repo / "models" / "action_expert.py",
        repo / "models" / "und_expert.py",
        repo / "models" / "wan_model.py",
        repo / "experiments" / "robotwin" / "policy_content_adapter" / "model.py",
        repo / "experiments" / "robotwin" / "policy_content_adapter" / "observation_content.py",
    ]
    source_identities = [_file_identity(path) for path in source_paths]
    return {
        "schema": LINEAGE_SCHEMA,
        "schema_version": LINEAGE_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "lineage_id": "motus_robotwin2_author_release_v1",
        "base_kind": "author_release_stage3_robotwin2",
        "source": {
            "repository": "https://github.com/thu-ml/Motus",
            "revision": _git_revision(repo),
            "release_model_id": "motus-robotics/Motus_robotwin2",
        },
        "checkpoint": _file_identity(checkpoint_root / CHECKPOINT_NAME),
        "checkpoint_config": _file_identity(checkpoint_config_path),
        "wan": {
            "root": str(wan),
            "config": _file_identity(wan / "config.json"),
            "vae": _file_identity(wan / "Wan2.2_VAE.pth"),
        },
        "vlm": {
            "root": str(vlm),
            "config": _file_identity(vlm / "config.json"),
        },
        "training_config": _file_identity(train_config_path),
        "inference_config": _file_identity(inference_config_path),
        "normalization_stats": _file_identity(stats_path),
        "source_files": source_identities,
        "source_inventory_sha256": canonical_json_sha256(source_identities),
        "model_contract": {
            "camera_layout": "head_full_top_left_right_half_bottom",
            "image_size_hw": [384, 320],
            "state_dim": 14,
            "action_dim": 14,
            "action_chunk": 16,
            "inference_steps": 10,
            "checkpoint_load": "strict_true_required_before_execution",
        },
        "strict_load_audit_required": True,
    }


def validate_author_release_lineage(
    manifest: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    _require(manifest.get("schema") == LINEAGE_SCHEMA, "lineage schema changed")
    _require(manifest.get("schema_version") == LINEAGE_VERSION, "lineage version changed")
    _require(manifest.get("status") == "PASS", "lineage is not PASS")
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "lineage protocol changed")
    _require(manifest.get("base_kind") == "author_release_stage3_robotwin2", "base kind changed")
    identities = [
        manifest.get("checkpoint", {}),
        manifest.get("checkpoint_config", {}),
        manifest.get("training_config", {}),
        manifest.get("inference_config", {}),
        manifest.get("normalization_stats", {}),
        manifest.get("wan", {}).get("config", {}),
        manifest.get("wan", {}).get("vae", {}),
        manifest.get("vlm", {}).get("config", {}),
    ] + list(manifest.get("source_files", []))
    for identity in identities:
        _require(isinstance(identity, Mapping), "lineage identity is missing")
        sha = identity.get("sha256")
        _require(isinstance(sha, str) and len(sha) == 64, "lineage SHA is invalid")
        path = Path(str(identity.get("path", "")))
        if verify_files:
            _require(path.is_file(), f"lineage file disappeared: {path}")
            _require(path.stat().st_size == int(identity.get("size_bytes", -1)), f"lineage size changed: {path}")
            _require(sha256_file(path) == sha, f"lineage SHA changed: {path}")
    source_files = manifest.get("source_files", [])
    _require(canonical_json_sha256(source_files) == manifest.get("source_inventory_sha256"), "lineage source inventory changed")
    return {
        "status": "PASS",
        "lineage_id": manifest["lineage_id"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "source_inventory_sha256": manifest["source_inventory_sha256"],
    }


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("probe", "build"):
        item = sub.add_parser(command)
        item.add_argument("--repo-root", required=True)
        item.add_argument("--checkpoint-dir", required=True)
        item.add_argument("--wan-dir", required=True)
        item.add_argument("--vlm-dir", required=True)
        if command == "build":
            item.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--skip-file-hashes", action="store_true")
    args = parser.parse_args()
    if args.command == "probe":
        print(json.dumps(probe_lineage_inputs(repo_root=args.repo_root, checkpoint_dir=args.checkpoint_dir, wan_dir=args.wan_dir, vlm_dir=args.vlm_dir), sort_keys=True))
        return
    if args.command == "build":
        manifest = build_author_release_lineage(repo_root=args.repo_root, checkpoint_dir=args.checkpoint_dir, wan_dir=args.wan_dir, vlm_dir=args.vlm_dir)
        output = Path(args.output).resolve()
        _write_create_only(output, manifest)
        print(json.dumps({"status": "PASS", "path": str(output), "sha256": sha256_file(output), "checkpoint_sha256": manifest["checkpoint"]["sha256"]}, sort_keys=True))
        return
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = validate_author_release_lineage(manifest, verify_files=not args.skip_file_hashes)
    result.update(path=str(path), sha256=sha256_file(path))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
