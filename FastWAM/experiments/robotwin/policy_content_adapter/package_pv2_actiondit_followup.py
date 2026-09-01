"""Create the small, reviewable P-v2 follow-up code/protocol archive.

The archive is deliberately allowlist-based.  It never traverses run outputs,
and it rejects checkpoint/cache/video/log extensions even if a future caller
adds one to the allowlist accidentally.  Scientific JSON/Markdown deliverables
are included only when they exist at the terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .runtime_utils import PROJECT_ROOT


KIND = "policy_pv2_actiondit_followup_review_archive_manifest"
SCHEMA_VERSION = 1
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "outputs/policy_content_adapter/release_base_v1/pv2_actiondit_followup_v1"
).resolve()
DEFAULT_OUTPUT = (
    DEFAULT_EXPERIMENT_ROOT / "pv2_actiondit_followup_core_code_and_docs.zip"
).resolve()

REQUIRED_PROJECT_FILES = (
    "experiments/robotwin/policy_content_adapter/config_audit.py",
    "experiments/robotwin/policy_content_adapter/data.py",
    "experiments/robotwin/policy_content_adapter/losses.py",
    "experiments/robotwin/policy_content_adapter/model.py",
    "experiments/robotwin/policy_content_adapter/train.py",
    "experiments/robotwin/policy_content_adapter/rollout_policy.py",
    "experiments/robotwin/policy_content_adapter/eval_robotwin_single.py",
    "experiments/robotwin/policy_content_adapter/materialize_pv2_actiondit_followup.py",
    "experiments/robotwin/policy_content_adapter/pv2_actiondit_followup_audit.py",
    "experiments/robotwin/policy_content_adapter/pv2_actiondit_followup_report.py",
    "experiments/robotwin/policy_content_adapter/pv2_followup_eval100_amendment.py",
    "experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup.sh",
    "experiments/robotwin/policy_content_adapter/run_pv2_eval100_gate_after_pid.sh",
    "experiments/robotwin/policy_content_adapter/tests/test_pv2_actiondit_followup.py",
    "experiments/robotwin/policy_content_adapter/tests/test_rollout.py",
    "docs/pv2_actiondit_followup.md",
)
OPTIONAL_PROJECT_FILES = (
    "experiments/robotwin/policy_content_adapter/pv2_actiondit_followup_expansion.py",
    "experiments/robotwin/policy_content_adapter/pv2_actiondit_followup_confirmatory.py",
    "experiments/robotwin/policy_content_adapter/pv2_actiondit_followup_final.py",
    "experiments/robotwin/policy_content_adapter/eval_robotwin_pv2_confirmatory.py",
    "experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup_expansion.sh",
    "experiments/robotwin/policy_content_adapter/run_pv2_actiondit_followup_confirmatory.sh",
    "experiments/robotwin/policy_content_adapter/package_pv2_actiondit_followup.py",
    "experiments/robotwin/policy_content_adapter/tests/test_pv2_confirmatory.py",
    "experiments/robotwin/policy_content_adapter/tests/test_package_pv2_followup.py",
)
REQUIRED_PARENT_DOCS = (
    "policy_method.md",
    "policy_protocol_v2.md",
    "实验规划.md",
)
REQUIRED_EXPERIMENT_FILES = (
    "configs/seed_1/c1.yaml",
    "configs/seed_1/c3.yaml",
    "configs/seed_2/c1.yaml",
    "configs/seed_2/c3.yaml",
    "configs/seed_3/c1.yaml",
    "configs/seed_3/c3.yaml",
    "materialization_manifest.json",
    "implementation_protocol_audit.json",
    "pilot_posttrain_audit.json",
    "manifests/mechanism_protocol.json",
    "manifests/action_dit_initialization_audit.json",
    "manifests/dev_seed53_bank.json",
    "manifests/dev_seed53_100ep_bank_v1.json",
    "manifests/eval100_user_amendment_v1.json",
    "manifests/seed2_seed3_expansion_protocol_v1.json",
    "manifests/confirmatory_seed59_bank_v1.json",
    "manifests/confirmatory_seed59_amendment_v1.json",
)
OPTIONAL_TERMINAL_FILES = (
    "confirmatory_cpu_test_audit.json",
    "expansion_materialization_audit.json",
    "expansion_posttrain_audit.json",
    "pilot_decision.json",
    "pilot_report/pilot_summary.json",
    "pilot_report/pilot_summary.md",
    "pilot_report/pilot_report_audit.json",
    "summary.json",
    "summary.md",
    "completion_audit.json",
)
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".mp4",
    ".avi",
    ".mkv",
    ".log",
}
MAX_MEMBER_BYTES = 16 * 1024 * 1024


class Pv2FollowupPackageError(ValueError):
    """The review archive allowlist or terminal inputs are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pv2FollowupPackageError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(path: Path, archive_name: str, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"required archive member missing: {resolved}")
    _require(
        resolved.suffix.lower() not in FORBIDDEN_SUFFIXES,
        f"forbidden large/binary artifact type: {resolved}",
    )
    size = int(resolved.stat().st_size)
    _require(size <= MAX_MEMBER_BYTES, f"archive member exceeds 16 MiB: {resolved}")
    _require(
        not archive_name.startswith("/") and ".." not in Path(archive_name).parts,
        f"unsafe archive name: {archive_name}",
    )
    return {
        "source": str(resolved),
        "archive_name": archive_name,
        "role": role,
        "size_bytes": size,
        "sha256": _sha256(resolved),
    }


def collect_members(
    *,
    project_root: str | Path = PROJECT_ROOT,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    parent_docs_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    project = Path(project_root).expanduser().resolve()
    experiment = Path(experiment_root).expanduser().resolve()
    parent_docs = (
        Path(parent_docs_root).expanduser().resolve()
        if parent_docs_root is not None
        else (project.parent / "docs").resolve()
    )
    members: list[dict[str, Any]] = []
    for relative in REQUIRED_PROJECT_FILES:
        members.append(
            _member(project / relative, f"FastWAM/{relative}", role="core_code_or_doc")
        )
    for relative in OPTIONAL_PROJECT_FILES:
        path = project / relative
        if path.is_file():
            members.append(
                _member(path, f"FastWAM/{relative}", role="core_code_or_doc")
            )
    for relative in REQUIRED_PARENT_DOCS:
        members.append(
            _member(
                parent_docs / relative,
                f"CausalWAM/docs/{relative}",
                role="protocol_documentation",
            )
        )
    for relative in REQUIRED_EXPERIMENT_FILES:
        members.append(
            _member(
                experiment / relative,
                f"artifacts/{relative}",
                role="immutable_protocol_or_audit",
            )
        )
    for relative in OPTIONAL_TERMINAL_FILES:
        path = experiment / relative
        if path.is_file():
            members.append(
                _member(path, f"artifacts/{relative}", role="terminal_result")
            )
    names = [row["archive_name"] for row in members]
    _require(len(names) == len(set(names)), "archive member names contain duplicates")
    return sorted(members, key=lambda row: str(row["archive_name"]))


def _manifest(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminal_names = {
        str(row["archive_name"])
        for row in members
        if row.get("role") == "terminal_result"
    }
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "member_count": len(members),
        "total_uncompressed_bytes": sum(int(row["size_bytes"]) for row in members),
        "members": [dict(row) for row in members],
        "exclusions": {
            "checkpoints": True,
            "optimizer_state": True,
            "token_or_layer16_caches": True,
            "rollout_videos": True,
            "logs": True,
            "forbidden_suffixes": sorted(FORBIDDEN_SUFFIXES),
            "maximum_member_bytes": MAX_MEMBER_BYTES,
        },
        "terminal_artifacts_present": sorted(terminal_names),
    }


def create_review_archive(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path = PROJECT_ROOT,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    parent_docs_root: str | Path | None = None,
    require_terminal_completion: bool = True,
) -> dict[str, Any]:
    destination = Path(output).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite archive: {destination}")
    members = collect_members(
        project_root=project_root,
        experiment_root=experiment_root,
        parent_docs_root=parent_docs_root,
    )
    manifest = _manifest(members)
    terminal_names = set(manifest["terminal_artifacts_present"])
    if require_terminal_completion:
        required = {"artifacts/summary.json", "artifacts/summary.md", "artifacts/completion_audit.json"}
        _require(
            required.issubset(terminal_names),
            "terminal summary/completion artifacts are not all present",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for row in members:
                raw = Path(str(row["source"])).read_bytes()
                _require(
                    hashlib.sha256(raw).hexdigest() == row["sha256"],
                    f"archive source changed during packaging: {row['source']}",
                )
                info = zipfile.ZipInfo(str(row["archive_name"]))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            info = zipfile.ZipInfo("MANIFEST.json")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, manifest_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise Pv2FollowupPackageError(
                f"refusing to overwrite archive: {destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    archive_identity = {
        "path": str(destination),
        "size_bytes": int(destination.stat().st_size),
        "sha256": _sha256(destination),
    }
    return {**manifest, "archive": archive_identity}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_review_archive(
        output=args.output,
        experiment_root=args.experiment_root,
        require_terminal_completion=not args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pv2FollowupPackageError",
    "collect_members",
    "create_review_archive",
]
