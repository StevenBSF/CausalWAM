"""Hash every active Motus adapter implementation/config source file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .paired_data import canonical_json_sha256, sha256_file


AUDIT_SCHEMA = "motus_policy_content_adapter_source_audit"


class SourceAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditError(message)


def source_paths(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).resolve()
    package = root / "experiments" / "robotwin" / "policy_content_adapter"
    allowed_suffixes = {".py", ".yaml", ".json", ".sh", ".md"}
    paths = [
        path
        for path in package.rglob("*")
        if path.is_file()
        and path.suffix in allowed_suffixes
        and "__pycache__" not in path.parts
    ]
    paths.extend(
        [
            root / "models" / "motus.py",
            root / "models" / "wan_model.py",
            root / "configs" / "robotwin.yaml",
            root / "train" / "train.py",
            root / "data" / "robotwin2" / "robotwin_agilex_dataset.py",
            root / "utils" / "scheduler.py",
            root / "inference" / "robotwin" / "Motus" / "deploy_policy.py",
            root / "inference" / "robotwin" / "Motus" / "deploy_policy.yml",
            root / "inference" / "robotwin" / "Motus" / "policy_content_adapter.py",
            root / "inference" / "robotwin" / "Motus" / "qwen_processor.py",
            root / "inference" / "robotwin" / "Motus" / "models" / "motus.py",
            root / "inference" / "robotwin" / "Motus" / "models" / "wan_model.py",
        ]
    )
    unique = sorted(set(path.resolve() for path in paths))
    for path in unique:
        _require(path.is_file(), f"active source is missing: {path}")
    return unique


def build_source_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    records = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths(root)
    ]
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "repo_root": str(root),
        "file_count": len(records),
        "inventory_sha256": canonical_json_sha256(records),
        "files": records,
    }


def validate_source_audit(
    audit: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    _require(audit.get("schema") == AUDIT_SCHEMA, "source audit schema changed")
    _require(audit.get("status") == "PASS", "source audit is not PASS")
    records = audit.get("files")
    _require(isinstance(records, list) and records, "source audit has no files")
    _require(len(records) == int(audit.get("file_count", -1)), "source file count changed")
    _require(canonical_json_sha256(records) == audit.get("inventory_sha256"), "source inventory SHA changed")
    root = Path(str(audit.get("repo_root", "")))
    if verify_files:
        current_paths = source_paths(root)
        _require(
            [str(path.relative_to(root)) for path in current_paths]
            == [record["relative_path"] for record in records],
            "active source file set changed",
        )
        for record in records:
            path = root / record["relative_path"]
            _require(path.stat().st_size == record["size_bytes"], f"source size changed: {path}")
            _require(sha256_file(path) == record["sha256"], f"source SHA changed: {path}")
    return {
        "status": "PASS",
        "file_count": len(records),
        "inventory_sha256": audit["inventory_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", required=True)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--audit", required=True)
    validate.add_argument("--skip-files", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        audit = build_source_audit(args.repo_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "output": str(output), "file_count": audit["file_count"]}, sort_keys=True))
        return
    path = Path(args.audit).resolve()
    audit = json.loads(path.read_text(encoding="utf-8"))
    result = validate_source_audit(audit, verify_files=not args.skip_files)
    result.update(path=str(path), sha256=sha256_file(path))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
