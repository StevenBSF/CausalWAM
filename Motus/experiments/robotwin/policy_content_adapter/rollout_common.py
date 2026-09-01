"""Shared create-only helpers for Motus RoboTwin rollout artifacts."""

import json
from pathlib import Path
from .paired_data import sha256_file


class RolloutError(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise RolloutError(message)


def read_json(path):
    path = Path(path).resolve()
    need(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text())
    need(isinstance(value, dict), "JSON must be an object")
    return value, path


def identity(path):
    path = Path(path).resolve()
    need(path.is_file(), f"missing file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_identity(value):
    current = identity(value.get("path", ""))
    need(current == value, "file identity changed")
    return current


def write_json(path, value):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path
