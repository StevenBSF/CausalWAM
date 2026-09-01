"""Small, atomic experiment I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def atomic_write_text(path: str | Path, content: str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_torch(path: str | Path) -> Any:
    source = Path(path).expanduser().resolve()
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(source, map_location="cpu")


def parameter_checksum(module: torch.nn.Module) -> float:
    """Cheap deterministic checksum used only to prove an optimizer update."""
    return sum(float(parameter.detach().double().sum().item()) for parameter in module.parameters())


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash every named tensor so InitHead and training can share one init."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Cheap immutable-run identity used to reject cross-cache comparisons."""

    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


__all__ = [
    "atomic_torch_save",
    "atomic_write_text",
    "file_identity",
    "load_torch",
    "module_state_sha256",
    "parameter_checksum",
    "write_csv",
    "write_json",
]
