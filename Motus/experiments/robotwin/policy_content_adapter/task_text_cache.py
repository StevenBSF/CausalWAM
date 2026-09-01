"""Three-prompt Motus UMT5 cache with immutable encoder provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .official_data import TASK_PROMPTS
from .paired_data import canonical_json_sha256, sha256_file
from .protocol import PROTOCOL_ID, TASKS


CACHE_SCHEMA = "motus_policy_task_text_cache"
CACHE_VERSION = 1


class TextCacheError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TextCacheError(message)


def _directory_identity(root: Path) -> dict[str, Any]:
    _require(root.is_dir(), f"tokenizer directory is missing: {root}")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _require(files, "tokenizer directory is empty")
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "sha256": canonical_json_sha256(files),
        "files": files,
    }


def build_task_text_cache(
    output_dir: str | Path,
    *,
    motus_repo_root: str | Path,
    wan_dir: str | Path,
    device: str,
    text_len: int = 512,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite text cache {output}")
    repo = Path(motus_repo_root).resolve()
    wan = Path(wan_dir).resolve()
    weight = wan / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer = wan / "google" / "umt5-xxl"
    _require(weight.is_file(), f"Motus T5 weights are missing: {weight}")
    tokenizer_identity = _directory_identity(tokenizer)
    if text_len <= 0:
        raise ValueError("text_len must be positive")
    # Use the exact author converter implementation rather than a second T5
    # wrapper.  Import is delayed so CPU validation never initializes CUDA.
    bak = repo / "bak"
    if str(bak) not in sys.path:
        sys.path.insert(0, str(bak))
    from wan.modules.t5 import T5EncoderModel

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("formal Motus T5 generation requires a CUDA device")
    torch.cuda.set_device(torch_device)
    encoder = T5EncoderModel(
        text_len=text_len,
        dtype=torch.bfloat16,
        device=torch_device,
        checkpoint_path=str(weight),
        tokenizer_path=str(tokenizer),
    )
    prompts = [TASK_PROMPTS[task] for task in TASKS]
    encoded = encoder(prompts, torch_device)
    _require(len(encoded) == len(TASKS), "T5 encoder did not return three embeddings")
    output.mkdir(parents=True)
    records = []
    for task, prompt, value in zip(TASKS, prompts, encoded, strict=True):
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
        tensor = tensor.detach().cpu().to(torch.bfloat16)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        _require(tensor.ndim == 2 and tensor.shape[1] == 4096, "T5 embedding shape changed")
        _require(0 < tensor.shape[0] <= text_len, "T5 embedding length changed")
        _require(bool(torch.isfinite(tensor).all()), "T5 embedding is non-finite")
        filename = f"{task}.pt"
        path = output / filename
        torch.save({"task": task, "prompt": prompt, "embedding": tensor}, path)
        records.append(
            {
                "task": task,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "file": filename,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
    audit = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "encoder": {
            "implementation": "Motus.bak.wan.modules.t5.T5EncoderModel",
            "weights": {
                "path": str(weight),
                "size_bytes": weight.stat().st_size,
                "sha256": sha256_file(weight),
            },
            "tokenizer": tokenizer_identity,
            "text_len": text_len,
            "dtype": "torch.bfloat16",
        },
        "records": records,
        "record_inventory_sha256": canonical_json_sha256(records),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def validate_task_text_cache(
    cache_dir: str | Path, *, verify_encoder_assets: bool = False
) -> dict[str, Any]:
    root = Path(cache_dir).resolve()
    audit_path = root / "audit.json"
    _require(audit_path.is_file(), "task text cache audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("schema") == CACHE_SCHEMA, "task text cache schema changed")
    _require(audit.get("schema_version") == CACHE_VERSION, "task text cache version changed")
    _require(audit.get("status") == "PASS", "task text cache is not PASS")
    _require(audit.get("protocol_id") == PROTOCOL_ID, "task text protocol changed")
    records = audit.get("records")
    _require(isinstance(records, list) and [item.get("task") for item in records] == list(TASKS), "task text records changed")
    _require(canonical_json_sha256(records) == audit.get("record_inventory_sha256"), "task text record SHA changed")
    for record in records:
        path = root / record["file"]
        _require(path.is_file(), f"task embedding disappeared: {path}")
        _require(path.stat().st_size == record["size_bytes"], "task embedding size changed")
        _require(sha256_file(path) == record["sha256"], "task embedding SHA changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("task") == record["task"] and payload.get("prompt") == record["prompt"], "task embedding metadata changed")
        embedding = payload.get("embedding")
        _require(isinstance(embedding, torch.Tensor) and list(embedding.shape) == record["shape"], "task embedding shape changed")
        _require(embedding.dtype == torch.bfloat16 and bool(torch.isfinite(embedding).all()), "task embedding values changed")
    if verify_encoder_assets:
        weight = audit["encoder"]["weights"]
        path = Path(weight["path"])
        _require(path.is_file() and path.stat().st_size == weight["size_bytes"], "T5 weights changed")
        _require(sha256_file(path) == weight["sha256"], "T5 weight SHA changed")
        current_tokenizer = _directory_identity(Path(audit["encoder"]["tokenizer"]["path"]))
        _require(current_tokenizer["sha256"] == audit["encoder"]["tokenizer"]["sha256"], "T5 tokenizer changed")
    return {
        "status": "PASS",
        "tasks": len(records),
        "audit_sha256": sha256_file(audit_path),
        "record_inventory_sha256": audit["record_inventory_sha256"],
    }


def load_task_embeddings(cache_dir: str | Path) -> dict[str, torch.Tensor]:
    root = Path(cache_dir).resolve()
    validate_task_text_cache(root)
    result = {}
    for task in TASKS:
        payload = torch.load(root / f"{task}.pt", map_location="cpu", weights_only=False)
        result[task] = payload["embedding"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", required=True)
    build.add_argument("--motus-repo-root", required=True)
    build.add_argument("--wan-dir", required=True)
    build.add_argument("--device", default="cuda:0")
    validate = sub.add_parser("validate")
    validate.add_argument("--cache-dir", required=True)
    validate.add_argument("--verify-encoder-assets", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        audit = build_task_text_cache(args.output, motus_repo_root=args.motus_repo_root, wan_dir=args.wan_dir, device=args.device)
        print(json.dumps({"status": "PASS", "output": str(Path(args.output).resolve()), "tasks": len(audit["records"])}, sort_keys=True))
        return
    print(json.dumps(validate_task_text_cache(args.cache_dir, verify_encoder_assets=args.verify_encoder_assets), sort_keys=True))


if __name__ == "__main__":
    main()
