"""GPU strict-load gate for the author Motus_robotwin2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .paired_data import sha256_file
from .runtime import instantiate_author_release, load_lineage


AUDIT_SCHEMA = "motus_robotwin2_strict_load_audit"


def run_strict_load_audit(
    *, lineage_path: str | Path, output_path: str | Path, local_cuda_index: int
) -> dict:
    lineage_path = Path(lineage_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    lineage = load_lineage(lineage_path, verify_files=True)
    model = instantiate_author_release(
        lineage,
        batch_size=1,
        local_cuda_index=local_cuda_index,
        strict=True,
    )
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "video": sum(parameter.numel() for parameter in model.video_model.parameters()),
        "action": sum(parameter.numel() for parameter in model.action_expert.parameters()),
        "vlm": sum(parameter.numel() for parameter in model.vlm_model.parameters()),
        "understanding": sum(
            parameter.numel() for parameter in model.und_expert.parameters()
        ),
    }
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError("strict-loaded Motus has an empty component")
    audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "lineage_manifest": {
            "path": str(lineage_path),
            "size_bytes": lineage_path.stat().st_size,
            "sha256": sha256_file(lineage_path),
        },
        "checkpoint": lineage["checkpoint"],
        "load_contract": {
            "strict": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "adapter_installed_during_load": False,
        },
        "parameter_counts": counts,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "cuda_device_name": torch.cuda.get_device_name(local_cuda_index),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    args = parser.parse_args()
    audit = run_strict_load_audit(
        lineage_path=args.lineage,
        output_path=args.output,
        local_cuda_index=args.local_cuda_index,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(Path(args.output).resolve()),
                "parameter_counts": audit["parameter_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
