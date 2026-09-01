from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.robotwin.policy_content_adapter.official_data import TASK_PROMPTS
from experiments.robotwin.policy_content_adapter.paired_data import (
    canonical_json_sha256,
    sha256_file,
)
from experiments.robotwin.policy_content_adapter.protocol import PROTOCOL_ID, TASKS
from experiments.robotwin.policy_content_adapter.task_text_cache import (
    CACHE_SCHEMA,
    CACHE_VERSION,
    load_task_embeddings,
    validate_task_text_cache,
)


def test_task_text_cache_validation_and_loading(tmp_path: Path) -> None:
    records = []
    for task in TASKS:
        path = tmp_path / f"{task}.pt"
        embedding = torch.randn(5, 4096).to(torch.bfloat16)
        torch.save({"task": task, "prompt": TASK_PROMPTS[task], "embedding": embedding}, path)
        records.append(
            {
                "task": task,
                "prompt": TASK_PROMPTS[task],
                "prompt_sha256": hashlib.sha256(TASK_PROMPTS[task].encode()).hexdigest(),
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": [5, 4096],
                "dtype": "torch.bfloat16",
            }
        )
    audit = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_VERSION,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "records": records,
        "record_inventory_sha256": canonical_json_sha256(records),
    }
    (tmp_path / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    result = validate_task_text_cache(tmp_path)
    assert result["status"] == "PASS" and result["tasks"] == 3
    loaded = load_task_embeddings(tmp_path)
    assert tuple(loaded) == TASKS
    assert all(value.shape == (5, 4096) for value in loaded.values())

