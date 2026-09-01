"""Immutable author-stock Motus rollout settings."""

from pathlib import Path
from .evaluation import DOMAINS
from .lineage import validate_author_release_lineage
from .paired_data import canonical_json_sha256, sha256_file
from .protocol import TASKS
from .rollout_common import identity, need, read_json

SCHEMA = "motus_policy_content_adapter_rollout_settings"


def build_settings(lineage_path, robotwin_root, motus_root, simulator_seed=42):
    lineage, lineage_file = read_json(lineage_path)
    validate_author_release_lineage(lineage, verify_files=False)
    robotwin = Path(robotwin_root).resolve()
    motus = Path(motus_root).resolve()
    paths = [
        robotwin / "script/eval_policy.py",
        robotwin / "task_config/demo_clean.yml",
        robotwin / "task_config/demo_randomized.yml",
        motus / "inference/robotwin/Motus/deploy_policy.py",
        motus / "inference/robotwin/Motus/deploy_policy.yml",
        motus / "inference/robotwin/Motus/utils/robotwin.yml",
    ]
    sources = [identity(path) for path in paths]
    contract = {
        "simulator_seed": int(simulator_seed),
        "episodes_per_cell": 100,
        "instruction_type": "unseen",
        "task_configs": {"clean": "demo_clean", "official_random": "demo_randomized"},
        "inference_steps": 10,
        "episode_selection": "author_stock_expert_filter",
        "episode_pairing": "shared_start_seed_not_exact_pairing",
        "tasks": list(TASKS),
        "domains": list(DOMAINS),
    }
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
        "lineage": identity(lineage_file),
        "base_checkpoint": lineage["checkpoint"],
        "robotwin_root": str(robotwin),
        "motus_root": str(motus),
        "source_files": sources,
        "source_inventory_sha256": canonical_json_sha256(sources),
    }


def validate_settings(value, verify_files=True):
    need(
        value.get("schema") == SCHEMA and value.get("status") == "PASS",
        "invalid settings",
    )
    contract = value.get("contract", {})
    need(
        canonical_json_sha256(contract) == value.get("contract_sha256"),
        "settings changed",
    )
    need(
        contract.get("episodes_per_cell") == 100
        and contract.get("simulator_seed") == 42,
        "formal settings changed",
    )
    need(
        contract.get("episode_pairing") == "shared_start_seed_not_exact_pairing",
        "pairing claim changed",
    )
    sources = value.get("source_files", [])
    need(
        canonical_json_sha256(sources) == value.get("source_inventory_sha256"),
        "source inventory changed",
    )
    if verify_files:
        for item in sources:
            path = Path(item["path"])
            need(
                path.stat().st_size == item["size_bytes"]
                and sha256_file(path) == item["sha256"],
                f"source changed: {path}",
            )
    return dict(value)
