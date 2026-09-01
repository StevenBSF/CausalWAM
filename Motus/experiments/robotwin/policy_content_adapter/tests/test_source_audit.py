from __future__ import annotations

from pathlib import Path

from experiments.robotwin.policy_content_adapter.source_audit import (
    build_source_audit,
    validate_source_audit,
)


def test_live_source_inventory_is_self_consistent() -> None:
    root = Path(__file__).parents[4]
    audit = build_source_audit(root)
    result = validate_source_audit(audit, verify_files=True)
    assert result["status"] == "PASS"
    assert result["file_count"] >= 30

