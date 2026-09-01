from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from experiments.robotwin.policy_content_adapter import package_pv2_actiondit_followup


def _tree(tmp_path: Path, *, terminal: bool) -> tuple[Path, Path, Path]:
    project = tmp_path / "FastWAM"
    experiment = tmp_path / "experiment"
    parent_docs = tmp_path / "docs"
    for relative in package_pv2_actiondit_followup.REQUIRED_PROJECT_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"project:{relative}\n", encoding="utf-8")
    for relative in package_pv2_actiondit_followup.REQUIRED_PARENT_DOCS:
        path = parent_docs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"doc:{relative}\n", encoding="utf-8")
    for relative in package_pv2_actiondit_followup.REQUIRED_EXPERIMENT_FILES:
        path = experiment / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "PASS", "name": relative}), encoding="utf-8")
    if terminal:
        for relative in ("summary.json", "summary.md", "completion_audit.json"):
            path = experiment / relative
            path.write_text(f"terminal:{relative}\n", encoding="utf-8")
    return project, experiment, parent_docs


def test_review_archive_is_deterministic_and_excludes_large_artifacts(
    tmp_path: Path,
) -> None:
    project, experiment, parent_docs = _tree(tmp_path, terminal=True)
    # These files exist beside the allowlisted inputs but must never be traversed.
    (experiment / "checkpoint.pt").write_bytes(b"large checkpoint placeholder")
    (experiment / "rollout.mp4").write_bytes(b"video placeholder")
    first = package_pv2_actiondit_followup.create_review_archive(
        output=tmp_path / "first.zip",
        project_root=project,
        experiment_root=experiment,
        parent_docs_root=parent_docs,
    )
    second = package_pv2_actiondit_followup.create_review_archive(
        output=tmp_path / "second.zip",
        project_root=project,
        experiment_root=experiment,
        parent_docs_root=parent_docs,
    )
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    with zipfile.ZipFile(first["archive"]["path"]) as archive:
        names = set(archive.namelist())
        assert "MANIFEST.json" in names
        assert "artifacts/summary.json" in names
        assert all(not name.endswith((".pt", ".mp4", ".log")) for name in names)
        manifest = json.loads(archive.read("MANIFEST.json"))
    assert manifest["exclusions"]["checkpoints"] is True
    assert manifest["exclusions"]["rollout_videos"] is True


def test_review_archive_refuses_before_terminal_completion(tmp_path: Path) -> None:
    project, experiment, parent_docs = _tree(tmp_path, terminal=False)
    with pytest.raises(
        package_pv2_actiondit_followup.Pv2FollowupPackageError,
        match="terminal summary/completion",
    ):
        package_pv2_actiondit_followup.create_review_archive(
            output=tmp_path / "incomplete.zip",
            project_root=project,
            experiment_root=experiment,
            parent_docs_root=parent_docs,
        )


def test_allowlist_rejects_forbidden_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "FastWAM"
    experiment = tmp_path / "experiment"
    docs = tmp_path / "docs"
    bad = project / "bad.pt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        package_pv2_actiondit_followup,
        "REQUIRED_PROJECT_FILES",
        ("bad.pt",),
    )
    monkeypatch.setattr(package_pv2_actiondit_followup, "REQUIRED_PARENT_DOCS", ())
    monkeypatch.setattr(package_pv2_actiondit_followup, "REQUIRED_EXPERIMENT_FILES", ())
    with pytest.raises(
        package_pv2_actiondit_followup.Pv2FollowupPackageError,
        match="forbidden",
    ):
        package_pv2_actiondit_followup.collect_members(
            project_root=project,
            experiment_root=experiment,
            parent_docs_root=docs,
        )
