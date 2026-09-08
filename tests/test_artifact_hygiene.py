"""Verify the artifact audit rejects accidental tracking and loose output."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_artifacts.py"
SPEC = importlib.util.spec_from_file_location("audit_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_force_added_output_is_reported_without_deleting_it(tmp_path: Path) -> None:
    """Force-adding an ignored artifact must fail even under an allowed root."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/results/\n")
    output = tmp_path / "results" / "runs" / "opaque-id" / "posterior.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    subprocess.run(["git", "add", "-f", str(output)], cwd=tmp_path, check=True)
    _, problems = MODULE.audit(tmp_path)
    assert problems == [
        "Tracked generated/local file: results/runs/opaque-id/posterior.json"
    ]
    assert output.read_text() == "{}"


def test_untracked_clutter_is_reported(tmp_path: Path) -> None:
    """Ignored clutter must remain visible to the audit rather than to Git."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "results" / "pf-test-old").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "loose.log").write_text("old output")
    _, problems = MODULE.audit(tmp_path)
    assert "Unexpected results root: results/pf-test-old" in problems
    assert "Loose output; use a run directory: logs/loose.log" in problems


def test_run_layout_is_accepted(tmp_path: Path) -> None:
    """Normal run data and temporary exports must not cause violations."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for directory in ("results/runs/opaque-id", "logs/opaque-id", "tmp"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "tmp" / "preview.pdf").write_bytes(b"preview")
    _, problems = MODULE.audit(tmp_path)
    assert problems == []


def test_repository_index_excludes_generated_artifacts() -> None:
    """The normal test suite must reject generated files added to the index."""
    root = SCRIPT.parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"], cwd=root, text=True,
    ).splitlines()
    assert tracked == [], f"Remove generated/local files from Git: {tracked}"


def test_retired_build_and_runtime_directories_are_reported(tmp_path: Path) -> None:
    """Old build trees and migrated runtime assets must not accumulate again."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for directory in ("build/lib", "sim", "data/manchester_nuclear_assets"):
        (tmp_path / directory).mkdir(parents=True)
    _, problems = MODULE.audit(tmp_path)
    assert problems == [
        "Retired or runtime-owned directory: build",
        "Retired or runtime-owned directory: sim",
        "Retired or runtime-owned directory: data/manchester_nuclear_assets",
    ]
