"""Repository ownership tests for the particle-filter package."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pf_repository_contains_no_simulation_implementation() -> None:
    """Geant4, observation generation, and raw-log writing stay shared."""
    forbidden = (
        ROOT / "native",
        ROOT / "obstacle_layouts",
        ROOT / "source_layouts",
        ROOT / "src" / "sim",
        ROOT / "src" / "measurement",
        ROOT / "src" / "spectrum",
        ROOT / "src" / "runtime",
    )

    assert all(not path.exists() for path in forbidden)


def test_pf_repository_contains_no_legacy_runtime_entry_points() -> None:
    """Retired simulation orchestration must not return outside the runtime."""
    forbidden = (
        ROOT / "environment.py",
        ROOT / "scripts" / "monitor_closed_loop_cui.py",
        ROOT / "src" / "realtime_demo.py",
        ROOT / "src" / "planning" / "measurement_workspace.py",
        ROOT / "src" / "planning" / "traversability.py",
    )

    assert all(not path.exists() for path in forbidden)


def test_pf_repository_owns_estimator_and_planner_only() -> None:
    """PF and its estimator-specific planner remain local."""
    assert (ROOT / "src" / "pf" / "particle_filter.py").is_file()
    assert (ROOT / "src" / "planning" / "dss_pp.py").is_file()
    assert (ROOT / "configs" / "pf" / "pf_strict_3d.json").is_file()


def test_service_adapter_delegates_without_importing_other_estimators() -> None:
    """The independent service must remain a thin local replay adapter."""
    source = (ROOT / "src" / "pf" / "service.py").read_text(encoding="utf-8")
    imported = {
        (node.module, alias.name)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("pf.replay", "replay_measurement_log") in imported
    assert "radiation_estimator_service_contracts" in source
    assert "three_d_estimation" not in source
    assert "orchestrator" not in source
    assert "ExecutionReceipt" not in source


def test_service_adapter_reuses_contract_file_boundaries() -> None:
    """Service control-file safety and artifact refs stay contract-owned."""
    source = (ROOT / "src" / "pf" / "service.py").read_text(encoding="utf-8")

    for shared_api in (
        "artifact_ref_from_path",
        "read_bounded_regular_file",
        "validate_new_file_path",
        "write_new_file",
    ):
        assert shared_api in source
    for removed_helper in (
        "def _assert_no_symlink_components(",
        "def _validate_response_target(",
        "def _read_regular_file(",
        "def _write_new_file(",
        "def _digest_regular_file(",
        "def _file_artifact(",
    ):
        assert removed_helper not in source


def test_service_contract_source_is_pinned_to_reviewed_revision() -> None:
    """The service wire contract must resolve from one immutable Git revision."""
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "e7184a117d5018ddef015182f357eb638b6fa377" in project


def test_service_contract_is_not_a_core_runtime_dependency() -> None:
    """Ordinary PF installs must not require the independent-service protocol."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert not any(
        dependency.startswith("radiation-estimator-service-contracts")
        for dependency in project["dependencies"]
    )
    assert project["optional-dependencies"]["service"] == [
        "radiation-estimator-service-contracts==0.1.0"
    ]
