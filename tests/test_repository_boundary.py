"""Repository ownership tests for the particle-filter package."""

from __future__ import annotations

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


def test_pf_repository_has_no_out_of_process_service_surface() -> None:
    """PF must expose only its in-process estimator and replay entry points."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert not (ROOT / "src" / "pf" / "service.py").exists()
    assert not (ROOT / "src" / "pf" / "service_cli.py").exists()
    assert "service" not in project.get("optional-dependencies", {})
    assert all(not name.endswith("-service") for name in project.get("scripts", {}))
