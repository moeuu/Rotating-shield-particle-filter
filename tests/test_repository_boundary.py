"""Repository ownership tests for the particle-filter package."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pf_repository_contains_no_simulation_implementation() -> None:
    """Geant4, observation generation, and raw-log writing stay shared."""
    forbidden = (
        ROOT / "native",
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
        ROOT / "src" / "realtime_demo.py",
    )

    assert all(not path.exists() for path in forbidden)


def test_pf_repository_owns_estimator_and_planner_only() -> None:
    """PF and its estimator-specific planner remain local."""
    assert (ROOT / "src" / "pf" / "particle_filter.py").is_file()
    assert (ROOT / "src" / "planning" / "dss_pp.py").is_file()
    assert (ROOT / "configs" / "pf" / "pf_strict_3d.json").is_file()
