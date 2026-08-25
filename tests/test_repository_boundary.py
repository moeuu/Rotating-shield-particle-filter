"""Repository ownership tests for the particle-filter package."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pf_package_requires_explicit_live_session_import() -> None:
    """The package must not dynamically facade live-session symbols."""
    import pf

    assert not hasattr(pf, "PFLiveSession")
    from pf.live_session import PFLiveSession

    assert PFLiveSession.__module__ == "pf.live_session"


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
        ROOT / "src" / "planning" / "candidate_generation.py",
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
    """PF must expose only its in-process estimator and live entry point."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert not (ROOT / "src" / "pf" / "service.py").exists()
    assert not (ROOT / "src" / "pf" / "service_cli.py").exists()
    assert "service" not in project.get("optional-dependencies", {})
    assert all(not name.endswith("-service") for name in project.get("scripts", {}))


def test_pf_repository_has_no_finalized_log_replay_surface() -> None:
    """Finalized logs must not have a batch inference API or command."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    forbidden = (
        ROOT / "main.py",
        ROOT / "src" / "pf" / "replay.py",
        ROOT / "scripts" / "evaluate_pure_pf_replay.py",
        ROOT / "scripts" / "run_pf_causal_replay_matrix.py",
        ROOT
        / "results"
        / "ral_ablation"
        / "diagnostic_runners"
        / "evaluate_exact_rj_shared_likelihood.py",
        ROOT / "configs" / "pf" / "diagnostics" / "causal_replay_matrix.example.json",
    )
    pf_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "pf").glob("*.py"))
    )
    forbidden_symbols = (
        "build_replay_estimator",
        "replay_measurement_log",
        "replay_records",
    )

    assert all(not path.exists() for path in forbidden)
    assert all(symbol not in pf_sources for symbol in forbidden_symbols)
    assert project.get("scripts", {}) == {
        "rotating-shield-pf-live": "pf.closed_loop:main"
    }
