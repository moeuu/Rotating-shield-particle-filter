"""Regression tests for simulation fidelity shortcuts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_demo import run_live_pf
from spectrum.runtime_counts import RuntimeCountExtractor


def test_codex_instructions_define_simulation_fidelity_guardrails() -> None:
    """Codex project instructions should forbid low-fidelity runtime shortcuts."""
    root = Path(__file__).resolve().parents[1]
    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    policy_text = (root / "docs" / "simulation_fidelity_policy.md").read_text(
        encoding="utf-8"
    )
    normalized_agents_text = " ".join(agents_text.split())
    required_phrases = (
        "Simulation fidelity policy",
        "Runtime simulation fidelity must take priority over speed shortcuts.",
        "CUI mode means \"run without the Isaac Sim GUI.\"",
        "surrogate transport",
        "expected-count observations",
        "weighted or capped Geant4 histories",
        "peak-window or full-spectrum-continuum runtime count extraction",
        "Full simulation",
        "uv run python main.py --full-simulation",
    )

    for phrase in required_phrases:
        assert phrase in normalized_agents_text
    assert "Prohibited Runtime Shortcuts" in policy_text
    assert "Allowed Performance Work" in policy_text


def test_codex_instructions_require_parallel_first_heavy_runtime_work() -> None:
    """Codex project instructions should require parallel-first heavy code."""
    root = Path(__file__).resolve().parents[1]
    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    parallel_policy = (root / "docs" / "compute_parallelism_policy.md").read_text(
        encoding="utf-8"
    )
    fidelity_policy = (root / "docs" / "simulation_fidelity_policy.md").read_text(
        encoding="utf-8"
    )
    combined = " ".join((agents_text + "\n" + parallel_policy).split())
    required_phrases = (
        "Compute parallelism policy",
        "batched, GPU, Geant4-threaded, or process-parallel",
        "Do not add scalar Python runtime loops over particles",
        "Parallelization must preserve the same physics",
        "serial-vs-parallel equivalence test",
    )

    for phrase in required_phrases:
        assert phrase in combined
    assert "docs/compute_parallelism_policy.md" in fidelity_policy


def test_demo_entrypoints_do_not_expose_expected_count_bypasses() -> None:
    """Demo entrypoints should feed PF updates from spectra, not expected counts."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = (
        root / "main.py",
        root / "src" / "realtime_demo.py",
        root / "src" / "baselines" / "legacy_no_shield" / "cli.py",
        root / "src" / "baselines" / "legacy_no_shield" / "realtime_demo.py",
    )
    forbidden_tokens = (
        "--count",
        "count_mode",
        "_detect_isotopes_from_expected",
        "def _expected_counts(",
    )

    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_runtime_count_extraction_defaults_to_response_poisson() -> None:
    """Runtime count extraction should standardize on response_poisson."""
    assert RuntimeCountExtractor.STANDARD_METHOD == "response_poisson"
    assert RuntimeCountExtractor.validate_count_method("response_poisson") == (
        "response_poisson"
    )


def test_runtime_rejects_peak_window_count_method(tmp_path: Path) -> None:
    """Runtime simulations should reject lower-fidelity peak-window counting."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "peak_window"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_runtime_rejects_photopeak_nnls_count_method(tmp_path: Path) -> None:
    """Runtime simulations should keep photopeak NNLS out of PF ingestion."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "photopeak_nnls"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_runtime_rejects_full_spectrum_response_count_method(tmp_path: Path) -> None:
    """Runtime simulations should reject continuum-fitting count extraction."""
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"spectrum_count_method": "response_matrix"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response_poisson"):
        run_live_pf(
            live=False,
            sim_config_path=config_path.as_posix(),
            max_steps=0,
            save_outputs=False,
        )


def test_wall_absorber_is_explicit_native_transport_mode() -> None:
    """Wall absorption should be an explicit mode, not a hidden shortcut."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(
        encoding="utf-8"
    )

    assert 'transport_mode) != "absorber"' in source
    assert "TransportSteppingAction" in source
    assert "SetTrackStatus(fStopAndKill)" in source
    assert 'result.metadata["absorbing_volume_count"]' in source
