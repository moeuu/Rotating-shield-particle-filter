"""Tests for run-level count, stability, and GPU diagnostics."""

from __future__ import annotations

import pytest

from evaluation_diagnostics import (
    finish_gpu_memory_tracking,
    start_gpu_memory_tracking,
    summarize_cluster_stability,
)










def test_summarize_cluster_stability_tracks_motion_and_count_stability() -> None:
    """Consecutive matched modes should expose motion and cardinality stability."""
    history = [
        {"Cs-137": ([[0.0, 0.0, 0.0]], [100.0])},
        {"Cs-137": ([[0.1, 0.0, 0.0]], [100.0])},
        {"Cs-137": ([[0.2, 0.0, 0.0]], [100.0])},
    ]

    diagnostics = summarize_cluster_stability(history, final_window=3)

    isotope = diagnostics["by_isotope"]["Cs-137"]
    assert isotope["final_window_count_stability_fraction"] == 1.0
    assert isotope["unmatched_cluster_event_count"] == 0
    shift = isotope["consecutive_matched_cluster_shift_m"]
    assert shift["count"] == 2
    assert shift["median"] == pytest.approx(0.1)


def test_cluster_stability_requires_one_transition() -> None:
    """Availability starts at two states and one consecutive transition."""
    one_state = summarize_cluster_stability(
        [{"Cs-137": ([[0.0, 0.0, 0.0]], [100.0])}]
    )
    two_states = summarize_cluster_stability(
        [
            {"Cs-137": ([[0.0, 0.0, 0.0]], [100.0])},
            {"Cs-137": ([[0.1, 0.0, 0.0]], [100.0])},
        ]
    )

    assert one_state["available"] is False
    assert one_state["by_isotope"]["Cs-137"]["available"] is False
    assert two_states["available"] is True
    assert two_states["by_isotope"]["Cs-137"]["transition_count"] == 1


def test_cluster_stability_separates_final_window_and_same_count_replacement() -> None:
    """A far same-count replacement is one birth and death, not stable motion."""
    history = [
        {"Cs-137": ([[0.0, 0.0, 0.0]], [100.0])},
        {"Cs-137": ([[10.0, 0.0, 0.0]], [200.0])},
        {"Cs-137": ([[10.1, 0.0, 0.0]], [220.0])},
        {"Cs-137": ([[10.2, 0.0, 0.0]], [200.0])},
    ]

    diagnostics = summarize_cluster_stability(
        history,
        final_window=2,
        match_gate_m=0.5,
    )

    isotope = diagnostics["by_isotope"]["Cs-137"]
    assert isotope["unmatched_cluster_appearance_count"] == 1
    assert isotope["unmatched_cluster_disappearance_count"] == 1
    assert isotope["unmatched_cluster_event_count"] == 2
    assert isotope["same_cardinality_cluster_replacement_transition_count"] == 1
    assert "birth_death_event_count" not in isotope
    assert isotope["all_history_consecutive_matched_cluster_shift_m"]["count"] == 2
    assert isotope["final_window_consecutive_matched_cluster_shift_m"]["count"] == 1
    final_strength = isotope[
        "final_window_consecutive_matched_strength_abs_drift_cps_1m"
    ]
    assert final_strength["count"] == 1
    assert final_strength["median"] == 20.0


def test_gpu_memory_tracking_is_explicitly_unavailable_for_cpu() -> None:
    """CPU runs should report unavailable GPU metrics instead of fake zeros."""
    baseline = start_gpu_memory_tracking("cpu")
    completed = finish_gpu_memory_tracking(baseline)

    assert completed["available"] is False
    assert completed["device"] == "cpu"
    assert completed["scope"] == "torch_cuda_allocator_current_process"
    assert completed["includes_external_cuda_allocations"] is False
    assert completed["includes_geant4_sidecar"] is False
