"""Tests for run-level count, stability, and GPU diagnostics."""

from __future__ import annotations

import json

import pytest

from evaluation_diagnostics import (
    finish_gpu_memory_tracking,
    start_gpu_memory_tracking,
    summarize_cluster_stability,
    summarize_count_bias,
)


def test_summarize_count_bias_groups_without_rowwise_relative_noise() -> None:
    """Bias should aggregate signed residuals by isotope, pair, and regime."""
    diagnostics = summarize_count_bias(
        observed_counts=[10.0, 20.0, 100.0, 200.0],
        predicted_counts=[11.0, 18.0, 120.0, 180.0],
        isotope_labels=["Cs-137", "Cs-137", "Co-60", "Co-60"],
        fe_indices=[0, 0, 1, 1],
        pb_indices=[0, 1, 0, 1],
        num_orientations=8,
        count_regime_lower_edges=(0.0, 50.0, 150.0),
    )

    assert diagnostics["available"] is True
    assert diagnostics["overall"]["signed_bias_counts"] == pytest.approx(-1.0)
    assert diagnostics["by_isotope"]["Cs-137"][
        "signed_relative_bias_pct"
    ] == pytest.approx(-100.0 / 30.0)
    assert diagnostics["by_shield_pair"]["9"]["fe_index"] == 1
    assert diagnostics["by_shield_pair"]["9"]["pb_index"] == 1
    assert (
        diagnostics["by_isotope_and_shield_pair"]["Co-60"]["9"]["signed_bias_counts"]
        == -20.0
    )
    assert diagnostics["by_count_regime"]["[0,50)"]["row_count"] == 2
    assert diagnostics["by_count_regime"]["[150,inf)"]["row_count"] == 1
    assert (
        diagnostics["by_isotope_and_count_regime"]["Cs-137"]["[0,50)"]["row_count"] == 2
    )
    coverage = diagnostics["shield_pair_coverage"]
    assert coverage["expected_pair_count"] == 64
    assert coverage["observed_pair_count"] == 4
    assert coverage["missing_pair_count"] == 60
    assert diagnostics["diagnostic_scope"] == "in_sample_final_fit_residual"
    assert diagnostics["calibration_bias_evidence"] is False
    assert diagnostics["count_regime_reference"] == "predicted_counts"
    json.dumps(diagnostics, allow_nan=False)


def test_count_regimes_use_prediction_or_explicit_reference() -> None:
    """Observed fluctuations must not choose their own residual regime by default."""
    default = summarize_count_bias(
        observed_counts=[150.0],
        predicted_counts=[40.0],
        isotope_labels=["Cs-137"],
        fe_indices=[0],
        pb_indices=[0],
        num_orientations=2,
        count_regime_lower_edges=(0.0, 50.0, 100.0),
    )
    explicit = summarize_count_bias(
        observed_counts=[150.0],
        predicted_counts=[40.0],
        isotope_labels=["Cs-137"],
        fe_indices=[0],
        pb_indices=[0],
        num_orientations=2,
        count_regime_lower_edges=(0.0, 50.0, 100.0),
        regime_reference_counts=[120.0],
    )

    assert default["by_count_regime"]["[0,50)"]["row_count"] == 1
    assert explicit["by_count_regime"]["[100,inf)"]["row_count"] == 1
    assert explicit["count_regime_reference"] == "provided_reference_counts"


def test_count_bias_rejects_invalid_counts_and_shield_indices() -> None:
    """Shield-pair decoding must not silently wrap invalid orientation indices."""
    kwargs = {
        "observed_counts": [1.0],
        "predicted_counts": [1.0],
        "isotope_labels": ["Cs-137"],
        "fe_indices": [0],
        "pb_indices": [0],
        "num_orientations": 8,
    }
    with pytest.raises(ValueError, match="pb_indices"):
        summarize_count_bias(**{**kwargs, "pb_indices": [8]})
    with pytest.raises(ValueError, match="non-negative"):
        summarize_count_bias(**{**kwargs, "observed_counts": [-1.0]})










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
