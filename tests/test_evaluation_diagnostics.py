"""Tests for run-level count, model-order, stability, and GPU diagnostics."""

from __future__ import annotations

import json

import pytest

from evaluation_diagnostics import (
    finish_gpu_memory_tracking,
    start_gpu_memory_tracking,
    summarize_cluster_stability,
    summarize_count_bias,
    summarize_model_diagnostics,
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


def test_summarize_model_diagnostics_selects_heldout_deviance() -> None:
    """Selected spectrum count should index held-out deviance explicitly."""
    diagnostics = summarize_model_diagnostics(
        {
            "Cs-137": {
                "selected_count": 1,
                "criterion_margin_to_runner_up": 2.0,
                "condition_number": 7.0,
            }
        },
        {
            "Cs-137": {
                "selected_count": 1,
                "bic_margin_to_runner_up": 3.5,
                "condition_number": 5.0,
                "measurement_count": 99,
                "heldout_observation_count": 3,
                "best_heldout_count": 1,
                "heldout_deviance_by_count": [10.0, 3.0, 4.0],
            },
            "joint_multi_isotope": {
                "available": True,
                "selected_cardinality_key": "Cs-137:1",
                "bic_margin_to_runner_up": 4.0,
            },
        },
    )

    isotope = diagnostics["by_isotope"]["Cs-137"]
    assert isotope["bic_margin_to_runner_up"] == 3.5
    assert isotope["response_condition_number"] == 5.0
    assert isotope["selected_spectrum_bin_heldout_deviance"] == 3.0
    assert isotope[
        "selected_spectrum_bin_heldout_deviance_per_observation"
    ] == 1.0
    assert diagnostics["spectrum_bin_heldout_deviance"]["median"] == 3.0
    assert diagnostics["spectrum_bin_heldout_deviance_per_observation"][
        "median"
    ] == 1.0
    assert diagnostics["joint_multi_isotope"]["bic_margin_to_runner_up"] == 4.0


def test_heldout_deviance_never_uses_training_measurement_count() -> None:
    """Per-observation deviance requires an explicit actual holdout count."""
    without_holdout_count = summarize_model_diagnostics(
        {},
        {
            "Cs-137": {
                "available": True,
                "selected_count": 0,
                "measurement_count": 100,
                "n_observations": 100,
                "heldout_deviance_by_count": [20.0],
            }
        },
    )
    with_holdout_count = summarize_model_diagnostics(
        {},
        {
            "Cs-137": {
                "available": True,
                "selected_count": 0,
                "measurement_count": 100,
                "heldout_observation_count": 4,
                "heldout_deviance_by_count": [20.0],
            }
        },
    )

    missing = without_holdout_count["by_isotope"]["Cs-137"]
    assert missing["heldout_deviance_observation_count"] is None
    assert missing["heldout_deviance_per_observation_available"] is False
    assert missing["selected_spectrum_bin_heldout_deviance_per_observation"] is None
    assert without_holdout_count[
        "spectrum_bin_heldout_deviance_per_observation"
    ]["count"] == 0
    present = with_holdout_count["by_isotope"]["Cs-137"]
    assert present["heldout_deviance_observation_count"] == 4
    assert present["selected_spectrum_bin_heldout_deviance_per_observation"] == 5.0


def test_model_diagnostics_availability_requires_an_evaluable_metric() -> None:
    """Placeholder isotope rows must not make model diagnostics available."""
    unavailable = summarize_model_diagnostics(
        {"Cs-137": {"selected_count": 1}},
        {"Co-60": {"available": False}},
    )
    joint = summarize_model_diagnostics(
        {},
        {"joint_multi_isotope": {"available": True}},
    )

    assert unavailable["available"] is False
    assert unavailable["by_isotope"]["Cs-137"]["available"] is False
    assert joint["available"] is True


def test_unavailable_sparse_evidence_falls_back_and_sanitizes_deviance() -> None:
    """Unavailable sparse payloads must not override valid report diagnostics."""
    diagnostics = summarize_model_diagnostics(
        {
            "Cs-137": {
                "selected_count": 2,
                "criterion_margin_to_runner_up": 2.0,
                "condition_number": 9.0,
            },
            "Co-60": {"selected_count": 1},
            "Eu-154": {"selected_count": 1, "condition_number": 12.0},
        },
        {
            "Cs-137": {
                "available": False,
                "selected_count": 0,
                "bic_margin_to_runner_up": 99.0,
                "condition_number": 1.0,
                "heldout_deviance_by_count": [float("nan")],
            },
            "Co-60": {
                "available": True,
                "selected_count": 1,
                "measurement_count": 2,
                "heldout_deviance_by_count": [8.0, float("nan")],
            },
        },
    )

    cesium = diagnostics["by_isotope"]["Cs-137"]
    assert cesium["sparse_evidence_available"] is False
    assert cesium["selected_count"] == 2
    assert cesium["bic_margin_to_runner_up"] == 2.0
    assert cesium["response_condition_number"] == 9.0
    cobalt = diagnostics["by_isotope"]["Co-60"]
    assert cobalt["heldout_deviance_by_count"] == [8.0, None]
    assert cobalt["selected_spectrum_bin_heldout_deviance"] is None
    europium = diagnostics["by_isotope"]["Eu-154"]
    assert europium["sparse_evidence_available"] is False
    assert europium["selected_count"] == 1
    assert europium["response_condition_number"] == 12.0
    json.dumps(diagnostics, allow_nan=False)


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
    assert isotope["birth_death_event_count"] == 0
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
    assert isotope["birth_event_count"] == 1
    assert isotope["death_event_count"] == 1
    assert isotope["birth_death_event_count"] == 2
    assert isotope["same_count_birth_death_transition_count"] == 1
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
