"""Tests for truth-free source-count residual and transition diagnostics."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from pf.estimator import (
    JointStationObservation,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.pure_estimator import PurePFEstimator
from pf.state import IsotopeState
from pf.strength_prior import BoundedUniformStrengthPriorTestConfig
from pure_pf_test_support import (
    approved_full_spectrum_model,
    runtime_observation_model,
)


def _diagnostic_estimator() -> PurePFEstimator:
    """Build a small physical two-component PF with two acquired stations."""
    isotope = "Cs-137"
    isotopes = (isotope,)
    model = approved_full_spectrum_model(isotopes)
    observation_model = runtime_observation_model(isotopes)
    observation_model = replace(
        observation_model,
        detector_geometry=replace(
            observation_model.detector_geometry,
            count_radius_m=0.025,
        ),
    )
    estimator = PurePFEstimator(
        isotopes=isotopes,
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        observation_model=observation_model,
        pf_config=RotatingShieldPFConfig(
            estimator_profile="pf_strict",
            num_particles=3,
            max_sources=2,
            variable_cardinality=True,
            init_num_sources=(0, 2),
            use_gpu=True,
            gpu_device="cpu",
            position_max=(2.0, 2.0, 2.0),
            structural_rj_surface_chart_max_edge_m=2.0,
            strength_prior=BoundedUniformStrengthPriorTestConfig(
                minimum_cps_1m=1.0,
                maximum_cps_1m=100.0,
            ),
            joint_strength_block_batch_size=64,
        ),
        full_spectrum_generative_model=model,
        measurement_log_schema_version=2,
        config_hash="a" * 64,
        resolved_config_hash="b" * 64,
        measurement_log_sha256="c" * 64,
        random_seed=17,
    )
    detector_positions = (
        np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        np.asarray([1.5, 1.0, 1.0], dtype=np.float64),
    )
    for position in detector_positions:
        estimator.add_measurement_pose(position)
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    source_positions = np.asarray(
        [[0.2, 0.2, 0.0], [0.8, 0.2, 0.0]],
        dtype=np.float64,
    )
    chart_ids, surface_uv = filt.structural_surface_chart_coordinates(source_positions)
    state = IsotopeState(
        num_sources=2,
        strengths=np.asarray([10.0, 20.0], dtype=np.float64),
        surface_chart_ids=chart_ids,
        surface_uv=surface_uv,
    )
    for particle in filt.continuous_particles:
        particle.state = state
        particle.log_weight = float(-np.log(3.0))
    estimator._invalidate_posterior_summary_cache()
    bin_count = int(np.asarray(model.energy_axis_keV).size)
    estimator._joint_station_history = [
        JointStationObservation(
            spectrum_vb=np.zeros((1, bin_count), dtype=np.float64),
            energy_axis_keV=np.asarray(model.energy_axis_keV, dtype=np.float64),
            generative_contract_hash_sha256=model.contract_hash_sha256,
            pose_idx=pose_idx,
            detector_position_xyz_m=tuple(float(value) for value in position),
            fe_indices=np.asarray([0], dtype=np.int64),
            pb_indices=np.asarray([0], dtype=np.int64),
            live_times_s=np.asarray([30.0], dtype=np.float64),
            station_sequence_id=pose_idx,
        )
        for pose_idx, position in enumerate(detector_positions)
    ]
    return estimator


def _particle_state_snapshot(
    estimator: PurePFEstimator,
) -> tuple[tuple[object, ...], ...]:
    """Return immutable numeric evidence that a diagnostic did not alter PF."""
    rows: list[tuple[object, ...]] = []
    for isotope in estimator.joint_isotope_order():
        for particle in estimator.filters[isotope].continuous_particles:
            rows.append(
                (
                    isotope,
                    int(particle.state.num_sources),
                    tuple(np.asarray(particle.state.strengths).tolist()),
                    tuple(np.asarray(particle.state.surface_chart_ids).tolist()),
                    tuple(np.asarray(particle.state.surface_uv).reshape(-1).tolist()),
                    float(particle.log_weight),
                )
            )
    return tuple(rows)


def test_adjacent_cardinality_transition_rows_are_direction_resolved() -> None:
    """Raw adjacent-K proposal rows must retain direction and acceptance."""
    diagnostics = {
        "birth": {
            "by_cardinality_transition": {
                "1->2": {"attempted": 4, "accepted": 2},
                "2->3": {"attempted": 3, "accepted": 1},
            }
        },
        "merge": {
            "by_cardinality_transition": {
                "3->2": {"attempted": 5, "accepted": 2},
                "4->2": {"attempted": 7, "accepted": 1},
                "2->2": {"attempted": 9, "accepted": 8},
            }
        },
    }

    counts = RotatingShieldPFEstimator._adjacent_cardinality_transition_counts(
        diagnostics
    )

    assert counts == {
        "k_to_k_minus_1_attempted_count": 5,
        "k_to_k_minus_1_accepted_count": 2,
        "k_minus_1_to_k_attempted_count": 7,
        "k_minus_1_to_k_accepted_count": 3,
        "k_transition_1_to_2_attempted_count": 4,
        "k_transition_1_to_2_accepted_count": 2,
        "k_transition_2_to_3_attempted_count": 3,
        "k_transition_2_to_3_accepted_count": 1,
        "k_transition_3_to_2_attempted_count": 5,
        "k_transition_3_to_2_accepted_count": 2,
    }


def test_latest_station_transition_counts_sum_every_sweep() -> None:
    """Station reporting must sum raw transition rows over all exact sweeps."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.isotopes = ["Cs-137", "Co-60"]
    estimator.last_joint_rejuvenation_diagnostics = [
        {
            "k_to_k_minus_1_attempted_count.Cs-137": 3.0,
            "k_to_k_minus_1_accepted_count.Cs-137": 1.0,
            "k_minus_1_to_k_attempted_count.Cs-137": 4.0,
            "k_minus_1_to_k_accepted_count.Cs-137": 2.0,
            "k_transition_2_to_1_attempted_count.Cs-137": 3.0,
            "k_transition_2_to_1_accepted_count.Cs-137": 1.0,
        },
        {
            "k_to_k_minus_1_attempted_count.Cs-137": 5.0,
            "k_to_k_minus_1_accepted_count.Cs-137": 2.0,
            "k_minus_1_to_k_attempted_count.Co-60": 7.0,
            "k_minus_1_to_k_accepted_count.Co-60": 3.0,
            "k_transition_2_to_1_attempted_count.Cs-137": 5.0,
            "k_transition_2_to_1_accepted_count.Cs-137": 2.0,
            "k_transition_1_to_2_attempted_count.Co-60": 7.0,
            "k_transition_1_to_2_accepted_count.Co-60": 3.0,
        },
    ]

    counts = estimator.latest_station_adjacent_cardinality_transition_counts()

    assert counts["Cs-137"]["k_to_k_minus_1"] == {
        "attempted": 8,
        "accepted": 3,
    }
    assert counts["Cs-137"]["k_minus_1_to_k"] == {
        "attempted": 4,
        "accepted": 2,
    }
    assert counts["Co-60"]["k_minus_1_to_k"] == {
        "attempted": 7,
        "accepted": 3,
    }
    assert counts["Co-60"]["k_to_k_minus_1"] == {
        "attempted": 0,
        "accepted": 0,
    }
    assert counts["Cs-137"]["by_cardinality_transition"] == {
        "2->1": {"attempted": 8, "accepted": 3}
    }
    assert counts["Co-60"]["by_cardinality_transition"] == {
        "1->2": {"attempted": 7, "accepted": 3}
    }


def test_conditional_source_count_diagnostic_is_batched_and_nonmutating() -> None:
    """One-source comparison must be batch-invariant and leave PF untouched."""
    estimator = _diagnostic_estimator()
    before = _particle_state_snapshot(estimator)

    scalar_oracle = estimator.conditional_source_count_residual_diagnostics(
        strength_grid_size=2,
        local_uv_grid_size=2,
        candidate_batch_size=1,
    )
    batched = estimator.conditional_source_count_residual_diagnostics(
        strength_grid_size=2,
        local_uv_grid_size=2,
        candidate_batch_size=64,
    )

    assert _particle_state_snapshot(estimator) == before
    assert batched["available"] is True
    assert batched["truth_used"] is False
    assert batched["changes_inference"] is False
    assert batched["candidate_pair_count"] == 1
    json.dumps(batched, allow_nan=False, sort_keys=True)
    pair = batched["candidate_pairs"][0]
    oracle_pair = scalar_oracle["candidate_pairs"][0]
    assert batched["two_component_reference"]["reference_id"] == (
        "posterior_representative"
    )
    assert pair["two_component_reference_id"] == "posterior_representative"
    assert [row["station_sequence_id"] for row in pair["views"]] == [0, 1]
    assert [row["view_index"] for row in pair["views"]] == [0, 0]
    energy_axis = np.asarray(batched["energy_axis_keV"], dtype=np.float64)
    energy_edges = np.asarray(
        batched["energy_bin_edges_keV"],
        dtype=np.float64,
    )
    assert energy_edges.size == energy_axis.size + 1
    np.testing.assert_array_equal(energy_edges[:-1], energy_axis)
    np.testing.assert_allclose(np.diff(energy_edges), 2.0, rtol=0.0, atol=0.0)
    assert batched["energy_bin_left_edges_keV"] == batched["energy_axis_keV"]
    assert batched["generative_contract_hash_sha256"] == (
        estimator._full_spectrum_model().contract_hash_sha256
    )
    reconstruction = batched["figure_reconstruction"]
    assert reconstruction["residual_formula"] == (
        "observed_count_minus_predicted_mean_count"
    )
    assert reconstruction["transformations"] == {
        "energy_rebinning": "none",
        "count_normalization": "none",
        "smoothing": "none",
        "energy_bin_exclusion": "none",
        "view_exclusion": "none",
    }
    reference_views = batched["two_component_reference"]["views"]
    for reference_view, pair_view in zip(
        reference_views,
        pair["views"],
        strict=True,
    ):
        observed = np.asarray(
            reference_view["observed_spectrum_count_by_bin"],
            dtype=np.float64,
        )
        two_prediction = np.asarray(
            reference_view["predicted_mean_count_by_bin"],
            dtype=np.float64,
        )
        two_residual = np.asarray(
            reference_view["residual"][
                "full_spectrum_residual_count_by_bin"
            ],
            dtype=np.float64,
        )
        one_prediction = np.asarray(
            pair_view["one_source"]["predicted_mean_count_by_bin"],
            dtype=np.float64,
        )
        one_residual = np.asarray(
            pair_view["one_source"][
                "full_spectrum_residual_count_by_bin"
            ],
            dtype=np.float64,
        )
        assert observed.size == energy_axis.size
        assert two_prediction.size == energy_axis.size
        assert one_prediction.size == energy_axis.size
        np.testing.assert_allclose(
            observed - two_prediction,
            two_residual,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            observed - one_prediction,
            one_residual,
            rtol=0.0,
            atol=0.0,
        )
    assert pair["conditional_one_source_fit"][
        "full_history_log_likelihood"
    ] == pytest.approx(
        oracle_pair["conditional_one_source_fit"]["full_history_log_likelihood"],
        rel=0.0,
        abs=1.0e-10,
    )
    np.testing.assert_allclose(
        pair["conditional_one_source_fit"]["position_xyz_m"],
        oracle_pair["conditional_one_source_fit"]["position_xyz_m"],
        rtol=0.0,
        atol=0.0,
    )
    for view, oracle_view in zip(
        pair["views"],
        oracle_pair["views"],
        strict=True,
    ):
        assert view["one_source"]["spectral_l1_count_residual"] == pytest.approx(
            oracle_view["one_source"]["spectral_l1_count_residual"],
            rel=0.0,
            abs=1.0e-10,
        )
        np.testing.assert_allclose(
            view["one_source"]["full_spectrum_residual_count_by_bin"],
            oracle_view["one_source"]["full_spectrum_residual_count_by_bin"],
            rtol=0.0,
            atol=1.0e-10,
        )
