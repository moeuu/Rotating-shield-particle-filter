"""Tests for online remaining-measurement budget estimation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from pf.estimator import MeasurementRecord, RotatingShieldPFConfig
from pf.estimator import RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticle
from pf.state import IsotopeState
from planning.remaining_measurements import RemainingMeasurementConfig
from planning.remaining_measurements import estimate_remaining_measurement_budget
from planning.remaining_measurements import _pairwise_signature_stats_batched


def _scalar_pairwise_signature_stats(
    response: np.ndarray,
    variance: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[float, float, int, float]:
    """Return a serial oracle for pairwise signature statistics."""
    pair_deficits = []
    pair_distances = []
    pair_weights = []
    for left in range(response.shape[1]):
        for right in range(left + 1, response.shape[1]):
            distance = 0.0
            for row in range(response.shape[0]):
                diff = float(response[row, left] - response[row, right])
                distance += diff * diff / float(variance[row])
            pair_distances.append(distance)
            pair_deficits.append(max(float(threshold) - distance, 0.0))
            pair_weights.append(float(weights[left]) * float(weights[right]))
    pair_weights_arr = np.asarray(pair_weights, dtype=float)
    pair_weights_arr /= float(np.sum(pair_weights_arr))
    distances_arr = np.asarray(pair_distances, dtype=float)
    deficits_arr = np.asarray(pair_deficits, dtype=float)
    return (
        float(np.sum(pair_weights_arr * deficits_arr)),
        float(np.min(distances_arr)),
        int(np.count_nonzero(distances_arr < float(threshold))),
        float(np.sum(pair_weights_arr * distances_arr)),
    )


def test_pairwise_signature_stats_match_scalar_oracle() -> None:
    """Batched signature deficits should match the serial pairwise oracle."""
    response = np.array(
        [
            [10.0, 13.0, 20.0],
            [11.0, 15.0, 21.0],
            [9.0, 12.0, 18.0],
        ],
        dtype=float,
    )
    variance = np.array([4.0, 9.0, 16.0], dtype=float)
    weights = np.array([0.5, 0.3, 0.2], dtype=float)
    threshold = 12.0

    expected = _scalar_pairwise_signature_stats(
        response,
        variance,
        weights,
        threshold,
    )
    actual = _pairwise_signature_stats_batched(
        response,
        variance,
        weights,
        threshold=threshold,
    )

    assert actual.deficit == pytest.approx(expected[0])
    assert actual.min_separation == pytest.approx(expected[1])
    assert actual.unresolved_pairs == expected[2]
    assert actual.weighted_increment == pytest.approx(expected[3])


def _build_two_mode_estimator() -> RotatingShieldPFEstimator:
    """Build a small estimator with two posterior source modes."""
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        shield_normals=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=2,
            planning_particles=2,
            planning_method="top_weight",
            pseudo_source_min_distinct_views=2,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    estimator.add_measurement_pose(np.array([1.0, 2.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.measurements.append(
        MeasurementRecord(
            z_k={"Cs-137": 120.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={"Cs-137": 120.0},
        )
    )
    filt = estimator.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([80.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.55)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([70.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([True], dtype=bool),
                verification_fail_streaks=np.array([1], dtype=int),
            ),
            log_weight=float(np.log(0.45)),
        ),
    ]
    filt.last_birth_residual_distinct_poses = 1
    return estimator


def test_remaining_measurement_estimate_is_json_safe() -> None:
    """Remaining budget estimates should include station and spectra ranges."""
    estimator = _build_two_mode_estimator()

    estimate = estimate_remaining_measurement_budget(
        estimator,
        next_pose_xyz=np.array([3.0, 2.0, 0.0], dtype=float),
        shield_program_pair_ids=(0, 1),
        live_time_s=1.0,
        config=RemainingMeasurementConfig(
            mode_cluster_radius_m=0.25,
            max_modes_per_isotope=4,
            max_particles=2,
            planning_method="top_weight",
            pairwise_separation_threshold=25.0,
            count_variance_floor=1.0,
            eta_default=0.7,
            max_reported_stations=20,
        ),
        current_station_count=1,
    )
    payload = estimate.to_dict()

    assert estimate.program_length == 2
    assert estimate.estimated_remaining_station_low <= (
        estimate.estimated_remaining_stations
    )
    assert estimate.estimated_remaining_spectra_high == (
        estimate.estimated_remaining_station_high * estimate.program_length
    )
    assert "same_isotope_separation" in estimate.components
    assert "Cs-137" in estimate.isotope_details
    assert payload["isotope_details"]["Cs-137"]["min_pairwise_separation"] >= 0.0
    json.dumps(payload, allow_nan=False)


def test_remaining_measurement_flags_high_surface_ambiguity() -> None:
    """Ceiling modes with weak pairwise separation should keep the mission open."""
    isotope = "Cs-137"
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array(
            [[2.0, 2.0, 10.0], [7.0, 2.0, 10.0]],
            dtype=float,
        ),
        shield_normals=np.eye(3, dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=2,
            planning_particles=2,
            planning_method="top_weight",
            position_max=(10.0, 10.0, 10.0),
            source_strength_prior_mean=100.0,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    estimator.add_measurement_pose(np.array([5.0, 5.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.measurements.append(
        MeasurementRecord(
            z_k={isotope: 120.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: 10.0},
        )
    )
    filt = estimator.filters[isotope]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 2.0, 10.0]], dtype=float),
                strengths=np.array([250.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[7.0, 2.0, 10.0]], dtype=float),
                strengths=np.array([250.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]

    estimate = estimate_remaining_measurement_budget(
        estimator,
        next_pose_xyz=np.array([5.0, 0.5, 0.5], dtype=float),
        shield_program_pair_ids=(0, 1),
        live_time_s=1.0,
        config=RemainingMeasurementConfig(
            mode_cluster_radius_m=0.25,
            max_modes_per_isotope=4,
            max_particles=2,
            planning_method="top_weight",
            high_surface_pairwise_separation_threshold=1.0e6,
            high_surface_absorption_q_multiple=2.0,
            eta_default=0.7,
            max_reported_stations=20,
        ),
        current_station_count=1,
    )

    assert estimate.components["high_surface_ambiguity"] > 0.0
    assert "high_surface_ambiguity" in estimate.unresolved_factors
    assert estimate.isotope_details[isotope]["high_surface_mode_count"] == 2


def test_remaining_measurement_residual_uses_surface_candidate_gain() -> None:
    """Residual budget should see batched rescue gain from known surface candidates."""
    estimator = _build_two_mode_estimator()
    estimator.measurements[0].z_k["Cs-137"] = 5000.0
    assert estimator.measurements[0].z_variance_k is not None
    estimator.measurements[0].z_variance_k["Cs-137"] = 100.0

    estimate = estimate_remaining_measurement_budget(
        estimator,
        next_pose_xyz=np.array([3.0, 2.0, 0.0], dtype=float),
        shield_program_pair_ids=(0, 1),
        live_time_s=1.0,
        config=RemainingMeasurementConfig(
            mode_cluster_radius_m=0.25,
            max_modes_per_isotope=4,
            max_particles=2,
            planning_method="top_weight",
            residual_chi2_threshold=9.0,
            count_variance_floor=1.0,
            residual_surface_gain_candidate_limit=16,
            eta_default=0.7,
            max_reported_stations=20,
        ),
        current_station_count=1,
    )

    assert estimate.gains["residual_surface"] > 0.0
    assert estimate.gains["residual"] >= estimate.gains["residual_surface"]


def test_remaining_measurement_marks_observed_zero_source_isotope_unresolved() -> None:
    """Observed isotope counts should not be considered resolved by a zero-source MAP."""
    isotope = "Eu-154"
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[5.0, 5.0, 10.0]], dtype=float),
        shield_normals=np.eye(3, dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(num_particles=2, max_sources=2),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    estimator.add_measurement_pose(np.array([5.0, 5.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.measurements.append(
        MeasurementRecord(
            z_k={isotope: 60.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: 9.0},
        )
    )
    filt = estimator.filters[isotope]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]

    estimate = estimate_remaining_measurement_budget(
        estimator,
        config=RemainingMeasurementConfig(
            max_particles=2,
            planning_method="top_weight",
            unresolved_absent_min_total_counts=25.0,
            unresolved_absent_min_max_counts=5.0,
            unresolved_absent_min_snr=2.0,
            eta_default=0.7,
            max_reported_stations=20,
        ),
        current_station_count=1,
    )

    assert estimate.components["isotope_absence"] > 0.0
    assert "isotope_absence" in estimate.unresolved_factors
    assert estimate.isotope_details[isotope]["map_source_count"] == 0
    assert estimate.isotope_details[isotope]["unresolved_absent_total_counts"] == 60.0


def test_remaining_measurement_ignores_low_snr_absent_isotope_counts() -> None:
    """Low-SNR decomposed counts should not keep an absent isotope unresolved."""
    isotope = "Eu-154"
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[5.0, 5.0, 10.0]], dtype=float),
        shield_normals=np.eye(3, dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(num_particles=2, max_sources=2),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    estimator.add_measurement_pose(np.array([5.0, 5.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.measurements.append(
        MeasurementRecord(
            z_k={isotope: 91.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: 1.0e8},
        )
    )
    filt = estimator.filters[isotope]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]

    estimate = estimate_remaining_measurement_budget(
        estimator,
        config=RemainingMeasurementConfig(
            max_particles=2,
            planning_method="top_weight",
            unresolved_absent_min_total_counts=25.0,
            unresolved_absent_min_max_counts=5.0,
            unresolved_absent_min_snr=2.0,
            eta_default=0.7,
            max_reported_stations=20,
        ),
        current_station_count=1,
    )

    assert estimate.components["isotope_absence"] == 0.0
    assert "isotope_absence" not in estimate.unresolved_factors
    assert estimate.isotope_details[isotope]["active_evidence"] == 0
    assert estimate.isotope_details[isotope]["unresolved_absent_total_counts"] == 0.0


def test_surface_stratified_rescue_keeps_high_surface_candidate() -> None:
    """Rescue candidate ranking should reserve slots for high surface strata."""
    candidates = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [5.0, 5.0, 10.0],
            [0.0, 5.0, 8.0],
        ],
        dtype=float,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=candidates,
        shield_normals=np.eye(3, dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            position_max=(10.0, 10.0, 10.0),
            report_mle_rescue_surface_quota_enable=True,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    scores = np.array([100.0, 90.0, 5.0, 4.0], dtype=float)
    selected = estimator._surface_stratified_rescue_indices(
        candidates,
        scores,
        np.ones(scores.size, dtype=bool),
        max_candidates=3,
    )

    assert 0 in selected
    assert 2 in selected
