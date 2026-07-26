"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from measurement.kernels import ShieldParams
from pf.likelihood import expected_counts_per_source
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import (
    IsotopeParticleFilter,
    IsotopeParticle,
    MeasurementData,
    PFConfig,
)
from pf.state import IsotopeState


def _build_filter(
    p_birth: float,
    min_strength: float,
    max_sources: int,
    num_particles: int = 10,
    **kwargs: object,
) -> IsotopeParticleFilter:
    """Utility to create an isotope PF with configurable birth/death parameters."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float
    )
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=num_particles,
        max_sources=max_sources,
        resample_threshold=0.5,
        strength_sigma=0.0,
        background_sigma=0.0,
        min_strength=min_strength,
        p_birth=p_birth,
        **kwargs,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    estimator._ensure_kernel_cache()
    return estimator.filters["Cs-137"]


@pytest.mark.parametrize("likelihood_model", ["gaussian", "student_t"])
def test_structural_count_likelihood_matches_runtime_sequence(
    likelihood_model: str,
) -> None:
    """Batched structural evidence should equal the runtime count likelihood."""
    import torch

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        count_likelihood_model=likelihood_model,
        count_likelihood_df=6.0,
        transport_model_rel_sigma=0.08,
        transport_model_abs_sigma=1.5,
        spectrum_count_rel_sigma=0.12,
        spectrum_count_abs_sigma=0.7,
        low_count_abs_sigma=0.5,
        low_count_transition_counts=25.0,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=0.4,
        shield_contrast_likelihood_enable=True,
        shield_contrast_likelihood_weight=0.3,
        shield_contrast_min_count=1.0,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=0.2,
        shield_view_ratio_likelihood_min_total_count=1.0,
        use_gpu=False,
    )
    detector_positions = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [2.0, 1.0, 0.5],
            [2.0, 1.0, 0.5],
            [2.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = np.array(
        [[0.0, 0.20, 0.10], [0.20, 0.0, 0.15], [0.10, 0.15, 0.0]],
        dtype=float,
    )
    covariance[3:, 3:] = np.array(
        [[0.0, 0.12, 0.08], [0.12, 0.0, 0.11], [0.08, 0.11, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([72.0, 48.0, 31.0, 26.0, 39.0, 57.0], dtype=float),
        observation_variances=np.array(
            [4.0, 5.0, 3.5, 2.5, 4.5, 6.0],
            dtype=float,
        ),
        detector_positions=detector_positions,
        fe_indices=np.array([0, 1, 2, 0, 1, 2], dtype=int),
        pb_indices=np.array([2, 1, 0, 2, 1, 0], dtype=int),
        live_times=np.ones(6, dtype=float),
        station_sequence_ids=np.array([0, 0, 0, 1, 1, 1], dtype=int),
        observation_count_covariance=covariance,
    )
    lambda_kp = np.array(
        [
            [68.0, 75.0, 58.0, 83.0],
            [51.0, 44.0, 55.0, 39.0],
            [28.0, 35.0, 24.0, 42.0],
            [29.0, 22.0, 34.0, 18.0],
            [36.0, 43.0, 31.0, 49.0],
            [61.0, 52.0, 67.0, 46.0],
        ],
        dtype=float,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(
        data,
        lambda_kp,
    )
    runtime = torch.zeros(lambda_kp.shape[1], dtype=torch.float64)
    for rows in (np.arange(3), np.arange(3, 6)):
        station_covariance = covariance[np.ix_(rows, rows)]
        runtime += filt._log_likelihood_sequence_gpu(
            torch.as_tensor(lambda_kp[rows], dtype=torch.float64),
            data.z_k[rows],
            data.observation_variances[rows],
            observation_count_covariance=station_covariance,
        )
    scalar = np.asarray(
        [
            filt._structural_count_log_likelihood_np(data, lambda_kp[:, idx])
            for idx in range(lambda_kp.shape[1])
        ],
        dtype=float,
    )

    assert np.allclose(structural, runtime.numpy(), rtol=1.0e-11, atol=1.0e-10)
    assert np.allclose(structural, scalar, rtol=1.0e-12, atol=1.0e-12)


def test_structural_per_row_blocks_match_runtime_likelihood_product() -> None:
    """Independent runtime rows must remain independent at one detector pose."""
    import torch

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="student_t",
        count_likelihood_df=6.0,
        spectrum_count_rel_sigma=0.2,
        spectrum_count_abs_sigma=0.8,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=0.75,
        shield_contrast_likelihood_enable=True,
        shield_contrast_likelihood_weight=0.4,
        shield_contrast_min_count=1.0,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=0.3,
        shield_view_ratio_likelihood_min_total_count=1.0,
        use_gpu=False,
    )
    data = MeasurementData(
        z_k=np.array([44.0, 19.0, 67.0], dtype=float),
        observation_variances=np.array([3.0, 2.0, 4.0], dtype=float),
        detector_positions=np.repeat(
            np.array([[1.0, 2.0, 0.5]], dtype=float),
            3,
            axis=0,
        ),
        fe_indices=np.array([0, 1, 2], dtype=int),
        pb_indices=np.array([2, 1, 0], dtype=int),
        live_times=np.ones(3, dtype=float),
        station_sequence_ids=np.arange(3, dtype=int),
    )
    lambda_kp = np.array(
        [[41.0, 52.0], [22.0, 16.0], [62.0, 73.0]],
        dtype=float,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    runtime = torch.zeros(lambda_kp.shape[1], dtype=torch.float64)
    lambda_t = torch.as_tensor(lambda_kp, dtype=torch.float64)
    for row_index in range(int(data.z_k.size)):
        rows = slice(row_index, row_index + 1)
        runtime += filt._log_likelihood_sequence_gpu(
            lambda_t[rows],
            data.z_k[rows],
            data.observation_variances[rows],
        )

    assert np.allclose(structural, runtime.numpy(), rtol=1.0e-11, atol=1.0e-10)


def test_same_xyz_revisited_station_sequences_are_not_merged() -> None:
    """Explicit sequence IDs must separate revisits to identical coordinates."""
    import torch

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="gaussian",
        spectrum_count_rel_sigma=0.25,
        spectrum_count_abs_sigma=1.0,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=0.8,
        shield_contrast_likelihood_enable=True,
        shield_contrast_likelihood_weight=0.5,
        shield_contrast_min_count=1.0,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=0.4,
        shield_view_ratio_likelihood_min_total_count=1.0,
        use_gpu=False,
    )
    covariance = np.zeros((4, 4), dtype=float)
    covariance[:2, :2] = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=float)
    covariance[2:, 2:] = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=float)
    data = MeasurementData(
        z_k=np.array([58.0, 23.0, 31.0, 71.0], dtype=float),
        observation_variances=np.array([4.0, 2.0, 3.0, 5.0], dtype=float),
        detector_positions=np.repeat(
            np.array([[1.5, 1.5, 0.5]], dtype=float),
            4,
            axis=0,
        ),
        fe_indices=np.array([0, 1, 2, 3], dtype=int),
        pb_indices=np.array([3, 2, 1, 0], dtype=int),
        live_times=np.ones(4, dtype=float),
        station_sequence_ids=np.array([10, 10, 11, 11], dtype=int),
        observation_count_covariance=covariance,
    )
    lambda_kp = np.array(
        [[54.0, 63.0], [26.0, 19.0], [35.0, 27.0], [66.0, 78.0]],
        dtype=float,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    lambda_t = torch.as_tensor(lambda_kp, dtype=torch.float64)
    runtime_separate = torch.zeros(lambda_kp.shape[1], dtype=torch.float64)
    for rows in (np.arange(2), np.arange(2, 4)):
        runtime_separate += filt._log_likelihood_sequence_gpu(
            lambda_t[rows],
            data.z_k[rows],
            data.observation_variances[rows],
            observation_count_covariance=covariance[np.ix_(rows, rows)],
        )
    runtime_merged = filt._log_likelihood_sequence_gpu(
        lambda_t,
        data.z_k,
        data.observation_variances,
        observation_count_covariance=covariance,
    )

    assert np.allclose(
        structural,
        runtime_separate.numpy(),
        rtol=1.0e-11,
        atol=1.0e-10,
    )
    assert not np.allclose(structural, runtime_merged.numpy())


def test_structural_mixed_station_covariance_routes_match_runtime() -> None:
    """Only station blocks that used covariance may take that runtime branch."""
    import torch

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="gaussian",
        station_view_covariance_enable=False,
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=False,
        use_gpu=False,
    )
    covariance = np.zeros((4, 4), dtype=float)
    covariance[:2, :2] = np.array(
        [[0.0, 4.0], [4.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([18.0, 27.0, 31.0, 14.0], dtype=float),
        observation_variances=np.array([3.0, 4.0, 5.0, 2.0], dtype=float),
        detector_positions=np.array(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        fe_indices=np.array([0, 1, 0, 1], dtype=int),
        pb_indices=np.array([1, 0, 1, 0], dtype=int),
        live_times=np.ones(4, dtype=float),
        station_sequence_ids=np.array([0, 0, 1, 1], dtype=int),
        runtime_likelihood_routes=np.array(
            ["count_covariance", "count_covariance", "count", "count"],
            dtype=str,
        ),
        observation_count_covariance=covariance,
    )
    lambda_kp = np.array(
        [[20.0, 16.0], [25.0, 30.0], [29.0, 35.0], [16.0, 12.0]],
        dtype=float,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    runtime = filt._log_likelihood_sequence_gpu(
        torch.as_tensor(lambda_kp[:2], dtype=torch.float64),
        data.z_k[:2],
        data.observation_variances[:2],
        observation_count_covariance=covariance[:2, :2],
    )
    runtime += filt._log_likelihood_sequence_gpu(
        torch.as_tensor(lambda_kp[2:], dtype=torch.float64),
        data.z_k[2:],
        data.observation_variances[2:],
    )

    np.testing.assert_allclose(
        structural,
        runtime.numpy(),
        rtol=1.0e-11,
        atol=1.0e-10,
    )


def test_structural_covariance_regularization_is_independent_per_station() -> None:
    """One invalid station covariance must not force fallback in another."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="gaussian",
        station_view_covariance_enable=False,
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=False,
        use_gpu=False,
    )
    covariance = np.zeros((4, 4), dtype=float)
    covariance[:2, :2] = np.array(
        [[0.0, 10.0], [10.0, 0.0]],
        dtype=float,
    )
    covariance[2:, 2:] = np.array(
        [[0.0, 1000.0], [1000.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([10.0, 12.0, 9.0, 11.0], dtype=float),
        observation_variances=np.ones(4, dtype=float),
        detector_positions=np.array(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        fe_indices=np.array([0, 1, 0, 1], dtype=int),
        pb_indices=np.array([1, 0, 1, 0], dtype=int),
        live_times=np.ones(4, dtype=float),
        station_sequence_ids=np.array([0, 0, 1, 1], dtype=int),
        runtime_likelihood_routes=np.asarray(
            ["count_covariance"] * 4,
            dtype=str,
        ),
        observation_count_covariance=covariance,
    )
    lambda_kp = np.array(
        [[10.0, 14.0], [12.0, 8.0], [9.0, 13.0], [11.0, 7.0]],
        dtype=float,
    )

    combined = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    separate = np.zeros(lambda_kp.shape[1], dtype=float)
    for block_id in (0, 1):
        mask = data.station_sequence_ids == block_id
        separate += filt._structural_count_log_likelihood_matrix_np(
            filt._measurement_rows(data, mask),
            lambda_kp[mask],
        )

    np.testing.assert_allclose(combined, separate, rtol=1.0e-12, atol=1.0e-12)


def test_measurement_history_records_runtime_likelihood_block_ids() -> None:
    """Joint rows should share an ID while per-row and delayed rows do not."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        max_sources=1,
        init_num_sources=(1, 1),
        birth_enable=False,
        resample_threshold=0.0,
        use_tempering=False,
        use_gpu=False,
        parallel_isotope_updates=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.update_pair_sequence(
        (
            ({"Cs-137": 10.0}, 0, 1, 1.0, {"Cs-137": 2.0}),
            ({"Cs-137": 12.0}, 1, 0, 1.0, {"Cs-137": 2.5}),
        ),
        pose_idx=0,
    )
    estimator.update_pair(
        {"Cs-137": 9.0},
        pose_idx=0,
        fe_index=2,
        pb_index=3,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 2.0},
    )
    estimator.begin_deferred_pose_update()
    estimator.update_pair(
        {"Cs-137": 8.0},
        pose_idx=0,
        fe_index=3,
        pb_index=2,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 2.0},
    )
    estimator.update_pair(
        {"Cs-137": 7.0},
        pose_idx=0,
        fe_index=4,
        pb_index=1,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 2.0},
    )
    assert estimator.finalize_deferred_pose_update() == 2
    estimator.update_pair(
        {"Cs-137": 6.0},
        pose_idx=0,
        fe_index=5,
        pb_index=0,
        live_time_s=1.0,
    )

    record_ids = [
        int(record.station_sequence_id)
        for record in estimator.measurements
        if record.station_sequence_id is not None
    ]
    data = estimator._measurement_data_for_iso("Cs-137", None)

    assert record_ids == [0, 0, 2, 3, 4, 5]
    assert data is not None
    assert data.station_sequence_ids is not None
    assert data.station_sequence_ids.tolist() == record_ids
    assert estimator.measurements[-1].z_variance_k == {"Cs-137": 0.0}
    assert data.observation_variances[-1] == pytest.approx(0.0)


def test_mixed_runtime_likelihood_history_matches_runtime_updates() -> None:
    """Structural replay must preserve mixed direct-spectrum and count routes."""
    import torch

    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        max_sources=1,
        init_num_sources=(1, 1),
        birth_enable=False,
        resample_threshold=0.0,
        use_tempering=False,
        count_likelihood_model="student_t",
        count_likelihood_df=6.0,
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=0.2,
        direct_spectrum_likelihood_enable=True,
        use_gpu=False,
        parallel_isotope_updates=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    template = (0.2, 0.5, 0.3)
    direct_with_variance = {
        "spectrum_counts": (8.0, 21.0, 13.0),
        "spectrum_variance": (1.5, 2.5, 3.5),
        "spectrum_background": (0.5, 0.5, 0.5),
        "spectrum_response_templates_by_isotope": {"Cs-137": template},
    }
    direct_without_variance = {
        "spectrum_counts": (11.0, 17.0, 10.0),
        "spectrum_background": (0.25, 0.25, 0.25),
        "spectrum_response_templates_by_isotope": {"Cs-137": template},
    }
    estimator.update_pair(
        {"Cs-137": 40.5},
        pose_idx=0,
        fe_index=0,
        pb_index=1,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 4.0},
        spectrum_payload=direct_with_variance,
    )
    estimator.update_pair(
        {"Cs-137": 31.0},
        pose_idx=0,
        fe_index=1,
        pb_index=0,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 3.0},
    )
    estimator.update_pair(
        {"Cs-137": 37.25},
        pose_idx=0,
        fe_index=2,
        pb_index=3,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 5.0},
        spectrum_payload=direct_without_variance,
    )
    data = estimator._measurement_data_for_iso("Cs-137", None)
    assert data is not None
    assert data.runtime_likelihood_routes is not None
    assert data.runtime_likelihood_routes.tolist() == [
        "direct_spectrum",
        "count",
        "direct_spectrum",
    ]
    assert data.spectrum_variance is not None
    assert data.spectrum_variance_present is not None
    assert data.spectrum_variance_present.tolist() == [True, False, False]
    np.testing.assert_allclose(
        data.spectrum_variance,
        np.array(
            [
                [1.5, 2.5, 3.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    filt = estimator.filters["Cs-137"]
    lambda_kp = np.array(
        [[39.0, 45.0], [29.0, 35.0], [36.0, 42.0]],
        dtype=float,
    )
    structural = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    lambda_t = torch.as_tensor(lambda_kp, dtype=torch.float64)
    runtime = torch.zeros(lambda_kp.shape[1], dtype=torch.float64)
    for row_index in (0, 2):
        rows = slice(row_index, row_index + 1)
        row_variance = (
            data.spectrum_variance[rows]
            if bool(data.spectrum_variance_present[row_index])
            else None
        )
        runtime += filt._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
            lambda_t[rows],
            data.spectrum_counts[rows],
            data.spectrum_response_template[rows],
            data.spectrum_background[rows],
            row_variance,
        )
    count_rows = slice(1, 2)
    runtime += filt._log_likelihood_sequence_gpu(
        lambda_t[count_rows],
        data.z_k[count_rows],
        data.observation_variances[count_rows],
    )

    assert np.allclose(structural, runtime.numpy(), rtol=1.0e-11, atol=1.0e-10)


def test_joint_spectrum_route_retains_present_row_variance() -> None:
    """A missing row variance must mean zero, not erase other joint variances."""
    import torch

    payloads = (
        {
            "spectrum_counts": np.array([3.0, 5.0], dtype=float),
            "spectrum_response_template": np.array([0.4, 0.6], dtype=float),
            "spectrum_background": np.array([0.1, 0.2], dtype=float),
            "spectrum_variance": np.array([1.0, 2.0], dtype=float),
        },
        {
            "spectrum_counts": np.array([4.0, 6.0], dtype=float),
            "spectrum_response_template": np.array([0.3, 0.7], dtype=float),
            "spectrum_background": np.array([0.2, 0.1], dtype=float),
        },
    )

    stacked = RotatingShieldPFEstimator._stack_pf_spectrum_sequence_payloads(
        payloads
    )

    assert stacked is not None
    assert "spectrum_variance" in stacked
    np.testing.assert_allclose(
        stacked["spectrum_variance"],
        np.array([[1.0, 2.0], [0.0, 0.0]], dtype=float),
    )
    assert (
        RotatingShieldPFEstimator._stack_pf_spectrum_sequence_payloads(
            (payloads[0], None)
        )
        is None
    )
    records = (
        MeasurementRecord(
            z_k={"Cs-137": 8.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            station_sequence_id=4,
            station_view_index=0,
            runtime_likelihood_route_by_isotope={
                "Cs-137": "direct_spectrum"
            },
        ),
        MeasurementRecord(
            z_k={"Cs-137": 10.0},
            pose_idx=0,
            orient_idx=1,
            live_time_s=1.0,
            station_sequence_id=4,
            station_view_index=1,
            runtime_likelihood_route_by_isotope={
                "Cs-137": "direct_spectrum"
            },
        ),
    )
    routes = np.asarray(["direct_spectrum"] * 2, dtype=str)
    variance_present = (
        RotatingShieldPFEstimator._runtime_spectrum_variance_usage_for_records(
            "Cs-137",
            records,
            payloads,
            routes,
        )
    )
    assert variance_present.tolist() == [True, True]

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="poisson",
        direct_spectrum_likelihood_enable=True,
        use_gpu=False,
    )
    assert stacked is not None
    data = MeasurementData(
        z_k=np.array([8.0, 10.0], dtype=float),
        observation_variances=np.zeros(2, dtype=float),
        detector_positions=np.repeat(
            np.array([[0.5, 0.0, 0.0]], dtype=float),
            2,
            axis=0,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([1, 0], dtype=int),
        live_times=np.ones(2, dtype=float),
        station_sequence_ids=np.array([4, 4], dtype=int),
        runtime_likelihood_routes=routes,
        spectrum_counts=stacked["spectrum_counts"],
        spectrum_response_template=stacked["spectrum_response_template"],
        spectrum_background=stacked["spectrum_background"],
        spectrum_variance=stacked["spectrum_variance"],
        spectrum_variance_present=variance_present,
    )
    lambda_kp = np.array([[7.0, 9.0], [11.0, 8.0]], dtype=float)
    structural = filt._structural_count_log_likelihood_matrix_np(data, lambda_kp)
    runtime = filt._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        torch.as_tensor(lambda_kp, dtype=torch.float64),
        stacked["spectrum_counts"],
        stacked["spectrum_response_template"],
        stacked["spectrum_background"],
        stacked["spectrum_variance"],
    )

    np.testing.assert_allclose(
        structural,
        runtime.numpy(),
        rtol=1.0e-11,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("likelihood_model", ["poisson", "student_t"])
def test_structural_direct_spectrum_likelihood_matches_runtime(
    likelihood_model: str,
) -> None:
    """Direct-spectrum structural evidence should equal the runtime GPU path."""
    import torch

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        count_likelihood_model=likelihood_model,
        count_likelihood_df=7.0,
        spectrum_count_rel_sigma=0.06,
        spectrum_count_abs_sigma=0.4,
        direct_spectrum_likelihood_enable=True,
        spectrum_likelihood_bin_chunk=2,
        shield_contrast_likelihood_enable=True,
        shield_contrast_likelihood_weight=0.25,
        shield_contrast_min_count=1.0,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=0.15,
        shield_view_ratio_likelihood_min_total_count=1.0,
        use_gpu=False,
    )
    lambda_kp = np.array(
        [
            [42.0, 55.0, 31.0],
            [28.0, 19.0, 37.0],
            [61.0, 48.0, 70.0],
            [35.0, 44.0, 26.0],
        ],
        dtype=float,
    )
    template = np.array(
        [
            [0.15, 0.25, 0.35, 0.25],
            [0.20, 0.30, 0.25, 0.25],
            [0.10, 0.20, 0.40, 0.30],
            [0.25, 0.20, 0.30, 0.25],
        ],
        dtype=float,
    )
    background = np.array(
        [
            [0.5, 0.2, 0.4, 0.3],
            [0.4, 0.3, 0.2, 0.5],
            [0.6, 0.2, 0.5, 0.4],
            [0.3, 0.4, 0.2, 0.6],
        ],
        dtype=float,
    )
    observed = np.array(
        [
            [7.0, 11.0, 16.0, 10.0],
            [6.0, 8.0, 7.0, 8.0],
            [7.0, 13.0, 25.0, 18.0],
            [9.0, 7.0, 11.0, 9.0],
        ],
        dtype=float,
    )
    spectrum_variance = (
        None
        if likelihood_model == "poisson"
        else np.full(observed.shape, 0.75, dtype=float)
    )
    data = MeasurementData(
        z_k=np.sum(observed - background, axis=1),
        observation_variances=np.array([3.0, 2.0, 4.0, 2.5], dtype=float),
        detector_positions=np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.0, 0.5],
            ],
            dtype=float,
        ),
        fe_indices=np.array([0, 1, 0, 1], dtype=int),
        pb_indices=np.array([1, 0, 1, 0], dtype=int),
        live_times=np.ones(4, dtype=float),
        station_sequence_ids=np.array([0, 0, 1, 1], dtype=int),
        spectrum_counts=observed,
        spectrum_response_template=template,
        spectrum_background=background,
        spectrum_variance=spectrum_variance,
    )

    structural = filt._structural_count_log_likelihood_matrix_np(
        data,
        lambda_kp,
    )
    lambda_t = torch.as_tensor(lambda_kp, dtype=torch.float64)
    runtime = filt._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        lambda_t,
        observed,
        template,
        background,
        spectrum_variance,
    )
    for rows in (np.arange(2), np.arange(2, 4)):
        runtime += filt._shield_shape_sequence_log_likelihood_gpu(
            lambda_t[rows],
            data.z_k[rows],
            data.observation_variances[rows],
        )

    assert np.allclose(structural, runtime.numpy(), rtol=1.0e-11, atol=1.0e-10)


def test_batched_structural_removal_matches_scalar_with_covariance() -> None:
    """Batched leave-one-out evidence should match the scalar structural oracle."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        count_likelihood_model="student_t",
        count_likelihood_df=5.0,
        transport_model_rel_sigma=0.1,
        spectrum_count_rel_sigma=0.15,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=0.5,
        use_gpu=False,
    )
    covariance = np.array(
        [
            [0.0, 0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.15],
            [0.0, 0.0, 0.15, 0.0],
        ],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([40.0, 24.0, 52.0, 31.0], dtype=float),
        observation_variances=np.array([3.0, 2.0, 4.0, 2.5], dtype=float),
        detector_positions=np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.0, 0.5],
            ],
            dtype=float,
        ),
        fe_indices=np.array([0, 1, 0, 1], dtype=int),
        pb_indices=np.array([1, 0, 1, 0], dtype=int),
        live_times=np.ones(4, dtype=float),
        station_sequence_ids=np.array([0, 0, 1, 1], dtype=int),
        observation_count_covariance=covariance,
    )
    lambda_total = np.array(
        [
            [38.0, 43.0, 35.0],
            [26.0, 21.0, 29.0],
            [49.0, 57.0, 45.0],
            [33.0, 28.0, 37.0],
        ],
        dtype=float,
    )
    component_fractions = np.array(
        [[0.18, 0.11], [0.22, 0.08], [0.15, 0.13], [0.20, 0.09]],
        dtype=float,
    )
    lambda_components = lambda_total[:, :, None] * component_fractions[:, None, :]

    batched = filt._delta_log_likelihood_remove_group(
        data,
        lambda_total,
        lambda_components,
    )
    scalar = np.vstack(
        [
            filt._structural_delta_log_likelihood_remove(
                data,
                lambda_total[:, particle_idx],
                lambda_components[:, particle_idx, :],
            )
            for particle_idx in range(lambda_total.shape[1])
        ]
    )
    batched_prune = filt._source_prune_allowed_mask_group(
        data,
        lambda_components,
        lambda_total,
        delta_ll=batched,
    )
    scalar_prune = np.vstack(
        [
            filt._source_prune_allowed_mask(
                IsotopeState(
                    num_sources=2,
                    positions=np.zeros((2, 3), dtype=float),
                    strengths=np.ones(2, dtype=float),
                    background=0.0,
                ),
                data,
                lambda_m=lambda_components[:, particle_idx, :],
                lambda_total=lambda_total[:, particle_idx],
                delta_ll=scalar[particle_idx],
            )
            for particle_idx in range(lambda_total.shape[1])
        ]
    )

    assert np.allclose(batched, scalar, rtol=1.0e-12, atol=1.0e-12)
    assert np.array_equal(batched_prune, scalar_prune)


def test_birth_residual_support_uses_effective_pf_covariance() -> None:
    """A large PF model variance should prevent unsupported residual births."""
    data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=np.array(
            [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    residual = np.array([10.0, 10.0], dtype=float)
    narrow = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        count_likelihood_model="student_t",
        transport_model_rel_sigma=0.0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
        birth_min_distinct_stations=2,
        birth_residual_support_sigma=1.0,
        birth_residual_gate_p_value=1.0,
        use_gpu=False,
    )
    broad = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        count_likelihood_model="student_t",
        transport_model_rel_sigma=1.0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
        birth_min_distinct_stations=2,
        birth_residual_support_sigma=1.0,
        birth_residual_gate_p_value=1.0,
        use_gpu=False,
    )

    assert narrow._birth_residual_gate_allows(residual, data)
    assert narrow.last_birth_residual_support == 2
    assert not broad._birth_residual_gate_allows(residual, data)
    assert broad.last_birth_residual_support == 0


def test_birth_residual_support_uses_offdiagonal_covariance_evidence() -> None:
    """Strong covariance contrast evidence must survive marginal-variance gating."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        count_likelihood_model="gaussian",
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=1.0,
        birth_residual_gate_p_value=0.05,
        birth_candidate_support_fraction=0.0,
        use_gpu=False,
    )
    common_kwargs = {
        "z_k": np.array([5.0, 0.0], dtype=float),
        "observation_variances": np.array([100.0, 100.0], dtype=float),
        "detector_positions": np.repeat(
            np.array([[0.5, 0.0, 0.0]], dtype=float),
            2,
            axis=0,
        ),
        "fe_indices": np.array([0, 1], dtype=int),
        "pb_indices": np.array([1, 0], dtype=int),
        "live_times": np.ones(2, dtype=float),
        "station_sequence_ids": np.array([0, 0], dtype=int),
        "runtime_likelihood_routes": np.array(
            ["count_covariance", "count_covariance"],
            dtype=str,
        ),
    }
    diagonal = MeasurementData(
        **common_kwargs,
        observation_count_covariance=np.zeros((2, 2), dtype=float),
    )
    correlated = MeasurementData(
        **common_kwargs,
        observation_count_covariance=np.array(
            [[0.0, 99.0], [99.0, 0.0]],
            dtype=float,
        ),
    )
    residual = np.array([5.0, 0.0], dtype=float)
    candidate_counts = np.array([[1.0], [0.25]], dtype=float)

    diagonal_support, *_ = filt._birth_residual_support_evidence(
        residual,
        diagonal,
    )
    correlated_support, *_ = filt._birth_residual_support_evidence(
        residual,
        correlated,
    )

    assert diagonal_support.tolist() == [False, False]
    assert correlated_support.tolist() == [True, False]
    assert not filt._birth_residual_gate_allows(residual, diagonal)
    assert filt._birth_residual_gate_allows(residual, correlated)
    assert not filt._birth_candidate_support_mask(
        data=diagonal,
        candidate_counts=candidate_counts,
        residual_mix=residual,
    )[0]
    assert filt._birth_candidate_support_mask(
        data=correlated,
        candidate_counts=candidate_counts,
        residual_mix=residual,
    )[0]


def test_removed_fit_rescue_and_forced_gate_apis_are_physically_absent() -> None:
    """Legacy fit, rescue, and forced-gate APIs should not remain as dead knobs."""
    field_names = set(PFConfig.__dataclass_fields__)
    exact_removed_fields = {
        "birth_window",
        "birth_residual_always_try",
        "split_residual_always_try",
        "death_low_q_streak",
        "death_strength_threshold",
        "death_require_low_strength",
        "death_delta_ll_threshold",
        "support_window",
        "report_exclude_unverified_sources",
        "mode_preserving_report_cardinality_strata",
        "mode_preserving_report_cardinality_extra_particles",
        "source_strength_prior_mean",
    }
    forbidden_field_fragments = (
        "refit",
        "global_rescue",
        "runtime_report_rescue",
        "force_proposal",
        "forced_min_delta",
        "suppress_death",
        "weak_source_prune",
        "structural_update_min",
        "strength_absorption",
        "observation_overshoot",
    )
    assert field_names.isdisjoint(exact_removed_fields)
    assert not any(
        fragment in field_name
        for field_name in field_names
        for fragment in forbidden_field_fragments
    )

    removed_methods = {
        "apply_birth_death",
        "apply_report_model_order_cluster_prune",
        "inject_runtime_report_rescue_particles",
        "refit_strengths_for_particles",
        "sync_particles_to_evidence_sources",
        "_birth_residual_survives_strength_refit",
        "_data_driven_visibility_strengths_from_unit_response",
        "_prune_floor_sources_by_expected_counts",
        "_solve_trial_strengths_from_unit_counts",
        "_structural_evidence_data",
    }
    assert not any(
        hasattr(IsotopeParticleFilter, method_name) for method_name in removed_methods
    )
    assert hasattr(IsotopeParticleFilter, "apply_structural_moves")


def test_birth_adds_source_when_particle_empty() -> None:
    """Birth move should inject a new source when a particle has none."""
    np.random.seed(0)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_residual_min_support=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.1,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert all(p.state.num_sources > 0 for p in filt.continuous_particles)


def test_birth_scoring_prefers_shield_coded_residual_shape() -> None:
    """Birth proposal scoring should prefer residual shape over raw count scale."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
    )
    candidate_counts = np.array(
        [
            [10.0, 30.0],
            [10.0, 0.0],
            [0.0, 300.0],
        ],
        dtype=float,
    )
    residual = np.array([10.0, 10.0, 0.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(3, dtype=float),
    )

    assert scores[0] > scores[1]
    assert q_hat[0] > q_hat[1]


def test_birth_scoring_uses_count_distance_prior_for_single_view() -> None:
    """Single-view residual birth should prefer high unit-response candidates."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
        birth_count_distance_prior_weight=1.0,
        birth_count_distance_strength_weight=1.0,
    )
    candidate_counts = np.array([[10.0, 1.0]], dtype=float)
    residual = np.array([100.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(1, dtype=float),
    )

    assert q_hat[0] < q_hat[1]
    assert scores[0] > scores[1]


def test_birth_scoring_can_disable_count_distance_prior() -> None:
    """Disabling the proposal prior should preserve pure least-squares ties."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
        birth_count_distance_prior_weight=0.0,
        birth_count_distance_strength_weight=0.0,
    )
    candidate_counts = np.array([[10.0, 1.0]], dtype=float)
    residual = np.array([100.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(1, dtype=float),
    )

    assert q_hat[0] < q_hat[1]
    assert np.isclose(scores[0], scores[1])


def test_peak_suppressed_residual_birth_reveals_noncollinear_weak_source() -> None:
    """Peak suppression should propose a weak candidate without rebirthing a strong source."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=2,
        peak_suppression_min_source_fraction=0.1,
        birth_num_local_jitter=0,
    )
    strong_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    weak_pos = np.array([[4.0, 4.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    strong_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=strong_pos,
        strengths=np.array([500.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    weak_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=weak_pos,
        strengths=np.array([80.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    filt.continuous_particles[0].state = IsotopeState(
        num_sources=1,
        positions=strong_pos.copy(),
        strengths=np.array([500.0], dtype=float),
        background=0.0,
    )
    data = MeasurementData(
        z_k=strong_counts + weak_counts,
        observation_variances=np.maximum(strong_counts + weak_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        live_times=np.ones(4, dtype=float),
    )

    proposal = filt._compute_birth_proposal(
        data,
        np.vstack([strong_pos, weak_pos]),
    )

    assert proposal is not None
    _, _, _, candidates, candidate_counts = proposal
    assert filt.last_birth_residual_layer.startswith("strong_suppressed")
    assert np.any(
        np.linalg.norm(candidates - weak_pos[0][None, :], axis=1) < 1.0e-9
    )
    accepted = filt._apply_matching_pursuit_births_to_state(
        filt.continuous_particles[0].state,
        data,
        candidates,
        max_new_sources=1,
        candidate_unit_counts=candidate_counts,
    )
    final_state = filt.continuous_particles[0].state
    assert accepted == 1
    assert final_state.num_sources == 2
    assert (
        np.min(np.linalg.norm(final_state.positions - weak_pos, axis=1))
        < 1.0e-9
    )


def test_birth_residual_layers_include_leave_one_cluster_out() -> None:
    """Peak suppression should include cluster-level leave-one-out residuals."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=2,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=4,
        peak_suppression_min_source_fraction=0.1,
        cluster_eps_m=0.8,
    )
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    strong_positions = [
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.array([[0.2, 0.1, 0.0]], dtype=float),
    ]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=pos.copy(),
                strengths=np.array([500.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for pos in strong_positions
    ]
    weak_pos = np.array([[4.0, 4.0, 0.0]], dtype=float)
    strong_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=strong_positions[0],
        strengths=np.array([500.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    weak_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=weak_pos,
        strengths=np.array([80.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    data = MeasurementData(
        z_k=strong_counts + weak_counts,
        observation_variances=np.maximum(strong_counts + weak_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        live_times=np.ones(4, dtype=float),
    )

    layers = filt._compute_birth_residual_layers(
        data=data,
        particle_indices=np.array([0, 1], dtype=int),
        particle_weights=np.array([0.5, 0.5], dtype=float),
    )

    assert any(layer.name.startswith("leave_one_cluster_out") for layer in layers)


def test_matching_pursuit_birth_can_add_multiple_sources() -> None:
    """Residual matching pursuit should add more than one supported source."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_matching_pursuit_max_new_sources=2,
        birth_matching_pursuit_topk_candidates=3,
        birth_min_sep_m=0.4,
        birth_bic_penalty_params=0,
        birth_q_min=1.0,
        birth_q_max=130.0,
    )
    state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    true_positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([120.0, 90.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        true_positions,
        max_new_sources=2,
    )

    assert accepted == 2
    assert state.num_sources == 2
    accepted_events = [
        event
        for event in filt.last_source_event_diagnostics
        if event["event"] == "source_birth_accepted"
    ]
    assert len(accepted_events) == 2
    assert all(event["reason"] == "matching_pursuit_birth" for event in accepted_events)


def test_pseudo_source_verification_prunes_unsupported_tentative_source() -> None:
    """Pseudo-source verification should quarantine before hard pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert changed
    assert state.num_sources == 2
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0
    assert any(
        event["event"] == "pseudo_source_quarantined"
        for event in filt.last_source_event_diagnostics
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert changed
    assert state.num_sources == 1
    assert filt.last_pseudo_source_pruned == 1
    assert any(
        event["event"] == "source_removed" and event["reason"] == "pseudo_source_pruned"
        for event in filt.last_source_event_diagnostics
    )


def test_pseudo_source_verification_requires_multiple_stations_to_prune() -> None:
    """A tentative source should quarantine when hard prune is not allowed."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.arange(3, dtype=int),
        pb_indices=np.arange(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.arange(3, dtype=int),
        pb_indices=np.arange(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt._verify_pseudo_sources_for_state(state, data, suppress_prune=False)

    assert state.num_sources == 2
    assert filt.last_pseudo_source_failed == 1
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0
    assert filt._quarantined_source_mask(state).tolist() == [False, True]


def test_pseudo_source_quarantine_does_not_require_prune_allowed() -> None:
    """Suppressed pseudo-source failures should quarantine before hard pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.arange(2, dtype=int),
        pb_indices=np.arange(2, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.array([10.0, 7.5], dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.arange(2, dtype=int),
        pb_indices=np.arange(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=True,
    )

    assert changed
    assert state.num_sources == 2
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0
    assert filt._quarantined_source_mask(state).tolist() == [False, True]
    assert np.allclose(state.support_scores, np.array([10.0, 7.5], dtype=float))


def test_pseudo_source_correlation_failure_requests_more_views() -> None:
    """Collinear tentative responses should not trigger quarantine or pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=3,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=1.0e9,
        pseudo_source_corr_max=0.995,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=position,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([position, position]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([5, 5], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert not changed
    assert state.num_sources == 2
    assert state.verification_fail_streaks.tolist() == [0, 0]
    assert filt.last_pseudo_source_failed == 1
    assert filt.last_pseudo_source_quarantined == 0
    assert filt.last_pseudo_source_pruned == 0
    assert filt.last_pseudo_source_fail_reasons["needs_discriminative_views"] == 1


def test_pseudo_source_verification_uses_cached_prune_allowed() -> None:
    """Pseudo-source verification should reuse cached prune decisions."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=1.0e9,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    lambda_m, lambda_total = filt._lambda_components(state, data)
    delta_ll = filt._structural_delta_log_likelihood_remove(
        data,
        lambda_total,
        lambda_m,
    )

    def fail_uncached_prune(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if pseudo verification recomputes prune permission."""
        raise AssertionError("uncached prune permission was recomputed")

    filt._source_prune_allowed_mask = fail_uncached_prune

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
        cached_lambda_m=lambda_m,
        cached_lambda_total=lambda_total,
        cached_delta_ll=delta_ll,
        cached_prune_allowed=np.array([False, True], dtype=bool),
    )

    assert changed
    assert filt.last_pseudo_source_quarantined == 1


def test_group_unit_response_matches_legacy_and_deduplicates_geometry(
    monkeypatch,
) -> None:
    """Unit-response reuse should preserve counts and evaluate unique positions once."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        use_gpu=False,
    )
    positions = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
    ]
    strengths = [
        np.array([20.0, 30.0], dtype=float),
        np.array([40.0, 50.0], dtype=float),
        np.array([60.0, 70.0], dtype=float),
    ]
    backgrounds = np.array([0.1, 0.2, 0.3], dtype=float)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=positions[idx].copy(),
                strengths=strengths[idx].copy(),
                background=float(backgrounds[idx]),
            ),
            log_weight=float(np.log(1.0 / 3.0)),
        )
        for idx in range(3)
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.array([11.0, 13.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.array([1.0, 2.0], dtype=float),
    )
    flat_positions = np.vstack(positions)
    flat_strengths = np.concatenate(strengths)
    legacy_components = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=data.detector_positions,
        sources=flat_positions,
        strengths=flat_strengths,
        live_times=data.live_times,
        fe_indices=data.fe_indices,
        pb_indices=data.pb_indices,
        source_scale=filt._measurement_source_scale_vector(
            data.fe_indices,
            data.pb_indices,
        ),
    ).reshape(2, 3, 2)
    legacy_total = data.live_times[:, None] * backgrounds[None, :] + np.sum(
        legacy_components, axis=2
    )

    source_counts: list[int] = []
    original = filt.continuous_kernel.kernel_values_selected_pairs_for_detectors

    def count_sources(*args: object, **kwargs: object) -> NDArray[np.float64]:
        """Record the number of geometry columns evaluated by the kernel."""
        sources_arg = np.asarray(kwargs["sources"], dtype=float)
        source_counts.append(int(sources_arg.shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        filt.continuous_kernel,
        "kernel_values_selected_pairs_for_detectors",
        count_sources,
    )

    actual_components, actual_total = filt._lambda_components_for_particle_group(
        data,
        particle_indices=[0, 1, 2],
        source_count=2,
    )

    assert np.array_equal(actual_components, legacy_components)
    assert np.array_equal(actual_total, legacy_total)
    assert source_counts == [3]


def test_matching_pursuit_birth_uses_cached_candidate_counts(monkeypatch) -> None:
    """Matching-pursuit birth should not recompute cached candidate responses."""
    import pf.particle_filter as particle_filter_module

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_matching_pursuit_max_new_sources=2,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.array([10.0, 12.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    candidates = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    candidate_counts = np.zeros((2, 2), dtype=float)
    original_expected_counts = particle_filter_module.expected_counts_per_source

    def guarded_expected_counts(*args: object, **kwargs: object) -> np.ndarray:
        """Reject the candidate-response recomputation that the cache avoids."""
        sources = np.asarray(kwargs.get("sources"), dtype=float)
        strengths = np.asarray(kwargs.get("strengths"), dtype=float)
        if sources.shape == candidates.shape and np.allclose(sources, candidates):
            if strengths.shape == (2,) and np.allclose(strengths, 1.0):
                raise AssertionError("candidate responses were recomputed")
        return original_expected_counts(*args, **kwargs)

    monkeypatch.setattr(
        particle_filter_module,
        "expected_counts_per_source",
        guarded_expected_counts,
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        candidates,
        max_new_sources=2,
        candidate_unit_counts=candidate_counts,
    )

    assert accepted == 0


def test_birth_existing_unit_response_counts_batched_match_scalar_oracle() -> None:
    """Existing-source birth response columns should be batched without drift."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=3,
        pseudo_source_fail_grace_stations=1,
        source_prune_fail_grace_stations=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
                    dtype=float,
                ),
                strengths=np.array([10.0, 5.0], dtype=float),
                background=0.0,
                ages=np.array([3, 1], dtype=int),
                support_scores=np.zeros(2, dtype=float),
                tentative_sources=np.array([False, True], dtype=bool),
                verification_fail_streaks=np.array([0, 1], dtype=int),
            ),
            log_weight=np.log(0.4),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 0.1, 0.0]], dtype=float),
                strengths=np.array([6.0], dtype=float),
                background=0.0,
                ages=np.array([2], dtype=int),
                support_scores=np.zeros(1, dtype=float),
            ),
            log_weight=np.log(0.35),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.25),
        ),
    ]
    filt.N = len(filt.continuous_particles)
    data = MeasurementData(
        z_k=np.array([8.0, 5.0, 3.0], dtype=float),
        observation_variances=np.array([8.0, 5.0, 3.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    indices = np.array([0, 1, 2], dtype=int)

    scalar = filt._birth_existing_unit_response_counts_scalar(
        data,
        particle_indices=indices,
    )
    batched = filt._birth_existing_unit_response_counts(
        data,
        particle_indices=indices,
    )

    assert np.allclose(batched, scalar)
    assert batched.shape == (3, 3)


def test_orthogonalized_residual_candidates_skip_existing_response_copy() -> None:
    """Orthogonalized birth ranking should avoid duplicate response columns."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_orthogonalize_residual_candidates=True,
        birth_orthogonal_candidate_corr_max=0.9,
    )
    existing = np.asarray([[1.0], [2.0], [3.0]], dtype=float)
    candidates = np.asarray(
        [
            [1.0, 0.1, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.1, 1.0],
        ],
        dtype=float,
    )

    selected = filt._orthogonalized_residual_candidate_indices(  # noqa: SLF001
        np.asarray([0, 1, 2], dtype=np.int64),
        candidate_counts=candidates,
        existing_response_counts=existing,
        observation_variances=np.ones(3, dtype=float),
        max_corr=0.9,
    )

    assert 0 not in selected.tolist()
    assert selected.tolist()[0] == 1


def test_merge_trial_batched_matches_scalar_oracle() -> None:
    """Batched merge candidate trials should match the scalar reference path."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=1,
        merge_distance_max=5.0,
        merge_response_corr_min=0.0,
        merge_search_topk_pairs=6,
    )
    state = IsotopeState(
        num_sources=3,
        positions=np.array(
            [[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [2.5, 0.2, 0.0]],
            dtype=float,
        ),
        strengths=np.array([80.0, 70.0, 35.0], dtype=float),
        background=0.0,
        ages=np.array([4, 3, 5], dtype=int),
        support_scores=np.array([3.0, 2.0, 1.0], dtype=float),
        tentative_sources=np.array([False, True, False], dtype=bool),
        verification_fail_streaks=np.array([0, 1, 0], dtype=int),
    )
    true_positions = np.array(
        [[0.15, 0.0, 0.0], [2.5, 0.2, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([150.0, 35.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.5, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    scalar_trial, scalar_delta = filt._best_merge_trial_scalar(state.copy(), data)
    batched_trial, batched_delta = filt._best_merge_trial(state.copy(), data)

    assert scalar_trial is not None
    assert batched_trial is not None
    assert np.isclose(batched_delta, scalar_delta)
    assert batched_trial.num_sources == scalar_trial.num_sources
    assert np.allclose(batched_trial.positions, scalar_trial.positions)
    assert np.allclose(batched_trial.strengths, scalar_trial.strengths)
    assert np.array_equal(
        batched_trial.tentative_sources, scalar_trial.tentative_sources
    )


def test_merge_trial_projects_weighted_position_to_surface_prior() -> None:
    """Merging sources on different walls must not create an interior source."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        source_position_prior="surface",
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array(
            [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0]],
            dtype=float,
        ),
        strengths=np.array([1.0, 1.0], dtype=float),
        background=0.0,
    )
    interior_average = np.mean(state.positions, axis=0)

    trial = filt._make_merge_trial_state(state, 0, 1)
    merged = trial.positions[-1]

    assert trial.num_sources == 1
    assert not np.allclose(merged, interior_average)
    assert np.any(
        np.isclose(merged, np.asarray(filt.config.position_min, dtype=float))
        | np.isclose(merged, np.asarray(filt.config.position_max, dtype=float))
    )


def test_merge_trial_evaluation_can_be_chunked_without_drift() -> None:
    """Independent merge trials should match when evaluated in worker chunks."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=1,
        merge_distance_max=5.0,
        merge_response_corr_min=0.0,
        merge_search_topk_pairs=6,
    )
    states = [
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.25, 0.05, 0.0], [2.0, 0.2, 0.0]],
                dtype=float,
            ),
            strengths=np.array([90.0, 80.0, 30.0], dtype=float),
            background=0.0,
            ages=np.array([4, 3, 5], dtype=int),
            support_scores=np.ones(3, dtype=float),
            tentative_sources=np.array([False, True, False], dtype=bool),
            verification_fail_streaks=np.zeros(3, dtype=int),
        ),
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.1, 0.0], [1.2, 0.1, 0.0], [1.35, 0.15, 0.0]],
                dtype=float,
            ),
            strengths=np.array([40.0, 60.0, 55.0], dtype=float),
            background=0.0,
            ages=np.array([5, 4, 3], dtype=int),
            support_scores=np.ones(3, dtype=float),
            tentative_sources=np.zeros(3, dtype=bool),
            verification_fail_streaks=np.zeros(3, dtype=int),
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[2.5, 0.0, 0.0], [2.75, 0.05, 0.0]], dtype=float),
            strengths=np.array([35.0, 33.0], dtype=float),
            background=0.0,
            ages=np.array([3, 3], dtype=int),
            support_scores=np.ones(2, dtype=float),
            tentative_sources=np.zeros(2, dtype=bool),
            verification_fail_streaks=np.zeros(2, dtype=int),
        ),
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.5, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=float,
    )
    true_positions = np.array(
        [[0.1, 0.02, 0.0], [1.3, 0.12, 0.0], [2.65, 0.02, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([170.0, 115.0, 68.0], dtype=float)
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    def signature(
        result: tuple[IsotopeState | None, float],
    ) -> tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]:
        """Return a compact deterministic signature for a merge trial result."""
        trial, delta = result
        if trial is None:
            return ((0, (), ()), float(delta))
        return (
            (
                int(trial.num_sources),
                tuple(np.round(trial.positions.reshape(-1), 12)),
                tuple(np.round(trial.strengths.reshape(-1), 12)),
            ),
            float(np.round(delta, 12)),
        )

    serial = [signature(filt._best_merge_trial(state.copy(), data)) for state in states]
    chunked: list[tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]] = []
    for chunk in ([0, 1], [2]):
        chunked.extend(
            signature(filt._best_merge_trial(states[idx].copy(), data)) for idx in chunk
        )
    filt.config.structural_trial_workers = 2
    filt.config.structural_trial_parallel_min_trials = 1
    threaded = [
        signature(filt._best_merge_trial(state.copy(), data)) for state in states
    ]

    assert chunked == serial
    assert threaded == serial


def test_cached_split_trial_evaluation_can_be_chunked_without_drift() -> None:
    """Independent residual split trials should match across worker chunks."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        split_residual_guided=True,
        split_residual_candidate_count=4,
        birth_min_sep_m=0.4,
        min_age_to_split=0,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([90.0], dtype=float),
            background=0.0,
            ages=np.array([3], dtype=int),
            support_scores=np.zeros(1, dtype=float),
            tentative_sources=np.array([False], dtype=bool),
            verification_fail_streaks=np.zeros(1, dtype=int),
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[1.0, 0.1, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
            ages=np.array([4], dtype=int),
            support_scores=np.zeros(1, dtype=float),
            tentative_sources=np.array([False], dtype=bool),
            verification_fail_streaks=np.zeros(1, dtype=int),
        ),
    ]
    candidates = np.array(
        [[1.8, 0.0, 0.0], [2.2, 0.2, 0.0], [3.5, 0.0, 0.0]],
        dtype=float,
    )
    true_positions = np.array([[0.0, 0.0, 0.0], [2.2, 0.2, 0.0]], dtype=float)
    true_strengths = np.array([80.0, 110.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.5, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    candidate_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=candidates,
        strengths=np.ones(candidates.shape[0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )

    def evaluate(state: IsotopeState) -> tuple[IsotopeState | None, float]:
        """Return the cached split trial result for one particle state."""
        return filt._best_residual_guided_split_trial(
            state.copy(),
            data,
            candidates,
            None,
            candidate_unit_counts=candidate_counts,
        )

    def signature(
        result: tuple[IsotopeState | None, float],
    ) -> tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]:
        """Return a compact deterministic signature for a split trial result."""
        trial, delta = result
        if trial is None:
            return ((0, (), ()), float(delta))
        return (
            (
                int(trial.num_sources),
                tuple(np.round(trial.positions.reshape(-1), 12)),
                tuple(np.round(trial.strengths.reshape(-1), 12)),
            ),
            float(np.round(delta, 12)),
        )

    serial = [signature(evaluate(state)) for state in states]
    chunked: list[tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]] = []
    for chunk in ([0], [1]):
        chunked.extend(signature(evaluate(states[idx])) for idx in chunk)
    filt.config.structural_trial_workers = 2
    filt.config.structural_trial_parallel_min_trials = 1
    threaded = [signature(evaluate(state)) for state in states]

    assert chunked == serial
    assert threaded == serial


def test_birth_residual_layers_batched_match_scalar_oracle() -> None:
    """Batched birth residual layers should match scalar per-particle layers."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=4,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=4,
        peak_suppression_min_source_fraction=0.0,
        peak_suppression_factor=1.0,
        birth_use_weighted_topk=True,
        birth_residual_clip_quantile=0.95,
        cluster_eps_m=1.0,
        birth_min_sep_m=0.5,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([20.0], dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]],
                dtype=float,
            ),
            strengths=np.array([15.0, 8.0], dtype=float),
            background=0.1,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array(
                [[1.5, 0.0, 0.0], [2.2, 0.4, 0.0]],
                dtype=float,
            ),
            strengths=np.array([12.0, 5.0], dtype=float),
            background=0.3,
        ),
        IsotopeState(
            num_sources=0,
            positions=np.zeros((0, 3), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=0.4,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=float(np.log(0.25))) for state in states
    ]
    filt.N = len(filt.continuous_particles)
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    true_sources = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_sources,
        strengths=np.array([18.0, 10.0, 7.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=true_counts + 1.0,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
    )
    particle_indices = np.arange(len(filt.continuous_particles), dtype=int)
    particle_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)

    batched_layers = filt._compute_birth_residual_layers(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
    )
    scalar_layers = filt._compute_birth_residual_layers_scalar(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
    )

    assert [layer.name for layer in batched_layers] == [
        layer.name for layer in scalar_layers
    ]
    for batched, scalar in zip(batched_layers, scalar_layers):
        assert np.allclose(batched.residual, scalar.residual)


def test_soft_quarantine_is_diagnostic_only() -> None:
    """Verification quarantine metadata must not suppress a physical PF source."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        cluster_min_samples=1,
        use_clustered_output=True,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 2], dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    positions, strengths = filt.estimate_clustered()

    assert positions.shape == (2, 3)
    assert strengths.shape == (2,)
    assert filt._active_source_mask(state).tolist() == [True, True]


def test_birth_response_counts_include_quarantined_sources() -> None:
    """Residual birth response columns should include soft-quarantined sources."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 2], dtype=int),
    )
    data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    counts = filt._birth_existing_unit_response_counts_for_state(state, data)

    assert counts.shape == (1, 2)


def test_planning_particles_include_quarantined_sources() -> None:
    """Planning subsets should retain soft-quarantined sources for separation."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        planning_particles=1,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0, 50.0], dtype=float),
                background=0.0,
                ages=np.array([3, 3], dtype=int),
                support_scores=np.zeros(2, dtype=float),
                tentative_sources=np.array([False, True], dtype=bool),
                verification_fail_streaks=np.array([0, 2], dtype=int),
            ),
            log_weight=0.0,
        )
    ]

    subsets = estimator.planning_particles(max_particles=1)
    states, _weights = subsets["Cs-137"]

    assert len(states) == 1
    assert int(states[0].num_sources) == 2


def test_convergence_requires_min_stations() -> None:
    """Convergence freeze should require the configured number of stations."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        converge_enable=True,
        converge_min_stations=2,
    )
    filt._cardinality_variance = lambda: 0.0  # type: ignore[method-assign]
    filt._has_unverified_sources = lambda: False  # type: ignore[method-assign]
    filt._cluster_convergence_supported = lambda: True  # type: ignore[method-assign]

    filt._observed_station_labels = {(0.0, 0.0)}
    assert filt._convergence_can_freeze() is False

    filt._observed_station_labels.add((1.0, 0.0))
    assert filt._convergence_can_freeze() is True


def test_connected_position_clusters_preserve_transitive_components() -> None:
    """Vectorized clustering should keep the same transitive eps-neighborhoods."""
    from scipy.spatial import cKDTree

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
    )

    assert [set(cluster.tolist()) for cluster in clusters] == [{0, 1, 2}]


def test_connected_position_clusters_handles_dense_component_without_pair_matrix() -> (
    None
):
    """Dense report clusters should not require materializing all neighbor pairs."""
    from scipy.spatial import cKDTree

    positions = np.zeros((2000, 3), dtype=float)
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
    )

    assert len(clusters) == 1
    assert clusters[0].size == positions.shape[0]


def test_connected_position_clusters_uses_large_point_fallback() -> None:
    """Large report clusters should use a bounded grid fallback."""
    from scipy.spatial import cKDTree

    positions = np.zeros((6000, 3), dtype=float)
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
        exact_max_points=5000,
    )

    assert len(clusters) == 1
    assert clusters[0].size == positions.shape[0]


def test_cardinality_preserving_resample_keeps_source_count_mass() -> None:
    """Resampling should not erase a low-mass source-count hypothesis."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=100,
        cardinality_preserving_resample=True,
    )
    filt.config.resample_threshold = 2.0
    particles: list[IsotopeParticle] = []
    log_weights: list[float] = []
    for idx in range(95):
        state = IsotopeState(
            num_sources=1,
            positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        )
        particles.append(
            IsotopeParticle(state=state, log_weight=float(np.log(0.95 / 95.0)))
        )
        log_weights.append(float(np.log(0.95 / 95.0)))
    for idx in range(5):
        state = IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=float,
            ),
            strengths=np.array([100.0, 80.0, 60.0], dtype=float),
            background=0.0,
        )
        particles.append(
            IsotopeParticle(state=state, log_weight=float(np.log(0.05 / 5.0)))
        )
        log_weights.append(float(np.log(0.05 / 5.0)))
    filt.continuous_particles = particles

    np.random.seed(123)
    filt._maybe_resample_continuous(disable_regularize=True)

    labels = np.array([p.state.num_sources for p in filt.continuous_particles])
    weights = filt.continuous_weights
    mass_k1 = float(np.sum(weights[labels == 1]))
    mass_k3 = float(np.sum(weights[labels == 3]))
    assert np.count_nonzero(labels == 3) > 0
    assert np.isclose(mass_k1, 0.95)
    assert np.isclose(mass_k3, 0.05)


def test_cardinality_preserving_resample_waits_for_min_stations() -> None:
    """Cardinality-preserving resampling should stay off during early exploration."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=20,
        cardinality_preserving_resample=True,
        cardinality_preserving_min_stations=2,
        cardinality_preserving_require_confirmed_structure=False,
    )
    particles: list[IsotopeParticle] = []
    for idx in range(18):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=1,
                    positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                    strengths=np.array([100.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    for _idx in range(2):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=3,
                    positions=np.array(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        dtype=float,
                    ),
                    strengths=np.array([100.0, 80.0, 60.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    filt.continuous_particles = particles
    weights = np.array([0.9 / 18.0] * 18 + [0.05, 0.05], dtype=float)

    assert filt._cardinality_preserving_resample_draw(weights) is None

    filt._observed_station_labels = {(0.0, 0.0), (1.0, 0.0)}
    assert filt._cardinality_preserving_resample_draw(weights) is not None


def test_cardinality_preserving_resample_keeps_protected_spatial_modes() -> None:
    """Cardinality-preserving draws should still retain protected spatial modes."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=20,
        cardinality_preserving_resample=True,
        mode_preserving_resample=True,
        mode_preserving_max_modes=3,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=0.5,
        mode_preserving_min_weight_fraction=0.0,
        mode_preserving_cardinality_strata=False,
    )
    particles: list[IsotopeParticle] = []
    for idx in range(18):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=1,
                    positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                    strengths=np.array([100.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    for idx, x_pos in enumerate((100.0, 200.0)):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=3,
                    positions=np.array(
                        [
                            [x_pos, 0.0, 0.0],
                            [x_pos, 1.0, 0.0],
                            [x_pos, 2.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    strengths=np.array([100.0, 80.0, 60.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    filt.continuous_particles = particles
    weights = np.array([0.9 / 18.0] * 18 + [0.0999, 0.0001], dtype=float)
    protected = np.array([18, 19], dtype=np.int64)

    np.random.seed(123)
    draw = filt._cardinality_preserving_resample_draw(
        weights,
        protected_indices=protected,
    )

    assert draw is not None
    indices, _ = draw
    labels = np.array([particles[int(idx)].state.num_sources for idx in indices])
    assert np.count_nonzero(labels == 3) == 2
    assert {18, 19}.issubset(set(indices.tolist()))
    assert filt.last_mode_preserved_count >= 1


def test_clustered_estimate_downsamples_report_points() -> None:
    """Report clustering should remain bounded without changing PF particles."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=10,
        cluster_min_samples=1,
        cluster_report_max_points=5,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[float(i), 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0 + i], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
        for i in range(10)
    ]
    positions, strengths = filt.estimate_clustered()

    assert len(filt.continuous_particles) == 10
    assert positions.shape[0] <= filt.config.max_sources
    assert strengths.size == positions.shape[0]


def test_convergence_does_not_skip_unverified_multisource_state() -> None:
    """Convergence gating should not freeze an unresolved tentative source."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        converge_enable=True,
        converge_freeze_updates=True,
        converge_require_no_tentative=True,
    )
    state = filt.continuous_particles[0].state
    state.num_sources = 2
    state.positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    state.strengths = np.array([100.0, 50.0], dtype=float)
    state.ages = np.array([10, 1], dtype=int)
    state.support_scores = np.zeros(2, dtype=float)
    state.tentative_sources = np.array([False, True], dtype=bool)
    state.verification_fail_streaks = np.zeros(2, dtype=int)
    filt.is_converged = True
    filt.frozen_estimate = (state.positions[:1].copy(), state.strengths[:1].copy())

    assert not filt._should_skip_converged_update()
    assert not filt.is_converged


def test_convergence_monitoring_does_not_freeze_updates_by_default() -> None:
    """Convergence checks should not discard later observations by default."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        converge_enable=True,
    )
    filt.is_converged = True

    assert not filt._should_skip_converged_update()
    assert filt.is_converged


def test_convergence_freeze_updates_requires_explicit_opt_in() -> None:
    """Legacy update freezing remains available only when explicitly enabled."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        converge_enable=True,
        converge_freeze_updates=True,
    )
    filt.is_converged = True

    assert filt._should_skip_converged_update()


def test_convergence_requires_compact_supported_clusters() -> None:
    """Cluster-level convergence should reject a spatially diffuse output cluster."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        converge_enable=True,
        cluster_eps_m=1.0,
        cluster_min_samples=1,
        converge_cluster_spread_max_m=0.1,
        converge_cluster_min_support_fraction=0.05,
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=positions[idx : idx + 1],
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.25)),
        )
        for idx in range(4)
    ]

    assert not filt._cluster_convergence_supported()
    filt.config.converge_cluster_spread_max_m = 1.0
    assert filt._cluster_convergence_supported()


def test_residual_birth_expands_beyond_topk_structural_particles(monkeypatch) -> None:
    """Residual-gated birth should not be limited to collapsed top-weight particles."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        structural_proposal_topk_particles=1,
        birth_residual_expand_structural_particles=True,
        birth_matching_pursuit_max_new_sources=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.99)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.01)),
        ),
    ]
    data = MeasurementData(
        z_k=np.array([20.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return a residual-gated proposal independent of top-k particles."""
        filt.last_birth_residual_gate_passed = True
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
            20.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

    def _matching_pursuit(
        st: IsotopeState,
        birth_data: MeasurementData,
        candidate_positions: np.ndarray,
        *,
        max_new_sources: int,
        candidate_unit_counts: np.ndarray | None = None,
    ) -> int:
        """Accept a birth only for the non-top empty particle."""
        assert candidate_unit_counts is None
        if st.num_sources > 0:
            return 0
        st.positions = np.array([[2.0, 0.0, 0.0]], dtype=float)
        st.strengths = np.array([100.0], dtype=float)
        st.ages = np.array([0], dtype=int)
        st.support_scores = np.array([0.0], dtype=float)
        st.num_sources = 1
        return 1

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)
    monkeypatch.setattr(
        filt,
        "_apply_matching_pursuit_births_to_state",
        _matching_pursuit,
    )
    monkeypatch.setattr(
        filt,
        "refresh_weights_from_measurements",
        lambda data, **kwargs: None,
    )

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert filt.last_birth_count == 1
    assert filt.last_birth_structural_eligible == 1
    assert filt.continuous_particles[1].state.num_sources == 1


def test_residual_birth_expansion_is_capped_and_cardinality_diverse(
    monkeypatch,
) -> None:
    """Residual-gated structural expansion should not evaluate every particle."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=5,
        structural_proposal_topk_particles=1,
        birth_residual_expand_structural_particles=True,
        birth_residual_expanded_structural_topk_particles=2,
        split_prob=0.0,
        split_residual_guided=False,
        merge_prob=0.0,
    )
    filt.continuous_particles = []
    weights = np.array([0.90, 0.05, 0.03, 0.01, 0.01], dtype=float)
    for idx, weight in enumerate(weights):
        if idx == 0:
            state = IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=float(idx),
            )
        else:
            state = IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=float(idx),
            )
        filt.continuous_particles.append(
            IsotopeParticle(state=state, log_weight=float(np.log(weight)))
        )
    data = MeasurementData(
        z_k=np.array([20.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return a residual-gated proposal for structural expansion."""
        filt.last_birth_residual_gate_passed = True
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
            20.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

    attempted: list[int] = []

    def _matching_pursuit(
        st: IsotopeState,
        birth_data: MeasurementData,
        candidate_positions: np.ndarray,
        *,
        max_new_sources: int,
        candidate_unit_counts: np.ndarray | None = None,
    ) -> int:
        """Record particles that receive exact structural birth evaluation."""
        assert candidate_unit_counts is None
        attempted.append(int(st.background))
        return 0

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)
    monkeypatch.setattr(
        filt,
        "_apply_matching_pursuit_births_to_state",
        _matching_pursuit,
    )

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert set(attempted) == {0, 1}


def test_birth_complexity_penalty_includes_bic() -> None:
    """Every birth should pay the configured and BIC model-order penalties."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_complexity_penalty=1.0e12,
        birth_bic_penalty_params=4,
    )

    penalty = filt._birth_complexity_penalty(
        measurement_count=16,
    )

    assert penalty == pytest.approx(1.0e12 + np.log(16.0) * 2.0)


def test_resampling_can_protect_distinct_low_weight_source_modes() -> None:
    """Mode-preserving resampling should retain spatially distinct source modes."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        mode_preserving_resample=True,
        mode_preserving_max_modes=3,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=0.5,
        mode_preserving_min_weight_fraction=0.0,
    )
    positions = [
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.array([[0.1, 0.0, 0.0]], dtype=float),
        np.array([[2.0, 0.0, 0.0]], dtype=float),
        np.array([[4.0, 0.0, 0.0]], dtype=float),
    ]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=pos,
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
        for pos in positions
    ]
    weights = np.array([0.94, 0.05, 0.006, 0.004], dtype=float)

    protected = filt._source_mode_preserving_indices(weights)
    injected = filt._inject_mode_preserving_indices(
        np.array([0, 0, 0, 1], dtype=np.int64),
        protected,
    )

    assert {0, 2, 3}.issubset(set(protected.tolist()))
    assert 2 in injected
    assert 3 in injected
    assert filt.last_mode_preserved_count == 2


def test_mode_preserving_resample_stratifies_surface_height_modes() -> None:
    """Surface-prior mode preservation should not merge distinct surface strata."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        mode_preserving_resample=True,
        mode_preserving_max_modes=2,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=2.0,
        mode_preserving_min_weight_fraction=0.0,
        mode_preserving_surface_strata=True,
        mode_preserving_height_bin_m=2.0,
        mode_preserving_cardinality_strata=False,
        source_position_prior="surface",
        position_max=(4.0, 4.0, 4.0),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 1.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 1.0, 1.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        ),
    ]

    protected = filt._source_mode_preserving_indices(np.array([0.5, 0.5]))

    assert set(protected.tolist()) == {0, 1}
    assert filt.last_mode_preserving_strata_summary
    selected = filt.last_mode_preserving_selected_strata
    assert len(selected) == 2
    assert {entry["height_bin"] for entry in selected} == {0}
    assert sum(int(entry["protected_count"]) for entry in selected) == 2


def test_mode_preserving_resample_adds_high_surface_particles() -> None:
    """Ceiling and high-wall modes should receive extra protected particles."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=3,
        mode_preserving_resample=True,
        mode_preserving_max_modes=1,
        mode_preserving_particles_per_mode=1,
        mode_preserving_high_surface_extra_particles=2,
        mode_preserving_high_surface_z_fraction=0.75,
        mode_preserving_radius_m=1.0,
        mode_preserving_min_weight_fraction=0.0,
        mode_preserving_surface_strata=True,
        mode_preserving_cardinality_strata=False,
        source_position_prior="surface",
        position_max=(4.0, 4.0, 4.0),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 1.0, 4.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.1, 1.0, 4.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.9, 1.0, 4.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        ),
    ]

    protected = filt._source_mode_preserving_indices(np.full(3, 1.0 / 3.0))

    assert set(protected.tolist()) == {0, 1, 2}
    assert filt.last_mode_preserving_selected_strata[0]["high_surface"] is True
    assert int(filt.last_mode_preserving_selected_strata[0]["protected_count"]) == 3


def test_mode_preserving_resample_protects_source_count_strata() -> None:
    """Mode-preserving resampling should retain low-mass source-count hypotheses."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=4,
        mode_preserving_resample=True,
        mode_preserving_max_modes=1,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=0.5,
        mode_preserving_min_weight_fraction=0.0,
        mode_preserving_cardinality_strata=True,
        mode_preserving_min_particles_per_cardinality=1,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0, 100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                dtype=float,
            ),
            strengths=np.array([100.0, 100.0, 100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=0,
            positions=np.zeros((0, 3), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=0.0,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=0.0) for state in states
    ]
    protected = filt._source_mode_preserving_indices(
        np.array([0.97, 0.02, 0.009, 0.001], dtype=float)
    )

    assert {0, 1, 2, 3}.issubset(set(protected.tolist()))
    assert filt.last_mode_preserving_cardinality_summary["3"] == pytest.approx(0.009)
    selected_counts = {
        int(entry["num_sources"])
        for entry in filt.last_mode_preserving_selected_cardinalities
    }
    assert {0, 1, 2, 3}.issubset(selected_counts)


def test_structural_weight_refresh_preserves_prior_history(monkeypatch) -> None:
    """Moved-particle refresh should apply a likelihood ratio, not reset weights."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=1.0,
            ),
            log_weight=float(np.log(0.9)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.5,
            ),
            log_weight=float(np.log(0.1)),
        ),
    ]
    data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([10.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def fake_lambda_components(
        state: IsotopeState,
        _data: MeasurementData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a deterministic one-bin expected count from background."""
        return np.zeros((1, 0), dtype=float), np.array(
            [10.0 * state.background], dtype=float
        )

    monkeypatch.setattr(filt, "_lambda_components", fake_lambda_components)
    old_ll = filt._count_log_likelihood_np(
        data.z_k,
        np.array([10.0], dtype=float),
        observation_count_variance=data.observation_variances,
    )

    filt.refresh_weights_from_measurements(
        data,
        reference_log_likelihood_by_index={0: old_ll},
        moved_indices={0},
    )

    weights = filt.continuous_weights
    assert np.allclose(weights, np.array([0.9, 0.1], dtype=float))


def test_structural_weight_refresh_reuses_metadata_only_likelihood(
    monkeypatch,
) -> None:
    """Metadata-only moves should preserve weights without a kernel recomputation."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                strengths=np.array([20.0 + idx], dtype=float),
                background=0.1,
            ),
            log_weight=float(np.log(weight)),
        )
        for idx, weight in enumerate((0.8, 0.2))
    ]
    data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([11.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def fail_window_likelihood(
        *_args: object,
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        """Fail if unchanged particles are sent through the response model."""
        raise AssertionError("metadata-only likelihood was recomputed")

    log_weights_before = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    reference_likelihoods = {0: -3.0, 1: -4.0}
    monkeypatch.setattr(
        filt,
        "_window_log_likelihoods_for_indices",
        lambda _data, _indices: np.array([-3.0, -4.0], dtype=float),
    )
    filt.refresh_weights_from_measurements(
        data,
        reference_log_likelihood_by_index=reference_likelihoods,
        moved_indices={0, 1},
    )
    reference_log_weights = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    for particle, log_weight in zip(filt.continuous_particles, log_weights_before):
        particle.log_weight = float(log_weight)
    monkeypatch.setattr(
        filt,
        "_window_log_likelihoods_for_indices",
        fail_window_likelihood,
    )

    filt.refresh_weights_from_measurements(
        data,
        reference_log_likelihood_by_index=reference_likelihoods,
        moved_indices={0, 1},
        likelihood_unchanged_indices={0, 1},
    )

    log_weights_after = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    assert np.array_equal(log_weights_after, reference_log_weights)


def test_pseudo_verification_marks_unchanged_likelihood_and_keeps_rng_schedule(
    monkeypatch,
) -> None:
    """Verification-only metadata changes should reuse likelihoods before resampling."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        pseudo_source_verification_enable=True,
        split_residual_guided=False,
        split_prob=0.0,
        merge_prob=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([40.0], dtype=float),
        background=0.1,
        ages=np.array([3], dtype=int),
        support_scores=np.zeros(1, dtype=float),
        tentative_sources=np.ones(1, dtype=bool),
        verification_fail_streaks=np.zeros(1, dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([11.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    positions_before = state.positions.copy()
    strengths_before = state.strengths.copy()
    refresh_kwargs: list[dict[str, object]] = []
    resample_draws: list[float] = []

    def verify_metadata_only(
        target: IsotopeState,
        _data: MeasurementData,
        **_kwargs: object,
    ) -> bool:
        """Mark a tentative source verified without changing its response."""
        target.tentative_sources[0] = False
        return True

    def record_refresh(
        _data: MeasurementData | None,
        **kwargs: object,
    ) -> None:
        """Record the cache classification passed to structural reweighting."""
        refresh_kwargs.append(kwargs)

    def record_resample() -> bool:
        """Record the next random draw while preserving the resample call."""
        resample_draws.append(float(np.random.rand()))
        return False

    monkeypatch.setattr(
        filt,
        "_verify_pseudo_sources_for_state",
        verify_metadata_only,
    )
    monkeypatch.setattr(filt, "refresh_weights_from_measurements", record_refresh)
    monkeypatch.setattr(
        filt,
        "_maybe_resample_after_structural_update",
        record_resample,
    )
    np.random.seed(123)
    expected_resample_draw = float(np.random.rand())
    np.random.seed(123)

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=filt.kernel.sources,
    )

    assert np.array_equal(state.positions, positions_before)
    assert np.array_equal(state.strengths, strengths_before)
    assert not bool(state.tentative_sources[0])
    assert refresh_kwargs
    assert refresh_kwargs[0]["moved_indices"] == {0}
    assert refresh_kwargs[0]["likelihood_unchanged_indices"] == {0}
    assert resample_draws == [expected_resample_draw]


def test_failed_birth_keeps_prior_move_weight_refresh(monkeypatch) -> None:
    """A rejected birth must not discard bookkeeping for an earlier PF move."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_matching_pursuit_max_new_sources=1,
        split_residual_guided=False,
        split_prob=0.0,
        merge_prob=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([40.0], dtype=float),
        background=0.1,
        ages=np.array([3], dtype=int),
        support_scores=np.zeros(1, dtype=float),
        tentative_sources=np.ones(1, dtype=bool),
        verification_fail_streaks=np.zeros(1, dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([11.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
        station_sequence_ids=np.array([0], dtype=int),
        runtime_likelihood_routes=np.array(["count"], dtype=str),
    )
    refresh_kwargs: list[dict[str, object]] = []

    def verify_metadata_only(
        target: IsotopeState,
        _data: MeasurementData,
        **_kwargs: object,
    ) -> bool:
        """Mark the existing source verified before the rejected birth."""
        target.tentative_sources[0] = False
        return True

    def rejected_birth_proposal(
        _data: MeasurementData,
        _candidates: NDArray[np.float64] | None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return a zero-response candidate that cannot produce a birth."""
        return (
            np.array([1.0], dtype=float),
            np.array([0.0], dtype=float),
            1.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
            np.zeros((1, 1), dtype=float),
        )

    def record_refresh(
        _data: MeasurementData | None,
        **kwargs: object,
    ) -> None:
        """Record reweighting after the failed birth attempt."""
        refresh_kwargs.append(kwargs)

    monkeypatch.setattr(
        filt,
        "_verify_pseudo_sources_for_state",
        verify_metadata_only,
    )
    monkeypatch.setattr(filt, "_compute_birth_proposal", rejected_birth_proposal)
    monkeypatch.setattr(filt, "refresh_weights_from_measurements", record_refresh)
    monkeypatch.setattr(
        filt,
        "_maybe_resample_after_structural_update",
        lambda: False,
    )

    filt.apply_structural_moves(data, candidate_positions=filt.kernel.sources)

    assert state.num_sources == 1
    assert not bool(state.tentative_sources[0])
    assert refresh_kwargs[0]["moved_indices"] == {0}


def test_structural_update_resamples_after_weight_collapse() -> None:
    """Delayed structural updates should not carry collapsed weights forward."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        deferred_resample_roughening_scale=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for log_weight in np.log(np.array([0.997, 0.001, 0.001, 0.001], dtype=float))
    ]

    resampled = filt._maybe_resample_after_structural_update()

    assert resampled
    assert filt.last_resample_ess
    assert filt.last_ess_post == float(filt.N)
    assert np.allclose(filt.continuous_weights, np.full(filt.N, 1.0 / filt.N))


def test_grid_initialization_repeats_strength_samples_per_cell() -> None:
    """Deterministic grid initialization should support repeated strength samples."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_grid_repeats=3,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
    )
    positions = np.vstack([p.state.positions[0] for p in filt.continuous_particles])
    grid_positions = filt._initial_grid_positions()

    assert len(filt.continuous_particles) == 3 * len(grid_positions)
    assert np.unique(positions, axis=0).shape[0] == len(grid_positions)


def test_grid_initialization_respects_source_count_prior() -> None:
    """Grid initialization should not force one source when count is unknown."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        init_num_sources=(0, 3),
        init_grid_spacing_m=1.0,
        init_grid_repeats=4,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
    )
    counts = [particle.state.num_sources for particle in filt.continuous_particles]

    assert len(filt.continuous_particles) == 4 * len(filt._initial_grid_positions())
    assert set(counts) == {0, 1, 2, 3}


def test_grid_source_counts_use_vectorized_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched grid cardinalities should match the scalar oracle used by tests."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        init_num_sources=(0, 5),
        init_grid_spacing_m=1.0,
        init_grid_repeats=4,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
    )
    particle_count = 19
    actual = filt._initial_source_counts_for_particles(
        particle_count,
        cyclic=True,
    )
    expected = np.asarray(
        [filt._initial_source_count_for_particle(idx) for idx in range(particle_count)],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(actual, expected)

    def fail_scalar_path(_particle_index: int) -> int:
        """Fail if standard grid initialization regresses to a particle loop."""
        pytest.fail("Grid initialization called the scalar source-count oracle.")

    monkeypatch.setattr(filt, "_initial_source_count_for_particle", fail_scalar_path)
    filt._init_continuous_particles()

    assert len(filt.continuous_particles) == 4 * len(filt._initial_grid_positions())
    assert {particle.state.num_sources for particle in filt.continuous_particles} == {
        0,
        1,
        2,
        3,
    }


def test_batched_joint_tuple_retry_selection_matches_scalar_oracle() -> None:
    """Batched retry selection should match pairwise scalar distance checks."""
    tuples = np.asarray(
        [
            [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
            ],
            [
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [9.0, 9.0, 0.0]],
                [[1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [9.0, 9.0, 0.0]],
                [[1.0, 1.0, 0.0], [4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [9.0, 9.0, 0.0]],
            ],
            [
                [[2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
                [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [8.0, 8.0, 0.0]],
                [[2.0, 2.0, 0.0], [1.0, 1.0, 0.0], [3.0, 3.0, 0.0], [7.0, 7.0, 0.0]],
            ],
        ],
        dtype=float,
    )
    source_counts = np.asarray([4, 3, 1], dtype=np.int64)
    threshold = 1.5
    expected: list[int] = []
    for particle_idx, source_count in enumerate(source_counts):
        selected = None
        for retry_idx in range(tuples.shape[1]):
            active = tuples[particle_idx, retry_idx, : int(source_count)]
            if active.shape[0] <= 1:
                min_distance = np.inf
            else:
                pair_left, pair_right = np.triu_indices(active.shape[0], k=1)
                min_distance = float(
                    np.min(
                        np.linalg.norm(active[pair_left] - active[pair_right], axis=1)
                    )
                )
            if min_distance >= threshold:
                selected = retry_idx
                break
        assert selected is not None
        expected.append(int(selected))

    actual = IsotopeParticleFilter._select_initial_tuple_retry_indices(
        tuples,
        source_counts,
        min_separation_m=threshold,
    )

    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int64))
    with pytest.raises(ValueError, match="Unable to construct separated"):
        IsotopeParticleFilter._select_initial_tuple_retry_indices(
            tuples[:1, :1],
            np.asarray([4], dtype=np.int64),
            min_separation_m=1.0,
        )


def test_latin_hypercube_joint_grid_initialization_is_reproducible() -> None:
    """K=3/4 tuples should retain anchors and the declared pair separation."""
    filters: list[IsotopeParticleFilter] = []
    for _ in range(2):
        np.random.seed(7)
        filters.append(
            _build_filter(
                p_birth=0.0,
                min_strength=0.01,
                max_sources=4,
                num_particles=1,
                init_num_sources=(3, 4),
                init_grid_spacing_m=1.0,
                init_grid_repeats=2,
                init_joint_position_design="latin_hypercube",
                init_joint_position_retries=32,
                init_source_min_separation_m=1.0,
                position_min=(0.0, 0.0, 0.0),
                position_max=(4.0, 4.0, 1.0),
            )
        )

    first, repeated = filters
    grid_positions = first._initial_grid_positions()
    expected_anchors = np.repeat(grid_positions, 2, axis=0)
    actual_anchors = np.vstack(
        [particle.state.positions[0] for particle in first.continuous_particles]
    )

    np.testing.assert_allclose(actual_anchors, expected_anchors, rtol=0.0, atol=0.0)
    assert np.unique(actual_anchors, axis=0).shape[0] == grid_positions.shape[0]
    assert {particle.state.num_sources for particle in first.continuous_particles} == {
        3,
        4,
    }
    for particle, repeated_particle in zip(
        first.continuous_particles,
        repeated.continuous_particles,
    ):
        np.testing.assert_allclose(
            particle.state.positions,
            repeated_particle.state.positions,
            rtol=0.0,
            atol=0.0,
        )
        pair_left, pair_right = np.triu_indices(particle.state.num_sources, k=1)
        pair_distances = np.linalg.norm(
            particle.state.positions[pair_left] - particle.state.positions[pair_right],
            axis=1,
        )
        assert float(np.min(pair_distances)) >= 1.0 - 1.0e-12


def test_grid_initialization_uses_batched_uniform_strength_prior() -> None:
    """A declared source-population range should bound all initial PF strengths."""
    np.random.seed(13)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=3,
        num_particles=1,
        init_num_sources=(1, 3),
        init_grid_spacing_m=1.0,
        init_grid_repeats=4,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
        init_strength_prior="uniform",
        init_strength_min=300000.0,
        init_strength_max=2000000.0,
    )
    strengths = np.concatenate(
        [particle.state.strengths for particle in filt.continuous_particles]
    )

    assert strengths.size > 0
    assert np.all(strengths >= 300000.0)
    assert np.all(strengths <= 2000000.0)


def test_birth_excludes_candidates_near_detector_poses() -> None:
    """Birth proposals should not place sources on measured detector poses."""
    np.random.seed(0)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_detector_min_sep_m=1.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.1,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    for particle in filt.continuous_particles:
        assert particle.state.num_sources == 1
        assert np.allclose(particle.state.positions[0], np.array([2.0, 0.0, 0.0]))


def test_evidence_death_prunes_neutral_source_and_keeps_supported_source() -> None:
    """The exact PF likelihood plus BIC should remove only a neutral source."""
    np.random.seed(23)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=1.0,
        min_strength=5.0,
        source_prune_bic_penalty_params=4,
        source_prune_delta_ll_threshold=0.0,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=2,
        pseudo_source_verification_enable=False,
        split_prob=0.0,
        merge_prob=0.0,
        num_particles=1,
        max_sources=2,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array(
            [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]],
            dtype=float,
        ),
        strengths=np.array([1.0e6, 0.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    template = MeasurementData(
        z_k=np.zeros(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    _, expected = filt._lambda_components(state, template)
    support_data = MeasurementData(
        z_k=expected,
        observation_variances=np.maximum(expected, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=support_data,
        candidate_positions=None,
    )
    final_state = filt.continuous_particles[0].state
    assert filt.last_kill_count == 1
    assert final_state.num_sources == 1
    assert final_state.strengths.tolist() == [pytest.approx(1.0e6)]

    filt.apply_structural_moves(
        evidence_data=support_data,
        candidate_positions=None,
    )
    assert filt.last_kill_count == 1
    assert filt.continuous_particles[0].state.num_sources == 1


def test_supported_source_inside_detector_exclusion_is_not_forced_dead() -> None:
    """A physical-prior violation must survive when full-history evidence supports it."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        source_detector_exclusion_m=0.25,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=2,
        pseudo_source_verification_enable=False,
        split_prob=0.0,
        merge_prob=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.51, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    detector_positions = np.array(
        [[0.5, 0.0, 0.0], [1.5, 0.5, 0.0]],
        dtype=float,
    )
    template = MeasurementData(
        z_k=np.zeros(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.array([7, 6], dtype=int),
        pb_indices=np.array([7, 6], dtype=int),
        live_times=np.ones(2, dtype=float),
        station_sequence_ids=np.array([0, 1], dtype=int),
    )
    _, expected = filt._lambda_components(state, template)
    support_data = MeasurementData(
        z_k=expected,
        observation_variances=np.maximum(expected, 1.0),
        detector_positions=detector_positions,
        fe_indices=template.fe_indices,
        pb_indices=template.pb_indices,
        live_times=template.live_times,
        station_sequence_ids=template.station_sequence_ids,
    )

    filt.apply_structural_moves(evidence_data=support_data, candidate_positions=None)

    final_state = filt.continuous_particles[0].state
    assert filt.last_kill_count == 0
    assert final_state.num_sources == 1
    assert np.allclose(final_state.positions[0], np.array([0.51, 0.0, 0.0]))


def test_unsupported_source_inside_detector_exclusion_is_evidence_pruned() -> None:
    """A physical-prior violation may be prioritized only after prune evidence."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        source_detector_exclusion_m=0.25,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=2,
        pseudo_source_verification_enable=False,
        split_prob=0.0,
        merge_prob=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.51, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    evidence_data = MeasurementData(
        z_k=np.zeros(2, dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.5, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([7, 6], dtype=int),
        pb_indices=np.array([7, 6], dtype=int),
        live_times=np.ones(2, dtype=float),
        station_sequence_ids=np.array([0, 1], dtype=int),
    )

    filt.apply_structural_moves(
        evidence_data=evidence_data,
        candidate_positions=None,
    )

    assert filt.last_kill_count == 1
    assert filt.continuous_particles[0].state.num_sources == 0
    assert any(
        event["reason"]
        == "leave_one_out_evidence_physical_prior_violation"
        for event in filt.last_source_event_diagnostics
    )


def test_estimate_returns_all_sources() -> None:
    """Estimator output should return all estimated sources without capping."""
    np.random.seed(2)
    filt = _build_filter(p_birth=1.0, min_strength=0.01, max_sources=1, num_particles=5)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                strengths=np.array([5.0, 2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                strengths=np.array([5.0, 2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    positions, strengths = filt.estimate()
    assert positions.shape[0] == 2
    assert strengths.shape[0] == positions.shape[0]


def test_weak_source_survives_with_support() -> None:
    """Weak sources should survive when delta-LL evidence is positive."""
    np.random.seed(2)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.1,
        max_sources=2,
        num_particles=1,
        support_ema_alpha=1.0,
        p_kill=1.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
                strengths=np.array([5.0, 0.5], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(1.0)),
        )
    ]
    kernel = filt.continuous_kernel
    det_pos = np.array([0.5, 0.0, 0.0], dtype=float)
    live_time = 1.0
    lam1 = (
        kernel.kernel_value_pair(
            "Cs-137", det_pos, filt.continuous_particles[0].state.positions[0], 7, 7
        )
        * filt.continuous_particles[0].state.strengths[0]
        * live_time
    )
    lam2 = (
        kernel.kernel_value_pair(
            "Cs-137", det_pos, filt.continuous_particles[0].state.positions[1], 7, 7
        )
        * filt.continuous_particles[0].state.strengths[1]
        * live_time
    )
    z_k = np.array([lam1 + lam2], dtype=float)
    support_data = MeasurementData(
        z_k=z_k,
        observation_variances=np.maximum(z_k, 1.0),
        detector_positions=np.array([det_pos], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([live_time], dtype=float),
    )
    for _ in range(3):
        filt.apply_structural_moves(
            evidence_data=support_data, candidate_positions=None
        )
    assert filt.continuous_particles[0].state.num_sources == 2


def test_birth_disabled_skips_all_structural_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-cardinality mode should return before structural trial work."""
    np.random.seed(3)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=2,
        birth_enable=False,
        p_kill=1.0,
        split_prob=1.0,
        merge_prob=1.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    particle_count_before = len(filt.continuous_particles)
    positions_before = [
        particle.state.positions.copy() for particle in filt.continuous_particles
    ]
    strengths_before = [
        particle.state.strengths.copy() for particle in filt.continuous_particles
    ]
    ages_before = [
        None if particle.state.ages is None else particle.state.ages.copy()
        for particle in filt.continuous_particles
    ]
    weights_before = [
        float(particle.log_weight) for particle in filt.continuous_particles
    ]

    def unexpected_structural_work(*_args: object, **_kwargs: object) -> None:
        """Fail if fixed-cardinality mode enters any structural helper."""
        raise AssertionError("structural helper called with birth_enable=False")

    for helper_name in (
        "_compute_birth_proposal",
        "_particle_indices_by_source_count",
        "_trial_log_likelihood",
        "_verify_pseudo_sources_for_state",
        "align_continuous_labels",
    ):
        monkeypatch.setattr(filt, helper_name, unexpected_structural_work)

    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert len(filt.continuous_particles) == particle_count_before
    for idx, particle in enumerate(filt.continuous_particles):
        assert particle.state.num_sources == 1
        assert np.array_equal(particle.state.positions, positions_before[idx])
        assert np.array_equal(particle.state.strengths, strengths_before[idx])
        if ages_before[idx] is None:
            assert particle.state.ages is None
        else:
            assert np.array_equal(particle.state.ages, ages_before[idx])
        assert particle.log_weight == weights_before[idx]
    assert filt.last_birth_count == 0
    assert filt.last_kill_count == 0
    assert filt.last_structural_timing_s["structural_moves_gated"] == 1.0


def test_standard_structural_cache_uses_grouped_prune_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard equal-cardinality path must not call scalar prune kernels."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=2,
        birth_enable=True,
        p_kill=0.0,
        split_prob=0.0,
        merge_prob=0.0,
        pseudo_source_verification_enable=False,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_min_distinct_stations=1,
        source_detector_exclusion_m=0.25,
        use_gpu=False,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0 + offset, 0.0, 0.0]], dtype=float),
                strengths=np.array([10.0 + offset], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for offset in (0.0, 0.5)
    ]
    data = MeasurementData(
        z_k=np.array([12.0, 8.0], dtype=float),
        observation_variances=np.array([2.0, 2.0], dtype=float),
        detector_positions=np.array(
            [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([1, 0], dtype=int),
        live_times=np.ones(2, dtype=float),
        station_sequence_ids=np.array([0, 1], dtype=int),
    )

    def unexpected_scalar_path(*_args: object, **_kwargs: object) -> None:
        """Fail if the standard grouped cache enters a scalar evidence helper."""
        raise AssertionError("scalar structural evidence helper was called")

    monkeypatch.setattr(
        filt,
        "_source_prune_allowed_mask",
        unexpected_scalar_path,
    )
    monkeypatch.setattr(
        filt,
        "_source_detector_exclusion_mask",
        unexpected_scalar_path,
    )

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=None,
    )

    assert len(filt.continuous_particles) == 2
    assert all(
        particle.state.num_sources == 1
        for particle in filt.continuous_particles
    )


def test_birth_enabled_adds_sources() -> None:
    """Birth mode enabled should allow adding sources."""
    np.random.seed(4)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([0.1], dtype=float),
                background=0.1,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert any(p.state.num_sources > 1 for p in filt.continuous_particles)
    assert filt.last_birth_count > 0


def test_birth_max_per_update_caps_structural_growth() -> None:
    """Birth proposals should be capped per structural update when configured."""
    np.random.seed(7)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=5,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_max_per_update=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 2
    assert any(p.state.num_sources > 0 for p in filt.continuous_particles)
    assert all(p.state.num_sources <= 2 for p in filt.continuous_particles)


def test_birth_proposal_skipped_when_all_particles_at_max_sources(monkeypatch) -> None:
    """Residual birth proposal should not run when no particle can accept birth."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        birth_enable=True,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("birth proposal should not be computed")

    monkeypatch.setattr(filt, "_compute_birth_proposal", fail_if_called)

    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 0
    assert all(p.state.num_sources == 1 for p in filt.continuous_particles)


def test_birth_gate_blocks_statistically_explained_residual() -> None:
    """Birth should not add sources when residuals are explained by count variance."""
    np.random.seed(5)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        count_likelihood_model="student_t",
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_residual_gate_p_value=0.05,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([1.0e6], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert not filt.last_birth_residual_gate_passed


def test_birth_gate_requires_multiple_supported_measurements_by_default() -> None:
    """Default birth evidence should require residual support in multiple observations."""
    np.random.seed(6)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_support == 1


def test_birth_gate_requires_residual_support_from_distinct_poses() -> None:
    """Birth evidence should be supported by residuals at multiple stations."""
    np.random.seed(8)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    same_pose_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=same_pose_data,
        candidate_positions=filt.kernel.sources,
    )

    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_support == 0
    assert filt.last_birth_residual_distinct_poses == 1

    distinct_pose_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=distinct_pose_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_count > 0


def test_birth_gate_counts_distinct_shield_views() -> None:
    """Residual birth should treat shield postures as independent views."""
    np.random.seed(9)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    shield_view_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=shield_view_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_count > 0


def test_surface_candidate_observability_diagnostics_are_truth_independent() -> None:
    """Estimator should report response observability over known surface candidates."""
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [4.0, 4.0, 0.0]],
            dtype=float,
        ),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=2,
            position_max=(4.0, 4.0, 4.0),
            source_position_prior="surface",
        ),
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    estimator.measurements.append(
        MeasurementRecord(
            z_k={"Cs-137": 10.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={"Cs-137": 10.0},
        )
    )

    diagnostics = estimator.surface_candidate_observability_diagnostics(
        max_candidates=3,
    )
    stats = diagnostics["Cs-137"]

    assert stats["candidate_count"] == 3
    assert stats["sampled_candidate_count"] == 3
    assert stats["measurement_count"] == 1
    assert stats["surface_counts"]["floor"] >= 1
    assert "max_abs_correlation" in stats


def test_birth_gate_can_require_distinct_robot_stations() -> None:
    """Residual birth can require support from more than one robot station."""
    np.random.seed(10)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
        birth_min_distinct_stations=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    one_station_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=one_station_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_residual_distinct_stations == 1

    two_station_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )
    filt.apply_structural_moves(
        evidence_data=two_station_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_stations == 2
    assert filt.last_birth_count > 0


def test_birth_readiness_does_not_gate_evidence_death(monkeypatch) -> None:
    """A birth-only station gate must not suppress leave-one-out death."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_min_distinct_poses=2,
        birth_min_distinct_stations=2,
        source_prune_min_distinct_views=1,
        source_prune_min_distinct_stations=1,
        pseudo_source_verification_enable=False,
        split_prob=0.0,
        merge_prob=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([40.0], dtype=float),
        background=0.1,
        ages=np.array([3], dtype=int),
        support_scores=np.zeros(1, dtype=float),
        tentative_sources=np.zeros(1, dtype=bool),
        verification_fail_streaks=np.zeros(1, dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    data = MeasurementData(
        z_k=np.array([1.0], dtype=float),
        observation_variances=np.array([2.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
        station_sequence_ids=np.array([0], dtype=int),
        runtime_likelihood_routes=np.array(["count"], dtype=str),
    )

    def allow_group_prune(
        _data: MeasurementData,
        lambda_m: NDArray[np.float64],
        _lambda_total: NDArray[np.float64],
        **_kwargs: object,
    ) -> NDArray[np.bool_]:
        """Allow the exact death path for every particle and source slot."""
        return np.ones(lambda_m.shape[1:], dtype=bool)

    monkeypatch.setattr(
        filt,
        "_source_prune_allowed_mask_group",
        allow_group_prune,
    )

    filt.apply_structural_moves(data, candidate_positions=filt.kernel.sources)

    assert filt.last_birth_residual_distinct_stations == 1
    assert filt.last_kill_count == 1
    assert state.num_sources == 0


def test_refresh_weights_batches_mixed_source_cardinality() -> None:
    """Batched structural reweighting should match scalar per-particle likelihoods."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        count_likelihood_model="student_t",
        spectrum_count_abs_sigma=1.0,
        use_gpu=False,
    )
    states = [
        IsotopeState(
            num_sources=0,
            positions=np.zeros((0, 3), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([80.0], dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([60.0, 110.0], dtype=float),
            background=0.2,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=np.log(1.0 / 3.0)) for state in states
    ]
    filt.N = len(states)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([31.0, 23.0, 9.0], dtype=float),
        observation_variances=np.array([32.0, 24.0, 10.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    scalar_ll = []
    for particle in filt.continuous_particles:
        _, lambda_total = filt._lambda_components(particle.state, data)
        scalar_ll.append(
            filt._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        )
    scalar_ll_arr = np.asarray(scalar_ll, dtype=float)
    max_ll = float(np.max(scalar_ll_arr))
    expected_logw = scalar_ll_arr - (
        max_ll + np.log(np.sum(np.exp(scalar_ll_arr - max_ll)))
    )

    filt.refresh_weights_from_measurements(data)

    actual_logw = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    assert np.allclose(actual_logw, expected_logw)


def test_refresh_weights_reuses_cached_lambda_totals(monkeypatch) -> None:
    """Cached exact expected counts should avoid duplicate kernel evaluation."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        use_gpu=False,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([20.0], dtype=float),
            background=0.1,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([45.0], dtype=float),
            background=0.2,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=np.log(0.5)) for state in states
    ]
    filt.N = len(filt.continuous_particles)
    data = MeasurementData(
        z_k=np.array([12.0, 7.0], dtype=float),
        observation_variances=np.array([13.0, 8.0], dtype=float),
        detector_positions=np.array(
            [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    cached_lambdas = {}
    scalar_ll = []
    for idx, particle in enumerate(filt.continuous_particles):
        _, lambda_total = filt._lambda_components(particle.state, data)
        cached_lambdas[int(idx)] = lambda_total.copy()
        scalar_ll.append(
            filt._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        )
    scalar_ll_arr = np.asarray(scalar_ll, dtype=float)
    max_ll = float(np.max(scalar_ll_arr))
    expected_logw = scalar_ll_arr - (
        max_ll + np.log(np.sum(np.exp(scalar_ll_arr - max_ll)))
    )

    def _fail_lambda_group(*args: object, **kwargs: object) -> None:
        """Fail if cached refresh recomputes grouped lambdas."""
        raise AssertionError("unexpected grouped lambda recomputation")

    def _fail_lambda_scalar(*args: object, **kwargs: object) -> None:
        """Fail if cached refresh recomputes scalar lambdas."""
        raise AssertionError("unexpected scalar lambda recomputation")

    monkeypatch.setattr(
        filt,
        "_lambda_components_for_particle_group",
        _fail_lambda_group,
    )
    monkeypatch.setattr(filt, "_lambda_components", _fail_lambda_scalar)

    filt.refresh_weights_from_measurements(
        data,
        lambda_total_by_index=cached_lambdas,
    )

    actual_logw = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    assert np.allclose(actual_logw, expected_logw)


def test_clustered_estimate_uses_robust_strength_summary() -> None:
    """Clustered output should not be dominated by rare high-strength tails."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=3,
        cluster_min_samples=1,
    )
    strengths = [100.0, 100.0, 300000.0]
    log_weights = np.log([0.80, 0.15, 0.05])
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for strength, log_weight in zip(strengths, log_weights)
    ]

    _, q_est = filt.estimate_clustered()

    assert q_est.shape == (1,)
    assert np.isclose(q_est[0], 100.0)


def test_clustered_estimate_conditions_strength_on_active_sources() -> None:
    """Clustered output should not let numeric floor sources erase an active mode."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=1,
        num_particles=4,
        cluster_min_samples=1,
    )
    strengths = [5.0, 5.0, 42000.0, 45000.0]
    log_weights = np.log([0.35, 0.25, 0.25, 0.15])
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.1, 0.0, 0.0]], dtype=float),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for strength, log_weight in zip(strengths, log_weights)
    ]

    _, q_est = filt.estimate_clustered()

    assert q_est.shape == (1,)
    assert q_est[0] > 10000.0


def test_residual_guided_split_separates_same_isotope_sources() -> None:
    """Residual-guided split should add a same-isotope source only when LL improves."""
    np.random.seed(4)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        split_prob=1.0,
        split_residual_guided=True,
        split_residual_candidate_count=3,
        split_delta_ll_threshold=0.0,
        min_age_to_split=0,
        birth_num_local_jitter=0,
        birth_min_sep_m=0.4,
        birth_detector_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        merge_prob=0.0,
    )
    initial_state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    filt.continuous_particles = [IsotopeParticle(state=initial_state, log_weight=0.0)]
    true_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    true_strengths = np.array([100.0, 120.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=filt.kernel.sources,
    )

    state = filt.continuous_particles[0].state
    assert state.num_sources == 2
    assert (
        np.min(np.linalg.norm(state.positions - true_positions[1][None, :], axis=1))
        < 0.5
    )


def test_residual_split_ranks_candidates_by_residual_support() -> None:
    """Residual split should test the strongest residual candidate first."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        split_residual_guided=True,
        split_prob=1.0,
        split_residual_candidate_count=1,
        split_delta_ll_threshold=0.0,
        min_age_to_split=0,
        birth_num_local_jitter=0,
        birth_min_sep_m=0.4,
        birth_detector_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        merge_prob=0.0,
    )
    initial_state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([90.0], dtype=float),
        background=0.0,
        ages=np.array([2], dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    filt.continuous_particles = [IsotopeParticle(state=initial_state, log_weight=0.0)]
    true_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    true_strengths = np.array([90.0, 150.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.5, 2.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    candidate_positions = np.array(
        [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )

    proposal = filt._compute_birth_proposal(data, candidate_positions)

    assert proposal is not None
    _, _, _, ranked_candidates, _ = proposal
    assert np.allclose(ranked_candidates[0], true_positions[1])

    filt.apply_structural_moves(
        evidence_data=data,
        candidate_positions=candidate_positions,
    )

    state = filt.continuous_particles[0].state
    assert state.num_sources == 2
    assert (
        np.min(np.linalg.norm(state.positions - true_positions[1][None, :], axis=1))
        < 0.5
    )


def test_birth_proposal_projects_and_deduplicates_surface_candidates() -> None:
    """Strict surface-PF candidates must be projected before response scoring."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        source_position_prior="surface",
        position_min=(0.0, 0.0, 0.0),
        position_max=(4.0, 4.0, 4.0),
        birth_num_local_jitter=0,
        birth_detector_min_sep_m=0.0,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=0.0,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        split_prob=0.0,
        merge_prob=0.0,
        use_gpu=False,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    data = MeasurementData(
        z_k=np.array([40.0], dtype=float),
        observation_variances=np.array([2.0], dtype=float),
        detector_positions=np.array([[0.5, 0.5, 0.5]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
        station_sequence_ids=np.array([0], dtype=int),
        runtime_likelihood_routes=np.array(["count"], dtype=str),
    )
    off_surface = np.array(
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        dtype=float,
    )
    projected = filt._project_positions_to_source_prior(off_surface[:1])
    assert not np.allclose(projected, off_surface[:1])

    proposal = filt._compute_birth_proposal(data, off_surface)

    assert proposal is not None
    _, _, _, proposed_positions, _ = proposal
    assert proposed_positions.shape == (1, 3)
    np.testing.assert_allclose(proposed_positions, projected)
