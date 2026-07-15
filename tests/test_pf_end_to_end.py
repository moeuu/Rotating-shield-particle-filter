"""簡易なPFエンドツーエンド動作を確認するスモークテスト。"""

import ast
from collections import Counter
from dataclasses import fields
import inspect
import textwrap
import types

import numpy as np
import pytest

import pf.estimator as estimator_module
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFEstimator,
    RotatingShieldPFConfig,
)
from pf.likelihood import expected_counts_per_source
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    MeasurementData,
    PFConfig,
)
from pf.state import IsotopeState
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from spectrum.pipeline import SpectralDecomposer
from measurement.model import EnvironmentConfig, PointSource


def test_pf_estimator_runs_one_step():
    """単一測定でPFが更新できることを確認する。"""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=RotatingShieldPFConfig(num_particles=50, max_sources=1),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    # PF observation should come from spectrum unfolding (Sec. 2.5.7)
    decomposer = SpectralDecomposer()
    env = EnvironmentConfig(detector_position=(0.5, 0.0, 0.0))
    sources = [PointSource("Cs-137", position=(0.0, 0.0, 0.0), intensity_cps_1m=20.0)]
    spectrum, _ = decomposer.simulate_spectrum(
        sources=sources,
        environment=env,
        acquisition_time=1.0,
        rng=np.random.default_rng(0),
    )
    z_k = decomposer.isotope_counts(spectrum)
    est.update_pair(z_k=z_k, pose_idx=0, fe_index=0, pb_index=0, live_time_s=1.0)
    estimates = est.estimates()
    assert "Cs-137" in estimates
    positions, strengths = estimates["Cs-137"]
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)


def test_estimator_can_start_without_active_detected_isotopes():
    """Estimator can activate spectrum-detected isotopes after an empty start."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5, "Co-60": 0.4},
        pf_config=RotatingShieldPFConfig(num_particles=8, max_sources=1, use_gpu=False),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    est.restrict_isotopes([], allow_empty=True)

    assert est.isotopes == []
    assert est.filters == {}

    est.add_isotopes(["Co-60"])
    assert est.isotopes == ["Co-60"]
    assert set(est.filters) == {"Co-60"}

    est.add_isotopes(["Cs-137"])
    assert est.isotopes == ["Cs-137", "Co-60"]
    assert set(est.filters) == {"Cs-137", "Co-60"}


def test_update_pair_sequence_uses_parallel_isotope_workers(monkeypatch):
    """Station joint updates should dispatch independent isotopes in parallel."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            parallel_isotope_updates=True,
            parallel_isotope_workers=2,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    calls = []

    def fake_sequence_update(self, **kwargs):
        """Record that this isotope received the station sequence update."""
        calls.append((self.isotope, tuple(np.asarray(kwargs["z_obs"], dtype=float))))

    def fake_birth_death(*_args, **_kwargs):
        """Skip structural moves so the test isolates dispatch policy."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_birth_death", fake_birth_death)

    est.update_pair_sequence(
        [
            ({"Cs-137": 1.0, "Co-60": 2.0}, 0, 0, 1.0, None),
            ({"Cs-137": 3.0, "Co-60": 4.0}, 0, 0, 1.0, None),
        ],
        pose_idx=0,
    )

    assert est.last_pair_sequence_update_workers == 2
    assert {isotope for isotope, _values in calls} == {"Cs-137", "Co-60"}
    report_summary = est.best_report_summary()
    assert report_summary["snapshot_count"] == 1
    assert report_summary["last"]["label"] == "measurement_2"


def test_update_pair_sequence_records_stage_timings(monkeypatch):
    """Station joint updates should expose stage-level wall-time diagnostics."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))

    def fake_sequence_update(self, **_kwargs):
        """Avoid heavy likelihood work while preserving dispatch."""
        return None

    def fake_birth_death(*_args, **_kwargs):
        """Avoid structural moves while preserving stage timing."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_birth_death", fake_birth_death)

    est.update_pair_sequence(
        [({"Cs-137": 1.0}, 0, 0, 1.0, None)],
        pose_idx=0,
    )

    stages = est.last_pair_sequence_stage_wall_s
    assert stages["isotope_sequence_update"] >= 0.0
    assert stages["sparse_poisson_refresh"] >= 0.0
    assert stages["birth_death"] >= 0.0
    assert stages["report_snapshot"] >= 0.0
    assert stages["total"] >= stages["isotope_sequence_update"]


def test_update_pair_sequence_passes_view_covariance(monkeypatch):
    """Station joint updates should pass same-shield-program view covariance."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    captured: dict[str, np.ndarray] = {}

    def fake_sequence_update(self, **kwargs):
        """Record the covariance matrix received by the isotope PF."""
        captured[self.isotope] = np.asarray(
            kwargs["observation_count_covariance"],
            dtype=float,
        )

    def fake_birth_death(*_args, **_kwargs):
        """Skip structural moves so this test isolates covariance routing."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_birth_death", fake_birth_death)
    view_covariance = np.array([[10.0, 3.0], [3.0, 20.0]], dtype=float)

    est.update_pair_sequence(
        [
            ({"Cs-137": 1.0}, 0, 0, 1.0, {"Cs-137": 10.0}),
            ({"Cs-137": 3.0}, 0, 0, 1.0, {"Cs-137": 20.0}),
        ],
        pose_idx=0,
        z_view_covariance_by_isotope={"Cs-137": view_covariance},
    )

    assert np.allclose(captured["Cs-137"], view_covariance)


def test_update_pair_sequence_records_spectrum_payload(monkeypatch):
    """Station sequence records should preserve direct spectrum-bin payloads."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            sparse_poisson_evidence_enable=False,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))

    def fake_sequence_update(self, **_kwargs):
        """Avoid stochastic PF updates while preserving measurement history."""
        return None

    def fake_birth_death(*_args, **_kwargs):
        """Skip structural moves so this test isolates history storage."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_birth_death", fake_birth_death)
    spectrum_payload = {
        "spectrum_counts": np.array([10.0, 3.0, 1.0], dtype=float),
        "spectrum_variance": np.array([10.0, 3.0, 1.0], dtype=float),
        "spectrum_background": np.array([1.0, 1.0, 1.0], dtype=float),
        "spectrum_response_templates_by_isotope": {
            "Cs-137": np.array([1.0, 0.2, 0.0], dtype=float)
        },
    }

    est.update_pair_sequence(
        [
            (
                {"Cs-137": 10.0},
                0,
                0,
                1.0,
                {"Cs-137": 10.0},
                None,
                spectrum_payload,
            ),
        ],
        pose_idx=0,
    )

    record = est.measurements[-1]
    assert record.spectrum_counts == pytest.approx((10.0, 3.0, 1.0))
    assert record.spectrum_background == pytest.approx((1.0, 1.0, 1.0))
    assert record.spectrum_response_templates_by_isotope is not None
    assert record.spectrum_response_templates_by_isotope["Cs-137"] == pytest.approx(
        (1.0, 0.2, 0.0)
    )


def test_update_pair_projects_isotope_covariance_to_pf_variance(monkeypatch):
    """Same-spectrum isotope covariance should widen independent PF variances."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    observed_variances = {}

    def fake_update(self, **kwargs):
        """Record the scalar variance that the independent isotope PF receives."""
        observed_variances[self.isotope] = float(
            kwargs["observation_count_variance"]
        )

    def fake_birth_death(*_args, **_kwargs):
        """Skip structural moves so this test isolates covariance projection."""
        return None

    monkeypatch.setattr(IsotopeParticleFilter, "update_continuous_pair", fake_update)
    monkeypatch.setattr(est, "_apply_birth_death", fake_birth_death)

    est.update_pair(
        z_k={"Cs-137": 100.0, "Co-60": 80.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        z_variance_k={"Cs-137": 25.0, "Co-60": 16.0},
        z_covariance_k={
            "Cs-137": {"Cs-137": 25.0, "Co-60": -12.0},
            "Co-60": {"Cs-137": -12.0, "Co-60": 16.0},
        },
    )

    assert observed_variances["Cs-137"] == pytest.approx(37.0)
    assert observed_variances["Co-60"] == pytest.approx(28.0)
    assert est.measurements[-1].z_variance_k == pytest.approx(
        {"Cs-137": 37.0, "Co-60": 28.0}
    )
    assert est.measurements[-1].z_covariance_k is not None
    assert est.measurements[-1].z_covariance_k["Cs-137"]["Co-60"] == pytest.approx(
        -12.0
    )


def test_runtime_report_rescue_queue_only_does_not_inject_particles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue-only report rescue should retain candidates without PF injection."""
    isotope = "Cs-137"
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=2,
            runtime_report_rescue_verification_queue_only=True,
            runtime_report_rescue_memory_enable=False,
            candidate_verification_queue_weight=0.07,
            report_best_so_far_enable=False,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    est.add_measurement_pose(np.array([0.0, 0.0, 0.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters[isotope]
    positions = np.array([[1.0, 2.0, 0.5], [3.0, 1.0, 0.5]], dtype=float)
    strengths = np.array([120.0, 80.0], dtype=float)

    def fake_rescue_estimate(
        rescue_isotope: str,
        _filt: IsotopeParticleFilter,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic rescue candidates for queue-only routing."""
        assert rescue_isotope == isotope
        return positions.copy(), strengths.copy()

    def forbidden_injection(*_args: object, **_kwargs: object) -> int:
        """Fail if queue-only mode tries to inject particles."""
        raise AssertionError("queue-only rescue must not inject PF particles")

    monkeypatch.setattr(est, "_runtime_report_rescue_estimate", fake_rescue_estimate)
    monkeypatch.setattr(
        filt,
        "inject_runtime_report_rescue_particles",
        forbidden_injection,
    )

    injected = est._inject_runtime_report_rescue(isotope, filt)

    assert injected == 0
    queued = est._candidate_verification_queue[isotope]
    np.testing.assert_allclose(queued[0], positions)
    np.testing.assert_allclose(queued[1], strengths)
    modes = est.runtime_report_rescue_modes()[isotope]
    np.testing.assert_allclose(modes[0], positions)
    np.testing.assert_allclose(modes[1], strengths)
    assert modes[2] == pytest.approx(0.07)


def test_sparse_evidence_prunes_contradicted_verification_queue_candidates() -> None:
    """Decisive sparse evidence should delete queued candidates it rejects."""
    isotope = "Cs-137"
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=2,
            sparse_poisson_evidence_authoritative=True,
            candidate_verification_queue_enable=True,
            report_mle_rescue_dedup_radius_m=0.25,
            report_best_so_far_enable=False,
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    near = np.array([1.0, 1.0, 0.5], dtype=float)
    far = np.array([3.0, 3.0, 0.5], dtype=float)
    est._candidate_verification_queue[isotope] = (
        np.vstack([near, far]),
        np.array([100.0, 90.0], dtype=float),
        np.array([100.0, 90.0], dtype=float),
    )
    payload = {
        "model_order_ready": True,
        "selected_positions": [near.tolist()],
    }

    removed = est._prune_candidate_verification_queue_with_sparse_evidence(
        isotope,
        payload,
    )

    assert removed == 1
    queued = est._candidate_verification_queue[isotope]
    np.testing.assert_allclose(queued[0], near.reshape(1, 3))
    np.testing.assert_allclose(queued[1], np.array([100.0], dtype=float))


def test_update_pair_refreshes_sparse_poisson_evidence_without_report_call():
    """All-history sparse evidence should refresh immediately after observations."""
    isotope = "Cs-137"
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=2,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            sparse_poisson_evidence_enable=True,
            sparse_poisson_evidence_authoritative=True,
            sparse_poisson_evidence_min_bic_margin=0.0,
            sparse_poisson_evidence_parameter_count_per_source=1,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    for pose in detector_positions:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    expected = expected_counts_per_source(
        kernel=est.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=candidate_sources[:1],
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    )

    for pose_idx, count in enumerate(np.sum(expected, axis=1)):
        est.update_pair(
            z_k={isotope: float(count)},
            pose_idx=pose_idx,
            fe_index=0,
            pb_index=0,
            live_time_s=1.0,
            z_variance_k={isotope: max(float(count), 1.0)},
        )

    diagnostics = est.sparse_poisson_evidence_diagnostics()
    payload = diagnostics[isotope]
    assert payload["available"] is True
    assert payload["method"] == "all_history_sparse_poisson"
    assert payload["measurement_count"] == detector_positions.shape[0]
    assert payload["selected_count"] == 1
    assert payload["model_order_ready"] is True


def test_sparse_evidence_requires_independent_station_support_for_authority():
    """Sparse evidence should not become authoritative from one shield-only station."""
    isotope = "Cs-137"
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=2,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            sparse_poisson_evidence_enable=True,
            sparse_poisson_evidence_authoritative=True,
            sparse_poisson_evidence_min_bic_margin=0.0,
            sparse_poisson_evidence_min_distinct_stations=2,
            sparse_poisson_evidence_parameter_count_per_source=1,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    station = np.array([1.0, 0.0, 0.0], dtype=float)
    est.add_measurement_pose(station)
    est._ensure_kernel_cache()
    detector_positions = np.repeat(station.reshape(1, 3), repeats=3, axis=0)
    expected = expected_counts_per_source(
        kernel=est.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=candidate_sources[:1],
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    )

    for count in np.sum(expected, axis=1):
        est.update_pair(
            z_k={isotope: float(count)},
            pose_idx=0,
            fe_index=0,
            pb_index=0,
            live_time_s=1.0,
            z_variance_k={isotope: max(float(count), 1.0)},
        )

    payload = est.sparse_poisson_evidence_diagnostics()[isotope]
    assert payload["available"] is True
    assert payload["distinct_station_count"] == 1
    assert payload["min_distinct_stations_for_ready"] == 2
    assert payload["model_order_ready"] is False


def test_sparse_evidence_zero_strengths_are_refit_for_report_and_sync():
    """Authoritative evidence positions should not disappear due to zero seed strength."""
    isotope = "Cs-137"
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=2,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            report_strength_refit=True,
            report_strength_refit_prior_weight=4.0,
            sparse_poisson_evidence_authoritative=True,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    for pose in detector_positions:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    true_strength = 180.0
    counts = expected_counts_per_source(
        kernel=est.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=candidate_sources[:1],
        strengths=np.array([true_strength], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    ).reshape(-1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
    )
    est._last_sparse_poisson_evidence_diagnostics = {
        isotope: {
            "available": True,
            "model_order_ready": True,
            "selected_count": 1,
            "selected_positions": [candidate_sources[0].tolist()],
            "selected_strengths": [0.0],
        }
    }
    background = est._background_counts_for_report_refit(isotope, data.live_times)

    report_override = est._sparse_poisson_report_override(
        isotope,
        filt=est.filters[isotope],
        data=data,
        background=background,
    )
    sync_ready, sync_positions, sync_strengths = est._authoritative_sparse_evidence_sources(
        isotope,
        filt=est.filters[isotope],
        data=data,
    )

    assert report_override is not None
    report_positions, report_strengths = report_override
    np.testing.assert_allclose(report_positions, candidate_sources[:1])
    assert report_strengths.shape == (1,)
    assert report_strengths[0] > est.pf_config.min_strength
    assert report_strengths[0] == pytest.approx(true_strength, rel=0.15)
    assert sync_ready is True
    np.testing.assert_allclose(sync_positions, candidate_sources[:1])
    assert sync_strengths[0] == pytest.approx(report_strengths[0], rel=1.0e-6)


def test_sparse_evidence_report_override_requires_ready_model_order():
    """Authoritative report override should wait for decisive sparse evidence."""
    isotope = "Cs-137"
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            sparse_poisson_evidence_authoritative=True,
            report_best_so_far_enable=False,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    est._last_sparse_poisson_evidence_diagnostics = {
        isotope: {
            "available": True,
            "model_order_ready": False,
            "selected_count": 1,
            "selected_positions": [[0.0, 0.0, 0.0]],
            "selected_strengths": [100.0],
        }
    }

    assert est._sparse_poisson_report_override(isotope) is None


def test_update_pair_prefers_spectrum_bin_sparse_evidence_when_available():
    """Sparse evidence should use direct spectrum bins when payloads are available."""
    isotope = "Cs-137"
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            sparse_poisson_evidence_enable=True,
            sparse_poisson_evidence_authoritative=True,
            sparse_poisson_evidence_min_bic_margin=0.0,
            sparse_poisson_evidence_parameter_count_per_source=1,
            sparse_poisson_spectral_evidence_enable=True,
            sparse_poisson_spectral_evidence_primary=True,
            sparse_poisson_spectral_nuisance_enable=False,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    for pose in detector_positions:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    scalar_counts = expected_counts_per_source(
        kernel=est.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=candidate_sources[:1],
        strengths=np.array([150.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    ).reshape(-1)
    template = np.array([1.0, 0.25, 0.02], dtype=float)

    for pose_idx, count in enumerate(scalar_counts):
        spectrum = count * template
        est.update_pair(
            z_k={isotope: float(count)},
            pose_idx=pose_idx,
            fe_index=0,
            pb_index=0,
            live_time_s=1.0,
            z_variance_k={isotope: max(float(count), 1.0)},
            spectrum_payload={
                "spectrum_counts": spectrum,
                "spectrum_background": np.zeros_like(template),
                "spectrum_response_templates_by_isotope": {isotope: template},
            },
        )

    payload = est.sparse_poisson_evidence_diagnostics()[isotope]
    assert payload["available"] is True
    assert payload["method"] == "spectral_bin_sparse_poisson"
    assert payload["selected_count"] == 1
    assert payload["spectrum_measurement_count"] == detector_positions.shape[0]
    assert payload["spectrum_bin_count"] == template.size
    assert payload["selected_indices"] == [0]
    assert "count_sparse_poisson_evidence" in payload


def test_sparse_poisson_offgrid_refinement_improves_selected_position():
    """Sparse evidence should profile coarse candidates in continuous space."""
    isotope = "Cs-137"
    candidate_sources = np.array(
        [[0.1, 0.1, 0.1], [2.5, 2.5, 0.5]],
        dtype=float,
    )
    true_source = np.array([[0.45, 0.5, 0.4]], dtype=float)
    est = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            sparse_poisson_evidence_enable=True,
            sparse_poisson_evidence_authoritative=True,
            sparse_poisson_evidence_min_bic_margin=0.0,
            sparse_poisson_spectral_evidence_enable=False,
            sparse_poisson_offgrid_refine_enable=True,
            sparse_poisson_offgrid_refine_radius_m=0.8,
            sparse_poisson_offgrid_refine_max_iter=96,
            position_min=(0.0, 0.0, 0.0),
            position_max=(3.0, 3.0, 2.0),
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.array(
        [
            [1.5, 0.2, 0.2],
            [0.2, 1.6, 0.2],
            [1.4, 1.4, 0.8],
            [2.0, 0.6, 0.6],
            [0.6, 2.0, 0.7],
        ],
        dtype=float,
    )
    for pose in detector_positions:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    true_counts = expected_counts_per_source(
        kernel=est.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=true_source,
        strengths=np.array([220.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    ).reshape(-1)

    for pose_idx, count in enumerate(true_counts):
        est.update_pair(
            z_k={isotope: float(count)},
            pose_idx=pose_idx,
            fe_index=0,
            pb_index=0,
            live_time_s=1.0,
            z_variance_k={isotope: max(float(count), 1.0)},
        )

    payload = est.sparse_poisson_evidence_diagnostics()[isotope]
    coarse_distance = float(np.linalg.norm(candidate_sources[0] - true_source[0]))
    refined_position = np.asarray(payload["selected_positions"], dtype=float).reshape(
        -1,
        3,
    )[0]
    refined_distance = float(np.linalg.norm(refined_position - true_source[0]))

    assert payload["available"] is True
    assert payload["selected_indices"] == [0]
    assert payload["offgrid_refined"] is True
    assert payload["offgrid_refinement"]["accepted"] is True
    assert refined_distance < coarse_distance


def test_joint_sparse_poisson_evidence_projects_multi_isotope_cardinality():
    """Joint spectrum-bin evidence should project cardinality back to isotopes."""
    isotopes = ["Cs-137", "Co-60"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            birth_enable=False,
            conditional_strength_refit=False,
            history_estimate_interval=0,
            report_best_so_far_enable=False,
            sparse_poisson_evidence_enable=True,
            sparse_poisson_evidence_authoritative=True,
            sparse_poisson_evidence_min_bic_margin=0.0,
            sparse_poisson_evidence_parameter_count_per_source=1,
            sparse_poisson_spectral_evidence_enable=True,
            sparse_poisson_spectral_evidence_primary=True,
            sparse_poisson_spectral_nuisance_enable=False,
            sparse_poisson_joint_evidence_enable=True,
            use_gpu=True,
            gpu_device="cpu",
        ),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=float,
    )
    for pose in detector_positions:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    live_times = np.ones(detector_positions.shape[0], dtype=float)
    fe_indices = np.zeros(detector_positions.shape[0], dtype=int)
    pb_indices = np.zeros(detector_positions.shape[0], dtype=int)
    cs_counts = expected_counts_per_source(
        kernel=est.filters["Cs-137"].continuous_kernel,
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=candidate_sources[:1],
        strengths=np.array([120.0], dtype=float),
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        source_scale=1.0,
    ).reshape(-1)
    co_counts = expected_counts_per_source(
        kernel=est.filters["Co-60"].continuous_kernel,
        isotope="Co-60",
        detector_positions=detector_positions,
        sources=candidate_sources[1:2],
        strengths=np.array([90.0], dtype=float),
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        source_scale=1.0,
    ).reshape(-1)
    cs_template = np.array([1.0, 0.2, 0.0, 0.0], dtype=float)
    co_template = np.array([0.0, 0.0, 0.3, 1.0], dtype=float)

    for pose_idx, (cs_count, co_count) in enumerate(zip(cs_counts, co_counts)):
        spectrum = cs_count * cs_template + co_count * co_template
        est.update_pair(
            z_k={"Cs-137": float(cs_count), "Co-60": float(co_count)},
            pose_idx=pose_idx,
            fe_index=0,
            pb_index=0,
            live_time_s=1.0,
            z_variance_k={
                "Cs-137": max(float(cs_count), 1.0),
                "Co-60": max(float(co_count), 1.0),
            },
            spectrum_payload={
                "spectrum_counts": spectrum,
                "spectrum_background": np.zeros_like(spectrum),
                "spectrum_response_templates_by_isotope": {
                    "Cs-137": cs_template,
                    "Co-60": co_template,
                },
            },
        )

    diagnostics = est.sparse_poisson_evidence_diagnostics()
    joint_payload = diagnostics["joint_multi_isotope"]
    assert joint_payload["available"] is True
    assert joint_payload["method"] == "joint_multi_isotope_sparse_poisson"
    assert joint_payload["selected_counts_by_isotope"] == {
        "Cs-137": 1,
        "Co-60": 1,
    }
    assert diagnostics["Cs-137"]["method"] == (
        "joint_multi_isotope_sparse_poisson_projection"
    )
    assert diagnostics["Co-60"]["selected_indices"] == [1]


def test_estimator_uses_clustered_output_when_birth_is_enabled():
    """Final PF estimates should honor the clustered-output configuration."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            min_particles=2,
            max_particles=2,
            max_sources=2,
            birth_enable=True,
            use_clustered_output=True,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]

    def _fake_clustered(self):
        """Return a distinctive clustered estimate."""
        return (
            np.array([[1.0, 2.0, 3.0]], dtype=float),
            np.array([42.0], dtype=float),
        )

    def _fake_mmse(self):
        """Return a fallback estimate that should not be used."""
        return (
            np.array([[9.0, 9.0, 9.0]], dtype=float),
            np.array([9.0], dtype=float),
        )

    filt.estimate_clustered = types.MethodType(_fake_clustered, filt)
    filt.estimate = types.MethodType(_fake_mmse, filt)

    positions, strengths = est.estimates()["Cs-137"]

    assert positions == pytest.approx(np.array([[1.0, 2.0, 3.0]], dtype=float))
    assert strengths == pytest.approx(np.array([42.0], dtype=float))


def test_step_diagnostics_can_skip_report_estimate_recomputation():
    """Per-step health logs should not require clustered report recomputation."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            min_particles=2,
            max_particles=2,
            max_sources=2,
            birth_enable=True,
            use_clustered_output=True,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]

    def _forbidden_clustered(self):
        """Raise if diagnostics accidentally enter report-only clustering."""
        raise AssertionError("clustered estimate should be skipped")

    filt.estimate_clustered = types.MethodType(_forbidden_clustered, filt)

    diagnostics = est.step_diagnostics(top_k=0, include_estimates=False)

    mmse_pos, mmse_strength = diagnostics["Cs-137"]["mmse"]
    assert mmse_pos.shape == (0, 3)
    assert mmse_strength.shape == (0,)
    assert diagnostics["Cs-137"]["r_mean"] >= 0.0
    assert "r_weighted_mean" in diagnostics["Cs-137"]
    assert "r_probability_by_count" in diagnostics["Cs-137"]
    assert sum(diagnostics["Cs-137"]["r_probability_by_count"].values()) == pytest.approx(
        1.0
    )


def test_report_refit_removes_redundant_duplicate_cluster():
    """Reported clusters should pay the same refit-after-remove model penalty."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            min_particles=2,
            max_particles=2,
            max_sources=2,
            birth_enable=True,
            report_strength_refit=True,
            report_strength_refit_preserve_cardinality=True,
            report_cluster_model_selection=True,
            use_clustered_output=True,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    poses = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 0.0], dtype=float),
    ]
    for pose in poses:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    true_source = np.array([[0.0, 0.0, 0.0]], dtype=float)
    z = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope="Cs-137",
        detector_positions=np.vstack(poses),
        sources=true_source,
        strengths=np.array([1000.0], dtype=float),
        live_times=np.ones(len(poses), dtype=float),
        fe_indices=np.zeros(len(poses), dtype=int),
        pb_indices=np.zeros(len(poses), dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    est.measurements = [
        MeasurementRecord(
            z_k={"Cs-137": float(value)},
            pose_idx=idx,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={"Cs-137": max(float(value), 1.0)},
        )
        for idx, value in enumerate(z)
    ]

    positions, strengths = est._refit_reported_strengths(
        "Cs-137",
        np.vstack([true_source, true_source]),
        np.array([500.0, 500.0], dtype=float),
    )

    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
    assert strengths[0] == pytest.approx(1000.0, rel=0.05)


def test_continuous_pair_expected_counts_supports_cpu_config():
    """Continuous expected counts should use the same model without CUDA."""
    dummy_kernel = types.SimpleNamespace(
        poses=[np.array([1.0, 0.0, 0.0], dtype=float)],
        orientations=[np.array([1.0, 0.0, 0.0], dtype=float)],
        num_sources=1,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
    )
    filt = IsotopeParticleFilter(
        "Cs-137",
        kernel=dummy_kernel,
        config=PFConfig(num_particles=1, use_gpu=False),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([5.0], dtype=float),
                background=1.0,
            ),
            log_weight=0.0,
        )
    ]

    lam = filt._continuous_expected_counts_pair(
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=2.0,
    )

    assert lam == pytest.approx(np.array([12.0], dtype=float))


def test_deferred_pose_update_delays_structural_update(monkeypatch):
    """Deferred pose updates should postpone only station-level structure moves."""
    update_defer_flags = []
    finalize_calls = []
    birth_calls = []

    def _fake_update_continuous_pair(
        self,
        z_obs,
        pose_idx,
        fe_index,
        pb_index,
        live_time_s,
        observation_count_variance=0.0,
        step_idx=None,
        defer_resample=False,
    ):
        """Record whether the estimator requested a deferred update."""
        update_defer_flags.append(bool(defer_resample))
        self.last_ess_pre = 3.0
        self.last_ess = 3.0
        self.last_resample_ess = False

    def _fake_finalize_deferred_update(self):
        """Record station-level finalization calls."""
        finalize_calls.append(self.isotope)
        self.last_resample_ess = True
        self.last_ess_post = float(len(self.continuous_particles))

    def _fake_apply_birth_death(self, birth_window_override=None):
        """Record birth/death applications."""
        _ = birth_window_override
        birth_calls.append(len(self.measurements))

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair",
        _fake_update_continuous_pair,
    )
    monkeypatch.setattr(
        IsotopeParticleFilter,
        "finalize_deferred_update",
        _fake_finalize_deferred_update,
    )
    monkeypatch.setattr(
        RotatingShieldPFEstimator,
        "_apply_birth_death",
        _fake_apply_birth_death,
    )

    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            adaptive_strength_prior=False,
        ),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0], dtype=float))

    est.begin_deferred_pose_update()
    est.update_pair(
        z_k={"Cs-137": 4.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    est.update_pair(
        z_k={"Cs-137": 5.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert update_defer_flags == [True, True]
    assert birth_calls == []
    assert len(est.measurements) == 2

    finalized = est.finalize_deferred_pose_update()

    assert finalized == 2
    assert finalize_calls == ["Cs-137"]
    assert birth_calls == [2]


def test_deferred_pose_update_defers_history_estimate_recompute(monkeypatch):
    """Deferred measurement updates should not recompute report estimates per posture."""

    def _fake_update_continuous_pair(self, *args, **kwargs):
        """Record no-op PF update for this report-history regression test."""
        _ = (self, args, kwargs)

    def _forbidden_estimates(self, *args, **kwargs):
        """Raise if update_pair enters expensive report estimate recomputation."""
        _ = (self, args, kwargs)
        raise AssertionError("deferred update should not recompute estimates")

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair",
        _fake_update_continuous_pair,
    )
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _forbidden_estimates)

    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            adaptive_strength_prior=False,
        ),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0], dtype=float))

    est.begin_deferred_pose_update()
    est.update_pair(
        z_k={"Cs-137": 4.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert len(est.history_estimates) == 0
    assert len(est.measurements) == 1
    assert est._deferred_measurement_count == 1


def test_report_history_interval_can_skip_exact_report_recompute():
    """Report-history recording should be configurable without changing PF state."""
    est = object.__new__(RotatingShieldPFEstimator)
    est.pf_config = RotatingShieldPFConfig(history_estimate_interval=0)
    est.history_estimates = []

    def _forbidden_estimates(self):
        """Raise when history recording unexpectedly computes a report estimate."""
        _ = self
        raise AssertionError("history estimate should be skipped")

    est.estimates = types.MethodType(_forbidden_estimates, est)
    est._record_history_estimate(1)

    assert est.history_estimates == []

    est.pf_config = RotatingShieldPFConfig(history_estimate_interval=2)
    calls = []

    def _fake_estimates(self):
        """Return a minimal estimate payload for history recording."""
        calls.append(1)
        _ = self
        return {"Cs-137": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float))}

    est.estimates = types.MethodType(_fake_estimates, est)
    est._record_history_estimate(1)
    est._record_history_estimate(2)

    assert calls == [1]
    assert len(est.history_estimates) == 1


def test_candidate_response_cache_reuses_full_surface_grid(monkeypatch):
    """Full-grid candidate responses should be cached without changing values."""
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=candidate_sources,
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            use_gpu=False,
            candidate_response_cache_max_entries=4,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    data = MeasurementData(
        z_k=np.array([3.0, 4.0], dtype=float),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=np.int64),
        pb_indices=np.array([1, 0], dtype=np.int64),
        live_times=np.ones(2, dtype=float),
    )
    calls = []

    def _fake_expected_counts_per_source(**kwargs):
        """Return deterministic response columns and count cache misses."""
        calls.append(kwargs)
        detectors = np.asarray(kwargs["detector_positions"], dtype=float)
        sources = np.asarray(kwargs["sources"], dtype=float)
        return np.full(
            (detectors.shape[0], sources.reshape(-1, 3).shape[0]),
            float(len(calls)),
            dtype=float,
        )

    monkeypatch.setattr(
        estimator_module,
        "expected_counts_per_source",
        _fake_expected_counts_per_source,
    )
    filt = types.SimpleNamespace(continuous_kernel=object())
    sources = np.asarray(est.candidate_sources, dtype=float).reshape(-1, 3)

    first = est._cached_expected_counts_per_source(
        filt=filt,
        isotope="Cs-137",
        data=data,
        sources=sources,
        strengths=np.ones(sources.shape[0], dtype=float),
    )
    first[0, 0] = 99.0
    second = est._cached_expected_counts_per_source(
        filt=filt,
        isotope="Cs-137",
        data=data,
        sources=sources,
        strengths=np.ones(sources.shape[0], dtype=float),
    )

    assert len(calls) == 1
    assert second == pytest.approx(np.ones_like(second))


def test_deferred_pose_update_runs_convergence_once_at_finalize(monkeypatch):
    """Deferred updates should move convergence clustering to station finalization."""
    convergence_steps = []

    def _fake_gpu_enabled(self):
        """Bypass hardware availability for the branch test."""
        _ = self
        return True

    def _fake_tempered_update(
        self,
        lam_fn,
        z_obs,
        observation_count_variance=0.0,
        disable_regularize_on_resample=None,
        roughening_scale_on_resample=1.0,
    ):
        """Avoid expected-count evaluation while exercising deferred control flow."""
        _ = (
            self,
            lam_fn,
            z_obs,
            observation_count_variance,
            disable_regularize_on_resample,
            roughening_scale_on_resample,
        )
        return 3.0, False

    def _fake_maybe_update_convergence(
        self,
        step_idx,
        detector_pos,
        fe_index,
        pb_index,
        live_time_s,
        z_obs,
    ):
        """Record convergence checks without running clustered reports."""
        _ = (self, detector_pos, fe_index, pb_index, live_time_s, z_obs)
        convergence_steps.append(step_idx)

    def _fake_apply_birth_death(self, birth_window_override=None):
        """Avoid structural moves so the test isolates convergence scheduling."""
        _ = (self, birth_window_override)

    monkeypatch.setattr(IsotopeParticleFilter, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(
        IsotopeParticleFilter,
        "_tempered_update",
        _fake_tempered_update,
    )
    monkeypatch.setattr(
        IsotopeParticleFilter,
        "_maybe_update_convergence",
        _fake_maybe_update_convergence,
    )
    monkeypatch.setattr(
        RotatingShieldPFEstimator,
        "_apply_birth_death",
        _fake_apply_birth_death,
    )

    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            max_sources=1,
            adaptive_strength_prior=False,
            use_tempering=True,
        ),
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0], dtype=float))

    est.begin_deferred_pose_update()
    est.update_pair(
        z_k={"Cs-137": 4.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    est.update_pair(
        z_k={"Cs-137": 5.0},
        pose_idx=0,
        fe_index=1,
        pb_index=1,
        live_time_s=1.0,
    )

    assert convergence_steps == []

    finalized = est.finalize_deferred_pose_update()

    assert finalized == 2
    assert convergence_steps == [1]


def test_tempered_update_batches_remainder_after_resample_cap():
    """Tempering should not loop in tiny beta steps after resampling is capped."""
    torch = pytest.importorskip("torch")
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=2,
            min_delta_beta=1.0e-3,
            target_ess_ratio=0.99,
            max_resamples_per_observation=0,
        ),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(-np.log(2.0)),
        )
        for _ in range(2)
    ]
    ll_t = torch.tensor([0.0, -50.0], dtype=torch.float64)

    ess_pre, resampled = filt._tempered_update_likelihood(lambda: ll_t)

    final_logw = np.asarray([p.log_weight for p in filt.continuous_particles])
    expected_logw = np.asarray([0.0, -50.0], dtype=float)
    expected_logw = expected_logw - np.log(np.sum(np.exp(expected_logw)))
    assert resampled is False
    assert ess_pre == pytest.approx(1.0)
    assert len(filt.last_temper_steps) <= 2
    assert final_logw == pytest.approx(expected_logw)


def test_deferred_pair_update_still_uses_tempered_resampling(monkeypatch):
    """Deferred station updates should still allow intra-station resampling."""
    calls = []

    def _fake_gpu_enabled(self):
        """Bypass hardware availability for the branch test."""
        return True

    def _fake_tempered_update(
        self,
        lam_fn,
        z_obs,
        observation_count_variance=0.0,
        disable_regularize_on_resample=None,
        roughening_scale_on_resample=1.0,
    ):
        """Record the deferred tempered-update request."""
        _ = lam_fn, z_obs, observation_count_variance
        calls.append(
            (
                bool(disable_regularize_on_resample),
                float(roughening_scale_on_resample),
            )
        )
        self.last_resample_ess = True
        self.last_ess_pre = 1.0
        self.last_ess_post = float(len(self.continuous_particles))
        return 1.0, True

    monkeypatch.setattr(IsotopeParticleFilter, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(
        IsotopeParticleFilter,
        "_tempered_update",
        _fake_tempered_update,
    )

    dummy_kernel = types.SimpleNamespace(
        poses=[np.array([0.0, 0.0, 0.0], dtype=float)],
        orientations=[np.array([1.0, 0.0, 0.0], dtype=float)],
        num_sources=1,
    )
    filt = IsotopeParticleFilter(
        "Cs-137",
        kernel=dummy_kernel,
        config=PFConfig(num_particles=2, use_tempering=True),
    )

    filt.update_continuous_pair(
        z_obs=1.0,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        defer_resample=True,
    )

    assert calls == [(False, 0.15)]
    assert filt._deferred_resampled_any


def test_estimator_passes_obstacle_attenuation_to_filters():
    """PF filters should include active concrete obstacle attenuation in their kernels."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[-1.0, 0.0, 1.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            min_particles=1,
            max_particles=1,
            max_sources=1,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
    )
    est.add_measurement_pose(np.array([2.0, 0.0, 1.0], dtype=float))
    est._ensure_kernel_cache()

    filt = est.filters["Cs-137"]
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)
    attenuated = filt.continuous_kernel.kernel_value_pair(
        "Cs-137", detector, source, 0, 0
    )
    free = 1.0 / 9.0
    np.testing.assert_allclose(attenuated, free * np.exp(-1.0), rtol=1e-12)


def test_rotating_config_passes_strength_and_label_parameters():
    """Estimator config should not silently drop PF parameters."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        init_strength_log_mean=2.5,
        init_strength_log_sigma=0.25,
        label_pos_weight=1.7,
        label_strength_weight=0.4,
        label_missing_cost=123.0,
        label_pos_scale=2.0,
        label_strength_scale=50.0,
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )

    pf_config = est._build_pf_config()

    assert pf_config.init_strength_log_mean == pytest.approx(2.5)
    assert pf_config.init_strength_log_sigma == pytest.approx(0.25)
    assert pf_config.label_pos_weight == pytest.approx(1.7)
    assert pf_config.label_strength_weight == pytest.approx(0.4)
    assert pf_config.label_missing_cost == pytest.approx(123.0)
    assert pf_config.label_pos_scale == pytest.approx(2.0)
    assert pf_config.label_strength_scale == pytest.approx(50.0)


def test_rotating_config_exposes_and_maps_all_pf_config_fields():
    """Every core PFConfig field must be exposed and forwarded by the estimator config."""
    pf_fields = {field.name for field in fields(PFConfig)}
    rotating_fields = {field.name for field in fields(RotatingShieldPFConfig)}
    assert pf_fields <= rotating_fields

    source = textwrap.dedent(
        inspect.getsource(RotatingShieldPFEstimator._build_pf_config)
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PFConfig"
    ]
    assert len(calls) == 1
    mapped_fields = {keyword.arg for keyword in calls[0].keywords if keyword.arg}
    assert pf_fields <= mapped_fields


def test_rotating_config_has_no_duplicate_field_annotations():
    """Duplicate dataclass annotations would silently overwrite earlier defaults."""
    source = textwrap.dedent(inspect.getsource(RotatingShieldPFConfig))
    tree = ast.parse(source)
    class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = [
        node.target.id
        for node in class_def.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert duplicates == []


def test_adaptive_strength_prior_matches_observed_count_scale():
    """Count-conditioned strength adaptation should infer scale from the observation."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        max_sources=1,
        use_gpu=False,
        position_min=(0.0, 0.0, 0.0),
        position_max=(1.0, 1.0, 1.0),
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_strength_log_mean=float(np.log(10000.0)),
        init_strength_log_sigma=0.0,
        adaptive_strength_prior=True,
        adaptive_strength_prior_steps=1,
        adaptive_strength_prior_min_counts=0.0,
        adaptive_strength_prior_log_sigma=0.0,
        background_level=0.0,
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.5, 0.5, 0.5]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([1.5, 0.5, 0.5], dtype=float))
    est._ensure_kernel_cache()
    before = est.filters["Cs-137"].continuous_particles[0].state.strengths[0]

    diagnostics = est.adapt_strength_prior_to_observation(
        z_k={"Cs-137": 42.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=2.0,
    )

    after = est.filters["Cs-137"].continuous_particles[0].state.strengths[0]
    assert before == pytest.approx(10000.0)
    assert after == pytest.approx(21.0)
    assert diagnostics["Cs-137"]["before_median_strength"] == pytest.approx(10000.0)
    assert diagnostics["Cs-137"]["after_median_strength"] == pytest.approx(21.0)
    assert diagnostics["Cs-137"]["observation_count_variance"] == pytest.approx(0.0)
    assert diagnostics["Cs-137"]["effective_log_sigma"] == pytest.approx(0.0)

    np.random.seed(0)
    uncertain_diagnostics = est.adapt_strength_prior_to_observation(
        z_k={"Cs-137": 42.0},
        z_variance_k={"Cs-137": 4200.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=2.0,
    )

    assert uncertain_diagnostics["Cs-137"][
        "observation_count_variance"
    ] == pytest.approx(4200.0)
    assert uncertain_diagnostics["Cs-137"]["effective_log_sigma"] > 0.0


def test_adaptive_strength_prior_floor_does_not_increase_strength():
    """The weak-count floor should only downscale, never create high-strength outliers."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        max_sources=1,
        use_gpu=False,
        position_min=(0.0, 0.0, 0.0),
        position_max=(1.0, 1.0, 1.0),
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_strength_log_mean=float(np.log(1.0)),
        init_strength_log_sigma=0.0,
        adaptive_strength_prior=True,
        adaptive_strength_prior_steps=1,
        adaptive_strength_prior_min_counts=3.0,
        adaptive_strength_prior_log_sigma=0.0,
        adaptive_strength_prior_max_upscale=10.0,
        background_level=0.0,
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.5, 0.5, 0.5]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([1.5, 0.5, 0.5], dtype=float))
    est._ensure_kernel_cache()
    before = float(est.filters["Cs-137"].continuous_particles[0].state.strengths[0])

    diagnostics = est.adapt_strength_prior_to_observation(
        z_k={"Cs-137": 0.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=2.0,
    )

    after = float(est.filters["Cs-137"].continuous_particles[0].state.strengths[0])
    assert after <= before
    assert diagnostics["Cs-137"]["floor_only_target"] == pytest.approx(1.0)


def test_pair_sequence_adaptive_strength_prior_counts_pending_records(monkeypatch):
    """Same-station sequences should consume the adaptive-prior step budget once."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        min_particles=1,
        max_particles=1,
        max_sources=1,
        use_gpu=False,
        position_min=(0.0, 0.0, 0.0),
        position_max=(1.0, 1.0, 1.0),
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        adaptive_strength_prior=True,
        adaptive_strength_prior_steps=2,
        background_level=0.0,
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.5, 0.5, 0.5]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([1.5, 0.5, 0.5], dtype=float))
    calls: list[int | None] = []

    def fake_adapt_strength_prior_to_observation(**kwargs):
        """Record adaptive-prior calls without running the heavy count kernel."""
        calls.append(kwargs.get("completed_measurement_count"))
        return {}

    monkeypatch.setattr(
        est,
        "adapt_strength_prior_to_observation",
        fake_adapt_strength_prior_to_observation,
    )
    monkeypatch.setattr(est, "_run_isotope_pair_sequence_update", lambda task: None)
    monkeypatch.setattr(est, "refresh_sparse_poisson_evidence", lambda: None)
    monkeypatch.setattr(est, "_apply_birth_death", lambda *args, **kwargs: None)
    monkeypatch.setattr(est, "record_report_snapshot", lambda *args, **kwargs: None)

    records = [
        ({"Cs-137": 10.0 + offset}, offset % 8, offset % 8, 1.0, None)
        for offset in range(8)
    ]

    est.update_pair_sequence(records, pose_idx=0)

    assert calls == [0, 1]
    assert len(est.measurements) == 8
