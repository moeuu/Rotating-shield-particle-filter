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
    RotatingShieldPFEstimator,
    RotatingShieldPFConfig,
)
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
        pf_config=RotatingShieldPFConfig(
            num_particles=50,
            max_sources=1,
            init_num_sources=(1, 1),
            variable_cardinality=False,
        ),
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
            variable_cardinality=False,
            init_num_sources=(0, 0),
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

    def fake_structural_moves(*_args, **_kwargs):
        """Skip structural moves so the test isolates dispatch policy."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_structural_moves", fake_structural_moves)

    est.update_pair_sequence(
        [
            ({"Cs-137": 1.0, "Co-60": 2.0}, 0, 0, 1.0, None),
            ({"Cs-137": 3.0, "Co-60": 4.0}, 0, 0, 1.0, None),
        ],
        pose_idx=0,
        runtime_likelihood_route_by_isotope={
            "Cs-137": "count",
            "Co-60": "count",
        },
    )

    assert est.last_pair_sequence_update_workers == 2
    assert {isotope for isotope, _values in calls} == {"Cs-137", "Co-60"}
    assert len(est.measurements) == 2
    assert [record.detector_position_xyz_m for record in est.measurements] == [
        (0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    ]
    assert [record.station_sequence_id for record in est.measurements] == [0, 0]
    assert [record.station_view_index for record in est.measurements] == [0, 1]


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
            variable_cardinality=False,
            init_num_sources=(0, 0),
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))

    def fake_sequence_update(self, **_kwargs):
        """Avoid heavy likelihood work while preserving dispatch."""
        return None

    def fake_structural_moves(*_args, **_kwargs):
        """Avoid structural moves while preserving stage timing."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_structural_moves", fake_structural_moves)

    est.update_pair_sequence(
        [({"Cs-137": 1.0}, 0, 0, 1.0, None)],
        pose_idx=0,
        runtime_likelihood_route_by_isotope={"Cs-137": "count"},
    )

    stages = est.last_pair_sequence_stage_wall_s
    assert stages["isotope_sequence_update"] >= 0.0
    assert stages["structural_moves"] >= 0.0
    assert stages["history_estimate"] >= 0.0
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
            variable_cardinality=False,
            init_num_sources=(0, 0),
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

    def fake_structural_moves(*_args, **_kwargs):
        """Skip structural moves so this test isolates covariance routing."""
        return None

    monkeypatch.setattr(
        IsotopeParticleFilter,
        "update_continuous_pair_sequence",
        fake_sequence_update,
    )
    monkeypatch.setattr(est, "_apply_structural_moves", fake_structural_moves)
    view_covariance = np.array([[10.0, 3.0], [3.0, 20.0]], dtype=float)

    est.update_pair_sequence(
        [
            ({"Cs-137": 1.0}, 0, 0, 1.0, {"Cs-137": 10.0}),
            ({"Cs-137": 3.0}, 0, 0, 1.0, {"Cs-137": 20.0}),
        ],
        pose_idx=0,
        runtime_likelihood_route_by_isotope={"Cs-137": "count"},
        z_view_covariance_by_isotope={"Cs-137": view_covariance},
    )

    assert np.allclose(captured["Cs-137"], view_covariance)


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
            variable_cardinality=False,
            init_num_sources=(0, 0),
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    observed_variances = {}

    def fake_update(self, **kwargs):
        """Record the scalar variance that the independent isotope PF receives."""
        observed_variances[self.isotope] = float(kwargs["observation_count_variance"])

    def fake_structural_moves(*_args, **_kwargs):
        """Skip structural moves so this test isolates covariance projection."""
        return None

    monkeypatch.setattr(IsotopeParticleFilter, "update_continuous_pair", fake_update)
    monkeypatch.setattr(est, "_apply_structural_moves", fake_structural_moves)

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
    assert est.measurements[-1].detector_position_xyz_m == (0.5, 0.0, 0.0)
    assert est.measurements[-1].station_sequence_id == 0
    assert est.measurements[-1].station_view_index == 0


def test_update_pair_updates_missing_configured_isotope_as_zero(monkeypatch):
    """A partial count mapping must update every filter exactly as history does."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137", "Co-60"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.0, "Co-60": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(0, 0),
            history_estimate_interval=0,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    observed_counts: dict[str, float] = {}

    def fake_update(self, **kwargs):
        """Record the count supplied to each configured isotope filter."""
        observed_counts[self.isotope] = float(kwargs["z_obs"])

    monkeypatch.setattr(IsotopeParticleFilter, "update_continuous_pair", fake_update)
    monkeypatch.setattr(est, "_apply_structural_moves", lambda: None)

    est.update_pair(
        z_k={"Cs-137": 7.0},
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    co_data = est._measurement_data_for_iso("Co-60", None)

    assert observed_counts == {"Cs-137": 7.0, "Co-60": 0.0}
    assert est.measurements[-1].z_variance_k == {
        "Cs-137": 0.0,
        "Co-60": 0.0,
    }
    assert co_data is not None
    np.testing.assert_allclose(co_data.z_k, [0.0])
    np.testing.assert_allclose(co_data.observation_variances, [0.0])


def test_estimator_uses_canonical_pf_posterior_projection():
    """Runtime estimates should use the MAP-cardinality PF posterior stratum."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            max_sources=2,
            init_num_sources=(0, 2),
            variable_cardinality=True,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    patches = filt._structural_rj_surface_patches
    assert patches is not None
    patch_kinds = np.asarray(patches.kinds, dtype=object)
    left_wall_indices = np.flatnonzero(
        (patch_kinds == "wall")
        & np.isclose(patches.centers_xyz[:, 0], 0.0)
    )
    state_positions = np.vstack(
        [
            patches.centers_xyz[
                left_wall_indices[
                    np.argmin(
                        np.linalg.norm(
                            patches.centers_xyz[left_wall_indices] - target,
                            axis=1,
                        )
                    )
                ]
            ]
            for target in (
                np.asarray([0.0, 1.0, 1.0], dtype=float),
                np.asarray([0.0, 3.0, 1.0], dtype=float),
            )
        ]
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=state_positions[[0]],
                strengths=np.array([20.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.40)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=state_positions[[1]],
                strengths=np.array([40.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.35)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.25)),
        ),
    ]

    positions, strengths = est.estimates()["Cs-137"]

    conditional_weights = np.asarray([0.40, 0.35], dtype=float)
    conditional_weights /= np.sum(conditional_weights)
    unprojected_mean = np.sum(
        conditional_weights[:, None] * state_positions,
        axis=0,
    )
    expected_patch_index = int(
        np.argmin(
            np.sum(
                (patches.centers_xyz - unprojected_mean[None, :]) ** 2,
                axis=1,
            )
        )
    )
    np.testing.assert_array_equal(
        positions,
        patches.centers_xyz[[expected_patch_index]],
    )
    assert filt.structural_surface_patch_indices(positions, strict=True)[0] == (
        expected_patch_index
    )
    assert strengths == pytest.approx(np.array([29.333333333333332], dtype=float))


def test_step_diagnostics_can_skip_posterior_projection_recomputation():
    """Per-step health logs should optionally skip PF posterior projection."""
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            max_sources=2,
            init_num_sources=(0, 2),
            variable_cardinality=True,
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]

    def _forbidden_estimate(self):
        """Raise if diagnostics accidentally enter posterior projection."""
        raise AssertionError("posterior estimate should be skipped")

    filt.estimate = types.MethodType(_forbidden_estimate, filt)

    diagnostics = est.step_diagnostics(top_k=0, include_estimates=False)

    mmse_pos, mmse_strength = diagnostics["Cs-137"]["mmse"]
    assert mmse_pos.shape == (0, 3)
    assert mmse_strength.shape == (0,)
    assert diagnostics["Cs-137"]["r_mean"] >= 0.0
    assert "r_weighted_mean" in diagnostics["Cs-137"]
    assert "r_probability_by_count" in diagnostics["Cs-137"]
    assert sum(
        diagnostics["Cs-137"]["r_probability_by_count"].values()
    ) == pytest.approx(1.0)


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
        config=PFConfig(
            num_particles=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
        ),
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


def test_posterior_history_interval_can_skip_exact_projection():
    """Posterior-history recording must not alter the PF state."""
    est = object.__new__(RotatingShieldPFEstimator)
    est.pf_config = RotatingShieldPFConfig(
        variable_cardinality=False,
        init_num_sources=(0, 0),
        history_estimate_interval=0,
    )
    est.history_estimates = []

    def _forbidden_estimates(self):
        """Raise when disabled history unexpectedly projects the posterior."""
        _ = self
        raise AssertionError("history estimate should be skipped")

    est.estimates = types.MethodType(_forbidden_estimates, est)
    est._record_history_estimate(1)

    assert est.history_estimates == []

    est.pf_config = RotatingShieldPFConfig(
        variable_cardinality=False,
        init_num_sources=(0, 0),
        history_estimate_interval=2,
    )
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
            variable_cardinality=False,
            init_num_sources=(0, 0),
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
        station_sequence_ids=np.array([0, 1], dtype=np.int64),
        runtime_likelihood_routes=np.asarray(
            ["count", "count"],
            dtype="<U16",
        ),
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


def test_tempered_update_batches_remainder_after_resample_cap():
    """Tempering should not loop in tiny beta steps after resampling is capped."""
    torch = pytest.importorskip("torch")
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=2,
            variable_cardinality=False,
            init_num_sources=(0, 0),
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


def test_estimator_passes_obstacle_attenuation_to_filters():
    """PF filters should include active concrete obstacle attenuation in their kernels."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
        transport_boxes_m=((0.0, 0.0, 0.0, 1.0, 1.0, 2.0),),
    )
    est = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[-1.0, 0.5, 1.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
    )
    est.add_measurement_pose(np.array([2.0, 0.5, 1.0], dtype=float))
    est._ensure_kernel_cache()

    filt = est.filters["Cs-137"]
    source = np.array([-1.0, 0.5, 1.0], dtype=float)
    detector = np.array([2.0, 0.5, 1.0], dtype=float)
    attenuated = filt.continuous_kernel.kernel_value_pair(
        "Cs-137", detector, source, 0, 0
    )
    free = 1.0 / 9.0
    np.testing.assert_allclose(attenuated, free * np.exp(-1.0), rtol=1e-12)


def test_rotating_config_passes_physical_strength_prior():
    """Estimator config must preserve the declared physical strength prior."""
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        variable_cardinality=False,
        init_num_sources=(1, 1),
        strength_prior_min_cps_1m=300000.0,
        strength_prior_max_cps_1m=2000000.0,
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

    assert pf_config.strength_prior_min_cps_1m == pytest.approx(300000.0)
    assert pf_config.strength_prior_max_cps_1m == pytest.approx(2000000.0)


def test_pf_configs_share_the_exact_runtime_kernel_fields():
    """Estimator configuration must expose every exact PF kernel setting."""
    pf_fields = {field.name for field in fields(PFConfig)}
    rotating_fields = {field.name for field in fields(RotatingShieldPFConfig)}

    assert pf_fields <= rotating_fields
    assert {
        "variable_cardinality",
        "strength_prior_min_cps_1m",
        "strength_prior_max_cps_1m",
        "structural_rj_position_move_probability",
        "structural_rj_strength_move_probability",
    } <= pf_fields


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
