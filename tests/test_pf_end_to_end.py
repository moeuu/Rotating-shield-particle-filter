"""簡易なPFエンドツーエンド動作を確認するスモークテスト。"""

import ast
from collections import Counter
from dataclasses import fields
import inspect
import textwrap
import types

import numpy as np
import pytest

from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFEstimator,
    RotatingShieldPFConfig,
)
from pf.likelihood import expected_counts_per_source
from pf.particle_filter import IsotopeParticle, IsotopeParticleFilter, PFConfig
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
