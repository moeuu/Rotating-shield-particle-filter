"""簡易なPFエンドツーエンド動作を確認するスモークテスト。"""

import ast
from collections import Counter
from dataclasses import fields
import inspect
import textwrap

import numpy as np
import pytest

from pf.estimator import RotatingShieldPFEstimator, RotatingShieldPFConfig
from pf.particle_filter import PFConfig
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
    spectrum, _ = decomposer.simulate_spectrum(sources=sources, environment=env, acquisition_time=1.0, rng=np.random.default_rng(0))
    z_k = decomposer.isotope_counts(spectrum)
    est.update_pair(z_k=z_k, pose_idx=0, fe_index=0, pb_index=0, live_time_s=1.0)
    estimates = est.estimates()
    assert "Cs-137" in estimates
    positions, strengths = estimates["Cs-137"]
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)


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
    attenuated = filt.continuous_kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
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

    source = textwrap.dedent(inspect.getsource(RotatingShieldPFEstimator._build_pf_config))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "PFConfig"
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

    assert uncertain_diagnostics["Cs-137"]["observation_count_variance"] == pytest.approx(4200.0)
    assert uncertain_diagnostics["Cs-137"]["effective_log_sigma"] > 0.0
