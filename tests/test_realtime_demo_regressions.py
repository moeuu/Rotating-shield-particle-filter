"""Regression coverage for isotope locking and missing-measurement handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import realtime_demo as realtime_demo_module
from measurement.obstacles import ObstacleGrid
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import IsotopeParticle, MeasurementData
from pf.profiles import resolve_estimator_profile
from pf.pure_estimator import PurePFEstimator
from pf.state import IsotopeState
from realtime_demo import (
    ADAPTIVE_STEP_ID_STRIDE,
    DeferredPFVisualizer,
    _acquire_spectrum_observation,
    _adaptive_mission_stop_reason,
    _apply_baseline_shield_program_to_dss_config,
    _argv_requests_cui,
    _build_effective_live_runtime_config,
    _build_intermediate_estimate_trace_payload,
    _build_robot_path_segment,
    _compute_shield_selection_grid,
    _complete_count_covariance,
    _diagnostic_detail_order,
    _evaluate_spectrum_count_result,
    _filter_reachable_candidates,
    _format_pf_timing_item,
    _format_estimate_trace_log_line,
    _format_truth_coverage_log_line,
    _best_dss_first_step_guard_candidate,
    _final_pf_cardinality_status,
    _inflate_low_signal_variances,
    _is_adaptive_spectrum_ready,
    _isotope_count_balance_penalty,
    _log_spectrum_isotope_channel_diagnostics,
    _log_precision_degradation_diagnostics,
    _log_particle_cloud_diagnostics,
    _log_surface_candidate_observability_diagnostics,
    _pf_obstacle_attenuation_enabled,
    _pf_obstacle_grid_for_runtime,
    _pure_pf_primary_estimates,
    _pure_pf_summary_provenance,
    _measurement_log_obstacle_layout_path,
    _measurement_transport_provenance,
    _resolve_ig_workers,
    _resolve_runtime_use_gpu,
    _resolve_mission_max_poses,
    _resolve_mission_max_steps,
    _resolve_plot_save_interval,
    _resolve_python_worker_count,
    _resolve_cui_split_view_enabled,
    _resolve_candidate_isotopes,
    _resolve_required_measurement_log_target,
    _particle_surface_diagnostics,
    _select_best_pair_from_scores,
    _signature_vector_is_dependent,
    _spectrum_config_from_runtime_config,
    _source_cardinality_dwell_status,
    _truth_free_live_runtime_config,
    _validate_surface_constrained_estimates,
    run_live_pf,
)
from planning.dss_pp import DSSPPConfig
from sim import SimulationCommand, SimulationObservation
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig
from spectrum.runtime_counts import RuntimeCountResult
from visualization.realtime_viz import PFFrame


def test_complete_count_covariance_preserves_cross_isotope_terms() -> None:
    """Build one full symmetric covariance without losing one-sided terms."""
    completed = _complete_count_covariance(
        {"Cs-137": 9.0, "Co-60": 16.0, "Eu-154": 25.0},
        {
            "Cs-137": {"Cs-137": 10.0, "Co-60": -3.0},
            "Co-60": {"Cs-137": -5.0},
        },
        ("Cs-137", "Co-60", "Eu-154"),
    )

    assert completed["Cs-137"]["Cs-137"] == pytest.approx(10.0)
    assert completed["Co-60"]["Co-60"] == pytest.approx(16.0)
    assert completed["Eu-154"]["Eu-154"] == pytest.approx(25.0)
    assert completed["Cs-137"]["Co-60"] == pytest.approx(-4.0)
    assert completed["Co-60"]["Cs-137"] == pytest.approx(-4.0)
    assert completed["Cs-137"]["Eu-154"] == pytest.approx(0.0)
    assert completed["Eu-154"]["Cs-137"] == pytest.approx(0.0)


def test_measurement_transport_provenance_keeps_full_history_evidence() -> None:
    """Measurement logs should retain transport fidelity for every PF update."""
    provenance = _measurement_transport_provenance(
        {
            "backend": "geant4",
            "source_rate_model": "detector_cps_1m",
            "expected_primary_semantics": "detector_equivalent_histories",
            "expected_detector_equivalent_primaries": 1234.0,
            "expected_sampled_primaries": 1234.0,
            "physics_profile": "balanced",
            "primary_sampling_fraction": 1.0,
            "primary_history_weight": 1.0,
            "requested_threads": 32,
            "source_bias_mode": "detector_cone",
            "multithreaded_run_manager": True,
            "unrelated": "omit",
        }
    )

    assert provenance == {
        "expected_detector_equivalent_primaries": 1234.0,
        "expected_primary_semantics": "detector_equivalent_histories",
        "expected_sampled_primaries": 1234.0,
        "multithreaded_run_manager": True,
        "physics_profile": "balanced",
        "primary_history_weight": 1.0,
        "primary_sampling_fraction": 1.0,
        "requested_threads": 32,
        "source_bias_mode": "detector_cone",
        "source_rate_model": "detector_cps_1m",
    }


def test_measurement_transport_provenance_keeps_dynamic_sampling_budget() -> None:
    """Measurement logs should distinguish configured and effective sampling."""
    provenance = _measurement_transport_provenance(
        {
            "primary_sampling_fraction": 0.04,
            "primary_history_weight": 25.0,
            "requested_primary_sampling_fraction": 0.2,
            "target_sampled_primaries": 100000,
            "primary_sampling_budget_enabled": True,
            "primary_sampling_fraction_resolution": "target_budget_limited",
            "unrelated": "omit",
        }
    )

    assert provenance == {
        "primary_history_weight": 25.0,
        "primary_sampling_budget_enabled": True,
        "primary_sampling_fraction": 0.04,
        "primary_sampling_fraction_resolution": "target_budget_limited",
        "requested_primary_sampling_fraction": 0.2,
        "target_sampled_primaries": 100000,
    }


def test_dss_history_weight_is_derived_from_transport_sampling_fraction() -> None:
    """RAL/DSS planning should inherit the configured minimum history weight."""
    assert realtime_demo_module._planning_primary_history_weight({}) == pytest.approx(
        1.0
    )
    assert realtime_demo_module._planning_primary_history_weight(
        {"primary_sampling_fraction": 0.02}
    ) == pytest.approx(50.0)
    assert realtime_demo_module._planning_primary_history_weight(
        {
            "primary_sampling_fraction": 0.2,
            "target_sampled_primaries": 100000,
        }
    ) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="primary_sampling_fraction"):
        realtime_demo_module._planning_primary_history_weight(
            {"primary_sampling_fraction": 0.0}
        )


def test_target_sampled_primaries_validation_is_fail_closed() -> None:
    """Dynamic transport budgets must be absent or positive JSON integers."""
    assert realtime_demo_module._target_sampled_primaries({}) is None
    assert (
        realtime_demo_module._target_sampled_primaries(
            {"target_sampled_primaries": 100000}
        )
        == 100000
    )
    for invalid in (
        0,
        -1,
        100000.0,
        "100000",
        True,
        float("inf"),
        float("nan"),
    ):
        with pytest.raises(ValueError, match="target_sampled_primaries"):
            realtime_demo_module._target_sampled_primaries(
                {"target_sampled_primaries": invalid}
            )


def test_transport_budget_radius_ignores_pf_count_radius_override() -> None:
    """DSS budgeting must use the native crystal instead of a PF-only override."""
    runtime_config = {
        "detector_model": {"crystal_radius_m": 0.041},
        "pf_detector_count_radius_m": 0.25,
    }

    assert realtime_demo_module._transport_detector_budget_radius_m(
        runtime_config
    ) == pytest.approx(0.041)
    assert realtime_demo_module._transport_detector_budget_radius_m(
        {}
    ) == pytest.approx(0.038)


def test_adaptive_target_budget_fails_before_transport_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-invocation primary budgets must reject adaptive dwell before transport."""
    runtime_config = {
        "target_sampled_primaries": 1_500_000,
        "primary_sampling_fraction": 1.0,
        "accelerated_weighted_transport_enable": True,
    }
    monkeypatch.setattr(
        realtime_demo_module,
        "load_runtime_config",
        lambda _path: dict(runtime_config),
    )
    monkeypatch.setattr(
        realtime_demo_module,
        "enforce_pure_runtime_settings",
        lambda config: dict(config),
    )
    transport_setup_called = False

    def _unexpected_transport_setup(**_kwargs: object) -> None:
        """Record transport setup if the budget guard runs too late."""
        nonlocal transport_setup_called
        transport_setup_called = True

    monkeypatch.setattr(
        realtime_demo_module,
        "build_runtime_obstacle_environment",
        _unexpected_transport_setup,
    )

    with pytest.raises(
        ValueError,
        match=(
            "target budget per Geant4 transport invocation.*adaptive planning.*"
            "unsupported"
        ),
    ):
        run_live_pf(
            live=False,
            adaptive_dwell=True,
            measurement_time_s=30.0,
            sim_config_path=None,
            save_outputs=False,
        )

    assert transport_setup_called is False


def test_weighted_pf_runtime_contract_accepts_single_evidence_path() -> None:
    """Weighted transport should pass only with complete covariance evidence."""
    realtime_demo_module._validate_weighted_pf_runtime_contract(
        {
            "primary_sampling_fraction": 0.2,
            "accelerated_weighted_transport_enable": True,
        },
        count_likelihood_model="student_t",
        observation_variance_semantics="complete_statistical",
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=False,
        planning_primary_history_weight=5.0,
    )
    realtime_demo_module._validate_weighted_pf_runtime_contract(
        {
            "primary_sampling_fraction": 1.0,
            "target_sampled_primaries": 100000,
            "accelerated_weighted_transport_enable": True,
        },
        count_likelihood_model="student_t",
        observation_variance_semantics="complete_statistical",
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=False,
        planning_primary_history_weight=1.0,
    )


@pytest.mark.parametrize(
    (
        "runtime_config",
        "count_likelihood_model",
        "variance_semantics",
        "contrast_enabled",
        "ratio_enabled",
        "history_weight",
        "error_match",
    ),
    [
        (
            {"primary_sampling_fraction": 0.2},
            "student_t",
            "complete_statistical",
            False,
            False,
            5.0,
            "accelerated_weighted_transport_enable",
        ),
        (
            {
                "primary_sampling_fraction": 0.2,
                "accelerated_weighted_transport_enable": True,
            },
            "student_t",
            "additional_model_uncertainty",
            False,
            False,
            5.0,
            "complete_statistical",
        ),
        (
            {
                "primary_sampling_fraction": 0.2,
                "accelerated_weighted_transport_enable": True,
            },
            "poisson",
            "complete_statistical",
            False,
            False,
            5.0,
            "gaussian or student_t",
        ),
        (
            {
                "primary_sampling_fraction": 0.2,
                "accelerated_weighted_transport_enable": True,
            },
            "student_t",
            "complete_statistical",
            True,
            False,
            5.0,
            "auxiliary likelihoods",
        ),
        (
            {
                "primary_sampling_fraction": 0.2,
                "accelerated_weighted_transport_enable": True,
            },
            "student_t",
            "complete_statistical",
            False,
            True,
            5.0,
            "auxiliary likelihoods",
        ),
        (
            {
                "primary_sampling_fraction": 0.2,
                "accelerated_weighted_transport_enable": True,
            },
            "student_t",
            "complete_statistical",
            False,
            False,
            50.0,
            "reciprocal",
        ),
        (
            {"primary_sampling_fraction": 1.0, "target_sampled_primaries": 100000},
            "student_t",
            "complete_statistical",
            False,
            False,
            1.0,
            "accelerated_weighted_transport_enable",
        ),
    ],
)
def test_weighted_pf_runtime_contract_fails_closed(
    runtime_config: dict[str, object],
    count_likelihood_model: str,
    variance_semantics: str,
    contrast_enabled: bool,
    ratio_enabled: bool,
    history_weight: float,
    error_match: str,
) -> None:
    """Each weighted-runtime double-counting guard should fail independently."""
    with pytest.raises(ValueError, match=error_match):
        realtime_demo_module._validate_weighted_pf_runtime_contract(
            runtime_config,
            count_likelihood_model=count_likelihood_model,
            observation_variance_semantics=variance_semantics,
            shield_contrast_likelihood_enable=contrast_enabled,
            shield_view_ratio_likelihood_enable=ratio_enabled,
            planning_primary_history_weight=history_weight,
        )


def test_pf_strength_prior_uses_predeclared_generator_population() -> None:
    """Generated simulations should share their declared strength population prior."""
    minimum, maximum = realtime_demo_module._resolve_pf_strength_prior_bounds(
        {
            "source_rate_model": "detector_cps_1m",
            "random_source_intensity_min_cps_1m": 300000.0,
            "random_source_intensity_max_cps_1m": 2000000.0,
        }
    )

    assert minimum == pytest.approx(300000.0)
    assert maximum == pytest.approx(2000000.0)


def test_pf_random_seed_controls_numpy_and_torch_planning_draws() -> None:
    """One PF seed should reproduce both CPU and GPU planning randomness."""
    torch = pytest.importorskip("torch")
    realtime_demo_module._seed_pf_random_generators(314159)
    numpy_first = np.random.random(4)
    torch_first = torch.rand(4)

    realtime_demo_module._seed_pf_random_generators(314159)
    numpy_second = np.random.random(4)
    torch_second = torch.rand(4)

    np.testing.assert_array_equal(numpy_first, numpy_second)
    torch.testing.assert_close(torch_first, torch_second, rtol=0.0, atol=0.0)


def test_explicit_pf_strength_prior_bounds_override_generator_population() -> None:
    """Explicit PF prior settings must take precedence over generator defaults."""
    minimum, maximum = realtime_demo_module._resolve_pf_strength_prior_bounds(
        {
            "source_rate_model": "detector_cps_1m",
            "random_source_intensity_min_cps_1m": 300000.0,
            "random_source_intensity_max_cps_1m": 2000000.0,
            "pf_strength_prior_min_cps_1m": 100000.0,
            "pf_strength_prior_max_cps_1m": 3000000.0,
        }
    )

    assert minimum == pytest.approx(100000.0)
    assert maximum == pytest.approx(3000000.0)


def test_pf_strength_prior_requires_predeclared_bounds() -> None:
    """Pure PF must fail closed when no bounded strength population is declared."""
    with pytest.raises(ValueError, match="requires explicit"):
        realtime_demo_module._resolve_pf_strength_prior_bounds(
            {"source_rate_model": "detector_cps_1m"}
        )


def test_surface_report_quality_gate_rejects_off_surface_estimate() -> None:
    """The runtime must fail closed before publishing an invalid surface report."""
    estimates = {
        "Cs-137": (
            np.asarray([[0.5, 0.5, 0.5]], dtype=float),
            np.asarray([10.0], dtype=float),
        )
    }

    with pytest.raises(RuntimeError, match="off-surface positions"):
        _validate_surface_constrained_estimates(
            estimates,
            EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
            None,
            obstacle_height_m=2.0,
            tolerance_m=1.0e-5,
            surface_prior_active=True,
        )


def test_surface_report_quality_gate_uses_exact_pf_dictionary() -> None:
    """The runtime gate must accept exact bottoms and reject non-patch wall points."""
    isotope = "Cs-137"
    env = EnvironmentConfig(size_x=3.0, size_y=3.0, size_z=3.0)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
    )
    estimator = PurePFEstimator(
        isotopes=(isotope,),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_patch_spacing_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
        measurement_log_sha256="d" * 64,
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    patches = estimator.filters[isotope]._structural_rj_surface_patches
    assert patches is not None
    patch_kinds = np.asarray(patches.kinds, dtype=object)
    bottom_center = patches.centers_xyz[
        int(np.flatnonzero(patch_kinds == "obstacle_bottom")[0])
    ]
    valid_estimate = {
        isotope: (
            bottom_center[None, :],
            np.asarray([100.0], dtype=float),
        )
    }

    _validate_surface_constrained_estimates(
        valid_estimate,
        env,
        obstacle_grid,
        obstacle_height_m=2.0,
        tolerance_m=1.0e-5,
        surface_prior_active=True,
        estimator=estimator,
    )

    invalid_estimate = {
        isotope: (
            np.asarray([[0.0, 0.3, 0.3]], dtype=float),
            np.asarray([100.0], dtype=float),
        )
    }
    with pytest.raises(RuntimeError, match="off-surface positions"):
        _validate_surface_constrained_estimates(
            invalid_estimate,
            env,
            obstacle_grid,
            obstacle_height_m=2.0,
            tolerance_m=1.0e-5,
            surface_prior_active=True,
            estimator=estimator,
        )


def test_cli_max_poses_overrides_runtime_config_pose_cap() -> None:
    """An explicit CLI pose cap should not be overwritten by runtime config."""
    runtime_config = {"mission_stop_max_poses": 10}

    assert _resolve_mission_max_poses(8, runtime_config) == 8
    assert _resolve_mission_max_poses(None, runtime_config) == 10


def test_cli_max_steps_overrides_runtime_measurement_budget() -> None:
    """The fixed config budget applies only when the CLI omits max steps."""
    runtime_config = {"measurement_budget_max_steps": 160}

    assert _resolve_mission_max_steps(80, runtime_config) == 80
    assert _resolve_mission_max_steps(None, runtime_config) == 160
    assert _resolve_mission_max_steps(0, runtime_config) is None


def test_effective_live_config_is_truth_free_and_binds_exact_pf_inputs(
    tmp_path: Path,
) -> None:
    """Live provenance strips source generation while hashing actual PF support."""
    raw = {
        "source_rate_model": "detector_cps_1m",
        "source_extent_radius_m": 0.05,
        "random_source_seed": 7,
        "random_source_count": 3,
        "random_source_intensity_cps_1m": 1000.0,
        "source_generation_mode": "surface_random",
        "source_layout_path": "secret-layout.json",
        "nested": {"random_source_isotopes": ["Cs-137"]},
    }
    sanitized = _truth_free_live_runtime_config(raw)
    serialized = json.dumps(sanitized, sort_keys=True)
    assert sanitized["source_rate_model"] == "detector_cps_1m"
    assert sanitized["source_extent_radius_m"] == pytest.approx(0.05)
    for fragment in ("random_source", "source_generation", "source_layout"):
        assert fragment not in serialized

    config = RotatingShieldPFConfig(num_particles=8, use_gpu=False)
    first_grid = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    first = _build_effective_live_runtime_config(
        raw,
        pf_config=config,
        candidate_sources_xyz=first_grid,
        source_position_bounds=(np.zeros(3), np.ones(3)),
        api_settings={
            "pf_random_seed": 5,
            "candidate_grid_spacing_m": [1.0, 1.0, 1.0],
        },
    )
    second = _build_effective_live_runtime_config(
        raw,
        pf_config=config,
        candidate_sources_xyz=np.asarray([[0.0, 0.0, 0.0]]),
        source_position_bounds=(np.zeros(3), np.ones(3)),
        api_settings={
            "pf_random_seed": 5,
            "candidate_grid_spacing_m": [1.0, 1.0, 1.0],
        },
    )
    assert first["effective_pf_replay"]["candidate_grid"]["point_count"] == 2
    assert first != second

    with pytest.raises(ValueError, match="require measurement_log_output"):
        _resolve_required_measurement_log_target(None, {}, repository_root=tmp_path)
    target = _resolve_required_measurement_log_target(
        None,
        {"measurement_log_output_dir": "logs/run"},
        repository_root=tmp_path,
    )
    assert target == tmp_path / "logs/run"


def test_measurement_log_obstacle_layout_path_is_portable_and_physical(
    tmp_path: Path,
) -> None:
    """Only a repository-local fixed asset should become a log pointer."""
    repository_root = tmp_path / "repository"
    fixed_layout = repository_root / "obstacle_layouts" / "fixed.json"
    fixed_environment = SimpleNamespace(mode="fixed", layout_path=fixed_layout)
    random_environment = SimpleNamespace(mode="random", layout_path=fixed_layout)

    assert (
        _measurement_log_obstacle_layout_path(
            fixed_environment,
            repository_root=repository_root,
        )
        == "obstacle_layouts/fixed.json"
    )
    assert (
        _measurement_log_obstacle_layout_path(
            random_environment,
            repository_root=repository_root,
        )
        is None
    )

    external_environment = SimpleNamespace(
        mode="fixed",
        layout_path=tmp_path / "external" / "fixed.json",
    )
    with pytest.raises(ValueError, match="must be inside the repository"):
        _measurement_log_obstacle_layout_path(
            external_environment,
            repository_root=repository_root,
        )


def test_pure_primary_estimates_preserve_low_strength_posterior_modes() -> None:
    """Primary pure-PF output must preserve every posterior source mode."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    expected_positions = np.asarray([[1.0, 2.0, 0.5]], dtype=float)
    expected_strengths = np.asarray([25.0], dtype=float)
    estimator = SimpleNamespace(
        profile_capabilities=capabilities,
        pf_config=SimpleNamespace(estimator_profile="pf_strict"),
        estimates=lambda: {
            "Cs-137": (expected_positions.copy(), expected_strengths.copy())
        },
    )

    actual = _pure_pf_primary_estimates(estimator, ("Cs-137", "Co-60"))

    assert actual is not None
    np.testing.assert_array_equal(actual["Cs-137"][0], expected_positions)
    np.testing.assert_array_equal(actual["Cs-137"][1], expected_strengths)
    assert actual["Co-60"][0].shape == (0, 3)
    assert actual["Co-60"][1].shape == (0,)


def test_pure_pf_summary_embeds_complete_posterior_provenance() -> None:
    """Every pure-PF result file must identify its log, config, and PF origin."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    payload = {
        "schema_version": 1,
        "pure_pf_schema_version": 1,
        "estimator_family": "particle_filter",
        "estimator_variant": "pf_strict",
        "estimator_profile": "pf_strict",
        "final_estimate_source": "pf_posterior",
        "posterior_semantics": (
            "fixed_cardinality_sequential_particle_filter_with_"
            "target_preserving_mh_rejuvenation"
        ),
        "structural_kernel_family": "fixed_cardinality_surface_position_strength_mh",
        "structural_kernel_target_preserving": True,
        "structural_kernel_exact_rj": False,
        "reversible_jump_mcmc_used": False,
        "structural_transition_provenance": {
            "posterior_semantics": (
                "fixed_cardinality_sequential_particle_filter_with_"
                "target_preserving_mh_rejuvenation"
            ),
            "structural_kernel_family": (
                "fixed_cardinality_surface_position_strength_mh"
            ),
            "structural_moves_enabled": True,
            "structural_kernel_target_preserving": True,
            "structural_kernel_exact_rj": False,
            "reversible_jump_mcmc_used": False,
            "structural_evidence_uses_pf_likelihood": True,
            "support_domain": "environment_surface",
            "variable_cardinality": False,
            "birth_death_moves_enabled": False,
            "within_cardinality_moves_enabled": True,
            "within_cardinality_kernel_exact_mh": True,
        },
        "planner_belief_sources": ["pf_posterior"],
        "repository_commit": "a" * 40,
        "measurement_log_schema_version": 1,
        "measurement_log_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
        "random_seed": 7,
        "profile_capability_map": capabilities.to_dict(),
        "provenance": {"estimator_commit": "a" * 40},
        "isotopes": {},
    }
    estimator = SimpleNamespace(
        profile_capabilities=capabilities,
        pf_config=SimpleNamespace(estimator_profile="pf_strict"),
        posterior_snapshot=lambda: SimpleNamespace(to_dict=lambda: dict(payload)),
    )

    summary = _pure_pf_summary_provenance(estimator)

    assert summary["final_estimate_source"] == "pf_posterior"
    assert summary["pf_posterior"]["pure_pf_schema_version"] == 1
    assert summary["measurement_log_sha256"] == "b" * 64
    assert summary["resolved_config_sha256"] == "d" * 64
    assert summary["structural_kernel_target_preserving"] is True
    assert summary["structural_kernel_exact_rj"] is False
    assert summary["pf_posterior"] == payload


def test_dss_one_step_guard_uses_ranked_node_diagnostics() -> None:
    """DSS one-step guard should reuse already computed first-step scores."""
    diagnostics = {
        "ranked_nodes": [
            {"pose_index": 5, "pose_xyz": [9.0, 8.0, 0.5], "score": 12.0},
            {"pose_index": 2, "pose_xyz": [3.0, 2.0, 0.5], "score": 11.0},
        ],
    }

    pose_index, score, pose_xyz = _best_dss_first_step_guard_candidate(
        diagnostics,
        candidate_poses_xyz=np.zeros((2, 3), dtype=float),
    )

    assert pose_index == 5
    assert score == pytest.approx(12.0)
    np.testing.assert_allclose(pose_xyz, np.array([9.0, 8.0, 0.5]))


def test_candidate_isotope_config_restricts_pf_labels() -> None:
    """Runtime candidate-isotope config should restrict online PF labels."""
    isotopes = _resolve_candidate_isotopes(
        {"candidate_isotopes": ["Cs-137"]},
        ["Cs-137", "Co-60", "Eu-154"],
    )

    assert isotopes == ("Cs-137",)


def test_candidate_isotope_config_rejects_unknown_labels() -> None:
    """Runtime candidate-isotope config should fail on unknown labels."""
    with pytest.raises(ValueError, match="candidate_isotopes contains"):
        _resolve_candidate_isotopes(
            {"candidate_isotopes": ["Unknown"]},
            ["Cs-137", "Co-60"],
        )


def test_particle_surface_diagnostics_use_all_posterior_sources() -> None:
    """Final surface diagnostics should cover every PF posterior source slot."""
    isotope = "Cs-137"
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=2,
            variable_cardinality=False,
            init_num_sources=(2, 2),
            use_gpu=False,
            position_max=(4.0, 4.0, 3.0),
            structural_rj_patch_spacing_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    patches = filt._structural_rj_surface_patches
    assert patches is not None
    patch_kinds = np.asarray(patches.kinds, dtype=object)
    bottom_center = patches.centers_xyz[
        int(np.flatnonzero(patch_kinds == "obstacle_bottom")[0])
    ]
    state = IsotopeState(
        num_sources=2,
        positions=np.asarray([bottom_center, [2.0, 2.0, 1.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=0.0)
    ]

    diagnostics = _particle_surface_diagnostics(
        estimator,
        env,
        obstacle_grid,
        obstacle_height_m=2.0,
    )[isotope]

    assert diagnostics["posterior_source_slots"] == 2
    assert diagnostics["surface_counts"]["obstacle_bottom"] == 1
    assert diagnostics["off_surface_count"] == 1
    assert diagnostics["weighted_surface_mass"]["obstacle_bottom"] == 1.0
    assert diagnostics["weighted_surface_mass"]["off_surface"] == 1.0


def test_intermediate_estimate_trace_reports_position_and_strength_error() -> None:
    """Intermediate estimate traces should expose source and strength accuracy."""
    isotope = "Cs-137"
    env = EnvironmentConfig(size_x=10.0, size_y=10.0, size_z=10.0)
    frame = PFFrame(
        step_index=7,
        time=123.0,
        robot_position=np.array([1.0, 2.0, 0.5], dtype=float),
        robot_orientation=None,
        RFe=np.eye(3),
        RPb=np.eye(3),
        duration=30.0,
        counts_by_isotope={isotope: 1000.0},
        particle_positions={isotope: np.zeros((0, 3), dtype=float)},
        particle_weights={isotope: np.zeros(0, dtype=float)},
        estimated_sources={
            isotope: np.array(
                [[0.0, 1.0, 1.0], [5.0, 5.0, 10.0]],
                dtype=float,
            )
        },
        estimated_strengths={isotope: np.array([12.0, 20.0], dtype=float)},
    )
    true_sources = {
        isotope: np.array(
            [[0.0, 1.0, 1.0], [5.0, 5.0, 10.0], [9.0, 9.0, 0.0]],
            dtype=float,
        )
    }
    true_strengths = {isotope: [10.0, 20.0, 30.0]}

    payload = _build_intermediate_estimate_trace_payload(
        frame,
        true_sources,
        true_strengths,
        env,
        None,
        obstacle_height_m=2.0,
        match_radius_m=0.5,
    )
    summary = payload["isotopes"][isotope]
    records = payload["estimates"]
    truth_records = payload["truth_sources"]
    line = _format_estimate_trace_log_line(
        7,
        isotope,
        summary,
        records,
    )
    truth_line = _format_truth_coverage_log_line(
        7,
        isotope,
        summary,
        truth_records,
    )

    assert summary["estimate_count"] == 2
    assert summary["truth_count"] == 3
    assert summary["source_count_error"] == -1
    assert summary["unmatched_truth_count"] == 1
    assert summary["truth_covered_count"] == 2
    assert summary["truth_uncovered_count"] == 1
    assert summary["total_est_strength"] == pytest.approx(32.0)
    assert summary["total_truth_strength"] == pytest.approx(60.0)
    assert records[0]["position_error_m"] == pytest.approx(0.0)
    assert records[0]["strength_rel_error"] == pytest.approx(0.2)
    assert records[0]["surface_kind"] == "wall"
    assert records[1]["surface_kind"] == "ceiling"
    assert truth_records[2]["covered"] is False
    assert truth_records[2]["nearest_estimate_distance_m"] == pytest.approx(
        np.sqrt(132.0)
    )
    assert "q=12.0" in line
    assert "source_count_error=-1" in line
    assert "pf_truth_coverage[Cs-137]" in truth_line
    assert "covered=2/3" in truth_line


def test_pf_obstacle_attenuation_config_defaults_to_fidelity_path() -> None:
    """PF obstacle attenuation should stay enabled unless explicitly ablated."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )

    assert _pf_obstacle_attenuation_enabled({}) is True
    assert _pf_obstacle_attenuation_enabled({"pf_obstacle_attenuation": None}) is True
    assert _pf_obstacle_grid_for_runtime(grid, {}) is grid
    assert (
        _pf_obstacle_grid_for_runtime(
            grid,
            {"pf_obstacle_attenuation": False},
        )
        is None
    )
    assert (
        _pf_obstacle_grid_for_runtime(
            grid,
            {"pf_obstacle_attenuation": "off"},
        )
        is None
    )


def test_robot_path_segment_uses_obstacle_aware_grid_path() -> None:
    """Robot travel timing should use an obstacle-aware path when a grid is available."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1)),
    )

    segment = _build_robot_path_segment(
        map_api=grid,
        from_pose_xyz=np.array([0.5, 0.5, 0.0], dtype=float),
        to_pose_xyz=np.array([4.5, 0.5, 0.0], dtype=float),
        nominal_motion_speed_m_s=1.0,
        path_planner="dss_pp",
        planned_shield_program=(0, 1),
        dss_diagnostics={"score": 1.0},
    )

    assert segment["obstacle_aware"] is True
    assert segment["euclidean_distance_m"] == pytest.approx(4.0)
    assert segment["distance_m"] > 4.0
    assert segment["travel_time_s"] == pytest.approx(segment["distance_m"])
    waypoints = np.asarray(segment["waypoints_xyz"], dtype=float)
    assert waypoints.ndim == 2
    assert np.max(waypoints[:, 1]) > 2.0


def test_full_simulation_cli_requests_cui_matplotlib_backend() -> None:
    """Full-simulation aliases should force a non-GUI Matplotlib backend."""
    assert _argv_requests_cui(["--full-simulation"]) is True
    assert _argv_requests_cui(["--standard-geant4-full"]) is True


def test_python_worker_auto_uses_all_logical_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python planning worker auto mode should not be capped below CPU count."""
    monkeypatch.setattr("realtime_demo.os.cpu_count", lambda: 32)

    assert _resolve_python_worker_count(0) == 32
    assert _resolve_python_worker_count(None) == 32
    assert _resolve_ig_workers(0) == 32
    assert _resolve_ig_workers(12) == 12


def test_runtime_config_can_disable_gpu_without_cuda_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime configs should be able to force CPU-only execution."""

    def _fail_cuda_probe() -> bool:
        """Fail if automatic CUDA detection is unexpectedly reached."""
        raise AssertionError("CUDA auto-detection should not run.")

    monkeypatch.setattr("realtime_demo._default_use_gpu", _fail_cuda_probe)

    assert _resolve_runtime_use_gpu({"use_gpu": False}) is False
    assert _resolve_runtime_use_gpu({"use_gpu": "off"}) is False
    assert _resolve_runtime_use_gpu({"use_gpu": 1}) is True


def test_runtime_gpu_default_uses_cuda_auto_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing use_gpu should preserve the historical auto-detect behavior."""
    monkeypatch.setattr("realtime_demo._default_use_gpu", lambda: True)

    assert _resolve_runtime_use_gpu({}) is True


def test_cui_split_view_defaults_to_saved_runs() -> None:
    """Saved runs should expose the URL-served CUI progress view by default."""
    assert _resolve_cui_split_view_enabled({}, save_outputs=True) is True
    assert _resolve_cui_split_view_enabled({}, save_outputs=False) is False
    assert (
        _resolve_cui_split_view_enabled(
            {"cui_split_view": False},
            save_outputs=True,
        )
        is False
    )
    assert (
        _resolve_cui_split_view_enabled(
            {"cui_split_view": True},
            save_outputs=False,
        )
        is True
    )


def test_diagnostic_detail_limit_uses_zero_as_no_details() -> None:
    """High-detail diagnostic limits should avoid accidental full log dumps."""
    order = np.array([4, 2, 0, 1, 3], dtype=int)

    assert _diagnostic_detail_order(order, 0).tolist() == []
    assert _diagnostic_detail_order(order, 2).tolist() == [4, 2]
    assert _diagnostic_detail_order(order, -1).tolist() == [4, 2, 0, 1, 3]


def test_source_event_diagnostics_bound_details_and_summarize_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source-event logs should retain complete counts without dumping particles."""
    events = [
        {
            "event": "rj_birth_accepted",
            "reason": "metropolis_hastings",
            "position": [float(index), 0.0, 0.0],
        }
        for index in range(3)
    ]
    estimator = SimpleNamespace(
        filters={
            "Cs-137": SimpleNamespace(last_source_event_diagnostics=events),
        }
    )

    realtime_demo_module._log_source_event_diagnostics(
        estimator,
        {"Cs-137": np.zeros((0, 3), dtype=float)},
        {"Cs-137": np.zeros(0, dtype=float)},
        step_index=9,
        event_log_limit=1,
    )

    output = capsys.readouterr().out
    assert "source_event_summary[Cs-137] total=3 logged=1 omitted=2" in output
    assert 'event_counts={"rj_birth_accepted": 3}' in output
    assert 'reason_counts={"metropolis_hastings": 3}' in output
    assert "raw_sha256=" in output
    assert output.count("source_event[Cs-137]") == 1
    assert "event_idx=0" in output
    assert "event_idx=1" not in output


def test_pf_timing_formatter_keeps_counters_unitless() -> None:
    """PF timing logs should not print diagnostic counters as seconds."""
    assert _format_pf_timing_item("total", 1.25) == "total=1.250s"
    assert (
        _format_pf_timing_item("rj_birth_attempted", 1.0)
        == "rj_birth_attempted=1"
    )
    assert (
        _format_pf_timing_item("rj_birth_accepted", 1.0)
        == "rj_birth_accepted=1"
    )


def test_precision_diagnostics_use_compact_spectrum_log_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime precision diagnostics should avoid full response JSON by default."""
    calls: list[str] = []

    def noop(*args: object, **kwargs: object) -> None:
        """Replace unrelated diagnostics during this routing test."""
        _ = (args, kwargs)

    def record_compact(*args: object, **kwargs: object) -> None:
        """Record compact spectrum-channel diagnostic routing."""
        _ = (args, kwargs)
        calls.append("compact")

    def record_full(*args: object, **kwargs: object) -> None:
        """Record full response-Poisson diagnostic routing."""
        _ = (args, kwargs)
        calls.append("full")

    for name in (
        "_log_current_map_prediction_residuals",
        "_log_truth_observability_diagnostics",
        "_log_posterior_truth_mass_diagnostics",
        "_log_particle_cloud_diagnostics",
        "_log_source_event_diagnostics",
    ):
        monkeypatch.setattr(realtime_demo_module, name, noop)
    monkeypatch.setattr(
        realtime_demo_module,
        "_log_spectrum_isotope_channel_diagnostics",
        record_compact,
    )
    monkeypatch.setattr(
        realtime_demo_module,
        "_log_spectrum_response_poisson_diagnostics",
        record_full,
    )

    _log_precision_degradation_diagnostics(
        object(),
        object(),
        None,
        {},
        {},
        EnvironmentConfig(),
        None,
        obstacle_height_m=2.0,
        step_index=0,
        particle_log_limit=0,
    )
    _log_precision_degradation_diagnostics(
        object(),
        object(),
        None,
        {},
        {},
        EnvironmentConfig(),
        None,
        obstacle_height_m=2.0,
        step_index=0,
        particle_log_limit=0,
        full_spectrum_response_diagnostics_enabled=True,
    )

    assert calls == ["compact", "full"]


def test_precision_diagnostics_route_source_event_log_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime diagnostics should route the configured source-event detail cap."""
    routed_limits: list[int] = []

    def noop(*args: object, **kwargs: object) -> None:
        """Replace unrelated diagnostic callbacks during this routing test."""
        _ = (args, kwargs)

    def record_source_events(*args: object, **kwargs: object) -> None:
        """Record the detail cap passed to source-event diagnostics."""
        _ = args
        routed_limits.append(int(kwargs["event_log_limit"]))

    for name in (
        "_log_spectrum_response_poisson_diagnostics",
        "_log_current_map_prediction_residuals",
        "_log_truth_observability_diagnostics",
        "_log_posterior_truth_mass_diagnostics",
        "_log_particle_cloud_diagnostics",
    ):
        monkeypatch.setattr(realtime_demo_module, name, noop)
    monkeypatch.setattr(
        realtime_demo_module,
        "_log_source_event_diagnostics",
        record_source_events,
    )

    _log_precision_degradation_diagnostics(
        object(),
        object(),
        None,
        {},
        {},
        EnvironmentConfig(),
        None,
        obstacle_height_m=2.0,
        step_index=0,
        particle_log_limit=0,
        source_event_log_limit=7,
    )

    assert routed_limits == [7]


def test_surface_observability_diagnostics_skip_zero_candidates() -> None:
    """Surface observability diagnostics should be fully skipped at zero cap."""

    class _Estimator:
        """Estimator whose observability diagnostic must not be invoked."""

        def surface_candidate_observability_diagnostics(
            self,
            *,
            window: int | None = None,
            max_candidates: int = 256,
        ) -> dict[str, dict[str, object]]:
            """Fail when the zero-candidate guard is not honored."""
            _ = (window, max_candidates)
            raise AssertionError("surface observability should be skipped.")

    _log_surface_candidate_observability_diagnostics(
        _Estimator(),  # type: ignore[arg-type]
        step_index=0,
        label="guard",
        max_candidates=0,
    )


def test_particle_cloud_diagnostics_zero_limit_skips_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Particle cloud diagnostics should skip slot and particle detail at zero limit."""

    class _State:
        """Minimal particle state exposing source cardinality."""

        num_sources = 1

    class _Particle:
        """Minimal continuous particle wrapper."""

        state = _State()
        log_weight = 0.0

    class _Filter:
        """Minimal isotope filter exposing particle weights."""

        continuous_particles = [_Particle(), _Particle()]
        continuous_weights = np.asarray([0.75, 0.25], dtype=float)

    class _Estimator:
        """Minimal estimator exposing one isotope filter."""

        filters = {"Cs-137": _Filter()}

    _log_particle_cloud_diagnostics(
        _Estimator(),  # type: ignore[arg-type]
        {},
        {},
        EnvironmentConfig(),
        None,
        obstacle_height_m=2.0,
        step_index=3,
        particle_log_limit=0,
    )

    output = capsys.readouterr().out
    assert "particle_cloud[Cs-137]" in output
    assert "particle_slot_cloud" not in output
    assert "particle_source" not in output


def test_plot_save_interval_can_disable_intermediate_pf_plots() -> None:
    """PF plot save intervals should allow disabling intermediate figures."""
    assert (
        _resolve_plot_save_interval(
            {"pf_plot_save_every": 0},
            "pf_plot_save_every",
            default=1,
            allow_disable=True,
        )
        == 0
    )
    assert (
        _resolve_plot_save_interval(
            {"pf_plot_save_every": 0},
            "pf_plot_save_every",
            default=1,
            allow_disable=False,
        )
        == 1
    )
    assert (
        _resolve_plot_save_interval(
            {"pf_plot_save_every": "bad"},
            "pf_plot_save_every",
            default=4,
            allow_disable=True,
        )
        == 4
    )


def test_deferred_pf_visualizer_renders_only_on_save() -> None:
    """Deferred visualizer should not create Matplotlib figures during updates."""
    calls: list[tuple[str, object]] = []

    class _DummyVisualizer:
        """Record update and save calls from the deferred wrapper."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Record construction."""
            calls.append(("init", (args, kwargs)))

        def update(self, frame: object) -> None:
            """Record rendered frames."""
            calls.append(("update", frame))

        def save_final(self, path: str) -> None:
            """Record final save calls."""
            calls.append(("save_final", path))

        def save_estimates_only(self, path: str) -> None:
            """Record estimates-only save calls."""
            calls.append(("save_estimates_only", path))

    wrapper = DeferredPFVisualizer(_DummyVisualizer, "arg", option=True)
    wrapper.update("frame-1")
    wrapper.update("frame-2")

    assert calls == []

    wrapper.save_final("out.png")

    assert calls[0][0] == "init"
    assert calls[1] == ("update", "frame-2")
    assert calls[2] == ("save_final", "out.png")


def test_reachable_candidate_filter_removes_disconnected_free_cells() -> None:
    """Pose candidates should be reachable, not merely outside obstacle cells."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 3),
        blocked_cells=((2, 0), (2, 1), (2, 2)),
    )
    candidates = np.array(
        [
            [1.5, 1.5, 0.0],
            [4.5, 1.5, 0.0],
        ],
        dtype=float,
    )

    filtered = _filter_reachable_candidates(
        current_pose_xyz=np.array([0.5, 1.5, 0.0], dtype=float),
        map_api=grid,
        candidates=candidates,
    )

    assert filtered.shape == (1, 3)
    assert filtered[0, 0] == pytest.approx(1.5)


def test_planning_candidate_batches_are_reproducible_from_explicit_rng() -> None:
    """Planning candidates should be replayable without the global PF RNG."""
    kwargs = {
        "current_pose_xyz": np.array([1.0, 1.0, 0.5], dtype=float),
        "map_api": None,
        "n_candidates": 24,
        "min_dist_from_visited": 0.5,
        "visited_poses_xyz": np.array([[1.0, 1.0, 0.5]], dtype=float),
        "bounds_xyz": (
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 2.0], dtype=float),
        ),
    }

    first = realtime_demo_module._generate_planning_candidates(
        **kwargs,
        rng=np.random.default_rng(91),
    )
    second = realtime_demo_module._generate_planning_candidates(
        **kwargs,
        rng=np.random.default_rng(91),
    )

    np.testing.assert_allclose(first[0], second[0], rtol=0.0, atol=0.0)
    assert first[1:] == second[1:]


def test_final_count_bias_preserves_full_isotope_names() -> None:
    """Count-bias grouping must not truncate Co-60 and Cs-137 to one character."""

    class _Estimator:
        """Provide the minimal count-bias reporting interface."""

        filters = {"Co-60": object(), "Cs-137": object()}
        pf_config = SimpleNamespace(background_level=0.0)
        num_orientations = 8
        measurements = [
            SimpleNamespace(z_k={"Co-60": 10.0, "Cs-137": 20.0}),
        ]

        @staticmethod
        def configured_isotope_order() -> list[str]:
            """Return two isotope labels sharing their first character."""
            return ["Co-60", "Cs-137"]

        @staticmethod
        def _measurement_data_for_iso(
            isotope: str,
            window: int | None,
            records: object,
        ) -> MeasurementData:
            """Return one deterministic record for either isotope."""
            del window, records
            count = 10.0 if isotope == "Co-60" else 20.0
            return MeasurementData(
                z_k=np.array([count], dtype=float),
                observation_variances=np.array([count], dtype=float),
                detector_positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                fe_indices=np.array([0], dtype=int),
                pb_indices=np.array([0], dtype=int),
                live_times=np.array([1.0], dtype=float),
                station_sequence_ids=np.array([0], dtype=np.int64),
                runtime_likelihood_routes=np.asarray(
                    ["count"],
                    dtype="<U16",
                ),
            )

    summary = realtime_demo_module._final_count_bias_diagnostics(
        _Estimator(),
        {
            "Co-60": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
            "Cs-137": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
        },
        count_regime_lower_edges=(0.0, 10.0),
    )

    assert set(summary["by_isotope"]) == {"Co-60", "Cs-137"}


def test_candidate_spacing_retry_triggers_for_height_only_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Height-only actions must not suppress the lateral-spacing retry."""
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    requested_distances: list[float] = []

    def _fake_generate_candidate_poses(**kwargs: object) -> np.ndarray:
        """Return a lateral action only after the spacing is relaxed."""
        requested = float(kwargs["min_dist_from_visited"])
        requested_distances.append(requested)
        if requested >= 3.0:
            return np.array([[1.0, 1.0, 1.5]], dtype=float)
        return np.array(
            [[1.0, 1.0, 1.5], [3.0, 1.0, 0.5]],
            dtype=float,
        )

    monkeypatch.setattr(
        realtime_demo_module,
        "generate_candidate_poses",
        _fake_generate_candidate_poses,
    )

    candidates, relaxed, resolved_distance = (
        realtime_demo_module._generate_planning_candidates(
            current_pose_xyz=current,
            map_api=None,
            n_candidates=16,
            min_dist_from_visited=3.0,
            visited_poses_xyz=current.reshape(1, 3),
            bounds_xyz=(
                np.array([0.0, 0.0, 0.5], dtype=float),
                np.array([10.0, 10.0, 2.0], dtype=float),
            ),
            continuous_height_anchor_count=8,
        )
    )

    assert relaxed is True
    assert requested_distances == pytest.approx([3.0, 1.5])
    assert resolved_distance == pytest.approx(1.5)
    assert np.any(np.linalg.norm(candidates[:, :2] - current[:2], axis=1) > 0.0)


def test_candidate_spacing_retry_triggers_when_lateral_actions_are_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One lateral action must not suppress the lateral-spacing retry."""
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    requested_distances: list[float] = []

    def _fake_generate_candidate_poses(**kwargs: object) -> np.ndarray:
        """Return the requested lateral inventory for each spacing."""
        requested = float(kwargs["min_dist_from_visited"])
        requested_distances.append(requested)
        if requested >= 3.0:
            return np.array(
                [[1.0, 1.0, 1.5], [4.0, 1.0, 0.5]],
                dtype=float,
            )
        lateral_x = np.arange(2.0, 11.0, dtype=float)
        return np.column_stack(
            [
                lateral_x,
                np.ones_like(lateral_x),
                np.full_like(lateral_x, 0.5),
            ]
        )

    monkeypatch.setattr(
        realtime_demo_module,
        "generate_candidate_poses",
        _fake_generate_candidate_poses,
    )

    candidates, relaxed, resolved_distance = (
        realtime_demo_module._generate_planning_candidates(
            current_pose_xyz=current,
            map_api=None,
            n_candidates=16,
            min_dist_from_visited=3.0,
            visited_poses_xyz=current.reshape(1, 3),
            bounds_xyz=(
                np.array([0.0, 0.0, 0.5], dtype=float),
                np.array([12.0, 12.0, 2.0], dtype=float),
            ),
            continuous_height_anchor_count=8,
        )
    )

    assert relaxed is True
    assert requested_distances == pytest.approx([3.0, 1.5])
    assert resolved_distance == pytest.approx(1.5)
    assert candidates.shape == (9, 3)


def test_candidate_generation_disables_consecutive_height_partners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A height-partner move must be followed by a lateral station move."""
    current = np.array([1.0, 1.0, 1.5], dtype=float)
    visited = np.array(
        [[1.0, 1.0, 0.5], [1.0, 1.0, 1.5]],
        dtype=float,
    )
    generation_options: list[tuple[bool, bool]] = []

    def _fake_generate_candidate_poses(**kwargs: object) -> np.ndarray:
        """Record whether local height actions are enabled."""
        generation_options.append(
            (
                bool(kwargs["include_current_xy_height_actions"]),
                bool(kwargs["allow_height_partners"]),
            )
        )
        lateral_x = np.arange(2.0, 18.0, dtype=float)
        return np.column_stack(
            [
                lateral_x,
                np.ones_like(lateral_x),
                np.full_like(lateral_x, 1.5),
            ]
        )

    monkeypatch.setattr(
        realtime_demo_module,
        "generate_candidate_poses",
        _fake_generate_candidate_poses,
    )

    candidates, relaxed, resolved_distance = (
        realtime_demo_module._generate_planning_candidates(
            current_pose_xyz=current,
            map_api=None,
            n_candidates=16,
            min_dist_from_visited=3.0,
            visited_poses_xyz=visited,
            bounds_xyz=(
                np.array([0.0, 0.0, 0.5], dtype=float),
                np.array([20.0, 20.0, 2.0], dtype=float),
            ),
            continuous_height_anchor_count=8,
            height_partner_min_z_separation_m=0.25,
        )
    )

    assert relaxed is False
    assert resolved_distance == pytest.approx(3.0)
    assert candidates.shape == (16, 3)
    assert generation_options == [(False, False)]


def test_selected_station_action_fails_fast_on_consecutive_height_move() -> None:
    """The runtime boundary must reject a reintroduced consecutive height move."""
    with pytest.raises(RuntimeError, match="consecutive same-xy height actions"):
        realtime_demo_module._validate_selected_station_action(
            current_pose_xyz=np.array([1.0, 1.0, 1.5], dtype=float),
            next_pose_xyz=np.array([1.0, 1.0, 2.5], dtype=float),
            previous_move_was_height_partner=True,
            xy_tolerance_m=1.0e-9,
            min_z_separation_m=0.25,
        )


def test_final_pf_cardinality_status_uses_only_pf_structural_evidence() -> None:
    """Final cardinality output should summarize only the PF posterior."""

    class _DummyEstimator:
        """Expose one normalized PF cardinality distribution."""

        def posterior_cardinality_distribution(
            self,
        ) -> dict[str, dict[int, float]]:
            """Return deliberately unnormalized posterior mass."""
            return {"Cs-137": {1: 1.0, 2: 3.0}}

    status = _final_pf_cardinality_status(_DummyEstimator())
    cs_status = status["pf_cardinality"]["Cs-137"]

    assert status["source"] == "pf_posterior"
    assert cs_status["distribution"] == {"1": 0.25, "2": 0.75}
    assert cs_status["mean"] == pytest.approx(1.75)
    assert cs_status["variance"] == pytest.approx(0.1875)
    assert cs_status["entropy_nats"] > 0.0


def test_cardinality_dwell_waits_for_diffuse_pf_count_posterior() -> None:
    """Adaptive dwell should wait while PF source-count variance is too large."""

    class _DummyConfig:
        """Set the accepted PF cardinality variance."""

        converge_cardinality_var_max = 0.05

    class _DummyState:
        """Store one particle source count."""

        def __init__(self, num_sources: int) -> None:
            """Initialize the source count."""
            self.num_sources = num_sources

    class _DummyParticle:
        """Wrap a PF state."""

        def __init__(self, num_sources: int) -> None:
            """Initialize the wrapped PF state."""
            self.state = _DummyState(num_sources)

    class _DummyFilter:
        """Expose a posterior split evenly between one and two sources."""

        continuous_particles = [_DummyParticle(1), _DummyParticle(2)]
        continuous_weights = np.asarray([0.5, 0.5], dtype=float)

    class _DummyEstimator:
        """Expose an unresolved PF cardinality posterior."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an available PF estimate."""
            return {}

    ready, reason = _source_cardinality_dwell_status(_DummyEstimator())

    assert ready is False
    assert reason == "pf_cardinality_variance:Cs-137"


def test_adaptive_mission_pf_convergence_waits_for_min_poses() -> None:
    """PF convergence should not stop before the guaranteed pose count."""

    class _DummyEstimator:
        """Minimal estimator state exposing a converged PF."""

        filters: dict[str, object] = {}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a converged global exploration state."""
            return True

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=None,
        min_poses=8,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_pf_cardinality_ready=False,
    )

    assert reason is None

    reason_after_min = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited * 8,
        map_api=None,
        min_poses=8,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_pf_cardinality_ready=False,
    )

    assert reason_after_min == "pf_converged_low_information_gain"


def test_adaptive_mission_coverage_can_stop_when_posterior_is_ready() -> None:
    """Coverage can stop a mission once PF posterior gates are ready."""

    class _DummyFilter:
        """Minimal filter state with a concentrated cardinality posterior."""

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=(),
    )
    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
    )

    assert reason == "environment_coverage:1.000"


def test_adaptive_mission_coverage_can_require_pf_convergence() -> None:
    """Coverage alone should not stop a mission when convergence is required."""

    class _DummyFilter:
        """Minimal filter state with a concentrated cardinality posterior."""

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=(),
    )
    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_pf_convergence_for_coverage=True,
    )

    assert reason is None


def test_demo_pure_pf_updates_all_configured_isotopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain every configured response-Poisson count in the pure PF history."""
    import realtime_demo

    class _DummyViz:
        """Minimal visualizer stub for fast regression testing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize the stub visualizer."""
            return None

        def update(self, frame: object) -> None:
            """Ignore frame updates in tests."""
            return None

        def save_final(self, path: str) -> None:
            """Skip saving final snapshots in tests."""
            return None

        def save_estimates_only(self, path: str) -> None:
            """Skip saving estimate snapshots in tests."""
            return None

    def _fake_update_pair_sequence(
        self: RotatingShieldPFEstimator,
        records: list[tuple[object, ...]],
        *,
        pose_idx: int,
        runtime_likelihood_route_by_isotope: dict[str, str],
        z_view_covariance_by_isotope: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Append one lightweight station sequence without PF particle work."""
        del z_view_covariance_by_isotope
        station_sequence_id = len(self.measurements)
        detector_position = tuple(
            float(value)
            for value in np.asarray(self.poses[pose_idx], dtype=float).reshape(3)
        )
        for station_view_index, raw_record in enumerate(records):
            z_k = dict(raw_record[0])
            fe_index = int(raw_record[1])
            pb_index = int(raw_record[2])
            live_time_s = float(raw_record[3])
            z_variance_k = (
                None if raw_record[4] is None else dict(raw_record[4])
            )
            self.measurements.append(
                MeasurementRecord(
                    z_k={iso: float(value) for iso, value in z_k.items()},
                    pose_idx=pose_idx,
                    live_time_s=live_time_s,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    detector_position_xyz_m=detector_position,
                    station_sequence_id=station_sequence_id,
                    station_view_index=station_view_index,
                    runtime_likelihood_route_by_isotope=dict(
                        runtime_likelihood_route_by_isotope
                    ),
                    z_variance_k=z_variance_k,
                )
            )

    def _fake_estimates(
        self: RotatingShieldPFEstimator,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty surface estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.0]], dtype=float)
        strengths = np.array([1.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in ANALYSIS_ISOTOPES}

    def _fake_sim(
        self: SpectralDecomposer, *args: object, **kwargs: object
    ) -> tuple[np.ndarray, None]:
        """Return a zero spectrum to avoid heavy simulation work."""
        return np.zeros_like(self.energy_axis, dtype=float), None

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic counts and a stable detection set."""
        counts = {iso: 10.0 for iso in ANALYSIS_ISOTOPES}
        self.last_count_variances = {iso: 2.0 for iso in ANALYSIS_ISOTOPES}
        return counts, {"Cs-137"}

    def _fake_ig_grid(
        estimator: RotatingShieldPFEstimator,
        rot_mats: list[np.ndarray],
        *,
        pose_idx: int,
        live_time_s: float,
        planning_isotopes: list[str] | None = None,
    ) -> np.ndarray:
        """Return a zero IG grid to bypass heavy IG evaluation."""
        planning_isotope_args.append(planning_isotopes)
        size = len(rot_mats)
        return np.zeros((size, size), dtype=float)

    def _fake_frame(*args: object, **kwargs: object) -> dict[str, object]:
        """Return an empty frame placeholder."""
        return {}

    def _fake_shield_grid(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Bypass response-heavy shield diagnostics in this loop-wiring test."""
        del args
        scores = np.asarray(kwargs["ig_scores"], dtype=float)
        zeros = np.zeros_like(scores)
        return scores.copy(), {
            "eig": scores.copy(),
            "signature": zeros.copy(),
            "signature_utility": zeros.copy(),
            "low_count_penalty": zeros.copy(),
            "count_balance_penalty": zeros.copy(),
            "rotation_cost": zeros.copy(),
        }

    def _fake_candidate_poses(*args: object, **kwargs: object) -> np.ndarray:
        """Return two deterministic candidate poses."""
        return np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]], dtype=float)

    def _fake_next_pose(*args: object, **kwargs: object) -> int:
        """Select the candidate that requires travel from the initial pose."""
        return 1

    def _fake_gpu_enabled(self: RotatingShieldPFEstimator) -> bool:
        """Pretend GPU is disabled to avoid CUDA checks in tests."""
        return False

    def _fake_runtime_config(_path: object) -> dict[str, object]:
        """Return the minimum schema-v1 physical runtime configuration."""
        return {
            "pure_pf_schema_version": 1,
            "estimator_profile": "pf_strict",
            "source_rate_model": "detector_cps_1m",
            "pf_strength_prior_min_cps_1m": 0.0,
            "pf_strength_prior_max_cps_1m": 2_000_000.0,
        }

    planning_isotope_args: list[list[str] | None] = []
    monkeypatch.setattr(realtime_demo, "RealTimePFVisualizer", _DummyViz)
    monkeypatch.setattr(realtime_demo, "build_frame_from_pf", _fake_frame)
    monkeypatch.setattr(realtime_demo, "_compute_ig_grid", _fake_ig_grid)
    monkeypatch.setattr(
        realtime_demo,
        "_compute_shield_selection_grid",
        _fake_shield_grid,
    )
    monkeypatch.setattr(
        realtime_demo,
        "generate_candidate_poses",
        _fake_candidate_poses,
    )
    monkeypatch.setattr(
        realtime_demo, "select_next_pose_from_candidates", _fake_next_pose
    )
    monkeypatch.setattr(SpectralDecomposer, "simulate_spectrum", _fake_sim)
    monkeypatch.setattr(
        SpectralDecomposer, "isotope_counts_with_detection", _fake_counts
    )
    pure_estimator_type = realtime_demo.PurePFEstimator
    monkeypatch.setattr(
        pure_estimator_type,
        "update_pair_sequence",
        _fake_update_pair_sequence,
    )
    monkeypatch.setattr(pure_estimator_type, "estimates", _fake_estimates)
    monkeypatch.setattr(pure_estimator_type, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(
        realtime_demo,
        "load_runtime_config",
        _fake_runtime_config,
    )

    estimator = run_live_pf(
        live=False,
        max_steps=None,
        max_poses=2,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        min_peaks_by_isotope={"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
        ig_threshold_mode="absolute",
        ig_threshold_min=0.0,
        obstacle_layout_path=None,
        num_particles=8,
        pf_config_overrides={
            "orientation_k": 1,
            "observation_count_variance_semantics": (
                "counting_noise_inclusive"
            ),
        },
        save_outputs=False,
        return_state=True,
        nominal_motion_speed_m_s=1.0,
        rotation_overhead_s=2.0,
        measurement_log_output=str(tmp_path / "measurement-log"),
    )
    assert estimator is not None
    assert (
        estimator.pf_config.observation_count_variance_semantics
        == "counting_noise_inclusive"
    )
    assert len(estimator.measurements) >= 2
    assert len(estimator.poses) >= 2
    metrics = estimator.mission_metrics
    assert metrics["total_measurements"] >= 2
    assert metrics["total_motion_distance_m"] == pytest.approx(np.sqrt(2.0))
    assert metrics["total_travel_time_s"] == pytest.approx(np.sqrt(2.0))
    assert metrics["total_shield_actuation_time_s"] == pytest.approx(
        metrics["total_measurements"] * 2.0
    )
    assert metrics["total_mission_time_s"] == pytest.approx(
        metrics["total_live_time_s"]
        + metrics["total_travel_time_s"]
        + metrics["total_shield_actuation_time_s"]
    )
    assert metrics["estimated_end_to_end_time_s"] == pytest.approx(
        metrics["total_mission_time_s"]
    )
    assert metrics["num_motion_segments"] == 1
    assert len(metrics["path_segments"]) == 1
    assert metrics["path_segments"][0]["travel_time_s"] == pytest.approx(np.sqrt(2.0))
    assert metrics["mean_orientation_selection_time_s"] >= 0.0
    assert metrics["mean_pf_update_time_s"] >= 0.0
    assert metrics["median_pf_update_time_s"] >= 0.0
    assert metrics["p95_pf_update_time_s"] >= 0.0
    assert metrics["station_count"] == 2
    assert metrics["detector_pose_station_count"] == 2
    assert metrics["height_change_count"] == 0
    assert metrics["station_visit_count"] == 2
    assert metrics["unique_xy_station_count"] == 2
    assert metrics["unique_xyz_action_count"] == 2
    assert metrics["height_transition_count"] == 0
    assert metrics["wall_clock_runtime_s"] == pytest.approx(
        metrics["online_wall_clock_s"]
    )
    assert metrics["end_to_end_wall_clock_s"] >= metrics["online_wall_clock_s"]
    assert metrics["final_posterior_projection_time_s"] >= 0.0
    assert metrics["gpu_memory"]["available"] is False
    evaluation = estimator.final_run_summary["evaluation_metrics"]
    assert "p95" in evaluation["accuracy"]["position_error"]
    assert "by_shield_pair" in evaluation["count_bias"]
    assert evaluation["count_bias"]["pf_isotopes_at_evaluation"] == list(
        ANALYSIS_ISOTOPES
    )
    assert "active_pf_isotopes_at_evaluation" not in evaluation["count_bias"]
    assert "inactive_configured_isotopes_scored" not in evaluation["count_bias"]
    assert "consecutive_matched_cluster_shift_m" in evaluation["cluster_stability"]
    assert evaluation["operational"]["station_count"] == 2
    assert evaluation["operational"]["station_visit_count"] == 2
    assert (
        evaluation["operational"]["online_wall_clock_s"]
        <= evaluation["operational"]["end_to_end_wall_clock_s"]
    )
    json.dumps(estimator.final_run_summary, allow_nan=False)
    assert estimator.configured_isotope_order() == tuple(ANALYSIS_ISOTOPES)
    assert estimator.isotopes == list(ANALYSIS_ISOTOPES)
    for rec in estimator.measurements:
        assert set(rec.z_k) == set(ANALYSIS_ISOTOPES)
        assert rec.z_variance_k is not None
        assert set(rec.z_variance_k) == set(ANALYSIS_ISOTOPES)
        assert rec.z_variance_k["Cs-137"] == pytest.approx(2.0)
    assert planning_isotope_args
    assert all(value is None for value in planning_isotope_args)
    estimates = estimator.estimates()
    positions, strengths = estimates.get("Cs-137", (np.zeros((0, 3)), np.zeros(0)))
    assert positions.size > 0
    assert strengths.size > 0


def test_baseline_shield_program_preserves_adapted_dss_length() -> None:
    """Shield ablations should not change the adapted spectra-per-station budget."""
    config = DSSPPConfig(program_length=16, forced_program_pair_ids=None)

    forced_config, baseline_program = _apply_baseline_shield_program_to_dss_config(
        config,
        {"name": "round_robin", "start_pair_id": 0, "advance_by_pose": True},
        total_pairs=64,
        pose_index=2,
        current_pair_id=None,
    )

    assert baseline_program is not None
    assert len(baseline_program.pair_ids) == 16
    assert forced_config.program_length == 16
    assert forced_config.forced_program_pair_ids == baseline_program.pair_ids


def test_shield_selection_uses_signature_floor_and_dependency() -> None:
    """Shield scoring should combine signature gain, count floor, and redundancy."""

    class _DummyConfig:
        """Minimal PF config stub for shield selection scoring."""

        planning_method = "top_weight"
        alpha_weights = None

    class _DummyEstimator:
        """Minimal estimator stub for shield selection scoring."""

        pf_config = _DummyConfig()
        isotopes = ["Cs-137", "Co-60"]

        def planning_particles(self, max_particles=None, method=None):
            """Return an empty planning subset for the dummy score."""
            return {}

        def orientation_signature_separation_score(
            self,
            pose_idx,
            fe_index,
            pb_index,
            *,
            live_time_s,
            particles_by_isotope=None,
            alpha_by_isotope=None,
            variance_floor=1.0,
        ):
            """Return a high signature score for one discriminative pair."""
            return 5.0 if int(fe_index) == 1 and int(pb_index) == 0 else 0.0

        def expected_observation_counts_by_isotope_at_pair(
            self,
            pose_idx,
            fe_index,
            pb_index,
            *,
            live_time_s,
            max_particles=None,
        ):
            """Return low Cs counts for one deliberately bad pair."""
            if int(fe_index) == 0 and int(pb_index) == 1:
                return {"Cs-137": 0.0, "Co-60": 10.0}
            return {"Cs-137": 10.0, "Co-60": 10.0}

    rot_mats = [
        np.eye(3, dtype=float),
        np.diag([1.0, -1.0, -1.0]),
    ]
    ig_scores = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float)

    scores, parts = _compute_shield_selection_grid(
        _DummyEstimator(),
        rot_mats,
        pose_idx=0,
        live_time_s=1.0,
        ig_scores=ig_scores,
        current_pair_id=None,
        min_observation_counts=5.0,
        signature_weight=1.0,
        low_count_penalty_weight=1.0,
        count_balance_weight=0.5,
        rotation_cost_weight=0.0,
        variance_floor=1.0,
        max_particles=None,
    )
    best_pair, best_score = _select_best_pair_from_scores(scores, None)

    assert best_pair == 2
    assert best_score == pytest.approx(scores[1, 0])
    assert parts["signature"][1, 0] == pytest.approx(5.0)
    assert parts["signature_utility"][1, 0] == pytest.approx(np.log1p(5.0))
    assert parts["low_count_penalty"][0, 1] > 0.0
    assert parts["count_balance_penalty"][0, 1] > parts["count_balance_penalty"][1, 0]
    assert _signature_vector_is_dependent(
        np.array([2.0, 2.0]),
        [np.array([1.0, 1.0])],
        cosine_threshold=0.99,
    )


def test_isotope_count_balance_penalty_is_not_nuclide_specific() -> None:
    """Dominance by any isotope should receive the same balance penalty."""
    balanced = {"Cs-137": 10.0, "Co-60": 10.0, "Eu-154": 10.0}
    co_dominated = {"Cs-137": 1.0, "Co-60": 98.0, "Eu-154": 1.0}
    cs_dominated = {"Cs-137": 98.0, "Co-60": 1.0, "Eu-154": 1.0}

    assert _isotope_count_balance_penalty(balanced) == pytest.approx(0.0)
    assert _isotope_count_balance_penalty(co_dominated) == pytest.approx(
        _isotope_count_balance_penalty(cs_dominated)
    )
    assert _isotope_count_balance_penalty(co_dominated) > 0.5


def test_spectrum_runtime_config_exposes_response_poisson_controls() -> None:
    """Runtime configs should be able to tune response-Poisson decomposition."""
    config = _spectrum_config_from_runtime_config(
        {
            "response_poisson_photopeak_anchor": False,
            "response_poisson_photopeak_anchor_weight": 0.5,
            "response_poisson_low_snr_suppress_count": False,
            "response_poisson_model_mismatch_variance_scale": 2.0,
            "response_poisson_crosstalk_corr_threshold": 0.9,
            "response_poisson_underallocation_count_guard_ratio": 1.1,
            "response_poisson_diagnostic_reduced_chi2_threshold": 3.0,
            "dead_time_tau_s": 0.0,
        }
    )

    assert config.response_poisson_photopeak_anchor is False
    assert config.response_poisson_photopeak_anchor_weight == pytest.approx(0.5)
    assert config.response_poisson_low_snr_suppress_count is False
    assert config.response_poisson_model_mismatch_variance_scale == pytest.approx(2.0)
    assert config.response_poisson_crosstalk_corr_threshold == pytest.approx(0.9)
    assert config.response_poisson_underallocation_count_guard_ratio == pytest.approx(
        1.1
    )
    assert config.response_poisson_diagnostic_reduced_chi2_threshold == pytest.approx(
        3.0
    )
    assert config.dead_time_tau_s == pytest.approx(0.0)


def test_spectrum_runtime_config_uses_geant4_background_cps() -> None:
    """Geant4 executable background cps should anchor response-Poisson background."""
    config = _spectrum_config_from_runtime_config(
        {
            "detector_scoring_mode": "incident_gamma_energy",
            "source_rate_model": "detector_cps_1m",
            "executable_args": ["--background-cps", "12.0"],
        }
    )

    assert config.response_poisson_background_rate_cps == pytest.approx(12.0)
    assert config.response_efficiency_model == "unit"
    assert config.use_incident_gamma_response_matrix is True
    assert config.normalize_line_intensities is True


def test_incident_gamma_runtime_uses_detector_response_folding() -> None:
    """Incident-energy spectra should be folded with detector response before unfolding."""
    config = _spectrum_config_from_runtime_config(
        {"detector_scoring_mode": "incident_gamma_energy"}
    )

    assert config.response_continuum_to_peak == pytest.approx(2.0)
    assert config.response_backscatter_fraction == pytest.approx(0.03)
    assert config.response_efficiency_model == "unit"
    assert config.apply_incident_gamma_detector_response is True


def test_adaptive_dwell_chunks_stop_at_ready_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive dwell should stop after accumulated isotope counts are usable."""
    decomposer = SpectralDecomposer()
    commands: list[SimulationCommand] = []

    class _FakeRuntime:
        """Return deterministic spectra proportional to requested dwell time."""

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Record the command and return one non-zero spectrum bin."""
            commands.append(command)
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            spectrum = np.zeros_like(energy, dtype=float)
            spectrum[0] = float(command.dwell_time_s) * 60.0
            spectrum_variance = np.zeros_like(energy, dtype=float)
            spectrum_variance[0] = float(command.dwell_time_s) * 25.0
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=spectrum.tolist(),
                energy_bin_edges_keV=np.concatenate(
                    [energy, [energy[-1] + step]]
                ).tolist(),
                metadata={
                    "backend": "fake",
                    "weighted_transport": True,
                    "num_primaries": float(command.dwell_time_s) * 10.0,
                    "run_time_s": float(command.dwell_time_s) * 0.5,
                    "source_equivalent_counts_Cs-137": float(command.dwell_time_s)
                    * 30.0,
                    "transport_detected_counts_Cs-137": float(command.dwell_time_s)
                    * 40.0,
                    "spectrum_count_variance": spectrum_variance.tolist(),
                },
            )

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return a Cs-137 count without relying on detection gating."""
        count = float(np.sum(spectrum))
        return {"Cs-137": count}, set()

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    def _fake_variance_floor(
        self: SpectralDecomposer,
        spectrum_variance: np.ndarray,
        *,
        isotopes: list[str],
    ) -> dict[str, float]:
        """Return a deterministic weighted-MC variance floor for the test."""
        assert float(np.sum(spectrum_variance)) > 0.0
        return {"Cs-137": 1000.0}

    monkeypatch.setattr(
        SpectralDecomposer,
        "estimate_count_variances_from_spectrum_variance",
        _fake_variance_floor,
    )
    observation, actual_live, counts, variances, detected, reason, chunks = (
        _acquire_spectrum_observation(
            simulation_runtime=_FakeRuntime(),
            decomposer=decomposer,
            step_id=7,
            pose_xyz=np.array([1.0, 2.0, 0.5], dtype=float),
            fe_idx=3,
            pb_idx=4,
            live_time_s=30.0,
            travel_time_s=5.0,
            shield_actuation_time_s=2.0,
            adaptive_dwell=True,
            adaptive_dwell_chunk_s=2.0,
            adaptive_min_dwell_s=2.0,
            adaptive_ready_min_counts=200.0,
            adaptive_ready_min_isotopes=1,
            adaptive_ready_min_snr=0.0,
            spectrum_count_method="response_poisson",
            detect_threshold_abs=0.0,
            detect_threshold_rel=0.0,
            detect_threshold_rel_by_isotope={},
            min_peaks_by_isotope=None,
            travel_waypoints_xyz=(
                (1.0, 2.0, 0.5),
                (1.5, 2.5, 0.5),
            ),
        )
    )

    assert actual_live == pytest.approx(4.0)
    assert counts["Cs-137"] == pytest.approx(240.0)
    assert variances["Cs-137"] == pytest.approx(1000.0)
    assert detected == set()
    assert reason == "isotope_count_estimates_ready"
    assert chunks == 2
    assert observation.step_id == 7
    assert observation.metadata["adaptive_dwell_chunks"] == 2
    assert "adaptive_dwell_count_variance_by_isotope" in observation.metadata
    assert observation.metadata["spectrum_count_variance_total"] > 0.0
    assert observation.metadata["num_primaries"] == pytest.approx(40.0)
    assert observation.metadata["run_time_s"] == pytest.approx(2.0)
    assert observation.metadata["primaries_per_sec"] == pytest.approx(20.0)
    assert observation.metadata["source_equivalent_counts_Cs-137"] == pytest.approx(
        120.0
    )
    assert observation.metadata["transport_detected_counts_Cs-137"] == pytest.approx(
        160.0
    )
    assert commands[0].step_id == 7 * ADAPTIVE_STEP_ID_STRIDE
    assert commands[1].step_id == 7 * ADAPTIVE_STEP_ID_STRIDE + 1
    assert commands[0].travel_time_s == pytest.approx(5.0)
    assert commands[1].travel_time_s == pytest.approx(0.0)
    assert commands[0].shield_actuation_time_s == pytest.approx(2.0)
    assert commands[1].shield_actuation_time_s == pytest.approx(0.0)
    assert commands[0].travel_waypoints_xyz == (
        (1.0, 2.0, 0.5),
        (1.5, 2.5, 0.5),
    )
    assert commands[1].travel_waypoints_xyz is None


def test_adaptive_chunk_metadata_aggregates_dynamic_transport_provenance() -> None:
    """Merged adaptive observations must not expose the final chunk as aggregate."""
    energy_edges = [0.0, 1.0, 2.0]

    def _observation(
        *,
        step_id: int,
        dwell_time_s: float,
        spectrum: list[float],
        spectrum_variance: list[float],
        expected_unthinned: float,
        expected_sampled: float,
        sampling_fraction: float,
        actual_primaries: int,
        dead_time_scale: float,
    ) -> SimulationObservation:
        """Build one native-like dynamic-budget transport chunk."""
        return SimulationObservation(
            step_id=step_id,
            detector_pose_xyz=(1.0, 2.0, 0.5),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=0,
            pb_orientation_index=1,
            spectrum_counts=spectrum,
            energy_bin_edges_keV=energy_edges,
            metadata={
                "dwell_time_s": dwell_time_s,
                "num_primaries": actual_primaries,
                "expected_detector_equivalent_primaries": expected_unthinned,
                "expected_unthinned_primaries": expected_unthinned,
                "expected_sampled_primaries": expected_sampled,
                "primary_sampling_fraction": sampling_fraction,
                "primary_history_weight": 1.0 / sampling_fraction,
                "requested_primary_sampling_fraction": 1.0,
                "target_sampled_primaries": 1_500_000,
                "primary_sampling_budget_enabled": True,
                "primary_sampling_fraction_resolution": (
                    "target_budget_limited"
                    if sampling_fraction < 1.0
                    else "maximum_fraction_limited"
                ),
                "history_thinning_enabled": sampling_fraction < 1.0,
                "transport_history_mode": (
                    "weighted_thinning"
                    if sampling_fraction < 1.0
                    else "full_unit_weight"
                ),
                "dead_time_observed_scale": dead_time_scale,
                "spectrum_count_variance": spectrum_variance,
                "spectrum_variance_semantics": (
                    "compound_poisson_sumw2_includes_counting"
                ),
                "spectrum_variance_dead_time_propagation": ("fixed_observed_scale"),
                "weighted_spectrum_sumw2": float(sum(spectrum_variance)),
                "run_time_s": 7.5,
                "total_spectrum_counts": float(sum(spectrum)),
            },
        )

    observations = [
        _observation(
            step_id=100,
            dwell_time_s=2.0,
            spectrum=[10.0, 20.0],
            spectrum_variance=[12.0, 24.0],
            expected_unthinned=2_000_000.0,
            expected_sampled=1_500_000.0,
            sampling_fraction=0.75,
            actual_primaries=1_499_000,
            dead_time_scale=0.98,
        ),
        _observation(
            step_id=101,
            dwell_time_s=1.0,
            spectrum=[5.0, 8.0],
            spectrum_variance=[5.0, 8.0],
            expected_unthinned=1_000_000.0,
            expected_sampled=1_000_000.0,
            sampling_fraction=1.0,
            actual_primaries=1_001_000,
            dead_time_scale=0.99,
        ),
    ]

    merged = realtime_demo_module._merge_adaptive_observation_chunks(
        logical_step_id=1,
        observations=observations,
        chunk_live_times_s=[2.0, 1.0],
        ready_reason="isotope_count_estimates_ready",
        counts_by_isotope={"Cs-137": 10.0},
        count_variance_by_isotope={"Cs-137": 4.0},
        detected_isotopes={"Cs-137"},
    )
    metadata = merged.metadata

    assert metadata["dwell_time_s"] == pytest.approx(3.0)
    assert metadata["expected_unthinned_primaries"] == pytest.approx(3_000_000.0)
    assert metadata["expected_sampled_primaries"] == pytest.approx(2_500_000.0)
    assert metadata["num_primaries"] == pytest.approx(2_500_000.0)
    assert metadata["primary_sampling_fraction"] == pytest.approx(5.0 / 6.0)
    assert metadata["primary_history_weight"] == pytest.approx(1.2)
    assert metadata["primary_sampling_fraction_resolution"] == (
        "adaptive_chunk_aggregate"
    )
    assert metadata["adaptive_dwell_chunk_primary_sampling_fractions"] == [
        0.75,
        1.0,
    ]
    assert metadata["adaptive_dwell_chunk_primary_history_weights"] == pytest.approx(
        [4.0 / 3.0, 1.0]
    )
    assert metadata["adaptive_dwell_target_sampled_primaries_semantics"] == (
        "per_geant4_transport_invocation_not_per_logical_observation"
    )
    assert "dead_time_observed_scale" not in metadata
    assert metadata["adaptive_dwell_dead_time_observed_scales"] == pytest.approx(
        [0.98, 0.99]
    )
    assert metadata["spectrum_variance_dead_time_propagation"] == (
        "independent_chunk_factored_dead_time_jacobians"
    )
    provenance = metadata["adaptive_dwell_transport_chunk_provenance"]
    assert len(provenance) == 2
    assert provenance[0]["step_id"] == 100
    assert provenance[0]["expected_unthinned_primaries"] == pytest.approx(2_000_000.0)
    assert provenance[0]["primary_sampling_fraction"] == pytest.approx(0.75)
    assert provenance[1]["step_id"] == 101
    assert provenance[1]["primary_history_weight"] == pytest.approx(1.0)

    logged_provenance = _measurement_transport_provenance(metadata)
    assert logged_provenance["adaptive_dwell_transport_chunk_provenance"] == (
        provenance
    )
    assert logged_provenance["adaptive_dwell_effective_primary_sampling_fraction"] == (
        pytest.approx(5.0 / 6.0)
    )


def test_adaptive_transport_decomposition_requires_every_chunk() -> None:
    """Adaptive transport decomposition totals must cover every child chunk."""
    complete_values = {
        "transport_detected_counts_Cs-137": (1.0, 2.0),
        "transport_uncollided_primary_counts_src0_Cs-137": (3.0, 4.0),
        "transport_interacted_primary_counts_src0_Cs-137_661p657keV": (
            5.0,
            6.0,
        ),
        "transport_secondary_counts_src0_Cs-137_661p657keV": (7.0, 8.0),
    }
    incomplete_keys = (
        "transport_detected_counts_missing_isotope",
        "transport_uncollided_primary_counts_missing_source",
        "transport_interacted_primary_counts_missing_line",
        "transport_secondary_counts_missing_line",
    )
    observations: list[SimulationObservation] = []
    for chunk_index in range(2):
        metadata = {key: values[chunk_index] for key, values in complete_values.items()}
        if chunk_index == 1:
            metadata.update({key: 100.0 for key in incomplete_keys})
        observations.append(
            SimulationObservation(
                step_id=200 + chunk_index,
                detector_pose_xyz=(1.0, 2.0, 0.5),
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=0,
                pb_orientation_index=0,
                spectrum_counts=[1.0, 2.0],
                energy_bin_edges_keV=[0.0, 1.0, 2.0],
                metadata=metadata,
            )
        )

    merged = realtime_demo_module._merge_adaptive_observation_chunks(
        logical_step_id=2,
        observations=observations,
        chunk_live_times_s=[1.0, 1.0],
        ready_reason="isotope_count_estimates_ready",
        counts_by_isotope={},
        count_variance_by_isotope={},
        detected_isotopes=set(),
    )

    for key, values in complete_values.items():
        assert merged.metadata[key] == pytest.approx(sum(values))
    for key in incomplete_keys:
        assert key not in merged.metadata


def test_adaptive_dwell_preserves_native_covariance_contract_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive evaluation should pass raw transport only for one native chunk."""
    config = _spectrum_config_from_runtime_config(
        {"detector_scoring_mode": "incident_gamma_energy"}
    )
    decomposer = SpectralDecomposer(config)
    calls: list[
        tuple[
            dict[str, object],
            np.ndarray | None,
            tuple[object, ...],
        ]
    ] = []

    class _FakeRuntime:
        """Return native-like incident-energy chunks with count variance."""

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return one deterministic weighted incident-energy tally."""
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            spectrum = np.zeros_like(energy, dtype=float)
            spectrum[100] = 60.0
            variance = np.zeros_like(energy, dtype=float)
            variance[100] = 300.0
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=spectrum.tolist(),
                energy_bin_edges_keV=np.concatenate(
                    [energy, [energy[-1] + step]]
                ).tolist(),
                metadata={
                    "backend": "geant4",
                    "detector_scoring_mode": "incident_gamma_energy",
                    "detector_fast_scoring": True,
                    "spectrum_count_variance": variance.tolist(),
                    "spectrum_variance_semantics": (
                        "compound_poisson_sumw2_includes_counting"
                    ),
                    "spectrum_variance_dead_time_propagation": ("fixed_observed_scale"),
                    "dead_time_observed_scale": 1.0,
                    "dead_time_tau_s": 0.0,
                    "dwell_time_s": float(command.dwell_time_s),
                },
            )

    def _fake_evaluate(
        _decomposer: SpectralDecomposer,
        _spectrum: np.ndarray,
        **kwargs: object,
    ) -> RuntimeCountResult:
        """Capture covariance inputs and become ready on the second chunk."""
        metadata = dict(kwargs["transport_metadata"])
        transport = kwargs.get("transport_spectrum")
        chunks = tuple(kwargs.get("transport_covariance_chunks", ()))
        calls.append(
            (
                metadata,
                None if transport is None else np.asarray(transport, dtype=float),
                chunks,
            )
        )
        count = 60.0 * len(calls)
        return RuntimeCountResult(
            counts={"Cs-137": count},
            variances={"Cs-137": 100.0},
            detected={"Cs-137"},
            covariance={"Cs-137": {"Cs-137": 100.0}},
        )

    monkeypatch.setattr(
        realtime_demo_module,
        "_evaluate_spectrum_count_result",
        _fake_evaluate,
    )
    result = _acquire_spectrum_observation(
        simulation_runtime=_FakeRuntime(),
        decomposer=decomposer,
        step_id=11,
        pose_xyz=np.array([1.0, 2.0, 0.5], dtype=float),
        fe_idx=0,
        pb_idx=0,
        live_time_s=4.0,
        travel_time_s=0.0,
        shield_actuation_time_s=0.0,
        adaptive_dwell=True,
        adaptive_dwell_chunk_s=2.0,
        adaptive_min_dwell_s=2.0,
        adaptive_ready_min_counts=100.0,
        adaptive_ready_min_isotopes=1,
        adaptive_ready_min_snr=0.0,
        spectrum_count_method="response_poisson",
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
    )

    assert result[-1] == 2
    assert len(calls) == 2
    assert calls[0][0]["spectrum_variance_dead_time_propagation"] == (
        "fixed_observed_scale"
    )
    assert calls[0][1] is not None
    assert calls[0][2] == ()
    assert calls[1][0]["spectrum_variance_dead_time_propagation"] == (
        "fixed_observed_scale"
    )
    assert calls[1][1] is None
    assert len(calls[1][2]) == 2
    assert result[0].metadata["spectrum_variance_dead_time_propagation"] == (
        "independent_chunk_factored_dead_time_jacobians"
    )


@pytest.mark.parametrize("native_first", (False, True))
def test_adaptive_dwell_rejects_mixed_covariance_chunks_in_either_order(
    monkeypatch: pytest.MonkeyPatch,
    native_first: bool,
) -> None:
    """Adaptive dwell must fail closed for either covariance-mode ordering."""
    config = _spectrum_config_from_runtime_config(
        {"detector_scoring_mode": "incident_gamma_energy"}
    )
    decomposer = SpectralDecomposer(config)

    class _FakeRuntime:
        """Return one native and one approximate covariance chunk."""

        def __init__(self) -> None:
            """Initialize the deterministic chunk counter."""
            self.calls = 0

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Return covariance metadata in the parameterized order."""
            is_native = native_first if self.calls == 0 else not native_first
            self.calls += 1
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            spectrum = np.zeros_like(energy, dtype=float)
            spectrum[100] = 1.0
            variance = np.zeros_like(energy, dtype=float)
            variance[100] = 1.0
            metadata: dict[str, object] = {
                "backend": "geant4",
                "spectrum_count_variance": variance.tolist(),
            }
            if is_native:
                metadata.update(
                    {
                        "detector_scoring_mode": "incident_gamma_energy",
                        "detector_fast_scoring": True,
                        "spectrum_variance_semantics": (
                            "compound_poisson_sumw2_includes_counting"
                        ),
                        "spectrum_variance_dead_time_propagation": (
                            "fixed_observed_scale"
                        ),
                        "dead_time_observed_scale": 1.0,
                        "dead_time_tau_s": 0.0,
                        "dwell_time_s": float(command.dwell_time_s),
                    }
                )
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=spectrum.tolist(),
                energy_bin_edges_keV=np.concatenate(
                    [energy, [energy[-1] + step]]
                ).tolist(),
                metadata=metadata,
            )

    def _fake_evaluate(
        _decomposer: SpectralDecomposer,
        _spectrum: np.ndarray,
        **_kwargs: object,
    ) -> RuntimeCountResult:
        """Keep acquisition running until the second chunk is inspected."""
        return RuntimeCountResult(
            counts={"Cs-137": 0.0},
            variances={"Cs-137": 1.0},
            detected=set(),
        )

    monkeypatch.setattr(
        realtime_demo_module,
        "_evaluate_spectrum_count_result",
        _fake_evaluate,
    )
    with pytest.raises(
        ValueError,
        match="cannot mix native and approximate covariance chunks",
    ):
        _acquire_spectrum_observation(
            simulation_runtime=_FakeRuntime(),
            decomposer=decomposer,
            step_id=12,
            pose_xyz=np.array([1.0, 2.0, 0.5], dtype=float),
            fe_idx=0,
            pb_idx=0,
            live_time_s=4.0,
            travel_time_s=0.0,
            shield_actuation_time_s=0.0,
            adaptive_dwell=True,
            adaptive_dwell_chunk_s=2.0,
            adaptive_min_dwell_s=2.0,
            adaptive_ready_min_counts=100.0,
            adaptive_ready_min_isotopes=1,
            adaptive_ready_min_snr=0.0,
            spectrum_count_method="response_poisson",
            detect_threshold_abs=0.0,
            detect_threshold_rel=0.0,
            detect_threshold_rel_by_isotope={},
            min_peaks_by_isotope=None,
        )


def test_adaptive_dwell_can_run_without_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncapped adaptive dwell should stop from readiness, not a time cap."""
    decomposer = SpectralDecomposer()
    commands: list[SimulationCommand] = []

    class _FakeRuntime:
        """Return deterministic spectra proportional to requested dwell time."""

        def step(self, command: SimulationCommand) -> SimulationObservation:
            """Record each chunk and return a proportional spectrum."""
            commands.append(command)
            energy = np.asarray(decomposer.energy_axis, dtype=float)
            step = float(np.median(np.diff(energy)))
            spectrum = np.zeros_like(energy, dtype=float)
            spectrum[0] = float(command.dwell_time_s) * 60.0
            return SimulationObservation(
                step_id=command.step_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                spectrum_counts=spectrum.tolist(),
                energy_bin_edges_keV=np.concatenate(
                    [energy, [energy[-1] + step]]
                ).tolist(),
                metadata={"backend": "fake"},
            )

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return counts proportional to the accumulated spectrum."""
        count = float(np.sum(spectrum))
        self.last_count_variances = {"Cs-137": max(count, 1.0)}
        return {"Cs-137": count}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    observation, actual_live, counts, _variances, _detected, reason, chunks = (
        _acquire_spectrum_observation(
            simulation_runtime=_FakeRuntime(),
            decomposer=decomposer,
            step_id=9,
            pose_xyz=np.array([1.0, 2.0, 0.5], dtype=float),
            fe_idx=1,
            pb_idx=2,
            live_time_s=0.0,
            travel_time_s=0.0,
            shield_actuation_time_s=0.0,
            adaptive_dwell=True,
            adaptive_dwell_chunk_s=2.0,
            adaptive_min_dwell_s=2.0,
            adaptive_ready_min_counts=200.0,
            adaptive_ready_min_isotopes=1,
            adaptive_ready_min_snr=0.0,
            spectrum_count_method="response_poisson",
            detect_threshold_abs=0.0,
            detect_threshold_rel=0.0,
            detect_threshold_rel_by_isotope={},
            min_peaks_by_isotope=None,
        )
    )

    assert actual_live == pytest.approx(4.0)
    assert counts["Cs-137"] == pytest.approx(240.0)
    assert reason == "isotope_count_estimates_ready"
    assert chunks == 2
    assert observation.metadata["adaptive_dwell_ready_reason"] == reason
    assert [command.dwell_time_s for command in commands] == [2.0, 2.0]


def test_adaptive_dwell_accepts_informative_low_isotope_count() -> None:
    """A high-statistics spectrum may make a low isotope count informative."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 0.0, "Co-60": 500.0, "Eu-154": 120.0},
        {"Cs-137": 1.0, "Co-60": 500.0, "Eu-154": 120.0},
        live_time_s=40.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=3,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=0.0,
        total_spectrum_counts=50000.0,
    )

    assert ready is True
    assert "informative_low=1" in reason


def test_adaptive_dwell_rejects_too_early_informative_low_count() -> None:
    """Informative low-count stopping should not trigger from a two-second glimpse."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 0.0, "Co-60": 500.0, "Eu-154": 40.0},
        {"Cs-137": 1.0, "Co-60": 500.0, "Eu-154": 40.0},
        live_time_s=2.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=3,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=0.0,
        total_spectrum_counts=50000.0,
    )

    assert ready is False
    assert reason == "insufficient_isotope_count_estimates:1/3"


def test_adaptive_dwell_stops_on_low_signal_upper_bound() -> None:
    """A long low-signal observation should be usable as a censored count."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 0.0, "Co-60": 2.0, "Eu-154": 0.0},
        {"Cs-137": 1.0, "Co-60": 4.0, "Eu-154": 1.0},
        live_time_s=120.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=3,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=3.0,
        total_spectrum_counts=2.0,
        allow_informative_low=False,
        allow_low_signal_stop=True,
        low_signal_min_live_s=120.0,
        low_signal_upper_sigma=3.0,
    )

    assert ready is True
    assert reason == "low_signal_upper_bound:positive=0,below=3"


def test_adaptive_dwell_stops_on_low_signal_count_floor() -> None:
    """A long low-count observation should stop even with conservative covariance."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 0.0, "Co-60": 2.0, "Eu-154": 0.0},
        {"Cs-137": 1.0e6, "Co-60": 1.0e6, "Eu-154": 1.0e6},
        live_time_s=120.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=3,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=3.0,
        total_spectrum_counts=2.0,
        allow_informative_low=False,
        allow_low_signal_stop=True,
        low_signal_min_live_s=120.0,
        low_signal_upper_sigma=3.0,
        low_signal_count_fraction=0.05,
    )

    assert ready is True
    assert reason == "low_signal_count_floor:positive=0,below=3"


def test_adaptive_dwell_stops_when_projected_live_time_is_unproductive() -> None:
    """A pose should stop when count-rate extrapolation cannot reach target soon."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 17.0, "Co-60": 2.0, "Eu-154": 8.0},
        {"Cs-137": 1.0e6, "Co-60": 1.0e6, "Eu-154": 1.0e6},
        live_time_s=120.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=1,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=5.0,
        total_spectrum_counts=10000.0,
        allow_informative_low=False,
        allow_low_signal_stop=True,
        low_signal_min_live_s=120.0,
        low_signal_upper_sigma=3.0,
        low_signal_count_fraction=0.05,
        low_signal_projected_live_factor=4.0,
    )

    assert ready is True
    assert reason.startswith("low_signal_projected_time:positive=0")


def test_adaptive_dwell_keeps_collecting_when_projected_live_time_is_reasonable() -> (
    None
):
    """A sub-threshold count should continue when extrapolated target time is modest."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 60.0, "Co-60": 2.0, "Eu-154": 8.0},
        {"Cs-137": 1.0e6, "Co-60": 1.0e6, "Eu-154": 1.0e6},
        live_time_s=120.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=1,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=5.0,
        total_spectrum_counts=10000.0,
        allow_informative_low=False,
        allow_low_signal_stop=True,
        low_signal_min_live_s=120.0,
        low_signal_upper_sigma=3.0,
        low_signal_count_fraction=0.05,
        low_signal_projected_live_factor=4.0,
    )

    assert ready is False
    assert reason == "insufficient_isotope_count_estimates:0/1"


def test_adaptive_dwell_stops_when_snr_projection_is_unproductive() -> None:
    """A high-count but low-SNR isotope should not keep uncapped dwell running forever."""
    ready, reason = _is_adaptive_spectrum_ready(
        {"Cs-137": 6200.0, "Co-60": 0.0, "Eu-154": 6.0},
        {"Cs-137": 2.5e6, "Co-60": 1.0, "Eu-154": 1.0},
        live_time_s=1000.0,
        min_live_time_s=2.0,
        min_counts_per_detected_isotope=100.0,
        min_detected_isotopes=3,
        candidate_isotopes=["Cs-137", "Co-60", "Eu-154"],
        min_snr=5.0,
        total_spectrum_counts=10000.0,
        allow_informative_low=True,
        allow_low_signal_stop=True,
        low_signal_min_live_s=30.0,
        low_signal_upper_sigma=3.0,
        low_signal_count_fraction=0.05,
        low_signal_projected_live_factor=4.0,
    )

    assert ready is True
    assert reason.startswith("low_signal_projected_time:")
    assert "best_iso=Cs-137" in reason


def test_low_signal_variance_inflation_marks_censored_observation() -> None:
    """Low-signal dwell stops should not pass near-zero variances to the PF."""
    inflated = _inflate_low_signal_variances(
        {"Cs-137": 3.0, "Co-60": 0.0, "Eu-154": 12.0},
        {"Cs-137": 1.0, "Co-60": 1.0, "Eu-154": 4.0},
        min_counts_per_detected_isotope=100.0,
        ready_reason="low_signal_projected_time:positive=0,best=12,projected=945",
    )

    assert inflated["Cs-137"] >= 10000.0
    assert inflated["Co-60"] >= 10000.0
    assert inflated["Eu-154"] >= 10000.0


def test_non_low_signal_variance_inflation_is_noop() -> None:
    """Ready high-signal spectra should keep their decomposition variance."""
    inflated = _inflate_low_signal_variances(
        {"Cs-137": 300.0},
        {"Cs-137": 450.0},
        min_counts_per_detected_isotope=100.0,
        ready_reason="isotope_count_estimates_ready",
    )

    assert inflated["Cs-137"] == pytest.approx(450.0)


def test_partial_ready_variance_inflation_marks_unresolved_isotopes() -> None:
    """Adaptive stops triggered by one isotope should soften unresolved isotopes."""
    inflated = _inflate_low_signal_variances(
        {"Cs-137": 300.0, "Co-60": 0.0, "Eu-154": 12.0},
        {"Cs-137": 450.0, "Co-60": 1.0, "Eu-154": 4.0},
        min_counts_per_detected_isotope=100.0,
        ready_reason="isotope_count_estimates_ready",
    )

    assert inflated["Cs-137"] == pytest.approx(450.0)
    assert inflated["Co-60"] >= 10000.0
    assert inflated["Eu-154"] >= 10000.0


def test_spectrum_counts_filter_to_candidate_isotopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PF count extraction should drop channels outside candidate_isotopes."""
    decomposer = SpectralDecomposer()
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic multi-isotope counts for candidate filtering."""
        return {"Cs-137": 100.0, "Co-60": 200.0}, {"Cs-137", "Co-60"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )
    decomposer.last_count_variances = {"Cs-137": 10.0, "Co-60": 20.0}

    result = _evaluate_spectrum_count_result(
        decomposer,
        spectrum,
        live_time_s=1.0,
        spectrum_count_method="response_poisson",
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        candidate_isotopes=("Cs-137",),
    )

    assert result.counts == {"Cs-137": 100.0}
    assert result.variances == {"Cs-137": 10.0}
    assert result.detected == {"Cs-137"}


def test_spectrum_count_result_filters_candidate_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PF count extraction should preserve covariance for retained isotopes only."""
    decomposer = SpectralDecomposer()
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic counts and response-Poisson covariance."""
        self.last_count_variances = {
            "Cs-137": 16.0,
            "Co-60": 25.0,
            "Eu-154": 36.0,
        }
        self.last_count_covariance = {
            "Cs-137": {"Cs-137": 16.0, "Co-60": -10.0, "Eu-154": 4.0},
            "Co-60": {"Cs-137": -10.0, "Co-60": 25.0, "Eu-154": -3.0},
            "Eu-154": {"Cs-137": 4.0, "Co-60": -3.0, "Eu-154": 36.0},
        }
        return (
            {"Cs-137": 100.0, "Co-60": 200.0, "Eu-154": 300.0},
            {"Cs-137", "Co-60", "Eu-154"},
        )

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )

    result = _evaluate_spectrum_count_result(
        decomposer,
        spectrum,
        live_time_s=1.0,
        spectrum_count_method="response_poisson",
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        candidate_isotopes=("Cs-137", "Co-60"),
    )

    assert set(result.counts) == {"Cs-137", "Co-60"}
    assert set(result.variances) == {"Cs-137", "Co-60"}
    assert result.covariance is not None
    assert set(result.covariance) == {"Cs-137", "Co-60"}
    assert result.covariance["Cs-137"]["Co-60"] == pytest.approx(-10.0)
    assert "Eu-154" not in result.covariance["Cs-137"]


def test_effective_entries_add_count_variance_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted effective entries should soften high-count PF observations."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(response_poisson_count_variance_ceiling_enable=False)
    )
    spectrum = np.zeros_like(decomposer.energy_axis, dtype=float)
    spectrum[0] = 1000.0

    def _fake_counts(
        self: SpectralDecomposer,
        spectrum: np.ndarray,
        *,
        live_time_s: float = 1.0,
        **kwargs: object,
    ) -> tuple[dict[str, float], set[str]]:
        """Return deterministic isotope counts for variance-floor testing."""
        return {"Cs-137": 1000.0}, {"Cs-137"}

    monkeypatch.setattr(
        SpectralDecomposer,
        "isotope_counts_with_detection",
        _fake_counts,
    )
    decomposer.last_count_variances = {"Cs-137": 1.0}

    result = _evaluate_spectrum_count_result(
        decomposer,
        spectrum,
        live_time_s=30.0,
        spectrum_count_method="response_poisson",
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_threshold_rel_by_isotope={},
        min_peaks_by_isotope=None,
        transport_metadata={"weighted_spectrum_effective_entries": "25"},
    )

    assert result.counts["Cs-137"] == pytest.approx(1000.0)
    assert result.variances["Cs-137"] == pytest.approx(40000.0)
    assert result.detected == {"Cs-137"}


def test_spectrum_isotope_channel_diagnostics_logs_photopeak_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compact spectrum diagnostics should expose per-isotope photopeak evidence."""

    class _DummyDecomposer:
        """Store deterministic response-Poisson diagnostics for logging."""

        last_response_poisson_diagnostics = {
            "counts": {"Cs-137": 120.0},
            "variances": {"Cs-137": 16.0},
            "photopeak_counts": {"Cs-137": 100.0},
            "photopeak_variances": {"Cs-137": 25.0},
            "snr": {"Cs-137": 30.0},
            "methods": {"Cs-137": "response_poisson_photopeak_fused"},
            "coefficient_correlation_by_isotope": {"Cs-137": 0.2},
        }

    _log_spectrum_isotope_channel_diagnostics(
        _DummyDecomposer(),  # type: ignore[arg-type]
        step_index=7,
        selected_counts={"Cs-137": 120.0},
        selected_variances={"Cs-137": 18.0},
    )

    output = capsys.readouterr().out
    assert "[step 7] spectrum_isotope_channels" in output
    assert "photopeak_over_response" in output
    assert "response_poisson_photopeak_fused" in output


def test_detector_height_partner_requires_same_xy_and_distinct_z() -> None:
    """Height-pair actions should not be confused with revisits or base motion."""
    low = np.array([1.0, 2.0, 0.5], dtype=float)

    assert realtime_demo_module._is_detector_height_partner(
        low,
        np.array([1.0, 2.0, 1.5], dtype=float),
        xy_tolerance_m=1.0e-6,
    )
    assert realtime_demo_module._is_detector_height_partner(
        low,
        np.array([1.0 + 5.0e-7, 2.0, 1.5], dtype=float),
        xy_tolerance_m=1.0e-6,
    )
    assert not realtime_demo_module._is_detector_height_partner(
        low,
        low,
        xy_tolerance_m=1.0e-6,
    )
    assert not realtime_demo_module._is_detector_height_partner(
        low,
        np.array([1.1, 2.0, 1.5], dtype=float),
        xy_tolerance_m=1.0e-6,
    )
    assert not realtime_demo_module._is_detector_height_partner(
        low,
        np.array([1.0, 2.0, 0.6], dtype=float),
        xy_tolerance_m=1.0e-6,
        min_z_separation_m=0.25,
    )


def test_operational_station_metrics_use_recorded_poses_and_planner_tolerances() -> (
    None
):
    """Operational counts should use actual measurement poses and tolerate jitter."""
    recorded_positions = [
        (1.0, 2.0, 0.5),
        (1.0 + 0.4e-6, 2.0, 0.5),
        (1.0 + 0.5e-6, 2.0, 1.5),
        (4.0, 2.0, 1.5),
        (1.0 + 0.3e-6, 2.0, 0.5),
    ]
    measurements = [
        MeasurementRecord(
            z_k={"Cs-137": 1.0},
            pose_idx=0,
            live_time_s=1.0,
            fe_index=index,
            pb_index=index,
            detector_position_xyz_m=position,
            station_sequence_id=index,
            station_view_index=0,
            runtime_likelihood_route_by_isotope={"Cs-137": "count"},
        )
        for index, position in enumerate(recorded_positions)
    ]
    metrics = realtime_demo_module._operational_station_height_metrics(
        measurements,
        [np.array([99.0, 99.0, 9.0], dtype=float)],
        xy_tolerance_m=1.0e-6,
        z_tolerance_m=1.0e-6,
    )

    assert metrics["observed_detector_heights_m"] == pytest.approx([0.5, 1.5])
    assert metrics["station_visit_count"] == 3
    assert metrics["unique_xy_station_count"] == 2
    assert metrics["unique_xyz_action_count"] == 3
    assert metrics["height_pair_station_count"] == 1
    assert metrics["height_transition_count"] == 2
    assert metrics["station_count"] == metrics["unique_xy_station_count"]
    assert metrics["detector_pose_station_count"] == metrics["unique_xyz_action_count"]
    assert metrics["height_change_count"] == metrics["height_transition_count"]
    assert "position_source" in metrics["station_height_count_definitions"]


def test_json_payload_sanitizer_is_recursive_and_strict() -> None:
    """Final summary sanitization should remove NumPy types and non-finite values."""
    payload = {
        "array": np.array([1.0, np.nan, np.inf]),
        "nested": ({"value": np.float64(-np.inf)}, np.int64(3)),
        "flags": {np.bool_(True), np.bool_(False)},
    }

    sanitized = realtime_demo_module._sanitize_json_payload(payload)

    assert sanitized["array"] == [1.0, None, None]
    assert sanitized["nested"] == [{"value": None}, 3]
    assert sorted(sanitized["flags"]) == [False, True]
    json.dumps(sanitized, allow_nan=False)


def test_surface_diagnostics_accept_the_posterior_annotation_tolerance() -> None:
    """Surface summaries should use the same tolerance as posterior annotation."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    positions = np.array([[1.0, 1.0, 5.0e-6]], dtype=float)

    loose = realtime_demo_module._surface_count_payload(
        positions,
        env,
        None,
        obstacle_height_m=1.0,
        tolerance_m=1.0e-5,
    )
    strict = realtime_demo_module._surface_count_payload(
        positions,
        env,
        None,
        obstacle_height_m=1.0,
        tolerance_m=1.0e-6,
    )

    assert loose["surface_counts"]["floor"] == 1
    assert loose["off_surface_count"] == 0
    assert strict["surface_counts"]["floor"] == 0
    assert strict["off_surface_count"] == 1


def test_detector_mast_heights_resolve_to_world_z_above_nonzero_ground() -> None:
    """PF actions should match the controller's ground-plus-mast world height."""
    ground_z, initial_world_z, mast_actions, world_actions = (
        realtime_demo_module._resolve_detector_height_world_actions(
            {
                "robot_ground_z_m": 0.2,
                "detector_height_m": 0.6,
                "detector_height_min_m": 0.5,
                "detector_height_max_m": 1.5,
                "detector_height_actions_m": [0.6, 1.4],
            },
            room_height_m=2.0,
        )
    )

    assert ground_z == pytest.approx(0.2)
    assert initial_world_z == pytest.approx(0.8)
    assert mast_actions == pytest.approx([0.6, 1.4])
    assert world_actions == pytest.approx([0.8, 1.6])


def test_continuous_detector_height_workspace_uses_full_mast_interval() -> None:
    """Continuous planning should sample the mast interval without action levels."""
    config = realtime_demo_module._resolve_detector_height_planning_config(
        {
            "robot_ground_z_m": 0.2,
            "detector_height_m": 0.6,
            "detector_height_min_m": 0.5,
            "detector_height_max_m": 1.5,
            "detector_height_sampling_mode": "continuous",
        },
        room_height_m=2.0,
    )

    assert config.mode == "continuous"
    assert config.initial_world_z_m == pytest.approx(0.8)
    assert config.candidate_world_z_bounds_m == pytest.approx((0.7, 1.7))
    assert config.candidate_world_heights_m is None
    assert config.discrete_mast_actions_m == ()


def test_continuous_detector_height_defaults_to_full_room_workspace() -> None:
    """Omitted mast bounds should expose the full room-height interval."""
    config = realtime_demo_module._resolve_detector_height_planning_config(
        {
            "robot_ground_z_m": 0.2,
            "detector_height_m": 0.6,
            "detector_height_sampling_mode": "continuous",
        },
        room_height_m=2.0,
    )

    assert config.minimum_mast_height_m == pytest.approx(0.0)
    assert config.maximum_mast_height_m == pytest.approx(1.8)
    assert config.candidate_world_z_bounds_m == pytest.approx((0.2, 2.0))


def test_continuous_detector_height_workspace_rejects_discrete_actions() -> None:
    """Ambiguous continuous and discrete height settings should fail early."""
    with pytest.raises(ValueError, match="must be omitted"):
        realtime_demo_module._resolve_detector_height_planning_config(
            {
                "detector_height_m": 0.5,
                "detector_height_min_m": 0.5,
                "detector_height_max_m": 1.5,
                "detector_height_sampling_mode": "continuous",
                "detector_height_actions_m": [0.5, 1.5],
            },
            room_height_m=2.0,
        )


def test_continuous_workspace_accepts_arbitrary_collision_free_xyz() -> None:
    """Room-only planning should accept continuous xy and z measurement poses."""
    height_config = realtime_demo_module._resolve_detector_height_planning_config(
        {
            "detector_height_m": 0.5,
            "detector_height_min_m": 0.5,
            "detector_height_max_m": 1.5,
            "detector_height_sampling_mode": "continuous",
            "measurement_pose_clearance_enabled": True,
        },
        room_height_m=10.0,
    )
    radius = realtime_demo_module._resolve_measurement_clearance_radius_m(
        {"measurement_pose_clearance_enabled": True},
        requested_robot_radius_m=0.35,
    )
    workspace, diagnostics = realtime_demo_module._build_measurement_workspace(
        {"measurement_pose_clearance_enabled": True},
        environment_size_xyz=(10.0, 20.0, 10.0),
        detector_height_config=height_config,
        obstacle_grid=None,
        base_map=None,
        shield_params=ShieldParams(),
        effective_robot_radius_m=radius,
    )

    arbitrary_poses = np.array(
        [
            [0.73, 0.81, 0.67],
            [4.321, 11.234, 1.137],
            [9.19, 19.27, 1.493],
        ],
        dtype=float,
    )
    assert diagnostics["continuous_measurement_volume"] is True
    assert diagnostics["route_grid_cell_size_m"] == pytest.approx(0.25)
    assert np.all(workspace.is_free_batch(arbitrary_poses))
    assert not workspace.is_free((0.1, 2.0, 1.0))
    waypoints = workspace.motion_waypoints(arbitrary_poses[0], arbitrary_poses[1])
    assert waypoints is not None
    assert waypoints[0] == pytest.approx(arbitrary_poses[0])
    assert waypoints[-1] == pytest.approx(arbitrary_poses[1])


def test_room_wide_continuous_workspace_accepts_high_free_measurement_pose() -> None:
    """Room-wide mode should retain high poses that clear the ceiling."""
    height_config = realtime_demo_module._resolve_detector_height_planning_config(
        {
            "detector_height_m": 0.5,
            "detector_height_sampling_mode": "continuous",
            "measurement_pose_clearance_enabled": True,
        },
        room_height_m=10.0,
    )
    radius = realtime_demo_module._resolve_measurement_clearance_radius_m(
        {"measurement_pose_clearance_enabled": True},
        requested_robot_radius_m=0.35,
    )
    workspace, _ = realtime_demo_module._build_measurement_workspace(
        {"measurement_pose_clearance_enabled": True},
        environment_size_xyz=(10.0, 20.0, 10.0),
        detector_height_config=height_config,
        obstacle_grid=None,
        base_map=None,
        shield_params=ShieldParams(),
        effective_robot_radius_m=radius,
    )

    assert height_config.candidate_world_z_bounds_m == pytest.approx((0.0, 10.0))
    assert workspace.is_free((4.321, 11.234, 9.7))


def test_count_error_model_reports_three_distinct_layers() -> None:
    """Final diagnostics should not collapse bias and model mismatch into variance."""
    config = RotatingShieldPFConfig(
        measurement_scale_by_isotope={"Cs-137": 1.01},
        measurement_scale_by_isotope_and_pair={"Cs-137": {3: 0.99}},
    )

    diagnostics = realtime_demo_module._count_error_model_diagnostics(
        config,
        obstacle_attenuation_active=True,
    )

    assert set(diagnostics) == {
        "statistical_uncertainty",
        "calibrated_systematic_response",
        "forward_model_mismatch",
    }
    assert diagnostics["calibrated_systematic_response"]["shield_pair_scale_configured"]
    assert diagnostics["forward_model_mismatch"]["obstacle_attenuation_active"]
