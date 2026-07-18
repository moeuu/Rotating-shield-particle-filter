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
from pf.likelihood import expected_counts_per_source
from pf.mixing import prune_spurious_sources_continuous
from pf.particle_filter import IsotopeParticle, MeasurementData
from pf.profiles import resolve_estimator_profile
from pf.state import IsotopeState
from realtime_demo import (
    ADAPTIVE_STEP_ID_STRIDE,
    DeferredPFVisualizer,
    _acquire_spectrum_observation,
    _adapt_dss_program_length_for_budget,
    _adaptive_mission_stop_reason,
    _all_pf_filters_converged,
    _apply_baseline_shield_program_to_dss_config,
    _argv_requests_cui,
    _build_candidate_sources,
    _build_effective_live_runtime_config,
    _build_intermediate_estimate_trace_payload,
    _build_robot_path_segment,
    _compute_shield_selection_grid,
    _diagnostic_detail_order,
    _evaluate_spectrum_count_result,
    _evaluate_spectrum_counts,
    _filter_absent_final_estimates,
    _filter_reachable_candidates,
    _format_pf_timing_item,
    _format_estimate_trace_log_line,
    _format_truth_coverage_log_line,
    _best_dss_first_step_guard_candidate,
    _final_model_order_status,
    _has_birth_residual_evidence,
    _has_unresolved_discriminative_pseudo_failures,
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
    _online_absent_pruning_supported_isotopes,
    _prune_online_absent_isotopes,
    _measurement_log_obstacle_layout_path,
    _resolve_ig_workers,
    _resolve_runtime_use_gpu,
    _resolve_mission_max_poses,
    _resolve_mission_max_steps,
    _resolve_plot_save_interval,
    _resolve_python_worker_count,
    _resolve_cui_split_view_enabled,
    _resolve_candidate_isotopes,
    _resolve_display_prune_refresh_interval,
    _resolve_structural_trial_parallelism,
    _resolve_station_update_modes,
    _resolve_required_measurement_log_target,
    _particle_surface_diagnostics,
    _report_model_order_simple_ready_for_stop,
    _select_best_pair_from_scores,
    _should_refresh_display_pruned_estimates,
    _signature_vector_is_dependent,
    _resolve_source_position_bounds,
    _spectrum_evidence_payload,
    _spectrum_config_from_runtime_config,
    _source_cardinality_dwell_status,
    _truth_free_live_runtime_config,
    _validate_surface_constrained_estimates,
    _remaining_measurement_progress,
    run_live_pf,
)
from mission_control import (
    report_model_order_ready_for_stop,
    sparse_cardinality_evidence_gap_unresolved,
)
from planning.dss_pp import DSSPPConfig
from sim import SimulationCommand, SimulationObservation
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig
from visualization.realtime_viz import PFFrame


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


def test_spectrum_evidence_payload_uses_independent_configured_background() -> None:
    """Final spectral history must not fix a background fitted to the same data."""
    decomposer = SpectralDecomposer(
        SpectrumConfig(response_poisson_background_rate_cps=12.0)
    )
    decomposer.last_response_poisson_background = np.full(
        decomposer.energy_axis.shape,
        1.0e6,
        dtype=float,
    )
    spectrum = np.ones_like(decomposer.energy_axis, dtype=float)

    payload = _spectrum_evidence_payload(
        decomposer,
        spectrum,
        live_time_s=2.0,
        spectrum_variance=None,
        isotopes=("Cs-137",),
    )

    assert payload is not None
    assert payload["spectrum_background_source"] == (
        "configured_rate_and_detector_background_shape"
    )
    assert payload["spectrum_background_observation_independent"] is True
    assert float(np.sum(payload["spectrum_background"])) == pytest.approx(24.0)


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
            "joint_observation_update": False,
            "delayed_resample_update": True,
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
            "joint_observation_update": False,
            "delayed_resample_update": True,
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

    assert _measurement_log_obstacle_layout_path(
        fixed_environment,
        repository_root=repository_root,
    ) == "obstacle_layouts/fixed.json"
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
    """Primary pure-PF output must bypass legacy report and display filters."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    expected_positions = np.asarray([[1.0, 2.0, 0.5]], dtype=float)
    expected_strengths = np.asarray([25.0], dtype=float)
    estimator = SimpleNamespace(
        profile_capabilities=capabilities,
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


def test_pure_legacy_summary_embeds_complete_posterior_provenance() -> None:
    """Every pure-PF result file must identify its log, config, and PF origin."""
    _profile, capabilities = resolve_estimator_profile("pf_strict")
    payload = {
        "schema_version": 1,
        "estimator_family": "particle_filter",
        "estimator_variant": "pf_strict",
        "estimator_profile": "pf_strict",
        "final_estimate_source": "pf_posterior",
        "uses_all_history_batch_fit": False,
        "uses_surface_map": False,
        "uses_batch_model_order": False,
        "batch_feedback_to_particles": False,
        "batch_methods_invoked": [],
        "planner_belief_sources": ["pf_posterior", "pf_tentative"],
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
        posterior_snapshot=lambda: SimpleNamespace(to_dict=lambda: dict(payload)),
    )

    summary = _pure_pf_summary_provenance(estimator)

    assert summary["final_estimate_source"] == "pf_posterior"
    assert summary["measurement_log_sha256"] == "b" * 64
    assert summary["resolved_config_sha256"] == "d" * 64
    assert summary["batch_methods_invoked"] == []
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


def test_joint_observation_update_disables_delayed_resample() -> None:
    """Station-window joint PF updates should not resample between postures."""
    assert _resolve_station_update_modes({}) == (False, True)
    assert _resolve_station_update_modes({"delayed_resample_update": False}) == (
        False,
        False,
    )
    assert _resolve_station_update_modes({"joint_observation_update": True}) == (
        True,
        False,
    )


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
    assert _resolve_station_update_modes(
        {"joint_observation_update": True, "delayed_resample_update": True}
    ) == (True, False)


def test_remaining_measurement_progress_uses_full_history_residual() -> None:
    """Soft extension progress should accept residual-budget improvement."""
    estimates = [
        {
            "components": {"residual": 5.0},
            "current_budget": 12.0,
            "estimated_remaining_stations": 3,
        },
        {
            "components": {"residual": 4.0},
            "current_budget": 12.5,
            "estimated_remaining_stations": 3,
        },
    ]

    progress = _remaining_measurement_progress(estimates)

    assert progress["available"] is True
    assert progress["has_progress"] is True
    assert progress["residual_improved"] is True
    assert progress["budget_improved"] is False
    assert progress["remaining_stations_improved"] is False


def test_remaining_measurement_progress_rejects_stalled_budget() -> None:
    """Soft extension progress should reject stalled residual and budget signals."""
    estimates = [
        {
            "components": {"residual": 5.0},
            "current_budget": 12.0,
            "estimated_remaining_stations": 3,
        },
        {
            "components": {"residual": 5.5},
            "current_budget": 12.0,
            "estimated_remaining_stations": 3,
        },
    ]

    progress = _remaining_measurement_progress(estimates)

    assert progress["available"] is True
    assert progress["has_progress"] is False
    assert progress["residual_improved"] is False
    assert progress["budget_improved"] is False
    assert progress["remaining_stations_improved"] is False


def test_remaining_measurement_progress_requires_full_history_improvement() -> None:
    """Soft extension progress should not accept a rebound above the best residual."""
    estimates = [
        {
            "components": {"residual": 5.0},
            "current_budget": 12.0,
            "estimated_remaining_stations": 3,
        },
        {
            "components": {"residual": 3.0},
            "current_budget": 9.0,
            "estimated_remaining_stations": 2,
        },
        {
            "components": {"residual": 4.0},
            "current_budget": 10.0,
            "estimated_remaining_stations": 2,
        },
    ]

    progress = _remaining_measurement_progress(estimates)

    assert progress["available"] is True
    assert progress["has_progress"] is False
    assert progress["residual_recent_improved"] is False
    assert progress["best_previous_residual_budget"] == pytest.approx(3.0)


def test_particle_surface_diagnostics_use_report_visible_sources() -> None:
    """Final particle surface diagnostics should count report-visible sources."""
    isotope = "Cs-137"
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=3.0)
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.array([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.array([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=2,
            report_exclude_unverified_sources=True,
            use_gpu=False,
        ),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 0], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 0], dtype=int),
    )
    estimator.filters[isotope].continuous_particles = [
        IsotopeParticle(state=state, log_weight=0.0)
    ]

    diagnostics = _particle_surface_diagnostics(
        estimator,
        env,
        None,
        obstacle_height_m=2.0,
    )[isotope]

    assert diagnostics["raw_source_slots"] == 2
    assert diagnostics["report_visible_source_slots"] == 1
    assert diagnostics["report_excluded_source_slots"] == 1
    assert diagnostics["surface_counts"]["floor"] == 1
    assert diagnostics["off_surface_count"] == 0


def test_report_mle_rescue_global_surface_candidates_recover_modes() -> None:
    """Global report rescue should recover separated sources from all candidates."""
    isotope = "Cs-137"
    candidates = np.array(
        [
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 1.0],
            [2.5, 2.5, 2.0],
            [1.0, 4.0, 1.0],
            [4.0, 1.2, 2.0],
        ],
        dtype=float,
    )
    truth = np.array([[1.0, 4.0, 1.0], [4.0, 1.2, 2.0]], dtype=float)
    strengths = np.array([450.0, 450.0], dtype=float)
    detector_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [2.5, 0.0, 1.0],
            [0.0, 2.5, 1.0],
            [5.0, 2.5, 1.0],
            [2.5, 5.0, 1.0],
        ],
        dtype=float,
    )
    fe_indices = np.zeros(detector_positions.shape[0], dtype=int)
    pb_indices = np.zeros(detector_positions.shape[0], dtype=int)
    live_times = np.full(detector_positions.shape[0], 5.0, dtype=float)
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=candidates,
        shield_normals=None,
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=4,
            use_gpu=False,
            report_mle_rescue_enable=True,
            report_mle_rescue_max_candidates=4,
            report_mle_rescue_max_residual_candidates=3,
            report_mle_rescue_dedup_radius_m=0.25,
            report_mle_rescue_min_residual_fraction=0.0,
        ),
    )
    estimator.add_measurement_pose(detector_positions[0])
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    design = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=isotope,
        detector_positions=detector_positions,
        sources=truth,
        strengths=np.ones(truth.shape[0], dtype=float),
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        source_scale=estimator.response_scale_for_isotope(isotope),
    )
    z_obs = design @ strengths
    data = MeasurementData(
        z_k=z_obs,
        observation_variances=np.maximum(z_obs, 1.0),
        detector_positions=detector_positions,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times=live_times,
    )

    rescued_pos, rescued_q, stats = estimator._rank_global_surface_candidates(
        isotope,
        filt,
        data,
        existing_positions=np.zeros((0, 3), dtype=float),
        background=np.zeros_like(z_obs),
        eps=1.0e-9,
        q_max=0.0,
    )

    assert rescued_pos.shape == (3, 3)
    assert rescued_q.shape == (3,)
    assert int(stats["global_rescue_candidate_count"]) == 3
    distances = np.linalg.norm(rescued_pos[:, None, :] - truth[None, :, :], axis=2)
    assert np.max(np.min(distances, axis=0)) < 0.25


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


def test_intermediate_estimate_trace_includes_source_slot_metadata() -> None:
    """Intermediate estimate traces should keep MAP source-slot diagnostics."""
    isotope = "Cs-137"
    env = EnvironmentConfig(size_x=10.0, size_y=10.0, size_z=10.0)
    frame = {
        "estimate_source": "post_finalize_map",
        "step_index": 8,
        "time": 130.0,
        "robot_position": np.array([1.0, 2.0, 0.5], dtype=float),
        "counts_by_isotope": {isotope: 500.0},
        "estimated_sources": {
            isotope: np.array([[0.0, 1.0, 1.0]], dtype=float),
        },
        "estimated_strengths": {isotope: np.array([12.0], dtype=float)},
        "estimated_metadata": {
            isotope: [
                {
                    "age": 4,
                    "tentative": True,
                    "verification_fail_streak": 2,
                    "support_score": 3.5,
                    "low_q_streak": 1,
                }
            ],
        },
    }

    payload = _build_intermediate_estimate_trace_payload(
        frame,
        {isotope: np.array([[0.0, 1.0, 1.0]], dtype=float)},
        {isotope: [10.0]},
        env,
        None,
        obstacle_height_m=2.0,
        match_radius_m=0.5,
    )
    record = payload["estimates"][0]
    line = _format_estimate_trace_log_line(
        8,
        isotope,
        {
            **payload["isotopes"][isotope],
            "estimate_source": payload["estimate_source"],
        },
        payload["estimates"],
    )

    assert record["age"] == 4
    assert record["tentative"] is True
    assert record["verification_fail_streak"] == 2
    assert record["support_score"] == pytest.approx(3.5)
    assert "mode=post_finalize_map" in line
    assert "age=4" in line
    assert "tent=True" in line
    assert "fail=2" in line


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


def test_display_pruned_estimate_refresh_interval_is_clamped() -> None:
    """Display pruning refresh intervals should parse safely."""
    assert _resolve_display_prune_refresh_interval({}) == 1
    assert (
        _resolve_display_prune_refresh_interval(
            {"display_pruned_estimates_every": 8},
        )
        == 8
    )
    assert (
        _resolve_display_prune_refresh_interval(
            {"display_pruned_estimates_every": 0},
        )
        == 0
    )
    assert (
        _resolve_display_prune_refresh_interval(
            {"display_pruned_estimates_every": "bad"},
        )
        == 1
    )


def test_display_pruned_estimate_disable_overrides_forced_refresh() -> None:
    """Disabled display pruning should not recompute expensive report previews."""
    assert (
        _should_refresh_display_pruned_estimates(
            step_index=8,
            refresh_every=0,
            cache_available=False,
            force_refresh=True,
        )
        is False
    )
    assert (
        _should_refresh_display_pruned_estimates(
            step_index=8,
            refresh_every=0,
            cache_available=True,
            force_refresh=True,
        )
        is False
    )


def test_diagnostic_detail_limit_uses_zero_as_no_details() -> None:
    """High-detail diagnostic limits should avoid accidental full log dumps."""
    order = np.array([4, 2, 0, 1, 3], dtype=int)

    assert _diagnostic_detail_order(order, 0).tolist() == []
    assert _diagnostic_detail_order(order, 2).tolist() == [4, 2]
    assert _diagnostic_detail_order(order, -1).tolist() == [4, 2, 0, 1, 3]


def test_pf_timing_formatter_keeps_counters_unitless() -> None:
    """PF timing logs should not print diagnostic counters as seconds."""
    assert _format_pf_timing_item("total", 1.25) == "total=1.250s"
    assert (
        _format_pf_timing_item("sparse_evidence_synced_particles", 6130.0)
        == "sparse_evidence_synced_particles=6130"
    )
    assert (
        _format_pf_timing_item("sparse_evidence_cardinality_ready", 1.0)
        == "sparse_evidence_cardinality_ready=1"
    )


def test_precision_diagnostics_skip_birth_candidate_grid_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime precision diagnostics should not score all birth candidates by default."""
    calls: list[str] = []

    def noop(*args: object, **kwargs: object) -> None:
        """Replace lightweight diagnostic callbacks during this routing test."""
        _ = (args, kwargs)

    def record_birth(*args: object, **kwargs: object) -> None:
        """Record whether the expensive birth-candidate diagnostic is invoked."""
        _ = (args, kwargs)
        calls.append("birth")

    for name in (
        "_log_spectrum_response_poisson_diagnostics",
        "_log_current_map_prediction_residuals",
        "_log_truth_observability_diagnostics",
        "_log_posterior_truth_mass_diagnostics",
        "_log_particle_cloud_diagnostics",
        "_log_source_event_diagnostics",
    ):
        monkeypatch.setattr(realtime_demo_module, name, noop)
    monkeypatch.setattr(
        realtime_demo_module,
        "_log_birth_candidate_diagnostics",
        record_birth,
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
        candidate_log_limit=64,
        particle_log_limit=0,
    )
    assert calls == []

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
        candidate_log_limit=64,
        particle_log_limit=0,
        birth_candidate_diagnostics_enabled=True,
    )
    assert calls == ["birth"]


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
        "_log_birth_candidate_diagnostics",
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
        candidate_log_limit=0,
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
        candidate_log_limit=0,
        particle_log_limit=0,
        full_spectrum_response_diagnostics_enabled=True,
    )

    assert calls == ["compact", "full"]


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


def test_structural_trial_parallelism_reads_runtime_config() -> None:
    """Full runtime config should reach PF structural-trial parallelism."""
    workers, min_trials = _resolve_structural_trial_parallelism(
        {
            "structural_trial_workers": 32,
            "structural_trial_parallel_min_trials": 4,
        }
    )

    assert workers == 32
    assert min_trials == 4
    assert _resolve_structural_trial_parallelism({}) == (1, 8)


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


def test_display_pruned_estimates_refresh_policy() -> None:
    """Display-only pruning should refresh on cache miss, force, or interval."""
    assert _should_refresh_display_pruned_estimates(
        step_index=3,
        refresh_every=8,
        cache_available=False,
        force_refresh=False,
    )
    assert _should_refresh_display_pruned_estimates(
        step_index=3,
        refresh_every=8,
        cache_available=True,
        force_refresh=True,
    )
    assert _should_refresh_display_pruned_estimates(
        step_index=16,
        refresh_every=8,
        cache_available=True,
        force_refresh=False,
    )
    assert not _should_refresh_display_pruned_estimates(
        step_index=17,
        refresh_every=8,
        cache_available=True,
        force_refresh=False,
    )
    assert not _should_refresh_display_pruned_estimates(
        step_index=16,
        refresh_every=0,
        cache_available=True,
        force_refresh=False,
    )


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


def test_adaptive_mission_coverage_waits_for_quiet_birth_residuals() -> None:
    """Coverage should not stop a mission while residual birth evidence remains."""

    class _DummyFilter:
        """Minimal filter state exposing residual-birth diagnostics."""

        last_birth_residual_gate_passed = True
        last_birth_residual_support = 3

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this non-model-order test."""
            return True

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
        require_quiet_birth_residual=True,
        birth_residual_min_support=2,
    )

    assert reason is None
    assert _has_birth_residual_evidence(
        _DummyEstimator(),  # type: ignore[arg-type]
        min_support=2,
    )


def test_adaptive_mission_waits_for_discriminative_pseudo_failures() -> None:
    """Mission stop should wait while source verification needs new views."""

    class _DummyFilter:
        """Minimal filter state exposing discriminative pseudo-source failures."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0
        last_pseudo_source_fail_reasons = {
            "needs_discriminative_views": 2,
            "high_response_corr": 1,
        }

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this discriminative-failure test."""
            return True

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=(),
    )
    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]
    estimator = _DummyEstimator()

    reason = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
    )

    assert reason is None
    assert _has_unresolved_discriminative_pseudo_failures(
        estimator,  # type: ignore[arg-type]
        min_count=1,
    )

    reason_without_guard = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_no_unresolved_discriminative_failures=False,
    )

    assert reason_without_guard == "environment_coverage:1.000"


def test_adaptive_mission_waits_for_remaining_measurement_budget() -> None:
    """Coverage should not stop while the remaining-measurement budget is unresolved."""

    class _DummyFilter:
        """Minimal filter state without residual or pseudo-source blockers."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0
        last_pseudo_source_fail_reasons: dict[str, int] = {}

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this remaining-budget test."""
            return True

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=(),
    )
    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]
    unresolved_budget = {
        "unresolved_factors": ["residual"],
        "estimated_remaining_stations": 3,
        "current_budget": 10.0,
    }

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
        remaining_measurement_estimate=unresolved_budget,
    )

    assert reason is None

    reason_without_guard = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        remaining_measurement_estimate=unresolved_budget,
        require_remaining_measurement_ready=False,
    )

    assert reason_without_guard == "environment_coverage:1.000"


def test_final_model_order_status_marks_unresolved_pseudo_sources_not_ready() -> None:
    """Final model-order status should include pseudo-source structural gates."""

    class _DummyFilter:
        """Minimal filter state exposing unresolved pseudo-source verification."""

        last_pseudo_source_verified = 0
        last_pseudo_source_failed = 2
        last_pseudo_source_pruned = 0
        last_pseudo_source_quarantined = 0
        last_pseudo_source_quarantine_active = 0
        last_pseudo_source_fail_reasons = {
            "needs_discriminative_views": 2,
            "high_response_corr": 1,
        }
        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0

    class _DummyEstimator:
        """Minimal estimator with ready BIC but unresolved structural evidence."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty report estimate."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return report-level BIC diagnostics that are locally ready."""
            return {
                "Cs-137": {
                    "model_order_ready": True,
                    "condition_number": 1.0,
                    "criterion_margin_to_simpler": float("inf"),
                    "selected_max_response_correlation": 0.0,
                }
            }

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return unresolved structural evidence as the estimator does."""
            return {
                "Cs-137": {
                    "pseudo_source_fail_reasons": {
                        "needs_discriminative_views": 2,
                        "high_response_corr": 1,
                    }
                }
            }

    status = _final_model_order_status(_DummyEstimator())

    assert status["all_model_order_ready"] is False
    assert status["all_model_order_ready_before_structural_gates"] is True
    assert status["unresolved_structural_evidence"]["Cs-137"]


def test_final_model_order_status_marks_missing_diagnostics_not_ready() -> None:
    """Final model-order status should not treat absent evidence as ready."""

    class _DummyEstimator:
        """Minimal estimator before report diagnostics have been generated."""

        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty report estimate."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return no model-order diagnostics."""
            return {}

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return no separately tracked structural evidence."""
            return {}

    status = _final_model_order_status(_DummyEstimator())

    assert status["all_model_order_ready"] is False
    assert status["all_model_order_ready_before_structural_gates"] is False


def test_cardinality_dwell_waits_for_unresolved_structural_evidence() -> None:
    """Adaptive cardinality dwell should not stop while structural evidence is open."""

    class _DummyEstimator:
        """Minimal estimator exposing unresolved report-underfit evidence."""

        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty report estimate."""
            return {}

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return unresolved report-level evidence."""
            return {"Cs-137": {"report_underfit": {"reason": "count_supported_zero"}}}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return ready-looking diagnostics that should still be blocked."""
            return {"Cs-137": {"model_order_ready": True, "selected_count": 0}}

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
    )

    assert ready is False
    assert reason == "unresolved_structural:Cs-137"


def test_cardinality_dwell_waits_for_model_order_diagnostics() -> None:
    """Adaptive cardinality dwell should wait until report evidence exists."""

    class _DummyEstimator:
        """Minimal estimator before report diagnostics have been generated."""

        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty report estimate."""
            return {}

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return no separately tracked structural evidence."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return no report-level model-order diagnostics."""
            return {}

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
    )

    assert ready is False
    assert reason == "no_model_order_diagnostics"


def test_cardinality_dwell_requires_explicit_model_order_ready() -> None:
    """Active multi-source diagnostics need an explicit ready flag."""

    class _DummyEstimator:
        """Minimal estimator with active diagnostics missing readiness."""

        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a two-source report estimate."""
            return {
                "Cs-137": (
                    np.zeros((2, 3), dtype=float),
                    np.ones(2, dtype=float),
                )
            }

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return no separately tracked structural evidence."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return active diagnostics without an explicit readiness decision."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 2,
                    "condition_number": 1.0,
                    "criterion_margin_to_simpler": 20.0,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
    )

    assert ready is False
    assert reason == "Cs-137:model_order"


def test_cardinality_dwell_ignores_inactive_zero_source_candidates() -> None:
    """Candidate pools without count support should not activate absent isotopes."""

    class _DummyEstimator:
        """Minimal estimator with candidate-only zero-source diagnostics."""

        filters: dict[str, object] = {}
        pf_config = RotatingShieldPFConfig(
            structural_update_min_counts=25.0,
            conditional_strength_refit_min_count=5.0,
            structural_update_min_snr=2.0,
        )

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty report estimate."""
            return {}

        def unresolved_structural_evidence(self) -> dict[str, dict[str, object]]:
            """Return no open structural evidence."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return zero-source diagnostics with candidate pool but no signal."""
            return {
                "Eu-154": {
                    "candidate_count": 10,
                    "selected_count": 0,
                    "model_order_ready": True,
                    "condition_number": 1.0,
                    "criterion_margin_to_simpler": float("inf"),
                    "observed_signal_total_counts": 0.0,
                    "observed_signal_max_count": 0.0,
                    "observed_signal_snr": 0.0,
                    "count_supported_zero_source": False,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
    )

    assert ready is True
    assert reason == "model_order_ready"


def test_adaptive_mission_waits_for_model_order_readiness() -> None:
    """Mission stop should not accept low IG while model order is unresolved."""

    class _DummyEstimator:
        """Minimal estimator with unresolved report-level source count."""

        filters: dict[str, object] = {}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a quiet global IG state."""
            return True

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a quiet local shield state."""
            return True

        def report_model_order_ready(self) -> bool:
            """Return unresolved model-order readiness."""
            return False

    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]
    estimator = _DummyEstimator()

    reason = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=None,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
    )

    assert reason is None

    reason_without_guard = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=None,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_model_order_ready=False,
    )

    assert reason_without_guard == "pf_converged_low_information_gain"


def test_adaptive_mission_accepts_strong_simple_report_bic() -> None:
    """A simple report BIC should stop despite low-weight pseudo-source failures."""

    class _DummyFilter:
        """Minimal filter with pseudo failures but no residual birth support."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0
        last_pseudo_source_fail_reasons = {"needs_discriminative_views": 3}

    class _DummyEstimator:
        """Estimator whose strict readiness is false but report BIC is simple."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Populate report diagnostics without changing test state."""
            return {}

        def report_model_order_ready(self) -> bool:
            """Return strict unresolved readiness."""
            return False

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a strong one-source BIC margin."""
            return {
                "Cs-137": {
                    "selected_count": 1,
                    "criterion_margin_to_simpler": 25.0,
                    "condition_number": 1.0,
                    "selected_max_response_correlation": 0.0,
                    "model_order_ready": False,
                }
            }

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

    remaining_ready = {
        "estimated_remaining_stations": 0,
        "current_budget": 0.0,
        "unresolved_factors": ["pseudo_source_verification"],
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=[np.array([0.0, 0.0, 0.5], dtype=float)],
        map_api=None,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        remaining_measurement_estimate=remaining_ready,
    )

    assert reason == "report_simple_model_order"


def test_adaptive_mission_simple_report_waits_for_residual_budget() -> None:
    """A strong simple BIC should not stop while residual evidence remains."""

    class _DummyFilter:
        """Minimal filter without direct birth gate support."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0
        last_pseudo_source_fail_reasons: dict[str, int] = {}

    class _DummyEstimator:
        """Estimator with simple BIC but unresolved residual budget."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Populate report diagnostics without changing test state."""
            return {}

        def report_model_order_ready(self) -> bool:
            """Return strict unresolved readiness."""
            return False

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a strong one-source BIC margin."""
            return {
                "Cs-137": {
                    "selected_count": 1,
                    "criterion_margin_to_simpler": 25.0,
                    "condition_number": 1.0,
                    "selected_max_response_correlation": 0.0,
                }
            }

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

    unresolved_residual = {
        "estimated_remaining_stations": 2,
        "current_budget": 10.0,
        "unresolved_factors": ["residual"],
        "components": {
            "residual": 3.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=[np.array([0.0, 0.0, 0.5], dtype=float)],
        map_api=None,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        remaining_measurement_estimate=unresolved_residual,
    )

    assert reason is None


def test_simple_report_stop_can_ignore_only_high_surface_budget() -> None:
    """Strong simple BIC may ignore high-surface ambiguity but not residuals."""

    class _DummyFilter:
        """Minimal filter without birth residual evidence."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0

    class _DummyEstimator:
        """Estimator with a confident one-source report model."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Populate report diagnostics without side effects."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a strong simple model-order diagnostic."""
            return {
                "Cs-137": {
                    "selected_count": 1,
                    "criterion_margin_to_simpler": 30.0,
                    "condition_number": 1.0,
                    "selected_max_response_correlation": 0.0,
                }
            }

    high_surface_only = {
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 4.0,
        },
    }
    residual_unresolved = {
        "components": {
            "residual": 4.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 4.0,
        },
    }

    blocked = _report_model_order_simple_ready_for_stop(
        _DummyEstimator(),
        remaining_measurement_estimate=high_surface_only,
        allow_high_surface_ambiguity=False,
    )
    allowed = _report_model_order_simple_ready_for_stop(
        _DummyEstimator(),
        remaining_measurement_estimate=high_surface_only,
        allow_high_surface_ambiguity=True,
    )
    residual_blocked = _report_model_order_simple_ready_for_stop(
        _DummyEstimator(),
        remaining_measurement_estimate=residual_unresolved,
        allow_high_surface_ambiguity=True,
    )

    assert blocked is False
    assert allowed is True
    assert residual_blocked is False


def test_simple_report_stop_can_ignore_birth_residual_noise() -> None:
    """Strong simple report BIC may override residual-birth structural noise."""

    class _DummyFilter:
        """Minimal filter with stale residual-birth evidence."""

        last_birth_residual_gate_passed = True
        last_birth_residual_support = 3

    class _DummyEstimator:
        """Estimator with a confident one-source report model."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Populate report diagnostics without side effects."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a strong simple model-order diagnostic."""
            return {
                "Cs-137": {
                    "selected_count": 1,
                    "criterion_margin_to_simpler": 30.0,
                    "condition_number": 1.0,
                    "selected_max_response_correlation": 0.0,
                }
            }

    quiet_budget = {
        "estimated_remaining_stations": 0,
        "current_budget": 0.0,
        "unresolved_factors": [],
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    blocked_by_birth = _report_model_order_simple_ready_for_stop(
        _DummyEstimator(),
        remaining_measurement_estimate=quiet_budget,
        require_no_birth_residual=True,
        birth_residual_min_support=2,
    )
    allowed_for_report = _report_model_order_simple_ready_for_stop(
        _DummyEstimator(),
        remaining_measurement_estimate=quiet_budget,
        require_no_birth_residual=False,
        birth_residual_min_support=2,
    )

    assert blocked_by_birth is False
    assert allowed_for_report is True


def test_adaptive_mission_stop_uses_simple_report_over_birth_noise() -> None:
    """Simple report readiness should stop instead of soft-extending on birth noise."""

    class _DummyFilter:
        """Minimal filter with stale residual-birth evidence."""

        last_birth_residual_gate_passed = True
        last_birth_residual_support = 3
        last_pseudo_source_fail_reasons: dict[str, int] = {
            "needs_discriminative_views": 2,
        }

    class _DummyEstimator:
        """Estimator with strong report BIC and non-converged runtime PF."""

        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Populate report diagnostics without side effects."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a strong simple model-order diagnostic."""
            return {
                "Cs-137": {
                    "selected_count": 1,
                    "criterion_margin_to_simpler": 30.0,
                    "condition_number": 1.0,
                    "selected_max_response_correlation": 0.0,
                }
            }

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=(),
    )
    quiet_budget = {
        "estimated_remaining_stations": 0,
        "current_budget": 0.0,
        "unresolved_factors": [],
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    reason = _adaptive_mission_stop_reason(
        _DummyEstimator(),  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=[np.array([0.5, 0.5, 0.0], dtype=float)],
        map_api=grid,
        min_poses=1,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=0.5,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
        require_quiet_birth_residual=True,
        birth_residual_min_support=2,
        remaining_measurement_estimate=quiet_budget,
        require_remaining_measurement_ready=True,
        allow_report_simple_stop=True,
    )

    assert reason == "report_simple_model_order"


def test_adaptive_dss_program_length_shortens_only_when_budget_is_resolved() -> None:
    """Adaptive shield programs should shorten only after report/budget readiness."""
    cfg = DSSPPConfig(program_length=8)
    quiet_budget = {
        "estimated_remaining_stations": 0,
        "current_budget": 0.0,
        "unresolved_factors": [],
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    shortened, reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=True,
        remaining_measurement_estimate=quiet_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
    )
    residual, residual_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=True,
        remaining_measurement_estimate={
            **quiet_budget,
            "components": {**quiet_budget["components"], "residual": 1.0},
        },
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
    )

    assert int(shortened.program_length) == 2
    assert reason == "simple_report"
    assert int(residual.program_length) == 16
    assert residual_reason == "residual"

    birth_noise, birth_noise_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=True,
        report_simple_ready=True,
        remaining_measurement_estimate=quiet_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
    )

    assert int(birth_noise.program_length) == 2
    assert birth_noise_reason == "simple_report"


def test_sparse_cardinality_evidence_gap_unresolved_checks_nested_payload() -> None:
    """Sparse cardinality gaps should mark only weak evidence as unresolved."""
    unresolved = {
        "Cs-137": {
            "sparse_poisson_evidence": {
                "available": True,
                "model_order_ready": False,
                "criterion_margin_to_runner_up": 2.0,
            }
        }
    }
    weak_gap = {
        "Co-60": {
            "available": True,
            "model_order_ready": True,
            "criterion_margin_to_runner_up": 4.0,
            "criterion_margin_to_simpler": 12.0,
        }
    }
    resolved = {
        "Eu-154": {
            "available": True,
            "model_order_ready": True,
            "criterion_margin_to_runner_up": 12.0,
            "criterion_margin_to_simpler": 14.0,
        }
    }
    missing_ready = {
        "Cs-137": {
            "available": True,
            "criterion_margin_to_runner_up": 12.0,
            "criterion_margin_to_simpler": 14.0,
        }
    }

    assert sparse_cardinality_evidence_gap_unresolved(unresolved, gap_target=10.0)
    assert sparse_cardinality_evidence_gap_unresolved(weak_gap, gap_target=10.0)
    assert sparse_cardinality_evidence_gap_unresolved(missing_ready, gap_target=10.0)
    assert not sparse_cardinality_evidence_gap_unresolved(resolved, gap_target=10.0)


def test_report_model_order_ready_fallback_waits_for_diagnostics() -> None:
    """Mission-stop fallback should not treat missing diagnostics as ready."""

    class _MissingDiagnosticsEstimator:
        """Estimator exposing no model-order diagnostics."""

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return no model-order diagnostics."""
            return {}

    class _ActiveMissingReadyEstimator:
        """Estimator exposing active diagnostics with no ready flag."""

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return active diagnostics without an explicit readiness decision."""
            return {"Cs-137": {"candidate_count": 3, "selected_count": 2}}

    assert report_model_order_ready_for_stop(_MissingDiagnosticsEstimator()) is False
    assert report_model_order_ready_for_stop(_ActiveMissingReadyEstimator()) is False


def test_adaptive_dss_residual_extension_requires_cardinality_gap() -> None:
    """Residual budgets alone should not trigger long programs under the new guard."""
    cfg = DSSPPConfig(program_length=8)
    residual_budget = {
        "estimated_remaining_stations": 1,
        "current_budget": 4.0,
        "unresolved_factors": ["residual"],
        "components": {
            "residual": 1.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 0.0,
        },
    }

    blocked, blocked_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=False,
        remaining_measurement_estimate=residual_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
        residual_extension_requires_cardinality_evidence=True,
        cardinality_evidence_unresolved=False,
    )
    extended, extended_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=False,
        remaining_measurement_estimate=residual_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
        residual_extension_requires_cardinality_evidence=True,
        cardinality_evidence_unresolved=True,
    )

    assert int(blocked.program_length) == 8
    assert blocked_reason == "residual_without_cardinality_gap"
    assert int(extended.program_length) == 16
    assert extended_reason == "residual"


def test_adaptive_dss_shortens_high_surface_budget_when_report_is_simple() -> None:
    """High-surface ambiguity alone should not force full programs for simple reports."""
    cfg = DSSPPConfig(program_length=8)
    high_surface_budget = {
        "estimated_remaining_stations": 1,
        "current_budget": 4.0,
        "unresolved_factors": ["high_surface_ambiguity"],
        "components": {
            "residual": 0.0,
            "isotope_absence": 0.0,
            "same_isotope_separation": 0.0,
            "high_surface_ambiguity": 4.0,
        },
    }

    blocked, blocked_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=True,
        remaining_measurement_estimate=high_surface_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
        allow_high_surface_simple=False,
    )
    shortened, shortened_reason = _adapt_dss_program_length_for_budget(
        cfg,
        enabled=True,
        simple_program_length=2,
        residual_program_length=16,
        residual_burst_active=False,
        report_simple_ready=True,
        remaining_measurement_estimate=high_surface_budget,
        residual_budget_threshold=1.0e-9,
        ambiguity_budget_threshold=1.0e-9,
        allow_high_surface_simple=True,
    )

    assert int(blocked.program_length) == 8
    assert blocked_reason == "ambiguity"
    assert int(shortened.program_length) == 2
    assert shortened_reason == "simple_report"


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

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this min-pose test."""
            return True

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
    )

    assert reason_after_min == "pf_converged_low_information_gain"


def test_adaptive_mission_stops_when_all_filter_flags_converged() -> None:
    """Per-isotope convergence flags should stop after the guaranteed pose count."""

    class _DummyConfig:
        """Minimal convergence config for filter and estimator dummies."""

        converge_enable = True

    class _DummyFilter:
        """Minimal filter exposing the per-isotope convergence flag."""

        config = _DummyConfig()
        is_converged = True
        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0

    class _DummyEstimator:
        """Estimator whose global IG condition is not yet quiet."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter(), "Co-60": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this convergence-flag test."""
            return True

    visited = [np.array([0.5, 0.5, 0.0], dtype=float)]
    estimator = _DummyEstimator()

    assert _all_pf_filters_converged(estimator) is True  # type: ignore[arg-type]
    reason = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited,
        map_api=None,
        min_poses=8,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
    )

    assert reason is None

    reason_after_min = _adaptive_mission_stop_reason(
        estimator,  # type: ignore[arg-type]
        current_pose_idx=0,
        visited_poses_xyz=visited * 8,
        map_api=None,
        min_poses=8,
        coverage_radius_m=10.0,
        coverage_fraction_threshold=1.0,
        ig_threshold=1e-3,
        planning_live_time_s=1.0,
    )

    assert reason_after_min == "pf_filters_converged"


def test_pf_convergence_rejects_report_cardinality_collapse() -> None:
    """Mission convergence should reject a report that collapses PF cardinality."""

    class _DummyConfig:
        """Minimal convergence config for filter and estimator dummies."""

        converge_enable = True
        converge_cardinality_var_max = 0.05
        report_model_order_require_posterior_match = True

    class _DummyState:
        """Minimal state with only an active source count."""

        def __init__(self, num_sources: int) -> None:
            """Store the active source count."""
            self.num_sources = int(num_sources)

    class _DummyParticle:
        """Minimal particle wrapping a source-count state."""

        def __init__(self, num_sources: int) -> None:
            """Store the particle state."""
            self.state = _DummyState(num_sources)

    class _DummyFilter:
        """Minimal converged filter whose posterior supports three sources."""

        config = _DummyConfig()
        is_converged = True
        continuous_particles = [_DummyParticle(3), _DummyParticle(3)]
        continuous_weights = np.array([0.5, 0.5], dtype=float)

        def state_without_quarantined_sources(self, state: _DummyState) -> _DummyState:
            """Return the state unchanged for this dummy."""
            return state

    class _DummyEstimator:
        """Estimator with report model-order diagnostics collapsed to one source."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a collapsed one-source report."""
            return {
                "Cs-137": (
                    np.zeros((1, 3), dtype=float),
                    np.ones(1, dtype=float),
                )
            }

        def report_model_order_ready(self) -> bool:
            """Return ready to exercise the posterior-cardinality guard."""
            return True

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return diagnostics indicating one selected source from three candidates."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 1,
                    "model_order_ready": True,
                }
            }

    assert _all_pf_filters_converged(_DummyEstimator()) is False  # type: ignore[arg-type]


def test_pf_convergence_accepts_matching_report_cardinality() -> None:
    """Mission convergence can stop when PF and report cardinality agree."""

    class _DummyConfig:
        """Minimal convergence config for filter and estimator dummies."""

        converge_enable = True
        converge_cardinality_var_max = 0.05
        report_model_order_require_posterior_match = True

    class _DummyState:
        """Minimal state with only an active source count."""

        num_sources = 2

    class _DummyParticle:
        """Minimal particle wrapping a source-count state."""

        state = _DummyState()

    class _DummyFilter:
        """Minimal converged filter whose posterior supports two sources."""

        config = _DummyConfig()
        is_converged = True
        continuous_particles = [_DummyParticle(), _DummyParticle()]
        continuous_weights = np.array([0.5, 0.5], dtype=float)

    class _DummyEstimator:
        """Estimator with report model-order diagnostics matching two sources."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a two-source report."""
            return {
                "Cs-137": (
                    np.zeros((2, 3), dtype=float),
                    np.ones(2, dtype=float),
                )
            }

        def report_model_order_ready(self) -> bool:
            """Return ready for this dummy."""
            return True

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return diagnostics indicating two selected sources."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 2,
                    "model_order_ready": True,
                }
            }

    assert _all_pf_filters_converged(_DummyEstimator()) is True  # type: ignore[arg-type]


def test_pf_convergence_rejects_report_count_above_posterior_cardinality() -> None:
    """Mission convergence should reject report clusters unsupported by PF K-mass."""

    class _DummyConfig:
        """Minimal convergence config for filter and estimator dummies."""

        converge_enable = True
        converge_cardinality_var_max = 0.05
        report_model_order_require_posterior_match = True

    class _DummyState:
        """Minimal state with only an active source count."""

        num_sources = 1

    class _DummyParticle:
        """Minimal particle wrapping a source-count state."""

        state = _DummyState()

    class _DummyFilter:
        """Minimal converged filter whose posterior supports one source."""

        config = _DummyConfig()
        is_converged = True
        continuous_particles = [_DummyParticle(), _DummyParticle()]
        continuous_weights = np.array([0.5, 0.5], dtype=float)

    class _DummyEstimator:
        """Estimator whose report overstates posterior source cardinality."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a three-source report."""
            return {
                "Cs-137": (
                    np.zeros((3, 3), dtype=float),
                    np.ones(3, dtype=float),
                )
            }

        def report_model_order_ready(self) -> bool:
            """Return ready to exercise the posterior-cardinality guard."""
            return True

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return diagnostics indicating three selected sources."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 3,
                    "model_order_ready": True,
                }
            }

    assert _all_pf_filters_converged(_DummyEstimator()) is False  # type: ignore[arg-type]


def test_pf_convergence_can_trust_report_model_order_without_posterior_match() -> None:
    """Mission convergence can use stable BIC report order as the cardinality source."""

    class _DummyConfig:
        """Minimal config that disables the report/PF cardinality equality guard."""

        converge_enable = True
        converge_cardinality_var_max = 0.05
        report_model_order_require_posterior_match = False

    class _DummyState:
        """Minimal state with only an active source count."""

        num_sources = 3

    class _DummyParticle:
        """Minimal particle wrapping a three-source state."""

        state = _DummyState()

    class _DummyFilter:
        """Minimal converged filter whose posterior supports three sources."""

        config = _DummyConfig()
        is_converged = True
        continuous_particles = [_DummyParticle()]
        continuous_weights = np.array([1.0], dtype=float)

    class _DummyEstimator:
        """Estimator whose BIC report selects fewer sources than PF K-mass."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return the BIC-selected two-source report."""
            return {
                "Cs-137": (
                    np.zeros((2, 3), dtype=float),
                    np.ones(2, dtype=float),
                )
            }

        def report_model_order_ready(self) -> bool:
            """Return a stable report-level model order."""
            return True

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return diagnostics indicating a BIC-selected two-source report."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 2,
                    "model_order_ready": True,
                }
            }

    assert _all_pf_filters_converged(_DummyEstimator()) is True  # type: ignore[arg-type]


def test_source_cardinality_dwell_rejects_unstable_posterior_when_report_collapses() -> (
    None
):
    """Adaptive dwell should not stop when report clusters miss multisource K-mass."""

    class _DummyConfig:
        """Minimal PF config for dwell status checks."""

        converge_cardinality_var_max = 0.05
        birth_enable = True
        max_sources = 3

    class _StateOne:
        """Single-source dummy state."""

        num_sources = 1

    class _StateThree:
        """Three-source dummy state."""

        num_sources = 3

    class _ParticleOne:
        """Particle wrapper for a single-source state."""

        state = _StateOne()

    class _ParticleThree:
        """Particle wrapper for a three-source state."""

        state = _StateThree()

    class _DummyFilter:
        """Filter whose posterior is still split across source counts."""

        config = _DummyConfig()
        continuous_particles = [_ParticleOne(), _ParticleThree()]
        continuous_weights = np.array([0.5, 0.5], dtype=float)

    class _DummyEstimator:
        """Estimator with a collapsed one-source report."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a collapsed one-source estimate."""
            return {
                "Cs-137": (
                    np.zeros((1, 3), dtype=float),
                    np.ones(1, dtype=float),
                )
            }

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a one-source report despite multisource posterior mass."""
            return {
                "Cs-137": {
                    "candidate_count": 1,
                    "selected_count": 1,
                    "model_order_ready": True,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),  # type: ignore[arg-type]
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=0.0,
    )

    assert ready is False
    assert "posterior_cardinality_var" in reason


def test_source_cardinality_dwell_allows_uncapped_max_sources() -> None:
    """Adaptive dwell should treat max_sources=None as an uncapped PF."""

    class _DummyConfig:
        """Minimal uncapped PF config for dwell status checks."""

        converge_cardinality_var_max = 0.05
        birth_enable = True
        max_sources = None

    class _State:
        """Two-source dummy state."""

        num_sources = 2

    class _Particle:
        """Particle wrapper for a two-source state."""

        state = _State()

    class _DummyFilter:
        """Filter whose posterior cardinality is stable."""

        config = _DummyConfig()
        continuous_particles = [_Particle()]
        continuous_weights = np.array([1.0], dtype=float)

    class _DummyEstimator:
        """Estimator with no report-visible source and uncapped birth enabled."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return an empty estimate to initialize report diagnostics."""
            return {}

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a no-source report below the dwell candidate threshold."""
            return {
                "Cs-137": {
                    "candidate_count": 0,
                    "selected_count": 0,
                    "model_order_ready": True,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),  # type: ignore[arg-type]
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=0.0,
    )

    assert ready is True
    assert reason == "model_order_ready"


def test_source_cardinality_dwell_can_use_report_order_without_posterior_match() -> (
    None
):
    """Adaptive dwell can ignore stable PF/report K mismatch when configured."""

    class _DummyConfig:
        """Minimal PF config with report-order cardinality as the stop source."""

        converge_cardinality_var_max = 0.05
        birth_enable = True
        max_sources = None
        report_model_order_require_posterior_match = False

    class _State:
        """Three-source dummy state."""

        num_sources = 3

    class _Particle:
        """Particle wrapper for a three-source state."""

        state = _State()

    class _DummyFilter:
        """Filter whose posterior cardinality disagrees with the report."""

        config = _DummyConfig()
        continuous_particles = [_Particle()]
        continuous_weights = np.array([1.0], dtype=float)

    class _DummyEstimator:
        """Estimator with stable two-source report diagnostics."""

        pf_config = _DummyConfig()
        filters = {"Cs-137": _DummyFilter()}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Return a two-source report."""
            return {
                "Cs-137": (
                    np.zeros((2, 3), dtype=float),
                    np.ones(2, dtype=float),
                )
            }

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return a stable two-source model-order report."""
            return {
                "Cs-137": {
                    "candidate_count": 3,
                    "selected_count": 2,
                    "model_order_ready": True,
                    "condition_number": 1.0,
                    "criterion_margin_to_simpler": 10.0,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),  # type: ignore[arg-type]
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
    )

    assert ready is True
    assert reason == "model_order_ready"


def test_source_cardinality_dwell_can_skip_estimate_refresh() -> None:
    """Runtime dwell checks should be able to use cached model-order diagnostics."""

    class _DummyConfig:
        """Minimal PF config for cached report diagnostics."""

        converge_cardinality_var_max = 0.05
        report_model_order_require_posterior_match = False

    class _DummyEstimator:
        """Estimator whose heavy report refresh must not be called."""

        pf_config = _DummyConfig()
        filters: dict[str, object] = {}

        def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Fail if a cached-only dwell check refreshes final estimates."""
            raise AssertionError("estimates refresh should be skipped")

        def report_model_order_diagnostics(self) -> dict[str, dict[str, object]]:
            """Return cached no-source report diagnostics."""
            return {
                "Cs-137": {
                    "candidate_count": 0,
                    "selected_count": 0,
                    "model_order_ready": True,
                }
            }

    ready, reason = _source_cardinality_dwell_status(
        _DummyEstimator(),  # type: ignore[arg-type]
        min_candidate_count=2,
        max_condition_number=100.0,
        min_bic_margin=2.0,
        refresh_estimates=False,
    )

    assert ready is True
    assert reason == "model_order_ready"


def test_adaptive_mission_coverage_can_stop_when_birth_residuals_are_quiet() -> None:
    """Coverage can stop a mission once residual birth evidence is quiet."""

    class _DummyFilter:
        """Minimal filter state exposing quiet residual-birth diagnostics."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0

    class _DummyEstimator:
        """Minimal estimator state for adaptive mission stop tests."""

        filters = {"Cs-137": _DummyFilter()}

        def should_stop_exploration(self, **kwargs: object) -> bool:
            """Return a non-converged global exploration state."""
            return False

        def should_stop_shield_rotation(self, **kwargs: object) -> bool:
            """Return a non-converged local rotation state."""
            return False

        def report_model_order_ready(self) -> bool:
            """Return settled model-order status for this coverage-stop test."""
            return True

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
        require_quiet_birth_residual=True,
        birth_residual_min_support=2,
    )

    assert reason == "environment_coverage:1.000"


def test_adaptive_mission_coverage_can_require_pf_convergence() -> None:
    """Coverage alone should not stop a mission when convergence is required."""

    class _DummyFilter:
        """Minimal filter state exposing quiet residual-birth diagnostics."""

        last_birth_residual_gate_passed = False
        last_birth_residual_support = 0

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
        require_quiet_birth_residual=True,
        birth_residual_min_support=2,
        require_pf_convergence_for_coverage=True,
    )

    assert reason is None


def test_source_position_support_limits_candidate_grid_z() -> None:
    """Configured source support should restrict PF candidates without using truth."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=5.0)
    bounds = _resolve_source_position_bounds(
        env,
        {"source_z_min_m": 0.0, "source_z_max_m": 1.5},
    )

    grid = _build_candidate_sources(
        env,
        spacing=(1.0, 1.0, 0.5),
        margin=0.0,
        position_min=bounds[0],
        position_max=bounds[1],
    )

    assert np.min(grid[:, 2]) >= 0.0
    assert np.max(grid[:, 2]) <= 1.5


def test_demo_pf_gate_retains_all_configured_counts_for_final_evaluation(
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

    spectrum_template_isotope_sets: list[set[str]] = []

    def _fake_update_pair(
        self: RotatingShieldPFEstimator,
        z_k: dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: dict[str, float] | None = None,
        z_covariance_k: dict[str, dict[str, float]] | None = None,
        spectrum_payload: dict[str, object] | None = None,
    ) -> None:
        """Append a lightweight measurement record without GPU updates."""
        del z_covariance_k
        if spectrum_payload is not None:
            templates = spectrum_payload.get(
                "spectrum_response_templates_by_isotope",
                {},
            )
            if isinstance(templates, dict):
                spectrum_template_isotope_sets.append(set(templates))
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                z_variance_k=z_variance_k,
            )
        )

    def _fake_estimates(
        self: RotatingShieldPFEstimator,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
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

    def _fake_add_isotopes(
        self: RotatingShieldPFEstimator,
        new_isotopes: list[str],
    ) -> None:
        """Activate isotopes without building heavy kernels in this test."""
        for iso in new_isotopes:
            if iso not in self.isotopes:
                self.isotopes.append(iso)

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
        "DETECT_CONSECUTIVE_BY_ISOTOPE",
        {"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
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
    pure_estimator_type = realtime_demo.RotatingShieldPFEstimator
    monkeypatch.setattr(pure_estimator_type, "update_pair", _fake_update_pair)
    monkeypatch.setattr(pure_estimator_type, "estimates", _fake_estimates)
    monkeypatch.setattr(pure_estimator_type, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(pure_estimator_type, "add_isotopes", _fake_add_isotopes)

    estimator = run_live_pf(
        live=False,
        max_steps=None,
        max_poses=2,
        detect_threshold_abs=0.0,
        detect_threshold_rel=0.0,
        detect_consecutive=1,
        detect_min_steps=1,
        min_peaks_by_isotope={"Cs-137": 1, "Co-60": 1, "Eu-154": 1},
        ig_threshold_mode="absolute",
        ig_threshold_min=0.0,
        obstacle_layout_path=None,
        num_particles=8,
        pf_config_overrides={
            "orientation_k": 1,
            "min_particles": 8,
            "max_particles": 8,
        },
        save_outputs=False,
        return_state=True,
        nominal_motion_speed_m_s=1.0,
        rotation_overhead_s=2.0,
        measurement_log_output=str(tmp_path / "measurement-log"),
    )
    assert estimator is not None
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
    assert metrics["final_point_reporting_pipeline_time_s"] >= 0.0
    assert metrics["final_point_estimate_time_s"] >= 0.0
    assert metrics["final_point_estimate_time_s"] == pytest.approx(
        metrics["final_point_reporting_pipeline_time_s"]
    )
    assert metrics["final_mle_time_s"] >= metrics["final_point_estimate_time_s"]
    assert metrics["gpu_memory"]["available"] is False
    evaluation = estimator.final_run_summary["evaluation_metrics"]
    assert "p95" in evaluation["accuracy"]["position_error"]
    assert "by_shield_pair" in evaluation["count_bias"]
    assert "spectrum_bin_heldout_deviance" in evaluation["model_identifiability"]
    assert "consecutive_matched_cluster_shift_m" in evaluation["cluster_stability"]
    assert evaluation["operational"]["station_count"] == 2
    assert evaluation["operational"]["station_visit_count"] == 2
    assert evaluation["operational"]["online_wall_clock_s"] <= evaluation[
        "operational"
    ]["end_to_end_wall_clock_s"]
    json.dumps(estimator.final_run_summary, allow_nan=False)
    assert estimator.isotopes == list(ANALYSIS_ISOTOPES)
    for rec in estimator.measurements:
        assert set(rec.z_k) == set(ANALYSIS_ISOTOPES)
        assert rec.z_variance_k is not None
        assert set(rec.z_variance_k) == set(ANALYSIS_ISOTOPES)
        assert rec.z_variance_k["Cs-137"] == pytest.approx(2.0)
    assert planning_isotope_args
    assert all(value is None for value in planning_isotope_args)
    assert spectrum_template_isotope_sets == []
    estimates = estimator.estimates()
    positions, strengths = estimates.get("Cs-137", (np.zeros((0, 3)), np.zeros(0)))
    assert positions.size > 0
    assert strengths.size > 0


def test_final_absent_filter_removes_unsupported_isotope() -> None:
    """Final reporting should drop isotopes without count and PF support."""
    measurements = [
        MeasurementRecord(
            z_k={"Cs-137": 120.0, "Co-60": 3.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={"Cs-137": 120.0, "Co-60": 9.0},
        ),
        MeasurementRecord(
            z_k={"Cs-137": 130.0, "Co-60": 2.0},
            pose_idx=0,
            orient_idx=1,
            live_time_s=1.0,
            fe_index=1,
            pb_index=0,
            z_variance_k={"Cs-137": 130.0, "Co-60": 9.0},
        ),
    ]
    estimates = {
        "Cs-137": (
            np.array([[1.0, 2.0, 3.0]], dtype=float),
            np.array([1000.0], dtype=float),
        ),
        "Co-60": (
            np.array([[4.0, 5.0, 6.0]], dtype=float),
            np.array([1000.0], dtype=float),
        ),
        "Eu-154": (
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        ),
    }

    filtered, diagnostics = _filter_absent_final_estimates(
        estimates,
        measurements,
        enabled=True,
        count_threshold_abs=30.0,
        min_support_measurements=2,
        min_total_counts=60.0,
        snr_threshold=3.0,
        min_strength=500.0,
    )

    assert set(filtered) == {"Cs-137"}
    assert diagnostics["Cs-137"]["kept"] is True
    assert diagnostics["Co-60"]["reason"] == "insufficient_spectral_support"
    assert diagnostics["Eu-154"]["reason"] == "no_final_pf_support"


class _TinyCoverageMap:
    """Small traversable map for online absent-isotope pruning tests."""

    grid_shape = (4, 1)
    cell_size = 1.0
    origin = (0.0, 0.0)
    traversable_cells = [(0, 0), (1, 0), (2, 0), (3, 0)]

    @staticmethod
    def cell_center(cell):
        """Return the center of one map cell."""
        return np.array([float(cell[0]) + 0.5, float(cell[1]) + 0.5])


class _DummyOnlineAbsentEstimator:
    """Minimal estimator stub for online absent-isotope pruning."""

    def __init__(self) -> None:
        """Create a two-isotope estimator with only Cs count support."""
        self.isotopes = ["Cs-137", "Co-60"]
        self.measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 120.0, "Co-60": 0.5},
                pose_idx=i,
                orient_idx=0,
                live_time_s=1.0,
                fe_index=0,
                pb_index=0,
                z_variance_k={"Cs-137": 120.0, "Co-60": 1.0},
            )
            for i in range(8)
        ]
        self.restricted: list[list[str]] = []

    def restrict_isotopes(
        self,
        active_isotopes,
        *,
        allow_empty: bool = False,
    ) -> None:
        """Record and apply isotope restrictions."""
        _ = allow_empty
        self.restricted.append(list(active_isotopes))
        self.isotopes = list(active_isotopes)


def test_online_absent_pruning_waits_for_environment_coverage() -> None:
    """Online pruning should not drop unsupported isotopes before coverage."""
    estimator = _DummyOnlineAbsentEstimator()
    removed = _prune_online_absent_isotopes(
        estimator,
        enabled=True,
        detected_isotopes={"Cs-137"},
        pruned_isotopes=set(),
        visited_poses_xyz=[np.array([0.5, 0.5, 0.5])],
        map_api=_TinyCoverageMap(),
        min_poses=4,
        coverage_radius_m=0.6,
        coverage_fraction_threshold=0.75,
        min_measurements=8,
        count_threshold_abs=50.0,
        min_support_measurements=2,
        min_total_counts=100.0,
        snr_threshold=3.0,
        label="low_coverage",
    )

    assert removed == set()
    assert estimator.isotopes == ["Cs-137", "Co-60"]
    assert estimator.restricted == []


def test_online_absent_pruning_support_guard_does_not_protect_all_active() -> None:
    """Absent pruning should not treat every active isotope as newly supported."""
    protected = _online_absent_pruning_supported_isotopes(
        raw_detected={"Cs-137"},
        last_candidates=set(),
    )

    assert protected == {"Cs-137"}
    assert "Co-60" not in protected


def test_online_absent_pruning_drops_only_after_covered_and_unsupported() -> None:
    """Online pruning should remove absent isotopes after support and coverage gates."""
    estimator = _DummyOnlineAbsentEstimator()
    pruned: set[str] = set()
    visited = [
        np.array([0.5, 0.5, 0.5]),
        np.array([1.5, 0.5, 0.5]),
        np.array([2.5, 0.5, 0.5]),
        np.array([3.5, 0.5, 0.5]),
    ]

    removed = _prune_online_absent_isotopes(
        estimator,
        enabled=True,
        detected_isotopes={"Cs-137"},
        pruned_isotopes=pruned,
        visited_poses_xyz=visited,
        map_api=_TinyCoverageMap(),
        min_poses=4,
        coverage_radius_m=0.6,
        coverage_fraction_threshold=0.75,
        min_measurements=8,
        count_threshold_abs=50.0,
        min_support_measurements=2,
        min_total_counts=100.0,
        snr_threshold=3.0,
        label="covered",
    )

    assert removed == {"Co-60"}
    assert pruned == {"Co-60"}
    assert estimator.isotopes == ["Cs-137"]
    assert estimator.restricted == [["Cs-137"]]


def test_online_absent_pruning_drops_low_snr_cumulative_crosstalk() -> None:
    """Cumulative low-SNR isotope leakage should not protect an absent filter."""

    class _DummyLowSnrEstimator:
        """Estimator with many low-SNR Eu leakage measurements."""

        def __init__(self) -> None:
            """Create many measurements whose Eu total count exceeds the count floor."""
            self.isotopes = ["Cs-137", "Eu-154"]
            self.measurements = [
                MeasurementRecord(
                    z_k={"Cs-137": 120.0, "Eu-154": 1.2},
                    pose_idx=i,
                    orient_idx=0,
                    live_time_s=1.0,
                    fe_index=0,
                    pb_index=0,
                    z_variance_k={"Cs-137": 120.0, "Eu-154": 1.0e6},
                )
                for i in range(120)
            ]
            self.restricted: list[list[str]] = []

        def restrict_isotopes(
            self,
            active_isotopes,
            *,
            allow_empty: bool = False,
        ) -> None:
            """Record and apply isotope restrictions."""
            _ = allow_empty
            self.restricted.append(list(active_isotopes))
            self.isotopes = list(active_isotopes)

    estimator = _DummyLowSnrEstimator()
    visited = [
        np.array([0.5, 0.5, 0.5]),
        np.array([1.5, 0.5, 0.5]),
        np.array([2.5, 0.5, 0.5]),
        np.array([3.5, 0.5, 0.5]),
    ]

    removed = _prune_online_absent_isotopes(
        estimator,  # type: ignore[arg-type]
        enabled=True,
        detected_isotopes={"Cs-137"},
        pruned_isotopes=set(),
        visited_poses_xyz=visited,
        map_api=_TinyCoverageMap(),
        min_poses=4,
        coverage_radius_m=0.6,
        coverage_fraction_threshold=0.75,
        min_measurements=8,
        count_threshold_abs=50.0,
        min_support_measurements=2,
        min_total_counts=100.0,
        snr_threshold=3.0,
        label="low_snr_crosstalk",
    )

    assert removed == {"Eu-154"}
    assert estimator.isotopes == ["Cs-137"]
    assert estimator.restricted == [["Cs-137"]]


def test_online_absent_pruning_keeps_newly_detected_isotope() -> None:
    """Online pruning should protect isotopes with current spectral support."""
    estimator = _DummyOnlineAbsentEstimator()
    visited = [
        np.array([0.5, 0.5, 0.5]),
        np.array([1.5, 0.5, 0.5]),
        np.array([2.5, 0.5, 0.5]),
        np.array([3.5, 0.5, 0.5]),
    ]

    removed = _prune_online_absent_isotopes(
        estimator,
        enabled=True,
        detected_isotopes={"Cs-137", "Co-60"},
        pruned_isotopes=set(),
        visited_poses_xyz=visited,
        map_api=_TinyCoverageMap(),
        min_poses=4,
        coverage_radius_m=0.6,
        coverage_fraction_threshold=0.75,
        min_measurements=8,
        count_threshold_abs=50.0,
        min_support_measurements=2,
        min_total_counts=100.0,
        snr_threshold=3.0,
        label="detected_guard",
    )

    assert removed == set()
    assert estimator.isotopes == ["Cs-137", "Co-60"]
    assert estimator.restricted == []


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


def test_prune_missing_isotope_does_not_zero_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing isotope keys should not be treated as zero-count measurements."""
    isotopes = ["Cs-137", "Co-60"]
    pf_conf = RotatingShieldPFConfig(
        num_particles=4, min_particles=4, max_particles=4, use_gpu=False
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=np.zeros((1, 3), dtype=float),
        shield_normals=None,
        mu_by_isotope=None,
        pf_config=pf_conf,
    )
    estimator.poses = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    estimator.measurements = [
        MeasurementRecord(
            z_k={"Cs-137": 10.0},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
        ),
        MeasurementRecord(
            z_k={"Cs-137": 12.0, "Co-60": 5.0},
            pose_idx=1,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
        ),
    ]

    def _fake_estimates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a deterministic estimate for the pruning regression."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([100.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in isotopes}

    monkeypatch.setattr(estimator, "estimates", _fake_estimates)
    keep_masks = prune_spurious_sources_continuous(
        estimator,
        method="deltaLL",
        params={"deltaLL_min": 1e9},
        min_support=2,
    )
    assert keep_masks["Co-60"].size == 1
    assert keep_masks["Co-60"].all()


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

    counts, variances, detected = _evaluate_spectrum_counts(
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

    assert counts == {"Cs-137": 100.0}
    assert variances == {"Cs-137": 10.0}
    assert detected == {"Cs-137"}


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

    counts, variances, detected = _evaluate_spectrum_counts(
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

    assert counts["Cs-137"] == pytest.approx(1000.0)
    assert variances["Cs-137"] == pytest.approx(40000.0)
    assert detected == {"Cs-137"}


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


def test_height_partner_reoptimizes_shield_program_by_default() -> None:
    """Height changes should not force the prior station's shield program."""
    pair_ids = (3, 4, 6, 13, 17, 36, 43, 52)

    assert realtime_demo_module._height_partner_program_for_scoring(
        reuse_enabled=False,
        executed_pair_ids=pair_ids,
        baseline_shield_policy=None,
    ) is None
    assert realtime_demo_module._height_partner_program_for_scoring(
        reuse_enabled=True,
        executed_pair_ids=pair_ids,
        baseline_shield_policy=None,
    ) == pair_ids


def test_operational_station_metrics_use_recorded_poses_and_planner_tolerances() -> None:
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
            orient_idx=index,
            live_time_s=1.0,
            detector_position_xyz_m=position,
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
    assert metrics["detector_pose_station_count"] == metrics[
        "unique_xyz_action_count"
    ]
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


def test_surface_map_runtime_configuration_is_explicit_and_validated() -> None:
    """Surface-map spacing and L1/TV weights should map to the public solver."""
    runtime_config = {
        "surface_map_spacing_m": [0.5, 1.0, 2.0],
        "surface_map_l1_weight": 0.25,
        "surface_map_tv_weight": 0.5,
        "surface_map_nuisance_l1_weight": 0.1,
        "surface_map_nuisance_l2_weight": 0.2,
        "surface_map_max_iterations": 75,
        "surface_map_max_spectrum_bins": 12,
    }

    spacing = realtime_demo_module._surface_map_spacing_from_runtime_config(
        runtime_config
    )
    config = realtime_demo_module._surface_map_config_from_runtime_config(
        runtime_config
    )

    assert spacing == pytest.approx((0.5, 1.0, 2.0))
    assert config.l1_weight == pytest.approx(0.25)
    assert config.tv_weight == pytest.approx(0.5)
    assert config.nuisance_l1_weight == pytest.approx(0.1)
    assert config.nuisance_l2_weight == pytest.approx(0.2)
    assert config.max_iterations == 75
    assert config.max_spectrum_bins == 12
    with pytest.raises(ValueError, match="surface_map_spacing_m"):
        realtime_demo_module._surface_map_spacing_from_runtime_config(
            {"surface_map_spacing_m": [1.0, 0.0, 1.0]}
        )


def test_final_surface_map_is_opt_in() -> None:
    """The memory-intensive full spectral map should remain an explicit action."""
    payload = realtime_demo_module._fit_final_surface_map(
        object(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        {},
        obstacle_height_m=1.0,
    )

    assert payload == {
        "enabled": False,
        "available": False,
        "reason": "disabled",
        "fit_time_s": None,
        "solver_time_s": None,
        "attempt_time_s": 0.0,
    }


def test_final_surface_map_checks_response_memory_before_fitting() -> None:
    """The spectral response tensor should not be allocated beyond its cap."""

    class _GuardedEstimator:
        """Expose enough history to trigger the pre-allocation memory guard."""

        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=tuple(np.ones(10, dtype=float)),
            )
        ]

        def fit_surface_map(self, *args: object, **kwargs: object) -> object:
            """Fail if the guarded solver is called after exceeding the cap."""
            raise AssertionError("surface-map solver should not be called")

    payload = realtime_demo_module._fit_final_surface_map(
        _GuardedEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 2.0,
            "surface_map_max_response_elements": 1,
        },
        obstacle_height_m=1.0,
    )

    assert payload["available"] is False
    assert payload["reason"] == "response_memory_budget_exceeded"
    assert payload["fit_time_s"] is None
    assert payload["solver_time_s"] is None
    assert payload["attempt_time_s"] >= 0.0
    assert payload["estimated_response_elements"] > 1
    assert payload["estimated_peak_response_elements"] == (
        payload["estimated_response_elements"]
        * payload["response_peak_array_multiplier"]
    )


def test_final_surface_map_memory_guard_counts_pruned_configured_isotopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response guard should count all configured, not only active, isotopes."""

    class _PrunedEstimator:
        """Expose one active filter from a three-isotope configured analysis."""

        all_isotopes = ["Cs-137", "Co-60", "Eu-154"]
        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=(1.0,),
            )
        ]

        def fit_surface_map(self, *args: object, **kwargs: object) -> object:
            """Fail if the all-isotope memory guard underestimates the tensor."""
            raise AssertionError("surface-map solver should not be called")

    def _fixed_patch_count(*_args: object, **_kwargs: object) -> int:
        """Return a small deterministic preflight patch count."""
        return 2

    monkeypatch.setattr(
        realtime_demo_module,
        "estimate_surface_patch_count_upper_bound",
        _fixed_patch_count,
    )

    payload = realtime_demo_module._fit_final_surface_map(
        _PrunedEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 2.0,
            "surface_map_max_response_elements": 10,
        },
        obstacle_height_m=1.0,
    )

    assert payload["reason"] == "response_memory_budget_exceeded"
    assert payload["isotope_count_for_memory_guard"] == 3
    assert payload["estimated_response_elements"] == 6
    assert payload["estimated_peak_response_elements"] == 24


def test_final_surface_map_memory_guard_uses_configured_spectrum_bin_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight response sizing should match the surface solver's bin cap."""

    class _CappedEstimator:
        """Expose one ten-bin spectrum for a capped preflight estimate."""

        all_isotopes = ["Cs-137"]
        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=tuple(np.ones(10, dtype=float)),
            )
        ]

        def fit_surface_map(self, *args: object, **kwargs: object) -> object:
            """Fail if the capped response guard does not stop the solver."""
            raise AssertionError("surface-map solver should not be called")

    def _fixed_patch_count(*_args: object, **_kwargs: object) -> int:
        """Return a deterministic preflight patch count."""
        return 2

    monkeypatch.setattr(
        realtime_demo_module,
        "estimate_surface_patch_count_upper_bound",
        _fixed_patch_count,
    )

    payload = realtime_demo_module._fit_final_surface_map(
        _CappedEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 2.0,
            "surface_map_max_spectrum_bins": 2,
            "surface_map_max_response_elements": 1,
        },
        obstacle_height_m=1.0,
    )

    assert payload["reason"] == "response_memory_budget_exceeded"
    assert payload["raw_maximum_spectrum_bin_count"] == 10
    assert payload["maximum_spectrum_bin_count_for_memory_guard"] == 2
    assert payload["estimated_response_elements"] == 4


def test_final_surface_map_checks_patch_count_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An excessively fine surface grid should be rejected before allocation."""

    class _GuardedEstimator:
        """Expose spectral history for the pre-construction patch guard."""

        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=(1.0,),
            )
        ]

    def _fail_patch_build(*_args: object, **_kwargs: object) -> object:
        """Fail if patch-array construction is reached after the count guard."""
        raise AssertionError("surface patches should not be constructed")

    monkeypatch.setattr(
        realtime_demo_module,
        "build_surface_patch_dictionary",
        _fail_patch_build,
    )
    payload = realtime_demo_module._fit_final_surface_map(
        _GuardedEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 0.1,
            "surface_map_max_patch_count": 10,
            "surface_map_max_response_elements": 1_000_000,
        },
        obstacle_height_m=1.0,
    )

    assert payload["available"] is False
    assert payload["reason"] == "patch_memory_budget_exceeded"
    assert payload["fit_time_s"] is None
    assert payload["solver_time_s"] is None
    assert payload["attempt_time_s"] >= 0.0
    assert payload["estimated_patch_count_upper_bound"] > 10


def test_final_surface_map_reports_memory_error_as_json_safe_unavailable() -> None:
    """Late allocation failures should become serializable unavailable payloads."""

    class _MemoryErrorEstimator:
        """Raise the allocation error that can occur inside response construction."""

        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=(1.0, 2.0),
            )
        ]

        def fit_surface_map(self, *args: object, **kwargs: object) -> object:
            """Simulate a NumPy allocation failure after passing both guards."""
            del args, kwargs
            raise MemoryError

    payload = realtime_demo_module._fit_final_surface_map(
        _MemoryErrorEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 1.0,
            "surface_map_max_patch_count": 100,
            "surface_map_max_response_elements": 1_000,
        },
        obstacle_height_m=1.0,
    )

    assert payload["enabled"] is True
    assert payload["available"] is False
    assert payload["reason"] == "surface_map_memory_error"
    assert payload["error"] == "memory_allocation_failed"
    assert payload["fit_time_s"] is None
    assert payload["solver_time_s"] is not None
    assert payload["attempt_time_s"] >= payload["solver_time_s"]
    assert json.loads(json.dumps(payload)) == payload


def test_final_surface_map_reports_successful_fit_time() -> None:
    """A completed all-history fit should expose finite wall-clock duration."""

    class _TimedEstimator:
        """Return a minimal successful surface-map payload."""

        isotopes = ["Cs-137"]
        measurements = [
            MeasurementRecord(
                z_k={"Cs-137": 1.0},
                pose_idx=0,
                orient_idx=0,
                live_time_s=1.0,
                spectrum_counts=(1.0,),
            )
        ]

        def fit_surface_map(self, *args: object, **kwargs: object) -> object:
            """Return the successful result timed by the runtime wrapper."""
            del args, kwargs
            return {"available": True, "reason": "ok"}

    payload = realtime_demo_module._fit_final_surface_map(
        _TimedEstimator(),  # type: ignore[arg-type]
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        {
            "surface_map_reconstruction_enable": True,
            "surface_map_spacing_m": 1.0,
            "surface_map_max_patch_count": 100,
            "surface_map_max_response_elements": 1_000,
        },
        obstacle_height_m=1.0,
    )

    assert payload["available"] is True
    assert np.isfinite(payload["fit_time_s"])
    assert payload["fit_time_s"] >= 0.0
    assert payload["solver_time_s"] == pytest.approx(payload["fit_time_s"])
    assert payload["attempt_time_s"] >= payload["solver_time_s"]


def test_count_error_model_reports_three_distinct_layers() -> None:
    """Final diagnostics should not collapse bias and model mismatch into variance."""
    config = RotatingShieldPFConfig(
        measurement_scale_by_isotope={"Cs-137": 1.01},
        measurement_scale_by_isotope_and_pair={"Cs-137": {3: 0.99}},
        sparse_poisson_spectral_nuisance_enable=True,
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
