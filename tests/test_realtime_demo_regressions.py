"""Regression coverage for isotope locking and missing-measurement handling."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from measurement.model import EnvironmentConfig
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.likelihood import expected_counts_per_source
from pf.mixing import prune_spurious_sources_continuous
from pf.particle_filter import IsotopeParticle, MeasurementData
from pf.state import IsotopeState
from realtime_demo import (
    ADAPTIVE_STEP_ID_STRIDE,
    DeferredPFVisualizer,
    _acquire_spectrum_observation,
    _adapt_dss_program_length_for_budget,
    _adaptive_mission_stop_reason,
    _all_pf_filters_converged,
    _argv_requests_cui,
    _build_candidate_sources,
    _build_intermediate_estimate_trace_payload,
    _build_robot_path_segment,
    _compute_shield_selection_grid,
    _diagnostic_detail_order,
    _evaluate_spectrum_counts,
    _filter_absent_final_estimates,
    _filter_reachable_candidates,
    _format_estimate_trace_log_line,
    _format_truth_coverage_log_line,
    _final_model_order_status,
    _has_birth_residual_evidence,
    _has_unresolved_discriminative_pseudo_failures,
    _inflate_low_signal_variances,
    _is_adaptive_spectrum_ready,
    _isotope_count_balance_penalty,
    _pf_obstacle_attenuation_enabled,
    _pf_obstacle_grid_for_runtime,
    _resolve_ig_workers,
    _resolve_mission_max_poses,
    _resolve_plot_save_interval,
    _resolve_python_worker_count,
    _resolve_cui_split_view_enabled,
    _resolve_display_prune_refresh_interval,
    _resolve_structural_trial_parallelism,
    _resolve_station_update_modes,
    _particle_surface_diagnostics,
    _report_model_order_simple_ready_for_stop,
    _select_best_pair_from_scores,
    _should_refresh_display_pruned_estimates,
    _signature_vector_is_dependent,
    _resolve_source_position_bounds,
    _spectrum_config_from_runtime_config,
    _source_cardinality_dwell_status,
    run_live_pf,
)
from planning.dss_pp import DSSPPConfig
from sim import SimulationCommand, SimulationObservation
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer
from visualization.realtime_viz import PFFrame


def test_cli_max_poses_overrides_runtime_config_pose_cap() -> None:
    """An explicit CLI pose cap should not be overwritten by runtime config."""
    runtime_config = {"mission_stop_max_poses": 10}

    assert _resolve_mission_max_poses(8, runtime_config) == 8
    assert _resolve_mission_max_poses(None, runtime_config) == 10


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
    assert _resolve_station_update_modes(
        {"joint_observation_update": True, "delayed_resample_update": True}
    ) == (True, False)


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


def test_source_cardinality_dwell_rejects_unstable_posterior_when_report_collapses() -> None:
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


def test_source_cardinality_dwell_can_use_report_order_without_posterior_match() -> None:
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


def test_demo_spectrum_counts_use_detected_isotopes_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure demo PF measurements include only spectrum-confirmed isotopes."""
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

    def _fake_update_pair(
        self: RotatingShieldPFEstimator,
        z_k: dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: dict[str, float] | None = None,
    ) -> None:
        """Append a lightweight measurement record without GPU updates."""
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
    monkeypatch.setattr(RotatingShieldPFEstimator, "update_pair", _fake_update_pair)
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _fake_estimates)
    monkeypatch.setattr(RotatingShieldPFEstimator, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(RotatingShieldPFEstimator, "add_isotopes", _fake_add_isotopes)

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
        pf_config_overrides={"orientation_k": 1},
        save_outputs=False,
        return_state=True,
        nominal_motion_speed_m_s=1.0,
        rotation_overhead_s=2.0,
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
    for rec in estimator.measurements:
        assert set(rec.z_k) == {"Cs-137"}
        assert rec.z_variance_k is not None
        assert set(rec.z_variance_k) == {"Cs-137"}
        assert rec.z_variance_k["Cs-137"] == pytest.approx(2.0)
    assert planning_isotope_args
    assert all(value is None for value in planning_isotope_args)
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
            "response_poisson_model_mismatch_variance_scale": 2.0,
            "response_poisson_crosstalk_corr_threshold": 0.9,
            "response_poisson_diagnostic_reduced_chi2_threshold": 3.0,
            "dead_time_tau_s": 0.0,
        }
    )

    assert config.response_poisson_photopeak_anchor is False
    assert config.response_poisson_photopeak_anchor_weight == pytest.approx(0.5)
    assert config.response_poisson_model_mismatch_variance_scale == pytest.approx(2.0)
    assert config.response_poisson_crosstalk_corr_threshold == pytest.approx(0.9)
    assert config.response_poisson_diagnostic_reduced_chi2_threshold == pytest.approx(
        3.0
    )
    assert config.dead_time_tau_s == pytest.approx(0.0)


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


def test_effective_entries_add_count_variance_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted effective entries should soften high-count PF observations."""
    decomposer = SpectralDecomposer()
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
