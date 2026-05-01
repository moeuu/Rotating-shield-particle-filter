"""Regression coverage for isotope locking and missing-measurement handling."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from measurement.model import EnvironmentConfig
from pf.estimator import MeasurementRecord, RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.mixing import prune_spurious_sources_continuous
from realtime_demo import (
    ADAPTIVE_STEP_ID_STRIDE,
    _acquire_spectrum_observation,
    _adaptive_mission_stop_reason,
    _build_candidate_sources,
    _build_robot_path_segment,
    _compute_shield_selection_grid,
    _evaluate_spectrum_counts,
    _filter_reachable_candidates,
    _has_birth_residual_evidence,
    _inflate_low_signal_variances,
    _is_adaptive_spectrum_ready,
    _isotope_count_balance_penalty,
    _select_best_pair_from_scores,
    _signature_vector_is_dependent,
    _resolve_source_position_bounds,
    _spectrum_config_from_runtime_config,
    run_live_pf,
)
from sim import SimulationCommand, SimulationObservation
from spectrum.library import ANALYSIS_ISOTOPES
from spectrum.pipeline import SpectralDecomposer


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


def test_demo_spectrum_counts_keep_all_isotopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure demo measurements keep all isotope keys after detection locking."""
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

    def _fake_estimates(self: RotatingShieldPFEstimator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return a non-empty estimate for each isotope."""
        positions = np.array([[0.5, 0.5, 0.5]], dtype=float)
        strengths = np.array([1.0], dtype=float)
        return {iso: (positions.copy(), strengths.copy()) for iso in ANALYSIS_ISOTOPES}

    def _fake_sim(self: SpectralDecomposer, *args: object, **kwargs: object) -> tuple[np.ndarray, None]:
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

    def _forbidden_restrict(
        self: RotatingShieldPFEstimator,
        active_isotopes: list[str],
    ) -> None:
        """Fail if runtime detection tries to remove isotope filters."""
        raise AssertionError("Detection must not restrict runtime isotopes.")

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
    monkeypatch.setattr(realtime_demo, "select_next_pose_from_candidates", _fake_next_pose)
    monkeypatch.setattr(SpectralDecomposer, "simulate_spectrum", _fake_sim)
    monkeypatch.setattr(SpectralDecomposer, "isotope_counts_with_detection", _fake_counts)
    monkeypatch.setattr(RotatingShieldPFEstimator, "update_pair", _fake_update_pair)
    monkeypatch.setattr(RotatingShieldPFEstimator, "estimates", _fake_estimates)
    monkeypatch.setattr(RotatingShieldPFEstimator, "_gpu_enabled", _fake_gpu_enabled)
    monkeypatch.setattr(RotatingShieldPFEstimator, "restrict_isotopes", _forbidden_restrict)

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
        for iso in ANALYSIS_ISOTOPES:
            assert iso in rec.z_k
            assert rec.z_variance_k is not None
            assert rec.z_variance_k[iso] == pytest.approx(2.0)
    assert planning_isotope_args
    assert all(value is None for value in planning_isotope_args)
    estimates = estimator.estimates()
    for iso in ANALYSIS_ISOTOPES:
        positions, strengths = estimates.get(iso, (np.zeros((0, 3)), np.zeros(0)))
        assert positions.size > 0
        assert strengths.size > 0


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
            "dead_time_tau_s": 0.0,
        }
    )

    assert config.response_poisson_photopeak_anchor is False
    assert config.response_poisson_photopeak_anchor_weight == pytest.approx(0.5)
    assert config.response_poisson_model_mismatch_variance_scale == pytest.approx(2.0)
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


def test_prune_missing_isotope_does_not_zero_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing isotope keys should not be treated as zero-count measurements."""
    isotopes = ["Cs-137", "Co-60"]
    pf_conf = RotatingShieldPFConfig(num_particles=4, min_particles=4, max_particles=4, use_gpu=False)
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
            spectrum_count_method="photopeak_nnls",
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
    assert observation.metadata["source_equivalent_counts_Cs-137"] == pytest.approx(120.0)
    assert observation.metadata["transport_detected_counts_Cs-137"] == pytest.approx(160.0)
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
            spectrum_count_method="photopeak_nnls",
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


def test_adaptive_dwell_keeps_collecting_when_projected_live_time_is_reasonable() -> None:
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


def test_effective_entries_add_count_variance_floor(monkeypatch: pytest.MonkeyPatch) -> None:
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
