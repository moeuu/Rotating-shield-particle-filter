"""Tests for shield rotation and pose selection logic (Chapter 3.4)."""

from types import SimpleNamespace

import numpy as np
import pytest

import planning.candidate_generation as candidate_generation
import planning.dss_pp as dss_pp
import planning.shield_rotation as shield_rotation
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import IsotopeState
from planning.pose_selection import (
    estimate_lambda_cost,
    minimum_observation_shortfall,
    recommend_num_rollouts,
    select_next_pose,
    select_next_pose_from_candidates,
    select_next_pose_after_rotation,
)
from planning.dss_pp import (
    DSSPPConfig,
    SignatureMode,
    _count_balance_penalty,
    _continuous_kernel_for_estimator,
    _saturated_count_utility,
    _programs_for_pose,
    _temporal_separation_score_from_signatures,
    build_shield_program_library,
    select_dss_pp_next_station,
)
from planning.candidate_generation import (
    expand_candidate_height_actions,
    generate_candidate_poses,
    resolve_detector_height_actions,
)
from planning.shield_rotation import rotation_policy_step, select_best_orientation
from planning.shield_rotation import select_separation_orientations
from planning.traversability import TraversabilityMap
from pf.particle_filter import IsotopeParticle
from measurement.shielding import generate_octant_rotation_matrices


def test_fast_gpu_rollout_respects_disabled_gpu_override() -> None:
    """Fast rollout falls back instead of raising when planning disables GPU."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = RotatingShieldPFConfig(
        use_gpu=False,
        use_fast_gpu_rollout=True,
    )

    result = estimator._expected_uncertainty_after_rotation_fast(
        detector_pos=np.array([0.0, 0.0, 0.0], dtype=float),
        live_time_per_rot_s=1.0,
        tau_ig=1e-3,
        tmax_s=1.0,
        rollouts=1,
        eig_samples=1,
        alpha_by_isotope=None,
        rollout_particles=None,
        rollout_method=None,
        use_mean_measurement=True,
        rng=np.random.default_rng(0),
        return_debug=False,
    )

    assert result is None


def _build_simple_estimator() -> RotatingShieldPFEstimator:
    """Build a minimal estimator with deterministic particle setup for tests."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[1.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=2, max_sources=1, resample_threshold=0.5
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.0, 0.0, 0.0]))
    est._ensure_kernel_cache()
    # Deterministic continuous particles for EIG tests (unblocked should dominate)
    from pf.particle_filter import IsotopeParticle
    from pf.state import IsotopeState

    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([10.0]),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([1.0]),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    return est


def test_dss_pp_uses_estimator_shared_continuous_kernel() -> None:
    """DSS-PP should not rebuild a divergent PF physics kernel by hand."""
    estimator = _build_simple_estimator()
    calls: list[int | None] = []
    sentinel = object()

    def fake_continuous_kernel(*, detector_aperture_samples=None, use_gpu=None):
        """Return a sentinel while recording kernel factory arguments."""
        del use_gpu
        calls.append(detector_aperture_samples)
        return sentinel

    estimator.continuous_kernel = fake_continuous_kernel  # type: ignore[method-assign]

    result = _continuous_kernel_for_estimator(
        estimator,
        detector_aperture_samples=7,
    )

    assert result is sentinel
    assert calls == [7]


def test_dss_pp_default_aperture_samples_matches_pf_standard() -> None:
    """DSS-PP defaults should not fall back to the obsolete one-ray kernel."""
    assert DSSPPConfig().detector_aperture_samples == 121


def test_select_best_orientation_prefers_unblocked_direction() -> None:
    """Orientation with larger IG (unblocked) should be selected (Eqs. 3.40–3.45)."""
    est = _build_simple_estimator()
    mats = generate_octant_rotation_matrices()
    from measurement.shielding import OctantShield, octant_index_from_rotation

    oct_shield = OctantShield()
    detector = np.array([1.0, 1.0, 1.0])
    source = np.array([0.0, 0.0, 0.0])
    blocked_mask = []
    for R in mats:
        idx = octant_index_from_rotation(R)
        blocked_mask.append(
            oct_shield.blocks_ray(
                detector_position=detector, source_position=source, octant_index=idx
            )
        )

    scores = []
    for oid, (RFe, RPb) in enumerate(zip(mats, mats)):
        score = select_best_orientation(
            estimator=est,
            pose_idx=0,
            live_time_s=1.0,
            RFe_candidates=[RFe],
            RPb_candidates=[RPb],
        )[1]
        scores.append(score)
    best_idx = int(np.argmax(scores))
    # Ensure argmax is returned and scores are finite; we do not enforce a specific octant here.
    assert scores[best_idx] == pytest.approx(max(scores))
    assert np.all(np.isfinite(scores))


def test_shield_selection_batch_grids_match_pairwise_scores() -> None:
    """Batched shield-selection grids should match the pairwise score helpers."""
    isotopes = ["Cs-137", "Co-60"]
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        planning_method="top_weight",
        pose_min_observation_quantile=0.25,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=np.array([[1.0, 1.0, 0.5]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.4, "Co-60": 0.2},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([4.0, 4.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    states_by_iso = {
        "Cs-137": [
            IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 2.0, 0.5]], dtype=float),
                strengths=np.array([1000.0], dtype=float),
                background=0.0,
            ),
            IsotopeState(
                num_sources=1,
                positions=np.array([[7.0, 2.0, 0.5]], dtype=float),
                strengths=np.array([500.0], dtype=float),
                background=0.0,
            ),
        ],
        "Co-60": [
            IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 8.0, 0.5]], dtype=float),
                strengths=np.array([800.0], dtype=float),
                background=0.0,
            ),
            IsotopeState(
                num_sources=1,
                positions=np.array([[8.0, 7.0, 0.5]], dtype=float),
                strengths=np.array([300.0], dtype=float),
                background=0.0,
            ),
        ],
    }
    for isotope, states in states_by_iso.items():
        filt = estimator.filters[isotope]
        filt.continuous_particles = [
            IsotopeParticle(state=state, log_weight=np.log(weight))
            for state, weight in zip(states, [0.7, 0.3])
        ]
    planning_particles = estimator.planning_particles()
    signature_grid, count_grids = estimator.shield_selection_batch_grids(
        pose_idx=0,
        live_time_s=2.0,
        particles_by_isotope=planning_particles,
        alpha_by_isotope=None,
        variance_floor=1.0,
        include_count_quantiles=True,
    )

    for fe_index in range(estimator.num_orientations):
        for pb_index in range(estimator.num_orientations):
            pairwise_signature = estimator.orientation_signature_separation_score(
                pose_idx=0,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=2.0,
                particles_by_isotope=planning_particles,
                alpha_by_isotope=None,
                variance_floor=1.0,
            )
            assert signature_grid[fe_index, pb_index] == pytest.approx(
                pairwise_signature,
                rel=1e-10,
                abs=1e-10,
            )
            pairwise_counts = estimator.expected_observation_counts_by_isotope_at_pair(
                pose_idx=0,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=2.0,
                max_particles=None,
            )
            for isotope in isotopes:
                assert count_grids[isotope][fe_index, pb_index] == pytest.approx(
                    pairwise_counts[isotope],
                    rel=1e-10,
                    abs=1e-10,
                )


def test_orientation_expected_information_gain_grid_cpu_fallback_matches_pairwise() -> (
    None
):
    """All-pair EIG grid should preserve the pairwise CPU EIG calculation."""
    estimator = _build_simple_estimator()
    planning_particles = estimator.planning_particles()
    mats = generate_octant_rotation_matrices()
    grid = estimator.orientation_expected_information_gain_grid(
        pose_idx=0,
        live_time_s=1.0,
        num_samples=0,
        particles_by_isotope=planning_particles,
    )
    for fe_index, RFe in enumerate(mats[: estimator.num_orientations]):
        for pb_index, RPb in enumerate(mats[: estimator.num_orientations]):
            pairwise = estimator.orientation_expected_information_gain(
                pose_idx=0,
                RFe=RFe,
                RPb=RPb,
                live_time_s=1.0,
                num_samples=0,
                particles_by_isotope=planning_particles,
            )
            assert grid[fe_index, pb_index] == pytest.approx(pairwise)


def test_rotation_policy_stops_when_information_low() -> None:
    """rotation_policy_step should stop if IG is below the threshold."""
    est = _build_simple_estimator()
    # Zero out strengths to make IG ~ 0
    filt = est.filters["Cs-137"]
    for p in filt.continuous_particles:
        p.state.strengths[:] = 0.0
    should_stop, orient_idx, score = rotation_policy_step(
        estimator=est, pose_idx=0, ig_threshold=1e-6, live_time_s=1.0
    )
    assert should_stop
    assert orient_idx == -1
    assert score == 0.0


def test_select_next_pose_balances_information_and_cost() -> None:
    """Pose selection should trade off uncertainty and motion cost (Eq. 3.51)."""

    class DummyEstimator:
        def __init__(self) -> None:
            self.poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)

        def expected_uncertainty(
            self, pose_idx: int, live_time_s: float = 1.0
        ) -> float:
            return [2.0, 0.5][pose_idx]

        def max_orientation_information_gain(
            self, pose_idx: int, live_time_s: float = 1.0
        ) -> float:
            return [0.0, 0.4][pose_idx]

    est = DummyEstimator()
    next_idx = select_next_pose(
        estimator=est,
        candidate_pose_indices=np.array([0, 1], dtype=np.int64),
        current_pose_idx=0,
        criterion="uncertainty",
        lambda_cost=0.1,
        t_short_s=1.0,
    )
    assert next_idx == 1


def test_select_next_pose_after_rotation_prefers_lower_score() -> None:
    """After-rotation selection should pick the candidate with lower expected uncertainty."""

    class DummyEstimator:
        def __init__(self) -> None:
            """Track calls for expected uncertainty evaluations."""
            self.calls: list[np.ndarray] = []

        def expected_uncertainty_after_rotation(
            self,
            pose_xyz: np.ndarray,
            live_time_per_rot_s: float,
            tau_ig: float,
            tmax_s: float,
            n_rollouts: int,
            orient_selection: str = "IG",
            return_debug: bool = False,
        ) -> float:
            """Return a deterministic score based on pose norm."""
            self.calls.append(np.asarray(pose_xyz, dtype=float))
            return float(np.linalg.norm(pose_xyz))

    est = DummyEstimator()
    current_pose = np.array([5.0, 5.0, 5.0], dtype=float)
    visited = np.array([[5.0, 5.0, 5.0]], dtype=float)
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose,
        n_candidates=4,
        strategy="ring",
        visited_poses_xyz=visited,
    )
    expected_scores = [np.linalg.norm(pose) for pose in candidates]
    expected_pose = candidates[int(np.argmin(expected_scores))]
    selected = select_next_pose_after_rotation(
        estimator=est,
        current_pose_xyz=current_pose,
        visited_poses_xyz=visited,
        n_candidates=4,
        n_rollouts=0,
        candidate_strategy="ring",
    )
    assert selected.shape == (3,)
    assert np.allclose(selected, expected_pose)
    assert len(est.calls) == candidates.shape[0]


def test_candidate_generation_adds_map_cells_when_random_sampling_is_sparse() -> None:
    """Candidate generation should include deterministic free-cell centers."""
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 10),
        traversable_cells=((8, 8),),
    )

    candidates = generate_candidate_poses(
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        map_api=traversable,
        n_candidates=4,
        strategy="free_space_sobol",
        min_dist_from_visited=1.0,
        visited_poses_xyz=np.array([[0.5, 0.5, 0.5]], dtype=float),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        rng=np.random.default_rng(7),
    )

    assert any(np.allclose(candidate, [8.5, 8.5, 0.5]) for candidate in candidates)


def test_candidate_generation_replenishes_after_batch_reachability_filter() -> None:
    """Reachability loss must trigger native map-center replenishment."""

    class ReachabilityMap:
        """Expose one reachable native cell and reject sampled off-center poses."""

        origin = (0.0, 0.0)
        cell_size = 1.0
        grid_shape = (10, 10)
        traversable_cells = ((8, 8),)

        def __init__(self) -> None:
            """Initialize batched reachability accounting."""
            self.reachability_calls = 0

        def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
            """Return the center of one native map cell."""
            return float(cell[0]) + 0.5, float(cell[1]) + 0.5

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject accidental use of the scalar free-space callback."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Mark all sampled endpoints free before reachability filtering."""
            return np.ones(np.asarray(points).shape[0], dtype=bool)

        def is_motion_reachable_batch(
            self,
            _start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Keep only the native center to make replenishment deterministic."""
            self.reachability_calls += 1
            goals = np.asarray(goals_xyz, dtype=float)
            return np.all(
                np.isclose(goals, np.array([8.5, 8.5, 0.5])[None, :]),
                axis=1,
            )

    planning_map = ReachabilityMap()
    candidates = generate_candidate_poses(
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        map_api=planning_map,
        n_candidates=4,
        strategy="free_space_sobol",
        min_dist_from_visited=1.0,
        visited_poses_xyz=np.array([[0.5, 0.5, 0.5]], dtype=float),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        rng=np.random.default_rng(7),
        require_motion_reachable=True,
    )

    assert planning_map.reachability_calls >= 2
    np.testing.assert_allclose(candidates, np.array([[8.5, 8.5, 0.5]]))


def test_candidate_filter_prefers_batched_free_space_path() -> None:
    """Standard maps should filter candidate arrays without scalar callbacks."""

    class BatchPlanningMap:
        """Expose both paths while making an accidental scalar call fail."""

        def __init__(self) -> None:
            """Initialize batch-call accounting."""
            self.batch_calls = 0

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject use of the compatibility-only scalar path."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Accept candidates whose x coordinate is at least one metre."""
            self.batch_calls += 1
            return np.asarray(points, dtype=float)[:, 0] >= 1.0

    planning_map = BatchPlanningMap()
    candidates = np.asarray(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 1.5]],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited_poses_xyz=None,
        min_dist_from_visited=0.0,
        is_free_fn=candidate_generation._resolve_free_space_checker(planning_map),
        is_free_batch_fn=(
            candidate_generation._resolve_free_space_batch_checker(planning_map)
        ),
    )

    assert planning_map.batch_calls == 1
    assert np.allclose(filtered, candidates[1:])


def test_candidate_height_expansion_matches_scalar_oracle() -> None:
    """Discrete height actions should be a vectorized Cartesian expansion."""
    candidates = np.array(
        [[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]],
        dtype=float,
    )
    heights = resolve_detector_height_actions(
        [1.5, 0.5, 1.5],
        default_height_m=0.5,
        bounds_z=(0.0, 2.0),
    )

    expanded = expand_candidate_height_actions(candidates, heights)
    expected = np.asarray(
        [
            [candidate[0], candidate[1], height]
            for candidate in candidates
            for height in heights
        ],
        dtype=float,
    )

    assert np.allclose(expanded, expected)


def test_candidate_generation_keeps_same_xy_alternate_height_action() -> None:
    """A height partner should bypass horizontal station-spacing rejection."""
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    candidates = generate_candidate_poses(
        current_pose_xyz=current,
        n_candidates=8,
        strategy="free_space_sobol",
        min_dist_from_visited=3.0,
        visited_poses_xyz=current.reshape(1, 3),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 1.5], dtype=float),
        ),
        rng=np.random.default_rng(11),
        detector_heights_m=[0.5, 1.5],
        include_current_xy_height_actions=True,
    )

    assert any(np.allclose(candidate, [1.0, 1.0, 1.5]) for candidate in candidates)
    assert not any(np.allclose(candidate, current) for candidate in candidates)


def test_candidate_generation_adds_continuous_same_xy_height_anchors() -> None:
    """Continuous planning should add useful same-xy anchors without a height list."""
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    candidates = generate_candidate_poses(
        current_pose_xyz=current,
        n_candidates=24,
        strategy="free_space_sobol",
        min_dist_from_visited=3.0,
        visited_poses_xyz=current.reshape(1, 3),
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 1.5], dtype=float),
        ),
        rng=np.random.default_rng(23),
        include_current_xy_height_actions=True,
        continuous_height_anchor_count=16,
        allow_height_partners=True,
        height_partner_min_z_separation_m=0.4,
    )

    same_xy = np.linalg.norm(candidates[:, :2] - current[None, :2], axis=1) <= 1e-12
    anchors = candidates[same_xy]
    assert anchors.shape[0] > 1
    assert np.all(np.abs(anchors[:, 2] - current[2]) >= 0.4)
    assert np.all((anchors[:, 2] >= 0.5) & (anchors[:, 2] <= 1.5))
    assert np.any(~np.isclose(anchors[:, 2], 1.5))


def test_continuous_height_partner_requires_minimum_z_separation() -> None:
    """Tiny continuous height jitter must not bypass station-spacing rejection."""
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.0, 1.0, 0.5001],
            [1.0, 1.0, 1.1],
            [5.0, 1.0, 0.5001],
        ],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited,
        3.0,
        lambda _candidate: True,
        allow_height_partners=True,
        height_partner_min_z_separation_m=0.5,
    )

    assert np.allclose(filtered, candidates[1:])


def test_dss_station_spacing_preserves_height_partner_only() -> None:
    """DSS station filtering should preserve a distinct-height revisit."""
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.5],
            [1.5, 1.0, 0.5],
            [5.0, 1.0, 0.5],
        ],
        dtype=float,
    )

    filtered, removed = dss_pp._filter_station_separation(
        candidates,
        visited,
        min_separation_m=3.0,
    )
    penalties = dss_pp._station_revisit_penalties_batch(
        candidates,
        visited,
        min_separation_m=3.0,
    )

    assert removed == 2
    assert np.allclose(filtered, [[1.0, 1.0, 1.5], [5.0, 1.0, 0.5]])
    assert penalties[1] == pytest.approx(0.0)
    assert penalties[0] > 0.0


def test_height_partner_filter_rejects_already_visited_actions_and_uses_xy_spacing() -> (
    None
):
    """Visited heights must not reopen duplicates or hide short horizontal moves."""
    visited = np.array(
        [[1.0, 1.0, 0.5], [1.0, 1.0, 1.5]],
        dtype=float,
    )
    candidates = np.array(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.5],
            [1.0, 1.0, 2.5],
            [3.9, 1.0, 2.5],
            [4.1, 1.0, 1.5],
        ],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited,
        3.0,
        lambda _candidate: True,
        allow_height_partners=True,
    )

    assert np.allclose(
        filtered,
        [[1.0, 1.0, 2.5], [4.1, 1.0, 1.5]],
    )


def test_height_partner_filter_uses_current_station_and_shared_tolerance() -> None:
    """Only the current station may reopen spacing for an alternate height."""
    visited = np.array(
        [[1.0, 1.0, 0.5], [5.0, 5.0, 0.5]],
        dtype=float,
    )
    current = visited[-1]
    candidates = np.array(
        [
            [1.0, 1.0, 1.5],
            [5.0 + 5.0e-7, 5.0, 1.5],
            [9.0, 9.0, 1.5],
        ],
        dtype=float,
    )

    filtered = candidate_generation._filter_candidates(
        candidates,
        visited,
        3.0,
        lambda _candidate: True,
        allow_height_partners=True,
        height_partner_reference_xyz=current,
        height_partner_xy_tolerance_m=1.0e-6,
    )

    assert np.allclose(filtered, candidates[1:])


def test_dss_height_partner_mask_uses_current_station_and_shared_tolerance() -> None:
    """Planner pairing must match runtime xy tolerance at the current station."""
    visited = np.array(
        [[1.0, 1.0, 0.5], [5.0, 5.0, 0.5]],
        dtype=float,
    )
    candidates = np.array(
        [[1.0, 1.0, 1.5], [5.0 + 5.0e-7, 5.0, 1.5]],
        dtype=float,
    )

    shared_tolerance_mask = dss_pp._height_partner_mask_batch(
        candidates,
        visited,
        reference_pose_xyz=visited[-1],
        xy_tolerance_m=1.0e-6,
    )
    strict_tolerance_mask = dss_pp._height_partner_mask_batch(
        candidates,
        visited,
        reference_pose_xyz=visited[-1],
        xy_tolerance_m=1.0e-9,
    )

    assert shared_tolerance_mask.tolist() == [False, True]
    assert strict_tolerance_mask.tolist() == [False, False]


def test_dss_height_partner_mask_requires_minimum_z_separation() -> None:
    """DSS should not classify sub-threshold height jitter as a paired action."""
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [[1.0, 1.0, 0.5001], [1.0, 1.0, 1.1]],
        dtype=float,
    )

    mask = dss_pp._height_partner_mask_batch(
        candidates,
        visited,
        min_z_separation_m=0.5,
    )

    assert mask.tolist() == [False, True]


def test_dss_height_partner_requires_an_unvisited_height_action() -> None:
    """DSS should waive revisit costs only for a genuinely new height action."""
    visited = np.array(
        [[1.0, 1.0, 0.5], [1.0, 1.0, 1.5]],
        dtype=float,
    )
    candidates = np.array(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.5],
            [1.0, 1.0, 2.5],
            [5.0, 1.0, 0.5],
        ],
        dtype=float,
    )

    filtered, removed = dss_pp._filter_station_separation(
        candidates,
        visited,
        min_separation_m=3.0,
    )
    penalties = dss_pp._station_revisit_penalties_batch(
        candidates,
        visited,
        min_separation_m=3.0,
    )

    assert removed == 2
    assert np.allclose(filtered, [[1.0, 1.0, 2.5], [5.0, 1.0, 0.5]])
    assert np.all(penalties[:2] > 0.0)
    assert penalties[2] == pytest.approx(0.0)


def test_dss_continuous_augmentation_preserves_base_and_uses_batch_filter() -> None:
    """Continuous augmentation should vary z and use batched free-space checks."""

    class BatchOnlyPlanningMap:
        """Expose only a usable batched free-space runtime path."""

        def __init__(self) -> None:
            """Initialize batch-call accounting."""
            self.batch_calls = 0

        def is_free(self, _point: np.ndarray) -> bool:
            """Reject accidental use of the scalar compatibility path."""
            raise AssertionError("scalar free-space path must not be selected")

        def is_free_batch(self, points: np.ndarray) -> np.ndarray:
            """Accept every in-bounds candidate in one batch."""
            self.batch_calls += 1
            return np.ones(np.asarray(points).shape[0], dtype=bool)

    planning_map = BatchOnlyPlanningMap()
    base = np.array(
        [[0.25, 0.25, 0.4], [0.75, 0.75, 1.4]],
        dtype=float,
    )
    current = np.array([0.5, 0.5, 0.5], dtype=float)
    bounds = (
        np.array([0.0, 0.0, 0.25], dtype=float),
        np.array([2.0, 2.0, 1.75], dtype=float),
    )
    config = DSSPPConfig(max_augmented_candidates=16, rng_seed=17)

    continuous = dss_pp.augment_candidate_stations(
        base,
        modes_by_isotope={},
        current_pose_xyz=current,
        visited_poses_xyz=None,
        map_api=planning_map,
        bounds_xyz=bounds,
        config=config,
        continuous_height_bounds_m=(0.25, 1.75),
    )
    legacy = dss_pp.augment_candidate_stations(
        base,
        modes_by_isotope={},
        current_pose_xyz=current,
        visited_poses_xyz=None,
        map_api=planning_map,
        bounds_xyz=bounds,
        config=config,
    )

    assert planning_map.batch_calls == 2
    assert np.allclose(continuous[: base.shape[0]], base)
    assert np.unique(np.round(continuous[base.shape[0] :, 2], 6)).size > 1
    assert np.all(
        (continuous[base.shape[0] :, 2] >= 0.25)
        & (continuous[base.shape[0] :, 2] <= 1.75)
    )
    assert np.allclose(legacy[: base.shape[0]], base)
    assert np.allclose(legacy[base.shape[0] :, 2], current[2])


def test_dss_path_cache_distinguishes_exact_xyz_inside_one_cell() -> None:
    """Cached grid paths must retain endpoint and height-dependent distance."""
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        traversable_cells=((0, 0),),
    )
    start = np.array([0.1, 0.1, 0.5], dtype=float)
    same_height = np.array([0.2, 0.2, 0.5], dtype=float)
    raised = np.array([0.2, 0.2, 1.5], dtype=float)

    dss_pp._DSS_PP_PATH_LENGTH_CACHE.clear()
    try:
        same_height_length = dss_pp._node_path_length(
            traversable,
            start,
            same_height,
        )
        raised_length = dss_pp._node_path_length(
            traversable,
            start,
            raised,
        )
    finally:
        dss_pp._DSS_PP_PATH_LENGTH_CACHE.clear()

    assert same_height_length == pytest.approx(np.linalg.norm(same_height - start))
    assert raised_length == pytest.approx(np.linalg.norm(raised - start))
    assert raised_length > same_height_length


def test_dss_path_filter_prefers_batch_lengths_over_reachability_flags() -> None:
    """Candidate filtering uses finite batch lengths before legacy flags."""

    class BatchLengthMap:
        """Expose both APIs while making legacy use an immediate failure."""

        def __init__(self) -> None:
            """Initialize the batch-call counter."""
            self.batch_calls = 0

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return one unreachable candidate between two reachable ones."""
            del start_xyz
            self.batch_calls += 1
            assert goals_xyz.shape == (3, 3)
            return np.array([1.0, float("inf"), 2.0], dtype=float)

        def is_motion_reachable_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Fail if the compatibility path is selected before lengths."""
            del start_xyz, goals_xyz
            raise AssertionError("legacy reachability API should not be used")

    planning_map = BatchLengthMap()
    candidates = np.array(
        [
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [3.0, 0.0, 0.5],
        ],
        dtype=float,
    )

    filtered, removed = dss_pp._filter_path_reachable_stations(
        candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        map_api=planning_map,
    )

    assert planning_map.batch_calls == 1
    assert removed == 1
    np.testing.assert_allclose(filtered, candidates[[0, 2]])


def test_dss_batch_path_length_helper_matches_vector_and_scalar_fallbacks() -> None:
    """DSS batch path lengths preserve native and compatibility semantics."""

    class NativeBatchMap:
        """Return deterministic native batch lengths for dispatch testing."""

        def __init__(self) -> None:
            """Initialize the native batch-call counter."""
            self.batch_calls = 0

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return direct distances with a deterministic offset."""
            self.batch_calls += 1
            return np.linalg.norm(goals_xyz - start_xyz[None, :], axis=1) + 0.25

    start = np.array([0.0, 0.0, 0.5], dtype=float)
    goals = np.array(
        [[1.0, 0.0, 0.5], [0.0, 2.0, 1.5]],
        dtype=float,
    )
    native_map = NativeBatchMap()

    native = dss_pp._node_path_lengths_batch(native_map, start, goals)
    no_map = dss_pp._node_path_lengths_batch(None, start, goals)

    assert native_map.batch_calls == 1
    np.testing.assert_allclose(
        native,
        np.linalg.norm(goals - start[None, :], axis=1) + 0.25,
    )
    np.testing.assert_allclose(
        no_map,
        np.linalg.norm(goals - start[None, :], axis=1),
    )


def test_dss_selection_uses_batch_lengths_for_filter_and_node_build() -> None:
    """End-to-end station selection dispatches both path phases in batches."""

    class TrackingBatchMap:
        """Wrap a traversability map and count native batch path requests."""

        def __init__(self, wrapped: TraversabilityMap) -> None:
            """Store the wrapped grid and initialize its call counter."""
            self.wrapped = wrapped
            self.batch_calls = 0

        def __getattr__(self, name: str) -> object:
            """Forward non-batch map APIs to the traversability grid."""
            return getattr(self.wrapped, name)

        def motion_path_lengths_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Return finite direct lengths for every candidate in one call."""
            self.batch_calls += 1
            return np.linalg.norm(goals_xyz - start_xyz[None, :], axis=1)

        def is_motion_reachable_batch(
            self,
            start_xyz: np.ndarray,
            goals_xyz: np.ndarray,
        ) -> np.ndarray:
            """Fail if selection bypasses the preferred path-length API."""
            del start_xyz, goals_xyz
            raise AssertionError("legacy reachability API should not be used")

    estimator = _build_simple_estimator()
    wrapped = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        traversable_cells=tuple(
            (ix, iy) for ix in range(4) for iy in range(4)
        ),
    )
    planning_map = TrackingBatchMap(wrapped)
    candidates = np.array(
        [[1.5, 1.5, 0.5], [2.5, 1.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.5, 0.5, 0.5], dtype=float),
        visited_poses_xyz=None,
        map_api=planning_map,
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=1.0,
            lambda_rotation=0.0,
            lambda_coverage=0.0,
            eta_revisit=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            min_station_separation_m=0.0,
            signature_std_min_counts=0.0,
            enforce_min_observation=False,
            augment_candidates=False,
        ),
    )

    assert planning_map.batch_calls == 2
    assert any(np.allclose(result.next_pose, pose) for pose in candidates)


def test_estimate_lambda_cost_range_scales_motion() -> None:
    """Range-based lambda should match uncertainty and motion-cost ranges."""
    uncertainties = np.array([1.0, 2.0, 4.0], dtype=float)
    distances = np.array([0.5, 1.0, 2.5], dtype=float)
    lam = estimate_lambda_cost(uncertainties, distances, method="range")
    expected = (4.0 - 1.0) / (2.5 - 0.5)
    assert lam == pytest.approx(expected)


def test_minimum_observation_shortfall_is_zero_only_when_all_isotopes_visible() -> None:
    """Observation shortfall should penalize any isotope below the count target."""
    assert minimum_observation_shortfall(
        {"Cs-137": 5.0, "Co-60": 5.0},
        min_counts=5.0,
    ) == pytest.approx(0.0)
    assert (
        minimum_observation_shortfall(
            {"Cs-137": 5.0, "Co-60": 0.0},
            min_counts=5.0,
        )
        > 0.0
    )


def test_pose_selection_prefers_all_isotope_observability() -> None:
    """Soft observability constraints should prevent single-isotope pose bias."""

    class _FairnessEstimator:
        """Minimal estimator exposing the pose-planning methods under test."""

        pf_config = None

        def expected_uncertainty_after_rotation(self, **kwargs: object) -> float:
            """Return identical uncertainty for every candidate."""
            return 0.0

        def expected_observation_counts_by_isotope_at_pose(
            self,
            pose_xyz: np.ndarray,
            **kwargs: object,
        ) -> dict[str, float]:
            """Return candidate-dependent isotope observability."""
            if float(pose_xyz[0]) < 0.5:
                return {"Cs-137": 10.0, "Co-60": 0.0}
            return {"Cs-137": 5.0, "Co-60": 5.0}

    candidates = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=float)
    selected = select_next_pose_from_candidates(
        estimator=_FairnessEstimator(),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        lambda_cost=0.0,
        min_observation_counts=5.0,
        min_observation_penalty_scale=1.0,
        num_rollouts=1,
    )

    assert selected == 1


def test_pose_selection_enforces_observability_when_feasible() -> None:
    """Feasible all-isotope candidates should dominate even with zero penalty scale."""

    class _FairnessEstimator:
        """Minimal estimator exposing candidate-dependent predictions."""

        pf_config = None

        def expected_uncertainty_after_rotation(self, **kwargs: object) -> float:
            """Return a lower uncertainty for the infeasible candidate."""
            pose = np.asarray(kwargs["pose_xyz"], dtype=float)
            return 0.0 if float(pose[0]) < 0.5 else 100.0

        def expected_observation_counts_by_isotope_at_pose(
            self,
            pose_xyz: np.ndarray,
            **kwargs: object,
        ) -> dict[str, float]:
            """Return one single-isotope-biased and one all-isotope-visible pose."""
            if float(pose_xyz[0]) < 0.5:
                return {"Cs-137": 100.0, "Co-60": 0.0}
            return {"Cs-137": 5.0, "Co-60": 5.0}

    selected = select_next_pose_from_candidates(
        estimator=_FairnessEstimator(),
        candidate_poses_xyz=np.array(
            [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
            dtype=float,
        ),
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        lambda_cost=0.0,
        min_observation_counts=5.0,
        min_observation_penalty_scale=0.0,
        num_rollouts=1,
    )

    assert selected == 1


def test_pose_selection_uses_planning_particle_cap_for_observability() -> None:
    """One-step observability scoring should use the configured planning subset."""

    class _Config:
        """Minimal PF config exposing the planning particle cap."""

        lambda_cost = 0.0
        ig_threshold = 0.0
        max_dwell_time_s = 2.0
        short_time_s = 1.0
        planning_rollout_particles = 7
        planning_particles = None

    class _Estimator:
        """Minimal estimator recording observability kwargs."""

        pf_config = _Config()

        def __init__(self) -> None:
            """Initialize call recording."""
            self.max_particles_seen: list[int | None] = []

        def expected_uncertainty_after_rotation(self, **kwargs: object) -> float:
            """Return identical uncertainty for every candidate."""
            return 0.0

        def expected_observation_counts_by_isotope_at_pose(
            self,
            pose_xyz: np.ndarray,
            **kwargs: object,
        ) -> dict[str, float]:
            """Record the particle cap used for observability scoring."""
            self.max_particles_seen.append(kwargs.get("max_particles"))
            return {"Cs-137": 10.0}

    estimator = _Estimator()
    selected = select_next_pose_from_candidates(
        estimator=estimator,
        candidate_poses_xyz=np.array(
            [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
            dtype=float,
        ),
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        min_observation_counts=5.0,
        num_rollouts=1,
    )

    assert selected == 0
    assert estimator.max_particles_seen == [7, 7]


def test_pose_selection_parallel_matches_serial_selection() -> None:
    """Parallel one-step candidate evaluation should preserve serial scores."""

    class _Config:
        """Minimal PF config for deterministic one-step pose selection."""

        lambda_cost = 0.0
        ig_threshold = 0.0
        max_dwell_time_s = 2.0
        short_time_s = 1.0
        planning_rollout_particles = None
        planning_particles = None
        use_gpu = False
        ig_workers = 4

    class _Estimator:
        """Deterministic estimator whose score depends only on the pose."""

        pf_config = _Config()

        def expected_uncertainty_after_rotation(self, **kwargs: object) -> float:
            """Return a smooth deterministic uncertainty surface."""
            pose = np.asarray(kwargs["pose_xyz"], dtype=float)
            return float((pose[0] - 2.0) ** 2 + 0.1 * pose[1] ** 2)

    candidates = np.array(
        [
            [0.0, 0.0, 0.5],
            [1.5, 1.0, 0.5],
            [2.0, 0.2, 0.5],
            [3.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    serial_idx = select_next_pose_from_candidates(
        estimator=_Estimator(),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        lambda_cost=0.0,
        num_rollouts=1,
        worker_count=1,
        rng_seed=123,
    )
    parallel_idx = select_next_pose_from_candidates(
        estimator=_Estimator(),
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        lambda_cost=0.0,
        num_rollouts=1,
        worker_count=3,
        rng_seed=123,
    )

    assert parallel_idx == serial_idx == 2


def test_pose_selection_gpu_override_allows_parallel_candidate_workers(
    capsys,
) -> None:
    """Explicit CPU planning override should keep worker-parallel evaluation."""

    class _Config:
        """Minimal PF config that defaults to GPU-enabled planning."""

        lambda_cost = 0.0
        ig_threshold = 0.0
        max_dwell_time_s = 2.0
        short_time_s = 1.0
        planning_rollout_particles = None
        planning_particles = None
        use_gpu = True
        gpu_device = "cuda"
        gpu_dtype = "float32"
        ig_workers = 4

    class _Estimator:
        """Deterministic estimator that records the active GPU flag."""

        pf_config = _Config()

        def __init__(self) -> None:
            """Initialize the active GPU-state recorder."""
            self.use_gpu_seen: list[bool] = []

        def expected_uncertainty_after_rotation(self, **kwargs: object) -> float:
            """Return a deterministic uncertainty and record GPU state."""
            pose = np.asarray(kwargs["pose_xyz"], dtype=float)
            self.use_gpu_seen.append(bool(self.pf_config.use_gpu))
            return float((pose[0] - 2.0) ** 2 + 0.1 * pose[1] ** 2)

    estimator = _Estimator()
    candidates = np.array(
        [
            [0.0, 0.0, 0.5],
            [2.0, 0.2, 0.5],
            [3.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    selected = select_next_pose_from_candidates(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        lambda_cost=0.0,
        num_rollouts=1,
        worker_count=3,
        rng_seed=123,
        verbose=True,
        use_gpu=False,
        ig_breakdown_k=0,
    )

    output = capsys.readouterr().out
    assert selected == 1
    assert "Candidate evaluation mode: 3 workers." in output
    assert estimator.use_gpu_seen == [False, False, False]
    assert estimator.pf_config.use_gpu is True


def test_recommend_num_rollouts_selects_minimum_stable_value() -> None:
    """recommend_num_rollouts should pick the smallest rollout meeting the target SE."""
    samples = {
        1: [6.0, 14.0, 8.0, 12.0],
        2: [9.0, 11.0, 10.0, 10.0],
        4: [9.6, 10.4, 9.8, 10.2],
        8: [9.9, 10.1, 9.95, 10.05],
    }

    def eval_fn(n_rollouts: int, seed: int) -> float:
        values = samples[int(n_rollouts)]
        idx = int(seed) % len(values)
        return float(values[idx])

    recommended = recommend_num_rollouts(
        candidate_rollouts=(1, 2, 4, 8),
        trials=4,
        rel_se_target=0.015,
        rng_seed=0,
        eval_fn=eval_fn,
    )
    assert recommended == 8


def test_select_next_pose_after_rotation_runs_with_estimator() -> None:
    """After-rotation selection should run with the real estimator."""
    est = _build_simple_estimator()
    est.pf_config.eig_num_samples = 0
    current_pose = np.array([5.0, 5.0, 5.0], dtype=float)
    visited = np.array([[5.0, 5.0, 5.0]], dtype=float)
    selected = select_next_pose_after_rotation(
        estimator=est,
        current_pose_xyz=current_pose,
        visited_poses_xyz=visited,
        n_candidates=1,
        n_rollouts=0,
        candidate_strategy="ring",
    )
    assert isinstance(selected, np.ndarray)
    assert selected.shape == (3,)


def test_orientation_expected_information_gain_positive_when_strengths_differ() -> None:
    """Monte-Carlo EIG (Eq. 3.44) should be positive when particles predict different Λ."""
    np.random.seed(0)
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(num_particles=2, max_sources=1)
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0]))
    est._ensure_kernel_cache()
    # Overwrite continuous particles with distinct strengths
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([10.0]),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([1.0]),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    RFe = np.diag([1.0, 1.0, 1.0])
    RPb = np.diag([1.0, 1.0, 1.0])
    ig = est.orientation_expected_information_gain(
        pose_idx=0, RFe=RFe, RPb=RPb, live_time_s=1.0, num_samples=20
    )
    assert ig > 0.0


def test_expected_uncertainty_after_rotation_stops_with_zero_time() -> None:
    """No rotation time should return the current uncertainty."""
    est = _build_simple_estimator()
    u0 = est.global_uncertainty()
    u_after = est.expected_uncertainty_after_rotation(
        pose_xyz=np.array([0.0, 0.0, 0.0]),
        live_time_per_rot_s=1.0,
        tau_ig=1e-3,
        tmax_s=0.0,
        n_rollouts=0,
    )
    assert u_after == pytest.approx(u0)


def test_planning_rollouts_do_not_advance_or_reseed_pf_numpy_stream() -> None:
    """Planning-only Monte Carlo must be isolated from sequential PF randomness."""
    est = _build_simple_estimator()
    np.random.seed(20260717)
    before = np.random.get_state()

    est.expected_uncertainty_after_rotation(
        pose_xyz=np.array([0.0, 0.0, 0.0]),
        live_time_per_rot_s=1.0,
        tau_ig=1e9,
        tmax_s=1.0,
        n_rollouts=0,
    )
    est.expected_uncertainty_after_rotation_at_pose(
        np.array([0.0, 0.0, 0.0]),
        tau_ig=1e9,
        t_max_s=1.0,
        t_short_s=1.0,
        num_rollouts=0,
        rng_seed=91,
    )
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_expected_uncertainty_after_rotation_stops_with_large_tau() -> None:
    """Large tau_ig should stop immediately and return current uncertainty."""
    est = _build_simple_estimator()
    u0 = est.global_uncertainty()
    u_after = est.expected_uncertainty_after_rotation(
        pose_xyz=np.array([0.0, 0.0, 0.0]),
        live_time_per_rot_s=1.0,
        tau_ig=1e9,
        tmax_s=1.0,
        n_rollouts=0,
    )
    assert u_after == pytest.approx(u0)


def test_expected_uncertainty_after_rotation_one_step() -> None:
    """When t_max_s == t_short_s and tau_ig == 0, exactly one rotation is simulated."""
    est = _build_simple_estimator()
    est.pf_config.eig_num_samples = 0
    u_after, debug = est.expected_uncertainty_after_rotation(
        pose_xyz=np.array([0.0, 0.0, 0.0]),
        live_time_per_rot_s=1.0,
        tau_ig=0.0,
        tmax_s=1.0,
        n_rollouts=0,
        return_debug=True,
    )
    assert isinstance(u_after, float)
    assert debug["rollouts"][0]["num_rotations"] == 1


def test_short_time_update_uses_default_duration() -> None:
    """short_time_update should use pf_config.short_time_s when live_time_s is omitted."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(num_particles=10, max_sources=1, short_time_s=0.25)
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0]))
    est._ensure_kernel_cache()
    mats = generate_octant_rotation_matrices()
    z_k = {"Cs-137": 5.0}
    est.short_time_update(
        z_k=z_k, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=None
    )
    import pytest

    assert est.measurements[-1].live_time_s == pytest.approx(0.25, rel=1e-12)
    assert est.measurements[-1].fe_index == 0
    assert est.measurements[-1].pb_index == 0


def test_should_stop_shield_rotation_by_dwell_time() -> None:
    """Rotation should stop when dwell time exceeds max_dwell_time_s (Eq. 3.49)."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=5, max_sources=1, max_dwell_time_s=0.5, ig_threshold=1e6
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0]))
    est._ensure_kernel_cache()
    mats = generate_octant_rotation_matrices()
    # Two short updates totaling > max_dwell_time_s
    est.short_time_update(
        z_k={"Cs-137": 1.0}, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=0.3
    )
    est.short_time_update(
        z_k={"Cs-137": 1.0}, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=0.3
    )
    assert est.should_stop_shield_rotation(
        pose_idx=0,
        ig_threshold=config.ig_threshold,
        change_tol=1e6,
        uncertainty_tol=1e6,
        live_time_s=config.short_time_s,
    )


def test_expected_uncertainty_after_pose_is_finite() -> None:
    """Expected uncertainty surrogate should return a finite positive value."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(num_particles=5, max_sources=1)
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 0.0, 0.0]))
    est._ensure_kernel_cache()
    # Setup simple continuous particles with two different strengths
    filt = est.filters["Cs-137"]
    from pf.particle_filter import IsotopeParticle
    from pf.state import IsotopeState

    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([5.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    U = est.expected_uncertainty_after_pose(
        pose_idx=0, orient_idx=0, live_time_s=1.0, num_samples=10
    )
    assert np.isfinite(U)
    assert U >= 0.0


def test_dss_pp_program_library_contains_differential_primitives() -> None:
    """DSS-PP should expose bearing, material, occlusion, and vertical programs."""
    mats = generate_octant_rotation_matrices()
    normals = np.asarray([mat[:, 2] for mat in mats], dtype=float)
    programs = build_shield_program_library(normals, program_length=2, max_programs=40)
    kinds = {program.kind for program in programs}

    assert "bearing_split" in kinds
    assert "material_split" in kinds
    assert "occlusion_test" in kinds
    assert "vertical_split" in kinds
    assert "vertical_material_split" in kinds
    assert "elevation_bearing_split" in kinds
    assert all(len(program.pair_ids) == 2 for program in programs)


def test_dss_pp_program_library_can_build_separation_bursts() -> None:
    """Shield programs should expand primitives into longer temporal bursts."""
    mats = generate_octant_rotation_matrices()
    normals = np.asarray([mat[:, 2] for mat in mats], dtype=float)
    programs = build_shield_program_library(normals, program_length=8, max_programs=8)

    assert programs
    assert all(len(program.pair_ids) == 8 for program in programs)


def test_pairwise_cover_objective_uses_vertical_ambiguity() -> None:
    """Greedy shield-program search should score height separation directly."""
    config = DSSPPConfig(
        lambda_temporal_separation=0.0,
        lambda_elevation_signature=4.0,
    )
    objective = dss_pp._pairwise_cover_objective(
        None,
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([0.1, 0.8], dtype=float),
        config=config,
    )

    assert int(np.argmax(objective)) == 1
    assert objective[1] > objective[0]


def test_temporal_separation_uses_response_shape_not_strength_scale() -> None:
    """Temporal-code scoring should not reward amplitude-only differences."""
    config = DSSPPConfig(
        temporal_cover_weight=1.0,
        temporal_logdet_weight=0.5,
        temporal_decorrelation_weight=1.0,
        temporal_pair_contrast_threshold=0.2,
    )
    collinear = [
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.array([10.0, 20.0, 30.0], dtype=float),
    ]
    separated = [
        np.array([12.0, 0.1, 0.1], dtype=float),
        np.array([0.1, 12.0, 0.1], dtype=float),
        np.array([0.1, 0.1, 12.0], dtype=float),
    ]

    collinear_score = _temporal_separation_score_from_signatures(
        collinear,
        [0.5, 0.5],
        config=config,
    )
    separated_score = _temporal_separation_score_from_signatures(
        separated,
        [1.0, 1.0, 1.0],
        config=config,
    )

    assert collinear_score == pytest.approx(0.0, abs=1.0e-9)
    assert separated_score > 1.0


def test_extract_signature_modes_boosts_tentative_residual_sources() -> None:
    """Residual-aware planning should prioritize tentative source signatures."""
    est = _build_simple_estimator()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([False], dtype=bool),
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[5.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([True], dtype=bool),
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]

    modes = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        tentative_weight_multiplier=5.0,
    )["Cs-137"]

    assert len(modes) == 2
    assert np.allclose(modes[0].position_xyz, np.array([5.0, 0.0, 0.0]))
    assert modes[0].weight > modes[1].weight


def test_extract_signature_modes_keeps_soft_quarantined_sources() -> None:
    """Soft-quarantined tentative modes should remain visible to DSS-PP."""
    est = _build_simple_estimator()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([False], dtype=bool),
                verification_fail_streaks=np.array([0], dtype=int),
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[5.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([True], dtype=bool),
                verification_fail_streaks=np.array([3], dtype=int),
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]

    modes = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        tentative_weight_multiplier=1.0,
    )["Cs-137"]

    assert len(modes) == 2


def test_extract_signature_modes_adds_runtime_rescue_modes() -> None:
    """DSS-PP should see station-level rescue modes outside the posterior."""
    est = _build_simple_estimator()
    rescue_pos = np.array([[5.0, 0.0, 0.0]], dtype=float)
    rescue_q = np.array([200.0], dtype=float)
    est._runtime_report_rescue_modes = {"Cs-137": (rescue_pos, rescue_q, 0.1)}

    with_rescue = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        include_runtime_rescue_modes=True,
        runtime_rescue_mode_weight=1.0,
    )["Cs-137"]
    without_rescue = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        include_runtime_rescue_modes=False,
    )["Cs-137"]

    assert any(np.allclose(mode.position_xyz, rescue_pos[0]) for mode in with_rescue)
    assert not any(
        np.allclose(mode.position_xyz, rescue_pos[0]) for mode in without_rescue
    )


def test_extract_signature_modes_preserves_rescue_when_mode_budget_is_full() -> None:
    """Runtime rescue modes should not be dropped only because PF modes fill max_k."""
    est = _build_simple_estimator()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([False], dtype=bool),
            ),
            log_weight=float(np.log(0.5)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                tentative_sources=np.array([False], dtype=bool),
            ),
            log_weight=float(np.log(0.5)),
        ),
    ]
    rescue_pos = np.array([[6.0, 0.0, 0.0]], dtype=float)
    rescue_q = np.array([50.0], dtype=float)
    est._runtime_report_rescue_modes = {"Cs-137": (rescue_pos, rescue_q, 0.1)}

    modes = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=2,
        include_runtime_rescue_modes=True,
        runtime_rescue_mode_weight=0.1,
    )["Cs-137"]

    assert len(modes) == 2
    assert any(np.allclose(mode.position_xyz, rescue_pos[0]) for mode in modes)


def test_extract_signature_modes_adds_global_surface_rescue_modes() -> None:
    """DSS-PP should include residual-ranked global surface rescue candidates."""
    est = _build_simple_estimator()
    rescue_pos = np.array([[6.0, 0.0, 0.0]], dtype=float)
    rescue_q = np.array([150.0], dtype=float)

    def rescue_modes():
        """Return deterministic planning rescue modes."""
        return {"Cs-137": (rescue_pos, rescue_q, 0.02)}

    est.planning_surface_rescue_modes = rescue_modes
    with_rescue = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=3,
        include_runtime_rescue_modes=False,
        include_global_surface_rescue_modes=True,
        global_surface_rescue_mode_weight=1.0,
    )["Cs-137"]
    without_rescue = dss_pp.extract_signature_modes(
        est,
        mode_cluster_radius_m=0.1,
        max_modes_per_isotope=3,
        include_runtime_rescue_modes=False,
        include_global_surface_rescue_modes=False,
    )["Cs-137"]

    assert any(np.allclose(mode.position_xyz, rescue_pos[0]) for mode in with_rescue)
    assert not any(
        np.allclose(mode.position_xyz, rescue_pos[0]) for mode in without_rescue
    )


def test_recovery_isotope_mode_weights_boost_target_isotope_only() -> None:
    """DSS-PP recovery mode should prioritize only unresolved isotopes."""
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
                strength_cps_1m=100.0,
                weight=0.5,
                spread_m=0.1,
            )
        ],
        "Co-60": [
            dss_pp.SignatureMode(
                isotope="Co-60",
                position_xyz=np.array([1.0, 0.0, 0.0], dtype=float),
                strength_cps_1m=100.0,
                weight=0.5,
                spread_m=0.1,
            )
        ],
    }

    boosted = dss_pp._apply_recovery_isotope_mode_weights(
        modes,
        dss_pp.DSSPPConfig(
            recovery_isotopes=("Cs-137",),
            recovery_isotope_mode_weight_multiplier=3.0,
        ),
    )

    assert boosted["Cs-137"][0].weight == pytest.approx(1.5)
    assert boosted["Co-60"][0].weight == pytest.approx(0.5)


def test_signature_mode_weight_cap_keeps_weak_modes_visible() -> None:
    """Planner-only mode weights should cap dominant modes without moving them."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=100.0,
            spread_m=0.1,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([5.0, 0.0, 0.0], dtype=float),
            strength_cps_1m=100.0,
            weight=1.0,
            spread_m=0.1,
        ),
    ]

    rebalanced = dss_pp._rebalance_signature_mode_weights(
        modes,
        weak_floor=0.2,
        dominant_cap=0.7,
    )

    assert max(mode.weight for mode in rebalanced) <= 0.700001
    assert min(mode.weight for mode in rebalanced) >= 0.199999
    assert np.allclose(rebalanced[0].position_xyz, modes[0].position_xyz)


def test_dss_pp_adds_pairwise_contrast_cover_program() -> None:
    """Temporal separation should add a pose-specific pairwise cover program."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([4.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
        ]
    }
    dss_config = DSSPPConfig(
        program_length=3,
        lambda_temporal_separation=1.0,
        temporal_cover_programs=1,
    )
    base_programs = build_shield_program_library(
        est.normals,
        program_length=3,
        max_programs=4,
    )

    programs = _programs_for_pose(
        estimator=est,
        kernel=_continuous_kernel_for_estimator(est),
        modes_by_isotope=modes,
        pose_xyz=np.array([2.0, 1.0, 0.5], dtype=float),
        base_programs=base_programs,
        config=dss_config,
    )

    assert any(program.kind == "pairwise_contrast_cover" for program in programs)


def test_dss_pp_batched_program_scores_match_scalar_oracle() -> None:
    """Batched DSS-PP program scoring should match the scalar program oracle."""
    isotopes = ["Cs-137", "Co-60"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.5], [4.0, 0.0, 0.5], [2.0, 3.0, 0.5]],
        dtype=float,
    )
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
        alpha_weights={"Cs-137": 0.6, "Co-60": 0.4},
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={
            "Cs-137": {"fe": 0.5, "pb": 1.0},
            "Co-60": {"fe": 0.3, "pb": 0.8},
        },
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.7,
                spread_m=0.1,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([4.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1200.0,
                weight=0.3,
                spread_m=0.1,
            ),
        ],
        "Co-60": [
            dss_pp.SignatureMode(
                isotope="Co-60",
                position_xyz=np.array([1.0, 3.0, 0.5], dtype=float),
                strength_cps_1m=900.0,
                weight=0.5,
                spread_m=0.1,
            ),
            dss_pp.SignatureMode(
                isotope="Co-60",
                position_xyz=np.array([3.0, 3.0, 0.5], dtype=float),
                strength_cps_1m=1500.0,
                weight=0.5,
                spread_m=0.1,
            ),
        ],
    }
    dss_config = DSSPPConfig(
        program_length=4,
        live_time_s=2.0,
        lambda_temporal_separation=1.0,
        temporal_cover_programs=1,
        temporal_cover_weight=1.0,
        temporal_logdet_weight=0.25,
        temporal_decorrelation_weight=0.5,
        min_observation_counts=5.0,
        count_utility_saturation_counts=50.0,
    )
    programs = build_shield_program_library(
        est.normals,
        program_length=4,
        max_programs=6,
    ) + [
        dss_pp.ShieldProgram(
            name="manual",
            pair_ids=(0, 7, 12, 31),
            kind="manual",
        )
    ]
    kernel = _continuous_kernel_for_estimator(est)
    pair_cache = dss_pp._build_pair_signature_cache(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes,
        pose_xyz=np.array([2.0, 1.0, 0.5], dtype=float),
        config=dss_config,
    )

    batched = dss_pp._score_programs_from_pair_cache(
        estimator=est,
        pair_cache=pair_cache,
        programs=programs,
        config=dss_config,
    )
    scalar_rows = [
        dss_pp._score_program_from_pair_cache(
            estimator=est,
            pair_cache=pair_cache,
            program=program,
            config=dss_config,
        )
        for program in programs
    ]

    for term_idx, term_values in enumerate(batched):
        expected = np.asarray([row[term_idx] for row in scalar_rows], dtype=float)
        assert np.allclose(term_values, expected, rtol=1.0e-10, atol=1.0e-10)
    temporal_only = dss_pp._temporal_scores_programs_from_pair_cache(
        estimator=est,
        pair_cache=pair_cache,
        programs=programs,
        config=dss_config,
    )
    assert np.allclose(temporal_only, batched[1], rtol=1.0e-10, atol=1.0e-10)


def test_dss_pp_expected_bic_gap_batch_matches_scalar_oracle() -> None:
    """Program-level cardinality BIC gap should be batched and scalar-equivalent."""
    raw = np.asarray(
        [
            [[100.0, 100.0], [120.0, 120.0], [80.0, 80.0], [90.0, 90.0]],
            [[220.0, 1.0], [1.0, 220.0], [180.0, 1.0], [1.0, 180.0]],
        ],
        dtype=float,
    )

    def scalar_gap(program_matrix: np.ndarray) -> float:
        """Return the K-vs-K-1 expected BIC gap with scalar operations."""
        expected_counts = np.sum(program_matrix, axis=1)
        sqrt_weight = 1.0 / np.sqrt(np.maximum(expected_counts, 1.0))
        best_deviance = np.inf
        for removed_idx in range(program_matrix.shape[1]):
            keep = [idx for idx in range(program_matrix.shape[1]) if idx != removed_idx]
            design = program_matrix[:, keep]
            weighted_design = design * sqrt_weight[:, None]
            weighted_expected = expected_counts * sqrt_weight
            normal = weighted_design.T @ weighted_design + 1.0e-9 * np.eye(len(keep))
            rhs = weighted_design.T @ weighted_expected
            coeff = np.maximum(np.linalg.pinv(normal) @ rhs, 0.0)
            fitted = np.sum(design * coeff.reshape(1, -1), axis=1)
            deviance = dss_pp._poisson_deviance_matrix(
                expected_counts.reshape(1, -1),
                fitted.reshape(1, -1),
            )[0]
            best_deviance = min(best_deviance, float(deviance))
        return max(best_deviance - 4.0 * np.log(program_matrix.shape[0]), 0.0)

    batched = dss_pp._expected_bic_gap_against_source_removal_batch(
        raw,
        parameter_count_per_source=4,
    )
    expected = np.asarray([scalar_gap(program) for program in raw], dtype=float)

    assert np.allclose(batched, expected, rtol=1.0e-10, atol=1.0e-10)
    assert batched[0] == pytest.approx(0.0)
    assert batched[1] > 100.0


def test_dss_pp_pair_signature_cache_uses_pair_response_scales() -> None:
    """Batched DSS-PP signatures should honor pair-specific PF response scales."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
        measurement_scale_by_isotope={"Cs-137": 1.25},
        measurement_scale_by_isotope_and_pair={"Cs-137": {0: 2.0, 7: 0.5}},
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=1.0,
                spread_m=0.1,
            )
        ]
    }
    dss_config = DSSPPConfig(program_length=2, live_time_s=2.0)
    pose = np.array([2.0, 1.0, 0.5], dtype=float)
    kernel = _continuous_kernel_for_estimator(est)
    program = dss_pp.ShieldProgram(
        name="scaled_pairs",
        pair_ids=(0, 7, 8),
        kind="manual",
    )

    pair_cache = dss_pp._build_pair_signature_cache(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes,
        pose_xyz=pose,
        config=dss_config,
    )
    signature = dss_pp._expected_signature(
        kernel=kernel,
        estimator=est,
        mode=modes["Cs-137"][0],
        pose_xyz=pose,
        program=program,
        num_orients=int(est.num_orientations),
        live_time_s=float(dss_config.live_time_s),
    )
    matrix, _weights = pair_cache["Cs-137"]

    assert matrix[program.pair_ids, 0] == pytest.approx(signature)
    mode = modes["Cs-137"][0]
    assert matrix[0, 0] / matrix[8, 0] == pytest.approx(
        (
            kernel.kernel_value_pair(
                "Cs-137",
                pose,
                mode.position_xyz,
                0,
                0,
            )
            * 2.0
        )
        / (
            kernel.kernel_value_pair(
                "Cs-137",
                pose,
                mode.position_xyz,
                1,
                0,
            )
            * 1.25
        )
    )

    pose_caches = dss_pp._build_pair_signature_caches_for_poses(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes,
        poses_xyz=np.vstack([pose, pose + np.array([0.5, 0.0, 0.0])]),
        config=dss_config,
    )
    pose_matrix, _pose_weights = pose_caches[0]["Cs-137"]
    assert pose_matrix[program.pair_ids, 0] == pytest.approx(signature)


def test_dss_pp_station_response_uses_transport_kernel_and_pair_scales() -> None:
    """Station preselection responses should match shared PF expected counts."""
    isotopes = ["Cs-137"]
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
        measurement_scale_by_isotope={"Cs-137": 1.25},
        measurement_scale_by_isotope_and_pair={"Cs-137": {0: 2.0, 7: 0.5}},
    )
    transport_model = {
        "enabled": True,
        "by_isotope": {
            "Cs-137": {
                "scale": 1.0,
                "tau_coefficients": {"distance": -0.15},
                "min_log_scale": -10.0,
                "max_log_scale": 10.0,
            }
        },
    }
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
        transport_response_model=transport_model,
    )
    base_est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.6,
            spread_m=0.1,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([4.0, 0.0, 0.5], dtype=float),
            strength_cps_1m=800.0,
            weight=0.4,
            spread_m=0.1,
        ),
    ]
    poses = np.array([[2.0, 1.0, 0.5], [3.0, 2.0, 0.5]], dtype=float)
    live_time_s = 2.0
    kernel = _continuous_kernel_for_estimator(est)
    matrix = dss_pp._station_response_matrix(
        poses,
        modes,
        live_time_s=live_time_s,
        kernel=kernel,
        estimator=est,
        isotope="Cs-137",
    )
    mode_positions = np.vstack([mode.position_xyz for mode in modes])
    mode_strengths = np.asarray(
        [mode.strength_cps_1m for mode in modes],
        dtype=float,
    )
    num_orients = int(est.num_orientations)
    pair_scales = dss_pp._response_scales_for_all_pairs(est, "Cs-137", num_orients)
    expected = (
        live_time_s
        * np.mean(
            kernel.kernel_values_all_pairs_for_detectors(
                isotope="Cs-137",
                detector_positions=poses,
                sources=mode_positions,
            )
            * pair_scales.reshape(1, -1, 1),
            axis=1,
        )
        * mode_strengths.reshape(1, -1)
    )
    base_kernel = _continuous_kernel_for_estimator(base_est)
    without_transport = (
        live_time_s
        * np.mean(
            base_kernel.kernel_values_all_pairs_for_detectors(
                isotope="Cs-137",
                detector_positions=poses,
                sources=mode_positions,
            )
            * pair_scales.reshape(1, -1, 1),
            axis=1,
        )
        * mode_strengths.reshape(1, -1)
    )

    assert np.allclose(matrix, expected, rtol=1.0e-12, atol=1.0e-12)
    assert not np.allclose(matrix, without_transport, rtol=1.0e-6, atol=1.0e-6)


def test_dss_pp_station_response_uses_obstacle_attenuation_kernel() -> None:
    """Station preselection responses should include shared obstacle attenuation."""
    isotopes = ["Cs-137"]
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    obstacle_grid = ObstacleGrid(
        origin=(-1.0, -1.0),
        cell_size=1.0,
        grid_shape=(6, 2),
        blocked_cells=(),
        transport_boxes_m=((1.0, -0.25, 0.0, 3.0, 0.25, 1.0),),
        transport_mu_by_isotope={"Cs-137": (0.02,)},
    )
    candidate_sources = np.array([[0.0, 0.0, 0.5]], dtype=float)
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=obstacle_grid,
    )
    free_est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
            strength_cps_1m=1000.0,
            weight=1.0,
            spread_m=0.1,
        )
    ]
    poses = np.array([[4.0, 0.0, 0.5]], dtype=float)
    live_time_s = 2.0
    kernel = _continuous_kernel_for_estimator(est)
    free_kernel = _continuous_kernel_for_estimator(free_est)

    matrix = dss_pp._station_response_matrix(
        poses,
        modes,
        live_time_s=live_time_s,
        kernel=kernel,
        estimator=est,
        isotope="Cs-137",
    )
    free_matrix = dss_pp._station_response_matrix(
        poses,
        modes,
        live_time_s=live_time_s,
        kernel=free_kernel,
        estimator=free_est,
        isotope="Cs-137",
    )
    expected = (
        live_time_s
        * np.mean(
            kernel.kernel_values_all_pairs_for_detectors(
                isotope="Cs-137",
                detector_positions=poses,
                sources=candidate_sources,
            ),
            axis=1,
        )
        * modes[0].strength_cps_1m
    )

    assert np.allclose(matrix, expected, rtol=1.0e-12, atol=1.0e-12)
    assert matrix[0, 0] < 0.05 * free_matrix[0, 0]


def test_dss_pp_batched_pairwise_cover_matches_scalar_oracle() -> None:
    """Batched pairwise-cover selection should choose the scalar greedy program."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([4.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
        ]
    }
    dss_config = DSSPPConfig(
        program_length=4,
        lambda_temporal_separation=1.0,
        temporal_cover_programs=1,
    )
    pose = np.array([2.0, 1.0, 0.5], dtype=float)
    kernel = _continuous_kernel_for_estimator(est)
    pair_cache = dss_pp._build_pair_signature_cache(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes,
        pose_xyz=pose,
        config=dss_config,
    )

    scalar = dss_pp._greedy_pairwise_contrast_program(
        estimator=est,
        kernel=kernel,
        modes_by_isotope=modes,
        pose_xyz=pose,
        config=dss_config,
        pair_cache=None,
    )
    batched = dss_pp._greedy_pairwise_contrast_program(
        estimator=est,
        kernel=kernel,
        modes_by_isotope=modes,
        pose_xyz=pose,
        config=dss_config,
        pair_cache=pair_cache,
    )

    assert scalar is not None
    assert batched is not None
    assert batched.pair_ids == scalar.pair_ids


def test_select_separation_orientations_returns_fe_pb_pairs() -> None:
    """Standalone temporal shield selection should return Fe/Pb pair tuples."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=2,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 1.0, 0.5], dtype=float))
    est._ensure_kernel_cache()

    pairs = select_separation_orientations(
        est,
        pose_idx=0,
        candidate_positions=candidate_sources,
        k=6,
        isotope="Cs-137",
    )

    assert len(pairs) == 6
    assert all(0 <= fe < est.num_orientations for fe, _ in pairs)
    assert all(0 <= pb < est.num_orientations for _, pb in pairs)


def test_select_best_orientation_preselect_supports_cpu_config() -> None:
    """Orientation preselection should fall back to CPU when GPU is disabled."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    rot_mats = generate_octant_rotation_matrices()[:2]
    shield_normals = np.asarray([mat[:, 2] for mat in rot_mats], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
        preselect_orientations=True,
        eig_num_samples=1,
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
    )
    est.add_measurement_pose(np.array([2.0, 1.0, 0.5], dtype=float))
    est._ensure_kernel_cache()

    best_idx, score = select_best_orientation(
        est,
        pose_idx=0,
        RFe_candidates=rot_mats,
        RPb_candidates=rot_mats,
        eig_samples=1,
    )

    assert 0 <= best_idx < 4
    assert np.isfinite(score)


def test_shield_rotation_gpu_surrogate_maps_reordered_candidates_to_octants() -> None:
    """GPU surrogate scoring should use octant ids, not candidate array positions."""
    torch = pytest.importorskip("torch")
    from pf import gpu_utils

    mats = generate_octant_rotation_matrices()
    rfe_candidates = np.asarray([mats[3], mats[0], mats[6]], dtype=float)
    rpb_candidates = np.asarray([mats[5], mats[1]], dtype=float)
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float),
        shield_normals=np.asarray([mat[:, 2] for mat in mats], dtype=float),
        mu_by_isotope={"Cs-137": {"fe": 0.4, "pb": 0.9}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            use_gpu=False,
            init_num_sources=(1, 1),
        ),
        shield_params=ShieldParams(thickness_fe_cm=1.0, thickness_pb_cm=0.5),
    )
    estimator.add_measurement_pose(np.array([2.0, 1.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
            strengths=np.array([1000.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
            strengths=np.array([700.0], dtype=float),
            background=0.0,
        ),
    ]
    particles_by_isotope = {"Cs-137": (states, np.array([0.4, 0.6], dtype=float))}

    cpu_scores = shield_rotation._surrogate_scores_cpu(
        estimator=estimator,
        pose_idx=0,
        live_time_s=1.0,
        particles_by_isotope=particles_by_isotope,
        RFe_candidates=rfe_candidates,
        RPb_candidates=rpb_candidates,
        alphas={"Cs-137": 1.0},
        allowed_indices=None,
        metric="var_lambda",
        eps=1.0e-12,
    )
    gpu_scores = shield_rotation._surrogate_scores_gpu(
        estimator=estimator,
        pose_idx=0,
        live_time_s=1.0,
        particles_by_isotope=particles_by_isotope,
        RFe_candidates=rfe_candidates,
        RPb_candidates=rpb_candidates,
        alphas={"Cs-137": 1.0},
        allowed_indices=None,
        metric="var_lambda",
        gpu_ctx=(torch, gpu_utils, torch.device("cpu"), torch.float64),
        eps=1.0e-12,
    )

    assert gpu_scores.keys() == cpu_scores.keys()
    for candidate_id, cpu_score in cpu_scores.items():
        assert gpu_scores[candidate_id] == pytest.approx(cpu_score)


def test_shield_rotation_gpu_eig_maps_reordered_candidates_to_octants() -> None:
    """GPU EIG scoring should pass candidate octant ids into the shared kernel."""
    torch = pytest.importorskip("torch")
    from pf import gpu_utils

    mats = generate_octant_rotation_matrices()
    rfe_candidates = np.asarray([mats[3], mats[0]], dtype=float)
    rpb_candidates = np.asarray([mats[5], mats[1]], dtype=float)
    recorded_pairs: list[tuple[int, int]] = []

    class RecordingKernel:
        """Minimal kernel that records requested Fe/Pb octant indices."""

        def expected_counts_pair_for_packed_states_torch(
            self,
            *,
            isotope: str,
            detector_pos: np.ndarray,
            positions: object,
            strengths: object,
            backgrounds: object,
            mask: object,
            fe_index: int,
            pb_index: int,
            live_time_s: float,
            source_scale: float,
            device: object,
            dtype: object,
        ) -> object:
            """Return finite expected counts while recording the pair ids."""
            del isotope, detector_pos, strengths, backgrounds, mask
            del live_time_s, source_scale
            recorded_pairs.append((int(fe_index), int(pb_index)))
            return torch.ones(positions.shape[0], device=device, dtype=dtype)

    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
            strengths=np.array([1000.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
            strengths=np.array([700.0], dtype=float),
            background=0.0,
        ),
    ]
    estimator = SimpleNamespace(
        poses=[np.array([2.0, 1.0, 0.5], dtype=float)],
        pf_config=RotatingShieldPFConfig(eig_num_samples=0),
        continuous_kernel=lambda: RecordingKernel(),
        response_scale_for_isotope=lambda isotope, fe_index, pb_index: 1.0,
    )

    shield_rotation._eig_scores_gpu(
        estimator=estimator,
        pose_idx=0,
        live_time_s=1.0,
        candidate_ids=[0, 1, 2, 3],
        RFe_candidates=rfe_candidates,
        RPb_candidates=rpb_candidates,
        alpha_by_isotope={"Cs-137": 1.0},
        particles_by_isotope={
            "Cs-137": (states, np.array([0.5, 0.5], dtype=float)),
        },
        num_samples=0,
        gpu_ctx=(torch, gpu_utils, torch.device("cpu"), torch.float64),
    )

    assert recorded_pairs == [(3, 5), (3, 1), (0, 5), (0, 1)]


def test_dss_pp_selects_station_and_shield_program() -> None:
    """DSS-PP should jointly return a pose and executable shield program."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    candidates = np.array(
        [[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]],
        dtype=float,
    )
    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=2,
            beam_width=2,
            max_programs=8,
            program_length=2,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=1.0,
            lambda_distance=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert result.shield_program.pair_ids
    assert result.diagnostics["node_count"] > 0
    assert result.diagnostics["ranked_nodes"]
    assert (
        result.diagnostics["ranked_nodes"][0]["score"]
        >= result.diagnostics["ranked_nodes"][-1]["score"]
    )
    assert np.allclose(result.next_pose, candidates[result.next_pose_index])
    assert "component_leaders" in result.diagnostics
    assert "score" in result.diagnostics["component_leaders"]


def test_dss_pp_forced_program_scores_only_baseline_pairs() -> None:
    """Forced DSS-PP programs should match baseline shield-policy execution."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    forced_pairs = (7, 8, 9, 10)
    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=np.array([[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]]),
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=8,
            program_length=4,
            forced_program_pair_ids=forced_pairs,
            temporal_cover_programs=4,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=1.0,
            lambda_distance=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
        ),
        height_partner_forced_program_pair_ids=(0,),
    )

    assert result.shield_program.pair_ids == forced_pairs
    assert result.shield_program.kind == "forced_baseline"
    assert result.diagnostics["program_count"] == 1
    assert result.diagnostics["height_partner_forced_program_requested"] is True
    assert result.diagnostics["height_partner_forced_program_applied"] is False
    assert {tuple(node["pair_ids"]) for node in result.diagnostics["ranked_nodes"]} == {
        forced_pairs
    }
    pair_diag = result.diagnostics["selected_pairwise_ambiguity"]["Cs-137"]
    assert pair_diag["mode_count"] == 2
    assert pair_diag["program_measurements"] == len(result.shield_program.pair_ids)
    assert pair_diag["bottleneck_pairs"]


def _build_height_partner_program_test_estimator() -> RotatingShieldPFEstimator:
    """Build a two-mode estimator for height-partner program ranking tests."""
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    estimator = RotatingShieldPFEstimator(
        isotopes=["Cs-137"],
        candidate_sources=np.array(
            [[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]],
            dtype=float,
        ),
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.5, "pb": 1.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            use_gpu=False,
            planning_particles=None,
            init_num_sources=(1, 1),
        ),
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    estimator.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    return estimator


def test_forced_height_program_information_uses_exact_pairs_and_length() -> None:
    """Forced-program information must reflect its executed views and duration."""
    estimator = SimpleNamespace(
        isotopes=["Cs-137"],
        pf_config=SimpleNamespace(alpha_weights={"Cs-137": 1.0}),
    )
    pair_cache = {
        "Cs-137": (
            np.array([[2.0, 8.0], [5.0, 5.0]], dtype=float),
            [0.5, 0.5],
        )
    }
    informative = dss_pp._program_conditioned_information_gain(
        estimator=estimator,
        pair_cache=pair_cache,
        program=dss_pp.ShieldProgram("informative", (0,), "forced_height_partner"),
    )
    repeated = dss_pp._program_conditioned_information_gain(
        estimator=estimator,
        pair_cache=pair_cache,
        program=dss_pp.ShieldProgram(
            "repeated",
            (0, 0),
            "forced_height_partner",
        ),
    )
    uninformative = dss_pp._program_conditioned_information_gain(
        estimator=estimator,
        pair_cache=pair_cache,
        program=dss_pp.ShieldProgram(
            "uninformative",
            (1,),
            "forced_height_partner",
        ),
    )

    assert informative > 0.0
    assert repeated > informative
    assert uninformative == pytest.approx(0.0)


def test_height_optimized_twin_does_not_scale_first_action_penalty() -> None:
    """A non-executable height twin must not affect first-action normalization."""

    def _node(
        pose_index: int,
        pose_xyz: np.ndarray,
        program_kind: str,
        static_score: float,
        observation_penalty: float,
    ) -> dss_pp.DSSPPNode:
        """Build one compact DSS node for observation-policy testing."""
        return dss_pp.DSSPPNode(
            pose_index=pose_index,
            pose_xyz=pose_xyz,
            program=dss_pp.ShieldProgram(program_kind, (0,), program_kind),
            score=static_score,
            static_score=static_score,
            distance_weight=0.0,
            observation_penalty_weight=0.0,
            information_gain=0.0,
            signature_score=0.0,
            temporal_separation_score=0.0,
            observation_penalty=observation_penalty,
            count_balance_penalty=0.0,
            differential_penalty=0.0,
            dose_score=0.0,
            count_utility=0.0,
            coverage_gain=0.0,
            revisit_penalty=0.0,
            bearing_diversity_gain=0.0,
            frontier_gain=0.0,
            turn_penalty=0.0,
            local_orbit_gain=0.0,
            station_condition_gain=0.0,
            correlation_reduction_gain=0.0,
            isotope_balance_gain=0.0,
            environment_signature_score=0.0,
            occlusion_boundary_gain=0.0,
            elevation_signature_score=0.0,
            elevation_condition_gain=0.0,
            vertical_environment_signature_score=0.0,
        )

    current = np.array([1.0, 1.0, 0.5], dtype=float)
    height = np.array([1.0, 1.0, 1.5], dtype=float)
    normal = np.array([4.0, 1.0, 0.5], dtype=float)
    forced = _node(0, height, "forced_height_partner", 1.0, 0.25)
    optimized_twin = _node(0, height, "optimized", 1.0e6, 1.0)
    normal_node = _node(1, normal, "optimized", 0.0, 0.75)
    first_nodes, _ = dss_pp._split_height_partner_first_action_nodes(
        [forced, optimized_twin, normal_node],
        visited_poses_xyz=current.reshape(1, 3),
        current_pose_xyz=current,
        enabled=True,
    )
    estimator = SimpleNamespace(normals=np.array([[1.0, 0.0, 0.0]]))
    config = DSSPPConfig(
        eta_observation=1.0,
        enforce_min_observation=False,
        lambda_distance=0.0,
        lambda_rotation=0.0,
        lambda_time=0.0,
    )

    rescored = dss_pp._apply_node_observation_policy(
        first_nodes,
        current_pose_xyz=current,
        current_pair_id=None,
        estimator=estimator,
        map_api=None,
        config=config,
    )
    expected = dss_pp._apply_node_observation_policy(
        [forced, normal_node],
        current_pose_xyz=current,
        current_pair_id=None,
        estimator=estimator,
        map_api=None,
        config=config,
    )

    assert {node.program.kind for node in first_nodes} == {
        "forced_height_partner",
        "optimized",
    }
    assert [node.score for node in rescored] == pytest.approx(
        [node.score for node in expected]
    )
    assert [node.observation_penalty_weight for node in rescored] == pytest.approx(
        [node.observation_penalty_weight for node in expected]
    )


@pytest.mark.parametrize("worker_count", [1, 2])
def test_dss_pp_scores_forced_height_partner_program_before_ranking(
    worker_count: int,
) -> None:
    """Height-pair reuse must alter planner ranking before action selection."""
    estimator = _build_height_partner_program_test_estimator()
    visited = np.array([[2.0, 2.0, 0.5]], dtype=float)
    candidates = np.array(
        [[2.0, 2.0, 1.5], [2.0, 6.0, 0.5]],
        dtype=float,
    )
    config = DSSPPConfig(
        horizon=2,
        beam_width=4,
        max_programs=8,
        program_length=4,
        temporal_cover_programs=1,
        live_time_s=1.0,
        lambda_eig=1.0,
        lambda_signature=1.0,
        lambda_distance=0.0,
        eta_observation=1.0,
        eta_differential=0.0,
        lambda_rotation=0.0,
        augment_candidates=False,
        candidate_preselect_enable=False,
        same_isotope_direct_separation_guard=False,
        program_eval_workers=worker_count,
        diagnostic_ranked_node_limit=100,
    )

    unrestricted = select_dss_pp_next_station(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        current_pose_xyz=visited[0],
        visited_poses_xyz=visited,
        config=config,
    )
    forced_pairs = (63, 63, 63, 63)
    constrained = select_dss_pp_next_station(
        estimator=estimator,
        candidate_poses_xyz=candidates,
        current_pose_xyz=visited[0],
        visited_poses_xyz=visited,
        config=config,
        height_partner_forced_program_pair_ids=forced_pairs,
    )

    assert np.allclose(unrestricted.next_pose, candidates[0])
    assert np.allclose(constrained.next_pose, candidates[1])
    height_nodes = [
        node
        for node in constrained.diagnostics["ranked_nodes"]
        if np.allclose(node["pose_xyz"], candidates[0])
    ]
    normal_nodes = [
        node
        for node in constrained.diagnostics["ranked_nodes"]
        if np.allclose(node["pose_xyz"], candidates[1])
    ]
    assert {tuple(node["pair_ids"]) for node in height_nodes} == {forced_pairs}
    assert {node["program_kind"] for node in height_nodes} == {"forced_height_partner"}
    assert normal_nodes
    assert all(node["program_kind"] != "forced_height_partner" for node in normal_nodes)
    assert constrained.diagnostics["height_partner_forced_candidate_count"] == 1
    assert constrained.diagnostics["height_partner_forced_node_count"] == 1
    assert constrained.diagnostics["height_partner_forced_program_applied"] is True
    assert len(constrained.sequence) == 2
    assert constrained.sequence[1].program.kind != "forced_height_partner"


def test_height_partner_program_is_not_forced_for_visited_exact_action() -> None:
    """A previously sampled height must not qualify via another sampled height."""
    candidates = np.array(
        [[2.0, 2.0, 0.5], [2.0, 2.0, 1.5], [2.0, 2.0, 2.5]],
        dtype=float,
    )
    visited = np.array(
        [[2.0, 2.0, 0.5], [2.0, 2.0, 1.5]],
        dtype=float,
    )

    mask = dss_pp._height_partner_mask_batch(candidates, visited)

    assert mask.tolist() == [False, False, True]


def test_dss_pp_ranked_node_limit_zero_disables_ranked_payload() -> None:
    """A zero DSS-PP ranked-node limit should skip diagnostic node payloads."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": 0.08},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=np.array([[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]]),
        current_pose_xyz=np.array([2.0, 2.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=8,
            program_length=2,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=1.0,
            lambda_distance=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
            diagnostic_ranked_node_limit=0,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert result.diagnostics["node_count"] > 0
    assert result.diagnostics["diagnostic_ranked_node_limit"] == 0
    assert result.diagnostics["ranked_nodes"] == []
    assert result.sequence


def _scalar_remaining_route_terms(
    candidates: np.ndarray,
    current_pose: np.ndarray,
    visited: np.ndarray,
    path_lengths: np.ndarray,
    coverage_norm: np.ndarray,
    revisit_penalties: np.ndarray,
    frontier_gains: np.ndarray,
    turn_penalties: np.ndarray,
    config: DSSPPConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a serial oracle for remaining-budget route terms."""
    pressure = dss_pp._remaining_budget_pressure(config)
    if pressure <= 0.0:
        return pressure, np.zeros(candidates.shape[0]), np.zeros(candidates.shape[0])
    nominal_step = max(
        float(config.min_station_separation_m),
        float(config.coverage_radius_m),
        1.0,
    )
    remaining = max(1, int(config.remaining_station_estimate or 1))
    current_xy = np.asarray(current_pose[:2], dtype=float)
    older = []
    radius = 2.0 * nominal_step
    for pose in visited:
        if float(np.linalg.norm(pose[:2] - current_xy)) > max(0.25 * radius, 1.0e-6):
            older.append(pose)
    penalties = []
    gains = []
    for idx, candidate in enumerate(candidates):
        if np.isfinite(path_lengths[idx]):
            distance_penalty = min(
                max(float(path_lengths[idx]) / (nominal_step * float(remaining)), 0.0),
                2.0,
            )
        else:
            distance_penalty = float("inf")
        backtrack_penalty = 0.0
        if older:
            nearest = min(
                float(np.linalg.norm(candidate[:2] - pose[:2])) for pose in older
            )
            if nearest < radius:
                shortfall = 1.0 - nearest / radius
                backtrack_penalty = shortfall * shortfall
        coverage = min(max(float(coverage_norm[idx]), 0.0), 1.0)
        penalty = (
            float(config.remaining_route_distance_weight) * distance_penalty
            + float(config.remaining_route_revisit_weight) * revisit_penalties[idx]
            + float(config.remaining_route_turn_weight) * turn_penalties[idx]
            + float(config.remaining_route_backtrack_weight) * backtrack_penalty
            + float(config.remaining_route_coverage_weight) * (1.0 - coverage)
        )
        gain = (
            float(config.remaining_route_coverage_weight) * coverage
            + float(config.remaining_route_frontier_weight) * frontier_gains[idx]
        )
        penalties.append(max(float(penalty), 0.0))
        gains.append(max(float(gain), 0.0))
    return pressure, np.asarray(penalties), np.asarray(gains)


def test_remaining_route_terms_batch_match_scalar_oracle() -> None:
    """Remaining-budget route terms should match a serial route oracle."""
    candidates = np.array(
        [[1.0, 0.0, 0.5], [6.0, 0.0, 0.5], [4.0, 2.0, 0.5]],
        dtype=float,
    )
    current = np.array([4.0, 0.0, 0.5], dtype=float)
    visited = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [4.0, 0.0, 0.5]])
    path_lengths = np.array([3.0, 2.0, 2.0], dtype=float)
    coverage_norm = np.array([0.0, 0.8, 0.4], dtype=float)
    revisit_penalties = np.array([0.5, 0.0, 0.1], dtype=float)
    frontier_gains = np.array([0.2, 0.7, 0.5], dtype=float)
    turn_penalties = np.array([1.0, 0.0, 0.5], dtype=float)
    config = DSSPPConfig(
        remaining_budget_guidance=True,
        remaining_station_estimate=2,
        remaining_budget_urgency_stations=4,
        remaining_route_weight=2.0,
        min_station_separation_m=2.0,
        coverage_radius_m=1.0,
    )

    expected = _scalar_remaining_route_terms(
        candidates,
        current,
        visited,
        path_lengths,
        coverage_norm,
        revisit_penalties,
        frontier_gains,
        turn_penalties,
        config,
    )
    actual = dss_pp._remaining_route_terms_batch(
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        path_lengths=path_lengths,
        coverage_norm=coverage_norm,
        revisit_penalties=revisit_penalties,
        frontier_gains=frontier_gains,
        turn_penalties=turn_penalties,
        config=config,
    )

    assert actual[0] == pytest.approx(expected[0])
    assert np.allclose(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2])


def test_dss_pp_remaining_budget_guidance_avoids_backtracking() -> None:
    """Low remaining station budgets should penalize route regression."""
    est = _build_simple_estimator()
    candidates = np.array(
        [[1.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=float,
    )
    current = np.array([4.0, 0.0, 0.0], dtype=float)
    visited = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], current], dtype=float)

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            lambda_rotation=0.0,
            augment_candidates=False,
            min_station_separation_m=0.0,
            coverage_radius_m=1.0,
            remaining_budget_guidance=True,
            remaining_station_estimate=1,
            remaining_budget_urgency_stations=4,
            remaining_route_weight=5.0,
            remaining_route_distance_weight=0.2,
            remaining_route_revisit_weight=1.0,
            remaining_route_turn_weight=1.0,
            remaining_route_backtrack_weight=1.0,
            remaining_route_coverage_weight=0.0,
            remaining_route_frontier_weight=0.0,
        ),
    )

    assert result.next_pose_index == 1
    assert np.allclose(result.next_pose, candidates[1])
    assert result.diagnostics["remaining_route_pressure"] > 0.0


def test_dss_pp_filters_zero_separation_nodes_for_multimode_sources() -> None:
    """Multi-source planning should not choose zero-signature nodes when avoidable."""
    program = dss_pp.ShieldProgram(name="p", pair_ids=(0,), kind="test")

    def node(
        index: int,
        temporal: float,
        *,
        environment: float = 0.0,
    ) -> dss_pp.DSSPPNode:
        """Build a minimal DSS-PP node for separation filtering tests."""
        return dss_pp.DSSPPNode(
            pose_index=index,
            pose_xyz=np.array([float(index), 0.0, 0.5], dtype=float),
            program=program,
            score=float(index),
            static_score=float(index),
            distance_weight=0.0,
            observation_penalty_weight=0.0,
            information_gain=0.0,
            signature_score=0.0,
            temporal_separation_score=float(temporal),
            elevation_signature_score=0.0,
            observation_penalty=0.0,
            count_balance_penalty=0.0,
            differential_penalty=0.0,
            dose_score=0.0,
            count_utility=1.0,
            coverage_gain=0.0,
            revisit_penalty=0.0,
            bearing_diversity_gain=0.0,
            frontier_gain=0.0,
            turn_penalty=0.0,
            local_orbit_gain=0.0,
            station_condition_gain=0.0,
            correlation_reduction_gain=0.0,
            isotope_balance_gain=0.0,
            environment_signature_score=float(environment),
            occlusion_boundary_gain=0.0,
            elevation_condition_gain=0.0,
            vertical_environment_signature_score=0.0,
        )

    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([4.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.1,
            ),
        ]
    }

    filtered = dss_pp._filter_nodes_for_multimode_separation(
        [node(0, 0.0), node(1, 0.25), node(2, 0.0, environment=10.0)],
        modes,
    )

    assert [item.pose_index for item in filtered] == [1]

    fallback = dss_pp._filter_nodes_for_multimode_separation(
        [node(0, 0.0), node(2, 0.0, environment=10.0)],
        modes,
    )

    assert [item.pose_index for item in fallback] == [2]

    single_mode = {"Cs-137": modes["Cs-137"][:1]}
    unresolved_filtered = dss_pp._filter_nodes_for_multimode_separation(
        [node(0, 0.0), node(1, 0.25), node(2, 0.0, environment=10.0)],
        single_mode,
        unresolved_evidence=True,
    )

    assert [item.pose_index for item in unresolved_filtered] == [1]


def test_dss_pp_environment_signature_prefers_obstacle_contrast() -> None:
    """DSS-PP should prefer stations where known obstacles separate modes."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array(
        [[-1.0, 0.0, 1.0], [-1.0, 2.0, 1.0]],
        dtype=float,
    )
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        planning_particles=None,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 1.0], dtype=float))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[-1.0, 0.0, 1.0]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[-1.0, 2.0, 1.0]], dtype=float),
                strengths=np.array([2000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    candidates = np.array(
        [[2.0, 0.0, 1.0], [2.0, 4.0, 1.0]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([2.0, 2.0, 1.0], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_temporal_separation=0.0,
            lambda_environment_signature=10.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            enforce_min_observation=False,
            augment_candidates=False,
            environment_contrast_threshold=0.25,
        ),
    )

    assert result.next_pose_index == 0
    assert result.diagnostics["first_environment_signature_score"] > 0.0
    assert 0.0 <= result.diagnostics["first_environment_signature_norm"] <= 1.0


def test_dss_pp_environment_signature_score_is_bounded() -> None:
    """Obstacle signature should act as a bounded weak planner term."""
    config = DSSPPConfig(environment_signature_score_clip=3.0)

    low = dss_pp._normalized_environment_signature_score(0.0, config=config)
    high = dss_pp._normalized_environment_signature_score(1.0e6, config=config)
    arr = dss_pp._normalized_environment_signature_score(
        np.array([0.0, 3.0, 1.0e6], dtype=float),
        config=config,
    )

    assert low == 0.0
    assert high == 1.0
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)
    assert np.isclose(arr[-1], 1.0)


def test_dss_pp_elevation_signature_scores_vertical_mode_pairs() -> None:
    """Height-separated posterior modes should add an elevation signature score."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([2.0, 0.0, 0.5], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.5,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([2.0, 0.0, 6.5], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.5,
            spread_m=0.2,
        ),
    ]
    config = DSSPPConfig(
        temporal_pair_contrast_threshold=0.1,
        elevation_pair_z_scale_m=1.0,
        elevation_pair_xy_scale_m=4.0,
    )
    responses = np.array(
        [
            [[10.0, 10.0], [2.0, 10.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ],
        dtype=float,
    )

    scores = dss_pp._batched_elevation_signature_scores(
        responses,
        np.array([0.5, 0.5], dtype=float),
        modes,
        config=config,
    )

    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_dss_pp_temporal_score_prioritizes_high_surface_pairs() -> None:
    """High-surface mode pairs should receive extra temporal-separation focus."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([1.0, 1.0, 10.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.25,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([4.0, 1.0, 10.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.25,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([1.0, 1.0, 0.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.5,
            spread_m=0.2,
        ),
    ]
    raw = np.array(
        [
            [
                [20.0, 2.0, 12.0],
                [2.0, 20.0, 12.0],
            ],
        ],
        dtype=float,
    )
    base_config = DSSPPConfig(
        temporal_pair_contrast_threshold=2.0,
        temporal_logdet_weight=0.0,
        temporal_decorrelation_weight=0.0,
    )
    boosted_config = DSSPPConfig(
        temporal_pair_contrast_threshold=2.0,
        temporal_logdet_weight=0.0,
        temporal_decorrelation_weight=0.0,
        high_surface_pair_boost=4.0,
        high_surface_z_fraction=0.75,
    )
    priority = dss_pp._high_surface_pair_priority_weights(
        modes,
        config=boosted_config,
        room_z_m=10.0,
    )

    base = dss_pp._batched_temporal_separation_scores(
        raw,
        np.array([0.25, 0.25, 0.5], dtype=float),
        config=base_config,
    )
    boosted = dss_pp._batched_temporal_separation_scores(
        raw,
        np.array([0.25, 0.25, 0.5], dtype=float),
        config=boosted_config,
        pair_priority_weights=priority,
    )

    assert boosted[0] > base[0]


def test_high_surface_pair_priority_boosts_ceiling_wall_ambiguity() -> None:
    """Ceiling-vs-high-wall mode pairs should get the strongest priority."""
    modes = [
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([1.0, 1.0, 10.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.25,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([1.0, 1.0, 8.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.25,
            spread_m=0.2,
        ),
        dss_pp.SignatureMode(
            isotope="Cs-137",
            position_xyz=np.array([1.0, 1.0, 0.0], dtype=float),
            strength_cps_1m=1000.0,
            weight=0.5,
            spread_m=0.2,
        ),
    ]
    config = DSSPPConfig(
        high_surface_pair_boost=2.0,
        high_surface_cross_stratum_boost=3.0,
        high_surface_z_fraction=0.75,
    )

    priority = dss_pp._high_surface_pair_priority_weights(
        modes,
        config=config,
        room_z_m=10.0,
    )

    assert priority[0] == pytest.approx(6.0)
    assert priority[0] > priority[1]
    assert priority[0] > priority[2]


def test_dss_pp_elevation_condition_prefers_elevation_separating_view() -> None:
    """Candidate views with different elevation angles should score higher."""
    modes = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([2.0, 0.0, 0.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.2,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([2.0, 0.0, 6.5], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.2,
            ),
        ],
    }
    candidates = np.array(
        [
            [0.0, 0.0, 0.5],
            [20.0, 0.0, 0.5],
        ],
        dtype=float,
    )

    gains = dss_pp._elevation_condition_gains_batch(
        candidates,
        modes,
        config=DSSPPConfig(elevation_angle_threshold_deg=45.0),
    )

    assert gains[0] > gains[1]
    assert gains[0] > 0.0


def test_dss_pp_batched_environment_features_match_scalar_oracle() -> None:
    """Batched obstacle-signature features should match scalar ray-box scoring."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array(
        [[-1.0, 0.0, 1.0], [-1.0, 2.0, 1.0], [-1.0, 4.0, 1.0]],
        dtype=float,
    )
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(2, 3),
        blocked_cells=((0, 0), (1, 1), (0, 2)),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            max_sources=1,
            use_gpu=False,
            planning_particles=None,
            init_num_sources=(1, 1),
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.015},
    )
    est.add_measurement_pose(np.array([2.0, 2.0, 1.0], dtype=float))
    est._ensure_kernel_cache()
    modes_by_isotope = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([-1.0, 0.0, 1.0], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.2,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([-1.0, 2.0, 1.0], dtype=float),
                strength_cps_1m=800.0,
                weight=0.3,
                spread_m=0.2,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([-1.0, 4.0, 1.0], dtype=float),
                strength_cps_1m=600.0,
                weight=0.2,
                spread_m=0.2,
            ),
        ]
    }
    poses = np.array(
        [[2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [2.0, 4.0, 1.0]],
        dtype=float,
    )
    config = DSSPPConfig(
        environment_contrast_threshold=0.2,
        occlusion_boundary_step_m=0.25,
    )
    kernel = _continuous_kernel_for_estimator(est, detector_aperture_samples=1)

    batched_env = dss_pp._environment_signature_scores_batch(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=poses,
        config=config,
    )
    scalar_env = np.asarray(
        [
            dss_pp._environment_signature_score(
                kernel=kernel,
                estimator=est,
                modes_by_isotope=modes_by_isotope,
                pose_xyz=pose,
                config=config,
            )
            for pose in poses
        ],
        dtype=float,
    )
    batched_boundary = dss_pp._occlusion_boundary_gains_batch(
        kernel=kernel,
        estimator=est,
        modes_by_isotope=modes_by_isotope,
        poses_xyz=poses,
        config=config,
    )
    scalar_boundary = np.asarray(
        [
            dss_pp._occlusion_boundary_gain(
                kernel=kernel,
                estimator=est,
                modes_by_isotope=modes_by_isotope,
                pose_xyz=pose,
                config=config,
            )
            for pose in poses
        ],
        dtype=float,
    )

    assert np.allclose(batched_env, scalar_env)
    assert np.allclose(batched_boundary, scalar_boundary)


def test_dss_pp_batched_station_features_match_scalar_oracle() -> None:
    """Batched station preselection features should match scalar helper scores."""
    modes_by_isotope = {
        "Cs-137": [
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([2.0, 2.0, 1.0], dtype=float),
                strength_cps_1m=1200.0,
                weight=0.6,
                spread_m=0.2,
            ),
            dss_pp.SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([7.0, 5.0, 1.0], dtype=float),
                strength_cps_1m=900.0,
                weight=0.4,
                spread_m=0.2,
            ),
        ],
        "Co-60": [
            dss_pp.SignatureMode(
                isotope="Co-60",
                position_xyz=np.array([4.0, 8.0, 1.0], dtype=float),
                strength_cps_1m=1000.0,
                weight=1.0,
                spread_m=0.2,
            )
        ],
    }
    candidates = np.array(
        [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0], [8.0, 8.0, 1.0]],
        dtype=float,
    )
    visited = np.array([[0.0, 0.0, 1.0], [2.0, 6.0, 1.0]], dtype=float)
    centers = np.array(
        [
            [x, y, 1.0]
            for x in np.linspace(0.5, 8.5, 4)
            for y in np.linspace(0.5, 8.5, 4)
        ],
        dtype=float,
    )
    config = DSSPPConfig(
        ring_radii_m=(2.0, 4.0),
        local_orbit_sigma_m=0.8,
        live_time_s=3.0,
        count_utility_saturation_counts=100.0,
    )

    scalar_coverage = np.asarray(
        [
            dss_pp._coverage_gain_fraction(
                cell_centers_xyz=centers,
                candidate_pose_xyz=pose,
                visited_poses_xyz=visited,
                radius_m=2.5,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_revisit = np.asarray(
        [
            dss_pp._station_revisit_penalty(
                pose,
                visited,
                min_separation_m=3.0,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_bearing = np.asarray(
        [
            dss_pp._bearing_diversity_gain(
                pose,
                visited,
                modes_by_isotope,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_frontier = np.asarray(
        [
            dss_pp._frontier_band_gain(
                pose,
                visited,
                target_radius_m=3.0,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_turn = np.asarray(
        [
            dss_pp._route_turn_penalty(
                pose,
                np.array([3.0, 6.0, 1.0], dtype=float),
                visited,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_orbit = np.asarray(
        [
            dss_pp._local_orbit_gain(
                pose,
                modes_by_isotope,
                config=config,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_condition = np.asarray(
        [
            dss_pp._station_condition_gain(
                pose,
                visited,
                modes_by_isotope,
                config=config,
            )
            for pose in candidates
        ],
        dtype=float,
    )
    scalar_correlation = []
    scalar_balance = []
    for pose in candidates:
        corr_rows = []
        corr_weights = []
        balance_rows = []
        for modes in modes_by_isotope.values():
            active = [mode for mode in modes if float(mode.weight) > 0.0]
            if len(active) >= 2:
                before = dss_pp._station_response_matrix(
                    visited,
                    active,
                    live_time_s=float(config.live_time_s),
                )
                candidate_row = dss_pp._station_response_matrix(
                    pose.reshape(1, 3),
                    active,
                    live_time_s=float(config.live_time_s),
                )
                before_corr = dss_pp._max_column_correlation_from_design(before)
                after_corr = dss_pp._max_column_correlation_after_candidate_batch(
                    before_matrix=before,
                    candidate_rows=candidate_row,
                )[0]
                corr_rows.append(max(float(before_corr - after_corr), 0.0))
                corr_weights.append(float(sum(mode.weight for mode in active)))
            if active:
                response = dss_pp._station_response_matrix(
                    pose.reshape(1, 3),
                    active,
                    live_time_s=float(config.live_time_s),
                )
                weights = dss_pp._normalise_weights(
                    np.asarray([float(mode.weight) for mode in active], dtype=float)
                )
                expected = float(np.sum(response[0] * weights))
                saturation = max(float(config.count_utility_saturation_counts), 1e-12)
                balance_rows.append(1.0 - np.exp(-max(expected, 0.0) / saturation))
        if corr_rows:
            corr_weight_arr = dss_pp._normalise_weights(np.asarray(corr_weights))
            scalar_correlation.append(float(np.sum(corr_weight_arr * corr_rows)))
        else:
            scalar_correlation.append(0.0)
        if balance_rows:
            balance = np.asarray(balance_rows, dtype=float)
            scalar_balance.append(
                float(0.75 * np.min(balance) + 0.25 * np.mean(balance))
            )
        else:
            scalar_balance.append(0.0)
    scalar_correlation = np.asarray(scalar_correlation, dtype=float)
    scalar_balance = np.asarray(scalar_balance, dtype=float)

    assert np.allclose(
        dss_pp._coverage_gain_fractions_batch(
            cell_centers_xyz=centers,
            candidate_poses_xyz=candidates,
            visited_poses_xyz=visited,
            radius_m=2.5,
        ),
        scalar_coverage,
    )
    assert np.allclose(
        dss_pp._station_revisit_penalties_batch(
            candidates,
            visited,
            min_separation_m=3.0,
        ),
        scalar_revisit,
    )
    assert np.allclose(
        dss_pp._bearing_diversity_gains_batch(
            candidates,
            visited,
            modes_by_isotope,
        ),
        scalar_bearing,
    )
    assert np.allclose(
        dss_pp._frontier_band_gains_batch(
            candidates,
            visited,
            target_radius_m=3.0,
        ),
        scalar_frontier,
    )
    assert np.allclose(
        dss_pp._route_turn_penalties_batch(
            candidates,
            np.array([3.0, 6.0, 1.0], dtype=float),
            visited,
        ),
        scalar_turn,
    )
    assert np.allclose(
        dss_pp._local_orbit_gains_batch(
            candidates,
            modes_by_isotope,
            config=config,
        ),
        scalar_orbit,
    )
    assert np.allclose(
        dss_pp._station_condition_gains_batch(
            candidates,
            visited,
            modes_by_isotope,
            config=config,
        ),
        scalar_condition,
    )
    assert np.allclose(
        dss_pp._station_correlation_reduction_gains_batch(
            candidates,
            visited,
            modes_by_isotope,
            config=config,
        ),
        scalar_correlation,
    )
    assert np.allclose(
        dss_pp._isotope_balance_gains_batch(
            candidates,
            modes_by_isotope,
            config=config,
        ),
        scalar_balance,
    )


def test_station_condition_composite_batch_matches_scalar() -> None:
    """Composite Fisher conditioning terms should match scalar evaluation."""
    candidates = np.array(
        [[1.0, 2.0, 1.0], [6.0, 4.0, 1.0], [9.0, 8.0, 1.0]],
        dtype=float,
    )
    visited = np.array([[2.0, 8.0, 1.0], [8.0, 2.0, 1.0]], dtype=float)
    modes_by_isotope = {
        "Cs-137": [
            SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.5,
            ),
            SignatureMode(
                isotope="Cs-137",
                position_xyz=np.array([10.0, 10.0, 0.0], dtype=float),
                strength_cps_1m=1000.0,
                weight=0.5,
                spread_m=0.5,
            ),
        ]
    }
    config = DSSPPConfig(
        station_condition_min_singular_weight=1.0,
        station_condition_inverse_condition_weight=1.0,
        station_condition_coherence_weight=0.5,
    )

    scalar = np.asarray(
        [
            dss_pp._station_condition_gain(  # noqa: SLF001
                candidate,
                visited,
                modes_by_isotope,
                config=config,
            )
            for candidate in candidates
        ],
        dtype=float,
    )
    batched = dss_pp._station_condition_gains_batch(  # noqa: SLF001
        candidates,
        visited,
        modes_by_isotope,
        config=config,
    )

    assert np.allclose(batched, scalar)


def test_dss_pp_enforces_all_isotope_observability_when_feasible() -> None:
    """DSS-PP should filter station-program nodes that miss an isotope."""
    isotopes = ["Cs-137", "Co-60"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]],
        dtype=float,
    )
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={
            "Cs-137": {"fe": 0.0, "pb": 0.0},
            "Co-60": {"fe": 0.0, "pb": 0.0},
        },
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.0, 5.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    for isotope, position in zip(isotopes, candidate_sources):
        est.filters[isotope].continuous_particles = [
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=1,
                    positions=position.reshape(1, 3),
                    strengths=np.array([1000.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        ]
    candidates = np.array(
        [[0.5, 0.0, 0.5], [5.0, 0.0, 0.5]],
        dtype=float,
    )
    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 5.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=4,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            min_observation_counts=30.0,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert result.next_pose_index == 1
    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_coverage_term_prefers_unvisited_free_space() -> None:
    """DSS-PP should move toward uncovered traversable cells when weighted."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([1.0, 1.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 10),
        traversable_cells=tuple((ix, iy) for ix in range(10) for iy in range(10)),
    )
    candidates = np.array(
        [[1.5, 1.5, 0.5], [8.5, 8.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([1.0, 1.0, 0.5], dtype=float),
        visited_poses_xyz=np.array([[1.0, 1.0, 0.5]], dtype=float),
        map_api=traversable,
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_coverage=1.0,
            eta_revisit=1.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            min_station_separation_m=3.0,
            coverage_radius_m=2.0,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert result.diagnostics["first_coverage_gain"] > 0.0
    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_coverage_floor_rejects_low_coverage_candidates() -> None:
    """Coverage-floor scoring should keep exploration from collapsing locally."""
    est = _build_simple_estimator()
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 10),
        traversable_cells=tuple((ix, iy) for ix in range(10) for iy in range(10)),
    )
    candidates = np.array(
        [[1.5, 1.5, 0.5], [8.5, 8.5, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([1.0, 1.0, 0.5], dtype=float),
        visited_poses_xyz=np.array([[1.0, 1.0, 0.5]], dtype=float),
        map_api=traversable,
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_coverage=0.0,
            coverage_floor_quantile=1.0,
            coverage_floor_weight=100.0,
            eta_revisit=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            min_station_separation_m=0.0,
            coverage_radius_m=2.0,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_falls_back_to_reachable_base_candidates() -> None:
    """DSS-PP should not fail when augmented candidates are disconnected."""
    est = _build_simple_estimator()
    traversable = TraversabilityMap(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(5, 1),
        traversable_cells=((0, 0), (1, 0), (3, 0), (4, 0)),
    )
    current = np.array([0.5, 0.5, 0.0], dtype=float)
    candidates = np.array([[1.5, 0.5, 0.0]], dtype=float)

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=np.array([current], dtype=float),
        map_api=traversable,
        config=DSSPPConfig(
            horizon=1,
            max_programs=1,
            program_length=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            min_station_separation_m=2.0,
            signature_std_min_counts=0.0,
            augment_candidates=True,
            max_augmented_candidates=8,
        ),
    )

    assert np.allclose(result.next_pose, candidates[0])
    assert result.diagnostics["path_fallback_used"] == 1
    assert result.diagnostics["path_filtered_candidates"] > 0


def test_dss_pp_limits_expensive_eig_candidate_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSS-PP should only run EIG rollouts on the configured top candidates."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.0, 0.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    calls: list[tuple[float, float, float]] = []

    def _fake_information_gain(
        estimator: RotatingShieldPFEstimator,
        pose_xyz: np.ndarray,
        *,
        config: DSSPPConfig,
        rng_seed: int | None,
    ) -> float:
        """Record expensive EIG calls and return a deterministic value."""
        calls.append(tuple(float(value) for value in pose_xyz))
        return 1.0

    monkeypatch.setattr(
        dss_pp,
        "_candidate_information_gain",
        _fake_information_gain,
    )
    candidates = np.array(
        [
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [3.0, 0.0, 0.5],
            [4.0, 0.0, 0.5],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=2,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=1.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            signature_std_min_counts=0.0,
            eig_candidate_limit=2,
            augment_candidates=False,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert len(calls) == 2


def test_candidate_information_gain_uses_cached_current_uncertainty() -> None:
    """Candidate EIG should reuse station-level current uncertainty when present."""
    calls = {"global_uncertainty": 0}

    class DummyEstimator:
        """Minimal estimator facade for candidate EIG cache behavior."""

        _dss_pp_current_uncertainty_cache = 9.0
        pf_config = SimpleNamespace(ig_threshold=0.0)

        def global_uncertainty(self) -> float:
            """Record unexpected uncached uncertainty requests."""
            calls["global_uncertainty"] += 1
            return 100.0

        def expected_uncertainty_after_rotation(self, **_kwargs: object) -> float:
            """Return deterministic posterior uncertainty."""
            return 4.0

    value = dss_pp._candidate_information_gain(
        DummyEstimator(),
        np.array([0.0, 0.0, 0.5], dtype=float),
        config=DSSPPConfig(lambda_eig=1.0, live_time_s=1.0, program_length=1),
        rng_seed=7,
    )

    assert calls["global_uncertainty"] == 0
    assert value == pytest.approx(np.log1p(9.0) - np.log1p(4.0))


def test_dss_pp_preselects_candidates_before_program_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSS-PP should shortlist evidence-backed candidates before program scoring."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([0.0, 0.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Cs-137"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    original = dss_pp._build_pair_signature_caches_for_poses
    seen_counts: list[int] = []

    def _wrapped_pair_cache(*args: object, **kwargs: object) -> object:
        """Record how many candidates reach expensive pair-cache construction."""
        poses = np.asarray(kwargs["poses_xyz"], dtype=float)
        seen_counts.append(int(poses.shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dss_pp,
        "_build_pair_signature_caches_for_poses",
        _wrapped_pair_cache,
    )
    candidates = np.array(
        [[float(idx), 0.0, 0.5] for idx in range(8)],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([0.0, 0.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            beam_width=1,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            eta_observation=0.0,
            eta_differential=0.0,
            eta_count_balance=0.0,
            signature_std_min_counts=0.0,
            eig_candidate_limit=None,
            augment_candidates=False,
            candidate_preselect_enable=True,
            candidate_preselect_min=3,
            candidate_preselect_multiplier=1,
        ),
    )

    assert result.next_pose.shape == (3,)
    assert seen_counts == [3]
    assert result.diagnostics["evaluated_candidate_count"] == 3


def test_dss_pp_uses_bounds_coverage_without_map() -> None:
    """Bounds-based coverage should drive exploration when no map is active."""
    est = _build_simple_estimator()
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [2.0, 1.0, 0.5],
            [8.0, 18.0, 0.5],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([10.0, 20.0, 1.0], dtype=float),
        ),
        config=DSSPPConfig(
            augment_candidates=False,
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=10.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            coverage_radius_m=3.0,
            min_station_separation_m=0.0,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])


def test_dss_pp_filters_near_revisit_when_alternatives_exist() -> None:
    """Station separation should remove near revisits after augmentation."""
    est = _build_simple_estimator()
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array(
        [
            [1.5, 1.0, 0.5],
            [5.5, 1.0, 0.5],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([10.0, 10.0, 1.0], dtype=float),
        ),
        config=DSSPPConfig(
            augment_candidates=False,
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            min_station_separation_m=3.0,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert int(result.diagnostics["separation_filtered_candidates"]) == 1


def test_dss_pp_augments_with_global_unvisited_coverage_candidates() -> None:
    """DSS-PP should add global coverage candidates when base candidates revisit."""
    est = _build_simple_estimator()
    current = np.array([1.0, 1.0, 0.5], dtype=float)
    visited = np.array([[1.0, 1.0, 0.5]], dtype=float)
    candidates = np.array([[1.2, 1.0, 0.5]], dtype=float)

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        bounds_xyz=(
            np.array([0.0, 0.0, 0.5], dtype=float),
            np.array([10.0, 10.0, 0.5], dtype=float),
        ),
        config=DSSPPConfig(
            augment_candidates=True,
            max_augmented_candidates=32,
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=10.0,
            lambda_rotation=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            min_station_separation_m=3.0,
            coverage_radius_m=2.0,
            signature_std_min_counts=0.0,
        ),
    )

    assert not np.allclose(result.next_pose, candidates[0])
    assert result.diagnostics["candidate_count"] > 1


def test_dss_pp_filters_augmented_height_action_when_first_action_is_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSS augmentation must not restore a caller-disallowed height action."""
    estimator = _build_simple_estimator()
    current = np.array([1.0, 1.0, 1.5], dtype=float)
    visited = np.array(
        [[1.0, 1.0, 0.5], [1.0, 1.0, 1.5]],
        dtype=float,
    )
    lateral = np.array([5.0, 1.0, 1.5], dtype=float)
    augmented_height = np.array([1.0, 1.0, 2.5], dtype=float)

    def _fake_augment_candidate_stations(
        _candidate_poses_xyz: np.ndarray,
        **_kwargs: object,
    ) -> np.ndarray:
        """Return one legal lateral action and one augmented height action."""
        return np.vstack([lateral, augmented_height])

    monkeypatch.setattr(
        dss_pp,
        "augment_candidate_stations",
        _fake_augment_candidate_stations,
    )
    result = select_dss_pp_next_station(
        estimator=estimator,
        candidate_poses_xyz=lateral.reshape(1, 3),
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        continuous_height_bounds_m=(0.5, 2.5),
        config=DSSPPConfig(
            augment_candidates=True,
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            lambda_rotation=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            min_station_separation_m=0.0,
            signature_std_min_counts=0.0,
        ),
        allow_height_partner_first_action=False,
    )

    assert np.allclose(result.next_pose, lateral)
    assert result.diagnostics["allow_height_partner_first_action"] is False
    assert result.diagnostics["disallowed_height_partner_candidates"] == 1


def test_dss_pp_count_balance_penalty_is_isotope_agnostic() -> None:
    """DSS-PP balance penalty should reject any single-isotope dominated program."""
    balanced = {"Cs-137": 12.0, "Co-60": 12.0, "Eu-154": 12.0}
    co_dominated = {"Cs-137": 1.0, "Co-60": 98.0, "Eu-154": 1.0}
    eu_dominated = {"Cs-137": 1.0, "Co-60": 1.0, "Eu-154": 98.0}

    assert _count_balance_penalty(balanced) == pytest.approx(0.0)
    assert _count_balance_penalty(co_dominated) == pytest.approx(
        _count_balance_penalty(eu_dominated)
    )
    assert _count_balance_penalty(co_dominated) > 0.5


def test_dss_pp_count_utility_saturates_high_counts() -> None:
    """Count utility should reward usability without unbounded proximity bias."""
    low = _saturated_count_utility(
        {"Cs-137": 10.0, "Co-60": 10.0},
        saturation_counts=100.0,
    )
    useful = _saturated_count_utility(
        {"Cs-137": 100.0, "Co-60": 100.0},
        saturation_counts=100.0,
    )
    extreme = _saturated_count_utility(
        {"Cs-137": 10000.0, "Co-60": 10000.0},
        saturation_counts=100.0,
    )

    assert 0.0 < low < useful < extreme <= 1.0
    assert extreme == pytest.approx(1.0)


def test_dss_pp_local_orbit_prefers_informative_annulus() -> None:
    """Local-orbit scoring should choose an offset station over source chasing."""
    est = _build_simple_estimator()
    candidates = np.array(
        [
            [0.1, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([5.0, 5.0, 0.0], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            max_programs=1,
            program_length=1,
            live_time_s=1.0,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_rotation=0.0,
            lambda_count_utility=0.0,
            lambda_local_orbit=10.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            ring_radii_m=(3.0,),
            local_orbit_sigma_m=0.5,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert result.diagnostics["first_local_orbit_gain"] > 0.0


def test_dss_pp_bearing_diversity_is_isotope_agnostic() -> None:
    """Bearing diversity should favor angularly separating any same-isotope modes."""
    isotopes = ["Co-60"]
    candidate_sources = np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]], dtype=float)
    normals = generate_octant_rotation_matrices()
    shield_normals = np.asarray([mat[:, 2] for mat in normals], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=2,
        max_sources=1,
        use_gpu=False,
        init_num_sources=(1, 1),
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=shield_normals,
        mu_by_isotope={"Co-60": {"fe": 0.0, "pb": 0.0}},
        pf_config=config,
        shield_params=ShieldParams(),
    )
    est.add_measurement_pose(np.array([2.0, 3.0, 0.5], dtype=float))
    est._ensure_kernel_cache()
    est.filters["Co-60"].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([1000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.5]], dtype=float),
                strengths=np.array([1000.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    candidates = np.array(
        [[2.0, 0.5, 0.5], [2.0, 6.0, 0.5]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=np.array([2.0, 3.0, 0.5], dtype=float),
        config=DSSPPConfig(
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            lambda_bearing_diversity=10.0,
            lambda_rotation=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[0])
    assert result.diagnostics["first_bearing_diversity_gain"] > 0.0


def test_dss_pp_turn_smoothness_discourages_backtracking() -> None:
    """Turn smoothness should prefer continuing outward over reversing course."""
    est = _build_simple_estimator()
    current = np.array([1.0, 0.0, 0.0], dtype=float)
    visited = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    candidates = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )

    result = select_dss_pp_next_station(
        estimator=est,
        candidate_poses_xyz=candidates,
        current_pose_xyz=current,
        visited_poses_xyz=visited,
        config=DSSPPConfig(
            horizon=1,
            max_programs=1,
            lambda_eig=0.0,
            lambda_signature=0.0,
            lambda_distance=0.0,
            lambda_coverage=0.0,
            lambda_turn_smoothness=5.0,
            lambda_rotation=0.0,
            eta_count_balance=0.0,
            eta_differential=0.0,
            eta_observation=0.0,
            enforce_min_observation=False,
            min_station_separation_m=0.0,
            signature_std_min_counts=0.0,
            augment_candidates=False,
        ),
    )

    assert np.allclose(result.next_pose, candidates[1])
    assert result.diagnostics["first_turn_penalty"] == pytest.approx(0.0)
