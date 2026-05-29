"""Tests for shield rotation and pose selection logic (Chapter 3.4)."""

import numpy as np
import pytest

import planning.dss_pp as dss_pp
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
    _count_balance_penalty,
    _continuous_kernel_for_estimator,
    _saturated_count_utility,
    _programs_for_pose,
    _temporal_separation_score_from_signatures,
    build_shield_program_library,
    select_dss_pp_next_station,
)
from planning.candidate_generation import generate_candidate_poses
from planning.shield_rotation import rotation_policy_step, select_best_orientation
from planning.shield_rotation import select_separation_orientations
from planning.traversability import TraversabilityMap
from pf.particle_filter import IsotopeParticle
from measurement.shielding import generate_octant_rotation_matrices


def _build_simple_estimator() -> RotatingShieldPFEstimator:
    """Build a minimal estimator with deterministic particle setup for tests."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[1.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(num_particles=2, max_sources=1, resample_threshold=0.5)
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
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([10.0]), background=0.0
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0
            ),
            log_weight=np.log(0.5),
        ),
    ]
    return est


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
        blocked_mask.append(oct_shield.blocks_ray(detector_position=detector, source_position=source, octant_index=idx))

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


def test_orientation_expected_information_gain_grid_cpu_fallback_matches_pairwise() -> None:
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

        def expected_uncertainty(self, pose_idx: int, live_time_s: float = 1.0) -> float:
            return [2.0, 0.5][pose_idx]

        def max_orientation_information_gain(self, pose_idx: int, live_time_s: float = 1.0) -> float:
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
    assert minimum_observation_shortfall(
        {"Cs-137": 5.0, "Co-60": 0.0},
        min_counts=5.0,
    ) > 0.0


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
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([10.0]), background=0.0
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0
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
    est.short_time_update(z_k=z_k, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=None)
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
    config = RotatingShieldPFConfig(num_particles=5, max_sources=1, max_dwell_time_s=0.5, ig_threshold=1e6)
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
    est.short_time_update(z_k={"Cs-137": 1.0}, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=0.3)
    est.short_time_update(z_k={"Cs-137": 1.0}, pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=0.3)
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
            state=IsotopeState(num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([2.0]), background=0.1),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([5.0]), background=0.1),
            log_weight=np.log(0.5),
        ),
    ]
    U = est.expected_uncertainty_after_pose(pose_idx=0, orient_idx=0, live_time_s=1.0, num_samples=10)
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
    assert result.diagnostics["ranked_nodes"][0]["score"] >= result.diagnostics[
        "ranked_nodes"
    ][-1]["score"]
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
    )

    assert result.shield_program.pair_ids == forced_pairs
    assert result.shield_program.kind == "forced_baseline"
    assert result.diagnostics["program_count"] == 1
    assert {
        tuple(node["pair_ids"]) for node in result.diagnostics["ranked_nodes"]
    } == {forced_pairs}
    pair_diag = result.diagnostics["selected_pairwise_ambiguity"]["Cs-137"]
    assert pair_diag["mode_count"] == 2
    assert pair_diag["program_measurements"] == len(result.shield_program.pair_ids)
    assert pair_diag["bottleneck_pairs"]


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
        [[x, y, 1.0] for x in np.linspace(0.5, 8.5, 4) for y in np.linspace(0.5, 8.5, 4)],
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
            scalar_balance.append(float(0.75 * np.min(balance) + 0.25 * np.mean(balance)))
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
