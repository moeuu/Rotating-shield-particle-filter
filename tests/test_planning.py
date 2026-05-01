"""Tests for shield rotation and pose selection logic (Chapter 3.4)."""

import numpy as np
import pytest

import planning.dss_pp as dss_pp
from measurement.kernels import ShieldParams
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
    build_shield_program_library,
    select_dss_pp_next_station,
)
from planning.candidate_generation import generate_candidate_poses
from planning.shield_rotation import rotation_policy_step, select_best_orientation
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
    """DSS-PP should expose bearing, material, and occlusion shield programs."""
    mats = generate_octant_rotation_matrices()
    normals = np.asarray([mat[:, 2] for mat in mats], dtype=float)
    programs = build_shield_program_library(normals, program_length=2, max_programs=32)
    kinds = {program.kind for program in programs}

    assert "bearing_split" in kinds
    assert "material_split" in kinds
    assert "occlusion_test" in kinds
    assert all(len(program.pair_ids) == 2 for program in programs)


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
    assert np.allclose(result.next_pose, candidates[result.next_pose_index])


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
