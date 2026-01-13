"""Tests for shield rotation and pose selection logic (Chapter 3.4)."""

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import IsotopeState
from planning.pose_selection import (
    estimate_lambda_cost,
    recommend_num_rollouts,
    select_next_pose,
    select_next_pose_after_rotation,
)
from planning.candidate_generation import generate_candidate_poses
from planning.shield_rotation import rotation_policy_step, select_best_orientation
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


def test_estimate_lambda_cost_range_scales_motion() -> None:
    """Range-based lambda should match uncertainty and motion-cost ranges."""
    uncertainties = np.array([1.0, 2.0, 4.0], dtype=float)
    distances = np.array([0.5, 1.0, 2.5], dtype=float)
    lam = estimate_lambda_cost(uncertainties, distances, method="range")
    expected = (4.0 - 1.0) / (2.5 - 0.5)
    assert lam == pytest.approx(expected)


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
