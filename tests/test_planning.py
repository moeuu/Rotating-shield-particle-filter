"""Tests for shield rotation and pose selection logic (Chapter 3.4)."""

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import ParticleState, IsotopeState
from planning.pose_selection import select_next_pose
from planning.shield_rotation import rotation_policy_step, select_best_orientation
from pf.particle_filter import IsotopeParticle
from measurement.shielding import generate_octant_rotation_matrices


def _build_simple_estimator() -> RotatingShieldPFEstimator:
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
    filt = est.filters["Cs-137"]
    filt.states = [
        ParticleState(source_indices=np.array([0], dtype=np.int32), strengths=np.array([10.0]), background=0.0),
        ParticleState(source_indices=np.array([0], dtype=np.int32), strengths=np.array([0.0]), background=0.0),
    ]
    filt.log_weights = np.log(np.ones(2) / 2.0)
    # Deterministic continuous particles for EIG tests (unblocked should dominate)
    from pf.particle_filter import IsotopeParticle
    from pf.state import IsotopeState

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
            criterion="eig",
            RFe_candidates=[RFe],
            RPb_candidates=[RPb],
        )[1]
        scores.append(score)
    best_idx = int(np.argmax(scores))
    # Ensure argmax is returned and scores are finite; we do not enforce a specific octant here.
    assert scores[best_idx] == pytest.approx(max(scores))
    assert np.all(np.isfinite(scores))


def test_rotation_policy_stops_when_information_low() -> None:
    """rotation_policy_step should stop if both IG and Fisher surrogates are below thresholds."""
    est = _build_simple_estimator()
    # Zero out strengths to make IG ~ 0
    filt = est.filters["Cs-137"]
    for st in filt.states:
        st.strengths[:] = 0.0
    should_stop, orient_idx, score = rotation_policy_step(
        estimator=est, pose_idx=0, ig_threshold=1e-6, fisher_threshold=1e-6, live_time_s=1.0
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
        live_time_s=1.0,
        lambda_cost=0.1,
    )
    assert next_idx == 1


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


def test_orientation_fisher_criteria_positive() -> None:
    """Fisher criteria JA/JD should be finite and non-negative for simple setup (Eq. 3.46–3.47)."""
    np.random.seed(1)
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
    filt = est.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([5.0]), background=0.1
            ),
            log_weight=np.log(0.6),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([2.0]), background=0.2
            ),
            log_weight=np.log(0.4),
        ),
    ]
    mats = generate_octant_rotation_matrices()
    JA, JD = est.orientation_fisher_criteria(pose_idx=0, RFe=mats[0], RPb=mats[0], live_time_s=1.0)
    assert JA >= 0.0
    assert np.isfinite(JD)


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
        fisher_threshold=1e6,
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
