"""Convergence criteria tests for RotatingShieldPFEstimator (Sec. 3.5–3.6)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import ParticleState


def _build_stable_estimator(strength: float = 10.0) -> RotatingShieldPFEstimator:
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
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
    est.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    est._ensure_kernel_cache()
    filt = est.filters["Cs-137"]
    filt.states = [
        ParticleState(source_indices=np.array([0], dtype=np.int32), strengths=np.array([strength]), background=0.0)
        for _ in range(filt.N)
    ]
    filt.log_weights = np.log(np.ones(filt.N) / filt.N)
    # Remove continuous particles to keep uncertainty zeroed for this grid-based convergence check
    filt.continuous_particles = []
    # populate history with identical estimates to simulate stabilization
    est.history_estimates = [est.estimates(), est.estimates()]
    return est


def test_should_stop_shield_rotation_when_stable() -> None:
    """Stable posterior with zero IG and low uncertainty should trigger stop."""
    est = _build_stable_estimator()
    assert est.should_stop_shield_rotation(
        pose_idx=0, ig_threshold=1e-6, change_tol=1e-6, uncertainty_tol=1e-6, live_time_s=1.0
    )


def test_should_not_stop_when_uncertain() -> None:
    """High variance across particles keeps exploration active."""
    est = _build_stable_estimator()
    filt = est.filters["Cs-137"]
    # Inject variance in strengths to raise U
    filt.states[0].strengths = np.array([1.0])
    filt.states[1].strengths = np.array([10.0])
    est.history_estimates = [est.estimates(), est.estimates()]
    assert not est.should_stop_exploration(
        ig_threshold=1e-6, change_tol=1e-6, uncertainty_tol=1e-3, live_time_s=1.0
    )
