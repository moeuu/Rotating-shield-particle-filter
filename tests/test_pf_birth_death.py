"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticleFilter, IsotopeParticle
from pf.state import IsotopeState


def _build_filter(p_birth: float, min_strength: float, max_sources: int, num_particles: int = 10) -> IsotopeParticleFilter:
    """Utility to create an isotope PF with configurable birth/death parameters."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=num_particles,
        max_sources=max_sources,
        resample_threshold=0.5,
        strength_sigma=0.0,
        background_sigma=0.0,
        min_strength=min_strength,
        p_birth=p_birth,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([0.5, 0.0, 0.0]))
    estimator._ensure_kernel_cache()
    return estimator.filters["Cs-137"]


def test_birth_adds_source_when_particle_empty() -> None:
    """Birth move should inject a new source when a particle has none."""
    np.random.seed(0)
    filt = _build_filter(p_birth=1.0, min_strength=0.01, max_sources=2, num_particles=3)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(num_sources=0, positions=np.zeros((0, 3)), strengths=np.zeros(0), background=0.1),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    filt.regularize_continuous(p_birth=1.0, p_kill=0.0, intensity_threshold=0.01)
    assert all(p.state.num_sources > 0 for p in filt.continuous_particles)


def test_death_removes_weak_sources() -> None:
    """Sources below the minimum strength threshold should be removed."""
    np.random.seed(1)
    filt = _build_filter(p_birth=0.0, min_strength=0.5, max_sources=2, num_particles=2)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                strengths=np.array([0.1, 0.2], dtype=float),
                background=0.1,
            ),
            log_weight=float(np.log(0.5)),
        )
        for _ in range(filt.N)
    ]
    filt.regularize_continuous(
        sigma_pos=0.0,
        sigma_int=0.0,
        p_birth=0.0,
        p_kill=1.0,
        intensity_threshold=0.5,
    )
    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)


def test_estimate_returns_all_sources() -> None:
    """Estimator output should return all estimated sources without capping."""
    np.random.seed(2)
    filt = _build_filter(p_birth=1.0, min_strength=0.01, max_sources=1, num_particles=5)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                strengths=np.array([5.0, 2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                strengths=np.array([5.0, 2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.5),
        ),
    ]
    positions, strengths = filt.estimate()
    assert positions.shape[0] == 2
    assert strengths.shape[0] == positions.shape[0]
