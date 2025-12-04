"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticleFilter, PFConfig
from pf.state import ParticleState


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
    filt.states = [
        ParticleState(source_indices=np.array([], dtype=np.int32), strengths=np.array([], dtype=float), background=0.1)
        for _ in range(filt.N)
    ]
    filt.log_weights = np.log(np.ones(filt.N) / filt.N)
    filt.predict()
    assert all(st.source_indices.size > 0 for st in filt.states)


def test_death_removes_weak_sources() -> None:
    """Sources below the minimum strength threshold should be removed."""
    np.random.seed(1)
    filt = _build_filter(p_birth=0.0, min_strength=0.5, max_sources=2, num_particles=2)
    filt.states = [
        ParticleState(
            source_indices=np.array([0, 1], dtype=np.int32),
            strengths=np.array([0.1, 0.2], dtype=float),
            background=0.1,
        )
        for _ in range(filt.N)
    ]
    filt.log_weights = np.log(np.ones(filt.N) / filt.N)
    filt.predict()
    assert all(st.source_indices.size == 0 for st in filt.states)


def test_estimate_respects_max_sources() -> None:
    """Estimator output should cap the number of sources at max_sources even with births."""
    np.random.seed(2)
    filt = _build_filter(p_birth=1.0, min_strength=0.01, max_sources=1, num_particles=5)
    filt.predict()  # allow births to occur
    positions, strengths = filt.estimate()
    assert positions.shape[0] <= 1
    assert strengths.shape[0] == positions.shape[0]
