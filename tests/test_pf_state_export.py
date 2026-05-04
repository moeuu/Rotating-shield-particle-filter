"""Tests for PF state export helper."""

import numpy as np

from pf.parallel import ParallelIsotopePF
from pf.particle_filter import PFConfig, IsotopeParticle
from pf.state import IsotopeState


def test_export_state_contains_particles_and_estimate() -> None:
    """export_state should return particle clouds, weights, and current estimate."""
    config = PFConfig(num_particles=2)
    pf = ParallelIsotopePF(isotope_names=["Cs-137"], config=config)
    # inject simple continuous particles
    filt = pf.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(num_sources=1, positions=np.array([[0.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.1),
            log_weight=np.log(0.6),
        ),
        IsotopeParticle(
            state=IsotopeState(num_sources=1, positions=np.array([[1.0, 0.0, 0.0]]), strengths=np.array([2.0]), background=0.2),
            log_weight=np.log(0.4),
        ),
    ]
    snap = pf.export_state()["Cs-137"]
    assert len(snap.particle_positions) == 2
    assert snap.weights.shape[0] == 2
    # weights normalized
    assert np.isclose(np.sum(snap.weights), 1.0)
    assert snap.estimate.positions.shape[1] == 3


def test_parallel_estimate_all_supports_cpu_config() -> None:
    """estimate_all should not require CUDA when PFConfig.use_gpu is false."""
    config = PFConfig(num_particles=2, use_gpu=False)
    pf = ParallelIsotopePF(isotope_names=["Cs-137"], config=config)
    filt = pf.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([2.0]),
                background=0.1,
            ),
            log_weight=np.log(0.25),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[4.0, 0.0, 0.0]]),
                strengths=np.array([6.0]),
                background=0.3,
            ),
            log_weight=np.log(0.75),
        ),
    ]

    estimate = pf.estimate_all()["Cs-137"]

    assert np.allclose(estimate.positions[0], [3.0, 0.0, 0.0])
    assert np.isclose(estimate.strengths[0], 5.0)
    assert np.isclose(estimate.background, 0.25)
    assert estimate.covariances is not None
    assert estimate.covariances.shape == (1, 4, 4)
