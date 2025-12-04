"""Tests for spurious-source pruning via the best-case measurement test (Chapter 3)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import ParticleState


def test_prune_spurious_source_best_case_measurement() -> None:
    """Verify that the best-case measurement test prunes a weak spurious source."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[-1.0, 0.0, 0.0]], dtype=float)  # avoid attenuation toward source 0
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(num_particles=2, max_sources=2)
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([1.0, 0.0, 0.0]))
    estimator._ensure_kernel_cache()

    true_strength = 200.0
    observed_z = estimator.kernel_cache.expected_counts(
        "Cs-137",
        pose_idx=0,
        orient_idx=0,
        source_strengths=np.array([true_strength, 0.0]),
        background=0.0,
        live_time_s=1.0,
    )
    estimator.update(z_k={"Cs-137": observed_z}, pose_idx=0, orient_idx=0, live_time_s=1.0)

    states = [
        ParticleState(
            source_indices=np.array([0, 1], dtype=np.int32),
            strengths=np.array([true_strength, 5.0], dtype=float),
            background=0.0,
        ),
        ParticleState(
            source_indices=np.array([0, 1], dtype=np.int32),
            strengths=np.array([true_strength, 5.0], dtype=float),
            background=0.0,
        ),
    ]
    filt = estimator.filters["Cs-137"]
    filt.states = states
    filt.log_weights = np.log(np.array([0.6, 0.4]))

    keep_mask = estimator.prune_spurious_sources(tau_mix=0.8)
    assert keep_mask["Cs-137"][0]
    assert not keep_mask["Cs-137"][1]
    for st in estimator.filters["Cs-137"].states:
        assert 0 in st.source_indices
        assert 1 not in st.source_indices
