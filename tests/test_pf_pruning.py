"""Tests for spurious-source pruning via the best-case measurement test (Chapter 3)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator, MeasurementRecord
from pf.particle_filter import IsotopeParticle
from pf.state import IsotopeState
from measurement.continuous_kernels import ContinuousKernel


def test_prune_spurious_source_best_case_measurement() -> None:
    """Verify that the best-case measurement test prunes a weak spurious source."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float)
    normals = np.array([[-1.0, 0.0, 0.0]], dtype=float)  # avoid attenuation toward source 0
    mu = {"Cs-137": {"fe": 0.5, "pb": 0.7}}
    config = RotatingShieldPFConfig(num_particles=2, max_sources=2)
    estimator = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=candidate_sources,
        shield_normals=normals,
        mu_by_isotope=mu,
        pf_config=config,
        shield_params=ShieldParams(),
    )
    estimator.add_measurement_pose(np.array([1.0, 1.0, 1.0]))
    estimator._ensure_kernel_cache()

    true_strength = 200.0
    filt = estimator.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=candidate_sources.copy(),
                strengths=np.array([true_strength, 5.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.6),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=candidate_sources.copy(),
                strengths=np.array([true_strength, 5.0], dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.4),
        ),
    ]
    kernel = ContinuousKernel(mu_by_isotope=mu, shield_params=ShieldParams())
    observed_z = kernel.expected_counts_pair(
        isotope="Cs-137",
        detector_pos=estimator.poses[0],
        sources=np.array([candidate_sources[0]]),
        strengths=np.array([true_strength]),
        fe_index=7,
        pb_index=7,
        live_time_s=1.0,
        background=0.0,
    )
    estimator.measurements = [
        MeasurementRecord(
            z_k={"Cs-137": float(observed_z)},
            pose_idx=0,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=7,
            pb_index=7,
        )
    ]

    keep_mask = estimator.prune_spurious_sources(tau_mix=0.8)
    assert keep_mask["Cs-137"][0]
    assert not keep_mask["Cs-137"][1]
    for p in estimator.filters["Cs-137"].continuous_particles:
        assert p.state.num_sources == 1
