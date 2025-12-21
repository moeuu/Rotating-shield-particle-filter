"""Basic tests for continuous measurement model and PF scaffold."""

import numpy as np

from measurement.continuous_kernels import expected_counts_single_isotope, ContinuousKernel
from pf.particle_filter import IsotopeParticleFilter, PFConfig, IsotopeParticle
from pf.state import IsotopeState


def test_geometric_scaling_inverse_square() -> None:
    """Expected counts should follow inverse-square scaling without shielding."""
    src = np.array([[0.0, 0.0, 0.0]])
    strength = np.array([10.0])
    d1 = 1.0
    d2 = 2.0
    lam1 = expected_counts_single_isotope(
        detector_position=np.array([d1, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam2 = expected_counts_single_isotope(
        detector_position=np.array([d2, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    assert np.allclose(lam1 / lam2, (d2**2) / (d1**2), rtol=1e-6)


def test_shield_attenuation_factor_both_materials() -> None:
    """When both Fe and Pb block, expected counts should follow exp(-mu*L)."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strength = np.array([5.0])
    lam_free = expected_counts_single_isotope(
        detector_position=det,
        RFe=np.array([1.0, 1.0, 1.0]),
        RPb=np.array([1.0, 1.0, 1.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam_blocked = expected_counts_single_isotope(
        detector_position=det,
        RFe=np.array([-1.0, -1.0, -1.0]),
        RPb=np.array([-1.0, -1.0, -1.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    from measurement.kernels import ShieldParams

    shield_params = ShieldParams()
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(lam_blocked, expected_ratio * lam_free, rtol=1e-6)


def test_poisson_weight_update_prefers_higher_lambda() -> None:
    """Weight update should favor particle with higher expected Λ for given z."""
    cfg = PFConfig(num_particles=2)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    # Override continuous particles with deterministic states
    p_hi = IsotopeState(num_sources=1, positions=np.array([[1.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0)
    p_lo = IsotopeState(num_sources=1, positions=np.array([[5.0, 0.0, 0.0]]), strengths=np.array([1.0]), background=0.0)
    pf.continuous_particles = [
        IsotopeParticle(state=p_hi, log_weight=np.log(0.5)),
        IsotopeParticle(state=p_lo, log_weight=np.log(0.5)),
    ]
    pf.kernel = dummy_kernel
    z_obs = 1.0
    pf.update_continuous(z_obs=z_obs, pose_idx=0, orient_idx=0, live_time_s=1.0)
    weights = pf.continuous_weights
    assert weights[0] > weights[1]


def test_resampling_increases_neff() -> None:
    """Highly skewed weights should be flattened after resampling."""
    cfg = PFConfig(num_particles=3)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    pf.continuous_particles = [
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.99)),
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.005)),
        IsotopeParticle(state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0), log_weight=np.log(0.005)),
    ]
    before = 1.0 / np.sum(pf.continuous_weights**2)
    pf._maybe_resample_continuous()
    after = 1.0 / np.sum(pf.continuous_weights**2)
    assert after > before
    assert np.allclose(pf.continuous_weights, np.ones(3) / 3, rtol=1e-3)
