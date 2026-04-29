"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticleFilter, IsotopeParticle, MeasurementData
from pf.state import IsotopeState


def _build_filter(
    p_birth: float,
    min_strength: float,
    max_sources: int,
    num_particles: int = 10,
    **kwargs: float,
) -> IsotopeParticleFilter:
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
        **kwargs,
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
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=None,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert all(p.state.num_sources > 0 for p in filt.continuous_particles)


def test_birth_excludes_candidates_near_detector_poses() -> None:
    """Birth proposals should not place sources on measured detector poses."""
    np.random.seed(0)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_detector_min_sep_m=1.0,
        birth_num_local_jitter=0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(num_sources=0, positions=np.zeros((0, 3)), strengths=np.zeros(0), background=0.1),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=None,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    for particle in filt.continuous_particles:
        assert particle.state.num_sources == 1
        assert np.allclose(particle.state.positions[0], np.array([2.0, 0.0, 0.0]))


def test_death_removes_weak_sources() -> None:
    """Sources below the minimum strength threshold should be removed."""
    np.random.seed(1)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.5,
        max_sources=2,
        num_particles=2,
        death_low_q_streak=1,
        death_delta_ll_threshold=0.0,
        support_ema_alpha=1.0,
        p_kill=1.0,
    )
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
    support_data = MeasurementData(
        z_k=np.array([0.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_birth_death(support_data=support_data, birth_data=None, candidate_positions=None)
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


def test_weak_source_survives_with_support() -> None:
    """Weak sources should survive when delta-LL evidence is positive."""
    np.random.seed(2)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.1,
        max_sources=2,
        num_particles=1,
        death_low_q_streak=2,
        death_delta_ll_threshold=0.0,
        support_ema_alpha=1.0,
        p_kill=1.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
                strengths=np.array([5.0, 0.5], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(1.0)),
        )
    ]
    kernel = filt.continuous_kernel
    det_pos = np.array([0.5, 0.0, 0.0], dtype=float)
    live_time = 1.0
    lam1 = (
        kernel.kernel_value_pair("Cs-137", det_pos, filt.continuous_particles[0].state.positions[0], 7, 7)
        * filt.continuous_particles[0].state.strengths[0]
        * live_time
    )
    lam2 = (
        kernel.kernel_value_pair("Cs-137", det_pos, filt.continuous_particles[0].state.positions[1], 7, 7)
        * filt.continuous_particles[0].state.strengths[1]
        * live_time
    )
    z_k = np.array([lam1 + lam2], dtype=float)
    support_data = MeasurementData(
        z_k=z_k,
        observation_variances=np.maximum(z_k, 1.0),
        detector_positions=np.array([det_pos], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([live_time], dtype=float),
    )
    for _ in range(3):
        filt.apply_birth_death(support_data=support_data, birth_data=None, candidate_positions=None)
    assert filt.continuous_particles[0].state.num_sources == 2


def test_birth_disabled_skips_moves() -> None:
    """Birth mode disabled should skip birth/kill/split/merge moves."""
    np.random.seed(3)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=2,
        birth_enable=False,
        p_kill=1.0,
        split_prob=1.0,
        merge_prob=1.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for _ in range(filt.N)
    ]
    support_data = MeasurementData(
        z_k=np.array([0.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=support_data,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert all(p.state.num_sources == 1 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert filt.last_kill_count == 0


def test_birth_enabled_adds_sources() -> None:
    """Birth mode enabled should allow adding sources."""
    np.random.seed(4)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]]),
                strengths=np.array([0.1], dtype=float),
                background=0.1,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=None,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )
    assert any(p.state.num_sources > 1 for p in filt.continuous_particles)
    assert filt.last_birth_count > 0
