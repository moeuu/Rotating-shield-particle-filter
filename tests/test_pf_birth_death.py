"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np

from measurement.kernels import ShieldParams
from pf.likelihood import expected_counts_per_source
from pf.estimator import MeasurementRecord, RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticleFilter, IsotopeParticle, MeasurementData
from pf.state import IsotopeState


def _build_filter(
    p_birth: float,
    min_strength: float,
    max_sources: int,
    num_particles: int = 10,
    **kwargs: object,
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
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_residual_min_support=1,
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
    assert all(p.state.num_sources > 0 for p in filt.continuous_particles)


def test_reported_strength_refit_uses_all_measurements() -> None:
    """Reported strengths should be refit from counts after position clustering."""
    np.random.seed(10)
    isotope = "Cs-137"
    sources = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0]], dtype=float)
    detector_positions = [
        np.array([1.0, 3.0, 0.0], dtype=float),
        np.array([4.0, 0.0, 0.0], dtype=float),
        np.array([2.0, -2.0, 0.0], dtype=float),
    ]
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        birth_enable=True,
        use_clustered_output=True,
        cluster_min_samples=1,
        report_strength_refit=True,
        report_strength_refit_iters=128,
        init_num_sources=(1, 1),
        use_gpu=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=sources,
        shield_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=config,
        shield_params=ShieldParams(thickness_pb_cm=0.0, thickness_fe_cm=0.0),
    )
    for pose in detector_positions:
        estimator.add_measurement_pose(pose)
    estimator._ensure_kernel_cache()
    true_strengths = np.array([120.0, 45.0], dtype=float)
    data_design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=sources,
        strengths=np.ones(2, dtype=float),
        live_times=np.ones(len(detector_positions), dtype=float),
        fe_indices=np.zeros(len(detector_positions), dtype=int),
        pb_indices=np.zeros(len(detector_positions), dtype=int),
    )
    counts = data_design @ true_strengths
    estimator.measurements = [
        MeasurementRecord(
            z_k={isotope: float(count)},
            pose_idx=idx,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: max(float(count), 1.0)},
        )
        for idx, count in enumerate(counts)
    ]
    filt = estimator.filters[isotope]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=sources.copy(),
                strengths=np.array([10.0, 10.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]

    positions, strengths = estimator.estimates()[isotope]

    order = np.argsort(positions[:, 0])
    assert np.allclose(positions[order], sources, atol=1.0e-6)
    assert np.allclose(strengths[order], true_strengths, rtol=1.0e-3, atol=1.0e-3)


def test_grid_initialization_repeats_strength_samples_per_cell() -> None:
    """Deterministic grid initialization should support repeated strength samples."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_grid_repeats=3,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
    )
    positions = np.vstack([p.state.positions[0] for p in filt.continuous_particles])

    assert len(filt.continuous_particles) == 6
    assert np.unique(positions, axis=0).shape[0] == 2


def test_grid_initialization_respects_source_count_prior() -> None:
    """Grid initialization should not force one source when count is unknown."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        init_num_sources=(0, 3),
        init_grid_spacing_m=1.0,
        init_grid_repeats=4,
        position_min=(0.0, 0.0, 0.0),
        position_max=(2.0, 1.0, 1.0),
    )
    counts = [particle.state.num_sources for particle in filt.continuous_particles]

    assert len(filt.continuous_particles) == 8
    assert set(counts) == {0, 1, 2, 3}


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
        birth_residual_min_support=1,
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


def test_source_detector_exclusion_removes_detector_collapsed_source() -> None:
    """Structural updates should reject sources inside measured detector poses."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        source_detector_exclusion_m=0.25,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.51, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([10.0, 10.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    support_data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([10.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    filt.apply_birth_death(support_data=support_data, birth_data=support_data, candidate_positions=None)

    state = filt.continuous_particles[0].state
    assert state.num_sources == 1
    assert np.allclose(state.positions[0], np.array([2.0, 0.0, 0.0]))


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
        birth_refit_residual_gate=False,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
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


def test_birth_max_per_update_caps_structural_growth() -> None:
    """Birth proposals should be capped per structural update when configured."""
    np.random.seed(7)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=5,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_max_per_update=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
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

    assert filt.last_birth_count == 2
    assert sum(p.state.num_sources for p in filt.continuous_particles) == 2


def test_birth_gate_blocks_statistically_explained_residual() -> None:
    """Birth should not add sources when residuals are explained by count variance."""
    np.random.seed(5)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_residual_gate_p_value=0.05,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    birth_data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([1.0e6], dtype=float),
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
    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert not filt.last_birth_residual_gate_passed


def test_birth_gate_requires_multiple_supported_measurements_by_default() -> None:
    """Default birth evidence should require residual support in multiple observations."""
    np.random.seed(6)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
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
    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_support == 1


def test_birth_gate_requires_residual_support_from_distinct_poses() -> None:
    """Birth evidence should be supported by residuals at multiple stations."""
    np.random.seed(8)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    same_pose_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_birth_death(
        support_data=None,
        birth_data=same_pose_data,
        candidate_positions=filt.kernel.sources,
    )

    assert all(p.state.num_sources == 0 for p in filt.continuous_particles)
    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_support == 0
    assert filt.last_birth_residual_distinct_poses == 1

    distinct_pose_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=None,
        birth_data=distinct_pose_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_count > 0


def test_birth_gate_counts_distinct_shield_views() -> None:
    """Residual birth should treat shield postures as independent views."""
    np.random.seed(9)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    shield_view_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_birth_death(
        support_data=None,
        birth_data=shield_view_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_count > 0


def test_birth_gate_can_require_distinct_robot_stations() -> None:
    """Residual birth can require support from more than one robot station."""
    np.random.seed(10)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=2,
        birth_min_distinct_poses=2,
        birth_min_distinct_stations=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    one_station_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )

    filt.apply_birth_death(
        support_data=None,
        birth_data=one_station_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 0
    assert filt.last_birth_residual_distinct_poses == 2
    assert filt.last_birth_residual_distinct_stations == 1

    two_station_data = MeasurementData(
        z_k=np.array([100.0, 100.0], dtype=float),
        observation_variances=np.array([100.0, 100.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 1.0], dtype=float),
    )
    filt.apply_birth_death(
        support_data=None,
        birth_data=two_station_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_residual_distinct_stations == 2
    assert filt.last_birth_count > 0


def test_structural_refresh_uses_birth_observation_block() -> None:
    """Moved particles should be judged by the station birth block when present."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        birth_enable=True,
        birth_min_sep_m=0.0,
        birth_detector_min_sep_m=0.0,
        birth_num_local_jitter=0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3)),
                strengths=np.zeros(0),
                background=0.0,
            ),
            log_weight=float(np.log(1.0 / filt.N)),
        )
        for _ in range(filt.N)
    ]
    support_data = MeasurementData(
        z_k=np.array([1.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    birth_data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[1.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )
    seen: list[MeasurementData] = []

    def _record_refresh(data: MeasurementData | None) -> None:
        """Record which data block was used for post-move weight refresh."""
        if data is not None:
            seen.append(data)

    filt.refresh_weights_from_measurements = _record_refresh  # type: ignore[method-assign]

    filt.apply_birth_death(
        support_data=support_data,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )

    assert seen and seen[-1] is birth_data


def test_conditional_strength_refit_matches_counts_for_fixed_position() -> None:
    """Strength refit should recover source rate when position is fixed."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        conditional_strength_refit=True,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=0.0),
    ]
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    true_strength = np.array([100.0], dtype=float)
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=state.positions,
        strengths=true_strength,
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.maximum(np.sum(expected, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=5)

    assert np.isclose(filt.continuous_particles[0].state.strengths[0], 100.0, rtol=1e-3)


def test_conditional_strength_refit_ignores_censored_low_signal_data() -> None:
    """Low-signal upper-limit observations should not collapse source rates."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=1,
        num_particles=1,
        conditional_strength_refit=True,
        conditional_strength_refit_min_count=5.0,
        conditional_strength_refit_min_snr=1.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    data = MeasurementData(
        z_k=np.array([0.0, 3.0], dtype=float),
        observation_variances=np.array([10000.0, 10000.0], dtype=float),
        detector_positions=np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=5)

    assert filt.continuous_particles[0].state.strengths[0] == 100.0


def test_conditional_strength_refit_prior_limits_overfit() -> None:
    """Strength MAP refit should not ignore the particle strength prior."""
    unregularized = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        conditional_strength_refit=True,
        conditional_strength_refit_prior_weight=0.0,
    )
    regularized = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        conditional_strength_refit=True,
        conditional_strength_refit_prior_weight=10.0,
        conditional_strength_refit_prior_rel_sigma=0.5,
    )
    detector_positions = np.array([[1.0, 0.0, 0.0]], dtype=float)
    data = MeasurementData(
        z_k=np.array([1000.0], dtype=float),
        observation_variances=np.array([1000.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        live_times=np.ones(1, dtype=float),
    )
    for filt in (unregularized, regularized):
        state = IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        )
        filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
        filt.refit_strengths_for_particles(data, iters=5)

    q_unregularized = unregularized.continuous_particles[0].state.strengths[0]
    q_regularized = regularized.continuous_particles[0].state.strengths[0]

    assert q_unregularized > 900.0
    assert q_regularized < q_unregularized
    assert q_regularized > 100.0


def test_conditional_strength_refit_does_not_anchor_numeric_floor() -> None:
    """Numerical strength floors should not become physical source priors."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=1,
        num_particles=1,
        conditional_strength_refit=True,
        conditional_strength_refit_prior_weight=100.0,
        conditional_strength_refit_prior_rel_sigma=0.5,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([5.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    detector_positions = np.array([[1.0, 0.0, 0.0]], dtype=float)
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=state.positions,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(1, dtype=float),
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.maximum(np.sum(expected, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        live_times=np.ones(1, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=5)

    assert filt.continuous_particles[0].state.strengths[0] > 90.0


def test_conditional_strength_refit_batches_multi_source_particles() -> None:
    """Batched refit should recover fixed-position strengths for multi-source particles."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=2,
        conditional_strength_refit=True,
    )
    states = [
        IsotopeState(
            num_sources=2,
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([10.0, 10.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([20.0, 20.0], dtype=float),
            background=0.0,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=np.log(0.5))
        for state in states
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([120.0, 240.0], dtype=float)
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=states[0].positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.maximum(np.sum(expected, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=8)

    for particle in filt.continuous_particles:
        assert np.allclose(particle.state.strengths, true_strengths, rtol=1e-2)


def test_refresh_weights_batches_mixed_source_cardinality() -> None:
    """Batched structural reweighting should match scalar per-particle likelihoods."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=3,
        count_likelihood_model="student_t",
        spectrum_count_abs_sigma=1.0,
        use_gpu=False,
    )
    states = [
        IsotopeState(
            num_sources=0,
            positions=np.zeros((0, 3), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([80.0], dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([60.0, 110.0], dtype=float),
            background=0.2,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=np.log(1.0 / 3.0))
        for state in states
    ]
    filt.N = len(states)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([31.0, 23.0, 9.0], dtype=float),
        observation_variances=np.array([32.0, 24.0, 10.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    scalar_ll = []
    for particle in filt.continuous_particles:
        _, lambda_total = filt._lambda_components(particle.state, data)
        scalar_ll.append(
            filt._count_log_likelihood_np(
                data.z_k,
                lambda_total,
                observation_count_variance=data.observation_variances,
            )
        )
    scalar_ll_arr = np.asarray(scalar_ll, dtype=float)
    max_ll = float(np.max(scalar_ll_arr))
    expected_logw = scalar_ll_arr - (
        max_ll + np.log(np.sum(np.exp(scalar_ll_arr - max_ll)))
    )

    filt.refresh_weights_from_measurements(data)

    actual_logw = np.asarray(
        [particle.log_weight for particle in filt.continuous_particles],
        dtype=float,
    )
    assert np.allclose(actual_logw, expected_logw)


def test_conditional_strength_refit_reweights_by_profile_likelihood() -> None:
    """Strength refit should update weights when fixed positions fit differently."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        conditional_strength_refit=True,
        conditional_strength_refit_reweight=True,
        conditional_strength_refit_reweight_clip=100.0,
    )
    true_state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
    )
    wrong_state = IsotopeState(
        num_sources=1,
        positions=np.array([[3.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(state=true_state, log_weight=np.log(0.5)),
        IsotopeParticle(state=wrong_state, log_weight=np.log(0.5)),
    ]
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_state.positions,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.ones(3, dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=5)

    weights = filt.continuous_weights
    assert weights[0] > weights[1]


def test_conditional_strength_refit_reweight_includes_strength_prior() -> None:
    """Profile-likelihood reweighting should penalize implausible strength jumps."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        conditional_strength_refit=True,
        conditional_strength_refit_reweight=True,
        conditional_strength_refit_reweight_clip=1.0e9,
        conditional_strength_refit_prior_weight=100.0,
        conditional_strength_refit_prior_rel_sigma=0.5,
    )
    near_prior = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0], dtype=float),
        background=0.0,
    )
    far_prior = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(state=near_prior, log_weight=np.log(0.999)),
        IsotopeParticle(state=far_prior, log_weight=np.log(0.001)),
    ]
    detector_positions = np.array([[1.0, 0.0, 0.0]], dtype=float)
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=near_prior.positions,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(1, dtype=float),
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.maximum(np.sum(expected, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        live_times=np.ones(1, dtype=float),
    )

    filt.refit_strengths_for_particles(data, iters=5)

    weights = filt.continuous_weights
    assert weights[0] > 0.95
    assert weights[0] > weights[1]


def test_clustered_estimate_uses_robust_strength_summary() -> None:
    """Clustered output should not be dominated by rare high-strength tails."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=3,
        cluster_min_samples=1,
    )
    strengths = [100.0, 100.0, 300000.0]
    log_weights = np.log([0.80, 0.15, 0.05])
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for strength, log_weight in zip(strengths, log_weights)
    ]

    _, q_est = filt.estimate_clustered()

    assert q_est.shape == (1,)
    assert np.isclose(q_est[0], 100.0)


def test_clustered_estimate_conditions_strength_on_active_sources() -> None:
    """Clustered output should not let numeric floor sources erase an active mode."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=1,
        num_particles=4,
        cluster_min_samples=1,
    )
    strengths = [5.0, 5.0, 42000.0, 45000.0]
    log_weights = np.log([0.35, 0.25, 0.25, 0.15])
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.1, 0.0, 0.0]], dtype=float),
                strengths=np.array([strength], dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for strength, log_weight in zip(strengths, log_weights)
    ]

    _, q_est = filt.estimate_clustered()

    assert q_est.shape == (1,)
    assert q_est[0] > 10000.0


def test_birth_is_blocked_when_strength_refit_explains_residual() -> None:
    """Residual birth should not fire when existing fixed sources absorb counts."""
    np.random.seed(3)
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_num_local_jitter=0,
        birth_detector_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_residual_support_sigma=1.0,
        birth_residual_gate_p_value=0.05,
        birth_refit_residual_gate=True,
        birth_refit_residual_min_fraction=0.5,
        split_prob=0.0,
        merge_prob=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([5.0], dtype=float),
        background=0.0,
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=state.positions,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.ones(2, dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    filt.apply_birth_death(
        support_data=None,
        birth_data=data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 0
    assert filt.continuous_particles[0].state.num_sources == 1
    assert not filt.last_birth_residual_refit_gate_passed


def test_residual_guided_split_separates_same_isotope_sources() -> None:
    """Residual-guided split should add a same-isotope source only when LL improves."""
    np.random.seed(4)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        split_prob=1.0,
        split_residual_guided=True,
        split_residual_candidate_count=3,
        split_delta_ll_threshold=0.0,
        min_age_to_split=0,
        birth_num_local_jitter=0,
        birth_min_sep_m=0.4,
        birth_detector_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_refit_residual_gate=False,
        merge_prob=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    initial_state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        low_q_streaks=np.zeros(1, dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    filt.continuous_particles = [IsotopeParticle(state=initial_state, log_weight=0.0)]
    true_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    true_strengths = np.array([100.0, 120.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 3.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=filt.kernel.sources,
    )

    state = filt.continuous_particles[0].state
    assert state.num_sources == 2
    assert np.min(np.linalg.norm(state.positions - true_positions[1][None, :], axis=1)) < 0.5


def test_residual_split_ranks_candidates_by_residual_support() -> None:
    """Residual split should test the strongest residual candidate first."""
    np.random.seed(11)
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        split_prob=0.0,
        split_residual_guided=True,
        split_residual_always_try=True,
        split_residual_candidate_count=1,
        split_delta_ll_threshold=0.0,
        min_age_to_split=0,
        birth_num_local_jitter=0,
        birth_min_sep_m=0.4,
        birth_detector_min_sep_m=0.0,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_refit_residual_gate=False,
        merge_prob=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    initial_state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([90.0], dtype=float),
        background=0.0,
        ages=np.array([2], dtype=int),
        low_q_streaks=np.zeros(1, dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    filt.continuous_particles = [IsotopeParticle(state=initial_state, log_weight=0.0)]
    true_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    true_strengths = np.array([90.0, 150.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.5, 2.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    candidate_positions = np.array(
        [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )

    proposal = filt._compute_birth_proposal(data, candidate_positions)

    assert proposal is not None
    _, _, _, ranked_candidates = proposal
    assert np.allclose(ranked_candidates[0], true_positions[1])

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=candidate_positions,
    )

    state = filt.continuous_particles[0].state
    assert state.num_sources == 2
    assert np.min(np.linalg.norm(state.positions - true_positions[1][None, :], axis=1)) < 0.5


def test_report_strength_refit_prunes_unsupported_component() -> None:
    """Reported Poisson strength refit should remove zero-supported components."""
    np.random.seed(5)
    isotope = "Cs-137"
    reported_positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    detector_positions = [
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 2.0, 0.0], dtype=float),
    ]
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        birth_enable=True,
        use_clustered_output=True,
        cluster_min_samples=1,
        report_strength_refit=True,
        report_strength_refit_iters=128,
        init_num_sources=(1, 1),
        min_strength=0.01,
        use_gpu=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=reported_positions,
        shield_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=config,
        shield_params=ShieldParams(thickness_pb_cm=0.0, thickness_fe_cm=0.0),
    )
    for pose in detector_positions:
        estimator.add_measurement_pose(pose)
    estimator._ensure_kernel_cache()
    true_strength = np.array([90.0], dtype=float)
    design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=reported_positions[:1],
        strengths=true_strength,
        live_times=np.ones(len(detector_positions), dtype=float),
        fe_indices=np.zeros(len(detector_positions), dtype=int),
        pb_indices=np.zeros(len(detector_positions), dtype=int),
    )
    counts = np.sum(design, axis=1)
    estimator.measurements = [
        MeasurementRecord(
            z_k={isotope: float(count)},
            pose_idx=idx,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: max(float(count), 1.0)},
        )
        for idx, count in enumerate(counts)
    ]

    positions, strengths = estimator._refit_reported_strengths(
        isotope,
        reported_positions,
        np.array([40.0, 40.0], dtype=float),
    )

    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
    assert np.allclose(positions[0], reported_positions[0], atol=1.0e-6)
    assert np.isclose(strengths[0], true_strength[0], rtol=1.0e-2)


def test_report_strength_refit_returns_empty_without_signal_support() -> None:
    """Reported source estimates should require signal-bearing observations."""
    isotope = "Co-60"
    positions = np.array([[2.0, 2.0, 0.0], [4.0, 4.0, 0.0]], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        birth_enable=True,
        report_strength_refit=True,
        conditional_strength_refit_min_count=5.0,
        conditional_strength_refit_min_snr=1.0,
        use_gpu=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=positions,
        shield_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=config,
        shield_params=ShieldParams(thickness_pb_cm=0.0, thickness_fe_cm=0.0),
    )
    detector_positions = [
        np.array([1.0, 1.0, 0.0], dtype=float),
        np.array([3.0, 1.0, 0.0], dtype=float),
    ]
    for pose in detector_positions:
        estimator.add_measurement_pose(pose)
    estimator._ensure_kernel_cache()
    estimator.measurements = [
        MeasurementRecord(
            z_k={isotope: 0.0},
            pose_idx=idx,
            orient_idx=0,
            live_time_s=1.0,
            fe_index=0,
            pb_index=0,
            z_variance_k={isotope: 10000.0},
        )
        for idx in range(len(detector_positions))
    ]

    refit_positions, refit_strengths = estimator._refit_reported_strengths(
        isotope,
        positions,
        np.array([1000.0, 1000.0], dtype=float),
    )

    assert refit_positions.shape == (0, 3)
    assert refit_strengths.shape == (0,)
