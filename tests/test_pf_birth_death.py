"""Tests for dynamic source cardinality with birth/death moves (Chapter 3, Sec. 3.4.2)."""

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import ShieldParams
from pf.likelihood import expected_counts_per_source
from pf.estimator import MeasurementRecord, RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import (
    BirthResidualLayer,
    IsotopeParticleFilter,
    IsotopeParticle,
    MeasurementData,
)
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


def test_birth_scoring_prefers_shield_coded_residual_shape() -> None:
    """Birth proposal scoring should prefer residual shape over raw count scale."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
    )
    candidate_counts = np.array(
        [
            [10.0, 30.0],
            [10.0, 0.0],
            [0.0, 300.0],
        ],
        dtype=float,
    )
    residual = np.array([10.0, 10.0, 0.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(3, dtype=float),
    )

    assert scores[0] > scores[1]
    assert q_hat[0] > q_hat[1]


def test_birth_scoring_uses_count_distance_prior_for_single_view() -> None:
    """Single-view residual birth should prefer high unit-response candidates."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
        birth_count_distance_prior_weight=1.0,
        birth_count_distance_strength_weight=1.0,
    )
    candidate_counts = np.array([[10.0, 1.0]], dtype=float)
    residual = np.array([100.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(1, dtype=float),
    )

    assert q_hat[0] < q_hat[1]
    assert scores[0] > scores[1]


def test_birth_scoring_can_disable_count_distance_prior() -> None:
    """Disabling the proposal prior should preserve pure least-squares ties."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_use_shield_coded_residual=True,
        birth_count_distance_prior_weight=0.0,
        birth_count_distance_strength_weight=0.0,
    )
    candidate_counts = np.array([[10.0, 1.0]], dtype=float)
    residual = np.array([100.0], dtype=float)

    scores, q_hat = filt._birth_residual_candidate_scores(
        candidate_counts=candidate_counts,
        residual_mix=residual,
        observation_variances=np.ones(1, dtype=float),
    )

    assert q_hat[0] < q_hat[1]
    assert np.isclose(scores[0], scores[1])


def test_peak_suppressed_residual_birth_reveals_noncollinear_weak_source() -> None:
    """Peak suppression should propose a weak candidate without rebirthing a strong source."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_refit_residual_gate=False,
        birth_existing_response_corr_max=0.95,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=2,
        peak_suppression_min_source_fraction=0.1,
        birth_num_local_jitter=0,
    )
    strong_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    weak_pos = np.array([[4.0, 4.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    strong_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=strong_pos,
        strengths=np.array([500.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    weak_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=weak_pos,
        strengths=np.array([80.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    filt.continuous_particles[0].state = IsotopeState(
        num_sources=1,
        positions=strong_pos.copy(),
        strengths=np.array([500.0], dtype=float),
        background=0.0,
    )
    data = MeasurementData(
        z_k=strong_counts + weak_counts,
        observation_variances=np.maximum(strong_counts + weak_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        live_times=np.ones(4, dtype=float),
    )

    proposal = filt._compute_birth_proposal(
        data,
        np.vstack([strong_pos, weak_pos]),
    )

    assert proposal is not None
    _, _, _, candidates, _ = proposal
    assert filt.last_birth_residual_layer.startswith("strong_suppressed")
    assert np.allclose(candidates[0], weak_pos[0])


def test_structural_updates_filter_low_count_rows(monkeypatch) -> None:
    """Structure-changing moves should ignore low-count rows, not PF weight updates."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        structural_update_min_counts=100.0,
    )
    filt.continuous_particles[0].state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    data = MeasurementData(
        z_k=np.array([4.0, 150.0, 6.0], dtype=float),
        observation_variances=np.array([4.0, 150.0, 6.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    seen_counts: list[np.ndarray] = []

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> None:
        """Record the rows passed to the residual birth proposal."""
        assert birth_data is not None
        seen_counts.append(np.asarray(birth_data.z_k, dtype=float).copy())
        return None

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=filt.kernel.sources,
    )

    assert len(seen_counts) == 1
    assert np.allclose(seen_counts[0], np.array([150.0], dtype=float))


def test_structural_updates_skip_when_only_low_count_rows(monkeypatch) -> None:
    """Birth proposals should not run when all structural evidence rows are weak."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        structural_update_min_counts=100.0,
        structural_update_min_snr=20.0,
    )
    data = MeasurementData(
        z_k=np.array([4.0, 6.0], dtype=float),
        observation_variances=np.array([4.0, 6.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    called = False

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> None:
        """Fail the test if low-count-only data reaches birth proposal."""
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=filt.kernel.sources,
    )

    assert not called
    assert not filt.last_birth_residual_gate_passed


def test_birth_response_condition_rejects_indistinguishable_candidate() -> None:
    """Same-isotope birth candidates should pass only when response columns differ."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_response_condition_max=10.0,
    )
    existing = np.array([[1.0], [2.0], [3.0]], dtype=float)
    candidates = np.array(
        [
            [2.0, 0.0],
            [4.0, 1.0],
            [6.0, 0.0],
        ],
        dtype=float,
    )

    keep = filt._birth_response_condition_mask(
        candidate_counts=candidates,
        existing_response_counts=existing,
        observation_variances=np.ones(3, dtype=float),
    )

    assert keep.tolist() == [False, True]


def test_birth_response_condition_is_incremental_with_bad_existing_matrix() -> None:
    """Ill-conditioned existing sources alone should not reject independent births."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=1,
        birth_response_condition_max=10.0,
    )
    existing = np.array(
        [
            [1.0, 1.01],
            [2.0, 2.02],
            [3.0, 3.03],
            [4.0, 4.04],
        ],
        dtype=float,
    )
    candidates = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )

    keep = filt._birth_response_condition_mask(
        candidate_counts=candidates,
        existing_response_counts=existing,
        observation_variances=np.ones(4, dtype=float),
    )

    assert keep.tolist() == [False, True]


def test_birth_residual_layers_include_leave_one_cluster_out() -> None:
    """Peak suppression should include cluster-level leave-one-out residuals."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=2,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=4,
        peak_suppression_min_source_fraction=0.1,
        cluster_eps_m=0.8,
    )
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    strong_positions = [
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.array([[0.2, 0.1, 0.0]], dtype=float),
    ]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=pos.copy(),
                strengths=np.array([500.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.5)),
        )
        for pos in strong_positions
    ]
    weak_pos = np.array([[4.0, 4.0, 0.0]], dtype=float)
    strong_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=strong_positions[0],
        strengths=np.array([500.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    weak_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=weak_pos,
        strengths=np.array([80.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    )[:, 0]
    data = MeasurementData(
        z_k=strong_counts + weak_counts,
        observation_variances=np.maximum(strong_counts + weak_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        live_times=np.ones(4, dtype=float),
    )

    layers = filt._compute_birth_residual_layers(
        data=data,
        particle_indices=np.array([0, 1], dtype=int),
        particle_weights=np.array([0.5, 0.5], dtype=float),
    )

    assert any(layer.name.startswith("leave_one_cluster_out") for layer in layers)


def test_matching_pursuit_birth_can_add_multiple_sources() -> None:
    """Residual matching pursuit should add more than one supported source."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_refit_residual_gate=False,
        birth_matching_pursuit_max_new_sources=2,
        birth_matching_pursuit_topk_candidates=3,
        birth_min_sep_m=0.4,
        birth_bic_penalty_params=0,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    true_positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([120.0, 90.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
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

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        true_positions,
        max_new_sources=2,
    )

    assert accepted == 2
    assert state.num_sources == 2


def test_pseudo_source_verification_prunes_unsupported_tentative_source() -> None:
    """Pseudo-source verification should quarantine before hard pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert changed
    assert state.num_sources == 2
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert changed
    assert state.num_sources == 1
    assert filt.last_pseudo_source_pruned == 1


def test_pseudo_source_verification_requires_multiple_stations_to_prune() -> None:
    """A tentative source should quarantine when hard prune is not allowed."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.arange(3, dtype=int),
        pb_indices=np.arange(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.arange(3, dtype=int),
        pb_indices=np.arange(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    filt._verify_pseudo_sources_for_state(state, data, suppress_prune=False)

    assert state.num_sources == 2
    assert filt.last_pseudo_source_failed == 1
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0
    assert filt._quarantined_source_mask(state).tolist() == [False, True]


def test_pseudo_source_quarantine_does_not_require_prune_allowed() -> None:
    """Suppressed pseudo-source failures should quarantine before hard pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=0.0,
        source_prune_min_distinct_stations=2,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.arange(2, dtype=int),
        pb_indices=np.arange(2, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.array([10.0, 7.5], dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.arange(2, dtype=int),
        pb_indices=np.arange(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=True,
    )

    assert changed
    assert state.num_sources == 2
    assert filt.last_pseudo_source_quarantined == 1
    assert filt.last_pseudo_source_pruned == 0
    assert filt._quarantined_source_mask(state).tolist() == [False, True]
    assert np.allclose(state.support_scores, np.array([10.0, 7.5], dtype=float))


def test_pseudo_source_correlation_failure_requests_more_views() -> None:
    """Collinear tentative responses should not trigger quarantine or pruning."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=3,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=1.0e9,
        pseudo_source_corr_max=0.995,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=position,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([position, position]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([5, 5], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
    )

    assert not changed
    assert state.num_sources == 2
    assert state.verification_fail_streaks.tolist() == [0, 0]
    assert filt.last_pseudo_source_failed == 1
    assert filt.last_pseudo_source_quarantined == 0
    assert filt.last_pseudo_source_pruned == 0
    assert filt.last_pseudo_source_fail_reasons["needs_discriminative_views"] == 1


def test_source_prune_mask_protects_collinear_tentative_sources() -> None:
    """Refit pruning should not delete tentative sources in collinear windows."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_corr_max=0.995,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([200.0, 100.0], dtype=float),
        background=0.0,
        ages=np.array([5, 5], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    lambda_m = np.array([[10.0, 5.0], [20.0, 10.0], [30.0, 15.0]], dtype=float)

    allowed = filt._tentative_response_separation_prune_mask(state, lambda_m)

    assert allowed.tolist() == [True, False]


def test_pseudo_source_verification_uses_cached_prune_allowed() -> None:
    """Pseudo-source verification should reuse cached prune decisions."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        pseudo_source_verification_enable=True,
        pseudo_source_min_distinct_views=1,
        pseudo_source_fail_grace_stations=1,
        pseudo_source_min_delta_ll=1.0e9,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_fail_grace_stations=1,
    )
    true_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    false_pos = np.array([[4.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_pos,
        strengths=np.array([200.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([true_pos, false_pos]),
        strengths=np.array([200.0, 200.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    lambda_m, lambda_total = filt._lambda_components(state, data)
    delta_ll = filt._delta_log_likelihood_remove(
        data.z_k,
        lambda_total,
        lambda_m,
        observation_count_variance=data.observation_variances,
    )

    def fail_uncached_prune(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if pseudo verification recomputes prune permission."""
        raise AssertionError("uncached prune permission was recomputed")

    filt._source_prune_allowed_mask = fail_uncached_prune

    changed = filt._verify_pseudo_sources_for_state(
        state,
        data,
        suppress_prune=False,
        cached_lambda_m=lambda_m,
        cached_lambda_total=lambda_total,
        cached_delta_ll=delta_ll,
        cached_prune_allowed=np.array([False, True], dtype=bool),
    )

    assert changed
    assert filt.last_pseudo_source_quarantined == 1


def test_refit_after_remove_prune_allows_redundant_source_removal() -> None:
    """Refit-after-remove pruning should catch collinear redundant components."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_refit_after_remove=True,
        source_prune_bic_penalty_params=4,
    )
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    source_pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=source_pos,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )[:, 0]
    state = IsotopeState(
        num_sources=2,
        positions=np.vstack([source_pos, source_pos]),
        strengths=np.array([50.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
    )
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    allowed = filt._source_prune_allowed_mask(state, data)

    assert np.all(allowed)


def test_batched_refit_uses_batched_refit_after_remove_prune() -> None:
    """Equal-cardinality refit should not fall back to scalar prune refits."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=2,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_refit_after_remove=True,
    )
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=positions[idx].copy(),
                strengths=np.array([50.0, 50.0], dtype=float),
                background=0.0,
                ages=np.array([3, 3], dtype=int),
                low_q_streaks=np.zeros(2, dtype=int),
                support_scores=np.zeros(2, dtype=float),
            ),
            log_weight=float(np.log(0.5)),
        )
        for idx in range(2)
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([120.0, 80.0, 45.0], dtype=float),
        observation_variances=np.array([120.0, 80.0, 45.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    def fail_scalar_refit(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if the old per-particle refit-after-remove path is used."""
        raise AssertionError("scalar refit-after-remove prune was called")

    filt._source_prune_refit_after_remove_mask = fail_scalar_refit

    filt._refit_fixed_source_count_particles_batched(
        data,
        particle_indices=[0, 1],
        source_count=2,
        iters=1,
        eps=1.0e-12,
    )

    assert all(p.state.num_sources >= 1 for p in filt.continuous_particles)


def test_refit_after_remove_vectorized_matches_loop_oracle() -> None:
    """Vectorized refit-after-remove masks should match source-loop masks."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=3,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_refit_after_remove=True,
        conditional_strength_refit_prior_weight=0.0,
    )
    base_positions = np.array(
        [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [2.0, 0.2, 0.0]],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=3,
                positions=base_positions + np.array([0.05 * idx, 0.0, 0.0]),
                strengths=np.array([80.0, 45.0 + idx, 25.0], dtype=float),
                background=0.0,
                ages=np.array([4, 4, 4], dtype=int),
                low_q_streaks=np.zeros(3, dtype=int),
                support_scores=np.ones(3, dtype=float),
            ),
            log_weight=float(np.log(1.0 / 3.0)),
        )
        for idx in range(3)
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=base_positions,
        strengths=np.array([80.0, 45.0, 25.0], dtype=float),
        live_times=np.ones(4, dtype=float),
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=true_counts,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(4, dtype=int),
        pb_indices=np.zeros(4, dtype=int),
        live_times=np.ones(4, dtype=float),
    )
    k_tensor, background_counts, strengths = filt._unit_kernel_tensor_for_particle_group(
        data,
        particle_indices=[0, 1, 2],
        source_count=3,
    )
    q_min = max(float(filt.config.min_strength), 0.0)
    q_max = float(filt.config.birth_q_max)
    full_strengths, full_lambda = filt._solve_strengths_for_kernel_tensor_batched(
        data,
        k_tensor=k_tensor,
        background_counts=background_counts,
        prior_mean=strengths,
        iters=2,
        eps=1.0e-12,
        q_min=q_min,
        q_max=q_max,
    )

    loop_mask = filt._source_prune_refit_after_remove_mask_loop(
        data,
        k_tensor=k_tensor,
        background_counts=background_counts,
        full_strengths=full_strengths,
        full_lambda_total=full_lambda,
        iters=2,
        eps=1.0e-12,
        q_min=q_min,
        q_max=q_max,
    )
    vectorized_mask = filt._source_prune_refit_after_remove_mask_batched(
        data,
        k_tensor=k_tensor,
        background_counts=background_counts,
        full_strengths=full_strengths,
        full_lambda_total=full_lambda,
        iters=2,
        eps=1.0e-12,
        q_min=q_min,
        q_max=q_max,
    )

    assert np.array_equal(vectorized_mask, loop_mask)


def test_apply_birth_death_reuses_batched_refit_prune_cache() -> None:
    """Structural updates should not scalar-refit prune masks twice."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=2,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_refit_after_remove=True,
        death_low_q_streak=99,
    )
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=positions[idx].copy(),
                strengths=np.array([50.0, 50.0], dtype=float),
                background=0.0,
                ages=np.array([3, 3], dtype=int),
                low_q_streaks=np.zeros(2, dtype=int),
                support_scores=np.zeros(2, dtype=float),
            ),
            log_weight=float(np.log(0.5)),
        )
        for idx in range(2)
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([120.0, 80.0, 45.0], dtype=float),
        observation_variances=np.array([120.0, 80.0, 45.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    def fail_scalar_refit(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if apply_birth_death uses the scalar prune-refit path."""
        raise AssertionError("scalar refit-after-remove prune was called")

    filt._source_prune_refit_after_remove_mask = fail_scalar_refit

    filt.apply_birth_death(
        support_data=data,
        birth_data=None,
        candidate_positions=filt.kernel.sources,
    )

    assert all(p.state.num_sources >= 1 for p in filt.continuous_particles)


def test_apply_birth_death_skips_refit_prune_without_candidates() -> None:
    """Structural updates should not refit-after-remove when no prune can occur."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=2,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
        source_prune_refit_after_remove=True,
        death_low_q_streak=99,
        pseudo_source_verification_enable=True,
    )
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=positions[idx].copy(),
                strengths=np.array([50.0, 50.0], dtype=float),
                background=0.0,
                ages=np.array([3, 3], dtype=int),
                low_q_streaks=np.zeros(2, dtype=int),
                support_scores=np.full(2, 100.0, dtype=float),
                tentative_sources=np.zeros(2, dtype=bool),
                verification_fail_streaks=np.zeros(2, dtype=int),
            ),
            log_weight=float(np.log(0.5)),
        )
        for idx in range(2)
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([120.0, 80.0, 45.0], dtype=float),
        observation_variances=np.array([120.0, 80.0, 45.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    def fail_batched_refit(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if prune-refit work runs without any prune candidates."""
        raise AssertionError("unnecessary refit-after-remove prune was called")

    filt._source_prune_refit_after_remove_mask_batched = fail_batched_refit

    filt.apply_birth_death(
        support_data=data,
        birth_data=None,
        candidate_positions=filt.kernel.sources,
    )

    assert all(p.state.num_sources == 2 for p in filt.continuous_particles)


def test_matching_pursuit_birth_uses_cached_candidate_counts(monkeypatch) -> None:
    """Matching-pursuit birth should not recompute cached candidate responses."""
    import pf.particle_filter as particle_filter_module

    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_matching_pursuit_max_new_sources=2,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([10.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        low_q_streaks=np.zeros(1, dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.array([10.0, 12.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    candidates = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    candidate_counts = np.zeros((2, 2), dtype=float)
    original_expected_counts = particle_filter_module.expected_counts_per_source

    def guarded_expected_counts(*args: object, **kwargs: object) -> np.ndarray:
        """Reject the candidate-response recomputation that the cache avoids."""
        sources = np.asarray(kwargs.get("sources"), dtype=float)
        strengths = np.asarray(kwargs.get("strengths"), dtype=float)
        if sources.shape == candidates.shape and np.allclose(sources, candidates):
            if strengths.shape == (2,) and np.allclose(strengths, 1.0):
                raise AssertionError("candidate responses were recomputed")
        return original_expected_counts(*args, **kwargs)

    monkeypatch.setattr(
        particle_filter_module,
        "expected_counts_per_source",
        guarded_expected_counts,
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        candidates,
        max_new_sources=2,
        residual_gate_forced=True,
        candidate_unit_counts=candidate_counts,
    )

    assert accepted == 0


def test_birth_existing_unit_response_counts_batched_match_scalar_oracle() -> None:
    """Existing-source birth response columns should be batched without drift."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=3,
        pseudo_source_fail_grace_stations=1,
        source_prune_fail_grace_stations=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
                    dtype=float,
                ),
                strengths=np.array([10.0, 5.0], dtype=float),
                background=0.0,
                ages=np.array([3, 1], dtype=int),
                low_q_streaks=np.zeros(2, dtype=int),
                support_scores=np.zeros(2, dtype=float),
                tentative_sources=np.array([False, True], dtype=bool),
                verification_fail_streaks=np.array([0, 1], dtype=int),
            ),
            log_weight=np.log(0.4),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 0.1, 0.0]], dtype=float),
                strengths=np.array([6.0], dtype=float),
                background=0.0,
                ages=np.array([2], dtype=int),
                low_q_streaks=np.zeros(1, dtype=int),
                support_scores=np.zeros(1, dtype=float),
            ),
            log_weight=np.log(0.35),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.25),
        ),
    ]
    filt.N = len(filt.continuous_particles)
    data = MeasurementData(
        z_k=np.array([8.0, 5.0, 3.0], dtype=float),
        observation_variances=np.array([8.0, 5.0, 3.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    indices = np.array([0, 1, 2], dtype=int)

    scalar = filt._birth_existing_unit_response_counts_scalar(
        data,
        particle_indices=indices,
    )
    batched = filt._birth_existing_unit_response_counts(
        data,
        particle_indices=indices,
    )

    assert np.allclose(batched, scalar)
    assert batched.shape == (3, 3)


def test_birth_proposal_reuses_raw_refit_gate(monkeypatch) -> None:
    """Raw residual refit gate should not be recomputed for the same layer."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_refit_residual_gate=True,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.array([10.0, 12.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    monkeypatch.setattr(
        filt,
        "_compute_birth_residual_layers",
        lambda **_kwargs: [
            BirthResidualLayer(
                name="raw",
                residual=np.array([5.0, 4.0], dtype=float),
            )
        ],
    )
    calls = {"gate": 0, "refit": 0}

    def count_residual_gate(*_args: object, **_kwargs: object) -> bool:
        """Record residual-gate calls while preserving a passing result."""
        calls["gate"] += 1
        return True

    monkeypatch.setattr(filt, "_birth_residual_gate_allows", count_residual_gate)

    def count_refit_gate(**_kwargs: object) -> bool:
        """Record refit-gate calls while preserving a passing result."""
        calls["refit"] += 1
        return True

    monkeypatch.setattr(
        filt,
        "_birth_residual_survives_strength_refit",
        count_refit_gate,
    )
    monkeypatch.setattr(
        filt,
        "_birth_candidate_support_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_existing_response_correlation_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_response_condition_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_residual_candidate_scores",
        lambda *, candidate_counts, **_kwargs: (
            np.ones(np.asarray(candidate_counts).shape[1], dtype=float),
            np.ones(np.asarray(candidate_counts).shape[1], dtype=float),
        ),
    )

    result = filt._compute_birth_proposal(
        data,
        np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float),
    )

    assert result is not None
    assert calls == {"gate": 1, "refit": 1}


def test_birth_proposal_falls_back_to_suppressed_layer_when_raw_refit_fails(
    monkeypatch,
) -> None:
    """A failed raw refit gate should not suppress shield-coded residual layers."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_refit_residual_gate=True,
        birth_residual_min_support=1,
        birth_min_distinct_poses=1,
        birth_min_distinct_stations=1,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.array([10.0, 12.0], dtype=float),
        detector_positions=np.array(
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    monkeypatch.setattr(
        filt,
        "_compute_birth_residual_layers",
        lambda **_kwargs: [
            BirthResidualLayer(name="raw", residual=np.array([5.0, 4.0], dtype=float)),
            BirthResidualLayer(
                name="strong_suppressed_0",
                residual=np.array([4.0, 3.0], dtype=float),
            ),
        ],
    )
    monkeypatch.setattr(filt, "_birth_residual_gate_allows", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        filt,
        "_birth_residual_survives_strength_refit",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        filt,
        "_birth_candidate_support_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_existing_response_correlation_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_response_condition_mask",
        lambda *, candidate_counts, **_kwargs: np.ones(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_residual_candidate_scores",
        lambda *, candidate_counts, **_kwargs: (
            np.ones(np.asarray(candidate_counts).shape[1], dtype=float),
            np.ones(np.asarray(candidate_counts).shape[1], dtype=float),
        ),
    )

    result = filt._compute_birth_proposal(
        data,
        np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float),
    )

    assert result is not None
    assert filt.last_birth_residual_refit_gate_passed is True
    assert filt.last_birth_residual_layer == "strong_suppressed_0"


def test_matching_pursuit_cached_refit_matches_scalar_path() -> None:
    """Cached matching-pursuit trial refits should match scalar refits."""
    common_kwargs = dict(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_residual_gate_p_value=1.0,
        birth_candidate_support_fraction=0.0,
        birth_refit_residual_gate=False,
        birth_matching_pursuit_max_new_sources=2,
        birth_matching_pursuit_topk_candidates=3,
        birth_min_sep_m=0.4,
        birth_bic_penalty_params=0,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    fast = _build_filter(**common_kwargs, birth_residual_suppress_death=True)
    scalar = _build_filter(**common_kwargs, birth_residual_suppress_death=False)
    true_positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([120.0, 90.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=fast.continuous_kernel,
        isotope=fast.isotope,
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
    candidate_counts = expected_counts_per_source(
        kernel=fast.continuous_kernel,
        isotope=fast.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=np.ones(2, dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    fast_state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    scalar_state = fast_state.copy()

    accepted_fast = fast._apply_matching_pursuit_births_to_state(
        fast_state,
        data,
        true_positions,
        max_new_sources=2,
        residual_gate_forced=True,
        candidate_unit_counts=candidate_counts,
    )
    accepted_scalar = scalar._apply_matching_pursuit_births_to_state(
        scalar_state,
        data,
        true_positions,
        max_new_sources=2,
        residual_gate_forced=True,
        candidate_unit_counts=candidate_counts,
    )

    assert accepted_fast == accepted_scalar == 2
    assert np.allclose(fast_state.positions, scalar_state.positions)
    assert np.allclose(fast_state.strengths, scalar_state.strengths)


def test_matching_pursuit_batched_trial_matches_scalar_oracle() -> None:
    """Batched matching-pursuit candidate trials should match scalar solves."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        birth_residual_suppress_death=True,
        birth_matching_pursuit_topk_candidates=3,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([50.0], dtype=float),
        background=0.0,
    )
    candidates = np.array(
        [[2.0, 0.0, 0.0], [3.0, 0.5, 0.0], [4.0, 0.0, 0.0]],
        dtype=float,
    )
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=np.vstack([state.positions, candidates[[1]]]),
        strengths=np.array([50.0, 70.0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    unit_existing = filt._unit_response_counts_for_state(state, data)
    unit_all = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=candidates,
        strengths=np.ones(candidates.shape[0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )
    q_hat = np.array([20.0, 70.0, 30.0], dtype=float)
    ranked = np.array([0, 1, 2], dtype=int)
    base_lambda = unit_existing @ state.strengths[: state.num_sources]
    base_ll = filt._trial_log_likelihood_from_lambda(data, base_lambda)

    batched_trial, batched_delta = (
        filt._best_cached_matching_pursuit_birth_trial_batched(
            state,
            data,
            candidates=candidates,
            ranked_candidate_indices=ranked,
            q_hat=q_hat,
            unit_counts_existing=unit_existing,
            unit_counts_all=unit_all,
            source_strengths=state.strengths[: state.num_sources],
            base_ll=base_ll,
        )
    )
    scalar_trial = None
    scalar_delta = -np.inf
    for cand_idx in ranked:
        trial_counts = np.column_stack([unit_existing, unit_all[:, int(cand_idx)]])
        trial_prior = np.concatenate(
            [state.strengths[: state.num_sources], [q_hat[int(cand_idx)]]]
        )
        trial_strengths, _lambda_total, ll_after = (
            filt._solve_trial_strengths_from_unit_counts(
                data,
                trial_counts,
                trial_prior,
                state.background,
                iters=filt.config.refit_iters,
                eps=filt.config.refit_eps,
            )
        )
        delta = float(ll_after - base_ll)
        if delta > scalar_delta:
            scalar_delta = delta
            scalar_trial = state.copy()
            scalar_trial.positions = np.vstack(
                [scalar_trial.positions[: scalar_trial.num_sources], candidates[int(cand_idx)]]
            )
            scalar_trial.strengths = trial_strengths
            scalar_trial.num_sources = int(scalar_trial.positions.shape[0])

    assert batched_trial is not None
    assert scalar_trial is not None
    assert np.isclose(batched_delta, scalar_delta)
    assert np.allclose(batched_trial.positions, scalar_trial.positions)
    assert np.allclose(batched_trial.strengths, scalar_trial.strengths)


def test_merge_trial_batched_matches_scalar_oracle() -> None:
    """Batched merge candidate trials should match the scalar reference path."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=1,
        merge_distance_max=5.0,
        merge_response_corr_min=0.0,
        merge_search_topk_pairs=6,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    state = IsotopeState(
        num_sources=3,
        positions=np.array(
            [[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [2.5, 0.2, 0.0]],
            dtype=float,
        ),
        strengths=np.array([80.0, 70.0, 35.0], dtype=float),
        background=0.0,
        ages=np.array([4, 3, 5], dtype=int),
        low_q_streaks=np.zeros(3, dtype=int),
        support_scores=np.array([3.0, 2.0, 1.0], dtype=float),
        tentative_sources=np.array([False, True, False], dtype=bool),
        verification_fail_streaks=np.array([0, 1, 0], dtype=int),
    )
    true_positions = np.array(
        [[0.15, 0.0, 0.0], [2.5, 0.2, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([150.0, 35.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.5, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    scalar_trial, scalar_delta = filt._best_merge_trial_scalar(state.copy(), data)
    batched_trial, batched_delta = filt._best_merge_trial(state.copy(), data)

    assert scalar_trial is not None
    assert batched_trial is not None
    assert np.isclose(batched_delta, scalar_delta)
    assert batched_trial.num_sources == scalar_trial.num_sources
    assert np.allclose(batched_trial.positions, scalar_trial.positions)
    assert np.allclose(batched_trial.strengths, scalar_trial.strengths)
    assert np.array_equal(batched_trial.tentative_sources, scalar_trial.tentative_sources)


def test_residual_split_cached_batched_matches_scalar_oracle() -> None:
    """Cached residual split trials should be batched without changing the result."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        split_residual_guided=True,
        split_residual_candidate_count=4,
        birth_min_sep_m=0.4,
        min_age_to_split=0,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([90.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        low_q_streaks=np.zeros(1, dtype=int),
        support_scores=np.zeros(1, dtype=float),
        tentative_sources=np.array([False], dtype=bool),
        verification_fail_streaks=np.zeros(1, dtype=int),
    )
    candidates = np.array(
        [[1.8, 0.0, 0.0], [2.2, 0.2, 0.0], [3.5, 0.0, 0.0]],
        dtype=float,
    )
    true_positions = np.array([[0.0, 0.0, 0.0], [2.2, 0.2, 0.0]], dtype=float)
    true_strengths = np.array([80.0, 110.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.5, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    candidate_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=candidates,
        strengths=np.ones(candidates.shape[0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )

    batched_trial, batched_delta = filt._best_residual_guided_split_trial(
        state.copy(),
        data,
        candidates,
        None,
        suppress_prune_after_refit=True,
        candidate_unit_counts=candidate_counts,
    )

    existing_counts = filt._unit_response_counts_for_state(state, data)
    base_lambda = existing_counts @ state.strengths[: state.num_sources]
    base_ll = filt._trial_log_likelihood_from_lambda(data, base_lambda)
    cand_strengths = filt._candidate_initial_strengths(
        candidate_count=candidates.shape[0],
        candidate_kernel_sums=None,
        residual_sum=float(np.sum(np.maximum(data.z_k, 0.0))),
    )
    scalar_trial = None
    scalar_delta = -np.inf
    for cand_idx in range(candidates.shape[0]):
        trial_counts = np.column_stack([existing_counts, candidate_counts[:, cand_idx]])
        q_new = max(float(cand_strengths[cand_idx]), float(filt.config.min_strength))
        keep_strength = max(
            float(state.strengths[0]) - q_new,
            float(filt.config.min_strength),
        )
        trial_prior = np.array([keep_strength, q_new], dtype=float)
        trial_strengths, _lambda_total, ll_after = (
            filt._solve_trial_strengths_from_unit_counts(
                data,
                trial_counts,
                trial_prior,
                state.background,
                iters=filt.config.refit_iters,
                eps=filt.config.refit_eps,
            )
        )
        delta = float(ll_after - base_ll)
        if delta > scalar_delta:
            scalar_delta = delta
            scalar_trial = state.copy()
            scalar_trial.positions = np.vstack([state.positions, candidates[cand_idx]])
            scalar_trial.strengths = trial_strengths
            scalar_trial.num_sources = 2

    assert batched_trial is not None
    assert scalar_trial is not None
    assert np.isclose(batched_delta, scalar_delta)
    assert np.allclose(batched_trial.positions, scalar_trial.positions)
    assert np.allclose(batched_trial.strengths, scalar_trial.strengths)


def test_merge_trial_evaluation_can_be_chunked_without_drift() -> None:
    """Independent merge trials should match when evaluated in worker chunks."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=1,
        merge_distance_max=5.0,
        merge_response_corr_min=0.0,
        merge_search_topk_pairs=6,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    states = [
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.0, 0.0], [0.25, 0.05, 0.0], [2.0, 0.2, 0.0]],
                dtype=float,
            ),
            strengths=np.array([90.0, 80.0, 30.0], dtype=float),
            background=0.0,
            ages=np.array([4, 3, 5], dtype=int),
            low_q_streaks=np.zeros(3, dtype=int),
            support_scores=np.ones(3, dtype=float),
            tentative_sources=np.array([False, True, False], dtype=bool),
            verification_fail_streaks=np.zeros(3, dtype=int),
        ),
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.1, 0.0], [1.2, 0.1, 0.0], [1.35, 0.15, 0.0]],
                dtype=float,
            ),
            strengths=np.array([40.0, 60.0, 55.0], dtype=float),
            background=0.0,
            ages=np.array([5, 4, 3], dtype=int),
            low_q_streaks=np.zeros(3, dtype=int),
            support_scores=np.ones(3, dtype=float),
            tentative_sources=np.zeros(3, dtype=bool),
            verification_fail_streaks=np.zeros(3, dtype=int),
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array([[2.5, 0.0, 0.0], [2.75, 0.05, 0.0]], dtype=float),
            strengths=np.array([35.0, 33.0], dtype=float),
            background=0.0,
            ages=np.array([3, 3], dtype=int),
            low_q_streaks=np.zeros(2, dtype=int),
            support_scores=np.ones(2, dtype=float),
            tentative_sources=np.zeros(2, dtype=bool),
            verification_fail_streaks=np.zeros(2, dtype=int),
        ),
    ]
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.5, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=float,
    )
    true_positions = np.array(
        [[0.1, 0.02, 0.0], [1.3, 0.12, 0.0], [2.65, 0.02, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([170.0, 115.0, 68.0], dtype=float)
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )

    def signature(result: tuple[IsotopeState | None, float]) -> tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]:
        """Return a compact deterministic signature for a merge trial result."""
        trial, delta = result
        if trial is None:
            return ((0, (), ()), float(delta))
        return (
            (
                int(trial.num_sources),
                tuple(np.round(trial.positions.reshape(-1), 12)),
                tuple(np.round(trial.strengths.reshape(-1), 12)),
            ),
            float(np.round(delta, 12)),
        )

    serial = [signature(filt._best_merge_trial(state.copy(), data)) for state in states]
    chunked: list[tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]] = []
    for chunk in ([0, 1], [2]):
        chunked.extend(
            signature(filt._best_merge_trial(states[idx].copy(), data)) for idx in chunk
        )
    filt.config.structural_trial_workers = 2
    filt.config.structural_trial_parallel_min_trials = 1
    threaded = [signature(filt._best_merge_trial(state.copy(), data)) for state in states]

    assert chunked == serial
    assert threaded == serial


def test_cached_split_trial_evaluation_can_be_chunked_without_drift() -> None:
    """Independent residual split trials should match across worker chunks."""
    filt = _build_filter(
        p_birth=0.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        split_residual_guided=True,
        split_residual_candidate_count=4,
        birth_min_sep_m=0.4,
        min_age_to_split=0,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([90.0], dtype=float),
            background=0.0,
            ages=np.array([3], dtype=int),
            low_q_streaks=np.zeros(1, dtype=int),
            support_scores=np.zeros(1, dtype=float),
            tentative_sources=np.array([False], dtype=bool),
            verification_fail_streaks=np.zeros(1, dtype=int),
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[1.0, 0.1, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
            ages=np.array([4], dtype=int),
            low_q_streaks=np.zeros(1, dtype=int),
            support_scores=np.zeros(1, dtype=float),
            tentative_sources=np.array([False], dtype=bool),
            verification_fail_streaks=np.zeros(1, dtype=int),
        ),
    ]
    candidates = np.array(
        [[1.8, 0.0, 0.0], [2.2, 0.2, 0.0], [3.5, 0.0, 0.0]],
        dtype=float,
    )
    true_positions = np.array([[0.0, 0.0, 0.0], [2.2, 0.2, 0.0]], dtype=float)
    true_strengths = np.array([80.0, 110.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.5, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
        strengths=true_strengths,
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    candidate_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=candidates,
        strengths=np.ones(candidates.shape[0], dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )

    def evaluate(state: IsotopeState) -> tuple[IsotopeState | None, float]:
        """Return the cached split trial result for one particle state."""
        return filt._best_residual_guided_split_trial(
            state.copy(),
            data,
            candidates,
            None,
            suppress_prune_after_refit=True,
            candidate_unit_counts=candidate_counts,
        )

    def signature(result: tuple[IsotopeState | None, float]) -> tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]:
        """Return a compact deterministic signature for a split trial result."""
        trial, delta = result
        if trial is None:
            return ((0, (), ()), float(delta))
        return (
            (
                int(trial.num_sources),
                tuple(np.round(trial.positions.reshape(-1), 12)),
                tuple(np.round(trial.strengths.reshape(-1), 12)),
            ),
            float(np.round(delta, 12)),
        )

    serial = [signature(evaluate(state)) for state in states]
    chunked: list[tuple[tuple[int, tuple[float, ...], tuple[float, ...]], float]] = []
    for chunk in ([0], [1]):
        chunked.extend(signature(evaluate(states[idx])) for idx in chunk)
    filt.config.structural_trial_workers = 2
    filt.config.structural_trial_parallel_min_trials = 1
    threaded = [signature(evaluate(state)) for state in states]

    assert chunked == serial
    assert threaded == serial


def test_birth_residual_layers_batched_match_scalar_oracle() -> None:
    """Batched birth residual layers should match scalar per-particle layers."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=4,
        num_particles=4,
        residual_decomposition_enable=True,
        peak_suppression_enable=True,
        residual_decomposition_max_layers=4,
        peak_suppression_min_source_fraction=0.0,
        peak_suppression_factor=1.0,
        birth_use_weighted_topk=True,
        birth_residual_clip_quantile=0.95,
        cluster_eps_m=1.0,
        birth_min_sep_m=0.5,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([20.0], dtype=float),
            background=0.2,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]],
                dtype=float,
            ),
            strengths=np.array([15.0, 8.0], dtype=float),
            background=0.1,
        ),
        IsotopeState(
            num_sources=2,
            positions=np.array(
                [[1.5, 0.0, 0.0], [2.2, 0.4, 0.0]],
                dtype=float,
            ),
            strengths=np.array([12.0, 5.0], dtype=float),
            background=0.3,
        ),
        IsotopeState(
            num_sources=0,
            positions=np.zeros((0, 3), dtype=float),
            strengths=np.zeros(0, dtype=float),
            background=0.4,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=float(np.log(0.25)))
        for state in states
    ]
    filt.N = len(filt.continuous_particles)
    detector_positions = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    true_sources = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.0]],
        dtype=float,
    )
    true_counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_sources,
        strengths=np.array([18.0, 10.0, 7.0], dtype=float),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        source_scale=1.0,
    ).sum(axis=1)
    data = MeasurementData(
        z_k=true_counts + 1.0,
        observation_variances=np.maximum(true_counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
        pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        live_times=np.ones(detector_positions.shape[0], dtype=float),
    )
    particle_indices = np.arange(len(filt.continuous_particles), dtype=int)
    particle_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)

    batched_layers = filt._compute_birth_residual_layers(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
    )
    scalar_layers = filt._compute_birth_residual_layers_scalar(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
    )

    assert [layer.name for layer in batched_layers] == [
        layer.name for layer in scalar_layers
    ]
    for batched, scalar in zip(batched_layers, scalar_layers):
        assert np.allclose(batched.residual, scalar.residual)


def test_residual_split_cached_refit_matches_scalar_path() -> None:
    """Cached residual split trial refits should match scalar trial refits."""
    common_kwargs = dict(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        split_residual_guided=True,
        split_residual_candidate_count=1,
        split_strength_min=1.0,
        min_age_to_split=1,
        birth_min_sep_m=0.4,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    fast = _build_filter(**common_kwargs, birth_residual_suppress_death=True)
    scalar = _build_filter(**common_kwargs, birth_residual_suppress_death=False)
    state = IsotopeState(
        num_sources=1,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([210.0], dtype=float),
        background=0.0,
        ages=np.array([3], dtype=int),
        low_q_streaks=np.zeros(1, dtype=int),
        support_scores=np.zeros(1, dtype=float),
    )
    true_positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    true_strengths = np.array([120.0, 90.0], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=fast.continuous_kernel,
        isotope=fast.isotope,
        detector_positions=detector_positions,
        sources=true_positions,
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
    candidates = np.array([[2.0, 0.0, 0.0]], dtype=float)
    candidate_counts = expected_counts_per_source(
        kernel=fast.continuous_kernel,
        isotope=fast.isotope,
        detector_positions=detector_positions,
        sources=candidates,
        strengths=np.ones(1, dtype=float),
        live_times=np.ones(3, dtype=float),
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        source_scale=1.0,
    )

    fast_trial, fast_delta = fast._best_residual_guided_split_trial(
        state.copy(),
        data,
        candidates,
        np.array([90.0], dtype=float),
        suppress_prune_after_refit=True,
        candidate_unit_counts=candidate_counts,
    )
    scalar_trial, scalar_delta = scalar._best_residual_guided_split_trial(
        state.copy(),
        data,
        candidates,
        np.array([90.0], dtype=float),
        suppress_prune_after_refit=False,
    )

    assert fast_trial is not None
    assert scalar_trial is not None
    assert np.isclose(fast_delta, scalar_delta)
    assert np.allclose(fast_trial.positions, scalar_trial.positions)
    assert np.allclose(fast_trial.strengths, scalar_trial.strengths)


def test_clustered_output_excludes_quarantined_sources() -> None:
    """Quarantined tentative sources should not appear in reported clusters."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        cluster_min_samples=1,
        use_clustered_output=True,
        pseudo_source_quarantine_excludes_runtime=True,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 2], dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    positions, strengths = filt.estimate_clustered()

    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
    assert np.allclose(positions[0], np.array([0.0, 0.0, 0.0]))


def test_report_excludes_unverified_sources_without_runtime_exclusion() -> None:
    """Report estimates should be able to hide tentative sources only in output."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        cluster_min_samples=1,
        use_clustered_output=True,
        pseudo_source_quarantine_excludes_runtime=False,
        report_exclude_unverified_sources=True,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 0], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 0], dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    runtime_state = filt.state_without_quarantined_sources(state)
    positions, strengths = filt.estimate_clustered()

    assert runtime_state.num_sources == 2
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
    assert np.allclose(positions[0], np.array([0.0, 0.0, 0.0]))


def test_soft_quarantine_remains_visible_to_runtime_by_default() -> None:
    """Soft-quarantined sources should remain available to runtime PF operations."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        cluster_min_samples=1,
        use_clustered_output=True,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 2], dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    positions, strengths = filt.estimate_clustered()

    assert positions.shape == (2, 3)
    assert strengths.shape == (2,)
    assert filt._active_source_mask(state, include_quarantined=False).tolist() == [
        True,
        True,
    ]


def test_birth_response_counts_include_quarantined_sources() -> None:
    """Residual birth response columns should include soft-quarantined sources."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        pseudo_source_quarantine_excludes_runtime=True,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.array([3, 3], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.array([False, True], dtype=bool),
        verification_fail_streaks=np.array([0, 2], dtype=int),
    )
    data = MeasurementData(
        z_k=np.array([5.0], dtype=float),
        observation_variances=np.array([5.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([7], dtype=int),
        pb_indices=np.array([7], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    counts = filt._birth_existing_unit_response_counts_for_state(state, data)

    assert counts.shape == (1, 2)


def test_planning_particles_include_quarantined_sources() -> None:
    """Planning subsets should retain soft-quarantined sources for separation."""
    isotopes = ["Cs-137"]
    candidate_sources = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    normals = np.array([[1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5}
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        pseudo_source_quarantine_excludes_runtime=True,
        planning_particles=1,
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
    filt = estimator.filters["Cs-137"]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0, 50.0], dtype=float),
                background=0.0,
                ages=np.array([3, 3], dtype=int),
                low_q_streaks=np.zeros(2, dtype=int),
                support_scores=np.zeros(2, dtype=float),
                tentative_sources=np.array([False, True], dtype=bool),
                verification_fail_streaks=np.array([0, 2], dtype=int),
            ),
            log_weight=0.0,
        )
    ]

    subsets = estimator.planning_particles(max_particles=1)
    states, _weights = subsets["Cs-137"]

    assert len(states) == 1
    assert int(states[0].num_sources) == 2


def test_convergence_requires_min_stations() -> None:
    """Convergence freeze should require the configured number of stations."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        converge_enable=True,
        converge_min_stations=2,
    )
    filt._cardinality_variance = lambda: 0.0  # type: ignore[method-assign]
    filt._has_unverified_sources = lambda: False  # type: ignore[method-assign]
    filt._cluster_convergence_supported = lambda: True  # type: ignore[method-assign]

    filt._observed_station_labels = {(0.0, 0.0)}
    assert filt._convergence_can_freeze() is False

    filt._observed_station_labels.add((1.0, 0.0))
    assert filt._convergence_can_freeze() is True


def test_connected_position_clusters_preserve_transitive_components() -> None:
    """Vectorized clustering should keep the same transitive eps-neighborhoods."""
    from scipy.spatial import cKDTree

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
    )

    assert [set(cluster.tolist()) for cluster in clusters] == [{0, 1, 2}]


def test_connected_position_clusters_handles_dense_component_without_pair_matrix() -> None:
    """Dense report clusters should not require materializing all neighbor pairs."""
    from scipy.spatial import cKDTree

    positions = np.zeros((2000, 3), dtype=float)
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
    )

    assert len(clusters) == 1
    assert clusters[0].size == positions.shape[0]


def test_connected_position_clusters_uses_large_point_fallback() -> None:
    """Large report clusters should use a bounded grid fallback."""
    from scipy.spatial import cKDTree

    positions = np.zeros((6000, 3), dtype=float)
    clusters = IsotopeParticleFilter._connected_position_clusters(
        cKDTree(positions),
        point_count=positions.shape[0],
        eps=0.5,
        min_samples=2,
        exact_max_points=5000,
    )

    assert len(clusters) == 1
    assert clusters[0].size == positions.shape[0]


def test_cardinality_preserving_resample_keeps_source_count_mass() -> None:
    """Resampling should not erase a low-mass source-count hypothesis."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=100,
        cardinality_preserving_resample=True,
    )
    filt.config.resample_threshold = 2.0
    particles: list[IsotopeParticle] = []
    log_weights: list[float] = []
    for idx in range(95):
        state = IsotopeState(
            num_sources=1,
            positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        )
        particles.append(IsotopeParticle(state=state, log_weight=float(np.log(0.95 / 95.0))))
        log_weights.append(float(np.log(0.95 / 95.0)))
    for idx in range(5):
        state = IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=float,
            ),
            strengths=np.array([100.0, 80.0, 60.0], dtype=float),
            background=0.0,
        )
        particles.append(IsotopeParticle(state=state, log_weight=float(np.log(0.05 / 5.0))))
        log_weights.append(float(np.log(0.05 / 5.0)))
    filt.continuous_particles = particles

    np.random.seed(123)
    filt._maybe_resample_continuous(disable_regularize=True)

    labels = np.array([p.state.num_sources for p in filt.continuous_particles])
    weights = filt.continuous_weights
    mass_k1 = float(np.sum(weights[labels == 1]))
    mass_k3 = float(np.sum(weights[labels == 3]))
    assert np.count_nonzero(labels == 3) > 0
    assert np.isclose(mass_k1, 0.95)
    assert np.isclose(mass_k3, 0.05)


def test_cardinality_preserving_resample_waits_for_min_stations() -> None:
    """Cardinality-preserving resampling should stay off during early exploration."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=20,
        cardinality_preserving_resample=True,
        cardinality_preserving_min_stations=2,
        cardinality_preserving_require_confirmed_structure=False,
    )
    particles: list[IsotopeParticle] = []
    for idx in range(18):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=1,
                    positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                    strengths=np.array([100.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    for _idx in range(2):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=3,
                    positions=np.array(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        dtype=float,
                    ),
                    strengths=np.array([100.0, 80.0, 60.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    filt.continuous_particles = particles
    weights = np.array([0.9 / 18.0] * 18 + [0.05, 0.05], dtype=float)

    assert filt._cardinality_preserving_resample_draw(weights) is None

    filt._observed_station_labels = {(0.0, 0.0), (1.0, 0.0)}
    assert filt._cardinality_preserving_resample_draw(weights) is not None


def test_cardinality_preserving_resample_keeps_protected_spatial_modes() -> None:
    """Cardinality-preserving draws should still retain protected spatial modes."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=20,
        cardinality_preserving_resample=True,
        mode_preserving_resample=True,
        mode_preserving_max_modes=3,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=0.5,
        mode_preserving_min_weight_fraction=0.0,
    )
    particles: list[IsotopeParticle] = []
    for idx in range(18):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=1,
                    positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                    strengths=np.array([100.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    for idx, x_pos in enumerate((100.0, 200.0)):
        particles.append(
            IsotopeParticle(
                state=IsotopeState(
                    num_sources=3,
                    positions=np.array(
                        [
                            [x_pos, 0.0, 0.0],
                            [x_pos, 1.0, 0.0],
                            [x_pos, 2.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    strengths=np.array([100.0, 80.0, 60.0], dtype=float),
                    background=0.0,
                ),
                log_weight=0.0,
            )
        )
    filt.continuous_particles = particles
    weights = np.array([0.9 / 18.0] * 18 + [0.0999, 0.0001], dtype=float)
    protected = np.array([18, 19], dtype=np.int64)

    np.random.seed(123)
    draw = filt._cardinality_preserving_resample_draw(
        weights,
        protected_indices=protected,
    )

    assert draw is not None
    indices, _ = draw
    labels = np.array([particles[int(idx)].state.num_sources for idx in indices])
    assert np.count_nonzero(labels == 3) == 2
    assert {18, 19}.issubset(set(indices.tolist()))
    assert filt.last_mode_preserved_count >= 1


def test_clustered_estimate_downsamples_report_points() -> None:
    """Report clustering should remain bounded without changing PF particles."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=10,
        cluster_min_samples=1,
        cluster_report_max_points=5,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[float(i), 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0 + i], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
        for i in range(10)
    ]
    positions, strengths = filt.estimate_clustered()

    assert len(filt.continuous_particles) == 10
    assert positions.shape[0] <= filt.config.max_sources
    assert strengths.size == positions.shape[0]


def test_convergence_does_not_skip_unverified_multisource_state() -> None:
    """Convergence gating should not freeze an unresolved tentative source."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=1,
        converge_enable=True,
        converge_freeze_updates=True,
        converge_require_no_tentative=True,
    )
    state = filt.continuous_particles[0].state
    state.num_sources = 2
    state.positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    state.strengths = np.array([100.0, 50.0], dtype=float)
    state.ages = np.array([10, 1], dtype=int)
    state.low_q_streaks = np.zeros(2, dtype=int)
    state.support_scores = np.zeros(2, dtype=float)
    state.tentative_sources = np.array([False, True], dtype=bool)
    state.verification_fail_streaks = np.zeros(2, dtype=int)
    filt.is_converged = True
    filt.frozen_estimate = (state.positions[:1].copy(), state.strengths[:1].copy())

    assert not filt._should_skip_converged_update()
    assert not filt.is_converged


def test_convergence_monitoring_does_not_freeze_updates_by_default() -> None:
    """Convergence checks should not discard later observations by default."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        converge_enable=True,
    )
    filt.is_converged = True

    assert not filt._should_skip_converged_update()
    assert filt.is_converged


def test_convergence_freeze_updates_requires_explicit_opt_in() -> None:
    """Legacy update freezing remains available only when explicitly enabled."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        converge_enable=True,
        converge_freeze_updates=True,
    )
    filt.is_converged = True

    assert filt._should_skip_converged_update()


def test_convergence_requires_compact_supported_clusters() -> None:
    """Cluster-level convergence should reject a spatially diffuse output cluster."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        converge_enable=True,
        cluster_eps_m=1.0,
        cluster_min_samples=1,
        converge_cluster_spread_max_m=0.1,
        converge_cluster_min_support_fraction=0.05,
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=float,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=positions[idx : idx + 1],
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.25)),
        )
        for idx in range(4)
    ]

    assert not filt._cluster_convergence_supported()
    filt.config.converge_cluster_spread_max_m = 1.0
    assert filt._cluster_convergence_supported()


def test_pre_finalize_guard_preserves_reported_cardinality() -> None:
    """Reported estimates should prefer pre-finalize modes after collapse."""
    isotope = "Cs-137"
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=3,
        birth_enable=True,
        report_pre_finalize_guard=True,
        report_strength_refit=False,
        use_gpu=False,
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=np.zeros((1, 3), dtype=float),
        shield_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=config,
        shield_params=ShieldParams(thickness_pb_cm=0.0, thickness_fe_cm=0.0),
    )
    guard_pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    guard_q = np.array([100.0, 50.0], dtype=float)
    estimator._pre_finalize_guard_estimates[isotope] = (guard_pos, guard_q)

    positions, strengths = estimator._guarded_report_estimate(
        isotope,
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.array([150.0], dtype=float),
        use_pre_finalize_guard=True,
    )

    assert positions.shape == (2, 3)
    assert strengths.shape == (2,)


def test_residual_birth_always_try_avoids_stochastic_miss(monkeypatch) -> None:
    """Residual-gated birth should not be skipped by the proposal probability."""
    filt = _build_filter(
        p_birth=0.01,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_residual_always_try=True,
        birth_matching_pursuit_max_new_sources=2,
        birth_matching_pursuit_topk_candidates=3,
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
        split_prob=0.0,
        merge_prob=0.0,
        conditional_strength_refit_prior_weight=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
                ages=np.array([3], dtype=int),
                low_q_streaks=np.zeros(1, dtype=int),
                support_scores=np.zeros(1, dtype=float),
            ),
            log_weight=0.0,
        )
    ]
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
    data = MeasurementData(
        z_k=np.sum(expected, axis=1),
        observation_variances=np.maximum(np.sum(expected, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    monkeypatch.setattr(np.random, "rand", lambda *args: 0.99)

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=true_positions,
    )

    state = filt.continuous_particles[0].state
    assert filt.last_birth_count > 0
    assert state.num_sources == 2


def test_residual_birth_expands_beyond_topk_structural_particles(monkeypatch) -> None:
    """Residual-gated birth should not be limited to collapsed top-weight particles."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        structural_proposal_topk_particles=1,
        birth_residual_expand_structural_particles=True,
        birth_residual_always_try=True,
        birth_matching_pursuit_max_new_sources=2,
        refit_after_moves=False,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.99)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(np.log(0.01)),
        ),
    ]
    data = MeasurementData(
        z_k=np.array([20.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return a residual-gated proposal independent of top-k particles."""
        filt.last_birth_residual_gate_passed = True
        filt.last_birth_residual_refit_gate_passed = True
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
            20.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

    def _matching_pursuit(
        st: IsotopeState,
        birth_data: MeasurementData,
        candidate_positions: np.ndarray,
        *,
        max_new_sources: int,
        residual_gate_forced: bool = False,
        candidate_unit_counts: np.ndarray | None = None,
    ) -> int:
        """Accept a birth only for the non-top empty particle."""
        assert residual_gate_forced
        assert candidate_unit_counts is None
        if st.num_sources > 0:
            return 0
        st.positions = np.array([[2.0, 0.0, 0.0]], dtype=float)
        st.strengths = np.array([100.0], dtype=float)
        st.ages = np.array([0], dtype=int)
        st.low_q_streaks = np.array([0], dtype=int)
        st.support_scores = np.array([0.0], dtype=float)
        st.num_sources = 1
        return 1

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)
    monkeypatch.setattr(
        filt,
        "_apply_matching_pursuit_births_to_state",
        _matching_pursuit,
    )
    monkeypatch.setattr(
        filt,
        "refresh_weights_from_measurements",
        lambda data, **kwargs: None,
    )

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert filt.last_birth_count == 1
    assert filt.last_birth_structural_eligible == 1
    assert filt.continuous_particles[1].state.num_sources == 1


def test_residual_birth_expansion_is_capped_and_cardinality_diverse(monkeypatch) -> None:
    """Residual-gated structural expansion should not evaluate every particle."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=5,
        structural_proposal_topk_particles=1,
        birth_residual_expand_structural_particles=True,
        birth_residual_expanded_structural_topk_particles=2,
        birth_residual_always_try=True,
        split_prob=0.0,
        split_residual_guided=False,
        merge_prob=0.0,
        refit_after_moves=False,
    )
    filt.continuous_particles = []
    weights = np.array([0.90, 0.05, 0.03, 0.01, 0.01], dtype=float)
    for idx, weight in enumerate(weights):
        if idx == 0:
            state = IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=float(idx),
            )
        else:
            state = IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=float(idx),
            )
        filt.continuous_particles.append(
            IsotopeParticle(state=state, log_weight=float(np.log(weight)))
        )
    data = MeasurementData(
        z_k=np.array([20.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return a residual-gated proposal for structural expansion."""
        filt.last_birth_residual_gate_passed = True
        filt.last_birth_residual_refit_gate_passed = True
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
            20.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

    attempted: list[int] = []

    def _matching_pursuit(
        st: IsotopeState,
        birth_data: MeasurementData,
        candidate_positions: np.ndarray,
        *,
        max_new_sources: int,
        residual_gate_forced: bool = False,
        candidate_unit_counts: np.ndarray | None = None,
    ) -> int:
        """Record particles that receive exact structural birth evaluation."""
        assert residual_gate_forced
        assert candidate_unit_counts is None
        attempted.append(int(st.background))
        return 0

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)
    monkeypatch.setattr(
        filt,
        "_apply_matching_pursuit_births_to_state",
        _matching_pursuit,
    )

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert set(attempted) == {0, 1}


def test_residual_birth_gate_suppresses_same_update_death(monkeypatch) -> None:
    """Residual birth evidence should delay death in the same structural update."""
    filt = _build_filter(
        p_birth=1.0,
        p_kill=1.0,
        min_strength=5.0,
        max_sources=1,
        num_particles=1,
        death_low_q_streak=1,
        death_delta_ll_threshold=1.0e9,
        support_ema_alpha=1.0,
        birth_residual_always_try=True,
        birth_residual_suppress_death=True,
        refit_after_moves=False,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
                ages=np.array([5], dtype=int),
                low_q_streaks=np.array([1], dtype=int),
                support_scores=np.array([-1.0], dtype=float),
            ),
            log_weight=0.0,
        )
    ]
    data = MeasurementData(
        z_k=np.array([50.0], dtype=float),
        observation_variances=np.array([1.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def _proposal(
        birth_data: MeasurementData | None,
        candidate_positions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return a gate-passing proposal so death is delayed."""
        filt.last_birth_residual_gate_passed = True
        filt.last_birth_residual_refit_gate_passed = True
        return (
            np.array([1.0], dtype=float),
            np.array([1.0], dtype=float),
            50.0,
            np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

    monkeypatch.setattr(filt, "_compute_birth_proposal", _proposal)

    filt.apply_birth_death(
        support_data=data,
        birth_data=data,
        candidate_positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert filt.last_kill_count == 0
    assert filt.continuous_particles[0].state.num_sources == 1


def test_residual_gate_suppresses_refit_floor_prune() -> None:
    """Residual-gated structural moves should delay weak-source refit pruning."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=3,
        num_particles=1,
        weak_source_prune_min_expected_count=3.0,
        weak_source_prune_min_fraction=0.0,
        weak_source_prune_min_age=1,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=float),
        strengths=np.array([5.0, 5.0], dtype=float),
        background=0.0,
        ages=np.array([5, 5], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
    )
    data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    filt._prune_floor_sources_after_refit(
        state,
        data,
        suppress_prune=True,
    )

    assert filt.last_kill_count == 0
    assert state.num_sources == 2


def test_residual_gate_scales_birth_complexity_penalty() -> None:
    """Residual-gated matching pursuit should avoid double complexity charging."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_candidate_support_fraction=0.0,
        birth_complexity_penalty=1.0e12,
        birth_residual_acceptance_complexity_scale=0.0,
        birth_min_sep_m=0.4,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    true_position = np.array([[0.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=true_position,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        true_position,
        max_new_sources=1,
        residual_gate_forced=True,
    )

    assert accepted == 1
    assert state.num_sources == 1


def test_residual_forced_birth_relaxes_empty_candidate_masks(monkeypatch) -> None:
    """Forced residual birth should not be blocked solely by proposal masks."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        birth_residual_force_proposal_on_gate=True,
        birth_residual_force_relax_candidate_masks=True,
        birth_residual_forced_min_delta_ll=-50.0,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_candidate_support_fraction=0.0,
        birth_matching_pursuit_topk_candidates=1,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    monkeypatch.setattr(
        filt,
        "_birth_candidate_support_mask",
        lambda *, candidate_counts, **_kwargs: np.zeros(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_existing_response_correlation_mask",
        lambda *, candidate_counts, **_kwargs: np.zeros(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        filt,
        "_birth_response_condition_mask",
        lambda *, candidate_counts, **_kwargs: np.zeros(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    source = np.array([[0.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=source,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        source,
        max_new_sources=1,
        residual_gate_forced=True,
    )

    assert accepted == 1
    assert state.num_sources == 1
    assert filt.last_birth_forced_mask_relaxations == 1
    assert filt.last_birth_forced_accepts == 1


def test_non_forced_birth_keeps_candidate_masks_strict(monkeypatch) -> None:
    """Non-forced matching pursuit should keep proposal masks as hard filters."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=1,
        birth_residual_force_relax_candidate_masks=True,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.1,
        birth_candidate_support_fraction=0.0,
        weak_source_prune_min_expected_count=0.0,
        weak_source_prune_min_fraction=0.0,
    )
    monkeypatch.setattr(
        filt,
        "_birth_candidate_support_mask",
        lambda *, candidate_counts, **_kwargs: np.zeros(
            np.asarray(candidate_counts).shape[1],
            dtype=bool,
        ),
    )
    state = IsotopeState(
        num_sources=0,
        positions=np.zeros((0, 3), dtype=float),
        strengths=np.zeros(0, dtype=float),
        background=0.0,
    )
    source = np.array([[0.0, 0.0, 0.0]], dtype=float)
    detector_positions = np.array([[0.0, 1.0, 0.0]], dtype=float)
    expected = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=source,
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(1, dtype=float),
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        source_scale=1.0,
    )
    counts = np.sum(expected, axis=1)
    data = MeasurementData(
        z_k=counts,
        observation_variances=np.maximum(counts, 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        live_times=np.ones(1, dtype=float),
    )

    accepted = filt._apply_matching_pursuit_births_to_state(
        state,
        data,
        source,
        max_new_sources=1,
        residual_gate_forced=False,
    )

    assert accepted == 0
    assert state.num_sources == 0


def test_birth_bic_penalty_survives_residual_gate_scaling() -> None:
    """Residual-forced births should still pay a BIC model-order penalty."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_complexity_penalty=1.0e12,
        birth_residual_acceptance_complexity_scale=0.0,
        birth_bic_penalty_params=4,
    )

    penalty = filt._birth_complexity_penalty(
        residual_gate_forced=True,
        measurement_count=16,
    )

    assert penalty == np.log(16.0) * 2.0


def test_resampling_can_protect_distinct_low_weight_source_modes() -> None:
    """Mode-preserving resampling should retain spatially distinct source modes."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        mode_preserving_resample=True,
        mode_preserving_max_modes=3,
        mode_preserving_particles_per_mode=1,
        mode_preserving_radius_m=0.5,
        mode_preserving_min_weight_fraction=0.0,
    )
    positions = [
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.array([[0.1, 0.0, 0.0]], dtype=float),
        np.array([[2.0, 0.0, 0.0]], dtype=float),
        np.array([[4.0, 0.0, 0.0]], dtype=float),
    ]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=pos,
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
        for pos in positions
    ]
    weights = np.array([0.94, 0.05, 0.006, 0.004], dtype=float)

    protected = filt._source_mode_preserving_indices(weights)
    injected = filt._inject_mode_preserving_indices(
        np.array([0, 0, 0, 1], dtype=np.int64),
        protected,
    )

    assert {0, 2, 3}.issubset(set(protected.tolist()))
    assert 2 in injected
    assert 3 in injected
    assert filt.last_mode_preserved_count == 2


def test_structural_weight_refresh_preserves_prior_history(monkeypatch) -> None:
    """Moved-particle refresh should apply a likelihood ratio, not reset weights."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=1.0,
            ),
            log_weight=float(np.log(0.9)),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.5,
            ),
            log_weight=float(np.log(0.1)),
        ),
    ]
    data = MeasurementData(
        z_k=np.array([10.0], dtype=float),
        observation_variances=np.array([10.0], dtype=float),
        detector_positions=np.array([[0.5, 0.0, 0.0]], dtype=float),
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times=np.array([1.0], dtype=float),
    )

    def fake_lambda_components(
        state: IsotopeState,
        _data: MeasurementData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a deterministic one-bin expected count from background."""
        return np.zeros((1, 0), dtype=float), np.array([10.0 * state.background], dtype=float)

    monkeypatch.setattr(filt, "_lambda_components", fake_lambda_components)
    old_ll = filt._count_log_likelihood_np(
        data.z_k,
        np.array([10.0], dtype=float),
        observation_count_variance=data.observation_variances,
    )

    filt.refresh_weights_from_measurements(
        data,
        reference_log_likelihood_by_index={0: old_ll},
        moved_indices={0},
    )

    weights = filt.continuous_weights
    assert np.allclose(weights, np.array([0.9, 0.1], dtype=float))


def test_structural_update_resamples_after_weight_collapse() -> None:
    """Delayed structural updates should not carry collapsed weights forward."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        deferred_resample_roughening_scale=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(log_weight),
        )
        for log_weight in np.log(np.array([0.997, 0.001, 0.001, 0.001], dtype=float))
    ]

    resampled = filt._maybe_resample_after_structural_update()

    assert resampled
    assert filt.last_resample_ess
    assert filt.last_ess_post == float(filt.N)
    assert np.allclose(filt.continuous_weights, np.full(filt.N, 1.0 / filt.N))


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
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
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


def test_weak_source_prune_respects_min_age() -> None:
    """Weak-source pruning should not delete newly proposed components immediately."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=3,
        num_particles=1,
        weak_source_prune_min_expected_count=3.0,
        weak_source_prune_min_fraction=0.0,
        weak_source_prune_min_age=2,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([5.0, 5.0], dtype=float),
        background=0.0,
        ages=np.array([0, 5], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
    )

    filt._prune_floor_sources_by_expected_counts(
        state,
        np.array([0.0, 0.0], dtype=float),
    )

    assert state.num_sources == 1
    assert np.allclose(state.positions[0], [0.0, 0.0, 0.0])


def test_batched_refit_respects_suppressed_weak_source_prune() -> None:
    """Batched strength refit should defer weak-source deletion when requested."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=5.0,
        max_sources=2,
        num_particles=1,
        weak_source_prune_min_expected_count=10.0,
        weak_source_prune_min_fraction=0.0,
        weak_source_prune_min_age=0,
        source_prune_refit_after_remove=False,
        source_prune_delta_ll_threshold=1.0,
        source_prune_min_distinct_stations=1,
        source_prune_min_distinct_views=1,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 5.0], dtype=float),
        background=0.0,
        ages=np.array([5, 5], dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]
    detector_positions = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    counts = expected_counts_per_source(
        kernel=filt.continuous_kernel,
        isotope=filt.isotope,
        detector_positions=detector_positions,
        sources=state.positions[:1],
        strengths=np.array([100.0], dtype=float),
        live_times=np.ones(2, dtype=float),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        source_scale=1.0,
    )
    data = MeasurementData(
        z_k=np.sum(counts, axis=1),
        observation_variances=np.maximum(np.sum(counts, axis=1), 1.0),
        detector_positions=detector_positions,
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )

    filt._refit_particle_indices_batched(
        data,
        [0],
        iters=3,
        eps=1.0e-12,
        suppress_prune_after_refit=True,
    )

    assert filt.continuous_particles[0].state.num_sources == 2
    assert filt.last_kill_count == 0


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
    assert any(p.state.num_sources > 0 for p in filt.continuous_particles)
    assert all(p.state.num_sources <= 2 for p in filt.continuous_particles)


def test_birth_proposal_skipped_when_all_particles_at_max_sources(monkeypatch) -> None:
    """Residual birth proposal should not run when no particle can accept birth."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        birth_enable=True,
        birth_residual_always_try=True,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
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

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("birth proposal should not be computed")

    monkeypatch.setattr(filt, "_compute_birth_proposal", fail_if_called)

    filt.apply_birth_death(
        support_data=None,
        birth_data=birth_data,
        candidate_positions=filt.kernel.sources,
    )

    assert filt.last_birth_count == 0
    assert all(p.state.num_sources == 1 for p in filt.continuous_particles)


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

    def _record_refresh(data: MeasurementData | None, **kwargs: object) -> None:
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


def test_refresh_weights_reuses_cached_lambda_totals(monkeypatch) -> None:
    """Cached exact expected counts should avoid duplicate kernel evaluation."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=2,
        use_gpu=False,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([20.0], dtype=float),
            background=0.1,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([45.0], dtype=float),
            background=0.2,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=np.log(0.5))
        for state in states
    ]
    filt.N = len(filt.continuous_particles)
    data = MeasurementData(
        z_k=np.array([12.0, 7.0], dtype=float),
        observation_variances=np.array([13.0, 8.0], dtype=float),
        detector_positions=np.array(
            [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.zeros(2, dtype=int),
        pb_indices=np.zeros(2, dtype=int),
        live_times=np.ones(2, dtype=float),
    )
    cached_lambdas = {}
    scalar_ll = []
    for idx, particle in enumerate(filt.continuous_particles):
        _, lambda_total = filt._lambda_components(particle.state, data)
        cached_lambdas[int(idx)] = lambda_total.copy()
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

    def _fail_lambda_group(*args: object, **kwargs: object) -> None:
        """Fail if cached refresh recomputes grouped lambdas."""
        raise AssertionError("unexpected grouped lambda recomputation")

    def _fail_lambda_scalar(*args: object, **kwargs: object) -> None:
        """Fail if cached refresh recomputes scalar lambdas."""
        raise AssertionError("unexpected scalar lambda recomputation")

    monkeypatch.setattr(
        filt,
        "_lambda_components_for_particle_group",
        _fail_lambda_group,
    )
    monkeypatch.setattr(filt, "_lambda_components", _fail_lambda_scalar)

    filt.refresh_weights_from_measurements(
        data,
        lambda_total_by_index=cached_lambdas,
    )

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


def test_conditional_strength_refit_resamples_after_reweight_collapse(monkeypatch) -> None:
    """Profile-likelihood refit should not leave a degenerate station posterior."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=1,
        num_particles=4,
        conditional_strength_refit=True,
        conditional_strength_refit_reweight=True,
        conditional_strength_refit_reweight_clip=100.0,
        deferred_resample_roughening_scale=0.0,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[float(idx), 0.0, 0.0]], dtype=float),
                strengths=np.array([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=float(-np.log(4.0)),
        )
        for idx in range(4)
    ]
    data = MeasurementData(
        z_k=np.array([100.0], dtype=float),
        observation_variances=np.array([100.0], dtype=float),
        detector_positions=np.array([[0.0, 1.0, 0.0]], dtype=float),
        fe_indices=np.zeros(1, dtype=int),
        pb_indices=np.zeros(1, dtype=int),
        live_times=np.ones(1, dtype=float),
    )

    def _collapse_weights(
        *_args: object,
        particle_indices: list[int],
        **_kwargs: object,
    ) -> np.ndarray:
        """Return deterministic corrections that collapse ESS."""
        corrections = np.full(len(particle_indices), -50.0, dtype=float)
        first = particle_indices.index(0)
        corrections[first] = 0.0
        return corrections

    monkeypatch.setattr(
        filt,
        "_refit_fixed_source_count_particles_batched",
        _collapse_weights,
    )

    filt.refit_strengths_for_particles(data, iters=1)

    assert filt.last_resample_ess
    assert np.allclose(filt.continuous_weights, np.full(4, 0.25))


def test_conditional_strength_refit_reweight_preserves_cardinality_mass() -> None:
    """Strength-refit reweighting should not act as a model-order selector."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=4,
        conditional_strength_refit_cardinality_neutral_reweight=True,
    )
    states = [
        IsotopeState(
            num_sources=1,
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=1,
            positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
            strengths=np.array([100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=float,
            ),
            strengths=np.array([100.0, 100.0, 100.0], dtype=float),
            background=0.0,
        ),
        IsotopeState(
            num_sources=3,
            positions=np.array(
                [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
                dtype=float,
            ),
            strengths=np.array([100.0, 100.0, 100.0], dtype=float),
            background=0.0,
        ),
    ]
    filt.continuous_particles = [
        IsotopeParticle(state=state, log_weight=float(-np.log(4.0)))
        for state in states
    ]
    corrections = np.array([20.0, 10.0, 0.0, -5.0], dtype=float)

    adjusted = filt._cardinality_neutral_refit_corrections(corrections)
    logw = np.array([particle.log_weight for particle in filt.continuous_particles])
    counts = np.array([particle.state.num_sources for particle in filt.continuous_particles])

    for source_count in (1, 3):
        mask = counts == source_count
        old_mass = np.sum(np.exp(logw[mask]))
        new_mass = np.sum(np.exp(logw[mask] + adjusted[mask]))
        assert np.isclose(new_mass, old_mass)


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


def test_birth_refit_residual_gate_batched_matches_scalar_oracle() -> None:
    """Batched residual-refit gate should match the scalar particle loop."""
    filt = _build_filter(
        p_birth=1.0,
        min_strength=0.01,
        max_sources=3,
        num_particles=3,
        birth_refit_residual_gate=True,
        birth_residual_min_support=1,
        birth_residual_support_sigma=0.25,
        birth_residual_gate_p_value=1.0,
        birth_refit_residual_min_fraction=0.1,
        refit_iters=4,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([10.0], dtype=float),
                background=0.2,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=np.array(
                    [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]],
                    dtype=float,
                ),
                strengths=np.array([8.0, 4.0], dtype=float),
                background=0.1,
            ),
            log_weight=np.log(0.3),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.5,
            ),
            log_weight=np.log(0.2),
        ),
    ]
    detector_positions = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    data = MeasurementData(
        z_k=np.array([80.0, 35.0, 20.0], dtype=float),
        observation_variances=np.array([80.0, 35.0, 20.0], dtype=float),
        detector_positions=detector_positions,
        fe_indices=np.zeros(3, dtype=int),
        pb_indices=np.zeros(3, dtype=int),
        live_times=np.ones(3, dtype=float),
    )
    particle_indices = np.array([0, 1, 2], dtype=int)
    particle_weights = np.array([5.0, 3.0, 2.0], dtype=float)

    scalar = filt._birth_residual_survives_strength_refit_scalar(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
        residual_sum_before=100.0,
    )
    scalar_fraction = filt.last_birth_residual_refit_fraction
    scalar_support = filt.last_birth_residual_support
    scalar_distinct = filt.last_birth_residual_distinct_poses
    scalar_stations = filt.last_birth_residual_distinct_stations

    batched = filt._birth_residual_survives_strength_refit_batched(
        data=data,
        particle_indices=particle_indices,
        particle_weights=particle_weights,
        residual_sum_before=100.0,
    )

    assert batched is scalar
    assert np.isclose(
        filt.last_birth_residual_refit_fraction,
        scalar_fraction,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert filt.last_birth_residual_support == scalar_support
    assert filt.last_birth_residual_distinct_poses == scalar_distinct
    assert filt.last_birth_residual_distinct_stations == scalar_stations


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
    _, _, _, ranked_candidates, _ = proposal
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


def test_report_strength_refit_can_preserve_posterior_cardinality() -> None:
    """Reported strength refit can keep PF clusters when regression is collinear."""
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
        report_strength_refit_preserve_cardinality=True,
        report_cluster_model_selection=False,
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
    design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=reported_positions[:1],
        strengths=np.array([90.0], dtype=float),
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

    assert positions.shape == (2, 3)
    assert strengths.shape == (2,)


def test_report_strength_refit_prior_regularizes_large_strength() -> None:
    """Report-level strength refit should support MAP regularization."""
    isotope = "Cs-137"
    source = np.array([[0.0, 0.0, 0.0]], dtype=float)
    estimator = RotatingShieldPFEstimator(
        isotopes=[isotope],
        candidate_sources=source,
        shield_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            report_strength_refit_iters=64,
            report_strength_refit_prior_weight=0.0,
            report_strength_refit_prior_rel_sigma=0.1,
            use_gpu=False,
        ),
        shield_params=ShieldParams(thickness_pb_cm=0.0, thickness_fe_cm=0.0),
    )
    design = np.ones((2, 1), dtype=float)
    z_obs = np.full(2, 1000.0, dtype=float)
    background = np.zeros(2, dtype=float)
    variances = np.ones(2, dtype=float)
    initial = np.array([100.0], dtype=float)

    q_unregularized = estimator._solve_report_strengths(
        design=design,
        z_obs=z_obs,
        background=background,
        observation_variances=variances,
        initial_strengths=initial,
        eps=1.0e-9,
        q_max=0.0,
    )
    estimator.pf_config.report_strength_refit_prior_weight = 1000.0
    q_regularized = estimator._solve_report_strengths(
        design=design,
        z_obs=z_obs,
        background=background,
        observation_variances=variances,
        initial_strengths=initial,
        eps=1.0e-9,
        q_max=0.0,
    )

    assert q_unregularized[0] > 900.0
    assert 100.0 <= q_regularized[0] < q_unregularized[0] * 0.5


def test_report_model_order_cluster_prune_updates_particle_source_slots() -> None:
    """BIC-dropped report clusters should be removable from PF source slots."""
    filt = _build_filter(
        p_birth=0.0,
        min_strength=0.01,
        max_sources=2,
        num_particles=1,
        birth_enable=True,
        use_gpu=False,
    )
    state = IsotopeState(
        num_sources=2,
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
        strengths=np.array([100.0, 50.0], dtype=float),
        background=0.0,
        ages=np.ones(2, dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.zeros(2, dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    removed = filt.apply_report_model_order_cluster_prune(
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
        np.array([True, False], dtype=bool),
        radius_m=0.5,
    )

    assert removed == 1
    assert state.num_sources == 1
    assert np.allclose(state.positions[0], np.array([0.0, 0.0, 0.0]))


def test_report_model_order_prune_particles_applies_after_bic_selection() -> None:
    """Estimator report refit should optionally push BIC drops back into PF state."""
    isotope = "Cs-137"
    reported_positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    detector_positions = [
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 2.0, 0.0], dtype=float),
        np.array([2.0, 2.0, 0.0], dtype=float),
    ]
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        birth_enable=True,
        use_clustered_output=True,
        cluster_min_samples=1,
        report_strength_refit=True,
        report_strength_refit_preserve_cardinality=False,
        report_cluster_model_selection=True,
        report_strength_refit_iters=128,
        report_cluster_bic_penalty_params=8,
        report_model_order_prune_particles=True,
        report_model_order_particle_prune_radius_m=0.5,
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
    design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=reported_positions[:1],
        strengths=np.array([90.0], dtype=float),
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
    state = IsotopeState(
        num_sources=2,
        positions=reported_positions.copy(),
        strengths=np.array([40.0, 40.0], dtype=float),
        background=0.0,
        ages=np.ones(2, dtype=int),
        low_q_streaks=np.zeros(2, dtype=int),
        support_scores=np.zeros(2, dtype=float),
        tentative_sources=np.zeros(2, dtype=bool),
        verification_fail_streaks=np.zeros(2, dtype=int),
    )
    filt = estimator.filters[isotope]
    filt.continuous_particles = [IsotopeParticle(state=state, log_weight=0.0)]

    positions, strengths = estimator._refit_reported_strengths(
        isotope,
        reported_positions,
        np.array([40.0, 40.0], dtype=float),
    )

    diagnostics = estimator.report_model_order_diagnostics()[isotope]
    assert positions.shape == (1, 3)
    assert strengths.shape == (1,)
    assert state.num_sources == 1
    assert int(diagnostics["particle_pruned_source_slots"]) == 1


def test_report_strength_refit_preserve_overrides_bic_collapse() -> None:
    """Cardinality preservation should keep supported PF clusters after BIC scoring."""
    isotope = "Cs-137"
    reported_positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    detector_positions = [
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 2.0, 0.0], dtype=float),
        np.array([2.0, 2.0, 0.0], dtype=float),
    ]
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=2,
        birth_enable=True,
        use_clustered_output=True,
        cluster_min_samples=1,
        report_strength_refit=True,
        report_strength_refit_preserve_cardinality=True,
        report_cluster_model_selection=True,
        report_strength_refit_iters=128,
        report_cluster_bic_penalty_params=8,
        report_model_order_min_bic_margin=0.0,
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
    design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=reported_positions[:1],
        strengths=np.array([90.0], dtype=float),
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

    diagnostics = estimator.report_model_order_diagnostics()[isotope]
    assert int(diagnostics["model_selected_count"]) < 2
    assert int(diagnostics["selected_count"]) == 2
    assert diagnostics["model_order_ready"] is False
    assert diagnostics["model_order_overridden"] is True
    assert positions.shape == (2, 3)
    assert strengths.shape == (2,)


def test_pruned_estimates_respect_preserved_report_cardinality(monkeypatch) -> None:
    """Non-destructive output pruning must not undo preserved model order."""
    isotope = "Cs-137"
    reported_positions = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 4.0, 0.0]],
        dtype=float,
    )
    reported_strengths = np.array([1000.0, 80.0, 60.0], dtype=float)
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=3,
        birth_enable=True,
        report_strength_refit=False,
        report_strength_refit_preserve_cardinality=True,
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

    def fake_estimates() -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return a three-source estimate with preserved model-order diagnostics."""
        estimator._last_report_model_order_diagnostics[isotope] = {
            "candidate_count": 3,
            "selected_count": 3,
            "selected_indices": [0, 1, 2],
            "preserve_cardinality": True,
        }
        return {isotope: (reported_positions, reported_strengths)}

    def fake_prune(*_args: object, **_kwargs: object) -> dict[str, NDArray[np.bool_]]:
        """Return an over-aggressive keep mask to exercise the guard."""
        return {isotope: np.array([True, False, False], dtype=bool)}

    monkeypatch.setattr(estimator, "estimates", fake_estimates)
    monkeypatch.setattr(
        "pf.mixing.prune_spurious_sources_continuous",
        fake_prune,
    )

    pruned_positions, pruned_strengths = estimator.pruned_estimates()[isotope]

    assert pruned_positions.shape == (3, 3)
    assert pruned_strengths.shape == (3,)


def test_report_model_order_selection_keeps_supported_three_sources() -> None:
    """Report-level BIC selection should keep any supported source count."""
    isotope = "Cs-137"
    sources = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.5, 3.5, 0.0]],
        dtype=float,
    )
    detector_positions = [
        np.array([0.5, 1.0, 0.0], dtype=float),
        np.array([1.5, 1.0, 0.0], dtype=float),
        np.array([3.5, 1.0, 0.0], dtype=float),
        np.array([5.0, 1.5, 0.0], dtype=float),
        np.array([1.0, 4.5, 0.0], dtype=float),
        np.array([3.0, 4.0, 0.0], dtype=float),
    ]
    config = RotatingShieldPFConfig(
        num_particles=1,
        max_sources=3,
        birth_enable=True,
        use_clustered_output=True,
        cluster_min_samples=1,
        report_strength_refit=True,
        report_strength_refit_iters=128,
        report_cluster_model_selection=True,
        report_cluster_model_selection_max_candidates=12,
        report_cluster_bic_penalty_params=4,
        report_model_order_min_bic_margin=0.0,
        init_num_sources=(1, 1),
        min_strength=0.01,
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
    true_strengths = np.array([1200.0, 850.0, 650.0], dtype=float)
    design = expected_counts_per_source(
        kernel=estimator.filters[isotope].continuous_kernel,
        isotope=isotope,
        detector_positions=np.vstack(detector_positions),
        sources=sources,
        strengths=np.ones(3, dtype=float),
        live_times=np.ones(len(detector_positions), dtype=float),
        fe_indices=np.zeros(len(detector_positions), dtype=int),
        pb_indices=np.zeros(len(detector_positions), dtype=int),
    )
    counts = design @ true_strengths
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
        sources,
        np.array([500.0, 500.0, 500.0], dtype=float),
    )

    diagnostics = estimator.report_model_order_diagnostics()[isotope]
    order = np.argsort(positions[:, 0])
    assert positions.shape == (3, 3)
    assert diagnostics["selected_count"] == 3
    assert set(diagnostics["best_by_k"]) >= {"0", "1", "2", "3"}
    assert np.allclose(positions[order], sources[np.argsort(sources[:, 0])], atol=1.0e-6)
    assert np.all(strengths > 0.0)


def test_report_model_order_parallel_matches_serial_selection() -> None:
    """Parallel report-level subset scoring should match the serial oracle."""
    isotope = "Cs-137"
    sources = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.5, 3.5, 0.0]],
        dtype=float,
    )
    detector_positions = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [3.5, 1.0, 0.0],
            [5.0, 1.5, 0.0],
            [1.0, 4.5, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    true_strengths = np.array([1200.0, 850.0, 650.0], dtype=float)
    results: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            dict[str, object],
        ]
    ] = []
    for workers in (1, 4):
        config = RotatingShieldPFConfig(
            num_particles=1,
            max_sources=3,
            birth_enable=True,
            use_clustered_output=True,
            cluster_min_samples=1,
            report_strength_refit=True,
            report_strength_refit_iters=64,
            report_cluster_model_selection=True,
            report_cluster_model_selection_max_candidates=12,
            report_cluster_bic_penalty_params=4,
            report_model_order_min_bic_margin=0.0,
            report_model_order_workers=workers,
            report_model_order_parallel_min_subsets=1,
            init_num_sources=(1, 1),
            min_strength=0.01,
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
        design = expected_counts_per_source(
            kernel=estimator.filters[isotope].continuous_kernel,
            isotope=isotope,
            detector_positions=detector_positions,
            sources=sources,
            strengths=np.ones(3, dtype=float),
            live_times=np.ones(detector_positions.shape[0], dtype=float),
            fe_indices=np.zeros(detector_positions.shape[0], dtype=int),
            pb_indices=np.zeros(detector_positions.shape[0], dtype=int),
        )
        counts = design @ true_strengths
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
            sources,
            np.array([500.0, 500.0, 500.0], dtype=float),
        )
        diagnostics = estimator.report_model_order_diagnostics()[isotope]
        results.append((positions, strengths, diagnostics))

    serial_positions, serial_strengths, serial_diagnostics = results[0]
    parallel_positions, parallel_strengths, parallel_diagnostics = results[1]
    assert parallel_diagnostics["workers"] == 4
    assert serial_diagnostics["evaluation_mode"] == "serial"
    assert parallel_diagnostics["evaluation_mode"] == "batched_numpy"
    assert serial_diagnostics["selected_indices"] == parallel_diagnostics["selected_indices"]
    assert serial_diagnostics["selected_count"] == parallel_diagnostics["selected_count"]
    assert np.allclose(serial_positions, parallel_positions, atol=1.0e-9)
    assert np.allclose(serial_strengths, parallel_strengths, rtol=1.0e-9, atol=1.0e-9)


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
