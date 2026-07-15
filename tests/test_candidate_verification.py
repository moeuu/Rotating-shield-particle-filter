"""Tests for independent verification of queued PF rescue candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from measurement.kernels import ShieldParams
from pf.estimator import (
    MeasurementRecord,
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator,
)
from pf.particle_filter import IsotopeParticle, IsotopeParticleFilter, MeasurementData
from pf.state import IsotopeState

ISOTOPE = "Cs-137"


def _build_estimator(**config_overrides: Any) -> RotatingShieldPFEstimator:
    """Return a deterministic estimator with independent verification enabled."""
    config_values: dict[str, Any] = {
        "num_particles": 4,
        "max_sources": 3,
        "min_strength": 0.01,
        "use_gpu": False,
        "runtime_report_rescue_verification_queue_only": True,
        "runtime_report_rescue_memory_enable": False,
        "runtime_report_rescue_min_particles_per_source": 1,
        "runtime_report_rescue_weight": 0.10,
        "candidate_verification_queue_decay": 1.0,
        "candidate_verification_independent_evidence_enable": True,
        "candidate_verification_min_xy_separation_m": 0.5,
        "candidate_verification_min_height_separation_m": 0.5,
        "candidate_verification_min_deviance_improvement": 2.0,
        "candidate_verification_min_positive_checks": 1,
        "candidate_verification_reject_after_negatives": 2,
        "candidate_verification_negative_deviance_threshold": 0.0,
        "candidate_verification_profile_l2": 1.0e-8,
        "candidate_verification_profile_max_iters": 48,
        "report_mle_rescue_dedup_radius_m": 0.2,
        "report_best_so_far_enable": False,
    }
    config_values.update(config_overrides)
    estimator = RotatingShieldPFEstimator(
        isotopes=[ISOTOPE],
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        mu_by_isotope={ISOTOPE: 0.0},
        pf_config=RotatingShieldPFConfig(**config_values),
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
    )
    for pose in (
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 1.5],
        [2.0, 1.0, 0.5],
        [2.0, 1.0, 1.5],
        [3.0, 1.0, 2.0],
    ):
        estimator.add_measurement_pose(np.asarray(pose, dtype=float))
    estimator._ensure_kernel_cache()
    estimator.filters[ISOTOPE].continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=float(-np.log(4.0)),
        )
        for _ in range(4)
    ]
    return estimator


def _candidate_unit_count(
    estimator: RotatingShieldPFEstimator,
    candidate: np.ndarray,
    *,
    pose_idx: int,
    fe_index: int,
    pb_index: int,
) -> float:
    """Return the shared unit-strength response for one synthetic row."""
    from pf.particle_filter import MeasurementData

    data = MeasurementData(
        z_k=np.zeros(1, dtype=float),
        observation_variances=np.ones(1, dtype=float),
        detector_positions=np.asarray([estimator.poses[pose_idx]], dtype=float),
        fe_indices=np.asarray([fe_index], dtype=int),
        pb_indices=np.asarray([pb_index], dtype=int),
        live_times=np.ones(1, dtype=float),
    )
    response = estimator._cached_expected_counts_per_source(
        filt=estimator.filters[ISOTOPE],
        isotope=ISOTOPE,
        data=data,
        sources=np.asarray(candidate, dtype=float).reshape(1, 3),
        strengths=np.ones(1, dtype=float),
    )
    return float(np.asarray(response, dtype=float)[0, 0])


def _append_station(
    estimator: RotatingShieldPFEstimator,
    candidate: np.ndarray,
    *,
    pose_idx: int,
    fe_index: int,
    pb_index: int,
    candidate_strength: float | None,
) -> None:
    """Append one synthetic station with optional candidate signal."""
    station_start = len(estimator.measurements)
    estimator._candidate_verification_station_start = station_start
    observed = 0.0
    if candidate_strength is not None:
        observed = _candidate_unit_count(
            estimator,
            candidate,
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
        ) * float(candidate_strength)
    estimator.measurements.append(
        MeasurementRecord(
            z_k={ISOTOPE: float(observed)},
            pose_idx=int(pose_idx),
            orient_idx=int(fe_index),
            live_time_s=1.0,
            fe_index=int(fe_index),
            pb_index=int(pb_index),
            z_variance_k={ISOTOPE: max(float(observed), 1.0)},
        )
    )


def _queue_origin_candidate(
    estimator: RotatingShieldPFEstimator,
    candidate: np.ndarray,
    strength: float = 100.0,
) -> None:
    """Queue a candidate at the origin station and record its provenance."""
    _append_station(
        estimator,
        candidate,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        candidate_strength=None,
    )
    estimator._merge_candidate_verification_queue(
        ISOTOPE,
        np.asarray(candidate, dtype=float).reshape(1, 3),
        np.asarray([strength], dtype=float),
    )


def _scalar_profile_scores(
    filt: IsotopeParticleFilter,
    observed: np.ndarray,
    baseline: np.ndarray,
    design: np.ndarray,
    mask: np.ndarray,
    observation_variances: np.ndarray,
    *,
    l2: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar bounded robust-profile fits as a test-only oracle."""
    strengths = np.zeros(design.shape[1], dtype=float)
    improvements = np.zeros(design.shape[1], dtype=float)
    for candidate_index in range(design.shape[1]):
        response_all = np.maximum(design[:, candidate_index], 0.0)
        active = np.asarray(mask[:, candidate_index], dtype=bool) & (
            response_all > 1.0e-12
        )
        if not np.any(active):
            continue
        z = np.maximum(observed[active], 0.0)
        base = np.maximum(baseline[active], 1.0e-12)
        response = response_all[active]
        variances = np.maximum(observation_variances[active], 0.0)

        def objective(strength: float) -> float:
            """Return the scalar penalized configured log likelihood."""
            fitted = np.maximum(base + response * float(strength), 1.0e-12)
            log_likelihood = filt._count_log_likelihood_np(
                z,
                fitted,
                observation_count_variance=variances,
            )
            return float(log_likelihood - 0.5 * l2 * float(strength) ** 2)

        result = minimize_scalar(
            lambda value: -objective(float(value)),
            bounds=(0.0, float(q_max)),
            method="bounded",
            options={"xatol": 1.0e-10, "maxiter": 500},
        )
        strength = float(result.x)
        boundary_scores = np.asarray(
            [objective(0.0), objective(strength), objective(float(q_max))],
            dtype=float,
        )
        boundary_strengths = np.asarray([0.0, strength, float(q_max)], dtype=float)
        best = int(np.argmax(boundary_scores))
        strength = float(boundary_strengths[best])
        strengths[candidate_index] = strength
        improvements[candidate_index] = 2.0 * (
            float(boundary_scores[best]) - objective(0.0)
        )
    return strengths, improvements


def test_candidate_verification_never_reuses_proposal_measurements() -> None:
    """A candidate must not be promoted by the data that proposed it."""
    estimator = _build_estimator()
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, candidate)

    promoted_positions, promoted_strengths, rejected = (
        estimator._evaluate_candidate_verification_queue(
            ISOTOPE,
            estimator.filters[ISOTOPE],
        )
    )
    provenance = estimator._candidate_verification_provenance[ISOTOPE]

    assert promoted_positions.size == 0
    assert promoted_strengths.size == 0
    assert rejected == 0
    assert provenance.proposal_measurement_cutoffs.tolist() == [1]
    assert provenance.last_evaluated_measurement_counts.tolist() == [1]
    assert provenance.positive_attempts.tolist() == [0]
    assert provenance.negative_attempts.tolist() == [0]
    np.testing.assert_allclose(provenance.origin_positions_xyz_m, [[1.0, 1.0, 0.5]])
    assert provenance.origin_fe_programs == ((0,),)
    assert provenance.origin_pb_programs == ((0,),)


def test_independent_verification_forces_queue_only_runtime_rescue() -> None:
    """Enabling independent evidence must make the queue the injection gate."""
    config = RotatingShieldPFConfig(
        candidate_verification_independent_evidence_enable=True,
        candidate_verification_queue_enable=False,
        runtime_report_rescue_verification_queue_only=False,
    )

    assert config.candidate_verification_queue_enable is True
    assert config.runtime_report_rescue_verification_queue_only is True


def test_independent_verification_prevents_direct_sparse_queue_pruning() -> None:
    """All-history sparse evidence must not bypass decisive-negative attempts."""
    estimator = _build_estimator(sparse_poisson_evidence_authoritative=True)
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, candidate)

    removed = estimator._prune_candidate_verification_queue_with_sparse_evidence(
        ISOTOPE,
        {
            "model_order_ready": True,
            "selected_positions": [],
        },
    )

    assert removed == 0
    assert ISOTOPE in estimator._candidate_verification_queue
    provenance = estimator._candidate_verification_provenance[ISOTOPE]
    assert provenance.positive_attempts.tolist() == [0]
    assert provenance.negative_attempts.tolist() == [0]


def test_authoritative_sparse_candidate_waits_in_independent_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ready sparse candidates must queue instead of synchronizing PF particles."""
    estimator = _build_estimator(sparse_poisson_evidence_authoritative=True)
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _append_station(
        estimator,
        candidate,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        candidate_strength=100.0,
    )
    data = estimator._measurement_data_for_iso(ISOTOPE, window=None)
    assert isinstance(data, MeasurementData)
    estimator._last_sparse_poisson_evidence_diagnostics[ISOTOPE] = {
        "available": True,
        "model_order_ready": True,
        "selected_count": 1,
    }
    monkeypatch.setattr(
        estimator,
        "_authoritative_sparse_evidence_sources",
        lambda *_args, **_kwargs: (
            True,
            candidate.reshape(1, 3),
            np.asarray([100.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        estimator.filters[ISOTOPE],
        "sync_particles_to_evidence_sources",
        lambda *_args, **_kwargs: pytest.fail("sparse sync bypassed the queue"),
    )
    structural: dict[str, object] = {}

    def capture_structural_update(**kwargs: object) -> None:
        """Capture whether sparse readiness still disables direct PF births."""
        structural.update(kwargs)

    monkeypatch.setattr(
        estimator.filters[ISOTOPE],
        "apply_birth_death",
        capture_structural_update,
    )
    monkeypatch.setattr(
        estimator,
        "_inject_runtime_report_rescue",
        lambda *_args, **_kwargs: 0,
    )

    estimator._run_isotope_structural_update(
        (ISOTOPE, estimator.filters[ISOTOPE], data, data, data)
    )

    queue = estimator._candidate_verification_queue[ISOTOPE]
    np.testing.assert_allclose(queue[0], candidate.reshape(1, 3))
    np.testing.assert_allclose(queue[1], [100.0])
    assert structural["allow_structural_birth_proposals"] is False


def test_candidate_verification_requires_all_three_independence_gates() -> None:
    """XY, detector height, and shield-program gates must all pass together."""
    estimator = _build_estimator(candidate_verification_min_positive_checks=2)
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, candidate)
    failing_stations = (
        (1, 1, 1),
        (2, 1, 1),
        (3, 0, 0),
    )
    for pose_idx, fe_index, pb_index in failing_stations:
        _append_station(
            estimator,
            candidate,
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            candidate_strength=100.0,
        )
        promoted, _, _ = estimator._evaluate_candidate_verification_queue(
            ISOTOPE,
            estimator.filters[ISOTOPE],
        )
        assert promoted.size == 0
        provenance = estimator._candidate_verification_provenance[ISOTOPE]
        assert provenance.positive_attempts.tolist() == [0]

    _append_station(
        estimator,
        candidate,
        pose_idx=3,
        fe_index=1,
        pb_index=1,
        candidate_strength=100.0,
    )
    promoted, _, _ = estimator._evaluate_candidate_verification_queue(
        ISOTOPE,
        estimator.filters[ISOTOPE],
    )

    assert promoted.size == 0
    provenance = estimator._candidate_verification_provenance[ISOTOPE]
    assert provenance.positive_attempts.tolist() == [1]
    assert provenance.negative_attempts.tolist() == [0]


def test_queue_only_verification_promotes_and_injects_old_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified old modes should inject while current new modes remain queued."""
    estimator = _build_estimator()
    old_candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    new_candidate = np.asarray([4.0, 4.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, old_candidate)
    _append_station(
        estimator,
        old_candidate,
        pose_idx=3,
        fe_index=1,
        pb_index=1,
        candidate_strength=100.0,
    )
    captured: dict[str, Any] = {}

    def fake_rescue_estimate(
        _isotope: str,
        _filt: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one genuinely new candidate for the current station."""
        return new_candidate.reshape(1, 3), np.asarray([40.0], dtype=float)

    def capture_injection(
        positions: np.ndarray,
        strengths: np.ndarray,
        **kwargs: Any,
    ) -> int:
        """Capture the existing rescue-particle injection call."""
        captured["positions"] = np.asarray(positions, dtype=float).copy()
        captured["strengths"] = np.asarray(strengths, dtype=float).copy()
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(
        estimator,
        "_runtime_report_rescue_estimate",
        fake_rescue_estimate,
    )
    monkeypatch.setattr(
        estimator.filters[ISOTOPE],
        "inject_runtime_report_rescue_particles",
        capture_injection,
    )

    injected = estimator._inject_runtime_report_rescue(
        ISOTOPE,
        estimator.filters[ISOTOPE],
    )

    assert injected == 2
    np.testing.assert_allclose(captured["positions"], old_candidate.reshape(1, 3))
    assert captured["strengths"][0] == pytest.approx(100.0, rel=1.0e-3)
    assert captured["combine_sources"] is True
    queue = estimator._candidate_verification_queue[ISOTOPE]
    np.testing.assert_allclose(queue[0], new_candidate.reshape(1, 3))
    provenance = estimator._candidate_verification_provenance[ISOTOPE]
    assert provenance.proposal_measurement_cutoffs.tolist() == [2]
    assert provenance.positive_attempts.tolist() == [0]


def test_candidate_verification_rejects_after_decisive_negatives() -> None:
    """Repeated independent null observations should reject a queued mode."""
    estimator = _build_estimator(
        candidate_verification_negative_deviance_threshold=0.0,
        candidate_verification_reject_after_negatives=2,
    )
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, candidate)

    _append_station(
        estimator,
        candidate,
        pose_idx=3,
        fe_index=1,
        pb_index=1,
        candidate_strength=None,
    )
    promoted, _, rejected = estimator._evaluate_candidate_verification_queue(
        ISOTOPE,
        estimator.filters[ISOTOPE],
    )
    assert promoted.size == 0
    assert rejected == 0
    provenance = estimator._candidate_verification_provenance[ISOTOPE]
    assert provenance.negative_attempts.tolist() == [1]

    _append_station(
        estimator,
        candidate,
        pose_idx=4,
        fe_index=1,
        pb_index=1,
        candidate_strength=None,
    )
    promoted, _, rejected = estimator._evaluate_candidate_verification_queue(
        ISOTOPE,
        estimator.filters[ISOTOPE],
    )

    assert promoted.size == 0
    assert rejected == 1
    assert ISOTOPE not in estimator._candidate_verification_queue
    assert ISOTOPE not in estimator._candidate_verification_provenance


def test_rejected_candidate_is_not_requeued_from_same_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decisive rejection must survive the current all-history rescue pass."""
    estimator = _build_estimator(
        candidate_verification_negative_deviance_threshold=0.0,
        candidate_verification_reject_after_negatives=1,
    )
    candidate = np.asarray([0.0, 0.0, 0.0], dtype=float)
    _queue_origin_candidate(estimator, candidate)
    _append_station(
        estimator,
        candidate,
        pose_idx=3,
        fe_index=1,
        pb_index=1,
        candidate_strength=None,
    )
    monkeypatch.setattr(
        estimator,
        "_runtime_report_rescue_estimate",
        lambda *_args, **_kwargs: (
            candidate.reshape(1, 3),
            np.asarray([100.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        estimator.filters[ISOTOPE],
        "inject_runtime_report_rescue_particles",
        lambda *_args, **_kwargs: pytest.fail("rejected candidate was injected"),
    )

    injected = estimator._inject_runtime_report_rescue(
        ISOTOPE,
        estimator.filters[ISOTOPE],
    )

    assert injected == 0
    assert ISOTOPE not in estimator._candidate_verification_queue
    assert ISOTOPE not in estimator._candidate_verification_provenance
    assert estimator._last_candidate_verification_diagnostics[ISOTOPE][
        "rejected_count"
    ] == 1


def test_candidate_verification_batch_scores_match_scalar_oracle() -> None:
    """Batched robust profile scores should match a scalar optimizer oracle."""
    estimator = _build_estimator(
        count_likelihood_model="student_t",
        count_likelihood_df=5.0,
        transport_model_rel_sigma=0.10,
        transport_model_abs_sigma=2.0,
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=1.0,
    )
    filt = estimator.filters[ISOTOPE]
    observed = np.asarray([20.0, 9.0, 18.0, 7.0], dtype=float)
    baseline = np.asarray([10.0, 12.0, 8.0, 6.0], dtype=float)
    observation_variances = np.asarray([4.0, 25.0, 9.0, 16.0], dtype=float)
    design = np.asarray(
        [
            [1.0, 0.2, 0.5],
            [0.8, 0.4, 0.1],
            [0.4, 1.2, 0.7],
            [0.2, 0.5, 1.0],
        ],
        dtype=float,
    )
    mask = np.asarray(
        [
            [True, True, False],
            [True, False, False],
            [True, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    l2 = 0.03
    max_iters = 64
    q_max = 100.0

    batch = RotatingShieldPFEstimator._profile_candidate_deviance_scores_batch(
        filt,
        observed,
        baseline,
        design,
        mask,
        observation_variances,
        l2=l2,
        max_iters=max_iters,
        q_max=q_max,
    )
    scalar = _scalar_profile_scores(
        filt,
        observed,
        baseline,
        design,
        mask,
        observation_variances,
        l2=l2,
        q_max=q_max,
    )

    np.testing.assert_allclose(batch[0], scalar[0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(batch[1], scalar[1], rtol=1.0e-6, atol=1.0e-6)


def test_candidate_verification_score_respects_propagated_variance() -> None:
    """A noisy unfolded count must carry less verification evidence."""
    estimator = _build_estimator(
        count_likelihood_model="student_t",
        count_likelihood_df=5.0,
        transport_model_rel_sigma=0.10,
        transport_model_abs_sigma=5.0,
    )
    filt = estimator.filters[ISOTOPE]
    observed = np.asarray([100.0], dtype=float)
    baseline = np.asarray([10.0], dtype=float)
    design = np.asarray([[1.0]], dtype=float)
    mask = np.asarray([[True]], dtype=bool)

    low_variance = RotatingShieldPFEstimator._profile_candidate_deviance_scores_batch(
        filt,
        observed,
        baseline,
        design,
        mask,
        np.asarray([1.0], dtype=float),
        l2=1.0e-6,
        max_iters=48,
        q_max=200.0,
    )
    high_variance = RotatingShieldPFEstimator._profile_candidate_deviance_scores_batch(
        filt,
        observed,
        baseline,
        design,
        mask,
        np.asarray([1.0e6], dtype=float),
        l2=1.0e-6,
        max_iters=48,
        q_max=200.0,
    )

    assert low_variance[1][0] > high_variance[1][0]
