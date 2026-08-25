"""Tests for conditional DSS contender and ambiguity integration."""

from __future__ import annotations

import numpy as np

from pf.provenance import strict_canonical_json_bytes, strict_json_loads
from planning.dss_pp import (
    _conditional_pose_ambiguity_mask,
    _proxy_replica_scores_payload,
    _select_contenders_from_kl_samples,
    _slice_joint_program_components,
)
from planning.dss_types import _JointProgramSpectrumComponents


def test_program_confirmation_is_requested_only_for_uncertain_distinct_gap() -> None:
    """Clear or duplicate contenders must not trigger independent sampling."""
    subsets = np.asarray(
        [
            [[0, 1], [0, 2], [0, 1]],
            [[0, 1], [0, 2], [2, 3]],
        ],
        dtype=np.int64,
    )
    samples = np.asarray(
        [
            [[2.0, 2.1, 1.9, 2.0], [0.1, 0.2, 0.0, 0.1], [2.0, 2.1, 1.9, 2.0]],
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        ],
        dtype=np.float64,
    )

    programs, gains, indices, ambiguous, lower = (
        _select_contenders_from_kl_samples(
            subsets,
            samples,
            confidence=0.95,
        )
    )

    assert np.array_equal(programs[0], [0, 1])
    assert gains[0] == 2.0
    assert indices[0] == 0
    assert bool(ambiguous[0]) is False
    assert lower[0] > 0.0
    assert bool(ambiguous[1]) is True


def test_pose_confirmation_mask_contains_leader_and_overlapping_pose_only() -> None:
    """Pose rechecks must exclude candidates with a separated paired gap."""
    score_samples = np.asarray(
        [
            [2.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [-2.0, -1.0, -2.0, -1.0],
        ],
        dtype=np.float64,
    )

    mask, lower = _conditional_pose_ambiguity_mask(
        score_samples,
        np.mean(score_samples, axis=1),
        confidence=0.95,
    )

    assert mask.tolist() == [True, True, False]
    assert np.isinf(lower[0])
    assert lower[1] <= 0.0
    assert lower[2] > 0.0


def test_sparse_proxy_replica_diagnostics_are_strict_json() -> None:
    """Unrefined poses must use JSON null instead of non-finite sentinels."""
    scores = np.asarray(
        [[3.0, 2.0, 1.0], [3.1, np.nan, np.nan], [2.9, np.nan, np.nan]],
        dtype=np.float64,
    )

    payload = _proxy_replica_scores_payload(scores, evaluated=True)

    assert payload == [[3.0, 2.0, 1.0], [3.1, None, None], [2.9, None, None]]
    assert strict_json_loads(
        strict_canonical_json_bytes({"proxy_replica_scores": payload})
    ) == {
        "proxy_replica_scores": payload
    }


def test_program_contender_deduplication_is_order_invariant() -> None:
    """One pair set in another execution order must not trigger confirmation."""
    subsets = np.asarray(
        [[[0, 1, 2], [2, 1, 0], [0, 1, 3]]],
        dtype=np.int64,
    )
    samples = np.asarray(
        [[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )

    programs, _gains, indices, ambiguous, _lower = (
        _select_contenders_from_kl_samples(
            subsets,
            samples,
            confidence=0.95,
        )
    )

    assert np.array_equal(programs, np.asarray([[0, 1, 2]], dtype=np.int64))
    assert indices.tolist() == [0]
    assert ambiguous.tolist() == [False]


def test_single_pose_confirmation_component_slice_is_zero_copy() -> None:
    """Ambiguity confirmation must not duplicate a full response component."""
    total = np.arange(4 * 2 * 3 * 1 * 1, dtype=np.float64).reshape(
        4,
        2,
        3,
        1,
        1,
    )
    components = _JointProgramSpectrumComponents(
        total_pnvsl=total,
        uncollided_pnvsl=0.5 * total,
        features_pnvslf=np.zeros(total.shape + (4,), dtype=np.float64),
        live_times_v=np.ones(3, dtype=np.float64),
        contract_hash_sha256="0" * 64,
    )

    selected = _slice_joint_program_components(
        components,
        np.asarray([2], dtype=np.int64),
    )

    assert selected.total_pnvsl.shape[0] == 1
    assert np.shares_memory(selected.total_pnvsl, components.total_pnvsl)
    assert np.shares_memory(
        selected.uncollided_pnvsl,
        components.uncollided_pnvsl,
    )
    assert np.shares_memory(
        selected.features_pnvslf,
        components.features_pnvslf,
    )
