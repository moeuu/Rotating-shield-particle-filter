"""Tests for paired-MC shield view-count shadow decisions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from planning.shield_view_count_shadow import select_shield_view_count_shadow


def _constant_samples(
    values_la: list[list[float]], sample_count: int = 8
) -> np.ndarray:
    """Expand pose means into zero-variance paired float64 samples."""
    values = np.asarray(values_la, dtype=np.float64)
    return np.repeat(values[:, :, np.newaxis], sample_count, axis=2)


def test_shadow_selects_shortest_passing_view_count() -> None:
    """The strict sequential rule must choose 2, then 4, then 8 views."""
    samples = _constant_samples(
        [
            [9.6, 9.0, 9.0],
            [9.8, 9.6, 9.0],
            [10.0, 10.0, 10.0],
        ]
    )

    result = select_shield_view_count_shadow(
        samples,
        candidate_view_counts=(2, 4, 8),
        retention_fraction=0.95,
        per_comparison_confidence=0.95,
    )

    assert result.point_selected_view_count_a.tolist() == [2, 4, 8]
    assert result.lcb_selected_view_count_a.tolist() == [2, 4, 8]
    assert result.retention_lcb_passed_sa.tolist() == [
        [True, False, False],
        [True, True, False],
    ]


def test_shadow_uses_paired_margin_student_t_lower_bound() -> None:
    """The saved bound must equal the direct common-random-number formula."""
    reference = np.asarray([1.0, 3.0, 2.0, 4.0], dtype=np.float64)
    short = 0.95 * reference + np.asarray(
        [0.2, -0.1, 0.1, -0.05],
        dtype=np.float64,
    )
    middle = 0.95 * reference + 0.5
    samples = np.stack((short, middle, reference), axis=0)[:, np.newaxis, :]

    result = select_shield_view_count_shadow(
        samples,
        candidate_view_counts=(2, 4, 8),
        retention_fraction=0.95,
        per_comparison_confidence=0.95,
    )

    paired = short - 0.95 * reference
    expected_mean = float(np.mean(paired))
    expected_se = float(np.std(paired, ddof=1) / np.sqrt(paired.size))
    expected_lcb = expected_mean - float(student_t.ppf(0.95, 3)) * expected_se
    assert result.retention_margin_mean_sa[0, 0] == pytest.approx(expected_mean)
    assert result.retention_margin_standard_error_sa[0, 0] == pytest.approx(expected_se)
    assert result.retention_margin_lower_confidence_sa[0, 0] == pytest.approx(
        expected_lcb
    )
    assert result.point_selected_view_count_a[0] == 2
    assert result.lcb_selected_view_count_a[0] == 4


def test_shadow_zero_reference_keeps_ratio_missing_and_falls_back() -> None:
    """Zero information must not create a false relative-equivalence claim."""
    samples = np.zeros((3, 1, 4), dtype=np.float64)

    result = select_shield_view_count_shadow(
        samples,
        candidate_view_counts=(2, 4, 8),
        retention_fraction=0.95,
        per_comparison_confidence=0.95,
    )

    assert np.all(np.isnan(result.retained_fraction_la[:, 0]))
    assert result.point_selected_view_count_a.tolist() == [2]
    assert result.lcb_selected_view_count_a.tolist() == [8]


def test_shadow_is_pose_batched_and_permutation_equivariant() -> None:
    """Vectorized pose evaluation must equal concatenated one-pose calls."""
    rng = np.random.default_rng(123)
    base = rng.uniform(0.2, 3.0, size=(1, 7, 20))
    fractions = np.asarray([0.93, 0.97, 1.0], dtype=np.float64)[:, None, None]
    samples = np.asarray(fractions * base, dtype=np.float64)
    permutation = np.asarray([5, 1, 6, 0, 4, 2, 3], dtype=np.int64)

    batch = select_shield_view_count_shadow(
        samples,
        candidate_view_counts=(2, 4, 8),
        retention_fraction=0.95,
        per_comparison_confidence=0.95,
    )
    permuted = select_shield_view_count_shadow(
        samples[:, permutation],
        candidate_view_counts=(2, 4, 8),
        retention_fraction=0.95,
        per_comparison_confidence=0.95,
    )
    serial = np.concatenate(
        [
            select_shield_view_count_shadow(
                samples[:, pose_index : pose_index + 1],
                candidate_view_counts=(2, 4, 8),
                retention_fraction=0.95,
                per_comparison_confidence=0.95,
            ).lcb_selected_view_count_a
            for pose_index in range(samples.shape[1])
        ]
    )

    assert np.array_equal(batch.lcb_selected_view_count_a, serial)
    assert np.array_equal(
        permuted.lcb_selected_view_count_a,
        batch.lcb_selected_view_count_a[permutation],
    )


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        (np.ones((3, 1, 1), dtype=np.float64), "at least two"),
        (np.ones((2, 1, 4), dtype=np.float64), "length, pose, sample"),
        (np.ones((3, 1, 4), dtype=np.float32), "float64"),
    ],
)
def test_shadow_rejects_invalid_sample_contract(
    samples: np.ndarray,
    match: str,
) -> None:
    """Malformed or lower-fidelity paired samples must fail closed."""
    with pytest.raises((TypeError, ValueError), match=match):
        select_shield_view_count_shadow(
            samples,
            candidate_view_counts=(2, 4, 8),
            retention_fraction=0.95,
            per_comparison_confidence=0.95,
        )
