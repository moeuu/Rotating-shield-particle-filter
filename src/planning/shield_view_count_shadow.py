"""Statistical shadow decisions for fixed-execution shield view counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t


@dataclass(frozen=True, slots=True)
class ShieldViewCountShadowDecision:
    """Store vectorized paired-MC evidence for several shield view counts."""

    candidate_view_counts: tuple[int, ...]
    reference_view_count: int
    information_gain_mean_la: NDArray[np.float64]
    information_gain_standard_error_la: NDArray[np.float64]
    retained_fraction_la: NDArray[np.float64]
    retention_margin_mean_sa: NDArray[np.float64]
    retention_margin_standard_error_sa: NDArray[np.float64]
    retention_margin_lower_confidence_sa: NDArray[np.float64]
    retention_point_passed_sa: NDArray[np.bool_]
    retention_lcb_passed_sa: NDArray[np.bool_]
    point_selected_view_count_a: NDArray[np.int64]
    lcb_selected_view_count_a: NDArray[np.int64]
    monotonicity_warning_a: NDArray[np.bool_]
    sample_count: int
    retention_fraction: float
    per_comparison_confidence: float


def select_shield_view_count_shadow(
    kl_samples_laq: NDArray[np.float64],
    *,
    candidate_view_counts: tuple[int, ...],
    retention_fraction: float,
    per_comparison_confidence: float,
) -> ShieldViewCountShadowDecision:
    """Select the shortest statistically equivalent nested view prefix.

    The input is ordered by ``candidate_view_counts`` and must contain common-
    random-number KL samples shaped ``(view_count, pose, predictive_sample)``.
    Each shorter prefix is compared with the largest view count through
    ``KL_short - retention_fraction * KL_reference``.  A strict positive
    one-sided Student-t lower bound is required before a shorter prefix is
    recommended.  The function is vectorized over every pose.
    """
    lengths = _validated_candidate_view_counts(candidate_view_counts)
    retention = _strict_probability(
        retention_fraction,
        name="retention_fraction",
        allow_one=True,
    )
    confidence = _strict_probability(
        per_comparison_confidence,
        name="per_comparison_confidence",
        allow_one=False,
    )
    samples = np.asarray(kl_samples_laq)
    if samples.dtype != np.float64:
        raise TypeError("kl_samples_laq must use float64.")
    if (
        samples.ndim != 3
        or samples.shape[0] != len(lengths)
        or samples.shape[1] < 1
        or samples.shape[2] < 2
        or np.any(~np.isfinite(samples))
        or np.any(samples < 0.0)
    ):
        raise ValueError(
            "kl_samples_laq must be finite nonnegative (length, pose, sample) "
            "data with at least two paired samples."
        )

    sample_count = int(samples.shape[2])
    means = np.mean(samples, axis=2, dtype=np.float64)
    standard_errors = np.std(samples, axis=2, ddof=1) / np.sqrt(float(sample_count))
    reference = samples[-1]
    shorter = samples[:-1]
    margins = shorter - retention * reference[np.newaxis, :, :]
    margin_means = np.mean(margins, axis=2, dtype=np.float64)
    margin_standard_errors = np.std(margins, axis=2, ddof=1) / np.sqrt(
        float(sample_count)
    )
    critical = float(student_t.ppf(confidence, sample_count - 1))
    margin_lower = margin_means - critical * margin_standard_errors
    point_passed = margin_means >= 0.0
    lcb_passed = margin_lower > 0.0

    pose_count = int(samples.shape[1])
    reference_count = int(lengths[-1])
    point_selected = np.full(pose_count, reference_count, dtype=np.int64)
    lcb_selected = np.full(pose_count, reference_count, dtype=np.int64)
    # Candidate count is a tiny, validated policy set. This loop does not span
    # particles, poses, shield pairs, or predictive samples.
    for index, view_count in reversed(tuple(enumerate(lengths[:-1]))):
        point_selected = np.where(
            point_passed[index],
            int(view_count),
            point_selected,
        )
        lcb_selected = np.where(
            lcb_passed[index],
            int(view_count),
            lcb_selected,
        )

    reference_means = means[-1]
    retained = np.full_like(means, np.nan)
    positive_reference = reference_means > 0.0
    np.divide(
        means,
        reference_means[np.newaxis, :],
        out=retained,
        where=positive_reference[np.newaxis, :],
    )
    monotonicity_warning = np.any(
        np.diff(means, axis=0) < -1.0e-12,
        axis=0,
    )
    return ShieldViewCountShadowDecision(
        candidate_view_counts=lengths,
        reference_view_count=reference_count,
        information_gain_mean_la=np.asarray(means, dtype=np.float64),
        information_gain_standard_error_la=np.asarray(
            standard_errors,
            dtype=np.float64,
        ),
        retained_fraction_la=np.asarray(retained, dtype=np.float64),
        retention_margin_mean_sa=np.asarray(margin_means, dtype=np.float64),
        retention_margin_standard_error_sa=np.asarray(
            margin_standard_errors,
            dtype=np.float64,
        ),
        retention_margin_lower_confidence_sa=np.asarray(
            margin_lower,
            dtype=np.float64,
        ),
        retention_point_passed_sa=np.asarray(point_passed, dtype=np.bool_),
        retention_lcb_passed_sa=np.asarray(lcb_passed, dtype=np.bool_),
        point_selected_view_count_a=np.asarray(point_selected, dtype=np.int64),
        lcb_selected_view_count_a=np.asarray(lcb_selected, dtype=np.int64),
        monotonicity_warning_a=np.asarray(
            monotonicity_warning,
            dtype=np.bool_,
        ),
        sample_count=sample_count,
        retention_fraction=float(retention),
        per_comparison_confidence=float(confidence),
    )


def _validated_candidate_view_counts(value: object) -> tuple[int, ...]:
    """Return a strictly increasing tuple with at least two view counts."""
    if not isinstance(value, tuple) or len(value) < 2:
        raise ValueError(
            "candidate_view_counts must be a tuple of at least two values."
        )
    resolved: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError("candidate_view_counts must contain only integers.")
        count = int(item)
        if count < 1:
            raise ValueError("candidate_view_counts must be positive.")
        resolved.append(count)
    if any(right <= left for left, right in zip(resolved, resolved[1:])):
        raise ValueError("candidate_view_counts must be strictly increasing.")
    return tuple(resolved)


def _strict_probability(
    value: object,
    *,
    name: str,
    allow_one: bool,
) -> float:
    """Return one finite probability under the requested endpoint rule."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be numeric.")
    resolved = float(value)
    valid_upper = resolved <= 1.0 if allow_one else resolved < 1.0
    if not np.isfinite(resolved) or resolved <= 0.0 or not valid_upper:
        endpoint = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} must lie in {endpoint}.")
    return resolved


__all__ = [
    "ShieldViewCountShadowDecision",
    "select_shield_view_count_shadow",
]
