"""MC-stability controlled pose shortlisting for DSS planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t


@dataclass(frozen=True, slots=True)
class ShortlistBoundaryDiagnostic:
    """Describe one attempted exact-pose shortlist boundary."""

    pose_count: int
    boundary_included_pose: int | None
    boundary_excluded_pose: int | None
    paired_gap_mean: float | None
    paired_gap_standard_error: float | None
    paired_gap_lower_confidence: float | None
    mean_top_k_jaccard: float
    stable: bool


@dataclass(frozen=True, slots=True)
class AdaptivePoseShortlist:
    """Store the deterministic adaptive exact-pose shortlist decision."""

    pose_indices: NDArray[np.int64]
    mean_scores_p: NDArray[np.float64]
    stop_reason: str
    coverage_reserve_pose: int | None
    boundary_diagnostics: tuple[ShortlistBoundaryDiagnostic, ...]


def _stable_descending_indices(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return descending indices with the original index as tie-break."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(scores)):
        raise ValueError("Ranking scores must be finite.")
    return np.lexsort(
        (np.arange(scores.size, dtype=np.int64), -scores)
    ).astype(np.int64, copy=False)


def _jaccard(left: NDArray[np.int64], right: NDArray[np.int64]) -> float:
    """Return the Jaccard similarity of two integer index sets."""
    left_set = set(int(value) for value in np.asarray(left).reshape(-1))
    right_set = set(int(value) for value in np.asarray(right).reshape(-1))
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _mean_available_scores(
    replica_scores_rp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average available replicas while requiring one score for every pose."""
    scores = np.asarray(replica_scores_rp, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("replica_scores_rp must be a nonempty replica-by-pose matrix.")
    if np.any(np.isinf(scores)):
        raise ValueError("Proxy replica scores may be finite or NaN only.")
    available = np.sum(np.isfinite(scores), axis=0)
    if np.any(available == 0):
        raise ValueError("Every pose requires at least one proxy score.")
    totals = np.nansum(scores, axis=0, dtype=np.float64)
    return np.asarray(totals / available, dtype=np.float64)


def _boundary_diagnostic(
    replica_scores_rp: NDArray[np.float64],
    mean_order_p: NDArray[np.int64],
    *,
    pose_count: int,
    confidence: float,
    minimum_jaccard: float,
) -> ShortlistBoundaryDiagnostic:
    """Evaluate paired MC gap and top-k membership at one boundary."""
    scores = np.asarray(replica_scores_rp, dtype=np.float64)
    pose_total = int(scores.shape[1])
    if pose_count >= pose_total:
        return ShortlistBoundaryDiagnostic(
            pose_count=pose_count,
            boundary_included_pose=None,
            boundary_excluded_pose=None,
            paired_gap_mean=None,
            paired_gap_standard_error=None,
            paired_gap_lower_confidence=None,
            mean_top_k_jaccard=1.0,
            stable=True,
        )
    included = int(mean_order_p[pose_count - 1])
    excluded = int(mean_order_p[pose_count])
    paired_mask = np.isfinite(scores[:, included]) & np.isfinite(scores[:, excluded])
    paired = scores[paired_mask, included] - scores[paired_mask, excluded]
    if paired.size >= 2:
        gap_mean = float(np.mean(paired, dtype=np.float64))
        gap_se = float(np.std(paired, ddof=1) / np.sqrt(float(paired.size)))
        critical = float(student_t.ppf(confidence, int(paired.size - 1)))
        lower = float(gap_mean - critical * gap_se)
    else:
        gap_mean = float(paired[0]) if paired.size == 1 else None
        gap_se = None
        lower = None

    mean_top = np.asarray(mean_order_p[:pose_count], dtype=np.int64)
    similarities: list[float] = []
    for replica in scores:
        available = np.isfinite(replica)
        if int(np.sum(available)) < pose_count:
            continue
        available_indices = np.flatnonzero(available)
        local_order = _stable_descending_indices(replica[available])
        replica_top = available_indices[local_order[:pose_count]]
        similarities.append(_jaccard(mean_top, replica_top))
    mean_jaccard = (
        float(np.mean(similarities, dtype=np.float64)) if similarities else 0.0
    )
    stable = bool(
        lower is not None
        and lower > 0.0
        and mean_jaccard >= float(minimum_jaccard)
    )
    return ShortlistBoundaryDiagnostic(
        pose_count=pose_count,
        boundary_included_pose=included,
        boundary_excluded_pose=excluded,
        paired_gap_mean=gap_mean,
        paired_gap_standard_error=gap_se,
        paired_gap_lower_confidence=lower,
        mean_top_k_jaccard=mean_jaccard,
        stable=stable,
    )


def select_adaptive_pose_shortlist(
    replica_scores_rp: NDArray[np.float64],
    coverage_gains_p: NDArray[np.float64],
    *,
    minimum_pose_count: int,
    maximum_pose_count: int,
    pose_count_step: int,
    coverage_reserve_count: int,
    boundary_confidence: float,
    minimum_top_k_jaccard: float,
) -> AdaptivePoseShortlist:
    """Select 8--16 style exact poses from paired proxy score replicas.

    Additional proxy replicas may contain NaN outside a preselected refinement
    pool. The first or another available replica must still score every pose.
    """
    scores = np.asarray(replica_scores_rp, dtype=np.float64)
    coverage = np.asarray(coverage_gains_p, dtype=np.float64).reshape(-1)
    mean_scores = _mean_available_scores(scores)
    if coverage.shape != mean_scores.shape or np.any(~np.isfinite(coverage)):
        raise ValueError("coverage_gains_p must be finite and align with poses.")
    integer_values = {
        "minimum_pose_count": minimum_pose_count,
        "maximum_pose_count": maximum_pose_count,
        "pose_count_step": pose_count_step,
        "coverage_reserve_count": coverage_reserve_count,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
    if minimum_pose_count <= 0 or maximum_pose_count < minimum_pose_count:
        raise ValueError("Pose shortlist bounds are invalid.")
    if pose_count_step <= 0 or coverage_reserve_count < 0:
        raise ValueError("Pose shortlist step and reserve are invalid.")
    if coverage_reserve_count > minimum_pose_count:
        raise ValueError("Coverage reserve must fit within the minimum shortlist.")
    if not 0.0 < float(boundary_confidence) < 1.0:
        raise ValueError("boundary_confidence must lie in (0, 1).")
    if not 0.0 <= float(minimum_top_k_jaccard) <= 1.0:
        raise ValueError("minimum_top_k_jaccard must lie in [0, 1].")

    pose_total = int(mean_scores.size)
    mean_order = _stable_descending_indices(mean_scores)
    if pose_total <= minimum_pose_count:
        selected = mean_order.copy()
        stop_reason = "all_poses_fit_minimum"
        diagnostics: list[ShortlistBoundaryDiagnostic] = []
    else:
        attempted_counts = list(
            range(
                int(minimum_pose_count),
                min(int(maximum_pose_count), pose_total) + 1,
                int(pose_count_step),
            )
        )
        resolved_maximum = min(int(maximum_pose_count), pose_total)
        if not attempted_counts or attempted_counts[-1] != resolved_maximum:
            attempted_counts.append(resolved_maximum)
        diagnostics = []
        selected_count = resolved_maximum
        stop_reason = "max_reached_unstable"
        for pose_count in attempted_counts:
            diagnostic = _boundary_diagnostic(
                scores,
                mean_order,
                pose_count=pose_count,
                confidence=float(boundary_confidence),
                minimum_jaccard=float(minimum_top_k_jaccard),
            )
            diagnostics.append(diagnostic)
            if diagnostic.stable:
                selected_count = pose_count
                stop_reason = "stable_boundary"
                break
        selected = mean_order[:selected_count].copy()

    coverage_reserve_pose: int | None = None
    if coverage_reserve_count > 0 and selected.size:
        coverage_order = _stable_descending_indices(coverage)
        reserve = coverage_order[: min(int(coverage_reserve_count), selected.size)]
        coverage_reserve_pose = int(reserve[0]) if reserve.size else None
        selected_set = set(int(value) for value in selected)
        missing = [int(value) for value in reserve if int(value) not in selected_set]
        if missing:
            retained = [
                int(value)
                for value in selected
                if int(value) not in set(int(item) for item in reserve)
            ]
            retained = retained[: int(selected.size) - len(missing)]
            selected = np.asarray(missing + retained, dtype=np.int64)
            selected = selected[
                np.lexsort((selected, -mean_scores[selected]))
            ].astype(np.int64, copy=False)

    return AdaptivePoseShortlist(
        pose_indices=np.asarray(selected, dtype=np.int64),
        mean_scores_p=np.asarray(mean_scores, dtype=np.float64),
        stop_reason=stop_reason,
        coverage_reserve_pose=coverage_reserve_pose,
        boundary_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "AdaptivePoseShortlist",
    "ShortlistBoundaryDiagnostic",
    "select_adaptive_pose_shortlist",
]
