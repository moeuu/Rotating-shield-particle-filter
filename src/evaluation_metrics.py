"""Compute and report evaluation metrics for PF source estimates."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

POSITION_ERROR_TARGET_M = 0.5
SURFACE_LABELS = (
    "floor",
    "wall",
    "ceiling",
    "obstacle_side",
    "obstacle_top",
    "off_surface",
)
ROOM_SURFACE_LABELS = ("floor", "wall", "ceiling")


@dataclass(frozen=True)
class Source:
    """Lightweight source representation for evaluation."""

    pos: NDArray[np.float64]
    strength: float
    surface_kind: str | None = None


def _as_array(value: Sequence[float]) -> NDArray[np.float64]:
    """Return a NumPy array for a position sequence."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Position must be a 3-element sequence.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Position coordinates must be finite.")
    return arr


def _non_negative_finite(value: Any, *, name: str) -> float:
    """Return a finite non-negative scalar for a physical metric input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def _extract_strength(obj: Any) -> float | None:
    """Extract strength from a dict/object if present."""
    for key in ("strength", "intensity_cps_1m", "intensity"):
        if isinstance(obj, Mapping) and key in obj:
            return float(obj[key])
        if hasattr(obj, key):
            return float(getattr(obj, key))
    return None


def _extract_position(obj: Any) -> NDArray[np.float64] | None:
    """Extract a position array from a dict/object if present."""
    if isinstance(obj, Mapping):
        if "pos" in obj:
            return _as_array(obj["pos"])
        if "position" in obj:
            return _as_array(obj["position"])
    if hasattr(obj, "pos"):
        return _as_array(getattr(obj, "pos"))
    if hasattr(obj, "position"):
        return _as_array(getattr(obj, "position"))
    return None


def _extract_surface_kind(obj: Any) -> str | None:
    """Extract an optional physical-surface label from a source-like object."""
    for key in ("surface_kind", "surface", "source_surface_kind"):
        if isinstance(obj, Mapping) and obj.get(key) is not None:
            return str(obj[key]).strip().lower()
        if hasattr(obj, key) and getattr(obj, key) is not None:
            return str(getattr(obj, key)).strip().lower()
    return None


def _normalize_source(entry: Any) -> Source:
    """Convert various source-like entries to Source."""
    if isinstance(entry, Source):
        return Source(
            pos=_as_array(entry.pos),
            strength=_non_negative_finite(entry.strength, name="source strength"),
            surface_kind=(
                None
                if entry.surface_kind is None
                else str(entry.surface_kind).strip().lower()
            ),
        )
    if isinstance(entry, (tuple, list, np.ndarray)) and len(entry) == 4:
        pos = _as_array(entry[:3])
        strength = _non_negative_finite(entry[3], name="source strength")
        return Source(pos=pos, strength=strength)
    pos = _extract_position(entry)
    strength = _extract_strength(entry)
    if pos is None or strength is None:
        raise ValueError("Unsupported source entry format.")
    return Source(
        pos=pos,
        strength=_non_negative_finite(strength, name="source strength"),
        surface_kind=_extract_surface_kind(entry),
    )


def _normalize_sources(entries: Iterable[Any] | None) -> List[Source]:
    """Normalize a list of source-like entries."""
    if entries is None:
        return []
    return [_normalize_source(entry) for entry in entries]


def _pairwise_distances(
    gt: List[Source],
    est: List[Source],
) -> NDArray[np.float64]:
    """Return pairwise distance matrix between GT and EST sources."""
    gt_pos = np.vstack([s.pos for s in gt]) if gt else np.zeros((0, 3), dtype=float)
    est_pos = np.vstack([s.pos for s in est]) if est else np.zeros((0, 3), dtype=float)
    if gt_pos.size == 0 or est_pos.size == 0:
        return np.zeros((len(gt), len(est)), dtype=float)
    diff = gt_pos[:, None, :] - est_pos[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    if np.any(~np.isfinite(distances)):
        raise ValueError("Source coordinates produce non-finite distances.")
    return distances


def _hungarian_assignment(cost: NDArray[np.float64]) -> List[Tuple[int, int]]:
    """Return deterministic minimal-cost Hungarian assignment pairs."""
    if cost.size == 0:
        return []
    if np.any(~np.isfinite(cost)):
        raise ValueError("Assignment cost must contain only finite values.")
    row_idx, col_idx = linear_sum_assignment(cost)
    return list(zip(row_idx.tolist(), col_idx.tolist()))


def _gated_distance_assignment(
    distances: NDArray[np.float64],
    *,
    radius_m: float,
) -> List[Tuple[int, int]]:
    """Maximize gated match cardinality, then minimize 3-D distance."""
    distance_matrix = np.asarray(distances, dtype=float)
    if distance_matrix.ndim != 2:
        raise ValueError("distances must be a two-dimensional matrix.")
    if distance_matrix.size == 0:
        return []
    if np.any(~np.isfinite(distance_matrix)) or np.any(distance_matrix < 0.0):
        raise ValueError("distances must be finite and non-negative.")
    radius = _non_negative_finite(radius_m, name="matching radius")
    valid = distance_matrix <= radius
    assignment_count = min(distance_matrix.shape)
    distance_scale = max(radius, 1.0)
    invalid_penalty = float(assignment_count + 1)
    cost = np.where(
        valid,
        distance_matrix / distance_scale,
        invalid_penalty,
    )
    assignments = _hungarian_assignment(cost)
    return [
        (int(row), int(column))
        for row, column in assignments
        if bool(valid[row, column])
    ]


def _summary_stats(values: Sequence[float]) -> Dict[str, float | None]:
    """Return summary statistics for a list of values."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "rmse": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Metric summaries require finite values.")
    scale = float(np.max(np.abs(arr), initial=0.0))
    rmse = (
        0.0
        if scale == 0.0
        else float(scale * np.sqrt(np.mean((arr / scale) ** 2)))
    )
    if not np.isfinite(rmse):
        raise ValueError("Metric summary overflowed to a non-finite value.")
    mean = 0.0 if scale == 0.0 else float(scale * np.mean(arr / scale))
    summary = {
        "mean": mean,
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "rmse": rmse,
        "max": float(np.max(arr)),
    }
    if any(not np.isfinite(value) for value in summary.values()):
        raise ValueError("Metric summary produced a non-finite value.")
    return summary


def _summary_abs(values: Sequence[float]) -> Dict[str, float | None]:
    """Return summary statistics for absolute errors."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Metric summaries require finite values.")
    scale = float(np.max(np.abs(arr), initial=0.0))
    mean = 0.0 if scale == 0.0 else float(scale * np.mean(arr / scale))
    summary = {
        "mean": mean,
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }
    if any(not np.isfinite(value) for value in summary.values()):
        raise ValueError("Metric summary produced a non-finite value.")
    return summary


def _summary_signed(values: Sequence[float]) -> Dict[str, float | None]:
    """Return signed-error summaries without discarding bias direction."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Metric summaries require finite values.")
    scale = float(np.max(np.abs(arr), initial=0.0))
    mean = 0.0 if scale == 0.0 else float(scale * np.mean(arr / scale))
    summary = {
        "mean": mean,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if any(not np.isfinite(value) for value in summary.values()):
        raise ValueError("Metric summary produced a non-finite value.")
    return summary


def _precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> Dict[str, float | int]:
    """Return detection counts and precision, recall, and F1."""
    tp = max(int(true_positive), 0)
    fp = max(int(false_positive), 0)
    fn = max(int(false_negative), 0)
    precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _position_error_summary(values: Sequence[float]) -> Dict[str, float | bool | None]:
    """Return position-error summary stats augmented with the fixed target check."""
    summary = _summary_stats(values)
    mean_error = summary["mean"]
    summary["target_m"] = float(POSITION_ERROR_TARGET_M)
    summary["within_target"] = (
        None if mean_error is None else bool(mean_error <= POSITION_ERROR_TARGET_M)
    )
    return summary


def _threshold_count_summary(
    *,
    gt_count: int,
    est_count: int,
    distances: NDArray[np.float64],
    thresholds_m: Sequence[float],
) -> Dict[str, Dict[str, float | int]]:
    """Recompute gated distance-only matching at every reporting threshold."""
    payload: Dict[str, Dict[str, float | int]] = {}
    for threshold in thresholds_m:
        radius = _non_negative_finite(threshold, name="distance threshold")
        assignments = _gated_distance_assignment(distances, radius_m=radius)
        true_positive = int(len(assignments))
        false_positive = max(0, int(est_count) - true_positive)
        false_negative = max(0, int(gt_count) - true_positive)
        detection = _precision_recall_f1(
            true_positive,
            false_positive,
            false_negative,
        )
        key = f"{radius:g}m"
        if key in payload:
            raise ValueError("distance thresholds must be unique.")
        payload[key] = {
            "radius_m": radius,
            **detection,
        }
    return payload


def _hotspot_distance_summary(
    gt: Sequence[Source],
    est: Sequence[Source],
) -> Dict[str, Any]:
    """Return each estimated hotspot's distance to its nearest truth source."""
    if not est:
        return {
            **_summary_stats([]),
            "hotspot_count": 0,
            "scored_hotspot_count": 0,
            "unscored_hotspot_count": 0,
        }
    if not gt:
        return {
            **_summary_stats([]),
            "hotspot_count": int(len(est)),
            "scored_hotspot_count": 0,
            "unscored_hotspot_count": int(len(est)),
        }
    distances = _pairwise_distances(list(gt), list(est))
    nearest = np.min(distances, axis=0)
    return {
        **_summary_stats([float(value) for value in nearest]),
        "hotspot_count": int(len(est)),
        "scored_hotspot_count": int(len(est)),
        "unscored_hotspot_count": 0,
    }


def _close_pair_separation_summary(
    gt: Sequence[Source],
    est: Sequence[Source],
    assignments: Sequence[Tuple[int, int, float]],
    *,
    close_pair_distance_m: float,
    min_estimated_separation_m: float,
) -> Dict[str, Any]:
    """Score whether close truth pairs map to two sufficiently separate modes."""
    close_distance = _non_negative_finite(
        close_pair_distance_m,
        name="close-pair distance",
    )
    minimum_separation = _non_negative_finite(
        min_estimated_separation_m,
        name="minimum estimated close-pair separation",
    )
    count = len(gt)
    if count < 2:
        return {
            "close_pair_distance_m": close_distance,
            "min_estimated_separation_m": minimum_separation,
            "eligible_pair_count": 0,
            "matched_pair_count": 0,
            "separated_pair_count": 0,
            "separation_rate": None,
            "estimated_separation_m": _summary_stats([]),
            "separation_abs_error_m": _summary_abs([]),
            "estimated_to_truth_separation_ratio": _summary_stats([]),
            "pairs": [],
        }
    positions = np.vstack([source.pos for source in gt])
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    eligible = upper & (distances <= close_distance)
    eligible_i, eligible_j = np.nonzero(eligible)
    assignment_by_truth = {
        int(gt_index): int(est_index)
        for gt_index, est_index, _ in assignments
    }
    pair_rows: List[Dict[str, Any]] = []
    estimated_separations: List[float] = []
    separation_errors: List[float] = []
    separation_ratios: List[float] = []
    separated_count = 0
    matched_pair_count = 0
    for first, second in zip(eligible_i.tolist(), eligible_j.tolist()):
        first_est = assignment_by_truth.get(int(first))
        second_est = assignment_by_truth.get(int(second))
        truth_separation = float(distances[first, second])
        estimated_separation: float | None = None
        separation_error: float | None = None
        separation_ratio: float | None = None
        resolved = False
        if first_est is not None and second_est is not None:
            matched_pair_count += 1
            estimated_separation = float(
                np.linalg.norm(est[first_est].pos - est[second_est].pos)
            )
            if not np.isfinite(estimated_separation):
                raise ValueError("Estimated close-pair separation is non-finite.")
            separation_error = abs(estimated_separation - truth_separation)
            separation_ratio = (
                estimated_separation / truth_separation
                if truth_separation > 0.0
                else None
            )
            resolved = bool(estimated_separation >= minimum_separation)
            estimated_separations.append(estimated_separation)
            separation_errors.append(separation_error)
            if separation_ratio is not None:
                separation_ratios.append(separation_ratio)
            separated_count += int(resolved)
        pair_rows.append(
            {
                "first_gt_index": int(first),
                "second_gt_index": int(second),
                "first_est_index": first_est,
                "second_est_index": second_est,
                "truth_separation_m": truth_separation,
                "estimated_separation_m": estimated_separation,
                "separation_abs_error_m": separation_error,
                "estimated_to_truth_separation_ratio": separation_ratio,
                "resolved": resolved,
            }
        )
    eligible_count = int(eligible_i.size)
    return {
        "close_pair_distance_m": close_distance,
        "min_estimated_separation_m": minimum_separation,
        "eligible_pair_count": eligible_count,
        "matched_pair_count": int(matched_pair_count),
        "separated_pair_count": int(separated_count),
        "separation_rate": (
            separated_count / float(eligible_count) if eligible_count else None
        ),
        "estimated_separation_m": _summary_stats(estimated_separations),
        "separation_abs_error_m": _summary_abs(separation_errors),
        "estimated_to_truth_separation_ratio": _summary_stats(separation_ratios),
        "pairs": pair_rows,
    }


def _surface_classification_summary(
    gt: Sequence[Source],
    est: Sequence[Source],
    assignments: Sequence[Tuple[int, int, float]],
    *,
    diagnostics: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, Any]:
    """Return hard surface confusion and optional posterior proper scores."""
    labels = SURFACE_LABELS
    confusion = {
        truth: {prediction: 0 for prediction in (*labels, "other", "missed")}
        for truth in labels
    }
    eligible_truth = [
        index for index, source in enumerate(gt) if source.surface_kind in labels
    ]
    assignment_by_truth = {
        int(gt_index): int(est_index)
        for gt_index, est_index, _ in assignments
    }
    diagnostic_by_mode = {
        int(item.get("mode_index", index)): item
        for index, item in enumerate(diagnostics or ())
        if isinstance(item, Mapping)
    }
    correct = 0
    localized = 0
    ceiling_total = 0
    ceiling_localized = 0
    ceiling_correct = 0
    posterior_brier: List[float] = []
    posterior_log_scores: List[float] = []
    posterior_skipped_count = 0
    for gt_index in eligible_truth:
        truth = str(gt[gt_index].surface_kind)
        if truth == "ceiling":
            ceiling_total += 1
        est_index = assignment_by_truth.get(gt_index)
        if est_index is None:
            confusion[truth]["missed"] += 1
            continue
        localized += 1
        if truth == "ceiling":
            ceiling_localized += 1
        prediction_raw = est[est_index].surface_kind
        prediction = str(prediction_raw) if prediction_raw in labels else "other"
        confusion[truth][prediction] += 1
        if prediction == truth:
            correct += 1
            if truth == "ceiling":
                ceiling_correct += 1
        diagnostic = diagnostic_by_mode.get(est_index)
        if (
            diagnostic is None
            or diagnostic.get("surface_posterior_available") is False
        ):
            posterior_skipped_count += 1
            continue
        probabilities_raw = diagnostic.get("surface_kind_posterior")
        if not isinstance(probabilities_raw, Mapping):
            posterior_skipped_count += 1
            continue
        probabilities = np.asarray(
            [float(probabilities_raw.get(label, 0.0)) for label in labels],
            dtype=float,
        )
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError(
                "Surface posterior probabilities must be finite and non-negative."
            )
        probability_total = float(np.sum(probabilities))
        if probability_total <= 0.0:
            posterior_skipped_count += 1
            continue
        probabilities /= probability_total
        target = np.zeros(len(labels), dtype=float)
        truth_index = labels.index(truth)
        target[truth_index] = 1.0
        posterior_brier.append(float(np.sum((probabilities - target) ** 2)))
        posterior_log_scores.append(
            float(-np.log(max(float(probabilities[truth_index]), 1.0e-12)))
        )
    eligible_count = int(len(eligible_truth))
    return {
        "available": bool(eligible_count > 0),
        "labels": list(labels),
        "confusion_matrix": confusion,
        "eligible_truth_count": eligible_count,
        "localized_truth_count": int(localized),
        "correct_count": int(correct),
        "accuracy_among_localized": (correct / float(localized) if localized else None),
        "end_to_end_accuracy": (
            correct / float(eligible_count) if eligible_count else None
        ),
        "ceiling_truth_count": int(ceiling_total),
        "ceiling_localization_recall": (
            ceiling_localized / float(ceiling_total) if ceiling_total else None
        ),
        "ceiling_classification_recall": (
            ceiling_correct / float(ceiling_total) if ceiling_total else None
        ),
        "posterior_scoring": {
            "available": bool(posterior_brier),
            "evaluable_count": int(len(posterior_brier)),
            "skipped_localized_count": int(posterior_skipped_count),
            "multiclass_brier_score": (
                float(np.mean(posterior_brier)) if posterior_brier else None
            ),
            "negative_log_score": (
                float(np.mean(posterior_log_scores))
                if posterior_log_scores
                else None
            ),
            "brier_definition": "sum_over_surface_labels_of_squared_probability_error",
        },
        "room_surface_3class": _room_surface_3class_summary(
            gt,
            est,
            assignments,
        ),
    }


def _excluded_room_surface_category(surface_kind: str | None) -> str:
    """Map a non-room surface label to an explicit exclusion category."""
    if surface_kind is None or surface_kind in {"", "unknown"}:
        return "unknown"
    label = str(surface_kind).strip().lower()
    if label.startswith("obstacle"):
        return "obstacle"
    if label.startswith("reference"):
        return "reference"
    return "other"


def _room_surface_3class_summary(
    gt: Sequence[Source],
    est: Sequence[Source],
    assignments: Sequence[Tuple[int, int, float]],
) -> Dict[str, Any]:
    """Score floor/wall/ceiling only and document all excluded examples."""
    labels = ROOM_SURFACE_LABELS
    confusion = {
        truth: {prediction: 0 for prediction in labels} for truth in labels
    }
    per_class_counts = {
        label: {
            "truth_count": 0,
            "localized_count": 0,
            "evaluable_localized_count": 0,
            "correct_count": 0,
        }
        for label in labels
    }
    excluded_truth_by_label: Dict[str, int] = {}
    excluded_prediction_by_label: Dict[str, int] = {}
    excluded_truth_by_category = {
        "obstacle": 0,
        "reference": 0,
        "unknown": 0,
        "other": 0,
    }
    excluded_prediction_by_category = dict(excluded_truth_by_category)
    assignment_by_truth = {
        int(gt_index): int(est_index)
        for gt_index, est_index, _ in assignments
    }
    eligible_count = 0
    localized_count = 0
    evaluable_localized_count = 0
    correct_count = 0
    for gt_index, source in enumerate(gt):
        truth = source.surface_kind
        if truth not in labels:
            truth_label = "unknown" if truth is None else str(truth)
            excluded_truth_by_label[truth_label] = (
                excluded_truth_by_label.get(truth_label, 0) + 1
            )
            category = _excluded_room_surface_category(truth)
            excluded_truth_by_category[category] += 1
            continue
        truth = str(truth)
        eligible_count += 1
        per_class_counts[truth]["truth_count"] += 1
        est_index = assignment_by_truth.get(gt_index)
        if est_index is None:
            continue
        localized_count += 1
        per_class_counts[truth]["localized_count"] += 1
        prediction = est[est_index].surface_kind
        if prediction not in labels:
            prediction_label = (
                "unknown" if prediction is None else str(prediction)
            )
            excluded_prediction_by_label[prediction_label] = (
                excluded_prediction_by_label.get(prediction_label, 0) + 1
            )
            category = _excluded_room_surface_category(prediction)
            excluded_prediction_by_category[category] += 1
            continue
        prediction = str(prediction)
        evaluable_localized_count += 1
        per_class_counts[truth]["evaluable_localized_count"] += 1
        confusion[truth][prediction] += 1
        if prediction == truth:
            correct_count += 1
            per_class_counts[truth]["correct_count"] += 1
    per_class: Dict[str, Dict[str, Any]] = {}
    for label, counts in per_class_counts.items():
        truth_count = int(counts["truth_count"])
        evaluable = int(counts["evaluable_localized_count"])
        correct = int(counts["correct_count"])
        per_class[label] = {
            **counts,
            "recall": correct / float(truth_count) if truth_count else None,
            "accuracy_among_evaluable_localized": (
                correct / float(evaluable) if evaluable else None
            ),
        }
    return {
        "available": bool(eligible_count),
        "labels": list(labels),
        "confusion_matrix": confusion,
        "eligible_truth_count": int(eligible_count),
        "localized_truth_count": int(localized_count),
        "evaluable_localized_count": int(evaluable_localized_count),
        "correct_count": int(correct_count),
        "accuracy_among_evaluable_localized": (
            correct_count / float(evaluable_localized_count)
            if evaluable_localized_count
            else None
        ),
        "end_to_end_accuracy": (
            correct_count / float(eligible_count) if eligible_count else None
        ),
        "per_class": per_class,
        "excluded": {
            "truth_count": int(sum(excluded_truth_by_label.values())),
            "truth_count_by_label": excluded_truth_by_label,
            "truth_count_by_category": excluded_truth_by_category,
            "prediction_count": int(sum(excluded_prediction_by_label.values())),
            "prediction_count_by_label": excluded_prediction_by_label,
            "prediction_count_by_category": excluded_prediction_by_category,
            "scope": (
                "obstacle, reference, unknown, and other non-floor/wall/ceiling "
                "truth labels are excluded; non-three-class predictions for "
                "eligible truths are excluded from the 3x3 confusion matrix"
            ),
        },
    }


def _aggregate_room_surface_3class(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate floor/wall/ceiling classification summaries across isotopes."""
    labels = ROOM_SURFACE_LABELS
    confusion = {
        truth: {
            prediction: int(
                sum(
                    int(row["confusion_matrix"][truth][prediction])
                    for row in rows
                )
            )
            for prediction in labels
        }
        for truth in labels
    }
    count_keys = (
        "eligible_truth_count",
        "localized_truth_count",
        "evaluable_localized_count",
        "correct_count",
    )
    counts = {
        key: int(sum(int(row[key]) for row in rows)) for key in count_keys
    }
    per_class: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        truth_count = int(
            sum(int(row["per_class"][label]["truth_count"]) for row in rows)
        )
        localized_count = int(
            sum(int(row["per_class"][label]["localized_count"]) for row in rows)
        )
        evaluable_count = int(
            sum(
                int(row["per_class"][label]["evaluable_localized_count"])
                for row in rows
            )
        )
        correct_count = int(
            sum(int(row["per_class"][label]["correct_count"]) for row in rows)
        )
        per_class[label] = {
            "truth_count": truth_count,
            "localized_count": localized_count,
            "evaluable_localized_count": evaluable_count,
            "correct_count": correct_count,
            "recall": (
                correct_count / float(truth_count) if truth_count else None
            ),
            "accuracy_among_evaluable_localized": (
                correct_count / float(evaluable_count) if evaluable_count else None
            ),
        }

    def sum_dynamic_counts(section: str) -> Dict[str, int]:
        """Sum a dynamic excluded-count mapping across isotope summaries."""
        keys = sorted(
            {
                str(key)
                for row in rows
                for key in row["excluded"].get(section, {})
            }
        )
        return {
            key: int(
                sum(int(row["excluded"].get(section, {}).get(key, 0)) for row in rows)
            )
            for key in keys
        }

    excluded_truth_by_label = sum_dynamic_counts("truth_count_by_label")
    excluded_prediction_by_label = sum_dynamic_counts("prediction_count_by_label")
    excluded_truth_by_category = sum_dynamic_counts("truth_count_by_category")
    excluded_prediction_by_category = sum_dynamic_counts(
        "prediction_count_by_category"
    )
    for category in ("obstacle", "reference", "unknown", "other"):
        excluded_truth_by_category.setdefault(category, 0)
        excluded_prediction_by_category.setdefault(category, 0)
    return {
        "available": bool(counts["eligible_truth_count"]),
        "labels": list(labels),
        "confusion_matrix": confusion,
        **counts,
        "accuracy_among_evaluable_localized": (
            counts["correct_count"] / float(counts["evaluable_localized_count"])
            if counts["evaluable_localized_count"]
            else None
        ),
        "end_to_end_accuracy": (
            counts["correct_count"] / float(counts["eligible_truth_count"])
            if counts["eligible_truth_count"]
            else None
        ),
        "per_class": per_class,
        "excluded": {
            "truth_count": int(sum(excluded_truth_by_label.values())),
            "truth_count_by_label": excluded_truth_by_label,
            "truth_count_by_category": excluded_truth_by_category,
            "prediction_count": int(sum(excluded_prediction_by_label.values())),
            "prediction_count_by_label": excluded_prediction_by_label,
            "prediction_count_by_category": excluded_prediction_by_category,
            "scope": (
                "obstacle, reference, unknown, and other non-floor/wall/ceiling "
                "truth labels are excluded; non-three-class predictions for "
                "eligible truths are excluded from the 3x3 confusion matrix"
            ),
        },
    }


def _ellipsoid_contains_position(
    diagnostic: Mapping[str, Any],
    position: NDArray[np.float64],
) -> bool | None:
    """Return whether a position lies in a serialized posterior ellipsoid."""
    mean_raw = diagnostic.get("weighted_mean_xyz_m")
    ellipsoid = diagnostic.get("ellipsoid_90", {})
    if not isinstance(ellipsoid, Mapping):
        return None
    axes_raw = ellipsoid.get("semi_axis_lengths_m")
    orientation_raw = ellipsoid.get("orientation_matrix_xyz_by_axis")
    if mean_raw is None or axes_raw is None or orientation_raw is None:
        return None
    mean = np.asarray(mean_raw, dtype=float)
    axes = np.asarray(axes_raw, dtype=float)
    orientation = np.asarray(orientation_raw, dtype=float)
    if mean.shape != (3,) or axes.shape != (3,) or orientation.shape != (3, 3):
        return None
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(axes)):
        return None
    if np.any(~np.isfinite(orientation)) or np.any(axes < 0.0):
        return None
    coordinates = orientation.T @ (np.asarray(position, dtype=float) - mean)
    positive = axes > 1.0e-12
    if np.any(~positive & (np.abs(coordinates) > 1.0e-9)):
        return False
    normalized_sq = np.sum((coordinates[positive] / axes[positive]) ** 2)
    return bool(float(normalized_sq) <= 1.0 + 1.0e-9)


def _uncertainty_coverage_summary(
    gt: Sequence[Source],
    assignments: Sequence[Tuple[int, int, float]],
    diagnostics: Sequence[Mapping[str, Any]] | None,
    *,
    est_count: int,
) -> Dict[str, Any]:
    """Return conditional and end-to-end interval and existence calibration."""
    if diagnostics is None:
        return {
            "available": False,
            "conditioning": "requires posterior diagnostics",
            "ground_truth_count": int(len(gt)),
            "matched_truth_count": int(len(assignments)),
            "position_interval_evaluable_count": 0,
            "position_interval_covered_count": 0,
            "position_interval_90_coverage": None,
            "position_interval_90_coverage_conditioned_on_match": None,
            "position_interval_end_to_end_evaluable_count": 0,
            "position_interval_end_to_end_covered_count": 0,
            "position_interval_90_coverage_end_to_end": None,
            "z_interval_evaluable_count": 0,
            "z_interval_covered_count": 0,
            "z_interval_90_coverage": None,
            "z_interval_90_coverage_conditioned_on_match": None,
            "z_interval_end_to_end_evaluable_count": 0,
            "z_interval_end_to_end_covered_count": 0,
            "z_interval_90_coverage_end_to_end": None,
            "excluded_diagnostic_count_by_reason": {},
            "excluded_diagnostic_count": 0,
            "existence_mass_calibration": {
                "available": False,
                "evaluable_count": 0,
                "positive_label_count": 0,
                "negative_label_count": 0,
                "brier_score": None,
                "negative_log_score": None,
                "missing_mass_count": 0,
            },
        }
    by_mode = {
        int(item.get("mode_index", index)): item
        for index, item in enumerate(diagnostics)
        if isinstance(item, Mapping)
    }
    position_results: List[bool] = []
    z_results: List[bool] = []
    position_end_to_end: List[bool] = []
    z_end_to_end: List[bool] = []
    excluded_reasons: Dict[str, int] = {}
    assignment_by_truth = {
        int(gt_index): int(est_index)
        for gt_index, est_index, _ in assignments
    }
    matched_est = set(assignment_by_truth.values())
    matched_count = int(len(assignment_by_truth))

    def exclusion_reason(diagnostic: Mapping[str, Any]) -> str | None:
        """Return the explicit reason a posterior diagnostic is not evaluable."""
        if diagnostic.get("reference_consistent") is False:
            return "reference_inconsistent"
        if diagnostic.get("posterior_support_available") is False:
            return "posterior_support_unavailable"
        return None

    for gt_index, source in enumerate(gt):
        est_index = assignment_by_truth.get(gt_index)
        if est_index is None:
            position_end_to_end.append(False)
            z_end_to_end.append(False)
            continue
        diagnostic = by_mode.get(est_index)
        if diagnostic is None:
            excluded_reasons["missing_diagnostic"] = (
                excluded_reasons.get("missing_diagnostic", 0) + 1
            )
            position_end_to_end.append(False)
            z_end_to_end.append(False)
            continue
        reason = exclusion_reason(diagnostic)
        if reason is not None:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            continue
        inside = _ellipsoid_contains_position(diagnostic, source.pos)
        if inside is not None:
            position_results.append(bool(inside))
            position_end_to_end.append(bool(inside))
        else:
            position_end_to_end.append(False)
        quantiles = diagnostic.get("z_quantiles_m", {})
        z_inside: bool | None = None
        if isinstance(quantiles, Mapping):
            lower = quantiles.get("q05")
            upper = quantiles.get("q95")
            if lower is not None and upper is not None:
                lower_f = float(lower)
                upper_f = float(upper)
                if (
                    np.isfinite(lower_f)
                    and np.isfinite(upper_f)
                    and lower_f <= upper_f
                ):
                    z_value = float(source.pos[2])
                    z_inside = bool(lower_f <= z_value <= upper_f)
        if z_inside is not None:
            z_results.append(z_inside)
            z_end_to_end.append(z_inside)
        else:
            z_end_to_end.append(False)

    existence_probabilities: List[float] = []
    existence_labels: List[int] = []
    missing_mass_count = 0
    for est_index in range(max(int(est_count), 0)):
        diagnostic = by_mode.get(est_index)
        if diagnostic is None:
            missing_mass_count += 1
            continue
        if exclusion_reason(diagnostic) is not None:
            continue
        mass_raw = diagnostic.get("existence_mass")
        if mass_raw is None:
            missing_mass_count += 1
            continue
        mass = float(mass_raw)
        if not np.isfinite(mass) or not 0.0 <= mass <= 1.0:
            raise ValueError("Posterior existence mass must be finite and in [0, 1].")
        existence_probabilities.append(mass)
        existence_labels.append(int(est_index in matched_est))
    unmatched_truth_count = int(len(gt) - matched_count)
    existence_probabilities.extend([0.0] * unmatched_truth_count)
    existence_labels.extend([1] * unmatched_truth_count)
    if existence_probabilities:
        probability_array = np.asarray(existence_probabilities, dtype=float)
        label_array = np.asarray(existence_labels, dtype=float)
        brier = float(np.mean((probability_array - label_array) ** 2))
        clipped = np.clip(probability_array, 1.0e-12, 1.0 - 1.0e-12)
        negative_log = float(
            np.mean(
                -(label_array * np.log(clipped))
                - ((1.0 - label_array) * np.log(1.0 - clipped))
            )
        )
    else:
        brier = None
        negative_log = None
    return {
        "available": True,
        "conditioning": (
            "position_interval_90_coverage and z_interval_90_coverage are "
            "conditioned on a gated match and an evaluable posterior interval; "
            "end_to_end variants count unmatched truths and missing intervals as false"
        ),
        "ground_truth_count": int(len(gt)),
        "matched_truth_count": int(matched_count),
        "position_interval_evaluable_count": int(len(position_results)),
        "position_interval_covered_count": int(sum(position_results)),
        "position_interval_90_coverage": (
            float(np.mean(position_results)) if position_results else None
        ),
        "position_interval_90_coverage_conditioned_on_match": (
            float(np.mean(position_results)) if position_results else None
        ),
        "position_interval_end_to_end_evaluable_count": int(
            len(position_end_to_end)
        ),
        "position_interval_end_to_end_covered_count": int(
            sum(position_end_to_end)
        ),
        "position_interval_90_coverage_end_to_end": (
            float(np.mean(position_end_to_end)) if position_end_to_end else None
        ),
        "z_interval_evaluable_count": int(len(z_results)),
        "z_interval_covered_count": int(sum(z_results)),
        "z_interval_90_coverage": (float(np.mean(z_results)) if z_results else None),
        "z_interval_90_coverage_conditioned_on_match": (
            float(np.mean(z_results)) if z_results else None
        ),
        "z_interval_end_to_end_evaluable_count": int(len(z_end_to_end)),
        "z_interval_end_to_end_covered_count": int(sum(z_end_to_end)),
        "z_interval_90_coverage_end_to_end": (
            float(np.mean(z_end_to_end)) if z_end_to_end else None
        ),
        "excluded_diagnostic_count_by_reason": excluded_reasons,
        "excluded_diagnostic_count": int(sum(excluded_reasons.values())),
        "existence_mass_calibration": {
            "available": bool(existence_probabilities),
            "evaluable_count": int(len(existence_probabilities)),
            "positive_label_count": int(sum(existence_labels)),
            "negative_label_count": int(
                len(existence_labels) - sum(existence_labels)
            ),
            "brier_score": brier,
            "negative_log_score": negative_log,
            "missing_mass_count": int(missing_mass_count),
            "label_definition": (
                "one for a gated matched estimate or missed truth, zero for an "
                "unmatched estimated mode"
            ),
        },
    }


def _isotope_confusion_summary(
    gt_by_iso: Mapping[str, Sequence[Source]],
    est_by_iso: Mapping[str, Sequence[Source]],
    *,
    match_radius_m: float,
) -> Dict[str, Any]:
    """Return global spatial matching and the resulting isotope confusion matrix."""
    isotopes = sorted(set(gt_by_iso) | set(est_by_iso))
    gt_flat = [
        (isotope, source)
        for isotope in isotopes
        for source in gt_by_iso.get(isotope, ())
    ]
    est_flat = [
        (isotope, source)
        for isotope in isotopes
        for source in est_by_iso.get(isotope, ())
    ]
    matrix = {
        truth: {prediction: 0 for prediction in (*isotopes, "missed")}
        for truth in isotopes
    }
    matrix["false_positive"] = {prediction: 0 for prediction in (*isotopes, "missed")}
    if gt_flat and est_flat:
        gt_positions = np.vstack([source.pos for _, source in gt_flat])
        est_positions = np.vstack([source.pos for _, source in est_flat])
        distances = np.linalg.norm(
            gt_positions[:, None, :] - est_positions[None, :, :],
            axis=2,
        )
        assignments = _gated_distance_assignment(
            distances,
            radius_m=match_radius_m,
        )
    else:
        distances = np.zeros((len(gt_flat), len(est_flat)), dtype=float)
        assignments = []
    matched_gt: set[int] = set()
    matched_est: set[int] = set()
    correct = 0
    localized = 0
    for gt_index, est_index in assignments:
        truth = str(gt_flat[gt_index][0])
        prediction = str(est_flat[est_index][0])
        matrix[truth][prediction] += 1
        matched_gt.add(int(gt_index))
        matched_est.add(int(est_index))
        localized += 1
        correct += int(truth == prediction)
    for gt_index, (truth, _) in enumerate(gt_flat):
        if gt_index not in matched_gt:
            matrix[str(truth)]["missed"] += 1
    for est_index, (prediction, _) in enumerate(est_flat):
        if est_index not in matched_est:
            matrix["false_positive"][str(prediction)] += 1
    return {
        "labels": isotopes,
        "matrix": matrix,
        "localized_pair_count": int(localized),
        "correct_isotope_count": int(correct),
        "accuracy_among_localized": (correct / float(localized) if localized else None),
    }


def compute_metrics(
    gt_by_iso: Dict[str, List[Any]],
    est_by_iso: Dict[str, List[Any]],
    *,
    match_radius_m: float,
    distance_thresholds_m: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    match_strength_weight: float = 2.0,
    match_distance_weight: float = 1.0,
    outside_radius_penalty: float = 1e3,
    close_pair_distance_m: float = 2.0,
    close_pair_min_estimated_separation_m: float = 0.5,
    uncertainty_by_iso: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Dict[str, Any]:
    """Compute distance-gated source, surface, and uncertainty metrics.

    The standard association is predeclared as a distance-only maximum-cardinality
    gated Hungarian match.  The same association is used for both position and
    strength errors.  Historical strength-weight arguments remain accepted for
    API compatibility but do not influence the standard association.
    """
    eps = 1e-12
    match_radius = _non_negative_finite(match_radius_m, name="match_radius_m")
    close_pair_distance = _non_negative_finite(
        close_pair_distance_m,
        name="close_pair_distance_m",
    )
    close_pair_minimum = _non_negative_finite(
        close_pair_min_estimated_separation_m,
        name="close_pair_min_estimated_separation_m",
    )
    thresholds = [
        _non_negative_finite(value, name="distance threshold")
        for value in distance_thresholds_m
    ]
    legacy_parameters = {
        "match_strength_weight": _non_negative_finite(
            match_strength_weight,
            name="match_strength_weight",
        ),
        "match_distance_weight": _non_negative_finite(
            match_distance_weight,
            name="match_distance_weight",
        ),
        "outside_radius_penalty": _non_negative_finite(
            outside_radius_penalty,
            name="outside_radius_penalty",
        ),
    }
    isotopes = sorted(set(gt_by_iso.keys()) | set(est_by_iso.keys()))
    metrics: Dict[str, Dict[str, Any]] = {}
    normalized_gt: Dict[str, List[Source]] = {}
    normalized_est: Dict[str, List[Source]] = {}
    global_position_errors: List[float] = []
    global_xy_errors: List[float] = []
    global_z_errors: List[float] = []
    global_abs_strength_errors: List[float] = []
    global_rel_strength_errors: List[float] = []
    global_signed_rel_strength_errors: List[float] = []
    global_hotspot_distances: List[float] = []
    global_tp = 0
    global_gt_count = 0
    global_est_count = 0
    global_close_pairs = 0
    global_matched_close_pairs = 0
    global_separated_pairs = 0
    global_close_pair_rows: List[Dict[str, Any]] = []
    surface_labels = SURFACE_LABELS
    global_surface_confusion = {
        truth: {prediction: 0 for prediction in (*surface_labels, "other", "missed")}
        for truth in surface_labels
    }
    global_surface_eligible = 0
    global_surface_localized = 0
    global_surface_correct = 0
    global_ceiling_total = 0
    global_ceiling_localized = 0
    global_ceiling_correct = 0
    global_surface_posterior_count = 0
    global_surface_brier_sum = 0.0
    global_surface_log_sum = 0.0
    global_surface_posterior_skipped = 0
    isotope_bias: Dict[str, Dict[str, float | None]] = {}
    for iso in isotopes:
        gt = _normalize_sources(gt_by_iso.get(iso, []))
        est = _normalize_sources(est_by_iso.get(iso, []))
        normalized_gt[iso] = gt
        normalized_est[iso] = est
        dist = _pairwise_distances(gt, est)
        assigned_all_pairs = _hungarian_assignment(dist)
        matched_pairs = _gated_distance_assignment(dist, radius_m=match_radius)
        matched = [(i, j, float(dist[i, j])) for i, j in matched_pairs]
        assigned_all_details = [
            {
                "gt_index": int(i),
                "est_index": int(j),
                "distance": float(dist[i, j]),
                "within_radius": bool(float(dist[i, j]) <= match_radius),
            }
            for i, j in assigned_all_pairs
        ]
        fp = max(0, len(est) - len(matched))
        fn = max(0, len(gt) - len(matched))
        pos_errors = [d for _, _, d in matched]
        localized_tp = int(len(matched))
        localized_fp = max(0, len(est) - localized_tp)
        localized_fn = max(0, len(gt) - localized_tp)
        detection = _precision_recall_f1(
            localized_tp,
            localized_fp,
            localized_fn,
        )
        xy_errors: List[float] = []
        z_errors: List[float] = []
        abs_errors: List[float] = []
        rel_errors: List[float] = []
        signed_rel_errors: List[float] = []
        match_details: List[Dict[str, Any]] = []
        for i, j, d in matched:
            q_true = float(gt[i].strength)
            q_hat = float(est[j].strength)
            abs_err = abs(q_hat - q_true)
            rel_err = abs_err / max(q_true, eps) * 100.0
            signed_rel_err = (q_hat - q_true) / max(q_true, eps) * 100.0
            delta = np.asarray(est[j].pos - gt[i].pos, dtype=float)
            xy_err = float(np.linalg.norm(delta[:2]))
            z_err = float(abs(delta[2]))
            xy_errors.append(xy_err)
            z_errors.append(z_err)
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            signed_rel_errors.append(signed_rel_err)
            match_details.append(
                {
                    "gt_index": i,
                    "est_index": j,
                    "distance": d,
                    "xy_distance": xy_err,
                    "z_abs_error": z_err,
                    "within_radius": True,
                    "q_true": q_true,
                    "q_hat": q_hat,
                    "abs_err": abs_err,
                    "rel_err_pct": rel_err,
                    "signed_rel_err_pct": signed_rel_err,
                    "gt_surface_kind": gt[i].surface_kind,
                    "est_surface_kind": est[j].surface_kind,
                }
            )
        hotspot_summary = _hotspot_distance_summary(gt, est)
        if gt and est:
            global_hotspot_distances.extend(
                float(value) for value in np.min(dist, axis=0)
            )
        close_pair_summary = _close_pair_separation_summary(
            gt,
            est,
            matched,
            close_pair_distance_m=close_pair_distance,
            min_estimated_separation_m=close_pair_minimum,
        )
        isotope_diagnostics = (
            None
            if uncertainty_by_iso is None
            else uncertainty_by_iso.get(iso, ())
        )
        surface_summary = _surface_classification_summary(
            gt,
            est,
            matched,
            diagnostics=isotope_diagnostics,
        )
        uncertainty_summary = _uncertainty_coverage_summary(
            gt,
            matched,
            isotope_diagnostics,
            est_count=len(est),
        )
        gt_strength_total = _non_negative_finite(
            sum(source.strength for source in gt),
            name=f"{iso} ground-truth total strength",
        )
        est_strength_total = _non_negative_finite(
            sum(source.strength for source in est),
            name=f"{iso} estimated total strength",
        )
        total_strength_bias_pct = (
            100.0 * (est_strength_total - gt_strength_total) / gt_strength_total
            if gt_strength_total > eps
            else None
        )
        if total_strength_bias_pct is not None and not np.isfinite(
            total_strength_bias_pct
        ):
            raise ValueError("Total strength bias produced a non-finite value.")
        isotope_bias[iso] = {
            "ground_truth_total_cps_1m": gt_strength_total,
            "estimated_total_cps_1m": est_strength_total,
            "signed_total_bias_cps_1m": est_strength_total - gt_strength_total,
            "signed_total_bias_pct": total_strength_bias_pct,
        }
        metrics[iso] = {
            "counts": {
                "gt": len(gt),
                "est": len(est),
                "assigned": len(matched),
                "assigned_all": len(assigned_all_pairs),
                "matched": len(matched),
                "fp": fp,
                "fn": fn,
                "source_count_error": int(len(est) - len(gt)),
                "source_count_abs_error": int(abs(len(est) - len(gt))),
                "cardinality_exact_match": bool(len(est) == len(gt)),
                "localized_tp": localized_tp,
                "localized_fp": localized_fp,
                "localized_fn": localized_fn,
            },
            "detection": detection,
            "threshold_counts": _threshold_count_summary(
                gt_count=len(gt),
                est_count=len(est),
                distances=dist,
                thresholds_m=thresholds,
            ),
            "position_error": _position_error_summary(pos_errors),
            "xy_error": _summary_stats(xy_errors),
            "z_abs_error": _summary_abs(z_errors),
            "hotspot_to_ground_truth_distance": hotspot_summary,
            "intensity_abs_error": _summary_abs(abs_errors),
            "intensity_rel_error_pct": _summary_abs(rel_errors),
            "intensity_signed_rel_error_pct": _summary_signed(signed_rel_errors),
            "intensity_total_bias": isotope_bias[iso],
            "same_isotope_close_pair_separation": close_pair_summary,
            "surface_classification": surface_summary,
            "uncertainty_coverage": uncertainty_summary,
            "matches": match_details,
            "assigned_all_matches": assigned_all_details,
        }
        global_position_errors.extend(pos_errors)
        global_xy_errors.extend(xy_errors)
        global_z_errors.extend(z_errors)
        global_abs_strength_errors.extend(abs_errors)
        global_rel_strength_errors.extend(rel_errors)
        global_signed_rel_strength_errors.extend(signed_rel_errors)
        global_tp += localized_tp
        global_gt_count += len(gt)
        global_est_count += len(est)
        global_close_pairs += int(close_pair_summary["eligible_pair_count"])
        global_matched_close_pairs += int(close_pair_summary["matched_pair_count"])
        global_separated_pairs += int(close_pair_summary["separated_pair_count"])
        global_close_pair_rows.extend(
            {"isotope": iso, **row} for row in close_pair_summary["pairs"]
        )
        surface_confusion = surface_summary["confusion_matrix"]
        for truth in surface_labels:
            for prediction in (*surface_labels, "other", "missed"):
                global_surface_confusion[truth][prediction] += int(
                    surface_confusion[truth][prediction]
                )
        global_surface_eligible += int(surface_summary["eligible_truth_count"])
        global_surface_localized += int(surface_summary["localized_truth_count"])
        global_surface_correct += int(surface_summary["correct_count"])
        global_ceiling_total += int(surface_summary["ceiling_truth_count"])
        global_ceiling_localized += sum(
            int(surface_summary["confusion_matrix"]["ceiling"][key])
            for key in (*surface_labels, "other")
        )
        global_ceiling_correct += int(
            surface_summary["confusion_matrix"]["ceiling"]["ceiling"]
        )
        posterior_scoring = surface_summary["posterior_scoring"]
        posterior_count = int(posterior_scoring["evaluable_count"])
        global_surface_posterior_count += posterior_count
        global_surface_posterior_skipped += int(
            posterior_scoring["skipped_localized_count"]
        )
        if posterior_count:
            global_surface_brier_sum += posterior_count * float(
                posterior_scoring["multiclass_brier_score"]
            )
            global_surface_log_sum += posterior_count * float(
                posterior_scoring["negative_log_score"]
            )
    global_fp = max(0, global_est_count - global_tp)
    global_fn = max(0, global_gt_count - global_tp)
    uncertainty_rows = [
        payload["uncertainty_coverage"] for payload in metrics.values()
    ]
    uncertainty_count_keys = (
        "ground_truth_count",
        "matched_truth_count",
        "position_interval_evaluable_count",
        "position_interval_covered_count",
        "position_interval_end_to_end_evaluable_count",
        "position_interval_end_to_end_covered_count",
        "z_interval_evaluable_count",
        "z_interval_covered_count",
        "z_interval_end_to_end_evaluable_count",
        "z_interval_end_to_end_covered_count",
    )
    global_uncertainty_counts = {
        key: int(sum(int(row.get(key, 0)) for row in uncertainty_rows))
        for key in uncertainty_count_keys
    }
    excluded_reasons: Dict[str, int] = {}
    for row in uncertainty_rows:
        for reason, count in row.get(
            "excluded_diagnostic_count_by_reason",
            {},
        ).items():
            excluded_reasons[str(reason)] = (
                excluded_reasons.get(str(reason), 0) + int(count)
            )
    existence_rows = [row["existence_mass_calibration"] for row in uncertainty_rows]
    existence_count = int(
        sum(int(row["evaluable_count"]) for row in existence_rows)
    )
    existence_brier_sum = sum(
        int(row["evaluable_count"]) * float(row["brier_score"])
        for row in existence_rows
        if int(row["evaluable_count"]) > 0
    )
    existence_log_sum = sum(
        int(row["evaluable_count"]) * float(row["negative_log_score"])
        for row in existence_rows
        if int(row["evaluable_count"]) > 0
    )

    def coverage_rate(covered_key: str, evaluable_key: str) -> float | None:
        """Return a global coverage rate for aggregate integer counters."""
        denominator = global_uncertainty_counts[evaluable_key]
        return (
            global_uncertainty_counts[covered_key] / float(denominator)
            if denominator
            else None
        )

    global_uncertainty = {
        "available": any(bool(row.get("available", False)) for row in uncertainty_rows),
        "conditioning": (
            "conditional fields require a gated match and evaluable posterior; "
            "end_to_end fields count misses and missing intervals as false"
        ),
        "position_interval_interpretation": (
            "Gaussian-equivalent 90% covariance ellipsoid scaled by the "
            "three-dimensional chi-square 0.9 quantile; it is not an empirical "
            "highest-posterior-density credible region"
        ),
        **global_uncertainty_counts,
        "position_interval_90_coverage": coverage_rate(
            "position_interval_covered_count",
            "position_interval_evaluable_count",
        ),
        "position_interval_90_coverage_conditioned_on_match": coverage_rate(
            "position_interval_covered_count",
            "position_interval_evaluable_count",
        ),
        "position_interval_90_coverage_end_to_end": coverage_rate(
            "position_interval_end_to_end_covered_count",
            "position_interval_end_to_end_evaluable_count",
        ),
        "z_interval_90_coverage": coverage_rate(
            "z_interval_covered_count",
            "z_interval_evaluable_count",
        ),
        "z_interval_90_coverage_conditioned_on_match": coverage_rate(
            "z_interval_covered_count",
            "z_interval_evaluable_count",
        ),
        "z_interval_90_coverage_end_to_end": coverage_rate(
            "z_interval_end_to_end_covered_count",
            "z_interval_end_to_end_evaluable_count",
        ),
        "excluded_diagnostic_count_by_reason": excluded_reasons,
        "excluded_diagnostic_count": int(sum(excluded_reasons.values())),
        "existence_mass_calibration": {
            "available": bool(existence_count),
            "evaluable_count": existence_count,
            "positive_label_count": int(
                sum(int(row["positive_label_count"]) for row in existence_rows)
            ),
            "negative_label_count": int(
                sum(int(row["negative_label_count"]) for row in existence_rows)
            ),
            "brier_score": (
                float(existence_brier_sum / existence_count)
                if existence_count
                else None
            ),
            "negative_log_score": (
                float(existence_log_sum / existence_count)
                if existence_count
                else None
            ),
            "missing_mass_count": int(
                sum(int(row["missing_mass_count"]) for row in existence_rows)
            ),
        },
    }
    evaluable_cardinality_isotopes = [
        isotope
        for isotope, payload in metrics.items()
        if int(payload["counts"]["gt"]) > 0 or int(payload["counts"]["est"]) > 0
    ]
    excluded_empty_cardinality_isotopes = sorted(
        set(metrics) - set(evaluable_cardinality_isotopes)
    )
    exact_cardinality_count = sum(
        int(bool(payload["counts"]["cardinality_exact_match"]))
        for isotope, payload in metrics.items()
        if isotope in evaluable_cardinality_isotopes
    )
    cardinality_errors = [
        int(payload["counts"]["source_count_error"]) for payload in metrics.values()
    ]
    close_pair_estimated = [
        float(row["estimated_separation_m"])
        for row in global_close_pair_rows
        if row["estimated_separation_m"] is not None
    ]
    close_pair_errors = [
        float(row["separation_abs_error_m"])
        for row in global_close_pair_rows
        if row["separation_abs_error_m"] is not None
    ]
    close_pair_ratios = [
        float(row["estimated_to_truth_separation_ratio"])
        for row in global_close_pair_rows
        if row["estimated_to_truth_separation_ratio"] is not None
    ]
    global_room_surface_3class = _aggregate_room_surface_3class(
        [
            payload["surface_classification"]["room_surface_3class"]
            for payload in metrics.values()
        ]
    )
    cardinality_available = bool(evaluable_cardinality_isotopes)
    run_cardinality_exact = (
        bool(exact_cardinality_count == len(evaluable_cardinality_isotopes))
        if cardinality_available
        else None
    )
    global_summary = {
        "available": bool(isotopes),
        "position_error": _position_error_summary(global_position_errors),
        "xy_error": _summary_stats(global_xy_errors),
        "z_abs_error": _summary_abs(global_z_errors),
        "hotspot_to_ground_truth_distance": {
            **_summary_stats(global_hotspot_distances),
            "hotspot_count": int(global_est_count),
            "scored_hotspot_count": int(len(global_hotspot_distances)),
            "unscored_hotspot_count": int(
                global_est_count - len(global_hotspot_distances)
            ),
        },
        "intensity_abs_error": _summary_abs(global_abs_strength_errors),
        "intensity_rel_error_pct": _summary_abs(global_rel_strength_errors),
        "intensity_signed_rel_error_pct": _summary_signed(
            global_signed_rel_strength_errors
        ),
        "isotope_strength_bias": isotope_bias,
        "cardinality": {
            "available": cardinality_available,
            "isotope_count": int(len(isotopes)),
            "evaluable_isotope_count": int(len(evaluable_cardinality_isotopes)),
            "evaluable_isotopes": sorted(evaluable_cardinality_isotopes),
            "excluded_empty_empty_isotope_count": int(
                len(excluded_empty_cardinality_isotopes)
            ),
            "excluded_empty_empty_isotopes": excluded_empty_cardinality_isotopes,
            "exact_match_isotope_count": int(exact_cardinality_count),
            "exact_match_rate": (
                float(run_cardinality_exact)
                if run_cardinality_exact is not None
                else None
            ),
            "exact_match_rate_definition": (
                "single_run_exact_cardinality_indicator_across_evaluable_isotopes"
            ),
            "per_evaluable_isotope_exact_match_rate": (
                exact_cardinality_count / float(len(evaluable_cardinality_isotopes))
                if cardinality_available
                else None
            ),
            "all_isotopes_exact_match": (
                run_cardinality_exact
            ),
            "ground_truth_source_count": int(global_gt_count),
            "estimated_source_count": int(global_est_count),
            "source_count_error": int(global_est_count - global_gt_count),
            "source_count_abs_error": int(sum(abs(value) for value in cardinality_errors)),
            "overestimated_source_count": int(
                sum(max(value, 0) for value in cardinality_errors)
            ),
            "underestimated_source_count": int(
                sum(max(-value, 0) for value in cardinality_errors)
            ),
            "overestimated_isotope_count": int(
                sum(int(value > 0) for value in cardinality_errors)
            ),
            "underestimated_isotope_count": int(
                sum(int(value < 0) for value in cardinality_errors)
            ),
        },
        "detection": {
            **_precision_recall_f1(global_tp, global_fp, global_fn),
            "match_radius_m": match_radius,
            "false_source_count": int(global_fp),
        },
        "same_isotope_close_pair_separation": {
            "close_pair_distance_m": close_pair_distance,
            "min_estimated_separation_m": close_pair_minimum,
            "eligible_pair_count": int(global_close_pairs),
            "matched_pair_count": int(global_matched_close_pairs),
            "separated_pair_count": int(global_separated_pairs),
            "separation_rate": (
                global_separated_pairs / float(global_close_pairs)
                if global_close_pairs
                else None
            ),
            "estimated_separation_m": _summary_stats(close_pair_estimated),
            "separation_abs_error_m": _summary_abs(close_pair_errors),
            "estimated_to_truth_separation_ratio": _summary_stats(
                close_pair_ratios
            ),
            "pairs": global_close_pair_rows,
        },
        "surface_classification": {
            "available": bool(global_surface_eligible > 0),
            "labels": list(surface_labels),
            "confusion_matrix": global_surface_confusion,
            "eligible_truth_count": int(global_surface_eligible),
            "localized_truth_count": int(global_surface_localized),
            "correct_count": int(global_surface_correct),
            "accuracy_among_localized": (
                global_surface_correct / float(global_surface_localized)
                if global_surface_localized
                else None
            ),
            "end_to_end_accuracy": (
                global_surface_correct / float(global_surface_eligible)
                if global_surface_eligible
                else None
            ),
            "ceiling_truth_count": int(global_ceiling_total),
            "ceiling_localization_recall": (
                global_ceiling_localized / float(global_ceiling_total)
                if global_ceiling_total
                else None
            ),
            "ceiling_classification_recall": (
                global_ceiling_correct / float(global_ceiling_total)
                if global_ceiling_total
                else None
            ),
            "posterior_scoring": {
                "available": bool(global_surface_posterior_count),
                "evaluable_count": int(global_surface_posterior_count),
                "skipped_localized_count": int(global_surface_posterior_skipped),
                "multiclass_brier_score": (
                    global_surface_brier_sum / float(global_surface_posterior_count)
                    if global_surface_posterior_count
                    else None
                ),
                "negative_log_score": (
                    global_surface_log_sum / float(global_surface_posterior_count)
                    if global_surface_posterior_count
                    else None
                ),
            },
            "room_surface_3class": global_room_surface_3class,
        },
        "uncertainty_coverage": global_uncertainty,
        "isotope_confusion": _isotope_confusion_summary(
            normalized_gt,
            normalized_est,
            match_radius_m=match_radius,
        ),
    }
    return {
        "available": bool(isotopes),
        "match_radius_m": match_radius,
        "close_pair_distance_m": close_pair_distance,
        "close_pair_min_estimated_separation_m": close_pair_minimum,
        "matching_policy": {
            "algorithm": "maximum_cardinality_distance_only_gated_hungarian",
            "distance": "euclidean_3d_m",
            "strength_used_for_assignment": False,
            "outside_radius_behavior": "unmatched",
            "standard_match_radius_m": match_radius,
            "threshold_matching": "recomputed_independently_at_each_threshold",
            "position_and_strength_use_same_assignment": True,
            "assigned_all_interpretation": (
                "optional ungated distance-only diagnostic; excluded from standard errors"
            ),
            "legacy_assignment_parameters_ignored": legacy_parameters,
        },
        "global": global_summary,
        "isotopes": metrics,
    }


def _format_value(value: float | None) -> str:
    """Format a numeric value or return 'n/a'."""
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable metrics report."""
    isotopes = metrics.get("isotopes", {})
    print("=== Evaluation Metrics (PF Final) ===")
    for iso in sorted(isotopes.keys()):
        data = isotopes[iso]
        counts = data["counts"]
        assigned = counts.get("assigned", None)
        pos_err = data["position_error"]
        abs_err = data["intensity_abs_error"]
        rel_err = data["intensity_rel_error_pct"]
        print(f"\n[Isotope: {iso}]")
        assigned_msg = ""
        if assigned is not None:
            assigned_msg = f", Assigned={assigned}"
        print(
            f"  GT={counts['gt']}, EST={counts['est']}{assigned_msg}, "
            f"TP={counts['matched']}, FP={counts['fp']}, FN={counts['fn']}"
        )
        print(
            "  Position error [m]: "
            f"mean={_format_value(pos_err['mean'])}, "
            f"median={_format_value(pos_err['median'])}, "
            f"p95={_format_value(pos_err.get('p95'))}, "
            f"rmse={_format_value(pos_err['rmse'])}, "
            f"max={_format_value(pos_err['max'])}"
        )
        xy_err = data.get("xy_error", {})
        z_err = data.get("z_abs_error", {})
        print(
            "  Axis errors [m]: "
            f"xy_median={_format_value(xy_err.get('median'))}, "
            f"xy_p95={_format_value(xy_err.get('p95'))}, "
            f"z_median={_format_value(z_err.get('median'))}, "
            f"z_p95={_format_value(z_err.get('p95'))}"
        )
        print(
            "  Position target [m]: "
            f"<={_format_value(pos_err.get('target_m'))}, "
            f"within_target={pos_err.get('within_target')}"
        )
        threshold_counts = data.get("threshold_counts", {})
        if threshold_counts:
            threshold_msg = ", ".join(
                f"{key}:P={value['precision']:.2f}/R={value['recall']:.2f}"
                for key, value in sorted(
                    threshold_counts.items(),
                    key=lambda item: float(item[1]["radius_m"]),
                )
            )
            print(f"  Threshold precision/recall: {threshold_msg}")
        print(
            "  Intensity abs error [cps@1m]: "
            f"mean={_format_value(abs_err['mean'])}, "
            f"median={_format_value(abs_err['median'])}, "
            f"max={_format_value(abs_err['max'])}"
        )
        print(
            "  Intensity rel error [%]: "
            f"mean={_format_value(rel_err['mean'])}, "
            f"median={_format_value(rel_err['median'])}, "
            f"p95={_format_value(rel_err.get('p95'))}, "
            f"max={_format_value(rel_err['max'])}"
        )
        matches = data.get("matches", [])
        if not matches:
            print("  Matches: none")
            continue
        print("  Matches:")
        for m in matches:
            print(
                "    "
                f"GT#{m['gt_index']} -> EST#{m['est_index']} : "
                f"d={m['distance']:.3f} m, "
                f"q_true={m['q_true']:.3f}, q_hat={m['q_hat']:.3f}, "
                f"abs={m['abs_err']:.3f}, rel={m['rel_err_pct']:.2f}%"
            )


def save_metrics_json(metrics: Dict[str, Any], out_path: str) -> None:
    """Save metrics to a JSON file."""
    import json
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
