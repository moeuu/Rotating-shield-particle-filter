"""Run-level diagnostics that complement source-matching accuracy metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float or ``None`` for JSON-safe diagnostics."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return count, mean, median, p95, and maximum for finite values."""
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _grouped_bias_rows(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    group_ids: NDArray[Any],
) -> tuple[NDArray[Any], list[dict[str, float | int | None]]]:
    """Aggregate signed prediction bias for arbitrary one-dimensional groups."""
    unique, inverse = np.unique(group_ids, return_inverse=True)
    group_count = int(unique.size)
    row_count = np.bincount(inverse, minlength=group_count).astype(np.int64)
    observed_total = np.bincount(
        inverse,
        weights=observed,
        minlength=group_count,
    )
    predicted_total = np.bincount(
        inverse,
        weights=predicted,
        minlength=group_count,
    )
    residual = predicted - observed
    residual_total = np.bincount(
        inverse,
        weights=residual,
        minlength=group_count,
    )
    absolute_total = np.bincount(
        inverse,
        weights=np.abs(residual),
        minlength=group_count,
    )
    squared_total = np.bincount(
        inverse,
        weights=residual * residual,
        minlength=group_count,
    )
    aggregate_arrays = (
        observed_total,
        predicted_total,
        residual_total,
        absolute_total,
        squared_total,
    )
    if any(np.any(~np.isfinite(array)) for array in aggregate_arrays):
        raise ValueError("Count-bias aggregation produced a non-finite value.")
    relative_bias_pct = np.divide(
        100.0 * residual_total,
        observed_total,
        out=np.full(group_count, np.nan, dtype=float),
        where=observed_total > 0.0,
    )
    rows = [
        {
            "row_count": int(row_count[index]),
            "observed_total_counts": float(observed_total[index]),
            "predicted_total_counts": float(predicted_total[index]),
            "signed_bias_counts": float(residual_total[index]),
            "signed_relative_bias_pct": _finite_or_none(relative_bias_pct[index]),
            "mean_signed_residual_counts": float(
                residual_total[index] / max(int(row_count[index]), 1)
            ),
            "mean_absolute_residual_counts": float(
                absolute_total[index] / max(int(row_count[index]), 1)
            ),
            "residual_rmse_counts": float(
                np.sqrt(squared_total[index] / max(int(row_count[index]), 1))
            ),
        }
        for index in range(group_count)
    ]
    return unique, rows


def summarize_count_bias(
    observed_counts: Sequence[float] | NDArray[np.float64],
    predicted_counts: Sequence[float] | NDArray[np.float64],
    isotope_labels: Sequence[str] | NDArray[np.str_],
    fe_indices: Sequence[int] | NDArray[np.int64],
    pb_indices: Sequence[int] | NDArray[np.int64],
    *,
    num_orientations: int,
    count_regime_lower_edges: Sequence[float] = (0.0, 10.0, 100.0, 1000.0),
    regime_reference_counts: Sequence[float] | NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Summarize in-sample residuals by isotope, shield pair, and count regime.

    Residual is defined as ``predicted - observed``.  The signed relative value uses
    the sum of observed counts as its denominator, so Poisson fluctuations are
    not amplified row by row in low-count regimes.  Regimes use predicted counts
    by default so observed Poisson fluctuations do not select their own bins.
    """
    observed = np.asarray(observed_counts, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_counts, dtype=float).reshape(-1)
    isotopes = np.asarray(isotope_labels, dtype=str).reshape(-1)
    fe = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
    pb = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
    size = int(observed.size)
    if any(array.size != size for array in (predicted, isotopes, fe, pb)):
        raise ValueError("all count-bias inputs must have the same length.")
    if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(predicted)):
        raise ValueError("observed and predicted counts must be finite.")
    if np.any(observed < 0.0) or np.any(predicted < 0.0):
        raise ValueError("observed and predicted counts must be non-negative.")
    orientations = int(num_orientations)
    if orientations <= 0:
        raise ValueError("num_orientations must be positive.")
    if np.any(fe < 0) or np.any(fe >= orientations):
        raise ValueError("fe_indices must be within [0, num_orientations).")
    if np.any(pb < 0) or np.any(pb >= orientations):
        raise ValueError("pb_indices must be within [0, num_orientations).")
    pair_ids = fe * orientations + pb
    edges = np.asarray(count_regime_lower_edges, dtype=float).reshape(-1)
    if edges.size == 0 or np.any(~np.isfinite(edges)):
        raise ValueError("count_regime_lower_edges must contain finite values.")
    if edges[0] > 0.0:
        edges = np.concatenate([np.array([0.0], dtype=float), edges])
    if np.any(np.diff(edges) <= 0.0) or edges[0] < 0.0:
        raise ValueError(
            "count regime lower edges must be non-negative and increasing."
        )
    if regime_reference_counts is None:
        regime_reference = predicted
        regime_reference_name = "predicted_counts"
    else:
        regime_reference = np.asarray(
            regime_reference_counts,
            dtype=float,
        ).reshape(-1)
        regime_reference_name = "provided_reference_counts"
        if regime_reference.size != size:
            raise ValueError("regime_reference_counts must match the count length.")
        if (
            np.any(~np.isfinite(regime_reference))
            or np.any(regime_reference < 0.0)
        ):
            raise ValueError(
                "regime_reference_counts must be finite and non-negative."
            )
    regime_ids = np.searchsorted(edges, regime_reference, side="right") - 1

    isotope_keys, isotope_rows = _grouped_bias_rows(observed, predicted, isotopes)
    pair_keys, pair_rows = _grouped_bias_rows(observed, predicted, pair_ids)
    regime_keys, regime_rows = _grouped_bias_rows(observed, predicted, regime_ids)
    isotope_pair_ids = np.char.add(
        np.char.add(isotopes.astype(str), "|"),
        pair_ids.astype(str),
    )
    isotope_pair_keys, isotope_pair_rows = _grouped_bias_rows(
        observed,
        predicted,
        isotope_pair_ids,
    )
    isotope_regime_ids = np.char.add(
        np.char.add(isotopes.astype(str), "|"),
        regime_ids.astype(str),
    )
    isotope_regime_keys, isotope_regime_rows = _grouped_bias_rows(
        observed,
        predicted,
        isotope_regime_ids,
    )
    total_keys, total_rows = _grouped_bias_rows(
        observed,
        predicted,
        np.zeros(size, dtype=np.int64),
    )
    del total_keys
    by_isotope = {
        str(key): row for key, row in zip(isotope_keys.tolist(), isotope_rows)
    }
    by_shield_pair: dict[str, dict[str, Any]] = {}
    for pair_id_raw, row in zip(pair_keys.tolist(), pair_rows):
        pair_id = int(pair_id_raw)
        fe_index = pair_id // orientations
        pb_index = pair_id % orientations
        by_shield_pair[str(pair_id)] = {
            "pair_id": pair_id,
            "fe_index": int(fe_index),
            "pb_index": int(pb_index),
            **row,
        }
    by_count_regime: dict[str, dict[str, Any]] = {}
    for regime_id_raw, row in zip(regime_keys.tolist(), regime_rows):
        regime_id = int(regime_id_raw)
        lower = float(edges[regime_id])
        upper = float(edges[regime_id + 1]) if regime_id + 1 < edges.size else None
        label = f"[{lower:g},{upper:g})" if upper is not None else f"[{lower:g},inf)"
        by_count_regime[label] = {
            "lower_inclusive_counts": lower,
            "upper_exclusive_counts": upper,
            **row,
        }
    by_isotope_and_shield_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for composite, row in zip(isotope_pair_keys.tolist(), isotope_pair_rows):
        isotope, pair_id_text = str(composite).rsplit("|", maxsplit=1)
        pair_id = int(pair_id_text)
        by_isotope_and_shield_pair.setdefault(isotope, {})[str(pair_id)] = {
            "pair_id": pair_id,
            "fe_index": int(pair_id // orientations),
            "pb_index": int(pair_id % orientations),
            **row,
        }
    by_isotope_and_count_regime: dict[str, dict[str, dict[str, Any]]] = {}
    for composite, row in zip(isotope_regime_keys.tolist(), isotope_regime_rows):
        isotope, regime_id_text = str(composite).rsplit("|", maxsplit=1)
        regime_id = int(regime_id_text)
        lower = float(edges[regime_id])
        upper = float(edges[regime_id + 1]) if regime_id + 1 < edges.size else None
        label = f"[{lower:g},{upper:g})" if upper is not None else f"[{lower:g},inf)"
        by_isotope_and_count_regime.setdefault(isotope, {})[label] = {
            "lower_inclusive_counts": lower,
            "upper_exclusive_counts": upper,
            **row,
        }
    observed_pair_ids = sorted(int(value) for value in pair_keys.tolist())
    expected_pair_count = orientations * orientations
    missing_pair_ids = sorted(set(range(expected_pair_count)) - set(observed_pair_ids))
    missing_pairs = [
        {
            "pair_id": pair_id,
            "fe_index": int(pair_id // orientations),
            "pb_index": int(pair_id % orientations),
        }
        for pair_id in missing_pair_ids
    ]
    return {
        "available": bool(size > 0),
        "comparison": "final_forward_prediction_vs_observed_unfolded_counts",
        "diagnostic_scope": "in_sample_final_fit_residual",
        "calibration_bias_evidence": False,
        "calibration_warning": (
            "These residuals reuse fitted observations and do not establish "
            "calibration bias; use an independent calibration or held-out set."
        ),
        "bias_definition": "predicted_minus_observed_in_sample_residual",
        "relative_bias_denominator": "group_observed_total_counts",
        "count_regime_reference": regime_reference_name,
        "row_count": size,
        "shield_pair_coverage": {
            "num_orientations": orientations,
            "expected_pair_count": int(expected_pair_count),
            "observed_pair_count": int(len(observed_pair_ids)),
            "coverage_fraction": (
                len(observed_pair_ids) / float(expected_pair_count)
            ),
            "observed_pair_ids": observed_pair_ids,
            "missing_pair_count": int(len(missing_pair_ids)),
            "missing_pair_ids": missing_pair_ids,
            "missing_pairs": missing_pairs,
        },
        "overall": total_rows[0] if total_rows else None,
        "by_isotope": by_isotope,
        "by_shield_pair": by_shield_pair,
        "by_count_regime": by_count_regime,
        "by_isotope_and_shield_pair": by_isotope_and_shield_pair,
        "by_isotope_and_count_regime": by_isotope_and_count_regime,
    }


def _selected_heldout_deviance(payload: Mapping[str, Any]) -> float | None:
    """Return held-out spectrum deviance at the selected source count."""
    values = payload.get("heldout_deviance_by_count", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    try:
        selected = int(payload.get("selected_count", -1))
    except (TypeError, ValueError):
        return None
    if selected < 0 or selected >= len(values):
        return None
    selected_value = _finite_or_none(values[selected])
    if selected_value is None or selected_value < 0.0:
        return None
    return selected_value


def _integer_or_none(value: Any, *, minimum: int = 0) -> int | None:
    """Return an integer at or above ``minimum``, otherwise ``None``."""
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(numeric) or numeric != integer or integer < minimum:
        return None
    return integer


def _finite_deviance_sequence(payload: Mapping[str, Any]) -> list[float | None]:
    """Return a strict-JSON sequence of finite non-negative deviances."""
    values = payload.get("heldout_deviance_by_count", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result: list[float | None] = []
    for value in values:
        numeric = _finite_or_none(value)
        result.append(numeric if numeric is not None and numeric >= 0.0 else None)
    return result


def summarize_model_diagnostics(
    report_model_order: Mapping[str, Mapping[str, Any]] | None,
    sparse_poisson_evidence: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Extract BIC margins, response conditioning, and held-out deviance."""
    report = {} if report_model_order is None else dict(report_model_order)
    sparse = {} if sparse_poisson_evidence is None else dict(sparse_poisson_evidence)
    isotope_names = sorted(
        set(str(key) for key in report)
        | {str(key) for key in sparse if str(key) != "joint_multi_isotope"}
    )
    margins: list[float] = []
    conditions: list[float] = []
    heldout: list[float] = []
    normalized_heldout: list[float] = []
    by_isotope: dict[str, dict[str, Any]] = {}
    for isotope in isotope_names:
        report_row = report.get(isotope, {})
        sparse_row_raw = sparse.get(isotope, {})
        sparse_available = bool(sparse_row_raw) and bool(
            sparse_row_raw.get("available", True)
        )
        sparse_row = sparse_row_raw if sparse_available else {}
        report_margin = _finite_or_none(report_row.get("criterion_margin_to_runner_up"))
        sparse_margin = _finite_or_none(sparse_row.get("bic_margin_to_runner_up"))
        condition = _finite_or_none(
            sparse_row.get("condition_number", report_row.get("condition_number"))
        )
        selected_heldout = _selected_heldout_deviance(sparse_row)
        observation_count = _integer_or_none(
            sparse_row.get("heldout_observation_count"),
            minimum=1,
        )
        normalized_selected_heldout = (
            selected_heldout / float(observation_count)
            if selected_heldout is not None and observation_count is not None
            else None
        )
        margin = sparse_margin if sparse_margin is not None else report_margin
        if margin is not None:
            margins.append(margin)
        if condition is not None:
            conditions.append(condition)
        if selected_heldout is not None:
            heldout.append(selected_heldout)
        if normalized_selected_heldout is not None:
            normalized_heldout.append(normalized_selected_heldout)
        sparse_selected_count = _integer_or_none(sparse_row.get("selected_count"))
        report_selected_count = _integer_or_none(report_row.get("selected_count"))
        selected_count = (
            sparse_selected_count
            if sparse_selected_count is not None
            else report_selected_count
        )
        by_isotope[isotope] = {
            "available": bool(
                margin is not None
                or condition is not None
                or selected_heldout is not None
            ),
            "sparse_evidence_available": sparse_available,
            "selected_count": selected_count,
            "bic_margin_to_runner_up": margin,
            "report_criterion_margin_to_runner_up": report_margin,
            "sparse_bic_margin_to_runner_up": sparse_margin,
            "response_condition_number": condition,
            "selected_spectrum_bin_heldout_deviance": selected_heldout,
            "selected_spectrum_bin_heldout_deviance_per_observation": (
                normalized_selected_heldout
            ),
            "heldout_deviance_observation_count": observation_count,
            "heldout_deviance_per_observation_available": bool(
                normalized_selected_heldout is not None
            ),
            "heldout_deviance_normalization": (
                "selected_heldout_deviance_divided_by_actual_heldout_observation_count"
            ),
            "best_heldout_count": _integer_or_none(
                sparse_row.get("best_heldout_count")
            ),
            "heldout_deviance_by_count": _finite_deviance_sequence(sparse_row),
        }
    joint = sparse.get("joint_multi_isotope", {})
    joint_available = bool(joint.get("available", False))
    joint_cardinality_key = joint.get("selected_cardinality_key")
    isotope_metric_available = any(
        bool(row.get("available", False)) for row in by_isotope.values()
    )
    return {
        "available": isotope_metric_available or joint_available,
        "by_isotope": by_isotope,
        "bic_margin_to_runner_up": _distribution_summary(margins),
        "response_condition_number": _distribution_summary(conditions),
        "spectrum_bin_heldout_deviance": _distribution_summary(heldout),
        "spectrum_bin_heldout_deviance_per_observation": _distribution_summary(
            normalized_heldout
        ),
        "joint_multi_isotope": {
            "available": joint_available,
            "selected_cardinality_key": (
                str(joint_cardinality_key)
                if joint_available and joint_cardinality_key is not None
                else None
            ),
            "bic_margin_to_runner_up": (
                _finite_or_none(joint.get("bic_margin_to_runner_up"))
                if joint_available
                else None
            ),
        },
    }


def _estimate_state(
    estimate: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Extract positions and optional strengths from one history estimate."""
    strength_raw: Any = None
    if isinstance(estimate, Mapping):
        position_raw = estimate.get("positions", estimate.get("position", ()))
        strength_raw = estimate.get("strengths", estimate.get("strength"))
    elif isinstance(estimate, tuple):
        position_raw = estimate[0] if len(estimate) >= 1 else ()
        strength_raw = estimate[1] if len(estimate) >= 2 else None
    else:
        position_raw = estimate
    positions = np.asarray(position_raw, dtype=float)
    if positions.size == 0:
        normalized_positions = np.zeros((0, 3), dtype=float)
    else:
        if positions.size % 3 != 0:
            raise ValueError("History estimate positions must have three coordinates.")
        normalized_positions = positions.reshape(-1, 3)
    if np.any(~np.isfinite(normalized_positions)):
        raise ValueError("History estimate positions must be finite.")
    if strength_raw is None:
        return normalized_positions, None
    strengths = np.asarray(strength_raw, dtype=float).reshape(-1)
    if strengths.size != normalized_positions.shape[0]:
        raise ValueError("History estimate positions and strengths must align.")
    if np.any(~np.isfinite(strengths)) or np.any(strengths < 0.0):
        raise ValueError("History estimate strengths must be finite and non-negative.")
    return normalized_positions, strengths


def _gated_transition_assignment(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    match_gate_m: float,
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Match consecutive modes by gated maximum cardinality and distance."""
    distances = np.linalg.norm(
        previous[:, None, :] - current[None, :, :],
        axis=2,
    )
    if distances.size == 0:
        return [], distances
    if np.any(~np.isfinite(distances)):
        raise ValueError("History positions produced non-finite distances.")
    valid = distances <= match_gate_m
    assignment_count = min(distances.shape)
    cost = np.where(
        valid,
        distances / max(match_gate_m, 1.0),
        float(assignment_count + 1),
    )
    rows, columns = linear_sum_assignment(cost)
    assignments = [
        (int(row), int(column))
        for row, column in zip(rows.tolist(), columns.tolist())
        if bool(valid[row, column])
    ]
    return assignments, distances


def _stability_scope_summary(
    transition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate one all-history or final-window transition scope."""
    shifts = [
        float(value)
        for row in transition_rows
        for value in row["matched_position_shifts_m"]
    ]
    absolute_strength_drifts = [
        float(value)
        for row in transition_rows
        for value in row["matched_strength_abs_drifts_cps_1m"]
    ]
    relative_strength_drifts = [
        float(value)
        for row in transition_rows
        for value in row["matched_strength_abs_relative_drifts_pct"]
    ]
    appearance_count = int(
        sum(int(row["birth_count"]) for row in transition_rows)
    )
    disappearance_count = int(
        sum(int(row["death_count"]) for row in transition_rows)
    )
    replacement_transition_count = int(
        sum(
            int(
                row["previous_count"] == row["current_count"]
                and (row["birth_count"] or row["death_count"])
            )
            for row in transition_rows
        )
    )
    return {
        "transition_count": int(len(transition_rows)),
        "matched_transition_mode_count": int(
            sum(int(row["matched_count"]) for row in transition_rows)
        ),
        "unmatched_cluster_appearance_count": appearance_count,
        "unmatched_cluster_disappearance_count": disappearance_count,
        "unmatched_cluster_event_count": appearance_count + disappearance_count,
        "same_cardinality_cluster_replacement_transition_count": (
            replacement_transition_count
        ),
        # These aliases keep old result readers working.  They count unmatched
        # reported clusters and must not be interpreted as accepted PF moves.
        "birth_event_count": appearance_count,
        "death_event_count": disappearance_count,
        "birth_death_event_count": appearance_count + disappearance_count,
        "same_count_birth_death_transition_count": replacement_transition_count,
        "legacy_birth_death_key_semantics": (
            "unmatched_reported_clusters_not_accepted_pf_transitions"
        ),
        "consecutive_matched_cluster_shift_m": _distribution_summary(shifts),
        "consecutive_matched_strength_abs_drift_cps_1m": _distribution_summary(
            absolute_strength_drifts
        ),
        "consecutive_matched_strength_abs_relative_drift_pct": (
            _distribution_summary(relative_strength_drifts)
        ),
    }


def summarize_cluster_stability(
    history_estimates: Sequence[Mapping[str, Any]],
    *,
    final_window: int = 5,
    match_gate_m: float = 0.5,
) -> dict[str, Any]:
    """Summarize gated cluster motion, replacement, and strength drift."""
    isotopes = sorted(
        {str(isotope) for estimate_map in history_estimates for isotope in estimate_map}
    )
    try:
        window = int(final_window)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("final_window must be a positive integer.") from exc
    try:
        window_matches_input = float(final_window) == window
    except (TypeError, ValueError, OverflowError):
        window_matches_input = False
    if window <= 0 or not window_matches_input:
        raise ValueError("final_window must be a positive integer.")
    gate = float(match_gate_m)
    if not np.isfinite(gate) or gate < 0.0:
        raise ValueError("match_gate_m must be finite and non-negative.")
    global_all_transition_rows: list[dict[str, Any]] = []
    global_final_transition_rows: list[dict[str, Any]] = []
    by_isotope: dict[str, dict[str, Any]] = {}
    for isotope in isotopes:
        states = [
            _estimate_state(estimate_map.get(isotope, ()))
            for estimate_map in history_estimates
        ]
        counts = [int(positions.shape[0]) for positions, _ in states]
        transition_rows: list[dict[str, Any]] = []
        for step_index, (previous_state, current_state) in enumerate(
            zip(states[:-1], states[1:])
        ):
            previous, previous_strengths = previous_state
            current, current_strengths = current_state
            assignments, distances = _gated_transition_assignment(
                previous,
                current,
                match_gate_m=gate,
            )
            shifts = [float(distances[row, column]) for row, column in assignments]
            absolute_strength_drifts: list[float] = []
            relative_strength_drifts: list[float] = []
            if previous_strengths is not None and current_strengths is not None:
                for row, column in assignments:
                    previous_strength = float(previous_strengths[row])
                    current_strength = float(current_strengths[column])
                    absolute = abs(current_strength - previous_strength)
                    relative = 100.0 * absolute / max(previous_strength, 1.0e-12)
                    if not np.isfinite(absolute) or not np.isfinite(relative):
                        raise ValueError("Strength drift produced a non-finite value.")
                    absolute_strength_drifts.append(absolute)
                    relative_strength_drifts.append(relative)
            matched_count = int(len(assignments))
            unmatched_appearances = int(current.shape[0] - matched_count)
            unmatched_disappearances = int(previous.shape[0] - matched_count)
            transition_rows.append(
                {
                    "previous_step_index": int(step_index),
                    "current_step_index": int(step_index + 1),
                    "previous_count": int(previous.shape[0]),
                    "current_count": int(current.shape[0]),
                    "matched_count": matched_count,
                    "unmatched_cluster_appearance_count": unmatched_appearances,
                    "unmatched_cluster_disappearance_count": (
                        unmatched_disappearances
                    ),
                    "birth_count": unmatched_appearances,
                    "death_count": unmatched_disappearances,
                    "legacy_birth_death_key_semantics": (
                        "unmatched_reported_clusters_not_accepted_pf_transitions"
                    ),
                    "matched_position_shifts_m": shifts,
                    "matched_strength_abs_drifts_cps_1m": absolute_strength_drifts,
                    "matched_strength_abs_relative_drifts_pct": (
                        relative_strength_drifts
                    ),
                }
            )
        final_counts = counts[-window:]
        modal_count = (
            Counter(final_counts).most_common(1)[0][0] if final_counts else None
        )
        stable_fraction = (
            sum(int(value == modal_count) for value in final_counts)
            / float(len(final_counts))
            if final_counts
            else None
        )
        final_start_step = max(0, len(states) - window)
        final_transition_rows = [
            row
            for row in transition_rows
            if int(row["previous_step_index"]) >= final_start_step
        ]
        all_summary = _stability_scope_summary(transition_rows)
        final_summary = _stability_scope_summary(final_transition_rows)
        global_all_transition_rows.extend(transition_rows)
        global_final_transition_rows.extend(final_transition_rows)
        by_isotope[isotope] = {
            "available": bool(len(states) >= 2),
            "history_length": int(len(states)),
            "transition_count": int(len(transition_rows)),
            "final_window": int(min(window, len(final_counts))),
            "final_count": counts[-1] if counts else None,
            "modal_final_window_count": modal_count,
            "final_window_count_stability_fraction": stable_fraction,
            "match_gate_m": gate,
            "unmatched_cluster_appearance_count": all_summary[
                "unmatched_cluster_appearance_count"
            ],
            "unmatched_cluster_disappearance_count": all_summary[
                "unmatched_cluster_disappearance_count"
            ],
            "unmatched_cluster_event_count": all_summary[
                "unmatched_cluster_event_count"
            ],
            "same_cardinality_cluster_replacement_transition_count": all_summary[
                "same_cardinality_cluster_replacement_transition_count"
            ],
            "birth_death_event_count": all_summary["birth_death_event_count"],
            "birth_event_count": all_summary["birth_event_count"],
            "death_event_count": all_summary["death_event_count"],
            "same_count_birth_death_transition_count": all_summary[
                "same_count_birth_death_transition_count"
            ],
            "legacy_birth_death_key_semantics": all_summary[
                "legacy_birth_death_key_semantics"
            ],
            "consecutive_matched_cluster_shift_m": all_summary[
                "consecutive_matched_cluster_shift_m"
            ],
            "all_history": all_summary,
            "final_window_dynamics": final_summary,
            "all_history_consecutive_matched_cluster_shift_m": all_summary[
                "consecutive_matched_cluster_shift_m"
            ],
            "final_window_consecutive_matched_cluster_shift_m": final_summary[
                "consecutive_matched_cluster_shift_m"
            ],
            "all_history_consecutive_matched_strength_abs_drift_cps_1m": (
                all_summary["consecutive_matched_strength_abs_drift_cps_1m"]
            ),
            "final_window_consecutive_matched_strength_abs_drift_cps_1m": (
                final_summary["consecutive_matched_strength_abs_drift_cps_1m"]
            ),
            "transitions": transition_rows,
        }
    global_all_summary = _stability_scope_summary(global_all_transition_rows)
    global_final_summary = _stability_scope_summary(global_final_transition_rows)
    return {
        "available": bool(len(history_estimates) >= 2 and isotopes),
        "minimum_history_length": 2,
        "minimum_transition_count": 1,
        "history_length": int(len(history_estimates)),
        "final_window": window,
        "match_gate_m": gate,
        "by_isotope": by_isotope,
        "consecutive_matched_cluster_shift_m": global_all_summary[
            "consecutive_matched_cluster_shift_m"
        ],
        "all_history": global_all_summary,
        "final_window_dynamics": global_final_summary,
    }


def start_gpu_memory_tracking(device_name: str | None) -> dict[str, Any]:
    """Reset torch CUDA peak statistics and return a baseline snapshot."""
    scope = {
        "scope": "torch_cuda_allocator_current_process",
        "includes_external_cuda_allocations": False,
        "includes_geant4_sidecar": False,
    }
    if device_name is None or not str(device_name).startswith("cuda"):
        return {"available": False, "device": device_name, **scope}
    try:
        import torch
    except ImportError:
        return {"available": False, "device": str(device_name), **scope}
    if not torch.cuda.is_available():
        return {"available": False, "device": str(device_name), **scope}
    try:
        device = torch.device(str(device_name))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return {
            "available": True,
            "device": str(device),
            **scope,
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    except (RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "device": str(device_name),
            **scope,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def finish_gpu_memory_tracking(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return current and peak torch CUDA memory since tracking started."""
    payload = dict(baseline)
    if not bool(payload.get("available", False)):
        return payload
    try:
        import torch
    except ImportError:
        payload["available"] = False
        return payload
    try:
        device = torch.device(str(payload["device"]))
        torch.cuda.synchronize(device)
        payload.update(
            {
                "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    except (RuntimeError, ValueError) as exc:
        payload.update(
            {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return payload
