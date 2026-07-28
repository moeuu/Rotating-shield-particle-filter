"""Run-level diagnostics that complement source-matching accuracy metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


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
        sum(
            int(row["unmatched_cluster_appearance_count"])
            for row in transition_rows
        )
    )
    disappearance_count = int(
        sum(
            int(row["unmatched_cluster_disappearance_count"])
            for row in transition_rows
        )
    )
    replacement_transition_count = int(
        sum(
            int(
                row["previous_count"] == row["current_count"]
                and (
                    row["unmatched_cluster_appearance_count"]
                    or row["unmatched_cluster_disappearance_count"]
                )
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
