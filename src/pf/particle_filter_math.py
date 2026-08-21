"""Numerical helpers shared by exact-RJ particle-filter algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ordered_source_pair_columns(
    cardinality: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return every ordered donor/receiver source-column pair."""
    if (
        isinstance(cardinality, (bool, np.bool_))
        or not isinstance(cardinality, (int, np.integer))
        or int(cardinality) < 2
    ):
        raise ValueError("Ordered source pairs require cardinality at least two.")
    count = int(cardinality)
    donor_grid = np.repeat(
        np.arange(count, dtype=np.int64)[:, None],
        count,
        axis=1,
    )
    receiver_grid = np.repeat(
        np.arange(count, dtype=np.int64)[None, :],
        count,
        axis=0,
    )
    distinct = donor_grid != receiver_grid
    return donor_grid[distinct], receiver_grid[distinct]


def extended_log_target_ratio(
    proposed_log_target: NDArray[np.float64],
    current_log_target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an MH log-target ratio on the finite-or-minus-infinity domain.

    A proposed state with zero target mass is rejected, while a finite proposal
    can recover a row whose current state has zero target mass. When both
    states have zero target mass the ratio is undefined mathematically, so the
    move is deterministically rejected instead of allowing ``-inf - -inf`` to
    produce a NaN.
    """
    proposed = np.asarray(proposed_log_target, dtype=np.float64)
    current = np.asarray(current_log_target, dtype=np.float64)
    if proposed.shape != current.shape:
        raise ValueError("Proposed and current log targets must be aligned.")
    if (
        np.any(np.isnan(proposed))
        or np.any(np.isnan(current))
        or np.any(np.isposinf(proposed))
        or np.any(np.isposinf(current))
    ):
        raise ValueError(
            "MH log targets may be finite or negative infinity, not NaN or "
            "positive infinity."
        )
    result = np.full(proposed.shape, float("-inf"), dtype=np.float64)
    proposed_finite = np.isfinite(proposed)
    current_finite = np.isfinite(current)
    both_finite = proposed_finite & current_finite
    result[both_finite] = proposed[both_finite] - current[both_finite]
    result[proposed_finite & ~current_finite] = float("inf")
    return result


__all__ = ["extended_log_target_ratio", "ordered_source_pair_columns"]
