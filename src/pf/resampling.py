"""Implement effective-sample-size checks and systematic resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def systematic_resample(
    log_weights: NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Return systematic-resampling ancestors using an explicit RNG stream."""
    weights = np.exp(np.asarray(log_weights, dtype=np.float64))
    particle_count = int(weights.size)
    if particle_count == 0:
        return np.zeros(0, dtype=np.int64)
    cumulative_sum = np.cumsum(weights)
    total = float(cumulative_sum[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Resampling weights must have a finite positive sum.")
    cumulative_sum /= total
    offset = float(rng.random())
    positions = (
        np.arange(particle_count, dtype=np.float64) + offset
    ) / particle_count
    indices = np.searchsorted(cumulative_sum, positions, side="right")
    return np.minimum(indices, particle_count - 1).astype(np.int64, copy=False)
