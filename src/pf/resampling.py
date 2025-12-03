"""Implement effective-sample-size checks and systematic resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def systematic_resample(log_weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """システマティックリサンプリング。"""
    weights = np.exp(log_weights)
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices
