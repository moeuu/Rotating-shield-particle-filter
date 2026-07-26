"""Validate vectors used by PF posterior diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def measurement_vector(
    values: float | NDArray[np.float64],
    count: int,
    name: str,
    *,
    min_value: float | None = None,
    allow_scalar: bool = True,
) -> NDArray[np.float64]:
    """Return a validated one-value-per-measurement vector."""
    expected = max(int(count), 0)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        if expected == 0:
            return np.zeros(0, dtype=float)
        raise ValueError(f"{name} must contain one value per measurement.")
    if arr.size == 1 and expected != 1 and allow_scalar:
        arr = np.full(expected, float(arr[0]), dtype=float)
    elif arr.size != expected:
        scalar_text = "scalar or " if allow_scalar else ""
        raise ValueError(f"{name} must be {scalar_text}one value per measurement.")
    if min_value is not None:
        arr = np.maximum(arr, float(min_value))
    return np.asarray(arr, dtype=float)
