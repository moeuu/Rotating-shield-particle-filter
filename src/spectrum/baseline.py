"""Baseline estimation utilities inspired by the ALS approach in Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve


def asymmetric_least_squares(
    signal: NDArray[np.float64],
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> NDArray[np.float64]:
    """
    Estimate a smooth baseline using asymmetric least squares.

    Args:
        signal: Input spectrum.
        lam: Smoothness parameter (larger = smoother).
        p: Asymmetry parameter (small values force baseline below data).
        niter: Number of reweighting iterations.

    Returns:
        Estimated baseline array.
    """
    y = signal
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).tocsc()
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = (W + lam * D @ D.T).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return np.asarray(z)
