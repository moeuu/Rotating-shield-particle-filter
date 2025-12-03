"""Smoothing utilities such as Gaussian convolution to suppress statistical noise while preserving peaks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth(signal: NDArray[np.float64], sigma_bins: float = 1.0) -> NDArray[np.float64]:
    """
    Apply 1D Gaussian smoothing to a spectrum.

    Args:
        signal: Input spectrum (counts per bin).
        sigma_bins: Standard deviation of the Gaussian kernel in bins.

    Returns:
        Smoothed spectrum.
    """
    return gaussian_filter1d(signal, sigma=sigma_bins, mode="nearest")
