"""Simple peak detection utilities for smoothed spectra."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def detect_peaks(signal: NDArray[np.float64], prominence: float = 0.01, distance: int = 5) -> NDArray[np.int_]:
    """
    Detect peaks using scipy's find_peaks with basic parameters.

    Args:
        signal: Smoothed spectrum.
        prominence: Minimum prominence relative to max to keep a peak.
        distance: Minimum separation between peaks in bins.

    Returns:
        Indices of detected peaks.
    """
    if signal.size == 0:
        return np.array([], dtype=int)
    prom = prominence * float(signal.max() if signal.max() > 0 else 1.0)
    peaks, _ = find_peaks(signal, prominence=prom, distance=distance)
    return peaks.astype(int)
