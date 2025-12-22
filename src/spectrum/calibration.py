"""Estimate and apply polynomial energy calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationModel:
    """Store calibration coefficients and map between channel and energy."""

    coefficients: Sequence[float]

    def channel_to_energy(self, channels: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert channel indices to energy (keV)."""
        poly = np.poly1d(self.coefficients)
        return poly(channels)

    def energy_to_channel(self, energies: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert energy values to channel indices."""
        # Inversion is not unique; use a simple Newton iteration for low-order polynomials.
        energies = np.asarray(energies, dtype=float)
        channels = np.zeros_like(energies, dtype=float)
        for _ in range(10):
            f = np.polyval(self.coefficients, channels) - energies
            df = np.polyval(np.polyder(self.coefficients), channels)
            df = np.where(df == 0, 1e-9, df)
            channels -= f / df
        return channels


def fit_polynomial_calibration(
    reference_peaks: Iterable[Tuple[float, float]],
    order: int = 2,
) -> CalibrationModel:
    """
    Fit a polynomial calibration from reference channel/energy pairs.

    Args:
        reference_peaks: Iterable of (channel, energy_keV) pairs.
        order: Polynomial order.

    Returns:
        CalibrationModel: Fitted calibration model.
    """
    refs = np.asarray(reference_peaks, dtype=float)
    if refs.shape[0] < order + 1:
        raise ValueError("Not enough reference peaks for the requested order.")
    channels = refs[:, 0]
    energies = refs[:, 1]
    coeffs = np.polyfit(channels, energies, order)
    return CalibrationModel(coefficients=coeffs)


def apply_calibration(model: CalibrationModel, num_channels: int) -> NDArray[np.float64]:
    """
    Generate an energy axis using the calibration model.

    Args:
        model: CalibrationModel instance.
        num_channels: Number of channels.

    Returns:
        Energy axis in keV.
    """
    channels = np.arange(num_channels, dtype=float)
    return model.channel_to_energy(channels)
