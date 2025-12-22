"""Represent the measurement environment for a non-directional detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    """Hold environment dimensions and detector position."""

    size_x: float = 10.0
    size_y: float = 20.0
    size_z: float = 10.0
    detector_position: Tuple[float, float, float] | None = None

    def detector(self) -> np.ndarray:
        """Return the detector position (defaults to the environment center)."""
        if self.detector_position is None:
            return np.array([self.size_x / 2.0, self.size_y / 2.0, self.size_z / 2.0])
        return np.array(self.detector_position, dtype=float)


@dataclass(frozen=True)
class PointSource:
    """Represent a point radiation source."""

    isotope: str
    position: Tuple[float, float, float]
    intensity_cps_1m: float

    def position_array(self) -> np.ndarray:
        """Return the position as a NumPy array."""
        return np.array(self.position, dtype=float)

    @property
    def strength(self) -> float:
        """Backward-compatible strength accessor (cps at 1 m)."""
        return self.intensity_cps_1m


def inverse_square_scale(detector: np.ndarray, source: PointSource) -> float:
    """
    Return the inverse-square geometric scale for a point source.

    Computes 1/(4πd^2) based on the detector distance.
    """
    distance = np.linalg.norm(detector - source.position_array())
    if distance == 0:
        # Clip zero distance to avoid unrealistic singularity.
        distance = 1e-6
    return 1.0 / (4.0 * np.pi * distance**2)
