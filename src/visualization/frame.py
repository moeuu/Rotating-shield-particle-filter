"""Stable data contract for one PF visualization frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from runtime.cui import CUIRoute


@dataclass
class PFFrame:
    """Store PF state, acquisition geometry, and optional raw spectrum data."""

    step_index: int
    time: float
    robot_position: NDArray[np.float64]
    robot_orientation: NDArray[np.float64] | None
    RFe: NDArray[np.float64]
    RPb: NDArray[np.float64]
    duration: float
    particle_positions: dict[str, NDArray[np.float64]]
    particle_weights: dict[str, NDArray[np.float64]]
    estimated_sources: dict[str, NDArray[np.float64]]
    estimated_strengths: dict[str, NDArray[np.float64]]
    path_waypoints_xyz: NDArray[np.float64] | None = None
    spectrum_energy_keV: NDArray[np.float64] | None = None
    spectrum_counts: NDArray[np.float64] | None = None
    particle_representative_positions: dict[str, NDArray[np.float64]] | None = None
    particle_representative_weights: dict[str, NDArray[np.float64]] | None = None
    cui_route: CUIRoute | None = None
    record_measurement: bool = True


__all__ = ["PFFrame"]
