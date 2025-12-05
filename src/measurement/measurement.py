"""Measurement data structure used by the PF (position, shield orientations, counts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray


@dataclass
class Measurement:
    """
    Single measurement step used by the PF.

    Only fields consumed by the PF:
        - position (q_k)
        - RFe, RPb (3x3 rotation matrices)
        - duration (T_k)
        - counts_by_isotope[h] = z_{k,h} (from spectrum unfolding, Sec. 2.5.7)
    """

    position: NDArray[np.float64]
    RFe: NDArray[np.float64]
    RPb: NDArray[np.float64]
    duration: float
    counts_by_isotope: Dict[str, float]
