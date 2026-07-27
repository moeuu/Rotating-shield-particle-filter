"""Define per-isotope PF state vectors."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class IsotopeState:
    """
    Continuous PF state for a single isotope (Sec. 3.3.2):
        θ_h = (r_h, {s_{h,m}}, {q_{h,m}}, b_h)
    """

    num_sources: int
    positions: NDArray[np.float64]  # shape (r_h,3)
    strengths: NDArray[np.float64]  # shape (r_h,)
    background: float
    # Optional shape (r_h, 4, 4) across (x, y, z, q).
    covariances: NDArray[np.float64] | None = None

    def copy(self) -> "IsotopeState":
        """Return a deep copy of the isotope state arrays."""
        return IsotopeState(
            num_sources=int(self.num_sources),
            positions=self.positions.copy(),
            strengths=self.strengths.copy(),
            background=float(self.background),
            covariances=None if self.covariances is None else self.covariances.copy(),
        )
