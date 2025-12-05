"""Define per-isotope particle state vectors (source count, positions, intensities, background).

Two representations exist while transitioning to the Chapter 3.3 formulation:
- `IsotopeState`: continuous 3D positions (s_{h,m}), strengths (q_{h,m}), background b_h.
- `ParticleState`: legacy grid-index representation (kept temporarily for compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    covariances: NDArray[np.float64] | None = None  # optional (r_h,4,4) across (x,y,z,q)

    def copy(self) -> "IsotopeState":
        return IsotopeState(
            num_sources=int(self.num_sources),
            positions=self.positions.copy(),
            strengths=self.strengths.copy(),
            background=float(self.background),
            covariances=None if self.covariances is None else self.covariances.copy(),
        )


@dataclass
class ParticleState:
    """Legacy grid-index particle state (Sec. 3.4, discrete candidate_sources)."""

    source_indices: NDArray[np.int32]  # shape (r_h,)
    strengths: NDArray[np.float64]  # shape (r_h,)
    background: float

    def copy(self) -> "ParticleState":
        return ParticleState(
            source_indices=self.source_indices.copy(),
            strengths=self.strengths.copy(),
            background=float(self.background),
        )


def jitter_state(
    state: ParticleState,
    strength_sigma: float = 0.1,
    background_sigma: float = 0.1,
) -> ParticleState:
    """リサンプリング後のわずかな揺らぎを加える。"""
    new_state = state.copy()
    if new_state.strengths.size:
        noise = np.random.normal(scale=strength_sigma, size=new_state.strengths.shape)
        new_state.strengths = np.clip(new_state.strengths + noise, a_min=0.0, a_max=None)
    new_state.background = max(0.0, new_state.background + np.random.normal(scale=background_sigma))
    return new_state
