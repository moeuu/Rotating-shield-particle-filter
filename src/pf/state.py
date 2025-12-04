"""Define per-isotope particle state vectors (source count, positions, intensities, background)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParticleState:
    """単一同位体の粒子状態（Sec. 3.4）。"""

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
