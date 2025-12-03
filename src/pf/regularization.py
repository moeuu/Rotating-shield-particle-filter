"""Manage post-resampling perturbations and particle-count adaptation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pf.state import ParticleState, jitter_state


def regularize_states(states: list[ParticleState], strength_sigma: float = 0.1, background_sigma: float = 0.1) -> list[ParticleState]:
    """リサンプリング後に小さな摂動を与える。"""
    grid_size = 0  # 位置は離散インデックスなので変更しない
    return [jitter_state(st, grid_size=grid_size, strength_sigma=strength_sigma, background_sigma=background_sigma) for st in states]
