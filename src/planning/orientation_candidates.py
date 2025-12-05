"""Generate discrete shield orientations (Sec. 3.4.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import generate_octant_orientations


def generate_ring_normals(num: int = 4, axis: str = "z") -> NDArray[np.float64]:
    """
    円環状に等間隔でシールド法線を生成する。

    Args:
        num: 生成する方位数。
        axis: 回転軸（"z"のみサポート）。

    Returns:
        (num, 3) 配列の単位ベクトル。
    """
    if axis != "z":
        raise NotImplementedError("Only z-axis ring supported.")
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    normals = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    return normals


def generate_candidate_orientations(mode: str = "octant", num_ring: int = 4) -> NDArray[np.float64]:
    """
    Generate the discrete shield orientations used in Chapter 3.4 (Eq. 3.39–3.41).

    - ``octant``: 8 orientations corresponding to the 1/8 spherical shell (Sec. 3.4.1).
    - ``ring``: keep legacy evenly spaced normals in the xy-plane (mainly for tests).
    """
    if mode == "octant":
        return generate_octant_orientations()
    if mode == "ring":
        return generate_ring_normals(num=num_ring, axis="z")
    raise ValueError(f"Unknown orientation generation mode: {mode}")
