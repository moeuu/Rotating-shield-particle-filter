"""Generate discrete shield orientations (Sec. 3.5.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
