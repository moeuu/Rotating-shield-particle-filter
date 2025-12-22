"""Generate discrete shield orientations (Sec. 3.4.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import generate_octant_orientations


def generate_ring_normals(num: int = 4, axis: str = "z") -> NDArray[np.float64]:
    """
    Generate evenly spaced shield normals on a ring perpendicular to the given axis.

    Args:
        num: Number of orientations to generate on the ring.
        axis: Axis the ring is perpendicular to ("x", "y", or "z").

    Returns:
        (num, 3) unit-normal vectors.
    """
    axis = axis.lower()
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    if axis == "z":
        normals = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    elif axis == "x":
        normals = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=1)
    elif axis == "y":
        normals = np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)], axis=1)
    else:
        raise ValueError(f"Unknown axis: {axis}")
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
