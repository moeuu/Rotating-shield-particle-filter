"""Shared spherical-octant shield geometry used by Python and Geant4 paths."""

from __future__ import annotations

import numpy as np

from measurement.shielding import CS137_TVL_FE_MM, CS137_TVL_PB_MM, path_length_cm
from sim.isaacsim_app.geometry import quaternion_wxyz_to_matrix

SHIELD_SHAPE_SPHERICAL_OCTANT = "spherical_octant_shell"
LOCAL_POSITIVE_OCTANT_CENTER_XYZ: tuple[float, float, float] = (
    1.0 / np.sqrt(3.0),
    1.0 / np.sqrt(3.0),
    1.0 / np.sqrt(3.0),
)

FE_SHIELD_INNER_RADIUS_M = 0.19
FE_SHIELD_THICKNESS_CM = float(CS137_TVL_FE_MM) / 10.0
FE_SHIELD_THICKNESS_M = FE_SHIELD_THICKNESS_CM / 100.0
FE_SHIELD_OUTER_RADIUS_M = FE_SHIELD_INNER_RADIUS_M + FE_SHIELD_THICKNESS_M

PB_SHIELD_INNER_RADIUS_M = 0.26
PB_SHIELD_THICKNESS_CM = float(CS137_TVL_PB_MM) / 10.0
PB_SHIELD_THICKNESS_M = PB_SHIELD_THICKNESS_CM / 100.0
PB_SHIELD_OUTER_RADIUS_M = PB_SHIELD_INNER_RADIUS_M + PB_SHIELD_THICKNESS_M


def shield_normal_from_quaternion_wxyz(
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return the world normal for a rotated local +X/+Y/+Z shield octant."""
    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    normal = rotation @ np.asarray(LOCAL_POSITIVE_OCTANT_CENTER_XYZ, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return LOCAL_POSITIVE_OCTANT_CENTER_XYZ
    normal /= norm
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def spherical_octant_path_length_cm(
    source_xyz: tuple[float, float, float],
    detector_xyz: tuple[float, float, float],
    shield_quat_wxyz: tuple[float, float, float, float],
    *,
    thickness_cm: float,
    use_angle_attenuation: bool = False,
) -> float:
    """Return the Python reference path length for a spherical octant shell."""
    direction = np.asarray(source_xyz, dtype=float) - np.asarray(detector_xyz, dtype=float)
    shield_normal = np.asarray(shield_normal_from_quaternion_wxyz(shield_quat_wxyz), dtype=float)
    return path_length_cm(
        direction,
        shield_normal,
        float(thickness_cm),
        use_angle_attenuation=bool(use_angle_attenuation),
    )
