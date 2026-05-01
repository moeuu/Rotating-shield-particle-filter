"""Shared spherical-octant shield geometry used by Python and Geant4 paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from measurement.shielding import (
    CS137_TVL_FE_MM,
    CS137_TVL_PB_MM,
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    path_length_cm,
    shield_blocks_radiation,
    spherical_shell_path_length_cm,
)
from sim.isaacsim_app.geometry import quaternion_wxyz_to_matrix

SHIELD_SHAPE_SPHERICAL_OCTANT = "spherical_octant_shell"
LOCAL_POSITIVE_OCTANT_CENTER_XYZ: tuple[float, float, float] = (
    1.0 / np.sqrt(3.0),
    1.0 / np.sqrt(3.0),
    1.0 / np.sqrt(3.0),
)

FE_SHIELD_INNER_RADIUS_M = DEFAULT_FE_SHIELD_INNER_RADIUS_CM / 100.0
FE_SHIELD_THICKNESS_CM = float(CS137_TVL_FE_MM) / 10.0
FE_SHIELD_THICKNESS_M = FE_SHIELD_THICKNESS_CM / 100.0
FE_SHIELD_OUTER_RADIUS_M = FE_SHIELD_INNER_RADIUS_M + FE_SHIELD_THICKNESS_M

PB_SHIELD_INNER_RADIUS_M = DEFAULT_PB_SHIELD_INNER_RADIUS_CM / 100.0
PB_SHIELD_THICKNESS_CM = float(CS137_TVL_PB_MM) / 10.0
PB_SHIELD_THICKNESS_M = PB_SHIELD_THICKNESS_CM / 100.0
PB_SHIELD_OUTER_RADIUS_M = PB_SHIELD_INNER_RADIUS_M + PB_SHIELD_THICKNESS_M


@dataclass(frozen=True)
class ShieldThicknessConfig:
    """Store Fe/Pb spherical-octant shield thickness overrides."""

    thickness_fe_cm: float = FE_SHIELD_THICKNESS_CM
    thickness_pb_cm: float = PB_SHIELD_THICKNESS_CM
    thickness_scale: float = 1.0
    transmission_target: float | None = None


def shield_thickness_scale_for_transmission(transmission_target: float) -> float:
    """Return the one-TVL thickness scale for a target single-shell transmission."""
    transmission = float(transmission_target)
    if not 0.0 < transmission <= 1.0:
        raise ValueError("shield_transmission_target must be in (0, 1].")
    if transmission == 1.0:
        return 0.0
    return float(math.log(1.0 / transmission) / math.log(10.0))


def resolve_shield_thickness_config(
    payload: Mapping[str, Any] | None = None,
) -> ShieldThicknessConfig:
    """Resolve shared shield thickness settings from a runtime config payload."""
    config = {} if payload is None else dict(payload)
    target_raw = config.get("shield_transmission_target")
    if target_raw in (None, ""):
        transmission_target = None
        default_scale = 1.0
    else:
        transmission_target = float(target_raw)
        default_scale = shield_thickness_scale_for_transmission(transmission_target)
    scale = float(config.get("shield_thickness_scale", default_scale))
    if scale < 0.0:
        raise ValueError("shield_thickness_scale must be non-negative.")
    thickness_fe_cm = float(config.get("fe_shield_thickness_cm", FE_SHIELD_THICKNESS_CM * scale))
    thickness_pb_cm = float(config.get("pb_shield_thickness_cm", PB_SHIELD_THICKNESS_CM * scale))
    if thickness_fe_cm < 0.0 or thickness_pb_cm < 0.0:
        raise ValueError("shield thickness values must be non-negative.")
    return ShieldThicknessConfig(
        thickness_fe_cm=thickness_fe_cm,
        thickness_pb_cm=thickness_pb_cm,
        thickness_scale=scale,
        transmission_target=transmission_target,
    )


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
    inner_radius_cm: float = 0.0,
    use_angle_attenuation: bool = False,
) -> float:
    """Return the path length for a rotated local +X/+Y/+Z spherical-octant shell."""
    direction = np.asarray(source_xyz, dtype=float) - np.asarray(detector_xyz, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1.0e-12:
        return 0.0
    rotation = quaternion_wxyz_to_matrix(shield_quat_wxyz)
    local_direction = rotation.T @ (direction / direction_norm)
    blocked = bool(np.all(local_direction >= -1.0e-9))
    shield_normal = np.asarray(shield_normal_from_quaternion_wxyz(shield_quat_wxyz), dtype=float)
    if not use_angle_attenuation:
        return spherical_shell_path_length_cm(
            direction_m=direction,
            inner_radius_cm=float(inner_radius_cm),
            outer_radius_cm=float(inner_radius_cm) + float(thickness_cm),
            blocked=blocked,
        )
    return path_length_cm(
        direction,
        shield_normal,
        float(thickness_cm),
        blocked=blocked,
        use_angle_attenuation=bool(use_angle_attenuation),
    )
