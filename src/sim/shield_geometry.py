"""Shared spherical-octant shield geometry used by Python and Geant4 paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

from measurement.shielding import (
    CS137_TVL_FE_CM,
    CS137_TVL_PB_CM,
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
    DEFAULT_SHIELD_TRANSMISSION_TARGET,
    DEFAULT_SHIELD_TVL_SCALE,
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    DEFAULT_SHIELD_CONTACT_RADIUS_CM,
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
FE_SHIELD_TVL_THICKNESS_CM = float(CS137_TVL_FE_CM)
FE_SHIELD_THICKNESS_CM = float(DEFAULT_FE_SHIELD_THICKNESS_CM)
FE_SHIELD_THICKNESS_M = FE_SHIELD_THICKNESS_CM / 100.0
FE_SHIELD_OUTER_RADIUS_M = FE_SHIELD_INNER_RADIUS_M + FE_SHIELD_THICKNESS_M
SHIELD_CONTACT_RADIUS_M = DEFAULT_SHIELD_CONTACT_RADIUS_CM / 100.0

PB_SHIELD_INNER_RADIUS_M = DEFAULT_PB_SHIELD_INNER_RADIUS_CM / 100.0
PB_SHIELD_TVL_THICKNESS_CM = float(CS137_TVL_PB_CM)
PB_SHIELD_THICKNESS_CM = float(DEFAULT_PB_SHIELD_THICKNESS_CM)
PB_SHIELD_THICKNESS_M = PB_SHIELD_THICKNESS_CM / 100.0
PB_SHIELD_OUTER_RADIUS_M = PB_SHIELD_INNER_RADIUS_M + PB_SHIELD_THICKNESS_M


def require_no_angle_attenuation(value: object) -> bool:
    """Require the production spherical-shell angle contract to be false."""
    if not isinstance(value, bool):
        raise TypeError("use_angle_attenuation must be a JSON boolean.")
    if value:
        raise ValueError(
            "use_angle_attenuation=true is incompatible with the production "
            "spherical-octant shell model."
        )
    return False


def _nonnegative_finite_real(value: object, *, field_name: str) -> float:
    """Return a strict nonnegative finite real geometry value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative.")
    return parsed


def nested_shield_inner_radii_cm(
    *,
    thickness_fe_cm: float = FE_SHIELD_THICKNESS_CM,
    detector_outer_radius_cm: float = DEFAULT_SHIELD_CONTACT_RADIUS_CM,
) -> tuple[float, float]:
    """Return nested Fe/Pb inner radii with Fe touching the detector housing."""
    fe_inner_cm = _nonnegative_finite_real(
        detector_outer_radius_cm,
        field_name="detector_outer_radius_cm",
    )
    fe_thickness_cm = _nonnegative_finite_real(
        thickness_fe_cm,
        field_name="thickness_fe_cm",
    )
    pb_inner_cm = fe_inner_cm + fe_thickness_cm
    return fe_inner_cm, pb_inner_cm


@dataclass(frozen=True)
class ShieldThicknessConfig:
    """Store Fe/Pb spherical-octant shield thickness overrides."""

    thickness_fe_cm: float = FE_SHIELD_THICKNESS_CM
    thickness_pb_cm: float = PB_SHIELD_THICKNESS_CM
    thickness_scale: float = DEFAULT_SHIELD_TVL_SCALE
    transmission_target: float | None = DEFAULT_SHIELD_TRANSMISSION_TARGET


def shield_thickness_scale_for_transmission(transmission_target: float) -> float:
    """Return the one-TVL thickness scale for a target single-shell transmission."""
    if isinstance(transmission_target, bool) or not isinstance(
        transmission_target,
        Real,
    ):
        raise TypeError("shield_transmission_target must be a real number.")
    transmission = float(transmission_target)
    if not math.isfinite(transmission) or not 0.0 < transmission <= 1.0:
        raise ValueError("shield_transmission_target must be in (0, 1].")
    if transmission == 1.0:
        return 0.0
    return float(math.log(1.0 / transmission) / math.log(10.0))


def resolve_shield_thickness_config(
    payload: Mapping[str, Any] | None = None,
) -> ShieldThicknessConfig:
    """Resolve shared shield thickness settings from a runtime config payload."""
    if payload is None:
        config: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        config = payload
    else:
        raise TypeError("shield thickness configuration must be a mapping or None.")
    target_raw = config.get("shield_transmission_target")
    has_override = any(
        key in config and config[key] is not None
        for key in (
            "shield_thickness_scale",
            "fe_shield_thickness_cm",
            "pb_shield_thickness_cm",
        )
    )
    if target_raw is None and not has_override:
        transmission_target = DEFAULT_SHIELD_TRANSMISSION_TARGET
        default_scale = DEFAULT_SHIELD_TVL_SCALE
    elif target_raw is None:
        transmission_target = None
        default_scale = 1.0
    else:
        if isinstance(target_raw, bool) or not isinstance(target_raw, Real):
            raise TypeError("shield_transmission_target must be a real number.")
        transmission_target = float(target_raw)
        default_scale = shield_thickness_scale_for_transmission(transmission_target)
    scale = _nonnegative_finite_real(
        config.get("shield_thickness_scale", default_scale),
        field_name="shield_thickness_scale",
    )
    thickness_fe_cm = _nonnegative_finite_real(
        config.get(
            "fe_shield_thickness_cm",
            FE_SHIELD_TVL_THICKNESS_CM * scale,
        ),
        field_name="fe_shield_thickness_cm",
    )
    thickness_pb_cm = _nonnegative_finite_real(
        config.get(
            "pb_shield_thickness_cm",
            PB_SHIELD_TVL_THICKNESS_CM * scale,
        ),
        field_name="pb_shield_thickness_cm",
    )
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
    require_no_angle_attenuation(use_angle_attenuation)
    direction = np.asarray(source_xyz, dtype=float) - np.asarray(detector_xyz, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1.0e-12:
        return 0.0
    rotation = quaternion_wxyz_to_matrix(shield_quat_wxyz)
    local_direction = rotation.T @ (direction / direction_norm)
    blocked = bool(np.all(local_direction >= -1.0e-9))
    return spherical_shell_path_length_cm(
        direction_m=direction,
        inner_radius_cm=float(inner_radius_cm),
        outer_radius_cm=float(inner_radius_cm) + float(thickness_cm),
        blocked=blocked,
    )
