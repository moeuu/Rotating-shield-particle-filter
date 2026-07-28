"""Shared shield parameters and immutable measurement geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import (
    CS137_TVL_FE_MM,
    CS137_TVL_PB_MM,
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    mu_from_tvl_mm,
)


CS137_MU_PB_CM_INV = mu_from_tvl_mm(CS137_TVL_PB_MM)
CS137_MU_FE_CM_INV = mu_from_tvl_mm(CS137_TVL_FE_MM)


@dataclass(frozen=True)
class ShieldParams:
    """Store the shared Fe/Pb shield material and geometry parameters."""

    mu_pb: float = CS137_MU_PB_CM_INV
    mu_fe: float = CS137_MU_FE_CM_INV
    thickness_pb_cm: float = DEFAULT_PB_SHIELD_THICKNESS_CM
    thickness_fe_cm: float = DEFAULT_FE_SHIELD_THICKNESS_CM
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM
    buildup_fe_coeff: float = 0.0
    buildup_pb_coeff: float = 0.0
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT
    use_angle_attenuation: bool = False

    def __post_init__(self) -> None:
        """Require the sole high-fidelity spherical-octant shield contract."""
        numeric_fields = (
            "mu_pb",
            "mu_fe",
            "thickness_pb_cm",
            "thickness_fe_cm",
            "inner_radius_fe_cm",
            "inner_radius_pb_cm",
            "buildup_fe_coeff",
            "buildup_pb_coeff",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            resolved = float(value)
            if not math.isfinite(resolved) or resolved < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, resolved)
        if (
            not isinstance(self.shield_geometry_model, str)
            or self.shield_geometry_model != SHIELD_GEOMETRY_SPHERICAL_OCTANT
        ):
            raise ValueError(
                "Pure PF requires the shared spherical-octant shell geometry."
            )
        if not isinstance(self.use_angle_attenuation, bool):
            raise TypeError("use_angle_attenuation must be a boolean.")
        if self.use_angle_attenuation:
            raise ValueError(
                "Pure PF spherical-octant geometry does not use the retired "
                "fixed-slab angle attenuation."
            )


@dataclass(frozen=True)
class MeasurementGeometry:
    """Store detector poses and shield orientations for the continuous PF."""

    poses: NDArray[np.float64]
    orientations: NDArray[np.float64]
    shield_params: ShieldParams
    mu_by_isotope: Dict[str, object]

    def __post_init__(self) -> None:
        """Validate and freeze canonical floating-point geometry arrays."""
        poses = np.asarray(self.poses, dtype=np.float64)
        orientations = np.asarray(self.orientations, dtype=np.float64)
        for name, values in (
            ("poses", poses),
            ("orientations", orientations),
        ):
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f"{name} must be shaped N x 3.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        object.__setattr__(self, "poses", np.ascontiguousarray(poses))
        object.__setattr__(
            self,
            "orientations",
            np.ascontiguousarray(orientations),
        )
        object.__setattr__(self, "mu_by_isotope", dict(self.mu_by_isotope))
