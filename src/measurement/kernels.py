"""Shared shield parameters and immutable measurement geometry."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class MeasurementGeometry:
    """Store source support, detector poses, and shield orientations for the PF."""

    candidate_sources: NDArray[np.float64]
    poses: NDArray[np.float64]
    orientations: NDArray[np.float64]
    shield_params: ShieldParams
    mu_by_isotope: Dict[str, object]

    def __post_init__(self) -> None:
        """Validate and freeze canonical floating-point geometry arrays."""
        sources = np.asarray(self.candidate_sources, dtype=np.float64)
        poses = np.asarray(self.poses, dtype=np.float64)
        orientations = np.asarray(self.orientations, dtype=np.float64)
        for name, values in (
            ("candidate_sources", sources),
            ("poses", poses),
            ("orientations", orientations),
        ):
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f"{name} must be shaped N x 3.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        object.__setattr__(
            self,
            "candidate_sources",
            np.ascontiguousarray(sources),
        )
        object.__setattr__(self, "poses", np.ascontiguousarray(poses))
        object.__setattr__(
            self,
            "orientations",
            np.ascontiguousarray(orientations),
        )
        object.__setattr__(self, "mu_by_isotope", dict(self.mu_by_isotope))

    @property
    def sources(self) -> NDArray[np.float64]:
        """Return the finite surface support used to initialize the PF."""
        return self.candidate_sources
