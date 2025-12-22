"""Legacy grid-based kernel calculations (Chapter 3, Sec. 3.2/3.4).

Note: This module uses discrete candidate_sources indices. New continuous PF code
uses measurement.continuous_kernels instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import OctantShield, octant_index_from_normal, path_length_cm, resolve_mu_values


@dataclass(frozen=True)
class ShieldParams:
    """Lightweight shield material and thickness parameters."""

    mu_pb: float = 0.7  # 1/cm at representative energies
    mu_fe: float = 0.5  # 1/cm at representative energies
    thickness_pb_cm: float = 2.0
    thickness_fe_cm: float = 2.0


class KernelPrecomputer:
    """
    Precompute geometric + shielding kernels for poses, orientations, and sources.

    This follows the model in Sec. 3.2 and 3.4.
    """

    def __init__(
        self,
        candidate_sources: NDArray[np.float64],
        poses: NDArray[np.float64],
        orientations: NDArray[np.float64],
        shield_params: ShieldParams,
        mu_by_isotope: Dict[str, object],
    ) -> None:
        """
        Args:
            candidate_sources: (J,3) candidate source positions.
            poses: (K,3) detector poses.
            orientations: (R,3) shield normal vectors.
            shield_params: ShieldParams instance.
            mu_by_isotope: Per-isotope linear attenuation (1/cm): float, (fe, pb), or {"fe","pb"}.
        """
        self.sources = candidate_sources
        self.poses = poses
        self.orientations = orientations
        self.shield_params = shield_params
        self.mu_by_isotope = mu_by_isotope
        self.num_sources = candidate_sources.shape[0]
        self.num_poses = poses.shape[0]
        self.num_orient = orientations.shape[0]
        self.octant_shield = OctantShield()

    def geometric_term(self, pose: NDArray[np.float64], source: NDArray[np.float64]) -> float:
        """Inverse-square geometric term 1/(4πd^2)."""
        d = np.linalg.norm(pose - source)
        if d == 0:
            d = 1e-6
        return float(1.0 / (4.0 * np.pi * d**2))

    def kernel(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
    ) -> NDArray[np.float64]:
        """
        Return the expected-count kernel (J,) for unit source strength.

        Includes geometric term and exponential attenuation based on shield thickness
        and per-isotope linear attenuation coefficients.
        """
        pose = self.poses[pose_idx]
        kernels = np.zeros(self.num_sources, dtype=float)
        oct_idx = octant_index_from_normal(self.orientations[orient_idx])
        mu_fe, mu_pb = resolve_mu_values(
            self.mu_by_isotope, isotope, default_fe=self.shield_params.mu_fe, default_pb=self.shield_params.mu_pb
        )
        normal = self.orientations[orient_idx]
        for j, src in enumerate(self.sources):
            vec = pose - src
            dist = np.linalg.norm(vec)
            if dist == 0:
                dist = 1e-6
            geom = 1.0 / (4.0 * np.pi * dist**2)
            blocked = self.octant_shield.blocks_ray(
                detector_position=pose, source_position=src, octant_index=oct_idx
            )
            L_fe = path_length_cm(vec, normal, self.shield_params.thickness_fe_cm, blocked=blocked)
            L_pb = path_length_cm(vec, normal, self.shield_params.thickness_pb_cm, blocked=blocked)
            att = float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))
            kernels[j] = geom * att
        return kernels

    def expected_counts(
        self,
        isotope: str,
        pose_idx: int,
        orient_idx: int,
        source_strengths: NDArray[np.float64],
        background: float = 0.0,
        live_time_s: float = 1.0,
    ) -> float:
        """
        Compute expected counts from source strengths (simulation/test helper).

        PF updates always use isotope-wise counts from spectrum unfolding (Sec. 2.5.7);
        this helper is not used directly for PF weight updates.
        """
        kvec = self.kernel(isotope, pose_idx, orient_idx)
        return float(live_time_s * (np.dot(kvec, source_strengths) + background))
