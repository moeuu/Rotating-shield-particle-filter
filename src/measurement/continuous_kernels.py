"""Continuous 3D kernel evaluations for the Chapter 3.3 measurement model.

Implements geometric and shielded kernels for arbitrary source coordinates,
consistent with Sec. 3.2–3.3 of the thesis (inverse-square law plus attenuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import ShieldParams
from measurement.shielding import (
    OctantShield,
    generate_octant_orientations,
    octant_index_from_normal,
    path_length_cm,
    resolve_mu_values,
)


def geometric_term(detector: NDArray[np.float64], source: NDArray[np.float64]) -> float:
    """Inverse-square geometric term 1/(4π d^2) (Eq. 3.6)."""
    d = float(np.linalg.norm(detector - source))
    if d == 0.0:
        d = 1e-6
    return float(1.0 / (4.0 * np.pi * d**2))


@dataclass
class ContinuousKernel:
    """
    Continuous-coordinate kernel for Poisson expected counts (Sec. 3.3).

    Shield attenuation is applied using an octant-based model with exponential
    attenuation exp(-mu * L) for Fe/Pb shells.
    """

    mu_by_isotope: Dict[str, object] | None = None
    shield_params: ShieldParams = field(default_factory=ShieldParams)
    octant_shield: OctantShield = OctantShield()
    orientations: NDArray[np.float64] = field(default_factory=generate_octant_orientations)

    def _mu_values(self, isotope: str) -> tuple[float, float]:
        """Return (mu_fe, mu_pb) for the given isotope with fallbacks."""
        return resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )

    def attenuation_factor(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        orient_idx: int,
    ) -> float:
        """
        Return attenuation factor A^{sh} (Sec. 3.2) for a single orientation.

        This treats Fe and Pb shells as sharing the same orientation index.
        """
        normal = self.orientations[orient_idx]
        blocked = self.octant_shield.blocks_ray(
            detector_position=detector_pos,
            source_position=source_pos,
            octant_index=orient_idx,
        )
        direction = detector_pos - source_pos
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        L_fe = path_length_cm(direction, normal, self.shield_params.thickness_fe_cm, blocked=blocked)
        L_pb = path_length_cm(direction, normal, self.shield_params.thickness_pb_cm, blocked=blocked)
        return float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))

    def attenuation_factor_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return combined Fe/Pb attenuation factor A^{sh} (Sec. 3.2)."""
        direction = detector_pos - source_pos
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        normal_fe = self.orientations[fe_index]
        normal_pb = self.orientations[pb_index]
        blocked_fe = self.octant_shield.blocks_ray(
            detector_position=detector_pos,
            source_position=source_pos,
            octant_index=fe_index,
        )
        blocked_pb = self.octant_shield.blocks_ray(
            detector_position=detector_pos,
            source_position=source_pos,
            octant_index=pb_index,
        )
        L_fe = path_length_cm(direction, normal_fe, self.shield_params.thickness_fe_cm, blocked=blocked_fe)
        L_pb = path_length_cm(direction, normal_pb, self.shield_params.thickness_pb_cm, blocked=blocked_pb)
        return float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))

    def kernel_value(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        source_pos: NDArray[np.float64],
        orient_idx: int,
    ) -> float:
        """
        Evaluate K_{k,j,h} = G_{k,j} * A^{sh}_{k,j,h} (Eq. 3.11).
        """
        geom = geometric_term(detector_pos, source_pos)
        att = self.attenuation_factor(isotope, source_pos, detector_pos, orient_idx)
        return geom * att

    def kernel_value_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        source_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Evaluate K_{k,j,h}(R_Fe, R_Pb) for a Fe/Pb orientation pair."""
        geom = geometric_term(detector_pos, source_pos)
        att = self.attenuation_factor_pair(isotope, source_pos, detector_pos, fe_index, pb_index)
        return geom * att

    def expected_rate(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        orient_idx: int,
        background: float = 0.0,
    ) -> float:
        """
        Compute λ_{k,h} = b_h + Σ_j K_{k,j,h} q_{h,j} (Eq. 3.12).
        """
        total = background
        for src_pos, q in zip(sources, strengths):
            total += self.kernel_value(isotope, detector_pos, src_pos, orient_idx) * float(q)
        return float(total)

    def expected_rate_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        background: float = 0.0,
    ) -> float:
        """
        Compute λ_{k,h} for a Fe/Pb orientation pair (Eq. 3.41 with separate R_Fe, R_Pb).
        """
        total = background
        for src_pos, q in zip(sources, strengths):
            total += self.kernel_value_pair(isotope, detector_pos, src_pos, fe_index, pb_index) * float(q)
        return float(total)

    def expected_counts(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        orient_idx: int,
        live_time_s: float = 1.0,
        background: float = 0.0,
    ) -> float:
        """
        Compute Λ_{k,h} = T_k λ_{k,h} (Eq. 3.13).
        """
        rate = self.expected_rate(isotope, detector_pos, sources, strengths, orient_idx, background=background)
        return float(live_time_s * rate)

    def expected_counts_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float = 1.0,
        background: float = 0.0,
    ) -> float:
        """
        Compute Λ_{k,h}(R_Fe, R_Pb) per Eq. (3.41) using octant indices for Fe/Pb.
        """
        rate = self.expected_rate_pair(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=fe_index,
            pb_index=pb_index,
            background=background,
        )
        return float(live_time_s * rate)

    def orient_index_from_vector(self, orientation: NDArray[np.float64]) -> int:
        """Map an orientation vector to the closest octant index."""
        return octant_index_from_normal(orientation)


def expected_counts_single_isotope(
    detector_position: NDArray[np.float64],
    RFe: NDArray[np.float64],
    RPb: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    background: float,
    duration: float,
    isotope_id: str | None = None,
    kernel: ContinuousKernel | None = None,
    mu_by_isotope: Dict[str, object] | None = None,
    shield_params: ShieldParams | None = None,
) -> float:
    """
    Continuous expected counts Λ_{k,h} for a single isotope and time step (Sec. 3.2–3.3).

    RFe / RPb are interpreted as orientation matrices; the third column is used as the
    shield normal. If a 3-vector is passed, it is used directly.
    mu_by_isotope and shield_params are used only when a kernel is not provided.
    """
    if kernel is None:
        k = ContinuousKernel(mu_by_isotope=mu_by_isotope, shield_params=shield_params or ShieldParams())
    else:
        k = kernel

    def _normal_from_R(R: NDArray[np.float64]) -> NDArray[np.float64]:
        if R.ndim == 1:
            return np.asarray(R, dtype=float)
        if R.shape == (3, 3):
            return np.asarray(R[:, 2], dtype=float)
        raise ValueError("RFe/RPb must be shape (3,) or (3,3)")

    n_fe = _normal_from_R(RFe)
    n_pb = _normal_from_R(RPb)
    idx_fe = k.orient_index_from_vector(n_fe)
    idx_pb = k.orient_index_from_vector(n_pb)

    return k.expected_counts_pair(
        isotope=isotope_id or "generic",
        detector_pos=detector_position,
        sources=sources,
        strengths=strengths,
        fe_index=idx_fe,
        pb_index=idx_pb,
        live_time_s=duration,
        background=background,
    )
