"""Continuous 3D kernel evaluations for the Chapter 3.3 measurement model.

Implements geometric and shielded kernels for arbitrary source coordinates,
consistent with Sec. 3.2–3.3 of the thesis (inverse-square law plus attenuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import OctantShield, generate_octant_orientations, octant_index_from_normal


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

    Shield attenuation is applied using an octant-based model: if the line-of-sight
    falls inside the selected octant, an attenuation factor of 0.1 is applied.
    """

    mu_by_isotope: Dict[str, float] | None = None  # kept for future use; not used in 0.1 model
    octant_shield: OctantShield = OctantShield()
    orientations: NDArray[np.float64] = field(default_factory=generate_octant_orientations)

    def attenuation_factor(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        orient_idx: int,
    ) -> float:
        """Return attenuation factor A^{sh} (Sec. 3.2) using a simple 0.1/1.0 model."""
        blocked = self.octant_shield.blocks_ray(detector_position=detector_pos, source_position=source_pos, octant_index=orient_idx)
        return 0.1 if blocked else 1.0

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
        att = self.attenuation_factor(source_pos, detector_pos, orient_idx)
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
) -> float:
    """
    Continuous expected counts Λ_{k,h} for a single isotope and time step (Sec. 3.2–3.3).

    Attenuation model:
        Fe blocks -> 0.1, Pb blocks -> 0.1, both -> 0.01, none -> 1.0.
    RFe / RPb are interpreted as orientation matrices; the third column is used as the
    shield normal. If a 3-vector is passed, it is used directly.
    """
    k = kernel or ContinuousKernel()

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

    lam = background
    for src_pos, q in zip(sources, strengths):
        geom = geometric_term(detector_position, src_pos)
        fe_block = k.octant_shield.blocks_ray(detector_position=detector_position, source_position=src_pos, octant_index=idx_fe)
        pb_block = k.octant_shield.blocks_ray(detector_position=detector_position, source_position=src_pos, octant_index=idx_pb)
        if fe_block and pb_block:
            att = 0.01
        elif fe_block or pb_block:
            att = 0.1
        else:
            att = 1.0
        lam += geom * att * float(q)
    return float(duration * lam)
