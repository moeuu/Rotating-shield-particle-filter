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

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


def geometric_term(detector: NDArray[np.float64], source: NDArray[np.float64]) -> float:
    """Inverse-square geometric term 1/d^2 (cps@1m scaling)."""
    d = float(np.linalg.norm(detector - source))
    if d == 0.0:
        d = 1e-6
    return float(1.0 / (d**2))


def _torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(_TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def _torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(_TORCH_AVAILABLE and torch is not None)


def _resolve_device(device: str | None) -> "torch.device":
    """Resolve a torch device string with CUDA fallback."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if device is None:
        device = "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> "torch.dtype":
    """Map a dtype string to a torch dtype."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


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
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"

    def _mu_values(self, isotope: str) -> tuple[float, float]:
        """Return (mu_fe, mu_pb) for the given isotope with fallbacks."""
        return resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        if not self.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu for ContinuousKernel.")
        if not _torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

    def _blocked_mask_torch(
        self,
        dir_unit: "torch.Tensor",
        octant_index: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return a boolean mask for rays blocked by the selected octant (torch)."""
        (theta_low, theta_high), (phi_low, phi_high) = self.octant_shield.theta_phi_ranges[octant_index]
        theta = torch.acos(torch.clamp(dir_unit[:, 2], -1.0, 1.0))
        phi = torch.remainder(torch.atan2(dir_unit[:, 1], dir_unit[:, 0]), 2.0 * np.pi)
        tol_t = torch.as_tensor(tol, device=dir_unit.device, dtype=dir_unit.dtype)
        return (
            (theta + tol_t >= theta_low)
            & (theta - tol_t < theta_high)
            & (phi + tol_t >= phi_low)
            & (phi - tol_t < phi_high)
        )

    def _expected_rate_pair_torch(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        background: float,
        tol: float = 1e-6,
    ) -> float:
        """Compute expected rate for a Fe/Pb orientation pair using torch."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return float(background)
        strengths_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 3)
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        dir_unit = direction / dist.unsqueeze(-1)
        geom = 1.0 / (dist**2)

        blocked_fe = self._blocked_mask_torch(dir_unit, fe_index, tol)
        blocked_pb = self._blocked_mask_torch(dir_unit, pb_index, tol)
        normal_fe = torch.as_tensor(self.orientations[fe_index], device=device, dtype=dtype)
        normal_pb = torch.as_tensor(self.orientations[pb_index], device=device, dtype=dtype)
        cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=1), 0.0, 1.0)
        cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=1), 0.0, 1.0)
        if self.shield_params.use_angle_attenuation:
            L_fe = torch.where(
                blocked_fe & (cos_fe > tol_t),
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype) / cos_fe,
                torch.zeros_like(cos_fe),
            )
            L_pb = torch.where(
                blocked_pb & (cos_pb > tol_t),
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype) / cos_pb,
                torch.zeros_like(cos_pb),
            )
        else:
            L_fe = torch.where(
                blocked_fe,
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype),
                torch.zeros_like(cos_fe),
            )
            L_pb = torch.where(
                blocked_pb,
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype),
                torch.zeros_like(cos_pb),
            )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        att = torch.exp(-(mu_fe * L_fe + mu_pb * L_pb))
        rate = torch.sum(geom * att * strengths_t) + float(background)
        return float(rate.detach().cpu().item())

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
        L_fe = path_length_cm(
            direction,
            normal,
            self.shield_params.thickness_fe_cm,
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )
        L_pb = path_length_cm(
            direction,
            normal,
            self.shield_params.thickness_pb_cm,
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )
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
        L_fe = path_length_cm(
            direction,
            normal_fe,
            self.shield_params.thickness_fe_cm,
            blocked=blocked_fe,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )
        L_pb = path_length_cm(
            direction,
            normal_pb,
            self.shield_params.thickness_pb_cm,
            blocked=blocked_pb,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )
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
        return self.expected_rate_pair(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=orient_idx,
            pb_index=orient_idx,
            background=background,
        )

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
        self._gpu_enabled()
        return self._expected_rate_pair_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            sources=sources,
            strengths=strengths,
            fe_index=fe_index,
            pb_index=pb_index,
            background=background,
        )

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
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> float:
    """
    Continuous expected counts Λ_{k,h} for a single isotope and time step (Sec. 3.2–3.3).

    RFe / RPb are interpreted as orientation matrices; the third column is used as the
    shield normal. If a 3-vector is passed, it is used directly.
    mu_by_isotope and shield_params are used only when a kernel is not provided.
    use_gpu controls optional CUDA acceleration for batch kernel evaluation.
    """
    if kernel is None:
        k = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params or ShieldParams(),
            use_gpu=bool(use_gpu) if use_gpu is not None else True,
            gpu_device=gpu_device,
            gpu_dtype=gpu_dtype,
        )
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
