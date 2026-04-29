"""Continuous 3D kernel evaluations for the Chapter 3.3 measurement model.

Implements geometric and shielded kernels for arbitrary source coordinates,
consistent with Sec. 3.2–3.3 of the thesis (inverse-square law plus attenuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    CONCRETE_MU_CM_INV,
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    octant_index_from_normal,
    path_length_cm,
    resolve_mu_values,
    spherical_shell_path_length_cm,
    spherical_shell_path_length_cm_torch,
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


def _normalize_isotope_key(isotope: str) -> str:
    """Return a normalized isotope key for table lookups."""
    return re.sub(r"[^A-Za-z0-9]", "", str(isotope)).upper()


def resolve_obstacle_mu_cm_inv(
    isotope: str,
    mu_by_isotope: Dict[str, float] | None = None,
) -> float:
    """Resolve concrete obstacle attenuation coefficient in 1/cm for an isotope."""
    table = mu_by_isotope if mu_by_isotope is not None else CONCRETE_MU_CM_INV
    if isotope in table:
        return float(table[isotope])
    normalized = {_normalize_isotope_key(key): float(value) for key, value in table.items()}
    norm_key = _normalize_isotope_key(isotope)
    if norm_key in normalized:
        return normalized[norm_key]
    raise ValueError(
        "Concrete obstacle attenuation is enabled but no coefficient is defined "
        f"for isotope {isotope!r}."
    )


def segment_box_intersection_length_m(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    box_m: NDArray[np.float64],
    tol: float = 1e-12,
) -> float:
    """Return the line-segment path length inside one axis-aligned box in meters."""
    source = np.asarray(source_pos, dtype=float)
    detector = np.asarray(detector_pos, dtype=float)
    box = np.asarray(box_m, dtype=float)
    if source.shape != (3,) or detector.shape != (3,) or box.shape != (6,):
        raise ValueError("source_pos, detector_pos, and box_m must have shapes (3,), (3,), and (6,).")
    direction = detector - source
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= tol:
        return 0.0
    lower = box[:3]
    upper = box[3:]
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        value = source[axis]
        delta = direction[axis]
        lo = lower[axis]
        hi = upper[axis]
        if abs(delta) <= tol:
            if value < lo or value > hi:
                return 0.0
            continue
        t0 = (lo - value) / delta
        t1 = (hi - value) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, float(t0))
        t_exit = min(t_exit, float(t1))
        if t_exit <= t_enter:
            return 0.0
    return max(0.0, t_exit - t_enter) * segment_length


def obstacle_path_length_cm(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
) -> float:
    """Return total source-detector path length inside obstacle boxes in centimeters."""
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    if boxes.size == 0:
        return 0.0
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    path_m = 0.0
    for box in boxes:
        path_m += segment_box_intersection_length_m(source_pos, detector_pos, box)
    return float(100.0 * path_m)


def obstacle_path_lengths_cm_torch(
    positions: "torch.Tensor",
    detector_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return batched obstacle path lengths through axis-aligned boxes in centimeters."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(positions.shape[:-1], device=positions.device, dtype=positions.dtype)
    if obstacle_boxes_m.ndim != 2 or obstacle_boxes_m.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    detector = detector_pos.to(device=positions.device, dtype=positions.dtype)
    detector = detector.view(*([1] * (positions.ndim - 1)), 3)
    direction = detector - positions
    distance = torch.linalg.norm(direction, dim=-1)
    p0 = positions.unsqueeze(-2)
    delta = direction.unsqueeze(-2)
    lower = obstacle_boxes_m[:, :3].to(device=positions.device, dtype=positions.dtype)
    upper = obstacle_boxes_m[:, 3:].to(device=positions.device, dtype=positions.dtype)
    tol_t = torch.as_tensor(tol, device=positions.device, dtype=positions.dtype)
    t_min_axes = []
    t_max_axes = []
    for axis in range(3):
        value = p0[..., axis]
        step = delta[..., axis]
        lo = lower[:, axis]
        hi = upper[:, axis]
        parallel = torch.abs(step) <= tol_t
        inside = (value >= lo) & (value <= hi)
        safe_step = torch.where(parallel, torch.ones_like(step), step)
        t0 = (lo - value) / safe_step
        t1 = (hi - value) / safe_step
        axis_min = torch.minimum(t0, t1)
        axis_max = torch.maximum(t0, t1)
        neg_inf = torch.full_like(axis_min, -float("inf"))
        pos_inf = torch.full_like(axis_max, float("inf"))
        axis_min = torch.where(parallel & inside, neg_inf, axis_min)
        axis_max = torch.where(parallel & inside, pos_inf, axis_max)
        axis_min = torch.where(parallel & ~inside, pos_inf, axis_min)
        axis_max = torch.where(parallel & ~inside, neg_inf, axis_max)
        t_min_axes.append(axis_min)
        t_max_axes.append(axis_max)
    t_enter = torch.maximum(torch.stack(t_min_axes, dim=-1).amax(dim=-1), torch.zeros_like(distance).unsqueeze(-1))
    t_exit = torch.minimum(torch.stack(t_max_axes, dim=-1).amin(dim=-1), torch.ones_like(distance).unsqueeze(-1))
    length_m = torch.where(t_exit > t_enter, (t_exit - t_enter) * distance.unsqueeze(-1), torch.zeros_like(t_exit))
    return 100.0 * torch.sum(length_m, dim=-1)


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
    obstacle_grid: ObstacleGrid | None = None
    obstacle_height_m: float = 2.0
    obstacle_mu_by_isotope: Dict[str, float] | None = None
    _obstacle_boxes_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate obstacle attenuation settings."""
        self.obstacle_height_m = float(self.obstacle_height_m)
        if self.obstacle_height_m < 0.0:
            raise ValueError("obstacle_height_m must be non-negative.")

    def _mu_values(self, isotope: str) -> tuple[float, float]:
        """Return (mu_fe, mu_pb) for the given isotope with fallbacks."""
        return resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )

    def obstacle_boxes_m(self) -> NDArray[np.float64]:
        """Return cached obstacle boxes in meters as (x0, y0, z0, x1, y1, z1)."""
        if self.obstacle_grid is None:
            return np.zeros((0, 6), dtype=float)
        if self._obstacle_boxes_cache is None:
            boxes = self.obstacle_grid.blocked_boxes(
                z_min=0.0,
                z_max=float(self.obstacle_height_m),
            )
            if boxes:
                self._obstacle_boxes_cache = np.asarray(boxes, dtype=float)
            else:
                self._obstacle_boxes_cache = np.zeros((0, 6), dtype=float)
        return self._obstacle_boxes_cache.copy()

    def obstacle_mu_cm_inv(self, isotope: str) -> float:
        """Return concrete obstacle attenuation coefficient in 1/cm for an isotope."""
        if self.obstacle_grid is None:
            return 0.0
        return resolve_obstacle_mu_cm_inv(isotope, self.obstacle_mu_by_isotope)

    def obstacle_path_length_cm(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return total source-detector path length inside configured obstacles in centimeters."""
        return obstacle_path_length_cm(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
        )

    def _obstacle_attenuation_factor(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return Beer-Lambert attenuation through concrete obstacle cells."""
        if self.obstacle_grid is None:
            return 1.0
        path_cm = self.obstacle_path_length_cm(source_pos, detector_pos)
        if path_cm <= 0.0:
            return 1.0
        return float(np.exp(-self.obstacle_mu_cm_inv(isotope) * path_cm))

    def obstacle_gpu_kwargs(self, isotope: str) -> dict[str, object]:
        """Return optional GPU kwargs for obstacle attenuation."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0:
            return {}
        return {
            "obstacle_boxes_m": boxes,
            "obstacle_mu_cm_inv": self.obstacle_mu_cm_inv(isotope),
        }

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

    def _shield_path_length_cm(
        self,
        direction_m: NDArray[np.float64],
        normal: NDArray[np.float64],
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return Pb/Fe path length through the configured shield geometry."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return spherical_shell_path_length_cm(
                direction_m=direction_m,
                inner_radius_cm=inner_radius_cm,
                outer_radius_cm=inner_radius_cm + thickness_cm,
                blocked=blocked,
            )
        return path_length_cm(
            direction_m,
            normal,
            thickness_cm,
            blocked=blocked,
            use_angle_attenuation=self.shield_params.use_angle_attenuation,
        )

    def _shield_path_lengths_torch(
        self,
        direction: "torch.Tensor",
        blocked_fe: "torch.Tensor",
        blocked_pb: "torch.Tensor",
        cos_fe: "torch.Tensor",
        cos_pb: "torch.Tensor",
        tol_t: "torch.Tensor",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return Fe/Pb path lengths through the configured shield geometry."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            l_fe = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_fe_cm,
                self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                blocked_fe,
            )
            l_pb = spherical_shell_path_length_cm_torch(
                direction,
                self.shield_params.inner_radius_pb_cm,
                self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                blocked_pb,
            )
            return l_fe, l_pb
        if self.shield_params.use_angle_attenuation:
            l_fe = torch.where(
                blocked_fe & (cos_fe > tol_t),
                torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype) / cos_fe,
                torch.zeros_like(cos_fe),
            )
            l_pb = torch.where(
                blocked_pb & (cos_pb > tol_t),
                torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype) / cos_pb,
                torch.zeros_like(cos_pb),
            )
            return l_fe, l_pb
        l_fe = torch.where(
            blocked_fe,
            torch.as_tensor(self.shield_params.thickness_fe_cm, device=device, dtype=dtype),
            torch.zeros_like(cos_fe),
        )
        l_pb = torch.where(
            blocked_pb,
            torch.as_tensor(self.shield_params.thickness_pb_cm, device=device, dtype=dtype),
            torch.zeros_like(cos_pb),
        )
        return l_fe, l_pb

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
        L_fe, L_pb = self._shield_path_lengths_torch(
            direction=direction,
            blocked_fe=blocked_fe,
            blocked_pb=blocked_pb,
            cos_fe=cos_fe,
            cos_pb=cos_pb,
            tol_t=tol_t,
            device=device,
            dtype=dtype,
        )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        att = torch.exp(-(mu_fe * L_fe + mu_pb * L_pb))
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            obstacle_path_cm = obstacle_path_lengths_cm_torch(
                positions=sources_t,
                detector_pos=detector_t.reshape(3),
                obstacle_boxes_m=boxes_t,
            )
            att = att * torch.exp(-self.obstacle_mu_cm_inv(isotope) * obstacle_path_cm)
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
        L_fe = self._shield_path_length_cm(
            direction_m=direction,
            normal=normal,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=blocked,
        )
        L_pb = self._shield_path_length_cm(
            direction_m=direction,
            normal=normal,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=blocked,
        )
        shield_att = float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))
        return shield_att * self._obstacle_attenuation_factor(isotope, source_pos, detector_pos)

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
        L_fe = self._shield_path_length_cm(
            direction_m=direction,
            normal=normal_fe,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=blocked_fe,
        )
        L_pb = self._shield_path_length_cm(
            direction_m=direction,
            normal=normal_pb,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=blocked_pb,
        )
        shield_att = float(np.exp(-(mu_fe * L_fe + mu_pb * L_pb)))
        return shield_att * self._obstacle_attenuation_factor(isotope, source_pos, detector_pos)

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
        if not self.use_gpu:
            rate = float(background)
            sources_arr = np.asarray(sources, dtype=float)
            strengths_arr = np.asarray(strengths, dtype=float)
            if sources_arr.size == 0:
                return rate
            for source_pos, strength in zip(sources_arr, strengths_arr):
                rate += float(strength) * self.kernel_value_pair(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    source_pos=source_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
            return float(rate)
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
