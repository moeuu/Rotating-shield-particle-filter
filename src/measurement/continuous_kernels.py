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
    LOCAL_POSITIVE_OCTANT_CENTER,
    OctantShield,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    octant_index_from_normal,
    path_length_cm,
    rotated_positive_octant_blocks_direction,
    resolve_mu_values,
    rotation_matrix_between_vectors,
    spherical_shell_path_length_cm,
    spherical_shell_path_length_cm_torch,
)

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


def _finite_sphere_geometric_term_torch(
    distance: "torch.Tensor",
    *,
    detector_radius_m: float,
    tol: "torch.Tensor",
) -> "torch.Tensor":
    """Return finite-sphere detector-cps@1m scaling for torch distances."""
    if torch is None:
        raise RuntimeError("torch is not available")
    radius = max(float(detector_radius_m), 0.0)
    if radius <= 0.0:
        dist = torch.clamp(distance, min=tol)
        return 1.0 / (dist**2)
    radius_t = torch.as_tensor(radius, device=distance.device, dtype=distance.dtype)
    d_eff = torch.maximum(torch.clamp(distance, min=tol), radius_t)
    ratio = torch.clamp(radius_t / torch.clamp(d_eff, min=tol), max=1.0)
    fraction = 0.5 * (1.0 - torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0)))
    reference = max(1.0, radius)
    ref_ratio = min(radius / reference, 1.0)
    ref_fraction = max(0.5 * (1.0 - float(np.sqrt(max(1.0 - ref_ratio * ref_ratio, 0.0)))), 1.0e-12)
    return fraction / ref_fraction


def geometric_term(detector: NDArray[np.float64], source: NDArray[np.float64]) -> float:
    """Inverse-square geometric term 1/d^2 for detector cps@1m scaling."""
    d = float(np.linalg.norm(detector - source))
    if d == 0.0:
        d = 1e-6
    return float(1.0 / (d**2))


def finite_sphere_geometric_term(
    detector: NDArray[np.float64],
    source: NDArray[np.float64],
    detector_radius_m: float,
) -> float:
    """
    Return detector-cps@1m scaling for a finite spherical detector.

    For a configured spherical detector, ``intensity_cps_1m`` is defined as the
    expected detector count rate at 1 m.  The near-field scaling should
    therefore use the sphere solid angle relative to the 1 m solid angle, not a
    point-detector singularity.  When no detector radius is configured this
    falls back to the inverse-square term.
    """
    radius = max(float(detector_radius_m), 0.0)
    if radius <= 0.0:
        return geometric_term(detector, source)
    d = float(np.linalg.norm(np.asarray(detector, dtype=float) - np.asarray(source, dtype=float)))
    d_eff = max(d, radius)
    reference = max(1.0, radius)

    def _sphere_fraction(distance: float) -> float:
        """Return the external point-source solid-angle fraction of a sphere."""
        if distance <= radius:
            return 0.5
        ratio = min(radius / max(distance, 1.0e-12), 1.0)
        return 0.5 * (1.0 - float(np.sqrt(max(1.0 - ratio * ratio, 0.0))))

    ref_fraction = max(_sphere_fraction(reference), 1.0e-12)
    return float(_sphere_fraction(d_eff) / ref_fraction)


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


def segment_sphere_intersection_length_m(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    radius_m: float,
    tol: float = 1e-12,
) -> float:
    """Return segment length inside a sphere in meters."""
    source = np.asarray(source_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    center = np.asarray(center_pos, dtype=float)
    radius = max(float(radius_m), 0.0)
    direction = target - source
    segment_length = float(np.linalg.norm(direction))
    if radius <= 0.0 or segment_length <= tol:
        return 0.0
    rel_source = source - center
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(rel_source, direction))
    c = float(np.dot(rel_source, rel_source)) - radius * radius
    rel_target = target - center
    source_inside = c <= 0.0
    target_inside = float(np.dot(rel_target, rel_target)) <= radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return segment_length if source_inside and target_inside else 0.0
    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    enter = max(0.0, min(t0, t1))
    exit_ = min(1.0, max(t0, t1))
    if exit_ <= enter:
        return segment_length if source_inside and target_inside else 0.0
    return max(0.0, exit_ - enter) * segment_length


def segment_spherical_shell_path_length_cm(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    blocked: bool,
) -> float:
    """Return segment length through a spherical shell centered at center_pos."""
    if not blocked:
        return 0.0
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return 0.0
    outer_length = segment_sphere_intersection_length_m(
        source_pos,
        target_pos,
        center_pos,
        outer_m,
    )
    inner_length = segment_sphere_intersection_length_m(
        source_pos,
        target_pos,
        center_pos,
        inner_m,
    )
    return float(100.0 * max(outer_length - inner_length, 0.0))


def segment_rotated_octant_shell_path_length_cm(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    shield_normal: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-12,
) -> float:
    """Return exact segment length inside a rotated local +X/+Y/+Z octant shell."""
    source = np.asarray(source_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    center = np.asarray(center_pos, dtype=float)
    rotation = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        np.asarray(shield_normal, dtype=float),
    )
    source_local = rotation.T @ (source - center)
    target_local = rotation.T @ (target - center)
    delta = target_local - source_local
    segment_length = float(np.linalg.norm(delta))
    if segment_length <= tol:
        return 0.0
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return 0.0
    breakpoints: list[float] = [0.0, 1.0]
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(source_local, delta))
    for radius in (inner_m, outer_m):
        c = float(np.dot(source_local, source_local)) - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = float(np.sqrt(max(discriminant, 0.0)))
        breakpoints.extend(
            value
            for value in (
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            )
            if -tol <= value <= 1.0 + tol
        )
    for axis in range(3):
        if abs(float(delta[axis])) <= tol:
            continue
        value = -float(source_local[axis]) / float(delta[axis])
        if -tol <= value <= 1.0 + tol:
            breakpoints.append(value)
    clipped = sorted({float(np.clip(value, 0.0, 1.0)) for value in breakpoints})
    length = 0.0
    for left, right in zip(clipped[:-1], clipped[1:]):
        if right - left <= tol:
            continue
        mid = 0.5 * (left + right)
        point = source_local + mid * delta
        radius_sq = float(np.dot(point, point))
        inside_shell = (inner_m * inner_m - tol) <= radius_sq <= (outer_m * outer_m + tol)
        inside_octant = bool(np.all(point >= -tol))
        if inside_shell and inside_octant:
            length += (right - left) * segment_length
    return float(100.0 * length)


def segment_rotated_octant_shell_path_length_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    center_pos: "torch.Tensor",
    shield_normal: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-9,
) -> "torch.Tensor":
    """Return exact segment length inside a rotated octant shell for torch tensors."""
    if torch is None:
        raise RuntimeError("torch is not available")
    rotation_np = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        np.asarray(shield_normal, dtype=float),
    )
    rotation = torch.as_tensor(rotation_np, device=source_pos.device, dtype=source_pos.dtype)
    source_local = (source_pos - center_pos) @ rotation
    target_local = (target_pos - center_pos) @ rotation
    delta = target_local - source_local
    segment_length = torch.linalg.norm(delta, dim=-1)
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return torch.zeros_like(segment_length)
    a = torch.sum(delta * delta, dim=-1)
    b = 2.0 * torch.sum(source_local * delta, dim=-1)
    breakpoints = [
        torch.zeros_like(segment_length),
        torch.ones_like(segment_length),
    ]
    for radius in (inner_m, outer_m):
        c = torch.sum(source_local * source_local, dim=-1) - float(radius * radius)
        discriminant = b * b - 4.0 * a * c
        root = torch.sqrt(torch.clamp(discriminant, min=0.0))
        denom = torch.clamp(2.0 * a, min=float(tol))
        valid = discriminant >= 0.0
        for sign in (-1.0, 1.0):
            value = (-b + sign * root) / denom
            value = torch.where(valid, value, torch.zeros_like(value))
            breakpoints.append(torch.clamp(value, 0.0, 1.0))
    for axis in range(3):
        axis_delta = delta[..., axis]
        valid = torch.abs(axis_delta) > float(tol)
        value = -source_local[..., axis] / torch.where(
            valid,
            axis_delta,
            torch.ones_like(axis_delta),
        )
        value = torch.where(valid, value, torch.zeros_like(value))
        breakpoints.append(torch.clamp(value, 0.0, 1.0))
    ordered = torch.sort(torch.stack(breakpoints, dim=-1), dim=-1).values
    left = ordered[..., :-1]
    right = ordered[..., 1:]
    width = torch.clamp(right - left, min=0.0)
    mid = 0.5 * (left + right)
    point = source_local.unsqueeze(-2) + mid.unsqueeze(-1) * delta.unsqueeze(-2)
    radius_sq = torch.sum(point * point, dim=-1)
    inside_shell = (radius_sq >= inner_m * inner_m - float(tol)) & (
        radius_sq <= outer_m * outer_m + float(tol)
    )
    inside_octant = torch.all(point >= -float(tol), dim=-1)
    length_m = torch.sum(
        torch.where(inside_shell & inside_octant, width, torch.zeros_like(width)),
        dim=-1,
    ) * segment_length
    return torch.where(segment_length > float(tol), 100.0 * length_m, torch.zeros_like(length_m))


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


def obstacle_path_lengths_between_points_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return path lengths through axis-aligned boxes for source-target segments."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(source_pos.shape[:-1], device=source_pos.device, dtype=source_pos.dtype)
    if obstacle_boxes_m.ndim != 2 or obstacle_boxes_m.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    p0 = source_pos.unsqueeze(-2)
    delta = (target_pos - source_pos).unsqueeze(-2)
    distance = torch.linalg.norm(target_pos - source_pos, dim=-1)
    lower = obstacle_boxes_m[:, :3].to(device=source_pos.device, dtype=source_pos.dtype)
    upper = obstacle_boxes_m[:, 3:].to(device=source_pos.device, dtype=source_pos.dtype)
    tol_t = torch.as_tensor(tol, device=source_pos.device, dtype=source_pos.dtype)
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
    t_enter = torch.maximum(
        torch.stack(t_min_axes, dim=-1).amax(dim=-1),
        torch.zeros_like(distance).unsqueeze(-1),
    )
    t_exit = torch.minimum(
        torch.stack(t_max_axes, dim=-1).amin(dim=-1),
        torch.ones_like(distance).unsqueeze(-1),
    )
    length_m = torch.where(
        t_exit > t_enter,
        (t_exit - t_enter) * distance.unsqueeze(-1),
        torch.zeros_like(t_exit),
    )
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
    obstacle_buildup_coeff: float = 0.0
    detector_radius_m: float = 0.0
    detector_aperture_samples: int = 1
    _obstacle_boxes_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate obstacle attenuation settings."""
        self.obstacle_height_m = float(self.obstacle_height_m)
        if self.obstacle_height_m < 0.0:
            raise ValueError("obstacle_height_m must be non-negative.")
        self.obstacle_buildup_coeff = max(float(self.obstacle_buildup_coeff), 0.0)
        self.shield_params = ShieldParams(
            mu_pb=float(self.shield_params.mu_pb),
            mu_fe=float(self.shield_params.mu_fe),
            thickness_pb_cm=float(self.shield_params.thickness_pb_cm),
            thickness_fe_cm=float(self.shield_params.thickness_fe_cm),
            inner_radius_fe_cm=float(self.shield_params.inner_radius_fe_cm),
            inner_radius_pb_cm=float(self.shield_params.inner_radius_pb_cm),
            buildup_fe_coeff=max(float(self.shield_params.buildup_fe_coeff), 0.0),
            buildup_pb_coeff=max(float(self.shield_params.buildup_pb_coeff), 0.0),
            shield_geometry_model=self.shield_params.shield_geometry_model,
            use_angle_attenuation=bool(self.shield_params.use_angle_attenuation),
        )
        self.detector_radius_m = max(float(self.detector_radius_m), 0.0)
        self.detector_aperture_samples = max(int(self.detector_aperture_samples), 1)

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
            "obstacle_buildup_coeff": self.obstacle_buildup_coeff,
        }

    def _buildup_factor(
        self,
        tau_fe: float,
        tau_pb: float,
        tau_obstacle: float,
    ) -> float:
        """Return a bounded broad-beam build-up factor from optical depths."""
        factor = 1.0
        factor += self.shield_params.buildup_fe_coeff * (1.0 - float(np.exp(-max(tau_fe, 0.0))))
        factor += self.shield_params.buildup_pb_coeff * (1.0 - float(np.exp(-max(tau_pb, 0.0))))
        factor += self.obstacle_buildup_coeff * (1.0 - float(np.exp(-max(tau_obstacle, 0.0))))
        return max(1.0, float(factor))

    def _buildup_factor_torch(
        self,
        tau_fe: "torch.Tensor",
        tau_pb: "torch.Tensor",
        tau_obstacle: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a broad-beam build-up factor for torch optical depths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        factor = torch.ones_like(tau_fe)
        factor = factor + self.shield_params.buildup_fe_coeff * (1.0 - torch.exp(-torch.clamp(tau_fe, min=0.0)))
        factor = factor + self.shield_params.buildup_pb_coeff * (1.0 - torch.exp(-torch.clamp(tau_pb, min=0.0)))
        factor = factor + self.obstacle_buildup_coeff * (
            1.0 - torch.exp(-torch.clamp(tau_obstacle, min=0.0))
        )
        return torch.clamp(factor, min=1.0)

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

    def _rotated_octant_blocked_mask_torch(
        self,
        detector_to_source_unit: "torch.Tensor",
        octant_index: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return a mask for the rotated local +X/+Y/+Z shield octant."""
        if torch is None:
            raise RuntimeError("torch is not available")
        physical_normal = -np.asarray(self.orientations[octant_index], dtype=float)
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            physical_normal,
        )
        rotation = torch.as_tensor(
            rotation_np,
            device=detector_to_source_unit.device,
            dtype=detector_to_source_unit.dtype,
        )
        local_direction = detector_to_source_unit @ rotation
        return torch.all(local_direction >= -float(tol), dim=-1)

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

    def _shield_segment_path_length_cm(
        self,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_center: NDArray[np.float64],
        incoming_normal: NDArray[np.float64],
        physical_normal: NDArray[np.float64],
        thickness_cm: float,
        inner_radius_cm: float,
        blocked: bool,
    ) -> float:
        """Return shield path length for a source-to-detector-aperture segment."""
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            return segment_rotated_octant_shell_path_length_cm(
                source_pos=source_pos,
                target_pos=target_pos,
                center_pos=detector_center,
                shield_normal=physical_normal,
                inner_radius_cm=inner_radius_cm,
                outer_radius_cm=inner_radius_cm + thickness_cm,
            )
        return self._shield_path_length_cm(
            direction_m=np.asarray(target_pos, dtype=float) - np.asarray(source_pos, dtype=float),
            normal=incoming_normal,
            thickness_cm=thickness_cm,
            inner_radius_cm=inner_radius_cm,
            blocked=blocked,
        )

    def _detector_aperture_targets(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return deterministic target points on the detector aperture disk."""
        detector = np.asarray(detector_pos, dtype=float)
        source = np.asarray(source_pos, dtype=float)
        if self.detector_radius_m <= 0.0 or self.detector_aperture_samples <= 1:
            return detector.reshape(1, 3)
        axis = detector - source
        distance = float(np.linalg.norm(axis))
        if distance <= 1.0e-12:
            return detector.reshape(1, 3)
        axis /= distance
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = np.cross(axis, helper)
        basis_u_norm = float(np.linalg.norm(basis_u))
        if basis_u_norm <= 1.0e-12:
            return detector.reshape(1, 3)
        basis_u /= basis_u_norm
        basis_v = np.cross(axis, basis_u)
        radius = min(self.detector_radius_m, 0.95 * distance)
        count = int(self.detector_aperture_samples)
        targets = np.empty((count, 3), dtype=float)
        targets[0] = detector
        if count == 1:
            return targets
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(1, count):
            fraction = (float(index) - 0.5) / float(count - 1)
            sample_radius = radius * float(np.sqrt(np.clip(fraction, 0.0, 1.0)))
            angle = golden_angle * float(index)
            offset = sample_radius * (
                float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v
            )
            targets[index] = detector + offset
        return targets

    def _attenuation_factor_for_target(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return attenuation for one source-to-aperture ray."""
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        normal_fe = self.orientations[fe_index]
        normal_pb = self.orientations[pb_index]
        detector_to_source = np.asarray(source_pos, dtype=float) - np.asarray(target_pos, dtype=float)
        blocked_fe = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_fe,
        )
        blocked_pb = rotated_positive_octant_blocks_direction(
            detector_to_source,
            -normal_pb,
        )
        l_fe = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_fe,
            physical_normal=-normal_fe,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            blocked=blocked_fe,
        )
        l_pb = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            incoming_normal=normal_pb,
            physical_normal=-normal_pb,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            blocked=blocked_pb,
        )
        tau_fe = float(mu_fe * l_fe)
        tau_pb = float(mu_pb * l_pb)
        tau_obstacle = 0.0
        if self.obstacle_grid is None:
            buildup = self._buildup_factor(tau_fe, tau_pb, tau_obstacle)
            return float(min(1.0, np.exp(-(tau_fe + tau_pb)) * buildup))
        obstacle_path_cm = obstacle_path_length_cm(
            source_pos=source_pos,
            detector_pos=target_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
        )
        if obstacle_path_cm > 0.0:
            tau_obstacle = float(self.obstacle_mu_cm_inv(isotope) * obstacle_path_cm)
        total_tau = tau_fe + tau_pb + tau_obstacle
        buildup = self._buildup_factor(tau_fe, tau_pb, tau_obstacle)
        return float(min(1.0, np.exp(-total_tau) * buildup))

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

    def _detector_aperture_targets_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", int]:
        """Return deterministic detector-aperture targets for torch sources."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if self.detector_radius_m <= 0.0 or self.detector_aperture_samples <= 1:
            return detector.expand_as(sources).unsqueeze(-2), 1
        sample_count = max(int(self.detector_aperture_samples), 1)
        axis = (detector - sources) / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u = basis_u / torch.clamp(torch.linalg.norm(basis_u, dim=-1, keepdim=True), min=tol)
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        indices = torch.arange(sample_count, device=sources.device, dtype=sources.dtype)
        fractions = torch.clamp((indices - 0.5) / float(sample_count - 1), min=0.0, max=1.0)
        radii = torch.sqrt(fractions)
        radii[0] = 0.0
        max_radius = torch.minimum(
            torch.as_tensor(self.detector_radius_m, device=sources.device, dtype=sources.dtype),
            0.95 * dist,
        )
        angles = indices * torch.as_tensor(np.pi * (3.0 - np.sqrt(5.0)), device=sources.device, dtype=sources.dtype)
        offsets = (
            max_radius.unsqueeze(-1).unsqueeze(-1)
            * radii.view(1, sample_count, 1)
            * (
                torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
            )
        )
        return detector.expand_as(sources).unsqueeze(-2) + offsets, sample_count

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
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        targets, sample_count = self._detector_aperture_targets_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_sources = sources_t.unsqueeze(-2).expand_as(targets)
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
        dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)

        detector_to_source_unit = -dir_unit
        blocked_fe = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, fe_index, tol)
        blocked_pb = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, pb_index, tol)
        normal_fe = torch.as_tensor(self.orientations[fe_index], device=device, dtype=dtype)
        normal_pb = torch.as_tensor(self.orientations[pb_index], device=device, dtype=dtype)
        cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=-1), 0.0, 1.0)
        cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=-1), 0.0, 1.0)
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            center = detector_t.expand_as(sources_t).unsqueeze(-2)
            L_fe = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=-np.asarray(self.orientations[fe_index], dtype=float),
                inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                outer_radius_cm=self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                tol=tol,
            )
            L_pb = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=-np.asarray(self.orientations[pb_index], dtype=float),
                inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                outer_radius_cm=self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                tol=tol,
            )
        else:
            L_fe, L_pb = self._shield_path_lengths_torch(
                direction=sampled_direction,
                blocked_fe=blocked_fe,
                blocked_pb=blocked_pb,
                cos_fe=cos_fe,
                cos_pb=cos_pb,
                tol_t=tol_t,
                device=device,
                dtype=dtype,
            )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = float(mu_fe) * L_fe
        tau_pb = float(mu_pb) * L_pb
        tau_obstacle = torch.zeros_like(tau_fe)
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            if sample_count > 1:
                obstacle_path_cm = obstacle_path_lengths_between_points_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
            else:
                obstacle_path_cm = obstacle_path_lengths_cm_torch(
                    positions=sources_t,
                    detector_pos=detector_t.reshape(3),
                    obstacle_boxes_m=boxes_t,
                ).unsqueeze(-1)
            tau_obstacle = float(self.obstacle_mu_cm_inv(isotope)) * obstacle_path_cm
        total_tau = tau_fe + tau_pb + tau_obstacle
        buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
        att = torch.clamp(torch.exp(-total_tau) * buildup, max=1.0)
        att = torch.mean(att, dim=-1)
        rate = torch.sum(geom * att * strengths_t) + float(background)
        return float(rate.detach().cpu().item())

    def _kernel_values_pair_torch_chunk(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return per-source kernel values for one GPU chunk."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return np.zeros(0, dtype=float)
        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 3)
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        targets, sample_count = self._detector_aperture_targets_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_sources = sources_t.unsqueeze(-2).expand_as(targets)
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
        dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)

        detector_to_source_unit = -dir_unit
        blocked_fe = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, fe_index, tol)
        blocked_pb = self._rotated_octant_blocked_mask_torch(detector_to_source_unit, pb_index, tol)
        normal_fe = torch.as_tensor(self.orientations[fe_index], device=device, dtype=dtype)
        normal_pb = torch.as_tensor(self.orientations[pb_index], device=device, dtype=dtype)
        cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=-1), 0.0, 1.0)
        cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=-1), 0.0, 1.0)
        if (
            self.shield_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
            and not self.shield_params.use_angle_attenuation
        ):
            center = detector_t.expand_as(sources_t).unsqueeze(-2)
            L_fe = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=-np.asarray(self.orientations[fe_index], dtype=float),
                inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                outer_radius_cm=self.shield_params.inner_radius_fe_cm + self.shield_params.thickness_fe_cm,
                tol=tol,
            )
            L_pb = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=-np.asarray(self.orientations[pb_index], dtype=float),
                inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                outer_radius_cm=self.shield_params.inner_radius_pb_cm + self.shield_params.thickness_pb_cm,
                tol=tol,
            )
        else:
            L_fe, L_pb = self._shield_path_lengths_torch(
                direction=sampled_direction,
                blocked_fe=blocked_fe,
                blocked_pb=blocked_pb,
                cos_fe=cos_fe,
                cos_pb=cos_pb,
                tol_t=tol_t,
                device=device,
                dtype=dtype,
            )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = float(mu_fe) * L_fe
        tau_pb = float(mu_pb) * L_pb
        tau_obstacle = torch.zeros_like(tau_fe)
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            if sample_count > 1:
                obstacle_path_cm = obstacle_path_lengths_between_points_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
            else:
                obstacle_path_cm = obstacle_path_lengths_cm_torch(
                    positions=sources_t,
                    detector_pos=detector_t.reshape(3),
                    obstacle_boxes_m=boxes_t,
                ).unsqueeze(-1)
            tau_obstacle = float(self.obstacle_mu_cm_inv(isotope)) * obstacle_path_cm
        total_tau = tau_fe + tau_pb + tau_obstacle
        buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
        att = torch.clamp(torch.exp(-total_tau) * buildup, max=1.0)
        att = torch.mean(att, dim=-1)
        values = geom * att
        return values.detach().cpu().numpy().astype(float, copy=False)

    def kernel_values_pair(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        chunk_size: int = 8192,
    ) -> NDArray[np.float64]:
        """Evaluate K values for many sources at one detector pose."""
        sources_arr = np.asarray(sources, dtype=float)
        if sources_arr.size == 0:
            return np.zeros(0, dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if not self.use_gpu:
            return np.asarray(
                [
                    self.kernel_value_pair(
                        isotope=isotope,
                        detector_pos=detector_pos,
                        source_pos=source,
                        fe_index=fe_index,
                        pb_index=pb_index,
                    )
                    for source in sources_arr
                ],
                dtype=float,
            )
        self._gpu_enabled()
        chunk = max(1, int(chunk_size))
        parts: list[NDArray[np.float64]] = []
        for start in range(0, sources_arr.shape[0], chunk):
            stop = min(start + chunk, sources_arr.shape[0])
            parts.append(
                self._kernel_values_pair_torch_chunk(
                    isotope=isotope,
                    detector_pos=detector_pos,
                    sources=sources_arr[start:stop],
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
            )
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

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
        return self.attenuation_factor_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
            fe_index=orient_idx,
            pb_index=orient_idx,
        )

    def attenuation_factor_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return combined Fe/Pb attenuation factor A^{sh} (Sec. 3.2)."""
        if self.detector_radius_m > 0.0 and self.detector_aperture_samples > 1:
            targets = self._detector_aperture_targets(source_pos, detector_pos)
            values = [
                self._attenuation_factor_for_target(
                    isotope=isotope,
                    source_pos=source_pos,
                    target_pos=target,
                    detector_pos=detector_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
                for target in targets
            ]
            return float(np.mean(values)) if values else 1.0
        return self._attenuation_factor_for_target(
            isotope=isotope,
            source_pos=source_pos,
            target_pos=detector_pos,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
        )

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
        geom = finite_sphere_geometric_term(
            detector_pos,
            source_pos,
            self.detector_radius_m,
        )
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
        geom = finite_sphere_geometric_term(
            detector_pos,
            source_pos,
            self.detector_radius_m,
        )
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
            use_gpu=bool(use_gpu) if use_gpu is not None else False,
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
