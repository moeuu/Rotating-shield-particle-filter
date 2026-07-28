"""Continuous 3D kernel evaluations for the Chapter 3.3 measurement model.

Implements geometric and shielded kernels for arbitrary source coordinates,
consistent with Sec. 3.2–3.3 of the thesis (inverse-square law plus attenuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
import re
from typing import Callable, Dict, TypeVar

import numpy as np
from numpy.typing import NDArray

from measurement.detector_geometry import normalize_detector_aperture_sampling
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    CONCRETE_MU_CM_INV,
    LOCAL_POSITIVE_OCTANT_CENTER,
    OctantShield,
    generate_octant_orientations,
    octant_index_from_normal,
    resolve_mu_values,
    rotation_matrix_between_vectors,
    spherical_shell_path_length_cm_torch,
)
from spectrum.additive_scatter import (
    AdditiveNoncollidedTransportResponse,
    physical_scatter_basis_numpy,
    physical_scatter_basis_torch,
)

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


_ChunkResult = TypeVar("_ChunkResult")
_CUDA_CHUNK_MAX_WORKING_BYTES = 4 * 1024**3
_CUDA_CHUNK_FREE_MEMORY_FRACTION = 0.20
_CUDA_CHUNK_LOW_MEMORY_FRACTION = 0.05
_CUDA_CHUNK_MIN_RESERVE_BYTES = 512 * 1024**2
_CUDA_CHUNK_RESERVE_FRACTION = 0.10
_LINE_RESPONSE_CACHE_MAX_ENTRIES = 8
_LINE_RESPONSE_CACHE_MAX_BYTES = 256 * 1024**2


def validate_orientation_pair_indices(
    fe_indices: object,
    pb_indices: object,
    *,
    orientation_count: int,
    expected_count: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return strict zero-based Fe/Pb orientation-index arrays.

    Production shield-pair identifiers are serialized as zero-based indices.
    Negative Python-style indexing and implicit floating-point truncation would
    silently reinterpret a corrupt live/replay record as another physical
    shield posture, so both are rejected at the physics boundary.
    """
    if isinstance(orientation_count, bool) or not isinstance(
        orientation_count,
        (int, np.integer),
    ):
        raise TypeError("orientation_count must be an integer.")
    count = int(orientation_count)
    if count <= 0:
        raise ValueError("orientation_count must be positive.")
    fe_raw = np.asarray(fe_indices)
    pb_raw = np.asarray(pb_indices)
    if (
        fe_raw.ndim != 1
        or pb_raw.ndim != 1
        or fe_raw.dtype == np.bool_
        or pb_raw.dtype == np.bool_
        or not np.issubdtype(fe_raw.dtype, np.integer)
        or not np.issubdtype(pb_raw.dtype, np.integer)
    ):
        raise ValueError(
            "Fe/Pb orientation indices must be one-dimensional integer arrays."
        )
    if fe_raw.size != pb_raw.size:
        raise ValueError("Fe and Pb index arrays must have matching lengths.")
    if expected_count is not None:
        if isinstance(expected_count, bool) or not isinstance(
            expected_count,
            (int, np.integer),
        ):
            raise TypeError("expected_count must be an integer.")
        expected = int(expected_count)
        if expected < 0:
            raise ValueError("expected_count must be nonnegative.")
        if fe_raw.size != expected:
            raise ValueError(
                "Fe/Pb index arrays must match the expected row count."
            )
    fe_arr = fe_raw.astype(np.int64, copy=False)
    pb_arr = pb_raw.astype(np.int64, copy=False)
    if (
        np.any(fe_arr < 0)
        or np.any(fe_arr >= count)
        or np.any(pb_arr < 0)
        or np.any(pb_arr >= count)
    ):
        raise IndexError(
            f"Fe/Pb orientation index values must lie in [0, {count})."
        )
    return fe_arr, pb_arr


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
    ref_fraction = max(
        0.5 * (1.0 - float(np.sqrt(max(1.0 - ref_ratio * ref_ratio, 0.0)))), 1.0e-12
    )
    return fraction / ref_fraction


def _source_scale_rows_torch(
    source_scale: float | NDArray[np.float64] | "torch.Tensor",
    num_rows: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Return a validated row-wise source scale tensor for pair batches."""
    if torch is None:
        raise RuntimeError("torch is not available")
    scale = torch.as_tensor(source_scale, device=device, dtype=dtype)
    if scale.numel() == 1:
        scale = scale.reshape(1)
    else:
        scale = scale.reshape(-1)
        if int(scale.numel()) != int(num_rows):
            raise ValueError(
                "source_scale must be scalar or contain one value per pair."
            )
    if not bool(torch.all(torch.isfinite(scale))) or bool(torch.any(scale < 0.0)):
        raise ValueError("source_scale values must be finite and nonnegative.")
    if int(scale.numel()) == 1:
        return scale.reshape(1, 1)
    scale = scale.reshape(-1)
    return scale.view(int(num_rows), 1)


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
    d = float(
        np.linalg.norm(
            np.asarray(detector, dtype=float) - np.asarray(source, dtype=float)
        )
    )
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
    normalized = {
        _normalize_isotope_key(key): float(value) for key, value in table.items()
    }
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
        raise ValueError(
            "source_pos, detector_pos, and box_m must have shapes (3,), (3,), and (6,)."
        )
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
    return float(
        np.sum(
            obstacle_path_lengths_by_box_cm(
                source_pos=source_pos,
                detector_pos=detector_pos,
                obstacle_boxes_m=obstacle_boxes_m,
            )
        )
    )


def obstacle_path_lengths_by_box_cm(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-box source-detector path lengths in centimeters."""
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    if boxes.size == 0:
        return np.zeros(0, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    return np.asarray(
        [
            100.0 * segment_box_intersection_length_m(source_pos, detector_pos, box)
            for box in boxes
        ],
        dtype=float,
    )


def obstacle_optical_depth(
    source_pos: NDArray[np.float64],
    detector_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
    obstacle_mu_cm_inv_by_box: NDArray[np.float64],
) -> float:
    """Return summed material optical depth through known obstacle components."""
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    mu_values = np.asarray(obstacle_mu_cm_inv_by_box, dtype=float)
    if boxes.size == 0:
        return 0.0
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")
    if mu_values.shape != (boxes.shape[0],):
        raise ValueError("obstacle_mu_cm_inv_by_box must match obstacle box count.")
    path_cm_by_box = obstacle_path_lengths_by_box_cm(
        source_pos=source_pos,
        detector_pos=detector_pos,
        obstacle_boxes_m=boxes,
    )
    return float(np.sum(mu_values * path_cm_by_box))


def obstacle_log_attenuation_matrix(
    sources_xyz: NDArray[np.float64],
    detector_poses_xyz: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
    obstacle_mu_cm_inv_by_box: NDArray[np.float64],
    *,
    element_budget: int = 4_000_000,
    tol: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Return log obstacle transmission for all detector-source pairs."""
    sources = np.asarray(sources_xyz, dtype=float)
    detectors = np.asarray(detector_poses_xyz, dtype=float)
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    mu_values = np.asarray(obstacle_mu_cm_inv_by_box, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources_xyz must be shaped (N, 3).")
    if detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detector_poses_xyz must be shaped (M, 3).")
    if boxes.size == 0:
        return np.zeros((detectors.shape[0], sources.shape[0]), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (B, 6).")
    if mu_values.shape != (boxes.shape[0],):
        raise ValueError("obstacle_mu_cm_inv_by_box must match obstacle box count.")

    pose_count = int(detectors.shape[0])
    source_count = int(sources.shape[0])
    box_count = int(boxes.shape[0])
    if pose_count == 0 or source_count == 0:
        return np.zeros((pose_count, source_count), dtype=float)

    budget = max(int(element_budget), 1)
    chunk = max(1, min(pose_count, budget // max(1, source_count * box_count)))
    out = np.zeros((pose_count, source_count), dtype=float)
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    mu = mu_values.reshape(1, 1, box_count)
    tol_value = float(tol)
    src = sources.reshape(1, source_count, 1, 3)
    for start in range(0, pose_count, chunk):
        stop = min(start + chunk, pose_count)
        det = detectors[start:stop].reshape(stop - start, 1, 1, 3)
        direction = det - src
        distance = np.linalg.norm(direction[:, :, 0, :], axis=2)
        t_min_axes: list[NDArray[np.float64]] = []
        t_max_axes: list[NDArray[np.float64]] = []
        for axis in range(3):
            value = src[..., axis]
            step = direction[..., axis]
            lo = lower[:, axis].reshape(1, 1, box_count)
            hi = upper[:, axis].reshape(1, 1, box_count)
            parallel = np.abs(step) <= tol_value
            inside = (value >= lo) & (value <= hi)
            safe_step = np.where(parallel, 1.0, step)
            t0 = (lo - value) / safe_step
            t1 = (hi - value) / safe_step
            axis_min = np.minimum(t0, t1)
            axis_max = np.maximum(t0, t1)
            axis_min = np.where(parallel & inside, -np.inf, axis_min)
            axis_max = np.where(parallel & inside, np.inf, axis_max)
            axis_min = np.where(parallel & ~inside, np.inf, axis_min)
            axis_max = np.where(parallel & ~inside, -np.inf, axis_max)
            t_min_axes.append(axis_min)
            t_max_axes.append(axis_max)
        t_enter = np.maximum(np.stack(t_min_axes, axis=-1).max(axis=-1), 0.0)
        t_exit = np.minimum(np.stack(t_max_axes, axis=-1).min(axis=-1), 1.0)
        valid = t_exit > t_enter
        span = np.zeros_like(t_exit)
        np.subtract(t_exit, t_enter, out=span, where=valid)
        span = np.where(np.isfinite(span), span, 0.0)
        length_cm = span * distance[:, :, None] * 100.0
        tau = np.sum(length_cm * mu, axis=2)
        out[start:stop] = -tau
    return out


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
        inside_shell = (
            (inner_m * inner_m - tol) <= radius_sq <= (outer_m * outer_m + tol)
        )
        inside_octant = bool(np.all(point >= -tol))
        if inside_shell and inside_octant:
            length += (right - left) * segment_length
    return float(100.0 * length)


def segment_rotated_octant_shell_path_length_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    center_pos: "torch.Tensor",
    shield_normal: NDArray[np.float64] | None,
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-9,
    rotation: "torch.Tensor" | None = None,
) -> "torch.Tensor":
    """Return exact segment length inside a rotated octant shell for torch tensors."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if rotation is None:
        if shield_normal is None:
            raise ValueError("shield_normal is required when rotation is not provided.")
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            np.asarray(shield_normal, dtype=float),
        )
        rotation = torch.as_tensor(
            rotation_np,
            device=source_pos.device,
            dtype=source_pos.dtype,
        )
    else:
        rotation = rotation.to(device=source_pos.device, dtype=source_pos.dtype)
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
    length_m = (
        torch.sum(
            torch.where(inside_shell & inside_octant, width, torch.zeros_like(width)),
            dim=-1,
        )
        * segment_length
    )
    return torch.where(
        segment_length > float(tol), 100.0 * length_m, torch.zeros_like(length_m)
    )


def obstacle_path_lengths_cm_torch(
    positions: "torch.Tensor",
    detector_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return batched obstacle path lengths through axis-aligned boxes in centimeters."""
    lengths_by_box = obstacle_path_lengths_by_box_cm_torch(
        positions=positions,
        detector_pos=detector_pos,
        obstacle_boxes_m=obstacle_boxes_m,
        tol=tol,
    )
    return torch.sum(lengths_by_box, dim=-1)


def obstacle_path_lengths_by_box_cm_torch(
    positions: "torch.Tensor",
    detector_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return batched per-box obstacle path lengths in centimeters."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(
            (*positions.shape[:-1], 0),
            device=positions.device,
            dtype=positions.dtype,
        )
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
    return 100.0 * length_m


def obstacle_path_lengths_between_points_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return path lengths through axis-aligned boxes for source-target segments."""
    lengths_by_box = obstacle_path_lengths_between_points_by_box_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        obstacle_boxes_m=obstacle_boxes_m,
        tol=tol,
    )
    return torch.sum(lengths_by_box, dim=-1)


def obstacle_path_lengths_between_points_by_box_cm_torch(
    source_pos: "torch.Tensor",
    target_pos: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float = 1e-9,
) -> "torch.Tensor":
    """Return per-box path lengths through axis-aligned boxes for segments."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(
            (*source_pos.shape[:-1], 0),
            device=source_pos.device,
            dtype=source_pos.dtype,
        )
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
    return 100.0 * length_m


def _finite_sphere_geometric_term_numpy(
    distance: NDArray[np.float64],
    *,
    detector_radius_m: float,
    point_distance_floor_m: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Return finite-sphere detector-cps@1m scaling for NumPy distances."""
    distances = np.asarray(distance, dtype=float)
    radius = max(float(detector_radius_m), 0.0)
    if radius <= 0.0:
        safe_distance = np.maximum(distances, float(point_distance_floor_m))
        return 1.0 / (safe_distance**2)
    effective_distance = np.maximum(distances, radius)
    ratio = np.clip(radius / np.maximum(effective_distance, 1.0e-12), 0.0, 1.0)
    fraction = 0.5 * (1.0 - np.sqrt(np.clip(1.0 - ratio * ratio, 0.0, None)))
    reference = max(1.0, radius)
    reference_ratio = min(radius / reference, 1.0)
    reference_fraction = max(
        0.5
        * (
            1.0
            - float(
                np.sqrt(max(1.0 - reference_ratio * reference_ratio, 0.0))
            )
        ),
        1.0e-12,
    )
    return fraction / reference_fraction


def _segment_rotated_octant_shell_path_length_cm_numpy(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    center_pos: NDArray[np.float64],
    rotations: NDArray[np.float64],
    inner_radius_cm: float,
    outer_radius_cm: float,
    tol: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Return exact rotated-octant shell lengths for matched NumPy ray rows."""
    sources = np.asarray(source_pos, dtype=float)
    targets = np.asarray(target_pos, dtype=float)
    centers = np.asarray(center_pos, dtype=float)
    rotation_arr = np.asarray(rotations, dtype=float)
    if sources.ndim != 3 or sources.shape[-1] != 3:
        raise ValueError("source_pos must be shaped (N, R, 3).")
    if targets.shape != sources.shape:
        raise ValueError("target_pos must match source_pos.")
    if centers.shape != (sources.shape[0], 1, 3):
        raise ValueError("center_pos must be shaped (N, 1, 3).")
    if rotation_arr.shape != (sources.shape[0], 3, 3):
        raise ValueError("rotations must be shaped (N, 3, 3).")

    source_local = np.einsum(
        "nri,nij->nrj",
        sources - centers,
        rotation_arr,
        optimize=True,
    )
    target_local = np.einsum(
        "nri,nij->nrj",
        targets - centers,
        rotation_arr,
        optimize=True,
    )
    delta = target_local - source_local
    segment_length = np.linalg.norm(delta, axis=-1)
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return np.zeros_like(segment_length)

    a = np.sum(delta * delta, axis=-1)
    b = 2.0 * np.sum(source_local * delta, axis=-1)
    breakpoints = [
        np.zeros_like(segment_length),
        np.ones_like(segment_length),
    ]
    denominator = np.maximum(2.0 * a, float(tol))
    for radius in (inner_m, outer_m):
        c = np.sum(source_local * source_local, axis=-1) - radius * radius
        discriminant = b * b - 4.0 * a * c
        valid = discriminant >= 0.0
        root = np.sqrt(np.clip(discriminant, 0.0, None))
        for sign in (-1.0, 1.0):
            value = (-b + sign * root) / denominator
            value = np.where(valid, value, 0.0)
            breakpoints.append(np.clip(value, 0.0, 1.0))
    for axis in range(3):
        axis_delta = delta[..., axis]
        valid = np.abs(axis_delta) > float(tol)
        safe_delta = np.where(valid, axis_delta, 1.0)
        value = -source_local[..., axis] / safe_delta
        breakpoints.append(np.clip(np.where(valid, value, 0.0), 0.0, 1.0))

    ordered = np.sort(np.stack(breakpoints, axis=-1), axis=-1)
    left = ordered[..., :-1]
    right = ordered[..., 1:]
    width = np.maximum(right - left, 0.0)
    midpoint = 0.5 * (left + right)
    points = (
        source_local[..., None, :]
        + midpoint[..., None] * delta[..., None, :]
    )
    radius_sq = np.sum(points * points, axis=-1)
    inside_shell = (
        (radius_sq >= inner_m * inner_m - float(tol))
        & (radius_sq <= outer_m * outer_m + float(tol))
    )
    inside_octant = np.all(points >= -float(tol), axis=-1)
    length_m = (
        np.sum(np.where(inside_shell & inside_octant, width, 0.0), axis=-1)
        * segment_length
    )
    return np.where(
        segment_length > float(tol),
        100.0 * length_m,
        np.zeros_like(length_m),
    )


def _obstacle_path_lengths_between_points_by_box_cm_numpy(
    source_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    obstacle_boxes_m: NDArray[np.float64],
    tol: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Return per-box segment lengths for batched matched NumPy ray rows."""
    sources = np.asarray(source_pos, dtype=float)
    targets = np.asarray(target_pos, dtype=float)
    boxes = np.asarray(obstacle_boxes_m, dtype=float)
    if sources.shape != targets.shape or sources.shape[-1] != 3:
        raise ValueError("source_pos and target_pos must match with last dimension 3.")
    if boxes.size == 0:
        return np.zeros((*sources.shape[:-1], 0), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("obstacle_boxes_m must be shaped (N, 6).")

    point = sources[..., None, :]
    direction = (targets - sources)[..., None, :]
    distance = np.linalg.norm(targets - sources, axis=-1)
    lower = boxes[:, :3]
    upper = boxes[:, 3:]
    parallel = np.abs(direction) <= float(tol)
    inside = (point >= lower) & (point <= upper)
    safe_direction = np.where(parallel, 1.0, direction)
    t0 = (lower - point) / safe_direction
    t1 = (upper - point) / safe_direction
    axis_min = np.minimum(t0, t1)
    axis_max = np.maximum(t0, t1)
    axis_min = np.where(parallel & inside, -np.inf, axis_min)
    axis_max = np.where(parallel & inside, np.inf, axis_max)
    axis_min = np.where(parallel & ~inside, np.inf, axis_min)
    axis_max = np.where(parallel & ~inside, -np.inf, axis_max)
    t_enter = np.maximum(np.max(axis_min, axis=-1), 0.0)
    t_exit = np.minimum(np.min(axis_max, axis=-1), 1.0)
    span = np.where(t_exit > t_enter, t_exit - t_enter, 0.0)
    span = np.where(np.isfinite(span), span, 0.0)
    return 100.0 * span * distance[..., None]


def _torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(_TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def _torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(_TORCH_AVAILABLE and torch is not None)


def _torch_device_available(device: str | None = None) -> bool:
    """Return True when torch can run on the requested device."""
    if not _torch_installed():
        return False
    device_name = "cuda" if device is None else str(device)
    if device_name.startswith("cuda"):
        return bool(torch is not None and torch.cuda.is_available())
    return True


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


@dataclass(frozen=True)
class LineTransportComponents:
    """Store physical transport components with one trailing line axis."""

    total_kernel: NDArray[np.float64]
    unattenuated_kernel: NDArray[np.float64]
    uncollided_kernel: NDArray[np.float64]
    tau_fe: NDArray[np.float64]
    tau_pb: NDArray[np.float64]
    tau_obstacle: NDArray[np.float64]
    tau_obstacle_compton: NDArray[np.float64]
    distance_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Require one common finite nonnegative shape ending in lines."""
        arrays = tuple(
            np.asarray(getattr(self, name), dtype=np.float64)
            for name in (
                "total_kernel",
                "unattenuated_kernel",
                "uncollided_kernel",
                "tau_fe",
                "tau_pb",
                "tau_obstacle",
                "tau_obstacle_compton",
                "distance_m",
            )
        )
        shape = arrays[0].shape
        if (
            len(shape) < 2
            or any(array.shape != shape for array in arrays)
            or any(np.any(~np.isfinite(array)) for array in arrays)
            or any(np.any(array < 0.0) for array in arrays)
        ):
            raise ValueError(
                "Line transport components must be finite nonnegative "
                "arrays with one common shape and a trailing line axis."
            )
        for name, array in zip(
            (
                "total_kernel",
                "unattenuated_kernel",
                "uncollided_kernel",
                "tau_fe",
                "tau_pb",
                "tau_obstacle",
                "tau_obstacle_compton",
                "distance_m",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, array)


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
    orientations: NDArray[np.float64] = field(
        default_factory=generate_octant_orientations
    )
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    obstacle_grid: ObstacleGrid | None = None
    obstacle_height_m: float = 2.0
    obstacle_mu_by_isotope: Dict[str, float] | None = None
    obstacle_buildup_coeff: float = 0.0
    detector_radius_m: float = 0.0
    detector_aperture_radius_m: float | None = None
    detector_aperture_samples: int = 1
    detector_aperture_sampling: str = "solid_angle_cone"
    source_extent_radius_m: float = 0.0
    source_extent_samples: int = 1
    line_mu_by_isotope: Dict[str, object] | None = None
    additive_scatter_response: AdditiveNoncollidedTransportResponse | None = None
    _obstacle_boxes_cache: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _torch_octant_rotation_cache: dict[
        tuple[str, str, tuple[float, float, float]], object
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _numpy_octant_rotations_cache: NDArray[np.float64] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _line_response_cache: dict[tuple[object, ...], NDArray[np.float64]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    line_response_cache_hits: int = field(default=0, init=False)
    line_response_cache_misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate every physical and execution field without silent repair."""
        for name in (
            "obstacle_height_m",
            "obstacle_buildup_coeff",
            "detector_radius_m",
            "source_extent_radius_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            resolved = float(value)
            if not np.isfinite(resolved) or resolved < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            setattr(self, name, resolved)
        if not isinstance(self.shield_params, ShieldParams):
            raise TypeError("shield_params must be a ShieldParams instance.")
        if self.detector_aperture_radius_m is None:
            self.detector_aperture_radius_m = self.detector_radius_m
        aperture_radius = self.detector_aperture_radius_m
        if isinstance(aperture_radius, bool) or not isinstance(
            aperture_radius,
            Real,
        ):
            raise TypeError(
                "detector_aperture_radius_m must be a real number or None."
            )
        self.detector_aperture_radius_m = float(aperture_radius)
        if (
            not np.isfinite(self.detector_aperture_radius_m)
            or self.detector_aperture_radius_m < 0.0
        ):
            raise ValueError(
                "detector_aperture_radius_m must be finite and nonnegative."
            )
        for name in ("detector_aperture_samples", "source_extent_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            resolved = int(value)
            if resolved <= 0:
                raise ValueError(f"{name} must be positive.")
            setattr(self, name, resolved)
        self.detector_aperture_sampling = normalize_detector_aperture_sampling(
            self.detector_aperture_sampling
        )
        if (self.source_extent_radius_m == 0.0) != (
            self.source_extent_samples == 1
        ):
            raise ValueError(
                "Source extent requires radius=0 with one sample, or a "
                "positive radius with at least two samples."
            )
        if not isinstance(self.use_gpu, bool):
            raise TypeError("use_gpu must be a boolean.")
        if not isinstance(self.gpu_device, str) or not self.gpu_device:
            raise TypeError("gpu_device must be a nonempty string.")
        if not isinstance(self.gpu_dtype, str) or not self.gpu_dtype:
            raise TypeError("gpu_dtype must be a nonempty string.")
        orientations = np.asarray(self.orientations, dtype=np.float64)
        if (
            orientations.ndim != 2
            or orientations.shape[1] != 3
            or orientations.shape[0] == 0
            or np.any(~np.isfinite(orientations))
            or np.any(np.linalg.norm(orientations, axis=1) <= 0.0)
        ):
            raise ValueError(
                "orientations must be a nonempty finite array shaped N x 3."
            )
        self.orientations = np.ascontiguousarray(orientations)

    def _rotated_octant_rotation_torch(
        self,
        shield_normal: NDArray[np.float64],
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return a cached torch rotation for a shield octant normal."""
        if torch is None:
            raise RuntimeError("torch is not available")
        normal = np.asarray(shield_normal, dtype=float).reshape(3)
        key = (
            str(device),
            str(dtype),
            tuple(float(np.round(value, 12)) for value in normal),
        )
        cached = self._torch_octant_rotation_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        rotation_np = rotation_matrix_between_vectors(
            LOCAL_POSITIVE_OCTANT_CENTER,
            normal,
        )
        rotation = torch.as_tensor(rotation_np, device=device, dtype=dtype)
        self._torch_octant_rotation_cache[key] = rotation
        return rotation

    def _rotated_octant_rotations_numpy(self) -> NDArray[np.float64]:
        """Return cached rotations for every fixed physical shield octant."""
        if self._numpy_octant_rotations_cache is None:
            # This one-time fixed-size geometry setup is not a runtime
            # detector/source/orientation evaluation loop.
            self._numpy_octant_rotations_cache = np.stack(
                [
                    rotation_matrix_between_vectors(
                        LOCAL_POSITIVE_OCTANT_CENTER,
                        -np.asarray(normal, dtype=float),
                    )
                    for normal in np.asarray(self.orientations, dtype=float)
                ],
                axis=0,
            )
        return self._numpy_octant_rotations_cache

    def _adaptive_numpy_chunk_size(
        self,
        requested: int,
        *,
        isotope: str,
    ) -> int:
        """Return a bounded NumPy row chunk for batched kernel evaluation."""
        source_samples = (
            max(int(self.source_extent_samples), 1)
            if float(self.source_extent_radius_m or 0.0) > 0.0
            else 1
        )
        aperture_samples = (
            max(int(self.detector_aperture_samples), 1)
            if float(self.detector_aperture_radius_m or 0.0) > 0.0
            else 1
        )
        ray_count = source_samples * aperture_samples
        obstacle_count = int(self.obstacle_boxes_m().shape[0])
        line_count = max(1, len(self._line_mu_values(isotope)))
        # The exact octant intersection holds coordinates, breakpoints and
        # masks concurrently; obstacle and line axes add their own work arrays.
        working_elements_per_row = ray_count * (
            48 + 12 * obstacle_count + 8 * line_count
        )
        element_budget = 8_000_000
        safe_chunk = max(1, element_budget // max(working_elements_per_row, 1))
        return max(1, min(int(requested), int(safe_chunk)))

    def _adaptive_torch_chunk_size(
        self,
        requested: int,
        *,
        isotope: str | None = None,
        orientation_pair_count: int | None = None,
        device: "torch.device | str | None" = None,
        dtype: "torch.dtype | None" = None,
        all_orientation_pairs: bool = False,
    ) -> int:
        """
        Return a memory-aware torch chunk size without changing kernel math.

        Obstacle attenuation with finite detector aperture expands each source
        into ``source_count * aperture_samples * obstacle_box_count`` segment-box
        intersections. The CUDA path budgets a conservative working set from
        currently free VRAM, while leaving headroom for the PF and driver. The
        estimate uses the requested isotope's actual line count and requested
        shield-pair count. CPU and unavailable-memory-query paths retain the
        fixed conservative budget used before adaptive CUDA sizing.
        """
        chunk = max(1, int(requested))
        aperture_sample_count = (
            max(int(self.detector_aperture_samples), 1)
            if float(self.detector_aperture_radius_m or 0.0) > 0.0
            else 1
        )
        source_sample_count = (
            max(int(self.source_extent_samples), 1)
            if float(self.source_extent_radius_m or 0.0) > 0.0
            else 1
        )
        sample_count = aperture_sample_count * source_sample_count
        obstacle_count = int(self.obstacle_boxes_m().shape[0])
        all_pair_count = int(len(self.orientations)) ** 2
        if orientation_pair_count is None:
            pair_count = all_pair_count if all_orientation_pairs else 1
        else:
            pair_count = max(1, int(orientation_pair_count))
        line_count = (
            max(1, len(self._line_mu_values(isotope)))
            if isotope is not None
            else self._max_line_count()
        )

        dtype_name = str(dtype if dtype is not None else self.gpu_dtype).lower()
        bytes_per_element = 8 if "64" in dtype_name else 4
        legacy_element_budget = (
            4_000_000 if bytes_per_element == 8 else 8_000_000
        )
        legacy_denom = max(1, sample_count * line_count)
        if obstacle_count > 0:
            legacy_denom = max(
                legacy_denom,
                sample_count * obstacle_count * line_count,
            )
        legacy_denom = max(
            legacy_denom,
            sample_count * pair_count * line_count,
        )
        legacy_safe_chunk = max(1, int(legacy_element_budget // legacy_denom))

        device_name = str(device if device is not None else self.gpu_device)
        if not device_name.startswith("cuda"):
            return max(1, min(chunk, legacy_safe_chunk))
        memory_info = self._torch_cuda_memory_info(device_name)
        if memory_info is None:
            return max(1, min(chunk, legacy_safe_chunk))
        free_bytes, total_bytes = memory_info
        reserve_bytes = max(
            _CUDA_CHUNK_MIN_RESERVE_BYTES,
            int(total_bytes * _CUDA_CHUNK_RESERVE_FRACTION),
        )
        unreserved_bytes = max(0, int(free_bytes) - reserve_bytes)
        if unreserved_bytes > 0:
            working_bytes = min(
                _CUDA_CHUNK_MAX_WORKING_BYTES,
                int(unreserved_bytes * _CUDA_CHUNK_FREE_MEMORY_FRACTION),
            )
        else:
            working_bytes = int(
                max(0, int(free_bytes)) * _CUDA_CHUNK_LOW_MEMORY_FRACTION
            )

        # Segment-box intersection retains several entry/exit tensors per axis.
        obstacle_working = 18 * sample_count * obstacle_count
        # Pair-resolved line attenuation retains optical-depth, buildup, and
        # response tensors concurrently. These factors intentionally overbound
        # the live fp64 tensors observed in the production RAL geometry.
        pair_working = 8 * sample_count * pair_count
        line_working = 8 * sample_count * pair_count * line_count
        ray_working = 32 * sample_count
        working_elements_per_source = max(
            1,
            obstacle_working + pair_working + line_working + ray_working,
        )
        cuda_safe_chunk = max(
            1,
            int(
                working_bytes
                // max(1, bytes_per_element * working_elements_per_source)
            ),
        )
        return max(1, min(chunk, cuda_safe_chunk))

    def _torch_cuda_memory_info(
        self,
        device: "torch.device | str",
    ) -> tuple[int, int] | None:
        """Return free and total CUDA bytes, or ``None`` when unavailable."""
        if torch is None or not str(device).startswith("cuda"):
            return None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except (RuntimeError, TypeError):
            return None
        return int(free_bytes), int(total_bytes)

    @staticmethod
    def _is_cuda_out_of_memory(error: RuntimeError) -> bool:
        """Return whether a runtime error represents CUDA allocation failure."""
        if torch is not None:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            if oom_type is not None and isinstance(error, oom_type):
                return True
        message = str(error).lower()
        return "cuda" in message and "out of memory" in message

    @staticmethod
    def _clear_cuda_cache_after_oom(device: "torch.device | str") -> None:
        """Release cached CUDA blocks after a recoverable chunk allocation OOM."""
        if torch is None or not str(device).startswith("cuda"):
            return
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            return

    def _evaluate_torch_chunks_with_oom_retry(
        self,
        *,
        total_size: int,
        initial_chunk: int,
        device: "torch.device | str",
        evaluator: Callable[[int, int], _ChunkResult],
    ) -> tuple[list[_ChunkResult], int]:
        """
        Evaluate ordered chunks, halving deterministically after a CUDA OOM.

        A failed attempt is discarded and restarted from index zero. Therefore
        output order and every per-row reduction remain identical to a run that
        selected the successful chunk size initially.
        """
        total = max(0, int(total_size))
        chunk = max(1, min(int(initial_chunk), max(total, 1)))
        while True:
            parts: list[_ChunkResult] = []
            try:
                for start in range(0, total, chunk):
                    stop = min(start + chunk, total)
                    parts.append(evaluator(start, stop))
                return parts, chunk
            except RuntimeError as error:
                if not self._is_cuda_out_of_memory(error) or chunk <= 1:
                    raise
                parts.clear()
                chunk = max(1, chunk // 2)
            # Run this after leaving the except block so the traceback and
            # failed evaluator frame no longer retain temporary tensors.
            self._clear_cuda_cache_after_oom(device)

    def _mu_values(self, isotope: str) -> tuple[float, float]:
        """Return (mu_fe, mu_pb) for the given isotope with fallbacks."""
        return resolve_mu_values(
            self.mu_by_isotope,
            isotope,
            default_fe=self.shield_params.mu_fe,
            default_pb=self.shield_params.mu_pb,
        )

    def _line_mu_values(self, isotope: str) -> tuple[tuple[float, float, float], ...]:
        """
        Return line-resolved ``(weight, mu_fe, mu_pb)`` entries for an isotope.

        If no line-resolved table is configured, callers fall back to the
        existing isotope-effective coefficients.
        """
        table = self.line_mu_by_isotope
        if not isinstance(table, dict):
            return ()
        entry = table.get(isotope)
        if entry is None:
            normalized = {
                _normalize_isotope_key(key): value for key, value in table.items()
            }
            entry = normalized.get(_normalize_isotope_key(isotope))
        if entry is None:
            return ()
        rows: list[tuple[float, float, float]] = []
        for item in entry if isinstance(entry, (list, tuple)) else ():
            if isinstance(item, dict):
                weight = float(item.get("weight", 0.0))
                mu_fe = float(
                    item.get("fe", item.get("mu_fe", self.shield_params.mu_fe))
                )
                mu_pb = float(
                    item.get("pb", item.get("mu_pb", self.shield_params.mu_pb))
                )
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 3:
                weight = float(item[0])
                mu_fe = float(item[1])
                mu_pb = float(item[2])
            else:
                continue
            if (
                weight > 0.0
                and np.isfinite(weight)
                and np.isfinite(mu_fe)
                and np.isfinite(mu_pb)
            ):
                rows.append((weight, mu_fe, mu_pb))
        total_weight = sum(weight for weight, _, _ in rows)
        if total_weight <= 0.0:
            return ()
        return tuple(
            (weight / total_weight, mu_fe, mu_pb) for weight, mu_fe, mu_pb in rows
        )

    def _line_energy_values_keV(self, isotope: str) -> tuple[float, ...]:
        """Return positive-line energies in the shared attenuation-table order."""
        table = self.line_mu_by_isotope
        if not isinstance(table, dict):
            return ()
        entry = table.get(isotope)
        if entry is None:
            normalized = {
                _normalize_isotope_key(key): value
                for key, value in table.items()
            }
            entry = normalized.get(_normalize_isotope_key(isotope))
        energies: list[float] = []
        for item in entry if isinstance(entry, (list, tuple)) else ():
            if not isinstance(item, dict) or "energy_keV" not in item:
                return ()
            energy = float(item["energy_keV"])
            if not np.isfinite(energy) or energy <= 0.0:
                return ()
            energies.append(energy)
        if len(energies) != len(self._line_mu_values(isotope)):
            return ()
        return tuple(energies)

    def _validated_positive_line_indices(
        self,
        isotope: str,
        line_indices: object,
    ) -> NDArray[np.int64]:
        """Return unique indices into the configured positive-line basis.

        The spectrum pipeline records line indices after filtering the isotope
        library to positive-intensity lines.  Production line attenuation
        tables use that same order.  This validator intentionally rejects
        implicit float truncation, duplicates, missing line tables, and
        out-of-range indices instead of silently changing the event basis.
        """
        entries = self._line_mu_values(str(isotope))
        if not entries:
            raise ValueError(
                f"No positive line-resolved transport basis exists for {isotope!r}."
            )
        raw = np.asarray(line_indices)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype == np.bool_:
            raise ValueError(
                "positive_line_indices must be a non-empty one-dimensional "
                "integer array."
            )
        if np.issubdtype(raw.dtype, np.integer):
            indices = raw.astype(np.int64, copy=False)
        elif np.issubdtype(raw.dtype, np.floating):
            if np.any(~np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                raise ValueError(
                    "positive_line_indices must contain exact integer values."
                )
            indices = raw.astype(np.int64)
        else:
            raise ValueError(
                "positive_line_indices must contain exact integer values."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError("positive_line_indices must not contain duplicates.")
        if np.any(indices < 0) or np.any(indices >= len(entries)):
            raise IndexError(
                "positive_line_indices contains an index outside the configured "
                f"positive-line basis [0, {len(entries)})."
            )
        return np.asarray(indices, dtype=np.int64)

    def line_branching_weights(
        self,
        isotope: str,
        positive_line_indices: object,
    ) -> NDArray[np.float64]:
        """Return normalized branching weights for selected positive lines."""
        indices = self._validated_positive_line_indices(
            isotope,
            positive_line_indices,
        )
        entries = np.asarray(self._line_mu_values(str(isotope)), dtype=np.float64)
        return np.asarray(entries[indices, 0], dtype=np.float64)

    def clear_line_response_cache(self) -> None:
        """Clear exact-input cached line responses and reset cache counters."""
        self._line_response_cache.clear()
        self.line_response_cache_hits = 0
        self.line_response_cache_misses = 0

    @staticmethod
    def _line_response_array_key(values: NDArray[np.generic]) -> tuple[object, ...]:
        """Return an exact dtype/shape/byte key for one response input array."""
        array = np.ascontiguousarray(values)
        return (str(array.dtype), tuple(int(value) for value in array.shape), array.tobytes())

    def _max_line_count(self) -> int:
        """Return the maximum configured line count used by attenuation batching."""
        table = self.line_mu_by_isotope
        if not isinstance(table, dict):
            return 1
        count = 1
        for isotope in table:
            count = max(count, len(self._line_mu_values(str(isotope))))
        return max(1, int(count))

    def obstacle_boxes_m(self) -> NDArray[np.float64]:
        """Return cached obstacle boxes in meters as (x0, y0, z0, x1, y1, z1)."""
        if self.obstacle_grid is None:
            return np.zeros((0, 6), dtype=float)
        if self._obstacle_boxes_cache is None:
            boxes = self.obstacle_grid.attenuation_boxes(
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

    def obstacle_mu_values_cm_inv(self, isotope: str) -> NDArray[np.float64]:
        """Return per-obstacle-box attenuation coefficients in 1/cm."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0:
            return np.zeros(0, dtype=float)
        if self.obstacle_grid is not None:
            values = self.obstacle_grid.transport_mu_values(isotope)
            if values is not None:
                return np.asarray(values, dtype=float)
        return np.full(boxes.shape[0], self.obstacle_mu_cm_inv(isotope), dtype=float)

    def obstacle_line_mu_values_cm_inv(self, isotope: str) -> NDArray[np.float64]:
        """Return per-line, per-box obstacle attenuation coefficients."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0 or self.obstacle_grid is None:
            return np.zeros((0, 0), dtype=float)
        values = self.obstacle_grid.transport_line_mu_values(isotope)
        if values is None:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != boxes.shape[0]:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        return array

    def obstacle_line_compton_mu_values_cm_inv(
        self,
        isotope: str,
    ) -> NDArray[np.float64]:
        """Return per-line, per-box physical Compton attenuation values."""
        boxes = self.obstacle_boxes_m()
        if boxes.size == 0 or self.obstacle_grid is None:
            return np.zeros((0, 0), dtype=float)
        values = self.obstacle_grid.transport_line_compton_mu_values(isotope)
        if values is None:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != boxes.shape[0]:
            return np.zeros((0, boxes.shape[0]), dtype=float)
        total = self.obstacle_line_mu_values_cm_inv(isotope)
        if array.shape != total.shape or np.any(
            array > total * (1.0 + 1.0e-12)
        ):
            raise ValueError(
                "Obstacle Compton and total line attenuation contracts disagree."
            )
        return array

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

    def obstacle_path_lengths_by_box_cm(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-obstacle-box path lengths in centimeters."""
        return obstacle_path_lengths_by_box_cm(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
        )

    def obstacle_optical_depth_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle-only optical depth for one source-detector ray."""
        if self.obstacle_grid is None:
            return 0.0
        return obstacle_optical_depth(
            source_pos=source_pos,
            detector_pos=detector_pos,
            obstacle_boxes_m=self.obstacle_boxes_m(),
            obstacle_mu_cm_inv_by_box=self.obstacle_mu_values_cm_inv(isotope),
        )

    def obstacle_log_attenuation_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return log obstacle transmission, log(A_env), for one ray."""
        return -float(
            self.obstacle_optical_depth_pair(
                isotope=isotope,
                source_pos=source_pos,
                detector_pos=detector_pos,
            )
        )

    def obstacle_log_attenuation_matrix(
        self,
        isotope: str,
        sources_xyz: NDArray[np.float64],
        detector_poses_xyz: NDArray[np.float64],
        *,
        element_budget: int = 4_000_000,
    ) -> NDArray[np.float64]:
        """Return obstacle-only log transmission for detector/source batches."""
        return obstacle_log_attenuation_matrix(
            sources_xyz=sources_xyz,
            detector_poses_xyz=detector_poses_xyz,
            obstacle_boxes_m=self.obstacle_boxes_m(),
            obstacle_mu_cm_inv_by_box=self.obstacle_mu_values_cm_inv(isotope),
            element_budget=element_budget,
        )

    def obstacle_attenuation_factor_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle-only Beer-Lambert attenuation for one ray."""
        tau = self.obstacle_optical_depth_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )
        if tau <= 0.0:
            return 1.0
        return float(np.exp(-tau))

    def obstacle_area_averaged_attenuation_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return obstacle attenuation averaged over source and aperture rays."""
        if self.obstacle_grid is None:
            return 1.0
        sampled_sources, targets = self._ray_sample_points(source_pos, detector_pos)
        transmissions = [
            float(
                np.exp(
                    -self.obstacle_optical_depth_pair(
                        isotope=isotope,
                        source_pos=sampled_source,
                        detector_pos=target,
                    )
                )
            )
            for sampled_source, target in zip(sampled_sources, targets)
        ]
        if not transmissions:
            return 1.0
        return float(np.mean(np.asarray(transmissions, dtype=float)))

    def obstacle_area_averaged_optical_depth_pair(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return the equivalent tau from area-averaged obstacle transmission."""
        attenuation = self.obstacle_area_averaged_attenuation_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )
        return float(-np.log(max(float(attenuation), 1.0e-300)))

    def _obstacle_attenuation_factor(
        self,
        isotope: str,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> float:
        """Return Beer-Lambert attenuation through known obstacle components."""
        return self.obstacle_attenuation_factor_pair(
            isotope=isotope,
            source_pos=source_pos,
            detector_pos=detector_pos,
        )

    def obstacle_gpu_kwargs(self, isotope: str) -> dict[str, object]:
        """Return optional GPU kwargs for obstacle attenuation."""
        boxes = self.obstacle_boxes_m()
        line_entries = self._line_mu_values(isotope)
        kwargs: dict[str, object] = {}
        if line_entries:
            kwargs.update(
                {
                    "line_weights": np.asarray(
                        [entry[0] for entry in line_entries],
                        dtype=float,
                    ),
                    "line_mu_fe": np.asarray(
                        [entry[1] for entry in line_entries],
                        dtype=float,
                    ),
                    "line_mu_pb": np.asarray(
                        [entry[2] for entry in line_entries],
                        dtype=float,
                    ),
                }
            )
            obstacle_line_mu = self.obstacle_line_mu_values_cm_inv(isotope)
            if obstacle_line_mu.shape == (len(line_entries), boxes.shape[0]):
                kwargs["obstacle_line_mu_cm_inv_by_box"] = obstacle_line_mu
        if boxes.size == 0:
            return kwargs
        kwargs.update(
            {
                "obstacle_boxes_m": boxes,
                "obstacle_mu_cm_inv": 0.0,
                "obstacle_mu_cm_inv_by_box": self.obstacle_mu_values_cm_inv(isotope),
                "obstacle_buildup_coeff": self.obstacle_buildup_coeff,
            }
        )
        return kwargs

    def _line_obstacle_tau_values(
        self,
        isotope: str,
        path_by_box_cm: NDArray[np.float64],
        *,
        line_count: int,
    ) -> tuple[float, ...] | None:
        """Return line-resolved obstacle optical depths for one ray."""
        mu_values = self.obstacle_line_mu_values_cm_inv(isotope)
        if mu_values.shape != (int(line_count), path_by_box_cm.shape[0]):
            return None
        return tuple(float(np.sum(row * path_by_box_cm)) for row in mu_values)

    def _buildup_factor(
        self,
        tau_fe: float,
        tau_pb: float,
        tau_obstacle: float,
    ) -> float:
        """Return a bounded broad-beam build-up factor from optical depths."""
        factor = 1.0
        factor += self.shield_params.buildup_fe_coeff * (
            1.0 - float(np.exp(-max(tau_fe, 0.0)))
        )
        factor += self.shield_params.buildup_pb_coeff * (
            1.0 - float(np.exp(-max(tau_pb, 0.0)))
        )
        factor += self.obstacle_buildup_coeff * (
            1.0 - float(np.exp(-max(tau_obstacle, 0.0)))
        )
        return max(1.0, float(factor))

    def _buildup_factor_numpy(
        self,
        tau_fe: NDArray[np.float64],
        tau_pb: NDArray[np.float64],
        tau_obstacle: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return broad-beam build-up factors for batched NumPy optical depths."""
        fe_arr = np.asarray(tau_fe, dtype=float)
        pb_arr = np.asarray(tau_pb, dtype=float)
        obstacle_arr = np.asarray(tau_obstacle, dtype=float)
        factor = np.ones(np.broadcast_shapes(
            fe_arr.shape,
            pb_arr.shape,
            obstacle_arr.shape,
        ), dtype=float)
        factor = factor + self.shield_params.buildup_fe_coeff * (
            1.0 - np.exp(-np.maximum(fe_arr, 0.0))
        )
        factor = factor + self.shield_params.buildup_pb_coeff * (
            1.0 - np.exp(-np.maximum(pb_arr, 0.0))
        )
        factor = factor + self.obstacle_buildup_coeff * (
            1.0 - np.exp(-np.maximum(obstacle_arr, 0.0))
        )
        return np.maximum(factor, 1.0)

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
        factor = factor + self.shield_params.buildup_fe_coeff * (
            1.0 - torch.exp(-torch.clamp(tau_fe, min=0.0))
        )
        factor = factor + self.shield_params.buildup_pb_coeff * (
            1.0 - torch.exp(-torch.clamp(tau_pb, min=0.0))
        )
        factor = factor + self.obstacle_buildup_coeff * (
            1.0 - torch.exp(-torch.clamp(tau_obstacle, min=0.0))
        )
        return torch.clamp(factor, min=1.0)

    def _obstacle_tau_from_lengths_torch(
        self,
        isotope: str,
        path_by_box_cm: "torch.Tensor",
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return material optical depth from per-box torch path lengths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if path_by_box_cm.shape[-1] == 0:
            return torch.zeros(path_by_box_cm.shape[:-1], device=device, dtype=dtype)
        mu_values = torch.as_tensor(
            self.obstacle_mu_values_cm_inv(isotope),
            device=device,
            dtype=dtype,
        )
        return torch.sum(path_by_box_cm * mu_values, dim=-1)

    def _obstacle_line_tau_from_lengths_torch(
        self,
        isotope: str,
        path_by_box_cm: "torch.Tensor",
        *,
        line_count: int,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor | None":
        """Return line-resolved obstacle optical depths from path lengths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if path_by_box_cm.shape[-1] == 0:
            return torch.zeros(
                (*path_by_box_cm.shape[:-1], int(line_count)),
                device=device,
                dtype=dtype,
            )
        mu_values = self.obstacle_line_mu_values_cm_inv(isotope)
        if mu_values.shape != (int(line_count), int(path_by_box_cm.shape[-1])):
            return None
        mu_t = torch.as_tensor(mu_values, device=device, dtype=dtype)
        return torch.sum(path_by_box_cm.unsqueeze(-2) * mu_t, dim=-1)

    def _obstacle_line_compton_tau_from_lengths_torch(
        self,
        isotope: str,
        path_by_box_cm: "torch.Tensor",
        *,
        line_count: int,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor | None":
        """Return line-resolved obstacle Compton depths from path lengths."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if path_by_box_cm.shape[-1] == 0:
            return torch.zeros(
                (*path_by_box_cm.shape[:-1], int(line_count)),
                device=device,
                dtype=dtype,
            )
        mu_values = self.obstacle_line_compton_mu_values_cm_inv(isotope)
        if mu_values.shape != (
            int(line_count),
            int(path_by_box_cm.shape[-1]),
        ):
            return None
        mu_t = torch.as_tensor(mu_values, device=device, dtype=dtype)
        return torch.sum(path_by_box_cm.unsqueeze(-2) * mu_t, dim=-1)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        if not self.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu for ContinuousKernel.")
        if not _torch_device_available(self.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _blocked_mask_torch(
        self,
        dir_unit: "torch.Tensor",
        octant_index: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return a boolean mask for rays blocked by the selected octant (torch)."""
        (theta_low, theta_high), (phi_low, phi_high) = (
            self.octant_shield.theta_phi_ranges[octant_index]
        )
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

    def _shield_segment_path_length_cm(
        self,
        source_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        detector_center: NDArray[np.float64],
        physical_normal: NDArray[np.float64],
        thickness_cm: float,
        inner_radius_cm: float,
    ) -> float:
        """Return path length through the shared spherical-octant shell."""
        return segment_rotated_octant_shell_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            center_pos=detector_center,
            shield_normal=physical_normal,
            inner_radius_cm=inner_radius_cm,
            outer_radius_cm=inner_radius_cm + thickness_cm,
        )

    def _detector_aperture_targets(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return deterministic source-to-detector aperture target points."""
        detector = np.asarray(detector_pos, dtype=float)
        source = np.asarray(source_pos, dtype=float)
        aperture_radius = float(self.detector_aperture_radius_m or 0.0)
        if aperture_radius <= 0.0 or self.detector_aperture_samples <= 1:
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
        if self.detector_aperture_sampling == "solid_angle_cone":
            return self._detector_aperture_targets_cone(
                source=source,
                detector=detector,
                axis=axis,
                basis_u=basis_u,
                basis_v=basis_v,
                distance=distance,
                aperture_radius=aperture_radius,
            )
        return self._detector_aperture_targets_disk(
            detector=detector,
            basis_u=basis_u,
            basis_v=basis_v,
            distance=distance,
            aperture_radius=aperture_radius,
        )

    @staticmethod
    def _ray_perpendicular_basis(
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return a stable basis perpendicular to the source-detector ray."""
        source = np.asarray(source_pos, dtype=float)
        detector = np.asarray(detector_pos, dtype=float)
        axis = detector - source
        distance = float(np.linalg.norm(axis))
        if distance <= 1.0e-12:
            return None
        axis /= distance
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        basis_u = np.cross(axis, helper)
        basis_u_norm = float(np.linalg.norm(basis_u))
        if basis_u_norm <= 1.0e-12:
            return None
        basis_u /= basis_u_norm
        basis_v = np.cross(axis, basis_u)
        return basis_u, basis_v

    def _source_extent_points(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return deterministic source-extent sample points for ray averaging."""
        source = np.asarray(source_pos, dtype=float)
        radius = float(self.source_extent_radius_m or 0.0)
        count = int(self.source_extent_samples)
        if radius <= 0.0 or count <= 1:
            return source.reshape(1, 3)
        basis = self._ray_perpendicular_basis(source, detector_pos)
        if basis is None:
            return source.reshape(1, 3)
        basis_u, basis_v = basis
        points = np.empty((count, 3), dtype=float)
        points[0] = source
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for index in range(1, count):
            fraction = (float(index) - 0.5) / float(count - 1)
            sample_radius = radius * float(np.sqrt(np.clip(fraction, 0.0, 1.0)))
            angle = golden_angle * float(index)
            offset = sample_radius * (
                float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v
            )
            points[index] = source + offset
        return points

    def _ray_sample_points(
        self,
        source_pos: NDArray[np.float64],
        detector_pos: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return flattened source and detector target pairs for area averaging."""
        source_points = self._source_extent_points(source_pos, detector_pos)
        sources: list[NDArray[np.float64]] = []
        targets: list[NDArray[np.float64]] = []
        for source_point in source_points:
            target_points = self._detector_aperture_targets(source_point, detector_pos)
            for target_point in target_points:
                sources.append(np.asarray(source_point, dtype=float))
                targets.append(np.asarray(target_point, dtype=float))
        if not sources:
            return (
                np.asarray(source_pos, dtype=float).reshape(1, 3),
                np.asarray(detector_pos, dtype=float).reshape(1, 3),
            )
        return np.vstack(sources), np.vstack(targets)

    def _detector_aperture_targets_disk(
        self,
        *,
        detector: NDArray[np.float64],
        basis_u: NDArray[np.float64],
        basis_v: NDArray[np.float64],
        distance: float,
        aperture_radius: float,
    ) -> NDArray[np.float64]:
        """Return legacy deterministic points on the detector aperture disk."""
        radius = min(aperture_radius, 0.95 * distance)
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

    def _detector_aperture_targets_cone(
        self,
        *,
        source: NDArray[np.float64],
        detector: NDArray[np.float64],
        axis: NDArray[np.float64],
        basis_u: NDArray[np.float64],
        basis_v: NDArray[np.float64],
        distance: float,
        aperture_radius: float,
    ) -> NDArray[np.float64]:
        """Return Geant4 detector-cone targets on the aperture sphere."""
        if distance <= aperture_radius:
            return detector.reshape(1, 3)
        count = int(self.detector_aperture_samples)
        targets = np.empty((count, 3), dtype=float)
        sin_theta_max = min(max(aperture_radius / max(distance, 1.0e-12), 0.0), 1.0)
        cos_theta_max = float(np.sqrt(max(1.0 - sin_theta_max * sin_theta_max, 0.0)))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        radius_sq = aperture_radius * aperture_radius
        for index in range(count):
            fraction = (float(index) + 0.5) / float(count)
            cos_theta = 1.0 - fraction * (1.0 - cos_theta_max)
            sin_theta = float(np.sqrt(max(1.0 - cos_theta * cos_theta, 0.0)))
            angle = golden_angle * float(index)
            direction = cos_theta * axis + sin_theta * (
                float(np.cos(angle)) * basis_u + float(np.sin(angle)) * basis_v
            )
            radial_sq = (distance * sin_theta) ** 2
            chord = float(np.sqrt(max(radius_sq - radial_sq, 0.0)))
            path_length = distance * cos_theta - chord
            targets[index] = source + path_length * direction
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
        l_fe = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            physical_normal=-normal_fe,
            thickness_cm=self.shield_params.thickness_fe_cm,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
        )
        l_pb = self._shield_segment_path_length_cm(
            source_pos=source_pos,
            target_pos=target_pos,
            detector_center=detector_pos,
            physical_normal=-normal_pb,
            thickness_cm=self.shield_params.thickness_pb_cm,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
        )
        obstacle_path_by_box = np.zeros(0, dtype=float)
        tau_obstacle = 0.0
        if self.obstacle_grid is None:
            tau_obstacle = 0.0
        else:
            obstacle_path_by_box = obstacle_path_lengths_by_box_cm(
                source_pos=source_pos,
                detector_pos=target_pos,
                obstacle_boxes_m=self.obstacle_boxes_m(),
            )
            tau_obstacle = float(
                np.sum(self.obstacle_mu_values_cm_inv(isotope) * obstacle_path_by_box)
            )
        line_entries = self._line_mu_values(isotope)
        if line_entries:
            line_obstacle_tau = self._line_obstacle_tau_values(
                isotope,
                obstacle_path_by_box,
                line_count=len(line_entries),
            )
            attenuation = 0.0
            for line_index, (weight, line_mu_fe, line_mu_pb) in enumerate(line_entries):
                tau_fe = float(line_mu_fe * l_fe)
                tau_pb = float(line_mu_pb * l_pb)
                tau_obs_line = (
                    tau_obstacle
                    if line_obstacle_tau is None
                    else float(line_obstacle_tau[line_index])
                )
                total_tau = tau_fe + tau_pb + tau_obs_line
                buildup = self._buildup_factor(tau_fe, tau_pb, tau_obs_line)
                attenuation += float(weight) * float(np.exp(-total_tau)) * buildup
            return float(np.clip(attenuation, 0.0, 1.0))
        tau_fe = float(mu_fe * l_fe)
        tau_pb = float(mu_pb * l_pb)
        total_tau = tau_fe + tau_pb + tau_obstacle
        buildup = self._buildup_factor(tau_fe, tau_pb, tau_obstacle)
        return float(
            np.clip(
                float(np.exp(-total_tau)) * buildup,
                0.0,
                1.0,
            )
        )

    def _shield_path_lengths_torch(
        self,
        direction: "torch.Tensor",
        blocked_fe: "torch.Tensor",
        blocked_pb: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return Fe/Pb path lengths through spherical-octant shells."""
        l_fe = spherical_shell_path_length_cm_torch(
            direction,
            self.shield_params.inner_radius_fe_cm,
            self.shield_params.inner_radius_fe_cm
            + self.shield_params.thickness_fe_cm,
            blocked_fe,
        )
        l_pb = spherical_shell_path_length_cm_torch(
            direction,
            self.shield_params.inner_radius_pb_cm,
            self.shield_params.inner_radius_pb_cm
            + self.shield_params.thickness_pb_cm,
            blocked_pb,
        )
        return l_fe, l_pb

    @staticmethod
    def _perpendicular_bases_numpy(
        sources: NDArray[np.float64],
        detectors: NDArray[np.float64],
        *,
        tol: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return matched ray axes, distances and stable perpendicular bases."""
        source_arr = np.asarray(sources, dtype=float)
        detector_arr = np.asarray(detectors, dtype=float)
        direction = detector_arr - source_arr
        distance = np.linalg.norm(direction, axis=-1)
        safe_distance = np.maximum(distance, float(tol))
        axis = direction / safe_distance[..., None]
        helper_z = np.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = np.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = np.where(
            (np.abs(axis[..., 2]) > 0.9)[..., None],
            helper_y,
            helper_z,
        )
        basis_u = np.cross(axis, helper)
        basis_norm = np.linalg.norm(basis_u, axis=-1)
        basis_u = basis_u / np.maximum(basis_norm, float(tol))[..., None]
        basis_v = np.cross(axis, basis_u)
        return axis, distance, basis_u, basis_v

    def _source_extent_points_numpy(
        self,
        sources: NDArray[np.float64],
        detectors: NDArray[np.float64],
        *,
        tol: float,
    ) -> NDArray[np.float64]:
        """Return deterministic source-extent points for matched NumPy rows."""
        source_arr = np.asarray(sources, dtype=float)
        detector_arr = np.asarray(detectors, dtype=float)
        radius = float(self.source_extent_radius_m or 0.0)
        count = max(int(self.source_extent_samples), 1)
        if radius <= 0.0 or count <= 1:
            return source_arr[:, None, :]
        _, distance, basis_u, basis_v = self._perpendicular_bases_numpy(
            source_arr,
            detector_arr,
            tol=tol,
        )
        indices = np.arange(count, dtype=float)
        fractions = np.zeros(count, dtype=float)
        fractions[1:] = (indices[1:] - 0.5) / float(count - 1)
        radii = radius * np.sqrt(np.clip(fractions, 0.0, 1.0))
        angles = indices * float(np.pi * (3.0 - np.sqrt(5.0)))
        offsets = radii[None, :, None] * (
            np.cos(angles)[None, :, None] * basis_u[:, None, :]
            + np.sin(angles)[None, :, None] * basis_v[:, None, :]
        )
        points = source_arr[:, None, :] + offsets
        degenerate = distance <= float(tol)
        if np.any(degenerate):
            points = np.where(
                degenerate[:, None, None],
                source_arr[:, None, :],
                points,
            )
        return points

    def _detector_aperture_targets_numpy(
        self,
        sources: NDArray[np.float64],
        detectors: NDArray[np.float64],
        *,
        tol: float,
    ) -> NDArray[np.float64]:
        """Return deterministic detector-aperture targets for matched NumPy rows."""
        source_arr = np.asarray(sources, dtype=float)
        detector_arr = np.asarray(detectors, dtype=float)
        aperture_radius = float(self.detector_aperture_radius_m or 0.0)
        sample_count = max(int(self.detector_aperture_samples), 1)
        if aperture_radius <= 0.0 or sample_count <= 1:
            return detector_arr[:, None, :]
        axis, distance, basis_u, basis_v = self._perpendicular_bases_numpy(
            source_arr,
            detector_arr,
            tol=tol,
        )
        indices = np.arange(sample_count, dtype=float)
        golden_angle = float(np.pi * (3.0 - np.sqrt(5.0)))
        angles = golden_angle * indices
        angular_basis = (
            np.cos(angles)[None, :, None] * basis_u[:, None, :]
            + np.sin(angles)[None, :, None] * basis_v[:, None, :]
        )
        if self.detector_aperture_sampling == "solid_angle_cone":
            safe_distance = np.maximum(distance, float(tol))
            sin_theta_max = np.clip(
                aperture_radius / safe_distance,
                0.0,
                1.0,
            )
            cos_theta_max = np.sqrt(
                np.clip(1.0 - sin_theta_max * sin_theta_max, 0.0, None)
            )
            fractions = (indices + 0.5) / float(sample_count)
            cos_theta = 1.0 - fractions[None, :] * (
                1.0 - cos_theta_max[:, None]
            )
            sin_theta = np.sqrt(
                np.clip(1.0 - cos_theta * cos_theta, 0.0, None)
            )
            direction = (
                cos_theta[..., None] * axis[:, None, :]
                + sin_theta[..., None] * angular_basis
            )
            radial_sq = (distance[:, None] * sin_theta) ** 2
            chord = np.sqrt(
                np.clip(aperture_radius * aperture_radius - radial_sq, 0.0, None)
            )
            path_length = distance[:, None] * cos_theta - chord
            targets = (
                source_arr[:, None, :]
                + path_length[..., None] * direction
            )
            inside = distance <= aperture_radius
            if np.any(inside):
                targets = np.where(
                    inside[:, None, None],
                    detector_arr[:, None, :],
                    targets,
                )
            return targets

        fractions = np.clip(
            (indices - 0.5) / float(sample_count - 1),
            0.0,
            1.0,
        )
        radii = np.sqrt(fractions)
        radii[0] = 0.0
        max_radius = np.minimum(aperture_radius, 0.95 * distance)
        offsets = (
            max_radius[:, None, None]
            * radii[None, :, None]
            * angular_basis
        )
        targets = detector_arr[:, None, :] + offsets
        degenerate = distance <= float(tol)
        if np.any(degenerate):
            targets = np.where(
                degenerate[:, None, None],
                detector_arr[:, None, :],
                targets,
            )
        return targets

    def _ray_sample_points_numpy(
        self,
        sources: NDArray[np.float64],
        detectors: NDArray[np.float64],
        *,
        tol: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Return flattened source/target rays for matched NumPy kernel rows."""
        source_arr = np.asarray(sources, dtype=float)
        detector_arr = np.asarray(detectors, dtype=float)
        source_points = self._source_extent_points_numpy(
            source_arr,
            detector_arr,
            tol=tol,
        )
        row_count, source_sample_count, _ = source_points.shape
        flat_sources = source_points.reshape(-1, 3)
        flat_detectors = np.broadcast_to(
            detector_arr[:, None, :],
            (row_count, source_sample_count, 3),
        ).reshape(-1, 3)
        flat_targets = self._detector_aperture_targets_numpy(
            flat_sources,
            flat_detectors,
            tol=tol,
        )
        aperture_sample_count = int(flat_targets.shape[1])
        sample_count = int(source_sample_count) * aperture_sample_count
        sampled_sources = np.broadcast_to(
            flat_sources[:, None, :],
            flat_targets.shape,
        )
        return (
            sampled_sources.reshape(row_count, sample_count, 3),
            flat_targets.reshape(row_count, sample_count, 3),
            sample_count,
        )

    def _detector_aperture_targets_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", int]:
        """Return deterministic source-to-detector aperture targets for torch."""
        if torch is None:
            raise RuntimeError("torch is not available")
        aperture_radius = float(self.detector_aperture_radius_m or 0.0)
        if aperture_radius <= 0.0 or self.detector_aperture_samples <= 1:
            return detector.expand_as(sources).unsqueeze(-2), 1
        sample_count = max(int(self.detector_aperture_samples), 1)
        detector_expanded = detector.expand_as(sources)
        axis = (detector_expanded - sources) / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u = basis_u / torch.clamp(
            torch.linalg.norm(basis_u, dim=-1, keepdim=True), min=tol
        )
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        if self.detector_aperture_sampling == "solid_angle_cone":
            return (
                self._detector_aperture_targets_cone_torch(
                    sources=sources,
                    detector=detector_expanded,
                    axis=axis,
                    basis_u=basis_u,
                    basis_v=basis_v,
                    dist=dist,
                    aperture_radius=aperture_radius,
                    sample_count=sample_count,
                    tol=tol,
                ),
                sample_count,
            )
        return (
            self._detector_aperture_targets_disk_torch(
                detector=detector_expanded,
                basis_u=basis_u,
                basis_v=basis_v,
                dist=dist,
                aperture_radius=aperture_radius,
                sample_count=sample_count,
            ),
            sample_count,
        )

    def _source_extent_points_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", int]:
        """Return deterministic source-extent sample points for torch kernels."""
        if torch is None:
            raise RuntimeError("torch is not available")
        radius = float(self.source_extent_radius_m or 0.0)
        if radius <= 0.0 or self.source_extent_samples <= 1:
            return sources.unsqueeze(-2), 1
        sample_count = max(int(self.source_extent_samples), 1)
        detector_expanded = (
            detector.expand_as(sources)
            if int(detector.shape[0]) == 1
            else detector.reshape_as(sources)
        )
        axis = (detector_expanded - sources) / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u = basis_u / torch.clamp(
            torch.linalg.norm(basis_u, dim=-1, keepdim=True),
            min=tol,
        )
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        indices = torch.arange(sample_count, device=sources.device, dtype=sources.dtype)
        fractions = torch.zeros_like(indices)
        if sample_count > 1:
            fractions[1:] = (indices[1:] - 0.5) / float(sample_count - 1)
        radii = float(radius) * torch.sqrt(torch.clamp(fractions, min=0.0, max=1.0))
        angles = indices * float(np.pi * (3.0 - np.sqrt(5.0)))
        offsets = radii.view(1, sample_count, 1) * (
            torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
            + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
        )
        return sources.unsqueeze(-2) + offsets, sample_count

    def _ray_sample_points_torch(
        self,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        dist: "torch.Tensor",
        tol: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", int]:
        """Return flattened source and detector target pairs for torch kernels."""
        if torch is None:
            raise RuntimeError("torch is not available")
        source_points, source_sample_count = self._source_extent_points_torch(
            sources=sources,
            detector=detector,
            dist=dist,
            tol=tol,
        )
        detector_expanded = (
            detector.expand_as(sources)
            if int(detector.shape[0]) == 1
            else detector.reshape_as(sources)
        )
        flat_sources = source_points.reshape(-1, 3)
        flat_detectors = (
            detector_expanded.unsqueeze(-2)
            .expand(-1, source_sample_count, -1)
            .reshape(-1, 3)
        )
        flat_dist = torch.linalg.norm(flat_detectors - flat_sources, dim=1)
        tol_t = torch.as_tensor(tol, device=sources.device, dtype=sources.dtype)
        flat_dist = torch.where(flat_dist <= tol_t, tol_t, flat_dist)
        flat_targets, aperture_sample_count = self._detector_aperture_targets_torch(
            sources=flat_sources,
            detector=flat_detectors,
            dist=flat_dist,
            tol=tol,
        )
        total_sample_count = int(source_sample_count) * int(aperture_sample_count)
        sampled_sources = flat_sources.unsqueeze(-2).expand_as(flat_targets)
        return (
            sampled_sources.reshape(int(sources.shape[0]), total_sample_count, 3),
            flat_targets.reshape(int(sources.shape[0]), total_sample_count, 3),
            total_sample_count,
        )

    def _detector_aperture_targets_disk_torch(
        self,
        *,
        detector: "torch.Tensor",
        basis_u: "torch.Tensor",
        basis_v: "torch.Tensor",
        dist: "torch.Tensor",
        aperture_radius: float,
        sample_count: int,
    ) -> "torch.Tensor":
        """Return legacy deterministic disk aperture targets for torch."""
        indices = torch.arange(
            sample_count,
            device=detector.device,
            dtype=detector.dtype,
        )
        fractions = torch.clamp(
            (indices - 0.5) / float(sample_count - 1),
            min=0.0,
            max=1.0,
        )
        radii = torch.sqrt(fractions)
        radii[0] = 0.0
        max_radius = torch.minimum(
            torch.as_tensor(
                aperture_radius, device=detector.device, dtype=detector.dtype
            ),
            0.95 * dist,
        )
        angles = indices * torch.as_tensor(
            np.pi * (3.0 - np.sqrt(5.0)),
            device=detector.device,
            dtype=detector.dtype,
        )
        offsets = (
            max_radius.unsqueeze(-1).unsqueeze(-1)
            * radii.view(1, sample_count, 1)
            * (
                torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
            )
        )
        return detector.unsqueeze(-2) + offsets

    def _detector_aperture_targets_cone_torch(
        self,
        *,
        sources: "torch.Tensor",
        detector: "torch.Tensor",
        axis: "torch.Tensor",
        basis_u: "torch.Tensor",
        basis_v: "torch.Tensor",
        dist: "torch.Tensor",
        aperture_radius: float,
        sample_count: int,
        tol: float,
    ) -> "torch.Tensor":
        """Return Geant4 detector-cone targets on the aperture sphere for torch."""
        radius_t = torch.as_tensor(
            aperture_radius,
            device=sources.device,
            dtype=sources.dtype,
        )
        indices = torch.arange(sample_count, device=sources.device, dtype=sources.dtype)
        fractions = (indices + 0.5) / float(sample_count)
        dist_safe = torch.clamp(dist, min=tol)
        sin_theta_max = torch.clamp(radius_t / dist_safe, min=0.0, max=1.0)
        cos_theta_max = torch.sqrt(
            torch.clamp(1.0 - sin_theta_max * sin_theta_max, min=0.0)
        )
        cos_theta = 1.0 - fractions.view(1, sample_count) * (
            1.0 - cos_theta_max.unsqueeze(-1)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
        angles = indices * torch.as_tensor(
            np.pi * (3.0 - np.sqrt(5.0)),
            device=sources.device,
            dtype=sources.dtype,
        )
        direction = cos_theta.unsqueeze(-1) * axis.unsqueeze(-2) + sin_theta.unsqueeze(
            -1
        ) * (
            torch.cos(angles).view(1, sample_count, 1) * basis_u.unsqueeze(-2)
            + torch.sin(angles).view(1, sample_count, 1) * basis_v.unsqueeze(-2)
        )
        radial_sq = (dist.unsqueeze(-1) * sin_theta) ** 2
        chord = torch.sqrt(torch.clamp(radius_t * radius_t - radial_sq, min=0.0))
        path_length = dist.unsqueeze(-1) * cos_theta - chord
        targets = sources.unsqueeze(-2) + path_length.unsqueeze(-1) * direction
        inside = dist <= radius_t
        if bool(torch.any(inside)):
            fallback = detector.unsqueeze(-2).expand_as(targets)
            targets = torch.where(inside.view(-1, 1, 1), fallback, targets)
        return targets

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
        fe_arr, pb_arr = validate_orientation_pair_indices(
            np.asarray([fe_index]),
            np.asarray([pb_index]),
            orientation_count=int(len(self.orientations)),
            expected_count=1,
        )
        fe_index = int(fe_arr[0])
        pb_index = int(pb_arr[0])
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return float(background)
        strengths_t = torch.as_tensor(strengths, device=device, dtype=dtype)
        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(
            1, 3
        )
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        sampled_sources, targets, sample_count = self._ray_sample_points_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)

        center = detector_t.expand_as(sources_t).unsqueeze(-2)
        fe_normal = -np.asarray(self.orientations[fe_index], dtype=float)
        pb_normal = -np.asarray(self.orientations[pb_index], dtype=float)
        fe_rotation = self._rotated_octant_rotation_torch(
            fe_normal,
            device=device,
            dtype=dtype,
        )
        pb_rotation = self._rotated_octant_rotation_torch(
            pb_normal,
            device=device,
            dtype=dtype,
        )
        L_fe = segment_rotated_octant_shell_path_length_cm_torch(
            source_pos=sampled_sources,
            target_pos=targets,
            center_pos=center,
            shield_normal=fe_normal,
            inner_radius_cm=self.shield_params.inner_radius_fe_cm,
            outer_radius_cm=self.shield_params.inner_radius_fe_cm
            + self.shield_params.thickness_fe_cm,
            tol=tol,
            rotation=fe_rotation,
        )
        L_pb = segment_rotated_octant_shell_path_length_cm_torch(
            source_pos=sampled_sources,
            target_pos=targets,
            center_pos=center,
            shield_normal=pb_normal,
            inner_radius_cm=self.shield_params.inner_radius_pb_cm,
            outer_radius_cm=self.shield_params.inner_radius_pb_cm
            + self.shield_params.thickness_pb_cm,
            tol=tol,
            rotation=pb_rotation,
        )
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = float(mu_fe) * L_fe
        tau_pb = float(mu_pb) * L_pb
        tau_obstacle = torch.zeros_like(tau_fe)
        obstacle_path_cm = None
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            if sample_count > 1:
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                )
            else:
                obstacle_path_cm = obstacle_path_lengths_by_box_cm_torch(
                    positions=sources_t,
                    detector_pos=detector_t.reshape(3),
                    obstacle_boxes_m=boxes_t,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                ).unsqueeze(-1)
        line_entries = self._line_mu_values(isotope)
        if line_entries:
            weights_t = torch.as_tensor(
                [entry[0] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_fe_t = torch.as_tensor(
                [entry[1] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_pb_t = torch.as_tensor(
                [entry[2] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            line_tau_fe = L_fe.unsqueeze(-1) * mu_fe_t.view(1, 1, -1)
            line_tau_pb = L_pb.unsqueeze(-1) * mu_pb_t.view(1, 1, -1)
            line_tau_obstacle = tau_obstacle.unsqueeze(-1)
            if obstacle_path_cm is not None:
                candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    line_count=len(line_entries),
                    device=device,
                    dtype=dtype,
                )
                if candidate_tau is not None:
                    line_tau_obstacle = (
                        candidate_tau
                        if sample_count > 1
                        else candidate_tau.unsqueeze(-2)
                    )
            line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
            line_buildup = self._buildup_factor_torch(
                line_tau_fe,
                line_tau_pb,
                line_tau_obstacle,
            )
            base_att = torch.sum(
                torch.exp(-line_total_tau) * line_buildup * weights_t.view(1, 1, -1),
                dim=-1,
            )
            att = torch.clamp(base_att, min=0.0, max=1.0)
        else:
            total_tau = tau_fe + tau_pb + tau_obstacle
            buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
            att = torch.clamp(
                torch.exp(-total_tau) * buildup,
                min=0.0,
                max=1.0,
            )
        att = torch.mean(att, dim=-1)
        rate = torch.sum(geom * att * strengths_t) + float(background)
        return float(rate.detach().cpu().item())

    def _kernel_values_selected_pairs_torch_tensor(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources_t: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        tol: float = 1e-6,
    ) -> "torch.Tensor":
        """Return selected Fe/Pb-pair kernels for flat torch source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        if sources_t.ndim != 2 or int(sources_t.shape[1]) != 3:
            raise ValueError("sources_t must be shaped (N, 3).")
        device = sources_t.device
        dtype = sources_t.dtype
        num_orients = int(len(self.orientations))
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=num_orients,
        )
        pair_count = int(fe_arr.size)
        if pair_count == 0:
            return torch.zeros((0, int(sources_t.shape[0])), device=device, dtype=dtype)
        if sources_t.numel() == 0:
            return torch.zeros((pair_count, 0), device=device, dtype=dtype)

        detector_t = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(
            1, 3
        )
        direction = detector_t - sources_t
        dist = torch.linalg.norm(direction, dim=1)
        tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
        dist = torch.where(dist <= tol_t, tol_t, dist)
        geom = _finite_sphere_geometric_term_torch(
            dist,
            detector_radius_m=self.detector_radius_m,
            tol=tol_t,
        )
        sampled_sources, targets, sample_count = self._ray_sample_points_torch(
            sources=sources_t,
            detector=detector_t,
            dist=dist,
            tol=tol,
        )
        sampled_direction = targets - sampled_sources
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
        unique_orients = np.unique(np.concatenate([fe_arr, pb_arr]))
        path_lengths: dict[int, tuple["torch.Tensor", "torch.Tensor"]] = {}
        for orient_idx in unique_orients:
            orient_int = int(orient_idx)
            center = detector_t.expand_as(sources_t).unsqueeze(-2)
            shield_normal = -np.asarray(
                self.orientations[orient_int],
                dtype=float,
            )
            rotation = self._rotated_octant_rotation_torch(
                shield_normal,
                device=device,
                dtype=dtype,
            )
            l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=shield_normal,
                inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                outer_radius_cm=(
                    self.shield_params.inner_radius_fe_cm
                    + self.shield_params.thickness_fe_cm
                ),
                tol=tol,
                rotation=rotation,
            )
            l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                center_pos=center,
                shield_normal=shield_normal,
                inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                outer_radius_cm=(
                    self.shield_params.inner_radius_pb_cm
                    + self.shield_params.thickness_pb_cm
                ),
                tol=tol,
                rotation=rotation,
            )
            path_lengths[orient_int] = (l_fe, l_pb)

        l_fe_pairs = torch.stack([path_lengths[int(idx)][0] for idx in fe_arr], dim=0)
        l_pb_pairs = torch.stack([path_lengths[int(idx)][1] for idx in pb_arr], dim=0)
        tau_fe = torch.zeros_like(l_fe_pairs)
        tau_pb = torch.zeros_like(l_pb_pairs)
        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = tau_fe + float(mu_fe) * l_fe_pairs
        tau_pb = tau_pb + float(mu_pb) * l_pb_pairs

        obstacle_path_cm = None
        tau_obstacle_base = torch.zeros(
            sampled_sources.shape[:-1],
            device=device,
            dtype=dtype,
        )
        boxes_np = self.obstacle_boxes_m()
        if boxes_np.size:
            boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
            obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                source_pos=sampled_sources,
                target_pos=targets,
                obstacle_boxes_m=boxes_t,
                tol=tol,
            )
            tau_obstacle_base = self._obstacle_tau_from_lengths_torch(
                isotope,
                obstacle_path_cm,
                device=device,
                dtype=dtype,
            )
        tau_obstacle = tau_obstacle_base.unsqueeze(0)

        line_entries = self._line_mu_values(isotope)
        if line_entries:
            weights_t = torch.as_tensor(
                [entry[0] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_fe_t = torch.as_tensor(
                [entry[1] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            mu_pb_t = torch.as_tensor(
                [entry[2] for entry in line_entries],
                device=device,
                dtype=dtype,
            )
            line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, 1, -1)
            line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, 1, -1)
            line_tau_obstacle = tau_obstacle.unsqueeze(-1)
            if obstacle_path_cm is not None:
                candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    line_count=len(line_entries),
                    device=device,
                    dtype=dtype,
                )
                if candidate_tau is not None:
                    line_tau_obstacle = candidate_tau.unsqueeze(0)
            line_buildup = self._buildup_factor_torch(
                line_tau_fe,
                line_tau_pb,
                line_tau_obstacle,
            )
            base_att = torch.sum(
                torch.exp(-(line_tau_fe + line_tau_pb + line_tau_obstacle))
                * line_buildup
                * weights_t.view(1, 1, 1, -1),
                dim=-1,
            )
            att = torch.clamp(base_att, min=0.0, max=1.0)
        else:
            total_tau = tau_fe + tau_pb + tau_obstacle
            buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
            att = torch.clamp(
                torch.exp(-total_tau) * buildup,
                min=0.0,
                max=1.0,
            )

        att = torch.mean(att, dim=-1)
        return geom.unsqueeze(0) * att

    def _expected_counts_selected_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute selected pair counts from the ContinuousKernel torch kernel."""
        if torch is None:
            raise RuntimeError("torch is not available")
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=int(len(self.orientations)),
        )
        pair_count = int(fe_arr.size)
        particle_count = int(positions.shape[0])
        slot_count = int(positions.shape[1]) if positions.ndim >= 2 else 0
        if pair_count == 0:
            return torch.zeros((0, particle_count), device=device, dtype=dtype)
        if particle_count == 0 or slot_count == 0:
            source_scale_t = _source_scale_rows_torch(
                source_scale,
                pair_count,
                device=device,
                dtype=dtype,
            )
            rate = source_scale_t * torch.zeros(
                (pair_count, particle_count),
                device=device,
                dtype=dtype,
            )
            return float(live_time_s) * (rate + backgrounds.unsqueeze(0))

        all_pairs = pair_count >= int(len(self.orientations)) ** 2
        source_chunk = self._adaptive_torch_chunk_size(
            8192,
            isotope=isotope,
            orientation_pair_count=pair_count,
            device=device,
            dtype=dtype,
            all_orientation_pairs=all_pairs,
        )
        particle_chunk = max(1, int(source_chunk) // max(slot_count, 1))
        strengths_masked = strengths * mask

        def _evaluate_particle_chunk(
            start: int,
            stop: int,
        ) -> "torch.Tensor":
            """Return source terms for one contiguous particle chunk."""
            flat_sources = positions[start:stop].reshape(-1, 3)
            kernel_values = self._kernel_values_selected_pairs_torch_tensor(
                isotope=isotope,
                detector_pos=detector_pos,
                sources_t=flat_sources,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
            )
            kernel_values = kernel_values.reshape(
                pair_count,
                stop - start,
                slot_count,
            )
            weighted = kernel_values * strengths_masked[start:stop].unsqueeze(0)
            return torch.sum(weighted, dim=-1)

        with torch.no_grad():
            source_terms, _ = self._evaluate_torch_chunks_with_oom_retry(
                total_size=particle_count,
                initial_chunk=particle_chunk,
                device=device,
                evaluator=_evaluate_particle_chunk,
            )
            source_term = torch.cat(source_terms, dim=1)
            source_scale_t = _source_scale_rows_torch(
                source_scale,
                pair_count,
                device=device,
                dtype=dtype,
            )
            rate = source_scale_t * source_term + backgrounds.unsqueeze(0)
        return float(live_time_s) * rate

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
        values = self._kernel_values_selected_pairs_torch_tensor(
            isotope=isotope,
            detector_pos=detector_pos,
            sources_t=sources_t,
            fe_indices=np.asarray([fe_index], dtype=np.int64),
            pb_indices=np.asarray([pb_index], dtype=np.int64),
            tol=tol,
        )
        return values[0].detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_all_pairs_torch_chunk(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return per-source kernel values for every Fe/Pb pair on the GPU."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_t.numel() == 0:
            return np.zeros((num_pairs, 0), dtype=float)
        pair_ids = np.arange(num_pairs, dtype=np.int64)
        values = self._kernel_values_selected_pairs_torch_tensor(
            isotope=isotope,
            detector_pos=detector_pos,
            sources_t=sources_t,
            fe_indices=pair_ids // num_orients,
            pb_indices=pair_ids % num_orients,
            tol=tol,
        )
        return values.detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_all_pairs_for_detector_source_torch_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Return all Fe/Pb pair kernels for matched detector-source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        detectors_t = torch.as_tensor(detector_positions, device=device, dtype=dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if detectors_t.ndim != 2 or detectors_t.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if sources_t.ndim != 2 or sources_t.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors_t.shape[0] != sources_t.shape[0]:
            raise ValueError(
                "detector_positions and sources must have the same row count."
            )
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_t.numel() == 0:
            return np.zeros((0, num_pairs), dtype=float)
        with torch.no_grad():
            direction = detectors_t - sources_t
            dist = torch.linalg.norm(direction, dim=1)
            tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
            dist = torch.where(dist <= tol_t, tol_t, dist)
            geom = _finite_sphere_geometric_term_torch(
                dist,
                detector_radius_m=self.detector_radius_m,
                tol=tol_t,
            )
            sampled_sources, targets, sample_count = self._ray_sample_points_torch(
                sources=sources_t,
                detector=detectors_t,
                dist=dist,
                tol=tol,
            )
            sampled_direction = targets - sampled_sources
            sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
            sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
            l_fe_by_orient: list[torch.Tensor] = []
            l_pb_by_orient: list[torch.Tensor] = []
            for orient_idx in range(num_orients):
                shield_normal = -np.asarray(
                    self.orientations[orient_idx],
                    dtype=float,
                )
                rotation = self._rotated_octant_rotation_torch(
                    shield_normal,
                    device=device,
                    dtype=dtype,
                )
                center = detectors_t.unsqueeze(-2)
                l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_fe_cm
                        + self.shield_params.thickness_fe_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
                l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_pb_cm
                        + self.shield_params.thickness_pb_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
                l_fe_by_orient.append(l_fe)
                l_pb_by_orient.append(l_pb)

            l_fe_stack = torch.stack(l_fe_by_orient, dim=0)
            l_pb_stack = torch.stack(l_pb_by_orient, dim=0)
            pair_ids = torch.arange(num_pairs, device=device, dtype=torch.long)
            fe_indices = torch.div(pair_ids, num_orients, rounding_mode="floor")
            pb_indices = torch.remainder(pair_ids, num_orients)
            l_fe_pairs = l_fe_stack.index_select(0, fe_indices)
            l_pb_pairs = l_pb_stack.index_select(0, pb_indices)

            mu_fe, mu_pb = self._mu_values(isotope=isotope)
            tau_fe = float(mu_fe) * l_fe_pairs
            tau_pb = float(mu_pb) * l_pb_pairs
            tau_obstacle = torch.zeros_like(tau_fe)
            obstacle_path_cm = None
            boxes_np = self.obstacle_boxes_m()
            if boxes_np.size:
                boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                ).unsqueeze(0)
            line_entries = self._line_mu_values(isotope)
            if line_entries:
                weights_t = torch.as_tensor(
                    [entry[0] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_fe_t = torch.as_tensor(
                    [entry[1] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_pb_t = torch.as_tensor(
                    [entry[2] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, 1, -1)
                line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, 1, -1)
                line_tau_obstacle = tau_obstacle.unsqueeze(-1)
                if obstacle_path_cm is not None:
                    candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                        isotope,
                        obstacle_path_cm,
                        line_count=len(line_entries),
                        device=device,
                        dtype=dtype,
                    )
                    if candidate_tau is not None:
                        line_tau_obstacle = candidate_tau.unsqueeze(0)
                line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
                line_buildup = self._buildup_factor_torch(
                    line_tau_fe,
                    line_tau_pb,
                    line_tau_obstacle,
                )
                base_att = torch.sum(
                    torch.exp(-line_total_tau)
                    * line_buildup
                    * weights_t.view(1, 1, 1, -1),
                    dim=-1,
                )
                att = torch.clamp(base_att, min=0.0, max=1.0)
            else:
                total_tau = tau_fe + tau_pb + tau_obstacle
                buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
                att = torch.clamp(
                    torch.exp(-total_tau) * buildup,
                    min=0.0,
                    max=1.0,
                )
            att = torch.mean(att, dim=-1)
            values = geom.unsqueeze(0) * att
        return values.transpose(0, 1).detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_unshielded_for_detector_source_numpy_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1.0e-12,
    ) -> NDArray[np.float64]:
        """Return matched-row physical kernels without Fe/Pb attenuation."""
        detectors = np.asarray(detector_positions, dtype=float)
        source_arr = np.asarray(sources, dtype=float)
        if detectors.ndim != 2 or detectors.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if source_arr.ndim != 2 or source_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors.shape[0] != source_arr.shape[0]:
            raise ValueError(
                "detector_positions and sources must have the same row count."
            )
        if (
            not np.all(np.isfinite(detectors))
            or not np.all(np.isfinite(source_arr))
        ):
            raise ValueError(
                "detector_positions and sources must contain finite values."
            )
        row_count = int(source_arr.shape[0])
        if row_count == 0:
            return np.zeros(0, dtype=float)

        center_distance = np.linalg.norm(detectors - source_arr, axis=1)
        geom = _finite_sphere_geometric_term_numpy(
            center_distance,
            detector_radius_m=self.detector_radius_m,
        )
        sampled_sources, targets, _ = self._ray_sample_points_numpy(
            source_arr,
            detectors,
            tol=tol,
        )
        obstacle_path_cm = None
        tau_obstacle = np.zeros(sampled_sources.shape[:-1], dtype=float)
        boxes = self.obstacle_boxes_m()
        if boxes.size:
            obstacle_path_cm = (
                _obstacle_path_lengths_between_points_by_box_cm_numpy(
                    sampled_sources,
                    targets,
                    boxes,
                    tol=tol,
                )
            )
            obstacle_mu = self.obstacle_mu_values_cm_inv(isotope)
            tau_obstacle = np.sum(
                obstacle_path_cm * obstacle_mu.reshape(1, 1, -1),
                axis=-1,
            )

        line_entries = self._line_mu_values(isotope)
        if line_entries:
            line_values = np.asarray(line_entries, dtype=float)
            weights = line_values[:, 0]
            line_tau_obstacle = tau_obstacle[..., None]
            if obstacle_path_cm is not None:
                obstacle_line_mu = self.obstacle_line_mu_values_cm_inv(isotope)
                if obstacle_line_mu.shape == (
                    len(line_entries),
                    obstacle_path_cm.shape[-1],
                ):
                    line_tau_obstacle = np.einsum(
                        "nrb,lb->nrl",
                        obstacle_path_cm,
                        obstacle_line_mu,
                        optimize=True,
                    )
            zeros = np.zeros_like(line_tau_obstacle)
            line_buildup = self._buildup_factor_numpy(
                zeros,
                zeros,
                line_tau_obstacle,
            )
            attenuation = np.sum(
                np.exp(-line_tau_obstacle)
                * line_buildup
                * weights.reshape(1, 1, -1),
                axis=-1,
            )
        else:
            zeros = np.zeros_like(tau_obstacle)
            buildup = self._buildup_factor_numpy(
                zeros,
                zeros,
                tau_obstacle,
            )
            attenuation = np.exp(-tau_obstacle) * buildup
        attenuation = np.clip(attenuation, 0.0, 1.0)
        values = geom * np.mean(attenuation, axis=-1)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(
                "Unshielded physical kernel produced an invalid response."
            )
        return values.astype(float, copy=False)

    def _kernel_values_selected_pairs_for_detector_source_numpy_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        tol: float = 1.0e-12,
        positive_line_indices: object | None = None,
        return_line_transport_components: bool = False,
    ) -> NDArray[np.float64] | LineTransportComponents:
        """Return selected-pair kernels for matched rows via batched NumPy."""
        line_selection = (
            None
            if positive_line_indices is None
            else self._validated_positive_line_indices(
                isotope,
                positive_line_indices,
            )
        )
        detectors = np.asarray(detector_positions, dtype=float)
        source_arr = np.asarray(sources, dtype=float)
        if detectors.ndim != 2 or detectors.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if source_arr.ndim != 2 or source_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors.shape[0] != source_arr.shape[0]:
            raise ValueError(
                "detector_positions and sources must have the same row count."
            )
        row_count = int(source_arr.shape[0])
        if row_count == 0:
            if line_selection is None:
                return np.zeros(0, dtype=float)
            if return_line_transport_components:
                empty = np.zeros(
                    (0, int(line_selection.size)),
                    dtype=np.float64,
                )
                return LineTransportComponents(
                    total_kernel=empty,
                    unattenuated_kernel=empty.copy(),
                    uncollided_kernel=empty.copy(),
                    tau_fe=empty.copy(),
                    tau_pb=empty.copy(),
                    tau_obstacle=empty.copy(),
                    tau_obstacle_compton=empty.copy(),
                    distance_m=empty.copy(),
                )
            return np.zeros((0, line_selection.size), dtype=float)
        if return_line_transport_components and line_selection is None:
            raise ValueError(
                "Line transport components require positive_line_indices."
            )
        num_orients = int(len(self.orientations))
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=num_orients,
            expected_count=row_count,
        )

        center_direction = detectors - source_arr
        center_distance = np.linalg.norm(center_direction, axis=1)
        geom = _finite_sphere_geometric_term_numpy(
            center_distance,
            detector_radius_m=self.detector_radius_m,
        )
        sampled_sources, targets, _ = self._ray_sample_points_numpy(
            source_arr,
            detectors,
            tol=tol,
        )
        response_distance = np.linalg.norm(
            sampled_sources - detectors[:, None, :],
            axis=-1,
        )
        rotations = self._rotated_octant_rotations_numpy()
        fe_rotations = rotations[fe_arr]
        pb_rotations = rotations[pb_arr]
        centers = detectors[:, None, :]
        l_fe = _segment_rotated_octant_shell_path_length_cm_numpy(
            sampled_sources,
            targets,
            centers,
            fe_rotations,
            self.shield_params.inner_radius_fe_cm,
            (
                self.shield_params.inner_radius_fe_cm
                + self.shield_params.thickness_fe_cm
            ),
            tol=tol,
        )
        l_pb = _segment_rotated_octant_shell_path_length_cm_numpy(
            sampled_sources,
            targets,
            centers,
            pb_rotations,
            self.shield_params.inner_radius_pb_cm,
            (
                self.shield_params.inner_radius_pb_cm
                + self.shield_params.thickness_pb_cm
            ),
            tol=tol,
        )

        mu_fe, mu_pb = self._mu_values(isotope=isotope)
        tau_fe = float(mu_fe) * l_fe
        tau_pb = float(mu_pb) * l_pb
        obstacle_path_cm = None
        tau_obstacle = np.zeros_like(tau_fe)
        boxes = self.obstacle_boxes_m()
        if boxes.size:
            obstacle_path_cm = (
                _obstacle_path_lengths_between_points_by_box_cm_numpy(
                    sampled_sources,
                    targets,
                    boxes,
                    tol=tol,
                )
            )
            obstacle_mu = self.obstacle_mu_values_cm_inv(isotope)
            tau_obstacle = np.sum(
                obstacle_path_cm * obstacle_mu.reshape(1, 1, -1),
                axis=-1,
            )

        line_entries = self._line_mu_values(isotope)
        if line_entries:
            line_values = np.asarray(line_entries, dtype=float)
            weights = line_values[:, 0]
            line_mu_fe = line_values[:, 1]
            line_mu_pb = line_values[:, 2]
            line_tau_fe = l_fe[..., None] * line_mu_fe.reshape(1, 1, -1)
            line_tau_pb = l_pb[..., None] * line_mu_pb.reshape(1, 1, -1)
            line_tau_obstacle = np.broadcast_to(
                tau_obstacle[..., None],
                line_tau_fe.shape,
            ).copy()
            line_tau_obstacle_compton = np.zeros_like(line_tau_fe)
            if obstacle_path_cm is not None:
                obstacle_line_mu = self.obstacle_line_mu_values_cm_inv(isotope)
                if obstacle_line_mu.shape == (
                    len(line_entries),
                    obstacle_path_cm.shape[-1],
                ):
                    line_tau_obstacle = np.einsum(
                        "nrb,lb->nrl",
                        obstacle_path_cm,
                        obstacle_line_mu,
                        optimize=True,
                    )
                obstacle_compton_mu = (
                    self.obstacle_line_compton_mu_values_cm_inv(isotope)
                )
                if obstacle_compton_mu.shape == (
                    len(line_entries),
                    obstacle_path_cm.shape[-1],
                ):
                    line_tau_obstacle_compton = np.einsum(
                        "nrb,lb->nrl",
                        obstacle_path_cm,
                        obstacle_compton_mu,
                        optimize=True,
                    )
            line_total_tau = (
                line_tau_fe + line_tau_pb + line_tau_obstacle
            )
            uncollided_line_attenuation = np.exp(-line_total_tau)
            if self.additive_scatter_response is not None:
                line_energies = np.asarray(
                    self._line_energy_values_keV(isotope),
                    dtype=np.float64,
                )
                if line_energies.shape != (len(line_entries),):
                    raise RuntimeError(
                        "Additive scatter requires exact positive-line energies."
                    )
                scatter_basis = physical_scatter_basis_numpy(
                    tau_fe=line_tau_fe,
                    tau_pb=line_tau_pb,
                    tau_obstacle=line_tau_obstacle,
                    tau_obstacle_compton=line_tau_obstacle_compton,
                    distance_m=np.broadcast_to(
                        response_distance[..., None],
                        line_total_tau.shape,
                    ),
                    energy_keV=line_energies.reshape(1, 1, -1),
                    mu_fe_cm_inv=line_mu_fe.reshape(1, 1, -1),
                    mu_pb_cm_inv=line_mu_pb.reshape(1, 1, -1),
                )
                line_base_attenuation = (
                    self.additive_scatter_response.total_kernel_numpy(
                        np.ones_like(uncollided_line_attenuation),
                        uncollided_line_attenuation,
                        scatter_basis,
                    )
                )
            else:
                line_buildup = self._buildup_factor_numpy(
                    line_tau_fe,
                    line_tau_pb,
                    line_tau_obstacle,
                )
                line_base_attenuation = (
                    uncollided_line_attenuation * line_buildup
                )
            weight_view = weights.reshape(1, 1, -1)
            base_attenuation = np.sum(
                line_base_attenuation * weight_view,
                axis=-1,
            )
        else:
            total_tau = tau_fe + tau_pb + tau_obstacle
            buildup = self._buildup_factor_numpy(
                tau_fe,
                tau_pb,
                tau_obstacle,
            )
            base_attenuation = np.exp(-total_tau) * buildup

        if line_selection is not None:
            if not line_entries:
                raise RuntimeError(
                    "Validated positive-line selection lacks line entries."
                )
            if self.additive_scatter_response is None:
                capped_base = np.clip(base_attenuation, 0.0, 1.0)
                aggregate_scale = np.divide(
                    capped_base,
                    base_attenuation,
                    out=np.zeros_like(capped_base),
                    where=base_attenuation > 0.0,
                )
            else:
                aggregate_scale = np.ones_like(base_attenuation)
            selected_attenuation = (
                line_base_attenuation[..., line_selection]
                * aggregate_scale[..., None]
            )
            if return_line_transport_components:
                selected = np.asarray(line_selection, dtype=np.int64)
                uncollided_attenuation = np.exp(
                    -line_total_tau[..., selected]
                )
                total_kernel = (
                    geom[:, None]
                    * np.mean(selected_attenuation, axis=1)
                )
                uncollided_kernel = (
                    geom[:, None]
                    * np.mean(uncollided_attenuation, axis=1)
                )
                unattenuated_kernel = np.broadcast_to(
                    geom[:, None],
                    total_kernel.shape,
                ).copy()
                direct_attenuation = np.minimum(
                    selected_attenuation,
                    uncollided_attenuation,
                )
                scatter_weights = np.maximum(
                    selected_attenuation - direct_attenuation,
                    0.0,
                )
                fallback_weights = np.maximum(
                    uncollided_attenuation,
                    0.0,
                )
                scatter_weight_sum = np.sum(
                    scatter_weights,
                    axis=1,
                    keepdims=True,
                )
                feature_weights = np.where(
                    scatter_weight_sum > np.finfo(np.float64).tiny,
                    scatter_weights,
                    fallback_weights,
                )
                feature_weight_sum = np.maximum(
                    np.sum(feature_weights, axis=1),
                    np.finfo(np.float64).tiny,
                )

                def _weighted_feature(
                    values: NDArray[np.float64],
                ) -> NDArray[np.float64]:
                    """Average one ray feature by its spectral contribution."""
                    return np.sum(
                        values * feature_weights,
                        axis=1,
                    ) / feature_weight_sum

                return LineTransportComponents(
                    total_kernel=np.asarray(
                        total_kernel,
                        dtype=np.float64,
                    ),
                    unattenuated_kernel=np.asarray(
                        unattenuated_kernel,
                        dtype=np.float64,
                    ),
                    uncollided_kernel=np.asarray(
                        uncollided_kernel,
                        dtype=np.float64,
                    ),
                    tau_fe=_weighted_feature(
                        line_tau_fe[..., selected],
                    ),
                    tau_pb=_weighted_feature(
                        line_tau_pb[..., selected],
                    ),
                    tau_obstacle=_weighted_feature(
                        line_tau_obstacle[..., selected],
                    ),
                    tau_obstacle_compton=_weighted_feature(
                        line_tau_obstacle_compton[..., selected],
                    ),
                    distance_m=_weighted_feature(
                        np.broadcast_to(
                            response_distance[..., None],
                            selected_attenuation.shape,
                        )
                    ),
                )
            mean_selected_attenuation = np.mean(
                selected_attenuation,
                axis=1,
            )
            return (
                geom[:, None] * mean_selected_attenuation
            ).astype(float, copy=False)

        attenuation = (
            np.maximum(base_attenuation, 0.0)
            if self.additive_scatter_response is not None
            else np.clip(base_attenuation, 0.0, 1.0)
        )
        mean_attenuation = np.mean(attenuation, axis=-1)
        return (geom * mean_attenuation).astype(float, copy=False)

    def _kernel_values_all_pairs_for_detectors_numpy(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        *,
        chunk_size: int,
    ) -> NDArray[np.float64]:
        """Return detector-by-pair-by-source kernels via batched NumPy rows."""
        detectors = np.asarray(detector_positions, dtype=float)
        source_arr = np.asarray(sources, dtype=float)
        if detectors.ndim != 2 or detectors.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if source_arr.ndim != 2 or source_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        detector_count = int(detectors.shape[0])
        source_count = int(source_arr.shape[0])
        orientation_count = int(len(self.orientations))
        pair_count = orientation_count * orientation_count
        if detector_count == 0 or source_count == 0:
            return np.zeros(
                (detector_count, pair_count, source_count),
                dtype=float,
            )
        pair_fe = np.repeat(
            np.arange(orientation_count, dtype=np.int64),
            orientation_count,
        )
        pair_pb = np.tile(
            np.arange(orientation_count, dtype=np.int64),
            orientation_count,
        )
        rows_per_detector = pair_count * source_count
        total_rows = detector_count * rows_per_detector
        chunk = self._adaptive_numpy_chunk_size(
            chunk_size,
            isotope=isotope,
        )
        parts: list[NDArray[np.float64]] = []
        for start in range(0, total_rows, chunk):
            stop = min(start + chunk, total_rows)
            flat_rows = np.arange(start, stop, dtype=np.int64)
            detector_indices = flat_rows // rows_per_detector
            local_rows = flat_rows % rows_per_detector
            pair_indices = local_rows // source_count
            source_indices = local_rows % source_count
            parts.append(
                self._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                    isotope=isotope,
                    detector_positions=detectors[detector_indices],
                    sources=source_arr[source_indices],
                    fe_indices=pair_fe[pair_indices],
                    pb_indices=pair_pb[pair_indices],
                )
            )
        flat_values = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
        return flat_values.reshape(detector_count, pair_count, source_count)

    def _kernel_values_unshielded_for_detector_source_torch_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        tol: float = 1.0e-6,
    ) -> NDArray[np.float64]:
        """Return matched-row unshielded physical kernels through Torch."""
        if torch is None:
            raise RuntimeError("torch is not available")
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        detectors_t = torch.as_tensor(
            detector_positions,
            device=device,
            dtype=dtype,
        )
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if detectors_t.ndim != 2 or detectors_t.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if sources_t.ndim != 2 or sources_t.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors_t.shape[0] != sources_t.shape[0]:
            raise ValueError(
                "detector_positions and sources must have the same row count."
            )
        if not bool(torch.all(torch.isfinite(detectors_t))) or not bool(
            torch.all(torch.isfinite(sources_t))
        ):
            raise ValueError(
                "detector_positions and sources must contain finite values."
            )
        row_count = int(sources_t.shape[0])
        if row_count == 0:
            return np.zeros(0, dtype=float)

        with torch.no_grad():
            direction = detectors_t - sources_t
            distance = torch.linalg.norm(direction, dim=1)
            tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
            safe_distance = torch.where(
                distance <= tol_t,
                tol_t,
                distance,
            )
            geom = _finite_sphere_geometric_term_torch(
                safe_distance,
                detector_radius_m=self.detector_radius_m,
                tol=tol_t,
            )
            sampled_sources, targets, _ = self._ray_sample_points_torch(
                sources=sources_t,
                detector=detectors_t,
                dist=safe_distance,
                tol=tol,
            )
            tau_obstacle = torch.zeros(
                sampled_sources.shape[:-1],
                device=device,
                dtype=dtype,
            )
            obstacle_path_cm = None
            boxes_np = self.obstacle_boxes_m()
            if boxes_np.size:
                boxes_t = torch.as_tensor(
                    boxes_np,
                    device=device,
                    dtype=dtype,
                )
                obstacle_path_cm = (
                    obstacle_path_lengths_between_points_by_box_cm_torch(
                        source_pos=sampled_sources,
                        target_pos=targets,
                        obstacle_boxes_m=boxes_t,
                        tol=tol,
                    )
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                )

            line_entries = self._line_mu_values(isotope)
            if line_entries:
                weights_t = torch.as_tensor(
                    [entry[0] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                line_tau_obstacle = tau_obstacle.unsqueeze(-1)
                if obstacle_path_cm is not None:
                    candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                        isotope,
                        obstacle_path_cm,
                        line_count=len(line_entries),
                        device=device,
                        dtype=dtype,
                    )
                    if candidate_tau is not None:
                        line_tau_obstacle = candidate_tau
                zeros = torch.zeros_like(line_tau_obstacle)
                line_buildup = self._buildup_factor_torch(
                    zeros,
                    zeros,
                    line_tau_obstacle,
                )
                attenuation = torch.sum(
                    torch.exp(-line_tau_obstacle)
                    * line_buildup
                    * weights_t.view(1, 1, -1),
                    dim=-1,
                )
            else:
                zeros = torch.zeros_like(tau_obstacle)
                buildup = self._buildup_factor_torch(
                    zeros,
                    zeros,
                    tau_obstacle,
                )
                attenuation = torch.exp(-tau_obstacle) * buildup
            attenuation = torch.clamp(attenuation, min=0.0, max=1.0)
            values = geom * torch.mean(attenuation, dim=-1)
            if not bool(torch.all(torch.isfinite(values))) or bool(
                torch.any(values < 0.0)
            ):
                raise RuntimeError(
                    "Unshielded physical Torch kernel produced an invalid response."
                )
        return values.detach().cpu().numpy().astype(float, copy=False)

    def _kernel_values_selected_pairs_for_detector_source_torch_chunk(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        tol: float = 1e-6,
        positive_line_indices: object | None = None,
        return_line_transport_components: bool = False,
    ) -> NDArray[np.float64] | LineTransportComponents:
        """Return selected Fe/Pb-pair kernels for matched detector-source rows."""
        if torch is None:
            raise RuntimeError("torch is not available")
        line_selection = (
            None
            if positive_line_indices is None
            else self._validated_positive_line_indices(
                isotope,
                positive_line_indices,
            )
        )
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        detectors_t = torch.as_tensor(detector_positions, device=device, dtype=dtype)
        sources_t = torch.as_tensor(sources, device=device, dtype=dtype)
        if detectors_t.ndim != 2 or detectors_t.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (N, 3).")
        if sources_t.ndim != 2 or sources_t.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if detectors_t.shape[0] != sources_t.shape[0]:
            raise ValueError(
                "detector_positions and sources must have the same row count."
            )
        row_count = int(sources_t.shape[0])
        if row_count == 0:
            if line_selection is None:
                return np.zeros(0, dtype=float)
            if return_line_transport_components:
                empty = np.zeros(
                    (0, int(line_selection.size)),
                    dtype=np.float64,
                )
                return LineTransportComponents(
                    total_kernel=empty,
                    unattenuated_kernel=empty.copy(),
                    uncollided_kernel=empty.copy(),
                    tau_fe=empty.copy(),
                    tau_pb=empty.copy(),
                    tau_obstacle=empty.copy(),
                    tau_obstacle_compton=empty.copy(),
                    distance_m=empty.copy(),
                )
            return np.zeros((0, line_selection.size), dtype=float)
        if return_line_transport_components and line_selection is None:
            raise ValueError(
                "Line transport components require positive_line_indices."
            )
        num_orients = int(len(self.orientations))
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=num_orients,
            expected_count=row_count,
        )
        unique_orients = np.unique(np.concatenate([fe_arr, pb_arr]))
        orient_to_row = {int(orient): idx for idx, orient in enumerate(unique_orients)}
        fe_select = torch.as_tensor(
            [orient_to_row[int(orient)] for orient in fe_arr],
            device=device,
            dtype=torch.long,
        )
        pb_select = torch.as_tensor(
            [orient_to_row[int(orient)] for orient in pb_arr],
            device=device,
            dtype=torch.long,
        )
        row_ids = torch.arange(row_count, device=device, dtype=torch.long)
        with torch.no_grad():
            direction = detectors_t - sources_t
            dist = torch.linalg.norm(direction, dim=1)
            tol_t = torch.as_tensor(tol, device=device, dtype=dtype)
            dist = torch.where(dist <= tol_t, tol_t, dist)
            geom = _finite_sphere_geometric_term_torch(
                dist,
                detector_radius_m=self.detector_radius_m,
                tol=tol_t,
            )
            sampled_sources, targets, sample_count = self._ray_sample_points_torch(
                sources=sources_t,
                detector=detectors_t,
                dist=dist,
                tol=tol,
            )
            sampled_direction = targets - sampled_sources
            sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
            sampled_dist = torch.where(sampled_dist <= tol_t, tol_t, sampled_dist)
            response_distance = torch.linalg.norm(
                sampled_sources - detectors_t.unsqueeze(-2),
                dim=-1,
            )

            l_fe_by_orient: list[torch.Tensor] = []
            l_pb_by_orient: list[torch.Tensor] = []
            for orient_idx in unique_orients:
                orient_int = int(orient_idx)
                shield_normal = -np.asarray(
                    self.orientations[orient_int],
                    dtype=float,
                )
                rotation = self._rotated_octant_rotation_torch(
                    shield_normal,
                    device=device,
                    dtype=dtype,
                )
                center = detectors_t.unsqueeze(-2)
                l_fe = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_fe_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_fe_cm
                        + self.shield_params.thickness_fe_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
                l_pb = segment_rotated_octant_shell_path_length_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    center_pos=center,
                    shield_normal=shield_normal,
                    inner_radius_cm=self.shield_params.inner_radius_pb_cm,
                    outer_radius_cm=(
                        self.shield_params.inner_radius_pb_cm
                        + self.shield_params.thickness_pb_cm
                    ),
                    tol=tol,
                    rotation=rotation,
                )
                l_fe_by_orient.append(l_fe)
                l_pb_by_orient.append(l_pb)

            l_fe_stack = torch.stack(l_fe_by_orient, dim=0)
            l_pb_stack = torch.stack(l_pb_by_orient, dim=0)
            l_fe_pairs = l_fe_stack[fe_select, row_ids, :]
            l_pb_pairs = l_pb_stack[pb_select, row_ids, :]

            mu_fe, mu_pb = self._mu_values(isotope=isotope)
            tau_fe = float(mu_fe) * l_fe_pairs
            tau_pb = float(mu_pb) * l_pb_pairs
            tau_obstacle = torch.zeros_like(tau_fe)
            obstacle_path_cm = None
            boxes_np = self.obstacle_boxes_m()
            if boxes_np.size:
                boxes_t = torch.as_tensor(boxes_np, device=device, dtype=dtype)
                obstacle_path_cm = obstacle_path_lengths_between_points_by_box_cm_torch(
                    source_pos=sampled_sources,
                    target_pos=targets,
                    obstacle_boxes_m=boxes_t,
                    tol=tol,
                )
                tau_obstacle = self._obstacle_tau_from_lengths_torch(
                    isotope,
                    obstacle_path_cm,
                    device=device,
                    dtype=dtype,
                )
            line_entries = self._line_mu_values(isotope)
            if line_entries:
                weights_t = torch.as_tensor(
                    [entry[0] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_fe_t = torch.as_tensor(
                    [entry[1] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                mu_pb_t = torch.as_tensor(
                    [entry[2] for entry in line_entries],
                    device=device,
                    dtype=dtype,
                )
                line_tau_fe = l_fe_pairs.unsqueeze(-1) * mu_fe_t.view(1, 1, -1)
                line_tau_pb = l_pb_pairs.unsqueeze(-1) * mu_pb_t.view(1, 1, -1)
                line_tau_obstacle = torch.broadcast_to(
                    tau_obstacle.unsqueeze(-1),
                    line_tau_fe.shape,
                )
                line_tau_obstacle_compton = torch.zeros_like(line_tau_fe)
                if obstacle_path_cm is not None:
                    candidate_tau = self._obstacle_line_tau_from_lengths_torch(
                        isotope,
                        obstacle_path_cm,
                        line_count=len(line_entries),
                        device=device,
                        dtype=dtype,
                    )
                    if candidate_tau is not None:
                        line_tau_obstacle = candidate_tau
                    candidate_compton_tau = (
                        self
                        ._obstacle_line_compton_tau_from_lengths_torch(
                            isotope,
                            obstacle_path_cm,
                            line_count=len(line_entries),
                            device=device,
                            dtype=dtype,
                        )
                    )
                    if candidate_compton_tau is not None:
                        line_tau_obstacle_compton = candidate_compton_tau
                line_total_tau = line_tau_fe + line_tau_pb + line_tau_obstacle
                uncollided_line_att = torch.exp(-line_total_tau)
                if self.additive_scatter_response is not None:
                    line_energies = self._line_energy_values_keV(isotope)
                    if len(line_energies) != len(line_entries):
                        raise RuntimeError(
                            "Additive scatter requires exact positive-line "
                            "energies."
                        )
                    scatter_basis = physical_scatter_basis_torch(
                        tau_fe=line_tau_fe,
                        tau_pb=line_tau_pb,
                        tau_obstacle=line_tau_obstacle,
                        tau_obstacle_compton=(
                            line_tau_obstacle_compton
                        ),
                        distance_m=torch.broadcast_to(
                            response_distance.unsqueeze(-1),
                            line_total_tau.shape,
                        ),
                        energy_keV=torch.as_tensor(
                            line_energies,
                            device=device,
                            dtype=dtype,
                        ).view(1, 1, -1),
                        mu_fe_cm_inv=mu_fe_t.view(1, 1, -1),
                        mu_pb_cm_inv=mu_pb_t.view(1, 1, -1),
                    )
                    line_base_att = (
                        self.additive_scatter_response.total_kernel_torch(
                            torch.ones_like(uncollided_line_att),
                            uncollided_line_att,
                            scatter_basis,
                        )
                    )
                else:
                    line_buildup = self._buildup_factor_torch(
                        line_tau_fe,
                        line_tau_pb,
                        line_tau_obstacle,
                    )
                    line_base_att = (
                        uncollided_line_att * line_buildup
                    )
                base_att = torch.sum(
                    line_base_att * weights_t.view(1, 1, -1),
                    dim=-1,
                )
                att = base_att
            else:
                total_tau = tau_fe + tau_pb + tau_obstacle
                buildup = self._buildup_factor_torch(tau_fe, tau_pb, tau_obstacle)
                att = torch.exp(-total_tau) * buildup
            if line_selection is not None:
                if not line_entries:
                    raise RuntimeError(
                        "Validated positive-line selection lacks line entries."
                    )
                if self.additive_scatter_response is None:
                    capped_base = torch.clamp(
                        base_att,
                        min=0.0,
                        max=1.0,
                    )
                    aggregate_scale = torch.where(
                        base_att > 0.0,
                        capped_base / base_att,
                        torch.zeros_like(base_att),
                    )
                else:
                    aggregate_scale = torch.ones_like(base_att)
                selected_attenuation = line_base_att.index_select(
                    -1,
                    torch.as_tensor(
                        line_selection,
                        device=device,
                        dtype=torch.long,
                    ),
                ) * aggregate_scale.unsqueeze(-1)
                if return_line_transport_components:
                    selected = torch.as_tensor(
                        line_selection,
                        device=device,
                        dtype=torch.long,
                    )
                    uncollided_attenuation = torch.exp(
                        -line_total_tau.index_select(-1, selected)
                    )
                    total_kernel = geom.unsqueeze(-1) * torch.mean(
                        selected_attenuation,
                        dim=1,
                    )
                    uncollided_kernel = geom.unsqueeze(-1) * torch.mean(
                        uncollided_attenuation,
                        dim=1,
                    )
                    unattenuated_kernel = torch.broadcast_to(
                        geom.unsqueeze(-1),
                        total_kernel.shape,
                    )
                    direct_attenuation = torch.minimum(
                        selected_attenuation,
                        uncollided_attenuation,
                    )
                    scatter_weights = torch.clamp(
                        selected_attenuation - direct_attenuation,
                        min=0.0,
                    )
                    fallback_weights = torch.clamp(
                        uncollided_attenuation,
                        min=0.0,
                    )
                    scatter_weight_sum = torch.sum(
                        scatter_weights,
                        dim=1,
                        keepdim=True,
                    )
                    feature_weights = torch.where(
                        scatter_weight_sum
                        > torch.finfo(dtype).tiny,
                        scatter_weights,
                        fallback_weights,
                    )
                    feature_weight_sum = torch.clamp(
                        torch.sum(feature_weights, dim=1),
                        min=torch.finfo(dtype).tiny,
                    )

                    def _weighted_feature(
                        values: "torch.Tensor",
                    ) -> "torch.Tensor":
                        """Average one ray feature by spectral contribution."""
                        return (
                            torch.sum(values * feature_weights, dim=1)
                            / feature_weight_sum
                        )

                    def _as_numpy(value: "torch.Tensor") -> NDArray[np.float64]:
                        """Return one detached float64 component array."""
                        return (
                            value.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64, copy=False)
                        )

                    return LineTransportComponents(
                        total_kernel=_as_numpy(total_kernel),
                        unattenuated_kernel=_as_numpy(
                            unattenuated_kernel
                        ),
                        uncollided_kernel=_as_numpy(
                            uncollided_kernel
                        ),
                        tau_fe=_as_numpy(
                            _weighted_feature(
                                line_tau_fe.index_select(-1, selected)
                            )
                        ),
                        tau_pb=_as_numpy(
                            _weighted_feature(
                                line_tau_pb.index_select(-1, selected)
                            )
                        ),
                        tau_obstacle=_as_numpy(
                            _weighted_feature(
                                line_tau_obstacle.index_select(
                                    -1,
                                    selected,
                                )
                            )
                        ),
                        tau_obstacle_compton=_as_numpy(
                            _weighted_feature(
                                line_tau_obstacle_compton.index_select(
                                    -1,
                                    selected,
                                )
                            )
                        ),
                        distance_m=_as_numpy(
                            _weighted_feature(
                                torch.broadcast_to(
                                    response_distance.unsqueeze(-1),
                                    selected_attenuation.shape,
                                )
                            )
                        ),
                    )
                selected_attenuation = torch.mean(
                    selected_attenuation,
                    dim=1,
                )
                selected_values = geom.unsqueeze(-1) * selected_attenuation
                return (
                    selected_values.detach()
                    .cpu()
                    .numpy()
                    .astype(float, copy=False)
                )
            att = torch.mean(att, dim=-1)
            values = geom * att
        return values.detach().cpu().numpy().astype(float, copy=False)

    def kernel_values_unshielded_for_detector_source_pairs(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate matched detector/source rows without shield attenuation."""
        detectors = np.asarray(detector_positions, dtype=np.float64)
        source_rows = np.asarray(sources, dtype=np.float64)
        if (
            detectors.ndim != 2
            or detectors.shape[1] != 3
            or source_rows.shape != detectors.shape
            or np.any(~np.isfinite(detectors))
            or np.any(~np.isfinite(source_rows))
        ):
            raise ValueError(
                "Matched detector/source inputs must be finite (N, 3) arrays."
            )
        row_count = int(detectors.shape[0])
        if row_count == 0:
            return np.zeros(0, dtype=np.float64)
        if not self.use_gpu:
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            return np.concatenate(
                [
                    self._kernel_values_unshielded_for_detector_source_numpy_chunk(
                        isotope=isotope,
                        detector_positions=detectors[start:stop],
                        sources=source_rows[start:stop],
                    )
                    for start in range(0, row_count, chunk)
                    for stop in [min(start + chunk, row_count)]
                ]
            )
        self._gpu_enabled()
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=1,
            device=device,
            dtype=dtype,
            all_orientation_pairs=False,
        )

        def _evaluate(start: int, stop: int) -> NDArray[np.float64]:
            """Evaluate one matched unshielded Torch chunk."""
            return self._kernel_values_unshielded_for_detector_source_torch_chunk(
                isotope=isotope,
                detector_positions=detectors[start:stop],
                sources=source_rows[start:stop],
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=row_count,
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate,
        )
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def kernel_values_selected_pairs_for_detector_source_pairs(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate one selected shield pair for each matched detector/source row."""
        detectors = np.asarray(detector_positions, dtype=np.float64)
        source_rows = np.asarray(sources, dtype=np.float64)
        if (
            detectors.ndim != 2
            or detectors.shape[1] != 3
            or source_rows.shape != detectors.shape
            or np.any(~np.isfinite(detectors))
            or np.any(~np.isfinite(source_rows))
        ):
            raise ValueError(
                "Matched detector/source inputs must be finite (N, 3) arrays."
            )
        row_count = int(detectors.shape[0])
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=int(len(self.orientations)),
            expected_count=row_count,
        )
        if row_count == 0:
            return np.zeros(0, dtype=np.float64)
        if not self.use_gpu:
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            return np.concatenate(
                [
                    self
                    ._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                        isotope=isotope,
                        detector_positions=detectors[start:stop],
                        sources=source_rows[start:stop],
                        fe_indices=fe_arr[start:stop],
                        pb_indices=pb_arr[start:stop],
                    )
                    for start in range(0, row_count, chunk)
                    for stop in [min(start + chunk, row_count)]
                ]
            )
        self._gpu_enabled()
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=1,
            device=device,
            dtype=dtype,
            all_orientation_pairs=False,
        )

        def _evaluate(start: int, stop: int) -> NDArray[np.float64]:
            """Evaluate one matched selected-pair Torch chunk."""
            return (
                self
                ._kernel_values_selected_pairs_for_detector_source_torch_chunk(
                    isotope=isotope,
                    detector_positions=detectors[start:stop],
                    sources=source_rows[start:stop],
                    fe_indices=fe_arr[start:stop],
                    pb_indices=pb_arr[start:stop],
                )
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=row_count,
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate,
        )
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)

    def kernel_values_unshielded_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate physical distance-plus-obstacle response without shields.

        The result is shaped ``(detector_count, source_count)``. It
        intentionally excludes Fe/Pb attenuation and pair-specific transport
        calibration, so it is a PF-independent physical observability kernel
        rather than an observation-likelihood shortcut.
        """
        detectors_arr = np.asarray(detector_positions, dtype=float)
        sources_arr = np.asarray(sources, dtype=float)
        if detectors_arr.ndim != 2 or detectors_arr.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        if not np.all(np.isfinite(detectors_arr)):
            raise ValueError("detector_positions must contain finite values.")
        if not np.all(np.isfinite(sources_arr)):
            raise ValueError("sources must contain finite values.")
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        if pose_count == 0 or source_count == 0:
            return np.zeros((pose_count, source_count), dtype=float)

        total_rows = pose_count * source_count
        if not self.use_gpu:
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            parts: list[NDArray[np.float64]] = []
            for start in range(0, total_rows, chunk):
                stop = min(start + chunk, total_rows)
                flat_rows = np.arange(start, stop, dtype=np.int64)
                detector_indices = flat_rows // source_count
                source_indices = flat_rows % source_count
                parts.append(
                    self._kernel_values_unshielded_for_detector_source_numpy_chunk(
                        isotope=isotope,
                        detector_positions=detectors_arr[detector_indices],
                        sources=sources_arr[source_indices],
                    )
                )
            flat_values = (
                np.concatenate(parts) if parts else np.zeros(0, dtype=float)
            )
            return flat_values.reshape(pose_count, source_count)

        self._gpu_enabled()
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=1,
            device=device,
            dtype=dtype,
            all_orientation_pairs=False,
        )

        def _evaluate_unshielded_detector_source_chunk(
            start: int,
            stop: int,
        ) -> NDArray[np.float64]:
            """Return unshielded kernels for one flat detector/source chunk."""
            flat_rows = np.arange(start, stop, dtype=np.int64)
            detector_indices = flat_rows // source_count
            source_indices = flat_rows % source_count
            return self._kernel_values_unshielded_for_detector_source_torch_chunk(
                isotope=isotope,
                detector_positions=detectors_arr[detector_indices],
                sources=sources_arr[source_indices],
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=total_rows,
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate_unshielded_detector_source_chunk,
        )
        flat_values = (
            np.concatenate(parts) if parts else np.zeros(0, dtype=float)
        )
        return flat_values.reshape(pose_count, source_count)

    def kernel_values_all_pairs(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 8192,
    ) -> NDArray[np.float64]:
        """Evaluate K values for every Fe/Pb orientation pair and source."""
        sources_arr = np.asarray(sources, dtype=float)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if sources_arr.size == 0:
            return np.zeros((num_pairs, 0), dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if not self.use_gpu:
            detector = np.asarray(detector_pos, dtype=float)
            if detector.shape != (3,):
                raise ValueError("detector_pos must be shaped (3,).")
            return self._kernel_values_all_pairs_for_detectors_numpy(
                isotope,
                detector.reshape(1, 3),
                sources_arr,
                chunk_size=chunk_size,
            )[0]
        self._gpu_enabled()
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=num_pairs,
            device=device,
            dtype=dtype,
            all_orientation_pairs=True,
        )

        def _evaluate_source_chunk(
            start: int,
            stop: int,
        ) -> NDArray[np.float64]:
            """Return every shield-pair kernel for one source chunk."""
            return self._kernel_values_all_pairs_torch_chunk(
                isotope=isotope,
                detector_pos=detector_pos,
                sources=sources_arr[start:stop],
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=int(sources_arr.shape[0]),
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate_source_chunk,
        )
        if not parts:
            return np.zeros((num_pairs, 0), dtype=float)
        return np.concatenate(parts, axis=1)

    def kernel_values_all_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate all Fe/Pb pair kernels for many detectors and sources."""
        detectors_arr = np.asarray(detector_positions, dtype=float)
        sources_arr = np.asarray(sources, dtype=float)
        num_orients = int(len(self.orientations))
        num_pairs = num_orients * num_orients
        if detectors_arr.ndim != 2 or detectors_arr.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        if pose_count == 0 or source_count == 0:
            return np.zeros((pose_count, num_pairs, source_count), dtype=float)
        if not self.use_gpu:
            return self._kernel_values_all_pairs_for_detectors_numpy(
                isotope,
                detectors_arr,
                sources_arr,
                chunk_size=chunk_size,
            )
        self._gpu_enabled()
        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (pose_count, 1))
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=num_pairs,
            device=device,
            dtype=dtype,
            all_orientation_pairs=True,
        )

        def _evaluate_detector_source_chunk(
            start: int,
            stop: int,
        ) -> NDArray[np.float64]:
            """Return all shield-pair kernels for matched rows in one chunk."""
            return self._kernel_values_all_pairs_for_detector_source_torch_chunk(
                isotope=isotope,
                detector_positions=detectors_flat[start:stop],
                sources=sources_flat[start:stop],
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=int(sources_flat.shape[0]),
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate_detector_source_chunk,
        )
        if not parts:
            return np.zeros((pose_count, num_pairs, source_count), dtype=float)
        flat_values = np.concatenate(parts, axis=0)
        return flat_values.reshape(pose_count, source_count, num_pairs).transpose(
            0,
            2,
            1,
        )

    def kernel_values_selected_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Evaluate one selected Fe/Pb pair per detector for many sources."""
        detectors_arr = np.asarray(detector_positions, dtype=float)
        sources_arr = np.asarray(sources, dtype=float)
        if detectors_arr.ndim != 2 or detectors_arr.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (P, 3).")
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (S, 3).")
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=int(len(self.orientations)),
            expected_count=pose_count,
        )
        if pose_count == 0 or source_count == 0:
            return np.zeros((pose_count, source_count), dtype=float)
        if not self.use_gpu:
            detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
            sources_flat = np.tile(sources_arr, (pose_count, 1))
            fe_flat = np.repeat(fe_arr, source_count)
            pb_flat = np.repeat(pb_arr, source_count)
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            parts = [
                self._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                    isotope=isotope,
                    detector_positions=detectors_flat[start:stop],
                    sources=sources_flat[start:stop],
                    fe_indices=fe_flat[start:stop],
                    pb_indices=pb_flat[start:stop],
                )
                for start in range(0, sources_flat.shape[0], chunk)
                for stop in [min(start + chunk, sources_flat.shape[0])]
            ]
            if not parts:
                return np.zeros((pose_count, source_count), dtype=float)
            return np.concatenate(parts).reshape(pose_count, source_count)
        self._gpu_enabled()
        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (pose_count, 1))
        fe_flat = np.repeat(fe_arr, source_count)
        pb_flat = np.repeat(pb_arr, source_count)
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=1,
            device=device,
            dtype=dtype,
            all_orientation_pairs=False,
        )

        def _evaluate_selected_detector_source_chunk(
            start: int,
            stop: int,
        ) -> NDArray[np.float64]:
            """Return selected-pair kernels for matched rows in one chunk."""
            return self._kernel_values_selected_pairs_for_detector_source_torch_chunk(
                isotope=isotope,
                detector_positions=detectors_flat[start:stop],
                sources=sources_flat[start:stop],
                fe_indices=fe_flat[start:stop],
                pb_indices=pb_flat[start:stop],
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=int(sources_flat.shape[0]),
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate_selected_detector_source_chunk,
        )
        if not parts:
            return np.zeros((pose_count, source_count), dtype=float)
        return np.concatenate(parts).reshape(pose_count, source_count)

    def kernel_values_selected_pairs_for_detectors_by_line(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        positive_line_indices: object,
        chunk_size: int = 262144,
    ) -> NDArray[np.float64]:
        """Return view-by-source-by-line source-equivalent kernel means.

        ``positive_line_indices`` addresses the isotope library after removing
        nonpositive-intensity lines, matching the spectrum identification
        contract.  The returned last axis preserves the requested index order.
        Values are evaluated before multiplying by branching weights.

        Each line shares the aggregate transport-response correction and, when
        buildup makes the raw mixture exceed unity, the aggregate transmission
        cap.  Consequently multiplying the complete line axis by
        :meth:`line_branching_weights` and summing reproduces the aggregate
        kernel exactly while retaining genuinely line-specific Fe/Pb and
        obstacle attenuation.
        """
        line_selection = self._validated_positive_line_indices(
            isotope,
            positive_line_indices,
        )
        detectors_arr = np.asarray(detector_positions, dtype=np.float64)
        sources_arr = np.asarray(sources, dtype=np.float64)
        if (
            detectors_arr.ndim != 2
            or detectors_arr.shape[1] != 3
            or np.any(~np.isfinite(detectors_arr))
        ):
            raise ValueError(
                "detector_positions must contain finite values with shape (P, 3)."
            )
        if (
            sources_arr.ndim != 2
            or sources_arr.shape[1] != 3
            or np.any(~np.isfinite(sources_arr))
        ):
            raise ValueError(
                "sources must contain finite values with shape (S, 3)."
        )
        pose_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        orientation_count = int(len(self.orientations))
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=orientation_count,
            expected_count=pose_count,
        )
        output_shape = (
            pose_count,
            source_count,
            int(line_selection.size),
        )
        if pose_count == 0 or source_count == 0:
            return np.zeros(output_shape, dtype=np.float64)

        cache_key = (
            str(isotope),
            tuple(int(value) for value in line_selection),
            bool(self.use_gpu),
            str(self.gpu_device),
            str(self.gpu_dtype),
            self._line_response_array_key(detectors_arr),
            self._line_response_array_key(sources_arr),
            self._line_response_array_key(fe_arr),
            self._line_response_array_key(pb_arr),
        )
        cached = self._line_response_cache.get(cache_key)
        if cached is not None:
            self.line_response_cache_hits += 1
            return cached.copy()
        self.line_response_cache_misses += 1

        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (pose_count, 1))
        fe_flat = np.repeat(fe_arr, source_count)
        pb_flat = np.repeat(pb_arr, source_count)
        if not self.use_gpu:
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            parts = [
                self._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                    isotope=isotope,
                    detector_positions=detectors_flat[start:stop],
                    sources=sources_flat[start:stop],
                    fe_indices=fe_flat[start:stop],
                    pb_indices=pb_flat[start:stop],
                    positive_line_indices=line_selection,
                )
                for start in range(0, sources_flat.shape[0], chunk)
                for stop in [min(start + chunk, sources_flat.shape[0])]
            ]
        else:
            self._gpu_enabled()
            device = _resolve_device(self.gpu_device)
            dtype = _resolve_dtype(self.gpu_dtype)
            chunk = self._adaptive_torch_chunk_size(
                chunk_size,
                isotope=isotope,
                orientation_pair_count=1,
                device=device,
                dtype=dtype,
                all_orientation_pairs=False,
            )

            def _evaluate_line_chunk(
                start: int,
                stop: int,
            ) -> NDArray[np.float64]:
                """Return matched-row line kernels for one Torch chunk."""
                return (
                    self
                    ._kernel_values_selected_pairs_for_detector_source_torch_chunk(
                        isotope=isotope,
                        detector_positions=detectors_flat[start:stop],
                        sources=sources_flat[start:stop],
                        fe_indices=fe_flat[start:stop],
                        pb_indices=pb_flat[start:stop],
                        positive_line_indices=line_selection,
                    )
                )

            parts, _ = self._evaluate_torch_chunks_with_oom_retry(
                total_size=int(sources_flat.shape[0]),
                initial_chunk=chunk,
                device=device,
                evaluator=_evaluate_line_chunk,
            )
        flat = (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros(
                (0, int(line_selection.size)),
                dtype=np.float64,
            )
        )
        result = np.asarray(flat, dtype=np.float64).reshape(output_shape)
        cached_result = result.copy()
        cached_result.setflags(write=False)
        self._line_response_cache[cache_key] = cached_result
        while (
            len(self._line_response_cache) > _LINE_RESPONSE_CACHE_MAX_ENTRIES
            or sum(
                int(values.nbytes)
                for values in self._line_response_cache.values()
            )
            > _LINE_RESPONSE_CACHE_MAX_BYTES
        ):
            self._line_response_cache.pop(next(iter(self._line_response_cache)))
        return result

    def line_transport_components_selected_pairs_for_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        positive_line_indices: object,
        chunk_size: int = 262144,
    ) -> LineTransportComponents:
        """Return view-by-source-by-line spectral transport components.

        The total kernel is bit-identical to
        :meth:`kernel_values_selected_pairs_for_detectors_by_line`.  The
        unattenuated term contains geometry only, the uncollided term contains
        exact material attenuation, and total adds the authenticated
        nonnegative scatter response.  The remaining arrays expose physical
        optical depths and source-detector distance.  Computation follows the
        same batched CPU/GPU ray path as the production scalar kernel.
        """
        line_selection = self._validated_positive_line_indices(
            isotope,
            positive_line_indices,
        )
        detectors_arr = np.asarray(detector_positions, dtype=np.float64)
        sources_arr = np.asarray(sources, dtype=np.float64)
        if (
            detectors_arr.ndim != 2
            or detectors_arr.shape[1] != 3
            or np.any(~np.isfinite(detectors_arr))
        ):
            raise ValueError(
                "detector_positions must contain finite values with shape (P, 3)."
            )
        if (
            sources_arr.ndim != 2
            or sources_arr.shape[1] != 3
            or np.any(~np.isfinite(sources_arr))
        ):
            raise ValueError(
                "sources must contain finite values with shape (S, 3)."
            )
        view_count = int(detectors_arr.shape[0])
        source_count = int(sources_arr.shape[0])
        orientation_count = int(len(self.orientations))
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=orientation_count,
            expected_count=view_count,
        )
        output_shape = (
            view_count,
            source_count,
            int(line_selection.size),
        )
        if view_count == 0 or source_count == 0:
            empty = np.zeros(output_shape, dtype=np.float64)
            return LineTransportComponents(
                total_kernel=empty,
                unattenuated_kernel=empty.copy(),
                uncollided_kernel=empty.copy(),
                tau_fe=empty.copy(),
                tau_pb=empty.copy(),
                tau_obstacle=empty.copy(),
                tau_obstacle_compton=empty.copy(),
                distance_m=empty.copy(),
            )
        detectors_flat = np.repeat(detectors_arr, source_count, axis=0)
        sources_flat = np.tile(sources_arr, (view_count, 1))
        fe_flat = np.repeat(fe_arr, source_count)
        pb_flat = np.repeat(pb_arr, source_count)
        if not self.use_gpu:
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            parts = [
                self._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                    isotope=isotope,
                    detector_positions=detectors_flat[start:stop],
                    sources=sources_flat[start:stop],
                    fe_indices=fe_flat[start:stop],
                    pb_indices=pb_flat[start:stop],
                    positive_line_indices=line_selection,
                    return_line_transport_components=True,
                )
                for start in range(0, sources_flat.shape[0], chunk)
                for stop in [min(start + chunk, sources_flat.shape[0])]
            ]
        else:
            self._gpu_enabled()
            device = _resolve_device(self.gpu_device)
            dtype = _resolve_dtype(self.gpu_dtype)
            chunk = self._adaptive_torch_chunk_size(
                chunk_size,
                isotope=isotope,
                orientation_pair_count=1,
                device=device,
                dtype=dtype,
                all_orientation_pairs=False,
            )

            def _evaluate_component_chunk(
                start: int,
                stop: int,
            ) -> LineTransportComponents:
                """Return physical line components for one Torch chunk."""
                result = (
                    self
                    ._kernel_values_selected_pairs_for_detector_source_torch_chunk(
                        isotope=isotope,
                        detector_positions=detectors_flat[start:stop],
                        sources=sources_flat[start:stop],
                        fe_indices=fe_flat[start:stop],
                        pb_indices=pb_flat[start:stop],
                        positive_line_indices=line_selection,
                        return_line_transport_components=True,
                    )
                )
                if not isinstance(result, LineTransportComponents):
                    raise RuntimeError(
                        "Torch line-component path returned scalar kernels."
                    )
                return result

            parts, _ = self._evaluate_torch_chunks_with_oom_retry(
                total_size=int(sources_flat.shape[0]),
                initial_chunk=chunk,
                device=device,
                evaluator=_evaluate_component_chunk,
            )
        if any(not isinstance(part, LineTransportComponents) for part in parts):
            raise RuntimeError(
                "Line-component chunk evaluation returned scalar kernels."
            )

        def _joined(field_name: str) -> NDArray[np.float64]:
            """Concatenate and reshape one component field."""
            values = [
                np.asarray(
                    getattr(part, field_name),
                    dtype=np.float64,
                )
                for part in parts
                if isinstance(part, LineTransportComponents)
            ]
            flat = (
                np.concatenate(values, axis=0)
                if values
                else np.zeros(
                    (0, int(line_selection.size)),
                    dtype=np.float64,
                )
            )
            return flat.reshape(output_shape)

        return LineTransportComponents(
            total_kernel=_joined("total_kernel"),
            unattenuated_kernel=_joined("unattenuated_kernel"),
            uncollided_kernel=_joined("uncollided_kernel"),
            tau_fe=_joined("tau_fe"),
            tau_pb=_joined("tau_pb"),
            tau_obstacle=_joined("tau_obstacle"),
            tau_obstacle_compton=_joined("tau_obstacle_compton"),
            distance_m=_joined("distance_m"),
        )

    def _kernel_values_pair_scalar_oracle(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        sources: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
    ) -> NDArray[np.float64]:
        """Return the scalar CPU kernel oracle for tests and explicit debugging."""
        sources_arr = np.asarray(sources, dtype=float)
        if sources_arr.size == 0:
            return np.zeros(0, dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
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
        fe_arr, pb_arr = validate_orientation_pair_indices(
            np.asarray([fe_index]),
            np.asarray([pb_index]),
            orientation_count=int(len(self.orientations)),
            expected_count=1,
        )
        fe_index = int(fe_arr[0])
        pb_index = int(pb_arr[0])
        sources_arr = np.asarray(sources, dtype=float)
        if sources_arr.size == 0:
            return np.zeros(0, dtype=float)
        if sources_arr.ndim != 2 or sources_arr.shape[1] != 3:
            raise ValueError("sources must be shaped (N, 3).")
        if not self.use_gpu:
            detector = np.asarray(detector_pos, dtype=float)
            if detector.shape != (3,):
                raise ValueError("detector_pos must be shaped (3,).")
            chunk = self._adaptive_numpy_chunk_size(
                chunk_size,
                isotope=isotope,
            )
            parts = [
                self._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                    isotope=isotope,
                    detector_positions=np.broadcast_to(
                        detector,
                        (stop - start, 3),
                    ),
                    sources=sources_arr[start:stop],
                    fe_indices=np.full(stop - start, int(fe_index), dtype=int),
                    pb_indices=np.full(stop - start, int(pb_index), dtype=int),
                )
                for start in range(0, sources_arr.shape[0], chunk)
                for stop in [min(start + chunk, sources_arr.shape[0])]
            ]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=float)
        self._gpu_enabled()
        device = _resolve_device(self.gpu_device)
        dtype = _resolve_dtype(self.gpu_dtype)
        chunk = self._adaptive_torch_chunk_size(
            chunk_size,
            isotope=isotope,
            orientation_pair_count=1,
            device=device,
            dtype=dtype,
            all_orientation_pairs=False,
        )

        def _evaluate_pair_source_chunk(
            start: int,
            stop: int,
        ) -> NDArray[np.float64]:
            """Return one shield-pair kernel for a contiguous source chunk."""
            return self._kernel_values_pair_torch_chunk(
                isotope=isotope,
                detector_pos=detector_pos,
                sources=sources_arr[start:stop],
                fe_index=fe_index,
                pb_index=pb_index,
            )

        parts, _ = self._evaluate_torch_chunks_with_oom_retry(
            total_size=int(sources_arr.shape[0]),
            initial_chunk=chunk,
            device=device,
            evaluator=_evaluate_pair_source_chunk,
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
        fe_arr, pb_arr = validate_orientation_pair_indices(
            np.asarray([fe_index]),
            np.asarray([pb_index]),
            orientation_count=int(len(self.orientations)),
            expected_count=1,
        )
        fe_index = int(fe_arr[0])
        pb_index = int(pb_arr[0])
        sampled_sources, targets = self._ray_sample_points(source_pos, detector_pos)
        values = [
            self._attenuation_factor_for_target(
                isotope=isotope,
                source_pos=sampled_source,
                target_pos=target,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
            )
            for sampled_source, target in zip(sampled_sources, targets)
        ]
        return float(np.mean(values)) if values else 1.0

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
        att = self.attenuation_factor_pair(
            isotope, source_pos, detector_pos, fe_index, pb_index
        )
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
        fe_arr, pb_arr = validate_orientation_pair_indices(
            np.asarray([fe_index]),
            np.asarray([pb_index]),
            orientation_count=int(len(self.orientations)),
            expected_count=1,
        )
        fe_index = int(fe_arr[0])
        pb_index = int(pb_arr[0])
        if not self.use_gpu:
            sources_arr = np.asarray(sources, dtype=float)
            strengths_arr = np.asarray(strengths, dtype=float)
            if sources_arr.size == 0:
                return float(background)
            if strengths_arr.shape != (sources_arr.shape[0],):
                raise ValueError("strengths must contain one value per source.")
            kernel_values = self.kernel_values_pair(
                isotope=isotope,
                detector_pos=detector_pos,
                sources=sources_arr,
                fe_index=fe_index,
                pb_index=pb_index,
            )
            return float(background) + float(np.dot(strengths_arr, kernel_values))
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
        rate = self.expected_rate(
            isotope, detector_pos, sources, strengths, orient_idx, background=background
        )
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

    def expected_counts_pair_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute packed-state pair counts through ContinuousKernel GPU math."""
        fe_arr, pb_arr = validate_orientation_pair_indices(
            np.asarray([fe_index]),
            np.asarray([pb_index]),
            orientation_count=int(len(self.orientations)),
            expected_count=1,
        )
        counts = self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )
        return counts[0]

    def expected_counts_all_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute all Fe/Pb-pair counts through ContinuousKernel GPU math."""
        num_orients = int(len(self.orientations))
        pair_ids = np.arange(num_orients * num_orients, dtype=np.int64)
        fe_indices = pair_ids // num_orients
        pb_indices = pair_ids % num_orients
        return self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )

    def expected_counts_selected_pairs_for_packed_states_torch(
        self,
        *,
        isotope: str,
        detector_pos: NDArray[np.float64],
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        backgrounds: "torch.Tensor",
        mask: "torch.Tensor",
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_time_s: float,
        source_scale: float | NDArray[np.float64] | "torch.Tensor" = 1.0,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Compute selected Fe/Pb-pair counts through ContinuousKernel GPU math."""
        fe_arr, pb_arr = validate_orientation_pair_indices(
            fe_indices,
            pb_indices,
            orientation_count=int(len(self.orientations)),
        )
        return self._expected_counts_selected_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=np.asarray(detector_pos, dtype=float),
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_time_s=float(live_time_s),
            source_scale=source_scale,
            device=device,
            dtype=dtype,
        )

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
        """Return a shield normal from either a vector or rotation matrix."""
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
