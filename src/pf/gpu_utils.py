"""GPU helpers for batched continuous-kernel computations."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    TORCH_AVAILABLE = False

from measurement.shielding import (
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    LOCAL_POSITIVE_OCTANT_CENTER,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    rotation_matrix_between_vectors,
)
from measurement.continuous_kernels import obstacle_path_lengths_cm_torch
from pf.state import IsotopeState


def _finite_sphere_geometric_term_torch(
    distance: "torch.Tensor",
    *,
    detector_radius_m: float,
    tol: float,
) -> "torch.Tensor":
    """Return finite-sphere detector-cps@1m scaling for torch distances."""
    if torch is None:
        raise RuntimeError("torch is not available")
    radius = max(float(detector_radius_m), 0.0)
    dist = torch.clamp(distance, min=float(tol))
    if radius <= 0.0:
        return 1.0 / (dist**2)
    radius_t = torch.as_tensor(radius, device=distance.device, dtype=distance.dtype)
    d_eff = torch.maximum(dist, radius_t)
    ratio = torch.clamp(radius_t / torch.clamp(d_eff, min=float(tol)), max=1.0)
    fraction = 0.5 * (1.0 - torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0)))
    reference = max(1.0, radius)
    ref_ratio = min(radius / reference, 1.0)
    ref_fraction = max(
        0.5 * (1.0 - float(np.sqrt(max(1.0 - ref_ratio * ref_ratio, 0.0)))),
        1.0e-12,
    )
    return fraction / ref_fraction


def _rotated_positive_octant_mask_torch(
    detector_to_source_unit: "torch.Tensor",
    octant_index: int,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
    tol: float,
) -> "torch.Tensor":
    """Return a mask for the physical rotated local +X/+Y/+Z shield octant."""
    normals = generate_octant_orientations()
    physical_normal = -np.asarray(normals[int(octant_index) % len(normals)], dtype=float)
    rotation_np = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        physical_normal,
    )
    rotation = torch.as_tensor(rotation_np, device=device, dtype=dtype)
    local_direction = detector_to_source_unit @ rotation
    return torch.all(local_direction >= -float(tol), dim=-1)


def _segment_sphere_length_m_torch(
    source: "torch.Tensor",
    target: "torch.Tensor",
    center: "torch.Tensor",
    radius_m: float,
    tol: float,
) -> "torch.Tensor":
    """Return batched segment lengths inside a sphere in meters."""
    rel_source = source - center
    direction = target - source
    a = torch.sum(direction * direction, dim=-1)
    segment_length = torch.sqrt(torch.clamp(a, min=tol))
    radius = torch.as_tensor(max(float(radius_m), 0.0), device=source.device, dtype=source.dtype)
    b = 2.0 * torch.sum(rel_source * direction, dim=-1)
    c = torch.sum(rel_source * rel_source, dim=-1) - radius * radius
    rel_target = target - center
    source_inside = c <= 0.0
    target_inside = torch.sum(rel_target * rel_target, dim=-1) <= radius * radius
    discriminant = b * b - 4.0 * a * c
    valid = (radius > 0.0) & (a > tol) & (discriminant >= 0.0)
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
    denom = torch.clamp(2.0 * a, min=tol)
    t0 = (-b - sqrt_disc) / denom
    t1 = (-b + sqrt_disc) / denom
    enter = torch.maximum(torch.minimum(t0, t1), torch.zeros_like(t0))
    exit_ = torch.minimum(torch.maximum(t0, t1), torch.ones_like(t0))
    crossed_length = torch.clamp(exit_ - enter, min=0.0) * segment_length
    contained_length = torch.where(source_inside & target_inside, segment_length, torch.zeros_like(segment_length))
    return torch.where(valid, crossed_length, contained_length)


def _segment_spherical_shell_length_cm_torch(
    source: "torch.Tensor",
    target: "torch.Tensor",
    center: "torch.Tensor",
    inner_radius_cm: float,
    outer_radius_cm: float,
    blocked: "torch.Tensor",
    tol: float,
) -> "torch.Tensor":
    """Return batched path lengths through a spherical shell in centimeters."""
    inner_m = max(0.0, float(inner_radius_cm) / 100.0)
    outer_m = max(inner_m, float(outer_radius_cm) / 100.0)
    if outer_m <= inner_m:
        return torch.zeros_like(blocked, dtype=source.dtype)
    outer_length = _segment_sphere_length_m_torch(source, target, center, outer_m, tol)
    inner_length = _segment_sphere_length_m_torch(source, target, center, inner_m, tol)
    shell_cm = 100.0 * torch.clamp(outer_length - inner_length, min=0.0)
    return torch.where(blocked, shell_cm, torch.zeros_like(shell_cm))


def _segment_rotated_octant_shell_length_cm_torch(
    source: "torch.Tensor",
    target: "torch.Tensor",
    center: "torch.Tensor",
    shield_normal: np.ndarray,
    inner_radius_cm: float,
    outer_radius_cm: float,
    device: "torch.device",
    dtype: "torch.dtype",
    tol: float,
) -> "torch.Tensor":
    """Return exact segment length inside a rotated local +X/+Y/+Z octant shell."""
    rotation_np = rotation_matrix_between_vectors(
        LOCAL_POSITIVE_OCTANT_CENTER,
        np.asarray(shield_normal, dtype=float),
    )
    rotation = torch.as_tensor(rotation_np, device=device, dtype=dtype)
    source_local = (source - center) @ rotation
    target_local = (target - center) @ rotation
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
    active = segment_length > float(tol)
    length_m = torch.sum(
        torch.where(inside_shell & inside_octant, width, torch.zeros_like(width)),
        dim=-1,
    ) * segment_length
    return torch.where(active, 100.0 * length_m, torch.zeros_like(length_m))


def _obstacle_path_lengths_between_points_cm_torch(
    source: "torch.Tensor",
    target: "torch.Tensor",
    obstacle_boxes_m: "torch.Tensor",
    tol: float,
) -> "torch.Tensor":
    """Return path lengths through obstacle boxes for batched source-target segments."""
    if obstacle_boxes_m.numel() == 0:
        return torch.zeros(source.shape[:-1], device=source.device, dtype=source.dtype)
    p0 = source.unsqueeze(-2)
    delta = (target - source).unsqueeze(-2)
    distance = torch.linalg.norm(target - source, dim=-1)
    lower = obstacle_boxes_m[:, :3].to(device=source.device, dtype=source.dtype)
    upper = obstacle_boxes_m[:, 3:].to(device=source.device, dtype=source.dtype)
    tol_t = torch.as_tensor(tol, device=source.device, dtype=source.dtype)
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


def torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(TORCH_AVAILABLE and torch is not None)


def resolve_device(device: str | None) -> "torch.device":
    """Resolve a torch device string with CUDA fallback."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if device is None:
        device = "cuda"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    return torch.device(device)


def resolve_dtype(dtype: str) -> "torch.dtype":
    """Map a dtype string to a torch dtype."""
    if torch is None:
        raise RuntimeError("torch is not available")
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def pack_states(
    states: Iterable[IsotopeState],
    device: "torch.device",
    dtype: "torch.dtype",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Pack IsotopeState list into padded tensors.

    Returns (positions, strengths, backgrounds, mask).
    """
    states_list = list(states)
    num_particles = len(states_list)
    max_r = max((st.num_sources for st in states_list), default=0)
    positions = np.zeros((num_particles, max_r, 3), dtype=float)
    strengths = np.zeros((num_particles, max_r), dtype=float)
    mask = np.zeros((num_particles, max_r), dtype=float)
    backgrounds = np.zeros(num_particles, dtype=float)
    for i, st in enumerate(states_list):
        r = st.num_sources
        if r > 0:
            positions[i, :r] = st.positions
            strengths[i, :r] = st.strengths
            mask[i, :r] = 1.0
        backgrounds[i] = st.background
    pos_t = torch.as_tensor(positions, device=device, dtype=dtype)
    str_t = torch.as_tensor(strengths, device=device, dtype=dtype)
    mask_t = torch.as_tensor(mask, device=device, dtype=dtype)
    bg_t = torch.as_tensor(backgrounds, device=device, dtype=dtype)
    return pos_t, str_t, bg_t, mask_t


def expected_counts_pair_torch(
    detector_pos: np.ndarray,
    positions: "torch.Tensor",
    strengths: "torch.Tensor",
    backgrounds: "torch.Tensor",
    mask: "torch.Tensor",
    fe_index: int,
    pb_index: int,
    mu_fe: float,
    mu_pb: float,
    thickness_fe_cm: float,
    thickness_pb_cm: float,
    live_time_s: float,
    device: "torch.device",
    dtype: "torch.dtype",
    use_angle_attenuation: bool = False,
    source_scale: float = 1.0,
    inner_radius_fe_cm: float = DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    inner_radius_pb_cm: float = DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    shield_geometry_model: str = SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    obstacle_boxes_m: np.ndarray | "torch.Tensor" | None = None,
    obstacle_mu_cm_inv: float = 0.0,
    detector_radius_m: float = 0.0,
    detector_aperture_samples: int = 1,
    buildup_fe_coeff: float = 0.0,
    buildup_pb_coeff: float = 0.0,
    obstacle_buildup_coeff: float = 0.0,
    tol: float = 1e-6,
) -> "torch.Tensor":
    """
    Compute Λ for all particles at a Fe/Pb orientation pair on GPU.

    When use_angle_attenuation is False, the shield path length is treated as a
    constant thickness for blocked rays (no 1/cos(theta) scaling).
    """
    if torch is None:
        raise RuntimeError("torch is not available")
    detector = torch.as_tensor(detector_pos, device=device, dtype=dtype).view(1, 1, 3)
    direction = detector - positions
    dist = torch.linalg.norm(direction, dim=-1)
    dist = torch.where(dist <= tol, torch.full_like(dist, tol), dist)
    if float(detector_radius_m) > 0.0 and int(detector_aperture_samples) > 1:
        sample_count = max(int(detector_aperture_samples), 1)
        axis = direction / dist.unsqueeze(-1)
        helper_z = torch.zeros_like(axis)
        helper_z[..., 2] = 1.0
        helper_y = torch.zeros_like(axis)
        helper_y[..., 1] = 1.0
        helper = torch.where(torch.abs(axis[..., 2:3]) > 0.9, helper_y, helper_z)
        basis_u = torch.linalg.cross(axis, helper, dim=-1)
        basis_u_norm = torch.linalg.norm(basis_u, dim=-1, keepdim=True)
        basis_u = basis_u / torch.clamp(basis_u_norm, min=tol)
        basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
        indices = torch.arange(sample_count, device=device, dtype=dtype)
        if sample_count == 1:
            radii = torch.zeros(1, device=device, dtype=dtype)
        else:
            fractions = torch.clamp((indices - 0.5) / float(sample_count - 1), min=0.0, max=1.0)
            radii = torch.sqrt(fractions)
            radii[0] = 0.0
        max_radius = torch.minimum(
            torch.as_tensor(max(float(detector_radius_m), 0.0), device=device, dtype=dtype),
            0.95 * dist,
        )
        angles = indices * torch.as_tensor(np.pi * (3.0 - np.sqrt(5.0)), device=device, dtype=dtype)
        offsets = (
            max_radius.unsqueeze(-1).unsqueeze(-1)
            * radii.view(*([1] * dist.ndim), sample_count, 1)
            * (
                torch.cos(angles).view(*([1] * dist.ndim), sample_count, 1) * basis_u.unsqueeze(-2)
                + torch.sin(angles).view(*([1] * dist.ndim), sample_count, 1) * basis_v.unsqueeze(-2)
            )
        )
        targets = detector.unsqueeze(-2) + offsets
        sampled_positions = positions.unsqueeze(-2).expand_as(targets)
        sampled_direction = targets - sampled_positions
        sampled_dist = torch.linalg.norm(sampled_direction, dim=-1)
        sampled_dist = torch.where(sampled_dist <= tol, torch.full_like(sampled_dist, tol), sampled_dist)
        dir_unit = sampled_direction / sampled_dist.unsqueeze(-1)
    else:
        sample_count = 1
        targets = detector.expand_as(positions).unsqueeze(-2)
        sampled_positions = positions.unsqueeze(-2)
        sampled_direction = direction.unsqueeze(-2)
        dir_unit = direction / dist.unsqueeze(-1)
        dir_unit = dir_unit.unsqueeze(-2)
    geom = _finite_sphere_geometric_term_torch(
        dist,
        detector_radius_m=detector_radius_m,
        tol=tol,
    )

    detector_to_source_unit = -dir_unit
    blocked_fe = _rotated_positive_octant_mask_torch(
        detector_to_source_unit,
        fe_index,
        device=device,
        dtype=dtype,
        tol=tol,
    )
    blocked_pb = _rotated_positive_octant_mask_torch(
        detector_to_source_unit,
        pb_index,
        device=device,
        dtype=dtype,
        tol=tol,
    )

    normals = torch.as_tensor(generate_octant_orientations(), device=device, dtype=dtype)
    normal_fe = normals[fe_index]
    normal_pb = normals[pb_index]
    cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=-1), 0.0, 1.0)
    cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=-1), 0.0, 1.0)
    thickness_fe = torch.as_tensor(thickness_fe_cm, device=device, dtype=dtype)
    thickness_pb = torch.as_tensor(thickness_pb_cm, device=device, dtype=dtype)
    if shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT and not use_angle_attenuation:
        normal_fe_np = -np.asarray(generate_octant_orientations()[fe_index], dtype=float)
        normal_pb_np = -np.asarray(generate_octant_orientations()[pb_index], dtype=float)
        if sample_count > 1:
            center = detector.expand_as(positions).unsqueeze(-2)
            L_fe = _segment_rotated_octant_shell_length_cm_torch(
                sampled_positions,
                targets,
                center,
                normal_fe_np,
                inner_radius_fe_cm,
                inner_radius_fe_cm + thickness_fe_cm,
                device,
                dtype,
                tol,
            )
            L_pb = _segment_rotated_octant_shell_length_cm_torch(
                sampled_positions,
                targets,
                center,
                normal_pb_np,
                inner_radius_pb_cm,
                inner_radius_pb_cm + thickness_pb_cm,
                device,
                dtype,
                tol,
            )
        else:
            center = detector.expand_as(positions)
            target = detector.expand_as(positions)
            L_fe = _segment_rotated_octant_shell_length_cm_torch(
                positions,
                target,
                center,
                normal_fe_np,
                inner_radius_fe_cm,
                inner_radius_fe_cm + thickness_fe_cm,
                device,
                dtype,
                tol,
            ).unsqueeze(-1)
            L_pb = _segment_rotated_octant_shell_length_cm_torch(
                positions,
                target,
                center,
                normal_pb_np,
                inner_radius_pb_cm,
                inner_radius_pb_cm + thickness_pb_cm,
                device,
                dtype,
                tol,
            ).unsqueeze(-1)
    elif use_angle_attenuation:
        L_fe = torch.where(blocked_fe & (cos_fe > tol), thickness_fe / cos_fe, torch.zeros_like(cos_fe))
        L_pb = torch.where(blocked_pb & (cos_pb > tol), thickness_pb / cos_pb, torch.zeros_like(cos_pb))
    else:
        L_fe = torch.where(blocked_fe, thickness_fe, torch.zeros_like(cos_fe))
        L_pb = torch.where(blocked_pb, thickness_pb, torch.zeros_like(cos_pb))
    tau_fe = float(mu_fe) * L_fe
    tau_pb = float(mu_pb) * L_pb
    tau_obstacle = torch.zeros_like(tau_fe)
    if obstacle_boxes_m is not None and float(obstacle_mu_cm_inv) > 0.0:
        boxes_t = torch.as_tensor(obstacle_boxes_m, device=device, dtype=dtype)
        if boxes_t.numel() > 0:
            if sample_count > 1:
                obstacle_path_cm = _obstacle_path_lengths_between_points_cm_torch(
                    sampled_positions,
                    targets,
                    boxes_t,
                    tol,
                )
            else:
                obstacle_path_cm = obstacle_path_lengths_cm_torch(
                    positions=positions,
                    detector_pos=detector.view(3),
                    obstacle_boxes_m=boxes_t,
                ).unsqueeze(-1)
            tau_obstacle = float(obstacle_mu_cm_inv) * obstacle_path_cm
    buildup = torch.ones_like(tau_fe)
    buildup = buildup + max(float(buildup_fe_coeff), 0.0) * (1.0 - torch.exp(-torch.clamp(tau_fe, min=0.0)))
    buildup = buildup + max(float(buildup_pb_coeff), 0.0) * (1.0 - torch.exp(-torch.clamp(tau_pb, min=0.0)))
    buildup = buildup + max(float(obstacle_buildup_coeff), 0.0) * (
        1.0 - torch.exp(-torch.clamp(tau_obstacle, min=0.0))
    )
    att = torch.clamp(torch.exp(-(tau_fe + tau_pb + tau_obstacle)) * buildup, max=1.0)
    att = torch.mean(att, dim=-1)

    strengths = strengths * mask
    source_scale_t = torch.as_tensor(max(float(source_scale), 0.0), device=device, dtype=dtype)
    rate = source_scale_t * torch.sum(geom * att * strengths, dim=-1) + backgrounds
    return live_time_s * rate
