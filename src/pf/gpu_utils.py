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
    OCTANT_THETA_PHI_RANGES,
    SHIELD_GEOMETRY_SPHERICAL_OCTANT,
    generate_octant_orientations,
    spherical_shell_path_length_cm_torch,
)
from measurement.continuous_kernels import obstacle_path_lengths_cm_torch
from pf.state import IsotopeState


_OCTANT_RANGES = OCTANT_THETA_PHI_RANGES


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
    dir_unit = direction / dist.unsqueeze(-1)
    geom = 1.0 / (dist**2)

    theta = torch.acos(torch.clamp(dir_unit[..., 2], -1.0, 1.0))
    phi = torch.atan2(dir_unit[..., 1], dir_unit[..., 0]) % (2.0 * np.pi)

    theta_low, theta_high = _OCTANT_RANGES[fe_index][0]
    phi_low, phi_high = _OCTANT_RANGES[fe_index][1]
    blocked_fe = (
        (theta + tol >= theta_low)
        & (theta - tol < theta_high)
        & (phi + tol >= phi_low)
        & (phi - tol < phi_high)
    )
    theta_low, theta_high = _OCTANT_RANGES[pb_index][0]
    phi_low, phi_high = _OCTANT_RANGES[pb_index][1]
    blocked_pb = (
        (theta + tol >= theta_low)
        & (theta - tol < theta_high)
        & (phi + tol >= phi_low)
        & (phi - tol < phi_high)
    )

    normals = torch.as_tensor(generate_octant_orientations(), device=device, dtype=dtype)
    normal_fe = normals[fe_index]
    normal_pb = normals[pb_index]
    cos_fe = torch.clamp(torch.sum(dir_unit * normal_fe, dim=-1), 0.0, 1.0)
    cos_pb = torch.clamp(torch.sum(dir_unit * normal_pb, dim=-1), 0.0, 1.0)
    thickness_fe = torch.as_tensor(thickness_fe_cm, device=device, dtype=dtype)
    thickness_pb = torch.as_tensor(thickness_pb_cm, device=device, dtype=dtype)
    if shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT and not use_angle_attenuation:
        L_fe = spherical_shell_path_length_cm_torch(
            direction,
            inner_radius_fe_cm,
            inner_radius_fe_cm + thickness_fe_cm,
            blocked_fe,
        )
        L_pb = spherical_shell_path_length_cm_torch(
            direction,
            inner_radius_pb_cm,
            inner_radius_pb_cm + thickness_pb_cm,
            blocked_pb,
        )
    elif use_angle_attenuation:
        L_fe = torch.where(blocked_fe & (cos_fe > tol), thickness_fe / cos_fe, torch.zeros_like(cos_fe))
        L_pb = torch.where(blocked_pb & (cos_pb > tol), thickness_pb / cos_pb, torch.zeros_like(cos_pb))
    else:
        L_fe = torch.where(blocked_fe, thickness_fe, torch.zeros_like(cos_fe))
        L_pb = torch.where(blocked_pb, thickness_pb, torch.zeros_like(cos_pb))
    att = torch.exp(-(mu_fe * L_fe + mu_pb * L_pb))
    if obstacle_boxes_m is not None and float(obstacle_mu_cm_inv) > 0.0:
        boxes_t = torch.as_tensor(obstacle_boxes_m, device=device, dtype=dtype)
        if boxes_t.numel() > 0:
            obstacle_path_cm = obstacle_path_lengths_cm_torch(
                positions=positions,
                detector_pos=detector.view(3),
                obstacle_boxes_m=boxes_t,
            )
            att = att * torch.exp(-float(obstacle_mu_cm_inv) * obstacle_path_cm)

    strengths = strengths * mask
    source_scale_t = torch.as_tensor(max(float(source_scale), 0.0), device=device, dtype=dtype)
    rate = source_scale_t * torch.sum(geom * att * strengths, dim=-1) + backgrounds
    return live_time_s * rate
