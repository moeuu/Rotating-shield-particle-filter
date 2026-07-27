"""GPU helpers for batched continuous-kernel computations."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from pf.state import IsotopeState

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    TORCH_AVAILABLE = False


def torch_available() -> bool:
    """Return True if torch is available and CUDA is usable."""
    return bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def torch_installed() -> bool:
    """Return True if torch is available (CUDA not required)."""
    return bool(TORCH_AVAILABLE and torch is not None)


def torch_device_available(device: str | None = None) -> bool:
    """Return True when torch can run on the requested device."""
    if not torch_installed():
        return False
    device_name = "cuda" if device is None else str(device)
    if device_name.startswith("cuda"):
        return bool(torch is not None and torch.cuda.is_available())
    return True


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
