"""GPU helpers for batched continuous-kernel computations."""

from __future__ import annotations

from functools import lru_cache

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
    """Resolve a torch device string without a silent CUDA fallback."""
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


@lru_cache(maxsize=None)
def require_torch_compute_device(
    device: str,
    dtype: str = "float64",
) -> None:
    """Fail unless a finite tensor operation runs on the requested device."""
    device_name = str(device).strip()
    dtype_name = str(dtype).strip().lower()
    if not device_name:
        raise RuntimeError("The requested torch compute device is empty.")
    if dtype_name != "float64":
        raise RuntimeError(
            "Pure PF compute requires torch float64; lower precision is "
            "not an allowed runtime fallback."
        )
    if torch is None:
        raise RuntimeError(
            "Pure PF use_gpu=true requires torch before simulation starts."
        )
    try:
        resolved_device = resolve_device(device_name)
        resolved_dtype = resolve_dtype(dtype_name)
        probe = torch.tensor(
            [1.0, 2.0],
            device=resolved_device,
            dtype=resolved_dtype,
        )
        result = torch.sum(probe * probe)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        if result.dtype != torch.float64 or not bool(
            torch.isfinite(result).detach().cpu().item()
        ):
            raise RuntimeError(
                "The requested torch device did not preserve finite float64 "
                "arithmetic."
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith(
            "The requested torch device did not preserve"
        ):
            raise
        raise RuntimeError(
            "Pure PF use_gpu=true could not execute float64 torch arithmetic "
            f"on device {device_name!r}."
        ) from exc


def preflight_compute_backend(
    *,
    use_gpu: bool,
    gpu_device: str,
    gpu_dtype: str,
) -> str:
    """Validate the selected PF compute backend before live inference starts."""
    dtype_name = str(gpu_dtype).strip().lower()
    if dtype_name != "float64":
        raise ValueError(
            "Production pure-PF runtime requires gpu_dtype='float64'; "
            "lower-precision posterior dynamics are forbidden."
        )
    if not bool(use_gpu):
        return "batched_numpy_float64"
    require_torch_compute_device(str(gpu_device), dtype_name)
    return "batched_torch_float64"
