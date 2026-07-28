"""Represent the measurement environment for a non-directional detector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import numpy as np

try:  # optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False


def _finite_real(value: object, *, field_name: str) -> float:
    """Return an exact finite real value without string or boolean coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _finite_tuple(
    value: object,
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    """Return an exact-length tuple of finite real values."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values.")
    return tuple(
        _finite_real(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Hold environment dimensions and detector position."""

    size_x: float = 10.0
    size_y: float = 20.0
    size_z: float = 10.0
    detector_position: Tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate finite positive room geometry and an in-room detector."""
        size_x = _finite_real(self.size_x, field_name="size_x")
        size_y = _finite_real(self.size_y, field_name="size_y")
        size_z = _finite_real(self.size_z, field_name="size_z")
        if size_x <= 0.0 or size_y <= 0.0 or size_z <= 0.0:
            raise ValueError("Environment dimensions must be positive.")
        detector_position = self.detector_position
        if detector_position is not None:
            detector_values = _finite_tuple(
                detector_position,
                length=3,
                field_name="detector_position",
            )
            if any(
                coordinate < 0.0 or coordinate > upper
                for coordinate, upper in zip(
                    detector_values,
                    (size_x, size_y, size_z),
                    strict=True,
                )
            ):
                raise ValueError(
                    "detector_position must lie inside the environment bounds."
                )
            detector_position = (
                detector_values[0],
                detector_values[1],
                detector_values[2],
            )
        object.__setattr__(self, "size_x", size_x)
        object.__setattr__(self, "size_y", size_y)
        object.__setattr__(self, "size_z", size_z)
        object.__setattr__(self, "detector_position", detector_position)

    def detector(self) -> np.ndarray:
        """Return the detector position (defaults to the environment center)."""
        if self.detector_position is None:
            return np.array([self.size_x / 2.0, self.size_y / 2.0, self.size_z / 2.0])
        return np.array(self.detector_position, dtype=float)


@dataclass(frozen=True)
class PointSource:
    """
    Represent a point radiation source on the detector-count-rate scale.

    ``intensity_cps_1m`` is the expected pre-dead-time detector pulse rate at
    a 1 m source-detector distance for the configured detector response. It is
    not total isotropic gamma/s activity and dead time must be applied exactly
    once after source and background spectra have been combined.
    """

    isotope: str
    position: Tuple[float, float, float]
    intensity_cps_1m: float
    surface_chart_id: int | None = None
    surface_uv: Tuple[float, float] | None = None
    surface_normal: Tuple[float, float, float] | None = None
    transport_position: Tuple[float, float, float] | None = None
    surface_emission_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate optional exact-anchor and native-transport provenance."""
        if not isinstance(self.isotope, str) or not self.isotope:
            raise ValueError("PointSource isotope must be a nonempty string.")
        position_values = _finite_tuple(
            self.position,
            length=3,
            field_name="PointSource.position",
        )
        position = (
            position_values[0],
            position_values[1],
            position_values[2],
        )
        intensity = _finite_real(
            self.intensity_cps_1m,
            field_name="PointSource.intensity_cps_1m",
        )
        if intensity <= 0.0:
            raise ValueError("PointSource intensity_cps_1m must be positive.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "intensity_cps_1m", intensity)
        metadata = (
            self.surface_chart_id,
            self.surface_uv,
            self.surface_normal,
            self.transport_position,
            self.surface_emission_policy_sha256,
        )
        if all(value is None for value in metadata):
            return
        if any(value is None for value in metadata):
            raise ValueError(
                "Surface-bound PointSource metadata must be provided as one "
                "complete chart/UV/normal/transport/policy contract."
            )
        chart_id = self.surface_chart_id
        if (
            isinstance(chart_id, (bool, np.bool_))
            or not isinstance(chart_id, (int, np.integer))
            or int(chart_id) < 0
        ):
            raise ValueError("surface_chart_id must be a nonnegative integer.")
        uv_values = _finite_tuple(
            self.surface_uv,
            length=2,
            field_name="surface_uv",
        )
        normal_values = _finite_tuple(
            self.surface_normal,
            length=3,
            field_name="surface_normal",
        )
        transport_values = _finite_tuple(
            self.transport_position,
            length=3,
            field_name="transport_position",
        )
        uv = np.asarray(uv_values, dtype=float)
        normal = np.asarray(normal_values, dtype=float)
        if (
            np.any(uv < 0.0)
            or np.any(uv > 1.0)
            or not np.isclose(
                np.linalg.norm(normal),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("PointSource surface coordinates are invalid.")
        digest = self.surface_emission_policy_sha256
        if not isinstance(digest, str):
            raise ValueError(
                "surface_emission_policy_sha256 must be a lowercase SHA-256 string."
            )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("surface_emission_policy_sha256 must be lowercase SHA-256.")
        object.__setattr__(self, "surface_chart_id", int(chart_id))
        object.__setattr__(self, "surface_uv", (uv_values[0], uv_values[1]))
        object.__setattr__(
            self,
            "surface_normal",
            (normal_values[0], normal_values[1], normal_values[2]),
        )
        object.__setattr__(
            self,
            "transport_position",
            (transport_values[0], transport_values[1], transport_values[2]),
        )

    def position_array(self) -> np.ndarray:
        """Return the exact evaluation-truth surface anchor."""
        return np.array(self.position, dtype=float)

    def transport_position_array(self) -> np.ndarray:
        """Return the native/PF forward-model source position in navigable air."""
        value = (
            self.position
            if self.transport_position is None
            else self.transport_position
        )
        return np.asarray(value, dtype=float)


def inverse_square_scale(detector: np.ndarray, source: PointSource) -> float:
    """
    Return the inverse-square geometric scale for a point source.

    Computes 1/d^2 based on detector distance for detector cps@1m scaling.
    """
    detector_array = np.asarray(detector)
    if detector_array.shape != (3,) or not np.issubdtype(
        detector_array.dtype,
        np.number,
    ):
        raise ValueError("detector must contain exactly three numeric values.")
    detector_array = detector_array.astype(float, copy=False)
    if np.any(~np.isfinite(detector_array)):
        raise ValueError("detector coordinates must be finite.")
    distance = float(
        np.linalg.norm(detector_array - source.transport_position_array())
    )
    if distance <= 0.0:
        raise ValueError(
            "Point-source inverse-square response is undefined at zero distance."
        )
    return 1.0 / (distance**2)


def inverse_square_scale_batch(
    detectors: np.ndarray,
    sources: np.ndarray,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> np.ndarray:
    """
    Return inverse-square scaling for paired detector/source arrays.

    Args:
        detectors: (N, 3) detector positions.
        sources: (N, 3) source positions.
        use_gpu: If True, require and compute on the requested CUDA device.
        gpu_device: Torch device string.
        gpu_dtype: Torch dtype string.
    """
    detector_input = np.asarray(detectors)
    source_input = np.asarray(sources)
    if not np.issubdtype(detector_input.dtype, np.number) or not np.issubdtype(
        source_input.dtype,
        np.number,
    ):
        raise ValueError("detectors and sources must contain numeric values.")
    detectors = detector_input.astype(float, copy=False)
    sources = source_input.astype(float, copy=False)
    if detectors.shape != sources.shape or detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("detectors and sources must be shape (N, 3)")
    if np.any(~np.isfinite(detectors)) or np.any(~np.isfinite(sources)):
        raise ValueError("detectors and sources must be finite.")
    dist = np.linalg.norm(detectors - sources, axis=1)
    if np.any(dist <= 0.0):
        raise ValueError(
            "Point-source inverse-square response is undefined at zero distance."
        )
    if use_gpu is not None and not isinstance(use_gpu, (bool, np.bool_)):
        raise TypeError("use_gpu must be a boolean or null.")
    if not isinstance(gpu_device, str) or not gpu_device.strip():
        raise TypeError("gpu_device must be a nonempty string.")
    if gpu_dtype not in {"float32", "float64"}:
        raise ValueError("gpu_dtype must be 'float32' or 'float64'.")
    if use_gpu is None:
        use_gpu = bool(
            _TORCH_AVAILABLE
            and torch is not None
            and torch.cuda.is_available()
        )
    if use_gpu:
        if not _TORCH_AVAILABLE or torch is None:
            raise RuntimeError("CUDA inverse-square batching requires torch.")
        device = torch.device(gpu_device)
        if device.type != "cuda":
            raise ValueError(
                "use_gpu=true requires gpu_device to name a CUDA device."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA inverse-square batching was requested but unavailable."
            )
        dtype = torch.float32 if gpu_dtype == "float32" else torch.float64
        det_t = torch.as_tensor(detectors, device=device, dtype=dtype)
        src_t = torch.as_tensor(sources, device=device, dtype=dtype)
        distance_t = torch.linalg.norm(det_t - src_t, dim=1)
        scale = 1.0 / (distance_t**2)
        return scale.detach().cpu().numpy()
    return 1.0 / (dist**2)
