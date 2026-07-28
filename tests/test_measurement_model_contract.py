"""Fail-closed tests for shared environment and point-source contracts."""

from __future__ import annotations

import numpy as np
import pytest

import measurement.model as model_module
from measurement.model import (
    EnvironmentConfig,
    PointSource,
    inverse_square_scale,
    inverse_square_scale_batch,
)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("size_x", "10.0"),
        ("size_y", True),
        ("size_z", float("nan")),
        ("size_x", 0.0),
        ("size_y", -1.0),
    ),
)
def test_environment_rejects_invalid_dimensions(
    field_name: str,
    invalid: object,
) -> None:
    """Room dimensions must remain finite positive physical lengths."""
    values: dict[str, object] = {
        "size_x": 10.0,
        "size_y": 20.0,
        "size_z": 10.0,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError):
        EnvironmentConfig(**values)


@pytest.mark.parametrize(
    "detector_position",
    (
        ("1.0", 1.0, 1.0),
        (True, 1.0, 1.0),
        (float("nan"), 1.0, 1.0),
        (-0.1, 1.0, 1.0),
        (10.1, 1.0, 1.0),
        (1.0, 20.1, 1.0),
        (1.0, 1.0, 10.1),
    ),
)
def test_environment_rejects_invalid_detector_position(
    detector_position: tuple[object, ...],
) -> None:
    """Detector coordinates must be finite numeric values inside the room."""
    with pytest.raises(ValueError):
        EnvironmentConfig(detector_position=detector_position)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("isotope", 137),
        ("isotope", ""),
        ("position", ("1.0", 1.0, 1.0)),
        ("position", (True, 1.0, 1.0)),
        ("position", (float("nan"), 1.0, 1.0)),
        ("intensity_cps_1m", "1000.0"),
        ("intensity_cps_1m", True),
        ("intensity_cps_1m", 0.0),
        ("intensity_cps_1m", float("inf")),
    ),
)
def test_point_source_rejects_implicit_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """Truth semantics must not change through permissive scalar conversion."""
    values: dict[str, object] = {
        "isotope": "Cs-137",
        "position": (1.0, 1.0, 1.0),
        "intensity_cps_1m": 1000.0,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError):
        PointSource(**values)


def test_surface_source_digest_requires_exact_string_type() -> None:
    """An integer resembling a digest must not satisfy provenance validation."""
    digest_as_integer = int("1" * 64)

    with pytest.raises(ValueError, match="SHA-256 string"):
        PointSource(
            isotope="Cs-137",
            position=(1.0, 1.0, 0.0),
            intensity_cps_1m=1000.0,
            surface_chart_id=0,
            surface_uv=(0.5, 0.5),
            surface_normal=(0.0, 0.0, 1.0),
            transport_position=(1.0, 1.0, 1.0e-6),
            surface_emission_policy_sha256=digest_as_integer,
        )


def test_inverse_square_response_rejects_zero_distance() -> None:
    """Coincident point source and detector must not become an arbitrary 1 µm ray."""
    source = PointSource(
        isotope="Cs-137",
        position=(1.0, 2.0, 3.0),
        intensity_cps_1m=1000.0,
    )

    with pytest.raises(ValueError, match="zero distance"):
        inverse_square_scale(np.asarray([1.0, 2.0, 3.0]), source)
    with pytest.raises(ValueError, match="zero distance"):
        inverse_square_scale_batch(
            np.asarray([[1.0, 2.0, 3.0]]),
            np.asarray([[1.0, 2.0, 3.0]]),
            use_gpu=False,
        )


def test_inverse_square_batch_matches_scalar_without_distance_floor() -> None:
    """CPU batching must preserve the exact finite-distance point response."""
    detectors = np.asarray([[0.0, 0.0, 0.0], [1.0e-8, 0.0, 0.0]])
    sources = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    actual = inverse_square_scale_batch(detectors, sources, use_gpu=False)

    assert np.allclose(actual, np.asarray([1.0, 1.0e16]), rtol=1.0e-15)


@pytest.mark.parametrize("invalid_dtype", ("float16", "double", "", 64))
def test_inverse_square_batch_rejects_implicit_dtype_fallback(
    invalid_dtype: object,
) -> None:
    """Unknown dtype values must not silently select float64 arithmetic."""
    with pytest.raises(ValueError, match="gpu_dtype"):
        inverse_square_scale_batch(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            use_gpu=False,
            gpu_dtype=invalid_dtype,
        )


def test_inverse_square_batch_rejects_requested_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CUDA request must not silently execute on the CPU."""
    if model_module.torch is None:
        pytest.skip("Torch is not installed.")
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requested but unavailable"):
        inverse_square_scale_batch(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            use_gpu=True,
            gpu_device="cuda",
            gpu_dtype="float64",
        )


@pytest.mark.skipif(
    model_module.torch is None
    or not bool(model_module.torch.cuda.is_available()),
    reason="CUDA is unavailable.",
)
def test_inverse_square_cuda_matches_vectorized_cpu() -> None:
    """CUDA batching must preserve the vectorized CPU physical response."""
    detectors = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]],
        dtype=np.float64,
    )
    sources = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 3.0, 2.5]],
        dtype=np.float64,
    )

    cpu = inverse_square_scale_batch(
        detectors,
        sources,
        use_gpu=False,
        gpu_dtype="float64",
    )
    cuda = inverse_square_scale_batch(
        detectors,
        sources,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
    )

    assert np.allclose(cuda, cpu, rtol=1.0e-13, atol=0.0)
