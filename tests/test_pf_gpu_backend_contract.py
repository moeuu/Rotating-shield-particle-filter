"""Fail-fast compute-backend contracts for the pure particle filter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from pf import gpu_utils
from pf.particle_filter import IsotopeParticleFilter
from realtime_demo import (
    _preflight_pure_pf_compute_backend,
    run_live_pf,
)


def test_explicit_numpy_backend_does_not_probe_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU-guarded NumPy route must not depend on torch availability."""

    def _unexpected_probe(device: str, dtype: str) -> None:
        """Fail if explicit NumPy mode probes a torch device."""
        raise AssertionError(f"unexpected torch probe: {device}/{dtype}")

    monkeypatch.setattr(
        gpu_utils,
        "require_torch_compute_device",
        _unexpected_probe,
    )

    backend = _preflight_pure_pf_compute_backend(
        use_gpu=False,
        gpu_device="cuda",
        gpu_dtype="float64",
    )

    assert backend == "batched_numpy_float64"


def test_requested_torch_backend_propagates_device_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested torch backend must never fall back silently to NumPy."""

    def _failed_probe(device: str, dtype: str) -> None:
        """Model a requested CUDA device that cannot execute."""
        raise RuntimeError(f"unavailable: {device}/{dtype}")

    monkeypatch.setattr(
        gpu_utils,
        "require_torch_compute_device",
        _failed_probe,
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        _preflight_pure_pf_compute_backend(
            use_gpu=True,
            gpu_device="cuda",
            gpu_dtype="float64",
        )


def test_filter_requested_torch_backend_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The isotope runtime must reject a failed requested torch device."""
    filt = object.__new__(IsotopeParticleFilter)
    filt.config = SimpleNamespace(
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
    )

    def _failed_probe(device: str, dtype: str) -> None:
        """Model a device failure after configuration was resolved."""
        raise RuntimeError(f"device failed: {device}/{dtype}")

    monkeypatch.setattr(
        gpu_utils,
        "require_torch_compute_device",
        _failed_probe,
    )

    with pytest.raises(RuntimeError, match="device failed"):
        filt._can_use_gpu()


def test_filter_explicit_numpy_backend_skips_torch_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The isotope runtime may use NumPy only when use_gpu is explicitly false."""
    filt = object.__new__(IsotopeParticleFilter)
    filt.config = SimpleNamespace(
        use_gpu=False,
        gpu_device="cuda",
        gpu_dtype="float64",
    )

    def _unexpected_probe(device: str, dtype: str) -> None:
        """Fail if explicit NumPy mode probes torch."""
        raise AssertionError(f"unexpected torch probe: {device}/{dtype}")

    monkeypatch.setattr(
        gpu_utils,
        "require_torch_compute_device",
        _unexpected_probe,
    )

    assert filt._can_use_gpu() is False


def test_compute_preflight_precedes_external_simulation_creation() -> None:
    """CUDA failure must occur before a Geant4 sidecar can be launched."""
    source = inspect.getsource(run_live_pf)
    preflight_offset = source.index("_preflight_pure_pf_compute_backend(")
    runtime_offset = source.index("create_simulation_runtime(")

    assert preflight_offset < runtime_offset


def test_torch_cpu_probe_executes_float64_when_installed() -> None:
    """The backend probe itself must perform finite float64 arithmetic."""
    pytest.importorskip("torch")
    gpu_utils.require_torch_compute_device.cache_clear()

    gpu_utils.require_torch_compute_device("cpu", "float64")
