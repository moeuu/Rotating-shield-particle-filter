"""Fail-fast compute-backend contracts for the pure particle filter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pf import gpu_utils
from pf.estimator import RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticleFilter


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

    backend = gpu_utils.preflight_compute_backend(
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
        gpu_utils.preflight_compute_backend(
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


def test_torch_cpu_probe_executes_float64_when_installed() -> None:
    """The backend probe itself must perform finite float64 arithmetic."""
    pytest.importorskip("torch")
    gpu_utils.require_torch_compute_device.cache_clear()

    gpu_utils.require_torch_compute_device("cpu", "float64")


def _scalar_tempering_bisection_oracle(
    log_weights: np.ndarray,
    likelihood: np.ndarray,
    *,
    minimum_delta: float,
    remaining: float,
    target_ess: float,
) -> tuple[float, np.ndarray, float]:
    """Return the former host-synchronized tempering bisection result."""

    def _normalized(values: np.ndarray) -> np.ndarray:
        """Normalize one deterministic log-weight vector."""
        shifted = values - float(np.max(values))
        return shifted - np.log(np.sum(np.exp(shifted)))

    def _ess(values: np.ndarray) -> float:
        """Return ESS for one normalized deterministic log-weight vector."""
        weights = np.exp(values)
        return float(1.0 / np.sum(weights**2))

    low = float(minimum_delta)
    high = float(remaining)
    logw_best = _normalized(log_weights + low * likelihood)
    ess_best = _ess(logw_best)
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        logw_mid = _normalized(log_weights + midpoint * likelihood)
        ess_mid = _ess(logw_mid)
        if ess_mid >= target_ess:
            low = midpoint
            logw_best = logw_mid
            ess_best = ess_mid
        else:
            high = midpoint
    return low, logw_best, ess_best


def test_device_tempering_bisection_matches_scalar_oracle() -> None:
    """The standard device-resident bisection must match its scalar oracle."""
    torch = pytest.importorskip("torch")
    particle_count = 16
    minimum_delta = 1.0e-8
    remaining = 0.9
    target_ess = 0.8 * particle_count
    log_weights = np.full(
        particle_count,
        -np.log(particle_count),
        dtype=np.float64,
    )
    likelihood = np.linspace(-8.0, 2.0, particle_count, dtype=np.float64)
    expected_delta, expected_logw, expected_ess = (
        _scalar_tempering_bisection_oracle(
            log_weights,
            likelihood,
            minimum_delta=minimum_delta,
            remaining=remaining,
            target_ess=target_ess,
        )
    )
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device_name in devices:
        filt = object.__new__(IsotopeParticleFilter)
        filt.config = SimpleNamespace(min_delta_beta=minimum_delta)
        actual_delta, actual_logw, actual_ess = filt._select_delta_beta(
            logw_prev=torch.as_tensor(
                log_weights,
                device=device_name,
                dtype=torch.float64,
            ),
            ll_t=torch.as_tensor(
                likelihood,
                device=device_name,
                dtype=torch.float64,
            ),
            remaining=remaining,
            target_ess=target_ess,
        )
        np.testing.assert_allclose(actual_delta, expected_delta, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            actual_logw.detach().cpu().numpy(),
            expected_logw,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(actual_ess, expected_ess, rtol=1e-13, atol=1e-13)


def _scalar_transport_is_invalid(
    total: np.ndarray,
    uncollided: np.ndarray,
    features: np.ndarray,
) -> bool:
    """Return the scalar-oracle result for the fused transport validator."""
    return bool(
        np.any(~np.isfinite(total))
        or np.any(~np.isfinite(uncollided))
        or np.any(~np.isfinite(features))
        or np.any(total < 0.0)
        or np.any(uncollided < 0.0)
    )


def test_fused_transport_validation_matches_scalar_oracle() -> None:
    """Fused CPU/CUDA validation must preserve every former scalar check."""
    torch = pytest.importorskip("torch")
    valid_total = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    valid_uncollided = 0.5 * valid_total
    valid_features = np.ones((2, 2, 4), dtype=np.float64)
    cases = [
        (valid_total, valid_uncollided, valid_features),
        (
            np.asarray([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64),
            valid_uncollided,
            valid_features,
        ),
        (
            valid_total,
            np.asarray([[0.5, -1.0], [1.5, 2.0]], dtype=np.float64),
            valid_features,
        ),
        (
            valid_total,
            valid_uncollided,
            np.full((2, 2, 4), np.inf, dtype=np.float64),
        ),
    ]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device_name in devices:
        for total, uncollided, features in cases:
            tensors = tuple(
                torch.as_tensor(values, device=device_name, dtype=torch.float64)
                for values in (total, uncollided, features)
            )
            expected_invalid = _scalar_transport_is_invalid(
                total,
                uncollided,
                features,
            )
            if expected_invalid:
                with pytest.raises(RuntimeError, match="invalid transport"):
                    RotatingShieldPFEstimator._validate_torch_transport_components(
                        *tensors,
                        error_message="invalid transport",
                    )
            else:
                RotatingShieldPFEstimator._validate_torch_transport_components(
                    *tensors,
                    error_message="invalid transport",
                )
