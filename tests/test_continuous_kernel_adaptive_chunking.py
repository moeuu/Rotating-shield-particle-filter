"""Tests for memory-aware ContinuousKernel torch batching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid


def _line_entries(count: int) -> tuple[dict[str, float], ...]:
    """Return a normalized synthetic line table with the requested size."""
    return tuple(
        {
            "weight": 1.0 / float(count),
            "fe": 0.01 + 0.001 * index,
            "pb": 0.02 + 0.001 * index,
        }
        for index in range(count)
    )


def _ral_shape_kernel() -> ContinuousKernel:
    """Return a kernel with the RAL aperture, obstacle, and line dimensions."""
    blocked = tuple((index, 0) for index in range(500))
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(500, 1),
        blocked_cells=blocked,
    )
    return ContinuousKernel(
        obstacle_grid=grid,
        detector_radius_m=0.038,
        detector_aperture_samples=121,
        gpu_dtype="float64",
        line_mu_by_isotope={
            "Cs-137": _line_entries(1),
            "Co-60": _line_entries(2),
            "Eu-154": _line_entries(6),
        },
    )


def test_cuda_chunk_uses_free_vram_and_actual_isotope_line_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA sizing should use headroom and the selected isotope's line count."""
    torch = pytest.importorskip("torch")
    kernel = _ral_shape_kernel()
    gib = 1024**3

    def _memory_info(_device: object) -> tuple[int, int]:
        """Return an otherwise idle 32 GiB test GPU."""
        return 32 * gib, 32 * gib

    monkeypatch.setattr(kernel, "_torch_cuda_memory_info", _memory_info)
    chunks = {
        isotope: kernel._adaptive_torch_chunk_size(
            8192,
            isotope=isotope,
            orientation_pair_count=64,
            device=torch.device("cuda"),
            dtype=torch.float64,
        )
        for isotope in ("Cs-137", "Co-60", "Eu-154")
    }

    assert chunks["Cs-137"] > chunks["Co-60"] > chunks["Eu-154"]
    assert chunks["Eu-154"] > 1
    assert chunks["Cs-137"] < 8192


def test_low_vram_shrinks_chunk_and_cpu_uses_fixed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low free VRAM should shrink safely while CPU avoids CUDA memory queries."""
    torch = pytest.importorskip("torch")
    kernel = _ral_shape_kernel()
    mib = 1024**2
    gib = 1024**3

    def _low_memory_info(_device: object) -> tuple[int, int]:
        """Return a CUDA device whose free memory is below reserved headroom."""
        return 512 * mib, 16 * gib

    monkeypatch.setattr(kernel, "_torch_cuda_memory_info", _low_memory_info)
    cuda_chunk = kernel._adaptive_torch_chunk_size(
        8192,
        isotope="Eu-154",
        orientation_pair_count=64,
        device=torch.device("cuda"),
        dtype=torch.float64,
    )

    def _unexpected_memory_query(_device: object) -> tuple[int, int]:
        """Fail if the CPU fallback attempts a CUDA memory query."""
        raise AssertionError("CPU fallback queried CUDA memory")

    monkeypatch.setattr(
        kernel,
        "_torch_cuda_memory_info",
        _unexpected_memory_query,
    )
    cpu_chunk = kernel._adaptive_torch_chunk_size(
        8192,
        isotope="Eu-154",
        orientation_pair_count=64,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert cuda_chunk == 2
    assert cpu_chunk == 11
    assert cuda_chunk < cpu_chunk


def test_cuda_oom_retry_halves_chunk_and_restarts_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA OOM recovery should halve deterministically and restart at zero."""
    torch = pytest.importorskip("torch")
    kernel = ContinuousKernel()
    attempts: list[tuple[int, int]] = []
    cache_clears: list[str] = []

    def _evaluate(start: int, stop: int) -> np.ndarray:
        """Fail chunks larger than two rows and return ordered rows otherwise."""
        attempts.append((start, stop))
        if stop - start > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return np.arange(start, stop, dtype=np.int64)

    def _record_cache_clear(device: object) -> None:
        """Record each cache clear without requiring an actual CUDA device."""
        cache_clears.append(str(device))

    monkeypatch.setattr(kernel, "_clear_cuda_cache_after_oom", _record_cache_clear)
    parts, successful_chunk = kernel._evaluate_torch_chunks_with_oom_retry(
        total_size=8,
        initial_chunk=8,
        device=torch.device("cuda"),
        evaluator=_evaluate,
    )

    assert successful_chunk == 2
    assert attempts[:3] == [(0, 8), (0, 4), (0, 2)]
    assert len(cache_clears) == 2
    assert np.array_equal(np.concatenate(parts), np.arange(8))


def test_standard_packed_path_selects_adaptive_chunk_without_math_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard packed path should select adaptive batching exactly."""
    torch = pytest.importorskip("torch")
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=((0, 0), (1, 0)),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"TestIso": 0.03},
        detector_radius_m=0.038,
        detector_aperture_samples=5,
        line_mu_by_isotope={"TestIso": _line_entries(2)},
        gpu_device="cpu",
        gpu_dtype="float64",
    )
    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(
        [
            [[-1.0, 0.0, 0.8], [-0.5, 0.4, 1.2]],
            [[0.2, -0.4, 0.7], [0.8, 0.3, 1.4]],
            [[1.1, -0.2, 1.0], [1.5, 0.5, 0.9]],
            [[2.0, 0.1, 1.3], [2.4, -0.3, 0.6]],
        ],
        device=device,
        dtype=dtype,
    )
    strengths = torch.as_tensor(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
        device=device,
        dtype=dtype,
    )
    backgrounds = torch.as_tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    mask = torch.ones((4, 2), device=device, dtype=dtype)
    fe_indices = np.asarray([0, 3, 7], dtype=np.int64)
    pb_indices = np.asarray([7, 2, 4], dtype=np.int64)

    def _one_particle_chunk(_requested: int, **_kwargs: Any) -> int:
        """Force the former conservative one-particle schedule."""
        return 2

    monkeypatch.setattr(kernel, "_adaptive_torch_chunk_size", _one_particle_chunk)
    legacy = kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=np.asarray([3.0, 0.0, 1.0]),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_time_s=2.0,
        device=device,
        dtype=dtype,
    )

    adaptive_calls: list[dict[str, Any]] = []
    original_adaptive = ContinuousKernel._adaptive_torch_chunk_size

    def _tracked_adaptive(requested: int, **kwargs: Any) -> int:
        """Record and invoke the production adaptive chunk selector."""
        adaptive_calls.append(dict(kwargs))
        return original_adaptive(kernel, requested, **kwargs)

    monkeypatch.setattr(kernel, "_adaptive_torch_chunk_size", _tracked_adaptive)
    adaptive = kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=np.asarray([3.0, 0.0, 1.0]),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_time_s=2.0,
        device=device,
        dtype=dtype,
    )

    assert torch.equal(adaptive, legacy)
    assert len(adaptive_calls) == 1
    assert adaptive_calls[0]["isotope"] == "TestIso"
    assert adaptive_calls[0]["orientation_pair_count"] == 3
    assert adaptive_calls[0]["device"] == device
    assert adaptive_calls[0]["dtype"] == dtype


def test_cuda_old_and_adaptive_chunks_are_strictly_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA chunk enlargement should preserve every fp64 output bit."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=0.5,
        grid_shape=(40, 1),
        blocked_cells=tuple((index, 0) for index in range(40)),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"TestIso": 0.03},
        detector_radius_m=0.038,
        detector_aperture_samples=31,
        line_mu_by_isotope={"TestIso": _line_entries(3)},
        gpu_device="cuda",
        gpu_dtype="float64",
    )
    device = torch.device("cuda")
    dtype = torch.float64
    positions_np = np.linspace(-2.0, 3.0, 20 * 3 * 3, dtype=float).reshape(
        20,
        3,
        3,
    )
    positions_np[:, :, 2] = np.abs(positions_np[:, :, 2]) + 0.2
    positions = torch.as_tensor(positions_np, device=device, dtype=dtype)
    strengths = torch.linspace(
        10.0,
        600.0,
        60,
        device=device,
        dtype=dtype,
    ).reshape(20, 3)
    backgrounds = torch.linspace(
        0.1,
        2.0,
        20,
        device=device,
        dtype=dtype,
    )
    mask = torch.ones((20, 3), device=device, dtype=dtype)
    fe_indices = np.arange(8, dtype=np.int64)
    pb_indices = np.arange(7, -1, -1, dtype=np.int64)

    def _legacy_chunk(_requested: int, **_kwargs: Any) -> int:
        """Force one particle per CUDA chunk."""
        return 3

    monkeypatch.setattr(kernel, "_adaptive_torch_chunk_size", _legacy_chunk)
    legacy = kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=np.asarray([4.0, 1.0, 1.5]),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    def _large_chunk(_requested: int, **_kwargs: Any) -> int:
        """Evaluate all particles in one CUDA chunk."""
        return 8192

    monkeypatch.setattr(kernel, "_adaptive_torch_chunk_size", _large_chunk)
    adaptive = kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=np.asarray([4.0, 1.0, 1.5]),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    assert torch.equal(adaptive, legacy)
