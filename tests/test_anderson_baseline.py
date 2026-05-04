"""Tests for the Anderson et al. RBE comparison baseline."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.anderson import (
    AndersonAttenuationKernel,
    AndersonFilterConfig,
    AndersonFisherConfig,
    AndersonKernelConfig,
    AndersonMeasurement,
    AndersonParallelConfig,
    AndersonParallelRBE,
    AndersonRBEParticleFilter,
    poisson_interval_log_likelihood,
    select_fisher_waypoint,
)
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid


def _build_kernel(
    obstacle_grid: ObstacleGrid | None = None,
    *,
    cpu_workers: int = 1,
    use_gpu: bool = False,
) -> AndersonAttenuationKernel:
    """Return a small Anderson attenuation kernel for tests."""
    env = EnvironmentConfig(
        size_x=4.0,
        size_y=4.0,
        size_z=2.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    return AndersonAttenuationKernel.from_environment(
        env=env,
        isotopes=["Cs-137", "Co-60"],
        obstacle_grid=obstacle_grid,
        config=AndersonKernelConfig(
            use_gpu=use_gpu,
            cpu_workers=cpu_workers,
            kernel_chunk_size=2,
        ),
    )


def test_poisson_interval_likelihood_prefers_matching_mean() -> None:
    """Anderson interval likelihood should prefer means near the observation."""
    means = np.asarray([20.0, 100.0, 300.0], dtype=float)
    log_likelihood = poisson_interval_log_likelihood(100.0, means, alpha=1.0)
    assert int(np.argmax(log_likelihood)) == 1


def test_anderson_kernel_models_obstacle_attenuation() -> None:
    """Obstacle cells should attenuate a source-detector segment."""
    obstacle = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 0),),
    )
    blocked_kernel = _build_kernel(obstacle)
    free_kernel = _build_kernel(None)
    source = np.asarray([[2.5, 0.5, 0.5]], dtype=float)
    detector = (0.5, 0.5, 0.5)

    blocked = blocked_kernel.response_vector(
        isotope="Cs-137",
        detector_pos=detector,
        sources=source,
    )[0]
    free = free_kernel.response_vector(
        isotope="Cs-137",
        detector_pos=detector,
        sources=source,
    )[0]

    assert blocked < free


def test_anderson_cpu_parallel_batch_matches_serial() -> None:
    """CPU-worker batch evaluation should preserve the serial expected counts."""
    serial = _build_kernel(None, cpu_workers=1)
    parallel = _build_kernel(None, cpu_workers=2)
    sources = np.asarray(
        [
            [[1.5, 1.0, 0.5], [2.5, 2.0, 0.5]],
            [[3.0, 1.0, 0.5], [1.0, 3.0, 0.5]],
        ],
        dtype=float,
    )
    activities = np.asarray([[1000.0, 500.0], [800.0, 300.0]], dtype=float)

    serial_counts = serial.expected_counts_batch(
        isotope="Cs-137",
        detector_pos=(1.0, 1.0, 0.5),
        sources=sources,
        activities=activities,
        live_time_s=2.0,
    )
    parallel_counts = parallel.expected_counts_batch(
        isotope="Cs-137",
        detector_pos=(1.0, 1.0, 0.5),
        sources=sources,
        activities=activities,
        live_time_s=2.0,
    )

    assert np.allclose(serial_counts, parallel_counts)


def test_anderson_gpu_batch_runs_when_cuda_available() -> None:
    """The Anderson kernel should expose an explicit CUDA batch path."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    kernel = _build_kernel(None, use_gpu=True)
    counts = kernel.expected_counts_batch(
        isotope="Cs-137",
        detector_pos=(1.0, 1.0, 0.5),
        sources=np.asarray([[[1.5, 1.0, 0.5]]], dtype=float),
        activities=np.asarray([[1000.0]], dtype=float),
        live_time_s=1.0,
    )

    assert counts.shape == (1,)
    assert np.all(np.isfinite(counts))


def test_anderson_filter_update_prefers_consistent_particle() -> None:
    """A deterministic update should weight the particle matching the count."""
    kernel = _build_kernel(None)
    config = AndersonFilterConfig(
        num_particles=2,
        num_sources=1,
        position_min=(0.0, 0.0, 0.0),
        position_max=(4.0, 4.0, 2.0),
        init_activity_log_mean=float(np.log(1000.0)),
        init_activity_log_sigma=0.1,
        resample_ess_fraction=0.0,
        rng_seed=1,
    )
    filt = AndersonRBEParticleFilter(
        isotope="Cs-137",
        kernel=kernel,
        config=config,
    )
    filt.set_particles_for_tests(
        positions=np.asarray(
            [
                [[1.5, 1.0, 0.5]],
                [[3.5, 3.5, 0.5]],
            ],
            dtype=float,
        ),
        activities=np.asarray([[1000.0], [1000.0]], dtype=float),
    )
    detector = (1.0, 1.0, 0.5)
    expected = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=detector,
        sources=np.asarray([[1.5, 1.0, 0.5]], dtype=float),
        activities=np.asarray([1000.0], dtype=float),
        live_time_s=1.0,
    )

    filt.update(
        AndersonMeasurement(
            detector_pos=detector,
            live_time_s=1.0,
            counts=expected,
        )
    )

    assert int(np.argmax(filt.weights)) == 0


def test_anderson_decay_prediction_reduces_activity_only() -> None:
    """The short-lived nuclide transition should decay activities only."""
    kernel = _build_kernel(None)
    config = AndersonFilterConfig(
        num_particles=2,
        num_sources=1,
        half_life_s=10.0,
        rng_seed=3,
    )
    filt = AndersonRBEParticleFilter(
        isotope="Cs-137",
        kernel=kernel,
        config=config,
    )
    positions_before = filt.positions.copy()
    activities_before = filt.activities.copy()

    filt.predict_decay(10.0)

    assert np.allclose(filt.positions, positions_before)
    assert np.allclose(filt.activities, activities_before * 0.5)


def test_parallel_rbe_initializes_only_detected_isotopes() -> None:
    """Parallel Anderson filters should lazily initialize detected isotopes."""
    kernel = _build_kernel(None)
    estimator = AndersonParallelRBE(
        isotopes=["Cs-137", "Co-60"],
        kernel=kernel,
        config=AndersonParallelConfig(
            filter_config=AndersonFilterConfig(num_particles=10, rng_seed=5),
            initialize_all_isotopes=False,
            detection_count_threshold=1.0,
            update_workers=2,
        ),
    )

    estimator.update(
        detector_pos=(1.0, 1.0, 0.5),
        live_time_s=1.0,
        counts_by_isotope={"Cs-137": 25.0, "Co-60": 0.0},
    )

    assert "Cs-137" in estimator.filters
    assert "Co-60" not in estimator.filters
    assert estimator.estimates_for_metrics()["Co-60"] == []


def test_parallel_rbe_cpu_workers_update_initialized_filters() -> None:
    """Parallel Anderson updates should preserve the initialized isotope bank."""
    kernel = _build_kernel(None)
    estimator = AndersonParallelRBE(
        isotopes=["Cs-137", "Co-60"],
        kernel=kernel,
        config=AndersonParallelConfig(
            filter_config=AndersonFilterConfig(num_particles=8, rng_seed=6),
            initialize_all_isotopes=True,
            update_workers=2,
        ),
    )

    diagnostics = estimator.update(
        detector_pos=(1.0, 1.0, 0.5),
        live_time_s=1.0,
        counts_by_isotope={"Cs-137": 10.0, "Co-60": 5.0},
    )

    assert set(diagnostics) == {"Cs-137", "Co-60"}
    assert all(item["compute_backend"] == "cpu" for item in diagnostics.values())


def test_fisher_waypoint_selector_returns_candidate() -> None:
    """Anderson Fisher planning should select one of the candidate points."""
    kernel = _build_kernel(None)
    config = AndersonFilterConfig(
        num_particles=4,
        num_sources=1,
        position_min=(0.0, 0.0, 0.0),
        position_max=(4.0, 4.0, 2.0),
        rng_seed=8,
    )
    filt = AndersonRBEParticleFilter(
        isotope="Cs-137",
        kernel=kernel,
        config=config,
    )
    filt.set_particles_for_tests(
        positions=np.asarray(
            [
                [[2.0, 2.0, 0.5]],
                [[2.1, 2.0, 0.5]],
                [[1.9, 2.0, 0.5]],
                [[2.0, 2.1, 0.5]],
            ],
            dtype=float,
        ),
        activities=np.full((4, 1), 1000.0, dtype=float),
    )
    candidates = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [3.0, 2.0, 0.5],
            [3.5, 3.5, 0.5],
        ],
        dtype=float,
    )

    selected, diagnostics = select_fisher_waypoint(
        filt=filt,
        current_pos=(0.5, 0.5, 0.5),
        candidate_positions=candidates,
        config=AndersonFisherConfig(live_time_s=5.0, ridge=1.0e-3),
    )

    assert any(np.allclose(selected, candidate) for candidate in candidates)
    assert diagnostics["score"] >= 0.0


def test_fisher_waypoint_cpu_workers_match_serial() -> None:
    """CPU-worker Fisher candidate evaluation should match serial selection."""
    kernel = _build_kernel(None)
    config = AndersonFilterConfig(
        num_particles=4,
        num_sources=1,
        position_min=(0.0, 0.0, 0.0),
        position_max=(4.0, 4.0, 2.0),
        rng_seed=9,
    )
    filt = AndersonRBEParticleFilter(
        isotope="Cs-137",
        kernel=kernel,
        config=config,
    )
    filt.set_particles_for_tests(
        positions=np.asarray(
            [
                [[2.0, 2.0, 0.5]],
                [[2.1, 2.0, 0.5]],
                [[1.9, 2.0, 0.5]],
                [[2.0, 2.1, 0.5]],
            ],
            dtype=float,
        ),
        activities=np.full((4, 1), 1000.0, dtype=float),
    )
    candidates = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [3.0, 2.0, 0.5],
            [3.5, 3.5, 0.5],
        ],
        dtype=float,
    )

    serial, _ = select_fisher_waypoint(
        filt=filt,
        current_pos=(0.5, 0.5, 0.5),
        candidate_positions=candidates,
        config=AndersonFisherConfig(live_time_s=5.0, ridge=1.0e-3),
    )
    parallel, diagnostics = select_fisher_waypoint(
        filt=filt,
        current_pos=(0.5, 0.5, 0.5),
        candidate_positions=candidates,
        config=AndersonFisherConfig(
            live_time_s=5.0,
            ridge=1.0e-3,
            cpu_workers=2,
        ),
    )

    assert np.allclose(serial, parallel)
    assert diagnostics["compute_backend"] == "cpu"
