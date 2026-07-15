"""Tests for the Kemp et al. comparison baseline implementation."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.kemp.filter import KempFilterConfig, KempMeasurement
from baselines.kemp.kernels import DiscreteAttenuationKernel, KempKernelConfig
from baselines.kemp.parallel import KempParallelConfig, KempParallelLogDDPF
from baselines.kemp.runner import _scene_reset_payload
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig, PointSource
from measurement.obstacles import ObstacleGrid


def _build_test_kernel(*, cpu_workers: int = 1) -> DiscreteAttenuationKernel:
    """Return a small obstacle-free Kemp kernel for unit tests."""
    env = EnvironmentConfig(size_x=4.0, size_y=4.0, size_z=2.0, detector_position=(1.0, 1.0, 0.5))
    return DiscreteAttenuationKernel.from_environment(
        env=env,
        isotopes=["Cs-137", "Co-60"],
        mu_by_isotope=None,
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
        obstacle_grid=None,
        config=KempKernelConfig(
            grid_spacing_m=(1.0, 1.0, 1.0),
            grid_margin_m=0.5,
            z_levels_m=(0.5,),
            use_gpu=False,
            cpu_workers=cpu_workers,
            kernel_chunk_size=2,
        ),
    )


def test_discrete_kernel_prefers_near_source() -> None:
    """Check that the kernel obeys detector-cps inverse-distance scaling."""
    kernel = _build_test_kernel()
    detector = (1.0, 1.0, 0.5)
    theta = kernel.kernel_vector("Cs-137", detector, 0, 0)
    near_idx = kernel.nearest_index((1.5, 1.5, 0.5))
    far_idx = kernel.nearest_index((3.5, 3.5, 0.5))
    assert theta[near_idx] > theta[far_idx]


def test_kemp_kernel_preserves_detector_observation_geometry() -> None:
    """Kemp baseline should pass shared detector aperture geometry to the kernel."""
    env = EnvironmentConfig(
        size_x=4.0,
        size_y=4.0,
        size_z=2.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    kernel = DiscreteAttenuationKernel.from_environment(
        env=env,
        isotopes=["Cs-137"],
        mu_by_isotope=None,
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
        obstacle_grid=None,
        config=KempKernelConfig(
            grid_spacing_m=(1.0, 1.0, 1.0),
            grid_margin_m=0.5,
            z_levels_m=(0.5,),
            detector_radius_m=0.05,
            detector_aperture_radius_m=0.052,
            detector_aperture_samples=33,
            use_gpu=False,
        ),
    )

    assert kernel._kernel.detector_radius_m == pytest.approx(0.05)
    assert kernel._kernel.detector_aperture_radius_m == pytest.approx(0.052)
    assert kernel._kernel.detector_aperture_samples == 33


def test_kemp_kernel_preserves_transport_response_model() -> None:
    """Kemp baseline should pass the shared transport sidecar to the kernel."""
    env = EnvironmentConfig(
        size_x=4.0,
        size_y=4.0,
        size_z=2.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    transport_model = {
        "enabled": True,
        "by_isotope": {"Cs-137": {"scale": 2.0}},
    }
    base = DiscreteAttenuationKernel.from_environment(
        env=env,
        isotopes=["Cs-137"],
        mu_by_isotope=None,
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
        obstacle_grid=None,
        config=KempKernelConfig(
            grid_spacing_m=(1.0, 1.0, 1.0),
            grid_margin_m=0.5,
            z_levels_m=(0.5,),
            use_gpu=False,
        ),
    )
    adjusted = DiscreteAttenuationKernel.from_environment(
        env=env,
        isotopes=["Cs-137"],
        mu_by_isotope=None,
        shield_params=ShieldParams(thickness_fe_cm=0.0, thickness_pb_cm=0.0),
        obstacle_grid=None,
        config=KempKernelConfig(
            grid_spacing_m=(1.0, 1.0, 1.0),
            grid_margin_m=0.5,
            z_levels_m=(0.5,),
            transport_response_model=transport_model,
            use_gpu=False,
        ),
    )

    detector = (1.0, 1.0, 0.5)
    base_vector = base.kernel_vector("Cs-137", detector, 0, 0)
    adjusted_vector = adjusted.kernel_vector("Cs-137", detector, 0, 0)

    assert adjusted._kernel.transport_response_model == transport_model
    assert np.allclose(adjusted_vector, 2.0 * base_vector)


def test_kemp_cpu_parallel_kernel_matches_serial() -> None:
    """CPU-worker kernel evaluation should preserve the serial response."""
    serial = _build_test_kernel(cpu_workers=1)
    parallel = _build_test_kernel(cpu_workers=2)
    detector = (1.0, 1.0, 0.5)

    serial_theta = serial.kernel_vector("Cs-137", detector, 0, 0)
    parallel_theta = parallel.kernel_vector("Cs-137", detector, 0, 0)

    assert np.allclose(serial_theta, parallel_theta)


def test_kemp_empty_obstacle_grid_disables_config_usd_fallback() -> None:
    """Kemp reset payload should not load demo obstacles for an empty grid."""
    env = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=3.0, detector_position=(1.0, 1.0, 0.5))
    sources = [PointSource("Cs-137", position=(2.0, 3.0, 1.0), intensity_cps_1m=1000.0)]
    empty_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(10, 20),
        blocked_cells=(),
    )

    payload = _scene_reset_payload(
        env=env,
        sources=sources,
        obstacle_grid=empty_grid,
        runtime_config={"obstacle_material": "air"},
    )

    assert payload["usd_path"] == ""
    assert payload["use_config_usd_fallback"] is False
    assert payload["obstacle_material"] == "air"
    assert payload["obstacle_cells"] == []


def test_kemp_filter_updates_toward_synthetic_source() -> None:
    """Verify that one isotope filter can localize a strong synthetic source."""
    kernel = _build_test_kernel()
    true_idx = kernel.nearest_index((2.5, 2.5, 0.5))
    true_strength = 20000.0
    config = KempFilterConfig(
        num_particles=500,
        max_sources=1,
        init_source_count_min=1,
        init_source_count_max=1,
        init_strength_log_mean=float(np.log(true_strength)),
        init_strength_log_sigma=0.4,
        p_birth=0.0,
        p_death=0.0,
        p_move=0.4,
        rng_seed=42,
    )
    from baselines.kemp.filter import KempLogDDPF

    filt = KempLogDDPF(isotope="Cs-137", kernel=kernel, config=config)
    poses = [(1.0, 1.0, 0.5), (3.0, 1.0, 0.5), (1.0, 3.0, 0.5), (3.0, 3.0, 0.5)]
    for pose in poses * 4:
        expected = kernel.expected_counts(
            isotope="Cs-137",
            detector_pos=pose,
            source_indices=[true_idx],
            strengths=[true_strength],
            live_time_s=10.0,
            fe_index=0,
            pb_index=0,
        )
        filt.update(
            KempMeasurement(
                detector_pos=pose,
                live_time_s=10.0,
                counts=expected,
                variance=max(expected, 1.0),
            )
        )
    positions, _, _ = filt.estimate_sources()
    assert positions.shape[0] == 1
    assert float(np.linalg.norm(positions[0] - kernel.source_grid[true_idx])) <= 1.5


def test_kemp_gpu_particle_rates_runs_when_cuda_available() -> None:
    """The Kemp filter should expose an explicit CUDA update path."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    kernel = _build_test_kernel()
    config = KempFilterConfig(
        num_particles=8,
        max_sources=1,
        init_source_count_min=1,
        init_source_count_max=1,
        use_gpu=True,
        rng_seed=11,
    )
    from baselines.kemp.filter import KempLogDDPF

    filt = KempLogDDPF(isotope="Cs-137", kernel=kernel, config=config)
    theta = kernel.kernel_vector("Cs-137", (1.0, 1.0, 0.5), 0, 0)
    rates = filt._particle_rates(theta)

    assert rates.shape == (config.num_particles,)
    assert np.all(np.isfinite(rates))


def test_parallel_mixing_removes_unsupported_isotope() -> None:
    """Verify the mixing step removes an isotope with only zero observations."""
    kernel = _build_test_kernel()
    config = KempParallelConfig(
        filter_config=KempFilterConfig(
            num_particles=200,
            max_sources=1,
            init_source_count_min=1,
            init_source_count_max=1,
            init_strength_log_mean=float(np.log(10000.0)),
            init_strength_log_sigma=0.2,
            p_birth=0.0,
            p_death=0.0,
            rng_seed=7,
        ),
        output_min_strength_cps_1m=50.0,
        mixing_min_expected_counts=1.0,
        update_workers=2,
    )
    estimator = KempParallelLogDDPF(isotopes=["Cs-137", "Co-60"], kernel=kernel, config=config)
    for _ in range(4):
        estimator.update(
            detector_pos=(1.0, 1.0, 0.5),
            live_time_s=5.0,
            counts_by_isotope={"Cs-137": 100.0, "Co-60": 0.0},
            variances_by_isotope={"Cs-137": 100.0, "Co-60": 1.0},
        )
    estimates = estimator.estimates_for_metrics()
    assert estimates["Co-60"] == []
