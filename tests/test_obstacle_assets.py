"""Tests for known Manchester-style obstacle assets."""

from __future__ import annotations

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacle_assets import (
    generate_manchester_obstacle_instances,
    known_obstacle_transport_model,
    known_obstacle_traversability_rects,
    material_mu_cm_inv,
)
from measurement.obstacles import ObstacleGrid


def test_gpu_obstacle_tau_chunks_known_components() -> None:
    """GPU obstacle attenuation should support many material-specific boxes."""
    torch = __import__("pytest").importorskip("torch")
    from pf import gpu_utils

    device = torch.device("cpu")
    dtype = torch.float64
    source = torch.as_tensor([[[0.0, 0.5, 0.5]]], device=device, dtype=dtype)
    target = torch.as_tensor([[[4.0, 0.5, 0.5]]], device=device, dtype=dtype)
    boxes = torch.as_tensor(
        [
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.5, 0.0, 0.0, 2.0, 1.0, 1.0],
            [2.5, 0.0, 0.0, 3.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    mu_values = torch.as_tensor([0.1, 0.2, 0.3], device=device, dtype=dtype)

    tau = gpu_utils._obstacle_tau_between_points_cm_torch(
        source,
        target,
        boxes,
        0.0,
        mu_values,
        device=device,
        dtype=dtype,
        tol=1.0e-9,
        chunk_size=1,
    )

    assert tau.item() == __import__("pytest").approx(30.0)


def test_manchester_assets_provide_hollow_transport_components() -> None:
    """Generated obstacle assets should separate motion footprints and transport boxes."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0), (1, 1)),
    )
    instances = generate_manchester_obstacle_instances(
        grid,
        room_size_xyz=(2.0, 2.0, 3.0),
        obstacle_height_m=2.0,
        rng_seed=4,
    )
    boxes_m, mu_by_isotope = known_obstacle_transport_model(instances)
    rects = known_obstacle_traversability_rects(instances)

    assert len(instances) == len(grid.blocked_cells)
    assert len(boxes_m) > len(grid.blocked_cells)
    assert len(rects) == len(grid.blocked_cells)
    assert set(mu_by_isotope) >= {"Cs-137", "Co-60", "Eu-154"}
    assert len(mu_by_isotope["Cs-137"]) == len(boxes_m)


def test_material_mu_uses_known_material_presets() -> None:
    """Material attenuation coefficients should depend on known material identity."""
    steel_mu = material_mu_cm_inv("steel", "Cs-137")
    water_mu = material_mu_cm_inv("water", "Cs-137")

    assert steel_mu > water_mu
    assert steel_mu > 0.0
    assert water_mu > 0.0


def test_continuous_kernel_uses_known_transport_components() -> None:
    """PF expected-count attenuation should use per-component obstacle materials."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.4, 0.0, 0.0, 0.6, 1.0, 1.0),),
        mu_by_isotope={"Cs-137": (0.5,)},
    )
    kernel = ContinuousKernel(obstacle_grid=grid)

    attenuation = kernel._obstacle_attenuation_factor(
        "Cs-137",
        np.asarray((0.5, -1.0, 0.5), dtype=float),
        np.asarray((0.5, 2.0, 0.5), dtype=float),
    )

    assert attenuation < 1.0
    expected = float(np.exp(-50.0))
    assert abs(attenuation / expected - 1.0) < 1.0e-6
