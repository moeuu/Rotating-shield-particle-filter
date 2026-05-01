"""Tests for continuous 3D kernel evaluation (Sec. 3.2–3.3)."""

import numpy as np
import pytest

from measurement.continuous_kernels import (
    ContinuousKernel,
    finite_sphere_geometric_term,
    geometric_term,
)
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from measurement.continuous_kernels import expected_counts_single_isotope
from pf.gpu_utils import expected_counts_pair_torch


def test_geometric_term_inverse_square() -> None:
    """Geometric term should follow 1/d^2 scaling."""
    det = np.array([0.0, 0.0, 0.0])
    s1 = np.array([1.0, 0.0, 0.0])
    s2 = np.array([2.0, 0.0, 0.0])
    g1 = geometric_term(det, s1)
    g2 = geometric_term(det, s2)
    assert np.isclose(g1 / g2, 4.0, rtol=1e-6)


def test_finite_sphere_geometric_term_preserves_one_meter_definition() -> None:
    """Finite detector geometry should remove the near-field point singularity."""
    det = np.array([0.0, 0.0, 0.0], dtype=float)
    radius = 0.04

    at_one_meter = finite_sphere_geometric_term(
        det,
        np.array([1.0, 0.0, 0.0], dtype=float),
        radius,
    )
    at_two_meters = finite_sphere_geometric_term(
        det,
        np.array([2.0, 0.0, 0.0], dtype=float),
        radius,
    )
    at_center = finite_sphere_geometric_term(det, det, radius)

    assert at_one_meter == pytest.approx(1.0)
    assert at_two_meters == pytest.approx(0.25, rel=1.0e-3)
    assert np.isfinite(at_center)
    assert at_center < 1.0e4


def test_attenuation_applies_blocking_factor() -> None:
    """Blocked orientation should reduce expected counts by exp(-mu*L)."""
    shield_params = ShieldParams()
    kernel = ContinuousKernel(shield_params=shield_params, use_gpu=False)
    orientations = generate_octant_orientations()
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([1.0, 1.0, 1.0])
    strengths = np.array([10.0])

    # Vector from src->det is (-,-,-), so orient 7 blocks, orient 0 unblocks
    blocked_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=7,
        live_time_s=1.0,
        background=0.0,
    )
    free_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=0,
        live_time_s=1.0,
        background=0.0,
    )
    expected_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    assert np.isclose(blocked_counts, expected_ratio * free_counts, rtol=1e-6)


def test_background_added_to_expected_counts() -> None:
    """Background should add directly to expected rate/counts."""
    kernel = ContinuousKernel()
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([10.0, 0.0, 0.0])
    counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=np.array([0.0]),
        orient_idx=0,
        live_time_s=2.0,
        background=5.0,
    )
    assert np.isclose(counts, 10.0, rtol=1e-6)


def test_expected_counts_single_isotope_attenuation_levels() -> None:
    """Fe/Pb blocking should scale expected counts via exp(-mu*L)."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strengths = np.array([10.0])
    # Orientation normal aligned with direction (-,-,-) from src to det
    orient_block = np.array([-1.0, -1.0, -1.0])
    orient_free = np.array([1.0, 1.0, 1.0])
    base = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_free,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    shield_params = ShieldParams()
    expected_fe_ratio = np.exp(-(shield_params.mu_fe * shield_params.thickness_fe_cm))
    expected_both_ratio = np.exp(
        -(shield_params.mu_fe * shield_params.thickness_fe_cm + shield_params.mu_pb * shield_params.thickness_pb_cm)
    )
    fe_only = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    both = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_block,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        use_gpu=False,
    )
    assert np.isclose(fe_only, expected_fe_ratio * base, rtol=1e-6)
    assert np.isclose(both, expected_both_ratio * base, rtol=1e-6)


def test_expected_counts_pair_matches_single_helper() -> None:
    """expected_counts_pair should match expected_counts_single_isotope with corresponding RFe/RPb."""
    kernel = ContinuousKernel()
    det = np.array([1.0, 1.0, 1.0])
    sources = np.array([[0.0, 0.0, 0.0]])
    strengths = np.array([1.0])
    fe_idx = 0  # (+,+,+) blocks
    pb_idx = 7  # (-,-,-) free
    pair_counts = kernel.expected_counts_pair(
        isotope="Cs-137",
        detector_pos=det,
        sources=sources,
        strengths=strengths,
        fe_index=fe_idx,
        pb_index=pb_idx,
        live_time_s=1.0,
        background=0.0,
    )
    from measurement.shielding import generate_octant_rotation_matrices

    mats = generate_octant_rotation_matrices()
    single = expected_counts_single_isotope(
        detector_position=det,
        RFe=mats[fe_idx],
        RPb=mats[pb_idx],
        sources=sources,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
        kernel=kernel,
    )
    assert pair_counts == pytest.approx(single, rel=1e-12)


def test_concrete_obstacle_path_reduces_kernel_value() -> None:
    """A blocked concrete cell should attenuate the source-detector kernel by its path length."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    shield_params = ShieldParams(mu_fe=0.0, mu_pb=0.0)
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    assert kernel.obstacle_path_length_cm(source, detector) == pytest.approx(100.0)
    blocked = kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
    unblocked = free_kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
    assert blocked == pytest.approx(unblocked * np.exp(-1.0), rel=1e-12)


def test_broad_beam_buildup_increases_but_bounds_attenuated_counts() -> None:
    """Build-up should increase attenuated broad-beam counts without exceeding unattenuated counts."""
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    base_params = ShieldParams(mu_fe=0.1, mu_pb=0.0, thickness_fe_cm=5.0, thickness_pb_cm=0.0)
    narrow_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=base_params,
        use_gpu=False,
    )
    buildup_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.1,
            mu_pb=0.0,
            thickness_fe_cm=5.0,
            thickness_pb_cm=0.0,
            buildup_fe_coeff=0.5,
        ),
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=False,
    )

    narrow = narrow_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)
    buildup = buildup_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)
    free = free_kernel.kernel_value_pair("Cs-137", detector, source, 7, 0)

    assert buildup > narrow
    assert buildup <= free


def test_spherical_shell_path_uses_radial_overlap_near_detector() -> None:
    """Shield path length should use exact radial overlap with the spherical shell."""
    shield_params = ShieldParams(
        mu_fe=0.1,
        mu_pb=0.0,
        thickness_fe_cm=5.0,
        thickness_pb_cm=0.0,
        inner_radius_fe_cm=19.0,
        inner_radius_pb_cm=26.0,
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=shield_params,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    direction = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    source = direction * 0.205

    attenuation = kernel.attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
        fe_index=7,
        pb_index=0,
    )
    assert attenuation == pytest.approx(np.exp(-0.1 * 1.5), rel=1e-12)


def test_concrete_obstacle_misses_off_axis_ray() -> None:
    """Obstacle attenuation should be unity when the ray does not cross blocked cells."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    source = np.array([-1.0, 2.0, 1.0], dtype=float)
    detector = np.array([2.0, 2.0, 1.0], dtype=float)

    assert kernel.obstacle_path_length_cm(source, detector) == pytest.approx(0.0)
    assert kernel.attenuation_factor_pair("Cs-137", source, detector, 0, 0) == pytest.approx(1.0)


def test_gpu_expected_counts_include_obstacle_attenuation_on_cpu_device() -> None:
    """The torch expected-count path should apply the same obstacle attenuation."""
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor([[[-1.0, 0.0, 1.0]]], device=device, dtype=dtype)
    strengths = torch.as_tensor([[9.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones(1, 1, device=device, dtype=dtype)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)
    boxes = np.array([[0.0, -0.5, 0.0, 1.0, 0.5, 2.0]], dtype=float)

    counts = expected_counts_pair_torch(
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=0,
        pb_index=0,
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=0.0,
        thickness_pb_cm=0.0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        obstacle_boxes_m=boxes,
        obstacle_mu_cm_inv=0.01,
    )
    expected = (9.0 / 9.0) * np.exp(-1.0)
    assert float(counts[0]) == pytest.approx(expected, rel=1e-12)


def test_gpu_expected_counts_use_exact_spherical_shell_overlap() -> None:
    """The torch expected-count path should match exact spherical-shell path length."""
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    dtype = torch.float64
    direction = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    source = direction * 0.205
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.ones(1, 1, device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones(1, 1, device=device, dtype=dtype)

    counts = expected_counts_pair_torch(
        detector_pos=np.zeros(3, dtype=float),
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=7,
        pb_index=0,
        mu_fe=0.1,
        mu_pb=0.0,
        thickness_fe_cm=5.0,
        thickness_pb_cm=0.0,
        inner_radius_fe_cm=19.0,
        inner_radius_pb_cm=26.0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )
    expected = (1.0 / (0.205**2)) * np.exp(-0.15)
    assert float(counts[0]) == pytest.approx(expected, rel=1e-12)


def test_continuous_kernel_cuda_matches_cpu_with_detector_aperture() -> None:
    """ContinuousKernel CUDA path should match the CPU finite-aperture geometry."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    rng = np.random.default_rng(77)
    detector = np.array([0.2, -0.3, 1.0], dtype=float)
    directions = rng.normal(size=(16, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = rng.uniform(0.8, 3.5, size=(16, 1))
    sources = detector + directions * distances
    strengths = rng.uniform(500.0, 30000.0, size=16)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )

    for fe_index in range(8):
        for pb_index in range(8):
            cpu_counts = cpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
                live_time_s=1.0,
            )
            gpu_counts = gpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
                live_time_s=1.0,
            )
            assert gpu_counts == pytest.approx(cpu_counts, rel=1e-10, abs=1e-10)


def test_continuous_kernel_cuda_matches_cpu_with_obstacles_and_aperture() -> None:
    """ContinuousKernel CUDA path should match CPU obstacle attenuation over aperture rays."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detector = np.array([2.2, 2.2, 1.0], dtype=float)
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    strengths = np.array([10000.0, 20000.0, 15000.0, 12000.0], dtype=float)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=31,
    )

    for fe_index in range(8):
        for pb_index in range(8):
            cpu_counts = cpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
            )
            gpu_counts = gpu_kernel.expected_counts_pair(
                "Cs-137",
                detector,
                sources,
                strengths,
                fe_index,
                pb_index,
            )
            assert gpu_counts == pytest.approx(cpu_counts, rel=1e-10, abs=1e-10)
