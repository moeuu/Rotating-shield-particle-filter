"""Tests for continuous 3D kernel evaluation (Sec. 3.2–3.3)."""

import numpy as np
import pytest

from measurement.continuous_kernels import (
    ContinuousKernel,
    finite_sphere_geometric_term,
    geometric_term,
    segment_rotated_octant_shell_path_length_cm_torch,
)
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
)
from measurement.continuous_kernels import expected_counts_single_isotope


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"shield_geometry_model": "nominal_tvl"}, ValueError),
        ({"shield_geometry_model": "spherical_octant"}, ValueError),
        ({"use_angle_attenuation": True}, ValueError),
        ({"use_angle_attenuation": 1}, TypeError),
        ({"thickness_fe_cm": "2.0"}, TypeError),
        ({"thickness_pb_cm": -1.0}, ValueError),
    ],
)
def test_shield_params_reject_retired_or_coerced_physics(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Shield parameters must expose only the shared Geant4 geometry."""
    with pytest.raises(error_type):
        ShieldParams(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"detector_radius_m": "0.04"}, TypeError),
        ({"detector_radius_m": -0.04}, ValueError),
        ({"detector_aperture_samples": 1.5}, TypeError),
        ({"detector_aperture_samples": 0}, ValueError),
        ({"detector_aperture_sampling": "cone"}, ValueError),
        ({"source_extent_radius_m": 0.05, "source_extent_samples": 1}, ValueError),
        ({"use_gpu": 1}, TypeError),
    ],
)
def test_continuous_kernel_rejects_implicit_runtime_coercions(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Continuous physics configuration must fail before inference changes."""
    with pytest.raises(error_type):
        ContinuousKernel(**kwargs)


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


def test_torch_rotated_octant_shell_cached_rotation_matches_direct() -> None:
    """Cached torch octant rotations should preserve exact path-length results."""
    torch = pytest.importorskip("torch")
    kernel = ContinuousKernel(use_gpu=True, gpu_device="cpu", gpu_dtype="float64")
    device = torch.device("cpu")
    dtype = torch.float64
    source_pos = torch.as_tensor(
        [[1.0, 1.0, 1.0], [-1.0, 1.5, 0.5]],
        device=device,
        dtype=dtype,
    )
    target_pos = torch.zeros_like(source_pos)
    center_pos = torch.zeros(1, 3, device=device, dtype=dtype)
    shield_normal = -np.asarray(kernel.orientations[0], dtype=float)
    rotation = kernel._rotated_octant_rotation_torch(
        shield_normal,
        device=device,
        dtype=dtype,
    )

    direct = segment_rotated_octant_shell_path_length_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        center_pos=center_pos,
        shield_normal=shield_normal,
        inner_radius_cm=1.0,
        outer_radius_cm=20.0,
    )
    cached = segment_rotated_octant_shell_path_length_cm_torch(
        source_pos=source_pos,
        target_pos=target_pos,
        center_pos=center_pos,
        shield_normal=None,
        inner_radius_cm=1.0,
        outer_radius_cm=20.0,
        rotation=rotation,
    )

    assert torch.allclose(cached, direct, rtol=1e-12, atol=1e-12)
    assert len(kernel._torch_octant_rotation_cache) == 1


def test_attenuation_applies_blocking_factor() -> None:
    """Blocked orientation should reduce expected counts by exp(-mu*L)."""
    shield_params = ShieldParams()
    kernel = ContinuousKernel(shield_params=shield_params, use_gpu=False)
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
        -(
            shield_params.mu_fe * shield_params.thickness_fe_cm
            + shield_params.mu_pb * shield_params.thickness_pb_cm
        )
    )
    assert np.isclose(blocked_counts, expected_ratio * free_counts, rtol=1e-6)


def test_line_resolved_attenuation_uses_weighted_transmission() -> None:
    """ContinuousKernel should average shield transmission over gamma lines."""
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
    )
    line_mu = {
        "TestIso": (
            {"weight": 1.0, "fe": 0.10, "pb": 0.0},
            {"weight": 3.0, "fe": 0.30, "pb": 0.0},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([[1.0, 1.0, 1.0]], dtype=float)
    strength = np.array([100.0], dtype=float)

    blocked = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source,
        strength,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )
    free = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source,
        strength,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    expected_ratio = 0.25 * np.exp(-0.10 * 2.0) + 0.75 * np.exp(-0.30 * 2.0)

    assert blocked == pytest.approx(free * expected_ratio, rel=1e-12)


def test_line_resolved_obstacle_attenuation_uses_line_mu_values() -> None:
    """ContinuousKernel should average obstacle transmission over gamma lines."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.0, -0.5, 0.0, 1.0, 0.5, 2.0),),
        mu_by_isotope={"TestIso": (0.0,)},
        line_mu_by_isotope={"TestIso": ((0.01,), (0.03,))},
    )
    line_mu = {
        "TestIso": (
            {"weight": 1.0, "fe": 0.0, "pb": 0.0},
            {"weight": 3.0, "fe": 0.0, "pb": 0.0},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    blocked = kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    free = free_kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    expected_ratio = 0.25 * np.exp(-1.0) + 0.75 * np.exp(-3.0)

    assert kernel.obstacle_path_lengths_by_box_cm(source, detector)[0] == pytest.approx(
        100.0
    )
    assert blocked == pytest.approx(free * expected_ratio, rel=1e-12)


def _line_resolved_full_physics_kernel(*, use_gpu: bool) -> ContinuousKernel:
    """Return a small line-resolved kernel exercising every transport term."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.2, 0.2, 0.0, 0.8, 0.8, 2.0),),
        mu_by_isotope={"TestIso": (0.015,)},
        line_mu_by_isotope={
            "TestIso": ((0.01,), (0.025,), (0.04,))
        },
    )
    return ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=1.5,
            buildup_fe_coeff=0.05,
            buildup_pb_coeff=0.03,
        ),
        obstacle_grid=grid,
        obstacle_buildup_coeff=0.02,
        detector_radius_m=0.04,
        detector_aperture_radius_m=0.025,
        detector_aperture_samples=5,
        source_extent_radius_m=0.05,
        source_extent_samples=3,
        line_mu_by_isotope={
            "TestIso": (
                {"weight": 0.2, "fe": 0.04, "pb": 0.07},
                {"weight": 0.3, "fe": 0.08, "pb": 0.11},
                {"weight": 0.5, "fe": 0.13, "pb": 0.18},
            )
        },
        use_gpu=use_gpu,
        gpu_device="cpu",
        gpu_dtype="float64",
    )


def _line_resolved_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return deterministic detector/source/pair arrays for line API tests."""
    return (
        np.asarray([[0.0, 0.0, 1.0], [1.8, 1.7, 1.2]], dtype=np.float64),
        np.asarray(
            [[1.0, 1.0, 1.0], [-0.8, 0.4, 0.7], [2.4, 0.2, 1.4]],
            dtype=np.float64,
        ),
        np.asarray([0, 3], dtype=np.int64),
        np.asarray([7, 2], dtype=np.int64),
    )


def test_matched_row_kernel_apis_match_cpu_and_torch_batches() -> None:
    """Sparse matched-row APIs must preserve the serial physical kernel."""
    pytest.importorskip("torch")
    detectors = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.8, 1.7, 1.2],
            [-0.5, 0.3, 0.8],
        ],
        dtype=np.float64,
    )
    sources = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-0.8, 0.4, 0.7],
            [2.4, 0.2, 1.4],
        ],
        dtype=np.float64,
    )
    fe_indices = np.asarray([0, 3, 7], dtype=np.int64)
    pb_indices = np.asarray([7, 2, 0], dtype=np.int64)
    cpu = _line_resolved_full_physics_kernel(use_gpu=False)
    torch_cpu = _line_resolved_full_physics_kernel(use_gpu=True)

    selected_serial = np.asarray(
        [
            cpu.kernel_value_pair(
                "TestIso",
                detector,
                source,
                int(fe_index),
                int(pb_index),
            )
            for detector, source, fe_index, pb_index in zip(
                detectors,
                sources,
                fe_indices,
                pb_indices,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    selected_cpu = cpu.kernel_values_selected_pairs_for_detector_source_pairs(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        chunk_size=1,
    )
    selected_torch = (
        torch_cpu.kernel_values_selected_pairs_for_detector_source_pairs(
            "TestIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            chunk_size=2,
        )
    )
    unshielded_cpu = (
        cpu.kernel_values_unshielded_for_detector_source_pairs(
            "TestIso",
            detectors,
            sources,
            chunk_size=1,
        )
    )
    unshielded_torch = (
        torch_cpu.kernel_values_unshielded_for_detector_source_pairs(
            "TestIso",
            detectors,
            sources,
            chunk_size=2,
        )
    )

    assert selected_cpu == pytest.approx(
        selected_serial,
        rel=5.0e-13,
        abs=5.0e-13,
    )
    assert selected_torch == pytest.approx(
        selected_cpu,
        rel=5.0e-12,
        abs=5.0e-12,
    )
    assert unshielded_torch == pytest.approx(
        unshielded_cpu,
        rel=5.0e-12,
        abs=5.0e-12,
    )


def test_line_kernel_shape_order_and_weighted_aggregate_identity() -> None:
    """Selected line means must be distinct and reconstruct the full kernel."""
    kernel = _line_resolved_full_physics_kernel(use_gpu=False)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    all_indices = np.asarray([0, 1, 2], dtype=np.int64)

    line_values = kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        all_indices,
        chunk_size=2,
    )
    aggregate = kernel.kernel_values_selected_pairs_for_detectors(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        chunk_size=2,
    )
    weights = kernel.line_branching_weights("TestIso", all_indices)

    assert line_values.shape == (2, 3, 3)
    np.testing.assert_allclose(
        np.einsum("psl,l->ps", line_values, weights),
        aggregate,
        rtol=3.0e-13,
        atol=3.0e-14,
    )
    assert not np.allclose(line_values[..., 0], line_values[..., 2])
    reversed_values = (
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            "TestIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            np.asarray([2, 0], dtype=np.int64),
        )
    )
    np.testing.assert_array_equal(reversed_values[..., 0], line_values[..., 2])
    np.testing.assert_array_equal(reversed_values[..., 1], line_values[..., 0])


def test_line_kernel_returns_source_equivalent_means_before_branching() -> None:
    """An unattenuated line estimator must predict source count, not line count."""
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=0.0,
            thickness_pb_cm=0.0,
        ),
        line_mu_by_isotope={
            "TestIso": (
                {"weight": 0.1, "fe": 0.0, "pb": 0.0},
                {"weight": 0.9, "fe": 0.0, "pb": 0.0},
            )
        },
        use_gpu=False,
    )
    values = kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(values, np.ones((1, 1, 2)), rtol=0.0, atol=1.0e-15)


def test_line_kernel_cpu_and_torch_float64_are_equivalent() -> None:
    """Torch float64 batching must match NumPy for every selected line."""
    pytest.importorskip("torch")
    cpu = _line_resolved_full_physics_kernel(use_gpu=False)
    torch_kernel = _line_resolved_full_physics_kernel(use_gpu=True)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    indices = np.asarray([0, 2], dtype=np.int64)

    expected = cpu.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        indices,
        chunk_size=2,
    )
    actual = torch_kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        indices,
        chunk_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-12)


def test_line_transport_components_preserve_existing_total_kernel() -> None:
    """Spectral components must preserve the production scalar line response."""
    kernel = _line_resolved_full_physics_kernel(use_gpu=False)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    indices = np.asarray([0, 1, 2], dtype=np.int64)

    expected = kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        indices,
        chunk_size=2,
    )
    components = (
        kernel.line_transport_components_selected_pairs_for_detectors(
            "TestIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            indices,
            chunk_size=2,
        )
    )

    np.testing.assert_array_equal(components.total_kernel, expected)
    assert components.total_kernel.shape == (2, 3, 3)
    assert np.all(components.uncollided_kernel >= 0.0)
    total_tau = (
        components.tau_fe
        + components.tau_pb
        + components.tau_obstacle
    )
    assert np.any(total_tau > 0.0)
    assert np.all(components.distance_m > 0.0)


def test_line_transport_components_cpu_and_torch_are_equivalent() -> None:
    """Every spectral transport feature must share the CPU/GPU physics."""
    pytest.importorskip("torch")
    cpu = _line_resolved_full_physics_kernel(use_gpu=False)
    torch_kernel = _line_resolved_full_physics_kernel(use_gpu=True)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    indices = np.asarray([0, 2], dtype=np.int64)

    expected = cpu.line_transport_components_selected_pairs_for_detectors(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        indices,
        chunk_size=2,
    )
    actual = (
        torch_kernel.line_transport_components_selected_pairs_for_detectors(
            "TestIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            indices,
            chunk_size=2,
        )
    )

    for field_name in (
        "total_kernel",
        "uncollided_kernel",
        "tau_fe",
        "tau_pb",
        "tau_obstacle",
        "distance_m",
    ):
        np.testing.assert_allclose(
            getattr(actual, field_name),
            getattr(expected, field_name),
            rtol=2.0e-10,
            atol=2.0e-12,
        )


@pytest.mark.parametrize(
    ("indices", "error_type", "message"),
    [
        (np.asarray([], dtype=np.int64), ValueError, "non-empty"),
        (np.asarray([0.5]), ValueError, "exact integer"),
        (np.asarray([0, 0]), ValueError, "duplicates"),
        (np.asarray([3]), IndexError, "outside"),
        (np.asarray([-1]), IndexError, "outside"),
    ],
)
def test_line_kernel_rejects_invalid_positive_line_indices(
    indices: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    """The line event basis must fail fast instead of being reindexed."""
    kernel = _line_resolved_full_physics_kernel(use_gpu=False)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()

    with pytest.raises(error_type, match=message):
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            "TestIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            indices,
        )


def test_line_kernel_rejects_missing_basis_and_invalid_pair_shape() -> None:
    """Missing physics metadata and pair-array mismatches must not fall back."""
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    missing = ContinuousKernel(use_gpu=False)
    with pytest.raises(ValueError, match="No positive line-resolved"):
        missing.kernel_values_selected_pairs_for_detectors_by_line(
            "MissingIso",
            detectors,
            sources,
            fe_indices,
            pb_indices,
            np.asarray([0], dtype=np.int64),
        )

    kernel = _line_resolved_full_physics_kernel(use_gpu=False)
    with pytest.raises(ValueError, match="matching lengths"):
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            "TestIso",
            detectors,
            sources,
            np.asarray([0], dtype=np.int64),
            pb_indices,
            np.asarray([0], dtype=np.int64),
        )
    with pytest.raises(IndexError, match="orientation index"):
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            "TestIso",
            detectors,
            sources,
            np.asarray([0, 8], dtype=np.int64),
            pb_indices,
            np.asarray([0], dtype=np.int64),
        )


@pytest.mark.parametrize("use_gpu", (False, True))
@pytest.mark.parametrize("bad_index", (-1, 8))
def test_selected_pair_kernels_reject_wrapped_orientation_indices(
    use_gpu: bool,
    bad_index: int,
) -> None:
    """CPU, Torch, and line kernels must not reinterpret corrupt pair IDs."""
    if use_gpu:
        pytest.importorskip("torch")
    kernel = _line_resolved_full_physics_kernel(use_gpu=use_gpu)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    invalid_fe = fe_indices.copy()
    invalid_fe[0] = int(bad_index)

    with pytest.raises(IndexError, match=r"must lie in \[0, 8\)"):
        kernel.kernel_values_selected_pairs_for_detectors(
            "TestIso",
            detectors,
            sources,
            invalid_fe,
            pb_indices,
        )
    with pytest.raises(IndexError, match=r"must lie in \[0, 8\)"):
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            "TestIso",
            detectors,
            sources,
            invalid_fe,
            pb_indices,
            np.asarray([0, 2], dtype=np.int64),
        )


def test_line_kernel_exact_input_cache_avoids_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated exact requests must reuse physics while protecting cache data."""
    kernel = _line_resolved_full_physics_kernel(use_gpu=False)
    detectors, sources, fe_indices, pb_indices = _line_resolved_inputs()
    original = (
        kernel._kernel_values_selected_pairs_for_detector_source_numpy_chunk
    )
    calls = 0

    def tracked(*args: object, **kwargs: object) -> np.ndarray:
        """Count production NumPy line batches and delegate unchanged."""
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        kernel,
        "_kernel_values_selected_pairs_for_detector_source_numpy_chunk",
        tracked,
    )
    indices = np.asarray([0, 1], dtype=np.int64)
    first = kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        indices,
        chunk_size=100,
    )
    first[0, 0, 0] = -1.0
    second = kernel.kernel_values_selected_pairs_for_detectors_by_line(
        "TestIso",
        detectors.copy(),
        sources.copy(),
        fe_indices.copy(),
        pb_indices.copy(),
        indices.copy(),
        chunk_size=1,
    )

    assert calls == 1
    assert kernel.line_response_cache_misses == 1
    assert kernel.line_response_cache_hits == 1
    assert second[0, 0, 0] >= 0.0
    kernel.clear_line_response_cache()
    assert kernel.line_response_cache_hits == 0
    assert kernel.line_response_cache_misses == 0


def test_cpu_attenuation_uses_explicit_aperture_radius_without_count_radius() -> None:
    """CPU attenuation should sample aperture rays independently of count radius."""
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.2, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.2,
            mu_pb=0.0,
            thickness_fe_cm=4.0,
            thickness_pb_cm=0.0,
            inner_radius_fe_cm=3.0,
        ),
        detector_radius_m=0.0,
        detector_aperture_radius_m=0.08,
        detector_aperture_samples=17,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([0.2, 0.2, 0.2], dtype=float)
    targets = kernel._detector_aperture_targets(source, detector)
    center_attenuation = kernel._attenuation_factor_for_target(
        "Cs-137",
        source,
        detector,
        detector,
        fe_index=7,
        pb_index=0,
    )
    aperture_attenuation = float(
        np.mean(
            [
                kernel._attenuation_factor_for_target(
                    "Cs-137",
                    source,
                    target,
                    detector,
                    fe_index=7,
                    pb_index=0,
                )
                for target in targets
            ]
        )
    )

    assert aperture_attenuation != pytest.approx(center_attenuation)
    assert kernel.attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
        fe_index=7,
        pb_index=0,
    ) == pytest.approx(aperture_attenuation, rel=1.0e-12)


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


def test_detector_cone_aperture_targets_match_outer_sphere() -> None:
    """Cone aperture targets should match the Geant4 detector-cone geometry."""
    detector = np.array([0.0, 0.0, 0.0], dtype=float)
    source = np.array([2.0, 0.3, -0.4], dtype=float)
    aperture_radius = 0.052
    kernel = ContinuousKernel(
        detector_radius_m=0.05,
        detector_aperture_radius_m=aperture_radius,
        detector_aperture_samples=17,
        detector_aperture_sampling="solid_angle_cone",
        use_gpu=False,
    )

    targets = kernel._detector_aperture_targets(source, detector)
    target_radii = np.linalg.norm(targets - detector, axis=1)
    ray_dirs = targets - source
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    axis = detector - source
    axis /= np.linalg.norm(axis)
    ray_angles = np.arccos(np.clip(ray_dirs @ axis, -1.0, 1.0))
    max_angle = np.arcsin(aperture_radius / np.linalg.norm(detector - source))

    assert targets.shape == (17, 3)
    assert np.allclose(target_radii, aperture_radius, rtol=0.0, atol=1.0e-10)
    assert float(np.max(ray_angles)) <= float(max_angle) + 1.0e-10
    assert np.linalg.matrix_rank(targets - targets.mean(axis=0)) == 3


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
        -(
            shield_params.mu_fe * shield_params.thickness_fe_cm
            + shield_params.mu_pb * shield_params.thickness_pb_cm
        )
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


def test_collision_boxes_replace_grid_columns_as_pf_attenuation_fallback() -> None:
    """PF counts should use the exact physical AABB before coarse blocked cells."""
    collision_box = (0.0, -0.5, 0.0, 1.0, 0.5, 2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
        collision_boxes_m=(collision_box,),
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    blocked = kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)
    free = free_kernel.kernel_value_pair("Cs-137", detector, source, 0, 0)

    assert kernel.obstacle_boxes_m() == pytest.approx(np.asarray([collision_box]))
    assert kernel.obstacle_path_length_cm(source, detector) == pytest.approx(100.0)
    assert blocked == pytest.approx(free * np.exp(-1.0), rel=1e-12)


def test_explicit_transport_model_prevents_pf_collision_double_count() -> None:
    """PF attenuation should exclusively use explicit boxes and per-box isotope mu."""
    collision_box = (0.0, -0.5, 0.0, 1.0, 0.5, 2.0)
    transport_box = (0.0, 1.5, 0.0, 1.0, 2.5, 2.0)
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
        collision_boxes_m=(collision_box,),
    ).with_transport_model(
        boxes_m=(transport_box,),
        mu_by_isotope={"Cs-137": (0.027,)},
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    collision_ray_source = np.array([-1.0, 0.0, 1.0], dtype=float)
    collision_ray_detector = np.array([2.0, 0.0, 1.0], dtype=float)
    transport_ray_source = np.array([-1.0, 2.0, 1.0], dtype=float)
    transport_ray_detector = np.array([2.0, 2.0, 1.0], dtype=float)

    assert kernel.obstacle_boxes_m() == pytest.approx(np.asarray([transport_box]))
    assert kernel.obstacle_mu_values_cm_inv("Cs-137") == pytest.approx([0.027])
    assert kernel.obstacle_path_length_cm(
        collision_ray_source,
        collision_ray_detector,
    ) == pytest.approx(0.0)
    assert kernel.obstacle_path_length_cm(
        transport_ray_source,
        transport_ray_detector,
    ) == pytest.approx(100.0)


def test_obstacle_only_optical_depth_diagnostics_match_kernel() -> None:
    """Public obstacle diagnostics should expose the same attenuation used by the kernel."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    tau = kernel.obstacle_optical_depth_pair("Cs-137", source, detector)

    assert tau == pytest.approx(1.0)
    assert kernel.obstacle_log_attenuation_pair(
        "Cs-137",
        source,
        detector,
    ) == pytest.approx(-1.0)
    assert kernel.obstacle_attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
    ) == pytest.approx(np.exp(-1.0))


def test_source_extent_obstacle_area_average_reduces_grazing_overattenuation() -> None:
    """Source extent sampling should expose partial obstacle occlusion."""
    grid = ObstacleGrid(
        origin=(0.0, -0.05),
        cell_size=0.1,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    center_kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.1},
        use_gpu=False,
    )
    area_kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.1},
        source_extent_radius_m=0.2,
        source_extent_samples=9,
        use_gpu=False,
    )
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    detector = np.array([2.0, 0.0, 1.0], dtype=float)

    center_tau = center_kernel.obstacle_optical_depth_pair(
        "Cs-137",
        source,
        detector,
    )
    area_tau = area_kernel.obstacle_area_averaged_optical_depth_pair(
        "Cs-137",
        source,
        detector,
    )

    assert center_tau > 0.0
    assert 0.0 <= area_tau < center_tau
    assert area_kernel.obstacle_area_averaged_attenuation_pair(
        "Cs-137",
        source,
        detector,
    ) > center_kernel.obstacle_attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
    )


def test_obstacle_log_attenuation_matrix_matches_pair_diagnostic() -> None:
    """Batched obstacle diagnostics should match the shared single-ray kernel."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(3, 1),
        blocked_cells=((0, 0), (1, 0), (2, 0)),
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_height_m=2.0,
        obstacle_mu_by_isotope={"Cs-137": 0.01},
        use_gpu=False,
    )
    sources = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.4, 1.0],
        ],
        dtype=float,
    )
    detectors = np.asarray(
        [
            [4.0, 0.0, 1.0],
            [4.0, 0.4, 1.0],
            [4.0, 1.5, 1.0],
        ],
        dtype=float,
    )

    matrix = kernel.obstacle_log_attenuation_matrix(
        "Cs-137",
        sources,
        detectors,
        element_budget=2,
    )
    expected = np.asarray(
        [
            [
                kernel.obstacle_log_attenuation_pair("Cs-137", source, detector)
                for source in sources
            ]
            for detector in detectors
        ],
        dtype=float,
    )
    wide_chunk = kernel.obstacle_log_attenuation_matrix(
        "Cs-137",
        sources,
        detectors,
        element_budget=10_000,
    )

    np.testing.assert_allclose(matrix, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wide_chunk, expected, rtol=1.0e-12, atol=1.0e-12)


def test_broad_beam_buildup_increases_but_bounds_attenuated_counts() -> None:
    """Build-up should increase attenuated broad-beam counts without exceeding unattenuated counts."""
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    base_params = ShieldParams(
        mu_fe=0.1, mu_pb=0.0, thickness_fe_cm=5.0, thickness_pb_cm=0.0
    )
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
        inner_radius_fe_cm=DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
        inner_radius_pb_cm=DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    )
    kernel = ContinuousKernel(
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.0}},
        shield_params=shield_params,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    direction = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
    source_distance_m = (DEFAULT_FE_SHIELD_INNER_RADIUS_CM + 0.55) / 100.0
    source = direction * source_distance_m

    attenuation = kernel.attenuation_factor_pair(
        "Cs-137",
        source,
        detector,
        fe_index=7,
        pb_index=0,
    )
    assert attenuation == pytest.approx(np.exp(-0.1 * 0.55), rel=1e-12)


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
    assert kernel.attenuation_factor_pair(
        "Cs-137", source, detector, 0, 0
    ) == pytest.approx(1.0)





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
            assert gpu_counts == pytest.approx(cpu_counts, rel=2e-6, abs=1e-6)


def test_kernel_values_all_pairs_matches_pair_values_cpu() -> None:
    """All-pair kernel evaluation should match per-pair CPU evaluations."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detector = np.array([0.1, -0.2, 0.8], dtype=float)
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    all_pair_values = kernel.kernel_values_all_pairs("Cs-137", detector, sources)
    assert all_pair_values.shape == (64, sources.shape[0])

    for fe_index in range(8):
        for pb_index in range(8):
            pair_id = fe_index * 8 + pb_index
            pair_values = kernel.kernel_values_pair(
                "Cs-137",
                detector,
                sources,
                fe_index,
                pb_index,
            )
            assert all_pair_values[pair_id] == pytest.approx(
                pair_values,
                rel=1.0e-12,
                abs=1.0e-12,
            )


def test_kernel_values_all_pairs_for_detectors_matches_cpu_pairs() -> None:
    """Batched detector all-pair evaluation should match scalar CPU calls."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detectors = np.array(
        [
            [0.1, -0.2, 0.8],
            [1.2, 0.4, 1.1],
            [-0.6, 0.7, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    batched = kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )

    assert batched.shape == (detectors.shape[0], 64, sources.shape[0])
    for pose_idx, detector in enumerate(detectors):
        expected = kernel.kernel_values_all_pairs("Cs-137", detector, sources)
        assert batched[pose_idx] == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-12,
        )


def test_kernel_values_all_pairs_preserves_source_axis_for_empty_detectors() -> None:
    """An empty detector batch should retain the declared source-axis length."""
    kernel = ContinuousKernel(use_gpu=False)
    sources = np.array(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.5]],
        dtype=float,
    )

    values = kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        np.empty((0, 3), dtype=float),
        sources,
    )

    assert values.shape == (0, len(kernel.orientations) ** 2, sources.shape[0])


def test_kernel_values_selected_pairs_for_detectors_matches_cpu_pairs() -> None:
    """Batched selected-pair detector evaluation should match scalar CPU calls."""
    from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

    detectors = np.array(
        [
            [0.1, -0.2, 0.8],
            [1.2, 0.4, 1.1],
            [-0.6, 0.7, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [1.0, 0.2, 0.8],
            [-0.5, 1.4, 1.0],
            [2.0, -1.0, 0.7],
        ],
        dtype=float,
    )
    fe_indices = np.array([0, 3, 7], dtype=int)
    pb_indices = np.array([7, 2, 0], dtype=int)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    kernel = ContinuousKernel(
        mu_by_isotope=mu,
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=5,
    )

    batched = kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )

    assert batched.shape == (detectors.shape[0], sources.shape[0])
    for pose_idx, detector in enumerate(detectors):
        expected = kernel.kernel_values_pair(
            "Cs-137",
            detector,
            sources,
            int(fe_indices[pose_idx]),
            int(pb_indices[pose_idx]),
        )
        assert batched[pose_idx] == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-12,
        )


@pytest.mark.parametrize(
    "shield_params",
    [
        ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=1.0,
            buildup_fe_coeff=0.2,
            buildup_pb_coeff=0.1,
        ),
    ],
    ids=["spherical-octant"],
)
def test_cpu_batched_selected_pairs_matches_scalar_oracle_with_full_physics(
    shield_params: ShieldParams,
) -> None:
    """NumPy batching should preserve every CPU attenuation component."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0), (1, 1)),
    ).with_transport_model(
        boxes_m=(
            (0.2, 0.2, 0.0, 0.8, 0.8, 2.0),
            (1.1, 1.1, 0.0, 1.7, 1.7, 2.0),
        ),
        mu_by_isotope={"TestIso": (0.01, 0.02)},
        line_mu_by_isotope={"TestIso": ((0.01, 0.02), (0.025, 0.04))},
    )
    line_mu = {
        "TestIso": (
            {"weight": 0.4, "fe": 0.05, "pb": 0.08},
            {"weight": 0.6, "fe": 0.09, "pb": 0.12},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        obstacle_grid=grid,
        detector_radius_m=0.04,
        detector_aperture_radius_m=0.03,
        detector_aperture_samples=9,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    detectors = np.array(
        [[0.0, 0.0, 1.0], [2.0, 2.0, 1.0]],
        dtype=float,
    )
    sources = np.array(
        [[1.0, 1.0, 1.0], [-1.0, 0.2, 0.7], [2.5, 0.3, 1.2]],
        dtype=float,
    )
    fe_indices = np.array([0, 3], dtype=int)
    pb_indices = np.array([7, 2], dtype=int)

    batched = kernel.kernel_values_selected_pairs_for_detectors(
        "TestIso",
        detectors,
        sources,
        fe_indices,
        pb_indices,
        chunk_size=2,
    )
    scalar = np.vstack(
        [
            kernel._kernel_values_pair_scalar_oracle(
                "TestIso",
                detector,
                sources,
                int(fe_index),
                int(pb_index),
            )
            for detector, fe_index, pb_index in zip(
                detectors,
                fe_indices,
                pb_indices,
            )
        ]
    )
    all_pairs_batched = kernel.kernel_values_all_pairs_for_detectors(
        "TestIso",
        detectors,
        sources,
        chunk_size=7,
    )
    all_pairs_scalar = np.stack(
        [
            np.vstack(
                [
                    kernel._kernel_values_pair_scalar_oracle(
                        "TestIso",
                        detector,
                        sources,
                        fe_index,
                        pb_index,
                    )
                    for fe_index in range(len(kernel.orientations))
                    for pb_index in range(len(kernel.orientations))
                ]
            )
            for detector in detectors
        ],
        axis=0,
    )

    assert batched == pytest.approx(scalar, rel=5.0e-13, abs=5.0e-13)
    assert all_pairs_batched == pytest.approx(
        all_pairs_scalar,
        rel=5.0e-13,
        abs=5.0e-13,
    )


def test_torch_source_extent_rays_match_numpy() -> None:
    """Torch kernels should match NumPy for sampled source-extent rays."""
    torch = pytest.importorskip("torch")
    isotope = "TestIso"
    kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        detector_radius_m=0.038,
        detector_aperture_radius_m=0.03,
        detector_aperture_samples=5,
        source_extent_radius_m=0.5,
        source_extent_samples=5,
        use_gpu=True,
        gpu_device="cpu",
        gpu_dtype="float64",
    )
    detectors = np.array(
        [[0.0, 0.0, 0.0], [0.5, -0.25, 0.1]],
        dtype=float,
    )
    matched_sources = np.array(
        [[2.0, 0.0, 0.0], [-1.0, 1.5, 0.3]],
        dtype=float,
    )
    fe_indices = np.array([0, 3], dtype=np.int64)
    pb_indices = np.array([7, 2], dtype=np.int64)

    numpy_selected = (
        kernel._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
            isotope,
            detectors,
            matched_sources,
            fe_indices,
            pb_indices,
        )
    )
    torch_selected = (
        kernel._kernel_values_selected_pairs_for_detector_source_torch_chunk(
            isotope,
            detectors,
            matched_sources,
            fe_indices,
            pb_indices,
            tol=1.0e-12,
        )
    )

    pair_count = len(kernel.orientations) ** 2
    pair_ids = np.arange(pair_count, dtype=np.int64)
    pair_fe = pair_ids // len(kernel.orientations)
    pair_pb = pair_ids % len(kernel.orientations)
    numpy_all = (
        kernel._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
            isotope,
            np.repeat(detectors, pair_count, axis=0),
            np.repeat(matched_sources, pair_count, axis=0),
            np.tile(pair_fe, detectors.shape[0]),
            np.tile(pair_pb, detectors.shape[0]),
        ).reshape(detectors.shape[0], pair_count)
    )
    torch_all = kernel._kernel_values_all_pairs_for_detector_source_torch_chunk(
        isotope,
        detectors,
        matched_sources,
        tol=1.0e-12,
    )

    common_detector_sources = np.array(
        [[2.0, 0.0, 0.0], [-1.0, 1.5, 0.3]],
        dtype=float,
    )
    tensor_fe = np.array([0, 3], dtype=np.int64)
    tensor_pb = np.array([7, 2], dtype=np.int64)
    numpy_tensor = np.vstack(
        [
            kernel._kernel_values_selected_pairs_for_detector_source_numpy_chunk(
                isotope,
                np.broadcast_to(detectors[0], common_detector_sources.shape),
                common_detector_sources,
                np.full(common_detector_sources.shape[0], fe_index, dtype=np.int64),
                np.full(common_detector_sources.shape[0], pb_index, dtype=np.int64),
            )
            for fe_index, pb_index in zip(tensor_fe, tensor_pb)
        ]
    )
    torch_tensor = (
        kernel._kernel_values_selected_pairs_torch_tensor(
            isotope,
            detectors[0],
            torch.as_tensor(common_detector_sources, dtype=torch.float64),
            tensor_fe,
            tensor_pb,
            tol=1.0e-12,
        )
        .detach()
        .cpu()
        .numpy()
    )

    strengths = np.array([2.0, 3.0], dtype=float)
    background = 0.25
    numpy_rate = float(background + numpy_tensor[0] @ strengths)
    torch_rate = kernel._expected_rate_pair_torch(
        isotope,
        detectors[0],
        common_detector_sources,
        strengths,
        int(tensor_fe[0]),
        int(tensor_pb[0]),
        background,
        tol=1.0e-12,
    )

    np.testing.assert_allclose(torch_selected, numpy_selected, rtol=1.0e-12)
    np.testing.assert_allclose(torch_all, numpy_all, rtol=1.0e-12)
    np.testing.assert_allclose(torch_tensor, numpy_tensor, rtol=1.0e-12)
    assert torch_rate == pytest.approx(numpy_rate, rel=1.0e-12)


def test_standard_cpu_kernel_paths_select_numpy_batching(monkeypatch) -> None:
    """Standard CPU multi-source APIs should not invoke the scalar primitive."""
    kernel = ContinuousKernel(
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=3,
    )
    detectors = np.array(
        [[0.1, -0.2, 0.8], [1.2, 0.4, 1.1]],
        dtype=float,
    )
    sources = np.array(
        [[1.0, 0.2, 0.8], [-0.5, 1.4, 1.0], [2.0, -1.0, 0.7]],
        dtype=float,
    )
    original_batch = (
        kernel._kernel_values_selected_pairs_for_detector_source_numpy_chunk
    )
    batch_calls = 0

    def _tracked_batch(*args, **kwargs):
        """Count standard-path calls into the batched NumPy implementation."""
        nonlocal batch_calls
        batch_calls += 1
        return original_batch(*args, **kwargs)

    def _forbid_scalar(*args, **kwargs):
        """Fail if a standard multi-source path selects the scalar primitive."""
        raise AssertionError("standard CPU path selected scalar kernel_value_pair")

    monkeypatch.setattr(
        kernel,
        "_kernel_values_selected_pairs_for_detector_source_numpy_chunk",
        _tracked_batch,
    )
    monkeypatch.setattr(kernel, "kernel_value_pair", _forbid_scalar)

    pair_values = kernel.kernel_values_pair(
        "Cs-137",
        detectors[0],
        sources,
        0,
        7,
        chunk_size=2,
    )
    selected_values = kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        np.array([0, 3], dtype=int),
        np.array([7, 2], dtype=int),
        chunk_size=2,
    )
    all_pairs = kernel.kernel_values_all_pairs(
        "Cs-137",
        detectors[0],
        sources,
        chunk_size=2,
    )
    all_detector_pairs = kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        chunk_size=2,
    )

    assert pair_values.shape == (sources.shape[0],)
    assert selected_values.shape == (detectors.shape[0], sources.shape[0])
    assert all_pairs.shape == (
        len(kernel.orientations) ** 2,
        sources.shape[0],
    )
    assert all_detector_pairs.shape == (
        detectors.shape[0],
        len(kernel.orientations) ** 2,
        sources.shape[0],
    )
    assert np.all(np.isfinite(pair_values))
    assert np.all(np.isfinite(selected_values))
    assert np.all(np.isfinite(all_pairs))
    assert np.all(np.isfinite(all_detector_pairs))
    assert batch_calls >= 4


def test_kernel_values_all_pairs_cuda_matches_cpu_with_obstacles_and_aperture() -> None:
    """All-pair CUDA kernel evaluation should match the CPU observation model."""
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
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )

    cpu_values = cpu_kernel.kernel_values_all_pairs("Cs-137", detector, sources)
    gpu_values = gpu_kernel.kernel_values_all_pairs("Cs-137", detector, sources)

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


def test_unshielded_batch_equals_best_pair_for_finite_ray_bundles() -> None:
    """Unshielded finite-ray response must equal the best physical pair."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1)),
    )
    detectors = np.asarray(
        [[2.3, 2.4, 1.2], [3.4, 0.7, 0.8]],
        dtype=float,
    )
    sources = np.asarray(
        [
            [0.2, 0.3, 0.9],
            [0.4, 3.5, 1.6],
            [3.6, 3.2, 0.4],
        ],
        dtype=float,
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )

    unshielded = kernel.kernel_values_unshielded_for_detectors(
        "Cs-137",
        detectors,
        sources,
        chunk_size=2,
    )
    all_pairs = kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        chunk_size=2,
    )

    np.testing.assert_allclose(
        unshielded,
        np.max(all_pairs, axis=1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_unshielded_torch_cpu_matches_numpy_with_obstacles_and_aperture() -> None:
    """Torch-batched unshielded kernels must match the NumPy physical path."""
    pytest.importorskip("torch")
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detectors = np.asarray(
        [[2.2, 2.2, 1.0], [3.2, 1.4, 1.1]],
        dtype=float,
    )
    sources = np.asarray(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    common = {
        "obstacle_grid": grid,
        "obstacle_mu_by_isotope": {"Cs-137": 0.17},
        "detector_radius_m": 0.038,
        "detector_aperture_samples": 7,
        "source_extent_radius_m": 0.08,
        "source_extent_samples": 5,
    }
    numpy_kernel = ContinuousKernel(use_gpu=False, **common)
    torch_kernel = ContinuousKernel(
        use_gpu=True,
        gpu_device="cpu",
        gpu_dtype="float64",
        **common,
    )

    numpy_values = numpy_kernel.kernel_values_unshielded_for_detectors(
        "Cs-137",
        detectors,
        sources,
        chunk_size=3,
    )
    torch_values = torch_kernel.kernel_values_unshielded_for_detectors(
        "Cs-137",
        detectors,
        sources,
        chunk_size=3,
    )

    np.testing.assert_allclose(
        torch_values,
        numpy_values,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_unshielded_cuda_matches_numpy_with_obstacles_and_aperture() -> None:
    """CUDA-batched unshielded kernels must match the NumPy physical path."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(4, 4),
        blocked_cells=((1, 1), (2, 1), (1, 2)),
    )
    detectors = np.asarray(
        [[2.2, 2.2, 1.0], [3.2, 1.4, 1.1]],
        dtype=float,
    )
    sources = np.asarray(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    common = {
        "obstacle_grid": grid,
        "obstacle_mu_by_isotope": {"Cs-137": 0.17},
        "detector_radius_m": 0.038,
        "detector_aperture_samples": 7,
        "source_extent_radius_m": 0.08,
        "source_extent_samples": 5,
    }
    numpy_kernel = ContinuousKernel(use_gpu=False, **common)
    cuda_kernel = ContinuousKernel(
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        **common,
    )

    numpy_values = numpy_kernel.kernel_values_unshielded_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )
    cuda_values = cuda_kernel.kernel_values_unshielded_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )

    np.testing.assert_allclose(
        cuda_values,
        numpy_values,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_kernel_values_all_pairs_for_detectors_cuda_matches_cpu() -> None:
    """Batched detector CUDA all-pair kernels should match CPU results."""
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
    detectors = np.array(
        [
            [2.2, 2.2, 1.0],
            [3.2, 1.4, 1.1],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
    )

    cpu_values = cpu_kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )
    gpu_values = gpu_kernel.kernel_values_all_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
    )

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


def test_kernel_values_selected_pairs_for_detectors_cuda_matches_cpu() -> None:
    """Batched selected-pair CUDA detector kernels should match CPU results."""
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
    detectors = np.array(
        [
            [2.2, 2.2, 1.0],
            [3.2, 1.4, 1.1],
            [1.2, 3.4, 0.9],
        ],
        dtype=float,
    )
    sources = np.array(
        [
            [0.2, 0.2, 1.0],
            [0.5, 3.5, 1.0],
            [3.5, 0.5, 1.0],
            [4.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    fe_indices = np.array([0, 4, 7], dtype=int)
    pb_indices = np.array([7, 3, 1], dtype=int)
    mu = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM)
    cpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=False,
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )
    gpu_kernel = ContinuousKernel(
        mu_by_isotope=mu,
        obstacle_grid=grid,
        obstacle_mu_by_isotope={"Cs-137": 0.17},
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        detector_radius_m=0.038,
        detector_aperture_samples=7,
        source_extent_radius_m=0.08,
        source_extent_samples=5,
    )

    cpu_values = cpu_kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )
    gpu_values = gpu_kernel.kernel_values_selected_pairs_for_detectors(
        "Cs-137",
        detectors,
        sources,
        fe_indices,
        pb_indices,
    )

    assert gpu_values == pytest.approx(cpu_values, rel=1.0e-10, abs=1.0e-10)


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


def test_torch_chunk_cpu_fallback_accounts_for_obstacle_aperture_expansion() -> None:
    """CPU fallback batching should retain the conservative fixed budget."""
    torch = pytest.importorskip("torch")
    blocked = tuple((idx, 0) for idx in range(494))
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(494, 1),
        blocked_cells=blocked,
    )
    kernel = ContinuousKernel(
        obstacle_grid=grid,
        detector_radius_m=0.038,
        detector_aperture_samples=121,
        gpu_dtype="float64",
    )

    chunk = kernel._adaptive_torch_chunk_size(
        8192,
        isotope="Cs-137",
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert chunk == 66
    assert chunk < 8192
