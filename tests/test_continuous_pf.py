"""Basic tests for continuous measurement model and PF scaffold."""

from pathlib import Path

import numpy as np
import pytest

import measurement.continuous_kernels as continuous_kernels
from measurement.continuous_kernels import (
    ContinuousKernel,
    expected_counts_single_isotope,
)
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import rotation_matrix_from_normal


def test_gpu_pair_counts_match_continuous_kernel_with_line_obstacles() -> None:
    """Torch PF counts should match the shared line-resolved ContinuousKernel."""
    torch = pytest.importorskip("torch")

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
    shield = ShieldParams(mu_fe=0.0, mu_pb=0.0)
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield,
        obstacle_grid=grid,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    device = torch.device("cpu")
    dtype = torch.float64
    detector = np.array([2.0, 0.0, 1.0], dtype=float)
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.as_tensor([[100.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones((1, 1), device=device, dtype=dtype)

    gpu_counts = kernel.expected_counts_pair_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )
    cpu_counts = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source.reshape(1, 3),
        np.array([100.0], dtype=float),
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert float(gpu_counts[0].detach().cpu().item()) == pytest.approx(
        cpu_counts, rel=1e-12
    )


def test_continuous_kernel_packed_gpu_pairs_are_consistent() -> None:
    """ContinuousKernel packed pair, selected-pair, and all-pair paths should agree."""
    torch = pytest.importorskip("torch")

    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(2, 1),
        blocked_cells=((0, 0), (1, 0)),
    ).with_transport_model(
        boxes_m=(
            (0.0, -0.5, 0.0, 1.0, 0.5, 2.0),
            (1.0, -0.5, 0.0, 2.0, 0.5, 2.0),
        ),
        mu_by_isotope={"TestIso": (0.0, 0.0)},
        line_mu_by_isotope={"TestIso": ((0.01, 0.02), (0.03, 0.04))},
    )
    line_mu = {
        "TestIso": (
            {"weight": 1.0, "fe": 0.01, "pb": 0.02},
            {"weight": 2.0, "fe": 0.03, "pb": 0.04},
        )
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.02, "pb": 0.03}},
        shield_params=ShieldParams(mu_fe=0.02, mu_pb=0.03),
        obstacle_grid=grid,
        line_mu_by_isotope=line_mu,
        detector_radius_m=0.04,
        detector_aperture_radius_m=0.05,
        detector_aperture_samples=5,
        use_gpu=False,
    )
    device = torch.device("cpu")
    dtype = torch.float64
    detector = np.array([3.0, 0.0, 1.0], dtype=float)
    positions = torch.as_tensor(
        [
            [[-1.0, 0.0, 1.0], [-0.5, 0.4, 0.8]],
            [[0.5, -0.2, 1.4], [1.4, 0.2, 1.1]],
        ],
        device=device,
        dtype=dtype,
    )
    strengths = torch.as_tensor(
        [[100.0, 40.0], [70.0, 30.0]], device=device, dtype=dtype
    )
    backgrounds = torch.as_tensor([0.5, 0.25], device=device, dtype=dtype)
    mask = torch.ones((2, 2), device=device, dtype=dtype)
    fe_indices = np.array([0, 3, 7], dtype=np.int64)
    pb_indices = np.array([7, 2, 4], dtype=np.int64)

    selected = kernel.expected_counts_selected_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=detector,
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
    pair_rows = []
    for fe_index, pb_index in zip(fe_indices, pb_indices):
        pair_rows.append(
            kernel.expected_counts_pair_for_packed_states_torch(
                isotope="TestIso",
                detector_pos=detector,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=2.0,
                device=device,
                dtype=dtype,
            )
        )
    pair_loop = torch.stack(pair_rows, dim=0)
    all_pairs = kernel.expected_counts_all_pairs_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        live_time_s=2.0,
        device=device,
        dtype=dtype,
    )
    pair_indices = fe_indices * 8 + pb_indices

    assert torch.allclose(selected, pair_loop, rtol=1e-10, atol=1e-10)
    assert torch.allclose(selected, all_pairs[pair_indices], rtol=1e-10, atol=1e-10)


def test_runtime_expected_counts_use_shared_kernel_component() -> None:
    """Runtime PF and planner code should not bypass ContinuousKernel counts."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = [
        Path(continuous_kernels.__file__).resolve(),
        root / "src/pf/particle_filter.py",
        root / "src/pf/estimator.py",
        root / "src/planning/dss_pp.py",
    ]
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "gpu_utils.expected_counts_" not in text, path


def test_geometric_scaling_inverse_square() -> None:
    """Expected counts should follow inverse-square scaling without shielding."""
    src = np.array([[0.0, 0.0, 0.0]])
    strength = np.array([10.0])
    d1 = 1.0
    d2 = 2.0
    lam1 = expected_counts_single_isotope(
        detector_position=np.array([d1, 0.0, 0.0]),
        RFe=rotation_matrix_from_normal(np.array([-1.0, 0.0, 0.0])),
        RPb=rotation_matrix_from_normal(np.array([-1.0, 0.0, 0.0])),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam2 = expected_counts_single_isotope(
        detector_position=np.array([d2, 0.0, 0.0]),
        RFe=rotation_matrix_from_normal(np.array([-1.0, 0.0, 0.0])),
        RPb=rotation_matrix_from_normal(np.array([-1.0, 0.0, 0.0])),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    assert np.allclose(lam1 / lam2, (d2**2) / (d1**2), rtol=1e-6)


def test_shield_attenuation_factor_both_materials() -> None:
    """When both Fe and Pb block, expected counts should follow exp(-mu*L)."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strength = np.array([5.0])
    lam_free = expected_counts_single_isotope(
        detector_position=det,
        RFe=rotation_matrix_from_normal(np.array([-1.0, -1.0, -1.0])),
        RPb=rotation_matrix_from_normal(np.array([-1.0, -1.0, -1.0])),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam_blocked = expected_counts_single_isotope(
        detector_position=det,
        RFe=rotation_matrix_from_normal(np.array([1.0, 1.0, 1.0])),
        RPb=rotation_matrix_from_normal(np.array([1.0, 1.0, 1.0])),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    from measurement.kernels import ShieldParams

    shield_params = ShieldParams()
    expected_ratio = np.exp(
        -(
            shield_params.mu_fe * shield_params.thickness_fe_cm
            + shield_params.mu_pb * shield_params.thickness_pb_cm
        )
    )
    assert np.isclose(lam_blocked, expected_ratio * lam_free, rtol=1e-6)
