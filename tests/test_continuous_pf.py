"""Basic tests for continuous measurement model and PF scaffold."""

from pathlib import Path
import math

import numpy as np
import pytest

from measurement.continuous_kernels import (
    ContinuousKernel,
    expected_counts_single_isotope,
)
from measurement.kernels import ShieldParams
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import is_allowed_source_surface_position
from pf.likelihood import (
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    count_log_likelihood,
    normalize_count_likelihood_model,
)
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import (
    IsotopeParticleFilter,
    PFConfig,
    IsotopeParticle,
    MeasurementData,
)
from pf.state import IsotopeState


def test_selected_pair_gpu_counts_match_all_pair_selection() -> None:
    """Selected-pair torch counts should equal all-pair counts indexed afterward."""
    torch = pytest.importorskip("torch")
    from measurement.kernels import ShieldParams
    from pf import gpu_utils

    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(
        [
            [[1.0, 0.5, 0.4], [2.0, 1.2, 1.4]],
            [[0.8, 1.5, 0.9], [2.5, 0.2, 1.8]],
        ],
        device=device,
        dtype=dtype,
    )
    strengths = torch.as_tensor(
        [[1000.0, 500.0], [800.0, 300.0]],
        device=device,
        dtype=dtype,
    )
    backgrounds = torch.as_tensor([2.0, 3.0], device=device, dtype=dtype)
    mask = torch.ones((2, 2), device=device, dtype=dtype)
    shield = ShieldParams()
    detector_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    fe_indices = np.array([0, 3, 7, 1], dtype=np.int64)
    pb_indices = np.array([7, 2, 4, 1], dtype=np.int64)

    all_counts = gpu_utils.expected_counts_all_pairs_torch(
        detector_pos=detector_pos,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        mu_fe=shield.mu_fe,
        mu_pb=shield.mu_pb,
        thickness_fe_cm=shield.thickness_fe_cm,
        thickness_pb_cm=shield.thickness_pb_cm,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        inner_radius_fe_cm=shield.inner_radius_fe_cm,
        inner_radius_pb_cm=shield.inner_radius_pb_cm,
        shield_geometry_model=shield.shield_geometry_model,
        use_angle_attenuation=shield.use_angle_attenuation,
    )
    selected_counts = gpu_utils.expected_counts_selected_pairs_torch(
        detector_pos=detector_pos,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        mu_fe=shield.mu_fe,
        mu_pb=shield.mu_pb,
        thickness_fe_cm=shield.thickness_fe_cm,
        thickness_pb_cm=shield.thickness_pb_cm,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        inner_radius_fe_cm=shield.inner_radius_fe_cm,
        inner_radius_pb_cm=shield.inner_radius_pb_cm,
        shield_geometry_model=shield.shield_geometry_model,
        use_angle_attenuation=shield.use_angle_attenuation,
    )
    pair_indices = fe_indices * 8 + pb_indices

    assert torch.allclose(selected_counts, all_counts[pair_indices], rtol=1e-10)

    pair_scales = np.linspace(0.5, 1.5, 64, dtype=float)
    all_scaled = gpu_utils.expected_counts_all_pairs_torch(
        detector_pos=detector_pos,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        mu_fe=shield.mu_fe,
        mu_pb=shield.mu_pb,
        thickness_fe_cm=shield.thickness_fe_cm,
        thickness_pb_cm=shield.thickness_pb_cm,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        source_scale=pair_scales,
        inner_radius_fe_cm=shield.inner_radius_fe_cm,
        inner_radius_pb_cm=shield.inner_radius_pb_cm,
        shield_geometry_model=shield.shield_geometry_model,
        use_angle_attenuation=shield.use_angle_attenuation,
    )
    selected_scaled = gpu_utils.expected_counts_selected_pairs_torch(
        detector_pos=detector_pos,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        mu_fe=shield.mu_fe,
        mu_pb=shield.mu_pb,
        thickness_fe_cm=shield.thickness_fe_cm,
        thickness_pb_cm=shield.thickness_pb_cm,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
        source_scale=pair_scales[pair_indices],
        inner_radius_fe_cm=shield.inner_radius_fe_cm,
        inner_radius_pb_cm=shield.inner_radius_pb_cm,
        shield_geometry_model=shield.shield_geometry_model,
        use_angle_attenuation=shield.use_angle_attenuation,
    )

    assert torch.allclose(
        selected_scaled,
        all_scaled[pair_indices],
        rtol=1e-10,
    )


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


def test_transport_response_model_matches_cpu_and_gpu_paths() -> None:
    """Transport-response optical-depth factors should be shared by CPU/GPU kernels."""
    torch = pytest.importorskip("torch")

    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.0, -0.5, 0.0, 1.0, 0.5, 2.0),),
        mu_by_isotope={"TestIso": (0.001,)},
        line_mu_by_isotope={"TestIso": ((0.001,),)},
    )
    transport_model = {
        "enabled": True,
        "by_isotope": {
            "TestIso": {
                "scale": 1.0,
                "tau_coefficients": {"shield": 0.0, "obstacle": -0.5},
                "min_log_scale": -2.0,
                "max_log_scale": 2.0,
            }
        },
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        line_mu_by_isotope={
            "TestIso": ({"weight": 1.0, "fe": 0.0, "pb": 0.0},),
        },
        transport_response_model=transport_model,
        use_gpu=False,
    )
    base_kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        obstacle_grid=grid,
        line_mu_by_isotope={
            "TestIso": ({"weight": 1.0, "fe": 0.0, "pb": 0.0},),
        },
        use_gpu=False,
    )
    detector = np.array([2.0, 0.0, 1.0], dtype=float)
    source = np.array([-1.0, 0.0, 1.0], dtype=float)
    base_value = base_kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    model_value = kernel.kernel_value_pair("TestIso", detector, source, 0, 0)
    expected_ratio = np.exp(-0.5 * 0.1)

    assert model_value / base_value == pytest.approx(expected_ratio, rel=1e-12)

    device = torch.device("cpu")
    dtype = torch.float64
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


def test_transport_response_model_can_exceed_unattenuated_cap_cpu_and_torch() -> None:
    """A fitted transport-response model should match validation semantics above unity."""
    torch = pytest.importorskip("torch")

    line_mu = {
        "TestIso": ({"weight": 1.0, "fe": 0.2, "pb": 0.0},),
    }
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
    )
    transport_model = {
        "enabled": True,
        "by_isotope": {
            "TestIso": {
                "scale": 1.0,
                "tau_coefficients": {"shield": 3.0},
                "min_log_scale": -5.0,
                "max_log_scale": 5.0,
            }
        },
    }
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    free_kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    strength = np.array([100.0], dtype=float)

    terms = kernel.transport_response_terms_pair("TestIso", detector, source, 7, 0)
    shield_tau = float(terms[0]["shield_tau_feature"])
    model_counts = kernel.expected_counts_pair(
        "TestIso",
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )
    free_counts = free_kernel.expected_counts_pair(
        "TestIso",
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert terms[0]["kernel"] > terms[0]["sample_scale"]
    assert model_counts / free_counts == pytest.approx(
        np.exp(-shield_tau) * np.exp(3.0 * shield_tau),
        rel=1.0e-12,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.as_tensor([[100.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones((1, 1), device=device, dtype=dtype)
    torch_counts = kernel.expected_counts_pair_for_packed_states_torch(
        isotope="TestIso",
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    assert float(torch_counts[0].detach().cpu().item()) == pytest.approx(
        model_counts,
        rel=1.0e-12,
    )


def test_identity_transport_response_preserves_capped_base_attenuation() -> None:
    """A no-op response model should not uncap broad-beam buildup attenuation."""
    torch = pytest.importorskip("torch")

    isotope = "TestIso"
    line_mu = {isotope: ({"weight": 1.0, "fe": 0.05, "pb": 0.0},)}
    shield_params = ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=2.0,
        thickness_pb_cm=0.0,
        buildup_fe_coeff=30.0,
    )
    identity_model = {
        "enabled": True,
        "by_isotope": {
            isotope: {
                "scale": 1.0,
                "tau_coefficients": {},
                "min_log_scale": -5.0,
                "max_log_scale": 5.0,
            }
        },
    }
    base_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    identity_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=shield_params,
        line_mu_by_isotope=line_mu,
        transport_response_model=identity_model,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    strength = np.array([100.0], dtype=float)

    base_counts = base_kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )
    identity_counts = identity_kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )
    identity_terms = identity_kernel.transport_response_terms_pair(
        isotope,
        detector,
        source,
        fe_index=7,
        pb_index=0,
    )

    assert identity_terms[0]["uncapped_kernel"] > identity_terms[0]["sample_scale"]
    assert identity_terms[0]["base_kernel"] == pytest.approx(
        identity_terms[0]["sample_scale"],
        rel=1.0e-12,
    )
    assert identity_terms[0]["kernel"] == pytest.approx(
        identity_terms[0]["sample_scale"],
        rel=1.0e-12,
    )
    assert identity_counts == pytest.approx(base_counts, rel=1.0e-12)

    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.as_tensor([[100.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones((1, 1), device=device, dtype=dtype)
    torch_counts = identity_kernel.expected_counts_pair_for_packed_states_torch(
        isotope=isotope,
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    assert float(torch_counts[0].detach().cpu().item()) == pytest.approx(
        identity_counts,
        rel=1.0e-12,
    )


def test_transport_response_feature_caps_match_cpu_and_torch() -> None:
    """Transport-response feature caps should apply in CPU and torch kernels."""
    torch = pytest.importorskip("torch")

    isotope = "TestIso"
    line_mu = {isotope: ({"weight": 1.0, "fe": 1.0, "pb": 0.0},)}
    transport_model = {
        "enabled": True,
        "by_isotope": {
            isotope: {
                "scale": 1.0,
                "tau_coefficients": {
                    "shield_squared": 1.0,
                    "distance_shield": 1.0,
                },
                "tau_feature_caps": {"shield": 0.5, "distance_shield": 0.25},
                "min_log_scale": -10.0,
                "max_log_scale": 10.0,
            }
        },
    }
    kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=0.0,
        ),
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    terms = kernel.transport_response_terms_pair(isotope, detector, source, 7, 0)
    shield_tau = float(terms[0]["shield_tau_feature"])
    distance_shield = float(terms[0]["distance_shield_feature"])

    assert shield_tau > 0.5
    assert distance_shield > 0.25
    assert terms[0]["shield_tau_feature_capped"] == pytest.approx(0.5)
    assert terms[0]["distance_shield_feature_capped"] == pytest.approx(0.25)
    assert terms[0]["response_factor"] == pytest.approx(np.exp(0.5), rel=1.0e-12)

    device = torch.device("cpu")
    dtype = torch.float64
    positions = torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype)
    strengths = torch.as_tensor([[100.0]], device=device, dtype=dtype)
    backgrounds = torch.zeros(1, device=device, dtype=dtype)
    mask = torch.ones((1, 1), device=device, dtype=dtype)
    torch_counts = kernel.expected_counts_pair_for_packed_states_torch(
        isotope=isotope,
        detector_pos=detector,
        positions=positions,
        strengths=strengths,
        backgrounds=backgrounds,
        mask=mask,
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )
    cpu_counts = kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        np.array([100.0], dtype=float),
        fe_index=7,
        pb_index=0,
        live_time_s=1.0,
    )

    assert float(torch_counts[0].detach().cpu().item()) == pytest.approx(
        cpu_counts,
        rel=1.0e-12,
    )


def test_transport_response_model_uses_separate_fe_pb_features() -> None:
    """Transport response should distinguish Fe and Pb optical-depth features."""
    kernel = ContinuousKernel(
        mu_by_isotope={"TestIso": {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=1.0,
            thickness_pb_cm=1.0,
        ),
        line_mu_by_isotope={
            "TestIso": ({"weight": 1.0, "fe": 0.2, "pb": 0.4},),
        },
        transport_response_model={
            "enabled": True,
            "by_isotope": {
                "TestIso": {
                    "scale": 1.0,
                    "tau_coefficients": {"fe": 0.5, "pb": -0.25},
                    "min_log_scale": -2.0,
                    "max_log_scale": 2.0,
                }
            },
        },
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)

    terms = kernel.transport_response_terms_pair("TestIso", detector, source, 7, 7)
    assert terms
    fe_tau = float(terms[0]["fe_tau_feature"])
    pb_tau = float(terms[0]["pb_tau_feature"])
    expected = float(np.exp(0.5 * fe_tau - 0.25 * pb_tau))

    assert terms[0]["response_factor"] == pytest.approx(expected, rel=1e-12)
    assert fe_tau > 0.0
    assert pb_tau > 0.0


def test_transport_response_model_uses_distance_features_in_cpu_and_torch() -> None:
    """Transport response should apply source-distance features on CPU and torch."""
    torch = pytest.importorskip("torch")

    isotope = "TestIso"
    distance_coeff = -0.2
    kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        transport_response_model={
            "enabled": True,
            "by_isotope": {
                isotope: {
                    "scale": 1.0,
                    "tau_coefficients": {"distance": distance_coeff},
                    "min_log_scale": -10.0,
                    "max_log_scale": 10.0,
                }
            },
        },
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([2.0, 0.0, 0.0], dtype=float)
    strength = np.array([100.0], dtype=float)
    distance = float(np.linalg.norm(source - detector))
    expected_factor = float(np.exp(distance_coeff * distance))

    terms = kernel.transport_response_terms_pair("TestIso", detector, source, 0, 0)
    cpu_counts = kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )
    base_counts = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        use_gpu=False,
    ).expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
    )

    assert terms[0]["distance_feature"] == pytest.approx(distance)
    assert terms[0]["response_factor"] == pytest.approx(expected_factor)
    assert cpu_counts == pytest.approx(base_counts * expected_factor)

    device = torch.device("cpu")
    dtype = torch.float64
    torch_counts = kernel.expected_counts_pair_for_packed_states_torch(
        isotope=isotope,
        detector_pos=detector,
        positions=torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype),
        strengths=torch.as_tensor([[100.0]], device=device, dtype=dtype),
        backgrounds=torch.zeros(1, device=device, dtype=dtype),
        mask=torch.ones((1, 1), device=device, dtype=dtype),
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    assert float(torch_counts[0].detach().cpu().item()) == pytest.approx(
        cpu_counts,
        rel=1.0e-12,
    )


def test_transport_response_distance_material_terms_match_cpu_and_torch() -> None:
    """Distance-material transport terms should match CPU and torch kernels."""
    torch = pytest.importorskip("torch")

    isotope = "TestIso"
    line_mu = {isotope: ({"weight": 1.0, "fe": 1.0, "pb": 0.5},)}
    transport_model = {
        "enabled": True,
        "by_isotope": {
            isotope: {
                "scale": 1.0,
                "tau_coefficients": {"distance_fe": 0.2, "distance_pb": -0.1},
                "min_log_scale": -10.0,
                "max_log_scale": 10.0,
            }
        },
    }
    kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=1.0,
        ),
        line_mu_by_isotope=line_mu,
        transport_response_model=transport_model,
        use_gpu=False,
    )
    base_kernel = ContinuousKernel(
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        shield_params=ShieldParams(
            mu_fe=0.0,
            mu_pb=0.0,
            thickness_fe_cm=2.0,
            thickness_pb_cm=1.0,
        ),
        line_mu_by_isotope=line_mu,
        use_gpu=False,
    )
    detector = np.zeros(3, dtype=float)
    source = np.array([1.0, 1.0, 1.0], dtype=float)
    strength = np.array([100.0], dtype=float)

    terms = kernel.transport_response_terms_pair(isotope, detector, source, 7, 7)
    distance_fe = float(terms[0]["distance_fe_feature"])
    distance_pb = float(terms[0]["distance_pb_feature"])
    expected_factor = float(np.exp(0.2 * distance_fe - 0.1 * distance_pb))
    cpu_counts = kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=7,
        pb_index=7,
        live_time_s=1.0,
    )
    base_counts = base_kernel.expected_counts_pair(
        isotope,
        detector,
        source.reshape(1, 3),
        strength,
        fe_index=7,
        pb_index=7,
        live_time_s=1.0,
    )

    assert distance_fe > 0.0
    assert distance_pb > 0.0
    assert terms[0]["response_factor"] == pytest.approx(expected_factor)
    assert cpu_counts == pytest.approx(base_counts * expected_factor)

    device = torch.device("cpu")
    dtype = torch.float64
    torch_counts = kernel.expected_counts_pair_for_packed_states_torch(
        isotope=isotope,
        detector_pos=detector,
        positions=torch.as_tensor(source.reshape(1, 1, 3), device=device, dtype=dtype),
        strengths=torch.as_tensor([[100.0]], device=device, dtype=dtype),
        backgrounds=torch.zeros(1, device=device, dtype=dtype),
        mask=torch.ones((1, 1), device=device, dtype=dtype),
        fe_index=7,
        pb_index=7,
        live_time_s=1.0,
        device=device,
        dtype=dtype,
    )

    assert float(torch_counts[0].detach().cpu().item()) == pytest.approx(
        cpu_counts,
        rel=1.0e-12,
    )


def test_estimator_exposes_transport_response_model_to_planner_kernel() -> None:
    """Estimator-created PF and planner kernels should share transport settings."""
    transport_model = {
        "enabled": True,
        "by_isotope": {
            "Cs-137": {
                "scale": 0.9,
                "tau_coefficients": {"shield": 0.1, "obstacle": -0.2},
            }
        },
    }
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.array([[1.0, 1.0, 1.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": {"fe": 0.01, "pb": 0.02}},
        pf_config=RotatingShieldPFConfig(num_particles=4, use_gpu=False),
        transport_response_model=transport_model,
    )
    estimator.add_measurement_pose(np.array([0.0, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()

    estimator_kernel = estimator.continuous_kernel(use_gpu=False)

    assert estimator_kernel.transport_response_model == transport_model
    assert (
        estimator.filters["Cs-137"].continuous_kernel.transport_response_model
        == transport_model
    )


def test_runtime_expected_counts_use_shared_kernel_component() -> None:
    """Runtime PF and planner code should not bypass ContinuousKernel counts."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = [
        "src/measurement/continuous_kernels.py",
        "src/pf/particle_filter.py",
        "src/pf/estimator.py",
        "src/pf/mixing.py",
        "src/pf/parallel.py",
        "src/planning/dss_pp.py",
        "src/planning/remaining_measurements.py",
        "src/planning/shield_rotation.py",
        "src/realtime_demo.py",
    ]
    for rel_path in checked_paths:
        text = (root / rel_path).read_text(encoding="utf-8")
        assert "gpu_utils.expected_counts_" not in text, rel_path


def test_geometric_scaling_inverse_square() -> None:
    """Expected counts should follow inverse-square scaling without shielding."""
    src = np.array([[0.0, 0.0, 0.0]])
    strength = np.array([10.0])
    d1 = 1.0
    d2 = 2.0
    lam1 = expected_counts_single_isotope(
        detector_position=np.array([d1, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam2 = expected_counts_single_isotope(
        detector_position=np.array([d2, 0.0, 0.0]),
        RFe=np.array([1.0, 0.0, 0.0]),
        RPb=np.array([1.0, 0.0, 0.0]),
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
        RFe=np.array([1.0, 1.0, 1.0]),
        RPb=np.array([1.0, 1.0, 1.0]),
        sources=src,
        strengths=strength,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    lam_blocked = expected_counts_single_isotope(
        detector_position=det,
        RFe=np.array([-1.0, -1.0, -1.0]),
        RPb=np.array([-1.0, -1.0, -1.0]),
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


def test_poisson_weight_update_prefers_higher_lambda() -> None:
    """Weight update should favor particle with higher expected Λ for given z."""
    cfg = PFConfig(num_particles=2)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    # Override continuous particles with deterministic states
    p_hi = IsotopeState(
        num_sources=1,
        positions=np.array([[1.0, 0.0, 0.0]]),
        strengths=np.array([1.0]),
        background=0.0,
    )
    p_lo = IsotopeState(
        num_sources=1,
        positions=np.array([[5.0, 0.0, 0.0]]),
        strengths=np.array([1.0]),
        background=0.0,
    )
    pf.continuous_particles = [
        IsotopeParticle(state=p_hi, log_weight=np.log(0.5)),
        IsotopeParticle(state=p_lo, log_weight=np.log(0.5)),
    ]
    pf.kernel = dummy_kernel
    z_obs = 1.0
    pf.update_continuous_pair(
        z_obs=z_obs, pose_idx=0, fe_index=0, pb_index=0, live_time_s=1.0
    )
    weights = pf.continuous_weights
    assert weights[0] > weights[1]


def test_sequence_covariance_likelihood_matches_numpy_oracle() -> None:
    """Batched same-station covariance likelihood should match a NumPy oracle."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=2,
        count_likelihood_model="gaussian",
        spectrum_count_rel_sigma=0.1,
        spectrum_count_abs_sigma=2.0,
        station_view_covariance_enable=True,
        station_view_correlated_spectrum_fraction=1.0,
        shield_contrast_likelihood_enable=False,
    )
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=None, config=cfg)
    lam = np.asarray(
        [[100.0, 120.0], [80.0, 82.0], [60.0, 65.0]],
        dtype=float,
    )
    z_obs = np.asarray([108.0, 78.0, 62.0], dtype=float)
    obs_var = np.asarray([9.0, 16.0, 25.0], dtype=float)
    obs_cov = np.asarray(
        [[9.0, 3.0, -2.0], [3.0, 16.0, 4.0], [-2.0, 4.0, 25.0]],
        dtype=float,
    )

    ll = (
        filt._log_likelihood_sequence_gpu(
            torch.as_tensor(lam, dtype=torch.float64),
            z_obs,
            obs_var,
            observation_count_covariance=obs_cov,
        )
        .cpu()
        .numpy()
    )

    expected = []
    obs_offdiag = obs_cov.copy()
    np.fill_diagonal(obs_offdiag, 0.0)
    for particle_idx in range(lam.shape[1]):
        lam_vec = lam[:, particle_idx]
        diag_var = count_likelihood_variance(
            z_obs,
            lam_vec,
            spectrum_count_rel_sigma=0.1,
            spectrum_count_abs_sigma=2.0,
            observation_count_variance=obs_var,
        )
        common = (0.1**2) * np.outer(lam_vec, lam_vec) + 2.0**2
        np.fill_diagonal(common, 0.0)
        covariance = np.diag(diag_var) + obs_offdiag + common
        residual = z_obs - lam_vec
        quad = float(residual @ np.linalg.solve(covariance, residual))
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign > 0.0
        expected.append(-0.5 * (quad + float(logdet)))

    assert np.allclose(ll, np.asarray(expected), rtol=1.0e-8, atol=1.0e-8)


def test_spectral_bin_sequence_likelihood_matches_poisson_oracle() -> None:
    """Direct PF spectrum-bin likelihood should match a batched Poisson oracle."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(num_particles=2, count_likelihood_model="poisson")
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=None, config=cfg)
    expected = np.asarray(
        [
            [[12.0, 8.0], [3.0, 5.0], [40.0, 42.0]],
            [[7.0, 10.0], [20.0, 18.0], [2.0, 4.0]],
        ],
        dtype=float,
    )
    observed = np.asarray([[11.0, 4.0, 39.0], [8.0, 19.0, 3.0]], dtype=float)

    actual = (
        filt._spectral_bin_sequence_log_likelihood_gpu(
            torch.as_tensor(expected, dtype=torch.float64),
            observed,
        )
        .detach()
        .cpu()
        .numpy()
    )
    oracle = np.sum(observed[:, :, None] * np.log(expected) - expected, axis=(0, 1))

    np.testing.assert_allclose(actual, oracle, rtol=1.0e-12, atol=1.0e-12)


def test_pair_sequence_update_uses_spectrum_bin_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PF sequence updates should prefer particles matching spectrum bins."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=2,
        count_likelihood_model="poisson",
        use_gpu=True,
        use_tempering=False,
        resample_threshold=0.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=0,
                positions=np.zeros((0, 3), dtype=float),
                strengths=np.zeros(0, dtype=float),
                background=0.0,
            ),
            log_weight=np.log(0.5),
        ),
    ]

    def fake_gpu_enabled() -> bool:
        """Pretend that the torch backend is available for this path test."""
        return True

    def fake_counts(**_kwargs: object) -> "torch.Tensor":
        """Return two particles whose scalar counts favor different spectra."""
        return torch.as_tensor([[10.0, 30.0]], dtype=torch.float64)

    def noop() -> None:
        """Skip non-likelihood side effects in this path test."""
        return None

    def noop_adapt_num_particles(**_kwargs: object) -> None:
        """Skip adaptive particle-count side effects in this path test."""
        return None

    def noop_maybe_update_convergence(**_kwargs: object) -> None:
        """Skip convergence side effects in this path test."""
        return None

    monkeypatch.setattr(filt, "_gpu_enabled", fake_gpu_enabled)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(filt, "_maybe_resample_continuous", noop)
    monkeypatch.setattr(filt, "align_continuous_labels", noop)
    monkeypatch.setattr(filt, "_advance_adapt_cooldown", noop)
    monkeypatch.setattr(filt, "adapt_num_particles", noop_adapt_num_particles)
    monkeypatch.setattr(
        filt,
        "_maybe_update_convergence",
        noop_maybe_update_convergence,
    )

    filt.update_continuous_pair_sequence(
        z_obs=np.array([10.0], dtype=float),
        pose_idx=0,
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times_s=np.array([1.0], dtype=float),
        observation_count_variances=np.array([1.0], dtype=float),
        spectrum_counts=np.array([[30.0, 0.0]], dtype=float),
        spectrum_response_template=np.array([[1.0, 0.0]], dtype=float),
        spectrum_background=np.zeros((1, 2), dtype=float),
    )

    assert filt.continuous_weights[1] > filt.continuous_weights[0]


def test_spectral_bin_chunked_likelihood_matches_full_tensor() -> None:
    """Chunked spectrum-bin likelihood should match the full KxBxN oracle."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        count_likelihood_model="student_t",
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=1.0,
        count_likelihood_df=5.0,
        spectrum_likelihood_bin_chunk=2,
    )
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=None, config=cfg)
    lam = torch.tensor(
        [[10.0, 20.0, 30.0], [4.0, 8.0, 12.0]],
        dtype=torch.float64,
    )
    observed = np.array(
        [[12.0, 3.0, 7.0, 1.0, 0.0], [2.0, 4.0, 5.0, 3.0, 1.0]],
        dtype=float,
    )
    template = np.array(
        [[0.5, 0.1, 0.2, 0.1, 0.0], [0.1, 0.3, 0.4, 0.1, 0.05]],
        dtype=float,
    )
    background = np.full_like(observed, 0.5, dtype=float)
    variance = np.full_like(observed, 1.25, dtype=float)

    expected = filt._expected_spectrum_sequence_torch(lam, template, background)
    full = filt._spectral_bin_sequence_log_likelihood_gpu(
        expected,
        observed,
        variance,
    )
    chunked = filt._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        lam,
        observed,
        template,
        background,
        variance,
    )

    assert torch.allclose(chunked, full, rtol=1.0e-12, atol=1.0e-12)


def test_deferred_pair_update_allows_scaled_roughening() -> None:
    """Deferred station updates should resample with small roughening, not freeze positions."""
    cfg = PFConfig(
        num_particles=2,
        use_tempering=True,
        deferred_resample_roughening_scale=0.15,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    captured: dict[str, float | bool] = {}

    def fake_tempered_update(
        *,
        lam_fn,
        z_obs,
        observation_count_variance=0.0,
        disable_regularize_on_resample=None,
        roughening_scale_on_resample=1.0,
    ):
        """Capture resampling options passed by the deferred update path."""
        captured["disable_regularize"] = bool(disable_regularize_on_resample)
        captured["roughening_scale"] = float(roughening_scale_on_resample)
        return 1.0, True

    pf._tempered_update = fake_tempered_update
    pf.update_continuous_pair(
        z_obs=1.0,
        pose_idx=0,
        fe_index=0,
        pb_index=0,
        live_time_s=1.0,
        defer_resample=True,
    )

    assert captured["disable_regularize"] is False
    assert captured["roughening_scale"] == 0.15


def test_surface_position_prior_initializes_and_roughens_on_surfaces() -> None:
    """Surface source prior should keep PF particles on known source surfaces."""
    env = EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0)
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((1, 1),),
    )
    cfg = PFConfig(
        num_particles=4,
        use_gpu=False,
        init_num_sources=(1, 1),
        init_grid_spacing_m=1.0,
        init_grid_repeats=1,
        position_min=(0.0, 0.0, 0.0),
        position_max=(env.size_x, env.size_y, env.size_z),
        source_position_prior="surface",
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1

    pf = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=dummy_kernel,
        config=cfg,
        obstacle_grid=grid,
        obstacle_height_m=1.0,
    )
    pf.regularize_continuous(
        sigma_pos=np.array([0.2, 0.2, 0.2]),
        strength_log_sigma=0.0,
    )

    assert pf.continuous_particles
    for particle in pf.continuous_particles:
        for position in particle.state.positions:
            assert is_allowed_source_surface_position(
                position,
                env,
                grid,
                obstacle_height_m=1.0,
            )


def test_student_t_count_likelihood_softens_model_mismatch() -> None:
    """Robust count likelihood should not over-trust a simplified transport kernel."""
    z_obs = np.array([100.0], dtype=float)
    lambda_good = np.array([100.0], dtype=float)
    lambda_mismatch = np.array([50.0], dtype=float)

    poisson_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="poisson",
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="poisson",
    )
    robust_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_rel_sigma=0.4,
        spectrum_count_rel_sigma=0.2,
        spectrum_count_abs_sigma=5.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_rel_sigma=0.4,
        spectrum_count_rel_sigma=0.2,
        spectrum_count_abs_sigma=5.0,
        student_t_df=5.0,
    )

    assert robust_gap > 0.0
    assert robust_gap < poisson_gap


def test_observation_count_variance_softens_spectrum_unfolding_update() -> None:
    """Spectrum decomposition variance should reduce overconfident count updates."""
    z_obs = np.array([100.0], dtype=float)
    lambda_good = np.array([100.0], dtype=float)
    lambda_mismatch = np.array([60.0], dtype=float)

    certain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        observation_count_variance=0.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        observation_count_variance=0.0,
        student_t_df=5.0,
    )
    uncertain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        observation_count_variance=400.0,
        student_t_df=5.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        observation_count_variance=400.0,
        student_t_df=5.0,
    )

    assert uncertain_gap > 0.0
    assert uncertain_gap < certain_gap


def test_matrix_count_likelihood_normal_alias_matches_scalar_gaussian() -> None:
    """The batched NumPy likelihood path should normalize model aliases."""
    cfg = PFConfig(
        num_particles=1,
        count_likelihood_model="normal",
        transport_model_rel_sigma=0.15,
        spectrum_count_abs_sigma=2.0,
        count_likelihood_df=3.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    z_obs = np.array([42.0, 18.0], dtype=float)
    lambda_kp = np.array(
        [
            [39.0, 23.0],
            [21.0, 12.0],
        ],
        dtype=float,
    )
    obs_var = np.array([4.0, 9.0], dtype=float)

    actual = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=obs_var,
    )
    expected = np.array(
        [
            count_log_likelihood(
                z_obs,
                lambda_kp[:, idx],
                model="normal",
                transport_model_rel_sigma=0.15,
                spectrum_count_abs_sigma=2.0,
                observation_count_variance=obs_var,
                student_t_df=3.0,
            )
            for idx in range(lambda_kp.shape[1])
        ],
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_matrix_count_likelihood_rejects_unknown_model() -> None:
    """The batched likelihood path should not silently reinterpret bad models."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="not_a_model")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)

    with pytest.raises(ValueError, match="Unknown count likelihood model"):
        filt._count_log_likelihood_matrix_np(
            np.array([1.0], dtype=float),
            np.array([[1.0]], dtype=float),
        )


def test_matrix_count_likelihood_scalar_variance_broadcasts() -> None:
    """Scalar unfolding variance should apply to every batched measurement."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="gaussian")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    z_obs = np.array([12.0, 18.0, 25.0], dtype=float)
    lambda_kp = np.array(
        [
            [11.0, 14.0],
            [17.0, 20.0],
            [23.0, 26.0],
        ],
        dtype=float,
    )

    scalar = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=4.0,
    )
    repeated = filt._count_log_likelihood_matrix_np(
        z_obs,
        lambda_kp,
        observation_count_variance=np.full(z_obs.shape, 4.0, dtype=float),
    )

    np.testing.assert_allclose(scalar, repeated, rtol=1.0e-12, atol=1.0e-12)


def test_matrix_count_likelihood_rejects_mismatched_variance() -> None:
    """Batched likelihoods should reject ambiguous observation variance shapes."""
    cfg = PFConfig(num_particles=1, count_likelihood_model="gaussian")
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)

    with pytest.raises(ValueError, match="observation_count_variance"):
        filt._count_log_likelihood_matrix_np(
            np.array([12.0, 18.0, 25.0], dtype=float),
            np.array([[11.0], [17.0], [23.0]], dtype=float),
            observation_count_variance=np.array([4.0, 9.0], dtype=float),
        )


def test_count_likelihood_aliases_are_consistent_across_gpu_increment() -> None:
    """Scalar, matrix, and torch increments should agree on model aliases."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        count_likelihood_model="normal",
        transport_model_rel_sigma=0.2,
        spectrum_count_abs_sigma=1.5,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    lam = torch.as_tensor([37.0, 41.0], dtype=torch.float64)

    actual = (
        filt._log_likelihood_increment_gpu(
            lam,
            z_obs=39.0,
            observation_count_variance=4.0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    expected = np.array(
        [
            count_log_likelihood(
                np.array([39.0], dtype=float),
                np.array([value], dtype=float),
                model="gaussian",
                transport_model_rel_sigma=0.2,
                spectrum_count_abs_sigma=1.5,
                observation_count_variance=4.0,
            )
            for value in (37.0, 41.0)
        ],
        dtype=float,
    )

    assert normalize_count_likelihood_model("") == "poisson"
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("model", ["poisson", "gaussian", "student_t"])
def test_sequence_gpu_likelihood_matches_scalar_sum(model: str) -> None:
    """Batched sequence likelihoods should equal summed scalar increments."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        count_likelihood_model=model,
        transport_model_rel_sigma=0.1,
        transport_model_abs_sigma=0.5,
        spectrum_count_rel_sigma=0.05,
        spectrum_count_abs_sigma=0.25,
        low_count_abs_sigma=1.0,
        low_count_transition_counts=20.0,
        count_likelihood_df=4.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    lam_kn = torch.as_tensor(
        [
            [37.0, 41.0],
            [8.0, 12.0],
            [120.0, 135.0],
        ],
        dtype=torch.float64,
    )
    z_obs = np.array([39.0, 9.0, 128.0], dtype=float)
    obs_var = np.array([4.0, 1.5, 16.0], dtype=float)

    actual = filt._log_likelihood_sequence_gpu(lam_kn, z_obs, obs_var)
    expected = torch.zeros(lam_kn.shape[1], dtype=torch.float64)
    for idx, z_val in enumerate(z_obs):
        expected = expected + filt._log_likelihood_increment_gpu(
            lam_kn[idx],
            z_obs=float(z_val),
            observation_count_variance=float(obs_var[idx]),
        )

    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_shield_contrast_likelihood_matches_numpy_signature_oracle() -> None:
    """Same-station contrast likelihood should use shield signatures robustly."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        shield_contrast_likelihood_enable=True,
        shield_contrast_likelihood_weight=1.0,
        shield_contrast_log_sigma_floor=0.5,
        shield_contrast_log_sigma_ceiling=2.0,
        shield_contrast_min_count=25.0,
        shield_contrast_likelihood_df=5.0,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    lambda_kn = np.array(
        [
            [10000.0, 10000.0, 5000.0],
            [100.0, 10000.0, 5000.0],
            [3000.0, 2500.0, 5000.0],
        ],
        dtype=float,
    )
    z_obs = np.array([10000.0, 100.0, 3000.0], dtype=float)
    obs_var = np.array([1.0e8, 1.0e8, 1.0e8], dtype=float)

    actual = (
        filt._shield_contrast_sequence_log_likelihood_gpu(
            torch.as_tensor(lambda_kn, dtype=torch.float64),
            z_obs,
            obs_var,
        )
        .detach()
        .cpu()
        .numpy()
    )

    min_count = 25.0
    sigma_floor = 0.5
    sigma_ceiling = 2.0
    df = 5.0
    z_safe = np.maximum(z_obs[:, None], min_count)
    lambda_safe = np.maximum(lambda_kn, min_count)
    log_z = np.log(z_safe)
    log_lambda = np.log(lambda_safe)
    log_var = np.clip(
        obs_var[:, None] / np.maximum(z_safe**2, 1.0e-12) + sigma_floor**2,
        sigma_floor**2,
        sigma_ceiling**2,
    )
    view_weight = 1.0 / log_var
    weight_sum = np.maximum(np.sum(view_weight, axis=0, keepdims=True), 1.0e-12)
    obs_center = log_z - np.sum(view_weight * log_z, axis=0, keepdims=True) / weight_sum
    pred_center = log_lambda - (
        np.sum(view_weight * log_lambda, axis=0, keepdims=True) / weight_sum
    )
    residual = obs_center - pred_center
    expected = np.sum(
        -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * log_var))
        - 0.5 * np.log(log_var),
        axis=0,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    assert actual[0] > actual[1]


def test_shield_view_ratio_likelihood_matches_dirichlet_multinomial_oracle() -> None:
    """Shield-view ratio likelihood should match a Dirichlet-multinomial oracle."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=2,
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=0.7,
        shield_view_ratio_likelihood_concentration=20.0,
        shield_view_ratio_likelihood_min_total_count=1.0,
        shield_view_ratio_likelihood_min_views=2,
    )
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=None, config=cfg)
    lambda_kn = np.asarray(
        [
            [90.0, 50.0],
            [10.0, 50.0],
        ],
        dtype=float,
    )
    z_obs = np.asarray([90.0, 10.0], dtype=float)

    actual = (
        filt._shield_view_ratio_sequence_log_likelihood_gpu(
            torch.as_tensor(lambda_kn, dtype=torch.float64),
            z_obs,
        )
        .detach()
        .cpu()
        .numpy()
    )

    total = float(np.sum(z_obs))
    concentration = 20.0
    expected = []
    for particle_idx in range(lambda_kn.shape[1]):
        probabilities = lambda_kn[:, particle_idx] / np.sum(lambda_kn[:, particle_idx])
        alpha = concentration * probabilities
        ll = math.lgamma(float(np.sum(alpha))) - math.lgamma(
            float(np.sum(alpha) + total)
        )
        ll += sum(
            math.lgamma(float(alpha_k + z_k)) - math.lgamma(float(alpha_k))
            for alpha_k, z_k in zip(alpha, z_obs)
        )
        expected.append(0.7 * ll)

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1.0e-12, atol=1.0e-12)
    assert actual[0] > actual[1]


def test_spectrum_sequence_update_keeps_shield_view_ratio_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct spectral PF updates should retain same-station shield-ratio evidence."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=2,
        use_tempering=True,
        shield_contrast_likelihood_enable=False,
        shield_view_ratio_likelihood_enable=True,
        shield_view_ratio_likelihood_weight=1.0,
        shield_view_ratio_likelihood_concentration=20.0,
        shield_view_ratio_likelihood_min_total_count=1.0,
        shield_view_ratio_likelihood_min_views=2,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=math.log(0.5),
        ),
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[2.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=math.log(0.5),
        ),
    ]
    calls: dict[str, object] = {}

    def fake_counts(
        *,
        pose_idx: int,
        fe_indices: np.ndarray,
        pb_indices: np.ndarray,
        live_times_s: np.ndarray,
    ) -> "torch.Tensor":
        """Return two particles with different shield-view count ratios."""
        del pose_idx, fe_indices, pb_indices, live_times_s
        return torch.as_tensor([[90.0, 50.0], [10.0, 50.0]], dtype=torch.float64)

    def fake_spectrum_ll(
        lam_kn: "torch.Tensor",
        observed_spectrum_kb: np.ndarray,
        response_template_kb: np.ndarray,
        background_spectrum_kb: np.ndarray,
        observation_spectrum_variance_kb: np.ndarray | None = None,
    ) -> "torch.Tensor":
        """Return a neutral spectral term so shield-ratio evidence is isolated."""
        del (
            observed_spectrum_kb,
            response_template_kb,
            background_spectrum_kb,
            observation_spectrum_variance_kb,
        )
        calls["spectrum_shape"] = tuple(lam_kn.shape)
        return torch.zeros(int(lam_kn.shape[1]), dtype=torch.float64)

    def fake_tempered_update_likelihood(
        ll_fn: object,
        **_kwargs: object,
    ) -> tuple[float, bool]:
        """Evaluate the likelihood callback and record particle log increments."""
        ll_t = ll_fn()
        calls["ll"] = ll_t.detach().cpu().numpy()
        return 2.0, False

    monkeypatch.setattr(filt, "_gpu_enabled", lambda: True)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(
        filt,
        "_spectral_bin_sequence_log_likelihood_from_lambda_gpu",
        fake_spectrum_ll,
    )
    monkeypatch.setattr(
        filt,
        "_tempered_update_likelihood",
        fake_tempered_update_likelihood,
    )
    monkeypatch.setattr(filt, "adapt_num_particles", lambda **_kwargs: None)
    monkeypatch.setattr(filt, "align_continuous_labels", lambda: None)
    monkeypatch.setattr(filt, "_advance_adapt_cooldown", lambda: None)
    monkeypatch.setattr(filt, "_maybe_update_convergence", lambda **_kwargs: None)

    filt.update_continuous_pair_sequence(
        z_obs=np.array([90.0, 10.0], dtype=float),
        pose_idx=0,
        fe_indices=np.array([0, 1], dtype=int),
        pb_indices=np.array([0, 1], dtype=int),
        live_times_s=np.array([1.0, 1.0], dtype=float),
        observation_count_variances=np.array([1.0, 1.0], dtype=float),
        spectrum_counts=np.ones((2, 4), dtype=float),
        spectrum_response_template=np.full((2, 4), 0.25, dtype=float),
        spectrum_background=np.zeros((2, 4), dtype=float),
    )

    assert calls["spectrum_shape"] == (2, 2)
    ll = np.asarray(calls["ll"], dtype=float)
    assert ll[0] > ll[1]


def test_pair_sequence_update_uses_batched_gpu_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint shield-program updates should call the batched sequence path once."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(num_particles=1, use_gpu=True, use_tempering=True)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    calls: dict[str, object] = {}

    def fake_counts(
        *,
        pose_idx: int,
        fe_indices: np.ndarray,
        pb_indices: np.ndarray,
        live_times_s: np.ndarray,
    ) -> "torch.Tensor":
        """Record batched expected-count inputs and return a deterministic tensor."""
        calls["counts"] = (
            int(pose_idx),
            tuple(np.asarray(fe_indices, dtype=int)),
            tuple(np.asarray(pb_indices, dtype=int)),
            tuple(np.asarray(live_times_s, dtype=float)),
        )
        return torch.as_tensor([[10.0], [20.0]], dtype=torch.float64)

    def fake_ll(
        lam_kn: "torch.Tensor",
        z_obs: np.ndarray,
        observation_count_variances: np.ndarray,
        observation_count_covariance: np.ndarray | None = None,
    ) -> "torch.Tensor":
        """Record batched likelihood inputs and return one particle increment."""
        calls["ll"] = (
            tuple(np.asarray(z_obs, dtype=float)),
            tuple(np.asarray(observation_count_variances, dtype=float)),
            None
            if observation_count_covariance is None
            else tuple(np.asarray(observation_count_covariance, dtype=float).shape),
            tuple(lam_kn.shape),
        )
        return torch.as_tensor([1.0], dtype=torch.float64)

    def fake_tempered_update_likelihood(
        ll_fn: object,
        **_kwargs: object,
    ) -> tuple[float, bool]:
        """Evaluate the likelihood callback once without resampling."""
        ll_t = ll_fn()
        calls["tempered_ll"] = tuple(ll_t.shape)
        return 1.0, False

    def fake_gpu_enabled() -> bool:
        """Pretend that the torch backend is available for this path test."""
        return True

    def noop_adapt_num_particles(**_kwargs: object) -> None:
        """Skip adaptive particle-count side effects in this path test."""
        return None

    def noop_align_continuous_labels() -> None:
        """Skip label alignment side effects in this path test."""
        return None

    def noop_advance_adapt_cooldown() -> None:
        """Skip adaptive cooldown side effects in this path test."""
        return None

    def noop_maybe_update_convergence(**_kwargs: object) -> None:
        """Skip convergence side effects in this path test."""
        return None

    monkeypatch.setattr(filt, "_gpu_enabled", fake_gpu_enabled)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(filt, "_log_likelihood_sequence_gpu", fake_ll)
    monkeypatch.setattr(
        filt,
        "_tempered_update_likelihood",
        fake_tempered_update_likelihood,
    )
    monkeypatch.setattr(filt, "adapt_num_particles", noop_adapt_num_particles)
    monkeypatch.setattr(
        filt,
        "align_continuous_labels",
        noop_align_continuous_labels,
    )
    monkeypatch.setattr(
        filt,
        "_advance_adapt_cooldown",
        noop_advance_adapt_cooldown,
    )
    monkeypatch.setattr(
        filt,
        "_maybe_update_convergence",
        noop_maybe_update_convergence,
    )

    filt.update_continuous_pair_sequence(
        z_obs=np.array([8.0, 18.0], dtype=float),
        pose_idx=0,
        fe_indices=np.array([1, 2], dtype=int),
        pb_indices=np.array([3, 4], dtype=int),
        live_times_s=np.array([1.5, 2.0], dtype=float),
        observation_count_variances=np.array([2.0, 3.0], dtype=float),
    )

    assert calls["counts"] == (0, (1, 2), (3, 4), (1.5, 2.0))
    assert calls["ll"] == ((8.0, 18.0), (2.0, 3.0), None, (2, 1))
    assert calls["tempered_ll"] == (1,)


def test_pair_sequence_profiles_strengths_before_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile strengths before every likelihood and after final adaptation."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        use_gpu=True,
        use_tempering=True,
        conditional_strength_profile_before_likelihood=True,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 1.5], dtype=float)]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([1.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    calls: list[tuple[str, float, float]] = []

    def fake_profile(
        data: MeasurementData,
        **kwargs: object,
    ) -> None:
        """Set strength from the current position and record profile ordering."""
        assert kwargs["reweight_override"] is False
        assert kwargs["suppress_prune_after_refit"] is True
        np.testing.assert_allclose(data.detector_positions[:, 2], [1.5, 1.5])
        state = filt.continuous_particles[0].state
        position_x = float(state.positions[0, 0])
        state.strengths[:] = 10.0 * position_x
        calls.append(("profile", position_x, float(state.strengths[0])))

    def fake_counts(**_kwargs: object) -> "torch.Tensor":
        """Observe the profiled strength when the likelihood is evaluated."""
        state = filt.continuous_particles[0].state
        position_x = float(state.positions[0, 0])
        strength = float(state.strengths[0])
        calls.append(("likelihood", position_x, strength))
        return torch.full((2, 1), float(strength), dtype=torch.float64)

    def fake_tempered(
        ll_fn: object,
        **kwargs: object,
    ) -> tuple[float, bool]:
        """Simulate roughening and the subsequent likelihood recomputation."""
        ll_fn()
        state = filt.continuous_particles[0].state
        state.positions[0, 0] = 2.0
        state.strengths[0] = -1.0
        calls.append(("roughen", 2.0, -1.0))
        refresh = kwargs["refresh_state_after_resample"]
        assert callable(refresh)
        refresh()
        ll_fn()
        return 1.0, True

    def fake_adapt(**_kwargs: object) -> None:
        """Simulate final particle adaptation changing the sampled position."""
        state = filt.continuous_particles[0].state
        state.positions[0, 0] = 3.0
        state.strengths[0] = -1.0
        calls.append(("adapt", 3.0, -1.0))

    monkeypatch.setattr(filt, "_gpu_enabled", lambda: True)
    monkeypatch.setattr(filt, "refit_strengths_for_particles", fake_profile)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(
        filt,
        "_log_likelihood_sequence_gpu",
        lambda lam, *_args, **_kwargs: torch.zeros(lam.shape[1]),
    )
    monkeypatch.setattr(filt, "_tempered_update_likelihood", fake_tempered)
    monkeypatch.setattr(filt, "adapt_num_particles", fake_adapt)
    monkeypatch.setattr(filt, "align_continuous_labels", lambda: None)
    monkeypatch.setattr(filt, "_advance_adapt_cooldown", lambda: None)
    monkeypatch.setattr(filt, "_maybe_update_convergence", lambda **_kwargs: None)

    filt.update_continuous_pair_sequence(
        z_obs=np.array([20.0, 30.0], dtype=float),
        pose_idx=0,
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times_s=np.array([1.0, 1.0], dtype=float),
        observation_count_variances=np.array([2.0, 3.0], dtype=float),
    )

    assert calls == [
        ("profile", 1.0, 10.0),
        ("likelihood", 1.0, 10.0),
        ("roughen", 2.0, -1.0),
        ("profile", 2.0, 20.0),
        ("likelihood", 2.0, 20.0),
        ("adapt", 3.0, -1.0),
        ("profile", 3.0, 30.0),
    ]
    assert filt.continuous_particles[0].state.strengths[0] == pytest.approx(30.0)


def test_tempering_refreshes_profile_state_after_roughening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tempering must refresh deterministic state before likelihood recomputation."""
    torch = pytest.importorskip("torch")
    filt = IsotopeParticleFilter(
        isotope="Cs-137",
        kernel=None,
        config=PFConfig(
            num_particles=1,
            target_ess_ratio=2.0,
            resample_threshold=2.0,
            max_resamples_per_observation=1,
            temper_resample_cooldown_steps=0,
        ),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([10.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    calls: list[tuple[str, float, float]] = []

    def likelihood() -> "torch.Tensor":
        """Record the position and strength used by each likelihood evaluation."""
        state = filt.continuous_particles[0].state
        calls.append(
            (
                "likelihood",
                float(state.positions[0, 0]),
                float(state.strengths[0]),
            )
        )
        return torch.zeros(1, dtype=torch.float64)

    def fake_resample(**_kwargs: object) -> None:
        """Replace the particle with an intentionally stale roughened state."""
        state = filt.continuous_particles[0].state
        state.positions[0, 0] = 2.0
        state.strengths[0] = -1.0
        filt.last_resample_ess = True
        calls.append(("roughen", 2.0, -1.0))

    def refresh_profile() -> None:
        """Condition strength on the newly roughened particle position."""
        state = filt.continuous_particles[0].state
        position_x = float(state.positions[0, 0])
        state.strengths[0] = 10.0 * position_x
        calls.append(("profile", position_x, float(state.strengths[0])))

    monkeypatch.setattr(filt, "_maybe_resample_continuous", fake_resample)

    _, resampled = filt._tempered_update_likelihood(
        likelihood,
        refresh_state_after_resample=refresh_profile,
    )

    assert resampled is True
    assert calls == [
        ("likelihood", 1.0, 10.0),
        ("roughen", 2.0, -1.0),
        ("profile", 2.0, 20.0),
        ("likelihood", 2.0, 20.0),
    ]


def test_non_tempered_pair_sequence_reprofiles_after_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-tempered final roughening must not leave stale strength state."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=1,
        use_gpu=True,
        use_tempering=False,
        conditional_strength_profile_before_likelihood=True,
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 1.5], dtype=float)]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 1
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
                strengths=np.array([-1.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]
    calls: list[tuple[str, float, float]] = []

    def fake_profile(_data: MeasurementData, **_kwargs: object) -> None:
        """Profile strength as a deterministic function of position."""
        state = filt.continuous_particles[0].state
        position_x = float(state.positions[0, 0])
        state.strengths[0] = 10.0 * position_x
        calls.append(("profile", position_x, float(state.strengths[0])))

    def fake_counts(**_kwargs: object) -> "torch.Tensor":
        """Record state used for the sole non-tempered likelihood."""
        state = filt.continuous_particles[0].state
        calls.append(
            (
                "likelihood",
                float(state.positions[0, 0]),
                float(state.strengths[0]),
            )
        )
        return torch.full((1, 1), float(state.strengths[0]), dtype=torch.float64)

    def fake_resample(**_kwargs: object) -> None:
        """Simulate the final non-tempered roughening operation."""
        state = filt.continuous_particles[0].state
        state.positions[0, 0] = 2.0
        state.strengths[0] = -1.0
        filt.last_resample_ess = True
        calls.append(("roughen", 2.0, -1.0))

    monkeypatch.setattr(filt, "_gpu_enabled", lambda: True)
    monkeypatch.setattr(filt, "refit_strengths_for_particles", fake_profile)
    monkeypatch.setattr(
        filt,
        "_continuous_expected_counts_pair_sequence_torch",
        fake_counts,
    )
    monkeypatch.setattr(
        filt,
        "_log_likelihood_sequence_gpu",
        lambda lam, *_args, **_kwargs: torch.zeros(lam.shape[1]),
    )
    monkeypatch.setattr(filt, "_maybe_resample_continuous", fake_resample)
    monkeypatch.setattr(filt, "adapt_num_particles", lambda **_kwargs: None)
    monkeypatch.setattr(filt, "align_continuous_labels", lambda: None)
    monkeypatch.setattr(filt, "_advance_adapt_cooldown", lambda: None)
    monkeypatch.setattr(filt, "_maybe_update_convergence", lambda **_kwargs: None)

    filt.update_continuous_pair_sequence(
        z_obs=np.array([20.0], dtype=float),
        pose_idx=0,
        fe_indices=np.array([0], dtype=int),
        pb_indices=np.array([0], dtype=int),
        live_times_s=np.array([1.0], dtype=float),
        observation_count_variances=np.array([2.0], dtype=float),
    )

    assert calls == [
        ("profile", 1.0, 10.0),
        ("likelihood", 1.0, 10.0),
        ("roughen", 2.0, -1.0),
        ("profile", 2.0, 20.0),
    ]
    assert filt.continuous_particles[0].state.strengths[0] == pytest.approx(20.0)


def test_pre_likelihood_profile_rejects_historical_refit_reweight() -> None:
    """Pre-profile state updates must not rebase weights against old data."""
    with pytest.raises(ValueError, match="cannot be combined"):
        PFConfig(
            conditional_strength_profile_before_likelihood=True,
            conditional_strength_refit_reweight=True,
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        RotatingShieldPFConfig(
            conditional_strength_profile_before_likelihood=True,
            conditional_strength_refit_reweight=True,
        )


def test_identical_source_state_gpu_compression_preserves_sequence_counts() -> None:
    """Duplicate source states should share response work without changing counts."""
    torch = pytest.importorskip("torch")
    cfg = PFConfig(
        num_particles=3,
        use_gpu=True,
        gpu_device="cpu",
        gpu_dtype="float64",
    )
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.orientations = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ]
    dummy_kernel.num_sources = 2
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    shared_positions = np.array(
        [[1.0, 0.2, 0.1], [1.5, 0.7, 0.4]],
        dtype=float,
    )
    shared_strengths = np.array([100.0, 40.0], dtype=float)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=shared_positions.copy(),
                strengths=shared_strengths.copy(),
                background=background,
            ),
            log_weight=-math.log(3.0),
        )
        for background in (0.0, 2.5, 8.0)
    ]

    compressed = filt._continuous_expected_counts_pair_sequence_torch(
        pose_idx=0,
        fe_indices=np.array([0, 1, 0], dtype=int),
        pb_indices=np.array([1, 0, 1], dtype=int),
        live_times_s=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    uncompressed = filt._continuous_expected_counts_pair_sequence_torch_uncompressed(
        pose_idx=0,
        fe_indices=np.array([0, 1, 0], dtype=int),
        pb_indices=np.array([1, 0, 1], dtype=int),
        live_times_s=np.array([1.0, 2.0, 3.0], dtype=float),
    )

    assert compressed.shape == (3, 3)
    assert torch.allclose(compressed, uncompressed, rtol=1.0e-12, atol=1.0e-12)


def test_identical_source_state_cpu_group_refresh_preserves_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped all-history refits should compress duplicate source rows exactly."""
    cfg = PFConfig(num_particles=3, use_gpu=False)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0], dtype=float)]
    dummy_kernel.num_sources = 2
    filt = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    shared_positions = np.array(
        [[1.0, 0.2, 0.1], [1.5, 0.7, 0.4]],
        dtype=float,
    )
    shared_strengths = np.array([100.0, 40.0], dtype=float)
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=2,
                positions=shared_positions.copy(),
                strengths=shared_strengths.copy(),
                background=background,
            ),
            log_weight=-math.log(3.0),
        )
        for background in (0.0, 2.5, 8.0)
    ]
    data = MeasurementData(
        z_k=np.array([10.0, 12.0], dtype=float),
        observation_variances=np.zeros(2, dtype=float),
        detector_positions=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=float,
        ),
        fe_indices=np.array([0, 0], dtype=int),
        pb_indices=np.array([0, 0], dtype=int),
        live_times=np.array([1.0, 2.0], dtype=float),
    )
    call_count = 0
    original = filt.continuous_kernel.kernel_values_selected_pairs_for_detectors

    def counted_kernel_values(*args: object, **kwargs: object) -> object:
        """Count response evaluations before delegating to the real kernel."""
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        filt.continuous_kernel,
        "kernel_values_selected_pairs_for_detectors",
        counted_kernel_values,
    )
    lambda_m, lambda_total = filt._lambda_components_for_particle_group(
        data,
        [0, 1, 2],
        2,
    )

    assert call_count == 1
    assert lambda_m.shape == (2, 3, 2)
    expected_total = np.stack(
        [
            np.sum(lambda_m[:, idx, :], axis=1)
            + filt.continuous_particles[idx].state.background * data.live_times
            for idx in range(3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(lambda_total, expected_total, rtol=1.0e-12)


def test_low_count_variance_floor_decays_for_informative_counts() -> None:
    """Low-count uncertainty should protect weak spectra without weakening high counts."""
    z_obs = np.array([5.0, 5000.0], dtype=float)
    lambda_obs = np.array([4.0, 5100.0], dtype=float)

    base_variance = count_likelihood_variance(z_obs, lambda_obs)
    robust_variance = count_likelihood_variance(
        z_obs,
        lambda_obs,
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
    )

    assert robust_variance[0] - base_variance[0] > 300.0
    assert robust_variance[1] - base_variance[1] < 110.0


def test_geant4_likelihood_profile_keeps_high_count_residuals_informative() -> None:
    """Validated Geant4 likelihood defaults should not mask large residuals."""
    z_obs = np.array([1_000_000.0], dtype=float)
    lambda_good = np.array([1_000_000.0], dtype=float)
    lambda_underfit = np.array([500_000.0], dtype=float)

    variance = count_likelihood_variance(
        z_obs,
        lambda_good,
        transport_model_rel_sigma=DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
        spectrum_count_rel_sigma=DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    )
    relative_sigma = float(np.sqrt(variance[0]) / z_obs[0])

    assert 0.10 <= relative_sigma <= 0.12

    previous_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_rel_sigma=0.50,
        spectrum_count_rel_sigma=0.15,
        student_t_df=3.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_underfit,
        model="student_t",
        transport_model_rel_sigma=0.50,
        spectrum_count_rel_sigma=0.15,
        student_t_df=3.0,
    )
    validated_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_rel_sigma=DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
        spectrum_count_rel_sigma=DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
        student_t_df=DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    ) - count_log_likelihood(
        z_obs,
        lambda_underfit,
        model="student_t",
        transport_model_rel_sigma=DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
        spectrum_count_rel_sigma=DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
        student_t_df=DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    )

    assert validated_gap > previous_gap * 2.0


def test_count_likelihood_variance_torch_matches_numpy() -> None:
    """Torch likelihood variance should match the shared NumPy equation."""
    torch = pytest.importorskip("torch")
    z_obs = np.array([[5.0], [500.0]], dtype=float)
    lambda_obs = np.array(
        [
            [4.0, 6.0],
            [510.0, 480.0],
        ],
        dtype=float,
    )
    observation_variance = np.array([[2.0], [7.0]], dtype=float)
    kwargs = {
        "transport_model_rel_sigma": 0.2,
        "transport_model_abs_sigma": 3.0,
        "spectrum_count_rel_sigma": 0.15,
        "spectrum_count_abs_sigma": 5.0,
        "low_count_abs_sigma": 20.0,
        "low_count_transition_counts": 100.0,
        "observation_count_variance": observation_variance,
    }

    expected = count_likelihood_variance(
        z_obs,
        lambda_obs,
        **kwargs,
    )
    actual = count_likelihood_variance_torch(
        torch.as_tensor(z_obs, dtype=torch.float64),
        torch.as_tensor(lambda_obs, dtype=torch.float64),
        **{
            **kwargs,
            "observation_count_variance": torch.as_tensor(
                observation_variance,
                dtype=torch.float64,
            ),
        },
    )

    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_transport_absolute_floor_softens_low_count_model_mismatch() -> None:
    """Absolute transport mismatch should prevent low counts from over-pruning particles."""
    z_obs = np.array([8.0], dtype=float)
    lambda_good = np.array([8.0], dtype=float)
    lambda_mismatch = np.array([20.0], dtype=float)

    certain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_abs_sigma=0.0,
        student_t_df=3.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_abs_sigma=0.0,
        student_t_df=3.0,
    )
    uncertain_gap = count_log_likelihood(
        z_obs,
        lambda_good,
        model="student_t",
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
        student_t_df=3.0,
    ) - count_log_likelihood(
        z_obs,
        lambda_mismatch,
        model="student_t",
        transport_model_abs_sigma=10.0,
        low_count_abs_sigma=20.0,
        low_count_transition_counts=100.0,
        student_t_df=3.0,
    )

    assert uncertain_gap > 0.0
    assert uncertain_gap < certain_gap


def test_resampling_increases_neff() -> None:
    """Highly skewed weights should be flattened after resampling."""
    cfg = PFConfig(num_particles=3)
    dummy_kernel = type("K", (), {})()
    dummy_kernel.poses = [np.array([0.0, 0.0, 0.0])]
    dummy_kernel.orientations = [np.array([1.0, 0.0, 0.0])]
    dummy_kernel.num_sources = 1
    pf = IsotopeParticleFilter(isotope="Cs-137", kernel=dummy_kernel, config=cfg)
    pf.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0),
            log_weight=np.log(0.99),
        ),
        IsotopeParticle(
            state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0),
            log_weight=np.log(0.005),
        ),
        IsotopeParticle(
            state=IsotopeState(0, np.zeros((0, 3)), np.zeros(0), 0.0),
            log_weight=np.log(0.005),
        ),
    ]
    before = 1.0 / np.sum(pf.continuous_weights**2)
    pf._maybe_resample_continuous()
    after = 1.0 / np.sum(pf.continuous_weights**2)
    assert after > before
    assert np.allclose(pf.continuous_weights, np.ones(3) / 3, rtol=1e-3)
