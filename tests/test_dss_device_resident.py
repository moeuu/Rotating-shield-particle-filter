"""Regression tests for multi-isotope device-resident DSS assembly."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from measurement.shielding import generate_octant_orientations
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticle
from planning.dss_pp import (
    ShieldProgram,
    _DeviceJointProgramSpectrumComponents,
    _full_spectrum_joint_program_components,
)
from pure_pf_test_support import approved_full_spectrum_model
from test_planning import _state_on_filter


_ISOTOPES = ("Co-60", "Cs-137")
_CARDINALITIES_BY_ISOTOPE = {
    "Co-60": (0, 1, 2),
    "Cs-137": (2, 0, 1),
}


def _line_mu_by_isotope(model: object) -> dict[str, tuple[dict[str, float], ...]]:
    """Return line attenuation metadata for the two-isotope test model."""
    line_identity = tuple(getattr(model, "line_identity"))
    return {
        isotope: tuple(
            {
                "energy_keV": float(line["energy_keV"]),
                "weight": float(line["branching_weight"]),
                "fe": float(line["mu_fe_cm_inv"]),
                "pb": float(line["mu_pb_cm_inv"]),
            }
            for line in line_identity
            if str(line["isotope"]) == isotope
        )
        for isotope in _ISOTOPES
    }


def _multi_isotope_estimator(
    *,
    use_gpu: bool,
    gpu_device: str,
) -> RotatingShieldPFEstimator:
    """Build aligned particles with heterogeneous isotope cardinalities."""
    model = approved_full_spectrum_model()
    estimator = RotatingShieldPFEstimator(
        isotopes=_ISOTOPES,
        surface_diagnostic_points=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        shield_normals=np.asarray(
            generate_octant_orientations(),
            dtype=np.float64,
        ),
        mu_by_isotope={
            isotope: {"fe": 0.0, "pb": 0.0} for isotope in _ISOTOPES
        },
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            max_sources=2,
            variable_cardinality=True,
            init_num_sources=(0, 2),
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            planning_eig_samples=3,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        line_mu_by_isotope=_line_mu_by_isotope(model),
        full_spectrum_generative_model=model,
        random_seed=9,
    )
    estimator.add_measurement_pose(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    )
    estimator._ensure_kernel_cache()
    for isotope in _ISOTOPES:
        particle_filter = estimator.filters[isotope]
        original_particles = particle_filter.continuous_particles
        particles = []
        for particle_index, cardinality in enumerate(
            _CARDINALITIES_BY_ISOTOPE[isotope]
        ):
            source_positions = np.zeros((cardinality, 3), dtype=np.float64)
            source_strengths = (
                np.arange(1, cardinality + 1, dtype=np.float64)
                * float(10 + particle_index)
            )
            particles.append(
                IsotopeParticle(
                    state=_state_on_filter(
                        particle_filter,
                        source_positions,
                        source_strengths,
                    ),
                    log_weight=float(np.log(1.0 / 3.0)),
                    joint_row_identity=(
                        original_particles[particle_index].joint_row_identity
                    ),
                )
            )
        particle_filter.continuous_particles = particles
    return estimator


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_multi_isotope_dss_components_remain_device_resident(
    device_name: str,
) -> None:
    """Match host physics while retaining padded multi-isotope tensors."""
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    detector_positions = np.asarray(
        [[1.0, 0.0, 0.5], [1.5, 0.5, 0.5]],
        dtype=np.float64,
    )
    programs = (
        ShieldProgram(name="first", pair_ids=(0, 9), kind="test"),
        ShieldProgram(name="second", pair_ids=(17, 63), kind="test"),
    )
    host_estimator = _multi_isotope_estimator(
        use_gpu=False,
        gpu_device="cpu",
    )
    device_estimator = _multi_isotope_estimator(
        use_gpu=True,
        gpu_device=device_name,
    )
    host = _full_spectrum_joint_program_components(
        host_estimator,
        detector_positions,
        programs,
        host_estimator.planning_joint_particles(),
        live_time_s=2.0,
        detector_aperture_samples=1,
    )
    device = _full_spectrum_joint_program_components(
        device_estimator,
        detector_positions,
        programs,
        device_estimator.planning_joint_particles(),
        live_time_s=2.0,
        detector_aperture_samples=1,
        device_resident=True,
    )

    assert isinstance(device, _DeviceJointProgramSpectrumComponents)
    expected_shapes = {
        "total_pnvsl": (2, 3, 2, 4, 9),
        "uncollided_pnvsl": (2, 3, 2, 4, 9),
        "features_pnvslf": (2, 3, 2, 4, 9, 4),
        "live_times_v": (2,),
    }
    for field_name, expected_shape in expected_shapes.items():
        device_value = getattr(device, field_name)
        host_value = getattr(host, field_name)
        assert torch.is_tensor(device_value)
        assert device_value.device.type == device_name
        assert device_value.dtype == torch.float64
        assert tuple(device_value.shape) == expected_shape
        assert device_value.is_contiguous()
        if device_name == "cpu":
            np.testing.assert_array_equal(device_value.numpy(), host_value)
        else:
            np.testing.assert_allclose(
                device_value.detach().cpu().numpy(),
                host_value,
                rtol=1.0e-12,
                atol=1.0e-14,
            )

    expected_active_slots = torch.as_tensor(
        [
            [False, False, True, True],
            [True, False, False, False],
            [True, True, True, False],
        ],
        device=device.total_pnvsl.device,
        dtype=torch.bool,
    )
    inactive_slots = ~expected_active_slots
    total_by_particle_slot = device.total_pnvsl.permute(1, 3, 0, 2, 4)
    uncollided_by_particle_slot = device.uncollided_pnvsl.permute(1, 3, 0, 2, 4)
    features_by_particle_slot = device.features_pnvslf.permute(1, 3, 0, 2, 4, 5)
    assert int(torch.count_nonzero(total_by_particle_slot[inactive_slots]).item()) == 0
    assert (
        int(torch.count_nonzero(uncollided_by_particle_slot[inactive_slots]).item())
        == 0
    )
    assert (
        int(torch.count_nonzero(features_by_particle_slot[inactive_slots]).item())
        == 0
    )
    active_totals = total_by_particle_slot[expected_active_slots].reshape(
        int(torch.count_nonzero(expected_active_slots).item()),
        -1,
    )
    assert bool(torch.all(torch.any(active_totals > 0.0, dim=1)).item())
