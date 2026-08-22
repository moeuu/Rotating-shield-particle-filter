"""Shared physical fixtures for DSS planning tests."""

from __future__ import annotations

import numpy as np

from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.state import IsotopeState
from pure_pf_test_support import approved_full_spectrum_model


def state_on_filter(
    particle_filter: object,
    positions_xyz: np.ndarray,
    strengths: np.ndarray,
) -> IsotopeState:
    """Build a continuous-surface state from physical test positions."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    strength_values = np.asarray(strengths, dtype=float).reshape(-1)
    chart_ids, surface_uv = particle_filter.structural_surface_chart_coordinates(  # type: ignore[attr-defined]
        positions
    )
    return IsotopeState(
        num_sources=int(strength_values.size),
        strengths=strength_values,
        surface_chart_ids=chart_ids,
        surface_uv=surface_uv,
    )


def build_full_spectrum_planning_estimator(
    *,
    shield_normals: np.ndarray | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> RotatingShieldPFEstimator:
    """Build a production-approved tiny estimator for DSS spectrum tests."""
    isotope = "Cs-137"
    model = approved_full_spectrum_model()
    line_mu = tuple(
        {
            "energy_keV": float(line["energy_keV"]),
            "weight": float(line["branching_weight"]),
            "fe": float(line["mu_fe_cm_inv"]),
            "pb": float(line["mu_pb_cm_inv"]),
        }
        for line in model.line_identity
        if str(line["isotope"]) == isotope
    )
    estimator = RotatingShieldPFEstimator(
        isotopes=(isotope,),
        surface_diagnostic_points=np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        shield_normals=(
            np.asarray([[0.0, 0.0, 1.0]], dtype=float)
            if shield_normals is None
            else np.asarray(shield_normals, dtype=float)
        ),
        mu_by_isotope={isotope: {"fe": 0.0, "pb": 0.0}},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            planning_eig_samples=4,
        ),
        shield_params=ShieldParams(mu_fe=0.0, mu_pb=0.0),
        line_mu_by_isotope={isotope: line_mu},
        full_spectrum_generative_model=model,
        random_seed=9,
    )
    estimator.add_measurement_pose(np.array([1.0, 0.0, 0.0], dtype=float))
    estimator._ensure_kernel_cache()
    particles = estimator.filters[isotope].continuous_particles
    particle_filter = estimator.filters[isotope]
    for index, particle in enumerate(particles):
        particle.state = state_on_filter(
            particle_filter,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([float(10 + 10 * index)]),
        )
        particle.log_weight = float(np.log(0.5))
    return estimator
