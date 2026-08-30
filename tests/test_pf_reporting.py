"""Tests for PF reporting helper functions."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import chi2

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kind
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.particle_filter import IsotopeParticle
from pf.posterior_uncertainty import (
    SURFACE_KINDS,
    posterior_mode_uncertainty_batched,
)
from pf.pure_estimator import PurePFEstimator
from pf.state import IsotopeState
from pure_pf_test_support import (
    approved_full_spectrum_model,
    runtime_observation_model,
)


def _canonical_scalar_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    """Choose the same deterministic eigenvector signs as the batched path."""
    vectors = np.asarray(eigenvectors, dtype=float).copy()
    for column in range(vectors.shape[1]):
        dominant = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[dominant, column] < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def _scalar_mode_oracle(
    positions: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    modes: np.ndarray,
    *,
    environment: EnvironmentConfig,
    match_radius_m: float,
) -> list[dict[str, object]]:
    """Return a simple loop-based oracle for posterior mode diagnostics."""
    normalized = np.maximum(np.asarray(weights, dtype=float), 0.0)
    normalized /= np.sum(normalized)
    selected: list[list[np.ndarray | None]] = [
        [None for _ in range(modes.shape[0])] for _ in range(positions.shape[0])
    ]
    selected_distances = np.full(
        (positions.shape[0], modes.shape[0]),
        np.inf,
        dtype=float,
    )
    for particle_index in range(positions.shape[0]):
        for slot_index in range(positions.shape[1]):
            if not mask[particle_index, slot_index]:
                continue
            point = positions[particle_index, slot_index]
            distances = np.linalg.norm(modes - point[None, :], axis=1)
            mode_index = int(np.argmin(distances))
            distance = float(distances[mode_index])
            if (
                distance <= match_radius_m
                and distance < selected_distances[particle_index, mode_index]
            ):
                selected[particle_index][mode_index] = point
                selected_distances[particle_index, mode_index] = distance

    output: list[dict[str, object]] = []
    for mode_index in range(modes.shape[0]):
        matched_indices = [
            particle_index
            for particle_index in range(positions.shape[0])
            if selected[particle_index][mode_index] is not None
        ]
        existence = float(np.sum(normalized[matched_indices]))
        conditional = normalized[matched_indices] / existence
        samples = np.asarray(
            [selected[index][mode_index] for index in matched_indices],
            dtype=float,
        )
        mean = np.sum(conditional[:, None] * samples, axis=0)
        centered = samples - mean
        covariance = centered.T @ (centered * conditional[:, None])
        order = np.argsort(samples[:, 2], kind="stable")
        sorted_z = samples[order, 2]
        cumulative = np.cumsum(conditional[order])
        quantile_values = [
            float(sorted_z[np.searchsorted(cumulative, quantile, side="left")])
            for quantile in (0.05, 0.50, 0.95)
        ]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        descending = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[descending]
        eigenvectors = _canonical_scalar_eigenvectors(eigenvectors[:, descending])
        surface_probability = {kind: 0.0 for kind in SURFACE_KINDS}
        for sample, sample_weight in zip(samples, conditional):
            kind = source_surface_kind(sample, environment)
            key = "off_surface" if kind is None else kind
            surface_probability[key] += float(sample_weight)
        output.append(
            {
                "existence_mass": existence,
                "matched_particle_count": len(matched_indices),
                "mean": mean,
                "covariance": covariance,
                "z_quantiles": np.asarray(quantile_values, dtype=float),
                "semi_axes": np.sqrt(
                    np.maximum(eigenvalues, 0.0) * chi2.ppf(0.9, df=3)
                ),
                "orientation": eigenvectors,
                "surface_probability": surface_probability,
            }
        )
    return output


def test_posterior_mode_uncertainty_batch_matches_scalar_oracle() -> None:
    """Batched mode matching and 3-D statistics should match a scalar oracle."""
    environment = EnvironmentConfig(size_x=5.0, size_y=5.0, size_z=5.0)
    modes = np.asarray([[1.0, 1.0, 0.1], [4.0, 4.0, 4.9]], dtype=float)
    positions = np.asarray(
        [
            [[0.90, 1.00, 0.15], [4.10, 4.00, 5.0], [1.05, 1.00, 0.05]],
            [[1.10, 0.80, 0.10], [3.90, 4.20, 4.9], [0.00, 0.00, 0.0]],
            [[0.80, 1.20, 0.00], [4.20, 3.90, 4.8], [0.00, 0.00, 0.0]],
            [[1.20, 1.10, 0.30], [8.00, 8.00, 8.0], [0.00, 0.00, 0.0]],
            [[3.80, 4.10, 4.7], [0.00, 0.00, 0.0], [0.00, 0.00, 0.0]],
        ],
        dtype=float,
    )
    mask = np.asarray(
        [
            [True, True, True],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, False, False],
        ],
        dtype=bool,
    )
    weights = np.asarray([0.10, 0.20, 0.30, 0.15, 0.25], dtype=float)
    radius = 0.5

    batched = posterior_mode_uncertainty_batched(
        positions,
        mask,
        weights,
        modes,
        environment=environment,
        match_radius_m=radius,
    )
    scalar = _scalar_mode_oracle(
        positions,
        mask,
        weights,
        modes,
        environment=environment,
        match_radius_m=radius,
    )

    for actual, expected in zip(batched, scalar):
        assert actual["posterior_support_available"] is True
        assert actual["location_posterior_available"] is True
        assert actual["surface_posterior_available"] is True
        assert actual["surface_posterior_normalized"] is True
        assert actual["existence_mass"] == pytest.approx(expected["existence_mass"])
        assert actual["matched_particle_count"] == expected["matched_particle_count"]
        assert np.allclose(actual["weighted_mean_xyz_m"], expected["mean"])
        assert np.allclose(
            actual["weighted_covariance_xyz_m2"],
            expected["covariance"],
        )
        assert np.allclose(
            list(actual["z_quantiles_m"].values()),
            expected["z_quantiles"],
        )
        assert np.allclose(
            actual["ellipsoid_90"]["semi_axis_lengths_m"],
            expected["semi_axes"],
        )
        assert np.allclose(
            actual["ellipsoid_90"]["orientation_matrix_xyz_by_axis"],
            expected["orientation"],
        )
        assert actual["ellipsoid_90"]["available"] is True
        assert (
            actual["ellipsoid_90"]["interpretation"]
            == "gaussian_equivalent_covariance_ellipsoid"
        )
        assert actual["ellipsoid_90"]["is_empirical_credible_region"] is False
        assert actual["ellipsoid_90"]["applicability_requirements"] == [
            "approximately_unimodal_conditional_position_posterior",
            "approximately_gaussian_conditional_position_posterior",
        ]
        assert actual["surface_kind_posterior"] == pytest.approx(
            expected["surface_probability"]
        )
        assert sum(actual["surface_kind_posterior"].values()) == pytest.approx(1.0)


def test_posterior_mode_uncertainty_marks_unsupported_payloads() -> None:
    """Modes without matched particle mass must expose unavailable diagnostics."""
    diagnostic = posterior_mode_uncertainty_batched(
        np.asarray([[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]], dtype=float),
        np.ones((2, 1), dtype=bool),
        np.asarray([0.4, 0.6], dtype=float),
        np.asarray([[4.0, 4.0, 4.0]], dtype=float),
        environment=EnvironmentConfig(size_x=5.0, size_y=5.0, size_z=5.0),
        match_radius_m=0.2,
    )[0]

    json.dumps(diagnostic, allow_nan=False)
    assert diagnostic["posterior_support_available"] is False
    assert diagnostic["location_posterior_available"] is False
    assert diagnostic["surface_posterior_available"] is False
    assert diagnostic["surface_posterior_normalized"] is False
    assert diagnostic["existence_mass"] == 0.0
    assert sum(diagnostic["surface_kind_posterior"].values()) == 0.0
    ellipsoid = diagnostic["ellipsoid_90"]
    assert ellipsoid["available"] is False
    assert ellipsoid["semi_axis_lengths_m"] is None
    assert ellipsoid["orientation_matrix_xyz_by_axis"] is None
    assert ellipsoid["interpretation"] == ("gaussian_equivalent_covariance_ellipsoid")
    assert ellipsoid["is_empirical_credible_region"] is False


def test_posterior_existence_mass_clips_normalization_roundoff() -> None:
    """A fully supported mode must report an exact unit-interval probability."""
    particle_count = 2000
    diagnostic = posterior_mode_uncertainty_batched(
        np.zeros((particle_count, 1, 3), dtype=float),
        np.ones((particle_count, 1), dtype=bool),
        np.full(particle_count, 1.0 / particle_count, dtype=float),
        np.zeros((1, 3), dtype=float),
        environment=EnvironmentConfig(size_x=5.0, size_y=5.0, size_z=5.0),
        match_radius_m=0.0,
    )[0]

    assert diagnostic["existence_mass"] == 1.0


def test_estimator_posterior_source_uncertainty_is_json_serializable() -> None:
    """Estimator diagnostics should expose conditional 3-D posterior summaries."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[1.0, 1.0, 0.0]], dtype=float),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
        ),
        detector_aperture_radius_m=0.0395,
        detector_aperture_samples=33,
        full_spectrum_generative_model=approved_full_spectrum_model(("Cs-137",)),
    )
    estimator.add_measurement_pose(np.asarray([2.5, 2.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    chart_geometry = filt._structural_rj_surface_atlas.geometry
    assert chart_geometry is not None
    chart_kinds = np.asarray(chart_geometry.kinds, dtype=object)
    floor_center = chart_geometry.centers_xyz[
        int(np.flatnonzero(chart_kinds == "floor")[0])
    ]
    ceiling_center = chart_geometry.centers_xyz[
        int(np.flatnonzero(chart_kinds == "ceiling")[0])
    ]
    particle_specs = (
        (
            0.2,
            [floor_center, ceiling_center],
            [12.0, 5.0],
        ),
        (0.3, [floor_center], [11.0]),
        (0.5, [ceiling_center], [6.0]),
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=len(strengths),
                strengths=np.asarray(strengths, dtype=float),
                surface_chart_ids=(
                    filt.structural_surface_chart_coordinates(
                        np.asarray(positions, dtype=float)
                    )[0]
                ),
                surface_uv=(
                    filt.structural_surface_chart_coordinates(
                        np.asarray(positions, dtype=float)
                    )[1]
                ),
            ),
            log_weight=float(np.log(weight)),
            joint_row_identity=filt.continuous_particles[
                row
            ].joint_row_identity,
        )
        for row, (weight, positions, strengths) in enumerate(particle_specs)
    ]
    reported = {
        "Cs-137": (
            np.asarray([floor_center, ceiling_center], dtype=float),
            np.asarray([11.5, 5.5], dtype=float),
        )
    }

    diagnostics = estimator.posterior_source_uncertainty(
        reported,
        match_radius_m=0.5,
    )

    json.dumps(diagnostics, allow_nan=False)
    assert len(diagnostics["Cs-137"]) == 2
    for mode in diagnostics["Cs-137"]:
        assert mode["reported_strength_cps_1m"] > 0.0
        assert mode["posterior_support_available"] is True
        assert mode["location_posterior_available"] is True
        assert mode["surface_posterior_available"] is True
        assert mode["surface_posterior_normalized"] is True
        assert mode["existence_mass"] > 0.0
        assert sum(mode["surface_kind_posterior"].values()) == pytest.approx(1.0)
        assert list(mode["z_quantiles_m"].values()) == sorted(
            mode["z_quantiles_m"].values()
        )
        orientation = np.asarray(
            mode["ellipsoid_90"]["orientation_matrix_xyz_by_axis"],
            dtype=float,
        )
        assert mode["ellipsoid_90"]["available"] is True
        assert mode["ellipsoid_90"]["nominal_gaussian_probability_mass"] == 0.9
        assert np.allclose(orientation.T @ orientation, np.eye(3))
    assert diagnostics["Cs-137"][0]["surface_kind_posterior"]["floor"] == 1.0
    assert diagnostics["Cs-137"][1]["surface_kind_posterior"]["ceiling"] == 1.0


def test_estimator_uncertainty_reports_exact_obstacle_bottom_kind() -> None:
    """Exact bottom charts must remain bottom in posterior uncertainty reports."""
    isotope = "Cs-137"
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
        transport_mu_by_isotope={"Cs-137": (0.1,)},
        transport_line_mu_by_isotope={"Cs-137": ((0.1,),)},
        transport_line_compton_mu_by_isotope={"Cs-137": ((0.05,),)},
    )
    estimator = PurePFEstimator(
        isotopes=(isotope,),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        observation_model=runtime_observation_model((isotope,)),
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_surface_chart_max_edge_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
        full_spectrum_generative_model=approved_full_spectrum_model((isotope,)),
        measurement_log_schema_version=2,
        config_hash="a" * 64,
        resolved_config_hash="b" * 64,
        measurement_log_sha256="c" * 64,
        random_seed=0,
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    atlas = filt._structural_rj_surface_atlas
    assert atlas is not None
    chart_kinds = np.asarray(atlas.geometry.kinds, dtype=object)
    bottom_chart_id = int(
        np.flatnonzero(chart_kinds == "obstacle_bottom")[0]
    )
    bottom_uv = np.asarray([[0.5, 0.5]], dtype=float)
    bottom_center = atlas.positions_xyz(
        np.asarray([bottom_chart_id], dtype=np.int64),
        bottom_uv,
    )[0]
    original_identities = [
        particle.joint_row_identity for particle in filt.continuous_particles
    ]
    bottom_state = IsotopeState(
        num_sources=1,
        strengths=np.asarray([300_000.0], dtype=float),
        surface_chart_ids=np.asarray(
            [bottom_chart_id],
            dtype=np.int64,
        ),
        surface_uv=bottom_uv,
    )
    filt.continuous_particles = [
        IsotopeParticle(
            state=bottom_state.copy(),
            log_weight=-np.log(2.0),
            joint_row_identity=original_identities[index],
        )
        for index in range(2)
    ]

    diagnostics = estimator.posterior_source_uncertainty(
        {
            isotope: (
                bottom_center[None, :],
                np.asarray([300_000.0], dtype=float),
            )
        },
        match_radius_m=0.2,
    )[isotope][0]

    assert diagnostics["surface_posterior_available"] is True
    assert diagnostics["surface_kind_posterior"]["obstacle_bottom"] == 1.0
    assert diagnostics["surface_kind_posterior"]["off_surface"] == 0.0
    assert sum(diagnostics["surface_kind_posterior"].values()) == pytest.approx(1.0)
