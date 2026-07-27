"""Tests for PF reporting helper functions."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
from pf.reporting import measurement_vector
from pf.state import IsotopeState


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


def test_measurement_vector_broadcasts_scalar() -> None:
    """Scalar inputs should broadcast to the requested measurement count."""
    vec = measurement_vector(2.5, 3, "background", min_value=0.0)
    assert np.allclose(vec, [2.5, 2.5, 2.5])


def test_measurement_vector_rejects_wrong_length() -> None:
    """Vector inputs must match the requested measurement count."""
    with pytest.raises(ValueError, match="one value per measurement"):
        measurement_vector(np.asarray([1.0, 2.0]), 3, "z", allow_scalar=False)


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


def test_estimator_posterior_source_uncertainty_is_json_serializable() -> None:
    """Estimator diagnostics should expose conditional 3-D posterior summaries."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        candidate_sources=np.asarray([[1.0, 1.0, 0.0]], dtype=float),
        shield_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        mu_by_isotope={"Cs-137": 0.5},
        pf_config=RotatingShieldPFConfig(
            num_particles=3,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
        ),
    )
    estimator.add_measurement_pose(np.asarray([2.5, 2.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters["Cs-137"]
    patches = filt._structural_rj_surface_patches
    assert patches is not None
    patch_kinds = np.asarray(patches.kinds, dtype=object)
    floor_center = patches.centers_xyz[
        int(np.flatnonzero(patch_kinds == "floor")[0])
    ]
    ceiling_center = patches.centers_xyz[
        int(np.flatnonzero(patch_kinds == "ceiling")[0])
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
                positions=np.asarray(positions, dtype=float),
                strengths=np.asarray(strengths, dtype=float),
                background=0.1,
            ),
            log_weight=float(np.log(weight)),
        )
        for weight, positions, strengths in particle_specs
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
    """Exact bottom patches must remain bottom in posterior uncertainty reports."""
    isotope = "Cs-137"
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        transport_boxes_m=((1.2, 1.3, 0.4, 1.8, 1.9, 1.4),),
    )
    estimator = PurePFEstimator(
        isotopes=(isotope,),
        candidate_sources=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={isotope: 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=1,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(1, 1),
            use_gpu=False,
            position_max=(3.0, 3.0, 3.0),
            structural_rj_patch_spacing_m=0.5,
        ),
        obstacle_grid=obstacle_grid,
        measurement_log_sha256="c" * 64,
    )
    estimator.add_measurement_pose(np.asarray([0.5, 0.5, 0.5], dtype=float))
    estimator._ensure_kernel_cache()
    filt = estimator.filters[isotope]
    patches = filt._structural_rj_surface_patches
    assert patches is not None
    patch_kinds = np.asarray(patches.kinds, dtype=object)
    bottom_center = patches.centers_xyz[
        int(np.flatnonzero(patch_kinds == "obstacle_bottom")[0])
    ]
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(
                num_sources=1,
                positions=bottom_center[None, :],
                strengths=np.asarray([100.0], dtype=float),
                background=0.0,
            ),
            log_weight=0.0,
        )
    ]

    diagnostics = estimator.posterior_source_uncertainty(
        {
            isotope: (
                bottom_center[None, :],
                np.asarray([100.0], dtype=float),
            )
        },
        match_radius_m=0.2,
    )[isotope][0]

    assert diagnostics["surface_posterior_available"] is True
    assert diagnostics["surface_kind_posterior"]["obstacle_bottom"] == 1.0
    assert diagnostics["surface_kind_posterior"]["off_surface"] == 0.0
    assert sum(diagnostics["surface_kind_posterior"].values()) == pytest.approx(1.0)


def test_fixed_cardinality_pure_estimator_dispatches_state_mh_moves() -> None:
    """Fixed-K estimator updates must still dispatch exact within-K state MH."""
    evidence = object()
    dispatched: list[object] = []

    def measurement_data_for_iso(
        isotope: str,
        window: int | None,
    ) -> object:
        """Return the sentinel history passed to the per-isotope kernel."""
        assert isotope == "Cs-137"
        assert window is None
        return evidence

    def apply_structural_moves(payload: object) -> None:
        """Record the exact state-kernel dispatch."""
        dispatched.append(payload)

    estimator = object.__new__(PurePFEstimator)
    estimator.pf_config = RotatingShieldPFConfig(
        num_particles=3,
        max_sources=1,
        variable_cardinality=False,
        init_num_sources=(1, 1),
        parallel_isotope_updates=False,
        use_gpu=False,
    )
    estimator.filters = {
        "Cs-137": SimpleNamespace(
            apply_structural_moves=apply_structural_moves,
        )
    }
    estimator.isotopes = ("Cs-137",)
    estimator._measurement_data_for_iso = measurement_data_for_iso
    estimator.last_structural_update_workers = 0
    estimator.last_structural_update_wall_s = 0.0

    estimator._apply_structural_moves()

    assert dispatched == [evidence]
    assert estimator.last_structural_update_workers == 1
    manifest = estimator.structural_model_manifest()
    assert manifest["pure_pf_schema_version"] == 1
    assert manifest["support_domain"] == "environment_surface"
    assert manifest["structural_kernel_family"] == (
        "fixed_cardinality_surface_position_strength_mh"
    )
    assert manifest["within_cardinality_kernel_exact_mh"] is True
    assert manifest["structural_kernel_exact_rj"] is False
    assert manifest["strength_prior"] == {
        "kind": "bounded_uniform",
        "minimum_cps_1m": 0.0,
        "maximum_cps_1m": 2_000_000.0,
        "units": "detector_cps_1m",
        "unit_definition": "expected_net_detector_count_rate_at_1m",
        "used_for_initialization": True,
        "shared_by_initialization_and_state_moves": True,
    }
    assert manifest["rj_move_kernel"]["enabled"] is True
    assert manifest["rj_move_kernel"]["birth_death_enabled"] is False
    assert manifest["rj_move_kernel"]["within_cardinality_exact_mh"] is True
