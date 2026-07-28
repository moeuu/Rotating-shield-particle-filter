"""Batched posterior diagnostics for reported particle-filter source modes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import (
    SOURCE_SURFACE_REPORT_LABELS,
    source_surface_kinds,
)
from pf.posterior import validated_probability

SURFACE_KINDS = SOURCE_SURFACE_REPORT_LABELS
ELLIPSOID_INTERPRETATION = "gaussian_equivalent_covariance_ellipsoid"
ELLIPSOID_APPLICABILITY_REQUIREMENTS = (
    "approximately_unimodal_conditional_position_posterior",
    "approximately_gaussian_conditional_position_posterior",
)


def _ellipsoid_diagnostic(
    *,
    available: bool,
    semi_axis_lengths_m: list[float] | None,
    orientation_matrix_xyz_by_axis: list[list[float]] | None,
) -> dict[str, Any]:
    """Return metadata that prevents covariance ellipsoids being read as HPD sets."""
    return {
        "available": bool(available),
        "confidence": 0.9,
        "nominal_gaussian_probability_mass": 0.9,
        "interpretation": ELLIPSOID_INTERPRETATION,
        "is_empirical_credible_region": False,
        "applicability_requirements": list(ELLIPSOID_APPLICABILITY_REQUIREMENTS),
        "semi_axis_lengths_m": semi_axis_lengths_m,
        "orientation_matrix_xyz_by_axis": orientation_matrix_xyz_by_axis,
    }


def _normalized_particle_weights(
    weights: NDArray[np.float64],
    num_particles: int,
) -> NDArray[np.float64]:
    """Return normalized particle weights or fail on an invalid posterior."""
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size != int(num_particles):
        raise ValueError("particle_weights must have one value per particle.")
    if values.size == 0:
        return values
    if np.any(~np.isfinite(values)):
        raise ValueError("particle_weights must all be finite.")
    if np.any(values < 0.0):
        raise ValueError("particle_weights must all be nonnegative.")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("particle_weights must have a finite positive sum.")
    return values / total


def _canonicalize_eigenvectors(
    eigenvectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Choose deterministic signs for batched covariance eigenvectors."""
    vectors = np.asarray(eigenvectors, dtype=float).copy()
    if vectors.size == 0:
        return vectors
    dominant_rows = np.argmax(np.abs(vectors), axis=1)
    dominant_values = np.take_along_axis(
        vectors,
        dominant_rows[:, None, :],
        axis=1,
    ).squeeze(axis=1)
    signs = np.where(dominant_values < 0.0, -1.0, 1.0)
    return vectors * signs[:, None, :]


def _weighted_quantiles_by_mode(
    values: NDArray[np.float64],
    conditional_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return batched q05, q50, and q95 values for every reported mode."""
    if values.shape != conditional_weights.shape:
        raise ValueError("values and conditional_weights must have matching shapes.")
    num_particles, num_modes = values.shape
    if num_particles == 0:
        return np.zeros((num_modes, 3), dtype=float)
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(conditional_weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    positive = np.sum(sorted_weights, axis=0) > 0.0
    cumulative[-1, positive] = 1.0
    quantiles = np.asarray([0.05, 0.50, 0.95], dtype=float)
    reached = cumulative[:, :, None] >= quantiles[None, None, :]
    indices = np.argmax(reached, axis=0)
    columns = np.arange(num_modes, dtype=int)[:, None]
    return sorted_values[indices, columns]


def _empty_mode_diagnostic(
    mode_index: int,
    reported_position: NDArray[np.float64],
) -> dict[str, Any]:
    """Return a JSON-safe diagnostic for a mode with no posterior support."""
    return {
        "mode_index": int(mode_index),
        "reported_position_xyz_m": [float(value) for value in reported_position],
        "posterior_support_available": False,
        "location_posterior_available": False,
        "surface_posterior_available": False,
        "surface_posterior_normalized": False,
        "existence_mass": 0.0,
        "matched_particle_count": 0,
        "z_quantiles_m": {"q05": None, "q50": None, "q95": None},
        "weighted_mean_xyz_m": None,
        "weighted_covariance_xyz_m2": None,
        "ellipsoid_90": _ellipsoid_diagnostic(
            available=False,
            semi_axis_lengths_m=None,
            orientation_matrix_xyz_by_axis=None,
        ),
        "surface_kind_posterior": {kind: 0.0 for kind in SURFACE_KINDS},
    }


def posterior_mode_uncertainty_batched(
    packed_positions: NDArray[np.float64],
    packed_mask: NDArray[np.bool_],
    particle_weights: NDArray[np.float64],
    reported_positions: NDArray[np.float64],
    *,
    packed_surface_kinds: NDArray[np.object_] | None = None,
    packed_surface_chart_ids: NDArray[np.int64] | None = None,
    packed_surface_uv: NDArray[np.float64] | None = None,
    reported_surface_chart_ids: NDArray[np.int64] | None = None,
    reported_surface_uv: NDArray[np.float64] | None = None,
    surface_coordinate_path_distance: Callable[
        [
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
    ]
    | None = None,
    environment: EnvironmentConfig,
    obstacle_grid: ObstacleGrid | None = None,
    obstacle_height_m: float = 2.0,
    match_radius_m: float = 0.8,
    surface_tolerance_m: float = 1.0e-5,
    posterior_reference_mass: float = 1.0,
) -> list[dict[str, Any]]:
    """Return JSON-safe posterior 3-D diagnostics for reported source modes.

    Each valid particle source slot is assigned to its nearest reported mode if
    it lies within ``match_radius_m``.  Continuous PF reports supply chart/UV
    states and an intrinsic surface-path callback, so nearby opposite faces and
    disconnected components cannot match through free space.  If a particle has
    multiple slots assigned to one mode, only the closest slot contributes.
    ``conditional_support_mass`` is measured inside the selected reporting
    stratum. ``existence_mass`` multiplies it by ``posterior_reference_mass`` so
    a low-probability joint-cardinality stratum cannot be reported as certain.
    All remaining statistics are conditional on that mode existing.

    The ellipsoid axes are sorted from longest to shortest.  Columns of
    ``orientation_matrix_xyz_by_axis`` are their corresponding unit vectors in
    room x/y/z coordinates.  It is a Gaussian-equivalent covariance ellipsoid,
    not an empirical highest-posterior-density region; its payload states the
    assumptions required to interpret the nominal 0.9 Gaussian mass.  Availability
    flags distinguish unsupported modes from valid zero-valued summaries.
    When ``packed_surface_kinds`` is supplied, those authoritative labels are
    propagated with the selected slots instead of reclassifying coordinates.
    """
    positions = np.asarray(packed_positions, dtype=float)
    mask = np.asarray(packed_mask, dtype=bool)
    modes = np.asarray(reported_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("packed_positions must have shape (P, S, 3).")
    if mask.shape != positions.shape[:2]:
        raise ValueError("packed_mask must have shape (P, S).")
    supplied_surface_kinds: NDArray[np.object_] | None = None
    if packed_surface_kinds is not None:
        supplied_surface_kinds = np.asarray(
            packed_surface_kinds,
            dtype=object,
        )
        if supplied_surface_kinds.shape != mask.shape:
            raise ValueError("packed_surface_kinds must have shape (P, S).")
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("reported_positions must have shape (M, 3).")
    if np.any(~np.isfinite(modes)):
        raise ValueError("reported_positions must contain only finite values.")
    coordinate_inputs = (
        packed_surface_chart_ids,
        packed_surface_uv,
        reported_surface_chart_ids,
        reported_surface_uv,
        surface_coordinate_path_distance,
    )
    surface_coordinates_active = all(
        value is not None for value in coordinate_inputs
    )
    if any(value is not None for value in coordinate_inputs) and not (
        surface_coordinates_active
    ):
        raise ValueError(
            "Intrinsic uncertainty matching requires packed/reported chart IDs, "
            "UV, and the surface-distance callback together."
        )
    packed_chart_ids: NDArray[np.int64] | None = None
    packed_uv: NDArray[np.float64] | None = None
    reported_chart_ids: NDArray[np.int64] | None = None
    reported_uv: NDArray[np.float64] | None = None
    if surface_coordinates_active:
        packed_chart_ids = np.asarray(
            packed_surface_chart_ids,
            dtype=np.int64,
        )
        packed_uv = np.asarray(packed_surface_uv, dtype=np.float64)
        reported_chart_ids = np.asarray(
            reported_surface_chart_ids,
            dtype=np.int64,
        ).reshape(-1)
        reported_uv = np.asarray(reported_surface_uv, dtype=np.float64)
        if packed_chart_ids.shape != mask.shape:
            raise ValueError(
                "packed_surface_chart_ids must have shape (P, S)."
            )
        if packed_uv.shape != mask.shape + (2,):
            raise ValueError(
                "packed_surface_uv must have shape (P, S, 2)."
            )
        if reported_chart_ids.shape != (modes.shape[0],):
            raise ValueError(
                "reported_surface_chart_ids must have shape (M,)."
            )
        if reported_uv.shape != (modes.shape[0], 2):
            raise ValueError(
                "reported_surface_uv must have shape (M, 2)."
            )
    radius = float(match_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("match_radius_m must be finite and non-negative.")
    tolerance = float(surface_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("surface_tolerance_m must be finite and non-negative.")
    reference_mass = float(posterior_reference_mass)
    if (
        not np.isfinite(reference_mass)
        or reference_mass < 0.0
        or reference_mass > 1.0
    ):
        raise ValueError("posterior_reference_mass must lie in [0, 1].")

    num_particles, num_slots = positions.shape[:2]
    num_modes = modes.shape[0]
    if num_modes == 0:
        return []
    weights = _normalized_particle_weights(particle_weights, num_particles)
    empty = [
        _empty_mode_diagnostic(mode_index, modes[mode_index])
        for mode_index in range(num_modes)
    ]
    matching_distance = (
        "intrinsic_surface_path_upper_bound_m"
        if surface_coordinates_active
        else "euclidean_3d_m"
    )
    for diagnostic in empty:
        diagnostic["matching_distance"] = matching_distance
        diagnostic["match_radius_m"] = radius
        diagnostic["posterior_reference_mass"] = reference_mass
        diagnostic["conditional_support_mass"] = 0.0
    if num_particles == 0 or num_slots == 0:
        return empty

    valid_slots = mask & np.all(np.isfinite(positions), axis=2)
    differences = positions[:, :, None, :] - modes[None, None, :, :]
    if (
        surface_coordinates_active
        and packed_chart_ids is not None
        and packed_uv is not None
        and reported_chart_ids is not None
        and reported_uv is not None
        and surface_coordinate_path_distance is not None
    ):
        distances = np.asarray(
            surface_coordinate_path_distance(
                reported_chart_ids[None, None, :],
                reported_uv[None, None, :, :],
                packed_chart_ids[:, :, None],
                packed_uv[:, :, None, :],
            ),
            dtype=np.float64,
        )
        if distances.shape != (
            num_particles,
            num_slots,
            num_modes,
        ):
            raise ValueError(
                "surface_coordinate_path_distance returned an unexpected shape."
            )
        if np.any(np.isnan(distances)) or np.any(distances < 0.0):
            raise ValueError(
                "Intrinsic surface distances must be nonnegative or infinity."
            )
    else:
        distances = np.linalg.norm(differences, axis=3)
    distances = np.where(valid_slots[:, :, None], distances, np.inf)
    nearest_modes = np.argmin(distances, axis=2)
    mode_indices = np.arange(num_modes, dtype=int)
    assigned = (nearest_modes[:, :, None] == mode_indices[None, None, :]) & (
        distances <= radius
    )
    assigned_distances = np.where(assigned, distances, np.inf)
    selected_slots = np.argmin(assigned_distances, axis=1)
    matched = np.any(assigned, axis=1)
    particle_indices = np.arange(num_particles, dtype=int)[:, None]
    selected_positions = positions[particle_indices, selected_slots]
    selected_positions = np.where(
        matched[:, :, None],
        selected_positions,
        0.0,
    )
    selected_surface_kinds: NDArray[np.object_] | None = None
    if supplied_surface_kinds is not None:
        selected_surface_kinds = supplied_surface_kinds[
            particle_indices,
            selected_slots,
        ]
        selected_surface_kinds = np.where(
            matched,
            selected_surface_kinds,
            None,
        )

    raw_conditional_existence_mass = np.einsum(
        "p,pm->m",
        weights,
        matched,
        optimize=True,
    )
    conditional_existence_mass = np.fromiter(
        (
            validated_probability(
                value,
                name=f"conditional mode existence mass[{index}]",
            )
            for index, value in enumerate(raw_conditional_existence_mass)
        ),
        dtype=np.float64,
        count=raw_conditional_existence_mass.size,
    )
    existence_mass = reference_mass * conditional_existence_mass
    joint_weights = weights[:, None] * matched
    conditional_weights = np.divide(
        joint_weights,
        conditional_existence_mass[None, :],
        out=np.zeros_like(joint_weights),
        where=conditional_existence_mass[None, :] > 0.0,
    )
    means = np.einsum(
        "pm,pmi->mi",
        conditional_weights,
        selected_positions,
        optimize=True,
    )
    centered = selected_positions - means[None, :, :]
    covariances = np.einsum(
        "pm,pmi,pmj->mij",
        conditional_weights,
        centered,
        centered,
        optimize=True,
    )
    covariances = 0.5 * (covariances + np.swapaxes(covariances, axis1=1, axis2=2))
    z_quantiles = _weighted_quantiles_by_mode(
        selected_positions[:, :, 2],
        conditional_weights,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    descending = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, descending, axis=1)
    eigenvectors = np.take_along_axis(
        eigenvectors,
        descending[:, None, :],
        axis=2,
    )
    eigenvectors = _canonicalize_eigenvectors(eigenvectors)
    ellipsoid_scale = float(chi2.ppf(0.9, df=3))
    semi_axes = np.sqrt(np.maximum(eigenvalues, 0.0) * ellipsoid_scale)

    kinds = (
        source_surface_kinds(
            selected_positions.reshape(-1, 3),
            environment,
            obstacle_grid,
            obstacle_height_m=float(obstacle_height_m),
            tolerance_m=tolerance,
        ).reshape(num_particles, num_modes)
        if selected_surface_kinds is None
        else selected_surface_kinds
    )
    known_surface = np.stack(
        [kinds == kind for kind in SURFACE_KINDS[:-1]],
        axis=2,
    )
    surface_indicators = np.concatenate(
        [known_surface, ~np.any(known_surface, axis=2, keepdims=True)],
        axis=2,
    )
    surface_probabilities = np.einsum(
        "pm,pmk->mk",
        conditional_weights,
        surface_indicators,
        optimize=True,
    )
    surface_totals = np.sum(surface_probabilities, axis=1)
    supported_modes = conditional_existence_mass > 0.0
    invalid_supported_surface = supported_modes & (
        ~np.isfinite(surface_totals) | (surface_totals <= 0.0)
    )
    if np.any(invalid_supported_surface):
        raise RuntimeError(
            "A supported posterior mode did not produce a finite surface posterior."
        )
    surface_probabilities = np.divide(
        surface_probabilities,
        surface_totals[:, None],
        out=np.zeros_like(surface_probabilities),
        where=surface_totals[:, None] > 0.0,
    )
    normalized_surface_totals = np.sum(surface_probabilities, axis=1)
    if not np.allclose(
        normalized_surface_totals[supported_modes],
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Supported surface posteriors must sum to one.")

    diagnostics: list[dict[str, Any]] = []
    for mode_index in range(num_modes):
        if conditional_existence_mass[mode_index] <= 0.0:
            diagnostics.append(empty[mode_index])
            continue
        diagnostics.append(
            {
                "mode_index": int(mode_index),
                "matching_distance": matching_distance,
                "match_radius_m": radius,
                "posterior_reference_mass": reference_mass,
                "conditional_support_mass": float(
                    conditional_existence_mass[mode_index]
                ),
                "reported_position_xyz_m": [
                    float(value) for value in modes[mode_index]
                ],
                "posterior_support_available": True,
                "location_posterior_available": True,
                "surface_posterior_available": True,
                "surface_posterior_normalized": True,
                "existence_mass": float(existence_mass[mode_index]),
                "matched_particle_count": int(np.count_nonzero(matched[:, mode_index])),
                "z_quantiles_m": {
                    "q05": float(z_quantiles[mode_index, 0]),
                    "q50": float(z_quantiles[mode_index, 1]),
                    "q95": float(z_quantiles[mode_index, 2]),
                },
                "weighted_mean_xyz_m": [float(value) for value in means[mode_index]],
                "weighted_covariance_xyz_m2": [
                    [float(value) for value in row] for row in covariances[mode_index]
                ],
                "ellipsoid_90": _ellipsoid_diagnostic(
                    available=True,
                    semi_axis_lengths_m=[
                        float(value) for value in semi_axes[mode_index]
                    ],
                    orientation_matrix_xyz_by_axis=[
                        [float(value) for value in row]
                        for row in eigenvectors[mode_index]
                    ],
                ),
                "surface_kind_posterior": {
                    kind: float(surface_probabilities[mode_index, kind_index])
                    for kind_index, kind in enumerate(SURFACE_KINDS)
                },
            }
        )
    return diagnostics
