"""Build deterministic reports directly from particle-filter posterior mass."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PFSourceMode:
    """Summarize one deterministically aligned source slot in a PF stratum."""

    label_index: int
    position_mean_xyz: tuple[float, float, float]
    position_covariance_xyz: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    credible_radius_95_m: float
    strength_mean_cps_1m: float
    strength_median_cps_1m: float
    strength_credible_interval_95_cps_1m: tuple[float, float]
    posterior_mass: float
    conditional_mass: float = 1.0
    belief_source: str = "pf_posterior"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mode payload."""
        return {
            "label_index": int(self.label_index),
            "position_mean_xyz": [float(value) for value in self.position_mean_xyz],
            "position_covariance_xyz": [
                [float(value) for value in row] for row in self.position_covariance_xyz
            ],
            "credible_radius_95_m": float(self.credible_radius_95_m),
            "credible_radius_m": float(self.credible_radius_95_m),
            "strength_mean_cps_1m": float(self.strength_mean_cps_1m),
            "strength_median_cps_1m": float(self.strength_median_cps_1m),
            "strength_credible_interval_95_cps_1m": [
                float(value) for value in self.strength_credible_interval_95_cps_1m
            ],
            "strength_credible_interval_cps_1m": [
                float(value) for value in self.strength_credible_interval_95_cps_1m
            ],
            "posterior_mass": float(self.posterior_mass),
            "conditional_mass": float(self.conditional_mass),
            "belief_source": str(self.belief_source),
        }


@dataclass(frozen=True)
class PFPointEstimate:
    """Store one isotope's PF-posterior-only point estimate and uncertainty."""

    map_cardinality: int
    cardinality_distribution: Mapping[int, float]
    modes: tuple[PFSourceMode, ...]
    background_rate_mean_cps: float
    background_rate_credible_interval_95_cps: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe isotope estimate payload."""
        return {
            "map_cardinality": int(self.map_cardinality),
            "cardinality_distribution": {
                str(int(key)): float(value)
                for key, value in sorted(self.cardinality_distribution.items())
            },
            "modes": [mode.to_dict() for mode in self.modes],
            "background_rate_mean_cps": float(self.background_rate_mean_cps),
            "background_rate_credible_interval_95_cps": [
                float(value) for value in self.background_rate_credible_interval_95_cps
            ],
        }


@dataclass(frozen=True)
class PFPosteriorSnapshot:
    """Store a complete PF posterior report with purity provenance."""

    estimator_variant: str
    isotopes: Mapping[str, PFPointEstimate]
    planner_belief_sources: tuple[str, ...]
    repository_commit: str
    measurement_log_schema_version: int
    config_hash: str
    resolved_config_hash: str
    measurement_log_sha256: str
    random_seed: int
    profile_capability_map: Mapping[str, bool]
    record_count: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return the required JSON-safe PF result contract."""
        provenance = {
            "estimator_repository": "moeuu/Rotating-shield-particle-filter",
            "estimator_commit": str(self.repository_commit),
            "measurement_log_schema_version": int(self.measurement_log_schema_version),
            "measurement_log_sha256": str(self.measurement_log_sha256),
            "resolved_config_sha256": str(self.resolved_config_hash),
            "config_sha256": str(self.config_hash),
            "random_seed": int(self.random_seed),
            "planner_belief_sources": list(self.planner_belief_sources),
            "batch_feedback_applied": False,
        }
        return {
            "schema_version": int(self.schema_version),
            "estimator_family": "particle_filter",
            "estimator_variant": str(self.estimator_variant),
            "estimator_profile": str(self.estimator_variant),
            "final_estimate_source": "pf_posterior",
            "uses_all_history_batch_fit": False,
            "uses_surface_map": False,
            "uses_batch_model_order": False,
            "batch_feedback_to_particles": False,
            "batch_feedback_applied": False,
            "batch_methods_invoked": [],
            "planner_belief_sources": list(self.planner_belief_sources),
            "repository_commit": str(self.repository_commit),
            "measurement_log_schema_version": int(self.measurement_log_schema_version),
            "resolved_config_hash": str(self.resolved_config_hash),
            "resolved_config_sha256": str(self.resolved_config_hash),
            "config_sha256": str(self.config_hash),
            "measurement_log_sha256": str(self.measurement_log_sha256),
            "random_seed": int(self.random_seed),
            "provenance": provenance,
            "profile_capability_map": {
                str(key): bool(value)
                for key, value in sorted(self.profile_capability_map.items())
            },
            "record_count": int(self.record_count),
            "isotopes": {
                str(isotope): estimate.to_dict()
                for isotope, estimate in sorted(self.isotopes.items())
            },
        }


def _normalized_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return finite normalized weights with a deterministic uniform fallback."""
    result = np.asarray(weights, dtype=float).reshape(-1)
    result = np.where(np.isfinite(result) & (result >= 0.0), result, 0.0)
    total = float(np.sum(result))
    if result.size == 0:
        return result
    if total <= 0.0:
        return np.full(result.size, 1.0 / float(result.size), dtype=float)
    return result / total


def weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a deterministic weighted quantile for one scalar posterior field."""
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = _normalized_weights(weights)
    if value_array.size == 0:
        return 0.0
    if value_array.size != weight_array.size:
        raise ValueError("values and weights must have equal lengths.")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    index = int(
        np.searchsorted(cumulative, np.clip(float(quantile), 0.0, 1.0), side="left")
    )
    return float(ordered_values[min(index, ordered_values.size - 1)])


def cardinality_distribution_from_states(
    states: Sequence[Any],
    weights: NDArray[np.float64],
    *,
    max_cardinality: int | None = None,
) -> dict[int, float]:
    """Accumulate particle weight by source count using one vectorized reduction."""
    normalized = _normalized_weights(weights)
    if len(states) != normalized.size:
        raise ValueError("states and weights must have equal lengths.")
    cardinalities = np.fromiter(
        (max(0, int(state.num_sources)) for state in states),
        dtype=np.int64,
        count=len(states),
    )
    observed_max = int(np.max(cardinalities)) if cardinalities.size else 0
    output_max = (
        observed_max
        if max_cardinality is None
        else max(
            observed_max,
            int(max_cardinality),
        )
    )
    masses = np.bincount(
        cardinalities,
        weights=normalized,
        minlength=output_max + 1,
    )
    return {index: float(value) for index, value in enumerate(masses)}


def align_spatial_modes_batched(
    positions: NDArray[np.float64],
    strengths: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    max_iterations: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align source labels with deterministic weighted spatial clustering.

    A deterministic weighted farthest-point initialization is refined by
    batched one-to-one assignments and weighted center updates. Assignments are
    evaluated over the small configured PF source-slot cap simultaneously for
    every particle; there is no scalar particle loop. This prevents label
    permutations and crossed spatial modes from being averaged together.
    """
    position_array = np.asarray(positions, dtype=float)
    strength_array = np.asarray(strengths, dtype=float)
    normalized = _normalized_weights(weights)
    if position_array.ndim != 3 or position_array.shape[2] != 3:
        raise ValueError("positions must have shape (particles, sources, 3).")
    particle_count, source_count, _ = position_array.shape
    if strength_array.shape != (particle_count, source_count):
        raise ValueError("strengths must have shape (particles, sources).")
    if normalized.size != particle_count:
        raise ValueError("weights must have one value per particle.")
    if source_count <= 1:
        return position_array.copy(), strength_array.copy()
    if source_count > 8:
        raise ValueError(
            "Batched exact source alignment supports at most eight source slots."
        )

    flat_positions = position_array.reshape(-1, 3)
    flat_weights = np.repeat(normalized / float(source_count), source_count)
    lexicographic = np.lexsort(
        (flat_positions[:, 2], flat_positions[:, 1], flat_positions[:, 0])
    )
    ordered_positions = flat_positions[lexicographic]
    ordered_weights = flat_weights[lexicographic]
    centers = np.empty((source_count, 3), dtype=float)
    first_index = int(np.argmax(ordered_weights))
    centers[0] = ordered_positions[first_index]
    min_distance_sq = np.sum(
        (ordered_positions - centers[0][None, :]) ** 2,
        axis=1,
    )
    for center_index in range(1, source_count):
        score = ordered_weights * min_distance_sq
        next_index = int(np.argmax(score))
        if float(score[next_index]) <= 0.0:
            next_index = min(center_index, ordered_positions.shape[0] - 1)
        centers[center_index] = ordered_positions[next_index]
        distance_sq = np.sum(
            (ordered_positions - centers[center_index][None, :]) ** 2,
            axis=1,
        )
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)

    permutations = np.asarray(
        tuple(itertools.permutations(range(source_count))),
        dtype=np.int64,
    )
    center_indices = np.arange(source_count, dtype=np.int64)

    def _assignment(current_centers: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return the minimum-cost source ordering for every particle."""
        cost = np.sum(
            (position_array[:, :, None, :] - current_centers[None, None, :, :]) ** 2,
            axis=3,
        )
        permutation_cost = np.sum(
            cost[:, permutations, center_indices],
            axis=2,
        )
        return permutations[np.argmin(permutation_cost, axis=1)]

    for _ in range(max(1, int(max_iterations))):
        source_order = _assignment(centers)
        aligned_positions = np.take_along_axis(
            position_array,
            source_order[:, :, None],
            axis=1,
        )
        updated = np.einsum(
            "n,nkd->kd",
            normalized,
            aligned_positions,
            optimize=True,
        )
        if np.allclose(updated, centers, rtol=0.0, atol=1.0e-12):
            centers = updated
            break
        centers = updated

    center_order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    centers = centers[center_order]
    final_order = _assignment(centers)
    return (
        np.take_along_axis(position_array, final_order[:, :, None], axis=1),
        np.take_along_axis(strength_array, final_order, axis=1),
    )


def posterior_point_estimate_from_states(
    states: Sequence[Any],
    weights: NDArray[np.float64],
    *,
    max_cardinality: int | None = None,
) -> PFPointEstimate:
    """Aggregate a PF-only estimate from a deterministic MAP-cardinality stratum.

    State extraction is necessarily linear in the particle container. All
    numerical aggregation, source ordering, covariance, and cardinality mass
    calculations use batched NumPy arrays; the only scalar loop is over the
    configured source-slot cap, which is tiny in the full runtime.
    """
    normalized = _normalized_weights(weights)
    if len(states) != normalized.size:
        raise ValueError("states and weights must have equal lengths.")
    distribution = cardinality_distribution_from_states(
        states,
        normalized,
        max_cardinality=max_cardinality,
    )
    if not states:
        return PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution=distribution,
            modes=(),
            background_rate_mean_cps=0.0,
            background_rate_credible_interval_95_cps=(0.0, 0.0),
        )
    max_mass = max(distribution.values(), default=0.0)
    map_cardinality = min(
        cardinality
        for cardinality, mass in distribution.items()
        if np.isclose(mass, max_mass, rtol=0.0, atol=1.0e-15)
    )
    selected_indices = np.fromiter(
        (
            index
            for index, state in enumerate(states)
            if int(state.num_sources) == map_cardinality
        ),
        dtype=np.int64,
    )
    if selected_indices.size == 0:
        raise RuntimeError("MAP cardinality stratum has no particles.")
    selected_weights = _normalized_weights(normalized[selected_indices])
    selected_states = [states[int(index)] for index in selected_indices]
    backgrounds = np.asarray(
        [float(state.background) for state in selected_states],
        dtype=float,
    )
    background_mean = float(np.sum(selected_weights * backgrounds))
    background_interval = (
        weighted_quantile(backgrounds, selected_weights, 0.025),
        weighted_quantile(backgrounds, selected_weights, 0.975),
    )
    if map_cardinality == 0:
        return PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution=distribution,
            modes=(),
            background_rate_mean_cps=background_mean,
            background_rate_credible_interval_95_cps=background_interval,
        )

    position_rows = np.stack(
        [
            np.asarray(state.positions[:map_cardinality], dtype=float)
            for state in selected_states
        ],
        axis=0,
    )
    strength_rows = np.stack(
        [
            np.asarray(state.strengths[:map_cardinality], dtype=float)
            for state in selected_states
        ],
        axis=0,
    )
    if position_rows.shape != (selected_indices.size, map_cardinality, 3):
        raise ValueError("particle positions do not match their cardinality.")
    if strength_rows.shape != (selected_indices.size, map_cardinality):
        raise ValueError("particle strengths do not match their cardinality.")

    aligned_positions, aligned_strengths = align_spatial_modes_batched(
        position_rows,
        strength_rows,
        selected_weights,
    )
    position_mean = np.einsum(
        "n,nkd->kd",
        selected_weights,
        aligned_positions,
        optimize=True,
    )
    position_delta = aligned_positions - position_mean[None, :, :]
    position_covariance = np.einsum(
        "n,nki,nkj->kij",
        selected_weights,
        position_delta,
        position_delta,
        optimize=True,
    )
    strength_mean = np.einsum(
        "n,nk->k",
        selected_weights,
        aligned_strengths,
        optimize=True,
    )
    radial_distance = np.linalg.norm(position_delta, axis=2)
    stratum_mass = float(distribution[map_cardinality])
    modes: list[PFSourceMode] = []
    for source_index in range(map_cardinality):
        covariance_tuple = tuple(
            tuple(float(value) for value in row)
            for row in position_covariance[source_index]
        )
        modes.append(
            PFSourceMode(
                label_index=int(source_index),
                position_mean_xyz=tuple(
                    float(value) for value in position_mean[source_index]
                ),
                position_covariance_xyz=covariance_tuple,
                credible_radius_95_m=weighted_quantile(
                    radial_distance[:, source_index],
                    selected_weights,
                    0.95,
                ),
                strength_mean_cps_1m=float(strength_mean[source_index]),
                strength_median_cps_1m=weighted_quantile(
                    aligned_strengths[:, source_index],
                    selected_weights,
                    0.5,
                ),
                strength_credible_interval_95_cps_1m=(
                    weighted_quantile(
                        aligned_strengths[:, source_index],
                        selected_weights,
                        0.025,
                    ),
                    weighted_quantile(
                        aligned_strengths[:, source_index],
                        selected_weights,
                        0.975,
                    ),
                ),
                posterior_mass=stratum_mass,
            )
        )
    return PFPointEstimate(
        map_cardinality=int(map_cardinality),
        cardinality_distribution=distribution,
        modes=tuple(modes),
        background_rate_mean_cps=background_mean,
        background_rate_credible_interval_95_cps=background_interval,
    )
