"""Build deterministic reports directly from particle-filter posterior mass."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


PROBABILITY_ROUNDOFF_ATOL = 1.0e-12
SurfaceCoordinateDistance = Callable[
    [
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.float64],
    ],
    NDArray[np.float64],
]


def validated_probability(value: object, *, name: str) -> float:
    """Return one probability, clipping only binary floating-point roundoff."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in [0, 1].")
    numeric = float(value)
    if (
        not np.isfinite(numeric)
        or numeric < -PROBABILITY_ROUNDOFF_ATOL
        or numeric > 1.0 + PROBABILITY_ROUNDOFF_ATOL
    ):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return float(np.clip(numeric, 0.0, 1.0))


def _validated_exact_boolean(value: object, *, name: str) -> bool:
    """Return a provenance boolean without accepting truthy substitutes."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def _validated_nonnegative_integer(value: object, *, name: str) -> int:
    """Return one exact nonnegative JSON integer."""
    if type(value) is not int:
        raise ValueError(f"{name} must be a nonnegative JSON integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative JSON integer.")
    return value


def _validated_nonempty_string(value: object, *, name: str) -> str:
    """Return one nonempty string without coercing another object type."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _validated_hex_digest(
    value: object,
    *,
    name: str,
    lengths: tuple[int, ...],
) -> str:
    """Return a lowercase hexadecimal provenance digest of an allowed length."""
    text = _validated_nonempty_string(value, name=name)
    normalized = text.strip().lower()
    if len(normalized) not in lengths or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise ValueError(
            f"{name} must be a {allowed}-character hexadecimal digest."
        )
    return normalized


def _validated_string_key_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    """Copy a mapping while rejecting silently stringified provenance keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    result: dict[str, Any] = {}
    for key, item in value.items():
        validated_key = _validated_nonempty_string(
            key,
            name=f"{name} key",
        )
        result[validated_key] = item
    return result


def validated_probability_distribution(
    values: Sequence[object] | NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a unit-mass probability vector or fail on material drift."""
    raw = np.asarray(values, dtype=object).reshape(-1)
    if raw.size == 0:
        raise ValueError(f"{name} must contain at least one probability.")
    probabilities = np.fromiter(
        (
            validated_probability(
                value,
                name=f"{name}[{index}]",
            )
            for index, value in enumerate(raw)
        ),
        dtype=np.float64,
        count=raw.size,
    )
    total = float(np.sum(probabilities, dtype=np.float64))
    if (
        not np.isfinite(total)
        or abs(total - 1.0) > PROBABILITY_ROUNDOFF_ATOL
    ):
        raise ValueError(
            f"{name} must sum to one within "
            f"{PROBABILITY_ROUNDOFF_ATOL:.0e}; got {total!r}."
        )
    return probabilities / total


@dataclass(frozen=True)
class PFSourceMode:
    """Summarize one deterministically aligned source slot in a PF stratum."""

    label_index: int
    position_medoid_xyz: tuple[float, float, float]
    position_covariance_xyz: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    credible_radius_95_m: float
    strength_representative_cps_1m: float
    strength_mean_cps_1m: float
    strength_median_cps_1m: float
    strength_credible_interval_95_cps_1m: tuple[float, float]
    posterior_mass: float
    conditional_mass: float = 1.0
    belief_source: str = "pf_posterior"
    credible_surface_path_radius_95_m: float | None = None
    surface_connected_mass: float = 1.0
    surface_chart_id: int | None = None
    surface_uv: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mode payload."""
        posterior_mass = validated_probability(
            self.posterior_mass,
            name="PFSourceMode.posterior_mass",
        )
        conditional_mass = validated_probability(
            self.conditional_mass,
            name="PFSourceMode.conditional_mass",
        )
        surface_connected_mass = validated_probability(
            self.surface_connected_mass,
            name="PFSourceMode.surface_connected_mass",
        )
        belief_source = _validated_nonempty_string(
            self.belief_source,
            name="PFSourceMode.belief_source",
        )
        return {
            "label_index": int(self.label_index),
            "position_medoid_xyz": [float(value) for value in self.position_medoid_xyz],
            "position_covariance_xyz": [
                [float(value) for value in row] for row in self.position_covariance_xyz
            ],
            "credible_radius_95_m": float(self.credible_radius_95_m),
            "credible_radius_m": float(self.credible_radius_95_m),
            "credible_surface_path_radius_95_m": (
                None
                if self.credible_surface_path_radius_95_m is None
                else float(self.credible_surface_path_radius_95_m)
            ),
            "surface_connected_mass": surface_connected_mass,
            "surface_chart_id": (
                None
                if self.surface_chart_id is None
                else int(self.surface_chart_id)
            ),
            "surface_uv": (
                None
                if self.surface_uv is None
                else [float(value) for value in self.surface_uv]
            ),
            "strength_representative_cps_1m": float(
                self.strength_representative_cps_1m
            ),
            "strength_mean_cps_1m": float(self.strength_mean_cps_1m),
            "strength_median_cps_1m": float(self.strength_median_cps_1m),
            "strength_credible_interval_95_cps_1m": [
                float(value) for value in self.strength_credible_interval_95_cps_1m
            ],
            "strength_credible_interval_cps_1m": [
                float(value) for value in self.strength_credible_interval_95_cps_1m
            ],
            "posterior_mass": posterior_mass,
            "conditional_mass": conditional_mass,
            "belief_source": belief_source,
        }


@dataclass(frozen=True)
class PFPointEstimate:
    """Store one isotope's PF-posterior-only point estimate and uncertainty."""

    map_cardinality: int
    cardinality_distribution: Mapping[int, float]
    selected_stratum_mass: float
    modes: tuple[PFSourceMode, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe isotope estimate payload."""
        if isinstance(self.map_cardinality, (bool, np.bool_)) or not isinstance(
            self.map_cardinality,
            (int, np.integer),
        ):
            raise ValueError("PFPointEstimate.map_cardinality must be an integer.")
        map_cardinality = int(self.map_cardinality)
        if map_cardinality < 0:
            raise ValueError(
                "PFPointEstimate.map_cardinality must be nonnegative."
            )
        if not isinstance(self.cardinality_distribution, Mapping) or not (
            self.cardinality_distribution
        ):
            raise ValueError(
                "PFPointEstimate.cardinality_distribution must be nonempty."
            )
        cardinality_entries: list[tuple[int, object]] = []
        for raw_cardinality, probability in (
            self.cardinality_distribution.items()
        ):
            if isinstance(raw_cardinality, (bool, np.bool_)) or not isinstance(
                raw_cardinality,
                (int, np.integer),
            ):
                raise ValueError(
                    "PFPointEstimate cardinality keys must be integers."
                )
            cardinality = int(raw_cardinality)
            if cardinality < 0:
                raise ValueError(
                    "PFPointEstimate cardinality keys must be nonnegative."
                )
            cardinality_entries.append((cardinality, probability))
        cardinality_entries.sort(key=lambda item: item[0])
        cardinality_probabilities = validated_probability_distribution(
            [probability for _, probability in cardinality_entries],
            name="PFPointEstimate.cardinality_distribution",
        )
        if map_cardinality not in {
            cardinality for cardinality, _ in cardinality_entries
        }:
            raise ValueError(
                "PFPointEstimate.map_cardinality is outside its distribution."
            )
        selected_stratum_mass = validated_probability(
            self.selected_stratum_mass,
            name="PFPointEstimate.selected_stratum_mass",
        )
        if not isinstance(self.modes, tuple):
            raise ValueError("PFPointEstimate.modes must be a tuple.")
        if len(self.modes) != map_cardinality:
            raise ValueError(
                "PFPointEstimate mode count must match map_cardinality."
            )
        return {
            "map_cardinality": map_cardinality,
            "cardinality_distribution": {
                str(cardinality): float(probability)
                for (cardinality, _), probability in zip(
                    cardinality_entries,
                    cardinality_probabilities,
                    strict=True,
                )
            },
            "selected_stratum_mass": selected_stratum_mass,
            "modes": [mode.to_dict() for mode in self.modes],
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
    structural_transition_provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    structural_model_manifest: Mapping[str, Any] = field(default_factory=dict)
    pure_pf_schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return the required JSON-safe PF result contract."""
        validated_schema_fields: dict[str, int] = {}
        for field_name, value, expected in (
            ("schema_version", self.schema_version, 1),
            ("pure_pf_schema_version", self.pure_pf_schema_version, 1),
            (
                "measurement_log_schema_version",
                self.measurement_log_schema_version,
                2,
            ),
        ):
            validated_value = _validated_nonnegative_integer(
                value,
                name=field_name,
            )
            if validated_value != expected:
                raise ValueError(
                    f"{field_name} must be the integer {expected}."
                )
            validated_schema_fields[field_name] = validated_value

        estimator_variant = _validated_nonempty_string(
            self.estimator_variant,
            name="estimator_variant",
        )
        repository_commit = _validated_hex_digest(
            self.repository_commit,
            name="repository_commit",
            lengths=(40, 64),
        )
        config_hash = _validated_hex_digest(
            self.config_hash,
            name="config_hash",
            lengths=(64,),
        )
        resolved_config_hash = _validated_hex_digest(
            self.resolved_config_hash,
            name="resolved_config_hash",
            lengths=(64,),
        )
        measurement_log_sha256 = _validated_hex_digest(
            self.measurement_log_sha256,
            name="measurement_log_sha256",
            lengths=(64,),
        )
        random_seed = _validated_nonnegative_integer(
            self.random_seed,
            name="random_seed",
        )
        record_count = _validated_nonnegative_integer(
            self.record_count,
            name="record_count",
        )
        if not isinstance(self.planner_belief_sources, tuple):
            raise ValueError("planner_belief_sources must be a tuple of strings.")
        planner_belief_sources = tuple(
            _validated_nonempty_string(
                value,
                name=f"planner_belief_sources[{index}]",
            )
            for index, value in enumerate(self.planner_belief_sources)
        )
        if not planner_belief_sources:
            raise ValueError(
                "planner_belief_sources must contain at least one source."
            )
        if len(set(planner_belief_sources)) != len(planner_belief_sources):
            raise ValueError("planner_belief_sources must not contain duplicates.")

        structural = dict(
            sorted(
                _validated_string_key_mapping(
                    self.structural_transition_provenance,
                    name="structural_transition_provenance",
                ).items()
            )
        )
        if not structural:
            raise ValueError(
                "Pure-PF posterior snapshots require structural-transition "
                "provenance."
            )
        required_structural_fields = {
            "posterior_semantics",
            "structural_kernel_exact_rj",
            "structural_kernel_family",
            "structural_kernel_target_preserving",
            "structural_moves_enabled",
            "reversible_jump_mcmc_used",
            "support_domain",
            "variable_cardinality",
            "birth_death_moves_enabled",
            "within_cardinality_moves_enabled",
            "within_cardinality_kernel_exact_mh",
        }
        missing_structural_fields = sorted(
            required_structural_fields.difference(structural)
        )
        if missing_structural_fields:
            raise ValueError(
                "Pure-PF structural provenance is incomplete: "
                + ", ".join(missing_structural_fields)
            )
        structural_boolean_fields = {
            "structural_kernel_exact_rj",
            "structural_kernel_target_preserving",
            "structural_moves_enabled",
            "reversible_jump_mcmc_used",
            "variable_cardinality",
            "birth_death_moves_enabled",
            "within_cardinality_moves_enabled",
            "within_cardinality_kernel_exact_mh",
        }
        for field_name in structural_boolean_fields:
            structural[field_name] = _validated_exact_boolean(
                structural[field_name],
                name=(
                    "structural_transition_provenance."
                    f"{field_name}"
                ),
            )
        if "structural_evidence_uses_pf_likelihood" in structural:
            structural["structural_evidence_uses_pf_likelihood"] = (
                _validated_exact_boolean(
                    structural["structural_evidence_uses_pf_likelihood"],
                    name=(
                        "structural_transition_provenance."
                        "structural_evidence_uses_pf_likelihood"
                    ),
                )
            )
        for field_name in (
            "posterior_semantics",
            "structural_kernel_family",
            "support_domain",
        ):
            structural[field_name] = _validated_nonempty_string(
                structural[field_name],
                name=(
                    "structural_transition_provenance."
                    f"{field_name}"
                ),
            )

        structural_model = _validated_string_key_mapping(
            self.structural_model_manifest,
            name="structural_model_manifest",
        )
        model_schema = structural_model.get("pure_pf_schema_version")
        if (
            type(model_schema) is not int
            or model_schema != validated_schema_fields["pure_pf_schema_version"]
            or structural_model.get("support_domain") != "environment_surface"
            or "strength_prior" not in structural_model
        ):
            raise ValueError(
                "Pure-PF posterior snapshots require a schema-v1 structural "
                "model over the environment surface with a declared strength "
                "prior."
            )
        if not isinstance(structural_model["strength_prior"], Mapping):
            raise ValueError(
                "structural_model_manifest.strength_prior must be a mapping."
            )

        capability_items = _validated_string_key_mapping(
            self.profile_capability_map,
            name="profile_capability_map",
        )
        if not capability_items:
            raise ValueError(
                "profile_capability_map must contain at least one capability."
            )
        profile_capability_map = {
            key: _validated_exact_boolean(
                value,
                name=f"profile_capability_map.{key}",
            )
            for key, value in sorted(capability_items.items())
        }
        isotope_items = _validated_string_key_mapping(
            self.isotopes,
            name="isotopes",
        )
        provenance = {
            "estimator_repository": "moeuu/Rotating-shield-particle-filter",
            "estimator_commit": repository_commit,
            "measurement_log_schema_version": validated_schema_fields[
                "measurement_log_schema_version"
            ],
            "measurement_log_sha256": measurement_log_sha256,
            "resolved_config_sha256": resolved_config_hash,
            "config_sha256": config_hash,
            "random_seed": random_seed,
            "pure_pf_schema_version": validated_schema_fields[
                "pure_pf_schema_version"
            ],
            "planner_belief_sources": list(planner_belief_sources),
            "posterior_semantics": structural["posterior_semantics"],
            "structural_transition_provenance": dict(structural),
            "structural_model_manifest": dict(structural_model),
        }
        return {
            "schema_version": validated_schema_fields["schema_version"],
            "pure_pf_schema_version": validated_schema_fields[
                "pure_pf_schema_version"
            ],
            "estimator_family": "particle_filter",
            "estimator_variant": estimator_variant,
            "estimator_profile": estimator_variant,
            "final_estimate_source": "pf_posterior",
            "posterior_semantics": structural["posterior_semantics"],
            "structural_kernel_family": structural["structural_kernel_family"],
            "structural_kernel_target_preserving": structural[
                "structural_kernel_target_preserving"
            ],
            "structural_kernel_exact_rj": structural[
                "structural_kernel_exact_rj"
            ],
            "reversible_jump_mcmc_used": structural[
                "reversible_jump_mcmc_used"
            ],
            "structural_transition_provenance": dict(structural),
            "structural_model_manifest": dict(structural_model),
            "planner_belief_sources": list(planner_belief_sources),
            "repository_commit": repository_commit,
            "measurement_log_schema_version": validated_schema_fields[
                "measurement_log_schema_version"
            ],
            "resolved_config_hash": resolved_config_hash,
            "resolved_config_sha256": resolved_config_hash,
            "config_sha256": config_hash,
            "measurement_log_sha256": measurement_log_sha256,
            "random_seed": random_seed,
            "provenance": provenance,
            "profile_capability_map": profile_capability_map,
            "record_count": record_count,
            "isotopes": {
                isotope: estimate.to_dict()
                for isotope, estimate in sorted(isotope_items.items())
            },
        }


def _normalized_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized posterior weights or fail on an invalid posterior."""
    result = np.asarray(weights, dtype=float).reshape(-1)
    if result.size == 0:
        return result
    if np.any(~np.isfinite(result)):
        raise ValueError("Posterior weights must all be finite.")
    if np.any(result < 0.0):
        raise ValueError("Posterior weights must all be nonnegative.")
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Posterior weights must have a finite positive sum.")
    return result / total


def validated_state_cardinality(
    state: Any,
    *,
    name: str,
    max_cardinality: int | None = None,
) -> int:
    """Return one exact PF source count inside the declared state support."""
    if not hasattr(state, "num_sources"):
        raise ValueError(f"{name} must expose num_sources.")
    raw_cardinality = state.num_sources
    if isinstance(raw_cardinality, (bool, np.bool_)) or not isinstance(
        raw_cardinality,
        (int, np.integer),
    ):
        raise TypeError(f"{name}.num_sources must be an integer.")
    cardinality = int(raw_cardinality)
    if cardinality < 0:
        raise ValueError(f"{name}.num_sources must be nonnegative.")
    if max_cardinality is not None:
        if isinstance(max_cardinality, (bool, np.bool_)) or not isinstance(
            max_cardinality,
            (int, np.integer),
        ):
            raise TypeError("max_cardinality must be an integer.")
        maximum = int(max_cardinality)
        if maximum < 0:
            raise ValueError("max_cardinality must be nonnegative.")
        if cardinality > maximum:
            raise ValueError(
                f"{name}.num_sources exceeds the configured maximum "
                f"{maximum}."
            )
    return cardinality


def weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a deterministic weighted quantile for one scalar posterior field."""
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = _normalized_weights(weights)
    if value_array.size == 0:
        raise ValueError("Weighted quantiles require at least one sample.")
    if value_array.size != weight_array.size:
        raise ValueError("values and weights must have equal lengths.")
    if np.any(~np.isfinite(value_array)):
        raise ValueError("Weighted-quantile values must all be finite.")
    if isinstance(quantile, (bool, np.bool_)) or not isinstance(
        quantile,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("quantile must be numeric.")
    resolved_quantile = float(quantile)
    if not np.isfinite(resolved_quantile) or not 0.0 <= resolved_quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    index = int(
        np.searchsorted(cumulative, resolved_quantile, side="left")
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
    if max_cardinality is not None:
        if isinstance(max_cardinality, (bool, np.bool_)) or not isinstance(
            max_cardinality,
            (int, np.integer),
        ):
            raise TypeError("max_cardinality must be an integer.")
        output_max = int(max_cardinality)
        if output_max < 0:
            raise ValueError("max_cardinality must be nonnegative.")
    else:
        output_max = None
    cardinalities = np.fromiter(
        (
            validated_state_cardinality(
                state,
                name=f"states[{index}]",
                max_cardinality=output_max,
            )
            for index, state in enumerate(states)
        ),
        dtype=np.int64,
        count=len(states),
    )
    observed_max = int(np.max(cardinalities)) if cardinalities.size else 0
    if output_max is None:
        output_max = observed_max
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
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations,
        (int, np.integer),
    ):
        raise TypeError("max_iterations must be an integer.")
    iteration_count = int(max_iterations)
    if iteration_count < 1:
        raise ValueError("max_iterations must be positive.")
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

    for _ in range(iteration_count):
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


def _validated_surface_distances(
    distance_function: SurfaceCoordinateDistance,
    first_chart_ids: NDArray[np.int64],
    first_uv: NDArray[np.float64],
    second_chart_ids: NDArray[np.int64],
    second_uv: NDArray[np.float64],
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Evaluate one batched intrinsic-distance call and reject bad geometry."""
    distances = np.asarray(
        distance_function(
            first_chart_ids,
            first_uv,
            second_chart_ids,
            second_uv,
        ),
        dtype=np.float64,
    )
    if distances.shape != expected_shape:
        raise ValueError(
            "surface_coordinate_path_distance returned an unexpected shape."
        )
    if np.any(np.isnan(distances)) or np.any(distances < 0.0):
        raise ValueError(
            "Surface distances must be nonnegative finite values or infinity."
        )
    return distances


def _surface_mode_medoid_coordinates_batched(
    chart_ids: NDArray[np.int64],
    surface_uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    distance_function: SurfaceCoordinateDistance,
    *,
    candidate_chunk_size: int = 64,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Select exact weighted medoids from every aligned posterior row.

    Candidate rows are processed in vectorized chunks so the runtime uses
    ``O(P * K * chunk_size)`` distance memory rather than materializing the
    full ``P * K * P`` tensor.  Every positive- or zero-weight posterior row
    remains a candidate: the chunk size changes only the execution schedule,
    never the mathematical medoid.
    """
    ids = np.asarray(chart_ids, dtype=np.int64)
    uv = np.asarray(surface_uv, dtype=np.float64)
    normalized = _normalized_weights(weights)
    if ids.ndim != 2 or uv.shape != ids.shape + (2,):
        raise ValueError(
            "Aligned surface coordinates must have shapes (P, K) and (P, K, 2)."
        )
    if ids.shape[0] != normalized.size or ids.shape[0] == 0:
        raise ValueError("Surface medoids require one weight per particle.")
    if isinstance(candidate_chunk_size, (bool, np.bool_)) or not isinstance(
        candidate_chunk_size,
        (int, np.integer),
    ):
        raise TypeError("candidate_chunk_size must be an integer.")
    chunk_size = int(candidate_chunk_size)
    if chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive.")

    particle_count, source_count = ids.shape
    connected_mass = np.empty(
        (source_count, particle_count),
        dtype=np.float64,
    )
    finite_cost = np.empty_like(connected_mass)
    for start in range(0, particle_count, chunk_size):
        stop = min(start + chunk_size, particle_count)
        distances = _validated_surface_distances(
            distance_function,
            ids[start:stop].T[None, :, :],
            np.swapaxes(uv[start:stop], 0, 1)[None, :, :, :],
            ids[:, :, None],
            uv[:, :, None, :],
            expected_shape=(
                particle_count,
                source_count,
                stop - start,
            ),
        )
        finite = np.isfinite(distances)
        connected_mass[:, start:stop] = np.einsum(
            "p,pkc->kc",
            normalized,
            finite,
            optimize=True,
        )
        finite_cost[:, start:stop] = np.einsum(
            "p,pkc->kc",
            normalized,
            np.where(finite, np.square(distances), 0.0),
            optimize=True,
        )
    maximum_connected_mass = np.max(connected_mass, axis=1, keepdims=True)
    dominant_component = np.isclose(
        connected_mass,
        maximum_connected_mass,
        rtol=0.0,
        atol=1.0e-15,
    )
    candidate_score = np.where(dominant_component, finite_cost, np.inf)
    minimum_score = np.min(candidate_score, axis=1, keepdims=True)
    score_ties = np.isclose(
        candidate_score,
        minimum_score,
        rtol=0.0,
        atol=1.0e-15,
    )
    tie_weights = np.where(score_ties, normalized[None, :], -np.inf)
    maximum_tie_weight = np.max(tie_weights, axis=1, keepdims=True)
    final_ties = score_ties & np.isclose(
        tie_weights,
        maximum_tie_weight,
        rtol=0.0,
        atol=1.0e-15,
    )
    chosen_rows = np.argmax(final_ties, axis=1)
    mode_indices = np.arange(source_count, dtype=np.int64)
    return (
        ids[chosen_rows, mode_indices],
        uv[chosen_rows, mode_indices],
    )


def align_surface_modes_batched(
    positions: NDArray[np.float64],
    strengths: NDArray[np.float64],
    surface_chart_ids: NDArray[np.int64],
    surface_uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    surface_coordinate_path_distance: SurfaceCoordinateDistance,
    *,
    max_iterations: int = 32,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Align continuous-surface modes using intrinsic topology and distance.

    Cartesian proximity is insufficient for thin obstacles and nearby room
    faces.  This routine therefore performs every assignment against explicit
    chart/UV states and treats disconnected surface components as infinitely
    far apart.  Mode centers are actual posterior surface states selected as
    deterministic weighted medoids.
    """
    position_array = np.asarray(positions, dtype=np.float64)
    strength_array = np.asarray(strengths, dtype=np.float64)
    chart_array = np.asarray(surface_chart_ids, dtype=np.int64)
    uv_array = np.asarray(surface_uv, dtype=np.float64)
    normalized = _normalized_weights(weights)
    if position_array.ndim != 3 or position_array.shape[2] != 3:
        raise ValueError("positions must have shape (particles, sources, 3).")
    particle_count, source_count, _ = position_array.shape
    if strength_array.shape != (particle_count, source_count):
        raise ValueError("strengths must have shape (particles, sources).")
    if chart_array.shape != (particle_count, source_count):
        raise ValueError(
            "surface_chart_ids must have shape (particles, sources)."
        )
    if uv_array.shape != (particle_count, source_count, 2):
        raise ValueError(
            "surface_uv must have shape (particles, sources, 2)."
        )
    if normalized.size != particle_count:
        raise ValueError("weights must have one value per particle.")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations,
        (int, np.integer),
    ):
        raise TypeError("max_iterations must be an integer.")
    iteration_count = int(max_iterations)
    if iteration_count < 1:
        raise ValueError("max_iterations must be positive.")
    if source_count <= 1:
        return (
            position_array.copy(),
            strength_array.copy(),
            chart_array.copy(),
            uv_array.copy(),
        )
    if source_count > 8:
        raise ValueError(
            "Batched exact source alignment supports at most eight source slots."
        )

    flat_ids = chart_array.reshape(-1)
    flat_uv = uv_array.reshape(-1, 2)
    flat_weights = np.repeat(normalized / float(source_count), source_count)
    lexicographic = np.lexsort(
        (flat_uv[:, 1], flat_uv[:, 0], flat_ids)
    )
    ordered_ids = flat_ids[lexicographic]
    ordered_uv = flat_uv[lexicographic]
    ordered_weights = flat_weights[lexicographic]
    center_ids = np.empty(source_count, dtype=np.int64)
    center_uv = np.empty((source_count, 2), dtype=np.float64)
    first_index = int(np.argmax(ordered_weights))
    center_ids[0] = ordered_ids[first_index]
    center_uv[0] = ordered_uv[first_index]
    min_distance = _validated_surface_distances(
        surface_coordinate_path_distance,
        center_ids[0],
        center_uv[0],
        ordered_ids,
        ordered_uv,
        expected_shape=(ordered_ids.size,),
    )
    for center_index in range(1, source_count):
        score = np.zeros_like(ordered_weights)
        np.multiply(
            ordered_weights,
            np.square(min_distance),
            out=score,
            where=ordered_weights > 0.0,
        )
        next_index = int(np.argmax(score))
        if float(score[next_index]) <= 0.0:
            next_index = min(center_index, ordered_ids.size - 1)
        center_ids[center_index] = ordered_ids[next_index]
        center_uv[center_index] = ordered_uv[next_index]
        distance = _validated_surface_distances(
            surface_coordinate_path_distance,
            center_ids[center_index],
            center_uv[center_index],
            ordered_ids,
            ordered_uv,
            expected_shape=(ordered_ids.size,),
        )
        min_distance = np.minimum(min_distance, distance)

    permutations = np.asarray(
        tuple(itertools.permutations(range(source_count))),
        dtype=np.int64,
    )
    center_indices = np.arange(source_count, dtype=np.int64)

    def _assignment(
        current_center_ids: NDArray[np.int64],
        current_center_uv: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Return the minimum intrinsic-cost source order for every particle."""
        distances = _validated_surface_distances(
            surface_coordinate_path_distance,
            current_center_ids[None, None, :],
            current_center_uv[None, None, :, :],
            chart_array[:, :, None],
            uv_array[:, :, None, :],
            expected_shape=(particle_count, source_count, source_count),
        )
        cost = np.square(distances)
        permutation_cost = np.sum(
            cost[:, permutations, center_indices],
            axis=2,
        )
        return permutations[np.argmin(permutation_cost, axis=1)]

    for _ in range(iteration_count):
        source_order = _assignment(center_ids, center_uv)
        aligned_ids = np.take_along_axis(chart_array, source_order, axis=1)
        aligned_uv = np.take_along_axis(
            uv_array,
            source_order[:, :, None],
            axis=1,
        )
        updated_ids, updated_uv = _surface_mode_medoid_coordinates_batched(
            aligned_ids,
            aligned_uv,
            normalized,
            surface_coordinate_path_distance,
        )
        if np.array_equal(updated_ids, center_ids) and np.allclose(
            updated_uv,
            center_uv,
            rtol=0.0,
            atol=1.0e-12,
        ):
            center_ids = updated_ids
            center_uv = updated_uv
            break
        center_ids = updated_ids
        center_uv = updated_uv

    center_order = np.lexsort(
        (center_uv[:, 1], center_uv[:, 0], center_ids)
    )
    final_order = _assignment(
        center_ids[center_order],
        center_uv[center_order],
    )
    return (
        np.take_along_axis(position_array, final_order[:, :, None], axis=1),
        np.take_along_axis(strength_array, final_order, axis=1),
        np.take_along_axis(chart_array, final_order, axis=1),
        np.take_along_axis(uv_array, final_order[:, :, None], axis=1),
    )


def surface_configuration_medoid_distance_batched(
    aligned_chart_ids: NDArray[np.int64],
    aligned_surface_uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    surface_coordinate_path_distance: SurfaceCoordinateDistance,
) -> NDArray[np.float64]:
    """Return each aligned row's intrinsic distance to surface-mode medoids."""
    ids = np.asarray(aligned_chart_ids, dtype=np.int64)
    uv = np.asarray(aligned_surface_uv, dtype=np.float64)
    normalized = _normalized_weights(weights)
    medoid_ids, medoid_uv = _surface_mode_medoid_coordinates_batched(
        ids,
        uv,
        normalized,
        surface_coordinate_path_distance,
    )
    distances = _validated_surface_distances(
        surface_coordinate_path_distance,
        medoid_ids[None, :],
        medoid_uv[None, :, :],
        ids,
        uv,
        expected_shape=ids.shape,
    )
    finite = np.isfinite(distances)
    finite_squared = np.where(finite, np.square(distances), 0.0)
    finite_scale = float(np.max(finite_squared, initial=0.0)) + 1.0
    return (
        np.sum(~finite, axis=1, dtype=np.int64)
        * finite_scale
        * float(max(1, ids.shape[1]))
        + np.sum(finite_squared, axis=1)
    )


def posterior_point_estimate_from_states(
    states: Sequence[Any],
    weights: NDArray[np.float64],
    *,
    positions_by_state: Sequence[NDArray[np.float64]],
    surface_chart_ids_by_state: Sequence[NDArray[np.int64]] | None = None,
    surface_uv_by_state: Sequence[NDArray[np.float64]] | None = None,
    surface_coordinate_path_distance: SurfaceCoordinateDistance | None = None,
    max_cardinality: int | None = None,
    surface_path_distance: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ]
    | None = None,
    selected_particle_indices: NDArray[np.int64] | None = None,
    representative_particle_index: int | None = None,
    selected_stratum_mass: float | None = None,
) -> PFPointEstimate:
    """Aggregate a PF-only estimate from a deterministic MAP-cardinality stratum.

    State extraction is necessarily linear in the particle container. All
    numerical aggregation, source ordering, covariance, and cardinality mass
    calculations use batched NumPy arrays; the only scalar loop is over the
    configured source-slot cap, which is tiny in the full runtime. Reported
    positions are taken together from one weighted squared-distance medoid in
    the joint source-configuration space.  Reporting therefore neither invents
    an off-surface mean nor combines source locations that never coexisted in a
    posterior particle.  When a surface-path callback is supplied, the report
    additionally records a conservative path-radius and the mass connected to
    its representative.
    """
    normalized = _normalized_weights(weights)
    if len(states) != normalized.size:
        raise ValueError("states and weights must have equal lengths.")
    if len(positions_by_state) != len(states):
        raise ValueError(
            "positions_by_state must contain one atlas-derived XYZ array per "
            "state."
        )
    surface_coordinate_inputs = (
        surface_chart_ids_by_state,
        surface_uv_by_state,
        surface_coordinate_path_distance,
    )
    if any(value is not None for value in surface_coordinate_inputs) and not all(
        value is not None for value in surface_coordinate_inputs
    ):
        raise ValueError(
            "Surface-aware reporting requires chart IDs, UV, and the intrinsic "
            "distance callback together."
        )
    if surface_chart_ids_by_state is not None and (
        len(surface_chart_ids_by_state) != len(states)
        or surface_uv_by_state is None
        or len(surface_uv_by_state) != len(states)
    ):
        raise ValueError(
            "Surface chart/UV rows must contain one entry per PF state."
        )
    distribution = cardinality_distribution_from_states(
        states,
        normalized,
        max_cardinality=max_cardinality,
    )
    for state_index, (state, positions) in enumerate(
        zip(states, positions_by_state, strict=True)
    ):
        cardinality = validated_state_cardinality(
            state,
            name=f"states[{state_index}]",
            max_cardinality=max_cardinality,
        )
        position_row = np.asarray(positions, dtype=np.float64)
        strength_row = np.asarray(state.strengths, dtype=np.float64)
        if (
            position_row.shape != (cardinality, 3)
            or np.any(~np.isfinite(position_row))
        ):
            raise ValueError(
                "Each particle position row must contain exactly one finite "
                "XYZ coordinate per active source."
            )
        if (
            strength_row.shape != (cardinality,)
            or np.any(~np.isfinite(strength_row))
            or np.any(strength_row <= 0.0)
        ):
            raise ValueError(
                "Each particle strength row must contain exactly one positive "
                "finite value per active source."
            )
        if (
            surface_chart_ids_by_state is not None
            and surface_uv_by_state is not None
        ):
            raw_chart_ids = np.asarray(
                surface_chart_ids_by_state[state_index]
            )
            surface_uv_row = np.asarray(
                surface_uv_by_state[state_index],
                dtype=np.float64,
            )
            if (
                not np.issubdtype(raw_chart_ids.dtype, np.integer)
                or raw_chart_ids.shape != (cardinality,)
                or np.any(raw_chart_ids < 0)
                or surface_uv_row.shape != (cardinality, 2)
                or np.any(~np.isfinite(surface_uv_row))
                or np.any(surface_uv_row < 0.0)
                or np.any(surface_uv_row > 1.0)
            ):
                raise ValueError(
                    "Each particle must provide authoritative chart/UV arrays "
                    "matching its active source count."
                )
    if not states:
        return PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution=distribution,
            selected_stratum_mass=0.0,
            modes=(),
        )
    if selected_particle_indices is None:
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
    else:
        selected_indices = np.asarray(
            selected_particle_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            selected_indices.size == 0
            or np.any(selected_indices < 0)
            or np.any(selected_indices >= len(states))
            or np.unique(selected_indices).size != selected_indices.size
        ):
            raise ValueError(
                "selected_particle_indices must be nonempty, unique, and valid."
            )
        selected_cardinalities = np.asarray(
            [
                int(states[int(index)].num_sources)
                for index in selected_indices
            ],
            dtype=np.int64,
        )
        if np.unique(selected_cardinalities).size != 1:
            raise ValueError(
                "Selected posterior particles must share one cardinality."
            )
        map_cardinality = int(selected_cardinalities[0])
    if selected_indices.size == 0:
        raise RuntimeError("MAP cardinality stratum has no particles.")
    selected_weights = _normalized_weights(normalized[selected_indices])
    selected_states = [states[int(index)] for index in selected_indices]
    if selected_stratum_mass is None:
        stratum_mass = float(distribution[map_cardinality])
    else:
        stratum_mass = validated_probability(
            selected_stratum_mass,
            name="selected_stratum_mass",
        )
    if map_cardinality == 0:
        return PFPointEstimate(
            map_cardinality=0,
            cardinality_distribution=distribution,
            selected_stratum_mass=stratum_mass,
            modes=(),
        )

    selected_position_rows = [
        np.asarray(positions_by_state[int(index)], dtype=float)
        for index in selected_indices
    ]
    position_rows = np.stack(selected_position_rows, axis=0)
    strength_rows = np.stack(
        [
            np.asarray(state.strengths, dtype=float)
            for state in selected_states
        ],
        axis=0,
    )
    if position_rows.shape != (selected_indices.size, map_cardinality, 3):
        raise ValueError("particle positions do not match their cardinality.")
    if strength_rows.shape != (selected_indices.size, map_cardinality):
        raise ValueError("particle strengths do not match their cardinality.")

    aligned_chart_ids: NDArray[np.int64] | None = None
    aligned_surface_uv: NDArray[np.float64] | None = None
    if (
        surface_chart_ids_by_state is None
        or surface_uv_by_state is None
        or surface_coordinate_path_distance is None
    ):
        aligned_positions, aligned_strengths = align_spatial_modes_batched(
            position_rows,
            strength_rows,
            selected_weights,
        )
    else:
        chart_id_rows = np.stack(
            [
                np.asarray(
                    surface_chart_ids_by_state[int(index)],
                    dtype=np.int64,
                )
                for index in selected_indices
            ],
            axis=0,
        )
        surface_uv_rows = np.stack(
            [
                np.asarray(
                    surface_uv_by_state[int(index)],
                    dtype=np.float64,
                )
                for index in selected_indices
            ],
            axis=0,
        )
        (
            aligned_positions,
            aligned_strengths,
            aligned_chart_ids,
            aligned_surface_uv,
        ) = align_surface_modes_batched(
            position_rows,
            strength_rows,
            chart_id_rows,
            surface_uv_rows,
            selected_weights,
            surface_coordinate_path_distance,
        )
    position_barycenter = np.einsum(
        "n,nkd->kd",
        selected_weights,
        aligned_positions,
        optimize=True,
    )
    if (
        aligned_chart_ids is not None
        and aligned_surface_uv is not None
        and surface_coordinate_path_distance is not None
    ):
        configuration_distance = (
            surface_configuration_medoid_distance_batched(
                aligned_chart_ids,
                aligned_surface_uv,
                selected_weights,
                surface_coordinate_path_distance,
            )
        )
    else:
        configuration_distance = np.sum(
            (aligned_positions - position_barycenter[None, :, :]) ** 2,
            axis=(1, 2),
        )
    if representative_particle_index is None:
        minimum_distance = float(np.min(configuration_distance))
        tied = np.flatnonzero(
            np.isclose(
                configuration_distance,
                minimum_distance,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        representative_index = int(tied[np.argmax(selected_weights[tied])])
    else:
        matches = np.flatnonzero(
            selected_indices == int(representative_particle_index)
        )
        if matches.size != 1:
            raise ValueError(
                "representative_particle_index must belong to the selected "
                "posterior stratum."
            )
        representative_index = int(matches[0])
    representative_positions = aligned_positions[representative_index].copy()
    covariance_delta = aligned_positions - position_barycenter[None, :, :]
    position_covariance = np.einsum(
        "n,nki,nkj->kij",
        selected_weights,
        covariance_delta,
        covariance_delta,
        optimize=True,
    )
    strength_mean = np.einsum(
        "n,nk->k",
        selected_weights,
        aligned_strengths,
        optimize=True,
    )
    representative_strengths = aligned_strengths[
        representative_index
    ].copy()
    radial_distance = np.linalg.norm(
        aligned_positions - representative_positions[None, :, :],
        axis=2,
    )
    if (
        aligned_chart_ids is not None
        and aligned_surface_uv is not None
        and surface_coordinate_path_distance is not None
    ):
        surface_path_distances = _validated_surface_distances(
            surface_coordinate_path_distance,
            aligned_chart_ids[representative_index][None, :],
            aligned_surface_uv[representative_index][None, :, :],
            aligned_chart_ids,
            aligned_surface_uv,
            expected_shape=radial_distance.shape,
        )
    elif surface_path_distance is None:
        surface_path_distances = None
    else:
        surface_path_distances = np.asarray(
            surface_path_distance(
                aligned_positions,
                representative_positions[None, :, :],
            ),
            dtype=float,
        )
        if surface_path_distances.shape != radial_distance.shape:
            raise ValueError(
                "surface_path_distance must return one value per particle/source."
            )
        if (
            np.any(np.isnan(surface_path_distances))
            or np.any(surface_path_distances < 0.0)
        ):
            raise ValueError(
                "surface_path_distance must return nonnegative values or infinity."
            )
    modes: list[PFSourceMode] = []
    for source_index in range(map_cardinality):
        surface_radius: float | None = None
        surface_connected_mass = 1.0
        if surface_path_distances is not None:
            source_path_distances = surface_path_distances[:, source_index]
            finite_path = np.isfinite(source_path_distances)
            surface_connected_mass = float(
                np.sum(selected_weights[finite_path])
            )
            path_order = np.argsort(source_path_distances, kind="stable")
            path_cdf = np.cumsum(selected_weights[path_order])
            path_quantile_index = min(
                int(np.searchsorted(path_cdf, 0.95, side="left")),
                path_order.size - 1,
            )
            path_quantile = float(
                source_path_distances[path_order[path_quantile_index]]
            )
            if np.isfinite(path_quantile):
                surface_radius = path_quantile
        covariance_tuple = tuple(
            tuple(float(value) for value in row)
            for row in position_covariance[source_index]
        )
        modes.append(
            PFSourceMode(
                label_index=int(source_index),
                position_medoid_xyz=tuple(
                    float(value) for value in representative_positions[source_index]
                ),
                position_covariance_xyz=covariance_tuple,
                credible_radius_95_m=weighted_quantile(
                    radial_distance[:, source_index],
                    selected_weights,
                    0.95,
                ),
                strength_representative_cps_1m=float(
                    representative_strengths[source_index]
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
                credible_surface_path_radius_95_m=surface_radius,
                surface_connected_mass=surface_connected_mass,
                surface_chart_id=(
                    None
                    if aligned_chart_ids is None
                    else int(
                        aligned_chart_ids[
                            representative_index,
                            source_index,
                        ]
                    )
                ),
                surface_uv=(
                    None
                    if aligned_surface_uv is None
                    else tuple(
                        float(value)
                        for value in aligned_surface_uv[
                            representative_index,
                            source_index,
                        ]
                    )
                ),
            )
        )
    return PFPointEstimate(
        map_cardinality=int(map_cardinality),
        cardinality_distribution=distribution,
        selected_stratum_mass=stratum_mass,
        modes=tuple(modes),
    )
