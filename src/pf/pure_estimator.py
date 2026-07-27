"""Expose the sequential particle-filter estimator used by scientific runtimes."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from pf.estimator import (
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator as _PFEstimatorCore,
)
from pf.posterior import (
    PFPointEstimate,
    PFPosteriorSnapshot,
    cardinality_distribution_from_states,
    posterior_point_estimate_from_states,
)
from pf.profiles import (
    PURE_PF_SCHEMA_VERSION,
    apply_profile_to_config,
    resolve_structural_transition_provenance,
)
from pf.provenance import canonical_json_bytes, repository_commit, sha256_json


class PurePFBoundaryError(RuntimeError):
    """Signal a violation of the sequential PF result contract."""


def _ordered_surface_dictionary_sha256(
    centers_xyz: NDArray[np.float64],
    areas_m2: NDArray[np.float64],
) -> str:
    """Hash ordered patch centers and areas using a stable binary encoding."""
    digest = hashlib.sha256()
    digest.update(b"pure-pf-surface-centers-areas-float64-le-v1\0")
    arrays = (
        (b"centers_xyz\0", np.asarray(centers_xyz, dtype="<f8")),
        (b"areas_m2\0", np.asarray(areas_m2, dtype="<f8")),
    )
    for label, values in arrays:
        contiguous = np.ascontiguousarray(values)
        shape = np.asarray(
            (contiguous.ndim, *contiguous.shape),
            dtype="<u8",
        )
        digest.update(label)
        digest.update(shape.tobytes(order="C"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _resolved_cardinality_prior(
    config: RotatingShieldPFConfig,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    """Return resolved cardinality support, normalized mass, and its source."""
    if not bool(config.variable_cardinality):
        fixed_cardinality = int(config.init_num_sources[0])
        return (fixed_cardinality,), (1.0,), "fixed_init_num_sources"
    max_sources = config.max_sources
    if max_sources is None:
        return (), (), "unbounded_support_unavailable"
    support = tuple(range(int(max_sources) + 1))
    configured = config.structural_cardinality_prior_probs
    if configured is None:
        probability = 1.0 / float(len(support))
        return (
            support,
            tuple(probability for _ in support),
            "uniform_default",
        )
    probabilities = np.asarray(configured, dtype=float).reshape(-1)
    if probabilities.size != len(support):
        raise PurePFBoundaryError(
            "Structural cardinality prior length differs from resolved support."
        )
    total = float(np.sum(probabilities))
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.isfinite(total)
        or total <= 0.0
    ):
        raise PurePFBoundaryError(
            "Structural cardinality prior must contain finite positive mass."
        )
    normalized = probabilities / total
    return support, tuple(float(value) for value in normalized), "explicit"


class PurePFEstimator(_PFEstimatorCore):
    """Run causal PF updates and report only the resulting PF posterior."""

    planner_belief_sources: tuple[str, ...] = ("pf_posterior",)

    def __init__(
        self,
        *,
        measurement_log_schema_version: int = 1,
        config_hash: str | None = None,
        resolved_config_hash: str | None = None,
        measurement_log_sha256: str = "unavailable",
        random_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the PF and its immutable result provenance."""
        pure_config = kwargs.get("pf_config")
        if pure_config is None:
            pure_config = RotatingShieldPFConfig()
            kwargs["pf_config"] = pure_config
        capabilities = apply_profile_to_config(pure_config)
        super().__init__(**kwargs)
        if apply_profile_to_config(self.pf_config) != capabilities:
            raise PurePFBoundaryError(
                "PF capabilities changed during estimator initialization."
            )
        self.profile_capabilities = capabilities
        if (
            isinstance(measurement_log_schema_version, bool)
            or not isinstance(measurement_log_schema_version, int)
            or measurement_log_schema_version != 1
        ):
            raise ValueError(
                "PurePFEstimator supports MeasurementLog schema version 1."
            )
        self.measurement_log_schema_version = measurement_log_schema_version
        self.resolved_config_hash = (
            str(resolved_config_hash)
            if resolved_config_hash is not None
            else sha256_json(self.pf_config)
        )
        self.config_hash = (
            str(config_hash)
            if config_hash is not None
            else str(self.resolved_config_hash)
        )
        self.repository_commit = repository_commit()
        self.measurement_log_sha256 = str(measurement_log_sha256)
        self.random_seed = int(random_seed)

    @property
    def estimator_variant(self) -> str:
        """Return the resolved scientific PF variant."""
        return str(self.pf_config.estimator_profile)

    def structural_transition_diagnostics(self) -> dict[str, bool | str]:
        """Return target-preservation provenance for structural moves."""
        provenance = resolve_structural_transition_provenance(
            self.pf_config,
            capabilities=self.profile_capabilities,
        ).to_dict()
        variable_cardinality = bool(self.pf_config.variable_cardinality)
        provenance.update(
            {
                "support_domain": "environment_surface",
                "structural_moves_enabled": True,
                "variable_cardinality": variable_cardinality,
                "birth_death_moves_enabled": variable_cardinality,
                "within_cardinality_moves_enabled": True,
                "within_cardinality_kernel_exact_mh": True,
            }
        )
        if not variable_cardinality:
            provenance.update(
                {
                    "posterior_semantics": (
                        "fixed_cardinality_sequential_particle_filter_with_"
                        "target_preserving_mh_rejuvenation"
                    ),
                    "structural_kernel_family": (
                        "fixed_cardinality_surface_position_strength_mh"
                    ),
                }
            )
        return provenance

    def structural_model_manifest(self) -> dict[str, Any]:
        """Return outcome-independent structural-prior and RJ-kernel provenance."""
        support, probabilities, prior_source = _resolved_cardinality_prior(
            self.pf_config
        )
        variable_cardinality = bool(self.pf_config.variable_cardinality)
        configured_isotopes = sorted(
            {str(isotope) for isotope in self.isotopes}
        )
        dictionary_groups: dict[str, dict[str, Any]] = {}
        missing_isotopes: list[str] = []
        for isotope in configured_isotopes:
            filt = self.filters.get(isotope)
            patches = (
                None
                if filt is None
                else getattr(filt, "_structural_rj_surface_patches", None)
            )
            if patches is None:
                missing_isotopes.append(isotope)
                continue
            centers = np.asarray(patches.centers_xyz, dtype=float)
            areas = np.asarray(patches.areas_m2, dtype=float)
            dictionary_hash = _ordered_surface_dictionary_sha256(
                centers,
                areas,
            )
            group = dictionary_groups.setdefault(
                dictionary_hash,
                {
                    "ordered_centers_areas_sha256": dictionary_hash,
                    "patch_count": int(patches.patch_count),
                    "total_area_m2": float(np.sum(areas, dtype=np.float64)),
                    "geometry_metadata": dict(patches.geometry_metadata),
                    "isotopes": [],
                },
            )
            group["isotopes"].append(isotope)
        if not dictionary_groups:
            dictionary_status = "not_initialized"
            dictionaries_identical: bool | None = None
        elif missing_isotopes:
            dictionary_status = "partially_initialized"
            dictionaries_identical = False if len(dictionary_groups) > 1 else None
        else:
            dictionary_status = "complete"
            dictionaries_identical = len(dictionary_groups) == 1
        grouped_dictionaries = sorted(
            dictionary_groups.values(),
            key=lambda item: str(item["ordered_centers_areas_sha256"]),
        )
        for group in grouped_dictionaries:
            group["isotopes"] = sorted(str(value) for value in group["isotopes"])

        birth_weight = float(self.pf_config.structural_rj_birth_probability)
        death_weight = float(self.pf_config.structural_rj_death_probability)
        interior_total = birth_weight + death_weight
        interior_birth = birth_weight / interior_total if interior_total > 0.0 else 0.0
        interior_death = death_weight / interior_total if interior_total > 0.0 else 0.0
        max_cardinality = None if not support else int(support[-1])
        manifest_completeness = (
            "complete"
            if dictionary_status in {"complete", "not_applicable"}
            else (
                "partial"
                if dictionary_status == "partially_initialized"
                else "config_only"
            )
        )
        return {
            "schema_version": 1,
            "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
            "manifest_completeness": manifest_completeness,
            "support_domain": "environment_surface",
            "structural_moves_enabled": True,
            "variable_cardinality": variable_cardinality,
            "birth_death_moves_enabled": variable_cardinality,
            "within_cardinality_moves_enabled": True,
            "structural_kernel_target_preserving": True,
            "within_cardinality_kernel_exact_mh": True,
            "structural_kernel_exact_rj": variable_cardinality,
            "structural_kernel_family": (
                "area_weighted_surface_birth_death_rj_mh"
                if variable_cardinality
                else "fixed_cardinality_surface_position_strength_mh"
            ),
            "cardinality_prior": {
                "support": [int(value) for value in support],
                "probabilities": [float(value) for value in probabilities],
                "configuration_source": prior_source,
                "applies_independently_per_isotope": True,
            },
            "strength_prior": {
                "kind": "bounded_uniform",
                "minimum_cps_1m": float(
                    self.pf_config.strength_prior_min_cps_1m
                ),
                "maximum_cps_1m": float(
                    self.pf_config.strength_prior_max_cps_1m
                ),
                "units": "detector_cps_1m",
                "unit_definition": ("expected_net_detector_count_rate_at_1m"),
                "used_for_initialization": True,
                "shared_by_initialization_and_state_moves": True,
            },
            "surface_set_prior": {
                "support": "environment_surface",
                "semantics": "area_product_distinct_patch_sets",
                "probability_mass": (
                    "product(patch_area_m2)/elementary_symmetric_normalizer(K)"
                ),
                "used_for_initialization": True,
                "canonical_strictly_increasing_patch_indices": True,
                "duplicate_patch_indices_allowed": False,
                "patch_spacing_m": float(self.pf_config.structural_rj_patch_spacing_m),
                "dictionary_hash_encoding": (
                    "ordered_centers_xyz_and_areas_m2_float64_little_endian_v1"
                ),
                "dictionary_status": dictionary_status,
                "configured_isotopes": configured_isotopes,
                "missing_isotopes": sorted(missing_isotopes),
                "dictionaries_identical_across_isotopes": (dictionaries_identical),
                "dictionary_groups": grouped_dictionaries,
            },
            "rj_move_kernel": {
                "enabled": True,
                "target_preserving": True,
                "within_cardinality_exact_mh": True,
                "variable_cardinality_enabled": variable_cardinality,
                "birth_death_enabled": variable_cardinality,
                "exact_reversible_jump_mh": variable_cardinality,
                "structural_attempt_probability": float(
                    self.pf_config.structural_rj_move_probability
                ),
                "birth_death_direction_weights": {
                    "birth": birth_weight,
                    "death": death_weight,
                },
                "interior_birth_death_probabilities": {
                    "birth": float(interior_birth),
                    "death": float(interior_death),
                },
                "position_move_attempt_probability": float(
                    self.pf_config.structural_rj_position_move_probability
                ),
                "position_move_proposal": (
                    "area_weighted_conditional_prior_independence"
                ),
                "local_position_move_attempt_probability": float(
                    self.pf_config.structural_rj_local_position_move_probability
                ),
                "local_position_move_proposal": (
                    "uniform_unoccupied_physical_surface_neighbor"
                ),
                "local_position_reverse_correction": (
                    "forward_available_degree_over_reverse_available_degree"
                ),
                "global_position_move_retained_for_irreducibility": True,
                "strength_move_attempt_probability": float(
                    self.pf_config.structural_rj_strength_move_probability
                ),
                "boundary_normalization": {
                    "rule": ("renormalize_admissible_birth_death_direction_weights"),
                    "at_k_zero": {"birth": 1.0, "death": 0.0},
                    "at_k_max": (
                        None
                        if max_cardinality is None
                        else {
                            "cardinality": max_cardinality,
                            "birth": 0.0,
                            "death": 1.0,
                        }
                    ),
                },
                "dimension_matching": {
                    "absolute_jacobian_determinant": 1.0,
                    "log_absolute_jacobian_determinant": 0.0,
                },
            },
        }

    def posterior_cardinality_distribution(self) -> dict[str, dict[int, float]]:
        """Return source-count posterior mass for every isotope PF."""
        result: dict[str, dict[int, float]] = {}
        for isotope, filt in self.filters.items():
            states = [particle.state for particle in filt.continuous_particles]
            result[str(isotope)] = cardinality_distribution_from_states(
                states,
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
            )
        return result

    def posterior_point_estimate(self) -> dict[str, PFPointEstimate]:
        """Return deterministic PF point estimates and uncertainty."""
        result: dict[str, PFPointEstimate] = {}
        for isotope, filt in self.filters.items():
            states = [particle.state for particle in filt.continuous_particles]
            result[str(isotope)] = posterior_point_estimate_from_states(
                states,
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
                position_projector=filt._project_positions_to_source_prior,
            )
        return result

    def posterior_snapshot(self) -> PFPosteriorSnapshot:
        """Return a schema-v1 PF posterior result with reproducibility metadata."""
        log_digest = str(self.measurement_log_sha256).strip().lower()
        if len(log_digest) != 64 or any(
            character not in "0123456789abcdef" for character in log_digest
        ):
            raise PurePFBoundaryError(
                "A publishable PF posterior requires a finalized "
                "MeasurementLog SHA-256 digest."
            )
        return PFPosteriorSnapshot(
            estimator_variant=self.estimator_variant,
            isotopes=self.posterior_point_estimate(),
            planner_belief_sources=self.planner_belief_sources,
            repository_commit=self.repository_commit,
            measurement_log_schema_version=self.measurement_log_schema_version,
            config_hash=self.config_hash,
            resolved_config_hash=self.resolved_config_hash,
            measurement_log_sha256=self.measurement_log_sha256,
            random_seed=self.random_seed,
            profile_capability_map=self.profile_capabilities.to_dict(),
            record_count=len(self.measurements),
            structural_transition_provenance=(self.structural_transition_diagnostics()),
            structural_model_manifest=self.structural_model_manifest(),
        )

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project the PF posterior into the historical array result format."""
        result: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, point_estimate in self.posterior_point_estimate().items():
            if not point_estimate.modes:
                result[isotope] = (
                    np.zeros((0, 3), dtype=float),
                    np.zeros(0, dtype=float),
                )
                continue
            result[isotope] = (
                np.asarray(
                    [mode.position_mean_xyz for mode in point_estimate.modes],
                    dtype=float,
                ),
                np.asarray(
                    [mode.strength_mean_cps_1m for mode in point_estimate.modes],
                    dtype=float,
                ),
            )
        return result

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the PF posterior projection for visualization."""
        return self.estimates()

    def serialized_state(self) -> bytes:
        """Return canonical bytes for causality and determinism tests."""
        isotope_payload: dict[str, Any] = {}
        for isotope, filt in sorted(self.filters.items()):
            particles: list[dict[str, Any]] = []
            for particle in filt.continuous_particles:
                state = particle.state
                particles.append(
                    {
                        "log_weight": float(particle.log_weight),
                        "num_sources": int(state.num_sources),
                        "positions": np.asarray(state.positions, dtype=float),
                        "strengths": np.asarray(state.strengths, dtype=float),
                        "background": float(state.background),
                    }
                )
            isotope_payload[str(isotope)] = particles
        measurement_history = [
            {
                "z_k": measurement.z_k,
                "pose_idx": int(measurement.pose_idx),
                "fe_index": measurement.fe_index,
                "pb_index": measurement.pb_index,
                "detector_position_xyz_m": measurement.detector_position_xyz_m,
                "live_time_s": float(measurement.live_time_s),
                "z_variance_k": measurement.z_variance_k,
                "z_covariance_k": measurement.z_covariance_k,
                "station_sequence_id": measurement.station_sequence_id,
                "station_view_index": measurement.station_view_index,
                "runtime_likelihood_route_by_isotope": (
                    measurement.runtime_likelihood_route_by_isotope
                ),
                "station_view_covariance_by_isotope": (
                    measurement.station_view_covariance_by_isotope
                ),
            }
            for measurement in self.measurements
        ]
        return canonical_json_bytes(
            {
                "schema_version": 1,
                "estimator_variant": self.estimator_variant,
                "measurement_count": len(self.measurements),
                "measurement_poses_xyz": [
                    np.asarray(pose, dtype=float) for pose in self.poses
                ],
                "measurement_pose_indices": [
                    int(measurement.pose_idx) for measurement in self.measurements
                ],
                "measurement_history_sha256": sha256_json(measurement_history),
                "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
                "isotopes": isotope_payload,
            }
        )

__all__ = [
    "PurePFBoundaryError",
    "PurePFEstimator",
    "RotatingShieldPFConfig",
]
