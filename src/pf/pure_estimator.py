"""Expose the sequential particle-filter estimator used by scientific runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from measurement.surface_charts import surface_chart_geometry_sha256
from pf.estimator import (
    RotatingShieldPFConfig,
    RotatingShieldPFEstimator as _PFEstimatorCore,
)
from pf.posterior import (
    PFPointEstimate,
    PFPosteriorSnapshot,
    align_surface_modes_batched,
    cardinality_distribution_from_states,
    posterior_point_estimate_from_states,
    surface_configuration_medoid_distance_batched,
    validated_probability_distribution,
)
from pf.profiles import (
    PURE_PF_SCHEMA_VERSION,
    apply_profile_to_config,
    resolve_structural_transition_provenance,
)
from pf.provenance import (
    canonical_json_bytes,
    repository_commit,
    repository_source_snapshot_sha256,
    sha256_json,
)
from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    truncated_poisson_cardinality_probabilities,
    poisson_geometric_tail_cardinality_probabilities,
    validate_cardinality_prior_policy,
)


class PurePFBoundaryError(RuntimeError):
    """Signal a violation of the sequential PF result contract."""


def _resolved_cardinality_prior(
    config: RotatingShieldPFConfig,
) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    """Return resolved cardinality support, normalized mass, and its source."""
    if not bool(config.variable_cardinality):
        fixed_cardinality = int(config.init_num_sources[0])
        return (fixed_cardinality,), (1.0,), "fixed_init_num_sources"
    max_sources = config.cardinality_capacity
    if max_sources is None:
        return (), (), "unbounded_support_unavailable"
    support = tuple(range(int(max_sources) + 1))
    configured = config.structural_cardinality_prior_probs
    policy = validate_cardinality_prior_policy(
        config.structural_cardinality_prior_policy,
        has_explicit_probabilities=configured is not None,
    )
    if configured is None:
        if policy not in {
            TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
            POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
        }:
            raise PurePFBoundaryError(
                "Resolved implicit cardinality prior has the wrong policy."
            )
        if policy == POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY:
            probabilities = poisson_geometric_tail_cardinality_probabilities(
                int(config.max_sources),
                int(max_sources),
                float(config.structural_cardinality_prior_mean),
                float(config.structural_cardinality_tail_ratio),
            )
        else:
            probabilities = truncated_poisson_cardinality_probabilities(
                int(max_sources),
                float(config.structural_cardinality_prior_mean),
            )
        probabilities = validated_probability_distribution(
            probabilities,
            name="resolved truncated-Poisson cardinality prior",
        )
        return (
            support,
            tuple(float(value) for value in probabilities),
            "truncated_poisson_surface_process",
        )
    probabilities = np.asarray(configured, dtype=float).reshape(-1)
    if probabilities.size != len(support):
        raise PurePFBoundaryError(
            "Structural cardinality prior length differs from resolved support."
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise PurePFBoundaryError(
            "Structural cardinality prior must contain finite positive mass."
        )
    normalized = validated_probability_distribution(
        probabilities,
        name="resolved explicit cardinality prior",
    )
    return support, tuple(float(value) for value in normalized), "explicit"


class PurePFEstimator(_PFEstimatorCore):
    """Run causal PF updates and report only the resulting PF posterior."""

    planner_belief_sources: tuple[str, ...] = ("pf_posterior",)

    def __init__(
        self,
        *,
        measurement_log_schema_version: int = 2,
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
        super().__init__(random_seed=random_seed, **kwargs)
        if apply_profile_to_config(self.pf_config) != capabilities:
            raise PurePFBoundaryError(
                "PF capabilities changed during estimator initialization."
            )
        self.profile_capabilities = capabilities
        if (
            isinstance(measurement_log_schema_version, bool)
            or not isinstance(measurement_log_schema_version, int)
            or measurement_log_schema_version != 2
        ):
            raise ValueError(
                "PurePFEstimator supports MeasurementLog schema version 2 only."
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
        self.repository_source_snapshot_sha256 = repository_source_snapshot_sha256()
        self.measurement_log_sha256 = str(measurement_log_sha256)

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
        configured_isotopes = sorted({str(isotope) for isotope in self.isotopes})
        atlas_groups: dict[str, dict[str, Any]] = {}
        missing_isotopes: list[str] = []
        for isotope in configured_isotopes:
            filt = self.filters.get(isotope)
            atlas = (
                None
                if filt is None
                else getattr(filt, "_structural_rj_surface_atlas", None)
            )
            chart_geometry = (
                getattr(atlas, "geometry", None) if atlas is not None else None
            )
            if chart_geometry is None:
                missing_isotopes.append(isotope)
                continue
            areas = np.asarray(chart_geometry.areas_m2, dtype=float)
            atlas_hash = surface_chart_geometry_sha256(chart_geometry)
            group = atlas_groups.setdefault(
                atlas_hash,
                {
                    "surface_atlas_contract_sha256": atlas_hash,
                    "chart_count": int(chart_geometry.chart_count),
                    "total_area_m2": float(np.sum(areas, dtype=np.float64)),
                    "geometry_metadata": dict(chart_geometry.geometry_metadata),
                    "isotopes": [],
                },
            )
            group["isotopes"].append(isotope)
        if not atlas_groups:
            atlas_status = "not_initialized"
            atlases_identical: bool | None = None
        elif missing_isotopes:
            atlas_status = "partially_initialized"
            atlases_identical = False if len(atlas_groups) > 1 else None
        else:
            atlas_status = "complete"
            atlases_identical = len(atlas_groups) == 1
        grouped_atlases = sorted(
            atlas_groups.values(),
            key=lambda item: str(item["surface_atlas_contract_sha256"]),
        )
        for group in grouped_atlases:
            group["isotopes"] = sorted(str(value) for value in group["isotopes"])

        birth_weight = float(self.pf_config.structural_rj_birth_probability)
        death_weight = float(self.pf_config.structural_rj_death_probability)
        interior_total = birth_weight + death_weight
        interior_birth = birth_weight / interior_total if interior_total > 0.0 else 0.0
        interior_death = death_weight / interior_total if interior_total > 0.0 else 0.0
        max_cardinality = None if not support else int(support[-1])
        spectrum_model = self._full_spectrum_model()
        full_spectrum_hash = getattr(
            spectrum_model,
            "contract_hash_sha256",
            None,
        )
        manifest_completeness = (
            "complete"
            if atlas_status in {"complete", "not_applicable"}
            else (
                "partial" if atlas_status == "partially_initialized" else "config_only"
            )
        )
        return {
            "schema_version": 1,
            "pure_pf_schema_version": PURE_PF_SCHEMA_VERSION,
            "repository_source_snapshot_sha256": (
                self.repository_source_snapshot_sha256
            ),
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
                "continuous_surface_birth_death_split_merge_rj_mh"
                if variable_cardinality
                else "fixed_cardinality_surface_position_strength_mh"
            ),
            "joint_observation_likelihood": {
                "isotope_order": list(self.joint_isotope_order()),
                "vector_layout": "view_major_then_energy_bin",
                "family": ("geometry_conditioned_joint_full_spectrum_generative"),
                "statistical_covariance_semantics": (
                    "candidate_complete_statistical_poisson_plus_model"
                ),
                "statistical_covariance_usage": "exactly_once",
                "projected_isotope_count_likelihood": False,
                "source_transport_layout": ("particle_view_source_slot_positive_line"),
                "transport_feature_order": list(spectrum_model.transport_feature_order),
                "shared_background_owned_by_generative_model": True,
                "station_assimilation_bridge": (
                    "exact_shared_latent_view_prefix_marginals"
                ),
                "final_prefix_target_equals_joint_station_likelihood": True,
                "full_spectrum_contract_hash_sha256": full_spectrum_hash,
            },
            "cardinality_prior": {
                "support": [int(value) for value in support],
                "probabilities": [float(value) for value in probabilities],
                "configuration_source": prior_source,
                "policy_name": str(self.pf_config.structural_cardinality_prior_policy),
                "truncated_poisson_mean_sources_per_isotope": (
                    float(self.pf_config.structural_cardinality_prior_mean)
                    if prior_source == "truncated_poisson_surface_process"
                    else None
                ),
                "fixed_before_observation": True,
                "applies_independently_per_isotope": True,
            },
            "joint_particle_initialization": {
                "proposal": (
                    "full_support_observation_guided_product_proposal"
                    if bool(self.pf_config.joint_guided_initialization)
                    else "direct_independent_isotope_product_prior_iid"
                ),
                "common_outer_weights": (
                    "exact_product_prior_over_proposal_importance_weights"
                    if bool(self.pf_config.joint_guided_initialization)
                    else "uniform"
                ),
                "isotope_cardinalities_coupled_by_row": False,
                "per_isotope_stratified_marginal_weights_reused": bool(
                    self.pf_config.joint_guided_initialization
                ),
                "external_optimizer": False,
            },
            "strength_prior": {
                "kind": str(self.pf_config.strength_prior_family),
                "minimum_cps_1m": float(self.pf_config.strength_prior_min_cps_1m),
                "maximum_cps_1m": (
                    None
                    if self.pf_config.strength_prior_family == "shifted_gamma"
                    else float(self.pf_config.strength_prior_max_cps_1m)
                ),
                "legacy_proposal_grid_maximum_cps_1m": float(
                    self.pf_config.strength_prior_max_cps_1m
                ),
                "support_maximum_cps_1m": (
                    None
                    if self.pf_config.strength_prior_family == "shifted_gamma"
                    else float(self.pf_config.strength_prior_max_cps_1m)
                ),
                "gamma_shape": (
                    float(self.pf_config.strength_prior_gamma_shape)
                    if self.pf_config.strength_prior_family == "shifted_gamma"
                    else None
                ),
                "gamma_scale_cps_1m": (
                    float(self.pf_config.strength_prior_gamma_scale_cps_1m)
                    if self.pf_config.strength_prior_family == "shifted_gamma"
                    else None
                ),
                "units": "detector_cps_1m",
                "unit_definition": ("expected_pre_dead_time_detector_pulse_rate_at_1m"),
                "used_for_initialization": True,
                "shared_by_initialization_and_state_moves": True,
            },
            "surface_position_prior": {
                "support": "environment_surface",
                "semantics": ("iid_uniform_physical_surface_area_canonical_unordered"),
                "used_for_initialization": True,
                "canonical_order": "surface_chart_id_then_continuous_u_v",
                "canonical_density_factor": "K_factorial",
                "same_chart_sources_allowed": True,
                "pair_interaction_prior": "none_iid_surface_positions",
                "proximity_used_only_in_target_preserving_proposals": True,
                "continuous_uv_support": True,
                "chart_max_edge_m": float(
                    self.pf_config.structural_rj_surface_chart_max_edge_m
                ),
                "chart_tessellation_role": (
                    "coordinates_continuous_max_edge_topology_only"
                ),
                "support_quantization": False,
                "continuous_coordinates_within_each_chart": True,
                "atlas_hash_encoding": (
                    "ordered_complete_chart_geometry_canonical_little_endian_v1"
                ),
                "atlas_status": atlas_status,
                "configured_isotopes": configured_isotopes,
                "missing_isotopes": sorted(missing_isotopes),
                "atlases_identical_across_isotopes": (atlases_identical),
                "atlas_groups": grouped_atlases,
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
                    "joint_state_independent_surface_and_chart_conditional_"
                    "strength_independence_mh"
                ),
                "position_proposal_prior_component_probability": float(
                    self.pf_config.structural_rj_position_proposal_prior_weight
                ),
                "position_proposal_data_component": (
                    "background_whitened_non_target_line_subspace_matched_filter_v1"
                ),
                "position_proposal_alignment_residual": (
                    "observed_full_spectrum_minus_shared_model_background_"
                    "after_fixed_non_target_line_subspace_projection"
                ),
                "position_proposal_alignment_response": (
                    "target_isotope_positive_transport_lines_only_at_chart_"
                    "centers_for_proposal_scoring"
                ),
                "position_proposal_state_dependence": (
                    "observations_target_beta_and_immutable_known_model_only_"
                    "never_current_particle_population"
                ),
                "guided_initialization": {
                    "enabled": bool(self.pf_config.joint_guided_initialization),
                    "proposal_support": (
                        "positive_product_prior_mixture_for_every_isotope"
                    ),
                    "weight_correction": (
                        "exact_product_position_and_strength_prior_over_q"
                    ),
                    "uses_external_optimizer": False,
                },
                "position_proposal_chart_conditional": (
                    "continuous_uniform_unit_square_uv"
                ),
                "position_proposal_full_support": True,
                "position_proposal_fixed_per_structural_sweep": True,
                "position_proposal_reverse_density": (
                    "same_state_independent_mixture_for_all_directions"
                ),
                "position_proposal_target_response": (
                    "direct_continuous_xyz_kernel_without_chart_interpolation"
                ),
                "strength_proposal": (
                    f"{self.pf_config.strength_prior_family}_prior_plus_"
                    "chart_conditional_truncated_normal_mixture"
                ),
                "strength_proposal_prior_component_probability": float(
                    self.pf_config.structural_rj_strength_proposal_prior_weight
                ),
                "strength_proposal_sigma_fraction": float(
                    self.pf_config.structural_rj_strength_proposal_sigma_fraction
                ),
                "strength_proposal_grid_size": int(
                    self.pf_config.structural_rj_strength_proposal_grid_size
                ),
                "proposal_score_cache": {
                    "unit": "isotope_station_chart_strength_grid",
                    "stores_spectra_or_particle_state": False,
                    "maximum_bytes": int(
                        self.pf_config.structural_rj_proposal_score_cache_max_bytes
                    ),
                },
                "local_position_move_attempt_probability": float(
                    self.pf_config.structural_rj_local_position_move_probability
                ),
                "local_position_move_proposal": (
                    "gaussian_tangent_geodesic_via_shared_edge_portals"
                ),
                "local_position_reverse_correction": (
                    "log_source_chart_area_over_destination_chart_area"
                ),
                "local_position_physical_area_jacobian": 1.0,
                "local_position_invalid_trace": (
                    "explicit_self_transition_without_redraw"
                ),
                "local_position_sigma_m": float(
                    self.pf_config.structural_rj_local_position_sigma_m
                ),
                "global_joint_position_strength_move_retained_for_irreducibility": True,
                "strength_move_attempt_probability": float(
                    self.pf_config.structural_rj_strength_move_probability
                ),
                "split_merge_attempt_probability": float(
                    self.pf_config.structural_rj_split_merge_probability
                ),
                "block_independence_attempt_probability": float(
                    self.pf_config.structural_rj_block_independence_probability
                ),
                "block_independence_proposal": (
                    "full_isotope_cardinality_position_strength_"
                    "independence_mh_with_explicit_forward_reverse_density"
                ),
                "split_merge_direction_weights": {
                    "split": float(self.pf_config.structural_rj_split_probability),
                    "merge": float(self.pf_config.structural_rj_merge_probability),
                },
                "split_merge_strength_map": (
                    "strength_transfer_with_exact_total_strength_jacobian"
                ),
                "split_position_proposal": (
                    "two_children_independently_from_parent_local_chart_"
                    "mixture_with_global_surface_support"
                ),
                "split_global_position_probability": float(
                    self.pf_config.structural_rj_split_global_position_probability
                ),
                "merge_pair_proposal": (
                    "exact_same_or_one_portal_surface_distance_weighted_"
                    "ordered_pair_with_uniform_global_support"
                ),
                "merge_position_proposal": (
                    "equal_mixture_of_local_chart_proposals_from_both_"
                    "children_with_exact_reverse_two_child_density"
                ),
                "merge_relocates_combined_source": True,
                "merge_uniform_pair_probability": float(
                    self.pf_config.structural_rj_merge_uniform_pair_probability
                ),
                "merge_distance_sigma_m": float(
                    self.pf_config.structural_rj_merge_distance_sigma_m
                ),
                "split_merge_selection_density_in_mh_ratio": True,
                "structural_sweep_order": (
                    "birth_death_then_split_merge_then_block_independence_"
                    "then_global_position_then_local_position_then_strength"
                ),
                "post_merge_same_sweep_refinement": True,
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
                    "birth_death": {
                        "absolute_jacobian_determinant": 1.0,
                        "log_absolute_jacobian_determinant": 0.0,
                    },
                    "split": {
                        "absolute_jacobian_determinant": "total_strength",
                        "log_absolute_jacobian_determinant": ("log_total_strength"),
                    },
                    "merge": {
                        "absolute_jacobian_determinant": (
                            "1_over_merged_total_strength"
                        ),
                        "log_absolute_jacobian_determinant": (
                            "minus_log_merged_total_strength"
                        ),
                    },
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
                max_cardinality=self.pf_config.cardinality_capacity,
            )
        return result

    def _joint_cardinality_partition(
        self,
    ) -> tuple[
        tuple[str, ...],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        """Return the strict aligned-particle partition by joint K tuple."""
        if not self.filters:
            raise PurePFBoundaryError(
                "Joint cardinality mass requires initialized isotope filters."
            )
        isotope_order = self.joint_isotope_order()
        self._assert_joint_particle_alignment()
        particle_count = len(self.filters[isotope_order[0]].continuous_particles)
        weights = validated_probability_distribution(
            self.filters[isotope_order[0]].continuous_weights,
            name="aligned joint PF particle weights",
        )
        cardinalities = np.column_stack(
            [
                np.asarray(
                    [
                        particle.state.num_sources
                        for particle in self.filters[isotope].continuous_particles
                    ],
                    dtype=np.int64,
                )
                for isotope in isotope_order
            ]
        )
        if cardinalities.shape != (particle_count, len(isotope_order)):
            raise PurePFBoundaryError(
                "Aligned joint PF cardinalities have an invalid shape."
            )
        if np.any(cardinalities < 0) or np.any(
            cardinalities > self.pf_config.cardinality_capacity
        ):
            raise PurePFBoundaryError(
                "Aligned joint PF cardinalities lie outside configured support."
            )
        joint_cardinality_vectors, inverse = np.unique(
            cardinalities,
            axis=0,
            return_inverse=True,
        )
        joint_mass = validated_probability_distribution(
            np.bincount(
                inverse,
                weights=weights,
                minlength=joint_cardinality_vectors.shape[0],
            ),
            name="joint cardinality posterior mass",
        )
        return (
            isotope_order,
            weights,
            cardinalities,
            joint_cardinality_vectors,
            np.asarray(inverse, dtype=np.int64),
            joint_mass,
        )

    def posterior_joint_cardinality_distribution(
        self,
    ) -> dict[tuple[int, ...], float]:
        """Return posterior mass over joint K tuples in ``joint_isotope_order``."""
        if not self.filters:
            return {}
        (
            _isotope_order,
            _weights,
            _cardinalities,
            joint_cardinality_vectors,
            _inverse,
            joint_mass,
        ) = self._joint_cardinality_partition()
        return {
            tuple(int(value) for value in vector): float(mass)
            for vector, mass in zip(
                joint_cardinality_vectors,
                joint_mass,
                strict=True,
            )
        }

    def _joint_map_cardinality_selection(
        self,
    ) -> tuple[
        tuple[str, ...],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
        float,
    ]:
        """Return the deterministic joint-MAP stratum and its posterior mass."""
        (
            isotope_order,
            weights,
            _cardinalities,
            joint_cardinality_vectors,
            inverse,
            joint_mass,
        ) = self._joint_cardinality_partition()
        maximum_mass = float(np.max(joint_mass))
        tied_vectors = np.flatnonzero(
            np.isclose(
                joint_mass,
                maximum_mass,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        joint_vector_index = int(tied_vectors[0])
        selected_indices = np.flatnonzero(inverse == joint_vector_index).astype(
            np.int64, copy=False
        )
        return (
            isotope_order,
            weights,
            joint_cardinality_vectors,
            selected_indices,
            maximum_mass,
        )

    def _posterior_reporting_particle_indices(
        self,
    ) -> NDArray[np.int64] | None:
        """Keep uncertainty summaries in the canonical joint-MAP stratum."""
        if not self.filters:
            return None
        return self._joint_map_cardinality_selection()[3].copy()

    def posterior_point_estimate(self) -> dict[str, PFPointEstimate]:
        """Return one coherent joint-particle configuration and uncertainty."""
        cached = self._cached_posterior_point_estimate()
        if cached is not None:
            return cached
        if not self.filters:
            return self._store_posterior_point_estimate({})
        (
            isotope_order,
            weights,
            joint_cardinality_vectors,
            selected_indices,
            maximum_mass,
        ) = self._joint_map_cardinality_selection()
        selected_cardinality_vector = np.asarray(
            [
                self.filters[isotope]
                .continuous_particles[int(selected_indices[0])]
                .state.num_sources
                for isotope in isotope_order
            ],
            dtype=np.int64,
        )
        matching_vectors = np.flatnonzero(
            np.all(
                joint_cardinality_vectors == selected_cardinality_vector[None, :],
                axis=1,
            )
        )
        if matching_vectors.size != 1:
            raise PurePFBoundaryError(
                "Joint-MAP cardinality stratum has no unique support vector."
            )
        joint_vector_index = int(matching_vectors[0])
        selected_weights = validated_probability_distribution(
            weights[selected_indices] / maximum_mass,
            name="selected joint-cardinality conditional weights",
        )
        joint_configuration_distance = np.zeros(
            selected_indices.size,
            dtype=np.float64,
        )
        for isotope_index, isotope in enumerate(isotope_order):
            cardinality = int(
                joint_cardinality_vectors[
                    joint_vector_index,
                    isotope_index,
                ]
            )
            if cardinality == 0:
                continue
            filt = self.filters[isotope]
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            if atlas is None:
                raise PurePFBoundaryError(
                    "Joint posterior reporting requires the shared continuous "
                    "surface atlas."
                )
            selected_states = [
                filt.continuous_particles[int(index)].state
                for index in selected_indices
            ]
            positions = np.stack(
                [
                    np.asarray(
                        filt.continuous_state_positions(state)[:cardinality],
                        dtype=np.float64,
                    )
                    for state in selected_states
                ],
                axis=0,
            )
            strengths = np.stack(
                [
                    np.asarray(
                        state.strengths[:cardinality],
                        dtype=np.float64,
                    )
                    for state in selected_states
                ],
                axis=0,
            )
            chart_ids = np.stack(
                [
                    np.asarray(
                        state.surface_chart_ids[:cardinality],
                        dtype=np.int64,
                    )
                    for state in selected_states
                ],
                axis=0,
            )
            surface_uv = np.stack(
                [
                    np.asarray(
                        state.surface_uv[:cardinality],
                        dtype=np.float64,
                    )
                    for state in selected_states
                ],
                axis=0,
            )
            (
                _,
                _,
                aligned_chart_ids,
                aligned_surface_uv,
            ) = align_surface_modes_batched(
                positions,
                strengths,
                chart_ids,
                surface_uv,
                selected_weights,
                atlas.surface_coordinate_path_distance_upper_bound_m,
            )
            joint_configuration_distance += (
                surface_configuration_medoid_distance_batched(
                    aligned_chart_ids,
                    aligned_surface_uv,
                    selected_weights,
                    atlas.surface_coordinate_path_distance_upper_bound_m,
                )
            )
        minimum_distance = float(np.min(joint_configuration_distance))
        tied_rows = np.flatnonzero(
            np.isclose(
                joint_configuration_distance,
                minimum_distance,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        representative_local_index = int(
            tied_rows[np.argmax(selected_weights[tied_rows])]
        )
        representative_particle_index = int(
            selected_indices[representative_local_index]
        )
        result: dict[str, PFPointEstimate] = {}
        for isotope in isotope_order:
            filt = self.filters[isotope]
            states = [particle.state for particle in filt.continuous_particles]
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            filt.validate_continuous_surface_states()
            if atlas is None:
                raise PurePFBoundaryError(
                    "Pure-PF reporting requires a continuous surface atlas."
                )
            result[str(isotope)] = posterior_point_estimate_from_states(
                states,
                weights,
                max_cardinality=self.pf_config.cardinality_capacity,
                positions_by_state=[
                    filt.continuous_state_positions(state) for state in states
                ],
                surface_chart_ids_by_state=[
                    np.asarray(state.surface_chart_ids, dtype=np.int64)
                    for state in states
                ],
                surface_uv_by_state=[
                    np.asarray(state.surface_uv, dtype=np.float64) for state in states
                ],
                surface_coordinate_path_distance=(
                    atlas.surface_coordinate_path_distance_upper_bound_m
                ),
                selected_particle_indices=selected_indices,
                representative_particle_index=(representative_particle_index),
                selected_stratum_mass=maximum_mass,
            )
        return self._store_posterior_point_estimate(result)

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
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project the PF posterior into the historical array result format."""
        return self._project_posterior_point_estimates(self.posterior_point_estimate())

    def estimate_all(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the PF posterior projection for visualization."""
        return self.estimates()

    def serialized_state(self) -> bytes:
        """Return canonical bytes for causality and determinism tests."""
        if self.filters:
            self._assert_joint_particle_alignment()
        isotope_payload: dict[str, Any] = {}
        for isotope, filt in sorted(self.filters.items()):
            particles: list[dict[str, Any]] = []
            for particle in filt.continuous_particles:
                state = particle.state
                particles.append(
                    {
                        "log_weight": float(particle.log_weight),
                        "joint_row_identity": (
                            None
                            if particle.joint_row_identity is None
                            else particle.joint_row_identity.to_dict()
                        ),
                        "num_sources": int(state.num_sources),
                        "strengths": np.asarray(state.strengths, dtype=float),
                        "surface_chart_ids": np.asarray(
                            state.surface_chart_ids,
                            dtype=np.int64,
                        ),
                        "surface_uv": np.asarray(
                            state.surface_uv,
                            dtype=np.float64,
                        ),
                    }
                )
            isotope_payload[str(isotope)] = particles
        measurement_history = [
            {
                "spectrum_counts_b": measurement.spectrum_counts_b,
                "pose_idx": int(measurement.pose_idx),
                "fe_index": measurement.fe_index,
                "pb_index": measurement.pb_index,
                "detector_position_xyz_m": measurement.detector_position_xyz_m,
                "live_time_s": float(measurement.live_time_s),
                "station_sequence_id": measurement.station_sequence_id,
                "station_view_index": measurement.station_view_index,
                "generative_contract_hash_sha256": (
                    measurement.generative_contract_hash_sha256
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
                "joint_row_identity_contract": {
                    "schema_version": 1,
                    "root_sha256": self._joint_row_identity_root_sha256,
                    "generation": self._joint_row_generation,
                    "cumulative_lineage_ids": (
                        None
                        if self._joint_cumulative_lineage_ids is None
                        else np.asarray(
                            self._joint_cumulative_lineage_ids,
                            dtype=np.int64,
                        )
                    ),
                },
                "rng_provenance": self.rng_provenance,
                "rng_states": {
                    "joint": self._joint_random_generator.bit_generator.state,
                    "isotope_conditionals": {
                        str(isotope): filt._random_generator.bit_generator.state
                        for isotope, filt in sorted(self.filters.items())
                    },
                },
                "full_spectrum_generative_contract": {
                    "contract_hash_sha256": (
                        self._full_spectrum_model().contract_hash_sha256
                    ),
                    "energy_axis_keV": np.asarray(
                        self._full_spectrum_model().energy_axis_keV,
                        dtype=np.float64,
                    ),
                    "transport_feature_order": list(
                        self._full_spectrum_model().transport_feature_order
                    ),
                },
                "isotopes": isotope_payload,
            }
        )


__all__ = [
    "PurePFBoundaryError",
    "PurePFEstimator",
    "RotatingShieldPFConfig",
]
