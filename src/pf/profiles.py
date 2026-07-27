"""Define the single supported pure particle-filter runtime profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import math
from typing import Any, Mapping


PURE_PF_SCHEMA_VERSION = 1


class EstimatorProfile(StrEnum):
    """Name the only supported scientific estimator profile."""

    PF_STRICT = "pf_strict"


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Declare the positive capabilities of the pure PF runtime."""

    sequential_updates_only: bool = True
    posterior_reporting_only: bool = True
    surface_constrained_source_prior: bool = True
    likelihood_consistent_structural_evidence: bool = True

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-safe capability mapping."""
        return {str(key): bool(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class StructuralTransitionProvenance:
    """Describe the statistical semantics of PF structural moves."""

    posterior_semantics: str
    structural_kernel_family: str
    structural_moves_enabled: bool
    variable_cardinality: bool
    birth_death_moves_enabled: bool
    within_cardinality_moves_enabled: bool
    within_cardinality_kernel_exact_mh: bool
    structural_kernel_target_preserving: bool
    structural_kernel_exact_rj: bool
    reversible_jump_mcmc_used: bool
    structural_evidence_uses_pf_likelihood: bool

    def to_dict(self) -> dict[str, bool | str]:
        """Return a JSON-safe structural-transition mapping."""
        return {
            str(key): value if isinstance(value, str) else bool(value)
            for key, value in asdict(self).items()
        }


_PURE_CAPABILITIES = EstimatorCapabilities()
_PURE_PF_RUNTIME_KEYS = frozenset(
    {
        "pf_buildup",
        "pf_count_likelihood",
        "pf_detector_aperture_radius_m",
        "pf_detector_aperture_samples",
        "pf_detector_aperture_sampling",
        "pf_detector_count_radius_m",
        "pf_line_resolved_shield_attenuation",
        "pf_max_sources",
        "pf_obstacle_attenuation",
        "pf_obstacle_material",
        "pf_obstacle_mu_by_isotope",
        "pf_obstacle_source_extent_radius_m",
        "pf_obstacle_source_extent_samples",
        "pf_plot_save_every",
        "pf_random_seed",
        "pf_shield_contrast_likelihood",
        "pf_shield_view_ratio_likelihood",
        "pf_source_extent_radius_m",
        "pf_source_extent_samples",
        "pf_strength_prior_max_cps_1m",
        "pf_strength_prior_min_cps_1m",
        "pf_transport_response_model",
        "pf_transport_response_model_path",
        "pf_visual_estimate_cross_size_m",
        "pf_visual_estimate_cross_width_m",
        "pf_visual_estimate_radius_m",
        "pf_visual_max_particles_per_isotope",
        "pf_visual_min_weight_fraction",
        "pf_visual_particle_radius_m",
        "pf_visualization_enabled",
    }
)
_PURE_PF_RUNTIME_MAPPING_KEYS = frozenset(
    {
        "dss_pp",
        "pf_buildup",
        "pf_count_likelihood",
        "pf_obstacle_mu_by_isotope",
        "pf_shield_contrast_likelihood",
        "pf_shield_view_ratio_likelihood",
        "pf_transport_response_model",
        "remaining_measurement_estimate",
    }
)
_PURE_PF_STRUCTURAL_RUNTIME_KEYS = frozenset(
    {
        "structural_cardinality_prior_probs",
        "structural_rj_birth_probability",
        "structural_rj_death_probability",
        "structural_rj_local_position_move_probability",
        "structural_rj_move_probability",
        "structural_rj_patch_spacing_m",
        "structural_rj_position_move_probability",
        "structural_rj_strength_move_probability",
    }
)
_DSS_PP_RUNTIME_KEYS = frozenset(
    {
        "augment_candidates",
        "beam_width",
        "bearing_diversity_weight",
        "candidate_preselect_enable",
        "candidate_preselect_min",
        "candidate_preselect_multiplier",
        "correlation_reduction_weight",
        "count_balance_weight",
        "count_utility_saturation_counts",
        "count_utility_weight",
        "count_variance_floor",
        "coverage_floor_quantile",
        "coverage_floor_weight",
        "coverage_grid_max_cells",
        "coverage_radius_m",
        "coverage_weight",
        "detector_aperture_samples",
        "diagnostic_ranked_node_limit",
        "differential_weight",
        "distance_weight",
        "dose_weight",
        "eig_candidate_limit",
        "eig_weight",
        "elevation_angle_threshold_deg",
        "elevation_condition_weight",
        "elevation_pair_xy_scale_m",
        "elevation_pair_z_scale_m",
        "elevation_signature_weight",
        "enforce_min_observation",
        "environment_contrast_threshold",
        "environment_signature_score_clip",
        "environment_signature_weight",
        "frontier_weight",
        "horizon",
        "isotope_balance_weight",
        "local_orbit_sigma_m",
        "local_orbit_weight",
        "max_augmented_candidates",
        "max_modes_per_isotope",
        "max_programs",
        "min_station_separation_m",
        "mode_cluster_radius_m",
        "observation_weight",
        "occlusion_boundary_step_m",
        "occlusion_boundary_weight",
        "one_step_guard_enable",
        "one_step_guard_score_abs_margin",
        "one_step_guard_score_rel_margin",
        "one_step_guard_use_gpu",
        "planning_method",
        "planning_particles",
        "program_eval_workers",
        "program_length",
        "remaining_budget_guidance",
        "remaining_budget_urgency_stations",
        "remaining_route_backtrack_weight",
        "remaining_route_coverage_weight",
        "remaining_route_distance_weight",
        "remaining_route_frontier_weight",
        "remaining_route_revisit_weight",
        "remaining_route_turn_weight",
        "remaining_route_weight",
        "revisit_penalty_weight",
        "rotation_weight",
        "signature_std_min_counts",
        "signature_weight",
        "station_condition_coherence_weight",
        "station_condition_inverse_condition_weight",
        "station_condition_min_singular_weight",
        "station_condition_ridge",
        "station_condition_weight",
        "temporal_cover_beam_width",
        "temporal_cover_programs",
        "temporal_cover_weight",
        "temporal_decorrelation_weight",
        "temporal_logdet_ridge",
        "temporal_logdet_weight",
        "temporal_pair_contrast_threshold",
        "temporal_separation_weight",
        "time_weight",
        "turn_smoothness_weight",
        "vertical_environment_signature_weight",
    }
)
_REMAINING_MEASUREMENT_RUNTIME_KEYS = frozenset(
    {
        "cardinality_weight",
        "count_variance_floor",
        "dss_information_gain_weight",
        "enabled",
        "eta_default",
        "eta_max",
        "eta_min",
        "gain_epsilon",
        "max_modes_per_isotope",
        "max_particles",
        "max_reported_stations",
        "mode_cluster_radius_m",
        "pairwise_separation_threshold",
        "planning_method",
        "range_scale",
        "separation_weight",
        "stop_budget",
        "target_cardinality_confidence",
        "target_position_spread_m",
        "target_strength_cv",
        "uncertainty_weight",
    }
)
_PURE_PF_NESTED_KEYS = {
    "dss_pp": _DSS_PP_RUNTIME_KEYS,
    "pf_buildup": frozenset({"fe_coeff", "pb_coeff", "obstacle_coeff"}),
    "pf_count_likelihood": frozenset(
        {
            "count_likelihood_df",
            "count_likelihood_model",
            "low_count_abs_sigma",
            "low_count_transition_counts",
            "observation_count_variance_semantics",
            "spectrum_count_abs_sigma",
            "spectrum_count_rel_sigma",
            "station_view_correlated_spectrum_fraction",
            "station_view_covariance_enable",
            "transport_model_abs_sigma",
            "transport_model_rel_sigma",
        }
    ),
    "pf_shield_contrast_likelihood": frozenset(
        {
            "df",
            "enabled",
            "log_sigma_ceiling",
            "log_sigma_floor",
            "min_count",
            "min_views",
            "weight",
        }
    ),
    "pf_shield_view_ratio_likelihood": frozenset(
        {
            "concentration",
            "enabled",
            "min_total_count",
            "min_views",
            "weight",
        }
    ),
    "remaining_measurement_estimate": _REMAINING_MEASUREMENT_RUNTIME_KEYS,
}
_TRANSPORT_MODEL_KEYS = frozenset(
    {"by_isotope", "enabled", "feature_semantics", "model"}
)
_TRANSPORT_ISOTOPE_KEYS = frozenset(
    {
        "max_log_scale",
        "min_log_scale",
        "num_fit_records",
        "scale",
        "scale_by_pair",
        "tau_coefficients",
        "tau_feature_caps",
    }
)
_TRANSPORT_TAU_KEYS = frozenset(
    {
        "distance",
        "distance_fe",
        "distance_obstacle",
        "distance_pb",
        "distance_shield",
        "fe",
        "fe_obstacle",
        "fe_pb",
        "fe_squared",
        "obstacle",
        "obstacle_squared",
        "pb",
        "pb_obstacle",
        "pb_squared",
        "shield",
        "shield_obstacle",
        "shield_squared",
    }
)
_TRANSPORT_CAP_KEYS = frozenset(
    {
        "distance_fe",
        "distance_obstacle",
        "distance_pb",
        "distance_shield",
        "fe",
        "obstacle",
        "pb",
        "shield",
    }
)
_RETIRED_RUNTIME_KEYS = frozenset(
    {
        "adapt_cooldown_steps",
        "adaptive_cardinality_condition_max",
        "adaptive_cardinality_min_bic_margin",
        "adaptive_cardinality_min_candidate_count",
        "birth_enable",
        "birth_enabled",
        "converge_enable",
        "converge_ess_ratio_high",
        "converge_ll_improve_eps",
        "converge_map_move_eps_m",
        "converge_min_stations",
        "converge_min_steps",
        "converge_require_all",
        "converge_window",
        "ess_high",
        "ess_low",
        "fixed_cardinality_no_structural_moves",
        "height_partner_reuse_shield_program",
        "init_grid_spacing_m",
        "init_num_sources_max",
        "init_num_sources_min",
        "init_strength_log_mean",
        "init_strength_log_sigma",
        "init_strength_max",
        "init_strength_min",
        "init_strength_prior",
        "joint_observation_update",
        "joint_station_update",
        "delayed_resample_update",
        "max_particles",
        "min_particles",
        "min_strength",
        "orientation_selection_mode",
        "position_min",
        "refit_after_moves",
        "short_time_s",
        "source_position_prior",
        "source_prior_mode",
        "source_surface_prior",
        "response_poisson_global_diagnostic_variance_enable",
        "source_detector_exclusion_m",
        "spectrum_likelihood_bin_chunk",
    }
)
_RETIRED_RUNTIME_PREFIXES = (
    "adaptive_strength_prior",
    "all_history_dictionary_proposal_",
    "batch_fit_",
    "birth_",
    "candidate_verification_",
    "cardinality_preserving_",
    "conditional_strength_",
    "converge_cluster_",
    "death_",
    "deferred_resample_",
    "delayed_resample_",
    "display_pruned_",
    "final_absent_",
    "global_surface_",
    "high_strength_split_",
    "high_surface_",
    "init_grid_",
    "init_num_sources_",
    "maximum_likelihood_",
    "merge_",
    "mission_stop_birth_residual_",
    "mission_stop_min_convergence_",
    "mission_stop_report_simple_",
    "mission_stop_require_model_order_",
    "mission_stop_require_no_unresolved_",
    "mission_stop_require_pf_convergence_",
    "mission_stop_require_quiet_birth_",
    "mission_stop_soft_",
    "mission_stop_unresolved_",
    "mle_",
    "mode_preserving_",
    "online_absent_",
    "precision_diagnostic_birth_candidate_",
    "pseudo_source_",
    "raw_count_residual_",
    "recovery_",
    "report_best_so_far_",
    "report_cluster_",
    "report_exclude_unverified_",
    "report_mle_",
    "report_model_order_",
    "report_refit_",
    "report_strength_",
    "report_surface_",
    "residual_birth_",
    "residual_decomposition_",
    "roughening_",
    "runtime_report_rescue_",
    "soft_extension_",
    "source_prune_",
    "source_strength_",
    "sparse_poisson_",
    "spectrum_likelihood_",
    "split_",
    "strength_refit_",
    "structural_proposal_",
    "structural_trial_",
    "structural_update_",
    "surface_rejuvenation_",
    "surface_map_",
    "verification_",
    "weak_source_prune_",
)


def _validate_allowed_keys(
    name: str,
    payload: Mapping[str, Any],
    allowed: frozenset[str],
) -> None:
    """Reject unknown keys from one versioned pure-PF object."""
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported {name} settings: " + ", ".join(unknown)
        )


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    """Return a mapping value or fail closed with a schema error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _validate_transport_response_model(payload: Mapping[str, Any]) -> None:
    """Validate the positive schema of an inline transport-response model."""
    _validate_allowed_keys(
        "pf_transport_response_model",
        payload,
        _TRANSPORT_MODEL_KEYS,
    )
    raw_by_isotope = payload.get("by_isotope")
    if raw_by_isotope is None:
        return
    by_isotope = _require_mapping(
        "pf_transport_response_model.by_isotope",
        raw_by_isotope,
    )
    for isotope, raw_isotope_payload in by_isotope.items():
        isotope_payload = _require_mapping(
            f"pf_transport_response_model.by_isotope.{isotope}",
            raw_isotope_payload,
        )
        _validate_allowed_keys(
            f"pf_transport_response_model.by_isotope.{isotope}",
            isotope_payload,
            _TRANSPORT_ISOTOPE_KEYS,
        )
        for field_name, allowed in (
            ("tau_coefficients", _TRANSPORT_TAU_KEYS),
            ("tau_feature_caps", _TRANSPORT_CAP_KEYS),
        ):
            raw_nested = isotope_payload.get(field_name)
            if raw_nested is None:
                continue
            nested = _require_mapping(
                "pf_transport_response_model.by_isotope."
                f"{isotope}.{field_name}",
                raw_nested,
            )
            _validate_allowed_keys(
                "pf_transport_response_model.by_isotope."
                f"{isotope}.{field_name}",
                nested,
                allowed,
            )
        raw_scale_by_pair = isotope_payload.get("scale_by_pair")
        if raw_scale_by_pair is not None:
            _require_mapping(
                "pf_transport_response_model.by_isotope."
                f"{isotope}.scale_by_pair",
                raw_scale_by_pair,
            )


def _validate_nested_runtime_settings(
    runtime_config: Mapping[str, Any],
) -> None:
    """Validate every versioned nested pure-PF configuration block."""
    for block_name, allowed in _PURE_PF_NESTED_KEYS.items():
        if block_name not in runtime_config:
            continue
        block = _require_mapping(block_name, runtime_config[block_name])
        _validate_allowed_keys(block_name, block, allowed)
    if "pf_obstacle_mu_by_isotope" in runtime_config:
        obstacle_mu = _require_mapping(
            "pf_obstacle_mu_by_isotope",
            runtime_config["pf_obstacle_mu_by_isotope"],
        )
        for isotope, value in obstacle_mu.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    "pf_obstacle_mu_by_isotope values must be finite "
                    f"nonnegative numbers; invalid value for {isotope!s}."
                )
    if "pf_transport_response_model" in runtime_config:
        transport = _require_mapping(
            "pf_transport_response_model",
            runtime_config["pf_transport_response_model"],
        )
        _validate_transport_response_model(transport)


def _validate_structural_runtime_values(
    runtime_config: Mapping[str, Any],
) -> None:
    """Validate pure-PF cardinality fields before runtime construction."""
    if "variable_cardinality" in runtime_config and not isinstance(
        runtime_config["variable_cardinality"],
        bool,
    ):
        raise ValueError("variable_cardinality must be a boolean.")
    if "pf_max_sources" in runtime_config:
        value = runtime_config["pf_max_sources"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("pf_max_sources must be a positive integer.")
    if "init_num_sources" in runtime_config:
        value = runtime_config["init_num_sources"]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(
                "init_num_sources must contain exactly two integer bounds."
            )


def resolve_estimator_profile(
    value: EstimatorProfile | str,
) -> tuple[EstimatorProfile, EstimatorCapabilities]:
    """Resolve the single pure-PF profile without compatibility aliases."""
    if isinstance(value, EstimatorProfile):
        profile = value
    elif value == EstimatorProfile.PF_STRICT.value:
        profile = EstimatorProfile.PF_STRICT
    else:
        raise ValueError(
            f"Unsupported estimator profile {value!r}; "
            "only 'pf_strict' is available."
        )
    return profile, _PURE_CAPABILITIES


def resolve_structural_transition_provenance(
    config: Any,
    *,
    capabilities: EstimatorCapabilities | None = None,
) -> StructuralTransitionProvenance:
    """Resolve provenance for the configured PF structural kernel."""
    del capabilities
    variable_cardinality = bool(getattr(config, "variable_cardinality", False))

    if variable_cardinality:
        kernel_family = "area_weighted_surface_birth_death_rj_mh"
        posterior_semantics = (
            "sequential_particle_filter_with_target_preserving_rj_mh_rejuvenation"
        )
        exact_rj = True
        reversible_jump_used = True
    else:
        kernel_family = "fixed_cardinality_surface_position_strength_mh"
        posterior_semantics = (
            "fixed_cardinality_sequential_particle_filter_with_"
            "target_preserving_mh_rejuvenation"
        )
        exact_rj = False
        reversible_jump_used = False

    return StructuralTransitionProvenance(
        posterior_semantics=posterior_semantics,
        structural_kernel_family=kernel_family,
        structural_moves_enabled=True,
        variable_cardinality=variable_cardinality,
        birth_death_moves_enabled=variable_cardinality,
        within_cardinality_moves_enabled=True,
        within_cardinality_kernel_exact_mh=True,
        structural_kernel_target_preserving=True,
        structural_kernel_exact_rj=exact_rj,
        reversible_jump_mcmc_used=reversible_jump_used,
        structural_evidence_uses_pf_likelihood=True,
    )


def apply_profile_to_config(config: Any) -> EstimatorCapabilities:
    """Validate and stamp the single supported estimator profile."""
    profile, capabilities = resolve_estimator_profile(
        getattr(config, "estimator_profile", EstimatorProfile.PF_STRICT.value)
    )
    config.estimator_profile = profile.value
    return capabilities


def enforce_pure_runtime_settings(
    runtime_config: Mapping[str, Any],
    *,
    profile: EstimatorProfile | str | None = None,
) -> dict[str, Any]:
    """Validate the pure-PF schema marker and strict estimator profile."""
    schema_version = runtime_config.get("pure_pf_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PURE_PF_SCHEMA_VERSION
    ):
        raise ValueError(
            "Runtime configuration requires pure_pf_schema_version=1."
        )
    retired_keys = sorted(
        str(key)
        for key in runtime_config
        if str(key) in _RETIRED_RUNTIME_KEYS
        or any(
            str(key).startswith(prefix)
            for prefix in _RETIRED_RUNTIME_PREFIXES
        )
    )
    if retired_keys:
        raise ValueError(
            "Retired particle-filter settings are not supported: "
            + ", ".join(retired_keys)
        )
    unknown_pf_keys = sorted(
        str(key)
        for key in runtime_config
        if str(key).startswith("pf_") and str(key) not in _PURE_PF_RUNTIME_KEYS
    )
    if unknown_pf_keys:
        raise ValueError(
            "Unsupported pure-PF runtime settings: "
            + ", ".join(unknown_pf_keys)
        )
    unknown_structural_keys = sorted(
        str(key)
        for key in runtime_config
        if (
            str(key).startswith("structural_rj_")
            or str(key).startswith("structural_cardinality_")
        )
        and str(key) not in _PURE_PF_STRUCTURAL_RUNTIME_KEYS
    )
    if unknown_structural_keys:
        raise ValueError(
            "Unsupported pure-PF structural settings: "
            + ", ".join(unknown_structural_keys)
        )
    invalid_mapping_keys = sorted(
        key
        for key in _PURE_PF_RUNTIME_MAPPING_KEYS
        if key in runtime_config and not isinstance(runtime_config[key], Mapping)
    )
    if invalid_mapping_keys:
        raise ValueError(
            "Pure-PF runtime settings must be objects: "
            + ", ".join(invalid_mapping_keys)
        )
    _validate_nested_runtime_settings(runtime_config)
    _validate_structural_runtime_values(runtime_config)
    configured_profile, _capabilities = resolve_estimator_profile(
        runtime_config.get("estimator_profile")
    )
    if profile is not None:
        requested_profile, _ = resolve_estimator_profile(profile)
        if requested_profile is not configured_profile:
            raise ValueError(
                "Requested estimator profile differs from the runtime schema."
            )
    result = dict(runtime_config)
    result["estimator_profile"] = configured_profile.value
    return result


__all__ = [
    "PURE_PF_SCHEMA_VERSION",
    "EstimatorCapabilities",
    "EstimatorProfile",
    "StructuralTransitionProvenance",
    "apply_profile_to_config",
    "enforce_pure_runtime_settings",
    "resolve_estimator_profile",
    "resolve_structural_transition_provenance",
]
