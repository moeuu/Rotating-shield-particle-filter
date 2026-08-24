"""Define the single supported pure particle-filter runtime profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import math
from typing import Any, Mapping

from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
    TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    validate_cardinality_prior_policy,
)


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
        "pf_detector_aperture_radius_m",
        "pf_detector_aperture_samples",
        "pf_detector_aperture_sampling",
        "pf_detector_count_radius_m",
        "pf_detected_isotopes_only",
        "pf_line_resolved_shield_attenuation",
        "pf_max_sources",
        "pf_hard_max_sources",
        "pf_obstacle_attenuation",
        "pf_obstacle_material",
        "pf_obstacle_mu_by_isotope",
        "pf_obstacle_source_extent_radius_m",
        "pf_obstacle_source_extent_samples",
        "pf_plot_save_every",
        "pf_random_seed",
        "pf_source_extent_radius_m",
        "pf_source_extent_samples",
        "pf_strength_prior_max_cps_1m",
        "pf_strength_prior_min_cps_1m",
        "pf_strength_prior_family",
        "pf_strength_prior_gamma_shape",
        "pf_strength_prior_gamma_scale_cps_1m",
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
        "adaptive_stop",
        "dss_pp",
        "pf_buildup",
        "pf_obstacle_mu_by_isotope",
    }
)
_PURE_PF_STRUCTURAL_RUNTIME_KEYS = frozenset(
    {
        "structural_cardinality_prior_probs",
        "structural_cardinality_prior_mean",
        "structural_cardinality_prior_policy",
        "structural_cardinality_tail_ratio",
        "structural_rj_birth_probability",
        "structural_rj_block_independence_probability",
        "structural_rj_death_probability",
        "structural_rj_local_position_move_probability",
        "structural_rj_local_position_sigma_m",
        "structural_rj_merge_distance_sigma_m",
        "structural_rj_merge_probability",
        "structural_rj_merge_response_sigma",
        "structural_rj_merge_uniform_pair_probability",
        "structural_rj_move_probability",
        "structural_rj_multi_component_max_group_size",
        "structural_rj_multi_component_probability",
        "structural_rj_position_move_probability",
        "structural_rj_position_proposal_prior_weight",
        "structural_rj_proposal_chart_batch_size",
        "structural_rj_proposal_score_cache_max_bytes",
        "structural_rj_strength_proposal_grid_size",
        "structural_rj_strength_proposal_prior_weight",
        "structural_rj_strength_proposal_sigma_fraction",
        "structural_rj_split_merge_probability",
        "structural_rj_split_global_position_probability",
        "structural_rj_split_probability",
        "structural_rj_strength_move_probability",
        "structural_rj_surface_chart_max_edge_m",
        "joint_strength_block_probability",
        "joint_strength_block_log_sigma",
        "joint_strength_block_batch_size",
        "joint_cross_isotope_state_block_probability",
    }
)
_DSS_PP_RUNTIME_KEYS = frozenset(
    {
        "augment_candidates",
        "bearing_diversity_weight",
        "coverage_floor_quantile",
        "coverage_floor_weight",
        "coverage_surface_quadrature_max_points",
        "coverage_surface_max_hausdorff_m",
        "coverage_radius_m",
        "coverage_weight",
        "detector_aperture_samples",
        "diagnostic_ranked_node_limit",
        "distance_weight",
        "eig_weight",
        "conditional_greedy_one_swap",
        "exact_eig_action_limit",
        "exact_eig_coverage_reserve",
        "exact_eig_memory_budget_bytes",
        "exact_eig_pose_max",
        "exact_eig_pose_min",
        "exact_eig_pose_limit",
        "exact_eig_pose_step",
        "exact_eig_program_diversity_reserve",
        "elevation_angle_threshold_deg",
        "elevation_condition_weight",
        "elevation_pair_xy_scale_m",
        "elevation_pair_z_scale_m",
        "frontier_weight",
        "horizontal_time_weight",
        "local_orbit_sigma_m",
        "local_orbit_weight",
        "legacy_program_guard_enabled",
        "mast_vertical_time_weight",
        "max_augmented_candidates",
        "max_modes_per_isotope",
        "max_programs",
        "measurement_time_weight",
        "min_station_separation_m",
        "mode_cluster_radius_m",
        "planning_method",
        "planning_particles",
        "program_length",
        "proxy_eig_samples",
        "proxy_boundary_confidence",
        "proxy_memory_budget_bytes",
        "proxy_planning_particles",
        "proxy_stability_refinement_pool",
        "proxy_stability_replicates",
        "proxy_top_k_jaccard_min",
        "revisit_penalty_weight",
        "rotation_weight",
        "settling_time_weight",
        "shield_program_search_policy",
        "time_weight",
        "turn_smoothness_weight",
    }
)
_PURE_PF_NESTED_KEYS = {
    "adaptive_stop": frozenset(
        {
            "assessment_start_station",
            "enabled",
            "innovation_confidence",
            "maximum_surface_path_radius_95_m",
            "maximum_upper_cardinality_mass",
            "minimum_joint_map_cardinality_probability",
            "required_consecutive_stations",
        }
    ),
    "dss_pp": _DSS_PP_RUNTIME_KEYS,
    "pf_buildup": frozenset({"fe_coeff", "pb_coeff", "obstacle_coeff"}),
}
_RETIRED_RUNTIME_KEYS = frozenset(
    {
        "adapt_cooldown_steps",
        "adaptive_allow_low_signal_stop",
        "adaptive_cardinality_min_live_s",
        "adaptive_cardinality_min_bic_margin",
        "adaptive_cardinality_condition_max",
        "adaptive_cardinality_min_candidate_count",
        "adaptive_low_signal_count_fraction",
        "adaptive_low_signal_min_live_s",
        "adaptive_low_signal_projected_live_factor",
        "adaptive_low_signal_upper_sigma",
        "adaptive_ready_allow_informative_low",
        "adaptive_mission_stop",
        "apply_incident_gamma_detector_response",
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
        "converge_cardinality_min_probability",
        "converge_cardinality_var_max",
        "converge_innovation_confidence",
        "converge_max_cardinality_boundary_mass",
        "converge_min_ess_ratio",
        "continuous_surface_chart_max_edge_m",
        "credible_surface_radius_threshold_m",
        "coverage_grid_max_cells",
        "response_backscatter_fraction",
        "response_continuum_to_peak",
        "response_efficiency_model",
        "calibration_count_method",
        "count_likelihood_model",
        "detector_height_sampling_mode",
        "ess_high",
        "ess_low",
        "fixed_cardinality_no_structural_moves",
        "height_partner_reuse_shield_program",
        "history_estimate_interval",
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
        "measurement_pose_clearance_enabled",
        "orientation_selection_mode",
        "path_planner",
        "pf_count_likelihood",
        "pose_selection_workers",
        "position_min",
        "python_worker_count",
        "refit_after_moves",
        "remaining_measurement_estimate",
        "short_time_s",
        "spectrum_count_method",
        "source_position_prior",
        "source_prior_mode",
        "source_surface_prior",
        "response_poisson_global_diagnostic_variance_enable",
        "source_detector_exclusion_m",
        "spectrum_likelihood_bin_chunk",
        "structural_rj_patch_spacing_m",
        "structural_rj_surface_chart_spacing_m",
        "surface_observability_diagnostic_candidates",
    }
)
_RETIRED_RUNTIME_PREFIXES = (
    "adaptive_strength_prior",
    "all_history_dictionary_proposal_",
    "batch_fit_",
    "birth_",
    "candidate_verification_",
    "cardinality_preserving_",
    "contrast_",
    "count_likelihood_",
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
    "response_poisson_",
    "residual_birth_",
    "residual_decomposition_",
    "roughening_",
    "runtime_report_rescue_",
    "soft_extension_",
    "source_prune_",
    "source_strength_",
    "shield_contrast_",
    "shield_view_ratio_",
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
    "view_ratio_",
    "weak_source_prune_",
)
_EXPERIMENT_ONLY_TOP_LEVEL_KEYS = frozenset(
    {
        "baseline_path_policy",
        "baseline_shield_policy",
        "metadata",
    }
)


def _validate_allowed_keys(
    name: str,
    payload: Mapping[str, Any],
    allowed: frozenset[str],
) -> None:
    """Reject unknown keys from one versioned pure-PF object."""
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unsupported {name} settings: " + ", ".join(unknown))


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    """Return a mapping value or fail closed with a schema error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


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


def _validate_structural_runtime_values(
    runtime_config: Mapping[str, Any],
) -> None:
    """Validate pure-PF cardinality fields before runtime construction."""
    validate_cardinality_prior_policy(
        runtime_config.get(
            "structural_cardinality_prior_policy",
            TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
        ),
        has_explicit_probabilities=(
            runtime_config.get("structural_cardinality_prior_probs") is not None
        ),
    )
    if "variable_cardinality" in runtime_config and not isinstance(
        runtime_config["variable_cardinality"],
        bool,
    ):
        raise ValueError("variable_cardinality must be a boolean.")
    if "pf_max_sources" in runtime_config:
        value = runtime_config["pf_max_sources"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("pf_max_sources must be a positive integer.")
    if "pf_hard_max_sources" in runtime_config:
        hard_value = runtime_config["pf_hard_max_sources"]
        if (
            isinstance(hard_value, bool)
            or not isinstance(hard_value, int)
            or hard_value < 1
        ):
            raise ValueError("pf_hard_max_sources must be a positive integer.")
        ordinary_value = runtime_config.get("pf_max_sources", hard_value)
        if hard_value < ordinary_value:
            raise ValueError("pf_hard_max_sources must be at least pf_max_sources.")
    if "init_num_sources" in runtime_config:
        value = runtime_config["init_num_sources"]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            )
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
            f"Unsupported estimator profile {value!r}; only 'pf_strict' is available."
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
        kernel_family = "continuous_surface_birth_death_split_merge_rj_mh"
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
        raise ValueError("Runtime configuration requires pure_pf_schema_version=1.")
    experiment_only = sorted(
        str(key)
        for key in runtime_config
        if str(key) in _EXPERIMENT_ONLY_TOP_LEVEL_KEYS
    )
    if experiment_only:
        raise ValueError(
            "Experiment-only fields must stay outside PF configuration: "
            + ", ".join(experiment_only)
        )
    retired_keys = sorted(
        str(key)
        for key in runtime_config
        if str(key) in _RETIRED_RUNTIME_KEYS
        or any(str(key).startswith(prefix) for prefix in _RETIRED_RUNTIME_PREFIXES)
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
            "Unsupported pure-PF runtime settings: " + ", ".join(unknown_pf_keys)
        )
    unknown_structural_keys = sorted(
        str(key)
        for key in runtime_config
        if (
            str(key).startswith("structural_rj_")
            or str(key).startswith("structural_cardinality_")
            or str(key).startswith("continuous_surface_")
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
    result.setdefault(
        "structural_cardinality_prior_policy",
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
    )
    if (
        result["structural_cardinality_prior_policy"]
        == POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
    ):
        result.setdefault(
            "pf_hard_max_sources",
            int(result.get("pf_max_sources", 5)),
        )
        result.setdefault("structural_cardinality_tail_ratio", 0.05)
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
