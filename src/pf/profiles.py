"""Define the single supported pure particle-filter runtime profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, MutableMapping


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
    structural_kernel_target_preserving: bool
    structural_kernel_exact_rj: bool
    reversible_jump_mcmc_used: bool
    data_conditioned_structural_proposal: bool
    data_conditioned_strength_proposal: bool
    data_conditioned_strength_proposal_importance_corrected: bool
    structural_evidence_uses_pf_likelihood: bool

    def to_dict(self) -> dict[str, bool | str]:
        """Return a JSON-safe structural-transition mapping."""
        return {
            str(key): value if isinstance(value, str) else bool(value)
            for key, value in asdict(self).items()
        }


_PURE_CAPABILITIES = EstimatorCapabilities()

_PROFILE_ALIASES = {
    "strict": EstimatorProfile.PF_STRICT,
    "pure_pf": EstimatorProfile.PF_STRICT,
    "pf_only": EstimatorProfile.PF_STRICT,
}

_REMOVED_KEY_PREFIXES = (
    "adaptive_strength_prior",
    "all_history_",
    "birth_global_rescue_",
    "birth_residual_force_",
    "candidate_verification_",
    "conditional_strength_",
    "final_absent_",
    "high_strength_split_",
    "mission_stop_report_simple_",
    "online_absent_",
    "report_best_so_far_",
    "report_cluster_",
    "report_mle_",
    "report_model_order_",
    "report_strength_",
    "report_surface_local_",
    "runtime_report_rescue_",
    "sparse_poisson_",
    "surface_map_",
    "structural_update_",
    "weak_source_prune_",
)

_REMOVED_HEURISTIC_KEYS = frozenset(
    {
        "background_sigma",
        "birth_alpha",
        "birth_bic_penalty_params",
        "birth_candidate_jitter_sigma",
        "birth_candidate_support_fraction",
        "birth_complexity_penalty",
        "birth_count_distance_log_clip",
        "birth_count_distance_prior_weight",
        "birth_count_distance_strength_sigma",
        "birth_count_distance_strength_weight",
        "birth_delta_ll_threshold",
        "birth_detector_min_sep_m",
        "birth_jitter_topk_candidates",
        "birth_matching_pursuit_max_new_sources",
        "birth_matching_pursuit_topk_candidates",
        "birth_max_per_update",
        "birth_min_distinct_poses",
        "birth_min_distinct_stations",
        "birth_min_score",
        "birth_min_sep_m",
        "birth_num_local_jitter",
        "birth_orthogonal_candidate_corr_max",
        "birth_orthogonalize_residual_candidates",
        "birth_q_max",
        "birth_q_min",
        "birth_residual_clip_quantile",
        "birth_residual_expand_structural_particles",
        "birth_residual_expanded_structural_topk_particles",
        "birth_residual_gate_p_value",
        "birth_residual_min_support",
        "birth_residual_support_sigma",
        "birth_softmax_temp",
        "birth_stage_single_station_as_quarantine",
        "birth_topk_particles",
        "birth_use_shield_coded_residual",
        "birth_use_weighted_topk",
        "cardinality_preserving_min_stations",
        "cardinality_preserving_require_confirmed_structure",
        "cardinality_preserving_resample",
        "converge_require_no_tentative",
        "deferred_resample_roughening_scale",
        "disable_regularize_on_temper_resample",
        "init_grid_repeats",
        "init_grid_spacing_m",
        "init_joint_position_design",
        "init_joint_position_retries",
        "init_source_min_separation_m",
        "label_alignment_iters",
        "label_enable",
        "label_missing_cost",
        "label_pos_scale",
        "label_pos_weight",
        "label_strength_scale",
        "label_strength_weight",
        "max_sigma_pos",
        "merge_delta_ll_threshold",
        "merge_distance_max",
        "merge_prob",
        "merge_response_corr_min",
        "merge_search_topk_pairs",
        "min_age_to_split",
        "min_sigma_pos",
        "mode_preserving_cardinality_strata",
        "mode_preserving_dynamic_cardinality_allocation",
        "mode_preserving_dynamic_cardinality_entropy_min",
        "mode_preserving_dynamic_cardinality_extra_particles",
        "mode_preserving_dynamic_cardinality_min_mass",
        "mode_preserving_dynamic_spatial_allocation",
        "mode_preserving_dynamic_spatial_extra_particles",
        "mode_preserving_dynamic_spatial_min_score_fraction",
        "mode_preserving_height_bin_m",
        "mode_preserving_high_surface_extra_particles",
        "mode_preserving_high_surface_z_fraction",
        "mode_preserving_max_modes",
        "mode_preserving_min_particles_per_cardinality",
        "mode_preserving_min_weight_fraction",
        "mode_preserving_particles_per_mode",
        "mode_preserving_radius_m",
        "mode_preserving_resample",
        "mode_preserving_residual_boost",
        "mode_preserving_support_score_weight",
        "mode_preserving_surface_strata",
        "mode_preserving_tentative_boost",
        "p_birth",
        "p_kill",
        "peak_suppression_enable",
        "peak_suppression_factor",
        "peak_suppression_min_source_fraction",
        "position_sigma",
        "pseudo_source_corr_max",
        "pseudo_source_fail_grace_stations",
        "pseudo_source_min_delta_ll",
        "pseudo_source_min_distinct_views",
        "pseudo_source_quarantine_on_suppress",
        "pseudo_source_temporal_sep_min",
        "pseudo_source_verification_enable",
        "residual_decomposition_enable",
        "residual_decomposition_max_layers",
        "roughening_decay",
        "roughening_k",
        "roughening_min_mult",
        "source_detector_exclusion_m",
        "source_prune_bic_penalty_params",
        "source_prune_delta_ll_threshold",
        "source_prune_fail_grace_stations",
        "source_prune_min_distinct_stations",
        "source_prune_min_distinct_views",
        "split_complexity_penalty",
        "split_delta_ll_threshold",
        "split_position_sigma",
        "split_prob",
        "split_residual_candidate_count",
        "split_residual_guided",
        "split_strength_max_frac",
        "split_strength_min",
        "split_strength_min_frac",
        "strength_log_sigma",
        "strength_sigma",
        "structural_kernel_mode",
        "structural_proposal_topk_particles",
        "structural_trial_parallel_min_trials",
        "structural_trial_workers",
        "support_ema_alpha",
        "surface_rejuvenation_enable",
    }
)

_REMOVED_EXACT_KEYS = (
    frozenset(
        {
            "birth_refit_residual_gate",
            "birth_refit_residual_min_fraction",
            "birth_existing_response_corr_max",
            "birth_response_condition_max",
            "birth_residual_acceptance_complexity_scale",
            "birth_residual_always_try",
            "birth_residual_forced_min_delta_ll",
            "birth_residual_suppress_death",
            "birth_window",
            "cluster_eps_m",
            "cluster_exact_max_points",
            "cluster_min_samples",
            "cluster_report_max_points",
            "converge_cluster_min_support_fraction",
            "converge_cluster_spread_max_m",
            "converge_freeze_updates",
            "death_delta_ll_threshold",
            "death_low_q_streak",
            "death_require_low_strength",
            "death_strength_threshold",
            "display_prune_refresh_every",
            "display_pruned_estimates_every",
            "final_absent_isotope_filter",
            "mode_preserving_report_cardinality_extra_particles",
            "mode_preserving_report_cardinality_strata",
            "mission_stop_require_model_order_ready",
            "mission_stop_soft_extension_require_report_progress",
            "pseudo_source_quarantine_excludes_runtime",
            "adaptive_cardinality_condition_max",
            "adaptive_cardinality_min_bic_margin",
            "adaptive_cardinality_min_candidate_count",
            "refit_after_moves",
            "refit_eps",
            "refit_iters",
            "report_pre_finalize_guard",
            "report_exclude_unverified_sources",
            "source_prune_refit_after_remove",
            "source_position_max",
            "source_position_min",
            "source_strength_absorption_penalty_weight",
            "source_strength_absorption_q_multiple",
            "source_strength_observation_overshoot_min_visible_fraction",
            "source_strength_observation_overshoot_min_visible_measurements",
            "source_strength_observation_overshoot_penalty_weight",
            "source_strength_observation_overshoot_quantile",
            "source_strength_observation_overshoot_sigma",
            "source_strength_prior_mean",
            "source_strength_prior_rel_sigma",
            "source_strength_prior_weight",
            "source_z_max_m",
            "source_z_min_m",
            "split_residual_always_try",
            "support_window",
            "use_clustered_output",
        }
    )
    | _REMOVED_HEURISTIC_KEYS
)

_REMOVED_DSS_KEYS = frozenset(
    {
        "include_global_surface_rescue_modes",
        "include_runtime_rescue_modes",
        "cardinality_bic_parameter_count_per_source",
        "cardinality_evidence_gap_target",
        "global_surface_rescue_mode_weight",
        "lambda_cardinality_discrimination",
        "runtime_rescue_mode_weight",
    }
)

_REMOVED_REMAINING_MEASUREMENT_KEYS = frozenset(
    {
        "high_surface_absorption_q_multiple",
        "report_positive_residual_fraction_threshold",
        "report_residual_weight",
        "report_response_correlation_threshold",
        "report_response_correlation_weight",
        "report_strength_concentration_threshold",
        "residual_surface_gain_candidate_limit",
        "strength_absorption_weight",
    }
)


def removed_estimator_config_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return removed estimator keys still present in a runtime mapping."""
    removed = {
        str(key)
        for key in config
        if str(key) in _REMOVED_EXACT_KEYS or str(key).startswith(_REMOVED_KEY_PREFIXES)
    }
    dss_payload = config.get("dss_pp")
    if isinstance(dss_payload, Mapping):
        removed.update(
            f"dss_pp.{key}" for key in dss_payload if key in _REMOVED_DSS_KEYS
        )
    removed.update(f"dss_{key}" for key in _REMOVED_DSS_KEYS if f"dss_{key}" in config)
    remaining_payload = config.get("remaining_measurement_estimate")
    if isinstance(remaining_payload, Mapping):
        removed.update(
            f"remaining_measurement_estimate.{key}"
            for key in remaining_payload
            if key in _REMOVED_REMAINING_MEASUREMENT_KEYS
        )
    removed.update(
        f"remaining_measurement_{key}"
        for key in _REMOVED_REMAINING_MEASUREMENT_KEYS
        if f"remaining_measurement_{key}" in config
    )
    return tuple(sorted(removed))


def reject_removed_estimator_config(config: Mapping[str, Any]) -> None:
    """Reject configuration for estimator implementations that no longer exist."""
    removed = removed_estimator_config_keys(config)
    if removed:
        joined = ", ".join(removed)
        raise ValueError(
            f"Removed non-PF estimator configuration is not supported: {joined}."
        )


def resolve_estimator_profile(
    value: EstimatorProfile | str | None,
) -> tuple[EstimatorProfile, EstimatorCapabilities]:
    """Resolve the strict pure-PF profile and reject removed variants."""
    if value is None:
        profile = EstimatorProfile.PF_STRICT
    elif isinstance(value, EstimatorProfile):
        profile = value
    else:
        normalized = str(value).strip().lower().replace("-", "_")
        profile = _PROFILE_ALIASES.get(normalized)
        if profile is None:
            try:
                profile = EstimatorProfile(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported estimator profile {value!r}; "
                    "only 'pf_strict' is available."
                ) from exc
    return profile, _PURE_CAPABILITIES


def resolved_profile_diagnostics(
    value: EstimatorProfile | str | None,
) -> dict[str, Any]:
    """Return JSON-safe pure-PF profile provenance."""
    profile, capabilities = resolve_estimator_profile(value)
    return {
        "estimator_family": "particle_filter",
        "estimator_variant": profile.value,
        "profile_capabilities": capabilities.to_dict(),
    }


def resolve_structural_transition_provenance(
    config: Any,
    *,
    capabilities: EstimatorCapabilities | None = None,
) -> StructuralTransitionProvenance:
    """Resolve truthful provenance for the configured PF structural kernel."""
    del capabilities
    structural_moves_enabled = bool(getattr(config, "birth_enable", False))
    raw_initial_support = getattr(config, "init_num_sources", (0, 0))
    try:
        initial_lower, initial_upper = raw_initial_support
        fixed_initial_cardinality = int(initial_lower) == int(initial_upper)
    except (TypeError, ValueError):
        fixed_initial_cardinality = True

    if structural_moves_enabled:
        kernel_family = "area_weighted_surface_birth_death_rj_mh"
        posterior_semantics = (
            "sequential_particle_filter_with_target_preserving_rj_mh_rejuvenation"
        )
        target_preserving = True
        exact_rj = True
        reversible_jump_used = True
        data_conditioned_proposal = False
        data_conditioned_strength = False
    else:
        kernel_family = (
            "fixed_cardinality_no_structural_moves"
            if fixed_initial_cardinality
            else "static_cardinality_mixture_no_structural_moves"
        )
        posterior_semantics = (
            "fixed_cardinality_sequential_particle_filter"
            if fixed_initial_cardinality
            else "static_cardinality_mixture_sequential_particle_filter"
        )
        target_preserving = True
        exact_rj = False
        reversible_jump_used = False
        data_conditioned_proposal = False
        data_conditioned_strength = False

    return StructuralTransitionProvenance(
        posterior_semantics=posterior_semantics,
        structural_kernel_family=kernel_family,
        structural_moves_enabled=structural_moves_enabled,
        structural_kernel_target_preserving=target_preserving,
        structural_kernel_exact_rj=exact_rj,
        reversible_jump_mcmc_used=reversible_jump_used,
        data_conditioned_structural_proposal=data_conditioned_proposal,
        data_conditioned_strength_proposal=data_conditioned_strength,
        data_conditioned_strength_proposal_importance_corrected=False,
        structural_evidence_uses_pf_likelihood=True,
    )


def apply_profile_to_config(config: Any) -> EstimatorCapabilities:
    """Validate and stamp the single supported estimator profile."""
    profile, capabilities = resolve_estimator_profile(
        getattr(config, "estimator_profile", EstimatorProfile.PF_STRICT.value)
    )
    source_prior = str(getattr(config, "source_position_prior", "surface")).strip()
    if source_prior.lower() != "surface":
        raise ValueError(
            "The pf_strict estimator requires source_position_prior='surface'."
        )
    config.estimator_profile = profile.value
    return capabilities


def enforce_pure_runtime_settings(
    runtime_config: Mapping[str, Any],
    *,
    profile: EstimatorProfile | str | None = None,
) -> dict[str, Any]:
    """Validate a runtime mapping and stamp the strict pure-PF profile."""
    reject_removed_estimator_config(runtime_config)
    if "source_surface_prior" in runtime_config:
        raw_surface_prior = runtime_config["source_surface_prior"]
        surface_prior_enabled = (
            raw_surface_prior
            if isinstance(raw_surface_prior, bool)
            else str(raw_surface_prior).strip().lower()
            in {"1", "true", "yes", "on", "surface", "surfaces"}
        )
        if not surface_prior_enabled:
            raise ValueError(
                "The pf_strict estimator requires source_surface_prior=true."
            )
    source_prior = runtime_config.get("source_position_prior")
    normalized_source_prior = (
        "surface"
        if source_prior is True
        else str(source_prior).strip().lower()
        if source_prior is not None
        else None
    )
    if normalized_source_prior is not None and normalized_source_prior != "surface":
        raise ValueError(
            "The pf_strict estimator requires source_position_prior='surface'."
        )
    resolved_profile, _capabilities = resolve_estimator_profile(
        profile
        if profile is not None
        else runtime_config.get("estimator_profile", EstimatorProfile.PF_STRICT.value)
    )
    result = dict(runtime_config)
    result["estimator_profile"] = resolved_profile.value
    return result


def enforce_pure_runtime_settings_in_place(
    runtime_config: MutableMapping[str, Any],
) -> EstimatorProfile:
    """Validate and stamp a mutable runtime mapping in place."""
    resolved = enforce_pure_runtime_settings(runtime_config)
    runtime_config.clear()
    runtime_config.update(resolved)
    profile, _ = resolve_estimator_profile(runtime_config["estimator_profile"])
    return profile


__all__ = [
    "EstimatorCapabilities",
    "EstimatorProfile",
    "StructuralTransitionProvenance",
    "apply_profile_to_config",
    "enforce_pure_runtime_settings",
    "enforce_pure_runtime_settings_in_place",
    "reject_removed_estimator_config",
    "removed_estimator_config_keys",
    "resolve_estimator_profile",
    "resolve_structural_transition_provenance",
    "resolved_profile_diagnostics",
]
