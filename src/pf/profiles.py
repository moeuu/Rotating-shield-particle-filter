"""Define immutable capabilities for the active particle-filter variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, MutableMapping


class EstimatorProfile(StrEnum):
    """Name the supported PF-family scientific variants."""

    PF_STRICT = "pf_strict"
    PF_PROFILED = "pf_profiled"


class ProposalOrigin(StrEnum):
    """Record the origin of a source proposal or planner mode."""

    PF_BIRTH = "pf_birth"
    PF_RESIDUAL = "pf_residual"
    PF_SPLIT = "pf_split"
    BATCH_SPARSE = "batch_sparse"
    REPORT_MLE = "report_mle"
    SURFACE_MAP = "surface_map"
    EXTERNAL_MLE = "external_mle"

    @property
    def is_pf_origin(self) -> bool:
        """Return whether the proposal is produced by a sequential PF move."""
        return self in {
            ProposalOrigin.PF_BIRTH,
            ProposalOrigin.PF_RESIDUAL,
            ProposalOrigin.PF_SPLIT,
        }


@dataclass(frozen=True)
class EstimatorCapabilities:
    """Declare every capability that can cross the PF/batch boundary."""

    conditional_strength_profile: bool
    all_history_sparse_evidence: bool
    spectral_sparse_evidence: bool
    joint_sparse_evidence: bool
    report_mle_rescue: bool
    runtime_report_rescue: bool
    all_history_dictionary_proposal: bool
    surface_map_reconstruction: bool
    batch_model_order_selection: bool
    batch_strength_refit: bool
    surface_local_refinement: bool
    offgrid_refinement: bool
    batch_feedback_to_particles: bool
    batch_candidates_in_planner: bool
    batch_evidence_in_mission_stop: bool
    batch_evidence_in_adaptive_dwell: bool

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-safe capability mapping."""
        return {str(key): bool(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class StructuralTransitionProvenance:
    """Describe whether PF structural moves preserve a declared target."""

    posterior_semantics: str
    structural_kernel_family: str
    structural_moves_enabled: bool
    structural_kernel_target_preserving: bool
    structural_kernel_exact_rj: bool
    reversible_jump_mcmc_used: bool
    data_conditioned_structural_proposal: bool
    data_conditioned_strength_proposal: bool
    data_conditioned_strength_proposal_importance_corrected: bool

    def to_dict(self) -> dict[str, bool | str]:
        """Return a JSON-safe structural-transition provenance mapping."""
        return {
            str(key): value if isinstance(value, str) else bool(value)
            for key, value in asdict(self).items()
        }


_STRICT_CAPABILITIES = EstimatorCapabilities(
    conditional_strength_profile=False,
    all_history_sparse_evidence=False,
    spectral_sparse_evidence=False,
    joint_sparse_evidence=False,
    report_mle_rescue=False,
    runtime_report_rescue=False,
    all_history_dictionary_proposal=False,
    surface_map_reconstruction=False,
    batch_model_order_selection=False,
    batch_strength_refit=False,
    surface_local_refinement=False,
    offgrid_refinement=False,
    batch_feedback_to_particles=False,
    batch_candidates_in_planner=False,
    batch_evidence_in_mission_stop=False,
    batch_evidence_in_adaptive_dwell=False,
)

_PROFILED_CAPABILITIES = EstimatorCapabilities(
    **{
        **_STRICT_CAPABILITIES.to_dict(),
        "conditional_strength_profile": True,
    }
)

_PROFILE_ALIASES = {
    "strict": EstimatorProfile.PF_STRICT,
    "pure_pf": EstimatorProfile.PF_STRICT,
    "pf_only": EstimatorProfile.PF_STRICT,
    "profiled": EstimatorProfile.PF_PROFILED,
    "pf_online_profiled": EstimatorProfile.PF_PROFILED,
}

_CONDITIONAL_STRENGTH_FIELDS = (
    "conditional_strength_refit",
    "conditional_strength_profile_before_likelihood",
    "conditional_strength_refit_reweight",
    "refit_after_moves",
)

_FORBIDDEN_BATCH_BOOLEAN_FIELDS = (
    "birth_global_rescue_enable",
    "birth_global_rescue_candidate_memory_enable",
    "runtime_report_rescue_enable",
    "runtime_report_rescue_memory_enable",
    "all_history_dictionary_proposal_enable",
    "candidate_verification_queue_enable",
    "candidate_verification_independent_evidence_enable",
    "report_mle_rescue_enable",
    "report_cluster_model_selection",
    "report_model_order_prune_particles",
    "report_strength_refit",
    "report_surface_local_refine",
    "sparse_poisson_evidence_enable",
    "sparse_poisson_evidence_authoritative",
    "sparse_poisson_spectral_evidence_enable",
    "sparse_poisson_spectral_evidence_primary",
    "sparse_poisson_joint_evidence_enable",
    "sparse_poisson_offgrid_refine_enable",
    "sparse_poisson_ambiguity_report_enable",
    "mode_preserving_report_cardinality_strata",
    "report_best_so_far_enable",
    "report_pre_finalize_guard",
    "runtime_report_rescue_verification_queue_only",
    "adaptive_strength_prior",
    "source_prune_refit_after_remove",
)

_FORBIDDEN_RUNTIME_BOOLEAN_FIELDS = (
    "adaptive_cardinality_dwell_enable",
    "adaptive_mission_stop",
    "mission_stop_require_model_order_ready",
    "mission_stop_report_simple_enable",
    "mission_stop_soft_extend_on_unresolved",
    "surface_map_reconstruction_enable",
    "pf_detected_isotopes_only",
    "pf_detected_isotope_activation_only",
    "online_absent_isotope_pruning",
    "final_absent_isotope_filter",
)


def resolve_estimator_profile(
    value: EstimatorProfile | str | None,
) -> tuple[EstimatorProfile, EstimatorCapabilities]:
    """Resolve a profile name and return its immutable capabilities."""
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
                supported = ", ".join(item.value for item in EstimatorProfile)
                raise ValueError(
                    f"Unsupported estimator profile {value!r}; choose {supported}."
                ) from exc
    capabilities = (
        _STRICT_CAPABILITIES
        if profile is EstimatorProfile.PF_STRICT
        else _PROFILED_CAPABILITIES
    )
    return profile, capabilities


def resolved_profile_diagnostics(
    value: EstimatorProfile | str | None,
) -> dict[str, Any]:
    """Return a JSON-safe resolved-profile provenance payload."""
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
    """Resolve truthful semantics for the configured PF structural kernel.

    The current residual birth/death/split/merge path selects and refits trial
    states using the observation window. It does not evaluate forward/reverse
    proposal densities, a reversible-jump Jacobian, or an importance correction.
    Consequently, enabling that path is recorded as heuristic even though the
    resulting states are scored by the configured count likelihood.
    """
    if capabilities is None:
        _profile, capabilities = resolve_estimator_profile(
            getattr(config, "estimator_profile", EstimatorProfile.PF_STRICT.value)
        )
    structural_moves_enabled = bool(getattr(config, "birth_enable", False))
    conditional_strength_profile = bool(capabilities.conditional_strength_profile) and (
        bool(getattr(config, "conditional_strength_refit", False))
        or bool(
            getattr(config, "conditional_strength_profile_before_likelihood", False)
        )
    )

    raw_initial_support = getattr(config, "init_num_sources", (0, 0))
    try:
        initial_lower, initial_upper = raw_initial_support
        fixed_initial_cardinality = int(initial_lower) == int(initial_upper)
    except (TypeError, ValueError):
        fixed_initial_cardinality = True

    if structural_moves_enabled:
        kernel_family = "residual_matching_pursuit_heuristic"
        posterior_semantics = (
            "approximate_sequential_particle_ensemble_with_heuristic_structural_moves"
        )
        target_preserving = False
        data_conditioned_structural = True
        data_conditioned_strength = True
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
        data_conditioned_structural = False
        data_conditioned_strength = conditional_strength_profile

    if conditional_strength_profile:
        posterior_semantics += "_with_deterministic_strength_profiling"

    return StructuralTransitionProvenance(
        posterior_semantics=posterior_semantics,
        structural_kernel_family=kernel_family,
        structural_moves_enabled=structural_moves_enabled,
        structural_kernel_target_preserving=target_preserving,
        # No current runtime path implements reversible-jump MCMC. For a
        # fixed-dimensional run this is false because RJ is not used, not
        # because a correction is missing.
        structural_kernel_exact_rj=False,
        reversible_jump_mcmc_used=False,
        data_conditioned_structural_proposal=data_conditioned_structural,
        data_conditioned_strength_proposal=data_conditioned_strength,
        data_conditioned_strength_proposal_importance_corrected=False,
    )


def apply_profile_to_config(config: Any) -> EstimatorCapabilities:
    """Make PF purity capabilities authoritative over legacy config booleans."""
    profile, capabilities = resolve_estimator_profile(
        getattr(config, "estimator_profile", EstimatorProfile.PF_STRICT.value)
    )
    config.estimator_profile = profile.value

    for field in _CONDITIONAL_STRENGTH_FIELDS:
        if hasattr(config, field):
            setattr(
                config,
                field,
                bool(getattr(config, field))
                and bool(capabilities.conditional_strength_profile),
            )

    for field in _FORBIDDEN_BATCH_BOOLEAN_FIELDS:
        if hasattr(config, field):
            setattr(config, field, False)
    if hasattr(config, "mode_preserving_report_cardinality_extra_particles"):
        config.mode_preserving_report_cardinality_extra_particles = 0
    return capabilities


def enforce_pure_runtime_settings(
    runtime_config: Mapping[str, Any],
    *,
    profile: EstimatorProfile | str | None = None,
) -> dict[str, Any]:
    """Return runtime settings with batch planner/dwell/stop paths disabled."""
    resolved_profile, capabilities = resolve_estimator_profile(
        profile
        if profile is not None
        else runtime_config.get("estimator_profile", EstimatorProfile.PF_STRICT.value)
    )
    result = dict(runtime_config)
    result["estimator_profile"] = resolved_profile.value
    for field in (
        *_FORBIDDEN_BATCH_BOOLEAN_FIELDS,
        *_FORBIDDEN_RUNTIME_BOOLEAN_FIELDS,
    ):
        result[field] = False
    if not capabilities.conditional_strength_profile:
        for field in _CONDITIONAL_STRENGTH_FIELDS:
            result[field] = False
    result["mode_preserving_report_cardinality_extra_particles"] = 0
    # PF update kernels share NumPy's deterministic RNG stream. Keep isotope
    # updates serial so scheduling cannot reorder draws.
    result["parallel_isotope_updates"] = False
    dss_payload = result.get("dss_pp", {})
    dss = dict(dss_payload) if isinstance(dss_payload, Mapping) else {}
    dss.update(
        {
            "include_runtime_rescue_modes": False,
            "include_global_surface_rescue_modes": False,
            "adaptive_program_length_enable": False,
        }
    )
    result["dss_pp"] = dss
    return result


def enforce_pure_runtime_settings_in_place(
    runtime_config: MutableMapping[str, Any],
) -> EstimatorProfile:
    """Apply pure runtime settings to a mutable mapping and return its profile."""
    resolved = enforce_pure_runtime_settings(runtime_config)
    runtime_config.clear()
    runtime_config.update(resolved)
    profile, _ = resolve_estimator_profile(runtime_config["estimator_profile"])
    return profile
