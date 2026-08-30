"""Define the single supported pure particle-filter runtime profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from pf.estimator_config import RotatingShieldPFConfig
from pf.structural_rj import (
    POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
)
from runtime.cui import CUIDashboardConfig


PURE_PF_SCHEMA_VERSION = 2

PRODUCTION_ADAPTIVE_STOP_KEYS = frozenset(
    {
        "assessment_start_station",
        "innovation_confidence",
        "maximum_surface_path_radius_95_m",
        "maximum_upper_cardinality_mass",
        "minimum_joint_map_cardinality_probability",
        "required_consecutive_stations",
    }
)
PRODUCTION_CUI_KEYS = frozenset(
    {
        "cui_split_view",
        "cui_split_view_host",
        "cui_split_view_max_particles_per_isotope",
        "cui_split_view_port",
        "cui_split_view_public_host",
        "cui_split_view_save_step_history",
        "cui_split_view_serve",
    }
)
PRODUCTION_SHIFTED_GAMMA_STRENGTH_PRIOR_KEYS = frozenset(
    {"kind", "minimum_cps_1m", "shape", "scale_cps_1m"}
)
ADAPTIVE_STOP_TO_PF_FIELD = {
    "innovation_confidence": "adaptive_stop_innovation_confidence",
    "maximum_surface_path_radius_95_m": (
        "adaptive_stop_maximum_surface_path_radius_95_m"
    ),
    "maximum_upper_cardinality_mass": (
        "adaptive_stop_maximum_upper_cardinality_mass"
    ),
    "minimum_joint_map_cardinality_probability": (
        "adaptive_stop_minimum_joint_map_cardinality_probability"
    ),
}
PRODUCTION_PF_SETTING_KEYS = frozenset(
    {
        "estimator_profile",
        "num_particles",
        "max_sources",
        "hard_max_sources",
        "strength_prior",
        "structural_cardinality_tail_ratio",
        "structural_cardinality_prior_mean",
        "structural_rj_surface_chart_max_edge_m",
        "structural_rj_move_probability",
        "structural_rj_birth_probability",
        "structural_rj_death_probability",
        "structural_rj_position_move_probability",
        "structural_rj_local_position_move_probability",
        "structural_rj_strength_move_probability",
        "structural_rj_split_merge_probability",
        "structural_rj_block_independence_probability",
        "structural_rj_multi_component_probability",
        "structural_rj_multi_component_max_group_size",
        "structural_rj_local_position_scales_m",
        "structural_rj_merge_probability",
        "structural_rj_split_global_position_probability",
        "structural_rj_merge_uniform_pair_probability",
        "structural_rj_merge_distance_sigma_m",
        "structural_rj_merge_response_sigma",
        "structural_rj_split_probability",
        "structural_rj_position_proposal_prior_weight",
        "structural_rj_proposal_chart_batch_size",
        "structural_rj_proposal_score_cache_max_bytes",
        "structural_rj_strength_proposal_grid_size",
        "structural_rj_strength_proposal_prior_weight",
        "structural_rj_strength_proposal_sigma_fraction",
        "target_ess_ratio",
        "max_temper_steps",
        "min_delta_beta",
        "joint_rejuvenation_min_sweeps",
        "joint_rejuvenation_min_state_change_weight_mass",
        "joint_rejuvenation_min_surface_esjd_m2",
        "joint_rejuvenation_min_log_strength_esjd",
        "joint_lineage_recovery_min_surviving_weight_mass",
        "joint_smc_rejuvenation_wall_time_limit_s",
        "joint_guided_initialization_prior_row_probability",
        "joint_strength_block_probability",
        "joint_strength_block_log_sigma",
        "joint_strength_block_batch_size",
        "joint_cross_isotope_state_block_probability",
        "surface_diagnostic_response_cache_max_entries",
        "planning_eig_samples",
    }
)
PRODUCTION_FIXED_PF_VALUES: Mapping[str, object] = {
    "variable_cardinality": True,
    "structural_cardinality_prior_policy": (
        POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY
    ),
    "structural_cardinality_prior_probs": None,
    "joint_guided_initialization": True,
    "gpu_dtype": "float64",
}
PRODUCTION_LIVE_TOP_LEVEL_KEYS = frozenset(
    PRODUCTION_PF_SETTING_KEYS
    | {
        "adaptive_stop",
        "compute_backend",
        "dss_pp",
        "planner_audit_top_k",
        "pure_pf_schema_version",
        "runtime_candidate_refinement_top_k",
    }
    | PRODUCTION_CUI_KEYS
)


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


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    """Return a mapping value or fail closed with a schema error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def production_pf_config_values(
    runtime_config: Mapping[str, Any],
    *,
    position_max: tuple[float, float, float],
) -> dict[str, Any]:
    """Build the sole production PF dataclass payload without raw fallbacks."""
    adaptive_stop = _require_mapping(
        "adaptive_stop",
        runtime_config["adaptive_stop"],
    )
    values = {
        key: runtime_config[key]
        for key in PRODUCTION_PF_SETTING_KEYS
    }
    values.update(PRODUCTION_FIXED_PF_VALUES)
    values.update(production_compute_backend_values(runtime_config))
    values["init_num_sources"] = (0, runtime_config["max_sources"])
    for external_name, field_name in ADAPTIVE_STOP_TO_PF_FIELD.items():
        values[field_name] = adaptive_stop[external_name]
    values["position_max"] = position_max
    return values


def production_compute_backend_values(
    runtime_config: Mapping[str, Any],
) -> dict[str, object]:
    """Resolve the explicit production compute-backend discriminated union."""
    backend = _require_mapping("compute_backend", runtime_config["compute_backend"])
    kind = backend.get("kind")
    if kind == "cuda_float64":
        expected = frozenset({"kind", "device"})
        device = backend.get("device")
        if not isinstance(device, str) or (
            device != "cuda"
            and not (
                device.startswith("cuda:")
                and device[5:].isdigit()
                and str(int(device[5:])) == device[5:]
            )
        ):
            raise ValueError(
                "compute_backend.device must be 'cuda' or canonical 'cuda:N'."
            )
        use_gpu = True
    else:
        raise ValueError("compute_backend.kind must be 'cuda_float64'.")
    actual = frozenset(backend)
    if actual != expected:
        raise ValueError(
            "compute_backend fields disagree with its kind: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )
    return {
        "use_gpu": use_gpu,
        "gpu_device": device,
    }


def _enforce_production_pf_invariants(config: RotatingShieldPFConfig) -> None:
    """Reject production combinations that disable or hide declared kernels."""
    active_probabilities = (
        "structural_rj_move_probability",
        "structural_rj_birth_probability",
        "structural_rj_death_probability",
        "structural_rj_position_move_probability",
        "structural_rj_local_position_move_probability",
        "structural_rj_strength_move_probability",
        "structural_rj_split_merge_probability",
        "structural_rj_block_independence_probability",
        "structural_rj_multi_component_probability",
        "structural_rj_split_probability",
        "structural_rj_merge_probability",
        "joint_strength_block_probability",
        "joint_cross_isotope_state_block_probability",
    )
    for name in active_probabilities:
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"Production {name} must be strictly positive.")
    for left, right in (
        ("structural_rj_birth_probability", "structural_rj_death_probability"),
        ("structural_rj_split_probability", "structural_rj_merge_probability"),
    ):
        if float(getattr(config, left)) + float(getattr(config, right)) != 1.0:
            raise ValueError(
                f"Production {left} and {right} must sum exactly to 1."
            )
    for name in (
        "structural_rj_position_proposal_prior_weight",
        "structural_rj_strength_proposal_prior_weight",
        "structural_rj_split_global_position_probability",
        "structural_rj_merge_uniform_pair_probability",
        "joint_guided_initialization_prior_row_probability",
    ):
        value = float(getattr(config, name))
        if not 0.0 < value < 1.0:
            raise ValueError(f"Production {name} must lie strictly in (0, 1).")
    hard_max_sources = int(config.hard_max_sources)
    if hard_max_sources <= int(config.max_sources):
        raise ValueError(
            "Production hard_max_sources must exceed max_sources so the fixed "
            "geometric capacity tail is active."
        )
    if hard_max_sources < 3:
        raise ValueError(
            "Production hard_max_sources must be at least 3 for active "
            "multi-component RJ moves."
        )
    if int(config.structural_rj_multi_component_max_group_size) > hard_max_sources:
        raise ValueError(
            "Production structural_rj_multi_component_max_group_size must not "
            "exceed hard_max_sources."
        )
    for name in (
        "joint_rejuvenation_min_state_change_weight_mass",
        "joint_rejuvenation_min_surface_esjd_m2",
        "joint_rejuvenation_min_log_strength_esjd",
        "joint_lineage_recovery_min_surviving_weight_mass",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"Production {name} must be strictly positive.")
    if float(config.adaptive_stop_maximum_upper_cardinality_mass) >= 1.0:
        raise ValueError(
            "Production adaptive-stop upper-cardinality mass must be below 1."
        )
    guided_rows = (
        float(config.joint_guided_initialization_prior_row_probability)
        * int(config.num_particles)
    )
    if not guided_rows.is_integer():
        raise ValueError(
            "Production guided-initialization probability times num_particles "
            "must be an exact integer."
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
    """Validate the one complete schema-v2 production-live configuration."""
    if not isinstance(runtime_config, Mapping) or any(
        not isinstance(key, str) for key in runtime_config
    ):
        raise ValueError(
            "Production live PF configuration must be a string-keyed object."
        )
    actual_top_level = frozenset(runtime_config)
    missing_top_level = sorted(
        PRODUCTION_LIVE_TOP_LEVEL_KEYS.difference(actual_top_level)
    )
    unknown_top_level = sorted(
        actual_top_level.difference(PRODUCTION_LIVE_TOP_LEVEL_KEYS)
    )
    if missing_top_level or unknown_top_level:
        raise ValueError(
            "Production live PF schema-v2 keys differ from the exact contract: "
            f"missing={missing_top_level}, unknown_or_retired={unknown_top_level}."
        )
    schema_version = runtime_config.get("pure_pf_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PURE_PF_SCHEMA_VERSION
    ):
        raise ValueError("Runtime configuration requires pure_pf_schema_version=2.")
    if runtime_config["hard_max_sources"] is None:
        raise ValueError(
            "Production live PF requires an explicit hard_max_sources capacity."
        )
    adaptive_stop = _require_mapping(
        "adaptive_stop",
        runtime_config["adaptive_stop"],
    )
    adaptive_keys = frozenset(str(key) for key in adaptive_stop)
    missing_adaptive = sorted(PRODUCTION_ADAPTIVE_STOP_KEYS - adaptive_keys)
    unknown_adaptive = sorted(adaptive_keys - PRODUCTION_ADAPTIVE_STOP_KEYS)
    if missing_adaptive or unknown_adaptive:
        raise ValueError(
            "adaptive_stop keys differ from the exact schema-v2 contract: "
            f"missing={missing_adaptive}, unknown_or_retired={unknown_adaptive}."
        )
    for key in ("assessment_start_station", "required_consecutive_stations"):
        value = adaptive_stop[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"adaptive_stop.{key} must be a positive integer.")
    production_compute_backend_values(runtime_config)
    strength_prior = _require_mapping(
        "strength_prior",
        runtime_config["strength_prior"],
    )
    if any(not isinstance(key, str) for key in strength_prior):
        raise ValueError("strength_prior must be a string-keyed object.")
    strength_prior_keys = frozenset(strength_prior)
    missing_strength_prior = sorted(
        PRODUCTION_SHIFTED_GAMMA_STRENGTH_PRIOR_KEYS - strength_prior_keys
    )
    unknown_strength_prior = sorted(
        strength_prior_keys - PRODUCTION_SHIFTED_GAMMA_STRENGTH_PRIOR_KEYS
    )
    if missing_strength_prior or unknown_strength_prior:
        raise ValueError(
            "strength_prior keys differ from the exact production contract: "
            f"missing={missing_strength_prior}, "
            f"unknown_or_retired={unknown_strength_prior}."
        )
    if strength_prior["kind"] != "shifted_gamma":
        raise ValueError(
            "Production live PF requires strength_prior.kind='shifted_gamma'."
        )
    from planning.configuration import (
        PRODUCTION_DSS_PP_SETTING_KEYS,
        validate_production_dss_setting_values,
    )

    dss_pp = runtime_config["dss_pp"]
    if dss_pp is not None:
        production_dss_keys = PRODUCTION_DSS_PP_SETTING_KEYS
        dss_pp = _require_mapping("dss_pp", dss_pp)
        dss_keys = frozenset(str(key) for key in dss_pp)
        missing_dss = sorted(production_dss_keys - dss_keys)
        unknown_dss = sorted(dss_keys - production_dss_keys)
        if missing_dss or unknown_dss:
            raise ValueError(
                "dss_pp keys differ from the exact schema-v2 contract: "
                f"missing={missing_dss}, unknown_or_retired={unknown_dss}."
            )
        try:
            validate_production_dss_setting_values(runtime_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid production dss_pp setting: {exc}") from exc
    configured_profile, _capabilities = resolve_estimator_profile(
        runtime_config.get("estimator_profile")
    )
    if profile is not None:
        requested_profile, _ = resolve_estimator_profile(profile)
        if requested_profile is not configured_profile:
            raise ValueError(
                "Requested estimator profile differs from the runtime schema."
            )
    for key in (
        "cui_split_view",
        "cui_split_view_save_step_history",
        "cui_split_view_serve",
    ):
        if type(runtime_config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    for key in (
        "planner_audit_top_k",
        "runtime_candidate_refinement_top_k",
    ):
        value = runtime_config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be an integer >= 0.")
    visual_particle_limit = runtime_config[
        "cui_split_view_max_particles_per_isotope"
    ]
    if visual_particle_limit is not None and (
        isinstance(visual_particle_limit, bool)
        or not isinstance(visual_particle_limit, int)
        or visual_particle_limit < 1
    ):
        raise ValueError(
            "cui_split_view_max_particles_per_isotope must be null or a "
            "positive integer."
        )
    if (
        visual_particle_limit is not None
        and isinstance(runtime_config["num_particles"], int)
        and not isinstance(runtime_config["num_particles"], bool)
        and visual_particle_limit > runtime_config["num_particles"]
    ):
        raise ValueError(
            "cui_split_view_max_particles_per_isotope must not exceed "
            "num_particles."
        )
    cui_enabled = runtime_config["cui_split_view"]
    cui_served = runtime_config["cui_split_view_serve"]
    if cui_served and not cui_enabled:
        raise ValueError("cui_split_view_serve requires cui_split_view=true.")
    if not cui_enabled:
        if runtime_config["cui_split_view_save_step_history"] is not False:
            raise ValueError(
                "Disabled CUI requires cui_split_view_save_step_history=false."
            )
        if runtime_config["cui_split_view_max_particles_per_isotope"] is not None:
            raise ValueError(
                "Disabled CUI requires cui_split_view_max_particles_per_isotope=null."
            )
    if cui_served:
        host = runtime_config["cui_split_view_host"]
        if (
            not isinstance(host, str)
            or not host
            or host != host.strip()
            or (host.startswith("[") and host.endswith("]"))
        ):
            raise ValueError(
                "cui_split_view_host must be a canonical nonempty host string."
            )
        port = runtime_config["cui_split_view_port"]
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError(
                "cui_split_view_port must be an integer in [1, 65535]."
            )
        public_host = runtime_config["cui_split_view_public_host"]
        if (
            not isinstance(public_host, str)
            or not public_host
            or public_host != public_host.strip()
            or (public_host.startswith("[") and public_host.endswith("]"))
        ):
            raise ValueError(
                "cui_split_view_public_host must be a canonical host string."
            )
        if public_host == "auto":
            raise ValueError(
                "Production CUI requires an explicit public host; auto discovery "
                "is not allowed."
            )
        try:
            CUIDashboardConfig(
                serve=True,
                host=host,
                port=port,
                public_host=public_host,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid production CUI setting: {exc}") from exc
    else:
        for key in (
            "cui_split_view_host",
            "cui_split_view_port",
            "cui_split_view_public_host",
        ):
            if runtime_config[key] is not None:
                raise ValueError(
                    f"A non-serving CUI requires {key}=null."
                )
    pf_values = production_pf_config_values(
        runtime_config,
        position_max=(1.0, 1.0, 1.0),
    )
    try:
        pf_config = RotatingShieldPFConfig(**pf_values)
        _enforce_production_pf_invariants(pf_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid production PF setting: {exc}") from exc
    result = dict(runtime_config)
    result["estimator_profile"] = configured_profile.value
    return result


__all__ = [
    "PURE_PF_SCHEMA_VERSION",
    "PRODUCTION_PF_SETTING_KEYS",
    "EstimatorCapabilities",
    "EstimatorProfile",
    "StructuralTransitionProvenance",
    "apply_profile_to_config",
    "enforce_pure_runtime_settings",
    "production_compute_backend_values",
    "production_pf_config_values",
    "resolve_estimator_profile",
    "resolve_structural_transition_provenance",
]
