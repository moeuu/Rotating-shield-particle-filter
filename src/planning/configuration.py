"""Translate PF-owned JSON settings into strict planner configuration."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from runtime.experiment_profiles import AcquisitionContract

from planning.dss_pp import DSSPPConfig


PRODUCTION_DSS_PP_SETTING_KEYS = frozenset(
    {
        "bearing_diversity_weight",
        "conditional_greedy_one_swap",
        "coverage_floor_quantile",
        "coverage_floor_weight",
        "coverage_surface_max_hausdorff_m",
        "coverage_surface_quadrature_max_points",
        "coverage_weight",
        "eig_weight",
        "elevation_angle_threshold_deg",
        "elevation_condition_weight",
        "elevation_pair_xy_scale_m",
        "elevation_pair_z_scale_m",
        "exact_eig_coverage_reserve",
        "exact_eig_memory_budget_bytes",
        "exact_eig_pose_max",
        "exact_eig_pose_min",
        "exact_eig_pose_step",
        "frontier_weight",
        "horizontal_time_weight",
        "local_orbit_ring_radii_m",
        "local_orbit_sigma_m",
        "local_orbit_weight",
        "mast_vertical_time_weight",
        "max_modes_per_isotope",
        "mode_cluster_radius_m",
        "planning_particles",
        "proxy_boundary_confidence",
        "proxy_eig_samples",
        "proxy_memory_budget_bytes",
        "proxy_planning_particles",
        "proxy_stability_refinement_pool",
        "proxy_stability_replicates",
        "proxy_top_k_jaccard_min",
        "revisit_penalty_weight",
        "settling_time_weight",
        "shield_view_count_shadow_enabled",
        "turn_smoothness_weight",
    }
)


def validate_production_dss_setting_values(
    settings: Mapping[str, Any],
) -> None:
    """Validate every PF-owned planner value before a runtime connection.

    Runtime-owned acquisition values are supplied by a fixed validation-only
    contract here and are replaced by the authenticated handshake when the
    live planner is built. The production builder performs the same validation
    again against those real values.
    """
    validation_contract = AcquisitionContract(
        max_stations=1,
        views_per_station=8,
        live_time_s=1.0,
        max_measurements=8,
        min_station_separation_m=1.0,
        coverage_radius_m=1.0,
    )
    dss_config_from_pf_settings(
        settings,
        acquisition_contract=validation_contract,
        detector_aperture_samples=1,
    )


def _validate_production_feature_branches(config: DSSPPConfig) -> None:
    """Reject production feature combinations with ignored subordinate values."""
    local_orbit_enabled = float(config.lambda_local_orbit) > 0.0
    local_orbit_disabled_state = (
        not config.ring_radii_m and config.local_orbit_sigma_m is None
    )
    if local_orbit_enabled and local_orbit_disabled_state:
        raise ValueError(
            "Positive local_orbit_weight requires nonempty ring radii and a "
            "numeric sigma."
        )
    if not local_orbit_enabled and not local_orbit_disabled_state:
        raise ValueError(
            "local_orbit_weight=0 requires local_orbit_ring_radii_m=[] and "
            "local_orbit_sigma_m=null."
        )

    elevation_enabled = float(config.lambda_elevation_condition) > 0.0
    elevation_parameters = (
        config.elevation_pair_z_scale_m,
        config.elevation_pair_xy_scale_m,
        config.elevation_angle_threshold_deg,
    )
    elevation_disabled_state = all(
        value is None for value in elevation_parameters
    )
    if elevation_enabled and elevation_disabled_state:
        raise ValueError(
            "Positive elevation_condition_weight requires numeric elevation "
            "parameters."
        )
    if not elevation_enabled and not elevation_disabled_state:
        raise ValueError(
            "elevation_condition_weight=0 requires all elevation scale and "
            "threshold fields to be null."
        )

    floor_enabled = float(config.coverage_floor_weight) > 0.0
    floor_quantile_enabled = float(config.coverage_floor_quantile) > 0.0
    if floor_enabled != floor_quantile_enabled:
        raise ValueError(
            "coverage_floor_weight and coverage_floor_quantile must either both "
            "be positive or both be zero."
        )
    coverage_enabled = float(config.lambda_coverage) > 0.0
    if not coverage_enabled and (
        floor_enabled or int(config.exact_eig_coverage_reserve) != 0
    ):
        raise ValueError(
            "coverage_weight=0 requires a zero coverage floor and "
            "exact_eig_coverage_reserve=0."
        )
    coverage_surface_parameters = (
        config.coverage_surface_quadrature_max_points,
        config.coverage_surface_max_hausdorff_m,
    )
    if coverage_enabled and any(
        value is None for value in coverage_surface_parameters
    ):
        raise ValueError(
            "Positive coverage_weight requires numeric surface quadrature fields."
        )
    if not coverage_enabled and any(
        value is not None for value in coverage_surface_parameters
    ):
        raise ValueError(
            "coverage_weight=0 requires coverage surface quadrature fields=null."
        )

    if float(config.proxy_top_k_jaccard_min) <= 0.0:
        raise ValueError(
            "Production proxy_top_k_jaccard_min must be strictly positive."
        )
    motion_weights = (
        config.lambda_horizontal_time,
        config.lambda_mast_vertical_time,
        config.lambda_settling_time,
    )
    if not any(float(value) > 0.0 for value in motion_weights):
        raise ValueError(
            "Production planning requires at least one positive runtime motion "
            "component weight."
        )


def dss_config_from_pf_settings(
    settings: Mapping[str, Any],
    *,
    acquisition_contract: AcquisitionContract,
    detector_aperture_samples: int,
) -> DSSPPConfig:
    """Build live DSS-PP settings for runtime-owned candidate poses."""
    raw = settings.get("dss_pp", {})
    if not isinstance(raw, Mapping):
        raise TypeError("dss_pp must be a mapping.")
    expected_keys = PRODUCTION_DSS_PP_SETTING_KEYS
    actual_keys = frozenset(str(key) for key in raw)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing or unknown:
        raise ValueError(
            "Production dss_pp settings differ from the exact contract: "
            f"missing={missing}, unknown_or_retired={unknown}."
        )
    for key in (
        "horizontal_time_weight",
        "mast_vertical_time_weight",
        "settling_time_weight",
    ):
        if raw[key] is None:
            raise ValueError(f"Production dss_pp.{key} must not be null.")
    particle_count = settings["num_particles"]
    planning_particles = raw["planning_particles"]
    proxy_particles = raw["proxy_planning_particles"]
    audit_top_k = settings["planner_audit_top_k"]
    refinement_top_k = settings["runtime_candidate_refinement_top_k"]
    for key, value in (
        ("num_particles", particle_count),
        ("dss_pp.planning_particles", planning_particles),
        ("dss_pp.proxy_planning_particles", proxy_particles),
        ("planner_audit_top_k", audit_top_k),
        ("runtime_candidate_refinement_top_k", refinement_top_k),
        ("detector_aperture_samples", detector_aperture_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{key} must be an integer.")
    if int(audit_top_k) < 0 or int(refinement_top_k) < 0:
        raise ValueError("Production planner top-k values must be nonnegative.")
    if int(detector_aperture_samples) < 1:
        raise ValueError("Authenticated detector_aperture_samples must be positive.")
    if not 2 <= int(planning_particles) < int(particle_count):
        raise ValueError(
            "Production dss_pp.planning_particles must be at least 2 and "
            "strictly less than num_particles."
        )
    if not 2 <= int(proxy_particles) <= int(planning_particles):
        raise ValueError(
            "Production dss_pp.proxy_planning_particles must be at least 2 "
            "and no greater than planning_particles."
        )
    # The shared runtime owns obstacle-aware reachability and publishes
    # time-valued travel costs. The PF must not add a second Euclidean
    # distance surrogate for those same runtime-authored actions.
    program_length = acquisition_contract.views_per_station
    live_time_s = acquisition_contract.live_time_s
    coverage_radius_m = acquisition_contract.coverage_radius_m
    min_station_separation_m = acquisition_contract.min_station_separation_m
    config = DSSPPConfig(
        program_length=program_length,
        mode_cluster_radius_m=raw["mode_cluster_radius_m"],
        max_modes_per_isotope=raw["max_modes_per_isotope"],
        planning_particles=planning_particles,
        planning_method="resample",
        live_time_s=live_time_s,
        lambda_eig=raw["eig_weight"],
        lambda_distance=0.0,
        lambda_horizontal_time=raw["horizontal_time_weight"],
        lambda_mast_vertical_time=raw["mast_vertical_time_weight"],
        lambda_settling_time=raw["settling_time_weight"],
        lambda_coverage=raw["coverage_weight"],
        lambda_bearing_diversity=raw["bearing_diversity_weight"],
        lambda_frontier=raw["frontier_weight"],
        lambda_turn_smoothness=raw["turn_smoothness_weight"],
        lambda_local_orbit=raw["local_orbit_weight"],
        lambda_elevation_condition=raw["elevation_condition_weight"],
        eta_revisit=raw["revisit_penalty_weight"],
        coverage_radius_m=coverage_radius_m,
        coverage_surface_quadrature_max_points=raw[
            "coverage_surface_quadrature_max_points"
        ],
        coverage_surface_max_hausdorff_m=raw[
            "coverage_surface_max_hausdorff_m"
        ],
        coverage_floor_quantile=raw["coverage_floor_quantile"],
        coverage_floor_weight=raw["coverage_floor_weight"],
        min_station_separation_m=min_station_separation_m,
        detector_aperture_samples=int(detector_aperture_samples),
        ring_radii_m=tuple(raw["local_orbit_ring_radii_m"]),
        local_orbit_sigma_m=raw["local_orbit_sigma_m"],
        elevation_pair_z_scale_m=raw["elevation_pair_z_scale_m"],
        elevation_pair_xy_scale_m=raw["elevation_pair_xy_scale_m"],
        elevation_angle_threshold_deg=raw["elevation_angle_threshold_deg"],
        diagnostic_ranked_node_limit=max(
            int(audit_top_k),
            int(refinement_top_k),
        ),
        exact_eig_coverage_reserve=raw["exact_eig_coverage_reserve"],
        exact_eig_memory_budget_bytes=raw["exact_eig_memory_budget_bytes"],
        proxy_memory_budget_bytes=raw["proxy_memory_budget_bytes"],
        proxy_planning_particles=raw["proxy_planning_particles"],
        proxy_eig_samples=raw["proxy_eig_samples"],
        conditional_greedy_one_swap=raw["conditional_greedy_one_swap"],
        exact_eig_pose_min=raw["exact_eig_pose_min"],
        exact_eig_pose_max=raw["exact_eig_pose_max"],
        exact_eig_pose_step=raw["exact_eig_pose_step"],
        proxy_stability_refinement_pool=raw[
            "proxy_stability_refinement_pool"
        ],
        proxy_stability_replicates=raw["proxy_stability_replicates"],
        proxy_boundary_confidence=raw["proxy_boundary_confidence"],
        proxy_top_k_jaccard_min=raw["proxy_top_k_jaccard_min"],
        shield_view_count_shadow_enabled=raw[
            "shield_view_count_shadow_enabled"
        ],
        shield_view_count_shadow_candidate_counts=(2, 4, 8),
        shield_view_count_shadow_retention_fraction=0.95,
        shield_view_count_shadow_per_comparison_confidence=0.95,
        forced_program_pair_ids=None,
    )
    _validate_production_feature_branches(config)
    exact_samples = settings["planning_eig_samples"]
    if (
        isinstance(exact_samples, bool)
        or not isinstance(exact_samples, Integral)
        or int(exact_samples) < 2
    ):
        raise ValueError(
            "Conditional-greedy shield search requires the PF-level "
            "planning_eig_samples setting to be an integer >= 2."
        )
    hard_max_sources = settings["hard_max_sources"]
    if (
        isinstance(hard_max_sources, bool)
        or not isinstance(hard_max_sources, Integral)
        or int(hard_max_sources) < 1
    ):
        raise ValueError("Production hard_max_sources must be a positive integer.")
    if int(config.max_modes_per_isotope) < int(hard_max_sources):
        raise ValueError(
            "Production dss_pp.max_modes_per_isotope must cover every "
            "hard_max_sources slot."
        )
    return config


__all__ = [
    "PRODUCTION_DSS_PP_SETTING_KEYS",
    "dss_config_from_pf_settings",
    "validate_production_dss_setting_values",
]
