"""Translate PF-owned JSON settings into strict planner configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from planning.dss_pp import DSSPPConfig


def dss_config_from_pf_settings(
    settings: Mapping[str, Any],
) -> DSSPPConfig:
    """Build live DSS-PP settings for runtime-owned candidate poses."""
    raw = settings.get("dss_pp", {})
    if not isinstance(raw, Mapping):
        raise TypeError("dss_pp must be a mapping.")
    pf_cardinality_capacity = settings.get(
        "pf_hard_max_sources",
        settings.get("pf_max_sources", 5),
    )
    planning_method = raw.get("planning_method", "resample")
    if planning_method != "resample":
        raise ValueError("Production PF planning_method must be exactly 'resample'.")
    # The shared runtime owns obstacle-aware reachability and publishes
    # time-valued travel costs. The PF must not add a second Euclidean
    # distance surrogate for those same runtime-authored actions.
    return DSSPPConfig(
        max_programs=raw.get("max_programs", 40),
        program_length=raw.get("program_length", 2),
        mode_cluster_radius_m=raw.get("mode_cluster_radius_m", 1.5),
        max_modes_per_isotope=raw.get(
            "max_modes_per_isotope",
            pf_cardinality_capacity,
        ),
        planning_particles=raw.get("planning_particles", 512),
        planning_method=planning_method,
        live_time_s=settings.get("measurement_live_time_s", 30.0),
        lambda_eig=raw.get("eig_weight", 1.0),
        lambda_distance=0.0,
        lambda_time=raw.get("time_weight", 0.0),
        lambda_rotation=raw.get("rotation_weight", 0.0),
        lambda_coverage=raw.get("coverage_weight", 0.0),
        lambda_bearing_diversity=raw.get("bearing_diversity_weight", 0.0),
        lambda_frontier=raw.get("frontier_weight", 0.0),
        lambda_turn_smoothness=raw.get("turn_smoothness_weight", 0.0),
        lambda_local_orbit=raw.get("local_orbit_weight", 0.75),
        lambda_elevation_condition=raw.get(
            "elevation_condition_weight",
            0.0,
        ),
        eta_revisit=raw.get("revisit_penalty_weight", 0.0),
        coverage_radius_m=raw.get("coverage_radius_m", 3.0),
        coverage_surface_quadrature_max_points=raw.get(
            "coverage_surface_quadrature_max_points",
            65536,
        ),
        coverage_surface_max_hausdorff_m=raw.get(
            "coverage_surface_max_hausdorff_m",
            0.75,
        ),
        coverage_floor_quantile=raw.get("coverage_floor_quantile", 0.0),
        coverage_floor_weight=raw.get("coverage_floor_weight", 0.0),
        min_station_separation_m=raw.get("min_station_separation_m", 0.0),
        detector_aperture_samples=raw.get("detector_aperture_samples", 121),
        robot_speed_m_s=raw.get("robot_speed_m_s", 0.5),
        rotation_overhead_s=raw.get("rotation_overhead_s", 0.0),
        augment_candidates=False,
        max_augmented_candidates=raw.get("max_augmented_candidates", 256),
        local_orbit_sigma_m=raw.get("local_orbit_sigma_m", 0.75),
        elevation_pair_z_scale_m=raw.get("elevation_pair_z_scale_m", 2.0),
        elevation_pair_xy_scale_m=raw.get("elevation_pair_xy_scale_m", 4.0),
        elevation_angle_threshold_deg=raw.get(
            "elevation_angle_threshold_deg",
            15.0,
        ),
        diagnostic_ranked_node_limit=raw.get(
            "diagnostic_ranked_node_limit",
            10,
        ),
        exact_eig_pose_limit=raw.get("exact_eig_pose_limit", 4),
        exact_eig_action_limit=raw.get("exact_eig_action_limit", 192),
        exact_eig_coverage_reserve=raw.get("exact_eig_coverage_reserve", 1),
        exact_eig_program_diversity_reserve=raw.get(
            "exact_eig_program_diversity_reserve",
            0,
        ),
        exact_eig_memory_budget_bytes=raw.get(
            "exact_eig_memory_budget_bytes",
            4 * 1024 * 1024 * 1024,
        ),
        proxy_memory_budget_bytes=raw.get(
            "proxy_memory_budget_bytes",
            256 * 1024 * 1024,
        ),
        proxy_planning_particles=raw.get("proxy_planning_particles", 16),
        proxy_eig_samples=raw.get("proxy_eig_samples", 2),
    )


__all__ = ["dss_config_from_pf_settings"]
