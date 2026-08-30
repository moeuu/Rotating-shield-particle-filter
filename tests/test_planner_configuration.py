"""Tests for the fail-closed production DSS configuration contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from runtime.experiment_profiles import AcquisitionContract

from planning.configuration import dss_config_from_pf_settings as _build_dss_config
from planning.dss_types import DSSPPConfig


def _planner_settings(
    dss_pp: dict[str, object],
    **top_level: object,
) -> dict[str, object]:
    """Return the complete production contract with selected overrides."""
    settings = json.loads(
        (Path(__file__).parents[1] / "configs/pf/pf_strict_3d.json").read_text(
            encoding="utf-8"
        )
    )
    settings["dss_pp"].update(dss_pp)
    settings.update(top_level)
    return settings


def _acquisition_contract(**overrides: object) -> AcquisitionContract:
    """Return the runtime-owned acquisition contract used by planner tests."""
    values: dict[str, object] = {
        "max_stations": 16,
        "views_per_station": 8,
        "live_time_s": 20.0,
        "max_measurements": 128,
        "min_station_separation_m": 3.0,
        "coverage_radius_m": 3.0,
    }
    values.update(overrides)
    return AcquisitionContract(**values)


def dss_config_from_pf_settings(
    settings: dict[str, object],
    *,
    acquisition_contract: AcquisitionContract | None = None,
    detector_aperture_samples: int = 121,
) -> DSSPPConfig:
    """Build a planner config with an explicit runtime acquisition contract."""
    return _build_dss_config(
        settings,
        acquisition_contract=acquisition_contract or _acquisition_contract(),
        detector_aperture_samples=detector_aperture_samples,
    )


@pytest.mark.parametrize(
    "retired_key",
    (
        "augment_candidates",
        "distance_weight",
        "diagnostic_ranked_node_limit",
        "detector_aperture_samples",
        "exact_eig_action_limit",
        "exact_eig_pose_limit",
        "exact_eig_program_diversity_reserve",
        "legacy_program_guard_enabled",
        "max_augmented_candidates",
        "max_programs",
        "measurement_time_weight",
        "planning_method",
        "rotation_weight",
        "shield_view_count_shadow_candidate_counts",
        "shield_view_count_shadow_per_comparison_confidence",
        "shield_view_count_shadow_retention_fraction",
        "shield_program_search_policy",
        "time_weight",
    ),
)
def test_runtime_planner_rejects_retired_settings(retired_key: str) -> None:
    """Removed planner controls must fail instead of being ignored or aliased."""
    with pytest.raises(ValueError, match="unknown_or_retired"):
        dss_config_from_pf_settings(_planner_settings({retired_key: 0}))


def test_runtime_planner_rejects_unknown_settings() -> None:
    """Planner typos must fail before a default can alter acquisition."""
    with pytest.raises(ValueError, match="unknown_option"):
        dss_config_from_pf_settings(
            _planner_settings({"unknown_option": 1})
        )


def test_planner_uses_authenticated_runtime_detector_aperture() -> None:
    """Planner geometry must come from the runtime model, never PF JSON."""
    config = dss_config_from_pf_settings(
        _planner_settings({}),
        detector_aperture_samples=37,
    )

    assert config.detector_aperture_samples == 37


def test_live_planner_uses_only_runtime_acquisition_values() -> None:
    """Shared station geometry and timing must come from the runtime contract."""
    contract = _acquisition_contract(live_time_s=20.0)
    config = dss_config_from_pf_settings(
        _planner_settings({}),
        acquisition_contract=contract,
    )

    assert config.program_length == 8
    assert config.live_time_s == pytest.approx(20.0)
    assert config.min_station_separation_m == pytest.approx(3.0)
    assert config.coverage_radius_m == pytest.approx(3.0)

    with pytest.raises(ValueError, match="unknown_or_retired"):
        dss_config_from_pf_settings(
            _planner_settings(
                {"program_length": 8},
            ),
            acquisition_contract=contract,
        )


def test_runtime_planner_defaults_to_pf_hard_cardinality_capacity() -> None:
    """Planner summaries must use the explicitly configured mode capacity."""
    config = dss_config_from_pf_settings(
        _planner_settings({}),
    )

    assert config.max_modes_per_isotope == 8


def test_runtime_planner_parses_current_all_pair_search_controls() -> None:
    """Production settings must expose only the current all-pair contract."""
    config = dss_config_from_pf_settings(
        _planner_settings(
            {
        "conditional_greedy_one_swap": True,
                "exact_eig_pose_min": 8,
                "exact_eig_pose_max": 16,
                "exact_eig_pose_step": 4,
                "proxy_stability_refinement_pool": 24,
                "proxy_stability_replicates": 3,
                "proxy_boundary_confidence": 0.95,
                "proxy_top_k_jaccard_min": 0.75,
                "shield_view_count_shadow_enabled": True,
            }
        ),
    )

    assert config.conditional_greedy_one_swap is True
    assert config.exact_eig_pose_min == 8
    assert config.exact_eig_pose_max == 16
    assert config.exact_eig_pose_step == 4
    assert config.proxy_stability_refinement_pool == 24
    assert config.proxy_stability_replicates == 3
    assert config.proxy_boundary_confidence == pytest.approx(0.95)
    assert config.proxy_top_k_jaccard_min == pytest.approx(0.75)
    assert config.shield_view_count_shadow_enabled is True
    assert config.shield_view_count_shadow_candidate_counts == (2, 4, 8)


def test_view_count_shadow_uses_one_fixed_research_audit_contract() -> None:
    """Production may enable the audit but cannot alter its fixed policy."""
    config = dss_config_from_pf_settings(
        _planner_settings({"shield_view_count_shadow_enabled": True})
    )

    assert config.shield_view_count_shadow_candidate_counts == (2, 4, 8)
    assert config.shield_view_count_shadow_retention_fraction == pytest.approx(0.95)
    assert config.shield_view_count_shadow_per_comparison_confidence == pytest.approx(
        0.95
    )


def test_conditional_search_requires_replicated_proxy_and_exact_samples() -> None:
    """Paired uncertainty checks require at least two MC draws per stage."""
    with pytest.raises(ValueError, match="proxy_eig_samples >= 2"):
        dss_config_from_pf_settings(_planner_settings({"proxy_eig_samples": 1}))

    with pytest.raises(ValueError, match="planning_eig_samples.*>= 2"):
        dss_config_from_pf_settings(
            _planner_settings({}, planning_eig_samples=1)
        )


def test_conditional_search_requires_positive_eig_weight() -> None:
    """Production shield selection cannot silently become a geometry policy."""
    with pytest.raises(ValueError, match="requires lambda_eig > 0"):
        DSSPPConfig(lambda_eig=0.0)


def test_all_pair_coverage_reserve_uses_adaptive_minimum() -> None:
    """The all-pair reserve is bounded by the current adaptive shortlist."""
    config = DSSPPConfig(
        exact_eig_coverage_reserve=5,
        exact_eig_pose_min=8,
        exact_eig_pose_max=16,
    )

    assert config.exact_eig_coverage_reserve == 5


def test_pf_strict_profile_uses_only_current_planner_contract() -> None:
    """The shipped strict profile must load without any retired control."""
    settings = json.loads(
        (Path(__file__).parents[1] / "configs/pf/pf_strict_3d.json").read_text(
            encoding="utf-8"
        )
    )
    contract = _acquisition_contract()

    config = dss_config_from_pf_settings(settings, acquisition_contract=contract)

    assert config.program_length == 8
    assert config.planning_particles == 512
    assert config.planning_method == "resample"
    assert config.diagnostic_ranked_node_limit == 10
    assert config.exact_eig_pose_min == 8
    assert config.exact_eig_pose_max == 16
    assert config.exact_eig_memory_budget_bytes == 4 * 1024**3
    assert config.proxy_memory_budget_bytes == 4 * 1024**3
    retired = {
        "augment_candidates",
        "distance_weight",
        "legacy_program_guard_enabled",
        "max_programs",
        "measurement_time_weight",
        "planning_method",
        "diagnostic_ranked_node_limit",
        "shield_view_count_shadow_candidate_counts",
        "shield_view_count_shadow_retention_fraction",
        "shield_view_count_shadow_per_comparison_confidence",
        "shield_program_search_policy",
    }
    assert retired.isdisjoint(settings["dss_pp"])


def test_runtime_planner_separates_motion_component_weights() -> None:
    """Production motion scoring must preserve component-specific weights."""
    config = dss_config_from_pf_settings(
        _planner_settings(
            {
                "horizontal_time_weight": 0.02,
                "mast_vertical_time_weight": 0.005,
                "settling_time_weight": 0.02,
                "local_orbit_ring_radii_m": [1.5, 3.0, 4.5],
            }
        ),
    )

    assert config.lambda_horizontal_time == pytest.approx(0.02)
    assert config.lambda_mast_vertical_time == pytest.approx(0.005)
    assert config.lambda_settling_time == pytest.approx(0.02)
    assert config.ring_radii_m == pytest.approx((1.5, 3.0, 4.5))


@pytest.mark.parametrize(
    "radii",
    ([2.0, 2.0, 3.5], [3.5, 2.0, 5.0]),
)
def test_runtime_planner_requires_canonical_ring_radii(
    radii: list[float],
) -> None:
    """Equivalent duplicate or reordered orbit rings must be rejected."""
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        dss_config_from_pf_settings(
            _planner_settings({"local_orbit_ring_radii_m": radii})
        )


def test_runtime_planner_accepts_one_explicit_geometry_disabled_state() -> None:
    """Zero compound weights must carry unambiguous null/empty sentinels."""
    config = dss_config_from_pf_settings(
        _planner_settings(
            {
                "coverage_weight": 0.0,
                "coverage_floor_quantile": 0.0,
                "coverage_floor_weight": 0.0,
                "coverage_surface_max_hausdorff_m": None,
                "coverage_surface_quadrature_max_points": None,
                "exact_eig_coverage_reserve": 0,
                "local_orbit_weight": 0.0,
                "local_orbit_ring_radii_m": [],
                "local_orbit_sigma_m": None,
                "elevation_condition_weight": 0.0,
                "elevation_pair_xy_scale_m": None,
                "elevation_pair_z_scale_m": None,
                "elevation_angle_threshold_deg": None,
            }
        )
    )

    assert config.lambda_coverage == pytest.approx(0.0)
    assert config.coverage_floor_weight == pytest.approx(0.0)
    assert config.exact_eig_coverage_reserve == 0
    assert config.coverage_surface_max_hausdorff_m is None
    assert config.coverage_surface_quadrature_max_points is None
    assert config.ring_radii_m == ()
    assert config.local_orbit_sigma_m is None
    assert config.elevation_pair_xy_scale_m is None
    assert config.elevation_pair_z_scale_m is None
    assert config.elevation_angle_threshold_deg is None


@pytest.mark.parametrize(
    "overrides",
    (
        {"local_orbit_weight": 0.0},
        {
            "local_orbit_weight": 0.0,
            "local_orbit_ring_radii_m": [],
        },
        {"elevation_condition_weight": 0.0},
        {
            "elevation_condition_weight": 0.0,
            "elevation_pair_xy_scale_m": None,
        },
    ),
)
def test_runtime_planner_rejects_ambiguous_disabled_geometry(
    overrides: dict[str, object],
) -> None:
    """A zero compound weight cannot leave arbitrary subordinate settings."""
    with pytest.raises(ValueError, match="local_orbit|[Ee]levation"):
        dss_config_from_pf_settings(_planner_settings(overrides))


@pytest.mark.parametrize(
    "overrides",
    (
        {"coverage_floor_quantile": 0.0},
        {"coverage_floor_weight": 0.0},
        {
            "coverage_weight": 0.0,
            "coverage_floor_quantile": 0.0,
            "coverage_floor_weight": 0.0,
        },
    ),
)
def test_runtime_planner_rejects_incomplete_coverage_disable(
    overrides: dict[str, object],
) -> None:
    """Coverage score, floor, and shortlist reserve must disable together."""
    with pytest.raises(ValueError, match="coverage"):
        dss_config_from_pf_settings(_planner_settings(overrides))


def test_runtime_planner_rejects_zero_stability_jaccard_gate() -> None:
    """Production adaptive shortlisting cannot disable its overlap gate."""
    with pytest.raises(ValueError, match="proxy_top_k_jaccard_min"):
        dss_config_from_pf_settings(
            _planner_settings({"proxy_top_k_jaccard_min": 0.0})
        )


def test_runtime_planner_requires_a_motion_cost_component() -> None:
    """Production pose scores cannot silently ignore all runtime motion time."""
    with pytest.raises(ValueError, match="runtime motion component"):
        dss_config_from_pf_settings(
            _planner_settings(
                {
                    "horizontal_time_weight": 0.0,
                    "mast_vertical_time_weight": 0.0,
                    "settling_time_weight": 0.0,
                }
            )
        )


@pytest.mark.parametrize(
    "key",
    ("horizontal_time_weight", "mast_vertical_time_weight", "settling_time_weight"),
)
def test_runtime_planner_rejects_null_motion_weights(key: str) -> None:
    """A null component weight must not silently remove a motion penalty."""
    with pytest.raises(ValueError, match="must not be null"):
        dss_config_from_pf_settings(_planner_settings({key: None}))


@pytest.mark.parametrize("planning_particles", (None, 4096, 8192))
def test_runtime_planner_requires_an_effective_particle_subset(
    planning_particles: object,
) -> None:
    """The configured resampling method must never collapse to use-all mode."""
    with pytest.raises(ValueError, match="planning_particles"):
        dss_config_from_pf_settings(
            _planner_settings({"planning_particles": planning_particles})
        )


def test_runtime_planner_requires_proxy_subset_within_exact_subset() -> None:
    """A proxy particle budget cannot exceed the exact planning budget."""
    with pytest.raises(ValueError, match="no greater than planning_particles"):
        dss_config_from_pf_settings(
            _planner_settings(
                {"planning_particles": 32, "proxy_planning_particles": 64}
            )
        )


def test_runtime_planner_mode_capacity_covers_the_hard_pf_capacity() -> None:
    """Mode truncation must fail at configuration time, not first planning."""
    with pytest.raises(ValueError, match="cover every hard_max_sources slot"):
        dss_config_from_pf_settings(
            _planner_settings({"max_modes_per_isotope": 7})
        )
