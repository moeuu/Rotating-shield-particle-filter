"""Tests for estimator-owned DSS configuration translation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from runtime.experiment_profiles import AcquisitionContract

from planning.configuration import dss_config_from_pf_settings
from planning.dss_types import DSSPPConfig


def test_runtime_workspace_disables_estimator_authored_candidate_poses() -> None:
    """A live PF may rank runtime poses but must not inject unvalidated poses."""
    config = dss_config_from_pf_settings(
        {
            "pf_max_sources": 7,
            "measurement_live_time_s": 45.0,
            "dss_pp": {
                "augment_candidates": True,
                "program_length": 8,
                "max_programs": 48,
                "planning_method": "resample",
                "diagnostic_ranked_node_limit": 12,
                "distance_weight": 0.4,
            },
        },
    )

    assert config.augment_candidates is False
    assert config.program_length == 8
    assert config.max_modes_per_isotope == 7
    assert config.live_time_s == 45.0
    assert config.diagnostic_ranked_node_limit == 12
    assert config.lambda_distance == 0.0


def test_live_planner_uses_only_runtime_acquisition_values() -> None:
    """Shared station geometry and timing must come from the runtime contract."""
    contract = AcquisitionContract(
        max_stations=16,
        views_per_station=8,
        live_time_s=20.0,
        max_measurements=128,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    )
    config = dss_config_from_pf_settings(
        {"dss_pp": {"planning_method": "resample"}},
        acquisition_contract=contract,
    )

    assert config.program_length == 8
    assert config.live_time_s == pytest.approx(20.0)
    assert config.min_station_separation_m == pytest.approx(3.0)
    assert config.coverage_radius_m == pytest.approx(3.0)

    with pytest.raises(ValueError, match="Runtime-owned acquisition settings"):
        dss_config_from_pf_settings(
            {
                "measurement_live_time_s": 30.0,
                "dss_pp": {"planning_method": "resample"},
            },
            acquisition_contract=contract,
        )


def test_runtime_planner_defaults_to_pf_hard_cardinality_capacity() -> None:
    """Planner summaries must retain PF states above the ordinary K limit."""
    config = dss_config_from_pf_settings(
        {
            "pf_max_sources": 5,
            "pf_hard_max_sources": 8,
            "dss_pp": {"planning_method": "resample"},
        },
    )

    assert config.max_modes_per_isotope == 8


def test_runtime_planner_retires_angular_rotation_penalty() -> None:
    """Production shield choice must not trade exact EIG for angular motion."""
    config = dss_config_from_pf_settings(
        {"dss_pp": {"planning_method": "resample"}},
    )

    assert config.lambda_rotation == 0.0
    assert config.exact_eig_pose_limit == 4
    assert config.exact_eig_action_limit == 192
    with pytest.raises(ValueError, match="lambda_rotation is retired"):
        dss_config_from_pf_settings(
            {
                "dss_pp": {
                    "planning_method": "resample",
                    "rotation_weight": 0.15,
                }
            },
        )


def test_runtime_planner_parses_conditional_all_pair_search_controls() -> None:
    """Production settings must expose the adaptive all-pair search contract."""
    config = dss_config_from_pf_settings(
        {
            "dss_pp": {
                "shield_program_search_policy": "conditional_greedy_all_pairs",
                "program_length": 8,
                "legacy_program_guard_enabled": True,
                "conditional_greedy_one_swap": True,
                "exact_eig_pose_min": 8,
                "exact_eig_pose_max": 16,
                "exact_eig_pose_step": 4,
                "proxy_stability_refinement_pool": 24,
                "proxy_stability_replicates": 3,
                "proxy_boundary_confidence": 0.95,
                "proxy_top_k_jaccard_min": 0.75,
                "shield_view_count_shadow_enabled": True,
                "shield_view_count_shadow_candidate_counts": [2, 4, 8],
                "shield_view_count_shadow_retention_fraction": 0.95,
                "shield_view_count_shadow_per_comparison_confidence": 0.95,
            }
        },
    )

    assert config.shield_program_search_policy == "conditional_greedy_all_pairs"
    assert config.legacy_program_guard_enabled is True
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
    assert config.shield_view_count_shadow_retention_fraction == pytest.approx(0.95)
    assert config.shield_view_count_shadow_per_comparison_confidence == pytest.approx(
        0.95
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"shield_program_search_policy": "predeclared_library"},
        {"program_length": 4},
        {"shield_view_count_shadow_candidate_counts": [2, 3, 8]},
    ),
)
def test_view_count_shadow_requires_the_fixed_eight_all_pair_contract(
    overrides: dict[str, object],
) -> None:
    """Shadow settings must not silently alter legacy or runtime acquisition."""
    values: dict[str, object] = {
        "shield_program_search_policy": "conditional_greedy_all_pairs",
        "shield_view_count_shadow_enabled": True,
        "shield_view_count_shadow_candidate_counts": [2, 4, 8],
        "program_length": 8,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="shadow|Shadow|program_length"):
        dss_config_from_pf_settings({"dss_pp": values})


@pytest.mark.parametrize(
    "policy",
    ("conditional_greedy_all_pairs", "conditional_greedy_shadow"),
)
def test_conditional_search_requires_replicated_proxy_and_exact_samples(
    policy: str,
) -> None:
    """Paired uncertainty checks require at least two MC draws per stage."""
    with pytest.raises(ValueError, match="proxy_eig_samples >= 2"):
        dss_config_from_pf_settings(
            {
                "dss_pp": {
                    "shield_program_search_policy": policy,
                    "proxy_eig_samples": 1,
                }
            }
        )

    with pytest.raises(ValueError, match="planning_eig_samples.*>= 2"):
        dss_config_from_pf_settings(
            {
                "planning_eig_samples": 1,
                "dss_pp": {"shield_program_search_policy": policy},
            }
        )


@pytest.mark.parametrize(
    "policy",
    ("conditional_greedy_all_pairs", "conditional_greedy_shadow"),
)
def test_conditional_search_requires_positive_eig_weight(policy: str) -> None:
    """Conditional shield selection cannot silently become a geometry policy."""
    with pytest.raises(ValueError, match="requires lambda_eig > 0"):
        DSSPPConfig(
            shield_program_search_policy=policy,
            lambda_eig=0.0,
        )


@pytest.mark.parametrize(
    "policy",
    ("conditional_greedy_all_pairs", "conditional_greedy_shadow"),
)
def test_conditional_shortlist_ignores_retired_legacy_pose_limit(
    policy: str,
) -> None:
    """The all-pair reserve is bounded by its adaptive minimum, not old top-4."""
    config = DSSPPConfig(
        shield_program_search_policy=policy,
        exact_eig_pose_limit=4,
        exact_eig_coverage_reserve=5,
        exact_eig_pose_min=8,
        exact_eig_pose_max=16,
    )

    assert config.exact_eig_coverage_reserve == 5


def test_all_pair_shortlist_ignores_retired_legacy_action_limit() -> None:
    """The all-pair path does not allocate old pose-by-program action slots."""
    config = DSSPPConfig(
        shield_program_search_policy="conditional_greedy_all_pairs",
        exact_eig_pose_limit=4,
        exact_eig_action_limit=1,
    )

    assert config.exact_eig_action_limit == 1


def test_legacy_shortlist_still_enforces_legacy_pose_limit() -> None:
    """The predeclared path must retain its own top-k reserve invariant."""
    with pytest.raises(ValueError, match="fit within exact_eig_pose_limit"):
        DSSPPConfig(
            shield_program_search_policy="predeclared_library",
            exact_eig_pose_limit=4,
            exact_eig_coverage_reserve=5,
        )


def test_pf_strict_profile_enables_conditional_all_pair_standard() -> None:
    """The shipped strict profile must execute the new policy with eight views."""
    settings = json.loads(
        (Path(__file__).parents[1] / "configs/pf/pf_strict_3d.json").read_text(
            encoding="utf-8"
        )
    )
    contract = AcquisitionContract(
        max_stations=16,
        views_per_station=8,
        live_time_s=30.0,
        max_measurements=128,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    )

    config = dss_config_from_pf_settings(
        settings,
        acquisition_contract=contract,
    )

    assert config.shield_program_search_policy == "conditional_greedy_all_pairs"
    assert config.program_length == 8
    assert config.planning_particles == 512
    assert config.exact_eig_pose_min == 8
    assert config.exact_eig_pose_max == 16
    assert config.legacy_program_guard_enabled is True
    assert config.exact_eig_memory_budget_bytes == 4 * 1024**3
    assert config.proxy_memory_budget_bytes == 4 * 1024**3


def test_runtime_planner_separates_mast_from_horizontal_time_weight() -> None:
    """Production motion scoring must discount mast time without hiding travel."""
    config = dss_config_from_pf_settings(
        {
            "dss_pp": {
                "planning_method": "resample",
                "measurement_time_weight": 0.02,
                "horizontal_time_weight": 0.02,
                "mast_vertical_time_weight": 0.005,
                "settling_time_weight": 0.02,
            }
        },
    )

    assert config.lambda_time == pytest.approx(0.02)
    assert config.lambda_horizontal_time == pytest.approx(0.02)
    assert config.lambda_mast_vertical_time == pytest.approx(0.005)
    assert config.lambda_settling_time == pytest.approx(0.02)
