"""Tests for estimator-owned DSS configuration translation."""

from __future__ import annotations

from planning.configuration import dss_config_from_pf_settings


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
        runtime_owned_candidates=True,
    )

    assert config.augment_candidates is False
    assert config.program_length == 8
    assert config.max_modes_per_isotope == 7
    assert config.live_time_s == 45.0
    assert config.diagnostic_ranked_node_limit == 12
    assert config.lambda_distance == 0.0


def test_runtime_planner_defaults_to_pf_hard_cardinality_capacity() -> None:
    """Planner summaries must retain PF states above the ordinary K limit."""
    config = dss_config_from_pf_settings(
        {
            "pf_max_sources": 5,
            "pf_hard_max_sources": 8,
            "dss_pp": {"planning_method": "resample"},
        },
        runtime_owned_candidates=True,
    )

    assert config.max_modes_per_isotope == 8


def test_replay_planner_can_retain_estimator_candidate_augmentation() -> None:
    """Offline planner studies may retain PF-authored candidate augmentation."""
    config = dss_config_from_pf_settings(
        {
            "dss_pp": {
                "augment_candidates": True,
                "planning_method": "resample",
                "distance_weight": 0.4,
            }
        },
        runtime_owned_candidates=False,
    )

    assert config.augment_candidates is True
    assert config.lambda_distance == 0.4
