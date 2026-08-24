"""Tests for estimator-owned DSS configuration translation."""

from __future__ import annotations

import pytest
from runtime.experiment_profiles import AcquisitionContract

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
