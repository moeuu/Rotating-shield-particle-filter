"""Tests for high-fidelity Geant4 runtime configuration defaults."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from measurement.kernels import ShieldParams
from measurement.detector_geometry import detector_active_radius_m
from measurement.detector_geometry import detector_observation_geometry_from_runtime_config
from measurement.detector_geometry import detector_outer_radius_cm
from measurement.observation_model import build_runtime_observation_model
from measurement.observation_model import continuous_kernel_from_observation_model
from measurement.obstacles import ObstacleGrid
from measurement.shielding import HVL_TVL_TABLE_MM, SHIELD_GEOMETRY_SPHERICAL_OCTANT
from pf.likelihood import (
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF,
    DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL,
    DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS,
    DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA,
    DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA,
    DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA,
)
from pf.profiles import enforce_pure_runtime_settings
from sim.geant4_app.app import Geant4AppConfig
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
    ExportedDetectorModel,
)
from sim.runtime import load_runtime_config
from sim.shield_geometry import (
    FE_SHIELD_INNER_RADIUS_M,
    FE_SHIELD_THICKNESS_M,
    PB_SHIELD_INNER_RADIUS_M,
    SHIELD_SHAPE_SPHERICAL_OCTANT,
    nested_shield_inner_radii_cm,
    resolve_shield_thickness_config,
)


def _geant4_runtime_config_paths() -> list[Path]:
    """Return Geant4 runtime configs, excluding calibration payloads."""
    root = Path(__file__).resolve().parents[1]
    return [
        path
        for path in sorted((root / "configs" / "geant4").glob("*.json"))
        if path.name != "net_response_calibration.json"
    ]


def _load_geant4_runtime_config(config_path: Path) -> dict[str, object]:
    """Load the effective pure-PF config after resolving inheritance."""
    return enforce_pure_runtime_settings(load_runtime_config(config_path))


def test_standard_runtime_uses_continuous_collision_checked_measurement_space() -> None:
    """The standard full simulation must plan over feasible continuous 3-D poses."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    assert payload["detector_height_sampling_mode"] == "continuous"
    assert "detector_height_actions_m" not in payload
    assert "detector_height_min_m" not in payload
    assert "detector_height_max_m" not in payload
    assert payload["measurement_pose_clearance_enabled"] is True
    assert float(payload["measurement_pose_clearance_margin_m"]) >= 0.0
    assert float(payload["measurement_route_grid_cell_size_m"]) == pytest.approx(
        0.25
    )
    assert int(payload["measurement_route_workers"]) > 1
    assert float(payload["detector_transport_height_m"]) >= 0.0


def test_geant4_configs_use_detector_cps_source_rate_by_default() -> None:
    """Geant4 configs should use detector cps@1m source-rate semantics."""
    forbidden_args = {
        "--min-histories-per-line",
        "--max-histories-per-line",
        "--no-poisson-background",
    }
    for config_path in _geant4_runtime_config_paths():
        payload = _load_geant4_runtime_config(config_path)
        executable_args = set(payload.get("executable_args", []))

        assert payload.get("engine_mode", "external") == "external"
        assert payload.get("persistent_process", False) is True
        assert payload.get("spectrum_count_method") == "response_poisson"
        assert payload.get("response_poisson_photopeak_fusion") is False
        assert payload.get("response_poisson_low_snr_photopeak_anchor") is True
        assert payload.get("response_poisson_low_snr_suppress_enable") is True
        assert payload.get("response_poisson_low_snr_suppress_count") is False
        assert (
            payload.get("precision_diagnostic_full_spectrum_response_enable", False)
            is False
        )
        assert (
            int(payload.get("precision_diagnostic_birth_candidate_log_limit", 0)) == 0
        )
        assert int(payload.get("precision_diagnostic_particle_log_limit", 0)) == 0
        assert (
            int(payload.get("surface_observability_diagnostic_candidates", 0)) == 0
        )
        assert (
            payload.get("response_poisson_crosstalk_count_guard_adjust_count", False)
            is True
        )
        assert (
            float(payload.get("pf_obstacle_source_extent_radius_m", 0.0))
            == pytest.approx(0.0)
        )
        assert int(payload.get("pf_obstacle_source_extent_samples", 1)) == 1
        suppress_fraction = float(
            payload.get("response_poisson_low_snr_suppress_fraction", 0.0)
        )
        assert suppress_fraction >= 0.15
        assert (
            float(
                payload.get(
                    "response_poisson_low_snr_suppress_photo_to_poisson_ratio",
                    0.0,
                )
            )
            <= 0.25
        )
        assert (
            1.3
            <= float(payload.get("response_poisson_crosstalk_count_guard_ratio", 9.0))
            <= 1.5
        )
        assert (
            1.0
            <= float(
                payload.get("response_poisson_crosstalk_count_guard_photo_snr", 9.0)
            )
            <= 2.0
        )
        assert payload.get("source_rate_model") == "detector_cps_1m"
        assert str(payload.get("physics_profile", "balanced")).lower() != "theory_tvl"
        assert not forbidden_args.intersection(executable_args)
        assert "net_response_calibration_path" not in payload
        assert "net_response_calibration" not in payload
        assert "response_poisson_truth_calibration_path" not in payload
        assert "response_poisson_truth_calibration" not in payload
        assert "measurement_scale_by_isotope" not in payload
        assert "measurement_scale_by_isotope_and_pair" not in payload
        assert payload.get("pf_line_resolved_shield_attenuation", True) is True
        assert payload.get("pf_obstacle_attenuation", True) is not False
        if "pf_count_likelihood" in payload:
            _assert_validated_geant4_count_likelihood(payload)
        if "response_poisson_count_variance_ceiling_enable" in payload:
            _assert_response_poisson_variance_ceiling(payload)
        assert payload["online_absent_isotope_pruning"] is False
        dss_pp = payload.get("dss_pp", {})
        if isinstance(dss_pp, dict):
            assert dss_pp.get("one_step_guard_enable", True) is True
            assert dss_pp.get("one_step_guard_use_gpu") is None
            observation_geometry = detector_observation_geometry_from_runtime_config(
                payload
            )
            assert int(
                dss_pp.get(
                    "detector_aperture_samples",
                    observation_geometry.aperture_samples,
                )
            ) == int(observation_geometry.aperture_samples)
        assert float(payload.get("scatter_gain", 0.0)) == 0.0
        source_bias_mode = str(payload.get("source_bias_mode", "detector_cone"))
        assert source_bias_mode == "detector_cone"
        if "high_fidelity" in config_path.name:
            assert payload.get("detector_scoring_mode") == "full_transport"
            assert payload.get("secondary_transport_mode") == "full_transport"
            assert float(payload.get("primary_sampling_fraction", 1.0)) == pytest.approx(1.0)
        else:
            assert payload.get("detector_scoring_mode") == "incident_gamma_energy"
            assert payload.get("secondary_transport_mode") == "gamma_only"
            assert 0.0 < float(payload.get("primary_sampling_fraction", 1.0)) < 1.0
            assert float(payload.get("source_bias_cone_half_angle_deg", 0.0)) >= 0.0


def _assert_validated_geant4_count_likelihood(payload: dict[str, object]) -> None:
    """Assert that configured Geant4 PF count uncertainty uses validated defaults."""
    pf_count_likelihood = payload.get("pf_count_likelihood", {})
    assert isinstance(pf_count_likelihood, dict)
    assert pf_count_likelihood["count_likelihood_model"] == (
        DEFAULT_GEANT4_COUNT_LIKELIHOOD_MODEL
    )
    assert float(pf_count_likelihood["count_likelihood_df"]) == pytest.approx(
        DEFAULT_GEANT4_COUNT_LIKELIHOOD_DF
    )
    assert float(pf_count_likelihood["transport_model_rel_sigma"]) == (
        pytest.approx(DEFAULT_GEANT4_TRANSPORT_MODEL_REL_SIGMA)
    )
    assert float(pf_count_likelihood["transport_model_abs_sigma"]) == (
        pytest.approx(DEFAULT_GEANT4_TRANSPORT_MODEL_ABS_SIGMA)
    )
    assert float(pf_count_likelihood["spectrum_count_rel_sigma"]) == (
        pytest.approx(DEFAULT_GEANT4_SPECTRUM_COUNT_REL_SIGMA)
    )
    assert float(pf_count_likelihood["spectrum_count_abs_sigma"]) == (
        pytest.approx(DEFAULT_GEANT4_SPECTRUM_COUNT_ABS_SIGMA)
    )
    assert float(pf_count_likelihood["low_count_abs_sigma"]) == pytest.approx(
        DEFAULT_GEANT4_LOW_COUNT_ABS_SIGMA
    )
    assert float(pf_count_likelihood["low_count_transition_counts"]) == (
        pytest.approx(DEFAULT_GEANT4_LOW_COUNT_TRANSITION_COUNTS)
    )


def _assert_validated_shield_contrast_likelihood(payload: dict[str, object]) -> None:
    """Assert that Geant4 runtime keeps same-station shield signatures active."""
    contrast = payload.get("pf_shield_contrast_likelihood", {})
    assert isinstance(contrast, dict)
    assert contrast["enabled"] is True
    assert float(contrast["weight"]) == pytest.approx(1.0)
    assert float(contrast["log_sigma_floor"]) == pytest.approx(0.5)
    assert float(contrast["log_sigma_ceiling"]) == pytest.approx(2.0)
    assert float(contrast["min_count"]) == pytest.approx(25.0)
    assert int(contrast["min_views"]) >= 2
    assert float(contrast["df"]) == pytest.approx(5.0)


def _assert_validated_shield_view_ratio_likelihood(payload: dict[str, object]) -> None:
    """Assert that Geant4 runtime uses conditional shield-view ratios."""
    ratio = payload.get("pf_shield_view_ratio_likelihood", {})
    assert isinstance(ratio, dict)
    assert ratio["enabled"] is True
    assert float(ratio["weight"]) == pytest.approx(1.0)
    assert float(ratio["concentration"]) == pytest.approx(128.0)
    assert float(ratio["min_total_count"]) == pytest.approx(25.0)
    assert int(ratio["min_views"]) >= 2


def _assert_pure_pf_estimator_boundary(payload: dict[str, object]) -> None:
    """Assert that a runtime config cannot activate a second batch estimator."""
    assert payload["estimator_profile"] == "pf_strict"
    forbidden = (
        "birth_global_rescue_enable",
        "birth_global_rescue_candidate_memory_enable",
        "conditional_strength_refit",
        "conditional_strength_profile_before_likelihood",
        "all_history_dictionary_proposal_enable",
        "candidate_verification_independent_evidence_enable",
        "candidate_verification_queue_enable",
        "final_absent_isotope_filter",
        "mode_preserving_report_cardinality_strata",
        "online_absent_isotope_pruning",
        "parallel_isotope_updates",
        "report_best_so_far_enable",
        "report_cluster_model_selection",
        "report_mle_rescue_enable",
        "report_model_order_prune_particles",
        "report_strength_refit",
        "report_surface_local_refine",
        "runtime_report_rescue_enable",
        "runtime_report_rescue_memory_enable",
        "sparse_poisson_evidence_authoritative",
        "sparse_poisson_evidence_enable",
        "sparse_poisson_joint_evidence_enable",
        "sparse_poisson_offgrid_refine_enable",
        "sparse_poisson_spectral_evidence_enable",
        "surface_map_reconstruction_enable",
    )
    for field in forbidden:
        assert payload[field] is False
    dss = payload["dss_pp"]
    assert dss["adaptive_program_length_enable"] is False
    assert dss["include_runtime_rescue_modes"] is False
    assert dss["include_global_surface_rescue_modes"] is False


def test_standard_full_simulation_count_likelihood_uncertainty_is_validated() -> None:
    """Standard full simulation should not use an over-broad count uncertainty."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    _assert_validated_geant4_count_likelihood(payload)
    _assert_validated_shield_contrast_likelihood(payload)
    _assert_validated_shield_view_ratio_likelihood(payload)
    _assert_pure_pf_estimator_boundary(payload)
    _assert_response_poisson_variance_ceiling(payload)


def test_guarded_full_simulation_config_forces_cpu_only_execution() -> None:
    """Guarded full simulation should preserve fidelity while avoiding CUDA use."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads_cpu_guarded.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    assert payload["use_gpu"] is False
    assert int(payload["thread_count"]) == 16
    assert int(payload["python_worker_count"]) == 16
    assert int(payload["pose_selection_workers"]) == 16
    assert int(payload["ig_workers"]) == 16
    assert int(payload["parallel_isotope_workers"]) == 16
    assert int(payload["structural_trial_workers"]) == 16
    assert payload["source_rate_model"] == "detector_cps_1m"
    assert payload["spectrum_count_method"] == "response_poisson"
    assert payload["pf_obstacle_attenuation"] is True


def _assert_response_poisson_variance_ceiling(payload: dict[str, object]) -> None:
    """Assert that runtime response-Poisson variances cannot become uninformative."""
    assert payload["response_poisson_count_variance_ceiling_enable"] is True
    assert float(payload["response_poisson_count_variance_max_rel_sigma"]) == (
        pytest.approx(0.15)
    )
    assert float(payload["response_poisson_count_variance_max_abs_sigma"]) == (
        pytest.approx(40.0)
    )
    assert (
        payload["response_poisson_count_variance_preserve_diagnostic_floors"]
        is True
    )
    assert payload["response_poisson_count_variance_preserve_guard_floors"] is True


def test_high_fidelity_external_config_uses_native_geometry() -> None:
    """The explicit high-fidelity external config should use balanced native transport."""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "geant4" / "high_fidelity_external_no_isaac.json"
    payload = _load_geant4_runtime_config(config_path)

    assert payload["engine_mode"] == "external"
    assert payload["physics_profile"] == "balanced"
    assert payload["start_isaacsim_sidecar_with_geant4"] is False
    assert payload["author_obstacle_prims"] is True
    assert payload["source_rate_model"] == "detector_cps_1m"
    assert payload["source_bias_mode"] == "detector_cone"
    assert payload["source_bias_isotropic_fraction"] == pytest.approx(1.0)
    assert payload["detector_scoring_mode"] == "full_transport"
    assert payload["secondary_transport_mode"] == "full_transport"
    assert payload["source_surface_prior"] is True
    assert payload["pf_line_resolved_shield_attenuation"] is True
    assert payload["pf_obstacle_attenuation"] is True
    assert payload["joint_observation_update"] is True
    assert payload["delayed_resample_update"] is False
    assert int(payload["mission_stop_soft_extension_poses"]) == 2
    assert payload["random_source_visibility_filter"] is True
    assert float(payload["random_source_min_visible_fraction"]) > 0.0
    assert int(payload["random_source_max_ceiling_sources"]) == 1
    assert float(payload["random_source_preferred_max_z_m"]) <= 5.0
    assert float(payload["random_source_same_isotope_min_distance_m"]) >= 2.0
    assert payload["random_source_response_observability_filter"] is True
    assert float(payload["random_source_response_max_pairwise_corr"]) <= 0.99
    assert float(payload["random_source_response_condition_max"]) >= 1.0
    assert int(payload["thread_count"]) == 32
    assert int(payload["python_worker_count"]) == 32
    assert int(payload["pose_selection_workers"]) == 32
    assert int(payload["ig_workers"]) == 32
    assert int(payload["parallel_isotope_workers"]) == 32
    assert int(payload["structural_trial_workers"]) == 32
    assert int(payload["structural_trial_parallel_min_trials"]) <= 8
    assert int(payload["dss_pp"]["program_eval_workers"]) == 32
    assert int(payload["birth_min_distinct_stations"]) >= 2
    assert int(payload["birth_min_distinct_poses"]) >= 5
    assert float(payload["birth_existing_response_corr_max"]) <= 0.99
    assert float(payload["pseudo_source_temporal_sep_min"]) > 0.0
    _assert_pure_pf_estimator_boundary(payload)
    assert float(payload["random_source_intensity_min_cps_1m"]) >= 3.0e5
    assert float(payload["random_source_intensity_max_cps_1m"]) >= 2.0e6
    assert float(payload["birth_q_max"]) >= float(
        payload["random_source_intensity_max_cps_1m"]
    )
    assert payload["high_strength_split_enable"] is True
    assert float(payload["high_strength_split_q_multiple"]) >= 1.0
    assert payload["weak_source_prune_require_observable"] is True
    assert int(payload["weak_source_prune_min_observable_measurements"]) >= 1
    assert payload["cardinality_preserving_resample"] is True
    assert int(payload["cardinality_preserving_min_stations"]) == 0
    assert payload["cardinality_preserving_require_confirmed_structure"] is False
    assert float(payload["source_strength_prior_mean"]) == 0.0
    assert float(payload["source_strength_prior_weight"]) == 0.0
    assert float(payload["source_strength_observation_overshoot_penalty_weight"]) > 0.0
    assert float(payload["weak_source_prune_visibility_reference_strength"]) == 0.0
    assert payload["mode_preserving_resample"] is True
    assert int(payload["mode_preserving_max_modes"]) >= 12
    assert int(payload["mode_preserving_particles_per_mode"]) >= 8
    assert float(payload["mode_preserving_min_weight_fraction"]) == pytest.approx(0.0)
    assert int(payload["mode_preserving_high_surface_extra_particles"]) >= 1
    assert payload["mode_preserving_cardinality_strata"] is True
    assert int(payload["mode_preserving_min_particles_per_cardinality"]) >= 4
    assert payload["split_residual_guided"] is True
    assert payload["split_residual_always_try"] is True
    assert int(payload["split_residual_candidate_count"]) >= 8
    assert payload["remaining_measurement_estimate"]["enabled"] is True
    assert (
        float(
            payload["remaining_measurement_estimate"][
                "report_response_correlation_weight"
            ]
        )
        > 0.0
    )
    assert (
        float(payload["remaining_measurement_estimate"]["report_residual_weight"])
        > 0.0
    )
    assert (
        float(
            payload["remaining_measurement_estimate"][
                "high_surface_ambiguity_weight"
            ]
        )
        > 0.0
    )
    assert (
        int(
            payload["remaining_measurement_estimate"][
                "residual_surface_gain_candidate_limit"
            ]
        )
        >= 512
    )
    assert float(payload["dss_pp"]["high_surface_pair_boost"]) > 1.0
    assert float(payload["dss_pp"]["high_surface_cross_stratum_boost"]) > 1.0
    assert float(payload["dss_pp"]["correlation_reduction_weight"]) > 0.0
    assert float(payload["dss_pp"]["isotope_balance_weight"]) > 0.0
    assert "executable_args" not in payload


def test_default_config_uses_detector_cps_source_rate() -> None:
    """The practical runtime default should use detector cps@1m semantics."""
    config = Geant4AppConfig.from_dict({})

    assert config.source_rate_model == "detector_cps_1m"
    assert config.source_bias_mode == "detector_cone"
    assert config.source_bias_isotropic_fraction == pytest.approx(0.1)
    assert config.source_bias_cone_half_angle_deg == pytest.approx(0.0)
    assert config.detector_scoring_mode == "full_transport"
    assert config.secondary_transport_mode == "full_transport"
    assert config.primary_sampling_fraction == pytest.approx(1.0)


def test_variance_reduction_config_is_explicit_weighted_mode() -> None:
    """The named variance-reduction config should document the default mode."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)
    config = Geant4AppConfig.from_dict(payload)

    assert config.source_rate_model == "detector_cps_1m"
    assert config.source_bias_mode == "detector_cone"
    assert config.source_bias_isotropic_fraction == pytest.approx(0.1)
    assert config.source_bias_cone_half_angle_deg == pytest.approx(0.0)
    assert config.physics_profile == "balanced"
    assert config.detector_scoring_mode == "incident_gamma_energy"
    assert config.secondary_transport_mode == "gamma_only"
    assert config.primary_sampling_fraction == pytest.approx(0.02)
    assert payload["source_surface_prior"] is True
    assert payload["pf_obstacle_attenuation"] is True
    assert payload["joint_observation_update"] is True
    assert payload["delayed_resample_update"] is False
    assert payload["random_source_visibility_filter"] is True
    assert float(payload["random_source_min_visible_fraction"]) > 0.0
    assert int(payload["random_source_max_ceiling_sources"]) == 1
    assert float(payload["random_source_preferred_max_z_m"]) <= 5.0
    assert float(payload["random_source_same_isotope_min_distance_m"]) >= 2.0
    assert payload["random_source_response_observability_filter"] is True
    assert float(payload["random_source_response_max_pairwise_corr"]) <= 0.99
    assert float(payload["random_source_response_condition_max"]) >= 1.0
    assert int(payload["python_worker_count"]) == 32
    assert int(payload["pose_selection_workers"]) == 32
    assert int(payload["ig_workers"]) == 32
    assert int(payload["parallel_isotope_workers"]) == 32
    assert int(payload["structural_trial_workers"]) == 32
    assert int(payload["structural_trial_parallel_min_trials"]) <= 8
    assert int(payload["dss_pp"]["program_eval_workers"]) == 32
    assert int(payload["birth_min_distinct_stations"]) >= 2
    assert int(payload["birth_min_distinct_poses"]) >= 5
    assert float(payload["birth_existing_response_corr_max"]) <= 0.99
    assert float(payload["pseudo_source_temporal_sep_min"]) > 0.0
    _assert_pure_pf_estimator_boundary(payload)
    assert float(payload["random_source_intensity_min_cps_1m"]) >= 3.0e5
    assert float(payload["random_source_intensity_max_cps_1m"]) >= 2.0e6
    assert float(payload["birth_q_max"]) >= float(
        payload["random_source_intensity_max_cps_1m"]
    )
    assert payload["high_strength_split_enable"] is True
    assert payload["weak_source_prune_require_observable"] is True
    assert int(payload["weak_source_prune_min_observable_measurements"]) >= 1
    assert payload["cardinality_preserving_resample"] is True
    assert int(payload["cardinality_preserving_min_stations"]) == 0
    assert payload["cardinality_preserving_require_confirmed_structure"] is False
    assert float(payload["source_strength_prior_mean"]) == 0.0
    assert float(payload["source_strength_prior_weight"]) == 0.0
    assert float(payload["source_strength_observation_overshoot_penalty_weight"]) > 0.0
    assert float(payload["weak_source_prune_visibility_reference_strength"]) == 0.0
    assert payload["split_residual_guided"] is True
    assert payload["split_residual_always_try"] is True
    assert int(payload["split_residual_candidate_count"]) >= 8
    assert payload["remaining_measurement_estimate"]["enabled"] is True
    assert (
        float(
            payload["remaining_measurement_estimate"][
                "report_response_correlation_weight"
            ]
        )
        > 0.0
    )
    assert (
        float(payload["remaining_measurement_estimate"]["report_residual_weight"])
        > 0.0
    )
    assert (
        float(
            payload["remaining_measurement_estimate"][
                "high_surface_ambiguity_weight"
            ]
        )
        > 0.0
    )
    assert (
        int(
            payload["remaining_measurement_estimate"][
                "residual_surface_gain_candidate_limit"
            ]
        )
        >= 512
    )
    assert int(payload["mission_stop_max_poses"]) == 20
    assert payload["mission_stop_require_model_order_ready"] is False
    assert payload["mission_stop_require_remaining_measurement_ready"] is True
    assert payload["mission_stop_soft_extend_on_unresolved"] is False
    assert int(payload["mission_stop_soft_extension_poses"]) == 2
    assert payload["mission_stop_require_pf_convergence_for_coverage"] is False
    assert payload["mission_stop_report_simple_enable"] is False
    assert payload["dss_pp"]["adaptive_program_length_enable"] is False
    assert int(payload["dss_pp"]["adaptive_simple_program_length"]) <= 2
    assert payload["dss_pp"]["same_isotope_direct_separation_guard"] is True
    assert float(payload["dss_pp"]["high_surface_pair_boost"]) > 1.0
    assert float(payload["dss_pp"]["high_surface_cross_stratum_boost"]) > 1.0
    assert float(payload["dss_pp"]["temporal_separation_weight"]) >= 8.0
    assert float(payload["dss_pp"]["correlation_reduction_weight"]) > 0.0
    assert float(payload["dss_pp"]["isotope_balance_weight"]) > 0.0
    assert float(payload["dss_pp"]["coverage_weight"]) <= 2.0
    assert payload["mode_preserving_resample"] is True
    assert int(payload["mode_preserving_max_modes"]) >= 12
    assert int(payload["mode_preserving_particles_per_mode"]) >= 8
    assert float(payload["mode_preserving_min_weight_fraction"]) == pytest.approx(0.0)
    assert int(payload["mode_preserving_high_surface_extra_particles"]) >= 1
    assert payload["mode_preserving_cardinality_strata"] is True
    assert int(payload["mode_preserving_min_particles_per_cardinality"]) >= 4


def test_gui_config_matches_standard_cui_except_isaacsim_sidecar() -> None:
    """The default GUI runtime should differ from CUI only by Isaac Sim controls."""
    root = Path(__file__).resolve().parents[1]
    standard_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    gui_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_gui_32threads.json"
    )
    standard_payload = _load_geant4_runtime_config(standard_path)
    gui_payload = _load_geant4_runtime_config(gui_path)
    gui_only_keys = {
        "start_isaacsim_sidecar_with_geant4",
        "isaacsim_sidecar_config_path",
        "isaacsim_sidecar_python_env",
        "isaacsim_sidecar_startup_timeout_s",
        "isaacsim_keep_sidecar_alive",
    }

    assert standard_payload["start_isaacsim_sidecar_with_geant4"] is False
    assert gui_payload["start_isaacsim_sidecar_with_geant4"] is True
    assert {
        key: value
        for key, value in gui_payload.items()
        if key not in gui_only_keys
    } == {
        key: value
        for key, value in standard_payload.items()
        if key not in gui_only_keys
    }


def test_standard_variance_reduction_loads_transport_response_model() -> None:
    """Standard CUI/GUI runtimes should load the shared PF transport model."""
    root = Path(__file__).resolve().parents[1]
    standard_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    gui_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_gui_32threads.json"
    )
    high_fidelity_path = (
        root
        / "configs"
        / "geant4"
        / "high_fidelity_external_no_isaac.json"
    )
    standard_payload = _load_geant4_runtime_config(standard_path)
    gui_payload = _load_geant4_runtime_config(gui_path)
    high_fidelity_payload = _load_geant4_runtime_config(high_fidelity_path)

    model_relpath = standard_payload.get("pf_transport_response_model_path")
    assert isinstance(model_relpath, str)
    assert model_relpath.startswith("configs/geant4/calibration/")
    assert (root / model_relpath).exists()
    assert gui_payload.get("pf_transport_response_model_path") == model_relpath
    assert "pf_transport_response_model_path" not in high_fidelity_payload

    observation_model = build_runtime_observation_model(
        standard_payload,
        isotopes=("Cs-137", "Co-60", "Eu-154"),
    )
    kernel = continuous_kernel_from_observation_model(
        observation_model,
        obstacle_grid=None,
        use_gpu=False,
    )
    model = kernel.transport_response_model

    assert model is not None
    assert model.get("enabled") is True
    assert set(model.get("by_isotope", {})) >= {"Cs-137", "Co-60", "Eu-154"}
    assert "tau_shield" in model.get("feature_semantics", {})
    assert "optical-depth" in str(model.get("model", ""))
    for payload in model.get("by_isotope", {}).values():
        caps = payload.get("tau_feature_caps", {})
        assert caps.get("shield") == pytest.approx(3.5)
        assert caps.get("distance_shield") == pytest.approx(8.0)


def test_standard_transport_response_model_does_not_stack_legacy_scales() -> None:
    """Standard transport-response runtime should not double-apply old scales."""
    root = Path(__file__).resolve().parents[1]
    standard_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    gui_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_gui_32threads.json"
    )
    for config_path in (standard_path, gui_path):
        payload = _load_geant4_runtime_config(config_path)

        assert payload.get("pf_transport_response_model_path")
        assert "measurement_scale_by_isotope" not in payload
        assert "measurement_scale_by_isotope_and_pair" not in payload


def test_standard_strict_profile_disables_strength_profiling() -> None:
    """Standard runtime must keep conditional strength optimization disabled."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    assert payload.get("conditional_strength_refit") is False
    assert payload.get("conditional_strength_profile_before_likelihood") is False
    assert payload.get("conditional_strength_refit_reweight") is False


def test_standard_full_simulation_disables_surface_reconstruction() -> None:
    """Standard final output must come from PF posterior particles only."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    assert payload.get("surface_map_reconstruction_enable") is False


def test_standard_runtime_declares_reproducible_evaluation_bins() -> None:
    """Evaluation thresholds should be explicit in the standard runtime config."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)

    assert payload.get("evaluation_close_pair_distance_m") == pytest.approx(2.0)
    assert payload.get(
        "evaluation_close_pair_min_estimated_separation_m"
    ) == pytest.approx(0.5)
    assert payload.get("evaluation_cluster_match_gate_m") == pytest.approx(0.5)
    assert payload.get("evaluation_cluster_stability_window") == 5
    assert payload.get("evaluation_count_regime_lower_edges") == [
        0.0,
        10.0,
        100.0,
        1000.0,
    ]


def test_standard_variance_reduction_uses_conservative_line_basis_margin() -> None:
    """Standard response-Poisson runtime should avoid weak line-basis wins."""
    root = Path(__file__).resolve().parents[1]
    standard_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    gui_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_gui_32threads.json"
    )
    high_fidelity_path = (
        root
        / "configs"
        / "geant4"
        / "high_fidelity_external_no_isaac.json"
    )
    standard_payload = _load_geant4_runtime_config(standard_path)
    gui_payload = _load_geant4_runtime_config(gui_path)
    high_fidelity_payload = _load_geant4_runtime_config(high_fidelity_path)

    assert standard_payload.get("response_poisson_line_resolved_fit") is True
    assert float(
        standard_payload.get("response_poisson_line_resolved_bic_margin", 0.0)
    ) >= 3500.0
    assert (
        gui_payload.get("response_poisson_line_resolved_bic_margin")
        == standard_payload.get("response_poisson_line_resolved_bic_margin")
    )
    assert "response_poisson_line_resolved_bic_margin" not in high_fidelity_payload


def test_transport_response_model_path_resolves_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repo-relative transport model paths should work outside the repo cwd."""
    monkeypatch.chdir(tmp_path)

    observation_model = build_runtime_observation_model(
        {
            "detector_model": {"crystal_radius_m": 0.038},
            "pf_transport_response_model_path": (
                "configs/geant4/calibration/"
                "pf_transport_response_model_dominanceguard_transport_20260608.json"
            ),
        },
        isotopes=("Cs-137", "Co-60", "Eu-154"),
    )

    assert observation_model.transport_response_model is not None
    assert observation_model.transport_response_model.get("enabled") is True


def test_geant4_configs_use_large_detector_model() -> None:
    """Runtime configs should use the large spherical CeBr3 detector."""
    for config_path in _geant4_runtime_config_paths():
        payload = _load_geant4_runtime_config(config_path)
        detector = payload.get("detector_model", {})

        assert detector["crystal_radius_m"] == pytest.approx(DEFAULT_DETECTOR_CRYSTAL_RADIUS_M)
        assert detector["crystal_length_m"] == pytest.approx(DEFAULT_DETECTOR_CRYSTAL_LENGTH_M)
        assert detector["crystal_shape"] == "sphere"


def test_geant4_default_detector_model_matches_native_sidecar() -> None:
    """Python defaults and native fallback defaults should describe the same detector."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")
    model = ExportedDetectorModel()
    config = Geant4AppConfig.from_dict({})

    assert model.crystal_radius_m == pytest.approx(DEFAULT_DETECTOR_CRYSTAL_RADIUS_M)
    assert model.crystal_length_m == pytest.approx(DEFAULT_DETECTOR_CRYSTAL_LENGTH_M)
    assert config.detector_model.crystal_radius_m == pytest.approx(
        DEFAULT_DETECTOR_CRYSTAL_RADIUS_M
    )
    assert config.detector_model.crystal_length_m == pytest.approx(
        DEFAULT_DETECTOR_CRYSTAL_LENGTH_M
    )
    assert model.crystal_shape == "sphere"
    assert config.detector_model.crystal_shape == "sphere"
    assert model.active_volume_m3 == pytest.approx(
        (4.0 / 3.0) * math.pi * DEFAULT_DETECTOR_CRYSTAL_RADIUS_M**3
    )
    assert "constexpr double kDefaultCrystalRadiusM = 0.038;" in source
    assert "constexpr double kDefaultCrystalLengthM = 0.076;" in source
    assert 'std::string crystal_shape = "sphere";' in source
    assert "constexpr double kDefaultFeShieldInnerRadiusM = kDefaultShieldContactRadiusM;" in source
    assert (
        "constexpr double kDefaultPbShieldInnerRadiusM =\n"
        "    kDefaultFeShieldInnerRadiusM + kDefaultFeShieldThicknessM;"
        in source
    )
    assert FE_SHIELD_INNER_RADIUS_M == pytest.approx(
        DEFAULT_DETECTOR_CRYSTAL_RADIUS_M + 0.0015
    )
    assert PB_SHIELD_INNER_RADIUS_M == pytest.approx(
        FE_SHIELD_INNER_RADIUS_M + FE_SHIELD_THICKNESS_M
    )


def test_pf_and_geant4_share_spherical_octant_shield_geometry() -> None:
    """PF shield likelihoods should use the same spherical-octant shell geometry."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = _load_geant4_runtime_config(config_path)
    detector_model = ExportedDetectorModel()
    shield_thickness = resolve_shield_thickness_config(payload)
    detector_outer_radius_cm_value = detector_outer_radius_cm(
        {
            "crystal_radius_m": detector_model.crystal_radius_m,
            "housing_thickness_m": detector_model.housing_thickness_m,
        }
    )
    fe_inner_cm, pb_inner_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=shield_thickness.thickness_fe_cm,
        detector_outer_radius_cm=detector_outer_radius_cm_value,
    )
    pf_params = ShieldParams(
        thickness_fe_cm=shield_thickness.thickness_fe_cm,
        thickness_pb_cm=shield_thickness.thickness_pb_cm,
        inner_radius_fe_cm=fe_inner_cm,
        inner_radius_pb_cm=pb_inner_cm,
    )

    assert SHIELD_SHAPE_SPHERICAL_OCTANT == SHIELD_GEOMETRY_SPHERICAL_OCTANT
    assert pf_params.shield_geometry_model == SHIELD_GEOMETRY_SPHERICAL_OCTANT
    assert pf_params.use_angle_attenuation is False
    assert pf_params.inner_radius_fe_cm == pytest.approx(
        100.0 * (DEFAULT_DETECTOR_CRYSTAL_RADIUS_M + DEFAULT_DETECTOR_HOUSING_THICKNESS_M)
    )
    assert pf_params.inner_radius_pb_cm == pytest.approx(
        pf_params.inner_radius_fe_cm + pf_params.thickness_fe_cm
    )


def test_spectrum_validation_uses_runtime_shield_inner_radii() -> None:
    """Validation PF targets should derive shield radii from runtime detector geometry."""
    payload = {
        "detector_model": {
            "crystal_radius_m": 0.05,
            "housing_thickness_m": 0.002,
        },
        "shield_transmission_target": 0.2,
    }
    shield_thickness = resolve_shield_thickness_config(payload)
    expected_fe_cm, expected_pb_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=shield_thickness.thickness_fe_cm,
        detector_outer_radius_cm=5.2,
    )

    params = build_runtime_observation_model(
        payload,
        isotopes=("Cs-137",),
    ).shield_params

    assert params.inner_radius_fe_cm == pytest.approx(expected_fe_cm)
    assert params.inner_radius_pb_cm == pytest.approx(expected_pb_cm)
    assert params.inner_radius_pb_cm == pytest.approx(
        params.inner_radius_fe_cm + params.thickness_fe_cm
    )


def test_detector_count_radius_excludes_housing() -> None:
    """PF count geometry should use active crystal radius, not housing radius."""
    detector_model = {
        "crystal_radius_m": 0.05,
        "housing_thickness_m": 0.002,
    }

    assert detector_active_radius_m(detector_model) == pytest.approx(0.05)
    assert detector_outer_radius_cm(detector_model) == pytest.approx(5.2)


def test_detector_observation_geometry_splits_count_and_aperture_radii() -> None:
    """PF count geometry and ray aperture geometry should match Geant4 semantics."""
    runtime_config = {
        "detector_model": {
            "crystal_radius_m": 0.05,
            "housing_thickness_m": 0.002,
        },
        "pf_detector_aperture_samples": 33,
    }

    geometry = detector_observation_geometry_from_runtime_config(runtime_config)

    assert geometry.count_radius_m == pytest.approx(0.05)
    assert geometry.aperture_radius_m == pytest.approx(0.052)
    assert geometry.aperture_samples == 33
    assert geometry.aperture_sampling == "solid_angle_cone"


def test_runtime_observation_model_builds_shared_continuous_kernel() -> None:
    """Runtime PF and planning should build kernels from one observation model."""
    runtime_config = {
        "detector_model": {
            "crystal_radius_m": 0.05,
            "housing_thickness_m": 0.002,
        },
        "pf_detector_aperture_samples": 33,
        "pf_buildup": {
            "fe_coeff": 0.1,
            "pb_coeff": 0.2,
            "obstacle_coeff": 0.3,
        },
        "pf_line_resolved_shield_attenuation": True,
        "pf_transport_response_model": {
            "enabled": True,
            "by_isotope": {
                "Cs-137": {
                    "scale": 0.95,
                    "tau_coefficients": {"shield": 0.1, "obstacle": -0.2},
                }
            },
        },
    }

    model = build_runtime_observation_model(
        runtime_config,
        isotopes=("Cs-137", "Co-60"),
    )
    kernel = continuous_kernel_from_observation_model(
        model,
        obstacle_grid=None,
        use_gpu=False,
    )

    assert model.detector_geometry.count_radius_m == pytest.approx(0.05)
    assert model.detector_geometry.aperture_radius_m == pytest.approx(0.052)
    assert model.detector_geometry.aperture_samples == 33
    assert model.detector_geometry.aperture_sampling == "solid_angle_cone"
    assert kernel.detector_radius_m == pytest.approx(model.detector_geometry.count_radius_m)
    assert kernel.detector_aperture_radius_m == pytest.approx(
        model.detector_geometry.aperture_radius_m
    )
    assert kernel.detector_aperture_samples == model.detector_geometry.aperture_samples
    assert kernel.detector_aperture_sampling == model.detector_geometry.aperture_sampling
    assert kernel.shield_params.buildup_fe_coeff == pytest.approx(0.1)
    assert kernel.shield_params.buildup_pb_coeff == pytest.approx(0.2)
    assert kernel.obstacle_buildup_coeff == pytest.approx(0.0)
    assert model.obstacle_buildup_coeff == pytest.approx(0.3)
    assert kernel.line_mu_by_isotope is not None
    assert kernel.transport_response_model == model.transport_response_model


def test_runtime_observation_model_loads_transport_response_model_path(
    tmp_path: Path,
) -> None:
    """PF and planning kernels should load one shared transport model file."""
    model_payload = {
        "pf_transport_response_model": {
            "enabled": True,
            "by_isotope": {
                "Cs-137": {
                    "scale": 1.05,
                    "tau_coefficients": {"fe": 0.1},
                }
            },
        }
    }
    model_path = tmp_path / "transport_response_model.json"
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    runtime_config = {
        "detector_model": {"crystal_radius_m": 0.05},
        "pf_transport_response_model_path": str(model_path),
    }
    model = build_runtime_observation_model(
        runtime_config,
        isotopes=("Cs-137",),
    )
    kernel = continuous_kernel_from_observation_model(
        model,
        obstacle_grid=None,
        use_gpu=False,
    )

    assert model.transport_response_model == model_payload[
        "pf_transport_response_model"
    ]
    assert kernel.transport_response_model == model.transport_response_model


def test_runtime_observation_model_resolves_obstacle_material_mu() -> None:
    """Runtime obstacle material should feed the shared ContinuousKernel."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    model = build_runtime_observation_model(
        {"obstacle_material": "air"},
        isotopes=("Cs-137",),
    )
    kernel = continuous_kernel_from_observation_model(
        model,
        obstacle_grid=grid,
        use_gpu=False,
    )

    assert model.obstacle_mu_by_isotope is not None
    assert 0.0 < model.obstacle_mu_by_isotope["Cs-137"] < 0.001
    assert kernel.obstacle_mu_cm_inv("Cs-137") == pytest.approx(
        model.obstacle_mu_by_isotope["Cs-137"]
    )


def test_runtime_observation_model_uses_explicit_obstacle_mu_override() -> None:
    """Explicit PF obstacle-mu config should override material defaults."""
    grid = ObstacleGrid(
        origin=(0.0, -0.5),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    model = build_runtime_observation_model(
        {
            "obstacle_material": "air",
            "pf_obstacle_mu_by_isotope": {"Cs-137": 0.025},
        },
        isotopes=("Cs-137",),
    )
    kernel = continuous_kernel_from_observation_model(
        model,
        obstacle_grid=grid,
        use_gpu=False,
    )

    assert model.obstacle_mu_by_isotope is not None
    assert model.obstacle_mu_by_isotope["Cs-137"] == pytest.approx(0.025)
    assert kernel.obstacle_mu_cm_inv("Cs-137") == pytest.approx(0.025)


def test_native_sidecar_detector_crystal_is_spherical() -> None:
    """The native detector crystal and housing should be Geant4 spheres."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert 'new G4Sphere(\n            "DetectorCrystalSolid"' in source
    assert 'new G4Sphere(\n            "DetectorHousingSolid"' in source
    assert "G4Tubs" not in source


def test_native_sidecar_does_not_expose_history_weighting_shortcuts() -> None:
    """The native sidecar should not expose capped history shortcuts."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")
    forbidden = (
        "--min-histories-per-line",
        "--max-histories-per-line",
        "--no-poisson-background",
        "ResolveHistoryCount",
    )

    for token in forbidden:
        assert token not in source


def test_native_sidecar_uses_physical_detector_deposit_pulses() -> None:
    """Native Geant4 spectra should keep physical deposits available for high fidelity."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert "DetectorEfficiency" not in source
    assert "spectrum[index] += deposit.weight;" in source
    assert 'std::string source_bias_mode = "detector_cone";' in source
    assert 'std::string source_rate_model = "detector_cps_1m";' in source
    assert "double source_bias_isotropic_fraction = 0.1;" in source
    assert 'result.metadata["isotropic_mixture_fraction"]' in source
    assert 'result.metadata["cone_half_angle_deg"]' in source
    assert "{1596.5, 0.02}" in source


def test_native_sidecar_exposes_detector_cps_source_rate_model() -> None:
    """Native Geant4 should not convert detector cps@1m through area acceptance."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "--source-rate-model",
        "NormalizeSourceRateModel",
        "detector_cps_1m",
        "detector_equivalent_cone",
        "DetectorCpsGeometryScale",
        "SphereSolidAngleFraction",
        'result.metadata["source_rate_model"]',
        'result.metadata["intensity_cps_1m_definition"]',
        "const double source_rate_scale = detector_cps_rate_model",
    ):
        assert token in source


def test_native_theory_tvl_table_matches_python_shielding_constants() -> None:
    """Native theory-TVL fallback should mirror the Python/PF shielding table."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")
    pattern = re.compile(
        r'if \(isotope == "([^"]+)"\) \{\s*'
        r"return is_fe \? ([0-9.]+) : ([0-9.]+);\s*"
        r"\}",
        re.MULTILINE,
    )
    native_table = {
        match.group(1): {"fe": float(match.group(2)), "pb": float(match.group(3))}
        for match in pattern.finditer(source)
    }

    for isotope, material_table in HVL_TVL_TABLE_MM.items():
        assert native_table[isotope]["fe"] == pytest.approx(
            float(material_table["fe"]["tvl"])
        )
        assert native_table[isotope]["pb"] == pytest.approx(
            float(material_table["pb"]["tvl"])
        )


def test_native_sidecar_exposes_fast_detector_scoring_mode() -> None:
    """Native Geant4 should expose an explicit fast detector scoring mode."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "--detector-scoring-mode",
        "NormalizeDetectorScoringMode",
        "incident_gamma_energy",
        'result.metadata["detector_scoring_mode"]',
        'result.metadata["detector_fast_scoring"]',
        'result.metadata["detector_crystal_radius_m"]',
        'result.metadata["detector_housing_thickness_m"]',
        'result.metadata["detector_target_radius_m"]',
        'result.metadata["reference_detector_acceptance"]',
        'result.metadata["fe_shield_normal_x"]',
        'result.metadata["pb_shield_normal_x"]',
    ):
        assert token in source


def test_native_sidecar_exposes_gamma_only_secondary_transport_mode() -> None:
    """Native Geant4 should expose explicit gamma-only secondary transport."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "--secondary-transport-mode",
        "NormalizeSecondaryTransportMode",
        "SecondaryTransportStackingAction",
        "gamma_only",
        'result.metadata["secondary_transport_mode"]',
        'result.metadata["killed_non_gamma_secondary_count"]',
    ):
        assert token in source


def test_native_sidecar_exposes_unbiased_primary_sampling_fraction() -> None:
    """Native Geant4 should expose primary thinning only as an explicit weighted mode."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "--primary-sampling-fraction",
        "primary_history_weight",
        'result.metadata["primary_sampling_fraction"]',
        'result.metadata["expected_sampled_primaries"]',
    ):
        assert token in source


def test_native_sidecar_updates_shield_pose_without_geometry_rebuild() -> None:
    """Shield rotations should be runtime-updated instead of cache-busting."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert "UpdateShieldPoses" in source
    assert "UpdatePhysicalPose" in source
    assert "detector_construction_->UpdateShieldPoses(request)" in source
    assert 'request.fe_pose.qw << "," << request.fe_pose.qx' not in source
    assert 'request.pb_pose.qw << "," << request.pb_pose.qx' not in source


def test_native_sidecar_reports_transport_diagnostics() -> None:
    """Native Geant4 metadata should expose transport/decomposition diagnostics."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "TransportSteppingAction",
        "GetProcessDefinedStep",
        "GetSecondaryInCurrentStep",
        'result.metadata["total_track_steps"]',
        'result.metadata["detector_hit_events"]',
        'result.metadata["transport_uncollided_primary_counts_" + item.first]',
        'result.metadata["transport_interacted_primary_counts_" + item.first]',
        'result.metadata["transport_secondary_counts_" + item.first]',
        'result.metadata["transport_non_uncollided_fraction_" + item.first]',
        'result.metadata["source_equivalent_counts_" + item.first]',
        'transport_detected_counts_by_source',
        'transport_detected_counts_by_line',
        'transport_uncollided_primary_counts_by_source',
        'transport_uncollided_primary_counts_by_line',
        'source_equivalent_counts_by_source',
        'source_equivalent_counts_by_line',
        "AddShieldAxisMetadata(result, \"fe\", request.fe_pose)",
        "AddShieldAxisMetadata(result, \"pb\", request.pb_pose)",
        '"_shield_axis_"',
        "AddGeant4MaterialMuMetadata(result, scene_)",
        "ComputeGammaAttenuationLength",
        '"geant4_mu_cm_inv_"',
        'result.metadata["process_count_compton"]',
        'result.metadata["process_count_rayleigh"]',
        'result.metadata["process_count_photoelectric"]',
        'result.metadata["volume_step_counts"]',
        'result.metadata["primaries_per_sec"]',
        'result.metadata["effective_entries_per_sec"]',
        'result.metadata["total_spectrum_counts"]',
    ):
        assert token in source


def test_native_sidecar_classifies_detector_entry_transport_history() -> None:
    """Fast detector scoring should separate uncollided and interacted entries."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    for token in (
        "enum class DetectorEntryClass",
        "kUncollidedPrimary",
        "kInteractedPrimary",
        "kSecondary",
        "TrackHadGammaInteraction",
        "ClassifyDetectorEntryTrack",
        "process_key != \"transportation\"",
        "SetUserAction(new EventAction(event_store_, diagnostics_))",
    ):
        assert token in source
