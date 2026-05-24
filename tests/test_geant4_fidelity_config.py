"""Tests for high-fidelity Geant4 runtime configuration defaults."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from measurement.shielding import HVL_TVL_TABLE_MM
from sim.geant4_app.app import Geant4AppConfig
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    ExportedDetectorModel,
)
from sim.runtime import load_runtime_config
from sim.shield_geometry import (
    FE_SHIELD_INNER_RADIUS_M,
    FE_SHIELD_THICKNESS_M,
    PB_SHIELD_INNER_RADIUS_M,
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
    """Load a Geant4 runtime config after resolving optional inheritance."""
    return load_runtime_config(config_path)


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
        assert payload.get("source_rate_model") == "detector_cps_1m"
        assert str(payload.get("physics_profile", "balanced")).lower() != "theory_tvl"
        assert not forbidden_args.intersection(executable_args)
        assert "net_response_calibration_path" not in payload
        assert "net_response_calibration" not in payload
        assert payload.get("pf_obstacle_attenuation", True) is not False
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
    assert payload["pf_obstacle_attenuation"] is True
    assert payload["joint_observation_update"] is True
    assert payload["delayed_resample_update"] is False
    assert payload["random_source_visibility_filter"] is True
    assert float(payload["random_source_min_visible_fraction"]) > 0.0
    assert int(payload["thread_count"]) == 32
    assert int(payload["python_worker_count"]) == 32
    assert int(payload["ig_workers"]) == 32
    assert int(payload["parallel_isotope_workers"]) == 32
    assert int(payload["structural_trial_workers"]) == 32
    assert int(payload["structural_trial_parallel_min_trials"]) <= 8
    assert int(payload["dss_pp"]["program_eval_workers"]) == 32
    assert int(payload["birth_min_distinct_stations"]) >= 2
    assert int(payload["birth_min_distinct_poses"]) >= 5
    assert float(payload["birth_existing_response_corr_max"]) <= 0.99
    assert float(payload["pseudo_source_temporal_sep_min"]) > 0.0
    assert payload["report_exclude_unverified_sources"] is True
    assert payload["report_strength_refit"] is True
    assert payload["report_strength_refit_use_all_measurements"] is True
    assert payload["report_strength_refit_preserve_cardinality"] is False
    assert float(payload["report_strength_refit_prior_weight"]) > 0.0
    assert payload["report_mle_rescue_enable"] is True
    assert int(payload["report_mle_rescue_max_candidates"]) >= 12
    assert payload["birth_global_rescue_enable"] is True
    assert int(payload["birth_global_rescue_max_candidates"]) >= 8
    assert float(payload["source_strength_prior_mean"]) > 0.0
    assert float(payload["source_strength_prior_weight"]) > 0.0
    assert payload["report_model_order_require_posterior_match"] is False
    assert payload["report_model_order_prune_particles"] is True
    assert payload["mode_preserving_resample"] is True
    assert int(payload["mode_preserving_max_modes"]) >= 12
    assert int(payload["mode_preserving_particles_per_mode"]) >= 8
    assert float(payload["mode_preserving_min_weight_fraction"]) == pytest.approx(0.0)
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
    assert int(payload["python_worker_count"]) == 32
    assert int(payload["ig_workers"]) == 32
    assert int(payload["parallel_isotope_workers"]) == 32
    assert int(payload["structural_trial_workers"]) == 32
    assert int(payload["structural_trial_parallel_min_trials"]) <= 8
    assert int(payload["dss_pp"]["program_eval_workers"]) == 32
    assert int(payload["birth_min_distinct_stations"]) >= 2
    assert int(payload["birth_min_distinct_poses"]) >= 5
    assert float(payload["birth_existing_response_corr_max"]) <= 0.99
    assert float(payload["pseudo_source_temporal_sep_min"]) > 0.0
    assert payload["report_exclude_unverified_sources"] is True
    assert payload["report_strength_refit"] is True
    assert payload["report_strength_refit_use_all_measurements"] is True
    assert payload["report_strength_refit_preserve_cardinality"] is False
    assert float(payload["report_strength_refit_prior_weight"]) > 0.0
    assert payload["report_mle_rescue_enable"] is True
    assert int(payload["report_mle_rescue_max_candidates"]) >= 12
    assert payload["birth_global_rescue_enable"] is True
    assert int(payload["birth_global_rescue_max_candidates"]) >= 8
    assert float(payload["source_strength_prior_mean"]) > 0.0
    assert float(payload["source_strength_prior_weight"]) > 0.0
    assert payload["report_model_order_require_posterior_match"] is False
    assert payload["report_model_order_prune_particles"] is True
    assert int(payload["mission_stop_max_poses"]) == 20
    assert payload["mission_stop_require_model_order_ready"] is True
    assert payload["mission_stop_require_pf_convergence_for_coverage"] is False
    assert payload["dss_pp"]["same_isotope_direct_separation_guard"] is True
    assert float(payload["dss_pp"]["temporal_separation_weight"]) >= 8.0
    assert float(payload["dss_pp"]["coverage_weight"]) <= 2.0
    assert payload["mode_preserving_resample"] is True
    assert int(payload["mode_preserving_max_modes"]) >= 12
    assert int(payload["mode_preserving_particles_per_mode"]) >= 8
    assert float(payload["mode_preserving_min_weight_fraction"]) == pytest.approx(0.0)


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
        'result.metadata["process_count_compton"]',
        'result.metadata["process_count_rayleigh"]',
        'result.metadata["process_count_photoelectric"]',
        'result.metadata["volume_step_counts"]',
        'result.metadata["primaries_per_sec"]',
        'result.metadata["effective_entries_per_sec"]',
        'result.metadata["total_spectrum_counts"]',
    ):
        assert token in source
