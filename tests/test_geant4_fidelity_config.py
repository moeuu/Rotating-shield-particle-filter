"""Tests for high-fidelity Geant4 runtime configuration defaults."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sim.geant4_app.app import Geant4AppConfig
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    ExportedDetectorModel,
)


def _geant4_runtime_config_paths() -> list[Path]:
    """Return Geant4 runtime configs, excluding calibration payloads."""
    root = Path(__file__).resolve().parents[1]
    return [
        path
        for path in sorted((root / "configs" / "geant4").glob("*.json"))
        if path.name != "net_response_calibration.json"
    ]


def test_geant4_configs_use_weighted_variance_reduction_by_default() -> None:
    """Geant4 configs should default to unbiased weighted variance reduction."""
    forbidden_args = {
        "--min-histories-per-line",
        "--max-histories-per-line",
        "--no-poisson-background",
    }
    for config_path in _geant4_runtime_config_paths():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        executable_args = set(payload.get("executable_args", []))

        assert payload.get("engine_mode", "external") == "external"
        assert payload.get("persistent_process", False) is True
        assert payload.get("spectrum_count_method", "photopeak_nnls") == "photopeak_nnls"
        assert str(payload.get("physics_profile", "balanced")).lower() != "theory_tvl"
        assert not forbidden_args.intersection(executable_args)
        assert "net_response_calibration_path" not in payload
        assert "net_response_calibration" not in payload
        assert float(payload.get("scatter_gain", 0.0)) == 0.0
        source_bias_mode = str(
            payload.get("source_bias_mode", "mixture_cone_isotropic")
        )
        if "high_fidelity" in config_path.name:
            assert source_bias_mode == "analog"
        else:
            assert source_bias_mode == "mixture_cone_isotropic"
            isotropic_fraction = float(
                payload.get("source_bias_isotropic_fraction", 0.1)
            )
            assert 0.05 <= isotropic_fraction <= 0.20
            assert float(payload.get("source_bias_cone_half_angle_deg", 0.0)) >= 0.0


def test_high_fidelity_external_config_uses_native_geometry() -> None:
    """The explicit high-fidelity external config should use balanced native transport."""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "geant4" / "high_fidelity_external_no_isaac.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["engine_mode"] == "external"
    assert payload["physics_profile"] == "balanced"
    assert payload["start_isaacsim_sidecar_with_geant4"] is False
    assert payload["author_obstacle_prims"] is True
    assert payload["source_bias_mode"] == "analog"
    assert payload["source_bias_isotropic_fraction"] == pytest.approx(1.0)
    assert "executable_args" not in payload


def test_default_config_uses_weighted_variance_reduction() -> None:
    """The practical runtime default should be weighted variance reduction."""
    config = Geant4AppConfig.from_dict({})

    assert config.source_bias_mode == "mixture_cone_isotropic"
    assert config.source_bias_isotropic_fraction == pytest.approx(0.1)
    assert config.source_bias_cone_half_angle_deg == pytest.approx(0.0)


def test_variance_reduction_config_is_explicit_weighted_mode() -> None:
    """The named variance-reduction config should document the default mode."""
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = Geant4AppConfig.from_dict(payload)

    assert config.source_bias_mode == "mixture_cone_isotropic"
    assert config.source_bias_isotropic_fraction == pytest.approx(0.1)
    assert config.source_bias_cone_half_angle_deg == pytest.approx(0.0)
    assert config.physics_profile == "balanced"


def test_geant4_configs_use_large_detector_model() -> None:
    """Runtime configs should use the large spherical CeBr3 detector."""
    for config_path in _geant4_runtime_config_paths():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
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
    """Native Geant4 spectra should not double-apply empirical efficiency weights."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").read_text(encoding="utf-8")

    assert "DetectorEfficiency" not in source
    assert "spectrum[index] += deposit.weight;" in source
    assert 'std::string source_bias_mode = "mixture_cone_isotropic";' in source
    assert "double source_bias_isotropic_fraction = 0.1;" in source
    assert 'result.metadata["isotropic_mixture_fraction"]' in source
    assert 'result.metadata["cone_half_angle_deg"]' in source
    assert "{1596.5, 0.02}" in source
