"""Tests for Geant4 calibration helper functions."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from measurement.model import PointSource


def _load_calibration_script() -> object:
    """Load the calibration script as a module for helper tests."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "calibrate_geant4_net_response.py"
    spec = spec_from_file_location("calibrate_geant4_net_response", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CALIBRATION_SCRIPT = _load_calibration_script()


def test_source_tally_counts_are_history_normalized() -> None:
    """Native source-equivalent tallies should be divided by history scale."""
    metadata = {
        "source_equivalent_counts_Cs-137": 500.0,
        "source_equivalent_counts_Co-60": 100.0,
    }

    counts = CALIBRATION_SCRIPT._source_tally_counts(
        metadata,
        ["Cs-137", "Co-60", "Eu-154"],
        history_scale=5.0,
    )

    assert counts["Cs-137"] == pytest.approx(100.0)
    assert counts["Co-60"] == pytest.approx(20.0)
    assert counts["Eu-154"] == pytest.approx(0.0)


def test_scale_sources_multiplies_intensity_only() -> None:
    """History scaling should keep source identity and position unchanged."""
    source = PointSource("Cs-137", position=(1.0, 2.0, 3.0), intensity_cps_1m=10.0)

    scaled = CALIBRATION_SCRIPT._scale_sources([source], 4.0)

    assert scaled[0].isotope == source.isotope
    assert scaled[0].position == source.position
    assert scaled[0].intensity_cps_1m == pytest.approx(40.0)
