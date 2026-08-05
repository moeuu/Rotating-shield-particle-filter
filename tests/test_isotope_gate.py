"""Tests for truth-free full-spectrum isotope activation."""

from __future__ import annotations

import numpy as np

from pf.isotope_gate import FullSpectrumIsotopeGate


def test_gate_activates_only_isotopes_crossing_corrected_threshold() -> None:
    """Activation must be one-way and use only accumulated score grids."""
    gate = FullSpectrumIsotopeGate(
        ("Co-60", "Cs-137", "Eu-154"),
        false_activation_probability=1.0e-3,
    )
    first = gate.update(
        {
            "Co-60": np.zeros((2, 5), dtype=np.float64),
            "Cs-137": np.full((2, 5), 30.0, dtype=np.float64),
            "Eu-154": np.full((2, 5), -5.0, dtype=np.float64),
        }
    )

    assert first["active_isotopes"] == ["Cs-137"]
    assert first["newly_active_isotopes"] == ["Cs-137"]
    assert first["truth_used"] is False

    second = gate.update(
        {
            "Co-60": np.full((2, 5), 30.0, dtype=np.float64),
            "Cs-137": np.full((2, 5), -100.0, dtype=np.float64),
            "Eu-154": np.zeros((2, 5), dtype=np.float64),
        }
    )

    assert second["active_isotopes"] == ["Co-60", "Cs-137"]
    assert second["newly_active_isotopes"] == ["Co-60"]
    assert "Eu-154" not in gate.active_isotopes


def test_gate_threshold_accounts_for_repeated_station_tests() -> None:
    """The sequential threshold must spend less false-alarm mass over time."""
    gate = FullSpectrumIsotopeGate(("Cs-137",), 1.0e-3)
    first = gate.update({"Cs-137": np.zeros((3, 5), dtype=np.float64)})
    second = gate.update({"Cs-137": np.zeros((3, 5), dtype=np.float64)})

    assert (
        second["activation_log_score_threshold"]
        > first["activation_log_score_threshold"]
    )
    assert (
        second["sequential_false_activation_probability"]
        < first["sequential_false_activation_probability"]
    )
