"""Tests for response-Poisson transport-truth calibration utilities."""

import numpy as np
import pytest

from spectrum.response_truth_calibration import apply_response_truth_calibration
from spectrum.response_truth_calibration import fit_local_knn_response_truth_calibration


def test_local_knn_response_truth_calibration_maps_counts() -> None:
    """Local response-truth calibration should scale counts and variances."""
    records = [
        {
            "isotope": "Cs-137",
            "shield_pair_id": 3,
            "response_count": 100.0 + 10.0 * idx,
            "truth_count": 110.0 + 11.0 * idx,
            "raw_count": 100.0 + 10.0 * idx,
            "photopeak_count": 100.0 + 10.0 * idx,
            "reduced_chi2": 1.0,
            "spectrum_total": 1000.0 + 100.0 * idx,
        }
        for idx in range(4)
    ]
    payload = fit_local_knn_response_truth_calibration(
        records,
        neighbor_count=2,
        min_group_points=2,
    )

    counts, variances, debug = apply_response_truth_calibration(
        {"Cs-137": 120.0},
        {"Cs-137": 25.0},
        payload,
        shield_pair_id=3,
        diagnostics={
            "coefficients": {"Cs-137": 120.0},
            "photopeak_counts": {"Cs-137": 120.0},
            "reduced_chi2": 1.0,
        },
        spectrum_total=1200.0,
    )

    assert counts["Cs-137"] == pytest.approx(132.0, rel=1.0e-6)
    assert variances["Cs-137"] == pytest.approx(25.0 * 1.1**2, rel=1.0e-6)
    assert debug["scales"]["Cs-137"] == pytest.approx(1.1, rel=1.0e-6)


def test_local_knn_response_truth_calibration_uses_fallback_scale() -> None:
    """Calibration should fall back to isotope scale when a pair is absent."""
    payload = fit_local_knn_response_truth_calibration(
        [
            {
                "isotope": "Co-60",
                "shield_pair_id": 1,
                "response_count": 100.0,
                "truth_count": 80.0,
                "spectrum_total": 500.0,
            },
            {
                "isotope": "Co-60",
                "shield_pair_id": 1,
                "response_count": 200.0,
                "truth_count": 160.0,
                "spectrum_total": 1000.0,
            },
        ],
        neighbor_count=2,
        min_group_points=2,
    )

    counts, _, debug = apply_response_truth_calibration(
        {"Co-60": 50.0},
        {"Co-60": 4.0},
        payload,
        shield_pair_id=99,
        diagnostics={},
        spectrum_total=250.0,
    )

    assert counts["Co-60"] == pytest.approx(40.0, rel=1.0e-6)
    assert debug["scales"]["Co-60"] == pytest.approx(0.8, rel=1.0e-6)


def test_response_truth_calibration_ignores_disabled_payload() -> None:
    """Disabled calibration payloads should leave counts unchanged."""
    counts, variances, debug = apply_response_truth_calibration(
        {"Eu-154": 20.0},
        {"Eu-154": 9.0},
        {"enabled": False},
        shield_pair_id=0,
        diagnostics={},
        spectrum_total=float(np.nan),
    )

    assert counts["Eu-154"] == pytest.approx(20.0)
    assert variances["Eu-154"] == pytest.approx(9.0)
    assert debug["enabled"] is False
