"""Tests for spectrum-net response calibration."""

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from pf.likelihood import expected_counts_per_source
from spectrum.net_response import NetResponseCalibration, fit_net_response_calibration


def test_fit_net_response_calibration_through_origin() -> None:
    """Calibration fit should recover a constant isotope-wise response scale."""
    calibration = fit_net_response_calibration(
        [
            {"isotope": "Cs-137", "theory_counts": 1000.0, "net_counts": 80.0, "weight": 1.0},
            {"isotope": "Cs-137", "theory_counts": 2000.0, "net_counts": 160.0, "weight": 1.0},
            {"isotope": "Co-60", "theory_counts": 500.0, "net_counts": 100.0, "weight": 1.0},
        ]
    )

    assert calibration.response_scale("Cs-137") == pytest.approx(0.08)
    assert calibration.response_scale("Co-60") == pytest.approx(0.2)


def test_net_response_calibration_applies_expected_counts() -> None:
    """Calibration should map ideal expected counts into net-count space."""
    calibration = NetResponseCalibration(scale_by_isotope={"Cs-137": 0.1})

    mapped = calibration.apply_expected_counts({"Cs-137": 1200.0, "Co-60": 50.0})

    assert mapped["Cs-137"] == 120.0
    assert mapped["Co-60"] == 50.0


def test_expected_counts_per_source_scales_only_source_terms() -> None:
    """Per-source expected counts should apply the response scale multiplicatively."""
    kernel = ContinuousKernel(use_gpu=False)
    detector_positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    sources = np.array([[1.0, 1.0, 1.0]], dtype=float)
    strengths = np.array([100.0], dtype=float)
    live_times = np.array([2.0], dtype=float)
    fe_indices = np.array([0], dtype=int)
    pb_indices = np.array([0], dtype=int)

    base = expected_counts_per_source(
        kernel=kernel,
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=sources,
        strengths=strengths,
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
    )
    scaled = expected_counts_per_source(
        kernel=kernel,
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=sources,
        strengths=strengths,
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        source_scale=0.25,
    )

    assert scaled == pytest.approx(0.25 * base)
