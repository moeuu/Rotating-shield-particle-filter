"""Tests for spectrum-net response calibration."""

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import PointSource
from pf.likelihood import expected_counts_per_source
from scripts.calibrate_geant4_net_response import _ideal_counts
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


def test_net_response_calibration_supports_shield_pair_scales() -> None:
    """Calibration should use pair-conditioned response scales when present."""
    calibration = fit_net_response_calibration(
        [
            {
                "isotope": "Cs-137",
                "shield_pair_id": 3,
                "theory_counts": 100.0,
                "net_counts": 10.0,
                "weight": 1.0,
            },
            {
                "isotope": "Cs-137",
                "shield_pair_id": 4,
                "theory_counts": 100.0,
                "net_counts": 30.0,
                "weight": 1.0,
            },
            {
                "isotope": "Cs-137",
                "shield_pair_id": 4,
                "theory_counts": 200.0,
                "net_counts": 60.0,
                "weight": 1.0,
            },
        ]
    )

    assert calibration.response_scale("Cs-137") == pytest.approx(4.0 / 15.0)
    assert calibration.response_scale("Cs-137", shield_pair_id=3) == pytest.approx(
        0.1
    )
    assert calibration.response_scale("Cs-137", shield_pair_id=4) == pytest.approx(
        0.3
    )
    mapped = calibration.apply_expected_counts(
        {"Cs-137": 100.0},
        shield_pair_id=4,
    )

    assert mapped["Cs-137"] == pytest.approx(30.0)


def test_pair_scales_require_support_and_shrink_to_isotope_scale() -> None:
    """Pair-conditioned scales should be support-gated and shrinkage-regularized."""
    calibration = fit_net_response_calibration(
        [
            {
                "isotope": "Cs-137",
                "shield_pair_id": 3,
                "theory_counts": 100.0,
                "net_counts": 100.0,
                "weight": 1.0,
            },
            {
                "isotope": "Cs-137",
                "shield_pair_id": 4,
                "theory_counts": 100.0,
                "net_counts": 200.0,
                "weight": 1.0,
            },
            {
                "isotope": "Cs-137",
                "shield_pair_id": 4,
                "theory_counts": 200.0,
                "net_counts": 400.0,
                "weight": 1.0,
            },
        ],
        min_pair_fit_points=2,
        pair_shrinkage_count=2.0,
    )

    isotope_scale = calibration.response_scale("Cs-137")
    pair_scale = calibration.response_scale("Cs-137", shield_pair_id=4)

    assert calibration.response_scale("Cs-137", shield_pair_id=3) == pytest.approx(
        isotope_scale
    )
    assert isotope_scale < pair_scale < 2.0


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


def test_calibration_ideal_counts_use_shared_continuous_kernel_value() -> None:
    """Calibration theory counts should match the shared ContinuousKernel API."""
    kernel = ContinuousKernel(detector_radius_m=0.5, use_gpu=False)
    source = PointSource(
        isotope="Cs-137",
        position=(0.25, 0.0, 0.0),
        intensity_cps_1m=20.0,
    )
    detector_pos = np.zeros(3, dtype=float)
    live_time_s = 3.0

    counts = _ideal_counts(
        [source],
        ["Cs-137"],
        detector_pos=detector_pos,
        fe_index=0,
        pb_index=0,
        live_time_s=live_time_s,
        kernel=kernel,
    )

    expected = (
        live_time_s
        * source.intensity_cps_1m
        * kernel.kernel_value_pair(
            "Cs-137",
            detector_pos,
            source.position_array(),
            0,
            0,
        )
    )
    assert counts["Cs-137"] == pytest.approx(expected)


def test_expected_counts_per_source_accepts_measurement_scale_vector() -> None:
    """Per-source expected counts should support one scale per measurement."""
    kernel = ContinuousKernel(use_gpu=False)
    detector_positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    sources = np.array([[1.0, 1.0, 1.0]], dtype=float)
    strengths = np.array([100.0], dtype=float)
    live_times = np.array([2.0, 2.0], dtype=float)
    fe_indices = np.array([0, 0], dtype=int)
    pb_indices = np.array([0, 0], dtype=int)

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
        source_scale=np.array([0.25, 2.0], dtype=float),
    )

    assert scaled[0] == pytest.approx(0.25 * base[0])
    assert scaled[1] == pytest.approx(2.0 * base[1])


def test_expected_counts_per_source_uses_selected_pair_batch_path() -> None:
    """Per-source expected counts should use the selected-pair batch kernel."""

    class RecordingKernel(ContinuousKernel):
        """ContinuousKernel that records selected-pair batch calls."""

        def __init__(self) -> None:
            """Initialize call counters for the recording kernel."""
            super().__init__(use_gpu=False)
            self.selected_pair_batch_calls = 0

        def kernel_values_selected_pairs_for_detectors(
            self,
            isotope: str,
            detector_positions: np.ndarray,
            sources: np.ndarray,
            fe_indices: np.ndarray,
            pb_indices: np.ndarray,
            chunk_size: int = 262144,
        ) -> np.ndarray:
            """Record the batch call and delegate to the production path."""
            self.selected_pair_batch_calls += 1
            return super().kernel_values_selected_pairs_for_detectors(
                isotope=isotope,
                detector_positions=detector_positions,
                sources=sources,
                fe_indices=fe_indices,
                pb_indices=pb_indices,
                chunk_size=chunk_size,
            )

    kernel = RecordingKernel()
    detector_positions = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=float,
    )
    sources = np.array(
        [[1.0, 1.0, 1.0], [1.5, 0.5, 1.0]],
        dtype=float,
    )
    strengths = np.array([100.0, 50.0], dtype=float)
    live_times = np.array([2.0, 3.0], dtype=float)
    fe_indices = np.array([0, 3], dtype=int)
    pb_indices = np.array([7, 1], dtype=int)

    counts = expected_counts_per_source(
        kernel=kernel,
        isotope="Cs-137",
        detector_positions=detector_positions,
        sources=sources,
        strengths=strengths,
        live_times=live_times,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
    )

    assert kernel.selected_pair_batch_calls == 1
    assert counts.shape == (detector_positions.shape[0], sources.shape[0])
