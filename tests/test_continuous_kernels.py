"""Tests for continuous 3D kernel evaluation (Sec. 3.2–3.3)."""

import numpy as np

from measurement.continuous_kernels import ContinuousKernel, geometric_term
from measurement.shielding import generate_octant_orientations
from measurement.continuous_kernels import expected_counts_single_isotope


def test_geometric_term_inverse_square() -> None:
    """Geometric term should follow 1/d^2 scaling."""
    det = np.array([0.0, 0.0, 0.0])
    s1 = np.array([1.0, 0.0, 0.0])
    s2 = np.array([2.0, 0.0, 0.0])
    g1 = geometric_term(det, s1)
    g2 = geometric_term(det, s2)
    assert np.isclose(g1 / g2, 4.0, rtol=1e-6)


def test_attenuation_applies_blocking_factor() -> None:
    """Blocked orientation should reduce expected counts by ~0.1."""
    kernel = ContinuousKernel()
    orientations = generate_octant_orientations()
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([1.0, 1.0, 1.0])
    strengths = np.array([10.0])

    # Vector from src->det is (-,-,-), so orient 7 blocks, orient 0 unblocks
    blocked_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=7,
        live_time_s=1.0,
        background=0.0,
    )
    free_counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=strengths,
        orient_idx=0,
        live_time_s=1.0,
        background=0.0,
    )
    assert np.isclose(blocked_counts, 0.1 * free_counts, rtol=1e-6)


def test_background_added_to_expected_counts() -> None:
    """Background should add directly to expected rate/counts."""
    kernel = ContinuousKernel()
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([10.0, 0.0, 0.0])
    counts = kernel.expected_counts(
        isotope="Cs-137",
        detector_pos=det,
        sources=np.array([src]),
        strengths=np.array([0.0]),
        orient_idx=0,
        live_time_s=2.0,
        background=5.0,
    )
    assert np.isclose(counts, 10.0, rtol=1e-6)


def test_expected_counts_single_isotope_attenuation_levels() -> None:
    """Fe/Pb blocking factors (0.1, 0.01) should scale expected counts accordingly."""
    det = np.array([0.0, 0.0, 0.0])
    src = np.array([[1.0, 1.0, 1.0]])
    strengths = np.array([10.0])
    # Orientation normal aligned with direction (-,-,-) from src to det
    orient_block = np.array([-1.0, -1.0, -1.0])
    orient_free = np.array([1.0, 1.0, 1.0])
    base = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_free,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    fe_only = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_free,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    both = expected_counts_single_isotope(
        detector_position=det,
        RFe=orient_block,
        RPb=orient_block,
        sources=src,
        strengths=strengths,
        background=0.0,
        duration=1.0,
        isotope_id="Cs-137",
    )
    assert np.isclose(fe_only, 0.1 * base, rtol=1e-6)
    assert np.isclose(both, 0.01 * base, rtol=1e-6)
