"""Tests for PF reporting helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from pf.estimator import RotatingShieldPFEstimator
from pf.reporting import dedupe_report_candidates, measurement_vector


def test_measurement_vector_broadcasts_scalar() -> None:
    """Scalar report inputs should broadcast to the requested measurement count."""
    vec = measurement_vector(2.5, 3, "background", min_value=0.0)
    assert np.allclose(vec, [2.5, 2.5, 2.5])


def test_measurement_vector_rejects_wrong_length() -> None:
    """Vector report inputs must match the requested measurement count."""
    with pytest.raises(ValueError, match="one value per measurement"):
        measurement_vector(np.asarray([1.0, 2.0]), 3, "z", allow_scalar=False)


def test_dedupe_report_candidates_keeps_strong_order() -> None:
    """Report candidate de-duplication should preserve deterministic input order."""
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    strengths = np.asarray([5.0, 9.0, 3.0], dtype=float)

    out_pos, out_q = dedupe_report_candidates(
        positions,
        strengths,
        radius_m=0.5,
        max_candidates=5,
    )

    assert out_pos.shape == (2, 3)
    assert np.allclose(out_pos, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(out_q, [5.0, 3.0])


def test_report_design_correlation_penalty_flags_collinear_sources() -> None:
    """Correlation penalty should activate only above the physical threshold."""
    collinear = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=float,
    )
    separated = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    assert (
        RotatingShieldPFEstimator._report_design_correlation_penalty(
            collinear,
            threshold=0.98,
            weight=24.0,
            power=1.0,
            eps=1.0e-12,
        )
        > 0.0
    )
    assert (
        RotatingShieldPFEstimator._report_design_correlation_penalty(
            separated,
            threshold=0.98,
            weight=24.0,
            power=1.0,
            eps=1.0e-12,
        )
        == 0.0
    )


def test_report_design_correlation_penalty_batch_matches_scalar() -> None:
    """Batched correlation penalties should match the scalar implementation."""
    designs = np.asarray(
        [
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=float,
    )
    batch = RotatingShieldPFEstimator._report_design_correlation_penalties_batch(
        designs,
        threshold=0.98,
        weight=24.0,
        power=1.0,
        eps=1.0e-12,
    )
    scalar = np.asarray(
        [
            RotatingShieldPFEstimator._report_design_correlation_penalty(
                design,
                threshold=0.98,
                weight=24.0,
                power=1.0,
                eps=1.0e-12,
            )
            for design in designs
        ],
        dtype=float,
    )

    assert np.allclose(batch, scalar)
