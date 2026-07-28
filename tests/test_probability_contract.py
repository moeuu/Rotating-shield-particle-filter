"""Tests for strict posterior-probability validation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pf.posterior import (
    cardinality_distribution_from_states,
    validated_probability,
    validated_probability_distribution,
    weighted_quantile,
)


def test_probability_validator_clips_only_boundary_roundoff() -> None:
    """Only tiny floating-point excursions outside the unit interval are valid."""
    assert validated_probability(
        -5.0e-13,
        name="lower",
    ) == 0.0
    assert validated_probability(
        1.0 + 5.0e-13,
        name="upper",
    ) == 1.0

    for invalid in (
        -2.0e-12,
        1.0 + 2.0e-12,
        np.nan,
        np.inf,
        "0.5",
        True,
        np.bool_(False),
    ):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validated_probability(invalid, name="invalid")


def test_probability_distribution_rejects_material_renormalization() -> None:
    """A posterior consumer must not turn arbitrary scores into probabilities."""
    normalized = validated_probability_distribution(
        [0.25, 0.75 + 5.0e-13],
        name="posterior",
    )
    np.testing.assert_allclose(
        normalized,
        np.asarray([0.25, 0.75 + 5.0e-13])
        / (1.0 + 5.0e-13),
        rtol=0.0,
        atol=0.0,
    )

    for invalid in (
        [1.0, 3.0],
        [0.0, 0.0],
        [0.5, np.nan],
        ["0.5", 0.5],
        [True, 0.0],
        [],
    ):
        with pytest.raises(ValueError, match="posterior"):
            validated_probability_distribution(invalid, name="posterior")


@pytest.mark.parametrize("invalid_cardinality", [-1, 1.5, True])
def test_cardinality_reporting_rejects_invalid_particle_state(
    invalid_cardinality: object,
) -> None:
    """Reporting must not clip or truncate a corrupt PF source count."""
    with pytest.raises((TypeError, ValueError), match="num_sources"):
        cardinality_distribution_from_states(
            [SimpleNamespace(num_sources=invalid_cardinality)],
            np.asarray([1.0], dtype=np.float64),
            max_cardinality=2,
        )


def test_cardinality_reporting_rejects_state_outside_declared_support() -> None:
    """A corrupt K above the PF support must not expand the report silently."""
    with pytest.raises(ValueError, match="configured maximum"):
        cardinality_distribution_from_states(
            [SimpleNamespace(num_sources=3)],
            np.asarray([1.0], dtype=np.float64),
            max_cardinality=2,
        )


@pytest.mark.parametrize("invalid_quantile", [-0.01, 1.01, np.nan, True])
def test_weighted_quantile_rejects_invalid_probability(
    invalid_quantile: object,
) -> None:
    """Posterior intervals must not clamp an invalid requested quantile."""
    with pytest.raises((TypeError, ValueError), match="quantile"):
        weighted_quantile(
            np.asarray([1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            invalid_quantile,
        )


def test_weighted_quantile_rejects_missing_or_nonfinite_samples() -> None:
    """Reporting must not invent a zero quantile for unavailable samples."""
    with pytest.raises(ValueError, match="at least one sample"):
        weighted_quantile(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        weighted_quantile(
            np.asarray([np.nan], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            0.5,
        )
