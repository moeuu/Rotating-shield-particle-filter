"""Preserve source-input validation used by current completed-run evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.source_normalization import Source, normalize_sources


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_source_positions_must_be_finite(coordinate: float) -> None:
    """Non-finite geometry must fail before truth-to-posterior matching."""
    with pytest.raises(ValueError, match="three finite coordinates"):
        normalize_sources([{"position": [coordinate, 0.0, 0.0], "strength": 1.0}])


@pytest.mark.parametrize("strength", [-1.0, float("nan"), float("inf")])
def test_source_strengths_must_be_finite_and_nonnegative(strength: float) -> None:
    """Malformed source strengths must fail even for preconstructed records."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalize_sources([Source(pos=np.zeros(3), strength=strength)])


def test_truth_and_posterior_source_names_normalize_consistently() -> None:
    """Runtime truth and PF posterior records must agree on physical units."""
    truth = normalize_sources([{"position": [1, 2, 3], "intensity_cps_1m": 12.5}])
    posterior = normalize_sources([{"pos": [1, 2, 3], "strength": 12.5}])
    np.testing.assert_array_equal(truth[0].pos, posterior[0].pos)
    assert truth[0].strength == posterior[0].strength == 12.5
