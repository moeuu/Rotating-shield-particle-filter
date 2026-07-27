"""Tests for fixed-size PF systematic resampling."""

from __future__ import annotations

import numpy as np

from pf.resampling import systematic_resample


def test_systematic_resample_preserves_particle_count() -> None:
    """Standard PF resampling should always return one index per particle."""
    probabilities = np.array([0.1, 0.2, 0.7], dtype=float)

    indices = systematic_resample(
        np.log(probabilities),
        rng=np.random.default_rng(3),
    )

    assert indices.dtype == np.int64
    assert indices.shape == (3,)
    assert np.all(indices >= 0)
    assert np.all(indices < probabilities.size)


def test_systematic_resample_uniform_weights_select_each_particle() -> None:
    """Uniform weights should preserve all particle representatives."""
    log_weights = np.full(6, -np.log(6.0), dtype=float)

    indices = systematic_resample(log_weights, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(indices, np.arange(6, dtype=np.int64))


def test_systematic_resample_is_seed_reproducible() -> None:
    """Equal NumPy states should produce identical ancestor indices."""
    log_weights = np.log(np.array([0.05, 0.15, 0.3, 0.5], dtype=float))
    first = systematic_resample(
        log_weights,
        rng=np.random.default_rng(20260727),
    )
    second = systematic_resample(
        log_weights,
        rng=np.random.default_rng(20260727),
    )

    np.testing.assert_array_equal(first, second)
