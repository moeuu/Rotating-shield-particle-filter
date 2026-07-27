"""Tests for PF resampling helper utilities."""

from __future__ import annotations

import numpy as np

from pf.resampling import systematic_resample_count


def test_systematic_resample_count_handles_empty_draw() -> None:
    """Zero requested draws should return an empty index array."""
    idx = systematic_resample_count(np.array([0.2, 0.8]), count=0)

    assert idx.dtype == np.int64
    assert idx.size == 0


def test_systematic_resample_count_normalizes_weights() -> None:
    """Positive non-normalized weights should be sampled as probabilities."""
    np.random.seed(3)

    idx = systematic_resample_count(np.array([0.0, 2.0, 8.0]), count=10)

    assert idx.shape == (10,)
    assert np.all(idx >= 0)
    assert np.all(idx < 3)
    assert np.count_nonzero(idx == 2) > np.count_nonzero(idx == 1)


def test_systematic_resample_count_falls_back_to_uniform() -> None:
    """Invalid total mass should fall back to uniform weights."""
    np.random.seed(1)

    idx = systematic_resample_count(np.array([0.0, 0.0, 0.0]), count=6)

    assert idx.shape == (6,)
    assert set(idx.tolist()).issubset({0, 1, 2})
