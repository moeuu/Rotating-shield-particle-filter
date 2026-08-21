"""Stratified sampling algorithms shared by PF estimator stages."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def _stratified_categorical_draws(
    probabilities: NDArray[np.float64],
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw a randomly permuted stratified categorical sample batch."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    count = int(sample_count)
    total = float(np.sum(values, dtype=np.float64))
    if (
        count < 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isfinite(total)
        or total <= 0.0
        or not isinstance(rng, np.random.Generator)
    ):
        raise ValueError("Stratified categorical inputs are invalid.")
    normalized = values / total
    uniforms = (np.arange(count, dtype=np.float64) + rng.random(count)) / float(count)
    draws = np.searchsorted(
        np.cumsum(normalized, dtype=np.float64),
        uniforms,
        side="right",
    ).astype(np.int64, copy=False)
    draws = np.minimum(draws, values.size - 1)
    return draws[rng.permutation(count)]


def _stratified_joint_cardinality_draws(
    marginal_probabilities: Sequence[NDArray[np.float64]],
    sample_count: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw product-prior K vectors with joint stratification.

    The vectorized Cartesian support is tiny for the configured isotope and
    source capacities. Sampling the flattened product distribution preserves
    the independent cardinality prior exactly while stratifying the joint
    vectors rather than only their separate isotope marginals.
    """
    probabilities = tuple(
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in marginal_probabilities
    )
    if not probabilities:
        raise ValueError("At least one cardinality marginal is required.")
    if any(
        values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1.0e-12)
        for values in probabilities
    ):
        raise ValueError("Cardinality marginals must be probability vectors.")
    support_shape = tuple(int(values.size) for values in probabilities)
    support_indices = (
        np.indices(
            support_shape,
            dtype=np.int64,
        )
        .reshape(len(probabilities), -1)
        .T
    )
    product_mass = np.ones(support_indices.shape[0], dtype=np.float64)
    for isotope_index, values in enumerate(probabilities):
        product_mass *= values[support_indices[:, isotope_index]]
    flat_draws = _stratified_categorical_draws(
        product_mass,
        sample_count,
        rng=rng,
    )
    return support_indices[flat_draws]
