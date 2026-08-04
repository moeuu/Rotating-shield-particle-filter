"""Tests for the normalized bounded physical source-strength prior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from pf.strength_prior import StrengthPrior


def test_strength_prior_log_prob_is_normalized() -> None:
    """The bounded-uniform density should integrate to one."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    integral, error = quad(
        lambda value: float(np.exp(prior.log_prob(value))),
        prior.minimum,
        prior.maximum,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )

    assert error < 1.0e-10
    assert integral == pytest.approx(1.0, abs=1.0e-12)


def test_strength_prior_support_matches_physical_bounds() -> None:
    """Support masks and log densities should agree at finite bounds."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    inside = np.array([2.0, 4.0, 8.0])
    outside = np.array([-1.0, 1.99, 8.01, np.inf, np.nan])

    assert np.all(prior.in_support(inside))
    assert not np.any(prior.in_support(outside))
    assert np.all(np.isfinite(prior.log_prob(inside)))
    assert np.all(np.isneginf(prior.log_prob(outside)))


def test_strength_prior_sampling_matches_uniform_mean() -> None:
    """Batched draws should match the bounded-uniform first moment."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    samples = prior.sample(50_000, rng=np.random.default_rng(781))

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (50_000,)
    assert np.all(prior.in_support(samples))
    assert not np.any(samples == prior.minimum)
    assert not np.any(samples == prior.maximum)
    assert float(np.mean(samples)) == pytest.approx(5.0, abs=0.02)


def test_shifted_gamma_strength_prior_is_normalized_and_unbounded() -> None:
    """The proper upper-unbounded prior must integrate to one."""
    prior = StrengthPrior(
        minimum=2.0,
        maximum=8.0,
        family="shifted_gamma",
        gamma_shape=2.0,
        gamma_scale=3.0,
    )
    integral, error = quad(
        lambda value: float(np.exp(prior.log_prob(value))),
        prior.minimum,
        np.inf,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
    )

    assert error < 1.0e-8
    assert integral == pytest.approx(1.0, abs=1.0e-10)
    assert prior.support_maximum == np.inf
    assert prior.in_support(80.0)
    assert np.isfinite(prior.log_prob(80.0))
    assert prior.finite_upper_quantile() > prior.maximum


def test_shifted_gamma_strength_prior_sampling_matches_mean() -> None:
    """Batched shifted-gamma draws should preserve their analytic mean."""
    prior = StrengthPrior(
        minimum=2.0,
        maximum=8.0,
        family="shifted_gamma",
        gamma_shape=2.0,
        gamma_scale=3.0,
    )
    samples = prior.sample(100_000, rng=np.random.default_rng(412))

    assert np.all(samples >= prior.minimum)
    assert np.any(samples > prior.maximum)
    assert float(np.mean(samples)) == pytest.approx(prior.mean, abs=0.04)


def test_strength_prior_sampling_is_seed_reproducible() -> None:
    """Equal NumPy generator states should produce bitwise-equal batches."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    first = prior.sample((7, 5), rng=np.random.default_rng(20260727))
    second = prior.sample((7, 5), rng=np.random.default_rng(20260727))

    np.testing.assert_array_equal(first, second)


def test_strength_prior_scalar_and_batch_interfaces() -> None:
    """Scalar inputs should return scalars while batches retain their shape."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    scalar_sample = prior.sample(rng=np.random.default_rng(14))
    batch_sample = prior.sample((2, 3), rng=np.random.default_rng(14))

    assert isinstance(scalar_sample, float)
    assert isinstance(prior.in_support(4.0), bool)
    assert isinstance(prior.log_prob(4.0), float)
    assert prior.in_support(4.0)
    assert np.isfinite(prior.log_prob(4.0))
    assert isinstance(batch_sample, np.ndarray)
    assert batch_sample.shape == (2, 3)
    batch_support = prior.in_support([[2.0, 4.0], [8.0, 9.0]])
    batch_log_prob = prior.log_prob([[2.0, 4.0], [8.0, 9.0]])
    assert isinstance(batch_support, np.ndarray)
    assert isinstance(batch_log_prob, np.ndarray)
    np.testing.assert_array_equal(
        batch_support,
        np.array([[True, True], [True, False]]),
    )
    assert np.isneginf(batch_log_prob[1, 1])


def test_batched_density_and_support_match_scalar_oracle() -> None:
    """Batched evaluation should match independent scalar evaluations."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)
    values = np.array(
        [[0.0, 1.0, 2.0], [5.0, 8.0, 9.0]],
        dtype=float,
    )
    batched_log_prob = prior.log_prob(values)
    batched_support = prior.in_support(values)
    scalar_log_prob = np.array(
        [prior.log_prob(float(value)) for value in values.ravel()],
        dtype=float,
    ).reshape(values.shape)
    scalar_support = np.array(
        [prior.in_support(float(value)) for value in values.ravel()],
        dtype=bool,
    ).reshape(values.shape)

    np.testing.assert_array_equal(batched_log_prob, scalar_log_prob)
    np.testing.assert_array_equal(batched_support, scalar_support)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"minimum": -1.0, "maximum": 2.0}, "nonnegative"),
        ({"minimum": np.inf, "maximum": 2.0}, "finite"),
        ({"minimum": 2.0, "maximum": 2.0}, "greater than minimum"),
        ({"minimum": 2.0, "maximum": np.inf}, "finite"),
    ],
)
def test_strength_prior_rejects_invalid_parameters(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Invalid physical bounds should fail before sampling."""
    with pytest.raises(ValueError, match=match):
        StrengthPrior(**kwargs)


def test_strength_prior_rejects_non_generator_rng() -> None:
    """Sampling should reject RNG objects with incompatible semantics."""
    prior = StrengthPrior(minimum=2.0, maximum=8.0)

    with pytest.raises(TypeError, match="numpy.random.Generator"):
        prior.sample(2, rng=object())  # type: ignore[arg-type]
