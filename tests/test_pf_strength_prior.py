"""Tests for normalized source-strength priors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import truncnorm

from pf.strength_prior import StrengthPrior


@pytest.mark.parametrize(
    "prior,integration_bounds",
    [
        (
            StrengthPrior("uniform", minimum=2.0, maximum=8.0),
            (2.0, 8.0),
        ),
        (
            StrengthPrior("log_uniform", minimum=1.0, maximum=100.0),
            (1.0, 100.0),
        ),
        (
            StrengthPrior(
                "lognormal",
                minimum=2.0,
                maximum=20.0,
                log_mean=1.5,
                log_sigma=0.7,
            ),
            (2.0, 20.0),
        ),
        (
            StrengthPrior(
                "lognormal",
                minimum=0.0,
                maximum=None,
                log_mean=1.5,
                log_sigma=0.7,
            ),
            (0.0, np.inf),
        ),
    ],
)
def test_strength_prior_log_prob_is_normalized(
    prior: StrengthPrior,
    integration_bounds: tuple[float, float],
) -> None:
    """Every supported prior density should integrate to one."""
    lower, upper = integration_bounds
    integral, error = quad(
        lambda value: float(np.exp(prior.log_prob(value))),
        lower,
        upper,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
    )

    assert error < 1.0e-8
    assert integral == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    "prior,inside,outside",
    [
        (
            StrengthPrior("uniform", minimum=2.0, maximum=8.0),
            np.array([2.0, 4.0, 8.0]),
            np.array([-1.0, 1.99, 8.01, np.inf, np.nan]),
        ),
        (
            StrengthPrior("log_uniform", minimum=1.0, maximum=100.0),
            np.array([1.0, 10.0, 100.0]),
            np.array([0.0, 0.99, 100.01, np.inf, np.nan]),
        ),
        (
            StrengthPrior(
                "lognormal",
                minimum=2.0,
                maximum=20.0,
                log_mean=1.5,
                log_sigma=0.7,
            ),
            np.array([2.0, 5.0, 20.0]),
            np.array([0.0, 1.99, 20.01, np.inf, np.nan]),
        ),
    ],
)
def test_strength_prior_support_matches_finite_bounds(
    prior: StrengthPrior,
    inside: np.ndarray,
    outside: np.ndarray,
) -> None:
    """Support masks and log densities should agree at all tested values."""
    assert np.all(prior.in_support(inside))
    assert not np.any(prior.in_support(outside))
    assert np.all(np.isfinite(prior.log_prob(inside)))
    assert np.all(np.isneginf(prior.log_prob(outside)))


@pytest.mark.parametrize(
    "prior,expected_transformed_mean",
    [
        (
            StrengthPrior("uniform", minimum=2.0, maximum=8.0),
            5.0,
        ),
        (
            StrengthPrior("log_uniform", minimum=1.0, maximum=100.0),
            0.5 * np.log(100.0),
        ),
    ],
)
def test_uniform_prior_sampling_matches_expected_mean(
    prior: StrengthPrior,
    expected_transformed_mean: float,
) -> None:
    """Uniform and log-uniform batched draws should match their first moments."""
    samples = prior.sample(50_000, rng=np.random.default_rng(781))
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (50_000,)
    assert np.all(prior.in_support(samples))
    assert not np.any(samples == prior.minimum)
    assert not np.any(samples == prior.maximum)

    transformed = samples if prior.kind == "uniform" else np.log(samples)
    assert float(np.mean(transformed)) == pytest.approx(
        expected_transformed_mean,
        abs=0.02,
    )


def test_truncated_lognormal_sampling_has_no_boundary_atoms() -> None:
    """Truncated lognormal draws should be direct draws, not clipped samples."""
    prior = StrengthPrior(
        "lognormal",
        minimum=2.0,
        maximum=20.0,
        log_mean=1.5,
        log_sigma=0.7,
    )
    samples = prior.sample((250, 200), rng=np.random.default_rng(992))
    alpha = (np.log(2.0) - 1.5) / 0.7
    beta = (np.log(20.0) - 1.5) / 0.7
    expected_log_mean = float(
        truncnorm.mean(alpha, beta, loc=1.5, scale=0.7)
    )

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (250, 200)
    assert np.all(prior.in_support(samples))
    assert not np.any(samples == prior.minimum)
    assert not np.any(samples == prior.maximum)
    assert np.unique(samples).size > 49_000
    assert float(np.mean(np.log(samples))) == pytest.approx(
        expected_log_mean,
        abs=0.01,
    )


@pytest.mark.parametrize(
    "prior",
    [
        StrengthPrior("uniform", minimum=2.0, maximum=8.0),
        StrengthPrior("log_uniform", minimum=1.0, maximum=100.0),
        StrengthPrior(
            "lognormal",
            minimum=2.0,
            maximum=20.0,
            log_mean=1.5,
            log_sigma=0.7,
        ),
    ],
)
def test_strength_prior_sampling_is_seed_reproducible(
    prior: StrengthPrior,
) -> None:
    """Equal NumPy generator states should produce bitwise-equal batches."""
    first = prior.sample((7, 5), rng=np.random.default_rng(20260727))
    second = prior.sample((7, 5), rng=np.random.default_rng(20260727))

    np.testing.assert_array_equal(first, second)


def test_strength_prior_scalar_and_batch_interfaces() -> None:
    """Scalar inputs should return scalars while batches retain their shape."""
    prior = StrengthPrior("uniform", minimum=2.0, maximum=8.0)
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
    assert batch_support.shape == (2, 2)
    assert batch_log_prob.shape == (2, 2)
    np.testing.assert_array_equal(
        batch_support,
        np.array([[True, True], [True, False]]),
    )
    assert np.isneginf(batch_log_prob[1, 1])


@pytest.mark.parametrize(
    "prior",
    [
        StrengthPrior("uniform", minimum=2.0, maximum=8.0),
        StrengthPrior("log_uniform", minimum=1.0, maximum=100.0),
        StrengthPrior(
            "lognormal",
            minimum=2.0,
            maximum=20.0,
            log_mean=1.5,
            log_sigma=0.7,
        ),
    ],
)
def test_batched_density_and_support_match_scalar_oracle(
    prior: StrengthPrior,
) -> None:
    """Batched evaluation should match independent scalar evaluations."""
    values = np.array(
        [[0.0, 1.0, 2.0], [5.0, 20.0, 101.0]],
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
        ({"kind": "unknown"}, "kind must"),
        ({"kind": "uniform", "maximum": None}, "finite maximum"),
        (
            {"kind": "uniform", "minimum": 2.0, "maximum": 2.0},
            "greater than minimum",
        ),
        (
            {"kind": "log_uniform", "minimum": 0.0, "maximum": 2.0},
            "positive minimum",
        ),
        (
            {"kind": "lognormal", "log_sigma": 0.0},
            "finite and positive",
        ),
    ],
)
def test_strength_prior_rejects_invalid_parameters(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Invalid continuous-prior definitions should fail before sampling."""
    with pytest.raises(ValueError, match=match):
        StrengthPrior(**kwargs)
