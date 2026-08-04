"""Proper physical priors for source strengths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincinv, gammaln


SampleShape: TypeAlias = int | tuple[int, ...] | None
FloatResult: TypeAlias = float | NDArray[np.float64]
BoolResult: TypeAlias = bool | NDArray[np.bool_]


def _float_result(
    value: NDArray[np.float64],
    *,
    scalar_input: bool,
) -> FloatResult:
    """Return a Python float for scalar input and an array otherwise."""
    if scalar_input:
        return float(np.asarray(value, dtype=np.float64))
    return np.asarray(value, dtype=np.float64)


def _bool_result(
    value: NDArray[np.bool_],
    *,
    scalar_input: bool,
) -> BoolResult:
    """Return a Python bool for scalar input and an array otherwise."""
    if scalar_input:
        return bool(np.asarray(value, dtype=np.bool_))
    return np.asarray(value, dtype=np.bool_)


@dataclass(frozen=True)
class StrengthPrior:
    """Represent a normalized physical source-strength prior.

    ``shifted_gamma`` removes the artificial upper support boundary while
    retaining a proper, normalized density.  ``maximum`` remains a required
    finite configuration field for backwards-compatible proposal-grid sizing;
    it is not part of shifted-gamma support.
    """

    minimum: float
    maximum: float
    family: str = "bounded_uniform"
    gamma_shape: float = 2.0
    gamma_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the selected normalized prior before inference starts."""
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        family = str(self.family).strip().lower()
        gamma_shape = float(self.gamma_shape)
        gamma_scale = float(self.gamma_scale)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum must be finite and nonnegative.")
        if not np.isfinite(maximum) or maximum <= minimum:
            raise ValueError("maximum must be finite and greater than minimum.")
        if family not in {"bounded_uniform", "shifted_gamma"}:
            raise ValueError(
                "family must be 'bounded_uniform' or 'shifted_gamma'."
            )
        if (
            not np.isfinite(gamma_shape)
            or gamma_shape < 1.0
            or not np.isfinite(gamma_scale)
            or gamma_scale <= 0.0
        ):
            raise ValueError(
                "gamma_shape must be finite and at least one, and "
                "gamma_scale must be finite and positive."
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "gamma_shape", gamma_shape)
        object.__setattr__(self, "gamma_scale", gamma_scale)

    @property
    def support_maximum(self) -> float:
        """Return the mathematical upper support endpoint."""
        if self.family == "shifted_gamma":
            return float("inf")
        return self.maximum

    @property
    def mean(self) -> float:
        """Return the prior mean in detector-count-rate units."""
        if self.family == "shifted_gamma":
            return self.minimum + self.gamma_shape * self.gamma_scale
        return 0.5 * (self.minimum + self.maximum)

    def finite_upper_quantile(self, probability: float = 0.995) -> float:
        """Return a finite prior quantile for proposal grids, never support."""
        quantile = float(probability)
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("probability must lie strictly between zero and one.")
        if self.family == "shifted_gamma":
            return float(
                self.minimum
                + self.gamma_scale
                * gammaincinv(self.gamma_shape, quantile)
            )
        return self.minimum + quantile * (self.maximum - self.minimum)

    def in_support(self, value: ArrayLike) -> BoolResult:
        """Return whether each scalar or batched strength is in prior support."""
        values = np.asarray(value, dtype=np.float64)
        support = (
            np.isfinite(values)
            & (values >= self.minimum)
            & (values <= self.support_maximum)
        )
        return _bool_result(
            np.asarray(support, dtype=np.bool_),
            scalar_input=values.ndim == 0,
        )

    def log_prob(self, value: ArrayLike) -> FloatResult:
        """Evaluate the selected normalized prior log density."""
        values = np.asarray(value, dtype=np.float64)
        support = np.asarray(
            np.isfinite(values)
            & (values >= self.minimum)
            & (values <= self.support_maximum),
            dtype=np.bool_,
        )
        if self.family == "bounded_uniform":
            log_density = -float(np.log(self.maximum - self.minimum))
            result = np.where(support, log_density, float("-inf"))
        else:
            shifted = values - self.minimum
            positive = support & (shifted > 0.0)
            safe_shifted = np.maximum(shifted, np.finfo(np.float64).tiny)
            log_density = (
                (self.gamma_shape - 1.0) * np.log(safe_shifted)
                - shifted / self.gamma_scale
                - gammaln(self.gamma_shape)
                - self.gamma_shape * np.log(self.gamma_scale)
            )
            at_boundary = support & (shifted == 0.0)
            boundary_log_density = (
                -np.log(self.gamma_scale)
                if self.gamma_shape == 1.0
                else float("-inf")
            )
            result = np.where(positive, log_density, float("-inf"))
            result = np.where(at_boundary, boundary_log_density, result)
        return _float_result(
            np.asarray(result, dtype=np.float64),
            scalar_input=values.ndim == 0,
        )

    def sample(
        self,
        size: SampleShape = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> FloatResult:
        """Draw scalar or batched strengths from the physical prior."""
        if rng is None:
            raise ValueError("Strength-prior sampling requires an explicit RNG.")
        generator = rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if self.family == "shifted_gamma":
            sampled = self.minimum + generator.gamma(
                shape=self.gamma_shape,
                scale=self.gamma_scale,
                size=size,
            )
        else:
            sampled = generator.uniform(self.minimum, self.maximum, size=size)
        if size is None:
            return float(sampled)
        return np.asarray(sampled, dtype=np.float64)
