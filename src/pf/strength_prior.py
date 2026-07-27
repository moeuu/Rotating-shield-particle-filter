"""Bounded physical prior for source strengths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    """Represent a normalized bounded-uniform physical strength prior."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate finite physical bounds before inference starts."""
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum must be finite and nonnegative.")
        if not np.isfinite(maximum) or maximum <= minimum:
            raise ValueError("maximum must be finite and greater than minimum.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def in_support(self, value: ArrayLike) -> BoolResult:
        """Return whether each scalar or batched strength is in prior support."""
        values = np.asarray(value, dtype=np.float64)
        support = (
            np.isfinite(values)
            & (values >= self.minimum)
            & (values <= self.maximum)
        )
        return _bool_result(
            np.asarray(support, dtype=np.bool_),
            scalar_input=values.ndim == 0,
        )

    def log_prob(self, value: ArrayLike) -> FloatResult:
        """Evaluate the normalized bounded-uniform log density."""
        values = np.asarray(value, dtype=np.float64)
        support = np.asarray(
            np.isfinite(values)
            & (values >= self.minimum)
            & (values <= self.maximum),
            dtype=np.bool_,
        )
        log_density = -float(np.log(self.maximum - self.minimum))
        result = np.where(support, log_density, float("-inf"))
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
        generator = np.random.default_rng() if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        sampled = generator.uniform(self.minimum, self.maximum, size=size)
        if size is None:
            return float(sampled)
        return np.asarray(sampled, dtype=np.float64)
