"""Prior distributions for positive source strengths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr
from scipy.stats import truncnorm


StrengthPriorKind: TypeAlias = Literal["uniform", "log_uniform", "lognormal"]
SampleShape: TypeAlias = int | tuple[int, ...] | None
FloatResult: TypeAlias = float | NDArray[np.float64]
BoolResult: TypeAlias = bool | NDArray[np.bool_]


def _log_subtract_exp(log_larger: float, log_smaller: float) -> float:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably."""
    if np.isneginf(log_smaller):
        return float(log_larger)
    if log_smaller >= log_larger:
        return float("-inf")
    return float(
        log_larger
        + np.log(-np.expm1(float(log_smaller) - float(log_larger)))
    )


def _standard_normal_interval_log_probability(
    lower: float,
    upper: float,
) -> float:
    """Return the log probability of one standard-normal interval."""
    if lower >= upper:
        return float("-inf")
    if lower >= 0.0:
        return _log_subtract_exp(
            float(log_ndtr(-lower)),
            float(log_ndtr(-upper)),
        )
    return _log_subtract_exp(
        float(log_ndtr(upper)),
        float(log_ndtr(lower)),
    )


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
    """Represent one normalized source-strength prior with optional bounds."""

    kind: StrengthPriorKind | str
    minimum: float = 0.0
    maximum: float | None = None
    log_mean: float = 9.0
    log_sigma: float = 1.0
    _log_normalizer: float = field(init=False, repr=False)
    _standard_lower: float = field(init=False, repr=False)
    _standard_upper: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize configuration and precompute distribution constants."""
        kind = str(self.kind).strip().lower().replace("-", "_")
        if kind not in {"uniform", "log_uniform", "lognormal"}:
            raise ValueError(
                "kind must be uniform, log_uniform, or lognormal."
            )

        minimum = float(self.minimum)
        maximum = None if self.maximum is None else float(self.maximum)
        log_mean = float(self.log_mean)
        log_sigma = float(self.log_sigma)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum must be finite and nonnegative.")
        if maximum is not None:
            if not np.isfinite(maximum) or maximum <= minimum:
                raise ValueError("maximum must be finite and greater than minimum.")

        standard_lower = float("-inf")
        standard_upper = float("inf")
        if kind == "uniform":
            if maximum is None:
                raise ValueError("uniform prior requires a finite maximum.")
            log_normalizer = float(np.log(maximum - minimum))
        elif kind == "log_uniform":
            if minimum <= 0.0:
                raise ValueError("log_uniform prior requires a positive minimum.")
            if maximum is None:
                raise ValueError("log_uniform prior requires a finite maximum.")
            log_normalizer = float(
                np.log(np.log(maximum) - np.log(minimum))
            )
        else:
            if not np.isfinite(log_mean):
                raise ValueError("log_mean must be finite.")
            if not np.isfinite(log_sigma) or log_sigma <= 0.0:
                raise ValueError("log_sigma must be finite and positive.")
            if minimum > 0.0:
                standard_lower = (
                    float(np.log(minimum)) - log_mean
                ) / log_sigma
            if maximum is not None:
                standard_upper = (
                    float(np.log(maximum)) - log_mean
                ) / log_sigma
            log_normalizer = _standard_normal_interval_log_probability(
                standard_lower,
                standard_upper,
            )
            if not np.isfinite(log_normalizer):
                raise ValueError(
                    "lognormal bounds contain no representable probability mass."
                )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "log_mean", log_mean)
        object.__setattr__(self, "log_sigma", log_sigma)
        object.__setattr__(self, "_log_normalizer", log_normalizer)
        object.__setattr__(self, "_standard_lower", standard_lower)
        object.__setattr__(self, "_standard_upper", standard_upper)

    def in_support(self, value: ArrayLike) -> BoolResult:
        """Return whether each scalar or batched strength is in prior support."""
        values = np.asarray(value, dtype=np.float64)
        support = np.isfinite(values) & (values >= self.minimum)
        if self.kind in {"log_uniform", "lognormal"}:
            support &= values > 0.0
        if self.maximum is not None:
            support &= values <= self.maximum
        return _bool_result(
            np.asarray(support, dtype=np.bool_),
            scalar_input=values.ndim == 0,
        )

    def log_prob(self, value: ArrayLike) -> FloatResult:
        """Evaluate normalized log density for scalar or batched strengths."""
        values = np.asarray(value, dtype=np.float64)
        support = np.asarray(
            np.isfinite(values)
            & (values >= self.minimum)
            & (
                True
                if self.maximum is None
                else values <= self.maximum
            ),
            dtype=np.bool_,
        )
        if self.kind in {"log_uniform", "lognormal"}:
            support &= values > 0.0

        safe_values = np.where(support, values, 1.0)
        if self.kind == "uniform":
            density = np.full(
                values.shape,
                -self._log_normalizer,
                dtype=np.float64,
            )
        elif self.kind == "log_uniform":
            density = -np.log(safe_values) - self._log_normalizer
        else:
            standardized = (
                np.log(safe_values) - self.log_mean
            ) / self.log_sigma
            density = (
                -np.log(safe_values)
                - np.log(self.log_sigma)
                - 0.5 * np.log(2.0 * np.pi)
                - 0.5 * np.square(standardized)
                - self._log_normalizer
            )
        result = np.where(support, density, float("-inf"))
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
        """Draw scalar or batched strengths without boundary clipping."""
        generator = np.random.default_rng() if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")

        if self.kind == "uniform":
            sampled = generator.uniform(
                self.minimum,
                float(self.maximum),
                size=size,
            )
        elif self.kind == "log_uniform":
            sampled = np.exp(
                generator.uniform(
                    np.log(self.minimum),
                    np.log(float(self.maximum)),
                    size=size,
                )
            )
        else:
            sampled_log = truncnorm.rvs(
                self._standard_lower,
                self._standard_upper,
                loc=self.log_mean,
                scale=self.log_sigma,
                size=size,
                random_state=generator,
            )
            sampled = np.exp(sampled_log)

        if size is None:
            return float(sampled)
        return np.asarray(sampled, dtype=np.float64)
