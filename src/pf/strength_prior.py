"""Proper physical priors for source strengths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincinv, gammaln


SampleShape: TypeAlias = int | tuple[int, ...] | None
FloatResult: TypeAlias = float | NDArray[np.float64]
BoolResult: TypeAlias = bool | NDArray[np.bool_]
STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY = 0.995


def _strict_positive_number(value: object, *, name: str) -> float:
    """Return one finite positive configuration number without coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be numeric.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


@dataclass(frozen=True, slots=True)
class ShiftedGammaStrengthPriorConfig:
    """Configure the production upper-unbounded source-strength prior."""

    minimum_cps_1m: float
    shape: float
    scale_cps_1m: float
    kind: str = field(default="shifted_gamma", init=False)

    def __post_init__(self) -> None:
        """Validate and freeze the exact shifted-gamma parameters."""
        minimum = _strict_positive_number(
            self.minimum_cps_1m,
            name="strength_prior.minimum_cps_1m",
        )
        shape = _strict_positive_number(
            self.shape,
            name="strength_prior.shape",
        )
        scale = _strict_positive_number(
            self.scale_cps_1m,
            name="strength_prior.scale_cps_1m",
        )
        if shape < 1.0:
            raise ValueError("strength_prior.shape must be at least one.")
        object.__setattr__(self, "minimum_cps_1m", minimum)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "scale_cps_1m", scale)

    def build(self) -> StrengthPrior:
        """Build the normalized prior with a deterministic proposal quantile."""
        proposal_upper = float(
            self.minimum_cps_1m
            + self.scale_cps_1m
            * gammaincinv(
                self.shape,
                STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY,
            )
        )
        return StrengthPrior(
            minimum=self.minimum_cps_1m,
            maximum=proposal_upper,
            family="shifted_gamma",
            gamma_shape=self.shape,
            gamma_scale=self.scale_cps_1m,
        )


@dataclass(frozen=True, slots=True)
class BoundedUniformStrengthPriorTestConfig:
    """Configure the bounded-uniform prior for deterministic test oracles only."""

    minimum_cps_1m: float
    maximum_cps_1m: float
    kind: str = field(default="bounded_uniform_test_only", init=False)

    def __post_init__(self) -> None:
        """Validate and freeze the explicitly test-only finite bounds."""
        minimum = _strict_positive_number(
            self.minimum_cps_1m,
            name="strength_prior.minimum_cps_1m",
        )
        maximum = _strict_positive_number(
            self.maximum_cps_1m,
            name="strength_prior.maximum_cps_1m",
        )
        if maximum <= minimum:
            raise ValueError(
                "strength_prior.maximum_cps_1m must exceed minimum_cps_1m."
            )
        object.__setattr__(self, "minimum_cps_1m", minimum)
        object.__setattr__(self, "maximum_cps_1m", maximum)

    def build(self) -> StrengthPrior:
        """Build the normalized bounded-uniform test prior."""
        return StrengthPrior(
            minimum=self.minimum_cps_1m,
            maximum=self.maximum_cps_1m,
            family="bounded_uniform",
        )


StrengthPriorConfig: TypeAlias = (
    ShiftedGammaStrengthPriorConfig | BoundedUniformStrengthPriorTestConfig
)


def resolve_strength_prior_config(value: object) -> StrengthPriorConfig:
    """Resolve one exact discriminated strength-prior configuration."""
    if isinstance(
        value,
        (ShiftedGammaStrengthPriorConfig, BoundedUniformStrengthPriorTestConfig),
    ):
        return value
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError("strength_prior must be a string-keyed object.")
    kind = value.get("kind")
    if kind == "shifted_gamma":
        expected = frozenset(
            {"kind", "minimum_cps_1m", "shape", "scale_cps_1m"}
        )
        actual = frozenset(value)
        if actual != expected:
            raise ValueError(
                "shifted_gamma strength_prior keys differ from the exact "
                f"contract: missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}."
            )
        return ShiftedGammaStrengthPriorConfig(
            minimum_cps_1m=value["minimum_cps_1m"],
            shape=value["shape"],
            scale_cps_1m=value["scale_cps_1m"],
        )
    if kind == "bounded_uniform_test_only":
        expected = frozenset({"kind", "minimum_cps_1m", "maximum_cps_1m"})
        actual = frozenset(value)
        if actual != expected:
            raise ValueError(
                "bounded_uniform_test_only strength_prior keys differ from the "
                f"exact contract: missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}."
            )
        return BoundedUniformStrengthPriorTestConfig(
            minimum_cps_1m=value["minimum_cps_1m"],
            maximum_cps_1m=value["maximum_cps_1m"],
        )
    raise ValueError(
        "strength_prior.kind must be 'shifted_gamma' in production or "
        "'bounded_uniform_test_only' in explicit test oracles."
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
    """Represent a normalized physical source-strength prior.

    ``shifted_gamma`` removes the artificial upper support boundary while
    retaining a proper, normalized density. For shifted gamma, ``maximum`` is
    an internal finite proposal reference derived from the fixed proposal
    quantile; it is never accepted from the production configuration and is
    not part of the mathematical support.
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

    def finite_upper_quantile(
        self,
        probability: float = STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY,
    ) -> float:
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
