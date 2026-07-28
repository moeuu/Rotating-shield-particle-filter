"""Probability bookkeeping for continuous-surface exact RJ/MH moves.

Source positions live at continuous coordinates inside a surface atlas.
This module evaluates cardinality, birth/death, position, and
split/merge proposal ratios; it neither evaluates measurement likelihoods nor
mutates particle-filter state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri


TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY = (
    "independent_truncated_poisson_surface_source_count_v1"
)
EXPLICIT_CARDINALITY_PRIOR_POLICY = (
    "explicit_pre_evaluation_cardinality_probability_vector_v1"
)
CARDINALITY_PRIOR_POLICIES = frozenset(
    {
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
        EXPLICIT_CARDINALITY_PRIOR_POLICY,
    }
)


def validate_cardinality_prior_policy(
    policy: object,
    *,
    has_explicit_probabilities: bool,
) -> str:
    """Validate that one named K-prior policy matches its parameterization."""
    normalized = str(policy).strip()
    if normalized not in CARDINALITY_PRIOR_POLICIES:
        raise ValueError(
            "structural_cardinality_prior_policy must be one of "
            f"{sorted(CARDINALITY_PRIOR_POLICIES)}."
        )
    if (
        normalized == TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY
        and has_explicit_probabilities
    ):
        raise ValueError(
            "The truncated-Poisson cardinality policy cannot be combined with "
            "structural_cardinality_prior_probs."
        )
    if (
        normalized == EXPLICIT_CARDINALITY_PRIOR_POLICY
        and not has_explicit_probabilities
    ):
        raise ValueError(
            "The explicit cardinality-vector policy requires "
            "structural_cardinality_prior_probs."
        )
    return normalized


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatResult: TypeAlias = float | FloatArray
SampleShape: TypeAlias = int | tuple[int, ...] | None
MoveKind: TypeAlias = Literal["birth", "death"]
SplitMergeMoveKind: TypeAlias = Literal["split", "merge"]


@dataclass(frozen=True)
class ContinuousSurfacePositionProposal:
    """Represent one sweep-fixed full-support chart/UV proposal.

    The proposal is expressed against the same ``(chart_id, u, v)`` base
    measure as the surface prior.  Conditional on a chart, ``u`` and ``v`` are
    uniform on the unit square.  Chart mass is an explicit mixture of the
    physical-area prior and a nonnegative data-matched-filter component.
    """

    area_prior_probabilities: ArrayLike
    alignment_scores: ArrayLike
    prior_component_probability: float = 0.5
    chart_probabilities: FloatArray = field(init=False, repr=False)
    log_chart_probabilities: FloatArray = field(init=False, repr=False)
    data_component_probabilities: FloatArray = field(init=False, repr=False)
    data_informative: bool = field(init=False)

    def __post_init__(self) -> None:
        """Validate and freeze the normalized mixture probabilities."""
        prior = np.array(
            self.area_prior_probabilities,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        alignment = np.array(
            self.alignment_scores,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        prior_probability = float(self.prior_component_probability)
        if (
            prior.size == 0
            or alignment.size != prior.size
            or np.any(~np.isfinite(prior))
            or np.any(prior <= 0.0)
        ):
            raise ValueError(
                "area_prior_probabilities must contain positive finite mass "
                "for every surface chart."
            )
        if np.any(~np.isfinite(alignment)) or np.any(alignment < 0.0):
            raise ValueError(
                "alignment_scores must be finite and nonnegative."
            )
        if (
            not np.isfinite(prior_probability)
            or prior_probability <= 0.0
            or prior_probability > 1.0
        ):
            raise ValueError(
                "prior_component_probability must lie in (0, 1]."
            )
        prior = prior / float(np.sum(prior, dtype=np.float64))
        maximum_alignment = float(np.max(alignment, initial=0.0))
        normalized_alignment = (
            alignment / maximum_alignment
            if maximum_alignment > 0.0
            else alignment
        )
        aligned_area_mass = prior * normalized_alignment
        alignment_mass = float(
            np.sum(aligned_area_mass, dtype=np.float64)
        )
        informative = bool(
            np.isfinite(alignment_mass) and alignment_mass > 0.0
        )
        if informative:
            data_component = aligned_area_mass / alignment_mass
            mixture = (
                prior_probability * prior
                + (1.0 - prior_probability) * data_component
            )
            mixture = mixture / float(
                np.sum(mixture, dtype=np.float64)
            )
        else:
            data_component = prior.copy()
            mixture = prior.copy()
        if (
            np.any(~np.isfinite(mixture))
            or np.any(mixture <= 0.0)
            or np.any(
                mixture
                < prior_probability * prior
                - 16.0 * np.finfo(np.float64).eps
            )
        ):
            raise ValueError(
                "Surface proposal mixture failed its full-support contract."
            )
        prior = np.asarray(prior, dtype=np.float64)
        alignment = np.asarray(alignment, dtype=np.float64)
        data_component = np.asarray(data_component, dtype=np.float64)
        mixture = np.asarray(mixture, dtype=np.float64)
        for values in (prior, alignment, data_component, mixture):
            values.setflags(write=False)
        log_mixture = np.log(mixture)
        log_mixture.setflags(write=False)
        object.__setattr__(self, "area_prior_probabilities", prior)
        object.__setattr__(self, "alignment_scores", alignment)
        object.__setattr__(
            self,
            "prior_component_probability",
            prior_probability,
        )
        object.__setattr__(
            self,
            "data_component_probabilities",
            data_component,
        )
        object.__setattr__(self, "chart_probabilities", mixture)
        object.__setattr__(
            self,
            "log_chart_probabilities",
            log_mixture,
        )
        object.__setattr__(self, "data_informative", informative)

    def log_density(self, chart_ids: ArrayLike) -> FloatArray:
        """Return log proposal density in chart/unit-square coordinates."""
        raw = np.asarray(chart_ids)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("chart_ids must contain integers.")
        ids = np.asarray(raw, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.chart_probabilities.size):
            raise ValueError("chart_ids lie outside proposal support.")
        return np.asarray(
            self.log_chart_probabilities[ids],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ContinuousStrengthProposal:
    """Represent a chart-conditional full-support birth-strength proposal.

    The proposal mixes the bounded-uniform physical prior with a truncated
    normal centered on a sweep-fixed full-spectrum residual estimate for each
    surface chart.  Both sampling and density evaluation use the same
    normalized mixture, so birth and reverse-death ratios remain exact.
    """

    minimum: float
    maximum: float
    data_locations_by_chart: ArrayLike
    data_sigma: float
    prior_component_probability: float = 0.5
    data_informative: bool = True

    def __post_init__(self) -> None:
        """Validate and freeze the chart-conditional proposal parameters."""
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        sigma = float(self.data_sigma)
        prior_probability = float(self.prior_component_probability)
        locations = np.array(
            self.data_locations_by_chart,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or maximum <= minimum
        ):
            raise ValueError(
                "Strength-proposal bounds must be finite and increasing."
            )
        if (
            locations.size == 0
            or np.any(~np.isfinite(locations))
            or np.any(locations < minimum)
            or np.any(locations > maximum)
        ):
            raise ValueError(
                "Every strength-proposal location must lie inside support."
            )
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Strength-proposal data_sigma must be positive.")
        if (
            not np.isfinite(prior_probability)
            or prior_probability <= 0.0
            or prior_probability > 1.0
        ):
            raise ValueError(
                "Strength prior_component_probability must lie in (0, 1]."
            )
        locations.setflags(write=False)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "data_sigma", sigma)
        object.__setattr__(
            self,
            "prior_component_probability",
            prior_probability,
        )
        object.__setattr__(self, "data_locations_by_chart", locations)
        object.__setattr__(
            self,
            "data_informative",
            bool(self.data_informative),
        )

    def _validated_chart_ids(self, chart_ids: ArrayLike) -> IntArray:
        """Return chart identifiers inside the proposal table."""
        raw = np.asarray(chart_ids)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("Strength-proposal chart_ids must be integers.")
        ids = np.asarray(raw, dtype=np.int64)
        if np.any(ids < 0) or np.any(
            ids >= np.asarray(self.data_locations_by_chart).size
        ):
            raise ValueError(
                "Strength-proposal chart_ids lie outside proposal support."
            )
        return ids

    def log_density(
        self,
        chart_ids: ArrayLike,
        strengths: ArrayLike,
    ) -> FloatArray:
        """Return the exact log mixture density for chart/strength pairs."""
        ids = self._validated_chart_ids(chart_ids)
        values = np.asarray(strengths, dtype=np.float64)
        try:
            ids, values = np.broadcast_arrays(ids, values)
        except ValueError as exc:
            raise ValueError(
                "Strengths must broadcast with proposal chart_ids."
            ) from exc
        in_support = (
            np.isfinite(values)
            & (values >= self.minimum)
            & (values <= self.maximum)
        )
        prior_density = 1.0 / (self.maximum - self.minimum)
        if (
            not self.data_informative
            or self.prior_component_probability >= 1.0
        ):
            return np.where(
                in_support,
                math.log(prior_density),
                float("-inf"),
            ).astype(np.float64)
        locations = np.asarray(self.data_locations_by_chart)[ids]
        lower_z = (self.minimum - locations) / self.data_sigma
        upper_z = (self.maximum - locations) / self.data_sigma
        normalization = ndtr(upper_z) - ndtr(lower_z)
        standardized = (values - locations) / self.data_sigma
        data_density = (
            np.exp(-0.5 * standardized**2)
            / (
                math.sqrt(2.0 * math.pi)
                * self.data_sigma
                * normalization
            )
        )
        mixture_density = (
            self.prior_component_probability * prior_density
            + (1.0 - self.prior_component_probability) * data_density
        )
        return np.where(
            in_support,
            np.log(mixture_density),
            float("-inf"),
        ).astype(np.float64)

    def sample(
        self,
        chart_ids: ArrayLike,
        *,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Draw strengths from the exact chart-conditional mixture."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "Strength-proposal sampling requires a NumPy Generator."
            )
        ids = self._validated_chart_ids(chart_ids)
        result = rng.uniform(
            self.minimum,
            self.maximum,
            size=ids.shape,
        )
        if (
            not self.data_informative
            or self.prior_component_probability >= 1.0
        ):
            return np.asarray(result, dtype=np.float64)
        use_data = (
            rng.random(ids.shape) >= self.prior_component_probability
        )
        if not np.any(use_data):
            return np.asarray(result, dtype=np.float64)
        locations = np.asarray(self.data_locations_by_chart)[ids[use_data]]
        lower_cdf = ndtr(
            (self.minimum - locations) / self.data_sigma
        )
        upper_cdf = ndtr(
            (self.maximum - locations) / self.data_sigma
        )
        uniforms = lower_cdf + rng.random(locations.shape) * (
            upper_cdf - lower_cdf
        )
        eps = np.finfo(np.float64).eps
        uniforms = np.clip(uniforms, eps, 1.0 - eps)
        result[use_data] = (
            locations + self.data_sigma * ndtri(uniforms)
        )
        return np.clip(
            np.asarray(result, dtype=np.float64),
            self.minimum,
            self.maximum,
        )


def truncated_poisson_cardinality_probabilities(
    max_cardinality: int,
    expected_cardinality: float,
) -> FloatArray:
    """Return a normalized truncated-Poisson source-count prior."""
    maximum = _positive_integer(
        max_cardinality,
        name="max_cardinality",
        allow_zero=False,
    )
    expected = float(expected_cardinality)
    if not np.isfinite(expected) or expected <= 0.0:
        raise ValueError("expected_cardinality must be finite and positive.")
    support = np.arange(maximum + 1, dtype=np.float64)
    log_mass = (
        support * math.log(expected)
        - np.asarray(
            [math.lgamma(float(value) + 1.0) for value in support],
            dtype=np.float64,
        )
    )
    log_mass -= float(np.max(log_mass))
    mass = np.exp(log_mass)
    mass /= float(np.sum(mass, dtype=np.float64))
    return np.asarray(mass, dtype=np.float64)


def _positive_integer(value: int, *, name: str, allow_zero: bool) -> int:
    """Return a validated integer configuration value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    lower = 0 if allow_zero else 1
    if result < lower:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _log_batch_vector(
    values: ArrayLike,
    *,
    batch_size: int,
    name: str,
) -> FloatArray:
    """Broadcast one log-probability term per batch row."""
    raw = np.asarray(values, dtype=np.float64)
    try:
        broadcast = np.broadcast_to(raw, (batch_size,))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to ({batch_size},)."
        ) from exc
    result = np.asarray(broadcast, dtype=np.float64)
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} cannot contain NaN.")
    return result


def _float_result(values: FloatArray, *, scalar_input: bool) -> FloatResult:
    """Return a Python float for scalar input and an array otherwise."""
    if scalar_input:
        return float(np.asarray(values, dtype=np.float64))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class CardinalityPrior:
    """Explicit normalized prior over integer cardinalities ``0..K_max``."""

    probabilities: ArrayLike
    _log_probabilities: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the configured cardinality probabilities."""
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError(
                "probabilities must be a nonempty one-dimensional array."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("cardinality probabilities must be finite.")
        if np.any(probabilities < 0.0) or not np.any(probabilities > 0.0):
            raise ValueError(
                "cardinality probabilities must be nonnegative with "
                "positive total mass."
            )
        normalized = np.array(
            probabilities / np.sum(probabilities),
            dtype=np.float64,
            copy=True,
        )
        log_probabilities = np.full(
            normalized.shape,
            float("-inf"),
            dtype=np.float64,
        )
        positive = normalized > 0.0
        log_probabilities[positive] = np.log(normalized[positive])
        normalized.setflags(write=False)
        log_probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", normalized)
        object.__setattr__(self, "_log_probabilities", log_probabilities)

    @classmethod
    def uniform(
        cls,
        max_cardinality: int,
        *,
        min_cardinality: int = 0,
    ) -> CardinalityPrior:
        """Construct a uniform prior on one inclusive cardinality interval."""
        maximum = _positive_integer(
            max_cardinality,
            name="max_cardinality",
            allow_zero=True,
        )
        minimum = _positive_integer(
            min_cardinality,
            name="min_cardinality",
            allow_zero=True,
        )
        if minimum > maximum:
            raise ValueError(
                "min_cardinality cannot exceed max_cardinality."
            )
        probabilities = np.zeros(maximum + 1, dtype=np.float64)
        probabilities[minimum : maximum + 1] = 1.0
        return cls(probabilities)

    @property
    def max_cardinality(self) -> int:
        """Return the largest representable cardinality."""
        return int(np.asarray(self.probabilities).size - 1)

    @property
    def log_probabilities(self) -> FloatArray:
        """Return read-only normalized log probabilities."""
        return self._log_probabilities

    def log_prob(self, cardinalities: ArrayLike) -> FloatResult:
        """Evaluate normalized log mass, returning ``-inf`` outside support."""
        raw = np.asarray(cardinalities)
        scalar_input = raw.ndim == 0
        values = np.asarray(raw, dtype=np.float64)
        integral = np.isfinite(values) & (values == np.floor(values))
        indices = np.where(integral, values, 0.0).astype(np.int64)
        in_range = (
            integral
            & (indices >= 0)
            & (indices <= self.max_cardinality)
        )
        safe_indices = np.where(in_range, indices, 0)
        result = np.where(
            in_range,
            self._log_probabilities[safe_indices],
            float("-inf"),
        )
        return _float_result(
            np.asarray(result, dtype=np.float64),
            scalar_input=scalar_input,
        )

    def sample(
        self,
        size: SampleShape = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> int | IntArray:
        """Draw scalar or batched cardinalities from the explicit prior."""
        if rng is None:
            raise ValueError("Cardinality-prior sampling requires an explicit RNG.")
        generator = rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        sampled = generator.choice(
            self.max_cardinality + 1,
            size=size,
            p=np.asarray(self.probabilities),
        )
        if size is None:
            return int(sampled)
        return np.asarray(sampled, dtype=np.int64)


@dataclass(frozen=True)
class BirthDeathMoveProbabilities:
    """Boundary-normalized probabilities for proposing birth or death."""

    max_cardinality: int
    min_cardinality: int = 0
    birth_weight: float = 1.0
    death_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate cardinality boundaries and positive interior weights."""
        maximum = _positive_integer(
            self.max_cardinality,
            name="max_cardinality",
            allow_zero=True,
        )
        minimum = _positive_integer(
            self.min_cardinality,
            name="min_cardinality",
            allow_zero=True,
        )
        birth_weight = float(self.birth_weight)
        death_weight = float(self.death_weight)
        if minimum > maximum:
            raise ValueError(
                "min_cardinality cannot exceed max_cardinality."
            )
        if (
            not np.isfinite(birth_weight)
            or not np.isfinite(death_weight)
            or birth_weight <= 0.0
            or death_weight <= 0.0
        ):
            raise ValueError("birth_weight and death_weight must be positive.")
        object.__setattr__(self, "max_cardinality", maximum)
        object.__setattr__(self, "min_cardinality", minimum)
        object.__setattr__(self, "birth_weight", birth_weight)
        object.__setattr__(self, "death_weight", death_weight)

    def probabilities(
        self,
        cardinalities: ArrayLike,
    ) -> tuple[FloatResult, FloatResult]:
        """Return normalized birth and death probabilities at each boundary."""
        raw = np.asarray(cardinalities)
        scalar_input = raw.ndim == 0
        values = np.asarray(raw, dtype=np.float64)
        if (
            np.any(~np.isfinite(values))
            or np.any(values != np.floor(values))
            or np.any(values < self.min_cardinality)
            or np.any(values > self.max_cardinality)
        ):
            raise ValueError(
                "cardinalities must be integers inside configured boundaries."
            )

        birth_available = values < self.max_cardinality
        death_available = values > self.min_cardinality
        birth_mass = self.birth_weight * birth_available
        death_mass = self.death_weight * death_available
        total = birth_mass + death_mass
        birth = np.divide(
            birth_mass,
            total,
            out=np.zeros_like(values, dtype=np.float64),
            where=total > 0.0,
        )
        death = np.divide(
            death_mass,
            total,
            out=np.zeros_like(values, dtype=np.float64),
            where=total > 0.0,
        )
        return (
            _float_result(
                np.asarray(birth, dtype=np.float64),
                scalar_input=scalar_input,
            ),
            _float_result(
                np.asarray(death, dtype=np.float64),
                scalar_input=scalar_input,
            ),
        )

    def log_probability(
        self,
        move: MoveKind,
        cardinalities: ArrayLike,
    ) -> FloatResult:
        """Return the log proposal probability of one structural move kind."""
        birth, death = self.probabilities(cardinalities)
        selected = birth if move == "birth" else death
        if move not in {"birth", "death"}:
            raise ValueError("move must be 'birth' or 'death'.")
        raw = np.asarray(cardinalities)
        probabilities = np.asarray(selected, dtype=np.float64)
        result = np.full(probabilities.shape, float("-inf"), dtype=np.float64)
        positive = probabilities > 0.0
        result[positive] = np.log(probabilities[positive])
        return _float_result(result, scalar_input=raw.ndim == 0)


def log_acceptance_probability(log_acceptance_ratio: ArrayLike) -> FloatResult:
    """Convert a raw MH log ratio to ``log(min(1, ratio))``."""
    raw = np.asarray(log_acceptance_ratio, dtype=np.float64)
    if np.any(np.isnan(raw)):
        raise ValueError("log_acceptance_ratio cannot contain NaN.")
    result = np.minimum(raw, 0.0)
    return _float_result(
        np.asarray(result, dtype=np.float64),
        scalar_input=raw.ndim == 0,
    )


@dataclass(frozen=True)
class SplitMergeMoveProbabilities:
    """Boundary-normalized probabilities for exact split/merge proposals."""

    max_cardinality: int
    split_weight: float = 1.0
    merge_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the cardinality bound and positive direction weights."""
        maximum = _positive_integer(
            self.max_cardinality,
            name="max_cardinality",
            allow_zero=False,
        )
        split_weight = float(self.split_weight)
        merge_weight = float(self.merge_weight)
        if (
            not np.isfinite(split_weight)
            or not np.isfinite(merge_weight)
            or split_weight <= 0.0
            or merge_weight <= 0.0
        ):
            raise ValueError("split_weight and merge_weight must be positive.")
        object.__setattr__(self, "max_cardinality", maximum)
        object.__setattr__(self, "split_weight", split_weight)
        object.__setattr__(self, "merge_weight", merge_weight)

    def probabilities(
        self,
        cardinalities: ArrayLike,
    ) -> tuple[FloatResult, FloatResult]:
        """Return split and merge proposal probabilities at each K boundary."""
        raw = np.asarray(cardinalities)
        scalar_input = raw.ndim == 0
        values = np.asarray(raw, dtype=np.float64)
        if (
            np.any(~np.isfinite(values))
            or np.any(values != np.floor(values))
            or np.any(values < 0.0)
            or np.any(values > self.max_cardinality)
        ):
            raise ValueError(
                "cardinalities must be integers inside configured boundaries."
            )
        split_mass = self.split_weight * (
            (values >= 1.0) & (values < self.max_cardinality)
        )
        merge_mass = self.merge_weight * (values >= 2.0)
        total = split_mass + merge_mass
        split = np.divide(
            split_mass,
            total,
            out=np.zeros_like(values, dtype=np.float64),
            where=total > 0.0,
        )
        merge = np.divide(
            merge_mass,
            total,
            out=np.zeros_like(values, dtype=np.float64),
            where=total > 0.0,
        )
        return (
            _float_result(
                np.asarray(split, dtype=np.float64),
                scalar_input=scalar_input,
            ),
            _float_result(
                np.asarray(merge, dtype=np.float64),
                scalar_input=scalar_input,
            ),
        )

    def log_probability(
        self,
        move: SplitMergeMoveKind,
        cardinalities: ArrayLike,
    ) -> FloatResult:
        """Return the log probability of one split/merge direction."""
        if move not in {"split", "merge"}:
            raise ValueError("move must be 'split' or 'merge'.")
        split, merge = self.probabilities(cardinalities)
        selected = split if move == "split" else merge
        raw = np.asarray(cardinalities)
        probabilities = np.asarray(selected, dtype=np.float64)
        result = np.full(probabilities.shape, float("-inf"), dtype=np.float64)
        positive = probabilities > 0.0
        result[positive] = np.log(probabilities[positive])
        return _float_result(result, scalar_input=raw.ndim == 0)


def continuous_birth_log_acceptance_ratio(
    *,
    current_cardinality: int,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: BirthDeathMoveProbabilities,
    log_position_prior_density: ArrayLike,
    log_strength_prior_density: ArrayLike,
    log_forward_position_proposal: ArrayLike,
    log_forward_strength_proposal: ArrayLike,
    log_abs_jacobian: ArrayLike = 0.0,
) -> FloatArray:
    """Return the exact birth ratio for canonical continuous surface states.

    Conditional on ``K``, source position/strength pairs are iid and then put
    into deterministic canonical order.  Their density on that ordered domain
    therefore contains ``K!``.  The explicit ``log(K + 1)`` target term and
    reverse ``-log(K + 1)`` death-index proposal below make that base-measure
    choice auditable instead of relying on an implicit cancellation.
    """
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=True,
    )
    proposed = current + 1
    if proposed > cardinality_prior.max_cardinality:
        raise ValueError("birth exceeds the cardinality prior support.")
    values = (
        log_likelihood_ratio,
        log_position_prior_density,
        log_strength_prior_density,
        log_forward_position_proposal,
        log_forward_strength_proposal,
        log_abs_jacobian,
    )
    batch_size = max(np.asarray(value).size for value in values)
    forward_move = float(move_probabilities.log_probability("birth", current))
    reverse_move = float(move_probabilities.log_probability("death", proposed))
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("birth and reverse death must both be available.")
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        + np.log(float(proposed))
        + _log_batch_vector(
            log_position_prior_density,
            batch_size=batch_size,
            name="log_position_prior_density",
        )
        + _log_batch_vector(
            log_strength_prior_density,
            batch_size=batch_size,
            name="log_strength_prior_density",
        )
        + reverse_move
        - np.log(float(proposed))
        - forward_move
        - _log_batch_vector(
            log_forward_position_proposal,
            batch_size=batch_size,
            name="log_forward_position_proposal",
        )
        - _log_batch_vector(
            log_forward_strength_proposal,
            batch_size=batch_size,
            name="log_forward_strength_proposal",
        )
        + _log_batch_vector(
            log_abs_jacobian,
            batch_size=batch_size,
            name="log_abs_jacobian",
        )
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous birth log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def continuous_death_log_acceptance_ratio(
    *,
    current_cardinality: int,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: BirthDeathMoveProbabilities,
    log_removed_position_prior_density: ArrayLike,
    log_removed_strength_prior_density: ArrayLike,
    log_reverse_position_proposal: ArrayLike,
    log_reverse_strength_proposal: ArrayLike,
    log_abs_reverse_jacobian: ArrayLike = 0.0,
) -> FloatArray:
    """Return the exact reciprocal death ratio for continuous surface states."""
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=False,
    )
    proposed = current - 1
    values = (
        log_likelihood_ratio,
        log_removed_position_prior_density,
        log_removed_strength_prior_density,
        log_reverse_position_proposal,
        log_reverse_strength_proposal,
        log_abs_reverse_jacobian,
    )
    batch_size = max(np.asarray(value).size for value in values)
    forward_move = float(move_probabilities.log_probability("death", current))
    reverse_move = float(move_probabilities.log_probability("birth", proposed))
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("death and reverse birth must both be available.")
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        - np.log(float(current))
        - _log_batch_vector(
            log_removed_position_prior_density,
            batch_size=batch_size,
            name="log_removed_position_prior_density",
        )
        - _log_batch_vector(
            log_removed_strength_prior_density,
            batch_size=batch_size,
            name="log_removed_strength_prior_density",
        )
        + reverse_move
        + _log_batch_vector(
            log_reverse_position_proposal,
            batch_size=batch_size,
            name="log_reverse_position_proposal",
        )
        + _log_batch_vector(
            log_reverse_strength_proposal,
            batch_size=batch_size,
            name="log_reverse_strength_proposal",
        )
        - forward_move
        + np.log(float(current))
        + _log_batch_vector(
            log_abs_reverse_jacobian,
            batch_size=batch_size,
            name="log_abs_reverse_jacobian",
        )
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous death log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def continuous_position_log_acceptance_ratio(
    *,
    log_likelihood_ratio: ArrayLike,
    log_old_position_prior_density: ArrayLike,
    log_new_position_prior_density: ArrayLike,
    log_reverse_proposal_density: ArrayLike,
    log_forward_proposal_density: ArrayLike,
    log_abs_jacobian: ArrayLike = 0.0,
) -> FloatArray:
    """Return an exact MH ratio for a continuous surface-position move."""
    values = (
        log_likelihood_ratio,
        log_old_position_prior_density,
        log_new_position_prior_density,
        log_reverse_proposal_density,
        log_forward_proposal_density,
        log_abs_jacobian,
    )
    batch_size = max(np.asarray(value).size for value in values)
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + _log_batch_vector(
            log_new_position_prior_density,
            batch_size=batch_size,
            name="log_new_position_prior_density",
        )
        - _log_batch_vector(
            log_old_position_prior_density,
            batch_size=batch_size,
            name="log_old_position_prior_density",
        )
        + _log_batch_vector(
            log_reverse_proposal_density,
            batch_size=batch_size,
            name="log_reverse_proposal_density",
        )
        - _log_batch_vector(
            log_forward_proposal_density,
            batch_size=batch_size,
            name="log_forward_proposal_density",
        )
        + _log_batch_vector(
            log_abs_jacobian,
            batch_size=batch_size,
            name="log_abs_jacobian",
        )
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous position log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def continuous_joint_position_strength_log_acceptance_ratio(
    *,
    log_likelihood_ratio: ArrayLike,
    log_old_position_prior_density: ArrayLike,
    log_new_position_prior_density: ArrayLike,
    log_old_strength_prior_density: ArrayLike,
    log_new_strength_prior_density: ArrayLike,
    log_reverse_position_proposal_density: ArrayLike,
    log_forward_position_proposal_density: ArrayLike,
    log_reverse_strength_proposal_density: ArrayLike,
    log_forward_strength_proposal_density: ArrayLike,
    log_abs_jacobian: ArrayLike = 0.0,
) -> FloatArray:
    """Return the exact MH ratio for a joint position-and-strength move."""
    values = (
        log_likelihood_ratio,
        log_old_position_prior_density,
        log_new_position_prior_density,
        log_old_strength_prior_density,
        log_new_strength_prior_density,
        log_reverse_position_proposal_density,
        log_forward_position_proposal_density,
        log_reverse_strength_proposal_density,
        log_forward_strength_proposal_density,
        log_abs_jacobian,
    )
    batch_size = max(np.asarray(value).size for value in values)
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + _log_batch_vector(
            log_new_position_prior_density,
            batch_size=batch_size,
            name="log_new_position_prior_density",
        )
        - _log_batch_vector(
            log_old_position_prior_density,
            batch_size=batch_size,
            name="log_old_position_prior_density",
        )
        + _log_batch_vector(
            log_new_strength_prior_density,
            batch_size=batch_size,
            name="log_new_strength_prior_density",
        )
        - _log_batch_vector(
            log_old_strength_prior_density,
            batch_size=batch_size,
            name="log_old_strength_prior_density",
        )
        + _log_batch_vector(
            log_reverse_position_proposal_density,
            batch_size=batch_size,
            name="log_reverse_position_proposal_density",
        )
        - _log_batch_vector(
            log_forward_position_proposal_density,
            batch_size=batch_size,
            name="log_forward_position_proposal_density",
        )
        + _log_batch_vector(
            log_reverse_strength_proposal_density,
            batch_size=batch_size,
            name="log_reverse_strength_proposal_density",
        )
        - _log_batch_vector(
            log_forward_strength_proposal_density,
            batch_size=batch_size,
            name="log_forward_strength_proposal_density",
        )
        + _log_batch_vector(
            log_abs_jacobian,
            batch_size=batch_size,
            name="log_abs_jacobian",
        )
    )
    if np.any(np.isnan(result)):
        raise ValueError(
            "continuous joint position-strength log acceptance ratio is "
            "undefined."
        )
    return np.asarray(result, dtype=np.float64)


def split_fraction_bounds(
    total_strength: ArrayLike,
    *,
    minimum_strength: float,
    maximum_strength: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Return the exact feasible uniform split-fraction interval."""
    total = np.asarray(total_strength, dtype=np.float64)
    minimum = float(minimum_strength)
    maximum = float(maximum_strength)
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum < 0.0
        or maximum <= minimum
    ):
        raise ValueError("Strength bounds must be finite and ordered.")
    safe_total = np.maximum(total, np.finfo(np.float64).tiny)
    lower = np.maximum(minimum / safe_total, 1.0 - maximum / safe_total)
    upper = np.minimum(maximum / safe_total, 1.0 - minimum / safe_total)
    feasible = (
        np.isfinite(total)
        & (total > 0.0)
        & (upper > lower)
        & (lower >= 0.0)
        & (upper <= 1.0)
    )
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(feasible, dtype=np.bool_),
    )


def continuous_split_log_acceptance_ratio(
    *,
    current_cardinality: int,
    total_strength: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: SplitMergeMoveProbabilities,
    log_new_position_prior_density: ArrayLike,
    log_old_strength_prior_density: ArrayLike,
    log_retained_strength_prior_density: ArrayLike,
    log_new_strength_prior_density: ArrayLike,
    log_forward_position_proposal: ArrayLike,
    log_forward_fraction_proposal: ArrayLike,
) -> FloatArray:
    """Return the exact split RJ ratio including the strength-map Jacobian."""
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=False,
    )
    proposed = current + 1
    if proposed > cardinality_prior.max_cardinality:
        raise ValueError("split exceeds the cardinality prior support.")
    values = (
        total_strength,
        log_likelihood_ratio,
        log_new_position_prior_density,
        log_old_strength_prior_density,
        log_retained_strength_prior_density,
        log_new_strength_prior_density,
        log_forward_position_proposal,
        log_forward_fraction_proposal,
    )
    batch_size = max(np.asarray(value).size for value in values)
    total = np.broadcast_to(
        np.asarray(total_strength, dtype=np.float64),
        (batch_size,),
    )
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("total_strength must be finite and positive.")
    forward_move = float(move_probabilities.log_probability("split", current))
    reverse_move = float(move_probabilities.log_probability("merge", proposed))
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("split and reverse merge must both be available.")
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        + np.log(float(proposed))
        + _log_batch_vector(
            log_new_position_prior_density,
            batch_size=batch_size,
            name="log_new_position_prior_density",
        )
        + _log_batch_vector(
            log_retained_strength_prior_density,
            batch_size=batch_size,
            name="log_retained_strength_prior_density",
        )
        + _log_batch_vector(
            log_new_strength_prior_density,
            batch_size=batch_size,
            name="log_new_strength_prior_density",
        )
        - _log_batch_vector(
            log_old_strength_prior_density,
            batch_size=batch_size,
            name="log_old_strength_prior_density",
        )
        + reverse_move
        - np.log(float(proposed * current))
        - forward_move
        + np.log(float(current))
        - _log_batch_vector(
            log_forward_position_proposal,
            batch_size=batch_size,
            name="log_forward_position_proposal",
        )
        - _log_batch_vector(
            log_forward_fraction_proposal,
            batch_size=batch_size,
            name="log_forward_fraction_proposal",
        )
        + np.log(total)
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous split log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def continuous_merge_log_acceptance_ratio(
    *,
    current_cardinality: int,
    merged_strength: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: SplitMergeMoveProbabilities,
    log_deleted_position_prior_density: ArrayLike,
    log_deleted_strength_prior_density: ArrayLike,
    log_retained_strength_prior_density: ArrayLike,
    log_merged_strength_prior_density: ArrayLike,
    log_reverse_position_proposal: ArrayLike,
    log_reverse_fraction_proposal: ArrayLike,
) -> FloatArray:
    """Return the exact merge RJ ratio, reciprocal to the matching split."""
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=False,
    )
    if current < 2:
        raise ValueError("merge requires at least two sources.")
    proposed = current - 1
    values = (
        merged_strength,
        log_likelihood_ratio,
        log_deleted_position_prior_density,
        log_deleted_strength_prior_density,
        log_retained_strength_prior_density,
        log_merged_strength_prior_density,
        log_reverse_position_proposal,
        log_reverse_fraction_proposal,
    )
    batch_size = max(np.asarray(value).size for value in values)
    total = np.broadcast_to(
        np.asarray(merged_strength, dtype=np.float64),
        (batch_size,),
    )
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("merged_strength must be finite and positive.")
    forward_move = float(move_probabilities.log_probability("merge", current))
    reverse_move = float(move_probabilities.log_probability("split", proposed))
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("merge and reverse split must both be available.")
    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        - np.log(float(current))
        - _log_batch_vector(
            log_deleted_position_prior_density,
            batch_size=batch_size,
            name="log_deleted_position_prior_density",
        )
        + _log_batch_vector(
            log_merged_strength_prior_density,
            batch_size=batch_size,
            name="log_merged_strength_prior_density",
        )
        - _log_batch_vector(
            log_deleted_strength_prior_density,
            batch_size=batch_size,
            name="log_deleted_strength_prior_density",
        )
        - _log_batch_vector(
            log_retained_strength_prior_density,
            batch_size=batch_size,
            name="log_retained_strength_prior_density",
        )
        + reverse_move
        - np.log(float(proposed))
        + _log_batch_vector(
            log_reverse_position_proposal,
            batch_size=batch_size,
            name="log_reverse_position_proposal",
        )
        + _log_batch_vector(
            log_reverse_fraction_proposal,
            batch_size=batch_size,
            name="log_reverse_fraction_proposal",
        )
        - forward_move
        + np.log(float(current * proposed))
        - np.log(total)
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous merge log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)
