"""Probability bookkeeping for continuous-surface exact RJ/MH moves.

Source positions live at continuous coordinates inside a surface atlas.
This module evaluates cardinality, birth/death, position, and
split/merge proposal ratios; it neither evaluates measurement likelihoods nor
mutates particle-filter state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from math import comb
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from pf.strength_prior import StrengthPrior


TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY = (
    "independent_truncated_poisson_surface_source_count_v1"
)
POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY = (
    "independent_poisson_with_thin_geometric_capacity_tail_v1"
)
EXPLICIT_CARDINALITY_PRIOR_POLICY = (
    "explicit_pre_evaluation_cardinality_probability_vector_v1"
)
CARDINALITY_PRIOR_POLICIES = frozenset(
    {
        TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
        POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
        EXPLICIT_CARDINALITY_PRIOR_POLICY,
    }
)


def shifted_log_strength_random_walk_log_reverse_ratio(
    current_strengths: ArrayLike,
    proposed_strengths: ArrayLike,
    active_mask: ArrayLike,
    *,
    minimum_strength: float,
) -> FloatArray:
    """Return ``log q(old|new)-log q(new|old)`` for shifted-log walks.

    Gaussian increments are symmetric in ``log(strength - minimum)``.  This
    function supplies the exact change-of-variables term in physical strength
    coordinates and is vectorized over arbitrary leading batch axes.
    """
    current = np.asarray(current_strengths, dtype=np.float64)
    proposed = np.asarray(proposed_strengths, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=np.bool_)
    if current.shape != proposed.shape or mask.shape != current.shape:
        raise ValueError("Strength random-walk arrays must have equal shapes.")
    if current.ndim < 1:
        raise ValueError("Strength random-walk arrays require a source axis.")
    minimum = float(minimum_strength)
    current_shifted = current - minimum
    proposed_shifted = proposed - minimum
    valid = (
        np.all(~mask | np.isfinite(current), axis=-1)
        & np.all(~mask | np.isfinite(proposed), axis=-1)
        & np.all(~mask | (current_shifted > 0.0), axis=-1)
        & np.all(~mask | (proposed_shifted > 0.0), axis=-1)
    )
    terms = np.where(
        mask,
        np.log(np.maximum(proposed_shifted, np.finfo(np.float64).tiny))
        - np.log(np.maximum(current_shifted, np.finfo(np.float64).tiny)),
        0.0,
    )
    result = np.sum(terms, axis=-1, dtype=np.float64)
    return np.where(valid, result, float("-inf"))


def independence_refresh_log_acceptance_ratio(
    *,
    log_target_ratio: ArrayLike,
    log_forward_proposal: ArrayLike,
    log_reverse_proposal: ArrayLike,
) -> FloatArray:
    """Return an exact MH ratio for a nonconserving independence refresh.

    The proposal may change cardinality, positions, and every surviving source
    strength simultaneously. It is a density directly on the old and new
    source coordinates rather than a deterministic strength-transfer map, so
    its Jacobian is exactly one and no cross-source strength conservation is
    implied.
    """
    target, forward, reverse = np.broadcast_arrays(
        np.asarray(log_target_ratio, dtype=np.float64),
        np.asarray(log_forward_proposal, dtype=np.float64),
        np.asarray(log_reverse_proposal, dtype=np.float64),
    )
    if (
        np.any(np.isnan(target))
        or np.any(np.isnan(forward))
        or np.any(np.isnan(reverse))
        or np.any(np.isposinf(forward))
        or np.any(np.isposinf(reverse))
    ):
        raise ValueError("Independence-refresh log densities are invalid.")
    result = target + reverse - forward
    return np.where(
        np.isfinite(forward) & np.isfinite(reverse),
        result,
        float("-inf"),
    ).astype(np.float64)


def validate_cardinality_prior_policy(
    policy: object,
    *,
    has_explicit_probabilities: bool,
) -> str:
    """Validate that one named K-prior policy matches its parameterization."""
    if not isinstance(policy, str) or policy not in CARDINALITY_PRIOR_POLICIES:
        raise ValueError(
            "structural_cardinality_prior_policy must be one of "
            f"{sorted(CARDINALITY_PRIOR_POLICIES)}."
        )
    if (
        policy
        in {
            TRUNCATED_POISSON_CARDINALITY_PRIOR_POLICY,
            POISSON_GEOMETRIC_TAIL_CARDINALITY_PRIOR_POLICY,
        }
        and has_explicit_probabilities
    ):
        raise ValueError(
            "The truncated-Poisson cardinality policy cannot be combined with "
            "structural_cardinality_prior_probs."
        )
    if (
        policy == EXPLICIT_CARDINALITY_PRIOR_POLICY
        and not has_explicit_probabilities
    ):
        raise ValueError(
            "The explicit cardinality-vector policy requires "
            "structural_cardinality_prior_probs."
        )
    return policy


def poisson_geometric_tail_cardinality_probabilities(
    typical_max_cardinality: int,
    hard_max_cardinality: int,
    expected_cardinality: float,
    tail_ratio: float,
) -> FloatArray:
    """Return a Poisson prior with a thin proper tail above the usual limit.

    Cardinalities through ``typical_max_cardinality`` retain their ordinary
    Poisson mass.  Above that point the unnormalized mass decreases by the
    fixed, pre-evaluation ratio ``tail_ratio`` at every extra source until the
    computational hard capacity.  This keeps five sources as the normal
    statistical boundary while avoiding a model space that makes ``K > 5``
    impossible regardless of the observations.
    """
    typical = _positive_integer(
        typical_max_cardinality,
        name="typical_max_cardinality",
        allow_zero=False,
    )
    hard = _positive_integer(
        hard_max_cardinality,
        name="hard_max_cardinality",
        allow_zero=False,
    )
    expected = float(expected_cardinality)
    ratio = float(tail_ratio)
    if hard < typical:
        raise ValueError(
            "hard_max_cardinality cannot be smaller than the typical maximum."
        )
    if not np.isfinite(expected) or expected <= 0.0:
        raise ValueError("expected_cardinality must be finite and positive.")
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("tail_ratio must lie strictly between zero and one.")
    support = np.arange(hard + 1, dtype=np.float64)
    log_mass = np.empty(hard + 1, dtype=np.float64)
    ordinary = support[: typical + 1]
    log_mass[: typical + 1] = (
        ordinary * math.log(expected)
        - np.asarray(
            [math.lgamma(float(value) + 1.0) for value in ordinary],
            dtype=np.float64,
        )
    )
    if hard > typical:
        tail_steps = np.arange(1, hard - typical + 1, dtype=np.float64)
        log_mass[typical + 1 :] = (
            log_mass[typical] + tail_steps * math.log(ratio)
        )
    log_mass -= float(np.max(log_mass))
    probabilities = np.exp(log_mass)
    probabilities /= float(np.sum(probabilities, dtype=np.float64))
    return np.asarray(probabilities, dtype=np.float64)


def bounded_simplex_probability(
    total_strength: ArrayLike,
    *,
    group_size: int,
    minimum_strength: float,
    maximum_strength: float,
) -> FloatArray:
    """Return Dirichlet(1) mass satisfying bounded child strengths.

    Inclusion-exclusion evaluates the exact volume of the simplex cut by the
    common lower and upper child-strength bounds. This normalization is needed
    for reversible multi-component split proposals.
    """
    size = int(group_size)
    if size < 2:
        raise ValueError("group_size must be at least two.")
    total = np.asarray(total_strength, dtype=np.float64)
    minimum = float(minimum_strength)
    maximum = float(maximum_strength)
    if (
        np.any(~np.isfinite(total))
        or not np.isfinite(minimum)
        or np.isnan(maximum)
        or minimum < 0.0
        or maximum <= minimum
    ):
        raise ValueError("Bounded-simplex inputs are invalid.")
    safe_total = np.maximum(total, np.finfo(np.float64).tiny)
    lower = minimum / safe_total
    residual = 1.0 - float(size) * lower
    if np.isposinf(maximum):
        feasible = total > size * minimum
        probability = np.power(np.maximum(residual, 0.0), size - 1)
        return np.where(feasible, np.clip(probability, 0.0, 1.0), 0.0)
    upper = maximum / safe_total
    width = upper - lower
    probability = np.zeros_like(total, dtype=np.float64)
    for subset_size in range(size + 1):
        remainder = np.maximum(
            residual - float(subset_size) * width,
            0.0,
        )
        probability += (
            (-1.0) ** subset_size
            * float(comb(size, subset_size))
            * np.power(remainder, size - 1)
        )
    feasible = (total > size * minimum) & (total < size * maximum)
    return np.where(feasible, np.clip(probability, 0.0, 1.0), 0.0)


def bounded_uniform_simplex_log_density(
    fractions: ArrayLike,
    *,
    total_strength: ArrayLike,
    minimum_strength: float,
    maximum_strength: float,
) -> FloatArray:
    """Return the exact conditional Dirichlet(1) log density."""
    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("fractions must end in a group axis of size at least two.")
    size = int(values.shape[-1])
    total = np.asarray(total_strength, dtype=np.float64)
    if total.shape != values.shape[:-1]:
        raise ValueError("total_strength must align with fraction rows.")
    probability = bounded_simplex_probability(
        total,
        group_size=size,
        minimum_strength=minimum_strength,
        maximum_strength=maximum_strength,
    )
    child_strengths = values * total[..., None]
    valid = (
        np.all(np.isfinite(values), axis=-1)
        & np.isclose(np.sum(values, axis=-1), 1.0, atol=1.0e-12)
        & np.all(child_strengths >= float(minimum_strength), axis=-1)
        & np.all(child_strengths <= float(maximum_strength), axis=-1)
        & (probability > 0.0)
    )
    density = math.lgamma(float(size)) - np.log(
        np.maximum(probability, np.finfo(np.float64).tiny)
    )
    return np.where(valid, density, float("-inf"))


def continuous_group_split_log_acceptance_ratio(
    *,
    log_target_ratio: ArrayLike,
    log_forward_proposal: ArrayLike,
    log_reverse_proposal: ArrayLike,
    total_strength: ArrayLike,
    group_size: int,
) -> FloatArray:
    """Return an exact multi-split RJ log ratio including its Jacobian."""
    size = int(group_size)
    if size < 2:
        raise ValueError("group_size must be at least two.")
    total = np.asarray(total_strength, dtype=np.float64)
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("total_strength must be finite and positive.")
    return (
        np.asarray(log_target_ratio, dtype=np.float64)
        + np.asarray(log_reverse_proposal, dtype=np.float64)
        - np.asarray(log_forward_proposal, dtype=np.float64)
        + float(size - 1) * np.log(total)
    )


def continuous_group_merge_log_acceptance_ratio(
    *,
    log_target_ratio: ArrayLike,
    log_forward_proposal: ArrayLike,
    log_reverse_proposal: ArrayLike,
    merged_strength: ArrayLike,
    group_size: int,
) -> FloatArray:
    """Return the inverse exact multi-merge RJ log ratio."""
    return continuous_group_split_log_acceptance_ratio(
        log_target_ratio=log_target_ratio,
        log_forward_proposal=log_forward_proposal,
        log_reverse_proposal=log_reverse_proposal,
        total_strength=merged_strength,
        group_size=group_size,
    ) - 2.0 * float(int(group_size) - 1) * np.log(
        np.asarray(merged_strength, dtype=np.float64)
    )


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

    The proposal mixes the configured proper physical prior with a truncated
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
    prior_family: str = "bounded_uniform"
    prior_gamma_shape: float = 2.0
    prior_gamma_scale: float = 1.0
    _prior: StrengthPrior = field(init=False, repr=False, compare=False)

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
        prior = StrengthPrior(
            minimum=minimum,
            maximum=maximum,
            family=self.prior_family,
            gamma_shape=self.prior_gamma_shape,
            gamma_scale=self.prior_gamma_scale,
        )
        if (
            locations.size == 0
            or np.any(~np.isfinite(locations))
            or np.any(locations < minimum)
            or np.any(locations > prior.support_maximum)
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
        object.__setattr__(self, "prior_family", prior.family)
        object.__setattr__(self, "prior_gamma_shape", prior.gamma_shape)
        object.__setattr__(self, "prior_gamma_scale", prior.gamma_scale)
        object.__setattr__(self, "_prior", prior)
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
            & (values <= self._prior.support_maximum)
        )
        prior_log_density = np.asarray(
            self._prior.log_prob(values),
            dtype=np.float64,
        )
        if (
            not self.data_informative
            or self.prior_component_probability >= 1.0
        ):
            return np.where(in_support, prior_log_density, float("-inf"))
        locations = np.asarray(self.data_locations_by_chart)[ids]
        lower_z = (self.minimum - locations) / self.data_sigma
        upper_z = (
            (self.maximum - locations) / self.data_sigma
            if self._prior.family == "bounded_uniform"
            else np.full_like(locations, np.inf)
        )
        normalization = ndtr(upper_z) - ndtr(lower_z)
        standardized = (values - locations) / self.data_sigma
        data_log_density = (
            -0.5 * standardized**2
            - math.log(math.sqrt(2.0 * math.pi) * self.data_sigma)
            - np.log(normalization)
        )
        mixture_log_density = np.logaddexp(
            math.log(self.prior_component_probability) + prior_log_density,
            math.log1p(-self.prior_component_probability) + data_log_density,
        )
        return np.where(
            in_support,
            mixture_log_density,
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
        result = np.asarray(
            self._prior.sample(ids.shape, rng=rng),
            dtype=np.float64,
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
        if self._prior.family == "bounded_uniform":
            upper_cdf = ndtr(
                (self.maximum - locations) / self.data_sigma
            )
        else:
            upper_cdf = np.ones_like(locations, dtype=np.float64)
        uniforms = lower_cdf + rng.random(locations.shape) * (
            upper_cdf - lower_cdf
        )
        eps = np.finfo(np.float64).eps
        uniforms = np.clip(uniforms, eps, 1.0 - eps)
        result[use_data] = (
            locations + self.data_sigma * ndtri(uniforms)
        )
        result = np.maximum(np.asarray(result, dtype=np.float64), self.minimum)
        if self._prior.family == "bounded_uniform":
            result = np.minimum(result, self.maximum)
        return result


@dataclass(frozen=True)
class ContinuousBlockStrengthProposal:
    """Represent a row-conditional full-support strength-vector proposal.

    One mixture component draws the complete source-strength vector from the
    proper physical prior.  The other draws the complete vector from
    independent truncated normals around an all-history conditional center.
    Choosing the component once per particle row avoids the exponentially
    unlikely mixture of data-informed and unrelated prior draws produced by
    selecting a component independently for every source.  The normalized
    block density is evaluated explicitly, so the proposal remains suitable
    for exact trans-dimensional MH ratios.
    """

    minimum: float
    maximum: float
    data_locations: ArrayLike
    data_sigma: float
    prior_component_probability: float = 0.5
    prior_family: str = "bounded_uniform"
    prior_gamma_shape: float = 2.0
    prior_gamma_scale: float = 1.0
    _prior: StrengthPrior = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and freeze the batched conditional proposal centers."""
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        sigma = float(self.data_sigma)
        prior_probability = float(self.prior_component_probability)
        locations = np.array(
            self.data_locations,
            dtype=np.float64,
            copy=True,
        )
        prior = StrengthPrior(
            minimum=minimum,
            maximum=maximum,
            family=self.prior_family,
            gamma_shape=self.prior_gamma_shape,
            gamma_scale=self.prior_gamma_scale,
        )
        if locations.ndim != 2 or locations.shape[1] < 1:
            raise ValueError(
                "Block strength-proposal centers require row and source axes."
            )
        if (
            np.any(~np.isfinite(locations))
            or np.any(locations < minimum)
            or np.any(locations > prior.support_maximum)
        ):
            raise ValueError(
                "Every block strength-proposal center must lie in support."
            )
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Block strength-proposal sigma must be positive.")
        if (
            not np.isfinite(prior_probability)
            or prior_probability <= 0.0
            or prior_probability > 1.0
        ):
            raise ValueError(
                "Block prior_component_probability must lie in (0, 1]."
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
        object.__setattr__(self, "data_locations", locations)
        object.__setattr__(self, "prior_family", prior.family)
        object.__setattr__(self, "prior_gamma_shape", prior.gamma_shape)
        object.__setattr__(self, "prior_gamma_scale", prior.gamma_scale)
        object.__setattr__(self, "_prior", prior)

    def _validated_values(self, strengths: ArrayLike) -> NDArray[np.float64]:
        """Return strength rows aligned with the immutable proposal centers."""
        values = np.asarray(strengths, dtype=np.float64)
        if values.shape != np.asarray(self.data_locations).shape:
            raise ValueError(
                "Block strengths must match proposal center row/source axes."
            )
        return values

    def log_density(self, strengths: ArrayLike) -> FloatArray:
        """Return one exact block-mixture log density per particle row."""
        values = self._validated_values(strengths)
        in_support = np.asarray(self._prior.in_support(values), dtype=np.bool_)
        row_in_support = np.all(in_support, axis=1)
        prior_log_density = np.sum(
            np.asarray(self._prior.log_prob(values), dtype=np.float64),
            axis=1,
        )
        if self.prior_component_probability >= 1.0:
            return np.where(
                row_in_support,
                prior_log_density,
                float("-inf"),
            ).astype(np.float64)
        locations = np.asarray(self.data_locations, dtype=np.float64)
        lower_z = (self.minimum - locations) / self.data_sigma
        if self._prior.family == "bounded_uniform":
            upper_z = (self.maximum - locations) / self.data_sigma
        else:
            upper_z = np.full_like(locations, np.inf)
        normalization = ndtr(upper_z) - ndtr(lower_z)
        standardized = (values - locations) / self.data_sigma
        data_log_density = np.sum(
            -0.5 * standardized**2
            - math.log(math.sqrt(2.0 * math.pi) * self.data_sigma)
            - np.log(normalization),
            axis=1,
        )
        mixture = np.logaddexp(
            math.log(self.prior_component_probability) + prior_log_density,
            math.log1p(-self.prior_component_probability)
            + data_log_density,
        )
        return np.where(
            row_in_support,
            mixture,
            float("-inf"),
        ).astype(np.float64)

    def sample(self, *, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw complete strength vectors from the normalized block mixture."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "Block strength-proposal sampling requires a NumPy Generator."
            )
        shape = np.asarray(self.data_locations).shape
        result = np.asarray(
            self._prior.sample(shape, rng=rng),
            dtype=np.float64,
        )
        if self.prior_component_probability >= 1.0:
            return result
        use_data = (
            rng.random(shape[0]) >= self.prior_component_probability
        )
        if not np.any(use_data):
            return result
        locations = np.asarray(self.data_locations)[use_data]
        lower_cdf = ndtr((self.minimum - locations) / self.data_sigma)
        if self._prior.family == "bounded_uniform":
            upper_cdf = ndtr(
                (self.maximum - locations) / self.data_sigma
            )
        else:
            upper_cdf = np.ones_like(locations, dtype=np.float64)
        uniforms = lower_cdf + rng.random(locations.shape) * (
            upper_cdf - lower_cdf
        )
        eps = np.finfo(np.float64).eps
        sampled = locations + self.data_sigma * ndtri(
            np.clip(uniforms, eps, 1.0 - eps)
        )
        sampled = np.maximum(sampled, self.minimum)
        if self._prior.family == "bounded_uniform":
            sampled = np.minimum(sampled, self.maximum)
        result[use_data] = sampled
        return np.asarray(result, dtype=np.float64)


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


def distance_weighted_ordered_pair_probabilities(
    surface_distances_m: ArrayLike,
    *,
    sigma_m: float,
    uniform_component_probability: float,
) -> FloatArray:
    """Return full-support probabilities over ordered source pairs.

    The last array dimension enumerates ordered donor/receiver pairs.  The
    local component favours small intrinsic surface distance, while a positive
    uniform component preserves irreducibility even across disconnected
    surface components.
    """
    distances = np.asarray(surface_distances_m, dtype=np.float64)
    if distances.ndim < 1 or distances.shape[-1] <= 0:
        raise ValueError(
            "surface_distances_m must contain a nonempty pair dimension."
        )
    if np.any(np.isnan(distances)) or np.any(distances < 0.0):
        raise ValueError(
            "surface_distances_m must be nonnegative or positive infinity."
        )
    sigma = float(sigma_m)
    uniform_probability = float(uniform_component_probability)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_m must be finite and positive.")
    if (
        not np.isfinite(uniform_probability)
        or uniform_probability <= 0.0
        or uniform_probability > 1.0
    ):
        raise ValueError(
            "uniform_component_probability must lie in (0, 1]."
        )
    pair_count = int(distances.shape[-1])
    local_mass = np.zeros(distances.shape, dtype=np.float64)
    finite = np.isfinite(distances)
    local_mass[finite] = np.exp(
        -0.5 * np.square(distances[finite] / sigma)
    )
    local_total = np.sum(local_mass, axis=-1, keepdims=True)
    uniform = np.full(
        distances.shape,
        1.0 / pair_count,
        dtype=np.float64,
    )
    local_probabilities = np.divide(
        local_mass,
        local_total,
        out=uniform.copy(),
        where=local_total > 0.0,
    )
    probabilities = (
        uniform_probability * uniform
        + (1.0 - uniform_probability) * local_probabilities
    )
    probabilities /= np.sum(
        probabilities,
        axis=-1,
        keepdims=True,
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.allclose(
            np.sum(probabilities, axis=-1),
            1.0,
            rtol=0.0,
            atol=1.0e-14,
        )
    ):
        raise RuntimeError("Ordered pair proposal probabilities are invalid.")
    return np.asarray(probabilities, dtype=np.float64)


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
        or np.isnan(maximum)
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
    log_forward_parent_selection: ArrayLike | None = None,
    log_reverse_pair_selection: ArrayLike | None = None,
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
    forward_parent_selection = (
        -math.log(float(current))
        if log_forward_parent_selection is None
        else log_forward_parent_selection
    )
    reverse_pair_selection = (
        -math.log(float(proposed * current))
        if log_reverse_pair_selection is None
        else log_reverse_pair_selection
    )
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
        + _log_batch_vector(
            reverse_pair_selection,
            batch_size=batch_size,
            name="log_reverse_pair_selection",
        )
        - forward_move
        - _log_batch_vector(
            forward_parent_selection,
            batch_size=batch_size,
            name="log_forward_parent_selection",
        )
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
    log_forward_pair_selection: ArrayLike | None = None,
    log_reverse_parent_selection: ArrayLike | None = None,
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
    forward_pair_selection = (
        -math.log(float(current * proposed))
        if log_forward_pair_selection is None
        else log_forward_pair_selection
    )
    reverse_parent_selection = (
        -math.log(float(proposed))
        if log_reverse_parent_selection is None
        else log_reverse_parent_selection
    )
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
        + _log_batch_vector(
            reverse_parent_selection,
            batch_size=batch_size,
            name="log_reverse_parent_selection",
        )
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
        - _log_batch_vector(
            forward_pair_selection,
            batch_size=batch_size,
            name="log_forward_pair_selection",
        )
        - np.log(total)
    )
    if np.any(np.isnan(result)):
        raise ValueError("continuous merge log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def continuous_relocated_split_log_acceptance_ratio(
    *,
    current_cardinality: int,
    total_strength: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: SplitMergeMoveProbabilities,
    log_parent_position_prior_density: ArrayLike,
    log_first_child_position_prior_density: ArrayLike,
    log_second_child_position_prior_density: ArrayLike,
    log_parent_strength_prior_density: ArrayLike,
    log_first_child_strength_prior_density: ArrayLike,
    log_second_child_strength_prior_density: ArrayLike,
    log_forward_first_position_proposal: ArrayLike,
    log_forward_second_position_proposal: ArrayLike,
    log_forward_fraction_proposal: ArrayLike,
    log_reverse_merged_position_proposal: ArrayLike,
    log_forward_parent_selection: ArrayLike,
    log_reverse_pair_selection: ArrayLike,
) -> FloatArray:
    """Return the exact RJ ratio for replacing one source by two children."""
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=False,
    )
    proposed = current + 1
    if proposed > cardinality_prior.max_cardinality:
        raise ValueError("relocated split exceeds cardinality support.")
    values = (
        total_strength,
        log_likelihood_ratio,
        log_parent_position_prior_density,
        log_first_child_position_prior_density,
        log_second_child_position_prior_density,
        log_parent_strength_prior_density,
        log_first_child_strength_prior_density,
        log_second_child_strength_prior_density,
        log_forward_first_position_proposal,
        log_forward_second_position_proposal,
        log_forward_fraction_proposal,
        log_reverse_merged_position_proposal,
        log_forward_parent_selection,
        log_reverse_pair_selection,
    )
    batch_size = max(np.asarray(value).size for value in values)
    total = np.broadcast_to(
        np.asarray(total_strength, dtype=np.float64),
        (batch_size,),
    )
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("total_strength must be finite and positive.")
    forward_move = float(
        move_probabilities.log_probability("split", current)
    )
    reverse_move = float(
        move_probabilities.log_probability("merge", proposed)
    )
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("Relocated split and merge must both be available.")

    def _vector(value: ArrayLike, name: str) -> FloatArray:
        """Return one validated vector on the common batch."""
        return _log_batch_vector(
            value,
            batch_size=batch_size,
            name=name,
        )

    result = (
        _vector(log_likelihood_ratio, "log_likelihood_ratio")
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        + np.log(float(proposed))
        + _vector(
            log_first_child_position_prior_density,
            "log_first_child_position_prior_density",
        )
        + _vector(
            log_second_child_position_prior_density,
            "log_second_child_position_prior_density",
        )
        - _vector(
            log_parent_position_prior_density,
            "log_parent_position_prior_density",
        )
        + _vector(
            log_first_child_strength_prior_density,
            "log_first_child_strength_prior_density",
        )
        + _vector(
            log_second_child_strength_prior_density,
            "log_second_child_strength_prior_density",
        )
        - _vector(
            log_parent_strength_prior_density,
            "log_parent_strength_prior_density",
        )
        + reverse_move
        + _vector(
            log_reverse_pair_selection,
            "log_reverse_pair_selection",
        )
        + _vector(
            log_reverse_merged_position_proposal,
            "log_reverse_merged_position_proposal",
        )
        - forward_move
        - _vector(
            log_forward_parent_selection,
            "log_forward_parent_selection",
        )
        - _vector(
            log_forward_first_position_proposal,
            "log_forward_first_position_proposal",
        )
        - _vector(
            log_forward_second_position_proposal,
            "log_forward_second_position_proposal",
        )
        - _vector(
            log_forward_fraction_proposal,
            "log_forward_fraction_proposal",
        )
        + np.log(total)
    )
    if np.any(np.isnan(result)):
        raise ValueError(
            "Relocated split log acceptance ratio is undefined."
        )
    return np.asarray(result, dtype=np.float64)


def continuous_relocated_merge_log_acceptance_ratio(
    *,
    current_cardinality: int,
    merged_strength: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    cardinality_prior: CardinalityPrior,
    move_probabilities: SplitMergeMoveProbabilities,
    log_first_child_position_prior_density: ArrayLike,
    log_second_child_position_prior_density: ArrayLike,
    log_merged_position_prior_density: ArrayLike,
    log_first_child_strength_prior_density: ArrayLike,
    log_second_child_strength_prior_density: ArrayLike,
    log_merged_strength_prior_density: ArrayLike,
    log_forward_pair_selection: ArrayLike,
    log_forward_merged_position_proposal: ArrayLike,
    log_reverse_parent_selection: ArrayLike,
    log_reverse_first_position_proposal: ArrayLike,
    log_reverse_second_position_proposal: ArrayLike,
    log_reverse_fraction_proposal: ArrayLike,
) -> FloatArray:
    """Return the reciprocal exact ratio for merging into a new position."""
    current = _positive_integer(
        current_cardinality,
        name="current_cardinality",
        allow_zero=False,
    )
    if current < 2:
        raise ValueError("relocated merge requires at least two sources.")
    proposed = current - 1
    values = (
        merged_strength,
        log_likelihood_ratio,
        log_first_child_position_prior_density,
        log_second_child_position_prior_density,
        log_merged_position_prior_density,
        log_first_child_strength_prior_density,
        log_second_child_strength_prior_density,
        log_merged_strength_prior_density,
        log_forward_pair_selection,
        log_forward_merged_position_proposal,
        log_reverse_parent_selection,
        log_reverse_first_position_proposal,
        log_reverse_second_position_proposal,
        log_reverse_fraction_proposal,
    )
    batch_size = max(np.asarray(value).size for value in values)
    total = np.broadcast_to(
        np.asarray(merged_strength, dtype=np.float64),
        (batch_size,),
    )
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("merged_strength must be finite and positive.")
    forward_move = float(
        move_probabilities.log_probability("merge", current)
    )
    reverse_move = float(
        move_probabilities.log_probability("split", proposed)
    )
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError("Relocated merge and split must both be available.")

    def _vector(value: ArrayLike, name: str) -> FloatArray:
        """Return one validated vector on the common batch."""
        return _log_batch_vector(
            value,
            batch_size=batch_size,
            name=name,
        )

    result = (
        _vector(log_likelihood_ratio, "log_likelihood_ratio")
        + float(cardinality_prior.log_prob(proposed))
        - float(cardinality_prior.log_prob(current))
        - np.log(float(current))
        + _vector(
            log_merged_position_prior_density,
            "log_merged_position_prior_density",
        )
        - _vector(
            log_first_child_position_prior_density,
            "log_first_child_position_prior_density",
        )
        - _vector(
            log_second_child_position_prior_density,
            "log_second_child_position_prior_density",
        )
        + _vector(
            log_merged_strength_prior_density,
            "log_merged_strength_prior_density",
        )
        - _vector(
            log_first_child_strength_prior_density,
            "log_first_child_strength_prior_density",
        )
        - _vector(
            log_second_child_strength_prior_density,
            "log_second_child_strength_prior_density",
        )
        + reverse_move
        + _vector(
            log_reverse_parent_selection,
            "log_reverse_parent_selection",
        )
        + _vector(
            log_reverse_first_position_proposal,
            "log_reverse_first_position_proposal",
        )
        + _vector(
            log_reverse_second_position_proposal,
            "log_reverse_second_position_proposal",
        )
        + _vector(
            log_reverse_fraction_proposal,
            "log_reverse_fraction_proposal",
        )
        - forward_move
        - _vector(
            log_forward_pair_selection,
            "log_forward_pair_selection",
        )
        - _vector(
            log_forward_merged_position_proposal,
            "log_forward_merged_position_proposal",
        )
        - np.log(total)
    )
    if np.any(np.isnan(result)):
        raise ValueError(
            "Relocated merge log acceptance ratio is undefined."
        )
    return np.asarray(result, dtype=np.float64)
