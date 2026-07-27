"""Exact finite-dictionary priors and birth/death RJ-MH bookkeeping.

The source-position state is a canonical, strictly increasing set of indices
into a finite surface dictionary.  Conditional on cardinality ``K``, its prior
is

``p(S | K) = prod(areas[S]) / e_K(areas)``,

where ``e_K`` is the elementary symmetric polynomial of order ``K``.  The
functions in this module only implement probability bookkeeping.  They do not
evaluate a measurement likelihood or mutate particle-filter state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
FloatResult: TypeAlias = float | FloatArray
SampleShape: TypeAlias = int | tuple[int, ...] | None
MoveKind: TypeAlias = Literal["birth", "death"]

BIRTH_DEATH_LOG_ABS_JACOBIAN = 0.0


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


def _validated_area_weights(areas: ArrayLike) -> FloatArray:
    """Return a read-only copy of finite, strictly positive area weights."""
    values = np.asarray(areas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("areas must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("areas must contain only finite positive values.")
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _logsumexp(values: FloatArray, *, axis: int | None = None) -> FloatArray:
    """Evaluate log-sum-exp with NumPy while preserving all-negative infinity."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    shifted = np.where(
        finite_maximum,
        array - maximum,
        float("-inf"),
    )
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(
        finite_maximum,
        maximum + np.log(total),
        float("-inf"),
    )
    if axis is None:
        result = np.squeeze(result)
    else:
        result = np.squeeze(result, axis=axis)
    return np.asarray(result, dtype=np.float64)


def _normalized_log_area_weights(areas: FloatArray) -> FloatArray:
    """Return log area masses normalized to sum to one."""
    log_areas = np.log(np.asarray(areas, dtype=np.float64))
    log_total = float(_logsumexp(log_areas))
    result = np.asarray(log_areas - log_total, dtype=np.float64)
    result.setflags(write=False)
    return result


def log_elementary_symmetric_normalizers(
    areas: ArrayLike,
    *,
    max_cardinality: int | None = None,
) -> FloatArray:
    """Return ``log(e_k)`` for normalized areas and orders zero through ``K``.

    The normalization of the area vector is immaterial to the resulting set
    probabilities.  Normalizing first prevents avoidable overflow.  Each
    recurrence order is evaluated for every dictionary prefix with a NumPy
    accumulate, so there is no Python loop over surface candidates.
    """
    area_values = _validated_area_weights(areas)
    dictionary_size = int(area_values.size)
    if max_cardinality is None:
        maximum = dictionary_size
    else:
        maximum = _positive_integer(
            max_cardinality,
            name="max_cardinality",
            allow_zero=True,
        )
    if maximum > dictionary_size:
        raise ValueError(
            "max_cardinality cannot exceed the surface dictionary size."
        )

    log_weights = _normalized_log_area_weights(area_values)
    normalizers = np.full(maximum + 1, float("-inf"), dtype=np.float64)
    normalizers[0] = 0.0

    # previous[n] is log(e_{order-1}) for the first n dictionary entries.
    previous = np.zeros(dictionary_size + 1, dtype=np.float64)
    for order in range(1, maximum + 1):
        contributions = log_weights + previous[:-1]
        current = np.full(
            dictionary_size + 1,
            float("-inf"),
            dtype=np.float64,
        )
        current[1:] = np.logaddexp.accumulate(contributions)
        normalizers[order] = current[-1]
        previous = current
    normalizers.setflags(write=False)
    return normalizers


def _surface_sets(
    values: ArrayLike,
    *,
    dictionary_size: int,
    name: str,
) -> IntArray:
    """Return validated batched canonical surface-index sets."""
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, cardinality).")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integer dictionary indices.")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= dictionary_size):
        raise ValueError(f"{name} contains an out-of-range dictionary index.")
    if result.shape[1] > 1 and np.any(np.diff(result, axis=1) <= 0):
        raise ValueError(
            f"{name} must be strictly increasing with no duplicate indices."
        )
    return result


def _integer_batch_vector(
    values: ArrayLike,
    *,
    batch_size: int,
    name: str,
) -> IntArray:
    """Broadcast one integer value per batch row."""
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integer values.")
    try:
        broadcast = np.broadcast_to(raw, (batch_size,))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to ({batch_size},)."
        ) from exc
    return np.asarray(broadcast, dtype=np.int64)


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
class SurfaceAdjacency:
    """Store an undirected finite-surface graph as padded neighbor rows."""

    dictionary_size: int
    edges: ArrayLike
    _neighbors: IntArray = field(init=False, repr=False)
    _degrees: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate, deduplicate, and batch-index undirected surface edges."""
        dictionary_size = _positive_integer(
            self.dictionary_size,
            name="dictionary_size",
            allow_zero=False,
        )
        raw_edges = np.asarray(self.edges)
        if raw_edges.size == 0:
            raw_edges = np.zeros((0, 2), dtype=np.int64)
        if (
            raw_edges.ndim != 2
            or raw_edges.shape[1] != 2
            or not np.issubdtype(raw_edges.dtype, np.integer)
        ):
            raise ValueError("edges must be an integer array shaped (E, 2).")
        canonical = np.sort(
            np.asarray(raw_edges, dtype=np.int64),
            axis=1,
        )
        if np.any(canonical < 0) or np.any(canonical >= dictionary_size):
            raise ValueError("edges contains an out-of-range patch index.")
        if np.any(canonical[:, 0] == canonical[:, 1]):
            raise ValueError("surface adjacency cannot contain self edges.")
        canonical = np.unique(canonical, axis=0)
        if canonical.shape[0] == 0:
            degrees = np.zeros(dictionary_size, dtype=np.int64)
            neighbors = np.full((dictionary_size, 0), -1, dtype=np.int64)
        else:
            directed = np.concatenate(
                [canonical, canonical[:, ::-1]],
                axis=0,
            )
            order = np.lexsort((directed[:, 1], directed[:, 0]))
            directed = directed[order]
            degrees = np.bincount(
                directed[:, 0],
                minlength=dictionary_size,
            ).astype(np.int64, copy=False)
            maximum_degree = int(np.max(degrees))
            neighbors = np.full(
                (dictionary_size, maximum_degree),
                -1,
                dtype=np.int64,
            )
            starts = np.cumsum(
                np.concatenate(
                    [
                        np.zeros(1, dtype=np.int64),
                        degrees[:-1],
                    ]
                )
            )
            slots = np.arange(directed.shape[0], dtype=np.int64) - np.repeat(
                starts,
                degrees,
            )
            neighbors[directed[:, 0], slots] = directed[:, 1]
        canonical.setflags(write=False)
        degrees.setflags(write=False)
        neighbors.setflags(write=False)
        object.__setattr__(self, "dictionary_size", dictionary_size)
        object.__setattr__(self, "edges", canonical)
        object.__setattr__(self, "_degrees", degrees)
        object.__setattr__(self, "_neighbors", neighbors)

    @property
    def neighbors(self) -> IntArray:
        """Return read-only padded neighbor indices, using ``-1`` as padding."""
        return self._neighbors

    @property
    def degrees(self) -> IntArray:
        """Return the undirected graph degree of every surface patch."""
        return self._degrees

    def _available_neighbors(
        self,
        current_surface_sets: ArrayLike,
        source_columns: ArrayLike,
    ) -> tuple[IntArray, IntArray, BoolArray]:
        """Return old patches, padded neighbors, and unoccupied masks in batch."""
        sets = _surface_sets(
            current_surface_sets,
            dictionary_size=self.dictionary_size,
            name="current_surface_sets",
        )
        cardinality = int(sets.shape[1])
        if cardinality == 0:
            raise ValueError("local position moves require positive cardinality.")
        columns = _integer_batch_vector(
            source_columns,
            batch_size=sets.shape[0],
            name="source_columns",
        )
        if np.any(columns < 0) or np.any(columns >= cardinality):
            raise ValueError("source_columns contains an out-of-range column.")
        rows = np.arange(sets.shape[0], dtype=np.int64)
        old_indices = sets[rows, columns]
        candidates = self._neighbors[old_indices]
        valid = candidates >= 0
        if cardinality > 1 and candidates.shape[1] > 0:
            active_occupied = (
                np.arange(cardinality, dtype=np.int64)[None, :]
                != columns[:, None]
            )
            occupied = np.any(
                (
                    candidates[:, :, None]
                    == sets[:, None, :]
                )
                & active_occupied[:, None, :],
                axis=2,
            )
            valid &= ~occupied
        return old_indices, candidates, valid

    def sample_unoccupied_neighbors(
        self,
        current_surface_sets: ArrayLike,
        source_columns: ArrayLike,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[IntArray, IntArray, BoolArray]:
        """Sample one uniform unoccupied adjacent patch per state in batch."""
        generator = np.random.default_rng() if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        old_indices, candidates, valid = self._available_neighbors(
            current_surface_sets,
            source_columns,
        )
        available_degrees = np.sum(valid, axis=1).astype(
            np.int64,
            copy=False,
        )
        movable = available_degrees > 0
        proposed = old_indices.copy()
        if candidates.shape[1] == 0 or not np.any(movable):
            return proposed, available_degrees, movable
        ranks = np.floor(
            generator.random(old_indices.size)
            * np.maximum(available_degrees, 1)
        ).astype(np.int64)
        cumulative = np.cumsum(valid, axis=1)
        selected_slots = np.argmax(
            cumulative > ranks[:, None],
            axis=1,
        )
        rows = np.arange(old_indices.size, dtype=np.int64)
        selected = candidates[rows, selected_slots]
        proposed[movable] = selected[movable]
        return proposed, available_degrees, movable

    def available_neighbor_degrees(
        self,
        center_indices: ArrayLike,
        occupied_surface_sets: ArrayLike,
    ) -> IntArray:
        """Count unoccupied neighbors of batched centers against reduced sets."""
        occupied = _surface_sets(
            occupied_surface_sets,
            dictionary_size=self.dictionary_size,
            name="occupied_surface_sets",
        )
        centers = _integer_batch_vector(
            center_indices,
            batch_size=occupied.shape[0],
            name="center_indices",
        )
        if np.any(centers < 0) or np.any(centers >= self.dictionary_size):
            raise ValueError("center_indices contains an out-of-range index.")
        candidates = self._neighbors[centers]
        valid = candidates >= 0
        if occupied.shape[1] > 0 and candidates.shape[1] > 0:
            valid &= ~np.any(
                candidates[:, :, None] == occupied[:, None, :],
                axis=2,
            )
        return np.sum(valid, axis=1).astype(np.int64, copy=False)


@dataclass(frozen=True)
class SurfaceSetPrior:
    """Area-weighted normalized prior over canonical surface-index sets."""

    areas: ArrayLike
    max_cardinality: int | None = None
    _log_area_masses: FloatArray = field(init=False, repr=False)
    _area_masses: FloatArray = field(init=False, repr=False)
    _log_normalizers: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the dictionary and precompute all requested normalizers."""
        area_values = _validated_area_weights(self.areas)
        dictionary_size = int(area_values.size)
        maximum = (
            dictionary_size
            if self.max_cardinality is None
            else _positive_integer(
                self.max_cardinality,
                name="max_cardinality",
                allow_zero=True,
            )
        )
        if maximum > dictionary_size:
            raise ValueError(
                "max_cardinality cannot exceed the surface dictionary size."
            )

        log_area_masses = _normalized_log_area_weights(area_values)
        area_masses = np.exp(log_area_masses)
        area_masses /= np.sum(area_masses)
        area_masses.setflags(write=False)
        log_normalizers = log_elementary_symmetric_normalizers(
            area_values,
            max_cardinality=maximum,
        )

        object.__setattr__(self, "areas", area_values)
        object.__setattr__(self, "max_cardinality", maximum)
        object.__setattr__(self, "_log_area_masses", log_area_masses)
        object.__setattr__(self, "_area_masses", area_masses)
        object.__setattr__(self, "_log_normalizers", log_normalizers)

    @property
    def dictionary_size(self) -> int:
        """Return the number of entries in the finite surface dictionary."""
        return int(np.asarray(self.areas).size)

    @property
    def area_masses(self) -> FloatArray:
        """Return read-only area weights normalized to sum to one."""
        return self._area_masses

    @property
    def log_area_masses(self) -> FloatArray:
        """Return read-only log area weights normalized to sum to one."""
        return self._log_area_masses

    @property
    def log_normalizers(self) -> FloatArray:
        """Return read-only ``log(e_k)`` values through the configured maximum."""
        return self._log_normalizers

    def log_prob(self, surface_sets: ArrayLike) -> FloatArray:
        """Evaluate normalized ``log p(S | K)`` for a batch of canonical sets."""
        sets = _surface_sets(
            surface_sets,
            dictionary_size=self.dictionary_size,
            name="surface_sets",
        )
        cardinality = int(sets.shape[1])
        if cardinality > int(self.max_cardinality):
            raise ValueError(
                "surface-set cardinality exceeds the precomputed maximum."
            )
        if cardinality == 0:
            return np.zeros(sets.shape[0], dtype=np.float64)
        return np.asarray(
            np.sum(self._log_area_masses[sets], axis=1)
            - self._log_normalizers[cardinality],
            dtype=np.float64,
        )

    def sample_rejection(
        self,
        cardinality: int,
        sample_count: int,
        *,
        rng: np.random.Generator | None = None,
        proposal_batch_size: int | None = None,
        max_proposals: int = 1_000_000,
    ) -> IntArray:
        """Draw exact canonical sets with a batched iid-then-reject sampler."""
        return sample_surface_sets_rejection(
            self.areas,
            cardinality,
            sample_count,
            rng=rng,
            proposal_batch_size=proposal_batch_size,
            max_proposals=max_proposals,
        )


def sample_surface_sets_rejection(
    areas: ArrayLike,
    cardinality: int,
    sample_count: int,
    *,
    rng: np.random.Generator | None = None,
    proposal_batch_size: int | None = None,
    max_proposals: int = 1_000_000,
) -> IntArray:
    """Sample area-weighted sets by rejecting duplicate iid categorical draws.

    Conditional on all ``K`` iid draws being distinct, every unordered set has
    ``K!`` orderings and therefore probability proportional to the product of
    its area weights.  Sorting accepted rows yields the canonical set without
    changing that distribution.
    """
    area_values = _validated_area_weights(areas)
    dictionary_size = int(area_values.size)
    order = _positive_integer(
        cardinality,
        name="cardinality",
        allow_zero=True,
    )
    count = _positive_integer(
        sample_count,
        name="sample_count",
        allow_zero=True,
    )
    proposal_limit = _positive_integer(
        max_proposals,
        name="max_proposals",
        allow_zero=False,
    )
    if order > dictionary_size:
        raise ValueError(
            "cardinality cannot exceed the surface dictionary size."
        )
    if count == 0 or order == 0:
        return np.empty((count, order), dtype=np.int64)

    if proposal_batch_size is None:
        batch_capacity = max(256, min(proposal_limit, 4 * count))
    else:
        batch_capacity = _positive_integer(
            proposal_batch_size,
            name="proposal_batch_size",
            allow_zero=False,
        )
    generator = np.random.default_rng() if rng is None else rng
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    log_probabilities = _normalized_log_area_weights(area_values)
    probabilities = np.exp(log_probabilities)
    probabilities /= np.sum(probabilities)

    accepted_batches: list[IntArray] = []
    accepted_count = 0
    proposal_count = 0
    while accepted_count < count and proposal_count < proposal_limit:
        batch_size = min(batch_capacity, proposal_limit - proposal_count)
        proposals = generator.choice(
            dictionary_size,
            size=(batch_size, order),
            replace=True,
            p=probabilities,
        )
        proposals.sort(axis=1)
        distinct = np.all(np.diff(proposals, axis=1) > 0, axis=1)
        accepted = np.asarray(proposals[distinct], dtype=np.int64)
        remaining = count - accepted_count
        if accepted.shape[0] > remaining:
            accepted = accepted[:remaining]
        if accepted.size > 0:
            accepted_batches.append(accepted)
            accepted_count += int(accepted.shape[0])
        proposal_count += batch_size

    if accepted_count < count:
        raise RuntimeError(
            "batched rejection sampler exhausted max_proposals; reduce "
            "cardinality, use less concentrated area weights, or raise the "
            "explicit proposal limit."
        )
    return np.concatenate(accepted_batches, axis=0)


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
        generator = np.random.default_rng() if rng is None else rng
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


def add_surface_indices(
    current_surface_sets: ArrayLike,
    birth_surface_indices: ArrayLike,
    *,
    dictionary_size: int,
) -> IntArray:
    """Insert one unused dictionary index per row and return canonical sets."""
    sets = _surface_sets(
        current_surface_sets,
        dictionary_size=dictionary_size,
        name="current_surface_sets",
    )
    birth_indices = _integer_batch_vector(
        birth_surface_indices,
        batch_size=sets.shape[0],
        name="birth_surface_indices",
    )
    if np.any(birth_indices < 0) or np.any(
        birth_indices >= dictionary_size
    ):
        raise ValueError("birth_surface_indices contains an out-of-range index.")
    if sets.shape[1] > 0 and np.any(
        np.any(sets == birth_indices[:, None], axis=1)
    ):
        raise ValueError("a birth surface index cannot duplicate an existing one.")
    proposed = np.concatenate((sets, birth_indices[:, None]), axis=1)
    proposed.sort(axis=1)
    return np.asarray(proposed, dtype=np.int64)


def remove_surface_columns(
    current_surface_sets: ArrayLike,
    death_columns: ArrayLike,
    *,
    dictionary_size: int,
) -> IntArray:
    """Remove one source column per row from canonical surface-index sets."""
    sets = _surface_sets(
        current_surface_sets,
        dictionary_size=dictionary_size,
        name="current_surface_sets",
    )
    cardinality = int(sets.shape[1])
    if cardinality == 0:
        raise ValueError("death is unavailable at zero cardinality.")
    columns = _integer_batch_vector(
        death_columns,
        batch_size=sets.shape[0],
        name="death_columns",
    )
    if np.any(columns < 0) or np.any(columns >= cardinality):
        raise ValueError("death_columns contains an out-of-range source column.")
    keep = np.arange(cardinality)[None, :] != columns[:, None]
    return np.asarray(sets[keep].reshape(sets.shape[0], cardinality - 1))


def conditional_birth_surface_log_probability(
    surface_prior: SurfaceSetPrior,
    current_surface_sets: ArrayLike,
    birth_surface_indices: ArrayLike,
) -> FloatArray:
    """Return area-weighted proposal log mass over currently unused indices."""
    sets = _surface_sets(
        current_surface_sets,
        dictionary_size=surface_prior.dictionary_size,
        name="current_surface_sets",
    )
    if sets.shape[1] >= surface_prior.dictionary_size:
        raise ValueError("birth is unavailable when every surface index is used.")
    birth_indices = _integer_batch_vector(
        birth_surface_indices,
        batch_size=sets.shape[0],
        name="birth_surface_indices",
    )
    add_surface_indices(
        sets,
        birth_indices,
        dictionary_size=surface_prior.dictionary_size,
    )

    if sets.shape[1] == 0:
        log_remaining_mass = np.zeros(sets.shape[0], dtype=np.float64)
    else:
        occupied_log_mass = _logsumexp(
            surface_prior.log_area_masses[sets],
            axis=1,
        )
        if np.any(occupied_log_mass >= 0.0):
            raise FloatingPointError(
                "unused area mass is not representable for this dictionary."
            )
        log_remaining_mass = np.log(-np.expm1(occupied_log_mass))
    return np.asarray(
        surface_prior.log_area_masses[birth_indices] - log_remaining_mass,
        dtype=np.float64,
    )


def uniform_death_index_log_probability(
    cardinalities: ArrayLike,
) -> FloatResult:
    """Return ``-log(K)`` for uniform selection among current source slots."""
    raw = np.asarray(cardinalities)
    scalar_input = raw.ndim == 0
    values = np.asarray(raw, dtype=np.float64)
    if (
        np.any(~np.isfinite(values))
        or np.any(values != np.floor(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("cardinalities must contain positive integers.")
    result = -np.log(values)
    return _float_result(
        np.asarray(result, dtype=np.float64),
        scalar_input=scalar_input,
    )


def local_position_log_acceptance_ratio(
    *,
    old_surface_indices: ArrayLike,
    new_surface_indices: ArrayLike,
    forward_available_degrees: ArrayLike,
    reverse_available_degrees: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    surface_prior: SurfaceSetPrior,
) -> FloatArray:
    """Return raw MH ratios for uniform unoccupied-neighbor position moves.

    Source-column selection is uniform in both directions and cancels.  For an
    undirected edge ``old <-> new``, the remaining proposal correction is the
    ratio of forward to reverse available-neighbor counts.
    """
    values = (
        old_surface_indices,
        new_surface_indices,
        forward_available_degrees,
        reverse_available_degrees,
        log_likelihood_ratio,
    )
    batch_size = max(np.asarray(value).size for value in values)
    old_indices = _integer_batch_vector(
        old_surface_indices,
        batch_size=batch_size,
        name="old_surface_indices",
    )
    new_indices = _integer_batch_vector(
        new_surface_indices,
        batch_size=batch_size,
        name="new_surface_indices",
    )
    forward_degrees = _integer_batch_vector(
        forward_available_degrees,
        batch_size=batch_size,
        name="forward_available_degrees",
    )
    reverse_degrees = _integer_batch_vector(
        reverse_available_degrees,
        batch_size=batch_size,
        name="reverse_available_degrees",
    )
    if (
        np.any(old_indices < 0)
        or np.any(old_indices >= surface_prior.dictionary_size)
        or np.any(new_indices < 0)
        or np.any(new_indices >= surface_prior.dictionary_size)
    ):
        raise ValueError("local position move contains an invalid patch index.")
    likelihood_ratio = _log_batch_vector(
        log_likelihood_ratio,
        batch_size=batch_size,
        name="log_likelihood_ratio",
    )
    valid = (forward_degrees > 0) & (reverse_degrees > 0)
    result = np.full(batch_size, float("-inf"), dtype=np.float64)
    result[valid] = (
        likelihood_ratio[valid]
        + surface_prior.log_area_masses[new_indices[valid]]
        - surface_prior.log_area_masses[old_indices[valid]]
        + np.log(forward_degrees[valid])
        - np.log(reverse_degrees[valid])
    )
    return result


def birth_log_acceptance_ratio(
    *,
    current_surface_sets: ArrayLike,
    birth_surface_indices: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    surface_prior: SurfaceSetPrior,
    cardinality_prior: CardinalityPrior,
    move_probabilities: BirthDeathMoveProbabilities,
    log_strength_prior_density: ArrayLike,
    log_forward_position_proposal: ArrayLike,
    log_forward_strength_proposal: ArrayLike,
    log_reverse_death_index_probability: ArrayLike,
) -> FloatArray:
    """Return the exact raw log MH ratio for batched source births.

    ``log_likelihood_ratio`` is ``log L(proposed) - log L(current)``.
    The dimension-matching map appends the proposed strength and its inverse
    removes it, so its absolute Jacobian is exactly one.
    """
    current = _surface_sets(
        current_surface_sets,
        dictionary_size=surface_prior.dictionary_size,
        name="current_surface_sets",
    )
    batch_size, current_cardinality = current.shape
    proposed_cardinality = current_cardinality + 1
    if proposed_cardinality > int(surface_prior.max_cardinality):
        raise ValueError("birth exceeds the surface-prior cardinality limit.")
    if proposed_cardinality > cardinality_prior.max_cardinality:
        raise ValueError("birth exceeds the cardinality-prior representation.")
    proposed = add_surface_indices(
        current,
        birth_surface_indices,
        dictionary_size=surface_prior.dictionary_size,
    )

    forward_move = float(
        move_probabilities.log_probability(
            "birth",
            current_cardinality,
        )
    )
    reverse_move = float(
        move_probabilities.log_probability(
            "death",
            proposed_cardinality,
        )
    )
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError(
            "birth and its reverse death must both have positive move "
            "probability."
        )

    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed_cardinality))
        - float(cardinality_prior.log_prob(current_cardinality))
        + surface_prior.log_prob(proposed)
        - surface_prior.log_prob(current)
        + _log_batch_vector(
            log_strength_prior_density,
            batch_size=batch_size,
            name="log_strength_prior_density",
        )
        + reverse_move
        + _log_batch_vector(
            log_reverse_death_index_probability,
            batch_size=batch_size,
            name="log_reverse_death_index_probability",
        )
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
        + BIRTH_DEATH_LOG_ABS_JACOBIAN
    )
    if np.any(np.isnan(result)):
        raise ValueError("birth log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


def death_log_acceptance_ratio(
    *,
    current_surface_sets: ArrayLike,
    death_columns: ArrayLike,
    log_likelihood_ratio: ArrayLike,
    surface_prior: SurfaceSetPrior,
    cardinality_prior: CardinalityPrior,
    move_probabilities: BirthDeathMoveProbabilities,
    log_removed_strength_prior_density: ArrayLike,
    log_forward_death_index_probability: ArrayLike,
    log_reverse_position_proposal: ArrayLike,
    log_reverse_strength_proposal: ArrayLike,
) -> FloatArray:
    """Return the exact raw log MH ratio for batched source deaths.

    ``log_likelihood_ratio`` is ``log L(proposed) - log L(current)``.
    The removed strength is the auxiliary variable of the reverse birth, and
    the identity dimension-matching map again has absolute Jacobian one.
    """
    current = _surface_sets(
        current_surface_sets,
        dictionary_size=surface_prior.dictionary_size,
        name="current_surface_sets",
    )
    batch_size, current_cardinality = current.shape
    if current_cardinality == 0:
        raise ValueError("death is unavailable at zero cardinality.")
    proposed_cardinality = current_cardinality - 1
    proposed = remove_surface_columns(
        current,
        death_columns,
        dictionary_size=surface_prior.dictionary_size,
    )

    forward_move = float(
        move_probabilities.log_probability(
            "death",
            current_cardinality,
        )
    )
    reverse_move = float(
        move_probabilities.log_probability(
            "birth",
            proposed_cardinality,
        )
    )
    if not np.isfinite(forward_move) or not np.isfinite(reverse_move):
        raise ValueError(
            "death and its reverse birth must both have positive move "
            "probability."
        )

    result = (
        _log_batch_vector(
            log_likelihood_ratio,
            batch_size=batch_size,
            name="log_likelihood_ratio",
        )
        + float(cardinality_prior.log_prob(proposed_cardinality))
        - float(cardinality_prior.log_prob(current_cardinality))
        + surface_prior.log_prob(proposed)
        - surface_prior.log_prob(current)
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
        - _log_batch_vector(
            log_forward_death_index_probability,
            batch_size=batch_size,
            name="log_forward_death_index_probability",
        )
        + BIRTH_DEATH_LOG_ABS_JACOBIAN
    )
    if np.any(np.isnan(result)):
        raise ValueError("death log acceptance ratio is undefined.")
    return np.asarray(result, dtype=np.float64)


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
