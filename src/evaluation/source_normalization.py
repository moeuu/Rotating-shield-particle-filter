"""Normalize heterogeneous source records at the evaluation boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROBABILITY_ROUNDOFF_ATOL = 1.0e-12


@dataclass(frozen=True)
class Source:
    """Store one normalized source for evaluation metrics."""

    pos: NDArray[np.float64]
    strength: float
    surface_kind: str | None = None


def as_position_array(value: Sequence[float]) -> NDArray[np.float64]:
    """Return one finite three-coordinate position array."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError("Position must contain exactly three finite coordinates.")
    return array


def non_negative_finite(value: Any, *, name: str) -> float:
    """Return a finite nonnegative scalar for a physical metric input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def unit_interval_probability(value: Any, *, name: str) -> float:
    """Return a probability, clipping only numeric boundary roundoff."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1].") from exc
    if (
        not np.isfinite(numeric)
        or numeric < -PROBABILITY_ROUNDOFF_ATOL
        or numeric > 1.0 + PROBABILITY_ROUNDOFF_ATOL
    ):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return float(np.clip(numeric, 0.0, 1.0))


def _extract_strength(value: Any) -> float | None:
    """Extract strength from a mapping or object when present."""
    for key in ("strength", "intensity_cps_1m", "intensity"):
        if isinstance(value, Mapping) and key in value:
            return float(value[key])
        if hasattr(value, key):
            return float(getattr(value, key))
    return None


def _extract_position(value: Any) -> NDArray[np.float64] | None:
    """Extract a position array from a mapping or object when present."""
    if isinstance(value, Mapping):
        if "pos" in value:
            return as_position_array(value["pos"])
        if "position" in value:
            return as_position_array(value["position"])
    if hasattr(value, "pos"):
        return as_position_array(value.pos)
    if hasattr(value, "position"):
        return as_position_array(value.position)
    return None


def _extract_surface_kind(value: Any) -> str | None:
    """Extract an optional normalized physical-surface label."""
    for key in ("surface_kind", "surface", "source_surface_kind"):
        if isinstance(value, Mapping) and value.get(key) is not None:
            return str(value[key]).strip().lower()
        if hasattr(value, key) and getattr(value, key) is not None:
            return str(getattr(value, key)).strip().lower()
    return None


def normalize_source(entry: Any) -> Source:
    """Convert one supported source-like value to the stable contract."""
    if isinstance(entry, Source):
        return Source(
            pos=as_position_array(entry.pos),
            strength=non_negative_finite(entry.strength, name="source strength"),
            surface_kind=(
                None
                if entry.surface_kind is None
                else str(entry.surface_kind).strip().lower()
            ),
        )
    if isinstance(entry, (tuple, list, np.ndarray)) and len(entry) == 4:
        return Source(
            pos=as_position_array(entry[:3]),
            strength=non_negative_finite(entry[3], name="source strength"),
        )
    position = _extract_position(entry)
    strength = _extract_strength(entry)
    if position is None or strength is None:
        raise ValueError("Unsupported source entry format.")
    return Source(
        pos=position,
        strength=non_negative_finite(strength, name="source strength"),
        surface_kind=_extract_surface_kind(entry),
    )


def normalize_sources(entries: Iterable[Any] | None) -> list[Source]:
    """Normalize a possibly absent iterable of source-like values."""
    if entries is None:
        return []
    return [normalize_source(entry) for entry in entries]


__all__ = [
    "Source",
    "as_position_array",
    "non_negative_finite",
    "normalize_source",
    "normalize_sources",
    "unit_interval_probability",
]
