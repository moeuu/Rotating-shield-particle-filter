"""Define the authoritative continuous-surface state of one isotope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class IsotopeState:
    """Store only source count, continuous surface coordinates, and strengths.

    Cartesian positions are deterministic images of ``surface_chart_ids`` and
    ``surface_uv`` under the filter's immutable surface atlas.  They are never
    stored in state, preventing chart/XYZ divergence.  Shared physical spectrum
    background belongs to the full-spectrum generative model rather than to an
    isotope state.
    """

    num_sources: int
    strengths: NDArray[np.float64]
    surface_chart_ids: NDArray[np.int64]
    surface_uv: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and own all variable-length state arrays."""
        if isinstance(self.num_sources, (bool, np.bool_)) or not isinstance(
            self.num_sources,
            (int, np.integer),
        ):
            raise TypeError("num_sources must be an integer.")
        cardinality = int(self.num_sources)
        strengths = np.array(self.strengths, dtype=np.float64, copy=True).reshape(
            -1
        )
        raw_chart_ids = np.asarray(self.surface_chart_ids)
        if not np.issubdtype(raw_chart_ids.dtype, np.integer):
            raise TypeError("surface_chart_ids must contain integers.")
        chart_ids = np.array(
            raw_chart_ids,
            dtype=np.int64,
            copy=True,
        ).reshape(-1)
        surface_uv = np.array(
            self.surface_uv,
            dtype=np.float64,
            copy=True,
        )
        if (
            cardinality < 0
            or strengths.shape != (cardinality,)
            or chart_ids.shape != (cardinality,)
            or surface_uv.shape != (cardinality, 2)
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
            or np.any(chart_ids < 0)
            or np.any(~np.isfinite(surface_uv))
            or np.any(surface_uv < 0.0)
            or np.any(surface_uv > 1.0)
        ):
            raise ValueError(
                "IsotopeState requires positive finite strengths and valid "
                "chart/UV arrays matching num_sources."
            )
        self.num_sources = cardinality
        self.strengths = strengths
        self.surface_chart_ids = chart_ids
        self.surface_uv = surface_uv

    def copy(self) -> "IsotopeState":
        """Return a deep copy of the authoritative isotope state arrays."""
        return IsotopeState(
            num_sources=int(self.num_sources),
            strengths=self.strengths.copy(),
            surface_chart_ids=self.surface_chart_ids.copy(),
            surface_uv=self.surface_uv.copy(),
        )
