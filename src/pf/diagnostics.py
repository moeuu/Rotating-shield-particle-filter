"""Diagnostic state helpers for isotope particle filters."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pf.state import IsotopeState


def reset_step_diagnostics(target: Any) -> None:
    """Reset per-step diagnostic counters on an isotope filter-like object."""
    target.last_ess = None
    target.last_ess_pre = None
    target.last_ess_post = None
    target.last_resample_ess = False
    target.last_resample_count = 0
    target.last_birth_count = 0
    target.last_death_count = 0
    target.last_temper_steps = []
    target.last_temper_resample_count = 0
    target.last_temper_min_ess = None
    target.last_station_unique_ancestor_count = None
    target.last_cumulative_unique_ancestor_count = None
    target.last_source_event_diagnostics = []
    target.last_structural_timing_s = {}
    target.last_structural_transition_weight_mass = {}
    target.last_structural_rejection_diagnostics = {}
    target._structural_mh_component_samples = {}
    target._resample_count_in_observation = 0


def build_source_event_record(
    *,
    event: str,
    isotope: str,
    state: IsotopeState,
    source_idx: int,
    position_xyz: NDArray[np.float64],
    reason: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Return a source-slot diagnostic record or None for an invalid source."""
    idx = int(source_idx)
    if idx < 0 or idx >= int(state.num_sources):
        return None
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    if np.any(~np.isfinite(position)):
        raise ValueError("Diagnostic source position must be finite.")
    record: dict[str, object] = {
        "event": str(event),
        "isotope": str(isotope),
        "reason": str(reason),
        "source_index": idx,
        "position": [float(value) for value in position],
        "strength": float(state.strengths[idx]),
    }
    if extra:
        record.update(extra)
    return record
