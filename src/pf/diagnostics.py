"""Diagnostic state helpers for isotope particle filters."""

from __future__ import annotations

from typing import Any

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
    target.last_n_after_adapt = None
    target.last_temper_steps = []
    target.last_temper_resample_count = 0
    target.last_source_event_diagnostics = []
    target.last_structural_timing_s = {}
    target._resample_count_in_observation = 0


def build_source_event_record(
    *,
    event: str,
    isotope: str,
    state: IsotopeState,
    source_idx: int,
    reason: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Return a source-slot diagnostic record or None for an invalid source."""
    idx = int(source_idx)
    if idx < 0 or idx >= int(state.num_sources):
        return None
    record: dict[str, object] = {
        "event": str(event),
        "isotope": str(isotope),
        "reason": str(reason),
        "source_index": idx,
        "position": [float(value) for value in state.positions[idx]],
        "strength": float(state.strengths[idx]),
    }
    if extra:
        record.update(extra)
    return record
