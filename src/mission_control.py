"""Mission-level stopping and adaptive shield-program helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def remaining_measurement_ready_for_stop(
    estimate: Mapping[str, Any] | None,
) -> bool:
    """Return True when the remaining-measurement estimate has no unresolved budget."""
    if not estimate:
        return True
    unresolved_raw = estimate.get("unresolved_factors", [])
    if isinstance(unresolved_raw, str):
        unresolved_factors = [unresolved_raw] if unresolved_raw else []
    else:
        unresolved_factors = [str(value) for value in unresolved_raw or []]
    try:
        remaining = int(estimate.get("estimated_remaining_stations", 0))
    except (TypeError, ValueError):
        remaining = 0
    try:
        budget = float(estimate.get("current_budget", 0.0))
    except (TypeError, ValueError):
        budget = 0.0
    return not (unresolved_factors and remaining > 0 and budget > 0.0)


def remaining_measurement_payload(
    estimate: Mapping[str, Any] | object | None,
) -> Mapping[str, Any]:
    """Return a mapping view of a remaining-measurement estimate."""
    if estimate is None:
        return {}
    if isinstance(estimate, Mapping):
        return estimate
    if hasattr(estimate, "to_dict"):
        try:
            payload = estimate.to_dict()
        except (TypeError, ValueError):
            return {}
        if isinstance(payload, Mapping):
            return payload
    return {}


def resolve_mission_max_steps(
    cli_max_steps: int | None,
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Resolve the fixed measurement budget, preserving explicit CLI input."""
    if cli_max_steps is not None:
        return max(1, int(cli_max_steps)) if int(cli_max_steps) > 0 else None
    measurement_budget_value = runtime_config.get(
        "measurement_budget_max_steps",
        None,
    )
    if measurement_budget_value is None:
        return None
    return max(1, int(measurement_budget_value))


def resolve_mission_max_poses(
    cli_max_poses: int | None,
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Resolve the mission pose cap while preserving explicit CLI overrides."""
    if cli_max_poses is not None:
        return max(1, int(cli_max_poses)) if int(cli_max_poses) > 0 else None
    mission_stop_max_poses_value = runtime_config.get(
        "mission_stop_max_poses",
        runtime_config.get("mission_stop_min_poses", None),
    )
    if mission_stop_max_poses_value is None:
        return None
    return max(1, int(mission_stop_max_poses_value))
