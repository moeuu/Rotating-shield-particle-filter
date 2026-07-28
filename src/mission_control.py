"""Mission-level stopping and adaptive shield-program helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _positive_optional_integer(value: object, *, name: str) -> int | None:
    """Return None or one exact positive mission-budget integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive JSON integer or null.")
    if value < 1:
        raise ValueError(f"{name} must be positive when provided.")
    return int(value)


def resolve_mission_max_steps(
    cli_max_steps: int | None,
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Resolve the fixed measurement budget, preserving explicit CLI input."""
    if cli_max_steps is not None:
        return _positive_optional_integer(
            cli_max_steps,
            name="max_steps",
        )
    measurement_budget_value = runtime_config.get(
        "measurement_budget_max_steps",
        None,
    )
    if measurement_budget_value is None:
        return None
    return _positive_optional_integer(
        measurement_budget_value,
        name="measurement_budget_max_steps",
    )


def resolve_mission_max_poses(
    cli_max_poses: int | None,
    runtime_config: Mapping[str, Any],
) -> int | None:
    """Resolve the mission pose cap while preserving explicit CLI overrides."""
    if cli_max_poses is not None:
        return _positive_optional_integer(
            cli_max_poses,
            name="max_poses",
        )
    mission_stop_max_poses_value = runtime_config.get(
        "mission_stop_max_poses",
        runtime_config.get("mission_stop_min_poses", None),
    )
    if mission_stop_max_poses_value is None:
        return None
    return _positive_optional_integer(
        mission_stop_max_poses_value,
        name="mission_stop_max_poses",
    )
