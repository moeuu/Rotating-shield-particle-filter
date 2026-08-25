"""Path-planning ablation policies for RA-L comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BaselinePathSelection:
    """Represent a baseline path-policy selection from candidate poses."""

    name: str
    next_pose: NDArray[np.float64]
    candidate_index: int
    score: float


def resolve_rotation_limit_for_active_program(
    *,
    base_rotation_limit: int,
    active_shield_program: Sequence[int] | None,
    strict_planned_shield_program: bool,
    baseline_shield_policy: Mapping[str, Any] | str | None,
) -> int:
    """Return the measurement count for one explicit shield program."""
    if baseline_shield_policy is not None:
        from baselines.ral_ablation.shield_policies import (
            validate_baseline_shield_policy,
        )

        validate_baseline_shield_policy(baseline_shield_policy)
    base_limit = max(1, int(base_rotation_limit))
    if not active_shield_program:
        return base_limit
    program_limit = max(1, len(active_shield_program))
    if strict_planned_shield_program or baseline_shield_policy is not None:
        return program_limit
    return max(base_limit, program_limit)


def _policy_name(policy_config: Mapping[str, Any] | str | None) -> str:
    """Return the exact canonical baseline path-policy name."""
    validated = validate_baseline_path_policy(policy_config)
    if validated is None:
        return ""
    return str(validated["name"])


def _positive_json_integer(value: object, *, field_name: str) -> int:
    """Return a strict positive JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return value


def validate_baseline_path_policy(
    policy_config: Mapping[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Validate one exact discriminated RA-L path-policy object."""
    if policy_config is None:
        return None
    if not isinstance(policy_config, Mapping):
        raise TypeError("baseline_path_policy must be a JSON object or null.")
    if any(not isinstance(key, str) for key in policy_config):
        raise TypeError("baseline_path_policy keys must be JSON strings.")
    name = policy_config.get("name")
    if name != "passive_serpentine":
        raise ValueError(
            "baseline_path_policy.name must be exactly 'passive_serpentine'."
        )
    expected = {"name", "row_count"}
    actual = set(policy_config)
    if actual != expected:
        raise ValueError(
            "passive_serpentine policy must contain exactly name and row_count; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )
    row_count = _positive_json_integer(
        policy_config["row_count"],
        field_name="baseline_path_policy.row_count",
    )
    return {"name": "passive_serpentine", "row_count": row_count}


def _serpentine_target(
    *,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    visited_count: int,
    row_count: int,
) -> NDArray[np.float64]:
    """Return the next nominal waypoint of a floor-plane serpentine path."""
    lo, hi = bounds_xyz
    rows = _positive_json_integer(row_count, field_name="row_count")
    if (
        isinstance(visited_count, bool)
        or not isinstance(visited_count, int)
        or visited_count < 0
    ):
        raise ValueError("visited_count must be a nonnegative integer.")
    row = visited_count % rows
    y_values = np.linspace(float(lo[1]), float(hi[1]), rows)
    x = float(hi[0] if row % 2 else lo[0])
    return np.asarray([x, float(y_values[row]), float(lo[2])], dtype=float)


def select_baseline_next_pose(
    policy_config: Mapping[str, Any] | str | None,
    *,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> BaselinePathSelection | None:
    """Select the next pose with a baseline path policy."""
    validated_policy = validate_baseline_path_policy(policy_config)
    policy = _policy_name(validated_policy)
    if policy == "":
        return None
    if not isinstance(validated_policy, Mapping):
        raise AssertionError("Validated path policy must be a mapping.")
    candidates = np.asarray(candidate_poses_xyz, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError("candidate_poses_xyz must be a non-empty (N, 3) array.")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_poses_xyz must contain only finite values.")
    current = np.asarray(current_pose_xyz, dtype=float)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise ValueError("current_pose_xyz must be one finite 3-D point.")
    if visited_poses_xyz is None:
        visited_count = 0
    else:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if (
            visited.ndim != 2
            or visited.shape[1] != 3
            or not np.all(np.isfinite(visited))
        ):
            raise ValueError("visited_poses_xyz must be a finite (N, 3) array.")
        visited_count = len(visited)
    if len(bounds_xyz) != 2:
        raise ValueError("bounds_xyz must contain lower and upper 3-D bounds.")
    lo = np.asarray(bounds_xyz[0], dtype=float)
    hi = np.asarray(bounds_xyz[1], dtype=float)
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.all(np.isfinite(lo))
        or not np.all(np.isfinite(hi))
        or np.any(hi < lo)
    ):
        raise ValueError("bounds_xyz must contain finite ordered 3-D bounds.")
    if policy == "passive_serpentine":
        row_count = _positive_json_integer(
            validated_policy["row_count"],
            field_name="baseline_path_policy.row_count",
        )
        target = _serpentine_target(
            bounds_xyz=(lo, hi),
            visited_count=visited_count,
            row_count=row_count,
        )
        distances = np.linalg.norm(candidates - target[None, :], axis=1)
        idx = int(np.argmin(distances))
        return BaselinePathSelection(
            name="passive_serpentine",
            next_pose=candidates[idx].astype(float, copy=True),
            candidate_index=idx,
            score=-float(distances[idx]),
        )
    raise ValueError(f"Unknown baseline_path_policy: {policy}")


__all__ = [
    "BaselinePathSelection",
    "resolve_rotation_limit_for_active_program",
    "select_baseline_next_pose",
    "validate_baseline_path_policy",
]
