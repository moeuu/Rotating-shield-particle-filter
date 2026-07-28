"""Shield-program ablation policies for RA-L comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineShieldProgram:
    """Represent a baseline shield program selected outside DSS-PP."""

    name: str
    pair_ids: tuple[int, ...]


def _read_policy_name(policy_config: Mapping[str, Any] | str | None) -> str:
    """Return the exact canonical baseline shield-policy name."""
    if policy_config is None:
        return ""
    if not isinstance(policy_config, Mapping):
        raise TypeError("baseline_shield_policy must be a JSON object or null.")
    name = policy_config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("baseline_shield_policy.name must be a nonempty string.")
    return name


def _read_int(
    policy_config: Mapping[str, Any] | str | None,
    key: str,
    default: int,
) -> int:
    """Read a strict JSON integer setting from a shield-policy payload."""
    if not isinstance(policy_config, Mapping):
        return default
    value = policy_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"baseline_shield_policy.{key} must be a JSON integer.")
    return value


def _positive_json_integer(value: object, *, field_name: str) -> int:
    """Return a strict positive JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return value


def _nonnegative_json_integer(value: object, *, field_name: str) -> int:
    """Return a strict nonnegative JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative JSON integer.")
    return value


def select_baseline_shield_program(
    policy_config: Mapping[str, Any] | str | None,
    *,
    total_pairs: int,
    program_length: int,
    pose_index: int,
    current_pair_id: int | None = None,
) -> BaselineShieldProgram | None:
    """Return a baseline shield program, or None when no baseline policy is active."""
    policy = _read_policy_name(policy_config)
    if policy == "":
        return None
    if not isinstance(policy_config, Mapping):
        raise AssertionError("Validated shield policy must be a mapping.")
    total = _positive_json_integer(total_pairs, field_name="total_pairs")
    length = _positive_json_integer(program_length, field_name="program_length")
    pose = _nonnegative_json_integer(pose_index, field_name="pose_index")
    if current_pair_id is not None:
        current = _nonnegative_json_integer(
            current_pair_id,
            field_name="current_pair_id",
        )
        if current >= total:
            raise ValueError("current_pair_id must be smaller than total_pairs.")
    else:
        current = None
    if policy == "fixed":
        unknown = sorted(set(policy_config) - {"name", "fixed_pair_id"})
        if unknown:
            raise ValueError(
                "Unsupported fixed shield settings: "
                + ", ".join(str(key) for key in unknown)
            )
        fixed_pair = _read_int(policy_config, "fixed_pair_id", 0)
        if not 0 <= fixed_pair < total:
            raise ValueError(
                "baseline_shield_policy.fixed_pair_id must be in "
                "[0, total_pairs)."
            )
        return BaselineShieldProgram(
            name=f"fixed_shield_{fixed_pair}",
            pair_ids=tuple(fixed_pair for _ in range(length)),
        )
    if policy == "round_robin":
        unknown = sorted(
            set(policy_config)
            - {"name", "start_pair_id", "advance_by_pose"}
        )
        if unknown:
            raise ValueError(
                "Unsupported round_robin shield settings: "
                + ", ".join(str(key) for key in unknown)
            )
        start = _read_int(
            policy_config,
            "start_pair_id",
            0 if current is None else (current + 1) % total,
        )
        if not 0 <= start < total:
            raise ValueError(
                "baseline_shield_policy.start_pair_id must be in "
                "[0, total_pairs)."
            )
        advance_by_pose = policy_config.get("advance_by_pose", True)
        if not isinstance(advance_by_pose, bool):
            raise ValueError(
                "baseline_shield_policy.advance_by_pose must be a JSON boolean."
            )
        if advance_by_pose:
            start += pose * length
        return BaselineShieldProgram(
            name="round_robin_shield",
            pair_ids=tuple((start + idx) % total for idx in range(length)),
        )
    raise ValueError(f"Unknown baseline_shield_policy: {policy}")
