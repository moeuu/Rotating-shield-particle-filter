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
    validated = validate_baseline_shield_policy(policy_config)
    if validated is None:
        return ""
    return str(validated["name"])


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


def validate_baseline_shield_policy(
    policy_config: Mapping[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Validate one exact discriminated RA-L shield-policy object."""
    if policy_config is None:
        return None
    if not isinstance(policy_config, Mapping):
        raise TypeError("baseline_shield_policy must be a JSON object or null.")
    if any(not isinstance(key, str) for key in policy_config):
        raise TypeError("baseline_shield_policy keys must be JSON strings.")
    name = policy_config.get("name")
    if name == "fixed":
        expected = {"name", "fixed_pair_id"}
        actual = set(policy_config)
        if actual != expected:
            raise ValueError(
                "fixed shield policy must contain exactly name and fixed_pair_id; "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}."
            )
        fixed_pair_id = _nonnegative_json_integer(
            policy_config["fixed_pair_id"],
            field_name="baseline_shield_policy.fixed_pair_id",
        )
        return {"name": "fixed", "fixed_pair_id": fixed_pair_id}
    if name == "round_robin":
        expected = {"name", "start_pair_id", "advance_by_pose"}
        actual = set(policy_config)
        if actual != expected:
            raise ValueError(
                "round_robin shield policy must contain exactly name, "
                "start_pair_id, and advance_by_pose; "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}."
            )
        start_pair_id = _nonnegative_json_integer(
            policy_config["start_pair_id"],
            field_name="baseline_shield_policy.start_pair_id",
        )
        advance_by_pose = policy_config["advance_by_pose"]
        if not isinstance(advance_by_pose, bool):
            raise ValueError(
                "baseline_shield_policy.advance_by_pose must be a JSON boolean."
            )
        return {
            "name": "round_robin",
            "start_pair_id": start_pair_id,
            "advance_by_pose": advance_by_pose,
        }
    raise ValueError(
        "baseline_shield_policy.name must be exactly 'fixed' or 'round_robin'."
    )


def select_baseline_shield_program(
    policy_config: Mapping[str, Any] | str | None,
    *,
    total_pairs: int,
    program_length: int,
    pose_index: int,
    current_pair_id: int | None = None,
) -> BaselineShieldProgram | None:
    """Return a baseline shield program, or None when no baseline policy is active."""
    validated_policy = validate_baseline_shield_policy(policy_config)
    policy = _read_policy_name(validated_policy)
    if policy == "":
        return None
    if not isinstance(validated_policy, Mapping):
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
    if policy == "fixed":
        fixed_pair = _nonnegative_json_integer(
            validated_policy["fixed_pair_id"],
            field_name="baseline_shield_policy.fixed_pair_id",
        )
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
        start = _nonnegative_json_integer(
            validated_policy["start_pair_id"],
            field_name="baseline_shield_policy.start_pair_id",
        )
        if not 0 <= start < total:
            raise ValueError(
                "baseline_shield_policy.start_pair_id must be in "
                "[0, total_pairs)."
            )
        advance_by_pose = validated_policy["advance_by_pose"]
        assert isinstance(advance_by_pose, bool)
        if advance_by_pose:
            start += pose * length
        return BaselineShieldProgram(
            name="round_robin_shield",
            pair_ids=tuple((start + idx) % total for idx in range(length)),
        )
    raise ValueError(f"Unknown baseline_shield_policy: {policy}")


__all__ = [
    "BaselineShieldProgram",
    "select_baseline_shield_program",
    "validate_baseline_shield_policy",
]
