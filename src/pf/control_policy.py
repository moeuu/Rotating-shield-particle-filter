"""Generic injection boundary for experiment-specific live control policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from planning.dss_pp import ShieldProgram


@dataclass(frozen=True, slots=True)
class PFExternalPathSelection:
    """Describe one externally selected candidate pose without experiment labels."""

    next_pose: NDArray[np.float64]
    candidate_index: int
    score: float
    policy_name: str


class PFControlPolicy(Protocol):
    """Define optional shield and path overrides for a live PF controller."""

    @property
    def has_fixed_path(self) -> bool:
        """Return whether this policy bypasses posterior-dependent path planning."""
        ...

    def select_shield_program(
        self,
        *,
        total_pairs: int,
        program_length: int,
        pose_index: int,
        current_pair_id: int | None,
    ) -> ShieldProgram | None:
        """Return an external shield program or defer to the standard planner."""
        ...

    def select_path(
        self,
        *,
        candidate_poses_xyz: NDArray[np.float64],
        current_pose_xyz: NDArray[np.float64],
        visited_poses_xyz: NDArray[np.float64],
        bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> PFExternalPathSelection | None:
        """Return an external path selection or defer to the standard planner."""
        ...


def validate_control_policy(policy: PFControlPolicy | None) -> None:
    """Fail early when an injected control policy lacks the required surface."""
    if policy is None:
        return
    for name in ("select_shield_program", "select_path"):
        if not callable(getattr(policy, name, None)):
            raise TypeError(f"control_policy.{name} must be callable.")
    if not isinstance(policy.has_fixed_path, bool):
        raise TypeError("control_policy.has_fixed_path must be a boolean.")


__all__ = [
    "PFControlPolicy",
    "PFExternalPathSelection",
    "validate_control_policy",
]
