"""RA-L-only control-policy adapter kept outside the generic PF package."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baselines.ral_ablation.path_policies import select_baseline_next_pose
from baselines.ral_ablation.shield_policies import select_baseline_shield_program
from pf.control_policy import PFExternalPathSelection
from planning.dss_pp import ShieldProgram


class RALControlPolicy:
    """Inject declared RA-L baseline choices through the generic policy protocol."""

    def __init__(
        self,
        *,
        path_policy: Mapping[str, Any] | None,
        shield_policy: Mapping[str, Any] | None,
    ) -> None:
        """Retain immutable copies of the two experiment-only policy objects."""
        self._path_policy = None if path_policy is None else dict(path_policy)
        self._shield_policy = None if shield_policy is None else dict(shield_policy)

    @property
    def has_fixed_path(self) -> bool:
        """Return whether this RA-L variant bypasses DSS-PP path selection."""
        return self._path_policy is not None

    def select_shield_program(
        self,
        *,
        total_pairs: int,
        program_length: int,
        pose_index: int,
        current_pair_id: int | None,
    ) -> ShieldProgram | None:
        """Return the declared RA-L shield baseline as a generic program."""
        selection = select_baseline_shield_program(
            self._shield_policy,
            total_pairs=total_pairs,
            program_length=program_length,
            pose_index=pose_index,
            current_pair_id=current_pair_id,
        )
        if selection is None:
            return None
        return ShieldProgram(
            name=selection.name,
            pair_ids=selection.pair_ids,
            kind="external_control",
        )

    def select_path(
        self,
        *,
        candidate_poses_xyz: NDArray[np.float64],
        current_pose_xyz: NDArray[np.float64],
        visited_poses_xyz: NDArray[np.float64],
        bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> PFExternalPathSelection | None:
        """Return the declared RA-L path baseline through the generic contract."""
        selection = select_baseline_next_pose(
            self._path_policy,
            candidate_poses_xyz=candidate_poses_xyz,
            current_pose_xyz=current_pose_xyz,
            visited_poses_xyz=visited_poses_xyz,
            bounds_xyz=bounds_xyz,
        )
        if selection is None:
            return None
        return PFExternalPathSelection(
            next_pose=selection.next_pose,
            candidate_index=selection.candidate_index,
            score=selection.score,
            policy_name=selection.name,
        )


def load_ral_control_policy(path: str | Path) -> RALControlPolicy:
    """Load one strict RA-L-only control policy from a separate artifact."""
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("RA-L control policy must be a JSON object.")
    expected = {"schema_version", "path_policy", "shield_policy"}
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("RA-L control policy must match schema version 1 exactly.")
    path_policy = payload["path_policy"]
    shield_policy = payload["shield_policy"]
    if path_policy is not None and not isinstance(path_policy, Mapping):
        raise TypeError("path_policy must be a JSON object or null.")
    if shield_policy is not None and not isinstance(shield_policy, Mapping):
        raise TypeError("shield_policy must be a JSON object or null.")
    if path_policy is not None and shield_policy is None:
        raise ValueError("A fixed RA-L path requires an explicit shield policy.")
    return RALControlPolicy(
        path_policy=path_policy,
        shield_policy=shield_policy,
    )


__all__ = ["RALControlPolicy", "load_ral_control_policy"]
