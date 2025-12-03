"""Choose the next robot pose while balancing uncertainty reduction and motion cost (Sec. 3.5.4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator


def select_next_pose(
    estimator: RotatingShieldPFEstimator,
    candidate_pose_indices: NDArray[np.int64],
    current_pose_idx: int,
    live_time_s: float = 1.0,
    lambda_cost: float = 1.0,
) -> int:
    """
    次姿勢を情報量と移動コストのトレードオフで選択する。
    """
    current_pos = estimator.poses[current_pose_idx]
    scores = []
    for idx in candidate_pose_indices:
        uncertainty = estimator.expected_uncertainty(pose_idx=int(idx), live_time_s=live_time_s)
        motion_cost = np.linalg.norm(estimator.poses[int(idx)] - current_pos)
        scores.append(uncertainty + lambda_cost * motion_cost)
    return int(candidate_pose_indices[int(np.argmin(scores))])
