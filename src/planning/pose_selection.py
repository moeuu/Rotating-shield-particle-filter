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
    lambda_cost: float | None = None,
) -> int:
    """
    次姿勢を不確実性と移動コストのトレードオフで選択する（Sec. 3.5.4, Eq. 3.51）。

    Score_k = E[U | q_k] + lambda_cost * C_move
    """
    current_pos = estimator.poses[current_pose_idx]
    lam_cost = estimator.pf_config.lambda_cost if lambda_cost is None else lambda_cost
    scores = []
    for idx in candidate_pose_indices:
        idx_int = int(idx)
        # Fallback to legacy expected_uncertainty if MC surrogate is unavailable (e.g., dummy estimator in tests)
        if hasattr(estimator, "expected_uncertainty_after_pose"):
            uncertainty = estimator.expected_uncertainty_after_pose(pose_idx=idx_int, orient_idx=0, live_time_s=live_time_s)
        else:
            uncertainty = estimator.expected_uncertainty(pose_idx=idx_int, live_time_s=live_time_s)
        motion_cost = float(np.linalg.norm(estimator.poses[idx_int] - current_pos))
        scores.append(uncertainty + lam_cost * motion_cost)
    return int(candidate_pose_indices[int(np.argmin(scores))])
