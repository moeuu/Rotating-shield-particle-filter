"""Choose the next robot pose while balancing uncertainty reduction and motion cost (Sec. 3.5.4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator


def select_next_pose(
    estimator: RotatingShieldPFEstimator,
    candidate_pose_indices: NDArray[np.int64],
    current_pose_idx: int,
    *,
    criterion: str = "after_rotation",
    lambda_cost: float | None = None,
    tau_ig: float | None = None,
    t_max_s: float | None = None,
    t_short_s: float | None = None,
    num_rollouts: int = 0,
    use_mean_measurement: bool = True,
    rng_seed: int | None = 0,
) -> int:
    """
    Select the next pose using either uncertainty or after-rotation uncertainty.

    criterion:
        - "after_rotation": uses E[U_after-rotation | q] with rotating-shield policy
        - "uncertainty": uses single-measurement E[U | q] (legacy)

    Score_k = E[U | q_k] + lambda_cost * C_move
    """
    current_pos = estimator.poses[current_pose_idx]
    pf_config = getattr(estimator, "pf_config", None)
    lam_cost = (pf_config.lambda_cost if pf_config is not None else 1.0) if lambda_cost is None else lambda_cost
    tau_ig = (pf_config.ig_threshold if pf_config is not None else 1e-3) if tau_ig is None else tau_ig
    t_max_s = (pf_config.max_dwell_time_s if pf_config is not None else 1.0) if t_max_s is None else t_max_s
    t_short_s = (pf_config.short_time_s if pf_config is not None else 1.0) if t_short_s is None else t_short_s
    scores = []
    for idx in candidate_pose_indices:
        idx_int = int(idx)
        if criterion == "after_rotation" and hasattr(estimator, "expected_uncertainty_after_rotation"):
            uncertainty = estimator.expected_uncertainty_after_rotation(
                pose_idx=idx_int,
                tau_ig=tau_ig,
                t_max_s=t_max_s,
                t_short_s=t_short_s,
                num_rollouts=num_rollouts,
                use_mean_measurement=use_mean_measurement,
                rng_seed=rng_seed,
            )
        elif criterion == "uncertainty" and hasattr(estimator, "expected_uncertainty_after_pose"):
            uncertainty = estimator.expected_uncertainty_after_pose(
                pose_idx=idx_int, orient_idx=0, live_time_s=t_short_s
            )
        else:
            uncertainty = estimator.expected_uncertainty(pose_idx=idx_int, live_time_s=t_short_s)
        motion_cost = float(np.linalg.norm(estimator.poses[idx_int] - current_pos))
        scores.append(uncertainty + lam_cost * motion_cost)
    return int(candidate_pose_indices[int(np.argmin(scores))])
