"""Choose the next robot pose while balancing uncertainty reduction and motion cost (Sec. 3.5.4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from planning.candidate_generation import generate_candidate_poses


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
    if rng_seed is not None:
        np.random.seed(rng_seed)
    rollouts = int(num_rollouts)
    if rollouts <= 0 and not use_mean_measurement:
        rollouts = 1
    scores = []
    for idx in candidate_pose_indices:
        idx_int = int(idx)
        if criterion == "after_rotation" and hasattr(estimator, "expected_uncertainty_after_rotation"):
            uncertainty = estimator.expected_uncertainty_after_rotation(
                pose_xyz=estimator.poses[idx_int],
                live_time_per_rot_s=t_short_s,
                tau_ig=tau_ig,
                tmax_s=t_max_s,
                n_rollouts=rollouts,
                orient_selection="IG",
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


def select_next_pose_from_candidates(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    lambda_cost: float | None = None,
    tau_ig: float | None = None,
    t_max_s: float | None = None,
    t_short_s: float | None = None,
    num_rollouts: int = 0,
    use_mean_measurement: bool = True,
    rng_seed: int | None = 0,
) -> int:
    """
    Select the next pose from explicit candidate coordinates (after-rotation criterion).

    Score_k = E[U_after-rotation | q_k] + lambda_cost * C_move
    """
    candidate_poses_xyz = np.asarray(candidate_poses_xyz, dtype=float)
    if candidate_poses_xyz.ndim != 2 or candidate_poses_xyz.shape[1] != 3:
        raise ValueError("candidate_poses_xyz must be shape (N, 3).")
    if candidate_poses_xyz.shape[0] == 0:
        raise ValueError("candidate_poses_xyz must contain at least one pose.")
    current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
    pf_config = getattr(estimator, "pf_config", None)
    lam_cost = (pf_config.lambda_cost if pf_config is not None else 1.0) if lambda_cost is None else lambda_cost
    tau_ig = (pf_config.ig_threshold if pf_config is not None else 1e-3) if tau_ig is None else tau_ig
    t_max_s = (pf_config.max_dwell_time_s if pf_config is not None else 1.0) if t_max_s is None else t_max_s
    t_short_s = (pf_config.short_time_s if pf_config is not None else 1.0) if t_short_s is None else t_short_s
    if rng_seed is not None:
        np.random.seed(rng_seed)
    rollouts = int(num_rollouts)
    if rollouts <= 0 and not use_mean_measurement:
        rollouts = 1
    scores = []
    for pose in candidate_poses_xyz:
        uncertainty = estimator.expected_uncertainty_after_rotation(
            pose_xyz=pose,
            live_time_per_rot_s=t_short_s,
            tau_ig=tau_ig,
            tmax_s=t_max_s,
            n_rollouts=rollouts,
            orient_selection="IG",
        )
        motion_cost = float(np.linalg.norm(pose - current_pose_xyz))
        scores.append(uncertainty + lam_cost * motion_cost)
    return int(np.argmin(scores))


def select_next_pose_after_rotation(
    estimator: RotatingShieldPFEstimator,
    current_pose_xyz: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64],
    n_candidates: int = 1024,
    n_rollouts: int = 64,
    live_time_per_rot_s: float = 1.0,
    tau_ig: float = 0.01,
    tmax_s: float = 10.0,
    lambda_cost: float = 0.0,
    candidate_strategy: str = "free_space_sobol",
) -> NDArray[np.float64]:
    """
    Choose q_{k+1} by minimizing after-rotation uncertainty plus motion cost.

    The score is:
        E[U_after-rotation | q] + lambda_cost * ||q - q_current||_2

    Candidate poses are generated on-demand using the requested strategy.
    """
    current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
    if current_pose_xyz.shape != (3,):
        raise ValueError("current_pose_xyz must be shape (3,).")
    visited_poses_xyz = np.asarray(visited_poses_xyz, dtype=float)
    if visited_poses_xyz.ndim != 2 or visited_poses_xyz.shape[1] != 3:
        raise ValueError("visited_poses_xyz must be shape (N, 3).")
    pf_config = getattr(estimator, "pf_config", None)
    bounds_xyz = None
    if pf_config is not None and hasattr(pf_config, "position_min") and hasattr(pf_config, "position_max"):
        bounds_xyz = (np.asarray(pf_config.position_min, dtype=float), np.asarray(pf_config.position_max, dtype=float))

    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        n_candidates=n_candidates,
        strategy=candidate_strategy,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=bounds_xyz,
    )
    if candidates.size == 0:
        raise ValueError("No candidate poses generated.")

    scores = []
    for pose in candidates:
        uncertainty = estimator.expected_uncertainty_after_rotation(
            pose_xyz=pose,
            live_time_per_rot_s=live_time_per_rot_s,
            tau_ig=tau_ig,
            tmax_s=tmax_s,
            n_rollouts=n_rollouts,
            orient_selection="IG",
        )
        motion_cost = float(np.linalg.norm(pose - current_pose_xyz))
        scores.append(float(uncertainty) + lambda_cost * motion_cost)
    return candidates[int(np.argmin(scores))]
