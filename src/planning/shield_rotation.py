"""Shield orientation selection based on information metrics (Sec. 3.5.2–3.5.3)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pf.estimator import RotatingShieldPFEstimator


def select_best_orientation(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float = 1.0,
) -> Tuple[int, float]:
    """
    候補方位の中から情報利得が最大となる方位を選択する。

    Returns:
        (best_orient_idx, best_score)
    """
    scores = []
    for orient_idx in range(estimator.num_orientations):
        score = estimator.orientation_information_gain(pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])
