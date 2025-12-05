"""Shield orientation selection based on information metrics (Sec. 3.4)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pf.estimator import RotatingShieldPFEstimator


def select_best_orientation(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float = 1.0,
    fisher_weight: float = 0.0,
    criterion: str = "eig",
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    beta_by_isotope=None,
) -> Tuple[int, float]:
    """
    Choose the shield orientation that maximises EIG/JA/JD or variance-based score (Eqs. 3.40–3.48).

    Returns:
        (best_orient_idx, best_score) where score = IG + fisher_weight * Fisher
    """
    scores: List[float] = []
    ids: List[int] = []
    if criterion == "variance":
        for orient_idx in range(estimator.num_orientations):
            ig, fisher = estimator.orientation_information_metrics(
                pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
            )
            scores.append(ig + fisher_weight * fisher)
            ids.append(orient_idx)
    elif criterion in {"eig", "ja", "jd"}:
        if RFe_candidates is None or RPb_candidates is None:
            from measurement.shielding import generate_octant_rotation_matrices

            RFe_candidates = generate_octant_rotation_matrices()
            RPb_candidates = generate_octant_rotation_matrices()
        for oid, (RFe, RPb) in enumerate(zip(RFe_candidates, RPb_candidates)):
            if criterion == "eig":
                score = estimator.orientation_expected_information_gain(
                    pose_idx=pose_idx,
                    RFe=RFe,
                    RPb=RPb,
                    live_time_s=live_time_s,
                    alpha_by_isotope=alpha_by_isotope,
                )
            else:
                JA, JD = estimator.orientation_fisher_criteria(
                    pose_idx=pose_idx,
                    RFe=RFe,
                    RPb=RPb,
                    live_time_s=live_time_s,
                    beta_by_isotope=beta_by_isotope,
                )
                score = JA if criterion == "ja" else JD
            scores.append(score)
            ids.append(oid)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    best_idx = int(np.argmax(scores)) if scores else -1
    return ids[best_idx], float(scores[best_idx] if scores else 0.0)


def rotation_policy_step(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    ig_threshold: float = 1e-3,
    fisher_threshold: float = 1e-3,
    live_time_s: float = 0.5,
    fisher_weight: float = 0.0,
) -> Tuple[bool, int, float]:
    """
    One step of the shield-rotation policy (Sec. 3.4.3, Eqs. 3.47–3.48).

    - Compute expected IG/Fisher for all orientations at the current pose.
    - If max metrics are below thresholds, stop rotating.
    - Otherwise select the best orientation (short acquisition suggested by live_time_s).

    Returns:
        (should_stop, orient_idx, score)
    """
    igs: List[float] = []
    fishers: List[float] = []
    scores: List[float] = []
    for oid in range(estimator.num_orientations):
        ig, fi = estimator.orientation_information_metrics(pose_idx=pose_idx, orient_idx=oid, live_time_s=live_time_s)
        igs.append(ig)
        fishers.append(fi)
        scores.append(ig + fisher_weight * fi)
    max_ig = max(igs) if igs else 0.0
    max_fi = max(fishers) if fishers else 0.0
    if (max_ig < ig_threshold) and (max_fi < fisher_threshold):
        return True, -1, 0.0
    best_idx = int(np.argmax(scores))
    return False, best_idx, float(scores[best_idx])
