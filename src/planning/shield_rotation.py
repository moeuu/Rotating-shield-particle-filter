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
    allowed_indices=None,
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
        # Build full Cartesian product so Fe/Pb can point independently (8x8=64 combos by default).
        allowed = set(allowed_indices) if allowed_indices is not None else None
        for fe_idx, RFe in enumerate(RFe_candidates):
            for pb_idx, RPb in enumerate(RPb_candidates):
                oid = fe_idx * len(RPb_candidates) + pb_idx
                if allowed is not None and oid not in allowed:
                    continue
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


def select_top_k_orientations(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    k: int = 4,
    live_time_s: float = 1.0,
    fisher_weight: float = 0.0,
    criterion: str = "eig",
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    beta_by_isotope=None,
    allowed_indices=None,
) -> List[int]:
    """
    Return the top-k orientation ids (Fe/Pb pairs) sorted by score (no replacement).

    This is useful for running multiple short measurements at one pose without repeating
    the same Fe/Pb pair. Uses the same scoring as select_best_orientation.
    """
    scores: List[float] = []
    ids: List[int] = []
    # Reuse select_best_orientation machinery with allowed filtering
    if criterion == "variance":
        for orient_idx in range(estimator.num_orientations):
            if allowed_indices is not None and orient_idx not in set(allowed_indices):
                continue
            ig, fisher = estimator.orientation_information_metrics(
                pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
            )
            scores.append(ig + fisher_weight * fisher)
            ids.append(orient_idx)
    else:
        if RFe_candidates is None or RPb_candidates is None:
            from measurement.shielding import generate_octant_rotation_matrices
            RFe_candidates = generate_octant_rotation_matrices()
            RPb_candidates = generate_octant_rotation_matrices()
        allowed = set(allowed_indices) if allowed_indices is not None else None
        for fe_idx, RFe in enumerate(RFe_candidates):
            for pb_idx, RPb in enumerate(RPb_candidates):
                oid = fe_idx * len(RPb_candidates) + pb_idx
                if allowed is not None and oid not in allowed:
                    continue
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
    if not scores:
        return []
    order = np.argsort(scores)[::-1]
    top_ids = [ids[i] for i in order[:k]]
    return top_ids


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

    - Compute expected IG/Fisher for all orientations at the current pose using
      the discrete particle states when available.
    - If max metrics are below thresholds, stop rotating.
    - Otherwise select the best orientation (short acquisition suggested by live_time_s).

    Returns:
        (should_stop, orient_idx, score)
    """
    igs: List[float] = []
    fishers: List[float] = []
    scores: List[float] = []
    for oid in range(estimator.num_orientations):
        ig, fi = estimator.orientation_information_metrics(
            pose_idx=pose_idx,
            orient_idx=oid,
            live_time_s=live_time_s,
            prefer_continuous=False,
        )
        igs.append(ig)
        fishers.append(fi)
        scores.append(ig + fisher_weight * fi)
    max_ig = max(igs) if igs else 0.0
    max_fi = max(fishers) if fishers else 0.0
    if (max_ig < ig_threshold) and (max_fi < fisher_threshold):
        return True, -1, 0.0
    best_idx = int(np.argmax(scores))
    return False, best_idx, float(scores[best_idx])
