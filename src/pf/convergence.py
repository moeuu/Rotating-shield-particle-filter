"""Compute convergence metrics, information measures, and stopping criteria (Chapter 3.5)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from pf.estimator import RotatingShieldPFEstimator


def entropy(weights: NDArray[np.float64]) -> float:
    """Shannon entropy H(w) = -Σ w log w (Eq. 3.42)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    w = w / max(np.sum(w), 1e-12)
    return float(-np.sum(w * np.log(w + 1e-12)))


def credible_volume(positions: NDArray[np.float64], weights: NDArray[np.float64], confidence: float = 0.95) -> float:
    """
    Compute 3D positional credible ellipsoid volume (4/3 π sqrt(det(cov * χ2))) for one source.
    """
    if positions.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    w = w / max(np.sum(w), 1e-12)
    mean = np.sum(w[:, None] * positions, axis=0)
    centered = positions - mean
    cov = centered.T @ (centered * w[:, None])
    det_val = np.linalg.det(cov * chi2.ppf(confidence, df=3))
    if det_val < 0:
        return 0.0
    return float((4.0 / 3.0) * np.pi * np.sqrt(det_val + 1e-12))


def global_uncertainty_strength(
    strengths: NDArray[np.float64], weights: NDArray[np.float64], max_sources: int
) -> float:
    """
    U = Σ_m Var(q_m) over strengths matrix shaped (N_particles, max_sources) (Eq. 3.38 surrogate).
    """
    if strengths.size == 0:
        return 0.0
    w = weights / max(np.sum(weights), 1e-12)
    mean = np.sum(w[:, None] * strengths, axis=0)
    var = np.sum(w[:, None] * (strengths - mean) ** 2, axis=0)
    return float(np.sum(var))


def summarize_estimates(estimator: RotatingShieldPFEstimator) -> Dict[str, List[Dict[str, NDArray[np.float64]]]]:
    """
    Export final estimates with covariances for downstream use.

    Returns:
        {iso: [{"position": (3,), "strength": float, "cov": (4,4)}], ...}
    """
    summary: Dict[str, List[Dict[str, NDArray[np.float64]]]] = {}
    estimates = estimator.estimates()
    for iso, (pos, strengths) in estimates.items():
        items: List[Dict[str, NDArray[np.float64]]] = []
        covs = None
        # try to fetch covariances from ParallelIsotopePF style history if available
        if hasattr(estimator, "history_estimates") and estimator.history_estimates:
            est_last = estimator.history_estimates[-1].get(iso)
            if est_last and hasattr(est_last, "covariances"):
                covs = est_last.covariances
        for i in range(pos.shape[0]):
            cov = covs[i] if covs is not None and covs.shape[0] > i else np.zeros((4, 4))
            items.append({"position": pos[i], "strength": strengths[i], "cov": cov})
        summary[iso] = items
    return summary


def has_converged(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int | None = None,
    ig_threshold: float | None = None,
    fisher_threshold: float | None = None,
    change_tol: float | None = None,
    uncertainty_tol: float | None = None,
    credible_volume_threshold: float | None = None,
    live_time_s: float = 1.0,
) -> bool:
    """
    Wrapper around should_stop_shield_rotation with explicit thresholds for clarity.
    """
    return estimator.should_stop_shield_rotation(
        pose_idx=pose_idx if pose_idx is not None else (len(estimator.poses) - 1),
        ig_threshold=ig_threshold if ig_threshold is not None else estimator.pf_config.ig_threshold,
        fisher_threshold=fisher_threshold if fisher_threshold is not None else 1e-3,
        change_tol=change_tol if change_tol is not None else 1e-2,
        uncertainty_tol=uncertainty_tol if uncertainty_tol is not None else 1e-3,
        live_time_s=live_time_s,
    )
