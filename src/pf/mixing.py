"""Merge PF outputs, reject spurious sources, and evaluate best-case measurement checks."""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from pf.estimator import RotatingShieldPFEstimator


def prune_spurious_sources_continuous(
    estimator: RotatingShieldPFEstimator,
    tau_mix: float = 0.9,
    epsilon: float = 1e-6,
) -> Dict[str, NDArray[np.bool_]]:
    """
    Apply the best-case measurement test to continuous PF estimates (Sec. 3.4.5).

    Uses MMSE estimates as candidate sources. Returns a keep mask per isotope that
    can be applied to continuous particle source indices by order.
    """
    if not estimator.measurements:
        return {iso: np.ones(0, dtype=bool) for iso in estimator.filters}

    kernel = ContinuousKernel(mu_by_isotope=estimator.mu_by_isotope, shield_params=estimator.shield_params)
    keep_masks: Dict[str, NDArray[np.bool_]] = {}
    estimates = estimator.estimates()

    for iso, (positions, strengths) in estimates.items():
        if positions.size == 0:
            keep_masks[iso] = np.ones(0, dtype=bool)
            continue
        keep_mask = np.ones(positions.shape[0], dtype=bool)
        for m, (pos, strength) in enumerate(zip(positions, strengths)):
            best_ratio = None
            for rec in estimator.measurements:
                z_obs = float(rec.z_k.get(iso, 0.0))
                det_pos = estimator.poses[rec.pose_idx]
                if rec.fe_index is not None and rec.pb_index is not None:
                    pred = kernel.expected_counts_pair(
                        isotope=iso,
                        detector_pos=det_pos,
                        sources=np.array([pos]),
                        strengths=np.array([strength]),
                        fe_index=rec.fe_index,
                        pb_index=rec.pb_index,
                        live_time_s=rec.live_time_s,
                        background=0.0,
                    )
                else:
                    pred = kernel.expected_counts(
                        isotope=iso,
                        detector_pos=det_pos,
                        sources=np.array([pos]),
                        strengths=np.array([strength]),
                        orient_idx=rec.orient_idx,
                        live_time_s=rec.live_time_s,
                        background=0.0,
                    )
                ratio = float(pred) / (z_obs + epsilon)
                if best_ratio is None or ratio > best_ratio:
                    best_ratio = ratio
            if best_ratio is not None and best_ratio < tau_mix:
                keep_mask[m] = False
        keep_masks[iso] = keep_mask
    return keep_masks
