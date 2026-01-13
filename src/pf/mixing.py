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
    min_support: int = 1,
    min_obs_count: float = 0.0,
    min_strength_abs: float | None = None,
    min_strength_ratio: float | None = None,
) -> Dict[str, NDArray[np.bool_]]:
    """
    Apply the best-case measurement test to continuous PF estimates (Sec. 3.4.5).

    Uses MMSE estimates as candidate sources. Returns a keep mask per isotope that
    can be applied to continuous particle source indices by order. Optionally drop
    sources with strengths below max(min_strength_abs, min_strength_ratio * max_strength).
    Measurements with z_obs <= min_obs_count are ignored for the best-case test.
    If all sources would be removed, keep the strongest one to avoid empty estimates.
    """
    if not estimator.measurements:
        return {iso: np.ones(0, dtype=bool) for iso in estimator.filters}

    kernel = ContinuousKernel(
        mu_by_isotope=estimator.mu_by_isotope,
        shield_params=estimator.shield_params,
        use_gpu=estimator._gpu_enabled(),
        gpu_device=estimator.pf_config.gpu_device,
        gpu_dtype=estimator.pf_config.gpu_dtype,
    )
    keep_masks: Dict[str, NDArray[np.bool_]] = {}
    estimates = estimator.estimates()

    for iso, (positions, strengths) in estimates.items():
        if positions.size == 0:
            keep_masks[iso] = np.ones(0, dtype=bool)
            continue
        keep_mask = np.ones(positions.shape[0], dtype=bool)
        max_strength = float(np.max(strengths)) if strengths.size else 0.0
        min_strength = 0.0
        if min_strength_abs is not None:
            min_strength = max(min_strength, float(min_strength_abs))
        if min_strength_ratio is not None:
            min_strength = max(min_strength, float(min_strength_ratio) * max_strength)
        for m, (pos, strength) in enumerate(zip(positions, strengths)):
            if min_strength > 0.0 and strength < min_strength:
                keep_mask[m] = False
                continue
            best_ratio = None
            support = 0
            for rec in estimator.measurements:
                z_obs = float(rec.z_k.get(iso, 0.0))
                if z_obs <= min_obs_count:
                    continue
                support += 1
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
            if support < max(1, int(min_support)):
                keep_mask[m] = False
                continue
            if best_ratio is not None and best_ratio < tau_mix:
                keep_mask[m] = False
        if not np.any(keep_mask) and strengths.size:
            keep_mask[int(np.argmax(strengths))] = True
        keep_masks[iso] = keep_mask
    return keep_masks
