"""Best-case measurement test for spurious source removal (Sec. 3.3.5)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from pf.parallel import Measurement, ParallelIsotopePF
from pf.state import IsotopeState


def prune_spurious_sources(
    pf: ParallelIsotopePF,
    measurements: List[Measurement],
    tau_mix: float = 0.8,
    epsilon: float = 1e-6,
) -> Dict[str, IsotopeState]:
    """
    Apply the best-case measurement test to each isotope estimate (Eqs. 3.33–3.36).

    For each estimated source m, find k* maximizing Λ_hat_k/(z_k+ε). If Λ_hat_{k*}/z_{k*} < τ_mix,
    mark spurious and remove.
    """
    pruned: Dict[str, IsotopeState] = {}
    kernel_helper = ContinuousKernel()

    for iso, filt in pf.filters.items():
        if not filt.continuous_particles:
            continue
        best = filt.best_particle().state
        if best.num_sources == 0:
            pruned[iso] = best
            continue
        keep_mask = np.ones(best.num_sources, dtype=bool)
        for m in range(best.num_sources):
            best_ratio = -np.inf
            for meas in measurements:
                z_obs = meas.counts_by_isotope.get(iso, 0.0)
                if filt.kernel is None:
                    continue
                det_pos = filt.kernel.poses[meas.pose_idx]
                orient_vec = filt.kernel.orientations[meas.orient_idx]
                k_val = kernel_helper.expected_counts(
                    isotope=iso,
                    detector_pos=det_pos,
                    sources=np.array([best.positions[m]]),
                    strengths=np.array([best.strengths[m]]),
                    orient_idx=kernel_helper.orient_index_from_vector(orient_vec),
                    live_time_s=meas.live_time_s,
                    background=0.0,
                )
                ratio = k_val / (z_obs + epsilon)
                if ratio > best_ratio:
                    best_ratio = ratio
            if best_ratio < tau_mix:
                keep_mask[m] = False
        pruned_positions = best.positions[keep_mask]
        pruned_strengths = best.strengths[keep_mask]
        pruned_state = IsotopeState(
            num_sources=pruned_positions.shape[0],
            positions=pruned_positions,
            strengths=pruned_strengths,
            background=best.background,
        )
        pruned[iso] = pruned_state
    return pruned
