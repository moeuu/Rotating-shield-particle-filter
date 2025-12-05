"""Parallel PF wrapper with one PF per isotope (Chapter 3.3.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pf.particle_filter import IsotopeParticleFilter, PFConfig
from pf.state import IsotopeState


@dataclass
class Measurement:
    """Simple container for isotope-wise counts at one time step."""

    counts_by_isotope: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float = 1.0


class ParallelIsotopePF:
    """
    Hold one PF per isotope and provide update/estimate helpers (Sec. 3.3.5).

    Currently uses the continuous PF scaffold via IsotopeParticleFilter.update_continuous.
    """

    def __init__(self, isotope_names: List[str], config: PFConfig) -> None:
        self.measurements: List[Measurement] = []
        self.filters: Dict[str, IsotopeParticleFilter] = {
            iso: IsotopeParticleFilter(isotope=iso, kernel=None, config=config)  # kernel to be injected later
            for iso in isotope_names
        }
        self.history: List[Dict[str, IsotopeState]] = []

    def attach_kernel(self, isotope: str, kernel) -> None:
        """Assign a measurement kernel to a specific isotope PF."""
        if isotope in self.filters:
            self.filters[isotope].kernel = kernel

    def update_all(self, measurement: Measurement) -> None:
        """Update all isotope PFs with isotope-wise counts z_{k,h}."""
        self.measurements.append(measurement)
        for iso, pf in self.filters.items():
            z_kh = measurement.counts_by_isotope.get(iso, 0.0)
            # continuous update assumes kernel/poses/orientations are set on pf.kernel
            pf.update_continuous(z_obs=z_kh, pose_idx=measurement.pose_idx, orient_idx=measurement.orient_idx, live_time_s=measurement.live_time_s)

    def estimate_all(self) -> Dict[str, IsotopeState]:
        """
        Return current weighted-average estimates for each isotope.

        For now, returns the weighted mean positions/strengths/background from continuous particles.
        """
        estimates: Dict[str, IsotopeState] = {}
        for iso, pf in self.filters.items():
            w = pf.continuous_weights
            if not pf.continuous_particles:
                continue
            # For simplicity, use expected num_sources = weighted average of counts
            num_sources = int(np.round(np.sum([p.state.num_sources * wi for p, wi in zip(pf.continuous_particles, w)])))
            # Aggregate positions/strengths up to num_sources by weighted mean (pad if needed)
            all_pos = []
            all_str = []
            for p, wi in zip(pf.continuous_particles, w):
                all_pos.append(wi * p.state.positions)
                all_str.append(wi * p.state.strengths)
            if all_pos:
                pos_mean = np.sum(np.vstack(all_pos), axis=0)
                str_mean = np.sum(np.concatenate(all_str)) if all_str else 0.0
                # naive reshape: replicate mean position/strength
                if num_sources <= 0:
                    positions = np.zeros((0, 3))
                    strengths = np.zeros(0)
                else:
                    positions = np.tile(pos_mean / max(len(all_pos), 1), (num_sources, 1))
                    strengths = np.full(num_sources, str_mean / max(num_sources, 1))
            else:
                positions = np.zeros((0, 3))
                strengths = np.zeros(0)
            bg = float(np.sum([wi * p.state.background for p, wi in zip(pf.continuous_particles, w)]))
            estimates[iso] = IsotopeState(num_sources=num_sources, positions=positions, strengths=strengths, background=bg)
        self.history.append(estimates)
        return estimates

    def has_converged(self, tau_conv: float = 1e-2, window: int = 5) -> bool:
        """
        Simple convergence check: if recent estimate changes are below tau_conv for all isotopes.

        Uses L2 norm over concatenated positions + strengths + background across last `window` estimates.
        """
        if len(self.history) < window + 1:
            return False
        recent = self.history[-(window + 1) :]
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            for iso in self.filters:
                if iso not in prev or iso not in curr:
                    continue
                p_state = prev[iso]
                c_state = curr[iso]
                vec_prev = np.concatenate(
                    [p_state.positions.ravel(), p_state.strengths.ravel(), np.array([p_state.background])]
                )
                vec_curr = np.concatenate(
                    [c_state.positions.ravel(), c_state.strengths.ravel(), np.array([c_state.background])]
                )
                # Pad to same length
                L = max(len(vec_prev), len(vec_curr))
                vec_prev = np.pad(vec_prev, (0, L - len(vec_prev)), mode="constant")
                vec_curr = np.pad(vec_curr, (0, L - len(vec_curr)), mode="constant")
                if np.linalg.norm(vec_curr - vec_prev) > tau_conv:
                    return False
        return True
