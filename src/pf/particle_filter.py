"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer
from pf.state import ParticleState
from pf.weights import effective_sample_size, log_weight_update_poisson
from pf.resampling import systematic_resample
from pf.regularization import regularize_states


@dataclass
class PFConfig:
    """粒子フィルタ設定（Sec. 3.4）。"""

    num_particles: int = 200
    max_sources: int = 1
    resample_threshold: float = 0.5  # relative to N
    strength_sigma: float = 0.1
    background_sigma: float = 0.1


class IsotopeParticleFilter:
    """同位体ごとの粒子フィルタを実装。"""

    def __init__(
        self,
        isotope: str,
        kernel: KernelPrecomputer,
        config: PFConfig | None = None,
    ) -> None:
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        self.states: List[ParticleState] = []
        self.log_weights: NDArray[np.float64] = np.zeros(self.N)
        self._init_particles()

    def _init_particles(self) -> None:
        """源位置と強度を乱択して初期化。"""
        self.states = []
        J = self.kernel.num_sources
        for _ in range(self.N):
            idx = np.random.choice(J, size=self.config.max_sources, replace=True)
            strengths = np.abs(np.random.normal(loc=1.0, scale=0.5, size=self.config.max_sources))
            self.states.append(ParticleState(source_indices=idx.astype(np.int32), strengths=strengths, background=0.1))
        self.log_weights = np.log(np.ones(self.N) / self.N)

    def predict(self) -> None:
        """位置は固定グリッドなので予測ステップは強度と背景の拡散に限定。"""
        self.states = regularize_states(
            self.states, strength_sigma=self.config.strength_sigma, background_sigma=self.config.background_sigma
        )

    def expected_counts(self, pose_idx: int, orient_idx: int, live_time_s: float) -> NDArray[np.float64]:
        """各粒子の期待計数を返す。"""
        lam = np.zeros(self.N, dtype=float)
        for i, st in enumerate(self.states):
            contrib = 0.0
            for idx_src, strength in zip(st.source_indices, st.strengths):
                kvec = self.kernel.kernel(self.isotope, pose_idx, orient_idx)
                contrib += kvec[idx_src] * strength
            lam[i] = live_time_s * (contrib + st.background)
        return lam

    def update(self, z_obs: float, pose_idx: int, orient_idx: int, live_time_s: float) -> None:
        """ポアソン重み更新。"""
        lam = self.expected_counts(pose_idx, orient_idx, live_time_s)
        self.log_weights = log_weight_update_poisson(self.log_weights, z_obs=z_obs, lambda_exp=lam)
        self._maybe_resample()

    def _maybe_resample(self) -> None:
        """有効サンプル数が閾値以下ならリサンプリング。"""
        ess = effective_sample_size(self.log_weights)
        if ess < self.config.resample_threshold * self.N:
            idx = systematic_resample(self.log_weights)
            self.states = [self.states[i].copy() for i in idx]
            self.log_weights = np.log(np.ones(self.N) / self.N)
            self.predict()

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """源位置と強度の単純加重平均推定を返す。"""
        w = np.exp(self.log_weights)
        pos_accum = np.zeros((len(self.states[0].source_indices), 3))
        strength_accum = np.zeros(len(self.states[0].source_indices))
        for st, wi in zip(self.states, w):
            for j, idx_src in enumerate(st.source_indices):
                pos_accum[j] += wi * self.kernel.sources[idx_src]
                strength_accum[j] += wi * st.strengths[j]
        return pos_accum, strength_accum
