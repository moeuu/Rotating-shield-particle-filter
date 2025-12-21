"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from pf.state import ParticleState, IsotopeState
from pf.weights import effective_sample_size, log_weight_update_poisson
from pf.resampling import systematic_resample
from pf.regularization import regularize_states


@dataclass
class PFConfig:
    """粒子フィルタ設定（Sec. 3.4）。"""

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    max_sources: int = 1
    resample_threshold: float = 0.5  # relative to N
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    min_strength: float = 0.01
    p_birth: float = 0.05
    ess_low: float = 0.5
    ess_high: float = 0.9
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (0, 3)  # inclusive range
    # Strength prior (cps@1m scale). Defaults cover ~1e3–1e5 cps via log-normal.
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
    use_discrete: bool = True  # set False to skip legacy discrete initialisation


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float


class IsotopeParticleFilter:
    """同位体ごとの粒子フィルタを実装（連続状態のPFを主体に運用）。"""

    def __init__(
        self,
        isotope: str,
        kernel: KernelPrecomputer | None,
        config: PFConfig | None = None,
    ) -> None:
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
        self.states: List[ParticleState] = []
        self.log_weights: NDArray[np.float64] = np.zeros(self.N)
        if self.kernel is not None and self.config.use_discrete:
            self._init_particles()
        mu_by_isotope = getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        shield_params = getattr(kernel, "shield_params", ShieldParams()) if kernel is not None else ShieldParams()
        self.continuous_kernel = ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        self.continuous_particles: List[IsotopeParticle] = []
        self._init_continuous_particles()

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a kernel and refresh the continuous-kernel configuration."""
        self.kernel = kernel
        self.continuous_kernel = ContinuousKernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )

    def _init_continuous_particles(self) -> None:
        """Sample continuous positions/strengths/background from broad priors (Sec. 3.3.2)."""
        self.continuous_particles = []
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        min_r, max_r = self.config.init_num_sources
        for _ in range(self.N):
            r_h = int(np.random.randint(min_r, max_r + 1))
            if self.config.max_sources > 0:
                r_h = min(r_h, self.config.max_sources)
            if r_h > 0:
                positions = lo + np.random.rand(r_h, 3) * (hi - lo)
                strengths = np.random.lognormal(
                    mean=self.config.init_strength_log_mean, sigma=self.config.init_strength_log_sigma, size=r_h
                )
            else:
                positions = np.zeros((0, 3), dtype=float)
                strengths = np.zeros(0, dtype=float)
            b_h = float(max(0.0, np.random.normal(loc=0.1, scale=0.05)))
            st = IsotopeState(num_sources=r_h, positions=positions, strengths=strengths, background=b_h)
            self.continuous_particles.append(IsotopeParticle(state=st, log_weight=float(np.log(1.0 / self.N))))

    def _continuous_expected_counts(self, pose_idx: int, orient_idx: int, live_time_s: float) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for each continuous particle using ContinuousKernel."""
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        detector_pos = self.kernel.poses[pose_idx]
        orient_vec = self.kernel.orientations[orient_idx]
        for i, p in enumerate(self.continuous_particles):
            st = p.state
            lam[i] = self.continuous_kernel.expected_counts(
                isotope=self.isotope,
                detector_pos=detector_pos,
                sources=st.positions,
                strengths=st.strengths,
                orient_idx=self.continuous_kernel.orient_index_from_vector(orient_vec),
                live_time_s=live_time_s,
                background=st.background,
            )
        return lam

    def _continuous_expected_counts_pair(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using Fe/Pb octant indices (Eq. 3.41)."""
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        detector_pos = self.kernel.poses[pose_idx]
        for i, p in enumerate(self.continuous_particles):
            st = p.state
            lam[i] = self.continuous_kernel.expected_counts_pair(
                isotope=self.isotope,
                detector_pos=detector_pos,
                sources=st.positions,
                strengths=st.strengths,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                background=st.background,
            )
        return lam

    def update_continuous(self, z_obs: float, pose_idx: int, orient_idx: int, live_time_s: float) -> None:
        """
        Poisson log-weight update for continuous particles (Sec. 3.3.3).

        z_obs should come from spectrum unfolding (Chapter 2.5.7). Expected Λ is
        computed via continuous kernel; no shortcut counts are used.
        """
        lam = self._continuous_expected_counts(pose_idx, orient_idx, live_time_s)
        log_unnorm = np.array([p.log_weight for p in self.continuous_particles], dtype=float)
        log_unnorm = log_unnorm + z_obs * np.log(lam + 1e-12) - lam
        log_unnorm -= np.max(log_unnorm)
        w = np.exp(log_unnorm)
        w /= np.sum(w)
        for p, wi in zip(self.continuous_particles, w):
            p.log_weight = float(np.log(wi + 1e-20))
        self._maybe_resample_continuous()

    def update_continuous_pair(self, z_obs: float, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float) -> None:
        """
        Poisson log-weight update using Fe/Pb orientation indices (Eq. 3.41–3.44).

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        """
        lam = self._continuous_expected_counts_pair(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        log_unnorm = np.array([p.log_weight for p in self.continuous_particles], dtype=float)
        log_unnorm = log_unnorm + z_obs * np.log(lam + 1e-12) - lam
        log_unnorm -= np.max(log_unnorm)
        w = np.exp(log_unnorm)
        w /= np.sum(w)
        for p, wi in zip(self.continuous_particles, w):
            p.log_weight = float(np.log(wi + 1e-20))
        self._maybe_resample_continuous()

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        w = np.exp([p.log_weight for p in self.continuous_particles])
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles)) / len(self.continuous_particles)
        return w / s

    def _maybe_resample_continuous(self) -> None:
        """ESS check and systematic resampling for continuous particles (Sec. 3.3.4, Eq. 3.29)."""
        w = self.continuous_weights
        ess = 1.0 / np.sum(w**2)
        if ess < self.config.resample_threshold * self.N:
            idx = systematic_resample(np.log(w))
            self.continuous_particles = [self.continuous_particles[i].state.copy() for i in idx]
            # reset weights to uniform
            self.continuous_particles = [
                IsotopeParticle(state=st, log_weight=float(-np.log(self.N))) for st in self.continuous_particles
            ]
            self.regularize_continuous(
                sigma_pos=self.config.strength_sigma,
                sigma_int=self.config.strength_sigma,
                p_birth=self.config.p_birth,
                p_kill=0.1,
                intensity_threshold=self.config.min_strength,
            )

    def adapt_num_particles(self) -> None:
        """
        Optional: adapt N based on variance/entropy of weights (Chapter 3.3.4).
        """
        if not self.continuous_particles:
            return
        min_particles = (
            max(1, int(self.config.min_particles))
            if self.config.min_particles is not None
            else max(1, int(self.config.num_particles))
        )
        max_particles = (
            max(1, int(self.config.max_particles))
            if self.config.max_particles is not None
            else max(1, int(self.config.num_particles))
        )
        w = self.continuous_weights
        ess = 1.0 / np.sum(w**2)
        ess_ratio = ess / max(len(w), 1)
        if ess_ratio < self.config.ess_low and len(w) < max_particles:
            target = min(max_particles, max(len(w) + 1, int(len(w) * 1.25)))
            self._resample_continuous_to(target, jitter=True)
        elif ess_ratio > self.config.ess_high and len(w) > min_particles:
            target = max(min_particles, int(len(w) * 0.8))
            self._resample_continuous_to(target, jitter=False)

    def _resample_continuous_to(self, target_n: int, jitter: bool = False) -> None:
        """Resample the continuous particles to a new population size."""
        target_n = max(1, int(target_n))
        w = self.continuous_weights
        idx = np.random.choice(len(self.continuous_particles), size=target_n, p=w)
        states = [self.continuous_particles[i].state.copy() for i in idx]
        self.continuous_particles = [
            IsotopeParticle(state=st, log_weight=float(-np.log(target_n))) for st in states
        ]
        self.N = target_n
        self.config.num_particles = target_n
        if jitter:
            self.regularize_continuous(
                sigma_pos=self.config.strength_sigma,
                sigma_int=self.config.strength_sigma,
                p_birth=self.config.p_birth,
                p_kill=0.1,
                intensity_threshold=self.config.min_strength,
            )

    def best_particle(self) -> IsotopeParticle:
        """Return the particle with maximum log_weight."""
        return max(self.continuous_particles, key=lambda p: p.log_weight)

    def regularize_continuous(
        self,
        sigma_pos: float = 0.05,
        sigma_int: float = 0.05,
        p_birth: float = 0.05,
        p_kill: float = 0.1,
        intensity_threshold: float = 0.05,
    ) -> None:
        """
        Apply small Gaussian jitter to positions/strengths and simple birth/death moves (Sec. 3.3.4).

        - positions: s <- s + N(0, sigma_pos^2 I)
        - strengths: q <- max(q + N(0, sigma_int^2), 0)
        - delete sources with q < intensity_threshold with prob p_kill
        - with prob p_birth, add a new source uniformly in workspace with small initial strength
        """
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        for p in self.continuous_particles:
            st = p.state
            if st.positions.size:
                st.positions = st.positions + np.random.normal(scale=sigma_pos, size=st.positions.shape)
                st.positions = np.clip(st.positions, lo, hi)
                st.strengths = np.maximum(st.strengths + np.random.normal(scale=sigma_int, size=st.strengths.shape), 0.0)
                # kill weak sources
                mask = np.ones(st.num_sources, dtype=bool)
                for i, q in enumerate(st.strengths):
                    if q < intensity_threshold and np.random.rand() < p_kill:
                        mask[i] = False
                st.positions = st.positions[mask]
                st.strengths = st.strengths[mask]
                st.num_sources = st.positions.shape[0]
            # birth
            if np.random.rand() < p_birth:
                new_pos = lo + np.random.rand(3) * (hi - lo)
                new_strength = float(np.abs(np.random.normal(loc=0.1, scale=0.05)))
                st.positions = np.vstack([st.positions, new_pos])
                st.strengths = np.append(st.strengths, new_strength)
                st.num_sources = st.positions.shape[0]

    def _init_particles(self) -> None:
        """源位置と強度を乱択して初期化。"""
        self.states = []
        J = self.kernel.num_sources
        for _ in range(self.N):
            r = np.random.randint(1, self.config.max_sources + 1)
            replace = r > J
            idx = np.random.choice(J, size=r, replace=replace)
            strengths = np.abs(np.random.normal(loc=1.0, scale=0.5, size=r))
            self.states.append(ParticleState(source_indices=idx.astype(np.int32), strengths=strengths, background=0.1))
        self.log_weights = np.log(np.ones(self.N) / self.N)

    def predict(self) -> None:
        """位置は固定グリッドなので予測ステップは強度と背景の拡散に限定。"""
        self.states = regularize_states(
            self.states,
            kernel=self.kernel,
            strength_sigma=self.config.strength_sigma,
            background_sigma=self.config.background_sigma,
            min_strength=self.config.min_strength,
            p_birth=self.config.p_birth,
            max_sources=self.config.max_sources,
        )

    def update(self, z_obs: float, pose_idx: int, orient_idx: int, live_time_s: float) -> None:
        """
        ポアソン重み更新。

        Note: z_obs はスペクトル展開（Sec. 2.5.7）から得た同位体別カウントを必須とする。
        このメソッド自身が幾何モデルから観測を合成することはなく、期待値計算は
        観測との対比のための内部モデルに限定される。
        """
        # PF now consumes isotope-wise counts from spectrum unfolding; expected rate
        # is approximated using current strengths and geometric kernels.
        lam = np.zeros(self.N, dtype=float)
        for i, st in enumerate(self.states):
            contrib = 0.0
            for idx_src, strength in zip(st.source_indices, st.strengths):
                kvec = self.kernel.kernel(self.isotope, pose_idx, orient_idx)
                contrib += kvec[idx_src] * strength
            lam[i] = live_time_s * (contrib + st.background)
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
        """
        Continuous MMSE estimate over positions/strengths using continuous_particles.
        """
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        w = self.continuous_weights
        max_r = max((p.state.num_sources for p in self.continuous_particles), default=0)
        positions = np.zeros((max_r, 3), dtype=float)
        strengths = np.zeros(max_r, dtype=float)
        for j in range(max_r):
            pos_stack = []
            str_stack = []
            w_stack = []
            for wi, p in zip(w, self.continuous_particles):
                if p.state.num_sources > j:
                    pos_stack.append(p.state.positions[j])
                    str_stack.append(p.state.strengths[j])
                    w_stack.append(wi)
            if not w_stack:
                continue
            wj = np.array(w_stack, dtype=float)
            wj = wj / max(np.sum(wj), 1e-12)
            pos_arr = np.vstack(pos_stack)
            str_arr = np.array(str_stack, dtype=float)
            positions[j] = np.sum(wj[:, None] * pos_arr, axis=0)
            strengths[j] = float(np.sum(wj * str_arr))
        # Trim zeros beyond max_sources
        mask = strengths > 0
        positions = positions[mask]
        strengths = strengths[mask]
        if positions.shape[0] > self.config.max_sources:
            order = np.argsort(strengths)[::-1][: self.config.max_sources]
            positions = positions[order]
            strengths = strengths[order]
        return positions, strengths
