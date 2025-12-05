"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer, ShieldParams
from pf.particle_filter import IsotopeParticleFilter, PFConfig


@dataclass
class RotatingShieldPFConfig:
    """設定パラメータ（Sec. 3.4–3.5）。"""

    num_particles: int = 200
    max_sources: int = 1
    resample_threshold: float = 0.5
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    min_strength: float = 0.01
    p_birth: float = 0.05


@dataclass(frozen=True)
class MeasurementRecord:
    """Store a single isotope-wise measurement and metadata."""

    z_k: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float


class RotatingShieldPFEstimator:
    """
    シールド回転を伴う並列PFによるオンライン源推定器（Sec. 3.4, 3.5, 3.6）。

    - 同位体ごとに独立PFを保持
    - 各更新で姿勢・遮蔽方位を受け取りPoisson重み更新
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        candidate_sources: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: Dict[str, float],
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
    ) -> None:
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        # 測定姿勢は逐次追加するので空で初期化
        self.poses: List[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = mu_by_isotope
        self.kernel_cache: KernelPrecomputer | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]] = []
        self.history_scores: List[float] = []
        self.measurements: List[MeasurementRecord] = []

    def _ensure_kernel_cache(self) -> None:
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = KernelPrecomputer(
            candidate_sources=self.candidate_sources,
            poses=poses_arr,
            orientations=self.normals,
            shield_params=self.shield_params,
            mu_by_isotope=self.mu_by_isotope,
        )
        pf_conf = PFConfig(
            num_particles=self.pf_config.num_particles,
            max_sources=self.pf_config.max_sources,
            resample_threshold=self.pf_config.resample_threshold,
            strength_sigma=self.pf_config.strength_sigma,
            background_sigma=self.pf_config.background_sigma,
            min_strength=self.pf_config.min_strength,
            p_birth=self.pf_config.p_birth,
        )
        for iso in self.isotopes:
            self.filters[iso] = IsotopeParticleFilter(iso, kernel=self.kernel_cache, config=pf_conf)

    def add_measurement_pose(self, pose: NDArray[np.float64]) -> None:
        """新しい測定姿勢を登録。最初の登録後にカーネルを構築。"""
        self.poses.append(np.asarray(pose, dtype=float))
        # 再構築は次回必要時に実施
        self.kernel_cache = None
        self.filters = {}

    def update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> None:
        """
        同位体別カウントz_kを用いてPF群を更新。

        z_k must come from the spectrum unfolding pipeline (Sec. 2.5.7); this method
        never fabricates observations from geometric kernels or ground truth.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            self.filters[iso].update(z_obs=val, pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
        self.history_estimates.append(self.estimates())
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=orient_idx,
                live_time_s=live_time_s,
            )
        )
        # reset cache if new pose added later

    def predict(self) -> None:
        """全PFで予測ステップを行う。"""
        for f in self.filters.values():
            f.predict()

    def estimates(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """同位体ごとの位置・強度推定を返す。"""
        return {iso: f.estimate() for iso, f in self.filters.items()}

    @property
    def num_orientations(self) -> int:
        return self.normals.shape[0]

    def orientation_information_gain(self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0) -> float:
        """簡易情報利得：予測強度の分散を用いた指標（Sec. 3.5.2簡略版）。"""
        score = 0.0
        for iso, f in self.filters.items():
            # Expected counts approximation using current strengths; used only for IG heuristic.
            lam = np.zeros(f.N, dtype=float)
            for i, st in enumerate(f.states):
                contrib = 0.0
                for idx_src, strength in zip(st.source_indices, st.strengths):
                    kvec = f.kernel.kernel(iso, pose_idx, orient_idx)
                    contrib += kvec[idx_src] * strength
                lam[i] = live_time_s * (contrib + st.background)
            w = np.exp(f.log_weights)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            score += var / (mean + 1e-6)
        return score

    def expected_uncertainty(self, pose_idx: int, live_time_s: float = 1.0) -> float:
        """全同位体の予測強度分散を足し上げた簡易不確実性指標。"""
        total = 0.0
        for iso, f in self.filters.items():
            lam = np.zeros(f.N, dtype=float)
            for i, st in enumerate(f.states):
                contrib = 0.0
                for idx_src, strength in zip(st.source_indices, st.strengths):
                    kvec = f.kernel.kernel(iso, pose_idx, 0)
                    contrib += kvec[idx_src] * strength
                lam[i] = live_time_s * (contrib + st.background)
            w = np.exp(f.log_weights)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            total += var
        return total

    def converged(self, ig_threshold: float = 1e-3, change_tol: float = 1e-2, window: int = 3) -> bool:
        """情報利得と推定変化に基づく簡易収束判定（Sec. 3.6簡略版）。"""
        if len(self.history_estimates) < window + 1:
            return False
        recent = self.history_estimates[-window:]
        # 位置・強度変化のノルムを確認
        diffs = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            for iso in self.isotopes:
                prev_pos, prev_str = prev.get(iso, (None, None))
                curr_pos, curr_str = curr.get(iso, (None, None))
                if prev_pos is None or curr_pos is None:
                    continue
                diffs.append(np.linalg.norm(prev_pos - curr_pos) + np.linalg.norm(prev_str - curr_str))
        if diffs and max(diffs) > change_tol:
            return False
        # 情報利得評価
        scores = []
        for orient_idx in range(self.num_orientations):
            scores.append(self.orientation_information_gain(pose_idx=len(self.poses) - 1, orient_idx=orient_idx))
        return max(scores) < ig_threshold

    def estimate_change_norm(self) -> float:
        """
        直近2回の推定差分ノルム ||Δs|| + ||Δq|| を返す（Sec. 3.6の収束判定の一部）。
        """
        if len(self.history_estimates) < 2:
            return float("inf")
        prev = self.history_estimates[-2]
        curr = self.history_estimates[-1]
        diff = 0.0
        for iso in self.isotopes:
            prev_pos, prev_str = prev.get(iso, (None, None))
            curr_pos, curr_str = curr.get(iso, (None, None))
            if prev_pos is None or curr_pos is None:
                continue
            m = min(len(prev_pos), len(curr_pos))
            if m > 0:
                diff += float(np.linalg.norm(prev_pos[:m] - curr_pos[:m]))
                diff += float(np.linalg.norm(prev_str[:m] - curr_str[:m]))
        return diff

    def global_uncertainty(self) -> float:
        """
        粒子群から強度分散を集計した不確実性 U = Σ_h Σ_j Var(q_{h,j}) を返す（Sec. 3.6のU）。
        """
        total = 0.0
        for iso, filt in self.filters.items():
            weights = np.exp(filt.log_weights)
            weights = weights / max(np.sum(weights), 1e-12)
            mean = np.zeros(self.candidate_sources.shape[0], dtype=float)
            second = np.zeros_like(mean)
            for st, wi in zip(filt.states, weights):
                for idx_src, strength in zip(st.source_indices, st.strengths):
                    mean[idx_src] += wi * strength
                    second[idx_src] += wi * (strength**2)
            var = np.clip(second - mean**2, a_min=0.0, a_max=None)
            total += float(np.sum(var))
        return total

    def should_stop_shield_rotation(
        self,
        pose_idx: int,
        ig_threshold: float = 1e-3,
        change_tol: float = 1e-2,
        uncertainty_tol: float = 1e-3,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        シールド回転を終了する条件（Sec. 3.5, Eq. for IG + Sec. 3.6の収束条件）。

        - 最大情報利得 max_φ IG_k(φ) が閾値未満
        - 直近推定差分 ||Δs|| + ||Δq|| < change_tol
        - グローバル不確実性 U が閾値未満
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if len(self.history_estimates) < 2:
            return False
        ig_scores = [
            self.orientation_information_gain(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
            for oidx in range(self.num_orientations)
        ]
        max_ig = max(ig_scores) if ig_scores else 0.0
        return (max_ig < ig_threshold) and (self.estimate_change_norm() < change_tol) and (
            self.global_uncertainty() < uncertainty_tol
        )

    def should_stop_exploration(
        self,
        ig_threshold: float = 5e-4,
        change_tol: float = 5e-3,
        uncertainty_tol: float = 5e-4,
        live_time_s: float = 1.0,
    ) -> bool:
        """
        探索全体を終了する条件（Sec. 3.6, Uと情報利得の収束に基づく）。

        - 最終ポーズでの最大IGが小さい（これ以上ポーズを変えても情報が増えない）
        - 推定変化が小さい
        - グローバル不確実性 U が十分小さい
        """
        if not self.poses:
            return False
        last_pose_idx = len(self.poses) - 1
        return self.should_stop_shield_rotation(
            pose_idx=last_pose_idx,
            ig_threshold=ig_threshold,
            change_tol=change_tol,
            uncertainty_tol=uncertainty_tol,
            live_time_s=live_time_s,
        )

    def prune_spurious_sources(self, tau_mix: float = 0.9, epsilon: float = 1e-6) -> Dict[str, NDArray[np.bool_]]:
        """
        Apply the best-case measurement test (Sec. 3.4.5) and zero out spurious sources.

        For each isotope h and each candidate grid point j with non-zero expected strength,
        find the measurement index k* that maximises the ratio \\hat{Λ}_{k,h,j}/(z_{k,h}+ε)
        (Eq. k* definition in Sec. 3.4.5), where \\hat{Λ}_{k,h,j} is the expected contribution
        of that source alone. If the best-case ratio \\hat{Λ}_{k*,h,j}/z_{k*,h} falls below τ_mix,
        the source is marked spurious and removed from every particle. Returns a boolean mask
        (length = num candidate grid points) of kept sources per isotope.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if not self.measurements:
            return {iso: np.ones(self.candidate_sources.shape[0], dtype=bool) for iso in self.filters}

        keep_masks: Dict[str, NDArray[np.bool_]] = {}
        for iso, filt in self.filters.items():
            weights = np.exp(filt.log_weights)
            weights = weights / max(np.sum(weights), 1e-12)

            expected_strength = np.zeros(self.candidate_sources.shape[0], dtype=float)
            for st, wi in zip(filt.states, weights):
                for idx_src, strength in zip(st.source_indices, st.strengths):
                    expected_strength[idx_src] += wi * strength

            keep_mask = np.ones_like(expected_strength, dtype=bool)
            active_indices = np.nonzero(expected_strength > 0.0)[0]

            for idx_src in active_indices:
                best_ratio: float | None = None
                for rec in self.measurements:
                    if iso not in rec.z_k:
                        continue
                    kvec = self.kernel_cache.kernel(iso, rec.pose_idx, rec.orient_idx)
                    pred = float(rec.live_time_s * kvec[idx_src] * expected_strength[idx_src])
                    obs = float(rec.z_k.get(iso, 0.0))
                    ratio = pred / (obs + epsilon)
                    if best_ratio is None or ratio > best_ratio:
                        best_ratio = ratio
                if best_ratio is not None and best_ratio < tau_mix:
                    keep_mask[idx_src] = False
                    for st in filt.states:
                        if idx_src in st.source_indices:
                            mask = st.source_indices != idx_src
                            st.source_indices = st.source_indices[mask]
                            st.strengths = st.strengths[mask]
            keep_masks[iso] = keep_mask
        return keep_masks
