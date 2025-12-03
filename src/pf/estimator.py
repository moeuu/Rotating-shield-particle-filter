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
        shield_normals: NDArray[np.float64],
        mu_by_isotope: Dict[str, float],
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
    ) -> None:
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        # 測定姿勢は逐次追加するので空で初期化
        self.poses: List[NDArray[np.float64]] = []
        self.normals = shield_normals
        self.mu_by_isotope = mu_by_isotope
        self.kernel_cache: KernelPrecomputer | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]] = []
        self.history_scores: List[float] = []

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
        """同位体別カウントz_kを用いてPF群を更新。"""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            self.filters[iso].update(z_obs=val, pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
        self.history_estimates.append(self.estimates())
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
            lam = f.expected_counts(pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
            w = np.exp(f.log_weights)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            score += var / (mean + 1e-6)
        return score

    def expected_uncertainty(self, pose_idx: int, live_time_s: float = 1.0) -> float:
        """全同位体の予測強度分散を足し上げた簡易不確実性指標。"""
        total = 0.0
        for iso, f in self.filters.items():
            lam = f.expected_counts(pose_idx=pose_idx, orient_idx=0, live_time_s=live_time_s)
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
