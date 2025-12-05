"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from pf.particle_filter import IsotopeParticleFilter, PFConfig


@dataclass
class RotatingShieldPFConfig:
    """
    設定パラメータ（Sec. 3.4–3.5）。

    Users can tune convergence thresholds:
        - ig_threshold: max IG below which rotation stops (Eq. 3.49)
        - max_dwell_time_s: per-pose dwell cap
        - credible_volume_threshold: max ellipsoid volume for positional credible regions
        - lambda_cost: motion-cost weight in Eq. 3.51
        - alpha_weights / beta_weights: isotope weights for IG / Fisher criteria
    """

    num_particles: int = 200
    max_sources: int = 1
    resample_threshold: float = 0.5
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    min_strength: float = 0.01
    p_birth: float = 0.05
    short_time_s: float = 0.5  # 推奨短時間計測時間（Sec. 3.4.3）
    ig_threshold: float = 1e-3  # ΔIG停止閾値（Sec. 3.4.4）
    max_dwell_time_s: float = 5.0  # 1ポーズあたりの最大滞留時間
    lambda_cost: float = 1.0  # 移動コスト重み（Eq. 3.51）
    alpha_weights: Dict[str, float] | None = None  # EIGの同位体重み α_h
    beta_weights: Dict[str, float] | None = None  # Fisher基準の同位体重み β_h
    credible_volume_threshold: float = 1e-3  # 95%位置信用領域体積の閾値（収束判定用）


@dataclass(frozen=True)
class MeasurementRecord:
    """Store a single isotope-wise measurement and metadata."""

    z_k: Dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float
    fe_index: int | None = None
    pb_index: int | None = None
    ig_value: float | None = None


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
                fe_index=None,
                pb_index=None,
            )
        )
        # reset cache if new pose added later

    def predict(self) -> None:
        """全PFで予測ステップを行う。"""
        for f in self.filters.values():
            f.predict()

    def short_time_update(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float | None = None,
    ) -> None:
        """
        短時間計測 (Sec. 3.4.3) を PF に反映するヘルパ。

        - シールド方位 (RFe, RPb) を設定し、短時間 T_k で取得した z_k（同位体別カウント）を使用。
        - T_k は pf_config.short_time_s（デフォルト0.5s）を既定値とする。
        - z_k は必ずスペクトル処理パイプライン (Sec. 2.5.7) の展開結果であり、幾何カーネルから
          直接計算したカウントを渡さないこと。
        """
        duration = live_time_s if live_time_s is not None else self.pf_config.short_time_s
        fe_index = octant_index_from_rotation(RFe)
        pb_index = octant_index_from_rotation(RPb)
        self.update_pair(z_k=z_k, pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=duration)

    def update_pair(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> None:
        """
        Update PFs using Fe/Pb orientation indices (RFe, RPb) and isotope-wise counts z_k.

        This feeds the continuous 3D PF path (Sec. 3.3.3) with Λ computed via expected_counts_pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            # Use continuous PF update that relies on spectrum-unfolded counts.
            self.filters[iso].update_continuous_pair(
                z_obs=val, pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
            )
        self.history_estimates.append(self.estimates())
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=pose_idx,
                orient_idx=fe_index,
                live_time_s=live_time_s,
                fe_index=fe_index,
                pb_index=pb_index,
                ig_value=None,
            )
        )

    def estimates(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """同位体ごとの位置・強度推定を返す。"""
        return {iso: f.estimate() for iso, f in self.filters.items()}

    @property
    def num_orientations(self) -> int:
        return self.normals.shape[0]

    def orientation_information_gain(self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        IG_k(phi) ~= 0.5 * log(1 + Var[Lambda_k(phi)] / E[Lambda_k(phi)]) aggregated over isotopes.
        """
        ig, _ = self.orientation_information_metrics(pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
        return ig

    def orientation_information_metrics(
        self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0
    ) -> Tuple[float, float]:
        """
        Compute (IG, Fisher) surrogates for a given orientation (Sec. 3.4.2, Eqs. 3.40–3.43).

        IG ≈ 0.5 log(1 + Var[Λ]/E[Λ]) and Fisher surrogate ≈ Var[Λ]/(E[Λ]^2+ε),
        where Λ are the per-particle expected counts under the current PF posterior.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        fisher_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            lam = np.zeros(filt.N, dtype=float)
            kvec = self.kernel_cache.kernel(iso, pose_idx, orient_idx)
            for i, st in enumerate(filt.states):
                contrib = 0.0
                for idx_src, strength in zip(st.source_indices, st.strengths):
                    contrib += kvec[idx_src] * strength
                lam[i] = live_time_s * (contrib + st.background)
            w = np.exp(filt.log_weights)
            w = w / max(np.sum(w), eps)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            ig_total += 0.5 * float(np.log1p(var / max(mean, eps)))
            fisher_total += float(var / (max(mean, eps) ** 2 + eps))
        return ig_total, fisher_total

    def max_orientation_information_gain(self, pose_idx: int, live_time_s: float = 1.0) -> float:
        """Return max_phi IG_k(phi) at pose k (Eq. 3.45 surrogate)."""
        scores = [
            self.orientation_information_gain(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
            for oidx in range(self.num_orientations)
        ]
        return float(np.max(scores)) if scores else 0.0

    def orientation_expected_information_gain(
        self,
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float = 1.0,
        num_samples: int = 50,
        alpha_by_isotope: Dict[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo approximation of EIG (Eq. 3.44) for a Fe/Pb orientation pair.

        - Uses continuous particles and ContinuousKernel expected counts (Eq. 3.41).
        - For each isotope h: IG_h = H(w_h) - E_z[H(w'_h(z; RFe, RPb))].
        - Global IG = Σ_h α_h IG_h, with α_h uniform if not provided.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        rng = rng or np.random.default_rng()
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        kernel = ContinuousKernel()
        detector_pos = self.kernel_cache.poses[pose_idx]
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        # normalize alphas
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}

        def _logsumexp(x: NDArray[np.float64]) -> float:
            m = float(np.max(x))
            return m + float(np.log(np.sum(np.exp(x - m))))

        total_ig = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            lam = np.zeros(len(filt.continuous_particles), dtype=float)
            for i, p in enumerate(filt.continuous_particles):
                st = p.state
                lam[i] = kernel.expected_counts_pair(
                    isotope=iso,
                    detector_pos=detector_pos,
                    sources=st.positions,
                    strengths=st.strengths,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_s,
                    background=st.background,
                )
            # Prior entropy H(w)
            H_prior = float(-np.sum(weights * np.log(weights + eps)))
            # Monte-Carlo expectation over z ~ mixture of Poissons
            H_post_accum = 0.0
            for _ in range(num_samples):
                idx = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[idx])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= _logsumexp(logw)
                w_post = np.exp(logw)
                H_post_accum += float(-np.sum(w_post * logw))
            H_post_mean = H_post_accum / max(num_samples, 1)
            ig_h = H_prior - H_post_mean
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)

    def orientation_fisher_criteria(
        self,
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float = 1.0,
        beta_by_isotope: Dict[str, float] | None = None,
        ridge: float = 1e-6,
    ) -> Tuple[float, float]:
        """
        Compute JA, JD criteria (Eq. 3.46–3.47) for a Fe/Pb orientation pair.

        Approximates Fisher information using weighted particles:
            I_h ≈ Σ_n w_n (1/Λ_n) g_n g_n^T
        where g_n = ∂Λ_n/∂θ_h with θ_h = [q_{h,1}, ..., q_{h,r_h}, b_h].
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        beta = beta_by_isotope or {iso: 1.0 for iso in self.filters}
        beta_sum = sum(beta.values()) or 1.0
        beta = {k: v / beta_sum for k, v in beta.items()}
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        detector_pos = self.kernel_cache.poses[pose_idx]
        kernel = ContinuousKernel()
        JA_total = 0.0
        JD_total = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            max_r = max(p.state.num_sources for p in filt.continuous_particles)
            dim = max_r + 1  # strengths + background
            I = np.zeros((dim, dim), dtype=float)
            for w, p in zip(weights, filt.continuous_particles):
                st = p.state
                lam = kernel.expected_counts_pair(
                    isotope=iso,
                    detector_pos=detector_pos,
                    sources=st.positions,
                    strengths=st.strengths,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_s,
                    background=st.background,
                )
                lam = max(lam, ridge)
                g = np.zeros(dim, dtype=float)
                for j in range(st.num_sources):
                    geom = 1.0 / (4.0 * np.pi * max(np.linalg.norm(detector_pos - st.positions[j]), 1e-6) ** 2)
                    fe_block = kernel.octant_shield.blocks_ray(
                        detector_position=detector_pos, source_position=st.positions[j], octant_index=fe_idx
                    )
                    pb_block = kernel.octant_shield.blocks_ray(
                        detector_position=detector_pos, source_position=st.positions[j], octant_index=pb_idx
                    )
                    if fe_block and pb_block:
                        att = 0.01
                    elif fe_block or pb_block:
                        att = 0.1
                    else:
                        att = 1.0
                    g[j] = live_time_s * geom * att
                g[-1] = live_time_s  # derivative w.r.t. background
                I += w * (1.0 / lam) * np.outer(g, g)
            I += ridge * np.eye(dim)
            try:
                inv_I = np.linalg.inv(I)
            except np.linalg.LinAlgError:
                inv_I = np.linalg.pinv(I)
            trace_inv = float(np.trace(inv_I))
            JA_h = 1.0 / max(trace_inv, ridge)
            sign, logdet = np.linalg.slogdet(I)
            JD_h = logdet if sign > 0 else np.log(ridge)
            JA_total += beta.get(iso, 0.0) * JA_h
            JD_total += beta.get(iso, 0.0) * JD_h
        return float(JA_total), float(JD_total)

    def _strength_matrix(self, filt: IsotopeParticleFilter) -> NDArray[np.float64]:
        """
        Build a (N, max_r) matrix of source strengths for variance computation (Eq. 3.38 surrogate).
        """
        max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
        mat = np.zeros((len(filt.continuous_particles), max_r), dtype=float)
        for i, p in enumerate(filt.continuous_particles):
            r = p.state.num_sources
            if r > 0:
                mat[i, :r] = p.state.strengths
        return mat

    def expected_uncertainty_after_pose(
        self,
        pose_idx: int,
        fe_index: int | None = None,
        pb_index: int | None = None,
        orient_idx: int = 0,
        live_time_s: float = 1.0,
        num_samples: int = 20,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Monte-Carlo estimate of E[U | q_cand] where U = Σ_h Σ_m Var(q_{h,m}) (Eq. 3.38 surrogate).

        Draw hypothetical Poisson observations at pose q_cand and average posterior variance of strengths.
        Uses either Fe/Pb indices (if provided) or orient_idx into the kernel orientations.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        rng = rng or np.random.default_rng()
        eps = 1e-12
        total_U = 0.0
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            if fe_index is not None and pb_index is not None:
                lam = filt._continuous_expected_counts_pair(
                    pose_idx=pose_idx, fe_index=fe_index, pb_index=pb_index, live_time_s=live_time_s
                )
            else:
                lam = filt._continuous_expected_counts(pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
            strengths_mat = self._strength_matrix(filt)
            U_accum = 0.0
            for _ in range(num_samples):
                n = int(rng.choice(len(lam), p=weights))
                z = rng.poisson(lam[n])
                logw = np.log(weights + eps) + z * np.log(lam + eps) - lam
                logw -= logsumexp(logw)
                w_post = np.exp(logw)
                if strengths_mat.size == 0:
                    continue
                mean = np.sum(w_post[:, None] * strengths_mat, axis=0)
                var = np.sum(w_post[:, None] * (strengths_mat - mean) ** 2, axis=0)
                U_accum += float(np.sum(var))
            total_U += U_accum / max(num_samples, 1)
        return float(total_U)

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
            # Prefer continuous PF if available; otherwise fallback to legacy grid
            if filt.continuous_particles:
                w = filt.continuous_weights
                max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
                if max_r == 0:
                    continue
                strengths = np.zeros((len(filt.continuous_particles), max_r), dtype=float)
                for i, p in enumerate(filt.continuous_particles):
                    r = p.state.num_sources
                    if r > 0:
                        strengths[i, :r] = p.state.strengths
                mean = np.sum(w[:, None] * strengths, axis=0)
                var = np.sum(w[:, None] * (strengths - mean) ** 2, axis=0)
                total += float(np.sum(var))
            else:
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

    def credible_region_volumes(
        self, confidence: float = 0.95
    ) -> Dict[str, List[float]]:
        """
        Compute 3D positional credible region volumes for each isotope/source (Sec. 3.5).

        For each source index m (up to max_r across particles), compute weighted mean/cov
        of positions and return ellipsoid volume using chi-square threshold. Used by
        should_stop_shield_rotation/should_stop_exploration to enforce small positional
        uncertainty before declaring convergence.
        """
        volumes: Dict[str, List[float]] = {}
        chi2_thresh = float(chi2.ppf(confidence, df=3))
        for iso, filt in self.filters.items():
            vols: List[float] = []
            if not filt.continuous_particles:
                volumes[iso] = vols
                continue
            w = filt.continuous_weights
            max_r = max((p.state.num_sources for p in filt.continuous_particles), default=0)
            for j in range(max_r):
                positions = []
                weights = []
                for wi, p in zip(w, filt.continuous_particles):
                    if p.state.num_sources > j:
                        positions.append(p.state.positions[j])
                        weights.append(wi)
                if not positions:
                    continue
                pos_arr = np.vstack(positions)
                weights_arr = np.asarray(weights)
                weights_arr = weights_arr / max(np.sum(weights_arr), 1e-12)
                mean = np.sum(weights_arr[:, None] * pos_arr, axis=0)
                centered = pos_arr - mean
                cov = centered.T @ (centered * weights_arr[:, None])
                # Ellipsoid volume = 4/3 π sqrt(det(cov * chi2_thresh))
                det_val = np.linalg.det(cov * chi2_thresh)
                if det_val < 0:
                    vol = 0.0
                else:
                    vol = float((4.0 / 3.0) * np.pi * np.sqrt(det_val + 1e-12))
                vols.append(vol)
            volumes[iso] = vols
        return volumes

    def should_stop_shield_rotation(
        self,
        pose_idx: int,
        ig_threshold: float = 1e-3,
        fisher_threshold: float = 1e-3,
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
        ig_scores = []
        fisher_scores = []
        for oidx in range(self.num_orientations):
            ig, fisher = self.orientation_information_metrics(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
            ig_scores.append(ig)
            fisher_scores.append(fisher)
        max_ig = max(ig_scores) if ig_scores else 0.0
        max_fisher = max(fisher_scores) if fisher_scores else 0.0
        dwell_time = sum(rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx)
        # Credible region volumes check (Sec. 3.5)
        volumes = self.credible_region_volumes()
        max_volume = 0.0
        for vols in volumes.values():
            if vols:
                max_volume = max(max_volume, max(vols))
        return (
            (max_ig < ig_threshold)
            and (max_fisher < fisher_threshold)
            and (self.estimate_change_norm() < change_tol)
            and (self.global_uncertainty() < uncertainty_tol)
            and (max_volume < self.pf_config.credible_volume_threshold)
            or (dwell_time >= self.pf_config.max_dwell_time_s)
        )

    def should_stop_exploration(
        self,
        ig_threshold: float = 5e-4,
        fisher_threshold: float = 5e-4,
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
            fisher_threshold=fisher_threshold,
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
