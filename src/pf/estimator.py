"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Any
import copy

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from pf.particle_filter import IsotopeParticleFilter, PFConfig
from pf.state import IsotopeState


@dataclass
class RotatingShieldPFConfig:
    """
    Configuration parameters for the rotating-shield PF (Sec. 3.4–3.5).

    Users can tune convergence thresholds and planning settings:
        - max_sources: optional cap on the number of sources per isotope (None = no cap)
        - ig_threshold: max IG below which rotation stops (Eq. 3.49)
        - max_dwell_time_s: per-pose dwell cap
        - credible_volume_threshold: max ellipsoid volume for positional credible regions
        - lambda_cost: motion-cost weight in Eq. 3.51
        - alpha_weights / beta_weights: isotope weights for IG / Fisher criteria
        - orientation_k: number of orientations to execute per pose
        - orientation_selection_mode: "eig", "fisher", or "hybrid"
        - planning_particles: particle count used for orientation scoring (None = all)
        - planning_method: how to select planning particles (top_weight/resample)
        - eig_num_samples: Monte-Carlo samples for EIG (Eq. 3.44)
        - fisher_screening_metric: "ja" or "jd" for Fisher screening
        - fisher_screening_k: number of top candidates kept after Fisher screening
        - hybrid_eig_samples: small-sample count for EIG refinement
        - hybrid_entropy_threshold: switch to Fisher-only when entropy ratio is below this
        - preselect_*: optional surrogate stage settings for candidate reduction
    """

    num_particles: int = 200
    max_sources: int | None = None
    resample_threshold: float = 0.5
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    min_strength: float = 0.01
    p_birth: float = 0.05
    short_time_s: float = 0.5  # Recommended short-time measurement (Sec. 3.4.3).
    ig_threshold: float = 1e-3  # ΔIG stopping threshold (Sec. 3.4.4).
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    lambda_cost: float = 1.0  # Motion-cost weight (Eq. 3.51).
    alpha_weights: Dict[str, float] | None = None  # EIG isotope weights alpha_h.
    beta_weights: Dict[str, float] | None = None  # Fisher weights beta_h.
    credible_volume_threshold: float = 1e-3  # Max 95% credible volume for convergence.
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    orientation_k: int = 16
    orientation_selection_mode: str = "eig"
    planning_particles: int | None = None
    planning_method: str = "top_weight"
    eig_num_samples: int = 50
    fisher_screening_metric: str = "ja"
    fisher_screening_k: int = 12
    hybrid_eig_samples: int = 20
    hybrid_entropy_threshold: float = 0.4
    preselect_orientations: bool = False
    preselect_metric: str = "var_log_lambda"
    preselect_delta: float = 0.05
    preselect_k_min: int = 8
    preselect_k_max: int = 16


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
    Online source estimator using parallel PFs with shield rotation (Sec. 3.4–3.6).

    - Maintains one PF per isotope.
    - Updates each PF with pose/orientation and Poisson weight updates.
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        candidate_sources: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: Dict[str, object],
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
    ) -> None:
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        # Measurement poses are appended incrementally.
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
            position_min=self.pf_config.position_min,
            position_max=self.pf_config.position_max,
            use_discrete=False,
        )
        for iso in self.isotopes:
            self.filters[iso] = IsotopeParticleFilter(iso, kernel=self.kernel_cache, config=pf_conf)

    def planning_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]:
        """
        Select per-isotope particle subsets for orientation evaluation.

        Args:
            max_particles: cap on particles per isotope; None uses config default.
            method: "top_weight" or "resample"; None uses config default.
            rng: optional RNG for resampling.
        """
        if max_particles is None:
            max_particles = self.pf_config.planning_particles
        method = method or self.pf_config.planning_method
        rng = rng or np.random.default_rng()
        subsets: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] = {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = filt.continuous_weights
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            weights = weights / total
            n_particles = len(weights)
            if max_particles is None or max_particles <= 0 or max_particles >= n_particles:
                states = [p.state for p in filt.continuous_particles]
                subsets[iso] = (states, weights)
                continue
            if method == "top_weight":
                idx = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[idx]
                sel_weights = sel_weights / max(np.sum(sel_weights), 1e-12)
            elif method == "resample":
                idx = rng.choice(n_particles, size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max_particles
            else:
                raise ValueError(f"Unknown planning particle selection method: {method}")
            states = [filt.continuous_particles[i].state for i in idx]
            subsets[iso] = (states, sel_weights)
        return subsets

    def weight_entropy_ratio(
        self,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
    ) -> float:
        """
        Return the mean normalized weight entropy across isotopes.

        The entropy ratio is H(w)/log(N) in [0, 1]. Lower values indicate a more
        concentrated posterior (less multi-modality).
        """
        entropies: List[float] = []
        eps = 1e-12
        for iso, filt in self.filters.items():
            if particles_by_isotope is not None and iso in particles_by_isotope:
                _, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                weights = filt.continuous_weights
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(float(np.sum(weights)), eps)
            if weights.size == 1:
                entropies.append(0.0)
                continue
            entropy = float(-np.sum(weights * np.log(weights + eps)))
            entropies.append(entropy / max(np.log(weights.size), eps))
        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def add_measurement_pose(self, pose: NDArray[np.float64]) -> None:
        """Register a new measurement pose (kernel built lazily on demand)."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
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
        Update per-isotope PFs using isotope-wise counts z_k.

        z_k must come from the spectrum unfolding pipeline (Sec. 2.5.7); this method
        never fabricates observations from geometric kernels or ground truth.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        for iso, val in z_k.items():
            if iso not in self.filters:
                continue
            filt = self.filters[iso]
            if filt.continuous_particles:
                filt.update_continuous(
                    z_obs=val, pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
            else:
                filt.update(z_obs=val, pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s)
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
        """Run the prediction step for all PFs."""
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
        Apply a short-time measurement update (Sec. 3.4.3).

        - Use shield orientations (RFe, RPb) and isotope-wise counts z_k.
        - T_k defaults to pf_config.short_time_s unless specified.
        - z_k must come from the spectrum pipeline (Sec. 2.5.7), not from geometry.
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
        """Return per-isotope position/strength estimates (MMSE over continuous particles)."""
        return {iso: f.estimate() for iso, f in self.filters.items()}

    def estimate_all(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

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
        self,
        pose_idx: int,
        orient_idx: int,
        live_time_s: float = 1.0,
        prefer_continuous: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute (IG, Fisher) surrogates for a given orientation (Sec. 3.4.2, Eqs. 3.40–3.43).

        IG ≈ 0.5 log(1 + Var[Λ]/E[Λ]) and Fisher surrogate ≈ Var[Λ]/(E[Λ]^2+ε),
        where Λ are the per-particle expected counts under the current PF posterior.
        If prefer_continuous is False and discrete particles are available, use the
        discrete states; otherwise fall back to continuous particles.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        fisher_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            use_continuous = bool(filt.continuous_particles)
            use_discrete = bool(filt.states)
            if use_continuous and (prefer_continuous or not use_discrete):
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
                w = filt.continuous_weights
            elif use_discrete:
                lam = np.zeros(filt.N, dtype=float)
                kvec = self.kernel_cache.kernel(iso, pose_idx, orient_idx)
                for i, st in enumerate(filt.states):
                    contrib = 0.0
                    for idx_src, strength in zip(st.source_indices, st.strengths):
                        contrib += kvec[idx_src] * strength
                    lam[i] = live_time_s * (contrib + st.background)
                w = np.exp(filt.log_weights)
                w = w / max(np.sum(w), eps)
            else:
                lam = np.zeros(0, dtype=float)
                w = np.zeros(0, dtype=float)
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
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
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
        num_samples = self.pf_config.eig_num_samples if num_samples is None else num_samples
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        kernel = ContinuousKernel(mu_by_isotope=self.mu_by_isotope, shield_params=self.shield_params)
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
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights = np.asarray(weights, dtype=float)
            weights = weights / max(np.sum(weights), eps)
            lam = np.zeros(len(states), dtype=float)
            for i, st in enumerate(states):
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
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]] | None = None,
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
        kernel = ContinuousKernel(mu_by_isotope=self.mu_by_isotope, shield_params=self.shield_params)
        JA_total = 0.0
        JD_total = 0.0
        for iso, filt in self.filters.items():
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights = np.asarray(weights, dtype=float)
            weights = weights / max(np.sum(weights), 1e-12)
            max_r = max(st.num_sources for st in states)
            dim = max_r + 1  # strengths + background
            I = np.zeros((dim, dim), dtype=float)
            for w, st in zip(weights, states):
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
                    g[j] = live_time_s * kernel.kernel_value_pair(
                        isotope=iso,
                        detector_pos=detector_pos,
                        source_pos=st.positions[j],
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                    )
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

    def expected_uncertainty_after_rotation(
        self,
        pose_idx: int,
        *,
        tau_ig: float,
        t_max_s: float,
        t_short_s: float,
        num_rollouts: int = 0,
        use_mean_measurement: bool = True,
        rng_seed: int | None = 0,
        return_debug: bool = False,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Estimate E[U_after-rotation | pose] for the full rotating-shield procedure.

        The rotation loop follows:
          - select the orientation with max IG (Eq. 3.48)
          - stop if ΔIG < tau_ig (Eq. 3.49)
          - simulate a short-time measurement and update the PF
          - stop if accumulated time >= t_max_s (Eq. 3.50)

        If num_rollouts == 0, uses a deterministic approximation that treats the
        mixture mean Λ̄ as the observation when use_mean_measurement is True.
        Otherwise, Monte Carlo rollouts are performed using mixture-Poisson sampling.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        alphas = self.pf_config.alpha_weights

        def _select_best_orientation(estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator) -> Tuple[int, int, float]:
            best_ig = -np.inf
            best_fe = 0
            best_pb = 0
            for fe_idx in range(num_fe):
                for pb_idx in range(num_pb):
                    ig_val = estimator.orientation_expected_information_gain(
                        pose_idx=pose_idx,
                        RFe=RFe_candidates[fe_idx],
                        RPb=RPb_candidates[pb_idx],
                        live_time_s=t_short_s,
                        alpha_by_isotope=alphas,
                        rng=rng_local,
                    )
                    if ig_val > best_ig:
                        best_ig = ig_val
                        best_fe = fe_idx
                        best_pb = pb_idx
            return best_fe, best_pb, float(best_ig)

        def _simulate_measurement(
            estimator: "RotatingShieldPFEstimator",
            fe_idx: int,
            pb_idx: int,
            rng_local: np.random.Generator,
        ) -> Dict[str, float]:
            z_k: Dict[str, float] = {}
            for iso, filt in estimator.filters.items():
                if not filt.continuous_particles:
                    z_k[iso] = 0.0
                    continue
                lam = filt._continuous_expected_counts_pair(
                    pose_idx=pose_idx,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=t_short_s,
                )
                if lam.size == 0:
                    z_k[iso] = 0.0
                    continue
                weights = filt.continuous_weights
                if num_rollouts == 0 and use_mean_measurement:
                    z_k[iso] = float(np.sum(weights * lam))
                else:
                    idx = int(rng_local.choice(len(lam), p=weights))
                    z_k[iso] = float(rng_local.poisson(lam[idx]))
            return z_k

        def _run_once(estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator) -> Tuple[float, Dict[str, Any]]:
            elapsed = 0.0
            rotations = 0
            iterations: List[Dict[str, Any]] = []
            while elapsed < t_max_s:
                fe_idx, pb_idx, ig_val = _select_best_orientation(estimator, rng_local)
                iterations.append(
                    {
                        "fe_idx": fe_idx,
                        "pb_idx": pb_idx,
                        "ig": ig_val,
                        "elapsed": elapsed,
                    }
                )
                if ig_val < tau_ig:
                    break
                z_k = _simulate_measurement(estimator, fe_idx, pb_idx, rng_local)
                estimator.update_pair(
                    z_k=z_k,
                    pose_idx=pose_idx,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=t_short_s,
                )
                elapsed += t_short_s
                rotations += 1
            return estimator.global_uncertainty(), {
                "iterations": iterations,
                "elapsed": elapsed,
                "num_rotations": rotations,
            }

        if num_rollouts <= 0:
            estimator_copy = copy.deepcopy(self)
            u_val, debug = _run_once(estimator_copy, rng)
            return (u_val, debug) if return_debug else u_val

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(num_rollouts):
            estimator_copy = copy.deepcopy(self)
            u_val, debug = _run_once(estimator_copy, rng)
            u_vals.append(u_val)
            debug_rollouts.append(debug)
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            return mean_u, {"rollouts": debug_rollouts, "u_vals": u_vals}
        return mean_u

    def estimate_change_norm(self) -> float:
        """
        Return ||Δs|| + ||Δq|| between the last two estimates (Sec. 3.6 convergence check).
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
        Return global uncertainty U = Σ_h Σ_j Var(q_{h,j}) (Sec. 3.6).
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
        Stop shield rotation when convergence criteria are met (Sec. 3.5–3.6).

        - max IG_k(φ) below threshold
        - estimate change ||Δs|| + ||Δq|| < change_tol
        - global uncertainty U below threshold
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
        Stop the overall exploration (Sec. 3.6) based on IG and uncertainty convergence.

        - Max IG at the last pose is small
        - Estimate change is small
        - Global uncertainty U is small
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

        For each isotope h and each candidate source, find the measurement index k* that
        maximises the ratio \\hat{Λ}_{k,h}/(z_{k,h}+ε) (Sec. 3.4.5). If the best-case ratio
        falls below τ_mix, the source is marked spurious and removed.
        """
        if any(filt.continuous_particles for filt in self.filters.values()):
            from pf.mixing import prune_spurious_sources_continuous

            keep_masks = prune_spurious_sources_continuous(self, tau_mix=tau_mix, epsilon=epsilon)
            for iso, filt in self.filters.items():
                if not filt.continuous_particles:
                    continue
                keep = keep_masks.get(iso)
                if keep is None or keep.size == 0:
                    continue
                for p in filt.continuous_particles:
                    r = p.state.num_sources
                    if r == 0:
                        continue
                    keep_idx = keep[:r] if keep.size >= r else np.pad(keep, (0, r - keep.size), constant_values=True)
                    p.state.positions = p.state.positions[keep_idx]
                    p.state.strengths = p.state.strengths[keep_idx]
                    p.state.num_sources = p.state.positions.shape[0]
            return keep_masks

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
