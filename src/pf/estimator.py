"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple, Any
import copy

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from pf.likelihood import expected_counts_per_source, poisson_log_likelihood
from pf.particle_filter import IsotopeParticleFilter, MeasurementData, PFConfig
from pf.resampling import systematic_resample
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
        - position_sigma: Gaussian jitter for positions (meters)
        - alpha_weights: isotope weights for IG criteria
        - death_low_q_streak: steps below min_strength before death is allowed
        - death_delta_ll_threshold: ΔLL threshold required to kill weak sources
        - support_ema_alpha: EMA weight for per-source ΔLL support
        - support_window: measurement window for per-source support scoring
        - birth_window: measurement window for residual-driven birth proposals
        - birth_softmax_temp: temperature for residual proposal sampling
        - birth_min_score: score floor for residual proposal sampling
        - split_prob: probability of split proposals per particle
        - split_strength_min: minimum strength for split candidates
        - split_position_sigma: position jitter for split proposals
        - split_strength_min_frac: min split fraction for q1/q2
        - split_strength_max_frac: max split fraction for q1/q2
        - split_delta_ll_threshold: ΔLL threshold for split acceptance
        - merge_prob: probability of merge proposals per particle
        - merge_distance_max: max distance for merge candidates
        - merge_delta_ll_threshold: ΔLL threshold for merge acceptance
        - init_num_sources: inclusive range for initial source count per particle
        - orientation_k: number of orientations to execute per pose
        - orientation_selection_mode: "eig"
        - planning_particles: particle count used for orientation scoring (None = all)
        - planning_method: how to select planning particles (top_weight/resample)
        - use_gpu: enable torch acceleration for continuous kernel evaluation
        - gpu_device: torch device string (e.g., "cuda" or "cpu")
        - gpu_dtype: torch dtype string ("float32" or "float64")
        - eig_num_samples: Monte-Carlo samples for EIG (Eq. 3.44)
        - planning_eig_samples: Monte-Carlo samples for EIG inside planning rollouts
        - planning_rollout_particles: particle cap for IG evaluation in rollouts
        - planning_rollout_method: selection method for rollout particles
        - preselect_*: optional surrogate stage settings for candidate reduction
        - use_fast_gpu_rollout: enable approximate fast GPU rollouts for uncertainty prediction
        - ig_workers: number of parallel workers for IG grid evaluation (0 = auto)
    """

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    ess_low: float = 0.5
    ess_high: float = 0.9
    max_sources: int | None = None
    resample_threshold: float = 0.5
    position_sigma: float = 0.1
    strength_sigma: float = 0.1
    background_sigma: float = 0.1
    background_level: float | dict[str, float] = 0.0
    min_strength: float = 0.01
    p_birth: float = 0.05
    p_kill: float = 0.1
    death_low_q_streak: int = 10
    death_delta_ll_threshold: float = 0.0
    support_ema_alpha: float = 0.3
    support_window: int = 1
    birth_window: int = 10
    birth_softmax_temp: float = 1.0
    birth_min_score: float = 1e-12
    split_prob: float = 0.05
    split_strength_min: float = 0.1
    split_position_sigma: float = 0.25
    split_strength_min_frac: float = 0.3
    split_strength_max_frac: float = 0.7
    split_delta_ll_threshold: float = 0.0
    merge_prob: float = 0.0
    merge_distance_max: float = 0.5
    merge_delta_ll_threshold: float = 0.0
    short_time_s: float = 0.5  # Recommended short-time measurement (Sec. 3.4.3).
    ig_threshold: float = 1e-3  # ΔIG stopping threshold (Sec. 3.4.4).
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    lambda_cost: float = 1.0  # Motion-cost weight (Eq. 3.51).
    alpha_weights: Dict[str, float] | None = None  # EIG isotope weights alpha_h.
    credible_volume_threshold: float = 1e-3  # Max 95% credible volume for convergence.
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (0, 3)
    orientation_k: int = 16
    orientation_selection_mode: str = "eig"
    planning_particles: int | None = None
    planning_method: str = "top_weight"
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    eig_num_samples: int = 50
    planning_eig_samples: int | None = None
    planning_rollout_particles: int | None = None
    planning_rollout_method: str | None = None
    preselect_orientations: bool = False
    preselect_metric: str = "var_log_lambda"
    preselect_delta: float = 0.05
    preselect_k_min: int = 8
    preselect_k_max: int = 16
    use_fast_gpu_rollout: bool = False
    ig_workers: int = 0

    def __post_init__(self) -> None:
        if self.min_particles is None:
            self.min_particles = max(1, int(self.num_particles * 0.5))
        if self.max_particles is None:
            self.max_particles = max(self.num_particles, int(self.num_particles * 2.0))
        self.ess_low = float(self.ess_low)
        self.ess_high = float(self.ess_high)
        if not 0.0 < self.ess_low < self.ess_high < 1.0:
            raise ValueError("ess_low and ess_high must satisfy 0 < ess_low < ess_high < 1.")
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")


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
        mu_by_isotope: Dict[str, object] | None,
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
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: KernelPrecomputer | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]] = []
        self.history_scores: List[float] = []
        self.measurements: List[MeasurementRecord] = []

    def _resolve_mu_by_isotope(self, mu_by_isotope: Dict[str, object] | None) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9]", "", name).upper()

        canonical_by_norm = {
            "CS137": "Cs-137",
            "CO60": "Co-60",
            "EU154": "Eu-154",
        }

        resolved: Dict[str, object] = {}
        if mu_by_isotope is not None:
            resolved.update(mu_by_isotope)
        normalized: Dict[str, object] = {}
        for key, value in resolved.items():
            normalized[_norm_key(key)] = value
        if self.isotopes:
            still_missing: List[str] = []
            for iso in self.isotopes:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=[canonical])
                    if canonical in table_vals:
                        resolved[iso] = table_vals[canonical]
                        normalized[norm] = table_vals[canonical]
                        if canonical not in resolved:
                            resolved[canonical] = table_vals[canonical]
                        continue
                still_missing.append(iso)
            if still_missing:
                missing_list = ", ".join(still_missing)
                raise ValueError(
                    "mu_by_isotope is missing entries for isotopes: "
                    f"{missing_list}. Ensure isotope names match the HVL/TVL table keys."
                )
        return resolved

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
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
        )
        pf_conf = self._build_pf_config()
        if self.filters:
            for iso in self.isotopes:
                if iso in self.filters:
                    self.filters[iso].set_kernel(self.kernel_cache)
                else:
                    self.filters[iso] = IsotopeParticleFilter(iso, kernel=self.kernel_cache, config=pf_conf)
        else:
            for iso in self.isotopes:
                self.filters[iso] = IsotopeParticleFilter(iso, kernel=self.kernel_cache, config=pf_conf)

    def _build_pf_config(self) -> PFConfig:
        """Build a per-isotope PFConfig from the estimator configuration."""
        return PFConfig(
            num_particles=self.pf_config.num_particles,
            min_particles=self.pf_config.min_particles,
            max_particles=self.pf_config.max_particles,
            ess_low=self.pf_config.ess_low,
            ess_high=self.pf_config.ess_high,
            max_sources=self.pf_config.max_sources,
            resample_threshold=self.pf_config.resample_threshold,
            position_sigma=self.pf_config.position_sigma,
            strength_sigma=self.pf_config.strength_sigma,
            background_sigma=self.pf_config.background_sigma,
            background_level=self.pf_config.background_level,
            min_strength=self.pf_config.min_strength,
            p_birth=self.pf_config.p_birth,
            p_kill=self.pf_config.p_kill,
            death_low_q_streak=self.pf_config.death_low_q_streak,
            death_delta_ll_threshold=self.pf_config.death_delta_ll_threshold,
            support_ema_alpha=self.pf_config.support_ema_alpha,
            support_window=self.pf_config.support_window,
            birth_window=self.pf_config.birth_window,
            birth_softmax_temp=self.pf_config.birth_softmax_temp,
            birth_min_score=self.pf_config.birth_min_score,
            split_prob=self.pf_config.split_prob,
            split_strength_min=self.pf_config.split_strength_min,
            split_position_sigma=self.pf_config.split_position_sigma,
            split_strength_min_frac=self.pf_config.split_strength_min_frac,
            split_strength_max_frac=self.pf_config.split_strength_max_frac,
            split_delta_ll_threshold=self.pf_config.split_delta_ll_threshold,
            merge_prob=self.pf_config.merge_prob,
            merge_distance_max=self.pf_config.merge_distance_max,
            merge_delta_ll_threshold=self.pf_config.merge_delta_ll_threshold,
            position_min=self.pf_config.position_min,
            position_max=self.pf_config.position_max,
            init_num_sources=self.pf_config.init_num_sources,
            use_gpu=self.pf_config.use_gpu,
            gpu_device=self.pf_config.gpu_device,
            gpu_dtype=self.pf_config.gpu_dtype,
        )

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in RotatingShieldPFConfig.")
        if not gpu_utils.torch_available():
            raise RuntimeError("GPU-only mode requires CUDA-enabled torch.")
        return True

    def expected_counts_pair_for_states(
        self,
        isotope: str,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ_{k,h}^{(n)} for an isotope over a list of states at a pose.

        Uses torch acceleration when enabled; otherwise falls back to CPU kernels.
        """
        if not states:
            return np.zeros(0, dtype=float)
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        kernel = ContinuousKernel(mu_by_isotope=self.mu_by_isotope, shield_params=self.shield_params)
        detector_pos = np.asarray(self.poses[pose_idx], dtype=float)
        self._gpu_enabled()
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
        mu_fe, mu_pb = kernel._mu_values(isotope=isotope)
        shield_params = kernel.shield_params
        lam_t = gpu_utils.expected_counts_pair_torch(
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            mu_fe=mu_fe,
            mu_pb=mu_pb,
            thickness_fe_cm=shield_params.thickness_fe_cm,
            thickness_pb_cm=shield_params.thickness_pb_cm,
            use_angle_attenuation=shield_params.use_angle_attenuation,
            live_time_s=live_time_s,
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy()

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
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
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

    def add_measurement_pose(self, pose: NDArray[np.float64], reset_filters: bool = True) -> None:
        """Register a new measurement pose and invalidate the kernel cache."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}

    def restrict_isotopes(self, active_isotopes: Sequence[str]) -> None:
        """
        Restrict estimator state to the specified isotopes.

        This drops filters and cached estimates for isotopes that are not in
        active_isotopes while preserving the original isotope ordering.
        """
        active_set = set(active_isotopes)
        if not active_set:
            raise ValueError("active_isotopes must contain at least one isotope.")
        self.isotopes = [iso for iso in self.isotopes if iso in active_set]
        if self.filters:
            self.filters = {iso: filt for iso, filt in self.filters.items() if iso in active_set}
        if self.history_estimates:
            self.history_estimates = [
                {iso: val for iso, val in est.items() if iso in active_set} for est in self.history_estimates
            ]

    def add_isotopes(self, new_isotopes: Sequence[str]) -> None:
        """
        Add isotopes to the estimator and initialize their PF filters.

        This is useful when new isotopes are detected after an initial restriction.
        """
        to_add = [iso for iso in new_isotopes if iso not in self.isotopes]
        if not to_add:
            return
        self.isotopes.extend(to_add)
        if self.kernel_cache is None:
            return
        pf_conf = self._build_pf_config()
        for iso in to_add:
            if iso not in self.filters:
                self.filters[iso] = IsotopeParticleFilter(iso, kernel=self.kernel_cache, config=pf_conf)

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
        raise RuntimeError(
            "Single-orientation updates are disabled. Use update_pair or short_time_update "
            "with Fe/Pb indices to preserve the 64-orientation shield model."
        )

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
        self._apply_birth_death()

    def _measurement_data_for_iso(
        self,
        isotope: str,
        window: int | None,
    ) -> MeasurementData | None:
        """Build measurement arrays for a single isotope with an optional window."""
        if not self.measurements:
            return None
        if window is None or window <= 0:
            records = self.measurements
        else:
            records = self.measurements[-int(window) :]
        if not records:
            return None
        z_list = []
        poses = []
        fe_indices = []
        pb_indices = []
        live_times = []
        for rec in records:
            z_list.append(float(rec.z_k.get(isotope, 0.0)))
            poses.append(self.poses[rec.pose_idx])
            live_times.append(float(rec.live_time_s))
            if rec.fe_index is not None and rec.pb_index is not None:
                fe_indices.append(int(rec.fe_index))
                pb_indices.append(int(rec.pb_index))
            else:
                fe_indices.append(int(rec.orient_idx))
                pb_indices.append(int(rec.orient_idx))
        return MeasurementData(
            z_k=np.asarray(z_list, dtype=float),
            detector_positions=np.asarray(poses, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            live_times=np.asarray(live_times, dtype=float),
        )

    def _apply_birth_death(self) -> None:
        """Apply per-isotope birth/death updates using recent measurements."""
        for iso, filt in self.filters.items():
            support_data = self._measurement_data_for_iso(iso, self.pf_config.support_window)
            birth_data = self._measurement_data_for_iso(iso, self.pf_config.birth_window)
            filt.apply_birth_death(
                support_data=support_data,
                birth_data=birth_data,
                candidate_positions=self.candidate_sources,
            )

    def estimates(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return per-isotope position/strength estimates (MMSE over continuous particles)."""
        return {iso: f.estimate() for iso, f in self.filters.items()}

    def estimate_all(self) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def step_diagnostics(self, top_k: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/kill counts, r distribution
        (mean/variance), MAP/MMSE estimates, and top-k particle summaries.
        """
        diagnostics: Dict[str, Dict[str, Any]] = {}
        eps = 1e-12
        k = max(0, int(top_k))
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                diagnostics[iso] = {
                    "ess_pre": 0.0,
                    "resampled": False,
                    "ess_post": None,
                    "n_after_adapt": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "kill_count": int(getattr(filt, "last_kill_count", 0)),
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "mmse": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "top_k": [],
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                weights = weights / total
            r_vals = np.array([p.state.num_sources for p in filt.continuous_particles], dtype=float)
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
            ess_pre = getattr(filt, "last_ess_pre", None)
            if ess_pre is None and weights.size:
                ess_pre = float(1.0 / max(np.sum(weights**2), eps))
            if ess_pre is None:
                ess_pre = 0.0
            resampled = bool(getattr(filt, "last_resample_ess", False))
            ess_post = getattr(filt, "last_ess_post", None)
            n_after_adapt = getattr(filt, "last_n_after_adapt", None)
            if n_after_adapt is None:
                n_after_adapt = int(len(filt.continuous_particles))
            best_state = filt.best_particle().state
            map_positions = best_state.positions.copy()
            map_strengths = best_state.strengths.copy()
            try:
                mmse_positions, mmse_strengths = filt.estimate()
            except RuntimeError:
                mmse_positions = np.zeros((0, 3), dtype=float)
                mmse_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.continuous_particles[int(idx)].state
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": int(state.num_sources),
                            "positions": state.positions.copy(),
                            "strengths": state.strengths.copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "n_after_adapt": int(n_after_adapt),
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "kill_count": int(getattr(filt, "last_kill_count", 0)),
                "r_mean": r_mean,
                "r_var": r_var,
                "map": (map_positions, map_strengths),
                "mmse": (mmse_positions, mmse_strengths),
                "top_k": top_entries,
            }
        return diagnostics

    def isotope_log_likelihood_gain(self, window: int | None = None) -> Dict[str, float]:
        """
        Return per-isotope log-likelihood gain vs background-only (evidence mixing).
        """
        if not self.measurements:
            return {iso: 0.0 for iso in self.filters}
        estimates = self.pruned_estimates(method="legacy")
        gains: Dict[str, float] = {}
        for iso, filt in self.filters.items():
            data = self._measurement_data_for_iso(iso, window)
            if data is None or data.z_k.size == 0:
                gains[iso] = 0.0
                continue
            positions, strengths = estimates.get(iso, (np.zeros((0, 3)), np.zeros(0)))
            if filt.continuous_particles:
                background_rate = float(filt.best_particle().state.background)
            else:
                background_rate = 0.0
            background_counts = background_rate * data.live_times
            if positions.size == 0:
                gains[iso] = 0.0
                continue
            lambda_m = expected_counts_per_source(
                kernel=filt.continuous_kernel,
                isotope=iso,
                detector_positions=data.detector_positions,
                sources=positions,
                strengths=strengths,
                live_times=data.live_times,
                fe_indices=data.fe_indices,
                pb_indices=data.pb_indices,
            )
            lambda_total = background_counts + np.sum(lambda_m, axis=1)
            ll = poisson_log_likelihood(data.z_k, lambda_total)
            ll_bg = poisson_log_likelihood(data.z_k, background_counts)
            gains[iso] = float(ll - ll_bg)
        return gains

    def isotopes_by_evidence(self, min_delta_ll: float = 0.0, window: int | None = None) -> List[str]:
        """
        Return isotopes whose LL gain exceeds min_delta_ll for the given window.
        """
        gains = self.isotope_log_likelihood_gain(window=window)
        return [iso for iso, gain in gains.items() if gain >= float(min_delta_ll)]

    @property
    def num_orientations(self) -> int:
        return self.normals.shape[0]

    def orientation_information_gain(self, pose_idx: int, orient_idx: int, live_time_s: float = 1.0) -> float:
        """
        Information gain surrogate using Eq. (3.40)–(3.42) style variance ratio.

        IG_k(phi) ~= 0.5 * log(1 + Var[Lambda_k(phi)] / E[Lambda_k(phi)]) aggregated over isotopes.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        ig_total = 0.0
        eps = 1e-9
        for iso, filt in self.filters.items():
            use_continuous = bool(filt.continuous_particles)
            if use_continuous:
                lam = filt._continuous_expected_counts(
                    pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
                )
                w = filt.continuous_weights
            else:
                lam = np.zeros(0, dtype=float)
                w = np.zeros(0, dtype=float)
            mean = float(np.sum(w * lam))
            var = float(np.sum(w * (lam - mean) ** 2))
            ig_total += 0.5 * float(np.log1p(var / max(mean, eps)))
        return ig_total

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
        detector_pos: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Monte-Carlo approximation of EIG (Eq. 3.44) for a Fe/Pb orientation pair.

        - Uses continuous particles and ContinuousKernel expected counts (Eq. 3.41).
        - For each isotope h: IG_h = H(w_h) - E_z[H(w'_h(z; RFe, RPb))].
        - Global IG = Σ_h α_h IG_h, with α_h uniform if not provided.
        - If detector_pos is provided, pose_idx is ignored.
        """
        if detector_pos is None:
            if self.kernel_cache is None:
                self._ensure_kernel_cache()
            detector_pos = self.kernel_cache.poses[pose_idx]
        detector_pos = np.asarray(detector_pos, dtype=float)
        rng = rng or np.random.default_rng()
        num_samples = self.pf_config.eig_num_samples if num_samples is None else num_samples
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        kernel = ContinuousKernel(mu_by_isotope=self.mu_by_isotope, shield_params=self.shield_params)
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        # normalize alphas
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils as gpu_mod
        import torch as torch_mod

        gpu_utils = gpu_mod
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        torch = torch_mod

        def _compute_lam_torch(states: Sequence[IsotopeState], isotope: str) -> "torch.Tensor":
            if not states:
                return torch.zeros(0, device=device, dtype=dtype)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
            mu_fe, mu_pb = kernel._mu_values(isotope=isotope)
            shield_params = kernel.shield_params
            return gpu_utils.expected_counts_pair_torch(
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                mu_fe=mu_fe,
                mu_pb=mu_pb,
                thickness_fe_cm=shield_params.thickness_fe_cm,
                thickness_pb_cm=shield_params.thickness_pb_cm,
                use_angle_attenuation=shield_params.use_angle_attenuation,
                live_time_s=live_time_s,
                device=device,
                dtype=dtype,
            )

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
            lam_t = _compute_lam_torch(states, iso)
            weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
            weight_sum = torch.sum(weights_t)
            if float(weight_sum) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum
            H_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
            if num_samples <= 0:
                H_post_mean = torch.zeros((), device=device, dtype=dtype)
            else:
                idx = torch.multinomial(weights_t, num_samples, replacement=True)
                z = torch.poisson(lam_t[idx])
                logw = torch.log(weights_t + eps) + z.unsqueeze(1) * torch.log(lam_t + eps) - lam_t
                logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
                w_post = torch.exp(logw)
                H_post = -torch.sum(w_post * torch.log(w_post + eps), dim=1)
                H_post_mean = torch.mean(H_post)
            ig_h = float((H_prior - H_post_mean).item())
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)


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
        pose_xyz: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        n_rollouts: int = 64,
        orient_selection: str = "IG",
        return_debug: bool = False,
        rng_seed: int | None = None,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Estimate E[U_after-rotation | pose_xyz] by Monte Carlo rollouts.

        This method has no side effects on the estimator state. Rotation policy:
        - choose the next orientation by maximizing IG
        - stop if max IG < tau_ig
        - stop if accumulated live time reaches tmax_s

        rng_seed can be set to make rollouts deterministic for debugging.
        """
        detector_pos = np.asarray(pose_xyz, dtype=float)
        if detector_pos.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        if orient_selection.lower() != "ig":
            raise ValueError("Only orient_selection='IG' is supported.")
        n_rollouts = int(n_rollouts)
        use_mean_measurement = n_rollouts <= 0
        rollouts = max(1, n_rollouts)
        if rng_seed is None:
            rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        else:
            rng = np.random.default_rng(int(rng_seed))
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        alphas = self.pf_config.alpha_weights
        eig_samples = (
            self.pf_config.planning_eig_samples
            if self.pf_config.planning_eig_samples is not None
            else self.pf_config.eig_num_samples
        )
        rollout_particles = self.pf_config.planning_rollout_particles
        if rollout_particles is None:
            rollout_particles = self.pf_config.planning_particles
        rollout_method = self.pf_config.planning_rollout_method or self.pf_config.planning_method

        fast_result = self._expected_uncertainty_after_rotation_fast(
            detector_pos=detector_pos,
            live_time_per_rot_s=live_time_per_rot_s,
            tau_ig=tau_ig,
            tmax_s=tmax_s,
            rollouts=rollouts,
            eig_samples=eig_samples,
            alpha_by_isotope=alphas,
            rollout_particles=rollout_particles,
            rollout_method=rollout_method,
            use_mean_measurement=use_mean_measurement,
            rng=rng,
            return_debug=return_debug,
        )
        if fast_result is not None:
            return fast_result

        def _select_best_orientation(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[int, int, float]:
            """Return the (fe_idx, pb_idx) pair with the maximum EIG at the given pose."""
            best_ig = -np.inf
            best_fe = 0
            best_pb = 0
            particles_by_iso = None
            if rollout_particles is not None and rollout_particles > 0:
                particles_by_iso = estimator.planning_particles(
                    max_particles=int(rollout_particles),
                    method=rollout_method,
                    rng=rng_local,
                )
            for fe_idx in range(num_fe):
                for pb_idx in range(num_pb):
                    ig_val = estimator.orientation_expected_information_gain(
                        pose_idx=0,
                        RFe=RFe_candidates[fe_idx],
                        RPb=RPb_candidates[pb_idx],
                        live_time_s=live_time_per_rot_s,
                        num_samples=eig_samples,
                        alpha_by_isotope=alphas,
                        particles_by_isotope=particles_by_iso,
                        rng=rng_local,
                        detector_pos=detector_pos,
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
            """Simulate isotope-wise Poisson observations at the candidate pose."""
            z_k: Dict[str, float] = {}
            for iso, filt in estimator.filters.items():
                if not filt.continuous_particles:
                    z_k[iso] = 0.0
                    continue
                lam = filt._continuous_expected_counts_pair_at_pose(
                    detector_pos=detector_pos,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    live_time_s=live_time_per_rot_s,
                )
                if lam.size == 0:
                    z_k[iso] = 0.0
                    continue
                weights = filt.continuous_weights
                if use_mean_measurement:
                    z_k[iso] = float(np.sum(weights * lam))
                else:
                    idx = int(rng_local.choice(len(lam), p=weights))
                    z_k[iso] = float(rng_local.poisson(lam[idx]))
            return z_k

        def _run_once(
            estimator: "RotatingShieldPFEstimator", rng_local: np.random.Generator
        ) -> Tuple[float, Dict[str, Any]]:
            """Run a single rotation rollout and return uncertainty plus debug metadata."""
            elapsed = 0.0
            rotations = 0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
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
                for iso, val in z_k.items():
                    if iso not in estimator.filters:
                        continue
                    estimator.filters[iso].update_continuous_pair_at_pose(
                        z_obs=val,
                        detector_pos=detector_pos,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=live_time_per_rot_s,
                    )
                elapsed += live_time_per_rot_s
                rotations += 1
            return estimator.global_uncertainty(), {
                "iterations": iterations,
                "elapsed": elapsed,
                "num_rotations": rotations,
            }

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(rollouts):
            estimator_copy = copy.deepcopy(self)
            u_val, debug = _run_once(estimator_copy, rng)
            u_vals.append(u_val)
            debug_rollouts.append(debug)
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def _expected_uncertainty_after_rotation_fast(
        self,
        detector_pos: NDArray[np.float64],
        live_time_per_rot_s: float,
        tau_ig: float,
        tmax_s: float,
        rollouts: int,
        eig_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        rollout_particles: int | None,
        rollout_method: str | None,
        use_mean_measurement: bool,
        rng: np.random.Generator,
        return_debug: bool,
    ) -> float | Tuple[float, Dict[str, Any]] | None:
        """
        Fast GPU rollout evaluation using precomputed lambdas and index-based updates.

        Returns None when the fast path cannot be used.
        """
        if not self.pf_config.use_fast_gpu_rollout:
            return None
        self._gpu_enabled()
        from pf import gpu_utils
        import torch
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        num_orients = num_fe * num_pb
        fe_indices = np.repeat(np.arange(num_fe), num_pb)
        pb_indices = np.tile(np.arange(num_pb), num_fe)
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)

        iso_data: Dict[str, Dict[str, Any]] = {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            if weights.size == 0:
                continue
            weights = weights / max(np.sum(weights), eps)
            states = [p.state for p in filt.continuous_particles]
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            mu_fe, mu_pb = filt.continuous_kernel._mu_values(isotope=iso)
            shield_params = filt.continuous_kernel.shield_params
            lam_list = []
            for fe_idx, pb_idx in zip(fe_indices, pb_indices):
                lam_t = gpu_utils.expected_counts_pair_torch(
                    detector_pos=detector_pos,
                    positions=positions,
                    strengths=strengths,
                    backgrounds=backgrounds,
                    mask=mask,
                    fe_index=int(fe_idx),
                    pb_index=int(pb_idx),
                    mu_fe=mu_fe,
                    mu_pb=mu_pb,
                    thickness_fe_cm=shield_params.thickness_fe_cm,
                    thickness_pb_cm=shield_params.thickness_pb_cm,
                    use_angle_attenuation=shield_params.use_angle_attenuation,
                    live_time_s=live_time_per_rot_s,
                    device=device,
                    dtype=dtype,
                )
                lam_list.append(lam_t)
            if not lam_list:
                continue
            lam_all = torch.stack(lam_list, dim=0)
            iso_data[iso] = {
                "lam": lam_all,
                "strengths": strengths,
                "weights": weights,
                "num_particles": weights.size,
                "resample_threshold": filt.config.resample_threshold,
            }
        if not iso_data:
            return 0.0 if not return_debug else (0.0, {"rollouts": [], "u_vals": []})

        def _select_subset(
            weights: NDArray[np.float64],
            indices: NDArray[np.int64],
            max_particles: int | None,
            method: str | None,
            rng_local: np.random.Generator,
        ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
            """Return subset indices and normalized weights for EIG evaluation."""
            if max_particles is None or max_particles <= 0 or max_particles >= len(weights):
                return indices, weights
            method = method or "top_weight"
            if method == "top_weight":
                sel = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[sel]
                sel_weights = sel_weights / max(np.sum(sel_weights), eps)
                return indices[sel], sel_weights
            if method == "resample":
                sel = rng_local.choice(len(weights), size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max(max_particles, 1)
                return indices[sel], sel_weights
            raise ValueError(f"Unknown planning particle selection method: {method}")

        def _ig_scores_from_lam(
            lam_all: "torch.Tensor",
            subset_indices: NDArray[np.int64],
            subset_weights: NDArray[np.float64],
            num_samples: int,
        ) -> "torch.Tensor":
            """Compute IG scores for all orientations from precomputed lambdas."""
            if num_samples <= 0:
                weights_t = torch.as_tensor(subset_weights, device=lam_all.device, dtype=lam_all.dtype)
                weights_t = weights_t / torch.sum(weights_t)
                h_prior = -torch.sum(weights_t * torch.log(weights_t + eps))
                return torch.full((lam_all.shape[0],), h_prior, device=lam_all.device, dtype=lam_all.dtype)
            idx_t = torch.as_tensor(subset_indices, device=lam_all.device, dtype=torch.long)
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(subset_weights, device=lam_all.device, dtype=lam_all.dtype)
            weights_t = weights_t / torch.sum(weights_t)
            log_weights = torch.log(weights_t + eps)
            h_prior = -torch.sum(weights_t * log_weights)
            weights_row = weights_t.expand(lam_sel.shape[0], -1)
            idx_samples = torch.multinomial(weights_row, num_samples, replacement=True)
            lam_samples = torch.gather(lam_sel, 1, idx_samples)
            z = torch.poisson(lam_samples)
            log_lam = torch.log(lam_sel + eps)
            logw = log_weights.view(1, 1, -1) + z.unsqueeze(2) * log_lam.unsqueeze(1) - lam_sel.unsqueeze(1)
            logw = logw - torch.logsumexp(logw, dim=2, keepdim=True)
            w_post = torch.exp(logw)
            h_post = -torch.sum(w_post * torch.log(w_post + eps), dim=2)
            h_post_mean = torch.mean(h_post, dim=1)
            return h_prior - h_post_mean

        def _update_weights(
            lam_curr: NDArray[np.float64],
            weights: NDArray[np.float64],
            z_obs: float,
        ) -> NDArray[np.float64]:
            """Update weights using Poisson log-likelihood and normalize."""
            logw = np.log(weights + eps) + z_obs * np.log(lam_curr + eps) - lam_curr
            logw -= np.max(logw)
            w = np.exp(logw)
            total = np.sum(w)
            if total <= 0.0:
                return np.ones_like(weights) / max(len(weights), 1)
            return w / total

        u_vals: List[float] = []
        debug_rollouts: List[Dict[str, Any]] = []
        for _ in range(int(rollouts)):
            weights_by_iso: Dict[str, NDArray[np.float64]] = {}
            indices_by_iso: Dict[str, NDArray[np.int64]] = {}
            for iso, data in iso_data.items():
                n_particles = int(data["num_particles"])
                weights_by_iso[iso] = data["weights"].copy()
                indices_by_iso[iso] = np.arange(n_particles, dtype=int)
            elapsed = 0.0
            iterations: List[Dict[str, Any]] = []
            while elapsed < tmax_s:
                total_ig: "torch.Tensor" | None = None
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    subset_idx, subset_w = _select_subset(
                        weights=weights,
                        indices=indices,
                        max_particles=rollout_particles,
                        method=rollout_method,
                        rng_local=rng,
                    )
                    if subset_w.size == 0:
                        continue
                    ig_scores = _ig_scores_from_lam(
                        lam_all=data["lam"],
                        subset_indices=subset_idx,
                        subset_weights=subset_w,
                        num_samples=int(eig_samples),
                    )
                    weight = float(alphas.get(iso, 0.0))
                    ig_scores = ig_scores * weight
                    if total_ig is None:
                        total_ig = ig_scores
                    else:
                        total_ig = total_ig + ig_scores
                if total_ig is None:
                    break
                best_orient = int(torch.argmax(total_ig).item())
                best_ig = float(total_ig[best_orient].detach().cpu().item())
                iterations.append(
                    {
                        "fe_idx": int(fe_indices[best_orient]),
                        "pb_idx": int(pb_indices[best_orient]),
                        "ig": best_ig,
                        "elapsed": elapsed,
                    }
                )
                if best_ig < tau_ig:
                    break
                for iso, data in iso_data.items():
                    weights = weights_by_iso[iso]
                    indices = indices_by_iso[iso]
                    if weights.size == 0:
                        continue
                    idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                    lam_curr_t = torch.index_select(data["lam"][best_orient], 0, idx_t)
                    lam_curr = lam_curr_t.detach().cpu().numpy()
                    if lam_curr.size == 0:
                        continue
                    if use_mean_measurement:
                        z_obs = float(np.sum(weights * lam_curr))
                    else:
                        idx = int(rng.choice(len(lam_curr), p=weights))
                        z_obs = float(rng.poisson(lam_curr[idx]))
                    weights = _update_weights(lam_curr, weights, z_obs)
                    ess = 1.0 / max(np.sum(weights**2), eps)
                    if ess < float(data["resample_threshold"]) * len(weights):
                        resampled = systematic_resample(np.log(weights + eps))
                        indices = indices[resampled]
                        weights = np.ones_like(weights) / max(len(weights), 1)
                    weights_by_iso[iso] = weights
                    indices_by_iso[iso] = indices
                elapsed += live_time_per_rot_s
            total_u = 0.0
            for iso, data in iso_data.items():
                weights = weights_by_iso[iso]
                indices = indices_by_iso[iso]
                if weights.size == 0:
                    continue
                idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
                strengths_t = torch.index_select(data["strengths"], 0, idx_t)
                weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
                weights_t = weights_t / torch.sum(weights_t)
                mean = torch.sum(weights_t[:, None] * strengths_t, dim=0)
                var = torch.sum(weights_t[:, None] * (strengths_t - mean) ** 2, dim=0)
                total_u += float(torch.sum(var).detach().cpu().item())
            u_vals.append(total_u)
            debug_rollouts.append(
                {
                    "iterations": iterations,
                    "elapsed": elapsed,
                    "num_rotations": len(iterations),
                }
            )
        mean_u = float(np.mean(u_vals)) if u_vals else 0.0
        if return_debug:
            debug_payload = {"rollouts": debug_rollouts, "u_vals": u_vals}
            return mean_u, debug_payload
        return mean_u

    def expected_uncertainty_after_rotation_at_pose(
        self,
        detector_pos: NDArray[np.float64],
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
        Backward-compatible wrapper for expected_uncertainty_after_rotation.
        """
        n_rollouts = int(num_rollouts)
        if n_rollouts <= 0 and not use_mean_measurement:
            n_rollouts = 1
        if rng_seed is not None:
            np.random.seed(rng_seed)
        return self.expected_uncertainty_after_rotation(
            pose_xyz=detector_pos,
            live_time_per_rot_s=t_short_s,
            tau_ig=tau_ig,
            tmax_s=t_max_s,
            n_rollouts=n_rollouts,
            orient_selection="IG",
            return_debug=return_debug,
            rng_seed=rng_seed,
        )

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
            if not filt.continuous_particles:
                continue
            self._gpu_enabled()
            from pf import gpu_utils
            import torch

            device = gpu_utils.resolve_device(self.pf_config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
            states = [p.state for p in filt.continuous_particles]
            _, strengths_t, _, _ = gpu_utils.pack_states(states, device=device, dtype=dtype)
            weights = torch.as_tensor(filt.continuous_weights, device=device, dtype=dtype)
            weight_sum = torch.sum(weights)
            if float(weight_sum) <= 0.0:
                weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
            else:
                weights = weights / weight_sum
            mean = torch.sum(weights[:, None] * strengths_t, dim=0)
            var = torch.sum(weights[:, None] * (strengths_t - mean) ** 2, dim=0)
            total += float(torch.sum(var).detach().cpu().item())
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
        for oidx in range(self.num_orientations):
            ig_scores.append(
                self.orientation_information_gain(pose_idx=pose_idx, orient_idx=oidx, live_time_s=live_time_s)
            )
        max_ig = max(ig_scores) if ig_scores else 0.0
        dwell_time = sum(rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx)
        # Credible region volumes check (Sec. 3.5)
        volumes = self.credible_region_volumes()
        max_volume = 0.0
        for vols in volumes.values():
            if vols:
                max_volume = max(max_volume, max(vols))
        return (
            (max_ig < ig_threshold)
            and (self.estimate_change_norm() < change_tol)
            and (self.global_uncertainty() < uncertainty_tol)
            and (max_volume < self.pf_config.credible_volume_threshold)
            or (dwell_time >= self.pf_config.max_dwell_time_s)
        )

    def should_stop_exploration(
        self,
        ig_threshold: float = 5e-4,
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
            change_tol=change_tol,
            uncertainty_tol=uncertainty_tol,
            live_time_s=live_time_s,
        )

    def prune_spurious_sources(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, NDArray[np.bool_]]:
        """
        Compute spurious-source keep masks using delta-LL, best-case residual gating, or legacy dominance.

        For each isotope h and each candidate source, the pruning method is applied using
        per-measurement expected counts from the continuous kernel. Optionally drop sources
        below max(min_strength_abs, min_strength_ratio * max_strength).

        Method params (passed via params):
        - deltaLL: deltaLL_min, penalty_d (BIC-style), epsilon
        - bestcase: alpha, lambda_min, lrt_threshold, epsilon
        - legacy: tau_mix, epsilon
        """
        from pf.mixing import prune_spurious_sources_continuous

        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
        )
        return keep_masks

    def pruned_estimates(
        self,
        method: str = "legacy",
        params: Dict[str, float] | None = None,
        tau_mix: float = 0.9,
        epsilon: float = 1e-6,
        min_support: int = 1,
        min_obs_count: float = 0.0,
        min_strength_abs: float | None = None,
        min_strength_ratio: float | None = None,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Return non-destructively pruned estimates derived from MMSE outputs.

        This uses prune_spurious_sources_continuous() in estimate space and does not
        mutate particle states.
        """
        from pf.mixing import prune_spurious_sources_continuous

        est = self.estimates()
        keep_masks = prune_spurious_sources_continuous(
            self,
            method=method,
            params=params,
            tau_mix=tau_mix,
            epsilon=epsilon,
            min_support=min_support,
            min_obs_count=min_obs_count,
            min_strength_abs=min_strength_abs,
            min_strength_ratio=min_strength_ratio,
        )
        pruned: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for iso, (pos, strg) in est.items():
            keep = keep_masks.get(iso)
            if keep is None or keep.size == 0:
                pruned[iso] = (pos, strg)
            else:
                pruned[iso] = (pos[keep], strg[keep])
        return pruned
