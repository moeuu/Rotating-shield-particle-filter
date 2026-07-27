"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
import hashlib
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence, Tuple
import copy
import os

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2

from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.model import EnvironmentConfig
from measurement.shielding import octant_index_from_rotation
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kinds
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.likelihood import (
    CountLikelihoodSpec,
    OBSERVATION_COUNT_VARIANCE_ADDITIONAL,
    OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL,
    count_log_likelihood_terms_np,
    count_log_likelihood_terms_torch,
    expected_counts_per_source,
    normalize_count_likelihood_model,
    predictive_count_likelihood_variance,
    predictive_count_likelihood_variance_torch,
    normalize_observation_count_variance_semantics,
)
from pf.particle_filter import IsotopeParticleFilter, MeasurementData, PFConfig
from pf.posterior import posterior_point_estimate_from_states
from pf.posterior_uncertainty import posterior_mode_uncertainty_batched
from pf.reporting import measurement_vector
from pf.resampling import systematic_resample
from pf.runtime_route import (
    COUNT_COVARIANCE_LIKELIHOOD_ROUTE,
    COUNT_LIKELIHOOD_ROUTE,
    canonical_runtime_likelihood_route_mapping,
)
from pf.state import IsotopeState

if TYPE_CHECKING:
    import torch


def _weighted_quantile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    quantile: float,
) -> float:
    """Return a weighted quantile for non-negative planning statistics."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        raise ValueError("weights must have the same size as values.")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(finite):
        return 0.0
    values = values[finite]
    weights = weights[finite]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.quantile(values, np.clip(float(quantile), 0.0, 1.0)))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / total
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, np.clip(float(quantile), 0.0, 1.0), side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


@dataclass
class RotatingShieldPFConfig:
    """Configure the exact finite-surface PF and its active planner."""

    estimator_profile: str = "pf_strict"
    num_particles: int = 200
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: Dict[str, float] | None = None
    measurement_scale_by_isotope_and_pair: Dict[str, Dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | Dict[str, float] = 0.0
    transport_model_abs_sigma: float | Dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | Dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_abs_sigma: float | Dict[str, float] = 0.0
    low_count_transition_counts: float | Dict[str, float] = 0.0
    observation_count_variance_semantics: str = (
        OBSERVATION_COUNT_VARIANCE_ADDITIONAL
    )
    count_likelihood_df: float = 5.0
    shield_contrast_likelihood_enable: bool = False
    shield_contrast_likelihood_weight: float = 1.0
    shield_contrast_log_sigma_floor: float = 0.5
    shield_contrast_log_sigma_ceiling: float = 2.0
    shield_contrast_min_count: float = 25.0
    shield_contrast_min_views: int = 2
    shield_contrast_likelihood_df: float = 5.0
    shield_view_ratio_likelihood_enable: bool = False
    shield_view_ratio_likelihood_weight: float = 1.0
    shield_view_ratio_likelihood_concentration: float = 128.0
    shield_view_ratio_likelihood_min_total_count: float = 25.0
    shield_view_ratio_likelihood_min_views: int = 2
    station_view_covariance_enable: bool = False
    station_view_correlated_spectrum_fraction: float = 0.0
    variable_cardinality: bool = True
    history_estimate_interval: int = 1
    candidate_response_cache_max_entries: int = 24
    structural_rj_patch_spacing_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_strength_move_probability: float = 1.0
    structural_cardinality_prior_probs: tuple[float, ...] | list[float] | None = None
    ig_threshold: float = 1e-3  # ΔIG stopping threshold (Sec. 3.4.4).
    max_dwell_time_s: float = 5.0  # Max dwell time per pose.
    lambda_cost: float = 1.0  # Motion-cost weight (Eq. 3.51).
    alpha_weights: Dict[str, float] | None = None  # EIG isotope weights alpha_h.
    credible_volume_threshold: float = 1e-3  # Max 95% credible volume for convergence.
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 16
    min_delta_beta: float = 1e-3
    use_tempering: bool = True
    max_resamples_per_observation: int = 2
    temper_resample_cooldown_steps: int = 2
    temper_resample_force_ratio: float = 0.1
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_num_sources: Tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    strength_prior_min_cps_1m: float = 0.0
    strength_prior_max_cps_1m: float = 2_000_000.0
    observation_covariance_projection_enable: bool = True
    observation_covariance_projection_weight: float = 1.0
    observation_covariance_projection_max_corr: float = 0.999
    pose_min_observation_counts: float = 0.0
    pose_min_observation_penalty_scale: float = 1.0
    pose_min_observation_aggregate: str = "max"
    pose_min_observation_max_particles: int | None = None
    pose_min_observation_quantile: float = 0.25
    orientation_k: int = 8
    min_rotations_per_pose: int = 0
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
    parallel_isotope_updates: bool = True
    parallel_isotope_workers: int | None = None
    converge_cardinality_var_max: float = 0.05

    def __post_init__(self) -> None:
        """Validate and normalize estimator configuration values."""
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        self.strength_prior_min_cps_1m = float(self.strength_prior_min_cps_1m)
        self.strength_prior_max_cps_1m = float(self.strength_prior_max_cps_1m)
        if (
            not np.isfinite(self.strength_prior_min_cps_1m)
            or self.strength_prior_min_cps_1m < 0.0
        ):
            raise ValueError(
                "strength_prior_min_cps_1m must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.strength_prior_max_cps_1m)
            or self.strength_prior_max_cps_1m
            <= self.strength_prior_min_cps_1m
        ):
            raise ValueError(
                "strength_prior_max_cps_1m must be finite and greater than "
                "strength_prior_min_cps_1m."
            )
        self.ig_workers = int(self.ig_workers)
        if self.ig_workers < 0:
            raise ValueError("ig_workers must be >= 0.")
        self.observation_covariance_projection_enable = bool(
            self.observation_covariance_projection_enable
        )
        self.observation_covariance_projection_weight = max(
            0.0,
            float(self.observation_covariance_projection_weight),
        )
        self.observation_covariance_projection_max_corr = float(
            np.clip(
                float(self.observation_covariance_projection_max_corr),
                0.0,
                1.0,
            )
        )
        self.pose_min_observation_counts = float(self.pose_min_observation_counts)
        if self.pose_min_observation_counts < 0.0:
            raise ValueError("pose_min_observation_counts must be >= 0.")
        self.pose_min_observation_penalty_scale = float(
            self.pose_min_observation_penalty_scale
        )
        if self.pose_min_observation_penalty_scale < 0.0:
            raise ValueError("pose_min_observation_penalty_scale must be >= 0.")
        self.pose_min_observation_aggregate = (
            str(self.pose_min_observation_aggregate).strip().lower()
        )
        if self.pose_min_observation_aggregate not in {"max", "mean"}:
            raise ValueError("pose_min_observation_aggregate must be max or mean.")
        if self.pose_min_observation_max_particles is not None:
            self.pose_min_observation_max_particles = int(
                self.pose_min_observation_max_particles
            )
            if self.pose_min_observation_max_particles < 0:
                raise ValueError("pose_min_observation_max_particles must be >= 0.")
        self.pose_min_observation_quantile = float(self.pose_min_observation_quantile)
        if not 0.0 <= self.pose_min_observation_quantile <= 1.0:
            raise ValueError("pose_min_observation_quantile must be in [0, 1].")
        self.count_likelihood_model = normalize_count_likelihood_model(
            self.count_likelihood_model
        )
        self.observation_count_variance_semantics = (
            normalize_observation_count_variance_semantics(
                self.observation_count_variance_semantics,
            )
        )
        if (
            self.observation_count_variance_semantics
            == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL
            and self.count_likelihood_model == "poisson"
        ):
            raise ValueError(
                "complete_statistical observation variance requires gaussian "
                "or student_t count likelihood."
            )
        self.count_likelihood_df = max(float(self.count_likelihood_df), 1.0)
        self.shield_contrast_likelihood_enable = bool(
            self.shield_contrast_likelihood_enable
        )
        self.shield_view_ratio_likelihood_enable = bool(
            self.shield_view_ratio_likelihood_enable
        )
        if (
            self.observation_count_variance_semantics
            == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL
        ):
            # These auxiliary terms reuse the same shield-view counts without
            # the supplied cross-view covariance. Applying them would count the
            # observation twice under complete statistical semantics.
            self.shield_contrast_likelihood_enable = False
            self.shield_view_ratio_likelihood_enable = False
        self.shield_view_ratio_likelihood_weight = max(
            0.0,
            float(self.shield_view_ratio_likelihood_weight),
        )
        self.shield_view_ratio_likelihood_concentration = max(
            1.0e-6,
            float(self.shield_view_ratio_likelihood_concentration),
        )
        self.shield_view_ratio_likelihood_min_total_count = max(
            0.0,
            float(self.shield_view_ratio_likelihood_min_total_count),
        )
        self.shield_view_ratio_likelihood_min_views = max(
            2,
            int(self.shield_view_ratio_likelihood_min_views),
        )
        self.station_view_covariance_enable = bool(self.station_view_covariance_enable)
        self.station_view_correlated_spectrum_fraction = max(
            0.0,
            float(self.station_view_correlated_spectrum_fraction),
        )
        self.structural_rj_patch_spacing_m = float(self.structural_rj_patch_spacing_m)
        if (
            not np.isfinite(self.structural_rj_patch_spacing_m)
            or self.structural_rj_patch_spacing_m <= 0.0
        ):
            raise ValueError("structural_rj_patch_spacing_m must be positive.")
        for probability_field in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
        ):
            probability = float(getattr(self, probability_field))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_field} must be in [0, 1].")
            setattr(self, probability_field, probability)
        if self.structural_cardinality_prior_probs is not None:
            if not isinstance(
                self.structural_cardinality_prior_probs,
                (tuple, list),
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must be a tuple, list, or None."
                )
            cardinality_prior = np.asarray(
                [float(value) for value in self.structural_cardinality_prior_probs],
                dtype=float,
            )
            if (
                cardinality_prior.size == 0
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain only "
                    "positive finite values."
                )
            cardinality_prior /= float(np.sum(cardinality_prior))
            self.structural_cardinality_prior_probs = tuple(
                float(value) for value in cardinality_prior
            )
        self.history_estimate_interval = max(0, int(self.history_estimate_interval))
        self.candidate_response_cache_max_entries = max(
            0,
            int(self.candidate_response_cache_max_entries),
        )
        self.converge_cardinality_var_max = max(
            0.0,
            float(self.converge_cardinality_var_max),
        )
        self.parallel_isotope_updates = bool(self.parallel_isotope_updates)
        if self.parallel_isotope_workers is not None:
            self.parallel_isotope_workers = max(1, int(self.parallel_isotope_workers))
        self.variable_cardinality = bool(self.variable_cardinality)
        if self.max_sources is None or int(self.max_sources) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        self.max_sources = int(self.max_sources)
        if (
            self.structural_cardinality_prior_probs is not None
            and len(self.structural_cardinality_prior_probs) != self.max_sources + 1
        ):
            raise ValueError(
                "structural_cardinality_prior_probs must contain "
                "max_sources + 1 entries."
            )
        initial_lower, initial_upper = (
            int(self.init_num_sources[0]),
            int(self.init_num_sources[1]),
        )
        if self.variable_cardinality:
            if (
                self.structural_rj_move_probability <= 0.0
                or self.structural_rj_birth_probability <= 0.0
                or self.structural_rj_death_probability <= 0.0
            ):
                raise ValueError(
                    "Variable-cardinality pure PF requires positive structural "
                    "move, birth, and death proposal probabilities."
                )
            if initial_lower != 0 or initial_upper != self.max_sources:
                raise ValueError(
                    "Variable-cardinality pure PF initialization must cover "
                    "every cardinality from zero through max_sources."
                )
            if self.num_particles < self.max_sources + 1:
                raise ValueError(
                    "Variable-cardinality pure PF requires at least one initial "
                    "particle per cardinality from zero through max_sources."
                )
        elif (
            initial_lower != initial_upper
            or initial_lower < 0
            or initial_upper > self.max_sources
        ):
            raise ValueError(
                "Fixed-cardinality pure PF requires "
                "init_num_sources=(K, K) within zero through max_sources."
            )
        self.init_num_sources = (initial_lower, initial_upper)


@dataclass(frozen=True)
class MeasurementRecord:
    """Store a single isotope-wise measurement and metadata."""

    z_k: Dict[str, float]
    pose_idx: int
    live_time_s: float
    fe_index: int
    pb_index: int
    detector_position_xyz_m: tuple[float, float, float]
    station_sequence_id: int
    station_view_index: int
    runtime_likelihood_route_by_isotope: Dict[str, str]
    z_variance_k: Dict[str, float] | None = None
    z_covariance_k: Dict[str, Dict[str, float]] | None = None
    station_view_covariance_by_isotope: (
        Dict[str, tuple[tuple[float, ...], ...]] | None
    ) = None


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
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: Dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: Dict[str, object] | None = None,
        transport_response_model: Dict[str, object] | None = None,
    ) -> None:
        """Initialize per-isotope filters and shared measurement-model state."""
        self.isotopes = list(isotopes)
        self.pf_config = pf_config or RotatingShieldPFConfig()
        self.shield_params = shield_params or ShieldParams()
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = float(obstacle_height_m)
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = max(float(obstacle_buildup_coeff), 0.0)
        self.detector_radius_m = max(float(detector_radius_m), 0.0)
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = max(float(detector_aperture_radius_m), 0.0)
        self.detector_aperture_samples = max(int(detector_aperture_samples), 1)
        self.detector_aperture_sampling = str(detector_aperture_sampling)
        self.source_extent_radius_m = max(float(source_extent_radius_m), 0.0)
        self.source_extent_samples = max(int(source_extent_samples), 1)
        self.line_mu_by_isotope = line_mu_by_isotope
        self.transport_response_model = transport_response_model
        # Measurement poses are appended incrementally.
        self.poses: List[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: MeasurementGeometry | None = None
        self.filters: Dict[str, IsotopeParticleFilter] = {}
        self.candidate_sources = candidate_sources
        self.history_estimates: List[
            Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]
        ] = []
        self.measurements: List[MeasurementRecord] = []
        self.last_pair_sequence_update_workers = 1
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_pair_sequence_stage_wall_s: Dict[str, float] = {}
        self.last_structural_update_workers = 1
        self.last_structural_update_wall_s = 0.0
        self._candidate_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._candidate_response_cache_order: list[tuple[Any, ...]] = []
        self._candidate_response_prefix_cache: dict[
            tuple[Any, ...],
            dict[str, NDArray[np.float64]],
        ] = {}
        self._candidate_response_prefix_cache_order: list[tuple[Any, ...]] = []

    def _record_history_estimate(self, measurement_count: int) -> None:
        """Record an exact report estimate when the configured history stride allows it."""
        interval = max(
            0,
            int(getattr(self.pf_config, "history_estimate_interval", 1)),
        )
        if interval <= 0:
            return
        count = max(0, int(measurement_count))
        if count <= 0 or count % interval != 0:
            return
        self.history_estimates.append(self.estimates())

    def _candidate_response_source_key(
        self,
        sources: NDArray[np.float64],
    ) -> tuple[str, int] | None:
        """Return a stable cache key for the full shared candidate-source grid."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        candidate_arr = np.asarray(self.candidate_sources, dtype=float).reshape(
            -1,
            3,
        )
        if (
            source_arr.shape == candidate_arr.shape
            and source_arr.size > 0
            and np.shares_memory(source_arr, candidate_arr)
        ):
            return ("candidate_sources", int(source_arr.shape[0]))
        return None

    @staticmethod
    def _measurement_geometry_digest(data: MeasurementData) -> bytes:
        """Return a compact digest for measurement geometry arrays used by responses."""
        digest = hashlib.blake2b(digest_size=16)
        for array in (
            np.asarray(data.detector_positions, dtype=np.float64),
            np.asarray(data.live_times, dtype=np.float64),
            np.asarray(data.fe_indices, dtype=np.int64),
            np.asarray(data.pb_indices, dtype=np.int64),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.digest()

    @staticmethod
    def _response_scale_vector(
        scale: NDArray[np.float64] | float,
        measurement_count: int,
    ) -> NDArray[np.float64]:
        """Return one response-scale value per measurement row."""
        count = max(0, int(measurement_count))
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if count == 0:
            return np.zeros(0, dtype=float)
        if scale_arr.size == 0:
            return np.ones(count, dtype=float)
        if scale_arr.size == 1 and count != 1:
            return np.full(count, float(scale_arr[0]), dtype=float)
        if scale_arr.size != count:
            raise ValueError(
                "source_scale must be scalar or one value per measurement."
            )
        return scale_arr.astype(float, copy=False)

    def _store_candidate_response_cache(
        self,
        cache_key: tuple[Any, ...],
        counts: NDArray[np.float64],
    ) -> None:
        """Store an exact deterministic candidate response with LRU eviction."""
        self._candidate_response_cache[cache_key] = np.asarray(
            counts,
            dtype=float,
        ).copy()
        self._candidate_response_cache_order.append(cache_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_cache_order) > max_entries:
            old_key = self._candidate_response_cache_order.pop(0)
            if old_key not in self._candidate_response_cache_order:
                self._candidate_response_cache.pop(old_key, None)

    @staticmethod
    def _candidate_response_prefix_matches(
        payload: Mapping[str, NDArray[np.float64]],
        data: MeasurementData,
        scale: NDArray[np.float64],
        row_count: int,
    ) -> bool:
        """Return True when cached response rows match the requested prefix."""
        rows = max(0, int(row_count))
        if rows < 0:
            return False
        try:
            return bool(
                np.allclose(
                    np.asarray(payload["detector_positions"], dtype=float)[:rows],
                    np.asarray(data.detector_positions, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.allclose(
                    np.asarray(payload["live_times"], dtype=float)[:rows],
                    np.asarray(data.live_times, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.array_equal(
                    np.asarray(payload["fe_indices"], dtype=int)[:rows],
                    np.asarray(data.fe_indices, dtype=int)[:rows],
                )
                and np.array_equal(
                    np.asarray(payload["pb_indices"], dtype=int)[:rows],
                    np.asarray(data.pb_indices, dtype=int)[:rows],
                )
                and np.allclose(
                    np.asarray(payload["source_scale"], dtype=float)[:rows],
                    np.asarray(scale, dtype=float)[:rows],
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _store_candidate_response_prefix_cache(
        self,
        prefix_key: tuple[Any, ...],
        data: MeasurementData,
        scale: NDArray[np.float64],
        counts: NDArray[np.float64],
    ) -> None:
        """Store a full-history response prefix for incremental extension."""
        self._candidate_response_prefix_cache[prefix_key] = {
            "detector_positions": np.asarray(
                data.detector_positions, dtype=float
            ).copy(),
            "live_times": np.asarray(data.live_times, dtype=float).copy(),
            "fe_indices": np.asarray(data.fe_indices, dtype=int).copy(),
            "pb_indices": np.asarray(data.pb_indices, dtype=int).copy(),
            "source_scale": np.asarray(scale, dtype=float).copy(),
            "counts": np.asarray(counts, dtype=float).copy(),
        }
        self._candidate_response_prefix_cache_order.append(prefix_key)
        max_entries = max(
            0,
            int(self.pf_config.candidate_response_cache_max_entries),
        )
        while len(self._candidate_response_prefix_cache_order) > max_entries:
            old_key = self._candidate_response_prefix_cache_order.pop(0)
            if old_key not in self._candidate_response_prefix_cache_order:
                self._candidate_response_prefix_cache.pop(old_key, None)

    def _cached_expected_counts_for_kernel(
        self,
        *,
        kernel: ContinuousKernel,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched expected counts from a particle-independent kernel."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._candidate_response_source_key(source_arr)
        scale = self.response_scales_for_measurements(
            isotope,
            data.fe_indices,
            data.pb_indices,
        )
        measurement_count = int(np.asarray(data.live_times, dtype=float).size)
        scale_arr = self._response_scale_vector(scale, measurement_count)
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.candidate_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        prefix_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(kernel)),
                source_key,
                self._measurement_geometry_digest(data),
                tuple(scale_arr.round(12).tolist()),
            )
            cached = self._candidate_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            prefix_key = (str(isotope), int(id(kernel)), source_key)
            prefix_payload = self._candidate_response_prefix_cache.get(prefix_key)
            if isinstance(prefix_payload, dict):
                cached_counts = np.asarray(
                    prefix_payload.get("counts", np.zeros((0, 0))),
                    dtype=float,
                )
                cached_rows = int(cached_counts.shape[0])
                if (
                    cached_rows >= measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        measurement_count,
                    )
                ):
                    counts_arr = cached_counts[:measurement_count].copy()
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr
                if (
                    cached_rows < measurement_count
                    and self._candidate_response_prefix_matches(
                        prefix_payload,
                        data,
                        scale_arr,
                        cached_rows,
                    )
                ):
                    suffix_counts = expected_counts_per_source(
                        kernel=kernel,
                        isotope=isotope,
                        detector_positions=np.asarray(
                            data.detector_positions,
                            dtype=float,
                        )[cached_rows:],
                        sources=source_arr,
                        strengths=strength_arr,
                        live_times=np.asarray(data.live_times, dtype=float)[
                            cached_rows:
                        ],
                        fe_indices=np.asarray(data.fe_indices, dtype=int)[cached_rows:],
                        pb_indices=np.asarray(data.pb_indices, dtype=int)[cached_rows:],
                        source_scale=scale_arr[cached_rows:],
                    )
                    counts_arr = np.vstack(
                        [
                            cached_counts,
                            np.asarray(suffix_counts, dtype=float),
                        ]
                    )
                    self._store_candidate_response_prefix_cache(
                        prefix_key,
                        data,
                        scale_arr,
                        counts_arr,
                    )
                    self._store_candidate_response_cache(cache_key, counts_arr)
                    return counts_arr.copy()
        counts = expected_counts_per_source(
            kernel=kernel,
            isotope=isotope,
            detector_positions=data.detector_positions,
            sources=source_arr,
            strengths=strength_arr,
            live_times=data.live_times,
            fe_indices=data.fe_indices,
            pb_indices=data.pb_indices,
            source_scale=scale_arr,
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._store_candidate_response_cache(cache_key, counts_arr)
            if prefix_key is not None:
                self._store_candidate_response_prefix_cache(
                    prefix_key,
                    data,
                    scale_arr,
                    counts_arr,
                )
        return counts_arr

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: MeasurementData,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts using one isotope filter's physical kernel."""
        return self._cached_expected_counts_for_kernel(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            data=data,
            sources=sources,
            strengths=strengths,
        )

    def _resolve_mu_by_isotope(
        self, mu_by_isotope: Dict[str, object] | None
    ) -> Dict[str, object]:
        """
        Ensure per-isotope attenuation coefficients are available for all isotopes.

        When missing, attempt to populate values from the HVL/TVL table; otherwise raise.
        """
        from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm

        def _norm_key(name: str) -> str:
            """Return a normalized isotope key for attenuation lookup."""
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
        isotope_names = self.isotopes
        if isotope_names:
            still_missing: List[str] = []
            for iso in isotope_names:
                if iso in resolved:
                    continue
                norm = _norm_key(iso)
                if norm in normalized:
                    resolved[iso] = normalized[norm]
                    continue
                canonical = canonical_by_norm.get(norm)
                if canonical is not None:
                    table_vals = mu_by_isotope_from_tvl_mm(
                        HVL_TVL_TABLE_MM, isotopes=[canonical]
                    )
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
        """Build the discrete kernel cache when it is first needed."""
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = MeasurementGeometry(
            candidate_sources=self.candidate_sources,
            poses=poses_arr,
            orientations=self.normals,
            shield_params=self.shield_params,
            mu_by_isotope=self.mu_by_isotope,
        )
        pf_conf = self._build_pf_config()
        if self.filters:
            expected = set(self.isotopes)
            actual = set(self.filters)
            if actual != expected:
                raise RuntimeError(
                    "Pure PF requires exactly one initialized filter per isotope."
                )
            for iso in self.isotopes:
                self.filters[iso].set_kernel(self.kernel_cache)
        else:
            for iso in self.isotopes:
                self.filters[iso] = self._build_filter(iso, pf_conf)

    def _build_filter(self, isotope: str, pf_conf: PFConfig) -> IsotopeParticleFilter:
        """Build an isotope filter with shared PF observation-model settings."""
        return IsotopeParticleFilter(
            isotope,
            kernel=self.kernel_cache,
            config=pf_conf,
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=self.detector_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
        )

    def _build_pf_config(self) -> PFConfig:
        """Build the per-isotope config from the shared PF field contract."""
        shared_values = {
            field.name: getattr(self.pf_config, field.name)
            for field in fields(PFConfig)
            if hasattr(self.pf_config, field.name)
        }
        return PFConfig(**shared_values)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        if not gpu_utils.torch_device_available(self.pf_config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether torch-backed estimator math is available."""
        from pf import gpu_utils

        return bool(
            self.pf_config.use_gpu
            and gpu_utils.torch_device_available(self.pf_config.gpu_device)
        )

    def response_scale_for_isotope(
        self,
        isotope: str,
        *,
        fe_index: int,
        pb_index: int,
    ) -> float:
        """Return the configured source response scale for one isotope."""
        pair_id = int(fe_index) * int(self.num_orientations) + int(pb_index)
        pair_scales = self.pf_config.measurement_scale_by_isotope_and_pair
        if isinstance(pair_scales, Mapping):
            iso_pair_scales = pair_scales.get(str(isotope), {})
            if isinstance(iso_pair_scales, Mapping):
                value = iso_pair_scales.get(int(pair_id))
                if value is None:
                    value = iso_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.pf_config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(isotope, 1.0)), 0.0)

    def response_scales_for_measurements(
        self,
        isotope: str,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale per Fe/Pb measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self.response_scale_for_isotope(
                    isotope,
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _project_observation_covariance_to_variance(
        self,
        z_k: Mapping[str, float],
        z_variance_k: Mapping[str, float] | None,
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> tuple[Dict[str, float] | None, Dict[str, Dict[str, float]] | None]:
        """
        Return diagonal PF variances that conservatively cover isotope covariance.

        The per-isotope filters are conditionally independent, while the
        response-Poisson spectrum regression reports a same-spectrum covariance
        across isotope count channels.  This projection keeps the observed count
        means unchanged and uses a Gershgorin-style diagonal envelope,
        ``var_i + sum_j |cov_ij|``, so ignoring off-diagonal terms cannot make
        the independent isotope filters more confident than the structured
        covariance supports.
        """
        if z_variance_k is None and z_covariance_k is None:
            return None, None
        isotopes = [str(isotope) for isotope in z_k]
        if not isotopes:
            return {}, None
        base_variances = np.asarray(
            [
                max(
                    float(
                        z_variance_k.get(
                            isotope, max(float(z_k.get(isotope, 0.0)), 1.0)
                        )
                        if z_variance_k is not None
                        else max(float(z_k.get(isotope, 0.0)), 1.0)
                    ),
                    1.0,
                )
                for isotope in isotopes
            ],
            dtype=float,
        )
        if z_covariance_k is None or not bool(
            self.pf_config.observation_covariance_projection_enable
        ):
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                self._sanitize_observation_covariance(
                    isotopes,
                    base_variances,
                    z_covariance_k,
                ),
            )
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return (
                {
                    isotope: float(variance)
                    for isotope, variance in zip(isotopes, base_variances)
                },
                None,
            )
        offdiag_abs = np.sum(np.abs(covariance), axis=1) - np.abs(np.diag(covariance))
        projected = np.maximum(
            base_variances,
            np.diag(covariance)
            + float(self.pf_config.observation_covariance_projection_weight)
            * offdiag_abs,
        )
        return (
            {
                isotope: float(variance)
                for isotope, variance in zip(isotopes, projected)
            },
            {
                row_iso: {
                    col_iso: float(covariance[row_idx, col_idx])
                    for col_idx, col_iso in enumerate(isotopes)
                }
                for row_idx, row_iso in enumerate(isotopes)
            },
        )

    def _observation_covariance_matrix(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]],
    ) -> NDArray[np.float64] | None:
        """Build a symmetric isotope covariance matrix for one spectrum."""
        covariance = np.diag(np.maximum(np.asarray(base_variances, dtype=float), 1.0))
        index_by_isotope = {str(isotope): idx for idx, isotope in enumerate(isotopes)}
        for row_iso, row_payload in z_covariance_k.items():
            if str(row_iso) not in index_by_isotope or not isinstance(
                row_payload,
                Mapping,
            ):
                continue
            row_idx = index_by_isotope[str(row_iso)]
            for col_iso, raw_value in row_payload.items():
                col_key = str(col_iso)
                if col_key not in index_by_isotope:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                col_idx = index_by_isotope[col_key]
                covariance[row_idx, col_idx] = value
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), base_variances))
        diag = np.maximum(np.diag(covariance), 1.0)
        corr_limit = float(self.pf_config.observation_covariance_projection_max_corr)
        if corr_limit < 1.0:
            scale = np.sqrt(diag[:, None] * diag[None, :])
            corr = np.divide(
                covariance,
                scale,
                out=np.zeros_like(covariance, dtype=float),
                where=scale > 0.0,
            )
            corr = np.clip(corr, -corr_limit, corr_limit)
            covariance = corr * scale
            np.fill_diagonal(covariance, diag)
        return covariance

    def _sanitize_observation_covariance(
        self,
        isotopes: Sequence[str],
        base_variances: NDArray[np.float64],
        z_covariance_k: Mapping[str, Mapping[str, float]] | None,
    ) -> Dict[str, Dict[str, float]] | None:
        """Return a JSON-safe covariance payload when one was supplied."""
        if z_covariance_k is None:
            return None
        covariance = self._observation_covariance_matrix(
            isotopes,
            base_variances,
            z_covariance_k,
        )
        if covariance is None:
            return None
        return {
            row_iso: {
                col_iso: float(covariance[row_idx, col_idx])
                for col_idx, col_iso in enumerate(isotopes)
            }
            for row_idx, row_iso in enumerate(isotopes)
        }

    def select_runtime_likelihood_routes(
        self,
        *,
        sequence_length: int,
        z_view_covariance_by_isotope: (
            Mapping[str, NDArray[np.float64]] | None
        ) = None,
    ) -> dict[str, str]:
        """Select one explicit likelihood route per isotope before PF ingestion."""
        length = int(sequence_length)
        if length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not self.filters:
            self._ensure_kernel_cache()
        routes: dict[str, str] = {}
        for isotope, filt in self.filters.items():
            view_covariance = self._view_covariance_for_isotope(
                isotope,
                sequence_length=length,
                z_view_covariance_by_isotope=z_view_covariance_by_isotope,
            )
            model = normalize_count_likelihood_model(
                str(filt.config.count_likelihood_model)
            )
            routes[str(isotope)] = (
                COUNT_COVARIANCE_LIKELIHOOD_ROUTE
                if model != "poisson"
                and filt._sequence_covariance_enabled(length, view_covariance)
                else COUNT_LIKELIHOOD_ROUTE
            )
        return canonical_runtime_likelihood_route_mapping(
            routes,
            self.configured_isotope_order(),
        )

    def _runtime_likelihood_routes_for_records(
        self,
        isotope: str,
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.str_]:
        """Return the exact per-row runtime likelihood route for one isotope."""
        configured_isotopes = self.configured_isotope_order()
        if str(isotope) not in configured_isotopes:
            raise ValueError(f"Isotope {isotope!r} is not configured.")
        routes: list[str] = []
        for record in records:
            mapping = canonical_runtime_likelihood_route_mapping(
                record.runtime_likelihood_route_by_isotope,
                configured_isotopes,
            )
            routes.append(mapping[str(isotope)])
        return np.asarray(routes, dtype="<U16")

    @staticmethod
    def _normalize_pair_sequence_record(
        record: Sequence[object],
    ) -> tuple[
        Dict[str, float],
        int,
        int,
        float,
        Dict[str, float] | None,
        Dict[str, Dict[str, float]] | None,
    ]:
        """Return a canonical same-pose shield-program observation record."""
        if len(record) == 5:
            z_k, fe_index, pb_index, live_time_s, z_variance_k = record
            z_covariance_k = None
        elif len(record) == 6:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
            ) = record
        else:
            raise ValueError(
                "Pair sequence records must have 5 fields "
                "(z, fe, pb, live, variance) or 6 fields with covariance."
            )
        return (
            {str(isotope): float(value) for isotope, value in dict(z_k).items()},
            int(fe_index),
            int(pb_index),
            float(live_time_s),
            None
            if z_variance_k is None
            else {
                str(isotope): float(value)
                for isotope, value in dict(z_variance_k).items()
            },
            None
            if z_covariance_k is None
            else {
                str(row_iso): {
                    str(col_iso): float(value)
                    for col_iso, value in dict(row_payload).items()
                }
                for row_iso, row_payload in dict(z_covariance_k).items()
            },
        )

    @staticmethod
    def _view_covariance_for_isotope(
        isotope: str,
        *,
        sequence_length: int,
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None,
    ) -> NDArray[np.float64] | None:
        """Return a same-station shield-view covariance matrix for one isotope."""
        if z_view_covariance_by_isotope is None:
            return None
        payload = z_view_covariance_by_isotope.get(str(isotope))
        if payload is None:
            return None
        covariance = np.asarray(payload, dtype=float)
        expected_shape = (int(sequence_length), int(sequence_length))
        if covariance.shape != expected_shape:
            raise ValueError(
                "z_view_covariance_by_isotope entries must be shaped K x K."
            )
        return 0.5 * (covariance + covariance.T)

    def continuous_kernel(
        self,
        *,
        detector_aperture_samples: int | None = None,
        use_gpu: bool | None = None,
    ) -> ContinuousKernel:
        """Build the shared ContinuousKernel for PF, planning, and diagnostics."""
        active_aperture_samples = (
            int(self.detector_aperture_samples)
            if detector_aperture_samples is None
            else int(detector_aperture_samples)
        )
        active_use_gpu = (
            bool(self.pf_config.use_gpu) if use_gpu is None else bool(use_gpu)
        )
        return ContinuousKernel(
            mu_by_isotope=self.mu_by_isotope,
            shield_params=self.shield_params,
            orientations=self.normals,
            use_gpu=active_use_gpu,
            gpu_device=str(self.pf_config.gpu_device),
            gpu_dtype=str(self.pf_config.gpu_dtype),
            obstacle_grid=self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
            obstacle_mu_by_isotope=self.obstacle_mu_by_isotope,
            obstacle_buildup_coeff=self.obstacle_buildup_coeff,
            detector_radius_m=self.detector_radius_m,
            detector_aperture_radius_m=self.detector_aperture_radius_m,
            detector_aperture_samples=max(1, active_aperture_samples),
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            transport_response_model=self.transport_response_model,
        )

    def configured_isotope_order(self) -> tuple[str, ...]:
        """Return the stable pure-PF isotope order."""
        return tuple(dict.fromkeys(str(isotope) for isotope in self.isotopes))

    def configured_isotope_response_counts(
        self,
        isotope: str,
        data: MeasurementData,
        source_positions: NDArray[np.float64],
        strengths: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return batched responses from the configured isotope PF kernel.

        Measurement rows and source positions are evaluated by the same continuous
        transport, obstacle, aperture, shield, and calibrated response-scale model
        used by the PF. Candidate positions remain batched and particle state is not
        read.
        """
        isotope_key = str(isotope)
        if isotope_key not in self.configured_isotope_order():
            raise KeyError(f"Isotope {isotope_key!r} is not configured.")
        filt = self.filters.get(isotope_key)
        if filt is None:
            raise RuntimeError(
                "Pure PF isotope filters must be initialized before response "
                "diagnostics are evaluated."
            )
        positions = np.asarray(source_positions, dtype=float).reshape(-1, 3)
        if strengths is None:
            strength_values = np.ones(positions.shape[0], dtype=float)
        else:
            strength_values = np.asarray(strengths, dtype=float).reshape(-1)
        if strength_values.size != positions.shape[0]:
            raise ValueError("strengths must contain one value per source position.")
        return self._cached_expected_counts_for_kernel(
            kernel=filt.continuous_kernel,
            isotope=isotope_key,
            data=data,
            sources=positions,
            strengths=strength_values,
        )

    def _continuous_kernel(self) -> ContinuousKernel:
        """Build a ContinuousKernel matching the estimator observation model."""
        return self.continuous_kernel()

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
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[pose_idx], dtype=float)
        return self.expected_counts_pair_for_states_at_detector(
            isotope=isotope,
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            states=states,
        )

    def expected_counts_pair_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute Λ for a state subset at an arbitrary detector position.

        This helper keeps candidate-pose and shield-selection scoring on the
        same GPU-accelerated transport approximation as normal PF updates,
        even when a planning particle subset is used.
        """
        if not states:
            return np.zeros(0, dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            values = np.zeros(len(states), dtype=float)
            source_scale = self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            )
            for idx, state in enumerate(states):
                rate = float(state.background)
                for pos, strength in zip(
                    state.positions[: state.num_sources],
                    state.strengths[: state.num_sources],
                ):
                    rate += (
                        source_scale
                        * float(strength)
                        * kernel.kernel_value_pair(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            source_pos=pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                        )
                    )
                values[idx] = float(live_time_s) * rate
            return values
        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        lam_t = kernel.expected_counts_pair_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self.response_scale_for_isotope(
                isotope,
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy()

    def expected_counts_all_pairs_for_states_at_detector(
        self,
        isotope: str,
        detector_pos: NDArray[np.float64],
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """
        Compute expected counts for all Fe/Pb orientation pairs for state subsets.

        The returned array is shaped ``(num_pairs, num_states)`` and uses the same
        continuous kernel, spherical-octant shield geometry, detector aperture,
        response scale, and obstacle attenuation as the per-pair helper.
        """
        num_pairs = int(self.num_orientations) * int(self.num_orientations)
        if not states:
            return np.zeros((num_pairs, 0), dtype=float)
        kernel = self._continuous_kernel()
        detector_pos = np.asarray(detector_pos, dtype=float)
        use_gpu = False
        if self.pf_config.use_gpu and int(self.num_orientations) == 8:
            try:
                use_gpu = bool(self._gpu_enabled())
            except RuntimeError:
                use_gpu = False
        if not use_gpu:
            rows: list[NDArray[np.float64]] = []
            for fe_index in range(int(self.num_orientations)):
                for pb_index in range(int(self.num_orientations)):
                    rows.append(
                        self.expected_counts_pair_for_states_at_detector(
                            isotope=isotope,
                            detector_pos=detector_pos,
                            fe_index=fe_index,
                            pb_index=pb_index,
                            live_time_s=live_time_s,
                            states=states,
                        )
                    )
            return np.vstack(rows).astype(float, copy=False)

        from pf import gpu_utils

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        num_orients = int(self.num_orientations)
        fe_indices = np.repeat(np.arange(num_orients), num_orients)
        pb_indices = np.tile(np.arange(num_orients), num_orients)
        lam_t = kernel.expected_counts_all_pairs_for_packed_states_torch(
            isotope=isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            live_time_s=live_time_s,
            source_scale=self.response_scales_for_measurements(
                isotope,
                fe_indices,
                pb_indices,
            ),
            device=device,
            dtype=dtype,
        )
        return lam_t.detach().cpu().numpy().astype(float, copy=False)

    def expected_counts_all_pairs_for_states_at_detectors(
        self,
        isotope: str,
        detector_positions: NDArray[np.float64],
        live_time_s: float,
        states: Sequence[IsotopeState],
    ) -> NDArray[np.float64]:
        """Compute all-pair counts for many detector positions in one batch.

        The result is shaped ``(detector, pair, state)``.  Particle source slots
        are packed once and the shared continuous kernel evaluates every
        detector/source/pair response together.  Callers should pass bounded
        detector chunks when many planning poses are under consideration.
        """
        detectors = np.asarray(detector_positions, dtype=float)
        if detectors.size == 0:
            detectors = np.zeros((0, 3), dtype=float)
        if detectors.ndim != 2 or detectors.shape[1] != 3:
            raise ValueError("detector_positions must be shaped (D, 3).")
        num_pairs = int(self.num_orientations) * int(self.num_orientations)
        state_count = len(states)
        if state_count == 0:
            return np.zeros((detectors.shape[0], num_pairs, 0), dtype=float)

        max_sources = max(
            (max(0, int(state.num_sources)) for state in states), default=0
        )
        backgrounds = np.asarray(
            [max(float(state.background), 0.0) for state in states],
            dtype=float,
        )
        source_rates = np.zeros(
            (detectors.shape[0], num_pairs, state_count),
            dtype=float,
        )
        if max_sources > 0 and detectors.shape[0] > 0:
            positions = np.zeros((state_count, max_sources, 3), dtype=float)
            strengths = np.zeros((state_count, max_sources), dtype=float)
            for state_index, state in enumerate(states):
                source_count = min(max(0, int(state.num_sources)), max_sources)
                if source_count <= 0:
                    continue
                positions[state_index, :source_count, :] = np.asarray(
                    state.positions[:source_count],
                    dtype=float,
                )
                strengths[state_index, :source_count] = np.maximum(
                    np.asarray(state.strengths[:source_count], dtype=float),
                    0.0,
                )
            kernel_values = (
                self._continuous_kernel().kernel_values_all_pairs_for_detectors(
                    isotope=isotope,
                    detector_positions=detectors,
                    sources=positions.reshape(-1, 3),
                )
            )
            expected_shape = (
                detectors.shape[0],
                num_pairs,
                state_count * max_sources,
            )
            if kernel_values.shape != expected_shape:
                raise RuntimeError(
                    "Batched all-pair kernel returned an unexpected shape: "
                    f"{kernel_values.shape} != {expected_shape}."
                )
            source_rates = np.einsum(
                "daps,ps->dap",
                kernel_values.reshape(
                    detectors.shape[0],
                    num_pairs,
                    state_count,
                    max_sources,
                ),
                strengths,
                optimize=True,
            )

        num_orients = int(self.num_orientations)
        fe_indices = np.repeat(np.arange(num_orients), num_orients)
        pb_indices = np.tile(np.arange(num_orients), num_orients)
        source_scales = self.response_scales_for_measurements(
            isotope,
            fe_indices,
            pb_indices,
        )
        rates = backgrounds[None, None, :] + source_scales[None, :, None] * np.maximum(
            source_rates, 0.0
        )
        return np.maximum(float(live_time_s) * rates, 0.0)

    def shield_selection_batch_grids(
        self,
        pose_idx: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
        include_count_quantiles: bool = True,
    ) -> tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """
        Return all-pair shield-signature and observability count grids.

        This batches the same per-pair expected-count calculation used by
        ``orientation_signature_separation_score`` and
        ``expected_observation_counts_by_isotope_at_pair``. It is a planning
        acceleration only; it does not change the observation model used for PF
        updates.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        if pose_idx < 0 or pose_idx >= len(self.poses):
            raise IndexError("pose_idx out of range")
        detector_pos = np.asarray(self.poses[int(pose_idx)], dtype=float)
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        floor = max(float(variance_floor), eps)
        signature_flat = np.zeros(num_pairs, dtype=float)
        count_quantiles: Dict[str, NDArray[np.float64]] = {}
        if particles_by_isotope is None:
            particles_by_isotope = self.planning_particles(
                max_particles=max_particles,
                method=self.pf_config.planning_method,
            )

        for iso, filt in self.filters.items():
            if iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                if not filt.continuous_particles:
                    continue
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            means = lambdas @ weights_arr
            centered = lambdas - means[:, None]
            variances = (centered * centered) @ weights_arr
            signature_flat += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * np.maximum(variances, 0.0)
                / np.maximum(means, floor)
            )
            if include_count_quantiles:
                quantile = float(self.pf_config.pose_min_observation_quantile)
                count_quantiles[iso] = np.asarray(
                    [
                        _weighted_quantile(lambdas[pair_idx], weights_arr, quantile)
                        for pair_idx in range(num_pairs)
                    ],
                    dtype=float,
                ).reshape(num_orients, num_orients)
        return (
            np.maximum(signature_flat, 0.0).reshape(num_orients, num_orients),
            count_quantiles,
        )

    def expected_observation_counts_by_isotope_at_pose(
        self,
        pose_xyz: NDArray[np.float64],
        *,
        live_time_s: float,
        fe_pb_pairs: Sequence[tuple[int, int]] | None = None,
        aggregate: str = "max",
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """
        Return posterior-mean expected counts for each isotope at a candidate pose.

        The value for one isotope is computed from the same inverse-square,
        spherical shield, and obstacle attenuation model used by PF updates.
        Across shield pairs, ``aggregate="max"`` returns the best achievable
        expected count at that pose, while ``aggregate="mean"`` returns the
        orientation-average expected count. Each pair uses a weighted posterior
        quantile rather than the posterior mean, so a few high-strength outlier
        particles cannot make the pose look observable for every isotope.
        """
        detector = np.asarray(pose_xyz, dtype=float)
        if detector.shape != (3,):
            raise ValueError("pose_xyz must be shape (3,).")
        live_time = float(live_time_s)
        if live_time <= 0.0:
            return {iso: 0.0 for iso in self.isotopes}
        aggregate = str(aggregate).strip().lower()
        if aggregate not in {"max", "mean"}:
            raise ValueError("aggregate must be max or mean.")
        num_orients = max(1, int(self.num_orientations))
        if fe_pb_pairs is None:
            pairs = [
                (fe_index, pb_index)
                for fe_index in range(num_orients)
                for pb_index in range(num_orients)
            ]
        else:
            pairs = [(int(fe), int(pb)) for fe, pb in fe_pb_pairs]
        if not pairs:
            return {iso: 0.0 for iso in self.isotopes}
        particles = self.planning_particles(max_particles=max_particles)
        counts_by_isotope: Dict[str, float] = {}
        eps = 1e-12
        for iso in self.isotopes:
            filt = self.filters.get(iso)
            use_gpu_quantile = False
            if (
                max_particles is None
                and filt is not None
                and filt.continuous_particles
                and self.pf_config.use_gpu
            ):
                try:
                    use_gpu_quantile = bool(self._gpu_enabled())
                except RuntimeError:
                    use_gpu_quantile = False
            if use_gpu_quantile:
                weights_arr = np.asarray(filt.continuous_weights, dtype=float)
                weight_sum = float(np.sum(weights_arr))
                if weight_sum <= eps:
                    weights_arr = np.ones(len(weights_arr), dtype=float) / max(
                        len(weights_arr),
                        1,
                    )
                else:
                    weights_arr = weights_arr / weight_sum
                pair_means = []
                for fe_index, pb_index in pairs:
                    lambdas = filt._continuous_expected_counts_pair_at_pose(
                        detector_pos=detector,
                        fe_index=fe_index,
                        pb_index=pb_index,
                        live_time_s=live_time,
                    )
                    pair_means.append(
                        _weighted_quantile(
                            lambdas,
                            weights_arr,
                            self.pf_config.pose_min_observation_quantile,
                        )
                    )
                if aggregate == "mean":
                    counts_by_isotope[iso] = float(np.mean(pair_means))
                else:
                    counts_by_isotope[iso] = float(np.max(pair_means))
                continue
            if iso not in particles:
                counts_by_isotope[iso] = 0.0
                continue
            states, weights = particles[iso]
            if not states:
                counts_by_isotope[iso] = 0.0
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            pair_means: list[float] = []
            for fe_index, pb_index in pairs:
                lambdas = self.expected_counts_pair_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time,
                    states=states,
                )
                pair_means.append(
                    _weighted_quantile(
                        lambdas,
                        weights_arr,
                        self.pf_config.pose_min_observation_quantile,
                    )
                )
            if aggregate == "mean":
                counts_by_isotope[iso] = float(np.mean(pair_means))
            else:
                counts_by_isotope[iso] = float(np.max(pair_means))
        return counts_by_isotope

    def expected_observation_counts_by_isotope_at_pair(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        max_particles: int | None = None,
    ) -> Dict[str, float]:
        """Return posterior-quantile expected counts for one Fe/Pb pair."""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        pose = np.asarray(self.poses[int(pose_idx)], dtype=float)
        return self.expected_observation_counts_by_isotope_at_pose(
            pose,
            live_time_s=float(live_time_s),
            fe_pb_pairs=[(int(fe_index), int(pb_index))],
            aggregate="max",
            max_particles=max_particles,
        )

    def orientation_signature_separation_score(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        *,
        live_time_s: float,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        variance_floor: float = 1.0,
    ) -> float:
        """
        Return a shield-signature separation score for one orientation pair.

        The score is a weighted posterior variance of predicted counts,
        normalized by the mean count scale. It favors shield postures whose
        response differs across currently plausible source hypotheses.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        score = 0.0
        floor = max(float(variance_floor), eps)
        for iso, filt in self.filters.items():
            if particles_by_isotope is not None and iso in particles_by_isotope:
                states, weights = particles_by_isotope[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = filt.continuous_weights
            if not states:
                continue
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lambdas = self.expected_counts_pair_for_states(
                isotope=iso,
                pose_idx=int(pose_idx),
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                live_time_s=float(live_time_s),
                states=states,
            )
            if lambdas.size == 0:
                continue
            mean = float(np.sum(weights_arr * lambdas))
            var = float(np.sum(weights_arr * (lambdas - mean) ** 2))
            score += (
                float(alphas.get(iso, 1.0))
                / alpha_sum
                * max(var, 0.0)
                / max(mean, floor)
            )
        return float(max(score, 0.0))

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
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= n_particles
            ):
                states = [p.state.copy() for p in filt.continuous_particles]
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
                raise ValueError(
                    f"Unknown planning particle selection method: {method}"
                )
            states = [filt.continuous_particles[i].state.copy() for i in idx]
            subsets[iso] = (states, sel_weights)
        return subsets

    def add_measurement_pose(
        self, pose: NDArray[np.float64], reset_filters: bool = True
    ) -> None:
        """Register a new measurement pose and invalidate the kernel cache."""
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self.filters = {}

    def _registered_detector_position_xyz(
        self,
        pose_idx: int,
    ) -> tuple[float, float, float]:
        """Return the canonical registered detector position for a pose index."""
        position = np.asarray(self.poses[int(pose_idx)], dtype=float).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError(
                "Registered measurement poses must contain three finite coordinates."
            )
        return tuple(float(value) for value in position)

    def update_pair(
        self,
        z_k: Dict[str, float],
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_variance_k: Dict[str, float] | None = None,
        z_covariance_k: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        """
        Update PFs using Fe/Pb orientation indices (RFe, RPb) and isotope-wise counts z_k.

        Configured isotopes omitted from ``z_k`` are observed as zero, matching
        joint-sequence and structural-history semantics. This feeds the
        continuous 3D PF path with expected counts from the shield pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        detector_position_xyz_m = self._registered_detector_position_xyz(pose_idx)
        effective_variance_k, sanitized_covariance_k = (
            self._project_observation_covariance_to_variance(
                z_k,
                z_variance_k,
                z_covariance_k,
            )
        )
        runtime_likelihood_routes = {
            str(isotope): COUNT_LIKELIHOOD_ROUTE for isotope in self.filters
        }
        for iso, filt in self.filters.items():
            val = float(z_k.get(iso, 0.0))
            # Use continuous PF update that relies on spectrum-unfolded counts.
            filt.update_continuous_pair(
                z_obs=val,
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                observation_count_variance=(
                    0.0
                    if effective_variance_k is None
                    else float(effective_variance_k.get(iso, 0.0))
                ),
                step_idx=len(self.measurements),
            )
        self.measurements.append(
            MeasurementRecord(
                z_k={iso: float(v) for iso, v in z_k.items()},
                pose_idx=int(pose_idx),
                live_time_s=float(live_time_s),
                fe_index=int(fe_index),
                pb_index=int(pb_index),
                detector_position_xyz_m=detector_position_xyz_m,
                station_sequence_id=int(len(self.measurements)),
                station_view_index=0,
                z_variance_k={
                    str(iso): (
                        0.0
                        if effective_variance_k is None
                        else float(effective_variance_k.get(iso, 0.0))
                    )
                    for iso in self.filters
                },
                z_covariance_k=sanitized_covariance_k,
                runtime_likelihood_route_by_isotope=runtime_likelihood_routes,
            )
        )
        self._apply_structural_moves()
        self._record_history_estimate(len(self.measurements))

    def update_pair_sequence(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        runtime_likelihood_route_by_isotope: Mapping[str, str],
        z_view_covariance_by_isotope: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """
        Jointly update PFs from a same-pose shield-orientation sequence.

        Each record is ``(z_k, fe_index, pb_index, live_time_s, z_variance_k)``.
        A sixth ``z_covariance_k`` field may be supplied for same-spectrum
        isotope covariance.
        ``z_view_covariance_by_isotope`` may also supply KxK same-station
        shield-view covariance for each isotope. The joint update uses one
        station-level likelihood over all postures and only applies birth/death
        after the full shield program is observed.
        """
        if not records:
            return
        runtime_likelihood_routes = canonical_runtime_likelihood_route_mapping(
            runtime_likelihood_route_by_isotope,
            self.configured_isotope_order(),
        )
        expected_runtime_likelihood_routes = self.select_runtime_likelihood_routes(
            sequence_length=len(records),
            z_view_covariance_by_isotope=z_view_covariance_by_isotope,
        )
        if runtime_likelihood_routes != expected_runtime_likelihood_routes:
            raise ValueError(
                "Explicit runtime likelihood routes do not match the configured "
                "count/covariance likelihood inputs."
            )
        sequence_start = time.perf_counter()
        stage_wall: Dict[str, float] = {}
        stage_start = sequence_start
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        detector_position_xyz_m = self._registered_detector_position_xyz(pose_idx)
        normalized_records = []
        for record in records:
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
            ) = self._normalize_pair_sequence_record(record)
            effective_variance_k, sanitized_covariance_k = (
                self._project_observation_covariance_to_variance(
                    z_k,
                    z_variance_k,
                    z_covariance_k,
                )
            )
            normalized_records.append(
                (
                    z_k,
                    fe_index,
                    pb_index,
                    live_time_s,
                    effective_variance_k,
                    sanitized_covariance_k,
                )
            )
        stage_wall["normalize_records"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        step_idx = len(self.measurements)
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64] | None,
                str,
                int,
                int,
            ]
        ] = []
        for iso, filt in self.filters.items():
            z_arr = np.asarray(
                [
                    float(z_k.get(iso, 0.0))
                    for z_k, _, _, _, _, _ in normalized_records
                ],
                dtype=float,
            )
            var_arr = np.asarray(
                [
                    0.0 if z_variance_k is None else float(z_variance_k.get(iso, 0.0))
                    for _, _, _, _, z_variance_k, _ in normalized_records
                ],
                dtype=float,
            )
            fe_arr = np.asarray(
                [int(fe_index) for _, fe_index, _, _, _, _ in normalized_records],
                dtype=int,
            )
            pb_arr = np.asarray(
                [int(pb_index) for _, _, pb_index, _, _, _ in normalized_records],
                dtype=int,
            )
            live_arr = np.asarray(
                [
                    float(live_time_s)
                    for _, _, _, live_time_s, _, _ in normalized_records
                ],
                dtype=float,
            )
            view_covariance = self._view_covariance_for_isotope(
                iso,
                sequence_length=z_arr.size,
                z_view_covariance_by_isotope=z_view_covariance_by_isotope,
            )
            tasks.append(
                (
                    iso,
                    filt,
                    z_arr,
                    fe_arr,
                    pb_arr,
                    live_arr,
                    var_arr,
                    view_covariance,
                    runtime_likelihood_routes[str(iso)],
                    int(pose_idx),
                    int(step_idx),
                )
            )
        stage_wall["build_isotope_tasks"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_pair_sequence_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_pair_sequence_update(task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(self._run_isotope_pair_sequence_update, tasks))
        self.last_pair_sequence_update_wall_s = time.perf_counter() - stage_start
        stage_wall["isotope_sequence_update"] = self.last_pair_sequence_update_wall_s
        stage_start = time.perf_counter()
        sequence_view_covariance = {
            str(task[0]): tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(task[7], dtype=float)
            )
            for task in tasks
            if task[7] is not None
        }
        for view_index, normalized_record in enumerate(normalized_records):
            (
                z_k,
                fe_index,
                pb_index,
                live_time_s,
                z_variance_k,
                z_covariance_k,
            ) = normalized_record
            self.measurements.append(
                MeasurementRecord(
                    z_k={iso: float(v) for iso, v in z_k.items()},
                    pose_idx=int(pose_idx),
                    live_time_s=float(live_time_s),
                    fe_index=int(fe_index),
                    pb_index=int(pb_index),
                    detector_position_xyz_m=detector_position_xyz_m,
                    station_sequence_id=int(step_idx),
                    station_view_index=int(view_index),
                    z_variance_k={
                        str(iso): (
                            0.0
                            if z_variance_k is None
                            else float(z_variance_k.get(iso, 0.0))
                        )
                        for iso in self.filters
                    },
                    z_covariance_k=z_covariance_k,
                    runtime_likelihood_route_by_isotope=runtime_likelihood_routes,
                    station_view_covariance_by_isotope=(
                        None
                        if not sequence_view_covariance
                        else dict(sequence_view_covariance)
                    ),
                )
            )
        stage_wall["append_measurements"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._apply_structural_moves()
        stage_wall["structural_moves"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        self._record_history_estimate(len(self.measurements))
        stage_wall["history_estimate"] = time.perf_counter() - stage_start
        stage_wall["total"] = time.perf_counter() - sequence_start
        self.last_pair_sequence_stage_wall_s = {
            key: float(value) for key, value in stage_wall.items()
        }

    @staticmethod
    def _run_isotope_pair_sequence_update(
        task: tuple[
            str,
            IsotopeParticleFilter,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
            str,
            int,
            int,
        ],
    ) -> None:
        """Run one isotope's same-station shield-program likelihood update."""
        (
            _isotope,
            filt,
            z_arr,
            fe_arr,
            pb_arr,
            live_arr,
            var_arr,
            view_covariance,
            runtime_likelihood_route,
            pose_idx,
            step_idx,
        ) = task
        filt.update_continuous_pair_sequence(
            z_obs=z_arr,
            pose_idx=pose_idx,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_times_s=live_arr,
            runtime_likelihood_route=runtime_likelihood_route,
            observation_count_variances=var_arr,
            observation_count_covariance=view_covariance,
            step_idx=step_idx,
        )

    @staticmethod
    def _station_view_covariance_for_records(
        isotope: str,
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.float64] | None:
        """Rebuild supplied station-view covariance for selected history rows."""
        grouped_rows: dict[
            int,
            list[
                tuple[
                    int,
                    int,
                    NDArray[np.float64],
                ]
            ],
        ] = {}
        for row_index, record in enumerate(records):
            if record.station_view_covariance_by_isotope is None:
                continue
            covariance_payload = record.station_view_covariance_by_isotope.get(
                str(isotope)
            )
            if covariance_payload is None:
                continue
            covariance = np.asarray(covariance_payload, dtype=float)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError(
                    "Stored station-view covariance must be a square matrix."
                )
            view_index = int(record.station_view_index)
            if view_index < 0 or view_index >= covariance.shape[0]:
                raise ValueError(
                    "Stored station-view index lies outside its covariance matrix."
                )
            grouped_rows.setdefault(int(record.station_sequence_id), []).append(
                (int(row_index), view_index, covariance)
            )
        if not grouped_rows:
            return None
        combined = np.zeros((len(records), len(records)), dtype=float)
        for rows in grouped_rows.values():
            reference = rows[0][2]
            if any(
                covariance.shape != reference.shape
                or not np.array_equal(covariance, reference)
                for _, _, covariance in rows[1:]
            ):
                raise ValueError(
                    "Stored station-view covariance changed within one sequence."
                )
            history_indices = np.fromiter(
                (row_index for row_index, _, _ in rows),
                dtype=np.int64,
                count=len(rows),
            )
            view_indices = np.fromiter(
                (view_index for _, view_index, _ in rows),
                dtype=np.int64,
                count=len(rows),
            )
            combined[np.ix_(history_indices, history_indices)] = reference[
                np.ix_(view_indices, view_indices)
            ]
        return combined

    @staticmethod
    def _station_sequence_ids_for_records(
        records: Sequence[MeasurementRecord],
    ) -> NDArray[np.int64]:
        """Return the required runtime-likelihood block ID for every history row."""
        return np.fromiter(
            (int(record.station_sequence_id) for record in records),
            dtype=np.int64,
            count=len(records),
        )

    def _measurement_data_for_iso(
        self,
        isotope: str,
        window: int | None,
        records: Sequence[MeasurementRecord] | None = None,
    ) -> MeasurementData | None:
        """Build measurement arrays for a single isotope with an optional window."""
        if records is None and not self.measurements:
            return None
        if records is not None:
            selected_records = list(records)
        elif window is None or window <= 0:
            selected_records = self.measurements
        else:
            selected_records = self.measurements[-int(window) :]
        if not selected_records:
            return None
        runtime_likelihood_routes = self._runtime_likelihood_routes_for_records(
            isotope,
            selected_records,
        )
        view_covariance = self._station_view_covariance_for_records(
            isotope,
            selected_records,
        )
        station_sequence_ids = self._station_sequence_ids_for_records(selected_records)
        z_list = []
        poses = []
        fe_indices = []
        pb_indices = []
        live_times = []
        variance_list = []
        for rec in selected_records:
            z_list.append(float(rec.z_k.get(isotope, 0.0)))
            if rec.z_variance_k is None:
                variance_list.append(0.0)
            else:
                variance_value = float(rec.z_variance_k.get(isotope, 0.0))
                variance_list.append(
                    max(variance_value, 0.0) if np.isfinite(variance_value) else 0.0
                )
            poses.append(rec.detector_position_xyz_m)
            live_times.append(float(rec.live_time_s))
            fe_indices.append(int(rec.fe_index))
            pb_indices.append(int(rec.pb_index))
        return MeasurementData(
            z_k=np.asarray(z_list, dtype=float),
            observation_variances=np.asarray(variance_list, dtype=float),
            detector_positions=np.asarray(poses, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            live_times=np.asarray(live_times, dtype=float),
            station_sequence_ids=station_sequence_ids,
            runtime_likelihood_routes=runtime_likelihood_routes,
            observation_count_covariance=view_covariance,
        )

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by surface-candidate diagnostics."""
        hi = np.asarray(self.pf_config.position_max, dtype=float).reshape(3)
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    @staticmethod
    def _response_design_observability_stats(
        design: NDArray[np.float64],
        *,
        eps: float,
    ) -> dict[str, float | int]:
        """Return condition and correlation statistics for a response design."""
        design_arr = np.maximum(np.asarray(design, dtype=float), 0.0)
        if design_arr.ndim != 2 or design_arr.shape[0] == 0:
            return {
                "candidate_count": int(
                    design_arr.shape[1] if design_arr.ndim == 2 else 0
                ),
                "active_candidate_count": 0,
                "weak_column_count": 0,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        column_norm = np.linalg.norm(design_arr, axis=0)
        valid = column_norm > max(float(eps), 1.0e-12)
        weak_count = int(np.count_nonzero(~valid))
        if np.count_nonzero(valid) <= 1:
            return {
                "candidate_count": int(design_arr.shape[1]),
                "active_candidate_count": int(np.count_nonzero(valid)),
                "weak_column_count": weak_count,
                "condition_number": 1.0,
                "max_abs_correlation": 0.0,
                "ambiguous_pair_count_corr_ge_0p99": 0,
                "ambiguous_pair_count_corr_ge_0p995": 0,
            }
        normalized = design_arr[:, valid] / np.maximum(column_norm[valid], eps)
        try:
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            positive = singular_values[singular_values > max(float(eps), 1.0e-12)]
            condition = (
                float(np.max(positive) / max(float(np.min(positive)), eps))
                if positive.size
                else float("inf")
            )
        except np.linalg.LinAlgError:
            condition = float("inf")
        corr = np.abs(normalized.T @ normalized)
        upper = np.triu_indices(corr.shape[0], k=1)
        upper_values = corr[upper] if upper[0].size else np.zeros(0, dtype=float)
        max_corr = float(np.max(upper_values)) if upper_values.size else 0.0
        return {
            "candidate_count": int(design_arr.shape[1]),
            "active_candidate_count": int(np.count_nonzero(valid)),
            "weak_column_count": weak_count,
            "condition_number": condition,
            "max_abs_correlation": max_corr,
            "ambiguous_pair_count_corr_ge_0p99": int(
                np.count_nonzero(upper_values >= 0.99)
            ),
            "ambiguous_pair_count_corr_ge_0p995": int(
                np.count_nonzero(upper_values >= 0.995)
            ),
        }

    def surface_candidate_observability_diagnostics(
        self,
        *,
        window: int | None = None,
        max_candidates: int = 256,
    ) -> dict[str, dict[str, Any]]:
        """Return truth-independent observability diagnostics over surface candidates."""
        diagnostics: dict[str, dict[str, Any]] = {}
        if self.candidate_sources.size == 0:
            return diagnostics
        self._ensure_kernel_cache()
        pool_all = np.asarray(self.candidate_sources, dtype=float).reshape(-1, 3)
        sample_count = max(1, min(int(max_candidates), int(pool_all.shape[0])))
        if pool_all.shape[0] > sample_count:
            sample_indices = np.linspace(
                0,
                pool_all.shape[0] - 1,
                sample_count,
                dtype=np.int64,
            )
            pool = pool_all[sample_indices]
        else:
            pool = pool_all
        env = self._source_prior_environment()
        surface_kinds = source_surface_kinds(
            pool,
            env,
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        surface_counts = {
            str(kind): int(np.count_nonzero(surface_kinds == kind))
            for kind in ("floor", "ceiling", "wall", "obstacle_side", "obstacle_top")
        }
        surface_counts["off_surface"] = int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
        eps = 1.0e-12
        for isotope, filt in self.filters.items():
            data = self._measurement_data_for_iso(isotope, window)
            if data is None or data.z_k.size == 0:
                diagnostics[isotope] = {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": 0,
                    "surface_counts": surface_counts,
                }
                continue
            candidate_counts = self._cached_expected_counts_per_source(
                filt=filt,
                isotope=isotope,
                data=data,
                sources=pool,
                strengths=np.ones(pool.shape[0], dtype=float),
            )
            variances = measurement_vector(
                data.observation_variances,
                data.z_k.size,
                "observation_variances",
                min_value=1.0,
            )
            whitened = np.asarray(candidate_counts, dtype=float) / np.sqrt(
                variances[:, None]
            )
            stats = self._response_design_observability_stats(whitened, eps=eps)
            stats.update(
                {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": int(data.z_k.size),
                    "surface_counts": surface_counts,
                    "window": None if window is None else int(window),
                }
            )
            diagnostics[isotope] = stats
        return diagnostics

    def _run_isotope_structural_update(
        self,
        task: tuple[
            str,
            IsotopeParticleFilter,
            MeasurementData | None,
        ],
    ) -> None:
        """Run one isotope's PF-native structural update."""
        _isotope, filt, evidence_data = task
        filt.apply_structural_moves(evidence_data)

    def _structural_update_worker_count(self, task_count: int) -> int:
        """Return the worker count for independent per-isotope structural updates."""
        if task_count <= 1 or not bool(self.pf_config.parallel_isotope_updates):
            return 1
        configured = self.pf_config.parallel_isotope_workers
        if configured is None:
            configured = os.cpu_count() or 1
        return max(1, min(int(configured), int(task_count)))

    def _apply_structural_moves(self) -> None:
        """Apply per-isotope structural updates using all measurement evidence."""
        structural_start = time.perf_counter()
        tasks: list[
            tuple[
                str,
                IsotopeParticleFilter,
                MeasurementData | None,
            ]
        ] = []
        for iso, filt in self.filters.items():
            structural_data = self._measurement_data_for_iso(iso, None)
            tasks.append((iso, filt, structural_data))
        worker_count = self._structural_update_worker_count(len(tasks))
        self.last_structural_update_workers = int(worker_count)
        if worker_count <= 1:
            for task in tasks:
                self._run_isotope_structural_update(task)
            self.last_structural_update_wall_s = time.perf_counter() - structural_start
            return
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(self._run_isotope_structural_update, tasks))
        self.last_structural_update_wall_s = time.perf_counter() - structural_start

    def estimates(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        estimates: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for isotope, filt in self.filters.items():
            point_estimate = posterior_point_estimate_from_states(
                [particle.state for particle in filt.continuous_particles],
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.max_sources,
                position_projector=filt._project_positions_to_source_prior,
            )
            positions = np.asarray(
                [mode.position_mean_xyz for mode in point_estimate.modes],
                dtype=float,
            ).reshape(-1, 3)
            strengths = np.asarray(
                [mode.strength_mean_cps_1m for mode in point_estimate.modes],
                dtype=float,
            )
            estimates[isotope] = (
                positions,
                strengths,
            )
        return estimates

    def posterior_source_uncertainty(
        self,
        reported_estimates: Mapping[
            str,
            Tuple[NDArray[np.float64], NDArray[np.float64]],
        ]
        | None = None,
        *,
        match_radius_m: float | None = None,
        surface_tolerance_m: float = 1.0e-5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return JSON-safe posterior 3-D diagnostics for reported source modes.

        Particle source slots are matched to the nearest reported mode in one
        batched distance calculation.  Existence mass is unconditional particle
        mass, while location, covariance, z quantiles, ellipsoid, and surface
        probabilities are conditional on a matched source being present.  Each
        mode includes availability flags so downstream evaluation can exclude
        unsupported summaries; the ellipsoid payload identifies itself as a
        Gaussian-equivalent covariance summary rather than an empirical credible
        region.
        """
        estimate_map = (
            self.estimates() if reported_estimates is None else dict(reported_estimates)
        )
        radius = 0.8 if match_radius_m is None else float(match_radius_m)
        environment = self._source_prior_environment()
        output: Dict[str, List[Dict[str, Any]]] = {}
        for isotope, estimate in estimate_map.items():
            positions = np.asarray(estimate[0], dtype=float)
            strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
            if positions.size == 0:
                positions = np.zeros((0, 3), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("reported estimate positions must have shape (M, 3).")
            if strengths.size != positions.shape[0]:
                raise ValueError(
                    "reported estimate strengths must have one value per mode."
                )
            if np.any(~np.isfinite(strengths)):
                raise ValueError("reported estimate strengths must be finite.")

            filt = self.filters.get(isotope)
            if filt is None or not filt.continuous_particles:
                packed_positions = np.zeros((0, 0, 3), dtype=float)
                packed_mask = np.zeros((0, 0), dtype=bool)
                weights = np.zeros(0, dtype=float)
            else:
                from pf import gpu_utils
                import torch

                report_states = [
                    particle.state for particle in filt.continuous_particles
                ]
                positions_tensor, _, _, mask_tensor = gpu_utils.pack_states(
                    report_states,
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
                packed_positions = positions_tensor.detach().cpu().numpy()
                packed_mask = mask_tensor.detach().cpu().numpy().astype(bool)
                weights = np.asarray(filt.continuous_weights, dtype=float)

            diagnostics = posterior_mode_uncertainty_batched(
                packed_positions,
                packed_mask,
                weights,
                positions,
                environment=environment,
                obstacle_grid=self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                match_radius_m=radius,
                surface_tolerance_m=surface_tolerance_m,
            )
            for mode_index, diagnostic in enumerate(diagnostics):
                diagnostic["reported_strength_cps_1m"] = float(strengths[mode_index])
            output[isotope] = diagnostics
        return output

    def estimate_all(
        self,
    ) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Alias for estimates() to align with visualization helpers."""
        return self.estimates()

    def step_diagnostics(
        self,
        top_k: int = 3,
        *,
        include_estimates: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/death counts, and the source
        count distribution. When include_estimates is false, the routine avoids
        the posterior point-estimate projection.
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
                    "particle_count": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "death_count": int(getattr(filt, "last_death_count", 0)),
                    "structural_timing_s": dict(
                        getattr(filt, "last_structural_timing_s", {})
                    ),
                    "temper_steps": [],
                    "temper_resamples": 0,
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "r_weighted_mean": 0.0,
                    "r_weighted_var": 0.0,
                    "r_probability_by_count": {},
                    "r_particle_count_by_count": {},
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "mmse": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "top_k": [],
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                weights = weights / total
            elif weights.size:
                weights = np.full(weights.size, 1.0 / float(weights.size), dtype=float)
            r_vals = np.array(
                [p.state.num_sources for p in filt.continuous_particles], dtype=float
            )
            if weights.size != r_vals.size and r_vals.size:
                weights = np.full(r_vals.size, 1.0 / float(r_vals.size), dtype=float)
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
            r_int = r_vals.astype(int, copy=False)
            r_weighted_mean = float(np.sum(weights * r_vals)) if r_vals.size else 0.0
            r_weighted_var = (
                float(np.sum(weights * (r_vals - r_weighted_mean) ** 2))
                if r_vals.size
                else 0.0
            )
            r_probability_by_count = {
                str(int(value)): float(np.sum(weights[r_int == int(value)]))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            r_particle_count_by_count = {
                str(int(value)): int(np.count_nonzero(r_int == int(value)))
                for value in sorted(set(int(v) for v in r_int.tolist()))
            }
            ess_pre = getattr(filt, "last_ess_pre", None)
            if ess_pre is None and weights.size:
                ess_pre = float(1.0 / max(np.sum(weights**2), eps))
            if ess_pre is None:
                ess_pre = 0.0
            resampled = bool(getattr(filt, "last_resample_ess", False))
            ess_post = getattr(filt, "last_ess_post", None)
            particle_count = int(len(filt.continuous_particles))
            best_state = filt.best_particle().state
            best_source_count = max(0, int(best_state.num_sources))
            map_positions = best_state.positions[:best_source_count].copy()
            map_strengths = best_state.strengths[:best_source_count].copy()
            if include_estimates:
                try:
                    mmse_positions, mmse_strengths = filt.estimate()
                except RuntimeError:
                    mmse_positions = np.zeros((0, 3), dtype=float)
                    mmse_strengths = np.zeros(0, dtype=float)
            else:
                mmse_positions = np.zeros((0, 3), dtype=float)
                mmse_strengths = np.zeros(0, dtype=float)
            top_entries: List[Dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.continuous_particles[int(idx)].state
                    source_count = max(0, int(state.num_sources))
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": source_count,
                            "positions": state.positions[:source_count].copy(),
                            "strengths": state.strengths[:source_count].copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "particle_count": particle_count,
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "death_count": int(getattr(filt, "last_death_count", 0)),
                "structural_timing_s": dict(
                    getattr(filt, "last_structural_timing_s", {})
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "r_mean": r_mean,
                "r_var": r_var,
                "r_weighted_mean": r_weighted_mean,
                "r_weighted_var": r_weighted_var,
                "r_probability_by_count": r_probability_by_count,
                "r_particle_count_by_count": r_particle_count_by_count,
                "map": (map_positions, map_strengths),
                "mmse": (mmse_positions, mmse_strengths),
                "top_k": top_entries,
            }
        return diagnostics

    @property
    def num_orientations(self) -> int:
        """Return the number of shield orientation normals."""
        return self.normals.shape[0]

    def count_likelihood_spec_for_isotope(
        self,
        isotope: str,
    ) -> CountLikelihoodSpec:
        """Return the normalized runtime count-likelihood spec for an isotope."""
        filt = self.filters[str(isotope)]
        return CountLikelihoodSpec(**filt._count_likelihood_kwargs())

    @staticmethod
    def _sample_planning_count_observations_np(
        selected_lambdas: NDArray[np.float64],
        predictive_variance: NDArray[np.float64] | float,
        *,
        spec: CountLikelihoodSpec,
        rng: np.random.Generator,
        epsilon: float = 1.0e-12,
    ) -> NDArray[np.float64]:
        """Sample non-negative future counts from the configured planning model."""
        lambdas = np.maximum(np.asarray(selected_lambdas, dtype=float), epsilon)
        if spec.model == "poisson":
            return np.asarray(rng.poisson(lambdas), dtype=float)
        scale = np.sqrt(
            np.maximum(np.asarray(predictive_variance, dtype=float), epsilon)
        )
        if spec.model == "gaussian":
            observations = rng.normal(loc=lambdas, scale=scale)
        else:
            df = max(float(spec.student_t_df), 1.0 + epsilon)
            observations = lambdas + scale * rng.standard_t(
                df,
                size=lambdas.shape,
            )
        return np.maximum(np.asarray(observations, dtype=float), 0.0)

    @staticmethod
    def _sample_planning_count_observations_torch(
        selected_lambdas: "torch.Tensor",
        predictive_variance: "torch.Tensor",
        *,
        spec: CountLikelihoodSpec,
        epsilon: float = 1.0e-12,
    ) -> "torch.Tensor":
        """Return the Torch equivalent of planning count observation sampling."""
        import torch

        lambdas = torch.clamp(selected_lambdas.to(dtype=torch.float64), min=epsilon)
        if spec.model == "poisson":
            return torch.poisson(lambdas)
        scale = torch.sqrt(
            torch.clamp(
                predictive_variance.to(device=lambdas.device, dtype=torch.float64),
                min=epsilon,
            )
        )
        if spec.model == "gaussian":
            observations = lambdas + scale * torch.randn_like(lambdas)
        else:
            df = max(float(spec.student_t_df), 1.0 + epsilon)
            distribution = torch.distributions.StudentT(
                torch.as_tensor(df, device=lambdas.device, dtype=torch.float64)
            )
            noise = distribution.sample(lambdas.shape)
            observations = lambdas + scale * noise
        return torch.clamp(observations, min=0.0)

    @staticmethod
    def _planning_eig_from_lambdas_np(
        lambdas_ap: NDArray[np.float64],
        weights_p: NDArray[np.float64],
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        rng: np.random.Generator,
        observations_as: NDArray[np.float64] | None = None,
        epsilon: float = 1.0e-12,
    ) -> NDArray[np.float64]:
        """Return batched action EIG using the configured count likelihood."""
        lambdas = np.maximum(np.asarray(lambdas_ap, dtype=float), epsilon)
        if lambdas.ndim == 1:
            lambdas = lambdas.reshape(1, -1)
        if lambdas.ndim != 2:
            raise ValueError("lambdas_ap must have shape action x particle.")
        weights = np.maximum(np.asarray(weights_p, dtype=float).reshape(-1), 0.0)
        if weights.size != lambdas.shape[1]:
            raise ValueError("weights_p must have one value per particle.")
        weight_sum = float(np.sum(weights))
        if weight_sum <= epsilon:
            weights = np.full(weights.size, 1.0 / max(weights.size, 1), dtype=float)
        else:
            weights = weights / weight_sum
        log_weights = np.log(weights + epsilon)
        h_prior = -float(np.sum(weights * log_weights))
        action_count = int(lambdas.shape[0])
        if observations_as is None:
            sample_count = int(num_samples)
            if sample_count <= 0:
                return np.full(action_count, h_prior, dtype=float)
            sample_indices = rng.choice(
                weights.size,
                size=(action_count, sample_count),
                replace=True,
                p=weights,
            )
            selected = np.take_along_axis(lambdas, sample_indices, axis=1)
            predictive_variance = predictive_count_likelihood_variance(
                selected,
                spec=spec,
                epsilon=epsilon,
            )
            observations = (
                RotatingShieldPFEstimator._sample_planning_count_observations_np(
                    selected,
                    predictive_variance,
                    spec=spec,
                    rng=rng,
                    epsilon=epsilon,
                )
            )
        else:
            observations = np.asarray(observations_as, dtype=float)
            if observations.ndim == 1 and action_count == 1:
                observations = observations.reshape(1, -1)
            if observations.ndim != 2 or observations.shape[0] != action_count:
                raise ValueError("observations_as must have shape action x sample.")
            if observations.shape[1] == 0:
                return np.full(action_count, h_prior, dtype=float)
        likelihood_terms = count_log_likelihood_terms_np(
            observations[:, :, None],
            lambdas[:, None, :],
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = log_weights[None, None, :] + likelihood_terms
        log_posterior -= logsumexp(log_posterior, axis=2, keepdims=True)
        posterior = np.exp(log_posterior)
        h_post = -np.sum(
            posterior * np.log(posterior + epsilon),
            axis=2,
        )
        return np.full(action_count, h_prior, dtype=float) - np.mean(h_post, axis=1)

    @staticmethod
    def _planning_eig_from_lambdas_torch(
        lambdas_ap: "torch.Tensor",
        weights_p: "torch.Tensor",
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        observations_as: "torch.Tensor | None" = None,
        epsilon: float = 1.0e-12,
    ) -> "torch.Tensor":
        """Return Torch action EIG equivalent to the batched NumPy helper."""
        import torch

        lambdas = torch.clamp(lambdas_ap.to(dtype=torch.float64), min=epsilon)
        if lambdas.ndim == 1:
            lambdas = lambdas.reshape(1, -1)
        if lambdas.ndim != 2:
            raise ValueError("lambdas_ap must have shape action x particle.")
        weights = torch.clamp(
            weights_p.to(device=lambdas.device, dtype=torch.float64).reshape(-1),
            min=0.0,
        )
        if int(weights.numel()) != int(lambdas.shape[1]):
            raise ValueError("weights_p must have one value per particle.")
        weight_sum = torch.sum(weights)
        if float(weight_sum.detach().cpu().item()) <= epsilon:
            weights = torch.full_like(weights, 1.0 / max(int(weights.numel()), 1))
        else:
            weights = weights / weight_sum
        log_weights = torch.log(weights + epsilon)
        h_prior = -torch.sum(weights * log_weights)
        action_count = int(lambdas.shape[0])
        if observations_as is None:
            sample_count = int(num_samples)
            if sample_count <= 0:
                return torch.full(
                    (action_count,),
                    float(h_prior.detach().cpu().item()),
                    device=lambdas.device,
                    dtype=torch.float64,
                )
            sample_indices = torch.multinomial(
                weights.expand(action_count, -1),
                sample_count,
                replacement=True,
            )
            selected = torch.gather(lambdas, 1, sample_indices)
            predictive_variance = predictive_count_likelihood_variance_torch(
                selected,
                spec=spec,
                epsilon=epsilon,
            )
            observations = (
                RotatingShieldPFEstimator._sample_planning_count_observations_torch(
                    selected,
                    predictive_variance,
                    spec=spec,
                    epsilon=epsilon,
                )
            )
        else:
            observations = observations_as.to(
                device=lambdas.device,
                dtype=torch.float64,
            )
            if observations.ndim == 1 and action_count == 1:
                observations = observations.reshape(1, -1)
            if observations.ndim != 2 or int(observations.shape[0]) != action_count:
                raise ValueError("observations_as must have shape action x sample.")
            if int(observations.shape[1]) == 0:
                return h_prior.expand(action_count).clone()
        likelihood_terms = count_log_likelihood_terms_torch(
            observations.unsqueeze(2),
            lambdas.unsqueeze(1),
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = log_weights.view(1, 1, -1) + likelihood_terms
        log_posterior = log_posterior - torch.logsumexp(
            log_posterior,
            dim=2,
            keepdim=True,
        )
        posterior = torch.exp(log_posterior)
        h_post = -torch.sum(
            posterior * torch.log(posterior + epsilon),
            dim=2,
        )
        return h_prior - torch.mean(h_post, dim=1)

    @staticmethod
    def _expected_strength_uncertainty_from_lambdas_np(
        lambdas_p: NDArray[np.float64],
        weights_p: NDArray[np.float64],
        strengths_pm: NDArray[np.float64],
        *,
        spec: CountLikelihoodSpec,
        num_samples: int,
        rng: np.random.Generator,
        epsilon: float = 1.0e-12,
    ) -> float:
        """Return batched expected posterior strength variance for one action."""
        lambdas = np.maximum(np.asarray(lambdas_p, dtype=float).reshape(-1), epsilon)
        weights = np.maximum(np.asarray(weights_p, dtype=float).reshape(-1), 0.0)
        strengths = np.asarray(strengths_pm, dtype=float)
        if strengths.ndim != 2 or strengths.shape[0] != lambdas.size:
            raise ValueError("strengths_pm must have one row per particle.")
        if weights.size != lambdas.size:
            raise ValueError("weights_p must have one value per particle.")
        sample_count = int(num_samples)
        if sample_count <= 0 or strengths.size == 0:
            return 0.0
        weight_sum = float(np.sum(weights))
        if weight_sum <= epsilon:
            weights = np.full(weights.size, 1.0 / max(weights.size, 1), dtype=float)
        else:
            weights = weights / weight_sum
        sample_indices = rng.choice(
            weights.size,
            size=sample_count,
            replace=True,
            p=weights,
        )
        selected_lambdas = lambdas[sample_indices]
        predictive_variance = predictive_count_likelihood_variance(
            selected_lambdas,
            spec=spec,
            epsilon=epsilon,
        )
        observations = RotatingShieldPFEstimator._sample_planning_count_observations_np(
            selected_lambdas,
            predictive_variance,
            spec=spec,
            rng=rng,
            epsilon=epsilon,
        )
        likelihood_terms = count_log_likelihood_terms_np(
            observations[:, None],
            lambdas[None, :],
            spec=spec,
            epsilon=epsilon,
        )
        log_posterior = np.log(weights + epsilon)[None, :] + likelihood_terms
        log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior_mean = posterior @ strengths
        posterior_second = posterior @ (strengths * strengths)
        posterior_variance = np.maximum(
            posterior_second - posterior_mean * posterior_mean,
            0.0,
        )
        return float(np.mean(np.sum(posterior_variance, axis=1)))

    def orientation_expected_information_gain(
        self,
        pose_idx: int,
        RFe: NDArray[np.float64],
        RPb: NDArray[np.float64],
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
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
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        fe_idx = octant_index_from_rotation(RFe)
        pb_idx = octant_index_from_rotation(RPb)
        if not self._can_use_gpu():
            return self._orientation_expected_information_gain_cpu(
                pose_idx=pose_idx,
                detector_pos=detector_pos,
                fe_idx=fe_idx,
                pb_idx=pb_idx,
                live_time_s=live_time_s,
                num_samples=int(num_samples),
                alpha_by_isotope=alpha_by_isotope,
                particles_by_isotope=particles_by_isotope,
                rng=rng,
                eps=eps,
            )
        kernel = self._continuous_kernel()
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

        def _compute_lam_torch(
            states: Sequence[IsotopeState], isotope: str
        ) -> "torch.Tensor":
            """Compute expected counts for a state subset on the torch backend."""
            if not states:
                return torch.zeros(0, device=device, dtype=dtype)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            return kernel.expected_counts_pair_for_packed_states_torch(
                isotope=isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                live_time_s=live_time_s,
                source_scale=self.response_scale_for_isotope(
                    isotope,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                ),
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
            ig_h = float(
                self._planning_eig_from_lambdas_torch(
                    lam_t,
                    weights_t,
                    spec=self.count_likelihood_spec_for_isotope(iso),
                    num_samples=int(num_samples),
                    epsilon=eps,
                )[0]
                .detach()
                .cpu()
                .item()
            )
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)

    def orientation_expected_information_gain_grid(
        self,
        pose_idx: int,
        *,
        live_time_s: float = 1.0,
        num_samples: int | None = None,
        alpha_by_isotope: Dict[str, float] | None = None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute MC-EIG for all Fe/Pb orientation pairs using shared lambdas.

        This evaluates the same likelihood-entropy estimator as
        ``orientation_expected_information_gain`` but avoids recomputing the
        continuous expected-count kernel separately for every orientation pair.
        """
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        num_orients = int(self.num_orientations)
        num_pairs = num_orients * num_orients
        if not self._can_use_gpu():
            detector_pos = np.asarray(
                self.kernel_cache.poses[int(pose_idx)],
                dtype=float,
            )
            sample_count = (
                self.pf_config.eig_num_samples
                if num_samples is None
                else int(num_samples)
            )
            eps = 1.0e-12
            alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
            alpha_sum = sum(float(value) for value in alphas.values()) or 1.0
            alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
            rng = np.random.default_rng()
            scores = np.zeros(num_pairs, dtype=float)
            for iso, filt in self.filters.items():
                if particles_by_isotope is not None and iso in particles_by_isotope:
                    states, weights = particles_by_isotope[iso]
                else:
                    if not filt.continuous_particles:
                        continue
                    states = [particle.state for particle in filt.continuous_particles]
                    weights = filt.continuous_weights
                if not states:
                    continue
                lambdas = self.expected_counts_all_pairs_for_states_at_detector(
                    isotope=iso,
                    detector_pos=detector_pos,
                    live_time_s=float(live_time_s),
                    states=states,
                )
                scores += float(alphas.get(iso, 0.0)) * (
                    self._planning_eig_from_lambdas_np(
                        lambdas,
                        np.asarray(weights, dtype=float),
                        spec=self.count_likelihood_spec_for_isotope(iso),
                        num_samples=sample_count,
                        rng=rng,
                        epsilon=eps,
                    )
                )
            return scores.reshape(num_orients, num_orients)

        detector_pos = np.asarray(self.kernel_cache.poses[int(pose_idx)], dtype=float)
        num_samples = (
            self.pf_config.eig_num_samples if num_samples is None else num_samples
        )
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(v) for v in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
        self._gpu_enabled()
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        iso_data: Dict[
            str,
            tuple["torch.Tensor", "torch.Tensor", CountLikelihoodSpec],
        ] = {}
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
            weights_arr = np.asarray(weights, dtype=float)
            weight_sum = float(np.sum(weights_arr))
            if weight_sum <= eps:
                weights_arr = np.ones(len(states), dtype=float) / max(len(states), 1)
            else:
                weights_arr = weights_arr / weight_sum
            lambdas_np = self.expected_counts_all_pairs_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                live_time_s=float(live_time_s),
                states=states,
            )
            lam_all = torch.as_tensor(lambdas_np, device=device, dtype=dtype)
            weights_t = torch.as_tensor(weights_arr, device=device, dtype=dtype)
            weight_sum_t = torch.sum(weights_t)
            if float(weight_sum_t.detach().cpu().item()) <= 0.0:
                weights_t = torch.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
            else:
                weights_t = weights_t / weight_sum_t
            iso_data[iso] = (
                lam_all,
                weights_t,
                self.count_likelihood_spec_for_isotope(iso),
            )
        if not iso_data:
            return np.zeros((num_orients, num_orients), dtype=float)

        scores_t = torch.zeros(num_pairs, device=device, dtype=torch.float64)
        for iso, (lam_all, weights_t, spec) in iso_data.items():
            scores_t = scores_t + float(alphas.get(iso, 0.0)) * (
                self._planning_eig_from_lambdas_torch(
                    lam_all,
                    weights_t,
                    spec=spec,
                    num_samples=int(num_samples),
                    epsilon=eps,
                )
            )
        return scores_t.detach().cpu().numpy().reshape(num_orients, num_orients)

    def _orientation_expected_information_gain_cpu(
        self,
        *,
        pose_idx: int,
        detector_pos: NDArray[np.float64],
        fe_idx: int,
        pb_idx: int,
        live_time_s: float,
        num_samples: int,
        alpha_by_isotope: Dict[str, float] | None,
        particles_by_isotope: Dict[str, Tuple[List[IsotopeState], NDArray[np.float64]]]
        | None,
        rng: np.random.Generator,
        eps: float,
    ) -> float:
        """Compute orientation EIG on CPU using the same expected-count kernel."""
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(float(value) for value in alphas.values()) or 1.0
        alphas = {key: float(value) / alpha_sum for key, value in alphas.items()}
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
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr = weights_arr / max(float(np.sum(weights_arr)), eps)
            lam = self.expected_counts_pair_for_states_at_detector(
                isotope=iso,
                detector_pos=detector_pos,
                fe_index=int(fe_idx),
                pb_index=int(pb_idx),
                live_time_s=float(live_time_s),
                states=states,
            )
            lam = np.maximum(np.asarray(lam, dtype=float).reshape(-1), eps)
            ig_h = float(
                self._planning_eig_from_lambdas_np(
                    lam,
                    weights_arr,
                    spec=self.count_likelihood_spec_for_isotope(iso),
                    num_samples=int(num_samples),
                    rng=rng,
                    epsilon=eps,
                )[0]
            )
            total_ig += alphas.get(iso, 0.0) * ig_h
        return float(total_ig)

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
        # Planning rollouts must never advance or reseed the global NumPy stream
        # used by the sequential PF.  Otherwise a live planner call between two
        # observations makes same-seed MeasurementLog replay diverge.
        rng = (
            np.random.default_rng()
            if rng_seed is None
            else np.random.default_rng(int(rng_seed))
        )
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
        rollout_method = (
            self.pf_config.planning_rollout_method or self.pf_config.planning_method
        )

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
            """Simulate isotope-wise observations from each runtime count model."""
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
                    spec = estimator.count_likelihood_spec_for_isotope(iso)
                    selected_lambda = np.asarray(lam[idx], dtype=float)
                    predictive_variance = predictive_count_likelihood_variance(
                        selected_lambda,
                        spec=spec,
                    )
                    z_k[iso] = float(
                        estimator._sample_planning_count_observations_np(
                            selected_lambda,
                            predictive_variance,
                            spec=spec,
                            rng=rng_local,
                        )
                    )
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
        if not self.pf_config.use_gpu:
            return None
        self._gpu_enabled()
        from pf import gpu_utils
        import torch
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
        num_fe = len(RFe_candidates)
        num_pb = len(RPb_candidates)
        fe_indices = np.repeat(np.arange(num_fe), num_pb)
        pb_indices = np.tile(np.arange(num_pb), num_fe)
        eps = 1e-12
        alphas = alpha_by_isotope or {iso: 1.0 for iso in self.filters}
        alpha_sum = sum(alphas.values()) or 1.0
        alphas = {k: v / alpha_sum for k, v in alphas.items()}
        device = gpu_utils.resolve_device(self.pf_config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.pf_config.gpu_dtype)
        planning_subset = self.planning_particles(
            max_particles=rollout_particles,
            method=rollout_method,
            rng=rng,
        )

        iso_data: Dict[str, Dict[str, Any]] = {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                continue
            if iso in planning_subset and planning_subset[iso][0]:
                states, weights = planning_subset[iso]
            else:
                states = [p.state for p in filt.continuous_particles]
                weights = np.asarray(filt.continuous_weights, dtype=float)
            weights = np.asarray(weights, dtype=float)
            if weights.size == 0 or not states:
                continue
            weights = weights / max(np.sum(weights), eps)
            positions, strengths, backgrounds, mask = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            lam_all = filt.continuous_kernel.expected_counts_all_pairs_for_packed_states_torch(
                isotope=iso,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                live_time_s=live_time_per_rot_s,
                source_scale=self.response_scales_for_measurements(
                    iso,
                    fe_indices,
                    pb_indices,
                ),
                device=device,
                dtype=dtype,
            )
            iso_data[iso] = {
                "lam": lam_all,
                "strengths": strengths,
                "weights": weights,
                "num_particles": weights.size,
                "resample_threshold": filt.config.resample_threshold,
                "likelihood_spec": self.count_likelihood_spec_for_isotope(iso),
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
            if (
                max_particles is None
                or max_particles <= 0
                or max_particles >= len(weights)
            ):
                return indices, weights
            method = method or "top_weight"
            if method == "top_weight":
                sel = np.argsort(weights)[::-1][:max_particles]
                sel_weights = weights[sel]
                sel_weights = sel_weights / max(np.sum(sel_weights), eps)
                return indices[sel], sel_weights
            if method == "resample":
                sel = rng_local.choice(len(weights), size=max_particles, p=weights)
                sel_weights = np.ones(max_particles, dtype=float) / max(
                    max_particles, 1
                )
                return indices[sel], sel_weights
            raise ValueError(f"Unknown planning particle selection method: {method}")

        def _ig_scores_from_lam(
            lam_all: "torch.Tensor",
            subset_indices: NDArray[np.int64],
            subset_weights: NDArray[np.float64],
            num_samples: int,
            spec: CountLikelihoodSpec,
        ) -> "torch.Tensor":
            """Compute model-aware IG for all orientations from cached lambdas."""
            idx_t = torch.as_tensor(
                subset_indices, device=lam_all.device, dtype=torch.long
            )
            lam_sel = torch.index_select(lam_all, 1, idx_t)
            weights_t = torch.as_tensor(
                subset_weights, device=lam_all.device, dtype=lam_all.dtype
            )
            return self._planning_eig_from_lambdas_torch(
                lam_sel,
                weights_t,
                spec=spec,
                num_samples=int(num_samples),
                epsilon=eps,
            )

        def _update_weights(
            lam_curr: NDArray[np.float64],
            weights: NDArray[np.float64],
            z_obs: float,
            spec: CountLikelihoodSpec,
        ) -> NDArray[np.float64]:
            """Update rollout weights using the configured count likelihood."""
            likelihood_terms = count_log_likelihood_terms_np(
                np.asarray(z_obs, dtype=float),
                lam_curr,
                spec=spec,
                epsilon=eps,
            )
            logw = np.log(weights + eps) + likelihood_terms
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
                        spec=data["likelihood_spec"],
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
                        spec = data["likelihood_spec"]
                        selected_lambda = np.asarray(lam_curr[idx], dtype=float)
                        predictive_variance = predictive_count_likelihood_variance(
                            selected_lambda,
                            spec=spec,
                            epsilon=eps,
                        )
                        z_obs = float(
                            self._sample_planning_count_observations_np(
                                selected_lambda,
                                predictive_variance,
                                spec=spec,
                                rng=rng,
                                epsilon=eps,
                            )
                        )
                    weights = _update_weights(
                        lam_curr,
                        weights,
                        z_obs,
                        data["likelihood_spec"],
                    )
                    ess = 1.0 / max(np.sum(weights**2), eps)
                    if ess < float(data["resample_threshold"]) * len(weights):
                        resampled = systematic_resample(
                            np.log(weights + eps),
                            rng=rng,
                        )
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
            _, strengths_t, _, _ = gpu_utils.pack_states(
                states, device=device, dtype=dtype
            )
            weights = torch.as_tensor(
                filt.continuous_weights, device=device, dtype=dtype
            )
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
            max_r = max(
                (p.state.num_sources for p in filt.continuous_particles), default=0
            )
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
        ig_grid = self.orientation_expected_information_gain_grid(
            pose_idx=pose_idx,
            live_time_s=live_time_s,
        )
        max_ig = float(np.max(ig_grid)) if ig_grid.size else 0.0
        dwell_time = sum(
            rec.live_time_s for rec in self.measurements if rec.pose_idx == pose_idx
        )
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
