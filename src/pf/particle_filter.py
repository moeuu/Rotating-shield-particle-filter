"""Coordinate the per-isotope particle filter main loop (predict, update, resample)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Tuple
from collections import deque
import os
import time

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

from measurement.model import EnvironmentConfig
from measurement.kernels import KernelPrecomputer, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    generate_octant_orientations,
    resolve_mu_values,
)
from measurement.source_surfaces import (
    project_positions_to_allowed_surfaces,
)
from measurement.surface_patches import (
    SurfacePatchDictionary,
    build_surface_patch_dictionary,
)
from pf.defaults import DEFAULT_MAX_SOURCES_PER_ISOTOPE
from pf.diagnostics import build_source_event_record, reset_step_diagnostics
from pf.likelihood import (
    CountLikelihoodSpec,
    OBSERVATION_COUNT_VARIANCE_ADDITIONAL,
    OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL,
    count_log_likelihood,
    count_likelihood_variance,
    count_likelihood_variance_torch,
    expected_counts_per_source,
    normalize_count_likelihood_model,
    normalize_observation_count_variance_semantics,
)
from pf.posterior import posterior_point_estimate_from_states
from pf.state import IsotopeState
from pf.resampling import systematic_resample, systematic_resample_count
from pf.strength_prior import StrengthPrior
from pf.structural_rj import (
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    SurfaceAdjacency,
    SurfaceSetPrior,
    add_surface_indices,
    birth_log_acceptance_ratio,
    conditional_birth_surface_log_probability,
    death_log_acceptance_ratio,
    local_position_log_acceptance_ratio,
    remove_surface_columns,
    uniform_death_index_log_probability,
)

if TYPE_CHECKING:
    import torch


def _pf_debug_timing_enabled() -> bool:
    """Return True when verbose PF phase timing should be printed."""
    return os.environ.get("PF_DEBUG_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass
class PFConfig:
    """Particle filter configuration (Sec. 3.4)."""

    num_particles: int = 200
    min_particles: int | None = None
    max_particles: int | None = None
    max_sources: int | None = DEFAULT_MAX_SOURCES_PER_ISOTOPE
    resample_threshold: float = 0.5  # relative to N
    background_level: float | dict[str, float] = 0.0
    measurement_scale_by_isotope: dict[str, float] | None = None
    measurement_scale_by_isotope_and_pair: dict[str, dict[int, float]] | None = None
    count_likelihood_model: str = "poisson"
    transport_model_rel_sigma: float | dict[str, float] = 0.0
    transport_model_abs_sigma: float | dict[str, float] = 0.0
    spectrum_count_rel_sigma: float | dict[str, float] = 0.0
    spectrum_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_abs_sigma: float | dict[str, float] = 0.0
    low_count_transition_counts: float | dict[str, float] = 0.0
    observation_count_variance_includes_counting_noise: bool = False
    observation_count_variance_semantics: str = ""
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
    direct_spectrum_likelihood_enable: bool = True
    spectrum_likelihood_bin_chunk: int = 512
    min_strength: float = 0.01
    birth_enable: bool = True
    structural_rj_patch_spacing_m: float = 1.0
    structural_rj_move_probability: float = 1.0
    structural_rj_birth_probability: float = 0.5
    structural_rj_death_probability: float = 0.5
    structural_rj_position_move_probability: float = 1.0
    structural_rj_local_position_move_probability: float = 1.0
    structural_rj_strength_move_probability: float = 1.0
    structural_cardinality_prior_probs: tuple[float, ...] | None = None
    ess_low: float = 0.5
    ess_high: float = 0.9
    target_ess_ratio: float = 0.5
    max_temper_steps: int = 16
    min_delta_beta: float = 1e-3
    use_tempering: bool = True
    max_resamples_per_observation: int = 2
    temper_resample_cooldown_steps: int = 2
    temper_resample_force_ratio: float = 0.1
    adapt_cooldown_steps: int = 0
    # Continuous PF priors (Sec. 3.3.2)
    position_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_max: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    source_position_prior: str = "surface"
    init_num_sources: Tuple[int, int] = (
        0,
        DEFAULT_MAX_SOURCES_PER_ISOTOPE,
    )
    # Strength prior (cps@1m scale). The uniform form is used when a simulation
    # declares its source-population bounds before evaluation.
    init_strength_prior: str = "lognormal"
    init_strength_min: float = 0.0
    init_strength_max: float | None = None
    init_strength_log_mean: float = 9.0  # exp(9) ~ 8e3
    init_strength_log_sigma: float = 1.0
    use_gpu: bool = True
    gpu_device: str = "cuda"
    gpu_dtype: str = "float32"
    converge_enable: bool = False
    converge_window: int = 8
    converge_map_move_eps_m: float = 0.4
    converge_ess_ratio_high: float = 0.2
    converge_ll_improve_eps: float = 1e5
    converge_min_steps: int = 30
    converge_require_all: bool = True
    converge_cardinality_var_max: float = 0.05
    converge_min_stations: int = 0

    def __post_init__(self) -> None:
        """Normalize the exact surface-PF configuration and likelihood semantics."""
        self.num_particles = int(self.num_particles)
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive.")
        if self.min_particles is not None:
            self.min_particles = int(self.min_particles)
            if self.min_particles < 1:
                raise ValueError("min_particles must be positive when provided.")
        if self.max_particles is not None:
            self.max_particles = int(self.max_particles)
            if self.max_particles < 1:
                raise ValueError("max_particles must be positive when provided.")
        if (
            self.min_particles is not None
            and self.max_particles is not None
            and self.min_particles > self.max_particles
        ):
            raise ValueError("min_particles cannot exceed max_particles.")
        self.structural_rj_patch_spacing_m = float(self.structural_rj_patch_spacing_m)
        if (
            not np.isfinite(self.structural_rj_patch_spacing_m)
            or self.structural_rj_patch_spacing_m <= 0.0
        ):
            raise ValueError(
                "structural_rj_patch_spacing_m must be finite and positive."
            )
        for probability_name in (
            "structural_rj_move_probability",
            "structural_rj_birth_probability",
            "structural_rj_death_probability",
            "structural_rj_position_move_probability",
            "structural_rj_local_position_move_probability",
            "structural_rj_strength_move_probability",
        ):
            probability = float(getattr(self, probability_name))
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{probability_name} must lie in [0, 1].")
            setattr(self, probability_name, probability)
        if self.max_sources is None or int(self.max_sources) < 1:
            raise ValueError("Pure PF requires a finite positive max_sources.")
        self.max_sources = int(self.max_sources)
        if not self._source_prior_name_is_surface(self.source_position_prior):
            raise ValueError("Pure PF requires source_position_prior='surface'.")
        self.source_position_prior = "surface"
        if self.structural_cardinality_prior_probs is not None:
            cardinality_prior = np.asarray(
                self.structural_cardinality_prior_probs,
                dtype=float,
            ).reshape(-1)
            if (
                cardinality_prior.size != self.max_sources + 1
                or np.any(~np.isfinite(cardinality_prior))
                or np.any(cardinality_prior <= 0.0)
            ):
                raise ValueError(
                    "structural_cardinality_prior_probs must contain "
                    "max_sources + 1 finite positive values."
                )
            cardinality_prior /= float(np.sum(cardinality_prior))
            self.structural_cardinality_prior_probs = tuple(
                float(value) for value in cardinality_prior
            )
        initial_lower, initial_upper = (
            int(self.init_num_sources[0]),
            int(self.init_num_sources[1]),
        )
        if self.birth_enable:
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
                "Structural moves disabled requires a fixed "
                "init_num_sources=(K, K) within zero through max_sources."
            )
        self.init_num_sources = (initial_lower, initial_upper)
        semantics = normalize_observation_count_variance_semantics(
            self.observation_count_variance_semantics,
            includes_counting_noise=(
                self.observation_count_variance_includes_counting_noise
            ),
        )
        self.observation_count_variance_semantics = semantics
        self.observation_count_variance_includes_counting_noise = (
            semantics != OBSERVATION_COUNT_VARIANCE_ADDITIONAL
        )
        self.direct_spectrum_likelihood_enable = bool(
            self.direct_spectrum_likelihood_enable
        )
        self.shield_contrast_likelihood_enable = bool(
            self.shield_contrast_likelihood_enable
        )
        self.shield_view_ratio_likelihood_enable = bool(
            self.shield_view_ratio_likelihood_enable
        )
        if semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
            model = normalize_count_likelihood_model(self.count_likelihood_model)
            if model == "poisson":
                raise ValueError(
                    "complete_statistical observation variance requires gaussian "
                    "or student_t count likelihood."
                )
            self.shield_contrast_likelihood_enable = False
            self.shield_view_ratio_likelihood_enable = False

    @staticmethod
    def _source_prior_name_is_surface(value: object) -> bool:
        """Return whether a configuration value denotes surface support."""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower().replace("-", "_") in {
            "surface",
            "surfaces",
            "surface_constrained",
        }


@dataclass
class PFConvergenceMonitor:
    """Track per-isotope convergence statistics over a sliding window."""

    window: int
    min_steps: int
    map_move_eps_m: float
    ess_ratio_high: float
    ll_improve_eps: float
    require_all: bool

    def __post_init__(self) -> None:
        """Initialize bounded convergence-history buffers."""
        self.positions: deque[NDArray[np.float64] | None] = deque(maxlen=self.window)
        self.ess_ratios: deque[float] = deque(maxlen=self.window)
        self.ll_values: deque[float] = deque(maxlen=self.window)

    def update_stats(
        self,
        step_idx: int,
        pos: NDArray[np.float64] | None,
        ess_ratio: float,
        ll_value: float,
    ) -> None:
        """Append the latest statistics to the window."""
        if step_idx < 0:
            return
        self.positions.append(pos.copy() if pos is not None else None)
        self.ess_ratios.append(float(ess_ratio))
        self.ll_values.append(float(ll_value))

    def is_converged(self, step_idx: int) -> bool:
        """Return True if all convergence criteria are satisfied."""
        if step_idx < self.min_steps:
            return False
        if len(self.positions) < self.window:
            return False
        if any(pos is None for pos in self.positions):
            return False
        pos_list = [pos for pos in self.positions if pos is not None]
        reference_shape = pos_list[0].shape
        if any(pos.shape != reference_shape for pos in pos_list):
            return False
        max_move = 0.0
        for prev, curr in zip(pos_list[:-1], pos_list[1:]):
            max_move = max(max_move, float(np.linalg.norm(curr - prev)))
        move_ok = max_move <= float(self.map_move_eps_m)
        ess_ok = min(self.ess_ratios) >= float(self.ess_ratio_high)
        ll_span = max(self.ll_values) - min(self.ll_values)
        ll_ok = ll_span <= float(self.ll_improve_eps)
        if self.require_all:
            return move_ok and ess_ok and ll_ok
        return sum([move_ok, ess_ok, ll_ok]) >= 2


@dataclass
class IsotopeParticle:
    """Continuous-state particle (Sec. 3.3.2)."""

    state: IsotopeState
    log_weight: float


@dataclass(frozen=True)
class MeasurementData:
    """Bundle full-history measurement arrays for structural PF moves."""

    z_k: NDArray[np.float64]
    observation_variances: NDArray[np.float64]
    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]
    station_sequence_ids: NDArray[np.int64] | None = None
    runtime_likelihood_routes: NDArray[np.str_] | None = None
    observation_count_covariance: NDArray[np.float64] | None = None
    spectrum_counts: NDArray[np.float64] | None = None
    spectrum_response_template: NDArray[np.float64] | None = None
    spectrum_background: NDArray[np.float64] | None = None
    spectrum_variance: NDArray[np.float64] | None = None
    spectrum_variance_present: NDArray[np.bool_] | None = None


class IsotopeParticleFilter:
    """Per-isotope particle filter (continuous state is the primary mode)."""

    def __init__(
        self,
        isotope: str,
        kernel: KernelPrecomputer | None,
        config: PFConfig | None = None,
        obstacle_grid: ObstacleGrid | None = None,
        obstacle_height_m: float = 2.0,
        obstacle_mu_by_isotope: dict[str, float] | None = None,
        obstacle_buildup_coeff: float = 0.0,
        detector_radius_m: float = 0.0,
        detector_aperture_radius_m: float | None = None,
        detector_aperture_samples: int = 1,
        detector_aperture_sampling: str = "solid_angle_cone",
        source_extent_radius_m: float = 0.0,
        source_extent_samples: int = 1,
        line_mu_by_isotope: dict[str, object] | None = None,
        transport_response_model: dict[str, object] | None = None,
    ) -> None:
        """Initialize particle state, priors, and continuous measurement kernels."""
        self.isotope = isotope
        self.kernel = kernel
        self.config = config or PFConfig()
        self.N = self.config.num_particles
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
        self._random_generator = np.random.default_rng(
            int(np.random.randint(0, np.iinfo(np.uint32).max))
        )
        self._strength_prior = self._build_strength_prior()
        self._structural_rj_surface_patches: SurfacePatchDictionary | None = None
        self._structural_rj_patch_key_to_index: dict[
            tuple[float, float, float], int
        ] = {}
        self._structural_rj_cardinality_prior_probs = (
            self._build_structural_cardinality_prior()
        )
        self._structural_rj_cardinality_prior: CardinalityPrior | None = None
        self._structural_rj_surface_prior: SurfaceSetPrior | None = None
        self._structural_rj_surface_adjacency: SurfaceAdjacency | None = None
        self._structural_rj_move_probabilities: BirthDeathMoveProbabilities | None = (
            None
        )
        self._structural_rj_response_cache: NDArray[np.float64] | None = None
        self._structural_rj_response_cache_signatures: NDArray[np.float64] | None = None
        self._structural_rj_response_evaluation_batches = 0
        self._structural_rj_response_evaluated_cells = 0
        self._structural_rj_response_touched_mask: NDArray[np.bool_] | None = None
        self._structural_rj_move_counts: dict[str, int] = {}
        self._initialize_structural_rj_surface_support()
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        )
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
        )
        self._continuous_kernel_physics_signature = (
            self._incoming_kernel_physics_signature(kernel)
        )
        self.continuous_particles: List[IsotopeParticle] = []
        self.last_ess: float | None = None
        self.last_ess_pre: float | None = None
        self.last_ess_post: float | None = None
        self.last_resample_ess = False
        self.last_resample_count = 0
        self.last_birth_count = 0
        self.last_death_count = 0
        self.last_n_after_adapt: int | None = None
        self.last_temper_steps: list[dict[str, float]] = []
        self.last_temper_resample_count = 0
        self.last_source_event_diagnostics: list[dict[str, object]] = []
        self.last_structural_timing_s: dict[str, float] = {}
        self.last_spectrum_likelihood_route = "none"
        self._deferred_resampled_any = False
        self._deferred_ess_min: float | None = None
        self._deferred_convergence_args: (
            tuple[
                int | None,
                NDArray[np.float64],
                int,
                int,
                float,
                float,
            ]
            | None
        ) = None
        self._adapt_cooldown_remaining = 0
        self._resample_count_in_observation = 0
        self._observed_station_labels: set[tuple[float, float]] = set()
        self.is_converged = False
        self._converge_monitor = (
            PFConvergenceMonitor(
                window=int(self.config.converge_window),
                min_steps=int(self.config.converge_min_steps),
                map_move_eps_m=float(self.config.converge_map_move_eps_m),
                ess_ratio_high=float(self.config.converge_ess_ratio_high),
                ll_improve_eps=float(self.config.converge_ll_improve_eps),
                require_all=bool(self.config.converge_require_all),
            )
            if self.config.converge_enable
            else None
        )
        self._init_continuous_particles()

    def _variable_cardinality_enabled(self) -> bool:
        """Return whether exact finite-surface RJ-MH rejuvenation is active."""
        return bool(self.config.birth_enable)

    def _build_strength_prior(self) -> StrengthPrior:
        """Build the normalized strength prior shared by initialization and moves."""
        return StrengthPrior(
            kind=str(self.config.init_strength_prior),
            minimum=float(self.config.init_strength_min),
            maximum=(
                None
                if self.config.init_strength_max is None
                else float(self.config.init_strength_max)
            ),
            log_mean=float(self.config.init_strength_log_mean),
            log_sigma=float(self.config.init_strength_log_sigma),
        )

    def _build_structural_cardinality_prior(self) -> NDArray[np.float64]:
        """Return the normalized prior mass for cardinalities zero through max."""
        max_sources = self.config.max_sources
        if max_sources is None:
            return np.zeros(0, dtype=float)
        count = max(0, int(max_sources)) + 1
        configured = self.config.structural_cardinality_prior_probs
        if configured is None:
            return np.full(count, 1.0 / max(count, 1), dtype=float)
        probabilities = np.asarray(configured, dtype=float).reshape(-1)
        if probabilities.size != count:
            raise ValueError(
                "structural_cardinality_prior_probs must have max_sources + 1 entries."
            )
        return probabilities / float(np.sum(probabilities))

    @staticmethod
    def _surface_patch_key(position: NDArray[np.float64]) -> tuple[float, float, float]:
        """Return a stable exact-mode lookup key for one patch center."""
        values = np.asarray(position, dtype=float).reshape(3)
        rounded = np.round(values, decimals=12)
        return float(rounded[0]), float(rounded[1]), float(rounded[2])

    def _initialize_structural_rj_surface_support(self) -> None:
        """Build complete area-aware surface support for exact structural moves."""
        position_min = np.asarray(self.config.position_min, dtype=float).reshape(3)
        if not np.allclose(position_min, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "rj_mh currently requires position_min=(0, 0, 0) so the "
                "environment surface measure is unambiguous."
            )
        patches = build_surface_patch_dictionary(
            self._source_prior_environment(),
            self.obstacle_grid,
            float(self.config.structural_rj_patch_spacing_m),
            obstacle_height_m=self.obstacle_height_m,
        )
        if not patches.obstacle_surfaces_available:
            warning = patches.obstacle_geometry_warning or (
                "Obstacle component surfaces are unavailable."
            )
            raise ValueError(
                f"rj_mh requires complete obstacle component geometry: {warning}"
            )
        if patches.patch_count <= int(self.config.max_sources or 0):
            raise ValueError(
                "rj_mh surface dictionary must contain more patches than max_sources."
            )
        keys = [
            self._surface_patch_key(position)
            for position in np.asarray(patches.centers_xyz, dtype=float)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "rj_mh surface dictionary contains duplicate patch centers."
            )
        self._structural_rj_surface_patches = patches
        self._structural_rj_patch_key_to_index = {
            key: int(index) for index, key in enumerate(keys)
        }
        max_sources = int(self.config.max_sources or 0)
        self._structural_rj_surface_prior = SurfaceSetPrior(
            patches.areas_m2,
            max_cardinality=max_sources,
        )
        self._structural_rj_surface_adjacency = SurfaceAdjacency(
            dictionary_size=patches.patch_count,
            edges=patches.adjacency_edges,
        )
        self._structural_rj_cardinality_prior = CardinalityPrior(
            self._structural_rj_cardinality_prior_probs
        )
        self._structural_rj_move_probabilities = BirthDeathMoveProbabilities(
            max_cardinality=max_sources,
            birth_weight=float(self.config.structural_rj_birth_probability),
            death_weight=float(self.config.structural_rj_death_probability),
        )

    def _structural_rj_patch_indices_for_state(
        self,
        state: IsotopeState,
    ) -> NDArray[np.int64]:
        """Resolve and validate the canonical surface indices of one exact state."""
        source_count = int(state.num_sources)
        positions = np.asarray(state.positions, dtype=float).reshape(-1, 3)
        if positions.shape[0] != source_count:
            raise ValueError("rj_mh state positions must match num_sources.")
        if source_count == 0:
            return np.zeros(0, dtype=np.int64)
        try:
            indices = np.asarray(
                [
                    self._structural_rj_patch_key_to_index[
                        self._surface_patch_key(position)
                    ]
                    for position in positions
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "rj_mh state contains a position outside the finite surface dictionary."
            ) from exc
        if np.unique(indices).size != indices.size:
            raise ValueError(
                "rj_mh does not permit duplicate surface patches in one state."
            )
        return indices

    def _canonicalize_structural_rj_state(
        self,
        state: IsotopeState,
    ) -> NDArray[np.int64]:
        """Sort one exact state by surface index and return the sorted indices."""
        indices = self._structural_rj_patch_indices_for_state(state)
        if indices.size <= 1:
            return indices
        order = np.argsort(indices, kind="stable")
        if not np.array_equal(order, np.arange(indices.size)):
            state.positions = np.asarray(state.positions, dtype=float)[order]
            state.strengths = np.asarray(state.strengths, dtype=float)[order]
            indices = indices[order]
        return np.asarray(indices, dtype=np.int64)

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the continuous kernel with the filter's environment attenuation settings."""
        kernel_kwargs: dict[str, object] = {}
        orientations = getattr(self.kernel, "orientations", None)
        if orientations is not None and len(orientations) > 1:
            kernel_kwargs["orientations"] = orientations
        return ContinuousKernel(
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
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
            **kernel_kwargs,
        )

    def _incoming_kernel_physics_signature(
        self,
        kernel: KernelPrecomputer | None,
    ) -> tuple[object, ...]:
        """Return canonical incoming physics that affects this isotope's kernel."""
        shield_params = (
            getattr(kernel, "shield_params", ShieldParams())
            if kernel is not None
            else ShieldParams()
        )
        mu_by_isotope = (
            getattr(kernel, "mu_by_isotope", None) if kernel is not None else None
        )
        mu_fe, mu_pb = resolve_mu_values(
            mu_by_isotope,
            self.isotope,
            default_fe=float(shield_params.mu_fe),
            default_pb=float(shield_params.mu_pb),
        )
        incoming_orientations = (
            getattr(kernel, "orientations", None) if kernel is not None else None
        )
        orientations = (
            generate_octant_orientations()
            if incoming_orientations is None or len(incoming_orientations) <= 1
            else np.asarray(incoming_orientations, dtype=np.float64)
        )
        orientation_array = np.asarray(
            orientations,
            dtype=np.float64,
        )
        canonical_orientations = np.ascontiguousarray(
            np.where(orientation_array == 0.0, 0.0, orientation_array),
            dtype="<f8",
        )
        shield_signature = (
            float(shield_params.mu_pb),
            float(shield_params.mu_fe),
            float(shield_params.thickness_pb_cm),
            float(shield_params.thickness_fe_cm),
            float(shield_params.inner_radius_fe_cm),
            float(shield_params.inner_radius_pb_cm),
            max(float(shield_params.buildup_fe_coeff), 0.0),
            max(float(shield_params.buildup_pb_coeff), 0.0),
            str(shield_params.shield_geometry_model),
            bool(shield_params.use_angle_attenuation),
        )
        return (
            (float(mu_fe), float(mu_pb)),
            shield_signature,
            canonical_orientations.shape,
            canonical_orientations.tobytes(order="C"),
        )

    def _measurement_source_scale(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> float:
        """Return the isotope-specific source response scale for PF likelihoods."""
        pair_id = self._measurement_shield_pair_id(
            fe_index=fe_index,
            pb_index=pb_index,
            shield_pair_id=shield_pair_id,
        )
        pair_scales = self.config.measurement_scale_by_isotope_and_pair
        if pair_id is not None and isinstance(pair_scales, Mapping):
            isotope_pair_scales = pair_scales.get(str(self.isotope), {})
            if isinstance(isotope_pair_scales, Mapping):
                value = isotope_pair_scales.get(int(pair_id))
                if value is None:
                    value = isotope_pair_scales.get(str(int(pair_id)))  # type: ignore[arg-type]
                if value is not None:
                    return max(float(value), 0.0)
        scales = self.config.measurement_scale_by_isotope
        if not isinstance(scales, dict):
            return 1.0
        return max(float(scales.get(self.isotope, 1.0)), 0.0)

    def _measurement_source_scale_vector(
        self,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return one source response scale for each measurement pair."""
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        if fe_arr.size != pb_arr.size:
            raise ValueError("fe_indices and pb_indices must have matching length.")
        return np.asarray(
            [
                self._measurement_source_scale(
                    fe_index=int(fe),
                    pb_index=int(pb),
                )
                for fe, pb in zip(fe_arr, pb_arr)
            ],
            dtype=float,
        )

    def _measurement_shield_pair_id(
        self,
        *,
        fe_index: int | None = None,
        pb_index: int | None = None,
        shield_pair_id: int | None = None,
    ) -> int | None:
        """Return the configured shield-pair id when orientation indices exist."""
        if shield_pair_id is not None:
            return int(shield_pair_id)
        if fe_index is None or pb_index is None:
            return None
        orientations = getattr(self.kernel, "orientations", None)
        num_orientations = len(orientations) if orientations is not None else 8
        return int(fe_index) * max(int(num_orientations), 1) + int(pb_index)

    @staticmethod
    def _measurement_vector(
        values: float | NDArray[np.float64],
        count: int,
        name: str,
        *,
        min_value: float | None = None,
        allow_scalar: bool = True,
    ) -> NDArray[np.float64]:
        """Return a validated one-value-per-measurement vector."""
        expected = max(int(count), 0)
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            if expected == 0:
                return np.zeros(0, dtype=float)
            raise ValueError(f"{name} must contain one value per measurement.")
        if arr.size == 1 and expected != 1 and allow_scalar:
            arr = np.full(expected, float(arr[0]), dtype=float)
        elif arr.size != expected:
            scalar_text = "scalar or " if allow_scalar else ""
            raise ValueError(f"{name} must be {scalar_text}one value per measurement.")
        if min_value is not None:
            arr = np.maximum(arr, float(min_value))
        return np.asarray(arr, dtype=float)

    def _measurement_rows(
        self, data: MeasurementData, mask: NDArray[np.bool_]
    ) -> MeasurementData:
        """Return a measurement bundle restricted to the selected row mask."""
        row_mask = np.asarray(mask, dtype=bool).reshape(-1)
        covariance = data.observation_count_covariance
        restricted_covariance = None
        if covariance is not None:
            covariance_arr = np.asarray(covariance, dtype=float)
            expected_shape = (int(row_mask.size), int(row_mask.size))
            if covariance_arr.shape != expected_shape:
                raise ValueError("observation_count_covariance must be shaped K x K.")
            restricted_covariance = covariance_arr[np.ix_(row_mask, row_mask)]
        restricted_sequence_ids = None
        if data.station_sequence_ids is not None:
            sequence_ids = np.asarray(
                data.station_sequence_ids,
                dtype=np.int64,
            ).reshape(-1)
            if sequence_ids.size != row_mask.size:
                raise ValueError(
                    "station_sequence_ids must contain one ID per measurement."
                )
            restricted_sequence_ids = sequence_ids[row_mask]
        restricted_routes = None
        if data.runtime_likelihood_routes is not None:
            routes = np.asarray(
                data.runtime_likelihood_routes,
                dtype=str,
            ).reshape(-1)
            if routes.size != row_mask.size:
                raise ValueError(
                    "runtime_likelihood_routes must contain one route per measurement."
                )
            restricted_routes = routes[row_mask]
        return MeasurementData(
            z_k=np.asarray(data.z_k, dtype=float)[row_mask],
            observation_variances=np.asarray(data.observation_variances, dtype=float)[
                row_mask
            ],
            detector_positions=np.asarray(data.detector_positions, dtype=float)[
                row_mask
            ],
            fe_indices=np.asarray(data.fe_indices, dtype=int)[row_mask],
            pb_indices=np.asarray(data.pb_indices, dtype=int)[row_mask],
            live_times=np.asarray(data.live_times, dtype=float)[row_mask],
            station_sequence_ids=restricted_sequence_ids,
            runtime_likelihood_routes=restricted_routes,
            observation_count_covariance=restricted_covariance,
            spectrum_counts=(
                None
                if data.spectrum_counts is None
                else np.asarray(data.spectrum_counts, dtype=float)[row_mask]
            ),
            spectrum_response_template=(
                None
                if data.spectrum_response_template is None
                else np.asarray(
                    data.spectrum_response_template,
                    dtype=float,
                )[row_mask]
            ),
            spectrum_background=(
                None
                if data.spectrum_background is None
                else np.asarray(data.spectrum_background, dtype=float)[row_mask]
            ),
            spectrum_variance=(
                None
                if data.spectrum_variance is None
                else np.asarray(data.spectrum_variance, dtype=float)[row_mask]
            ),
            spectrum_variance_present=(
                None
                if data.spectrum_variance_present is None
                else np.asarray(
                    data.spectrum_variance_present,
                    dtype=bool,
                )[row_mask]
            ),
        )

    @staticmethod
    def _optional_arrays_equal(
        first: NDArray[np.float64] | None,
        second: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether two optional arrays are both absent or exactly equal."""
        if first is None or second is None:
            return first is None and second is None
        return bool(np.array_equal(first, second))

    @staticmethod
    def _same_measurement_block(
        first: MeasurementData | None,
        second: MeasurementData | None,
    ) -> bool:
        """Return whether two bundles contain exactly the same response rows."""
        if first is second:
            return first is not None
        if first is None or second is None:
            return False
        return bool(
            np.array_equal(first.z_k, second.z_k)
            and np.array_equal(
                first.observation_variances,
                second.observation_variances,
            )
            and np.array_equal(
                first.detector_positions,
                second.detector_positions,
            )
            and np.array_equal(first.fe_indices, second.fe_indices)
            and np.array_equal(first.pb_indices, second.pb_indices)
            and np.array_equal(first.live_times, second.live_times)
            and IsotopeParticleFilter._optional_arrays_equal(
                first.station_sequence_ids,
                second.station_sequence_ids,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.runtime_likelihood_routes,
                second.runtime_likelihood_routes,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.observation_count_covariance,
                second.observation_count_covariance,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_counts,
                second.spectrum_counts,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_response_template,
                second.spectrum_response_template,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_background,
                second.spectrum_background,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_variance,
                second.spectrum_variance,
            )
            and IsotopeParticleFilter._optional_arrays_equal(
                first.spectrum_variance_present,
                second.spectrum_variance_present,
            )
        )

    @staticmethod
    def _isotope_float_config_for(
        value: float | Mapping[str, float],
        isotope: str,
        default: float = 0.0,
    ) -> float:
        """Resolve a scalar or isotope-indexed float for one isotope."""
        if isinstance(value, Mapping):
            return max(float(value.get(str(isotope), default)), 0.0)
        return max(float(value), 0.0)

    def _isotope_float_config(
        self, value: float | dict[str, float], default: float = 0.0
    ) -> float:
        """Resolve a scalar or isotope-indexed float config value."""
        return self._isotope_float_config_for(
            value,
            self.isotope,
            default,
        )

    def _count_likelihood_kwargs(self) -> dict[str, bool | float | str]:
        """Return likelihood keyword arguments for this isotope filter."""
        return {
            "model": str(self.config.count_likelihood_model),
            "transport_model_rel_sigma": self._isotope_float_config(
                self.config.transport_model_rel_sigma,
            ),
            "transport_model_abs_sigma": self._isotope_float_config(
                self.config.transport_model_abs_sigma,
            ),
            "spectrum_count_rel_sigma": self._isotope_float_config(
                self.config.spectrum_count_rel_sigma,
            ),
            "spectrum_count_abs_sigma": self._isotope_float_config(
                self.config.spectrum_count_abs_sigma,
            ),
            "low_count_abs_sigma": self._isotope_float_config(
                self.config.low_count_abs_sigma,
            ),
            "low_count_transition_counts": self._isotope_float_config(
                self.config.low_count_transition_counts,
            ),
            "observation_count_variance_includes_counting_noise": bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            "observation_count_variance_semantics": str(
                self.config.observation_count_variance_semantics
            ),
            "student_t_df": max(float(self.config.count_likelihood_df), 1.0),
        }

    def count_likelihood_spec(self) -> CountLikelihoodSpec:
        """Return the resolved isotope-specific likelihood configuration."""
        return CountLikelihoodSpec(**self._count_likelihood_kwargs())

    def _count_log_likelihood_np(
        self,
        z_k: NDArray[np.float64],
        lambda_k: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> float:
        """Evaluate this filter's configured count log-likelihood in NumPy."""
        return count_log_likelihood(
            z_k,
            lambda_k,
            observation_count_variance=observation_count_variance,
            **self._count_likelihood_kwargs(),
        )

    def _count_log_likelihood_matrix_np(
        self,
        z_k: NDArray[np.float64],
        lambda_kp: NDArray[np.float64],
        observation_count_variance: float | NDArray[np.float64] = 0.0,
    ) -> NDArray[np.float64]:
        """Evaluate per-particle count log-likelihoods for a KxP lambda matrix."""
        z_arr = np.asarray(z_k, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != z_arr.size:
            raise ValueError("lambda_kp must have one row per measurement.")
        obs_var = np.asarray(observation_count_variance, dtype=float).reshape(-1)
        if obs_var.size == 0:
            obs_var = np.zeros(z_arr.size, dtype=float)
        elif obs_var.size == 1:
            obs_var = np.full(z_arr.size, float(obs_var[0]), dtype=float)
        elif obs_var.size != z_arr.size:
            raise ValueError(
                "observation_count_variance must be scalar or have one value per measurement."
            )
        kwargs = self._count_likelihood_kwargs()
        model = normalize_count_likelihood_model(str(kwargs["model"]))
        z_col = z_arr[:, None]
        if model == "poisson":
            return np.sum(z_col * np.log(lam) - lam, axis=0)
        variance = count_likelihood_variance(
            z_col,
            lam,
            transport_model_rel_sigma=float(kwargs["transport_model_rel_sigma"]),
            transport_model_abs_sigma=float(kwargs["transport_model_abs_sigma"]),
            spectrum_count_rel_sigma=float(kwargs["spectrum_count_rel_sigma"]),
            spectrum_count_abs_sigma=float(kwargs["spectrum_count_abs_sigma"]),
            low_count_abs_sigma=float(kwargs["low_count_abs_sigma"]),
            low_count_transition_counts=float(kwargs["low_count_transition_counts"]),
            observation_count_variance=obs_var[:, None],
            observation_count_variance_includes_counting_noise=bool(
                kwargs["observation_count_variance_includes_counting_noise"]
            ),
            observation_count_variance_semantics=str(
                kwargs["observation_count_variance_semantics"]
            ),
        )
        residual = z_col - lam
        if model == "gaussian":
            terms = -0.5 * ((residual**2) / variance + np.log(variance))
            return np.sum(terms, axis=0)
        df = max(float(kwargs["student_t_df"]), 1.0 + 1.0e-12)
        terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
        terms -= 0.5 * np.log(variance)
        return np.sum(terms, axis=0)

    def _structural_effective_variance_np(
        self,
        z_k: NDArray[np.float64],
        lambda_kp: NDArray[np.float64],
        observation_count_variance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the configured marginal count variance for structural evidence."""
        z_arr = np.asarray(z_k, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != z_arr.size:
            raise ValueError("lambda_kp must have one row per measurement.")
        obs_var = self._measurement_vector(
            observation_count_variance,
            z_arr.size,
            "observation_count_variance",
            min_value=0.0,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "poisson":
            return lam
        kwargs = self._count_likelihood_kwargs()
        return count_likelihood_variance(
            z_arr[:, None],
            lam,
            transport_model_rel_sigma=float(kwargs["transport_model_rel_sigma"]),
            transport_model_abs_sigma=float(kwargs["transport_model_abs_sigma"]),
            spectrum_count_rel_sigma=float(kwargs["spectrum_count_rel_sigma"]),
            spectrum_count_abs_sigma=float(kwargs["spectrum_count_abs_sigma"]),
            low_count_abs_sigma=float(kwargs["low_count_abs_sigma"]),
            low_count_transition_counts=float(kwargs["low_count_transition_counts"]),
            observation_count_variance=obs_var[:, None],
            observation_count_variance_includes_counting_noise=bool(
                kwargs["observation_count_variance_includes_counting_noise"]
            ),
            observation_count_variance_semantics=str(
                kwargs["observation_count_variance_semantics"]
            ),
        )

    def _structural_spectrum_log_likelihood_matrix_np(
        self,
        lambda_kp: NDArray[np.float64],
        observed_spectrum_kb: NDArray[np.float64],
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
        spectrum_variance_kb: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the runtime direct-spectrum likelihood for structural states."""
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        observed = np.asarray(observed_spectrum_kb, dtype=float)
        template = np.asarray(response_template_kb, dtype=float)
        background = np.asarray(background_spectrum_kb, dtype=float)
        if observed.ndim != 2 or observed.shape[0] != lam.shape[0]:
            raise ValueError("observed spectrum rows must match expected counts.")
        if template.shape != observed.shape or background.shape != observed.shape:
            raise ValueError(
                "spectrum template and background must match observations."
            )
        spectrum_variance = None
        if spectrum_variance_kb is not None:
            spectrum_variance = np.asarray(spectrum_variance_kb, dtype=float)
            if spectrum_variance.shape != observed.shape:
                raise ValueError("spectrum variance must match observed bins.")

        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        rel_sigma = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        abs_sigma = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        chunk_size = max(1, int(self.config.spectrum_likelihood_bin_chunk))
        result = np.zeros(int(lam.shape[1]), dtype=float)
        for start in range(0, int(observed.shape[1]), chunk_size):
            stop = min(start + chunk_size, int(observed.shape[1]))
            expected = np.maximum(
                lam[:, None, :] * template[:, start:stop, None]
                + background[:, start:stop, None],
                1.0e-12,
            )
            observed_chunk = observed[:, start:stop, None]
            if model == "poisson" and spectrum_variance is None:
                result += np.sum(
                    observed_chunk * np.log(expected) - expected,
                    axis=(0, 1),
                )
                continue
            observation_variance = (
                0.0
                if spectrum_variance is None
                else spectrum_variance[:, start:stop, None]
            )
            variance = np.maximum(
                expected
                + observation_variance
                + (float(rel_sigma) * expected) ** 2
                + float(abs_sigma) ** 2,
                1.0e-12,
            )
            residual = observed_chunk - expected
            if model == "gaussian":
                terms = -0.5 * ((residual**2) / variance + np.log(variance))
            else:
                df = max(
                    float(self.config.count_likelihood_df),
                    1.0 + 1.0e-12,
                )
                terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * variance))
                terms -= 0.5 * np.log(variance)
            result += np.sum(terms, axis=(0, 1))
        return result

    def _shield_shape_log_likelihood_batch_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        row_indices_bk: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Return runtime-equivalent shield-shape evidence for station batches."""
        rows = np.asarray(row_indices_bk, dtype=np.int64)
        lam_all = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if rows.ndim != 2 or lam_all.ndim != 2:
            raise ValueError("Station rows and expected counts must be matrices.")
        block_count, view_count = rows.shape
        particle_count = int(lam_all.shape[1])
        result = np.zeros((block_count, particle_count), dtype=float)
        if block_count == 0 or particle_count == 0:
            return np.zeros(particle_count, dtype=float)

        z_bk = np.asarray(data.z_k, dtype=float)[rows]
        var_bk = np.asarray(data.observation_variances, dtype=float)[rows]
        lam_bkp = lam_all[rows, :]

        contrast_weight = max(
            float(self.config.shield_contrast_likelihood_weight),
            0.0,
        )
        contrast_min_views = max(
            int(self.config.shield_contrast_min_views),
            2,
        )
        if (
            bool(self.config.shield_contrast_likelihood_enable)
            and contrast_weight > 0.0
            and view_count >= contrast_min_views
        ):
            min_count = max(
                float(self.config.shield_contrast_min_count),
                1.0e-6,
            )
            sigma_floor = max(
                float(self.config.shield_contrast_log_sigma_floor),
                1.0e-6,
            )
            sigma_ceiling = max(
                float(self.config.shield_contrast_log_sigma_ceiling),
                sigma_floor,
            )
            df = max(
                float(self.config.shield_contrast_likelihood_df),
                1.0 + 1.0e-12,
            )
            z_safe = np.maximum(z_bk, min_count)
            lam_safe = np.maximum(lam_bkp, min_count)
            log_z = np.log(z_safe)[:, :, None]
            log_lam = np.log(lam_safe)
            log_var = np.clip(
                var_bk / np.maximum(z_safe**2, 1.0e-12) + sigma_floor**2,
                sigma_floor**2,
                sigma_ceiling**2,
            )[:, :, None]
            view_weight = np.reciprocal(log_var)
            weight_sum = np.maximum(np.sum(view_weight, axis=1, keepdims=True), 1e-12)
            observed_center = log_z - (
                np.sum(view_weight * log_z, axis=1, keepdims=True) / weight_sum
            )
            predicted_center = log_lam - (
                np.sum(view_weight * log_lam, axis=1, keepdims=True) / weight_sum
            )
            residual = observed_center - predicted_center
            terms = -0.5 * (df + 1.0) * np.log1p((residual**2) / (df * log_var))
            terms -= 0.5 * np.log(log_var)
            result += contrast_weight * np.sum(terms, axis=1)

        ratio_weight = max(
            float(self.config.shield_view_ratio_likelihood_weight),
            0.0,
        )
        ratio_min_views = max(
            int(self.config.shield_view_ratio_likelihood_min_views),
            2,
        )
        if (
            bool(self.config.shield_view_ratio_likelihood_enable)
            and ratio_weight > 0.0
            and view_count >= ratio_min_views
        ):
            concentration = max(
                float(self.config.shield_view_ratio_likelihood_concentration),
                1.0e-6,
            )
            min_total = max(
                float(self.config.shield_view_ratio_likelihood_min_total_count),
                0.0,
            )
            z_nonnegative = np.maximum(
                np.where(np.isfinite(z_bk), z_bk, 0.0),
                0.0,
            )
            totals = np.sum(z_nonnegative, axis=1)
            valid = np.isfinite(totals) & (totals >= min_total)
            if np.any(valid):
                lam_valid = lam_bkp[valid]
                probabilities = lam_valid / np.maximum(
                    np.sum(lam_valid, axis=1, keepdims=True),
                    1.0e-12,
                )
                alpha = np.maximum(concentration * probabilities, 1.0e-12)
                alpha0 = np.maximum(np.sum(alpha, axis=1), 1.0e-12)
                total_valid = totals[valid, None]
                ratio_ll = gammaln(alpha0) - gammaln(alpha0 + total_valid)
                ratio_ll += np.sum(
                    gammaln(alpha + z_nonnegative[valid, :, None]) - gammaln(alpha),
                    axis=1,
                )
                result[valid] += ratio_weight * ratio_ll
        return np.sum(result, axis=0)

    def _structural_shield_shape_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return shield-shape terms grouped exactly by station sequence."""
        lam = np.asarray(lambda_kp, dtype=float)
        if lam.ndim == 1:
            lam = lam[:, None]
        result = np.zeros(int(lam.shape[1]), dtype=float)
        if int(data.z_k.size) < 2 or not (
            self.config.shield_contrast_likelihood_enable
            or self.config.shield_view_ratio_likelihood_enable
        ):
            return result
        blocks_by_length = self._station_likelihood_block_rows(data)
        for block_length, rows in blocks_by_length.items():
            if int(block_length) <= 1:
                continue
            result += self._shield_shape_log_likelihood_batch_np(
                data,
                lam,
                rows,
            )
        return result

    @staticmethod
    def _contiguous_station_blocks(
        detector_positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return start indices and lengths for contiguous same-position views."""
        positions = np.asarray(detector_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            count = int(positions.shape[0]) if positions.ndim > 0 else 0
            return (
                np.arange(count, dtype=np.int64),
                np.ones(count, dtype=np.int64),
            )
        count = int(positions.shape[0])
        if count == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        rounded = np.round(positions, decimals=8)
        changes = np.any(rounded[1:] != rounded[:-1], axis=1)
        starts = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.flatnonzero(changes).astype(np.int64) + 1,
            ]
        )
        stops = np.concatenate([starts[1:], np.array([count], dtype=np.int64)])
        return starts, stops - starts

    def _station_likelihood_block_rows(
        self,
        data: MeasurementData,
    ) -> dict[int, NDArray[np.int64]]:
        """
        Return station likelihood rows batched by equal block length.

        Estimator-produced data carries explicit sequence IDs that reproduce the
        runtime update boundary exactly. Directly constructed legacy bundles
        without IDs retain the contiguous same-position fallback.
        """
        measurement_count = int(data.z_k.size)
        sequence_ids = data.station_sequence_ids
        if sequence_ids is None:
            starts, lengths = self._contiguous_station_blocks(data.detector_positions)
            labels = np.repeat(
                np.arange(starts.size, dtype=np.int64),
                lengths,
            )
        else:
            ids = np.asarray(sequence_ids, dtype=np.int64).reshape(-1)
            if ids.size != measurement_count:
                raise ValueError(
                    "station_sequence_ids must contain one ID per measurement."
                )
            if measurement_count == 0:
                return {}
            _, labels = np.unique(ids, return_inverse=True)
            labels = labels.astype(np.int64, copy=False)
        if labels.size != measurement_count:
            raise ValueError("Station likelihood block labels are inconsistent.")
        if measurement_count == 0:
            return {}
        _, inverse, lengths = np.unique(
            labels,
            return_inverse=True,
            return_counts=True,
        )
        rows_by_length: dict[int, NDArray[np.int64]] = {}
        for block_length in np.unique(lengths):
            selected_blocks = np.flatnonzero(lengths == int(block_length))
            membership = inverse[None, :] == selected_blocks[:, None]
            rows = np.nonzero(membership)[1].reshape(
                int(selected_blocks.size),
                int(block_length),
            )
            rows_by_length[int(block_length)] = rows.astype(
                np.int64,
                copy=False,
            )
        return rows_by_length

    @staticmethod
    def _regularize_station_covariance_np(
        covariance_bpkk: NDArray[np.float64],
        diagonal_bpk: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return positive-definite station covariance using the runtime policy.

        Each station is regularized independently because runtime observes one
        station sequence at a time. Particles remain batched within a station,
        matching ``_regularize_sequence_covariance_torch`` exactly.
        """
        covariance = np.asarray(covariance_bpkk, dtype=float)
        diagonal = np.asarray(diagonal_bpk, dtype=float)
        if covariance.ndim != 4 or diagonal.shape != covariance.shape[:3]:
            raise ValueError("Station covariance arrays have incompatible shapes.")
        size = int(covariance.shape[-1])
        eye = np.eye(size, dtype=float).reshape(1, size, size)
        regularized_blocks: list[NDArray[np.float64]] = []
        for block_index in range(int(covariance.shape[0])):
            block = covariance[block_index]
            block_diagonal = diagonal[block_index]
            diag_scale = np.maximum(np.mean(block_diagonal, axis=1), 1.0)
            jitter = 1.0e-9 * diag_scale
            accepted: NDArray[np.float64] | None = None
            for _ in range(6):
                candidate = block + jitter[:, None, None] * eye
                try:
                    np.linalg.cholesky(candidate)
                except np.linalg.LinAlgError:
                    jitter = jitter * 10.0
                    continue
                accepted = candidate
                break
            if accepted is None:
                fallback = np.zeros_like(block, dtype=float)
                diag_indices = np.arange(size)
                fallback[:, diag_indices, diag_indices] = np.maximum(
                    block_diagonal,
                    1.0e-12,
                )
                accepted = fallback + jitter[:, None, None] * eye
            regularized_blocks.append(accepted)
        if not regularized_blocks:
            return np.zeros_like(covariance, dtype=float)
        return np.stack(regularized_blocks, axis=0)

    def _station_covariance_log_likelihood_batch_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        row_indices_bk: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Return summed likelihoods for a batch of equal-size station blocks.

        Arrays are batched over both station blocks and PF particles. No scalar
        particle, candidate, source-slot, or shield-view loop is used.
        """
        rows = np.asarray(row_indices_bk, dtype=np.int64)
        lam_all = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if rows.ndim != 2 or lam_all.ndim != 2:
            raise ValueError("Station rows and expected counts must be matrices.")
        block_count, view_count = rows.shape
        particle_count = int(lam_all.shape[1])
        z_bk = np.asarray(data.z_k, dtype=float)[rows]
        obs_var_bk = np.asarray(data.observation_variances, dtype=float)[rows]
        lam_bkp = lam_all[rows, :]
        variance_bkp = self._structural_effective_variance_np(
            z_bk.reshape(-1),
            lam_bkp.reshape(block_count * view_count, particle_count),
            obs_var_bk.reshape(-1),
        ).reshape(block_count, view_count, particle_count)
        variance_bpk = np.transpose(variance_bkp, (0, 2, 1))
        covariance = np.zeros(
            (block_count, particle_count, view_count, view_count),
            dtype=float,
        )
        diag_indices = np.arange(view_count)
        covariance[:, :, diag_indices, diag_indices] = variance_bpk

        supplied = data.observation_count_covariance
        if supplied is not None:
            supplied_arr = np.asarray(supplied, dtype=float)
            expected_shape = (int(data.z_k.size), int(data.z_k.size))
            if supplied_arr.shape != expected_shape:
                raise ValueError("observation_count_covariance must be shaped K x K.")
            supplied_blocks = supplied_arr[rows[:, :, None], rows[:, None, :]]
            supplied_blocks = 0.5 * (
                supplied_blocks + np.swapaxes(supplied_blocks, 1, 2)
            )
            supplied_blocks[:, diag_indices, diag_indices] = 0.0
            covariance += supplied_blocks[:, None, :, :]

        fraction = max(
            float(self.config.station_view_correlated_spectrum_fraction),
            0.0,
        )
        if bool(self.config.station_view_covariance_enable) and fraction > 0.0:
            spectrum_rel = fraction * self._isotope_float_config(
                self.config.spectrum_count_rel_sigma
            )
            spectrum_abs = fraction * self._isotope_float_config(
                self.config.spectrum_count_abs_sigma
            )
            lam_bpk = np.transpose(lam_bkp, (0, 2, 1))
            common = (spectrum_rel**2) * (
                lam_bpk[:, :, :, None] * lam_bpk[:, :, None, :]
            )
            if spectrum_abs > 0.0:
                common = common + spectrum_abs**2
            common[:, :, diag_indices, diag_indices] = 0.0
            covariance += common

        covariance = self._regularize_station_covariance_np(
            covariance,
            variance_bpk,
        )
        residual = (z_bk[:, None, :] - np.transpose(lam_bkp, (0, 2, 1)))[..., None]
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, residual)
        quadratic = np.sum(whitened[..., 0] ** 2, axis=2)
        logdet = 2.0 * np.sum(
            np.log(np.diagonal(chol, axis1=2, axis2=3)),
            axis=2,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "gaussian":
            block_ll = -0.5 * (quadratic + logdet)
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            block_ll = (
                -0.5 * (df + float(view_count)) * np.log1p(quadratic / df)
                - 0.5 * logdet
            )
        return np.sum(block_ll, axis=0)

    def _structural_count_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate structural evidence using each recorded runtime route."""
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != int(data.z_k.size):
            raise ValueError("lambda_kp must have one row per measurement.")
        if data.runtime_likelihood_routes is None:
            return self._structural_single_route_log_likelihood_matrix_np(
                data,
                lam,
                use_direct_spectrum=None,
            )
        routes = np.asarray(data.runtime_likelihood_routes, dtype=str).reshape(-1)
        if routes.size != int(data.z_k.size):
            raise ValueError(
                "runtime_likelihood_routes must contain one route per measurement."
            )
        allowed_routes = np.isin(
            routes,
            np.asarray(
                ["count", "count_covariance", "direct_spectrum"],
                dtype=str,
            ),
        )
        if not np.all(allowed_routes):
            invalid = np.unique(routes[~allowed_routes]).tolist()
            raise ValueError(f"Unsupported runtime likelihood routes: {invalid}.")
        for rows in self._station_likelihood_block_rows(data).values():
            block_routes = routes[rows]
            if np.any(block_routes != block_routes[:, :1]):
                raise ValueError(
                    "Rows in one station sequence must share one runtime "
                    "likelihood route."
                )
        variance_presence = data.spectrum_variance_present
        if variance_presence is not None:
            variance_presence = np.asarray(
                variance_presence,
                dtype=bool,
            ).reshape(-1)
            if variance_presence.size != int(data.z_k.size):
                raise ValueError(
                    "spectrum_variance_present must contain one flag per measurement."
                )
            for rows in self._station_likelihood_block_rows(data).values():
                direct_blocks = routes[rows[:, 0]] == "direct_spectrum"
                if np.any(direct_blocks):
                    block_presence = variance_presence[rows[direct_blocks]]
                    if np.any(block_presence != block_presence[:, :1]):
                        raise ValueError(
                            "Rows in one direct-spectrum station sequence must "
                            "share spectrum-variance semantics."
                        )
        result = np.zeros(int(lam.shape[1]), dtype=float)
        direct_mask = routes == "direct_spectrum"
        if np.any(direct_mask):
            if variance_presence is None:
                direct_data = self._measurement_rows(data, direct_mask)
                result += self._structural_single_route_log_likelihood_matrix_np(
                    direct_data,
                    lam[direct_mask, :],
                    use_direct_spectrum=True,
                )
            else:
                for uses_variance in (False, True):
                    direct_variance_mask = direct_mask & (
                        variance_presence == uses_variance
                    )
                    if not np.any(direct_variance_mask):
                        continue
                    direct_data = self._measurement_rows(
                        data,
                        direct_variance_mask,
                    )
                    result += self._structural_single_route_log_likelihood_matrix_np(
                        direct_data,
                        lam[direct_variance_mask, :],
                        use_direct_spectrum=True,
                        use_spectrum_variance=uses_variance,
                    )
        count_covariance_mask = routes == "count_covariance"
        if np.any(count_covariance_mask):
            count_covariance_data = self._measurement_rows(
                data,
                count_covariance_mask,
            )
            result += self._structural_single_route_log_likelihood_matrix_np(
                count_covariance_data,
                lam[count_covariance_mask, :],
                use_direct_spectrum=False,
                use_count_covariance=True,
            )
        count_mask = routes == "count"
        if np.any(count_mask):
            count_data = self._measurement_rows(data, count_mask)
            result += self._structural_single_route_log_likelihood_matrix_np(
                count_data,
                lam[count_mask, :],
                use_direct_spectrum=False,
                use_count_covariance=False,
            )
        return result

    def _structural_single_route_log_likelihood_matrix_np(
        self,
        data: MeasurementData,
        lambda_kp: NDArray[np.float64],
        *,
        use_direct_spectrum: bool | None,
        use_spectrum_variance: bool | None = None,
        use_count_covariance: bool | None = None,
    ) -> NDArray[np.float64]:
        """
        Evaluate one homogeneous runtime likelihood route in a single batch.

        Direct spectrum payloads use the same independent-bin model as runtime.
        Otherwise same-position shield sequences use the configured multivariate
        Student-t/Gaussian count covariance.
        """
        lam = np.maximum(np.asarray(lambda_kp, dtype=float), 1.0e-12)
        if lam.ndim == 1:
            lam = lam[:, None]
        if lam.shape[0] != int(data.z_k.size):
            raise ValueError("lambda_kp must have one row per measurement.")
        spectrum_arrays = self._spectrum_update_arrays(
            data.spectrum_counts,
            data.spectrum_response_template,
            data.spectrum_background,
            (data.spectrum_variance if use_spectrum_variance is not False else None),
            sequence_length=int(data.z_k.size),
        )
        direct_route = (
            spectrum_arrays is not None
            and self._direct_spectrum_route_admissible(
                sequence_length=int(data.z_k.size),
                observation_count_covariance=data.observation_count_covariance,
            )
            if use_direct_spectrum is None
            else bool(use_direct_spectrum)
        )
        if direct_route and spectrum_arrays is None:
            raise ValueError(
                "The recorded direct-spectrum route lacks complete spectrum arrays."
            )
        if direct_route:
            assert spectrum_arrays is not None
            observed, template, background, spectrum_variance = spectrum_arrays
            if use_spectrum_variance is True and spectrum_variance is None:
                raise ValueError(
                    "The recorded direct-spectrum route used a variance array, "
                    "but the history lacks it."
                )
            return self._structural_spectrum_log_likelihood_matrix_np(
                lam,
                observed,
                template,
                background,
                spectrum_variance,
            ) + self._structural_shield_shape_log_likelihood_matrix_np(
                data,
                lam,
            )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        shield_shape_enabled = bool(
            self.config.shield_contrast_likelihood_enable
            or self.config.shield_view_ratio_likelihood_enable
        )
        if model == "poisson" and not shield_shape_enabled:
            return self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        if use_count_covariance is None:
            covariance_enabled = model != "poisson" and (
                data.observation_count_covariance is not None
                or (
                    bool(self.config.station_view_covariance_enable)
                    and float(self.config.station_view_correlated_spectrum_fraction)
                    > 0.0
                )
            )
        else:
            covariance_enabled = model != "poisson" and bool(use_count_covariance)
        if (not covariance_enabled and not shield_shape_enabled) or int(
            data.z_k.size
        ) < 2:
            return self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )

        blocks_by_length = self._station_likelihood_block_rows(data)
        if covariance_enabled:
            result = np.zeros(int(lam.shape[1]), dtype=float)
        else:
            result = self._count_log_likelihood_matrix_np(
                data.z_k,
                lam,
                observation_count_variance=data.observation_variances,
            )
        single_rows = blocks_by_length.get(1)
        if covariance_enabled and single_rows is not None and single_rows.size:
            single_indices = single_rows[:, 0]
            result += self._count_log_likelihood_matrix_np(
                np.asarray(data.z_k, dtype=float)[single_indices],
                lam[single_indices, :],
                observation_count_variance=np.asarray(
                    data.observation_variances,
                    dtype=float,
                )[single_indices],
            )
        # Station programs normally share one small view count. Grouping by
        # length keeps station and particle evaluation batched while preserving
        # the product of station-level multivariate likelihoods.
        for block_length, rows in blocks_by_length.items():
            if int(block_length) <= 1:
                continue
            if covariance_enabled:
                result += self._station_covariance_log_likelihood_batch_np(
                    data,
                    lam,
                    rows,
                )
            if shield_shape_enabled:
                result += self._shield_shape_log_likelihood_batch_np(
                    data,
                    lam,
                    rows,
                )
        return result

    def set_kernel(self, kernel: KernelPrecomputer) -> None:
        """Attach a discrete kernel and refresh only changed continuous physics."""
        incoming_signature = self._incoming_kernel_physics_signature(kernel)
        self.kernel = kernel
        if incoming_signature == self._continuous_kernel_physics_signature:
            return
        self.continuous_kernel = self._build_continuous_kernel(
            mu_by_isotope=getattr(kernel, "mu_by_isotope", None),
            shield_params=getattr(kernel, "shield_params", ShieldParams()),
        )
        self._continuous_kernel_physics_signature = incoming_signature
        self._structural_rj_response_cache = None
        self._structural_rj_response_cache_signatures = None

    def _source_prior_is_surface(self) -> bool:
        """Return True when source positions should be constrained to surfaces."""
        raw = getattr(self.config, "source_position_prior", "volume")
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {
            "surface",
            "surfaces",
            "surface_constrained",
            "surface-constrained",
        }

    def _source_prior_environment(self) -> EnvironmentConfig:
        """Return the room geometry used by the source-position prior."""
        hi = np.array(self.config.position_max, dtype=float)
        if hi.shape != (3,):
            raise ValueError("position_max must be a 3-element vector.")
        if np.any(hi <= 0.0):
            raise ValueError("position_max must define positive room dimensions.")
        return EnvironmentConfig(
            size_x=float(hi[0]),
            size_y=float(hi[1]),
            size_z=float(hi[2]),
        )

    def _project_positions_to_source_prior(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Clip or project source positions according to the configured prior."""
        arr = np.asarray(positions, dtype=float)
        lo = np.array(self.config.position_min, dtype=float)
        hi = np.array(self.config.position_max, dtype=float)
        clipped = np.clip(arr, lo, hi)
        if not self._source_prior_is_surface() or clipped.size == 0:
            return clipped
        projected = project_positions_to_allowed_surfaces(
            clipped,
            self._source_prior_environment(),
            self.obstacle_grid,
            obstacle_height_m=self.obstacle_height_m,
        )
        return np.clip(projected, lo, hi)

    def _sample_initial_strengths(
        self,
        shape: int | tuple[int, ...],
    ) -> NDArray[np.float64]:
        """Draw a batched initial-strength array from the configured PF prior."""
        return np.asarray(
            self._strength_prior.sample(
                shape,
                rng=self._random_generator,
            ),
            dtype=float,
        )

    def _exact_initial_cardinality_counts(
        self,
        particle_count: int,
    ) -> NDArray[np.int64]:
        """Allocate at least one exact-mode particle to every positive K prior."""
        count = max(1, int(particle_count))
        probabilities = np.asarray(
            self._structural_rj_cardinality_prior_probs,
            dtype=float,
        )
        positive = probabilities > 0.0
        positive_count = int(np.count_nonzero(positive))
        if count < positive_count:
            raise ValueError(
                "rj_mh needs at least one initial particle for every positive "
                "cardinality-prior entry."
            )
        allocation = np.zeros(probabilities.size, dtype=np.int64)
        allocation[positive] = 1
        remaining = count - positive_count
        if remaining <= 0:
            return allocation
        expected = remaining * probabilities
        extra = np.floor(expected).astype(np.int64)
        allocation += extra
        leftover = remaining - int(np.sum(extra))
        if leftover > 0:
            fractions = expected - extra
            order = np.argsort(fractions, kind="stable")[::-1]
            allocation[order[:leftover]] += 1
        return allocation

    def _init_exact_structural_particles(self) -> None:
        """Initialize the exact PF from its K, surface-set, and strength priors."""
        patches = self._structural_rj_surface_patches
        surface_prior = self._structural_rj_surface_prior
        cardinality_prior = self._structural_rj_cardinality_prior
        if patches is None or surface_prior is None or cardinality_prior is None:
            raise RuntimeError("rj_mh surface and cardinality priors are unavailable.")
        # Exact-mode particles are weighted Monte Carlo samples from the
        # normalized cardinality, surface-set, and strength priors.
        target_n = int(self.config.num_particles)
        allocation = self._exact_initial_cardinality_counts(target_n)
        particles: list[IsotopeParticle] = []
        for cardinality, cardinality_count in enumerate(allocation.tolist()):
            if cardinality_count <= 0:
                continue
            surface_sets = surface_prior.sample_rejection(
                cardinality,
                cardinality_count,
                rng=self._random_generator,
            )
            strengths = self._sample_initial_strengths((cardinality_count, cardinality))
            per_particle_mass = float(
                cardinality_prior.probabilities[cardinality]
            ) / float(cardinality_count)
            log_weight = float(np.log(per_particle_mass))
            for row in range(cardinality_count):
                patch_indices = surface_sets[row]
                positions = np.asarray(
                    patches.centers_xyz[patch_indices],
                    dtype=float,
                ).reshape(cardinality, 3)
                state = IsotopeState(
                    num_sources=cardinality,
                    positions=positions,
                    strengths=np.asarray(strengths[row], dtype=float).copy(),
                    background=self._background_level(),
                )
                particles.append(IsotopeParticle(state=state, log_weight=log_weight))
        permutation = self._random_generator.permutation(len(particles))
        self.continuous_particles = [particles[int(index)] for index in permutation]
        self.N = len(self.continuous_particles)
        self.config.num_particles = self.N
        self.config.min_particles = self.N
        self.config.max_particles = self.N

    def _init_fixed_cardinality_particles(self) -> None:
        """Initialize a fixed-K PF from the finite surface and strength priors."""
        patches = self._structural_rj_surface_patches
        surface_prior = self._structural_rj_surface_prior
        if patches is None or surface_prior is None:
            raise RuntimeError("Finite surface support is unavailable.")
        cardinality = int(self.config.init_num_sources[0])
        particle_count = max(1, int(self.config.num_particles))
        surface_sets = surface_prior.sample_rejection(
            cardinality,
            particle_count,
            rng=self._random_generator,
        )
        strengths = self._sample_initial_strengths((particle_count, cardinality))
        log_weight = float(-np.log(particle_count))
        self.continuous_particles = []
        for row in range(particle_count):
            patch_indices = surface_sets[row]
            state = IsotopeState(
                num_sources=cardinality,
                positions=np.asarray(
                    patches.centers_xyz[patch_indices],
                    dtype=float,
                ).reshape(cardinality, 3),
                strengths=np.asarray(strengths[row], dtype=float).copy(),
                background=self._background_level(),
            )
            self.continuous_particles.append(
                IsotopeParticle(state=state, log_weight=log_weight)
            )
        self.N = particle_count
        self.config.num_particles = particle_count

    def _init_continuous_particles(self) -> None:
        """Initialize exact variable-K or fixed-K particles from PF priors."""
        self.continuous_particles = []
        if self._variable_cardinality_enabled():
            self._init_exact_structural_particles()
        else:
            self._init_fixed_cardinality_particles()

    def reset_step_stats(self) -> None:
        """Reset per-step diagnostic counters."""
        reset_step_diagnostics(self)

    def _advance_adapt_cooldown(self) -> None:
        """Decrement the adapt cooldown counter after each update."""
        if self._adapt_cooldown_remaining > 0:
            self._adapt_cooldown_remaining -= 1

    def _trigger_adapt_cooldown(self) -> None:
        """Start the adapt cooldown after a resampling event."""
        steps = max(0, int(self.config.adapt_cooldown_steps))
        if steps > 0:
            self._adapt_cooldown_remaining = max(
                self._adapt_cooldown_remaining, steps + 1
            )

    def _record_source_event(
        self,
        event: str,
        st: IsotopeState,
        source_idx: int,
        *,
        reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Record an accepted exact-RJ source birth or death event."""
        record = build_source_event_record(
            event=event,
            isotope=self.isotope,
            state=st,
            source_idx=int(source_idx),
            reason=reason,
            extra=extra,
        )
        if record is None:
            return
        self.last_source_event_diagnostics.append(record)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.config.use_gpu:
            raise RuntimeError("GPU-only mode: enable use_gpu in PFConfig.")
        if not gpu_utils.torch_device_available(self.config.gpu_device):
            raise RuntimeError("GPU-only mode requires torch on the requested device.")
        return True

    def _can_use_gpu(self) -> bool:
        """Return whether this filter can use the configured torch device."""
        from pf import gpu_utils

        return bool(
            self.config.use_gpu
            and gpu_utils.torch_device_available(self.config.gpu_device)
        )

    def _ll_proxy_pair(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> float:
        """Return a Poisson log-likelihood proxy for convergence checks."""
        if not self.continuous_particles:
            return 0.0
        state = self.best_particle().state
        lam_rate = float(state.background)
        source_scale = self._measurement_source_scale(
            fe_index=fe_index,
            pb_index=pb_index,
        )
        if state.num_sources > 0:
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value_pair(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                )
                lam_rate += source_scale * float(kernel_val) * float(strength)
        lam = float(live_time_s) * lam_rate
        return self._count_log_likelihood_np(
            np.array([float(z_obs)], dtype=float),
            np.array([lam], dtype=float),
        )

    def _mmse_primary_position(self) -> NDArray[np.float64] | None:
        """Return the MMSE position for the first source slot, if available."""
        if not self.continuous_particles:
            return None
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size == 0:
            return None
        pos_stack: list[NDArray[np.float64]] = []
        w_stack: list[float] = []
        for weight, particle in zip(weights, self.continuous_particles):
            state = particle.state
            if state.num_sources > 0:
                pos_stack.append(state.positions[0])
                w_stack.append(float(weight))
        if not w_stack:
            return None
        w = np.asarray(w_stack, dtype=float)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.full_like(w, 1.0 / max(len(w), 1))
        else:
            w = w / w_sum
        pos_arr = np.vstack(pos_stack)
        return np.sum(w[:, None] * pos_arr, axis=0)

    def _cardinality_variance(self) -> float:
        """Return posterior variance of the active source count."""
        if not self.continuous_particles:
            return 0.0
        weights = np.asarray(self.continuous_weights, dtype=float)
        if weights.size != len(self.continuous_particles):
            weights = np.ones(len(self.continuous_particles), dtype=float)
        weights = weights / max(float(np.sum(weights)), 1.0e-12)
        counts = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=float,
        )
        mean = float(np.sum(weights * counts))
        return float(np.sum(weights * (counts - mean) ** 2))

    def _convergence_state_vector(self) -> NDArray[np.float64] | None:
        """Return a cardinality-aware vector for convergence monitoring."""
        if not self.continuous_particles:
            return None
        point_estimate = posterior_point_estimate_from_states(
            [particle.state for particle in self.continuous_particles],
            np.asarray(self.continuous_weights, dtype=float),
            max_cardinality=self.config.max_sources,
            position_projector=self._project_positions_to_source_prior,
        )
        cardinality = np.asarray(
            [
                point_estimate.cardinality_distribution.get(source_count, 0.0)
                for source_count in range(int(self.config.max_sources) + 1)
            ],
            dtype=float,
        )
        if not point_estimate.modes:
            return cardinality
        positions = np.asarray(
            [mode.position_mean_xyz for mode in point_estimate.modes],
            dtype=float,
        )
        strengths = np.asarray(
            [mode.strength_mean_cps_1m for mode in point_estimate.modes],
            dtype=float,
        )
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        pos_sorted = np.asarray(positions[order], dtype=float)
        str_sorted = np.asarray(strengths, dtype=float).reshape(-1)[order]
        strength_scale = max(float(np.max(np.abs(str_sorted))), 1.0)
        return np.concatenate(
            [cardinality, pos_sorted.reshape(-1), str_sorted / strength_scale]
        )

    def _distinct_observed_station_count(self) -> int:
        """Return the number of distinct detector stations seen by this filter."""
        return int(len(self._observed_station_labels))

    def _record_observed_station(
        self,
        detector_pos: NDArray[np.float64] | None,
    ) -> None:
        """Record a detector station using rounded XY coordinates."""
        if detector_pos is None:
            return
        pos = np.asarray(detector_pos, dtype=float).reshape(-1)
        if pos.size < 2 or not np.all(np.isfinite(pos[:2])):
            return
        self._observed_station_labels.add(
            (round(float(pos[0]), 3), round(float(pos[1]), 3))
        )

    def _convergence_structure_supported(self) -> bool:
        """Return whether posterior cardinality is stable enough to stop."""
        if self._cardinality_variance() > float(
            self.config.converge_cardinality_var_max
        ):
            return False
        return True

    def _convergence_stopping_supported(self) -> bool:
        """Return whether posterior diagnostics support acquisition stopping."""
        if not self.config.converge_enable:
            return False
        min_stations = max(0, int(getattr(self.config, "converge_min_stations", 0)))
        if self._distinct_observed_station_count() < min_stations:
            return False
        if not self._convergence_structure_supported():
            return False
        return True

    def _maybe_update_convergence(
        self,
        step_idx: int | None,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        z_obs: float,
    ) -> None:
        """Update posterior-derived convergence diagnostics after an observation."""
        if not self.config.converge_enable or self._converge_monitor is None:
            return
        if step_idx is None:
            return
        if not self.continuous_particles:
            return
        summary_vec = self._convergence_state_vector()
        ess_pre = self.last_ess_pre
        if ess_pre is None:
            w = self.continuous_weights
            ess_pre = float(1.0 / max(np.sum(w**2), 1e-12)) if w.size else 0.0
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        ll_value = self._ll_proxy_pair(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )
        self._converge_monitor.update_stats(step_idx, summary_vec, ess_ratio, ll_value)
        self.is_converged = bool(
            self._converge_monitor.is_converged(step_idx)
            and self._convergence_stopping_supported()
        )

    def _continuous_expected_counts_torch(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        if self.kernel is None:
            from pf import gpu_utils

            device = gpu_utils.resolve_device(self.config.gpu_device)
            dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
            import torch

            return torch.zeros(0, device=device, dtype=dtype)
        orient_vec = self.kernel.orientations[orient_idx]
        octant_idx = self.continuous_kernel.orient_index_from_vector(orient_vec)
        return self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=octant_idx,
            pb_index=octant_idx,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_torch(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles or self.kernel is None:
            return torch.zeros(0, device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states, device=device, dtype=dtype
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                source_scale=self._measurement_source_scale(
                    fe_index=fe_index,
                    pb_index=pb_index,
                ),
                device=device,
                dtype=dtype,
            )
        zero_backgrounds = torch.zeros(
            int(packed_positions.shape[0]),
            device=device,
            dtype=dtype,
        )
        source_counts = (
            self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=zero_backgrounds,
                mask=packed_mask,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                source_scale=self._measurement_source_scale(
                    fe_index=fe_index,
                    pb_index=pb_index,
                ),
                device=device,
                dtype=dtype,
            )
        )
        return source_counts.index_select(0, inverse) + float(live_time_s) * backgrounds

    def _continuous_expected_counts_pair_sequence_torch(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute expected counts for a same-station shield sequence in one GPU batch."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
        else:
            zero_backgrounds = torch.zeros(
                int(packed_positions.shape[0]),
                device=device,
                dtype=dtype,
            )
            source_unit_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=detector_pos,
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=zero_backgrounds,
                mask=packed_mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
            unit_live_counts = source_unit_counts.index_select(1, inverse)
            unit_live_counts = unit_live_counts + backgrounds.unsqueeze(0)
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _compress_identical_packed_sources_torch(
        self,
        positions: "torch.Tensor",
        strengths: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor | None",
    ]:
        """Return one source row plus inverse when all packed source states match."""
        import torch

        particle_count = int(positions.shape[0])
        if particle_count <= 1:
            return positions, strengths, mask, None
        if positions.ndim != 3 or strengths.ndim != 2 or mask.ndim != 2:
            return positions, strengths, mask, None
        flat = torch.cat(
            (
                positions.reshape(particle_count, -1),
                strengths.reshape(particle_count, -1),
                mask.reshape(particle_count, -1),
            ),
            dim=1,
        )
        if flat.numel() == 0:
            return positions, strengths, mask, None
        identical = bool(torch.all(flat == flat[:1]).detach().cpu().item())
        if not identical:
            return positions, strengths, mask, None
        inverse = torch.zeros(particle_count, device=positions.device, dtype=torch.long)
        return positions[:1], strengths[:1], mask[:1], inverse

    def _continuous_expected_counts_pair_sequence_torch_uncompressed(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Compute same-station expected counts without duplicate-state compression."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        fe_arr = np.asarray(fe_indices, dtype=int).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=int).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=float).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        if fe_arr.size == 0 or not self.continuous_particles or self.kernel is None:
            return torch.zeros((0, 0), device=device, dtype=dtype)
        states = [p.state for p in self.continuous_particles]
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            states,
            device=device,
            dtype=dtype,
        )
        detector_pos = np.asarray(self.kernel.poses[pose_idx], dtype=float)
        unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=detector_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
            live_time_s=1.0,
            source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
            device=device,
            dtype=dtype,
        )
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return unit_live_counts * live_t

    def _current_log_weights_torch(self, device: "torch.device") -> "torch.Tensor":
        """Return log-weights as a float64 torch tensor on the requested device."""
        import torch

        return torch.as_tensor(
            [p.log_weight for p in self.continuous_particles],
            device=device,
            dtype=torch.float64,
        )

    def _log_likelihood_increment_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        observation_count_variance: float = 0.0,
    ) -> "torch.Tensor":
        """Return the per-particle count log-likelihood increment in float64."""
        import torch

        lam_t = lam_t.to(dtype=torch.float64)
        lam_t = torch.clamp(lam_t, min=1e-12)
        z = torch.as_tensor(z_obs, device=lam_t.device, dtype=torch.float64)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""}:
            return z * torch.log(lam_t) - lam_t

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = max(float(observation_count_variance), 0.0)
        variance = count_likelihood_variance_torch(
            z,
            lam_t,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )
        residual = z - lam_t
        if model == "gaussian":
            return -0.5 * ((residual**2) / variance + torch.log(variance))

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        return -0.5 * (df + 1.0) * torch.log1p(
            (residual**2) / (df * variance)
        ) - 0.5 * torch.log(variance)

    def _log_likelihood_sequence_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        observation_count_covariance: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return summed per-particle log-likelihoods for a measurement sequence."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        if lam.numel() == 0:
            return torch.zeros(0, device=lam.device, dtype=torch.float64)
        lam = torch.clamp(lam, min=1e-12)
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""}:
            count_ll = torch.sum(z * torch.log(lam) - lam, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )
        if self._sequence_covariance_enabled(
            z_arr.size,
            observation_count_covariance,
        ):
            return self._log_likelihood_sequence_covariance_gpu(
                lam,
                z_arr,
                var_arr,
                observation_count_covariance=observation_count_covariance,
            )

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        variance = count_likelihood_variance_torch(
            z,
            lam,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )
        residual = z - lam
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            count_ll = torch.sum(ll, dim=0)
            return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
                lam,
                z_arr,
                var_arr,
            )

        df = max(float(self.config.count_likelihood_df), 1.0 + 1e-12)
        ll = -0.5 * (df + 1.0) * torch.log1p(
            (residual**2) / (df * variance)
        ) - 0.5 * torch.log(variance)
        count_ll = torch.sum(ll, dim=0)
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_covariance_enabled(
        self,
        sequence_length: int,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether same-station view covariance should be evaluated."""
        if int(sequence_length) < 2:
            return False
        if observation_count_covariance is not None:
            return True
        return bool(self.config.station_view_covariance_enable) and (
            float(self.config.station_view_correlated_spectrum_fraction) > 0.0
        )

    def _spectral_bin_sequence_log_likelihood_gpu(
        self,
        expected_spectrum_kbn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods for each particle."""
        import torch

        expected = expected_spectrum_kbn.to(dtype=torch.float64)
        if expected.ndim == 2:
            expected = expected.unsqueeze(0)
        if expected.ndim != 3:
            raise ValueError("expected_spectrum_kbn must have shape K x B x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.shape != tuple(expected.shape[:2]):
            raise ValueError("observed_spectrum_kb must match K x B expected bins.")
        expected = torch.clamp(expected, min=1.0e-12)
        observed = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=expected.device,
            dtype=torch.float64,
        ).unsqueeze(-1)
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model in {"poisson", ""} and observation_spectrum_variance_kb is None:
            return torch.sum(observed * torch.log(expected) - expected, dim=(0, 1))

        if observation_spectrum_variance_kb is None:
            obs_var = torch.zeros_like(observed)
        else:
            obs_var_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if obs_var_arr.ndim == 1:
                obs_var_arr = obs_var_arr.reshape(1, -1)
            if obs_var_arr.shape != tuple(expected.shape[:2]):
                raise ValueError("observation_spectrum_variance_kb must match K x B.")
            obs_var = torch.as_tensor(
                np.maximum(np.where(np.isfinite(obs_var_arr), obs_var_arr, 0.0), 0.0),
                device=expected.device,
                dtype=torch.float64,
            ).unsqueeze(-1)
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        variance = torch.clamp(
            expected
            + obs_var
            + (float(spectrum_rel) * expected) ** 2
            + float(spectrum_abs) ** 2,
            min=1.0e-12,
        )
        residual = observed - expected
        if model == "gaussian":
            ll = -0.5 * ((residual**2) / variance + torch.log(variance))
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            ll = -0.5 * (df + 1.0) * torch.log1p(
                (residual**2) / (df * variance)
            ) - 0.5 * torch.log(variance)
        return torch.sum(ll, dim=(0, 1))

    @classmethod
    def _direct_spectrum_likelihood_config_enabled(
        cls,
        config: Any,
        isotope: str,
    ) -> bool:
        """Return whether config permits independent-bin spectrum likelihoods.

        Complete statistical covariance from weighted transport generally has
        response-folding correlations between spectrum bins. The current
        direct-bin likelihood accepts only diagonal variance, so that semantic
        must route through isotope counts and their propagated covariance.
        Gaussian and Student-t transport discrepancy is also defined on the
        isotope count, not independently on every spectrum bin, and therefore
        must use the count likelihood.
        """
        if not bool(config.direct_spectrum_likelihood_enable):
            return False
        semantics = normalize_observation_count_variance_semantics(
            config.observation_count_variance_semantics,
            includes_counting_noise=(
                config.observation_count_variance_includes_counting_noise
            ),
        )
        if semantics == OBSERVATION_COUNT_VARIANCE_COMPLETE_STATISTICAL:
            return False
        model = normalize_count_likelihood_model(str(config.count_likelihood_model))
        if model != "poisson" and (
            cls._isotope_float_config_for(
                config.transport_model_rel_sigma,
                isotope,
            )
            > 0.0
            or cls._isotope_float_config_for(
                config.transport_model_abs_sigma,
                isotope,
            )
            > 0.0
        ):
            return False
        return True

    def _direct_spectrum_likelihood_enabled(self) -> bool:
        """Return whether this filter permits independent spectrum bins."""
        return self._direct_spectrum_likelihood_config_enabled(
            self.config,
            self.isotope,
        )

    def _direct_spectrum_route_admissible(
        self,
        *,
        sequence_length: int,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> bool:
        """Return whether one runtime update may use independent spectrum bins."""
        return self._direct_spectrum_likelihood_enabled() and not (
            self._sequence_covariance_enabled(
                int(sequence_length),
                observation_count_covariance,
            )
        )

    @staticmethod
    def _spectrum_update_arrays(
        observed_spectrum: NDArray[np.float64] | None,
        response_template: NDArray[np.float64] | None,
        background_spectrum: NDArray[np.float64] | None,
        spectrum_variance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
        ]
        | None
    ):
        """Return validated spectrum-bin arrays for a PF likelihood update."""
        if observed_spectrum is None or response_template is None:
            return None
        rows = max(1, int(sequence_length))
        observed = np.asarray(observed_spectrum, dtype=float)
        if observed.ndim == 1:
            observed = observed.reshape(1, -1)
        template = np.asarray(response_template, dtype=float)
        if template.ndim == 1:
            template = np.broadcast_to(template.reshape(1, -1), observed.shape)
        background = (
            np.zeros_like(observed, dtype=float)
            if background_spectrum is None
            else np.asarray(background_spectrum, dtype=float)
        )
        if background.ndim == 1:
            background = np.broadcast_to(background.reshape(1, -1), observed.shape)
        variance = None
        if spectrum_variance is not None:
            variance = np.asarray(spectrum_variance, dtype=float)
            if variance.ndim == 1:
                variance = np.broadcast_to(variance.reshape(1, -1), observed.shape)
        if observed.shape[0] != rows:
            raise ValueError("observed_spectrum row count must match sequence length.")
        if template.shape != observed.shape:
            raise ValueError("response_template must match observed spectrum bins.")
        if background.shape != observed.shape:
            raise ValueError("background_spectrum must match observed spectrum bins.")
        if variance is not None and variance.shape != observed.shape:
            raise ValueError("spectrum_variance must match observed spectrum bins.")
        observed = np.maximum(np.where(np.isfinite(observed), observed, 0.0), 0.0)
        template = np.maximum(np.where(np.isfinite(template), template, 0.0), 0.0)
        background = np.maximum(
            np.where(np.isfinite(background), background, 0.0),
            0.0,
        )
        if variance is not None:
            variance = np.maximum(np.where(np.isfinite(variance), variance, 0.0), 0.0)
        return observed, template, background, variance

    def _spectral_bin_sequence_log_likelihood_from_lambda_gpu(
        self,
        lam_kn: "torch.Tensor",
        observed_spectrum_kb: NDArray[np.float64],
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
        observation_spectrum_variance_kb: NDArray[np.float64] | None = None,
    ) -> "torch.Tensor":
        """Return direct spectrum-bin log likelihoods without a full KxBxN tensor."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        if lam.ndim != 2:
            raise ValueError("lam_kn must have shape K x N.")
        observed_arr = np.asarray(observed_spectrum_kb, dtype=float)
        if observed_arr.ndim == 1:
            observed_arr = observed_arr.reshape(1, -1)
        if observed_arr.ndim != 2 or observed_arr.shape[0] != int(lam.shape[0]):
            raise ValueError("observed_spectrum_kb rows must match lambda rows.")
        template_arr = np.asarray(response_template_kb, dtype=float)
        if template_arr.ndim == 1:
            template_arr = np.broadcast_to(
                template_arr.reshape(1, -1),
                observed_arr.shape,
            )
        background_arr = np.asarray(background_spectrum_kb, dtype=float)
        if background_arr.ndim == 1:
            background_arr = np.broadcast_to(
                background_arr.reshape(1, -1),
                observed_arr.shape,
            )
        if template_arr.shape != observed_arr.shape:
            raise ValueError("response_template_kb must match observed spectrum bins.")
        if background_arr.shape != observed_arr.shape:
            raise ValueError(
                "background_spectrum_kb must match observed spectrum bins."
            )
        variance_arr = None
        if observation_spectrum_variance_kb is not None:
            variance_arr = np.asarray(observation_spectrum_variance_kb, dtype=float)
            if variance_arr.ndim == 1:
                variance_arr = np.broadcast_to(
                    variance_arr.reshape(1, -1),
                    observed_arr.shape,
                )
            if variance_arr.shape != observed_arr.shape:
                raise ValueError("observation_spectrum_variance_kb must match bins.")
        observed_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(observed_arr), observed_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        template_all = torch.as_tensor(
            np.maximum(np.where(np.isfinite(template_arr), template_arr, 0.0), 0.0),
            device=lam.device,
            dtype=torch.float64,
        )
        background_all = torch.as_tensor(
            np.maximum(
                np.where(np.isfinite(background_arr), background_arr, 0.0),
                0.0,
            ),
            device=lam.device,
            dtype=torch.float64,
        )
        variance_all = None
        if variance_arr is not None:
            variance_all = torch.as_tensor(
                np.maximum(np.where(np.isfinite(variance_arr), variance_arr, 0.0), 0.0),
                device=lam.device,
                dtype=torch.float64,
            )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        rel_sigma = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        abs_sigma = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        chunk = max(1, int(getattr(self.config, "spectrum_likelihood_bin_chunk", 512)))
        bin_count = int(observed_all.shape[1])
        total_ll = torch.zeros(
            int(lam.shape[1]), device=lam.device, dtype=torch.float64
        )
        for start in range(0, bin_count, chunk):
            stop = min(start + chunk, bin_count)
            expected = torch.clamp(
                lam[:, None, :] * template_all[:, start:stop, None]
                + background_all[:, start:stop, None],
                min=1.0e-12,
            )
            observed = observed_all[:, start:stop, None]
            if model in {"poisson", ""} and variance_all is None:
                total_ll = total_ll + torch.sum(
                    observed * torch.log(expected) - expected,
                    dim=(0, 1),
                )
                continue
            if variance_all is None:
                obs_var = torch.zeros_like(observed)
            else:
                obs_var = variance_all[:, start:stop, None]
            variance = torch.clamp(
                expected
                + obs_var
                + (float(rel_sigma) * expected) ** 2
                + float(abs_sigma) ** 2,
                min=1.0e-12,
            )
            residual = observed - expected
            if model == "gaussian":
                ll = -0.5 * ((residual**2) / variance + torch.log(variance))
            else:
                df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
                ll = -0.5 * (df + 1.0) * torch.log1p(
                    (residual**2) / (df * variance)
                ) - 0.5 * torch.log(variance)
            total_ll = total_ll + torch.sum(ll, dim=(0, 1))
        return total_ll

    def _expected_spectrum_sequence_torch(
        self,
        lam_kn: "torch.Tensor",
        response_template_kb: NDArray[np.float64],
        background_spectrum_kb: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxBxN expected spectra from particle count rates."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim == 1:
            lam = lam.view(1, -1)
        template = torch.as_tensor(
            response_template_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        background = torch.as_tensor(
            background_spectrum_kb,
            device=lam.device,
            dtype=torch.float64,
        )
        if template.ndim == 1:
            template = template.view(1, -1)
        if background.ndim == 1:
            background = background.view(1, -1)
        if template.shape[0] != lam.shape[0] or background.shape[0] != lam.shape[0]:
            raise ValueError("spectrum template rows must match lambda rows.")
        expected = lam[:, None, :] * template[:, :, None] + background[:, :, None]
        return torch.clamp(expected, min=1.0e-12)

    def _log_likelihood_sequence_covariance_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
        *,
        observation_count_covariance: NDArray[np.float64] | None,
    ) -> "torch.Tensor":
        """Return a batched multivariate count likelihood for a shield sequence."""
        import torch

        lam = torch.clamp(lam_kn.to(dtype=torch.float64), min=1.0e-12)
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        variance = self._sequence_diagonal_variance_torch(lam, z, var_arr)
        covariance_nkk = torch.diag_embed(variance.T)
        covariance_nkk = covariance_nkk + self._sequence_observation_offdiag_torch(
            observation_count_covariance,
            sequence_length=z_arr.size,
            device=lam.device,
        )
        covariance_nkk = covariance_nkk + self._sequence_common_mode_offdiag_torch(
            lam,
        )
        covariance_nkk = self._regularize_sequence_covariance_torch(
            covariance_nkk,
            variance.T,
        )
        residual = (z - lam).T.unsqueeze(-1)
        chol = torch.linalg.cholesky(covariance_nkk)
        solved = torch.cholesky_solve(residual, chol)
        quad = torch.sum(residual * solved, dim=(1, 2))
        logdet = 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=1, dim2=2)),
            dim=1,
        )
        model = normalize_count_likelihood_model(
            str(self.config.count_likelihood_model)
        )
        if model == "gaussian":
            count_ll = -0.5 * (quad + logdet)
        else:
            df = max(float(self.config.count_likelihood_df), 1.0 + 1.0e-12)
            dim = float(z_arr.size)
            count_ll = -0.5 * (df + dim) * torch.log1p(quad / df) - 0.5 * logdet
        return count_ll + self._shield_shape_sequence_log_likelihood_gpu(
            lam,
            z_arr,
            var_arr,
        )

    def _sequence_diagonal_variance_torch(
        self,
        lam_kn: "torch.Tensor",
        z_kn: "torch.Tensor",
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return KxN diagonal variances for same-station sequence likelihood."""
        import torch

        transport_rel = self._isotope_float_config(
            self.config.transport_model_rel_sigma
        )
        transport_abs = self._isotope_float_config(
            self.config.transport_model_abs_sigma
        )
        spectrum_rel = self._isotope_float_config(self.config.spectrum_count_rel_sigma)
        spectrum_abs = self._isotope_float_config(self.config.spectrum_count_abs_sigma)
        low_count_abs = self._isotope_float_config(self.config.low_count_abs_sigma)
        low_count_transition = self._isotope_float_config(
            self.config.low_count_transition_counts
        )
        obs_var = torch.clamp(
            torch.as_tensor(
                observation_count_variances,
                device=lam_kn.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        return count_likelihood_variance_torch(
            z_kn,
            lam_kn,
            transport_model_rel_sigma=float(transport_rel),
            transport_model_abs_sigma=float(transport_abs),
            spectrum_count_rel_sigma=float(spectrum_rel),
            spectrum_count_abs_sigma=float(spectrum_abs),
            low_count_abs_sigma=float(low_count_abs),
            low_count_transition_counts=float(low_count_transition),
            observation_count_variance=obs_var,
            observation_count_variance_includes_counting_noise=bool(
                self.config.observation_count_variance_includes_counting_noise
            ),
            observation_count_variance_semantics=str(
                self.config.observation_count_variance_semantics
            ),
        )

    def _sequence_observation_offdiag_torch(
        self,
        observation_count_covariance: NDArray[np.float64] | None,
        *,
        sequence_length: int,
        device: "torch.device",
    ) -> "torch.Tensor":
        """Return fixed off-diagonal observation covariance for one sequence."""
        import torch

        size = int(sequence_length)
        if observation_count_covariance is None:
            return torch.zeros((1, size, size), device=device, dtype=torch.float64)
        covariance = np.asarray(observation_count_covariance, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(
                "observation_count_covariance must be shaped sequence x sequence."
            )
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, 0.0)
        return torch.as_tensor(
            covariance,
            device=device,
            dtype=torch.float64,
        ).view(1, size, size)

    def _sequence_common_mode_offdiag_torch(
        self,
        lam_kn: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return particle-wise off-diagonal covariance from common spectrum error."""
        import torch

        fraction = max(
            0.0,
            float(self.config.station_view_correlated_spectrum_fraction),
        )
        if not bool(self.config.station_view_covariance_enable) or fraction <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        spectrum_rel = fraction * self._isotope_float_config(
            self.config.spectrum_count_rel_sigma
        )
        spectrum_abs = fraction * self._isotope_float_config(
            self.config.spectrum_count_abs_sigma
        )
        if spectrum_rel <= 0.0 and spectrum_abs <= 0.0:
            size = int(lam_kn.shape[0])
            return torch.zeros(
                (int(lam_kn.shape[1]), size, size),
                device=lam_kn.device,
                dtype=torch.float64,
            )
        lam_nk = lam_kn.T
        common = (float(spectrum_rel) ** 2) * (lam_nk[:, :, None] * lam_nk[:, None, :])
        if spectrum_abs > 0.0:
            common = common + float(spectrum_abs) ** 2
        diag = torch.arange(int(lam_kn.shape[0]), device=lam_kn.device)
        common[:, diag, diag] = 0.0
        return common

    def _regularize_sequence_covariance_torch(
        self,
        covariance_nkk: "torch.Tensor",
        diagonal_nk: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a positive-definite covariance batch for Cholesky solves."""
        import torch

        size = int(covariance_nkk.shape[1])
        eye = torch.eye(size, device=covariance_nkk.device, dtype=torch.float64)
        diag_scale = torch.clamp(torch.mean(diagonal_nk, dim=1), min=1.0)
        jitter = 1.0e-9 * diag_scale
        regularized = covariance_nkk
        for _ in range(6):
            candidate = regularized + jitter.view(-1, 1, 1) * eye.view(1, size, size)
            _chol, info = torch.linalg.cholesky_ex(candidate)
            if bool(torch.all(info == 0)):
                return candidate
            jitter = jitter * 10.0
            regularized = covariance_nkk
        fallback = torch.diag_embed(torch.clamp(diagonal_nk, min=1.0e-12))
        return fallback + jitter.view(-1, 1, 1) * eye.view(1, size, size)

    def _shield_shape_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return same-station shield signature likelihood terms."""
        return self._shield_contrast_sequence_log_likelihood_gpu(
            lam_kn,
            z_obs,
            observation_count_variances,
        ) + self._shield_view_ratio_sequence_log_likelihood_gpu(lam_kn, z_obs)

    def _shield_contrast_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a robust same-station shield-signature likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(getattr(self.config, "shield_contrast_likelihood_enable", False)):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_contrast_likelihood_weight", 1.0)),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_contrast_min_views", 2)),
            2,
        )
        if weight <= 0.0 or int(lam.shape[0]) < min_views or particle_count == 0:
            return zeros

        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        var_arr = np.asarray(observation_count_variances, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]) or var_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        min_count = max(
            float(getattr(self.config, "shield_contrast_min_count", 25.0)),
            1.0e-6,
        )
        sigma_floor = max(
            float(getattr(self.config, "shield_contrast_log_sigma_floor", 0.5)),
            1.0e-6,
        )
        sigma_ceiling = max(
            float(getattr(self.config, "shield_contrast_log_sigma_ceiling", 2.0)),
            sigma_floor,
        )
        df = max(
            float(getattr(self.config, "shield_contrast_likelihood_df", 5.0)),
            1.0 + 1.0e-12,
        )

        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        obs_var = torch.clamp(
            torch.as_tensor(
                var_arr,
                device=lam.device,
                dtype=torch.float64,
            ).view(-1, 1),
            min=0.0,
        )
        z_safe = torch.clamp(z, min=min_count)
        lam_safe = torch.clamp(lam, min=min_count)
        log_z = torch.log(z_safe)
        log_lam = torch.log(lam_safe)
        log_var = obs_var / torch.clamp(z_safe**2, min=1.0e-12)
        log_var = torch.clamp(
            log_var + sigma_floor**2,
            min=sigma_floor**2,
            max=sigma_ceiling**2,
        )
        view_weight = torch.reciprocal(log_var)
        weight_sum = torch.clamp(
            torch.sum(view_weight, dim=0, keepdim=True),
            min=1e-12,
        )
        obs_center = log_z - (
            torch.sum(view_weight * log_z, dim=0, keepdim=True) / weight_sum
        )
        pred_center = (
            log_lam
            - torch.sum(
                view_weight * log_lam,
                dim=0,
                keepdim=True,
            )
            / weight_sum
        )
        residual = obs_center - pred_center
        terms = -0.5 * (df + 1.0) * torch.log1p((residual**2) / (df * log_var))
        terms -= 0.5 * torch.log(log_var)
        return float(weight) * torch.sum(terms, dim=0)

    def _shield_view_ratio_sequence_log_likelihood_gpu(
        self,
        lam_kn: "torch.Tensor",
        z_obs: NDArray[np.float64],
    ) -> "torch.Tensor":
        """Return a Dirichlet-multinomial shield-view ratio likelihood term."""
        import torch

        lam = lam_kn.to(dtype=torch.float64)
        if lam.ndim != 2:
            raise ValueError("Sequence expected counts must have shape K x N.")
        particle_count = int(lam.shape[1]) if lam.ndim == 2 else 0
        zeros = torch.zeros(particle_count, device=lam.device, dtype=torch.float64)
        if not bool(getattr(self.config, "shield_view_ratio_likelihood_enable", False)):
            return zeros
        weight = max(
            float(getattr(self.config, "shield_view_ratio_likelihood_weight", 1.0)),
            0.0,
        )
        concentration = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_concentration",
                    128.0,
                )
            ),
            1.0e-6,
        )
        min_total = max(
            float(
                getattr(
                    self.config,
                    "shield_view_ratio_likelihood_min_total_count",
                    25.0,
                )
            ),
            0.0,
        )
        min_views = max(
            int(getattr(self.config, "shield_view_ratio_likelihood_min_views", 2)),
            2,
        )
        if weight <= 0.0 or int(lam.shape[0]) < min_views or particle_count == 0:
            return zeros
        z_arr = np.asarray(z_obs, dtype=float).reshape(-1)
        if z_arr.size != int(lam.shape[0]):
            raise ValueError("Observation arrays must match the sequence length.")
        z_arr = np.maximum(np.where(np.isfinite(z_arr), z_arr, 0.0), 0.0)
        total = float(np.sum(z_arr))
        if total < min_total or not np.isfinite(total):
            return zeros
        z = torch.as_tensor(
            z_arr,
            device=lam.device,
            dtype=torch.float64,
        ).view(-1, 1)
        lam_safe = torch.clamp(lam, min=1.0e-12)
        lam_total = torch.clamp(torch.sum(lam_safe, dim=0, keepdim=True), min=1.0e-12)
        probabilities = lam_safe / lam_total
        alpha = torch.clamp(concentration * probabilities, min=1.0e-12)
        alpha0 = torch.clamp(torch.sum(alpha, dim=0), min=1.0e-12)
        total_t = torch.as_tensor(total, device=lam.device, dtype=torch.float64)
        ll = torch.lgamma(alpha0) - torch.lgamma(alpha0 + total_t)
        ll = ll + torch.sum(torch.lgamma(alpha + z) - torch.lgamma(alpha), dim=0)
        return float(weight) * ll

    def _normalized_log_weights_torch(self, logw: "torch.Tensor") -> "torch.Tensor":
        """Normalize log-weights using logsumexp in float64."""
        import torch

        return logw - torch.logsumexp(logw, dim=0)

    def _ess_from_logw_torch(self, logw: "torch.Tensor") -> float:
        """Return the effective sample size from normalized log-weights."""
        import torch

        w = torch.exp(logw)
        ess = 1.0 / torch.sum(w**2)
        return float(ess.detach().cpu().item())

    def _assign_logw_from_torch(self, logw: "torch.Tensor") -> None:
        """Copy log-weights from torch back into particle objects."""
        logw_cpu = logw.detach().cpu().numpy()
        for p, lw in zip(self.continuous_particles, logw_cpu):
            p.log_weight = float(lw)

    def _update_continuous_weights_gpu(
        self,
        lam_t: "torch.Tensor",
        z_obs: float,
        *,
        observation_count_variance: float = 0.0,
        delta_beta: float = 1.0,
        logw_prev: "torch.Tensor | None" = None,
        ll_t: "torch.Tensor | None" = None,
        return_logw: bool = False,
    ) -> "torch.Tensor | None":
        """
        Update continuous log-weights using tempered Poisson increments.

        When return_logw is True, returns the normalized log-weights after the update.
        """
        if lam_t.numel() == 0:
            return
        logw_prev = (
            logw_prev
            if logw_prev is not None
            else self._current_log_weights_torch(lam_t.device)
        )
        ll_t = (
            ll_t
            if ll_t is not None
            else self._log_likelihood_increment_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
            )
        )
        logw = self._normalized_log_weights_torch(logw_prev + float(delta_beta) * ll_t)
        self._assign_logw_from_torch(logw)
        if return_logw:
            return logw
        return None

    def _select_delta_beta(
        self,
        logw_prev: "torch.Tensor",
        ll_t: "torch.Tensor",
        remaining: float,
        target_ess: float,
    ) -> tuple[float, "torch.Tensor", float]:
        """
        Return the largest delta_beta that keeps ESS above the target.

        Returns (delta_beta, logw_new, ess).
        """
        remaining = float(remaining)
        min_delta = max(float(self.config.min_delta_beta), 0.0)
        if remaining <= min_delta:
            logw_new = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
            ess = self._ess_from_logw_torch(logw_new)
            return remaining, logw_new, ess

        logw_full = self._normalized_log_weights_torch(logw_prev + remaining * ll_t)
        ess_full = self._ess_from_logw_torch(logw_full)
        if ess_full >= target_ess:
            return remaining, logw_full, ess_full

        logw_low = self._normalized_log_weights_torch(logw_prev + min_delta * ll_t)
        ess_low = self._ess_from_logw_torch(logw_low)
        if ess_low < target_ess:
            return min_delta, logw_low, ess_low

        low = min_delta
        high = remaining
        logw_best = logw_low
        ess_best = ess_low
        for _ in range(24):
            mid = 0.5 * (low + high)
            logw_mid = self._normalized_log_weights_torch(logw_prev + mid * ll_t)
            ess_mid = self._ess_from_logw_torch(logw_mid)
            if ess_mid >= target_ess:
                low = mid
                logw_best = logw_mid
                ess_best = ess_mid
            else:
                high = mid
        return low, logw_best, ess_best

    def _tempered_update(
        self,
        lam_fn: Callable[[], "torch.Tensor"],
        z_obs: float,
        observation_count_variance: float = 0.0,
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering for a single Poisson update.

        The update increments beta from 0 to 1 using delta_beta steps that
        maintain ESS above the configured target ratio when possible.

        Returns (ess_pre, resampled_any) for downstream adaptation logic.
        """

        def _ll_fn() -> "torch.Tensor":
            """Return per-particle log-likelihood increments for one count."""
            import torch

            lam_t_inner = lam_fn()
            if lam_t_inner.numel() == 0:
                return lam_t_inner.to(dtype=torch.float64)
            return self._log_likelihood_increment_gpu(
                lam_t_inner,
                z_obs,
                observation_count_variance=observation_count_variance,
            )

        return self._tempered_update_likelihood(ll_fn=_ll_fn)

    def _tempered_update_likelihood(
        self,
        ll_fn: Callable[[], "torch.Tensor"],
    ) -> tuple[float, bool]:
        """
        Apply ESS-targeted tempering to a precomputed likelihood increment.

        ``ll_fn`` is re-evaluated after a tempering resample so the likelihood
        remains consistent with the resampled particle array.
        """
        beta_total = 0.0
        steps: list[dict[str, float]] = []
        resamples = 0
        resampled_any = False
        ess_min: float | None = None
        target_ess = float(self.config.target_ess_ratio) * max(self.N, 1)
        resample_threshold = float(self.config.resample_threshold) * max(self.N, 1)
        max_resamples = max(0, int(self.config.max_resamples_per_observation))
        cooldown_steps = max(0, int(self.config.temper_resample_cooldown_steps))
        force_resample_ess = float(self.config.temper_resample_force_ratio) * max(
            self.N, 1
        )
        ll_t = ll_fn()
        if ll_t.numel() == 0:
            self.last_temper_steps = []
            self.last_temper_resample_count = 0
            return 0.0, False
        logw = self._current_log_weights_torch(ll_t.device)

        cooldown_remaining = 0
        while beta_total < 1.0 - 1e-12:
            remaining = 1.0 - beta_total
            delta_beta, logw_new, ess = self._select_delta_beta(
                logw_prev=logw,
                ll_t=ll_t,
                remaining=remaining,
                target_ess=target_ess,
            )
            logw = logw_new
            self._assign_logw_from_torch(logw)
            beta_total += delta_beta
            ess_min = ess if ess_min is None else min(ess_min, ess)
            steps.append(
                {
                    "beta_total": float(beta_total),
                    "delta_beta": float(delta_beta),
                    "ess": float(ess),
                }
            )
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            do_resample = (
                ess < resample_threshold
                and resamples < max_resamples
                and (cooldown_remaining == 0 or ess < force_resample_ess)
            )
            can_jump_without_intermediate_state_change = (
                not do_resample
                and cooldown_remaining == 0
                and delta_beta < remaining - 1.0e-12
            )
            if can_jump_without_intermediate_state_change:
                remaining_after_delta = 1.0 - beta_total
                logw_full = self._normalized_log_weights_torch(
                    logw + remaining_after_delta * ll_t
                )
                ess_full = self._ess_from_logw_torch(logw_full)
                logw = logw_full
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess_min = ess_full if ess_min is None else min(ess_min, ess_full)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining_after_delta),
                        "ess": float(ess_full),
                    }
                )
                if ess_full < resample_threshold and resamples < max_resamples:
                    self._maybe_resample_continuous()
                    if self.last_resample_ess:
                        resampled_any = True
                        resamples += 1
                break
            if do_resample:
                self._maybe_resample_continuous()
                if self.last_resample_ess:
                    resampled_any = True
                    resamples += 1
                    cooldown_remaining = max(cooldown_remaining, cooldown_steps)
                    ll_t = ll_fn()
                    if ll_t.numel() == 0:
                        break
                    logw = self._current_log_weights_torch(ll_t.device)
            if resamples >= max_resamples and beta_total < 1.0 - 1e-12:
                remaining = 1.0 - beta_total
                logw = self._normalized_log_weights_torch(logw + remaining * ll_t)
                self._assign_logw_from_torch(logw)
                beta_total = 1.0
                ess = self._ess_from_logw_torch(logw)
                ess_min = ess if ess_min is None else min(ess_min, ess)
                steps.append(
                    {
                        "beta_total": float(beta_total),
                        "delta_beta": float(remaining),
                        "ess": float(ess),
                    }
                )
                break
        self.last_temper_steps = steps
        self.last_temper_resample_count = resamples
        if ess_min is None:
            ess_min = 0.0
        return float(ess_min), resampled_any

    def _continuous_expected_counts_gpu(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for a single orientation index."""
        lam_t = self._continuous_expected_counts_torch(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_pair_gpu(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using torch for Fe/Pb orientation indices."""
        lam_t = self._continuous_expected_counts_pair_torch(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        return lam_t.detach().cpu().numpy()

    def _continuous_expected_counts_cpu(
        self,
        pose_idx: int,
        orient_idx: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute single-orientation expected counts on CPU."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        lam = np.zeros(len(self.continuous_particles), dtype=float)
        source_scale = self._measurement_source_scale(
            fe_index=orient_idx,
            pb_index=orient_idx,
        )
        for particle_idx, particle in enumerate(self.continuous_particles):
            state = particle.state
            rate = float(state.background)
            for pos, strength in zip(
                state.positions[: state.num_sources],
                state.strengths[: state.num_sources],
            ):
                kernel_val = self.continuous_kernel.kernel_value(
                    isotope=self.isotope,
                    detector_pos=detector_pos,
                    source_pos=pos,
                    orient_idx=int(orient_idx),
                )
                rate += source_scale * float(kernel_val) * float(strength)
            lam[particle_idx] = float(live_time_s) * rate
        return lam

    def _continuous_expected_counts_pair_cpu(
        self,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute one Fe/Pb-pair row through the batched CPU kernel."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        counts = self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=detector_pos,
            fe_indices=np.asarray([int(fe_index)], dtype=np.int64),
            pb_indices=np.asarray([int(pb_index)], dtype=np.int64),
            live_times_s=np.asarray([float(live_time_s)], dtype=np.float64),
        )
        return counts[0]

    def _continuous_expected_counts_pair_sequence_cpu(
        self,
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute all views x particles x source slots in one CPU batch."""
        if self.kernel is None:
            raise RuntimeError("Continuous PF update requires an attached kernel.")
        detector_pos = np.asarray(self.kernel.poses[int(pose_idx)], dtype=float)
        return self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=detector_pos,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            live_times_s=live_times_s,
        )

    def _continuous_expected_counts_pair_sequence_at_pose_cpu(
        self,
        detector_pos: NDArray[np.float64],
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return selected-pair counts from the production kernel on CPU.

        The production packed-state kernel vectorizes shield views, particles,
        and source slots.  Running that same kernel on a CPU device avoids a
        scalar view/particle loop while retaining line, aperture, obstacle, and
        transport-response semantics.
        """
        from pf import gpu_utils
        import torch

        fe_arr = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
        pb_arr = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
        live_arr = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
        if not (fe_arr.size == pb_arr.size == live_arr.size):
            raise ValueError("Fe, Pb, and live-time arrays must have matching lengths.")
        particle_count = len(self.continuous_particles)
        if fe_arr.size == 0 or particle_count == 0:
            return np.zeros((fe_arr.size, particle_count), dtype=np.float64)
        device = torch.device("cpu")
        dtype = torch.float64
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            [particle.state for particle in self.continuous_particles],
            device=device,
            dtype=dtype,
        )
        (
            packed_positions,
            packed_strengths,
            packed_mask,
            inverse,
        ) = self._compress_identical_packed_sources_torch(
            positions,
            strengths,
            mask,
        )
        if inverse is None:
            unit_live_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=np.asarray(detector_pos, dtype=float),
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
        else:
            source_unit_counts = self.continuous_kernel.expected_counts_selected_pairs_for_packed_states_torch(
                isotope=self.isotope,
                detector_pos=np.asarray(detector_pos, dtype=float),
                positions=packed_positions,
                strengths=packed_strengths,
                backgrounds=torch.zeros(
                    int(packed_positions.shape[0]),
                    device=device,
                    dtype=dtype,
                ),
                mask=packed_mask,
                fe_indices=fe_arr,
                pb_indices=pb_arr,
                live_time_s=1.0,
                source_scale=self._measurement_source_scale_vector(fe_arr, pb_arr),
                device=device,
                dtype=dtype,
            )
            unit_live_counts = source_unit_counts.index_select(1, inverse)
            unit_live_counts = unit_live_counts + backgrounds.unsqueeze(0)
        live_t = torch.as_tensor(live_arr, device=device, dtype=dtype).view(-1, 1)
        return (
            unit_live_counts.mul(live_t)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    def _continuous_expected_counts_pair_at_pose_cpu(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute one explicit-pose pair through the batched CPU kernel."""
        counts = self._continuous_expected_counts_pair_sequence_at_pose_cpu(
            detector_pos=np.asarray(detector_pos, dtype=float),
            fe_indices=np.asarray([int(fe_index)], dtype=np.int64),
            pb_indices=np.asarray([int(pb_index)], dtype=np.int64),
            live_times_s=np.asarray([float(live_time_s)], dtype=np.float64),
        )
        return counts[0]

    def _continuous_expected_counts(
        self, pose_idx: int, orient_idx: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for each continuous particle using ContinuousKernel."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_cpu(
                pose_idx=pose_idx,
                orient_idx=orient_idx,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_gpu(
            pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
        )

    def _continuous_expected_counts_pair(
        self, pose_idx: int, fe_index: int, pb_index: int, live_time_s: float
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} using Fe/Pb octant indices (Eq. 3.41)."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_cpu(
                pose_idx=pose_idx,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        return self._continuous_expected_counts_pair_gpu(
            pose_idx=pose_idx,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )

    def _continuous_expected_counts_pair_at_pose_torch(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> "torch.Tensor":
        """Compute Λ_{k,h}^{(n)} using torch for explicit detector position."""
        from pf import gpu_utils
        import torch

        device = gpu_utils.resolve_device(self.config.gpu_device)
        dtype = gpu_utils.resolve_dtype(self.config.gpu_dtype)
        if not self.continuous_particles:
            return torch.zeros(0, device=device, dtype=dtype)
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(
            [p.state for p in self.continuous_particles],
            device=device,
            dtype=dtype,
        )
        det_pos = np.asarray(detector_pos, dtype=float)
        return self.continuous_kernel.expected_counts_pair_for_packed_states_torch(
            isotope=self.isotope,
            detector_pos=det_pos,
            positions=positions,
            strengths=strengths,
            backgrounds=backgrounds,
            mask=mask,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            source_scale=self._measurement_source_scale(
                fe_index=fe_index,
                pb_index=pb_index,
            ),
            device=device,
            dtype=dtype,
        )

    def _continuous_expected_counts_pair_at_pose(
        self,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
    ) -> NDArray[np.float64]:
        """Compute Λ_{k,h}^{(n)} for explicit detector position."""
        if not self._can_use_gpu():
            return self._continuous_expected_counts_pair_at_pose_cpu(
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
            )
        lam_t = self._continuous_expected_counts_pair_at_pose_torch(
            detector_pos=detector_pos,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
        )
        return lam_t.detach().cpu().numpy()

    def update_continuous_pair(
        self,
        z_obs: float,
        pose_idx: int,
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
        defer_resample: bool = False,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Count-likelihood weight update using Fe/Pb orientation indices.

        z_obs must come from spectrum unfolding; expected Λ_{k,h} is computed via expected_counts_pair.
        When ``defer_resample`` is True, particle-count adaptation and
        convergence bookkeeping are deferred to the caller's end-of-station
        finalization. ESS-triggered resampling remains active for each posture.
        """
        self.reset_step_stats()
        detector_pos = (
            np.asarray(self.kernel.poses[pose_idx], dtype=float)
            if self.kernel
            else None
        )
        self._record_observed_station(detector_pos)

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            if self.config.use_gpu:
                self._gpu_enabled()
                return self._continuous_expected_counts_pair_torch(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            import torch

            return torch.as_tensor(
                self._continuous_expected_counts_pair_cpu(
                    pose_idx=pose_idx,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                ),
                dtype=torch.float64,
                device="cpu",
            )

        candidate_spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=1,
        )
        spectrum_arrays = (
            candidate_spectrum_arrays
            if self._direct_spectrum_route_admissible(
                sequence_length=1,
                observation_count_covariance=None,
            )
            else None
        )
        if spectrum_arrays is not None:
            self.last_spectrum_likelihood_route = "direct_spectrum"
        else:
            self.last_spectrum_likelihood_route = "count"

        def _spectral_ll_fn() -> "torch.Tensor":
            """Return direct spectrum-bin likelihood increments for one view."""
            if spectrum_arrays is None:
                raise RuntimeError("spectrum_arrays must be available.")
            observed, template, background, variance = spectrum_arrays
            lam_t_inner = _lam_fn()
            return self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                lam_t_inner,
                observed,
                template,
                background,
                variance,
            )

        if self.config.use_tempering and spectrum_arrays is not None:
            ess_pre, resampled_any = self._tempered_update_likelihood(
                ll_fn=_spectral_ll_fn,
            )
        elif self.config.use_tempering:
            debug_timing = _pf_debug_timing_enabled()
            debug_start = time.perf_counter()
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_start defer={defer_resample} "
                    f"fe={fe_index} pb={pb_index} z={float(z_obs):.6g}",
                    flush=True,
                )
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
            )
            if debug_timing:
                print(
                    f"[pf_internal] isotope={self.isotope} step={step_idx} "
                    f"phase=tempered_done elapsed={time.perf_counter() - debug_start:.3f}s "
                    f"resampled={resampled_any} ess={float(ess_pre):.3f}",
                    flush=True,
                )
        elif spectrum_arrays is not None:
            ll_t = _spectral_ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous()
            resampled_any = bool(self.last_resample_ess)
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous()
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        if defer_resample:
            self._deferred_resampled_any = self._deferred_resampled_any or bool(
                resampled_any
            )
            if np.isfinite(float(ess_pre)):
                if self._deferred_ess_min is None:
                    self._deferred_ess_min = float(ess_pre)
                else:
                    self._deferred_ess_min = min(
                        float(self._deferred_ess_min),
                        float(ess_pre),
                    )
        else:
            self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
            self._advance_adapt_cooldown()
        if detector_pos is not None:
            if defer_resample:
                self._deferred_convergence_args = (
                    step_idx,
                    np.asarray(detector_pos, dtype=float).copy(),
                    int(fe_index),
                    int(pb_index),
                    float(live_time_s),
                    float(z_obs),
                )
                return
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )

    def finalize_deferred_update(self) -> None:
        """Finalize deferred particle adaptation and convergence bookkeeping."""
        if self._deferred_ess_min is not None:
            ess_pre = float(self._deferred_ess_min)
        elif self.last_ess_pre is not None:
            ess_pre = float(self.last_ess_pre)
        else:
            weights = np.asarray(self.continuous_weights, dtype=float)
            ess_pre = (
                float(1.0 / max(np.sum(weights**2), 1.0e-12)) if weights.size else 0.0
            )
        resampled_any = bool(self._deferred_resampled_any)
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self._advance_adapt_cooldown()
        convergence_args = self._deferred_convergence_args
        self._deferred_convergence_args = None
        if convergence_args is not None:
            step_idx, detector_pos, fe_index, pb_index, live_time_s, z_obs = (
                convergence_args
            )
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=fe_index,
                pb_index=pb_index,
                live_time_s=live_time_s,
                z_obs=z_obs,
            )
        self._deferred_resampled_any = False
        self._deferred_ess_min = None

    def update_continuous_pair_sequence(
        self,
        z_obs: NDArray[np.float64],
        pose_idx: int,
        fe_indices: NDArray[np.int64],
        pb_indices: NDArray[np.int64],
        live_times_s: NDArray[np.float64],
        observation_count_variances: NDArray[np.float64] | None = None,
        observation_count_covariance: NDArray[np.float64] | None = None,
        step_idx: int | None = None,
        spectrum_counts: NDArray[np.float64] | None = None,
        spectrum_response_template: NDArray[np.float64] | None = None,
        spectrum_background: NDArray[np.float64] | None = None,
        spectrum_variance: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Jointly update weights using a same-pose shield-orientation sequence.

        The measurement model evaluates the shield program as one station-level
        observation. When covariance is supplied or configured, same-station
        shield-view correlations are handled by a batched multivariate
        likelihood; otherwise the update reduces to the product likelihood over
        views. Updating the views jointly evaluates their shared likelihood
        before any station-level particle-count adaptation.
        """
        self.reset_step_stats()
        detector_pos = (
            np.asarray(self.kernel.poses[pose_idx], dtype=float)
            if self.kernel
            else None
        )
        self._record_observed_station(detector_pos)
        z_arr = np.asarray(z_obs, dtype=float).ravel()
        fe_arr = np.asarray(fe_indices, dtype=int).ravel()
        pb_arr = np.asarray(pb_indices, dtype=int).ravel()
        live_arr = np.asarray(live_times_s, dtype=float).ravel()
        if observation_count_variances is None:
            var_arr = np.zeros_like(z_arr, dtype=float)
        else:
            var_arr = np.asarray(observation_count_variances, dtype=float).ravel()
        covariance = None
        if observation_count_covariance is not None:
            covariance = np.asarray(observation_count_covariance, dtype=float)
            if covariance.shape != (z_arr.size, z_arr.size):
                raise ValueError("observation_count_covariance must be shaped K x K.")
        if not (
            z_arr.size == fe_arr.size == pb_arr.size == live_arr.size == var_arr.size
        ):
            raise ValueError("Joint PF update arrays must have matching lengths.")
        if z_arr.size == 0:
            return
        candidate_spectrum_arrays = self._spectrum_update_arrays(
            spectrum_counts,
            spectrum_response_template,
            spectrum_background,
            spectrum_variance,
            sequence_length=z_arr.size,
        )
        spectrum_arrays = (
            candidate_spectrum_arrays
            if self._direct_spectrum_route_admissible(
                sequence_length=int(z_arr.size),
                observation_count_covariance=covariance,
            )
            else None
        )
        if spectrum_arrays is not None:
            self.last_spectrum_likelihood_route = "direct_spectrum"
        elif self._sequence_covariance_enabled(
            z_arr.size,
            covariance,
        ):
            self.last_spectrum_likelihood_route = "count_covariance"
        else:
            self.last_spectrum_likelihood_route = "count"

        def _ll_fn() -> "torch.Tensor":
            """Return summed per-particle log-likelihood for the shield sequence."""
            if self.config.use_gpu:
                self._gpu_enabled()
                lam_kn = self._continuous_expected_counts_pair_sequence_torch(
                    pose_idx=pose_idx,
                    fe_indices=fe_arr,
                    pb_indices=pb_arr,
                    live_times_s=live_arr,
                )
            else:
                import torch

                lam_kn = torch.as_tensor(
                    self._continuous_expected_counts_pair_sequence_cpu(
                        pose_idx=pose_idx,
                        fe_indices=fe_arr,
                        pb_indices=pb_arr,
                        live_times_s=live_arr,
                    ),
                    dtype=torch.float64,
                    device="cpu",
                )
            if spectrum_arrays is not None:
                observed, template, background_spectrum, variance = spectrum_arrays
                spectrum_ll = (
                    self._spectral_bin_sequence_log_likelihood_from_lambda_gpu(
                        lam_kn,
                        observed,
                        template,
                        background_spectrum,
                        variance,
                    )
                )
                return spectrum_ll + self._shield_shape_sequence_log_likelihood_gpu(
                    lam_kn,
                    z_arr,
                    var_arr,
                )
            return self._log_likelihood_sequence_gpu(
                lam_kn,
                z_arr,
                var_arr,
                observation_count_covariance=covariance,
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update_likelihood(
                ll_fn=_ll_fn,
            )
        else:
            ll_t = _ll_fn()
            if ll_t.numel() == 0:
                ess_pre = 0.0
                resampled_any = False
            else:
                logw_prev = self._current_log_weights_torch(ll_t.device)
                logw = self._normalized_log_weights_torch(logw_prev + ll_t)
                self._assign_logw_from_torch(logw)
                ess_pre = self._ess_from_logw_torch(logw)
                self._maybe_resample_continuous()
                resampled_any = bool(self.last_resample_ess)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self._advance_adapt_cooldown()
        if detector_pos is not None:
            self._maybe_update_convergence(
                step_idx=step_idx,
                detector_pos=detector_pos,
                fe_index=int(fe_arr[-1]),
                pb_index=int(pb_arr[-1]),
                live_time_s=float(np.sum(live_arr)),
                z_obs=float(np.sum(z_arr)),
            )

    def update_continuous_pair_at_pose(
        self,
        z_obs: float,
        detector_pos: NDArray[np.float64],
        fe_index: int,
        pb_index: int,
        live_time_s: float,
        observation_count_variance: float = 0.0,
        step_idx: int | None = None,
    ) -> None:
        """
        Count-likelihood weight update using explicit detector position.

        This avoids reliance on pose indices for planning-time evaluations.
        """
        self.reset_step_stats()
        self.last_spectrum_likelihood_route = "count"

        def _lam_fn() -> "torch.Tensor":
            """Return expected counts for the current particle set."""
            if self.config.use_gpu:
                self._gpu_enabled()
                return self._continuous_expected_counts_pair_at_pose_torch(
                    detector_pos=detector_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                )
            import torch

            return torch.as_tensor(
                self._continuous_expected_counts_pair_at_pose_cpu(
                    detector_pos=detector_pos,
                    fe_index=fe_index,
                    pb_index=pb_index,
                    live_time_s=live_time_s,
                ),
                dtype=torch.float64,
                device="cpu",
            )

        if self.config.use_tempering:
            ess_pre, resampled_any = self._tempered_update(
                lam_fn=_lam_fn,
                z_obs=z_obs,
                observation_count_variance=observation_count_variance,
            )
        else:
            lam_t = _lam_fn()
            logw = self._update_continuous_weights_gpu(
                lam_t,
                z_obs,
                observation_count_variance=observation_count_variance,
                return_logw=True,
            )
            if logw is None:
                ess_pre = 0.0
            else:
                ess_pre = self._ess_from_logw_torch(logw)
            self._maybe_resample_continuous()
            resampled_any = bool(self.last_resample_ess)
            if logw is None and self.last_ess_pre is not None:
                ess_pre = float(self.last_ess_pre)
        if resampled_any:
            self._trigger_adapt_cooldown()
        self.adapt_num_particles(ess_pre=ess_pre, resampled=resampled_any)
        self._advance_adapt_cooldown()
        self._maybe_update_convergence(
            step_idx=step_idx,
            detector_pos=np.asarray(detector_pos, dtype=float),
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=live_time_s,
            z_obs=z_obs,
        )

    @property
    def continuous_weights(self) -> NDArray[np.float64]:
        """Return normalized weights for continuous particles."""
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles], dtype=np.float64
        )
        if logw.size == 0:
            return np.zeros(0, dtype=float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        s = np.sum(w)
        if s <= 0:
            return np.ones(len(self.continuous_particles), dtype=float) / len(
                self.continuous_particles
            )
        return w / s

    def _normalize_continuous_log_weights(self) -> None:
        """Normalize continuous-particle log weights in place."""
        if not self.continuous_particles:
            return
        logw = np.asarray(
            [p.log_weight for p in self.continuous_particles],
            dtype=np.float64,
        )
        norm = logsumexp(logw)
        if not np.isfinite(norm):
            uniform = -np.log(max(len(self.continuous_particles), 1))
            for particle in self.continuous_particles:
                particle.log_weight = float(uniform)
            return
        for particle, value in zip(self.continuous_particles, logw - norm):
            particle.log_weight = float(value)

    def _maybe_resample_continuous(self) -> None:
        """Apply standard ESS-triggered systematic resampling."""
        weights = np.asarray(self.continuous_weights, dtype=np.float64)
        if weights.size == 0:
            self.last_ess = 0.0
            self.last_ess_pre = 0.0
            self.last_ess_post = 0.0
            self.last_resample_ess = False
            return
        ess = 1.0 / max(float(np.sum(weights**2)), 1.0e-12)
        self.last_ess = float(ess)
        self.last_ess_pre = float(ess)
        self.last_ess_post = None
        self.last_resample_ess = False
        if ess >= float(self.config.resample_threshold) * self.N:
            return
        self.last_resample_ess = True
        self.last_resample_count += 1
        indices = systematic_resample(np.log(np.clip(weights, 1.0e-300, 1.0)))
        uniform_log_weight = float(-np.log(max(indices.size, 1)))
        self.continuous_particles = [
            IsotopeParticle(
                state=self.continuous_particles[int(index)].state.copy(),
                log_weight=uniform_log_weight,
            )
            for index in indices
        ]
        post_weights = np.asarray(self.continuous_weights, dtype=np.float64)
        self.last_ess_post = float(1.0 / max(float(np.sum(post_weights**2)), 1.0e-12))
        self._resample_count_in_observation += 1

    def adapt_num_particles(
        self, *, ess_pre: float | None = None, resampled: bool = False
    ) -> None:
        """
        Optional: adapt N based on variance/entropy of weights (Chapter 3.3.4).

        Uses ess_pre when provided to avoid the resampling inflation of ESS.
        Resampling or cooldown windows only allow growth.
        """
        if not self.continuous_particles:
            self.last_n_after_adapt = 0
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
        if ess_pre is None:
            w = self.continuous_weights
            if w.size == 0:
                ess_pre = 0.0
            else:
                ess_pre = float(1.0 / max(np.sum(w**2), 1e-12))
        ess_ratio = float(ess_pre) / max(len(self.continuous_particles), 1)
        allow_shrink = not resampled and self._adapt_cooldown_remaining <= 0
        if (
            ess_ratio < self.config.ess_low
            and len(self.continuous_particles) < max_particles
        ):
            grown = max(
                len(self.continuous_particles) + 1,
                int(len(self.continuous_particles) * 1.25),
            )
            target = min(max_particles, grown)
            self._resample_continuous_to(target)
        elif (
            allow_shrink
            and ess_ratio > self.config.ess_high
            and len(self.continuous_particles) > min_particles
        ):
            target = max(min_particles, int(len(self.continuous_particles) * 0.8))
            self._resample_continuous_to(target)
        self.last_n_after_adapt = int(len(self.continuous_particles))

    def _resample_continuous_to(self, target_n: int) -> None:
        """Systematically resample the particle array to a requested size."""
        requested = max(1, int(target_n))
        if not self.continuous_particles:
            self.N = requested
            self.config.num_particles = requested
            self._init_continuous_particles()
            return
        weights = np.asarray(self.continuous_weights, dtype=np.float64)
        indices = systematic_resample_count(weights, count=requested)
        uniform_log_weight = float(-np.log(requested))
        self.continuous_particles = [
            IsotopeParticle(
                state=self.continuous_particles[int(index)].state.copy(),
                log_weight=uniform_log_weight,
            )
            for index in indices
        ]
        self.N = requested
        self.config.num_particles = requested

    def best_particle(self) -> IsotopeParticle:
        """Return the particle with maximum log_weight."""
        return max(self.continuous_particles, key=lambda p: p.log_weight)

    def _structural_rj_response_signatures(
        self,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return response-relevant row signatures for safe prefix caching."""
        measurement_count = int(data.z_k.size)
        detector_positions = np.asarray(
            data.detector_positions,
            dtype=float,
        ).reshape(measurement_count, 3)
        live_times = np.asarray(
            data.live_times,
            dtype=float,
        ).reshape(measurement_count)
        fe_indices = np.asarray(
            data.fe_indices,
            dtype=np.int64,
        ).reshape(measurement_count)
        pb_indices = np.asarray(
            data.pb_indices,
            dtype=np.int64,
        ).reshape(measurement_count)
        source_scales = self._measurement_source_scale_vector(
            fe_indices,
            pb_indices,
        )
        signatures = np.column_stack(
            [
                detector_positions,
                live_times,
                fe_indices,
                pb_indices,
                source_scales,
            ]
        ).astype(np.float64, copy=False)
        if not np.all(np.isfinite(signatures)):
            raise ValueError("rj_mh response signatures must contain finite values.")
        return signatures

    def _structural_rj_prepare_response_cache(
        self,
        data: MeasurementData,
    ) -> NDArray[np.float64]:
        """Return a prefix-safe lazy response cache for the supplied history."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        signatures = self._structural_rj_response_signatures(data)
        measurement_count = int(signatures.shape[0])
        patch_count = int(patches.patch_count)
        cached = self._structural_rj_response_cache
        cached_signatures = self._structural_rj_response_cache_signatures

        matching_prefix = 0
        if cached is not None and cached_signatures is not None:
            matching_prefix = min(
                measurement_count,
                int(cached_signatures.shape[0]),
            )
            prefix_matches = np.array_equal(
                signatures[:matching_prefix],
                cached_signatures[:matching_prefix],
            )
        else:
            prefix_matches = False

        if (
            cached is None
            or cached_signatures is None
            or cached.shape[1] != patch_count
            or not prefix_matches
        ):
            cached = np.full(
                (measurement_count, patch_count),
                np.nan,
                dtype=float,
            )
            cached_signatures = signatures.copy()
        elif measurement_count > int(cached.shape[0]):
            added_rows = measurement_count - int(cached.shape[0])
            cached = np.vstack(
                [
                    cached,
                    np.full((added_rows, patch_count), np.nan, dtype=float),
                ]
            )
            cached_signatures = signatures.copy()

        self._structural_rj_response_cache = cached
        self._structural_rj_response_cache_signatures = cached_signatures
        return cached[:measurement_count]

    def _structural_rj_evaluate_response_columns(
        self,
        data: MeasurementData,
        patch_indices: NDArray[np.int64],
        *,
        row_start: int,
    ) -> NDArray[np.float64]:
        """Evaluate one batched suffix for selected global surface patches."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        indices = np.unique(np.asarray(patch_indices, dtype=np.int64).reshape(-1))
        start = max(0, int(row_start))
        measurement_count = int(data.z_k.size)
        if start > measurement_count:
            raise ValueError("row_start exceeds the measurement history.")
        if np.any(indices < 0) or np.any(indices >= patches.patch_count):
            raise ValueError("rj_mh response request contains an invalid patch.")
        if indices.size == 0 or start == measurement_count:
            return np.zeros(
                (measurement_count - start, indices.size),
                dtype=float,
            )
        fe_indices = np.asarray(data.fe_indices, dtype=np.int64)[start:]
        pb_indices = np.asarray(data.pb_indices, dtype=np.int64)[start:]
        responses = expected_counts_per_source(
            kernel=self.continuous_kernel,
            isotope=self.isotope,
            detector_positions=np.asarray(
                data.detector_positions,
                dtype=float,
            )[start:],
            sources=np.asarray(
                patches.centers_xyz,
                dtype=float,
            )[indices],
            strengths=np.ones(indices.size, dtype=float),
            live_times=np.asarray(data.live_times, dtype=float)[start:],
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            source_scale=self._measurement_source_scale_vector(
                fe_indices,
                pb_indices,
            ),
        )
        response_array = np.asarray(responses, dtype=float)
        expected_shape = (measurement_count - start, int(indices.size))
        if response_array.shape != expected_shape:
            raise ValueError("rj_mh response columns have an unexpected shape.")
        if not np.all(np.isfinite(response_array)):
            raise ValueError("rj_mh response columns must be finite.")
        self._structural_rj_response_evaluation_batches += 1
        self._structural_rj_response_evaluated_cells += int(response_array.size)
        touched_mask = self._structural_rj_response_touched_mask
        if touched_mask is None or touched_mask.size != patches.patch_count:
            touched_mask = np.zeros(patches.patch_count, dtype=bool)
            self._structural_rj_response_touched_mask = touched_mask
        touched_mask[indices] = True
        return response_array

    def _structural_rj_ensure_response_columns(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
        patch_indices: NDArray[np.int64],
    ) -> None:
        """Fill only missing cached row suffixes for required patch columns."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        response_array = np.asarray(response_dictionary, dtype=float)
        expected_shape = (int(data.z_k.size), int(patches.patch_count))
        if response_array.shape != expected_shape:
            raise ValueError("rj_mh response dictionary has an unexpected shape.")
        required = np.unique(np.asarray(patch_indices, dtype=np.int64).reshape(-1))
        if required.size == 0:
            return
        if np.any(required < 0) or np.any(required >= patches.patch_count):
            raise ValueError("rj_mh response request contains an invalid patch.")

        finite = np.isfinite(response_array[:, required])
        valid_prefixes = np.sum(finite, axis=0).astype(
            np.int64,
            copy=False,
        )
        row_indices = np.arange(response_array.shape[0])[:, None]
        prefix_mask = row_indices < valid_prefixes[None, :]
        if not np.array_equal(finite, prefix_mask):
            raise ValueError(
                "rj_mh response cache contains a non-prefix validity pattern."
            )
        for valid_prefix in np.unique(valid_prefixes).tolist():
            if int(valid_prefix) >= response_array.shape[0]:
                continue
            selected = required[valid_prefixes == int(valid_prefix)]
            evaluated = self._structural_rj_evaluate_response_columns(
                data,
                selected,
                row_start=int(valid_prefix),
            )
            response_array[int(valid_prefix) :, selected] = evaluated

    def _structural_rj_response_dictionary(
        self,
        data: MeasurementData,
        patch_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return a prefix-cached response matrix, eagerly filling requested columns."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        response_dictionary = self._structural_rj_prepare_response_cache(data)
        requested = (
            np.arange(patches.patch_count, dtype=np.int64)
            if patch_indices is None
            else np.asarray(patch_indices, dtype=np.int64).reshape(-1)
        )
        self._structural_rj_ensure_response_columns(
            data,
            response_dictionary,
            requested,
        )
        return response_dictionary

    def _structural_rj_group_arrays(
        self,
        particle_indices: NDArray[np.int64],
        cardinality: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """Return canonical patch, strength, and background arrays for one K."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        source_count = max(0, int(cardinality))
        patch_rows: list[NDArray[np.int64]] = []
        strength_rows: list[NDArray[np.float64]] = []
        backgrounds = np.empty(indices.size, dtype=float)
        for row, particle_index in enumerate(indices.tolist()):
            state = self.continuous_particles[int(particle_index)].state
            if int(state.num_sources) != source_count:
                raise ValueError("rj_mh particle group mixes cardinalities.")
            patch_rows.append(self._canonicalize_structural_rj_state(state))
            strength_rows.append(
                np.asarray(state.strengths, dtype=float).reshape(source_count)
            )
            backgrounds[row] = float(state.background)
        if source_count == 0:
            patch_sets = np.zeros((indices.size, 0), dtype=np.int64)
            strengths = np.zeros((indices.size, 0), dtype=float)
        else:
            patch_sets = np.vstack(patch_rows).astype(np.int64, copy=False)
            strengths = np.vstack(strength_rows).astype(float, copy=False)
        if strengths.size and not np.all(self._strength_prior.in_support(strengths)):
            raise ValueError(
                "rj_mh particle strength lies outside the configured prior."
            )
        return patch_sets, strengths, backgrounds

    @staticmethod
    def _structural_rj_lambda_from_arrays(
        response_dictionary: NDArray[np.float64],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
        backgrounds: NDArray[np.float64],
        live_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return expected counts for an equal-cardinality state batch."""
        responses = np.asarray(response_dictionary, dtype=float)
        patches = np.asarray(patch_sets, dtype=np.int64)
        strength_array = np.asarray(strengths, dtype=float)
        background_array = np.asarray(backgrounds, dtype=float).reshape(-1)
        if patches.ndim != 2 or strength_array.shape != patches.shape:
            raise ValueError("rj_mh patch and strength arrays must share P x K.")
        if background_array.size != patches.shape[0]:
            raise ValueError("rj_mh background array must contain P entries.")
        background_counts = (
            np.asarray(live_times, dtype=float)[:, None] * background_array[None, :]
        )
        if patches.shape[1] == 0:
            return background_counts
        selected = responses[:, patches]
        source_counts = np.einsum(
            "mpk,pk->mp",
            selected,
            strength_array,
            optimize=True,
        )
        return background_counts + source_counts

    def _structural_rj_group_log_likelihood(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
        backgrounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one equal-cardinality group with the shared PF likelihood."""
        self._structural_rj_ensure_response_columns(
            data,
            response_dictionary,
            patch_sets,
        )
        lambda_total = self._structural_rj_lambda_from_arrays(
            response_dictionary,
            patch_sets,
            strengths,
            backgrounds,
            data.live_times,
        )
        return self._structural_count_log_likelihood_matrix_np(data, lambda_total)

    def _sample_structural_rj_unused_indices(
        self,
        occupied_sets: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Sample area-weighted unused patches for a same-cardinality batch."""
        surface_prior = self._structural_rj_surface_prior
        if surface_prior is None:
            raise RuntimeError("rj_mh surface prior is unavailable.")
        occupied = np.asarray(occupied_sets, dtype=np.int64)
        if occupied.ndim != 2:
            raise ValueError("occupied_sets must have shape P x K.")
        if occupied.shape[1] >= surface_prior.dictionary_size:
            raise ValueError("rj_mh birth has no unused surface patch.")
        sample_count = int(occupied.shape[0])
        sampled = self._random_generator.choice(
            surface_prior.dictionary_size,
            size=sample_count,
            p=surface_prior.area_masses,
        ).astype(np.int64, copy=False)
        if occupied.shape[1] == 0 or sample_count == 0:
            return sampled
        rejected = np.any(occupied == sampled[:, None], axis=1)
        while np.any(rejected):
            sampled[rejected] = self._random_generator.choice(
                surface_prior.dictionary_size,
                size=int(np.count_nonzero(rejected)),
                p=surface_prior.area_masses,
            )
            rejected = np.any(occupied == sampled[:, None], axis=1)
        return sampled

    @staticmethod
    def _structural_rj_insert_values(
        current_sets: NDArray[np.int64],
        current_values: NDArray[np.float64],
        new_indices: NDArray[np.int64],
        new_values: NDArray[np.float64],
        *,
        dictionary_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Insert one indexed scalar per row while preserving canonical order."""
        sets = np.asarray(current_sets, dtype=np.int64)
        values = np.asarray(current_values, dtype=float)
        indices = np.asarray(new_indices, dtype=np.int64).reshape(-1)
        additions = np.asarray(new_values, dtype=float).reshape(-1)
        if values.shape != sets.shape:
            raise ValueError("current_values must match current_sets.")
        if indices.size != sets.shape[0] or additions.size != sets.shape[0]:
            raise ValueError("new_indices and new_values must contain P entries.")
        proposed_sets = add_surface_indices(
            sets,
            indices,
            dictionary_size=dictionary_size,
        )
        combined_indices = np.concatenate([sets, indices[:, None]], axis=1)
        combined_values = np.concatenate([values, additions[:, None]], axis=1)
        order = np.argsort(combined_indices, axis=1, kind="stable")
        proposed_values = np.take_along_axis(combined_values, order, axis=1)
        return proposed_sets, np.asarray(proposed_values, dtype=float)

    @staticmethod
    def _structural_rj_remove_values(
        current_sets: NDArray[np.int64],
        current_values: NDArray[np.float64],
        columns: NDArray[np.int64],
        *,
        dictionary_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Remove one indexed scalar per row from canonical state arrays."""
        sets = np.asarray(current_sets, dtype=np.int64)
        values = np.asarray(current_values, dtype=float)
        removed_columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if values.shape != sets.shape:
            raise ValueError("current_values must match current_sets.")
        if removed_columns.size != sets.shape[0]:
            raise ValueError("columns must contain one entry per row.")
        proposed_sets = remove_surface_columns(
            sets,
            removed_columns,
            dictionary_size=dictionary_size,
        )
        keep = np.arange(sets.shape[1])[None, :] != removed_columns[:, None]
        proposed_values = values[keep].reshape(sets.shape[0], sets.shape[1] - 1)
        return proposed_sets, np.asarray(proposed_values, dtype=float)

    def _commit_structural_rj_states(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        patch_sets: NDArray[np.int64],
        strengths: NDArray[np.float64],
    ) -> int:
        """Commit accepted canonical state arrays after batched RJ/MH scoring."""
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
        sets = np.asarray(patch_sets, dtype=np.int64)
        strength_array = np.asarray(strengths, dtype=float)
        if (
            acceptance.size != indices.size
            or sets.shape[0] != indices.size
            or strength_array.shape != sets.shape
        ):
            raise ValueError("rj_mh commit arrays must share one particle axis.")
        accepted_rows = np.flatnonzero(acceptance)
        cardinality = int(sets.shape[1])
        for row in accepted_rows.tolist():
            state = self.continuous_particles[int(indices[row])].state
            state.num_sources = cardinality
            state.positions = np.asarray(
                patches.centers_xyz[sets[row]],
                dtype=float,
            ).reshape(cardinality, 3)
            state.strengths = strength_array[row].copy()
            # Background is not proposed by this kernel and must stay fixed.
        return int(accepted_rows.size)

    def _apply_structural_rj_birth_death(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> tuple[int, int]:
        """Apply one exact batched birth/death RJ-MH attempt per selected particle."""
        surface_prior = self._structural_rj_surface_prior
        cardinality_prior = self._structural_rj_cardinality_prior
        move_probabilities = self._structural_rj_move_probabilities
        if (
            surface_prior is None
            or cardinality_prior is None
            or move_probabilities is None
        ):
            raise RuntimeError("rj_mh priors or move probabilities are unavailable.")
        particle_count = len(self.continuous_particles)
        attempt = self._random_generator.random(particle_count) < float(
            self.config.structural_rj_move_probability
        )
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        accepted_births = 0
        accepted_deaths = 0
        attempted_births = 0
        attempted_deaths = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            group_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            if group_indices.size == 0:
                continue
            birth_probability, _ = move_probabilities.probabilities(int(cardinality))
            birth_move = self._random_generator.random(group_indices.size) < float(
                birth_probability
            )
            for is_birth in (True, False):
                selected_rows = np.flatnonzero(birth_move == is_birth)
                if selected_rows.size == 0:
                    continue
                selected_indices = group_indices[selected_rows]
                if is_birth:
                    attempted_births += int(selected_indices.size)
                else:
                    attempted_deaths += int(selected_indices.size)
                patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                    selected_indices,
                    int(cardinality),
                )
                base_ll = self._structural_rj_group_log_likelihood(
                    data,
                    response_dictionary,
                    patch_sets,
                    strengths,
                    backgrounds,
                )
                if is_birth:
                    new_patch_indices = self._sample_structural_rj_unused_indices(
                        patch_sets
                    )
                    new_strengths = np.asarray(
                        self._strength_prior.sample(
                            selected_indices.size,
                            rng=self._random_generator,
                        ),
                        dtype=float,
                    )
                    proposed_sets, proposed_strengths = (
                        self._structural_rj_insert_values(
                            patch_sets,
                            strengths,
                            new_patch_indices,
                            new_strengths,
                            dictionary_size=surface_prior.dictionary_size,
                        )
                    )
                    proposed_ll = self._structural_rj_group_log_likelihood(
                        data,
                        response_dictionary,
                        proposed_sets,
                        proposed_strengths,
                        backgrounds,
                    )
                    position_log_proposal = conditional_birth_surface_log_probability(
                        surface_prior,
                        patch_sets,
                        new_patch_indices,
                    )
                    strength_log_density = np.asarray(
                        self._strength_prior.log_prob(new_strengths),
                        dtype=float,
                    )
                    log_ratio = birth_log_acceptance_ratio(
                        current_surface_sets=patch_sets,
                        birth_surface_indices=new_patch_indices,
                        log_likelihood_ratio=proposed_ll - base_ll,
                        surface_prior=surface_prior,
                        cardinality_prior=cardinality_prior,
                        move_probabilities=move_probabilities,
                        log_strength_prior_density=strength_log_density,
                        log_forward_position_proposal=position_log_proposal,
                        log_forward_strength_proposal=strength_log_density,
                        log_reverse_death_index_probability=(
                            uniform_death_index_log_probability(int(cardinality) + 1)
                        ),
                    )
                    accepted = np.log(
                        self._random_generator.random(selected_indices.size)
                    ) < np.minimum(log_ratio, 0.0)
                    accepted_births += self._commit_structural_rj_states(
                        selected_indices,
                        accepted,
                        proposed_sets,
                        proposed_strengths,
                    )
                    for row in np.flatnonzero(accepted).tolist():
                        state = self.continuous_particles[
                            int(selected_indices[row])
                        ].state
                        source_column = int(
                            np.flatnonzero(
                                proposed_sets[row] == new_patch_indices[row]
                            )[0]
                        )
                        self._record_source_event(
                            "source_birth_accepted",
                            state,
                            source_column,
                            reason="rj_mh_birth",
                            extra={
                                "delta_ll": float(proposed_ll[row] - base_ll[row]),
                                "log_acceptance_ratio": float(log_ratio[row]),
                                "surface_patch_index": int(new_patch_indices[row]),
                            },
                        )
                    continue

                death_columns = self._random_generator.integers(
                    0,
                    int(cardinality),
                    size=selected_indices.size,
                    dtype=np.int64,
                )
                row_indices = np.arange(selected_indices.size, dtype=np.int64)
                removed_patch_indices = patch_sets[row_indices, death_columns]
                removed_strengths = strengths[row_indices, death_columns]
                proposed_sets, proposed_strengths = self._structural_rj_remove_values(
                    patch_sets,
                    strengths,
                    death_columns,
                    dictionary_size=surface_prior.dictionary_size,
                )
                proposed_ll = self._structural_rj_group_log_likelihood(
                    data,
                    response_dictionary,
                    proposed_sets,
                    proposed_strengths,
                    backgrounds,
                )
                removed_strength_log_density = np.asarray(
                    self._strength_prior.log_prob(removed_strengths),
                    dtype=float,
                )
                reverse_position_log_proposal = (
                    conditional_birth_surface_log_probability(
                        surface_prior,
                        proposed_sets,
                        removed_patch_indices,
                    )
                )
                log_ratio = death_log_acceptance_ratio(
                    current_surface_sets=patch_sets,
                    death_columns=death_columns,
                    log_likelihood_ratio=proposed_ll - base_ll,
                    surface_prior=surface_prior,
                    cardinality_prior=cardinality_prior,
                    move_probabilities=move_probabilities,
                    log_removed_strength_prior_density=(removed_strength_log_density),
                    log_forward_death_index_probability=(
                        uniform_death_index_log_probability(int(cardinality))
                    ),
                    log_reverse_position_proposal=(reverse_position_log_proposal),
                    log_reverse_strength_proposal=(removed_strength_log_density),
                )
                accepted = np.log(
                    self._random_generator.random(selected_indices.size)
                ) < np.minimum(log_ratio, 0.0)
                for row in np.flatnonzero(accepted).tolist():
                    state = self.continuous_particles[int(selected_indices[row])].state
                    self._record_source_event(
                        "source_removed",
                        state,
                        int(death_columns[row]),
                        reason="rj_mh_death",
                        extra={
                            "delta_ll": float(proposed_ll[row] - base_ll[row]),
                            "log_acceptance_ratio": float(log_ratio[row]),
                            "surface_patch_index": int(removed_patch_indices[row]),
                        },
                    )
                accepted_deaths += self._commit_structural_rj_states(
                    selected_indices,
                    accepted,
                    proposed_sets,
                    proposed_strengths,
                )
        self._structural_rj_move_counts.update(
            {
                "birth_attempted": int(attempted_births),
                "birth_accepted": int(accepted_births),
                "death_attempted": int(attempted_deaths),
                "death_accepted": int(accepted_deaths),
            }
        )
        return accepted_births, accepted_deaths

    def _apply_structural_rj_position_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Relocate sources with an exact area-prior-cancelling MH proposal."""
        surface_prior = self._structural_rj_surface_prior
        if surface_prior is None:
            raise RuntimeError("rj_mh surface prior is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            relocated_strengths = strengths[rows, source_columns]
            old_patch_indices = patch_sets[rows, source_columns]
            reduced_sets, reduced_strengths = self._structural_rj_remove_values(
                patch_sets,
                strengths,
                source_columns,
                dictionary_size=surface_prior.dictionary_size,
            )
            new_patch_indices = self._sample_structural_rj_unused_indices(reduced_sets)
            proposed_sets, proposed_strengths = self._structural_rj_insert_values(
                reduced_sets,
                reduced_strengths,
                new_patch_indices,
                relocated_strengths,
                dictionary_size=surface_prior.dictionary_size,
            )
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                proposed_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = proposed_ll - base_ll
            accepted = (
                np.log(self._random_generator.random(particle_indices.size))
                < np.minimum(log_ratio, 0.0)
            ) & (new_patch_indices != old_patch_indices)
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                proposed_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "global_position_attempted": int(attempted_count),
                "global_position_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_structural_rj_local_position_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Relocate sources to adjacent free patches with exact local MH."""
        surface_prior = self._structural_rj_surface_prior
        adjacency = self._structural_rj_surface_adjacency
        if surface_prior is None or adjacency is None:
            raise RuntimeError("rj_mh surface prior or adjacency is unavailable.")
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_local_position_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        movable_count = 0
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            rows = np.arange(particle_indices.size, dtype=np.int64)
            old_patch_indices = patch_sets[rows, source_columns]
            relocated_strengths = strengths[rows, source_columns]
            reduced_sets, reduced_strengths = self._structural_rj_remove_values(
                patch_sets,
                strengths,
                source_columns,
                dictionary_size=surface_prior.dictionary_size,
            )
            (
                new_patch_indices,
                forward_degrees,
                movable,
            ) = adjacency.sample_unoccupied_neighbors(
                patch_sets,
                source_columns,
                rng=self._random_generator,
            )
            movable_count += int(np.count_nonzero(movable))
            proposed_sets, proposed_strengths = self._structural_rj_insert_values(
                reduced_sets,
                reduced_strengths,
                new_patch_indices,
                relocated_strengths,
                dictionary_size=surface_prior.dictionary_size,
            )
            reverse_degrees = adjacency.available_neighbor_degrees(
                new_patch_indices,
                reduced_sets,
            )
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                proposed_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = local_position_log_acceptance_ratio(
                old_surface_indices=old_patch_indices,
                new_surface_indices=new_patch_indices,
                forward_available_degrees=forward_degrees,
                reverse_available_degrees=reverse_degrees,
                log_likelihood_ratio=proposed_ll - base_ll,
                surface_prior=surface_prior,
            )
            accepted = (
                movable
                & (new_patch_indices != old_patch_indices)
                & (
                    np.log(self._random_generator.random(particle_indices.size))
                    < np.minimum(log_ratio, 0.0)
                )
            )
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                proposed_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "local_position_attempted": int(attempted_count),
                "local_position_movable": int(movable_count),
                "local_position_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_structural_rj_strength_moves(
        self,
        data: MeasurementData,
        response_dictionary: NDArray[np.float64],
    ) -> int:
        """Update strengths with an exact prior-independence MH proposal."""
        particle_count = len(self.continuous_particles)
        cardinalities = np.asarray(
            [particle.state.num_sources for particle in self.continuous_particles],
            dtype=np.int64,
        )
        attempt = (
            self._random_generator.random(particle_count)
            < float(self.config.structural_rj_strength_move_probability)
        ) & (cardinalities > 0)
        attempted_count = int(np.count_nonzero(attempt))
        accepted_count = 0
        for cardinality in np.unique(cardinalities[attempt]).tolist():
            particle_indices = np.flatnonzero(
                attempt & (cardinalities == int(cardinality))
            ).astype(np.int64, copy=False)
            patch_sets, strengths, backgrounds = self._structural_rj_group_arrays(
                particle_indices,
                int(cardinality),
            )
            base_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                strengths,
                backgrounds,
            )
            source_columns = self._random_generator.integers(
                0,
                int(cardinality),
                size=particle_indices.size,
                dtype=np.int64,
            )
            proposed_strengths = strengths.copy()
            proposed_values = np.asarray(
                self._strength_prior.sample(
                    particle_indices.size,
                    rng=self._random_generator,
                ),
                dtype=float,
            )
            proposed_strengths[
                np.arange(particle_indices.size),
                source_columns,
            ] = proposed_values
            proposed_ll = self._structural_rj_group_log_likelihood(
                data,
                response_dictionary,
                patch_sets,
                proposed_strengths,
                backgrounds,
            )
            log_ratio = proposed_ll - base_ll
            accepted = np.log(
                self._random_generator.random(particle_indices.size)
            ) < np.minimum(log_ratio, 0.0)
            accepted_count += self._commit_structural_rj_states(
                particle_indices,
                accepted,
                patch_sets,
                proposed_strengths,
            )
        self._structural_rj_move_counts.update(
            {
                "strength_attempted": int(attempted_count),
                "strength_accepted": int(accepted_count),
            }
        )
        return accepted_count

    def _apply_exact_structural_rj_moves(
        self,
        evidence_data: MeasurementData,
    ) -> None:
        """Apply target-preserving RJ/MH rejuvenation without changing PF weights."""
        structural_start = time.perf_counter()
        original_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        self._structural_rj_response_evaluation_batches = 0
        self._structural_rj_response_evaluated_cells = 0
        self._structural_rj_move_counts = {
            "birth_attempted": 0,
            "birth_accepted": 0,
            "death_attempted": 0,
            "death_accepted": 0,
            "global_position_attempted": 0,
            "global_position_accepted": 0,
            "local_position_attempted": 0,
            "local_position_movable": 0,
            "local_position_accepted": 0,
            "strength_attempted": 0,
            "strength_accepted": 0,
        }
        patches = self._structural_rj_surface_patches
        if patches is None:
            raise RuntimeError("rj_mh surface patches are unavailable.")
        self._structural_rj_response_touched_mask = np.zeros(
            patches.patch_count,
            dtype=bool,
        )
        response_start = time.perf_counter()
        response_dictionary = self._structural_rj_response_dictionary(
            evidence_data,
            patch_indices=np.zeros(0, dtype=np.int64),
        )
        response_elapsed = time.perf_counter() - response_start
        birth_count = 0
        death_count = 0
        move_start = time.perf_counter()
        birth_count, death_count = self._apply_structural_rj_birth_death(
            evidence_data,
            response_dictionary,
        )
        birth_death_elapsed = time.perf_counter() - move_start
        position_start = time.perf_counter()
        position_count = self._apply_structural_rj_position_moves(
            evidence_data,
            response_dictionary,
        )
        position_elapsed = time.perf_counter() - position_start
        local_position_start = time.perf_counter()
        local_position_count = self._apply_structural_rj_local_position_moves(
            evidence_data,
            response_dictionary,
        )
        local_position_elapsed = time.perf_counter() - local_position_start
        strength_start = time.perf_counter()
        strength_count = self._apply_structural_rj_strength_moves(
            evidence_data,
            response_dictionary,
        )
        strength_elapsed = time.perf_counter() - strength_start
        current_log_weights = np.asarray(
            [particle.log_weight for particle in self.continuous_particles],
            dtype=float,
        )
        outer_weight_array_equal = bool(
            np.array_equal(original_log_weights, current_log_weights)
        )
        with np.errstate(invalid="ignore"):
            outer_weight_differences = np.where(
                original_log_weights == current_log_weights,
                0.0,
                np.abs(original_log_weights - current_log_weights),
            )
        outer_weight_differences = np.where(
            np.isfinite(outer_weight_differences),
            outer_weight_differences,
            float("inf"),
        )
        outer_weight_max_abs_diff = (
            float(np.max(outer_weight_differences))
            if outer_weight_differences.size
            else 0.0
        )
        self.last_birth_count += int(birth_count)
        self.last_death_count += int(death_count)
        self.last_structural_timing_s = {
            "total": float(time.perf_counter() - structural_start),
            "response_dictionary": float(response_elapsed),
            "rj_birth_death": float(birth_death_elapsed),
            "rj_position": float(position_elapsed),
            "rj_global_position": float(position_elapsed),
            "rj_local_position": float(local_position_elapsed),
            "rj_strength": float(strength_elapsed),
            "rj_birth_attempted": float(
                self._structural_rj_move_counts["birth_attempted"]
            ),
            "rj_birth_accepted": float(birth_count),
            "rj_death_attempted": float(
                self._structural_rj_move_counts["death_attempted"]
            ),
            "rj_death_accepted": float(death_count),
            "rj_global_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_global_position_accepted": float(position_count),
            "rj_position_attempted": float(
                self._structural_rj_move_counts["global_position_attempted"]
            ),
            "rj_position_accepted": float(position_count),
            "rj_local_position_attempted": float(
                self._structural_rj_move_counts["local_position_attempted"]
            ),
            "rj_local_position_movable": float(
                self._structural_rj_move_counts["local_position_movable"]
            ),
            "rj_local_position_accepted": float(local_position_count),
            "rj_strength_attempted": float(
                self._structural_rj_move_counts["strength_attempted"]
            ),
            "rj_strength_accepted": float(strength_count),
            "rj_response_evaluation_batches": float(
                self._structural_rj_response_evaluation_batches
            ),
            "rj_response_evaluated_cells": float(
                self._structural_rj_response_evaluated_cells
            ),
            "rj_response_touched_columns": float(
                np.count_nonzero(self._structural_rj_response_touched_mask)
            ),
            "outer_log_weight_max_abs_diff": float(outer_weight_max_abs_diff),
            "outer_log_weight_array_equal": float(outer_weight_array_equal),
            "weights_preserved": float(outer_weight_array_equal),
        }
        if not outer_weight_array_equal:
            raise RuntimeError("rj_mh rejuvenation must not alter PF weights.")

    def apply_structural_moves(
        self,
        evidence_data: MeasurementData | None,
    ) -> None:
        """Apply exact finite-surface RJ-MH rejuvenation when enabled."""
        if not self.continuous_particles:
            return
        if not bool(self.config.birth_enable):
            self.last_structural_timing_s = {
                "total": 0.0,
                "structural_moves_gated": 1.0,
                "weights_preserved": 1.0,
            }
            return
        if evidence_data is None or evidence_data.z_k.size == 0:
            self.last_structural_timing_s = {
                "total": 0.0,
                "rj_mh_no_evidence": 1.0,
                "weights_preserved": 1.0,
            }
            return
        self._apply_exact_structural_rj_moves(evidence_data)

    def _background_level(self) -> float:
        """Resolve per-isotope background level."""
        level = self.config.background_level
        if isinstance(level, dict):
            return float(level.get(self.isotope, 0.0))
        return float(level)

    def estimate(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        if not self.continuous_particles:
            return np.zeros((0, 3)), np.zeros(0)
        point_estimate = posterior_point_estimate_from_states(
            [particle.state for particle in self.continuous_particles],
            np.asarray(self.continuous_weights, dtype=float),
            max_cardinality=self.config.max_sources,
            position_projector=self._project_positions_to_source_prior,
        )
        positions = np.asarray(
            [mode.position_mean_xyz for mode in point_estimate.modes],
            dtype=float,
        ).reshape(-1, 3)
        strengths = np.asarray(
            [mode.strength_mean_cps_1m for mode in point_estimate.modes],
            dtype=float,
        )
        return positions, strengths
