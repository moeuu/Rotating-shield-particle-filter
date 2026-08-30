"""High-level estimator coordinating parallel PFs and shield rotation (Chapter 3)."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from spectrum.air_attenuation import (
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
)
from pf.estimator_types import (
    JointPlanningParticles as JointPlanningParticles,
    JointStationObservation as JointStationObservation,
    MeasurementRecord as MeasurementRecord,
)
from pf.full_spectrum import (
    FullSpectrumGenerativeModel,
    validate_full_spectrum_model,
)
from pf.particle_filter import (
    IsotopeParticle,
    IsotopeParticleFilter,
    JointRowIdentity,
    StructuralGeometryBatch,
)
from pf.posterior import (
    PFPointEstimate,
    validated_probability_distribution,
)
from pf.provenance import strict_sha256_json
from pf.randomness import (
    named_random_generator,
    normalize_pf_random_seed,
    pf_rng_provenance,
)
from pf.state import IsotopeState
from pf.estimator_config import (
    RotatingShieldPFConfig as RotatingShieldPFConfig,
    _strict_config_boolean,
    _strict_config_number,
    _strict_nonnegative_integer,
)
from pf.estimator_likelihood import (
    JOINT_HISTORY_STATION_ACTION_BATCH_SIZE as JOINT_HISTORY_STATION_ACTION_BATCH_SIZE,
    JointLikelihoodMixin,
)
from pf.joint_transport_cache import JointTransportCache
from pf.estimator_rejuvenation import JointRejuvenationMixin
from pf.estimator_reporting import EstimatorReportingMixin
from pf.estimator_sampling import (
    _stratified_categorical_draws as _stratified_categorical_draws,
    _stratified_joint_cardinality_draws as _stratified_joint_cardinality_draws,
)
from pf.estimator_structural import EstimatorStructuralProposalMixin
from pf.estimator_structural import (
    JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE,
    JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER,
    JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
    JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES,
    JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES,
)
from pf.estimator_surface import (
    SurfaceAtlasQuadrature as SurfaceAtlasQuadrature,
    build_complete_surface_atlas_quadrature as build_complete_surface_atlas_quadrature,
)
from pf.transport_response import expected_counts_per_source

__all__ = (
    "JOINT_HISTORY_STATION_ACTION_BATCH_SIZE",
    "JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE",
    "JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER",
    "JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE",
    "JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES",
    "JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES",
    "JointPlanningParticles",
    "JointStationObservation",
    "MeasurementRecord",
    "RotatingShieldPFConfig",
    "RotatingShieldPFEstimator",
    "SurfaceAtlasQuadrature",
    "build_complete_surface_atlas_quadrature",
)

class RotatingShieldPFEstimator(
    JointLikelihoodMixin,
    EstimatorStructuralProposalMixin,
    JointRejuvenationMixin,
    EstimatorReportingMixin,
):
    """
    Online exact-RJ particle filter using rotating shield spectra.

    Isotope state blocks share one outer particle weight and one resampling
    ancestry. Each station is assimilated once through the immutable joint
    full-spectrum generative likelihood.
    """

    def __init__(
        self,
        isotopes: Sequence[str],
        surface_diagnostic_points: NDArray[np.float64],
        shield_normals: NDArray[np.float64] | None,
        mu_by_isotope: dict[str, object] | None,
        pf_config: RotatingShieldPFConfig | None = None,
        shield_params: ShieldParams | None = None,
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
        strict_catalog_line_contract: bool = False,
        dry_air_total_attenuation_contract_id: str | None = None,
        dry_air_total_attenuation_contract_sha256: str | None = None,
        full_spectrum_generative_model: object | None = None,
        random_seed: int = 0,
    ) -> None:
        """Initialize per-isotope filters and shared measurement-model state."""
        configured_isotopes = tuple(isotopes)
        if (
            not configured_isotopes
            or any(
                not isinstance(isotope, str) or not isotope.strip()
                for isotope in configured_isotopes
            )
            or len(set(configured_isotopes)) != len(configured_isotopes)
        ):
            raise ValueError("Joint PF isotopes must be unique nonempty strings.")
        self.isotopes = list(configured_isotopes)
        self.random_seed = normalize_pf_random_seed(random_seed)
        self.rng_provenance = pf_rng_provenance(
            self.random_seed,
            self.isotopes,
        )
        if pf_config is not None and not isinstance(
            pf_config,
            RotatingShieldPFConfig,
        ):
            raise TypeError(
                "pf_config must be a RotatingShieldPFConfig instance."
            )
        self.pf_config = (
            RotatingShieldPFConfig() if pf_config is None else pf_config
        )
        self.shield_params = shield_params or ShieldParams()
        if not isinstance(self.shield_params, ShieldParams):
            raise TypeError("shield_params must be a ShieldParams instance.")
        self.obstacle_grid = obstacle_grid
        self.obstacle_height_m = _strict_config_number(
            obstacle_height_m,
            name="obstacle_height_m",
        )
        if self.obstacle_height_m < 0.0:
            raise ValueError("obstacle_height_m must be nonnegative.")
        self.obstacle_mu_by_isotope = obstacle_mu_by_isotope
        self.obstacle_buildup_coeff = _strict_config_number(
            obstacle_buildup_coeff,
            name="obstacle_buildup_coeff",
        )
        if self.obstacle_buildup_coeff < 0.0:
            raise ValueError("obstacle_buildup_coeff must be nonnegative.")
        self.detector_radius_m = _strict_config_number(
            detector_radius_m,
            name="detector_radius_m",
        )
        if self.detector_radius_m < 0.0:
            raise ValueError("detector_radius_m must be nonnegative.")
        if detector_aperture_radius_m is None:
            detector_aperture_radius_m = self.detector_radius_m
        self.detector_aperture_radius_m = _strict_config_number(
            detector_aperture_radius_m,
            name="detector_aperture_radius_m",
        )
        if self.detector_aperture_radius_m < 0.0:
            raise ValueError("detector_aperture_radius_m must be nonnegative.")
        self.detector_aperture_samples = _strict_nonnegative_integer(
            detector_aperture_samples,
            name="detector_aperture_samples",
        )
        if self.detector_aperture_samples == 0:
            raise ValueError("detector_aperture_samples must be positive.")
        if not isinstance(detector_aperture_sampling, str):
            raise TypeError("detector_aperture_sampling must be a string.")
        self.detector_aperture_sampling = detector_aperture_sampling
        self.source_extent_radius_m = _strict_config_number(
            source_extent_radius_m,
            name="source_extent_radius_m",
        )
        if self.source_extent_radius_m < 0.0:
            raise ValueError("source_extent_radius_m must be nonnegative.")
        self.source_extent_samples = _strict_nonnegative_integer(
            source_extent_samples,
            name="source_extent_samples",
        )
        if self.source_extent_samples == 0:
            raise ValueError("source_extent_samples must be positive.")
        if (self.source_extent_radius_m == 0.0) != (self.source_extent_samples == 1):
            raise ValueError(
                "Source extent requires radius=0 with one sample, or a "
                "positive radius with at least two samples."
            )
        self.line_mu_by_isotope = line_mu_by_isotope
        if not isinstance(strict_catalog_line_contract, bool):
            raise TypeError("strict_catalog_line_contract must be a boolean.")
        self.strict_catalog_line_contract = strict_catalog_line_contract
        self.dry_air_total_attenuation_contract_id = (
            dry_air_total_attenuation_contract_id
        )
        self.dry_air_total_attenuation_contract_sha256 = (
            dry_air_total_attenuation_contract_sha256
        )
        air_contract = (
            self.dry_air_total_attenuation_contract_id,
            self.dry_air_total_attenuation_contract_sha256,
        )
        authenticated_air_contract = (
            NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
            NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
        )
        if self.strict_catalog_line_contract:
            if not isinstance(self.line_mu_by_isotope, dict) or not (
                self.line_mu_by_isotope
            ):
                raise ValueError(
                    "Strict PF transport requires a nonempty exact catalog "
                    "line table."
                )
            if air_contract != authenticated_air_contract:
                raise ValueError(
                    "Strict PF transport requires the exact authenticated "
                    "XCOM dry-air contract."
                )
        elif air_contract != (None, None):
            raise ValueError(
                "A dry-air contract cannot be enabled while exact catalog "
                "line transport is disabled."
            )
        self.full_spectrum_generative_model = validate_full_spectrum_model(
            full_spectrum_generative_model
        )
        self._authenticated_full_spectrum_model_object = (
            self.full_spectrum_generative_model
        )
        self._authenticated_full_spectrum_contract_hash_sha256 = str(
            self.full_spectrum_generative_model.contract_hash_sha256
        )
        if not np.isclose(
            self.detector_aperture_radius_m,
            float(self.full_spectrum_generative_model.detector_target_radius_m),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "PF detector aperture radius does not match the authenticated "
                "detector Green boundary."
            )
        if self.source_extent_radius_m != 0.0 or self.source_extent_samples != 1:
            raise ValueError(
                "Full-spectrum detector-impact conditioning requires the "
                "authenticated point-source contract."
            )
        self.additive_scatter_response = getattr(
            self.full_spectrum_generative_model,
            "additive_scatter_response",
            None,
        )
        # Measurement poses are appended incrementally.
        self.poses: list[NDArray[np.float64]] = []
        if shield_normals is None:
            from measurement.shielding import generate_octant_orientations

            self.normals = generate_octant_orientations()
        else:
            self.normals = shield_normals
        self.mu_by_isotope = self._resolve_mu_by_isotope(mu_by_isotope)
        self.kernel_cache: MeasurementGeometry | None = None
        self.filters: dict[str, IsotopeParticleFilter] = {}
        self._joint_particles_initialized = False
        self._joint_row_identity_root_sha256: str | None = None
        self._joint_row_generation: int | None = None
        diagnostic_points = np.asarray(
            surface_diagnostic_points,
            dtype=np.float64,
        )
        if diagnostic_points.ndim != 2 or diagnostic_points.shape[1] != 3:
            raise ValueError("surface_diagnostic_points must be shaped (N, 3).")
        if not np.all(np.isfinite(diagnostic_points)):
            raise ValueError(
                "surface_diagnostic_points must contain only finite values."
            )
        self.surface_diagnostic_points = np.ascontiguousarray(diagnostic_points)
        self._posterior_point_estimate_cache: dict[str, PFPointEstimate] | None = None
        self._posterior_point_estimate_cache_fingerprint: str | None = None
        self.measurements: list[MeasurementRecord] = []
        self._joint_station_history: list[JointStationObservation] = []
        self._active_joint_station_history: (
            tuple[JointStationObservation, ...] | None
        ) = None
        self._active_joint_structural_geometry: StructuralGeometryBatch | None = None
        self._joint_structural_transport_cache: JointTransportCache | None = None
        self._joint_persistent_structural_transport_cache: (
            JointTransportCache | None
        ) = None
        self.joint_transport_cache_preflight: dict[str, object] | None = None
        self.last_joint_persistent_cache_reuse_count = 0
        self.last_joint_persistent_cache_append_count = 0
        self.last_joint_persistent_cache_reindex_count = 0
        self.last_joint_station_transport_cache_reuse_count = 0
        self._joint_structural_unit_transport_cache: dict[
            str,
            dict[str, dict[str, Any]],
        ] = {}
        self._joint_device_unit_transport_cache: dict[
            tuple[str, str, str, str, str],
            tuple[object, object, object],
        ] = {}
        self.last_joint_device_unit_cache_hits = 0
        self.last_joint_device_unit_cache_misses = 0
        self._joint_cuda_accepted_unit_transport_cache: dict[
            tuple[str, str], dict[str, object]
        ] = {}
        self._joint_structural_unit_cache_access_generation = 0
        self.last_joint_structural_unit_cache_hits = 0
        self.last_joint_structural_unit_cache_misses = 0
        self.last_joint_staged_transport_commit_rows = 0
        self.last_joint_slot_overlay_likelihood_calls = 0
        self.last_joint_full_history_clone_count = 0
        self.last_joint_station_likelihood_cache_reuse_count = 0
        self.last_joint_station_likelihood_append_count = 0
        self.last_joint_station_likelihood_full_refresh_count = 0
        self.last_joint_strength_grid_source_slots_before = 0
        self.last_joint_strength_grid_source_slots_after = 0
        self._joint_strength_grid_batch_size_cache: dict[tuple[object, ...], int] = {}
        self._joint_torch_observation_context_cache: dict[
            tuple[object, ...], object
        ] = {}
        self._joint_torch_context_station_ids: tuple[int, ...] = ()
        self._joint_torch_history_layout_cache: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}
        self.last_joint_strength_grid_batch_diagnostics: dict[str, object] = {}
        self._joint_birth_proposal_station_score_cache: dict[
            tuple[str, str],
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_station_score_cache_order: list[tuple[str, str]] = []
        self._joint_birth_proposal_prefix_scores: dict[
            str,
            NDArray[np.float64],
        ] = {}
        self._joint_birth_proposal_prefix_station_count = 0
        self.last_joint_birth_proposal_cache_hits = 0
        self.last_joint_birth_proposal_cache_misses = 0
        self._joint_birth_proposal_reference_mean_vb: NDArray[np.float64] | None = None
        self._joint_external_surface_guidance_by_isotope: dict[
            str,
            NDArray[np.float64],
        ] | None = None
        self._joint_external_surface_guidance_mass = 0.0
        self.last_external_surface_guidance_diagnostics: dict[
            str,
            dict[str, float],
        ] = {}
        self.last_external_surface_guidance_evaluated_isotopes: set[str] = set()
        self._joint_random_generator = named_random_generator(
            self.random_seed,
            "joint_isotope_particle_filter",
        )
        self._joint_torch_generator: object | None = None
        self._joint_tpht_generator: object | None = None
        self.last_joint_resample_indices = np.zeros(0, dtype=np.int64)
        self.last_joint_temper_steps: list[dict[str, float]] = []
        self.last_joint_rejuvenation_diagnostics: list[dict[str, float]] = []
        self.last_joint_smc_wall_time_limit_exceeded = False
        self.last_joint_rejuvenation_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete = False
        self.last_joint_structural_mixing_incomplete_by_isotope = {
            str(isotope): False for isotope in self.isotopes
        }
        self._joint_guided_initialization_applied = False
        self.last_joint_guided_initialization_ess: float | None = None
        self.last_joint_cross_isotope_state_attempted_weight_mass = 0.0
        self.last_joint_cross_isotope_state_accepted_weight_mass = 0.0
        self.last_joint_cross_isotope_state_rejection_diagnostics: dict[
            str,
            object,
        ] = {}
        self.last_joint_strength_block_attempted_weight_mass = 0.0
        self.last_joint_strength_block_accepted_weight_mass = 0.0
        self.last_joint_device_mh_acceptance_calls = 0
        self.last_joint_device_mh_acceptance_rows = 0
        self.last_joint_tpht_diagnostics: dict[str, int] = {}
        self._joint_tpht_linear_scaling_streak = 0
        self._joint_initial_product_prior_state_sha256: str | None = None
        self.last_joint_station_unique_ancestor_count: int | None = None
        self.last_joint_cumulative_unique_ancestor_count: int | None = None
        self._joint_cumulative_lineage_ids: NDArray[np.int64] | None = None
        self._joint_lineage_recovery_active = False
        self._joint_lineage_recovery_epoch = 0
        self._joint_lineage_recovery_certified_mask_by_isotope: dict[
            str,
            NDArray[np.bool_],
        ] = {
            str(isotope): np.zeros(0, dtype=np.bool_)
            for isotope in self.isotopes
        }
        self.last_pair_sequence_update_wall_s = 0.0
        self.last_pair_sequence_stage_wall_s: dict[str, float] = {}
        self.last_structural_update_wall_s = 0.0
        self._surface_diagnostic_response_cache: dict[
            tuple[Any, ...],
            NDArray[np.float64],
        ] = {}
        self._surface_diagnostic_response_cache_order: list[tuple[Any, ...]] = []
        self._surface_diagnostic_response_prefix_cache: dict[
            tuple[Any, ...],
            dict[str, NDArray[np.float64]],
        ] = {}
        self._surface_diagnostic_response_prefix_cache_order: list[tuple[Any, ...]] = []

    def authenticated_full_spectrum_model(self) -> FullSpectrumGenerativeModel:
        """Return the construction-authenticated immutable model binding.

        Full schema and production evidence are checked once at the estimator
        boundary. Internal PF/planning calls then verify the exact object and
        contract identity without repeating the complete acceptance audit.
        """
        model = self.full_spectrum_generative_model
        if model is not self._authenticated_full_spectrum_model_object:
            raise RuntimeError(
                "The full-spectrum model was replaced after estimator "
                "authentication."
            )
        if (
            str(model.contract_hash_sha256)
            != self._authenticated_full_spectrum_contract_hash_sha256
        ):
            raise RuntimeError(
                "The full-spectrum model contract changed after estimator "
                "authentication."
            )
        return model

    def _named_planning_rng(
        self,
        operation: str,
        *components: object,
    ) -> np.random.Generator:
        """Return an order-independent planning stream from the logged PF seed."""
        return named_random_generator(
            self.random_seed,
            "estimator_planning",
            str(operation),
            len(self.measurements),
            *components,
        )

    def _refresh_posterior_summary(self) -> None:
        """Prepare the exact cached summary used by reporting and planning."""
        self.estimates()

    def _invalidate_posterior_summary_cache(self) -> None:
        """Discard report-only summaries after posterior state changes."""
        self._posterior_point_estimate_cache = None
        self._posterior_point_estimate_cache_fingerprint = None

    def _posterior_summary_state_fingerprint(self) -> str:
        """Hash aligned weights and continuous states in batched fixed slots."""
        digest = hashlib.blake2b(digest_size=16)
        for isotope in sorted(self.filters):
            filt = self.filters[isotope]
            (
                _positions,
                strengths,
                active_mask,
                chart_ids,
                surface_uv,
            ) = filt._packed_continuous_surface_state_arrays()
            for values in (
                np.asarray(filt.continuous_weights, dtype=np.float64),
                strengths,
                active_mask,
                chart_ids,
                surface_uv,
            ):
                array = np.ascontiguousarray(values)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
                digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _cached_posterior_point_estimate(
        self,
        *,
        validate_state: bool = True,
    ) -> dict[str, PFPointEstimate] | None:
        """Return a shallow copy of the immutable cached posterior summary."""
        if self._posterior_point_estimate_cache is None:
            return None
        if validate_state and (
            self._posterior_point_estimate_cache_fingerprint
            != self._posterior_summary_state_fingerprint()
        ):
            self._invalidate_posterior_summary_cache()
            return None
        return dict(self._posterior_point_estimate_cache)

    def _store_posterior_point_estimate(
        self,
        estimate: Mapping[str, PFPointEstimate],
    ) -> dict[str, PFPointEstimate]:
        """Store and return one exact posterior summary generation."""
        cached = {
            str(isotope): point_estimate for isotope, point_estimate in estimate.items()
        }
        self._posterior_point_estimate_cache = cached
        self._posterior_point_estimate_cache_fingerprint = (
            self._posterior_summary_state_fingerprint()
        )
        return dict(cached)

    @staticmethod
    def _project_posterior_point_estimates(
        point_estimates: Mapping[str, PFPointEstimate],
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Project immutable posterior summaries into visualization arrays."""
        projected: dict[
            str,
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        for isotope, point_estimate in point_estimates.items():
            projected[str(isotope)] = (
                np.asarray(
                    [mode.position_medoid_xyz for mode in point_estimate.modes],
                    dtype=float,
                ).reshape(-1, 3),
                np.asarray(
                    [
                        mode.strength_representative_cps_1m
                        for mode in point_estimate.modes
                    ],
                    dtype=float,
                ),
            )
        return projected

    def visualization_estimates(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the latest committed exact summary without recomputation.

        Frames collected before the first completed station intentionally show
        no point estimates. Particle clouds remain available, while the live
        rendering path never triggers the quadratic intrinsic-medoid report.
        """
        cached = self._cached_posterior_point_estimate(
            validate_state=False,
        )
        if cached is None:
            return {
                str(isotope): (
                    np.zeros((0, 3), dtype=float),
                    np.zeros(0, dtype=float),
                )
                for isotope in self.filters
            }
        return self._project_posterior_point_estimates(cached)

    def _surface_diagnostic_response_source_key(
        self,
        sources: NDArray[np.float64],
    ) -> tuple[str, int] | None:
        (
            "Return a stable cache key for the full shared surface-diagnostic "
            "atlas sample."
        )
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        candidate_arr = np.asarray(self.surface_diagnostic_points, dtype=float).reshape(
            -1,
            3,
        )
        if (
            source_arr.shape == candidate_arr.shape
            and source_arr.size > 0
            and np.shares_memory(source_arr, candidate_arr)
        ):
            return ("surface_diagnostic_points", int(source_arr.shape[0]))
        return None

    @staticmethod
    def _measurement_geometry_digest(data: StructuralGeometryBatch) -> bytes:
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

    def _store_surface_diagnostic_response_cache(
        self,
        cache_key: tuple[Any, ...],
        counts: NDArray[np.float64],
    ) -> None:
        (
            "Store an exact deterministic surface-diagnostic response with "
            "LRU eviction."
        )
        self._surface_diagnostic_response_cache[cache_key] = np.asarray(
            counts,
            dtype=float,
        ).copy()
        self._surface_diagnostic_response_cache_order.append(cache_key)
        max_entries = max(
            0,
            int(self.pf_config.surface_diagnostic_response_cache_max_entries),
        )
        while len(self._surface_diagnostic_response_cache_order) > max_entries:
            old_key = self._surface_diagnostic_response_cache_order.pop(0)
            if old_key not in self._surface_diagnostic_response_cache_order:
                self._surface_diagnostic_response_cache.pop(old_key, None)

    @staticmethod
    def _surface_diagnostic_response_prefix_matches(
        payload: Mapping[str, NDArray[np.float64]],
        data: StructuralGeometryBatch,
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
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _store_surface_diagnostic_response_prefix_cache(
        self,
        prefix_key: tuple[Any, ...],
        data: StructuralGeometryBatch,
        counts: NDArray[np.float64],
    ) -> None:
        """Store a full-history response prefix for incremental extension."""
        self._surface_diagnostic_response_prefix_cache[prefix_key] = {
            "detector_positions": np.asarray(
                data.detector_positions, dtype=float
            ).copy(),
            "live_times": np.asarray(data.live_times, dtype=float).copy(),
            "fe_indices": np.asarray(data.fe_indices, dtype=int).copy(),
            "pb_indices": np.asarray(data.pb_indices, dtype=int).copy(),
            "counts": np.asarray(counts, dtype=float).copy(),
        }
        self._surface_diagnostic_response_prefix_cache_order.append(prefix_key)
        max_entries = max(
            0,
            int(self.pf_config.surface_diagnostic_response_cache_max_entries),
        )
        while len(self._surface_diagnostic_response_prefix_cache_order) > max_entries:
            old_key = self._surface_diagnostic_response_prefix_cache_order.pop(0)
            if old_key not in self._surface_diagnostic_response_prefix_cache_order:
                self._surface_diagnostic_response_prefix_cache.pop(old_key, None)

    def _cached_expected_counts_for_kernel(
        self,
        *,
        kernel: ContinuousKernel,
        isotope: str,
        data: StructuralGeometryBatch,
        sources: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched expected counts from a particle-independent kernel."""
        source_arr = np.asarray(sources, dtype=float).reshape(-1, 3)
        strength_arr = np.asarray(strengths, dtype=float).reshape(-1)
        source_key = self._surface_diagnostic_response_source_key(source_arr)
        measurement_count = int(np.asarray(data.live_times, dtype=float).size)
        cache_enabled = (
            source_key is not None
            and strength_arr.size == source_arr.shape[0]
            and np.allclose(strength_arr, 1.0)
            and int(self.pf_config.surface_diagnostic_response_cache_max_entries) > 0
        )
        cache_key: tuple[Any, ...] | None = None
        prefix_key: tuple[Any, ...] | None = None
        if cache_enabled:
            cache_key = (
                str(isotope),
                int(id(kernel)),
                source_key,
                self._measurement_geometry_digest(data),
            )
            cached = self._surface_diagnostic_response_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            prefix_key = (str(isotope), int(id(kernel)), source_key)
            prefix_payload = self._surface_diagnostic_response_prefix_cache.get(
                prefix_key
            )
            if isinstance(prefix_payload, dict):
                cached_counts = np.asarray(
                    prefix_payload.get("counts", np.zeros((0, 0))),
                    dtype=float,
                )
                cached_rows = int(cached_counts.shape[0])
                if (
                    cached_rows >= measurement_count
                    and self._surface_diagnostic_response_prefix_matches(
                        prefix_payload,
                        data,
                        measurement_count,
                    )
                ):
                    counts_arr = cached_counts[:measurement_count].copy()
                    self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
                    return counts_arr
                if (
                    cached_rows < measurement_count
                    and self._surface_diagnostic_response_prefix_matches(
                        prefix_payload,
                        data,
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
                    )
                    counts_arr = np.vstack(
                        [
                            cached_counts,
                            np.asarray(suffix_counts, dtype=float),
                        ]
                    )
                    self._store_surface_diagnostic_response_prefix_cache(
                        prefix_key,
                        data,
                        counts_arr,
                    )
                    self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
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
        )
        counts_arr = np.asarray(counts, dtype=float)
        if cache_enabled and cache_key is not None:
            self._store_surface_diagnostic_response_cache(cache_key, counts_arr)
            if prefix_key is not None:
                self._store_surface_diagnostic_response_prefix_cache(
                    prefix_key,
                    data,
                    counts_arr,
                )
        return counts_arr

    def _cached_expected_counts_per_source(
        self,
        *,
        filt: IsotopeParticleFilter,
        isotope: str,
        data: StructuralGeometryBatch,
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
        self, mu_by_isotope: dict[str, object] | None
    ) -> dict[str, object]:
        """Require one authenticated attenuation entry for every isotope."""
        if not isinstance(mu_by_isotope, dict) or any(
            not isinstance(key, str) for key in mu_by_isotope
        ):
            raise TypeError("mu_by_isotope must be a string-keyed dictionary.")
        expected = set(self.isotopes)
        actual = set(mu_by_isotope)
        if actual != expected:
            raise ValueError(
                "mu_by_isotope keys must exactly match configured isotopes: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        return dict(mu_by_isotope)

    def _ensure_kernel_cache(self) -> None:
        """Build the discrete kernel cache when it is first needed."""
        if self.kernel_cache is not None:
            return
        if len(self.poses) == 0:
            raise ValueError("No poses added; cannot build kernel cache.")
        poses_arr = np.stack(self.poses, axis=0)
        self.kernel_cache = MeasurementGeometry(
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
        self._configure_joint_particle_filters()

    def _build_filter(
        self,
        isotope: str,
        pf_conf: RotatingShieldPFConfig,
    ) -> IsotopeParticleFilter:
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
            detector_impact_parameter_edges_fraction=(
                self.full_spectrum_generative_model
                .detector_impact_parameter_edges_fraction
            ),
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            strict_catalog_line_contract=self.strict_catalog_line_contract,
            dry_air_total_attenuation_contract_id=(
                self.dry_air_total_attenuation_contract_id
            ),
            dry_air_total_attenuation_contract_sha256=(
                self.dry_air_total_attenuation_contract_sha256
            ),
            additive_scatter_response=self.additive_scatter_response,
            random_seed=self.random_seed,
        )

    def _build_pf_config(self) -> RotatingShieldPFConfig:
        """Build an independent canonical config for one isotope filter."""
        return replace(self.pf_config)

    def _gpu_enabled(self) -> bool:
        """Return True if GPU computation is enabled and available."""
        from pf import gpu_utils

        if not self.pf_config.use_gpu:
            raise RuntimeError(
                "GPU-only mode: enable use_gpu in RotatingShieldPFConfig."
            )
        gpu_utils.require_torch_compute_device(
            str(self.pf_config.gpu_device),
            str(self.pf_config.gpu_dtype),
        )
        return True

    def _can_use_gpu(self) -> bool:
        """Select explicit NumPy mode or require the configured torch device."""
        if not self.pf_config.use_gpu:
            return False
        return self._gpu_enabled()

    def continuous_kernel(
        self,
        *,
        detector_aperture_samples: int | None = None,
        use_gpu: bool | None = None,
    ) -> ContinuousKernel:
        """Build the shared ContinuousKernel for PF, planning, and diagnostics."""
        requested_aperture_samples = (
            self.detector_aperture_samples
            if detector_aperture_samples is None
            else detector_aperture_samples
        )
        active_aperture_samples = _strict_nonnegative_integer(
            requested_aperture_samples,
            name="detector_aperture_samples",
        )
        if active_aperture_samples < 1:
            raise ValueError("detector_aperture_samples must be positive.")
        active_use_gpu = _strict_config_boolean(
            self.pf_config.use_gpu if use_gpu is None else use_gpu,
            name="use_gpu",
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
            detector_aperture_samples=active_aperture_samples,
            detector_aperture_sampling=self.detector_aperture_sampling,
            source_extent_radius_m=self.source_extent_radius_m,
            source_extent_samples=self.source_extent_samples,
            line_mu_by_isotope=self.line_mu_by_isotope,
            strict_catalog_line_contract=self.strict_catalog_line_contract,
            dry_air_total_attenuation_contract_id=(
                self.dry_air_total_attenuation_contract_id
            ),
            dry_air_total_attenuation_contract_sha256=(
                self.dry_air_total_attenuation_contract_sha256
            ),
            additive_scatter_response=self.additive_scatter_response,
        )

    def surface_transport_positions(
        self,
        isotope: str,
        anchors_xyz: NDArray[np.float64],
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map exact surface anchors to the shared air-side physics positions."""
        isotope_key = str(isotope)
        try:
            filt = self.filters[isotope_key]
        except KeyError as error:
            raise ValueError(
                f"Unknown isotope for surface transport: {isotope_key!r}."
            ) from error
        atlas = filt._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        anchors = np.asarray(anchors_xyz, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids)
        uv = np.asarray(surface_uv, dtype=np.float64)
        if (
            anchors.ndim != 2
            or anchors.shape[1] != 3
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != anchors.shape[:1]
            or uv.shape != anchors.shape[:1] + (2,)
        ):
            raise ValueError(
                "Surface anchors, chart IDs, and UV coordinates are misaligned."
            )
        chart_id_array = np.asarray(raw_chart_ids, dtype=np.int64)
        mapped = atlas.positions_xyz(chart_id_array, uv)
        if not np.allclose(mapped, anchors, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Surface chart/UV coordinates do not map to their exact anchors."
            )
        return filt._surface_transport_positions(
            anchors,
            chart_ids=chart_id_array,
        )

    def configured_isotope_order(self) -> tuple[str, ...]:
        """Return the stable pure-PF isotope order."""
        return tuple(dict.fromkeys(str(isotope) for isotope in self.isotopes))

    def joint_isotope_order(self) -> tuple[str, ...]:
        """Return the canonical order used by every joint likelihood vector."""
        order = tuple(sorted(self.configured_isotope_order()))
        if not order or len(order) != len(self.isotopes):
            raise RuntimeError("Joint PF requires unique configured isotope names.")
        return order

    def _configure_joint_particle_filters(self) -> None:
        """Attach the conditional joint target and verify aligned initialization."""
        if not self.filters:
            return
        if not self._joint_particles_initialized:
            self._initialize_joint_particles_from_product_prior()
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            filt.set_joint_target_evaluator(self._joint_structural_target_evaluator)
            filt.set_joint_history_tree_evaluator(
                self._joint_structural_history_tree_evaluator
            )
            filt.set_joint_strength_grid_target_evaluator(
                self._joint_structural_strength_grid_target_evaluator
            )
            filt.set_joint_proposal_evaluator(self._joint_structural_proposal_evaluator)
        self._assert_joint_particle_alignment()

    def _joint_row_identity_root(self, *, particle_count: int) -> str:
        """Return the deterministic contract root for joint row identities."""
        atlas_sha256 = self._assert_joint_surface_atlas_alignment()
        model_sha256 = self._full_spectrum_model().contract_hash_sha256
        return strict_sha256_json(
            {
                "schema_version": 1,
                "identity_domain": "pure_pf_joint_row_identity_root_v1",
                "random_seed": self.random_seed,
                "isotope_order": list(self.joint_isotope_order()),
                "particle_count": int(particle_count),
                "surface_atlas_sha256": atlas_sha256,
                "full_spectrum_contract_sha256": model_sha256,
                "pf_config": asdict(self.pf_config),
            }
        )

    def _initialize_joint_particles_from_product_prior(self) -> None:
        """Draw aligned rows directly from the independent isotope product prior."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint prior initialization requires every isotope filter."
            )
        particle_counts = {
            int(filt.config.num_particles) for filt in self.filters.values()
        }
        if len(particle_counts) != 1:
            raise RuntimeError(
                "Joint prior initialization requires one common particle count."
            )
        particle_count = next(iter(particle_counts))
        if particle_count <= 0:
            raise RuntimeError(
                "Joint prior initialization requires positive particle count."
            )
        common_log_weight = float(-np.log(particle_count))
        identity_root = self._joint_row_identity_root(particle_count=particle_count)
        row_identities = tuple(
            JointRowIdentity.initial(
                root_sha256=identity_root,
                ordinal=row,
            )
            for row in range(particle_count)
        )
        particles_by_isotope: dict[str, list[IsotopeParticle]] = {}
        for isotope in order:
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            cardinality_prior = filt._structural_rj_cardinality_prior
            if atlas is None or cardinality_prior is None:
                raise RuntimeError(
                    "Joint product-prior initialization requires a surface "
                    "atlas and cardinality prior."
                )
            rng = named_random_generator(
                self.random_seed,
                "joint_product_prior",
                isotope,
            )
            if filt.config.variable_cardinality:
                cardinalities = rng.choice(
                    cardinality_prior.probabilities.size,
                    size=particle_count,
                    replace=True,
                    p=cardinality_prior.probabilities,
                ).astype(np.int64, copy=False)
            else:
                cardinalities = np.full(
                    particle_count,
                    int(filt.config.init_num_sources[0]),
                    dtype=np.int64,
                )
            source_offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    np.cumsum(cardinalities, dtype=np.int64),
                )
            )
            total_sources = int(source_offsets[-1])
            chart_ids, surface_uv, _ = atlas.sample(
                total_sources,
                rng=rng,
            )
            strengths = np.asarray(
                filt._strength_prior.sample(total_sources, rng=rng),
                dtype=np.float64,
            )
            particles: list[IsotopeParticle] = []
            # Numerical sampling is batched above; this loop only creates the
            # variable-length Python state objects used by the RJ kernel.
            for row, cardinality in enumerate(cardinalities.tolist()):
                begin = int(source_offsets[row])
                end = int(source_offsets[row + 1])
                state = IsotopeState(
                    num_sources=int(cardinality),
                    strengths=np.asarray(
                        strengths[begin:end],
                        dtype=np.float64,
                    ).copy(),
                    surface_chart_ids=np.asarray(
                        chart_ids[begin:end],
                        dtype=np.int64,
                    ).copy(),
                    surface_uv=np.asarray(
                        surface_uv[begin:end],
                        dtype=np.float64,
                    ).copy(),
                )
                filt._canonicalize_structural_rj_state(state)
                particles.append(
                    IsotopeParticle(
                        state=state,
                        log_weight=common_log_weight,
                        joint_row_identity=row_identities[row],
                    )
                )
            particles_by_isotope[isotope] = particles
        for isotope in order:
            filt = self.filters[isotope]
            filt.continuous_particles = particles_by_isotope[isotope]
            filt.N = particle_count
            filt.config.num_particles = particle_count
        self._joint_row_identity_root_sha256 = identity_root
        self._joint_row_generation = 0
        self._joint_cumulative_lineage_ids = np.arange(
            particle_count,
            dtype=np.int64,
        )
        self._joint_lineage_recovery_active = False
        self._joint_lineage_recovery_epoch = 0
        self._joint_lineage_recovery_certified_mask_by_isotope = {
            isotope: np.zeros(particle_count, dtype=np.bool_)
            for isotope in order
        }
        self._joint_particles_initialized = True
        self._invalidate_posterior_summary_cache()
        state_matrix = np.asarray(
            self._joint_mixing_snapshot()["state_matrix"],
            dtype=np.float64,
        )
        self._joint_initial_product_prior_state_sha256 = hashlib.sha256(
            np.ascontiguousarray(state_matrix).tobytes(order="C")
        ).hexdigest()

    def _assert_joint_particle_alignment(self) -> NDArray[np.float64]:
        """Return common log weights or reject any broken joint-row alignment."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint PF requires exactly one filter for every configured isotope."
            )
        self._assert_joint_surface_atlas_alignment()
        particle_counts = {
            len(self.filters[isotope].continuous_particles) for isotope in order
        }
        if len(particle_counts) != 1:
            raise RuntimeError(
                "Joint isotope filters must have the same particle count."
            )
        particle_count = next(iter(particle_counts))
        if particle_count <= 0:
            raise RuntimeError("Joint PF requires at least one aligned particle.")
        identity_root = self._joint_row_identity_root_sha256
        identity_generation = self._joint_row_generation
        if identity_root is None or identity_generation is None:
            raise RuntimeError("Joint PF row identity contract is not initialized.")
        reference_identities: tuple[JointRowIdentity, ...] | None = None
        for isotope in order:
            identities: list[JointRowIdentity] = []
            for row, particle in enumerate(self.filters[isotope].continuous_particles):
                identity = particle.joint_row_identity
                if not isinstance(identity, JointRowIdentity):
                    raise RuntimeError(
                        "Every joint isotope particle requires an immutable "
                        "joint row identity."
                    )
                if (
                    identity.validate() != identity.row_sha256
                    or identity.root_sha256 != identity_root
                    or identity.generation != identity_generation
                    or identity.ordinal != row
                ):
                    raise RuntimeError(
                        "Joint row identity does not match the current "
                        "estimator generation and row ordering."
                    )
                identities.append(identity)
            identity_tuple = tuple(identities)
            if reference_identities is None:
                if (
                    len({identity.row_sha256 for identity in identity_tuple})
                    != particle_count
                ):
                    raise RuntimeError(
                        "Current joint PF row identities must be unique."
                    )
                reference_identities = identity_tuple
            elif identity_tuple != reference_identities:
                raise RuntimeError(
                    "Joint isotope filters lost their authenticated row "
                    "identity ordering."
                )
        reference = np.asarray(
            [
                particle.log_weight
                for particle in self.filters[order[0]].continuous_particles
            ],
            dtype=np.float64,
        )
        if (
            reference.shape != (particle_count,)
            or np.any(np.isnan(reference))
            or np.any(np.isposinf(reference))
            or not np.any(np.isfinite(reference))
        ):
            raise RuntimeError("Joint PF common log weights are invalid.")
        for isotope in order[1:]:
            candidate = np.asarray(
                [
                    particle.log_weight
                    for particle in self.filters[isotope].continuous_particles
                ],
                dtype=np.float64,
            )
            if not np.array_equal(candidate, reference):
                raise RuntimeError(
                    "Joint isotope filters lost their exact common weights/order."
                )
        return reference

    def _assert_joint_surface_atlas_alignment(self) -> str:
        """Return the shared atlas digest or reject isotope geometry drift."""
        order = self.joint_isotope_order()
        if set(self.filters) != set(order):
            raise RuntimeError(
                "Joint PF requires exactly one filter for every configured isotope."
            )
        digests = tuple(
            str(
                getattr(
                    self.filters[isotope],
                    "structural_rj_surface_atlas_sha256",
                    "",
                )
            ).strip()
            for isotope in order
        )
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for digest in digests
        ):
            raise RuntimeError(
                "Every joint isotope filter requires a valid surface-atlas digest."
            )
        if len(set(digest.lower() for digest in digests)) != 1:
            raise RuntimeError(
                "Joint isotope rows cannot use different continuous surface atlases."
            )
        return digests[0].lower()

    def initialize_joint_particle_filters(self) -> str:
        """Initialize every joint PF row and return its shared atlas digest.

        Measurement poses are registered before the first station is observed,
        while the particle filters themselves are intentionally built lazily.
        Runtime preflight must use this method instead of inspecting ``filters``
        directly so atlas validation cannot race that lazy initialization.
        """
        self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        return self._assert_joint_surface_atlas_alignment()

    def initialize_joint_exact_transport_cache(self) -> int:
        """Reserve the fixed live CUDA transport cache before acquisition."""
        if self._joint_persistent_structural_transport_cache is not None:
            raise RuntimeError(
                "The fixed joint transport cache is already initialized."
            )
        self.initialize_joint_particle_filters()
        if not bool(self.pf_config.use_gpu):
            raise RuntimeError(
                "Live joint exact inference requires CUDA; CPU fallback is "
                "not permitted."
            )
        import torch

        model = self._full_spectrum_model()
        source_slots = (
            len(self.joint_isotope_order())
            * int(self.pf_config.cardinality_capacity)
        )
        cache = JointTransportCache.allocate_empty_torch(
            particle_count=int(self.pf_config.num_particles),
            source_slots=source_slots,
            line_count=len(tuple(model.line_identity)),
            feature_count=len(tuple(model.transport_feature_order)),
            device=torch.device(str(self.pf_config.gpu_device)),
            dtype=torch.float64,
            state_sha256=self._joint_structural_state_sha256(),
            row_generation=self._joint_row_generation,
        )
        self._joint_persistent_structural_transport_cache = cache
        return cache.allocated_bytes

    def continuous_surface_atlas(self) -> Any:
        """Return the one authoritative atlas shared by all isotope filters."""
        if not self.filters:
            raise RuntimeError(
                "The continuous surface atlas is unavailable before PF initialization."
            )
        self._assert_joint_surface_atlas_alignment()
        atlas = next(iter(self.filters.values()))._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return atlas

    def _assign_joint_log_weights(
        self,
        normalized_log_weights: NDArray[np.float64],
    ) -> None:
        """Copy one normalized log-weight vector to every aligned isotope row."""
        self._invalidate_posterior_summary_cache()
        values = np.asarray(normalized_log_weights, dtype=np.float64).reshape(-1)
        self._assert_joint_particle_alignment()
        if (
            values.size
            != len(self.filters[self.joint_isotope_order()[0]].continuous_particles)
            or np.any(np.isnan(values))
            or np.any(np.isposinf(values))
            or not np.any(np.isfinite(values))
        ):
            raise ValueError("Joint normalized log weights are invalid.")
        finite = np.isfinite(values)
        normalizer = float(logsumexp(values[finite]))
        if not np.isclose(normalizer, 0.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("Joint log weights must already be normalized.")
        for isotope in self.joint_isotope_order():
            for particle, log_weight in zip(
                self.filters[isotope].continuous_particles,
                values,
            ):
                particle.log_weight = float(log_weight)
        self._assert_joint_particle_alignment()

    def _strict_joint_particle_weights(self) -> NDArray[np.float64]:
        """Return normalized joint weights or reject invalid posterior mass."""
        log_weights = self._assert_joint_particle_alignment()
        finite = np.isfinite(log_weights)
        weights = np.zeros(log_weights.size, dtype=np.float64)
        weights[finite] = np.exp(
            log_weights[finite] - float(logsumexp(log_weights[finite]))
        )
        return validated_probability_distribution(
            weights,
            name="joint PF particle weights",
        )




    def configured_isotope_response_counts(
        self,
        isotope: str,
        data: StructuralGeometryBatch,
        source_positions: NDArray[np.float64],
        strengths: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return batched responses from the configured isotope PF kernel.

        Measurement rows and source positions are evaluated by the same continuous
        transport, obstacle, aperture, and shield model used by the PF.
        Candidate positions remain batched and particle state is not read.
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

    def planning_joint_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> JointPlanningParticles:
        """Return one common-index numeric snapshot of the joint posterior."""
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        weights = self._strict_joint_particle_weights()
        particle_count = int(weights.size)
        if max_particles is None:
            if method is not None or rng is not None:
                raise ValueError(
                    "method and rng must be omitted when all planning particles "
                    "are requested."
                )
            indices = np.arange(particle_count, dtype=np.int64)
            selected_weights = weights.copy()
        else:
            if isinstance(max_particles, bool) or not isinstance(
                max_particles,
                (int, np.integer),
            ):
                raise TypeError("max_particles must be an integer or None.")
            resolved_max_particles = int(max_particles)
            if not 1 <= resolved_max_particles <= particle_count:
                raise ValueError(
                    "max_particles must lie between 1 and the particle count."
                )
            if method not in {"top_weight", "resample"}:
                raise ValueError(
                    "A strict 'top_weight' or 'resample' method is required "
                    "when planning selects a particle count."
                )
        if max_particles is not None and method == "top_weight":
            indices = np.argsort(weights)[::-1][:resolved_max_particles].astype(
                np.int64,
                copy=False,
            )
            selected_weights = weights[indices]
            selected_weights /= float(np.sum(selected_weights))
        elif max_particles is not None and method == "resample":
            if rng is not None and not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator.")
            if rng is None:
                rng = self._named_planning_rng(
                    "joint_particle_subset",
                    resolved_max_particles,
                )
            indices = np.asarray(
                rng.choice(
                    particle_count,
                    size=resolved_max_particles,
                    replace=True,
                    p=weights,
                ),
                dtype=np.int64,
            )
            selected_weights = np.full(
                resolved_max_particles,
                1.0 / float(resolved_max_particles),
                dtype=np.float64,
            )
        max_sources = self.pf_config.cardinality_capacity
        positions_by_isotope: dict[str, NDArray[np.float64]] = {}
        chart_ids_by_isotope: dict[str, NDArray[np.int64]] = {}
        surface_uv_by_isotope: dict[str, NDArray[np.float64]] = {}
        strengths_by_isotope: dict[str, NDArray[np.float64]] = {}
        masks_by_isotope: dict[str, NDArray[np.bool_]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            (
                all_positions,
                all_strengths,
                all_mask,
                all_chart_ids,
                all_surface_uv,
            ) = filt._packed_continuous_surface_state_arrays()
            selected_positions = all_positions[indices]
            selected_strengths = all_strengths[indices]
            selected_mask = all_mask[indices]
            selected_chart_ids = all_chart_ids[indices]
            selected_surface_uv = all_surface_uv[indices]
            current_slots = int(selected_positions.shape[1])
            if current_slots > max_sources:
                raise RuntimeError(
                    "Joint planning state exceeds configured max_sources."
                )
            positions = np.zeros(
                (indices.size, max_sources, 3),
                dtype=np.float64,
            )
            chart_ids = np.zeros(
                (indices.size, max_sources),
                dtype=np.int64,
            )
            surface_uv = np.zeros(
                (indices.size, max_sources, 2),
                dtype=np.float64,
            )
            strengths = np.zeros(
                (indices.size, max_sources),
                dtype=np.float64,
            )
            mask = np.zeros(
                (indices.size, max_sources),
                dtype=bool,
            )
            if current_slots:
                positions[:, :current_slots, :] = selected_positions
                chart_ids[:, :current_slots] = selected_chart_ids
                surface_uv[:, :current_slots, :] = selected_surface_uv
                strengths[:, :current_slots] = selected_strengths
                mask[:, :current_slots] = selected_mask
            positions_by_isotope[isotope] = positions
            chart_ids_by_isotope[isotope] = chart_ids
            surface_uv_by_isotope[isotope] = surface_uv
            strengths_by_isotope[isotope] = strengths
            masks_by_isotope[isotope] = mask
        return JointPlanningParticles(
            isotope_order=self.joint_isotope_order(),
            weights_n=np.ascontiguousarray(selected_weights),
            positions_nk3_by_isotope=positions_by_isotope,
            surface_chart_ids_nk_by_isotope=chart_ids_by_isotope,
            surface_uv_nk2_by_isotope=surface_uv_by_isotope,
            strengths_nk_by_isotope=strengths_by_isotope,
            source_mask_nk_by_isotope=masks_by_isotope,
            original_particle_indices=np.ascontiguousarray(indices),
        )

    def planning_particles(
        self,
        max_particles: int | None = None,
        method: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[list[IsotopeState], NDArray[np.float64]]]:
        """
        Select per-isotope particle subsets for orientation evaluation.

        Args:
            max_particles: cap on particles per isotope; None keeps every particle.
            method: "top_weight" or "resample" when a subset is requested.
            rng: optional RNG for resampling.
        """
        joint = self.planning_joint_particles(
            max_particles=max_particles,
            method=method,
            rng=rng,
        )
        return {
            isotope: (
                [
                    self.filters[isotope].continuous_particles[int(index)].state.copy()
                    for index in joint.original_particle_indices
                ],
                joint.weights_n.copy(),
            )
            for isotope in joint.isotope_order
        }

    def add_measurement_pose(
        self, pose: NDArray[np.float64], reset_filters: bool = False
    ) -> None:
        """Register a pose without erasing posterior state unless requested."""
        if reset_filters and (self._joint_station_history or self.measurements):
            raise RuntimeError(
                "Pure PF cannot reset particles after observations have "
                "entered the posterior."
            )
        self.poses.append(np.asarray(pose, dtype=float))
        # Rebuild lazily on the next access.
        self.kernel_cache = None
        if reset_filters:
            self._invalidate_posterior_summary_cache()
            self.filters = {}
            self._joint_particles_initialized = False
            self._joint_row_identity_root_sha256 = None
            self._joint_row_generation = None
            self._joint_cumulative_lineage_ids = None
            self._joint_lineage_recovery_active = False
            self._joint_lineage_recovery_epoch = 0
            self._joint_lineage_recovery_certified_mask_by_isotope = {
                str(isotope): np.zeros(0, dtype=np.bool_)
                for isotope in self.isotopes
            }
            self._joint_birth_proposal_prefix_scores = {}
            self._joint_birth_proposal_prefix_station_count = 0
            self._joint_birth_proposal_reference_mean_vb = None

    def _registered_detector_position_xyz(
        self,
        pose_idx: int,
    ) -> tuple[float, float, float]:
        """Return the canonical registered detector position for a pose index."""
        resolved_pose_idx = _strict_nonnegative_integer(
            pose_idx,
            name="pose_idx",
        )
        if resolved_pose_idx >= len(self.poses):
            raise IndexError("pose_idx lies outside the registered measurement poses.")
        position = np.asarray(
            self.poses[resolved_pose_idx],
            dtype=float,
        ).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError(
                "Registered measurement poses must contain three finite coordinates."
            )
        return tuple(float(value) for value in position)

    def update_spectrum_station(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> None:
        """Assimilate one shield program through the sole spectrum likelihood."""
        if not records:
            raise ValueError(
                "A spectrum station must contain at least one shield view."
            )
        sequence_start = time.perf_counter()
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        station_sequence_id = len(self._joint_station_history)
        station = self._joint_station_from_spectrum_records(
            records,
            pose_idx=pose_idx,
            station_sequence_id=station_sequence_id,
            generative_contract_hash_sha256=(generative_contract_hash_sha256),
        )
        update_start = time.perf_counter()
        self._joint_tempered_station_update(station)
        update_wall = time.perf_counter() - update_start
        self.last_pair_sequence_update_wall_s = float(update_wall)
        self.last_structural_update_wall_s = float(
            sum(
                float(
                    self.filters[isotope].last_structural_timing_s.get(
                        "total",
                        0.0,
                    )
                )
                for isotope in self.joint_isotope_order()
            )
        )
        detector_position = station.detector_position_xyz_m
        for view_index, record in enumerate(records):
            self.measurements.append(
                MeasurementRecord(
                    spectrum_counts_b=np.ascontiguousarray(
                        station.spectrum_vb[view_index]
                    ),
                    pose_idx=int(station.pose_idx),
                    live_time_s=float(station.live_times_s[view_index]),
                    fe_index=int(station.fe_indices[view_index]),
                    pb_index=int(station.pb_indices[view_index]),
                    detector_position_xyz_m=detector_position,
                    station_sequence_id=int(station_sequence_id),
                    station_view_index=int(view_index),
                    generative_contract_hash_sha256=str(
                        generative_contract_hash_sha256
                    ),
                )
            )
        report_start = time.perf_counter()
        self._refresh_posterior_summary()
        report_wall = time.perf_counter() - report_start
        self.last_pair_sequence_stage_wall_s = {
            "normalize_and_validate": float(update_start - sequence_start),
            "joint_smc_and_conditional_rj": float(update_wall),
            "posterior_report": float(report_wall),
            "total": float(time.perf_counter() - sequence_start),
        }
