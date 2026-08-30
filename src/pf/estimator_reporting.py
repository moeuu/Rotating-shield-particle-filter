"""Posterior summaries and convergence diagnostics for the PF estimator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from measurement.model import EnvironmentConfig
from measurement.source_surfaces import (
    SOURCE_SURFACE_REPORT_LABELS,
    source_surface_kinds,
)
from pf.estimator_config import (
    _strict_config_number,
    _strict_nonnegative_integer,
)
from pf.estimator_sampling import _stratified_categorical_draws
from pf.estimator_surface import (
    SurfaceAtlasQuadrature,
    build_complete_surface_atlas_quadrature,
)
from pf.estimator_types import MeasurementRecord
from pf.particle_filter import (
    IsotopeParticleFilter,
    StructuralGeometryBatch,
)
from pf.posterior import (
    PFPointEstimate,
    posterior_point_estimate_from_states,
    validated_probability_distribution,
    validated_state_cardinality,
)
from pf.posterior_uncertainty import posterior_mode_uncertainty_batched
from pf.randomness import named_random_generator


class EstimatorReportingMixin:
    """Provide posterior reporting, uncertainty, and stopping diagnostics."""

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

    def _structural_geometry_for_records(
        self,
        isotope: str,
        window: int | None,
        records: Sequence[MeasurementRecord] | None = None,
    ) -> StructuralGeometryBatch | None:
        """Return geometry-only rows without an independent isotope target."""
        isotope_key = str(isotope)
        if isotope_key not in self.joint_isotope_order():
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        if records is not None:
            selected_records = list(records)
        elif window is None or int(window) <= 0:
            selected_records = list(self.measurements)
        else:
            selected_records = self.measurements[-int(window) :]
        if not selected_records:
            return None
        return StructuralGeometryBatch(
            detector_positions=np.asarray(
                [record.detector_position_xyz_m for record in selected_records],
                dtype=np.float64,
            ),
            fe_indices=np.asarray(
                [record.fe_index for record in selected_records],
                dtype=np.int64,
            ),
            pb_indices=np.asarray(
                [record.pb_index for record in selected_records],
                dtype=np.int64,
            ),
            live_times=np.asarray(
                [record.live_time_s for record in selected_records],
                dtype=np.float64,
            ),
            station_sequence_ids=self._station_sequence_ids_for_records(
                selected_records
            ),
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

    def surface_atlas_observability_diagnostics(
        self,
        *,
        window: int | None = None,
        max_candidates: int = 256,
    ) -> dict[str, dict[str, Any]]:
        (
            "Return truth-independent observability diagnostics over surface "
            "candidates."
        )
        diagnostics: dict[str, dict[str, Any]] = {}
        if self.surface_diagnostic_points.size == 0:
            return diagnostics
        self._ensure_kernel_cache()
        pool_all = np.asarray(self.surface_diagnostic_points, dtype=float).reshape(
            -1, 3
        )
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
            for kind in SOURCE_SURFACE_REPORT_LABELS[:-1]
        }
        surface_counts["off_surface"] = int(
            np.count_nonzero(np.equal(surface_kinds, None))
        )
        eps = 1.0e-12
        for isotope, filt in self.filters.items():
            data = self._structural_geometry_for_records(isotope, window)
            if data is None or data.row_count == 0:
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
            stats = self._response_design_observability_stats(
                np.asarray(candidate_counts, dtype=float),
                eps=eps,
            )
            stats.update(
                {
                    "candidate_count": int(pool_all.shape[0]),
                    "sampled_candidate_count": int(pool.shape[0]),
                    "measurement_count": data.row_count,
                    "surface_counts": surface_counts,
                    "window": None if window is None else int(window),
                }
            )
            diagnostics[isotope] = stats
        return diagnostics

    def source_response_signatures(
        self,
        isotope: str,
        positions_xyz_m: NDArray[np.float64],
        *,
        window: int | None = None,
    ) -> NDArray[np.float64]:
        """Return normalized measured-view response columns for source points.

        The signature uses the same physical transport kernel and completed
        measurement geometry as inference.  It is intended for reporting
        whether two nearby components are physically distinguishable; it does
        not alter particles, weights, cardinality, or the PF target.
        """
        positions = np.asarray(positions_xyz_m, dtype=np.float64)
        if positions.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_xyz_m must have shape (source, 3).")
        if np.any(~np.isfinite(positions)):
            raise ValueError("Source-response positions must be finite.")
        filt = self.filters.get(str(isotope))
        if filt is None:
            raise KeyError(f"Unknown PF isotope: {isotope}")
        data = self._structural_geometry_for_records(str(isotope), window)
        if data is None or data.row_count == 0:
            return np.zeros((0, positions.shape[0]), dtype=np.float64)
        counts = np.asarray(
            self._cached_expected_counts_per_source(
                filt=filt,
                isotope=str(isotope),
                data=data,
                sources=positions,
                strengths=np.ones(positions.shape[0], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        if counts.shape != (data.row_count, positions.shape[0]):
            raise RuntimeError("Source-response signature shape is invalid.")
        norms = np.linalg.norm(counts, axis=0)
        return np.divide(
            counts,
            norms[None, :],
            out=np.zeros_like(counts),
            where=norms[None, :] > 0.0,
        )

    def surface_atlas_area_quadrature(
        self,
        *,
        max_points: int,
        maximum_hausdorff_bound_m: float,
    ) -> SurfaceAtlasQuadrature:
        """Return every chart center with its exact physical chart area.

        A finite area-quantile sample can omit small charts entirely and create
        an exploration absorbing state. Production coverage therefore uses one
        center per continuous chart and fails before planning when its explicit
        budget or chart-center Hausdorff bound cannot cover the atlas.
        """
        if not self.filters:
            raise RuntimeError(
                "Surface-atlas coverage requires initialized PF filters."
            )
        self._assert_joint_surface_atlas_alignment()
        atlases = [filt._structural_rj_surface_atlas for filt in self.filters.values()]
        if any(atlas is None for atlas in atlases):
            raise RuntimeError(
                "Surface-atlas coverage requires a continuous surface atlas."
            )
        atlas = atlases[0]
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        return build_complete_surface_atlas_quadrature(
            atlas,
            max_points=int(max_points),
            maximum_hausdorff_bound_m=float(maximum_hausdorff_bound_m),
        )

    def posterior_point_estimate(self) -> dict[str, PFPointEstimate]:
        """Return deterministic posterior summaries for every isotope.

        The pure runtime subclass overrides this method to select one aligned
        joint-cardinality stratum and one common representative particle row.
        Keeping all report consumers on this virtual method prevents stopping
        and visualization from silently reverting to independent isotope
        marginals.
        """
        cached = self._cached_posterior_point_estimate()
        if cached is not None:
            return cached
        estimates: dict[str, PFPointEstimate] = {}
        for isotope, filt in self.filters.items():
            filt.validate_continuous_surface_states()
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            estimates[isotope] = posterior_point_estimate_from_states(
                [particle.state for particle in filt.continuous_particles],
                np.asarray(filt.continuous_weights, dtype=float),
                max_cardinality=self.pf_config.cardinality_capacity,
                positions_by_state=[
                    filt.continuous_state_positions(particle.state)
                    for particle in filt.continuous_particles
                ],
                surface_chart_ids_by_state=(
                    None
                    if atlas is None
                    else [
                        np.asarray(
                            particle.state.surface_chart_ids,
                            dtype=np.int64,
                        )
                        for particle in filt.continuous_particles
                    ]
                ),
                surface_uv_by_state=(
                    None
                    if atlas is None
                    else [
                        np.asarray(
                            particle.state.surface_uv,
                            dtype=np.float64,
                        )
                        for particle in filt.continuous_particles
                    ]
                ),
                surface_coordinate_path_distance=(
                    None
                    if atlas is None
                    else atlas.surface_coordinate_path_distance_upper_bound_m
                ),
            )
        return self._store_posterior_point_estimate(estimates)

    def estimates(
        self,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the canonical MAP-cardinality PF posterior projection."""
        return self._project_posterior_point_estimates(self.posterior_point_estimate())

    def structural_surface_kinds(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        *,
        strict: bool = True,
    ) -> NDArray[np.object_]:
        """Return authoritative continuous-surface kinds for one isotope."""
        filt = self.filters.get(str(isotope))
        if filt is None:
            raise KeyError(f"Unknown PF isotope: {isotope}")
        return filt.structural_surface_kinds(
            np.asarray(positions, dtype=np.float64),
            strict=strict,
        )

    def _posterior_reporting_particle_indices(
        self,
    ) -> NDArray[np.int64] | None:
        """Return a shared posterior-report stratum or all rows by default."""
        return None

    def posterior_source_uncertainty(
        self,
        reported_estimates: Mapping[
            str,
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ]
        | None = None,
        *,
        match_radius_m: float | None = None,
        surface_tolerance_m: float = 1.0e-5,
    ) -> dict[str, list[dict[str, Any]]]:
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
        point_estimate_map = self.posterior_point_estimate()
        estimate_map = (
            {
                isotope: (
                    np.asarray(
                        [mode.position_medoid_xyz for mode in point_estimate.modes],
                        dtype=np.float64,
                    ).reshape(-1, 3),
                    np.asarray(
                        [
                            mode.strength_representative_cps_1m
                            for mode in point_estimate.modes
                        ],
                        dtype=np.float64,
                    ),
                )
                for isotope, point_estimate in point_estimate_map.items()
            }
            if reported_estimates is None
            else dict(reported_estimates)
        )
        radius = 0.8 if match_radius_m is None else float(match_radius_m)
        environment = self._source_prior_environment()
        reporting_indices = self._posterior_reporting_particle_indices()
        output: dict[str, list[dict[str, Any]]] = {}
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
                packed_chart_ids = np.zeros((0, 0), dtype=np.int64)
                packed_surface_uv = np.zeros((0, 0, 2), dtype=np.float64)
                weights = np.zeros(0, dtype=float)
                atlas = None
            else:
                (
                    packed_positions,
                    _,
                    packed_mask,
                    packed_chart_ids,
                    packed_surface_uv,
                ) = filt._packed_continuous_surface_state_arrays()
                weights = np.asarray(filt.continuous_weights, dtype=float)
                atlas = getattr(filt, "_structural_rj_surface_atlas", None)
                if atlas is None:
                    raise RuntimeError(
                        "Posterior uncertainty requires the continuous surface atlas."
                    )
                if reporting_indices is not None:
                    indices = np.asarray(
                        reporting_indices,
                        dtype=np.int64,
                    ).reshape(-1)
                    if (
                        indices.size == 0
                        or np.any(indices < 0)
                        or np.any(indices >= weights.size)
                    ):
                        raise RuntimeError(
                            "Posterior uncertainty received an invalid shared "
                            "reporting stratum."
                        )
                    packed_positions = packed_positions[indices]
                    packed_mask = packed_mask[indices]
                    packed_chart_ids = packed_chart_ids[indices]
                    packed_surface_uv = packed_surface_uv[indices]
                    reporting_mass = float(np.sum(weights[indices]))
                    if not np.isfinite(reporting_mass) or reporting_mass <= 0.0:
                        raise RuntimeError(
                            "Posterior reporting stratum has no probability mass."
                        )
                    weights = validated_probability_distribution(
                        weights[indices] / reporting_mass,
                        name="posterior reporting-stratum weights",
                    )
            packed_surface_kinds = np.full(packed_mask.shape, None, dtype=object)
            if np.any(packed_mask):
                packed_surface_kinds[packed_mask] = filt.structural_surface_kinds(
                    packed_positions[packed_mask],
                    strict=True,
                )
            reported_chart_ids = np.zeros(
                positions.shape[0],
                dtype=np.int64,
            )
            reported_surface_uv = np.zeros(
                (positions.shape[0], 2),
                dtype=np.float64,
            )
            if positions.shape[0] > 0:
                if atlas is None:
                    raise RuntimeError(
                        "Reported continuous-surface modes require an atlas."
                    )
                point_estimate = point_estimate_map.get(isotope)
                point_positions = np.asarray(
                    []
                    if point_estimate is None
                    else [mode.position_medoid_xyz for mode in point_estimate.modes],
                    dtype=np.float64,
                ).reshape(-1, 3)
                point_modes_have_coordinates = bool(
                    point_estimate is not None
                    and len(point_estimate.modes) == positions.shape[0]
                    and np.array_equal(point_positions, positions)
                    and all(
                        mode.surface_chart_id is not None
                        and mode.surface_uv is not None
                        for mode in point_estimate.modes
                    )
                )
                if point_modes_have_coordinates and point_estimate is not None:
                    reported_chart_ids = np.asarray(
                        [int(mode.surface_chart_id) for mode in point_estimate.modes],
                        dtype=np.int64,
                    )
                    reported_surface_uv = np.asarray(
                        [mode.surface_uv for mode in point_estimate.modes],
                        dtype=np.float64,
                    )
                else:
                    reported_chart_ids, reported_surface_uv = atlas.locate_positions(
                        positions
                    )

            diagnostics = posterior_mode_uncertainty_batched(
                packed_positions,
                packed_mask,
                weights,
                positions,
                packed_surface_kinds=packed_surface_kinds,
                packed_surface_chart_ids=(None if atlas is None else packed_chart_ids),
                packed_surface_uv=(None if atlas is None else packed_surface_uv),
                reported_surface_chart_ids=(
                    None if atlas is None else reported_chart_ids
                ),
                reported_surface_uv=(None if atlas is None else reported_surface_uv),
                surface_coordinate_path_distance=(
                    None
                    if atlas is None
                    else atlas.surface_coordinate_path_distance_upper_bound_m
                ),
                environment=environment,
                obstacle_grid=self.obstacle_grid,
                obstacle_height_m=self.obstacle_height_m,
                match_radius_m=radius,
                surface_tolerance_m=surface_tolerance_m,
                posterior_reference_mass=(
                    1.0
                    if point_estimate_map.get(isotope) is None
                    else float(point_estimate_map[isotope].selected_stratum_mass)
                ),
            )
            for mode_index, diagnostic in enumerate(diagnostics):
                diagnostic["reported_strength_cps_1m"] = float(strengths[mode_index])
            output[isotope] = diagnostics
        return output

    def step_diagnostics(
        self,
        top_k: int = 3,
        *,
        include_estimates: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Return per-isotope diagnostics for the current PF state.

        The diagnostics include ESS, resample/birth/death counts, and the source
        count distribution. When include_estimates is false, the routine avoids
        the posterior point-estimate projection.
        """
        diagnostics: dict[str, dict[str, Any]] = {}
        eps = 1e-12
        k = max(0, int(top_k))
        posterior_estimates = self.estimates() if include_estimates else {}
        for iso, filt in self.filters.items():
            if not filt.continuous_particles:
                diagnostics[iso] = {
                    "ess_pre": 0.0,
                    "resampled": False,
                    "ess_post": None,
                    "current_ess": 0.0,
                    "current_ess_ratio": 0.0,
                    "particle_count": 0,
                    "resample_count": int(getattr(filt, "last_resample_count", 0)),
                    "birth_count": int(getattr(filt, "last_birth_count", 0)),
                    "death_count": int(getattr(filt, "last_death_count", 0)),
                    "structural_timing_s": dict(
                        getattr(filt, "last_structural_timing_s", {})
                    ),
                    "transition_weight_mass": {},
                    "temper_steps": [],
                    "joint_rejuvenation_diagnostics": [],
                    "joint_smc_wall_time_limit_exceeded": False,
                    "joint_guided_initialization_ess": None,
                    "joint_cross_isotope_state_rejection_diagnostics": {},
                    "joint_transport_cache": {},
                    "temper_resamples": 0,
                    "temper_min_ess": None,
                    "station_unique_ancestor_count": None,
                    "cumulative_unique_ancestor_count": None,
                    "r_mean": 0.0,
                    "r_var": 0.0,
                    "r_weighted_mean": 0.0,
                    "r_weighted_var": 0.0,
                    "r_probability_by_count": {},
                    "r_particle_count_by_count": {},
                    "map": (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)),
                    "posterior": (
                        np.zeros((0, 3), dtype=float),
                        np.zeros(0, dtype=float),
                    ),
                    "top_k": [],
                }
                continue
            weights = np.asarray(filt.continuous_weights, dtype=float)
            total = float(np.sum(weights))
            if (
                weights.size != len(filt.continuous_particles)
                or not np.isfinite(total)
                or total <= 0.0
            ):
                raise RuntimeError(
                    "Step diagnostics require valid normalized PF weights."
                )
            weights = weights / total
            current_ess = float(1.0 / max(np.sum(weights**2), eps))
            current_ess_ratio = current_ess / float(weights.size)
            maximum_cardinality = self.pf_config.cardinality_capacity
            r_int = np.fromiter(
                (
                    validated_state_cardinality(
                        particle.state,
                        name=f"{iso} particle[{index}]",
                        max_cardinality=maximum_cardinality,
                    )
                    for index, particle in enumerate(filt.continuous_particles)
                ),
                dtype=np.int64,
                count=len(filt.continuous_particles),
            )
            r_vals = r_int.astype(np.float64)
            if weights.size != r_vals.size:
                raise RuntimeError(
                    "Step diagnostics require one weight per PF particle."
                )
            r_mean = float(np.mean(r_vals)) if r_vals.size else 0.0
            r_var = float(np.var(r_vals)) if r_vals.size else 0.0
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
            if include_estimates:
                posterior_positions, posterior_strengths = posterior_estimates[iso]
            else:
                posterior_positions = np.zeros((0, 3), dtype=float)
                posterior_strengths = np.zeros(0, dtype=float)
            top_entries: list[dict[str, Any]] = []
            if k > 0 and weights.size:
                order = np.argsort(weights)[::-1][:k]
                for idx in order:
                    state = filt.continuous_particles[int(idx)].state
                    source_count = validated_state_cardinality(
                        state,
                        name=f"{iso} particle[{int(idx)}]",
                        max_cardinality=maximum_cardinality,
                    )
                    top_entries.append(
                        {
                            "weight": float(weights[idx]),
                            "num_sources": source_count,
                            "positions": filt.continuous_state_positions(state)[
                                :source_count
                            ],
                            "strengths": state.strengths[:source_count].copy(),
                        }
                    )
            diagnostics[iso] = {
                "ess_pre": float(ess_pre),
                "resampled": resampled,
                "ess_post": ess_post,
                "current_ess": current_ess,
                "current_ess_ratio": current_ess_ratio,
                "particle_count": particle_count,
                "resample_count": int(getattr(filt, "last_resample_count", 0)),
                "birth_count": int(getattr(filt, "last_birth_count", 0)),
                "death_count": int(getattr(filt, "last_death_count", 0)),
                "structural_timing_s": dict(
                    getattr(filt, "last_structural_timing_s", {})
                ),
                "transition_weight_mass": dict(
                    getattr(
                        filt,
                        "last_structural_transition_weight_mass",
                        {},
                    )
                ),
                "structural_rejection_diagnostics": dict(
                    getattr(
                        filt,
                        "last_structural_rejection_diagnostics",
                        {},
                    )
                ),
                "temper_steps": list(getattr(filt, "last_temper_steps", [])),
                "joint_rejuvenation_diagnostics": [
                    dict(entry) for entry in self.last_joint_rejuvenation_diagnostics
                ],
                "joint_smc_wall_time_limit_exceeded": bool(
                    self.last_joint_smc_wall_time_limit_exceeded
                ),
                "joint_rejuvenation_mixing_incomplete": bool(
                    self.last_joint_rejuvenation_mixing_incomplete
                ),
                "joint_structural_mixing_incomplete": bool(
                    self.last_joint_structural_mixing_incomplete_by_isotope.get(
                        str(iso),
                        self.last_joint_structural_mixing_incomplete,
                    )
                ),
                "joint_guided_initialization_ess": (
                    self.last_joint_guided_initialization_ess
                ),
                "joint_cross_isotope_state_rejection_diagnostics": dict(
                    self.last_joint_cross_isotope_state_rejection_diagnostics
                ),
                "joint_transport_cache": {
                    "preflight": (
                        None
                        if self.joint_transport_cache_preflight is None
                        else dict(self.joint_transport_cache_preflight)
                    ),
                    "unit_hits": int(self.last_joint_structural_unit_cache_hits),
                    "unit_misses": int(self.last_joint_structural_unit_cache_misses),
                    "staged_transport_commit_rows": int(
                        self.last_joint_staged_transport_commit_rows
                    ),
                    "accepted_state_reuses": int(
                        self.last_joint_persistent_cache_reuse_count
                    ),
                    "history_appends": int(
                        self.last_joint_persistent_cache_append_count
                    ),
                    "ancestor_reindexes": int(
                        self.last_joint_persistent_cache_reindex_count
                    ),
                    "slot_overlay_likelihood_calls": int(
                        self.last_joint_slot_overlay_likelihood_calls
                    ),
                    "full_history_clone_count": int(
                        self.last_joint_full_history_clone_count
                    ),
                    "past_slab_recopy_count": 0,
                    "station_likelihood_reuses": int(
                        self.last_joint_station_likelihood_cache_reuse_count
                    ),
                    "station_likelihood_appends": int(
                        self.last_joint_station_likelihood_append_count
                    ),
                    "station_likelihood_full_refreshes": int(
                        self.last_joint_station_likelihood_full_refresh_count
                    ),
                    "last_slot_overlay": dict(
                        getattr(
                            self._full_spectrum_model(),
                            "last_torch_slot_overlay_diagnostics",
                            {},
                        )
                        or {}
                    ),
                    "valid_views": (
                        int(
                            self._joint_persistent_structural_transport_cache
                            .valid_view_count
                        )
                        if hasattr(
                            self._joint_persistent_structural_transport_cache,
                            "valid_view_count",
                        )
                        else None
                    ),
                    "allocated_bytes": (
                        int(
                            self._joint_persistent_structural_transport_cache
                            .allocated_bytes
                        )
                        if hasattr(
                            self._joint_persistent_structural_transport_cache,
                            "allocated_bytes",
                        )
                        else None
                    ),
                    "resident_device": (
                        "cuda"
                        if self._joint_persistent_structural_transport_cache
                        is not None
                        and hasattr(
                            self._joint_persistent_structural_transport_cache[0],
                            "detach",
                        )
                        else "cpu"
                        if self._joint_persistent_structural_transport_cache
                        is not None
                        else None
                    ),
                },
                "temper_resamples": int(getattr(filt, "last_temper_resample_count", 0)),
                "temper_min_ess": getattr(filt, "last_temper_min_ess", None),
                "station_unique_ancestor_count": getattr(
                    filt,
                    "last_station_unique_ancestor_count",
                    None,
                ),
                "cumulative_unique_ancestor_count": getattr(
                    filt,
                    "last_cumulative_unique_ancestor_count",
                    None,
                ),
                "r_mean": r_mean,
                "r_var": r_var,
                "r_weighted_mean": r_weighted_mean,
                "r_weighted_var": r_weighted_var,
                "r_probability_by_count": r_probability_by_count,
                "r_particle_count_by_count": r_particle_count_by_count,
                "posterior": (
                    posterior_positions,
                    posterior_strengths,
                ),
                "top_k": top_entries,
            }
        return diagnostics

    @property
    def num_orientations(self) -> int:
        """Return the number of shield orientation normals."""
        return self.normals.shape[0]

    @staticmethod
    def _normalized_stopping_weights(
        filt: IsotopeParticleFilter,
    ) -> NDArray[np.float64]:
        """Return one filter's valid normalized posterior weights."""
        particle_count = len(filt.continuous_particles)
        weights = np.asarray(filt.continuous_weights, dtype=float).reshape(-1)
        if weights.size != particle_count:
            raise RuntimeError(
                "Stopping diagnostics require one weight per PF particle."
            )
        if weights.size == 0 or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise RuntimeError(
                "Stopping diagnostics require finite nonnegative PF weights."
            )
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "Stopping diagnostics require positive posterior weight mass."
            )
        return weights / total

    def credible_surface_radii(
        self,
        confidence: float = 0.95,
    ) -> dict[str, list[float]]:
        """Return conservative path radii along connected physical surfaces.

        Each radius is computed inside the MAP-cardinality stratum after
        deterministic source alignment.  Its center is an actual posterior
        surface state. Same-chart and one-portal paths are evaluated directly;
        longer paths use a realizable portal-graph path. Disconnected posterior
        mass makes the radius infinite. This fail-closed statistic cannot
        collapse merely because a broad posterior lies on a two-dimensional
        wall, floor, or ceiling.
        """
        probability = float(confidence)
        if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("confidence must be in (0, 1].")
        radii: dict[str, list[float]] = {}
        point_estimates = self.posterior_point_estimate()
        for isotope, filt in self.filters.items():
            if not filt.continuous_particles:
                radii[isotope] = []
                continue
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            if atlas is None:
                raise RuntimeError(
                    "Surface convergence requires a continuous surface atlas."
                )
            filt.validate_continuous_surface_states()
            estimate = point_estimates.get(isotope)
            if estimate is None:
                raise RuntimeError(
                    "Canonical posterior report omitted an initialized isotope."
                )
            if np.isclose(probability, 0.95, rtol=0.0, atol=1.0e-15):
                radii[isotope] = [
                    (
                        float("inf")
                        if mode.credible_surface_path_radius_95_m is None
                        else float(mode.credible_surface_path_radius_95_m)
                    )
                    for mode in estimate.modes
                ]
                continue

            # The reporting contract stores the 95% radius.  Other confidence
            # levels are intentionally unavailable here instead of being
            # inferred from a Gaussian ellipsoid on a non-Euclidean surface.
            raise ValueError(
                "credible_surface_radii currently supports confidence=0.95 only."
            )
        return radii

    def _latest_joint_station_innovation(
        self,
    ) -> dict[str, float | int | bool | None]:
        """Return strict renewal-total and conditional-mark innovation gates."""
        if not self._joint_station_history:
            return {
                "available": False,
                "passed": False,
                "view_count": 0,
                "dimension": 0,
                "renewal_total_max_abs_z": None,
                "renewal_total_within_confidence": False,
                "conditional_mark_pearson": None,
                "conditional_mark_degrees_of_freedom": 0,
                "conditional_mark_tail_probability": None,
                "conditional_mark_upper_tail_probability": None,
                "confidence": float(
                    self.pf_config.adaptive_stop_innovation_confidence
                ),
            }
        station = self._joint_station_history[-1]
        weights = self._strict_joint_particle_weights()
        components = tuple(
            value.detach().cpu().numpy().astype(np.float64, copy=False)
            for value in self._joint_station_transport_components_torch(station)
        )
        raw_result = dict(
            self._full_spectrum_model().posterior_predictive_innovation_numpy(
                station.spectrum_vb,
                components[0],
                components[1],
                components[2],
                station.live_times_s,
                weights,
                confidence=float(
                    self.pf_config.adaptive_stop_innovation_confidence
                ),
            )
        )
        required_raw = {
            "renewal_total_max_abs_z",
            "renewal_total_within_confidence",
            "conditional_mark_pearson",
            "conditional_mark_degrees_of_freedom",
            "conditional_mark_tail_probability",
            "conditional_mark_upper_tail_probability",
            "confidence",
        }
        if set(raw_result) != required_raw:
            raise RuntimeError(
                "Full-spectrum innovation returned an incompatible diagnostic schema."
            )
        confidence = float(raw_result["confidence"])
        total_z = float(raw_result["renewal_total_max_abs_z"])
        mark_pearson = float(raw_result["conditional_mark_pearson"])
        mark_degrees = raw_result["conditional_mark_degrees_of_freedom"]
        mark_tail_raw = raw_result["conditional_mark_tail_probability"]
        mark_upper_tail_raw = raw_result["conditional_mark_upper_tail_probability"]
        if (
            not np.isfinite(confidence)
            or not np.isclose(
                confidence,
                float(self.pf_config.adaptive_stop_innovation_confidence),
                rtol=0.0,
                atol=1.0e-15,
            )
            or not np.isfinite(total_z)
            or total_z < 0.0
            or type(raw_result["renewal_total_within_confidence"]) is not bool
            or not np.isfinite(mark_pearson)
            or mark_pearson < 0.0
            or isinstance(mark_degrees, (bool, np.bool_))
            or not isinstance(mark_degrees, (int, np.integer))
        ):
            raise RuntimeError("Full-spectrum innovation contains invalid diagnostics.")
        mark_tail: float | None
        if mark_tail_raw is None:
            mark_tail = None
        else:
            mark_tail = float(mark_tail_raw)
            if not np.isfinite(mark_tail) or mark_tail < 0.0 or mark_tail > 1.0:
                raise RuntimeError(
                    "Full-spectrum conditional-mark tail probability is invalid."
                )
        mark_upper_tail: float | None
        if mark_upper_tail_raw is None:
            mark_upper_tail = None
        else:
            mark_upper_tail = float(mark_upper_tail_raw)
            if (
                not np.isfinite(mark_upper_tail)
                or mark_upper_tail < 0.0
                or mark_upper_tail > 1.0
            ):
                raise RuntimeError(
                    "Full-spectrum conditional-mark upper-tail probability is invalid."
                )
        mark_passed = bool(
            mark_tail is not None and mark_tail + 1.0e-15 >= 1.0 - confidence
        )
        total_passed = bool(raw_result["renewal_total_within_confidence"])
        return {
            "available": True,
            "passed": bool(total_passed and mark_passed),
            "view_count": int(station.spectrum_vb.shape[0]),
            "dimension": int(station.spectrum_vb.size),
            "renewal_total_max_abs_z": total_z,
            "renewal_total_within_confidence": total_passed,
            "conditional_mark_pearson": mark_pearson,
            "conditional_mark_degrees_of_freedom": int(mark_degrees),
            "conditional_mark_tail_probability": mark_tail,
            "conditional_mark_upper_tail_probability": mark_upper_tail,
            "confidence": confidence,
        }

    def posterior_predictive_check(
        self,
        *,
        sample_count: int = 128,
        confidence: float = 0.95,
        worst_bin_count: int = 32,
    ) -> dict[str, object]:
        """Return a model-native posterior predictive residual audit.

        Every predictive spectrum is sampled through the immutable generative
        model, so detector response, background, dead time, and configured
        discrepancy covariance are preserved. Stations are evaluated one at a
        time because station-shared latent variables must not be coupled across
        acquisition boundaries; particles, views, source slots, lines, and
        energy bins remain batched inside each station call.
        """
        count = _strict_nonnegative_integer(
            sample_count,
            name="posterior predictive sample_count",
        )
        if count < 2:
            raise ValueError("posterior predictive sample_count must be at least two.")
        probability = _strict_config_number(
            confidence,
            name="posterior predictive confidence",
        )
        if not 0.0 < probability < 1.0:
            raise ValueError("posterior predictive confidence must lie in (0, 1).")
        maximum_worst_bins = _strict_nonnegative_integer(
            worst_bin_count,
            name="posterior predictive worst_bin_count",
        )
        if not self._joint_station_history:
            return {
                "available": False,
                "sample_count": count,
                "confidence": probability,
                "stations": [],
                "shield_pair_summary": {},
                "obstacle_line_of_sight_summary": {},
                "worst_standardized_bin_residuals": [],
            }
        weights = self._strict_joint_particle_weights()
        rng = named_random_generator(
            self.random_seed,
            "posterior_predictive_residual_audit",
        )
        particle_indices = _stratified_categorical_draws(
            weights,
            count,
            rng=rng,
        )
        model = self._full_spectrum_model()
        feature_names = tuple(model.transport_feature_order)
        obstacle_feature_index = (
            feature_names.index("tau_obstacle")
            if "tau_obstacle" in feature_names
            else None
        )
        alpha = 0.5 * (1.0 - probability)
        station_results: list[dict[str, object]] = []
        pair_values: dict[int, list[NDArray[np.float64]]] = {}
        pair_coverages: dict[int, list[NDArray[np.bool_]]] = {}
        obstacle_values: dict[str, list[NDArray[np.float64]]] = {
            "crosses_obstacle": [],
            "clear_line_of_sight": [],
        }
        isotope_ablation_values: dict[str, list[float]] = {
            isotope: [] for isotope in self.joint_isotope_order()
        }
        worst_rows: list[dict[str, object]] = []
        for station in self._joint_station_history:
            components = tuple(
                value.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
                for value in self._joint_station_transport_components_torch(station)
            )
            selected_components = tuple(value[particle_indices] for value in components)
            sampled = np.asarray(
                model.sample_predictive_numpy(
                    selected_components[0],
                    selected_components[1],
                    selected_components[2],
                    station.live_times_s,
                    sample_count=1,
                    rng=rng,
                )
            )
            expected_shape = (
                count,
                1,
                int(station.fe_indices.size),
                int(station.energy_axis_keV.size),
            )
            if (
                sampled.shape != expected_shape
                or not np.issubdtype(sampled.dtype, np.integer)
                or np.any(sampled < 0)
            ):
                raise RuntimeError(
                    "Posterior predictive sampler returned an invalid batch."
                )
            draws = sampled[:, 0].astype(np.float64, copy=False)
            observed = np.asarray(station.spectrum_vb, dtype=np.float64)
            predictive_mean = np.mean(draws, axis=0)
            predictive_std = np.std(draws, axis=0, ddof=1)
            standardized = (observed - predictive_mean) / np.maximum(
                predictive_std,
                1.0,
            )
            lower = np.quantile(draws, alpha, axis=0)
            upper = np.quantile(draws, 1.0 - alpha, axis=0)
            covered = (observed >= lower) & (observed <= upper)
            observed_totals = np.sum(observed, axis=1, dtype=np.float64)
            predictive_totals = np.sum(draws, axis=2, dtype=np.float64)
            predictive_total_mean = np.mean(predictive_totals, axis=0)
            predictive_total_std = np.std(
                predictive_totals,
                axis=0,
                ddof=1,
            )
            total_standardized = (observed_totals - predictive_total_mean) / np.maximum(
                predictive_total_std, 1.0
            )
            pair_ids = np.asarray(station.fe_indices, dtype=np.int64) * int(
                self.num_orientations
            ) + np.asarray(station.pb_indices, dtype=np.int64)
            obstacle_probability = np.zeros(
                int(station.fe_indices.size),
                dtype=np.float64,
            )
            if obstacle_feature_index is not None:
                tau_obstacle = selected_components[2][..., obstacle_feature_index]
                contributes = selected_components[0] > 0.0
                crosses = np.any(
                    contributes & (tau_obstacle > 1.0e-12),
                    axis=(2, 3),
                )
                obstacle_probability = np.mean(crosses, axis=0)
            view_rows: list[dict[str, object]] = []
            for view_index in range(int(station.fe_indices.size)):
                pair_id = int(pair_ids[view_index])
                pair_values.setdefault(pair_id, []).append(
                    standardized[view_index].copy()
                )
                pair_coverages.setdefault(pair_id, []).append(
                    covered[view_index].copy()
                )
                obstacle_label = (
                    "crosses_obstacle"
                    if obstacle_probability[view_index] >= 0.5
                    else "clear_line_of_sight"
                )
                obstacle_values[obstacle_label].append(standardized[view_index].copy())
                view_rows.append(
                    {
                        "view_index": int(view_index),
                        "fe_orientation_index": int(station.fe_indices[view_index]),
                        "pb_orientation_index": int(station.pb_indices[view_index]),
                        "shield_pair_id": pair_id,
                        "observed_total_count": float(observed_totals[view_index]),
                        "predictive_total_mean": float(
                            predictive_total_mean[view_index]
                        ),
                        "predictive_total_std": float(predictive_total_std[view_index]),
                        "standardized_total_residual": float(
                            total_standardized[view_index]
                        ),
                        "maximum_abs_standardized_bin_residual": float(
                            np.max(np.abs(standardized[view_index]))
                        ),
                        "p95_abs_standardized_bin_residual": float(
                            np.quantile(
                                np.abs(standardized[view_index]),
                                0.95,
                            )
                        ),
                        "marginal_bin_coverage_fraction": float(
                            np.mean(covered[view_index])
                        ),
                        "posterior_obstacle_crossing_probability": float(
                            obstacle_probability[view_index]
                        ),
                    }
                )
            flat_count = min(
                maximum_worst_bins,
                int(standardized.size),
            )
            if flat_count:
                flat_abs = np.abs(standardized).reshape(-1)
                candidate_flat = np.argpartition(
                    flat_abs,
                    -flat_count,
                )[-flat_count:]
                view_indices, bin_indices = np.unravel_index(
                    candidate_flat,
                    standardized.shape,
                )
                for view_index, bin_index in zip(
                    view_indices.tolist(),
                    bin_indices.tolist(),
                    strict=True,
                ):
                    worst_rows.append(
                        {
                            "station_sequence_id": int(station.station_sequence_id),
                            "view_index": int(view_index),
                            "shield_pair_id": int(pair_ids[view_index]),
                            "energy_keV": float(station.energy_axis_keV[bin_index]),
                            "bin_index": int(bin_index),
                            "observed_count": float(observed[view_index, bin_index]),
                            "predictive_mean": float(
                                predictive_mean[view_index, bin_index]
                            ),
                            "predictive_std": float(
                                predictive_std[view_index, bin_index]
                            ),
                            "standardized_residual": float(
                                standardized[view_index, bin_index]
                            ),
                        }
                    )
            innovation = dict(
                model.posterior_predictive_innovation_numpy(
                    observed,
                    components[0],
                    components[1],
                    components[2],
                    station.live_times_s,
                    weights,
                    confidence=probability,
                )
            )
            full_log_likelihood = np.asarray(
                model.log_likelihood_numpy(
                    observed,
                    components[0],
                    components[1],
                    components[2],
                    station.live_times_s,
                ),
                dtype=np.float64,
            )
            log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
            full_log_predictive_density = float(
                logsumexp(log_weights + full_log_likelihood)
            )
            station_isotope_ablation: dict[str, object] = {}
            line_identity = tuple(model.line_identity)
            for isotope in self.joint_isotope_order():
                isotope_line_mask = np.asarray(
                    [str(payload["isotope"]) == isotope for payload in line_identity],
                    dtype=np.bool_,
                )
                if not np.any(isotope_line_mask):
                    raise RuntimeError(
                        f"No full-spectrum line columns exist for {isotope}."
                    )
                ablated_total = components[0].copy()
                ablated_uncollided = components[1].copy()
                ablated_total[..., isotope_line_mask] = 0.0
                ablated_uncollided[..., isotope_line_mask] = 0.0
                ablated_log_likelihood = np.asarray(
                    model.log_likelihood_numpy(
                        observed,
                        ablated_total,
                        ablated_uncollided,
                        components[2],
                        station.live_times_s,
                    ),
                    dtype=np.float64,
                )
                ablated_log_predictive_density = float(
                    logsumexp(log_weights + ablated_log_likelihood)
                )
                density_delta = (
                    full_log_predictive_density - ablated_log_predictive_density
                )
                isotope_ablation_values[isotope].append(density_delta)
                ablated_innovation = dict(
                    model.posterior_predictive_innovation_numpy(
                        observed,
                        ablated_total,
                        ablated_uncollided,
                        components[2],
                        station.live_times_s,
                        weights,
                        confidence=probability,
                    )
                )
                station_isotope_ablation[isotope] = {
                    "full_minus_ablation_log_predictive_density": float(density_delta),
                    "ablated_model_native_innovation": ablated_innovation,
                }
            station_results.append(
                {
                    "station_sequence_id": int(station.station_sequence_id),
                    "pose_index": int(station.pose_idx),
                    "view_count": int(station.fe_indices.size),
                    "energy_bin_count": int(station.energy_axis_keV.size),
                    "observed_total_count": float(np.sum(observed_totals)),
                    "predictive_total_mean": float(np.sum(predictive_total_mean)),
                    "maximum_abs_standardized_bin_residual": float(
                        np.max(np.abs(standardized))
                    ),
                    "p95_abs_standardized_bin_residual": float(
                        np.quantile(np.abs(standardized), 0.95)
                    ),
                    "marginal_bin_coverage_fraction": float(np.mean(covered)),
                    "model_native_innovation": innovation,
                    "isotope_response_ablation": station_isotope_ablation,
                    "views": view_rows,
                }
            )

        def _group_summary(
            values: Sequence[NDArray[np.float64]],
            coverage: Sequence[NDArray[np.bool_]] | None = None,
        ) -> dict[str, float | int | None]:
            """Summarize residual arrays for one physical view group."""
            if not values:
                return {
                    "view_count": 0,
                    "mean_standardized_bin_residual": None,
                    "maximum_abs_standardized_bin_residual": None,
                    "p95_abs_standardized_bin_residual": None,
                    "marginal_bin_coverage_fraction": None,
                }
            flattened = np.concatenate(
                [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
            )
            return {
                "view_count": int(len(values)),
                "mean_standardized_bin_residual": float(np.mean(flattened)),
                "maximum_abs_standardized_bin_residual": float(
                    np.max(np.abs(flattened))
                ),
                "p95_abs_standardized_bin_residual": float(
                    np.quantile(np.abs(flattened), 0.95)
                ),
                "marginal_bin_coverage_fraction": (
                    None
                    if coverage is None
                    else float(
                        np.mean(
                            np.concatenate(
                                [
                                    np.asarray(value, dtype=np.bool_).reshape(-1)
                                    for value in coverage
                                ]
                            )
                        )
                    )
                ),
            }

        pair_summary = {
            str(pair_id): _group_summary(
                pair_values[pair_id],
                pair_coverages[pair_id],
            )
            for pair_id in sorted(pair_values)
        }
        obstacle_summary = {
            label: _group_summary(values) for label, values in obstacle_values.items()
        }
        isotope_ablation_summary = {
            isotope: {
                "station_count": int(len(values)),
                "sum_full_minus_ablation_log_predictive_density": float(
                    np.sum(values, dtype=np.float64)
                ),
                "median_full_minus_ablation_log_predictive_density": float(
                    np.median(values)
                ),
                "minimum_full_minus_ablation_log_predictive_density": float(
                    np.min(values)
                ),
                "maximum_full_minus_ablation_log_predictive_density": float(
                    np.max(values)
                ),
            }
            for isotope, values in isotope_ablation_values.items()
            if values
        }
        worst_rows.sort(
            key=lambda row: abs(float(row["standardized_residual"])),
            reverse=True,
        )
        return {
            "available": True,
            "sampling_semantics": (
                "stratified_posterior_particle_draw_then_one_exact_"
                "generative_spectrum_per_draw"
            ),
            "sample_count": count,
            "confidence": probability,
            "isotopes": list(self.joint_isotope_order()),
            "stations": station_results,
            "shield_pair_summary": pair_summary,
            "obstacle_line_of_sight_summary": obstacle_summary,
            "isotope_response_ablation_summary": isotope_ablation_summary,
            "worst_standardized_bin_residuals": worst_rows[:maximum_worst_bins],
            "isotope_response_ablation_semantics": (
                "diagnostic full-spectrum response-column ablation with all "
                "other posterior source states and weights held fixed; this "
                "is not leave-one-isotope-out model evidence"
            ),
        }

    def _stopping_joint_cardinality_distribution(
        self,
    ) -> dict[tuple[int, ...], float]:
        """Return aligned joint-cardinality mass for adaptive stopping."""
        isotope_order = tuple(self.joint_isotope_order())
        if not isotope_order:
            return {}
        weights = self._strict_joint_particle_weights()
        cardinalities = np.column_stack(
            [
                np.fromiter(
                    (
                        int(particle.state.num_sources)
                        for particle in self.filters[isotope].continuous_particles
                    ),
                    dtype=np.int64,
                    count=weights.size,
                )
                for isotope in isotope_order
            ]
        )
        expected_shape = (weights.size, len(isotope_order))
        if cardinalities.shape != expected_shape or np.any(cardinalities < 0):
            raise RuntimeError(
                "Adaptive stopping requires valid aligned joint cardinalities."
            )
        vectors, inverse = np.unique(cardinalities, axis=0, return_inverse=True)
        mass = validated_probability_distribution(
            np.bincount(inverse, weights=weights, minlength=vectors.shape[0]),
            name="adaptive-stop joint cardinality mass",
        )
        return {
            tuple(int(value) for value in vector): float(probability)
            for vector, probability in zip(vectors, mass, strict=True)
        }

    def posterior_convergence_diagnostics(self) -> dict[str, Any]:
        """Return model-native adaptive-stop gates without simulation truth."""
        isotope_diagnostics: dict[str, dict[str, Any]] = {}
        joint_innovation = self._latest_joint_station_innovation()
        point_estimates = self.posterior_point_estimate()
        isotope_order = tuple(self.joint_isotope_order())
        joint_distribution = self._stopping_joint_cardinality_distribution()
        joint_map_vector = tuple(
            int(point_estimates[isotope].map_cardinality)
            for isotope in isotope_order
            if isotope in point_estimates
        )
        joint_map_probability = float(joint_distribution.get(joint_map_vector, 0.0))
        sampler_health = {
            "smc_rejuvenation_wall_time_respected": bool(
                not self.last_joint_smc_wall_time_limit_exceeded
            ),
            "rejuvenation_mixing_complete": bool(
                not self.last_joint_rejuvenation_mixing_incomplete
            ),
            "structural_mixing_complete": bool(
                not self.last_joint_structural_mixing_incomplete
            ),
        }
        joint_gates = {
            "joint_map_cardinality_probability": bool(
                joint_map_probability
                >= self.pf_config
                .adaptive_stop_minimum_joint_map_cardinality_probability
            ),
            "full_spectrum_innovation": bool(joint_innovation["passed"]),
            **sampler_health,
        }
        all_isotopes_ready = True
        for isotope, filt in self.filters.items():
            if not filt.continuous_particles:
                isotope_diagnostics[isotope] = {
                    "ready": False,
                    "reason": "missing_particles",
                }
                all_isotopes_ready = False
                continue
            weights = self._normalized_stopping_weights(filt)
            particle_count = int(weights.size)
            ess = float(1.0 / np.sum(weights**2))
            ess_ratio = ess / float(particle_count)
            atlas = getattr(filt, "_structural_rj_surface_atlas", None)
            if atlas is None:
                raise RuntimeError(
                    "Surface convergence requires a continuous surface atlas."
                )
            filt.validate_continuous_surface_states()
            point_estimate = point_estimates.get(isotope)
            if point_estimate is None:
                raise RuntimeError(
                    "Canonical posterior report omitted an initialized isotope."
                )
            distribution = {
                int(cardinality): float(mass)
                for cardinality, mass in point_estimate.cardinality_distribution.items()
            }
            ordinary_maximum = int(filt.config.max_sources or 0)
            boundary_mass = float(
                sum(
                    mass
                    for cardinality, mass in distribution.items()
                    if int(cardinality) >= ordinary_maximum
                )
            )
            radii = [
                (
                    None
                    if mode.credible_surface_path_radius_95_m is None
                    else float(mode.credible_surface_path_radius_95_m)
                )
                for mode in point_estimate.modes
            ]
            connected_masses = [
                float(mode.surface_connected_mass) for mode in point_estimate.modes
            ]
            maximum_radius = (
                None
                if any(radius is None for radius in radii)
                else max(
                    (float(radius) for radius in radii if radius is not None),
                    default=0.0,
                )
            )
            minimum_connected_mass = min(connected_masses, default=1.0)
            gates = {
                "cardinality_not_at_upper_boundary": bool(
                    not filt.config.variable_cardinality
                    or boundary_mass
                    <= self.pf_config.adaptive_stop_maximum_upper_cardinality_mass
                ),
                "surface_path_concentration": bool(
                    maximum_radius is not None
                    and maximum_radius
                    <= self.pf_config.adaptive_stop_maximum_surface_path_radius_95_m
                    and minimum_connected_mass + 1.0e-15 >= 0.95
                ),
            }
            ready = bool(all(gates.values()))
            isotope_diagnostics[isotope] = {
                "ready": ready,
                "current_ess": ess,
                "particle_count": particle_count,
                "current_ess_ratio": ess_ratio,
                "cardinality_distribution": distribution,
                "selected_joint_map_cardinality": int(
                    point_estimate.map_cardinality
                ),
                "marginal_map_cardinality_probability": max(
                    distribution.values(),
                    default=0.0,
                ),
                "maximum_cardinality_boundary_mass": boundary_mass,
                "credible_surface_radii_95_m": radii,
                "surface_connected_masses": connected_masses,
                "minimum_surface_connected_mass": minimum_connected_mass,
                "maximum_credible_surface_radius_95_m": maximum_radius,
                "gates": gates,
            }
            all_isotopes_ready &= ready
        joint_ready = bool(joint_distribution) and all(joint_gates.values())
        return {
            "ready": bool(isotope_diagnostics) and all_isotopes_ready and joint_ready,
            "metric": "surface_path_upper_bound_credible_distance",
            "joint_cardinality": {
                "isotope_order": list(isotope_order),
                "map_cardinalities": list(joint_map_vector),
                "map_probability": joint_map_probability,
                "distribution": [
                    {
                        "cardinalities": list(cardinalities),
                        "probability": float(probability),
                    }
                    for cardinalities, probability in sorted(
                        joint_distribution.items()
                    )
                ],
            },
            "innovation": dict(joint_innovation),
            "sampler_health": sampler_health,
            "joint_gates": joint_gates,
            "isotopes": isotope_diagnostics,
        }
