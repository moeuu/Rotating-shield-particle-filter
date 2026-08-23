"""Exact-RJ structural proposal and cached target algorithms."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.estimator_compat import runtime_estimator_export
from pf.estimator_types import JointStationObservation
from pf.particle_filter import IsotopeParticleFilter, StructuralGeometryBatch
from pf.strength_prior import StrengthPrior

if TYPE_CHECKING:
    import torch


def _runtime_facade_constants(*names: str) -> tuple[object, ...]:
    """Resolve compatibility constants from the public estimator facade."""
    return tuple(runtime_estimator_export(name) for name in names)


class EstimatorStructuralProposalMixin:
    """Provide structural proposals, transport caching, and target scoring."""

    def _mix_external_surface_guidance(
        self,
        *,
        isotope: str,
        alignment: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], bool]:
        """Mix one external grid into a full-support proposal, never its target."""
        internal = np.asarray(alignment, dtype=np.float64).reshape(-1)
        external_by_isotope = self._joint_external_surface_guidance_by_isotope
        if external_by_isotope is None:
            return internal, False
        raw_external = external_by_isotope.get(str(isotope))
        if raw_external is None:
            raise RuntimeError(
                "External surface guidance lacks one configured PF isotope."
            )
        external = np.asarray(raw_external, dtype=np.float64).reshape(-1)
        if (
            external.shape != internal.shape
            or np.any(~np.isfinite(external))
            or np.any(external < 0.0)
        ):
            raise RuntimeError(
                "External surface guidance is misaligned with the PF atlas."
            )
        self.last_external_surface_guidance_evaluated_isotopes.add(str(isotope))
        external_maximum = float(np.max(external, initial=0.0))
        if external_maximum <= 0.0:
            return internal, False
        mass = float(self._joint_external_surface_guidance_mass)
        if not 0.0 < mass <= 1.0:
            raise RuntimeError("External surface-guidance proposal mass is invalid.")
        external = external / external_maximum
        internal_maximum = float(np.max(internal, initial=0.0))
        if internal_maximum > 0.0:
            internal = internal / internal_maximum
            mixed = (1.0 - mass) * internal + mass * external
        else:
            mixed = external
        self.last_external_surface_guidance_diagnostics[str(isotope)] = {
            "proposal_mass": mass,
            "external_maximum": external_maximum,
            "mapped_chart_count": float(internal.size),
            "target_preserving_proposal_only": 1.0,
        }
        return np.asarray(mixed, dtype=np.float64), True

    @staticmethod
    def _joint_birth_proposal_station_digest(
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        strength_grid: NDArray[np.float64],
        reference_mean_vb: NDArray[np.float64] | None = None,
    ) -> str:
        """Hash every immutable input to one station proposal-score grid."""
        digest = hashlib.sha256(b"joint_full_spectrum_birth_proposal_station_v1\0")
        for text in (
            str(filt.isotope),
            str(station.generative_contract_hash_sha256),
            str(filt.structural_rj_surface_atlas_sha256),
            str(id(filt.continuous_kernel)),
        ):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(
            np.asarray(
                [station.pose_idx, station.station_sequence_id],
                dtype="<i8",
            ).tobytes()
        )
        for values in (
            station.spectrum_vb,
            station.energy_axis_keV,
            station.detector_position_xyz_m,
            station.fe_indices,
            station.pb_indices,
            station.live_times_s,
            strength_grid,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        if reference_mean_vb is None:
            digest.update(b"proposal_reference=physical_background\0")
        else:
            reference = np.ascontiguousarray(
                reference_mean_vb,
                dtype="<f8",
            )
            digest.update(b"proposal_reference=sequential_mean\0")
            digest.update(np.asarray(reference.shape, dtype="<i8").tobytes())
            digest.update(reference.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _joint_station_structural_geometry(
        station: JointStationObservation,
    ) -> StructuralGeometryBatch:
        """Return the geometry-only carrier for one immutable station."""
        view_count = int(station.fe_indices.size)
        return StructuralGeometryBatch(
            detector_positions=np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                view_count,
                axis=0,
            ),
            fe_indices=np.asarray(station.fe_indices, dtype=np.int64),
            pb_indices=np.asarray(station.pb_indices, dtype=np.int64),
            live_times=np.asarray(station.live_times_s, dtype=np.float64),
            station_sequence_ids=np.full(
                view_count,
                int(station.station_sequence_id),
                dtype=np.int64,
            ),
        )

    @property
    def joint_birth_proposal_cache_bytes(self) -> int:
        """Return bytes occupied by cached chart-by-strength score arrays."""
        return int(
            sum(
                int(values.nbytes)
                for values in (self._joint_birth_proposal_station_score_cache.values())
            )
        )

    def _store_joint_birth_proposal_station_scores(
        self,
        cache_key: tuple[str, str],
        score_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Store one immutable score grid under a strict LRU memory bound."""
        scores = np.ascontiguousarray(score_grid, dtype=np.float64)
        maximum_bytes = int(self.pf_config.structural_rj_proposal_score_cache_max_bytes)
        if int(scores.nbytes) > maximum_bytes:
            raise MemoryError(
                "One full-spectrum birth-proposal score grid exceeds "
                "structural_rj_proposal_score_cache_max_bytes."
            )
        scores.setflags(write=False)
        self._joint_birth_proposal_station_score_cache[cache_key] = scores
        self._joint_birth_proposal_station_score_cache_order = [
            key
            for key in self._joint_birth_proposal_station_score_cache_order
            if key != cache_key
        ]
        self._joint_birth_proposal_station_score_cache_order.append(cache_key)
        while self.joint_birth_proposal_cache_bytes > maximum_bytes:
            oldest = self._joint_birth_proposal_station_score_cache_order.pop(0)
            self._joint_birth_proposal_station_score_cache.pop(oldest, None)
        if cache_key not in self._joint_birth_proposal_station_score_cache:
            raise RuntimeError(
                "Birth-proposal LRU evicted the score grid being inserted."
            )
        return scores

    def _joint_station_birth_proposal_score_grid(
        self,
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        chart_centers_xyz: NDArray[np.float64],
        strength_grid: NDArray[np.float64],
        reference_mean_vb: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return one cached proposal-only chart-by-strength score grid."""
        centers = np.asarray(
            chart_centers_xyz,
            dtype=np.float64,
        ).reshape(-1, 3)
        atlas = filt._structural_rj_surface_atlas
        if atlas is None or int(centers.shape[0]) != int(atlas.chart_count):
            raise ValueError(
                "Birth-proposal centers must contain the authoritative atlas "
                "charts in chart-ID order."
            )
        strengths = np.asarray(
            strength_grid,
            dtype=np.float64,
        ).reshape(-1)
        cache_key = (
            str(filt.isotope),
            self._joint_birth_proposal_station_digest(
                filt=filt,
                station=station,
                strength_grid=strengths,
                reference_mean_vb=reference_mean_vb,
            ),
        )
        cached = self._joint_birth_proposal_station_score_cache.get(cache_key)
        expected_shape = (int(centers.shape[0]), int(strengths.size))
        if cached is not None:
            if cached.shape != expected_shape:
                raise RuntimeError(
                    "Cached birth-proposal score grid has an invalid shape."
                )
            self.last_joint_birth_proposal_cache_hits += 1
            self._joint_birth_proposal_station_score_cache_order = [
                key
                for key in (self._joint_birth_proposal_station_score_cache_order)
                if key != cache_key
            ]
            self._joint_birth_proposal_station_score_cache_order.append(cache_key)
            return cached

        self.last_joint_birth_proposal_cache_misses += 1
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        target_line_mask = np.zeros(line_count, dtype=np.bool_)
        target_line_mask[global_columns] = True
        view_count = int(station.fe_indices.size)
        geometry = self._joint_station_structural_geometry(station)
        score_cg = np.empty(expected_shape, dtype=np.float64)
        batch_size = int(self.pf_config.structural_rj_proposal_chart_batch_size)
        for chart_start in range(0, int(centers.shape[0]), batch_size):
            chart_stop = min(
                chart_start + batch_size,
                int(centers.shape[0]),
            )
            batch_centers = centers[chart_start:chart_stop]
            components = filt._continuous_rj_line_transport_component_columns(
                geometry,
                batch_centers,
                local_indices,
                chart_ids=np.arange(
                    chart_start,
                    chart_stop,
                    dtype=np.int64,
                ),
            )
            local_shape = (
                view_count,
                int(batch_centers.shape[0]),
                int(local_indices.size),
            )
            unit_total = np.asarray(
                components.total_kernel,
                dtype=np.float64,
            ).reshape(local_shape)
            unit_uncollided = np.asarray(
                components.uncollided_kernel,
                dtype=np.float64,
            ).reshape(local_shape)
            unit_features = np.stack(
                (
                    np.asarray(components.tau_fe, dtype=np.float64),
                    np.asarray(components.tau_pb, dtype=np.float64),
                    np.asarray(
                        components.tau_obstacle,
                        dtype=np.float64,
                    ),
                    np.asarray(components.distance_m, dtype=np.float64),
                ),
                axis=-1,
            ).reshape(local_shape + (feature_count,))
            batch_count = int(batch_centers.shape[0])
            candidate_count = batch_count * int(strengths.size)
            total = np.zeros(
                (candidate_count, view_count, 1, line_count),
                dtype=np.float64,
            )
            uncollided = np.zeros_like(total)
            features = np.zeros(
                (
                    candidate_count,
                    view_count,
                    1,
                    line_count,
                    feature_count,
                ),
                dtype=np.float64,
            )
            scale = (
                strengths[None, :, None, None] * branching_weights[None, None, None, :]
            )
            total_local = (
                np.transpose(unit_total, (1, 0, 2))[:, None, :, :] * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            uncollided_local = (
                np.transpose(unit_uncollided, (1, 0, 2))[:, None, :, :] * scale
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
            )
            feature_local = np.broadcast_to(
                np.transpose(unit_features, (1, 0, 2, 3))[:, None, :, :, :],
                (
                    batch_count,
                    int(strengths.size),
                    view_count,
                    int(local_indices.size),
                    feature_count,
                ),
            ).reshape(
                candidate_count,
                view_count,
                int(local_indices.size),
                feature_count,
            )
            total[..., global_columns] = total_local[:, :, None, :]
            uncollided[..., global_columns] = uncollided_local[:, :, None, :]
            features[..., global_columns, :] = feature_local[:, :, None, :, :]
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                scores = model.birth_proposal_log_scores_torch(
                    station.spectrum_vb,
                    torch.as_tensor(
                        total,
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.as_tensor(
                        uncollided,
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.as_tensor(
                        features,
                        dtype=torch.float64,
                        device=device,
                    ),
                    station.live_times_s,
                    target_line_mask_l=torch.as_tensor(
                        target_line_mask,
                        dtype=torch.bool,
                        device=device,
                    ),
                    reference_mean_vb=(
                        None
                        if reference_mean_vb is None
                        else torch.as_tensor(
                            reference_mean_vb,
                            dtype=torch.float64,
                            device=device,
                        )
                    ),
                )
                batch_scores = (
                    scores.detach().cpu().numpy().astype(np.float64, copy=False)
                )
            else:
                batch_scores = np.asarray(
                    model.birth_proposal_log_scores_numpy(
                        station.spectrum_vb,
                        total,
                        uncollided,
                        features,
                        station.live_times_s,
                        target_line_mask_l=target_line_mask,
                        reference_mean_vb=reference_mean_vb,
                    ),
                    dtype=np.float64,
                )
            if batch_scores.shape != (candidate_count,) or np.any(
                ~np.isfinite(batch_scores)
            ):
                raise RuntimeError(
                    "Full-spectrum birth proposal returned invalid scores."
                )
            score_cg[chart_start:chart_stop, :] = batch_scores.reshape(
                batch_count,
                strengths.size,
            )
        return self._store_joint_birth_proposal_station_scores(
            cache_key,
            score_cg,
        )

    def _strength_birth_proposal_grid(
        self,
    ) -> tuple[NDArray[np.float64], float]:
        """Return a finite design grid without truncating prior support."""
        prior = StrengthPrior(
            minimum=float(self.pf_config.strength_prior_min_cps_1m),
            maximum=float(self.pf_config.strength_prior_max_cps_1m),
            family=str(self.pf_config.strength_prior_family),
            gamma_shape=float(self.pf_config.strength_prior_gamma_shape),
            gamma_scale=float(self.pf_config.strength_prior_gamma_scale_cps_1m),
        )
        upper = prior.finite_upper_quantile(0.995)
        grid = np.linspace(
            prior.minimum,
            upper,
            int(self.pf_config.structural_rj_strength_proposal_grid_size),
            dtype=np.float64,
        )
        return grid, float(prior.mean)

    def full_spectrum_isotope_detection_score_grids(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        generative_contract_hash_sha256: str,
    ) -> dict[str, NDArray[np.float64]]:
        """Return truth-free chart-by-strength detection scores for one station."""
        if not records:
            raise ValueError(
                "An isotope-detection station must contain at least one view."
            )
        if self.kernel_cache is None:
            self._ensure_kernel_cache()
        self._assert_joint_particle_alignment()
        station = self._joint_station_from_spectrum_records(
            records,
            pose_idx=pose_idx,
            station_sequence_id=0,
            generative_contract_hash_sha256=(generative_contract_hash_sha256),
        )
        strength_grid, _ = self._strength_birth_proposal_grid()
        result: dict[str, NDArray[np.float64]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Full-spectrum isotope detection requires a surface atlas."
                )
            centers = np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            )
            result[isotope] = np.asarray(
                self._joint_station_birth_proposal_score_grid(
                    filt=filt,
                    station=station,
                    chart_centers_xyz=centers,
                    strength_grid=strength_grid,
                    reference_mean_vb=None,
                ),
                dtype=np.float64,
            ).copy()
        return result

    def _joint_structural_proposal_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        chart_centers_xyz: NDArray[np.float64],
        target_beta: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        bool,
    ]:
        """Build one cached state-independent full-spectrum birth proposal."""
        stations = self._active_joint_station_history
        if stations is None:
            raise RuntimeError("Joint full-spectrum proposal target is not active.")
        self._validate_joint_structural_geometry(data, stations)
        centers = np.asarray(
            chart_centers_xyz,
            dtype=np.float64,
        ).reshape(-1, 3)
        chart_count = int(centers.shape[0])
        atlas = filt._structural_rj_surface_atlas
        if atlas is None or not np.array_equal(
            centers,
            np.asarray(
                atlas.geometry.centers_xyz,
                dtype=np.float64,
            ),
        ):
            raise ValueError(
                "Birth-proposal chart centers must be the immutable active "
                "continuous-surface atlas centers."
            )
        strength_grid, midpoint = self._strength_birth_proposal_grid()
        if chart_count == 0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                False,
            )
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if data.row_count != total_views:
            raise ValueError(
                "Residual proposal geometry differs from joint station history."
            )
        beta = float(target_beta)
        active_prefix = self._active_joint_tempering_prefix_count
        if active_prefix is None:
            newest_station_power = beta
        else:
            newest_view_count = int(stations[-1].fe_indices.size)
            newest_station_power = (int(active_prefix) - 1 + beta) / float(
                newest_view_count
            )
        has_active_likelihood = len(stations) > 1 or newest_station_power > 0.0
        if not has_active_likelihood:
            return (
                np.zeros(chart_count, dtype=np.float64),
                np.full(chart_count, midpoint, dtype=np.float64),
                False,
            )
        completed_count = len(self._joint_station_history)
        if (
            len(stations) != completed_count + 1
            or self._joint_birth_proposal_prefix_station_count != completed_count
        ):
            raise RuntimeError(
                "Birth-proposal prefix does not match the completed station history."
            )
        prefix = self._joint_birth_proposal_prefix_scores.get(str(filt.isotope))
        expected_shape = (chart_count, strength_grid.size)
        if prefix is None:
            if completed_count:
                raise RuntimeError(
                    "Birth-proposal prefix is missing for a configured isotope."
                )
            score_cg = np.zeros(expected_shape, dtype=np.float64)
        else:
            score_cg = np.asarray(prefix, dtype=np.float64).copy()
            if score_cg.shape != expected_shape or np.any(~np.isfinite(score_cg)):
                raise RuntimeError("Birth-proposal prefix score grid is invalid.")
        if newest_station_power > 0.0:
            if (
                self._joint_birth_proposal_reference_mean_vb is not None
                and completed_count != 0
            ):
                raise RuntimeError(
                    "Sequential guided residuals are valid only for the first station."
                )
            score_cg += (
                newest_station_power
                * self._joint_station_birth_proposal_score_grid(
                    filt=filt,
                    station=stations[-1],
                    chart_centers_xyz=centers,
                    strength_grid=strength_grid,
                    reference_mean_vb=(self._joint_birth_proposal_reference_mean_vb),
                )
            )
        best_grid_indices = np.argmax(score_cg, axis=1)
        best_scores = score_cg[
            np.arange(chart_count, dtype=np.int64),
            best_grid_indices,
        ]
        best_locations = strength_grid[best_grid_indices]
        maximum = float(np.max(best_scores))
        informative = bool(np.isfinite(maximum) and maximum > 1.0e-9)
        alignment = (
            np.exp(np.clip(best_scores - maximum, -745.0, 0.0))
            if informative
            else np.zeros(chart_count, dtype=np.float64)
        )
        alignment, external_informative = self._mix_external_surface_guidance(
            isotope=str(filt.isotope),
            alignment=np.asarray(alignment, dtype=np.float64),
        )
        informative = informative or external_informative
        if not informative:
            return (
                np.zeros(chart_count, dtype=np.float64),
                np.full(chart_count, midpoint, dtype=np.float64),
                False,
            )
        return (
            np.asarray(alignment, dtype=np.float64),
            np.asarray(best_locations, dtype=np.float64),
            True,
        )

    def _promote_joint_birth_proposal_station(
        self,
        station: JointStationObservation,
    ) -> None:
        """Add one completed station to each isotope's exact score prefix.

        Birth proposal scores are additive over conditionally independent
        stations.  Keeping the full chart-by-strength sum for completed
        stations is mathematically identical to rescanning every old station,
        while preventing a bounded LRU from cyclically evicting and recomputing
        the entire history during every intermediate rejuvenation.
        """
        completed_count = len(self._joint_station_history)
        if self._joint_birth_proposal_prefix_station_count != completed_count:
            raise RuntimeError("Birth-proposal prefix promotion is out of sequence.")
        strength_grid, _ = self._strength_birth_proposal_grid()
        promoted: dict[str, NDArray[np.float64]] = {}
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            atlas = filt._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Birth-proposal prefix requires a continuous surface atlas."
                )
            station_scores = self._joint_station_birth_proposal_score_grid(
                filt=filt,
                station=station,
                chart_centers_xyz=np.asarray(
                    atlas.geometry.centers_xyz,
                    dtype=np.float64,
                ),
                strength_grid=strength_grid,
            )
            previous = self._joint_birth_proposal_prefix_scores.get(isotope)
            if previous is None:
                if completed_count:
                    raise RuntimeError(
                        "Birth-proposal prefix is missing for a configured isotope."
                    )
                combined = np.asarray(
                    station_scores,
                    dtype=np.float64,
                ).copy()
            else:
                if previous.shape != station_scores.shape:
                    raise RuntimeError(
                        "Birth-proposal prefix and station grids disagree."
                    )
                combined = np.asarray(previous, dtype=np.float64) + np.asarray(
                    station_scores, dtype=np.float64
                )
            if np.any(~np.isfinite(combined)):
                raise RuntimeError("Birth-proposal prefix contains non-finite scores.")
            combined = np.ascontiguousarray(combined, dtype=np.float64)
            combined.setflags(write=False)
            promoted[isotope] = combined
        self._joint_birth_proposal_prefix_scores = promoted
        self._joint_birth_proposal_prefix_station_count = completed_count + 1
        self._joint_birth_proposal_station_score_cache.clear()
        self._joint_birth_proposal_station_score_cache_order.clear()

    @staticmethod
    def _joint_structural_unit_cache_signature(
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positive_line_indices: NDArray[np.int64],
    ) -> str:
        """Hash all immutable inputs to one unit-transport cache generation."""
        digest = hashlib.sha256(b"joint_continuous_surface_unit_transport_cache_v1\0")
        for text in (
            str(filt.isotope),
            str(filt.structural_rj_surface_atlas_sha256),
            str(id(filt.continuous_kernel)),
        ):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for values in (
            data.detector_positions,
            data.fe_indices,
            data.pb_indices,
            data.station_sequence_ids,
            positive_line_indices,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def _joint_cached_continuous_unit_components_shard(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_s3: NDArray[np.float64],
        chart_ids_s: NDArray[np.int64],
        positive_line_indices: NDArray[np.int64],
        prefetched_keys: NDArray[Any] | None = None,
        prefetched_values: tuple[NDArray[np.float64], ...] | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        """Return exact unit transport for one immutable station shard.

        The cache changes only scheduling.  Keys contain the authoritative
        chart and continuous XYZ coordinates, while values are the exact
        continuous-kernel outputs for one immutable station geometry.
        Missing positions are evaluated together by the batched CPU/GPU
        transport kernel, or consumed from an exact multi-station prefetch.
        """
        (
            JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER,
            JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
            JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES,
            JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES,
        ) = _runtime_facade_constants(
            "JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER",
            "JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE",
            "JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES",
            "JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES",
        )
        positions = np.asarray(positions_s3, dtype=np.float64).reshape(-1, 3)
        raw_chart_ids = np.asarray(chart_ids_s)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.size != positions.shape[0]
            or np.any(~np.isfinite(positions))
            or line_indices.size == 0
        ):
            raise ValueError(
                "Cached continuous components require aligned finite surface "
                "positions, integer chart IDs, and positive line indices."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64).reshape(-1)
        row_count = int(data.row_count)
        line_count = int(line_indices.size)
        request_count = int(positions.shape[0])
        if request_count == 0:
            return tuple(
                np.zeros(
                    (row_count, 0, line_count),
                    dtype=np.float64,
                )
                for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
        keys = np.empty(
            request_count,
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        keys["chart"] = chart_ids
        keys["x"] = positions[:, 0]
        keys["y"] = positions[:, 1]
        keys["z"] = positions[:, 2]
        unique_keys, first_indices, inverse = np.unique(
            keys,
            return_index=True,
            return_inverse=True,
        )
        signature = self._joint_structural_unit_cache_signature(
            filt=filt,
            data=data,
            positive_line_indices=line_indices,
        )
        isotope_key = str(filt.isotope)
        isotope_cache = self._joint_structural_unit_transport_cache.setdefault(
            isotope_key,
            {},
        )
        cache = isotope_cache.get(signature)
        if cache is None:
            cache = {
                "signature": signature,
                "keys": np.empty(
                    0,
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                ),
                "recency": np.empty(0, dtype=np.int64),
                "values": tuple(
                    np.zeros(
                        (0, row_count, line_count),
                        dtype=np.float64,
                    )
                    for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
                ),
                "generation": 0,
                "last_access": 0,
            }
        cached_keys = np.asarray(
            cache["keys"],
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        cached_values = tuple(
            np.asarray(values, dtype=np.float64) for values in cache["values"]
        )
        cached_count = int(cached_keys.size)
        lookup = np.searchsorted(cached_keys, unique_keys)
        if cached_count:
            safe_lookup = np.minimum(lookup, cached_count - 1)
            hit = (lookup < cached_count) & (cached_keys[safe_lookup] == unique_keys)
        else:
            safe_lookup = np.zeros(unique_keys.size, dtype=np.int64)
            hit = np.zeros(unique_keys.size, dtype=bool)
        missing = ~hit
        unique_component_values = tuple(
            np.empty(
                (unique_keys.size, row_count, line_count),
                dtype=np.float64,
            )
            for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
        )
        if np.any(hit):
            for output, values in zip(
                unique_component_values,
                cached_values,
                strict=True,
            ):
                output[hit] = values[safe_lookup[hit]]
        missing_first_indices = first_indices[missing]
        missing_values: tuple[NDArray[np.float64], ...]
        if missing_first_indices.size:
            if prefetched_keys is not None or prefetched_values is not None:
                if prefetched_keys is None or prefetched_values is None:
                    raise ValueError(
                        "Prefetched transport keys and values must be paired."
                    )
                supplied_keys = np.asarray(
                    prefetched_keys,
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                ).reshape(-1)
                supplied_values = tuple(
                    np.asarray(values, dtype=np.float64) for values in prefetched_values
                )
                supplied_lookup = np.searchsorted(
                    supplied_keys,
                    unique_keys[missing],
                )
                if (
                    len(supplied_values) != len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES)
                    or np.any(supplied_lookup >= supplied_keys.size)
                    or np.any(supplied_keys[supplied_lookup] != unique_keys[missing])
                ):
                    raise RuntimeError(
                        "Exact transport prefetch does not cover shard misses."
                    )
                missing_values = tuple(
                    values[supplied_lookup] for values in supplied_values
                )
            else:
                evaluated = filt._continuous_rj_line_transport_component_columns(
                    data,
                    positions[missing_first_indices],
                    line_indices,
                    chart_ids=chart_ids[missing_first_indices],
                )
                missing_values = tuple(
                    np.transpose(
                        np.asarray(
                            getattr(evaluated, name),
                            dtype=np.float64,
                        ),
                        (1, 0, 2),
                    )
                    for name in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
                )
            expected_missing_shape = (
                int(missing_first_indices.size),
                row_count,
                line_count,
            )
            if any(
                values.shape != expected_missing_shape
                or np.any(~np.isfinite(values))
                or np.any(values < 0.0)
                for values in missing_values
            ):
                raise RuntimeError(
                    "Batched unit-transport cache fill returned invalid "
                    "physical components."
                )
            for output, values in zip(
                unique_component_values,
                missing_values,
                strict=True,
            ):
                output[missing] = values
        else:
            missing_values = tuple(
                np.zeros(
                    (0, row_count, line_count),
                    dtype=np.float64,
                )
                for _ in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
        # A cache hit normally identifies a source in the currently accepted
        # PF state, whereas a miss is commonly a one-shot rejected proposal.
        # Give hits the newer generation so proposal churn cannot evict the
        # accepted state and force exact old-station transport to be recomputed
        # at every tempering step.
        generation = int(cache["generation"]) + 2
        recency = np.asarray(cache["recency"], dtype=np.int64).copy()
        if np.any(hit):
            recency[safe_lookup[hit]] = generation
        merged_keys = np.concatenate((cached_keys, unique_keys[missing]))
        merged_recency = np.concatenate(
            (
                recency,
                np.full(
                    int(np.count_nonzero(missing)),
                    generation - 1,
                    dtype=np.int64,
                ),
            )
        )
        merged_values = tuple(
            np.concatenate((old, new), axis=0)
            for old, new in zip(
                cached_values,
                missing_values,
                strict=True,
            )
        )
        bytes_per_entry = (
            int(JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE.itemsize)
            + np.dtype(np.int64).itemsize
            + len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES)
            * row_count
            * line_count
            * np.dtype(np.float64).itemsize
        )
        byte_capacity = max(
            1,
            JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES // max(bytes_per_entry, 1),
        )
        active_state_capacity = max(
            1,
            JOINT_STRUCTURAL_UNIT_CACHE_ACTIVE_STATE_MULTIPLIER
            * int(filt.config.num_particles)
            * int(filt.config.hard_max_sources),
        )
        capacity = min(byte_capacity, active_state_capacity)
        if merged_keys.size > capacity:
            retain = np.argsort(
                merged_recency,
                kind="stable",
            )[-capacity:]
            merged_keys = merged_keys[retain]
            merged_recency = merged_recency[retain]
            merged_values = tuple(values[retain] for values in merged_values)
        order = np.argsort(merged_keys, kind="stable")
        cache = {
            "signature": signature,
            "keys": merged_keys[order],
            "recency": merged_recency[order],
            "values": tuple(values[order] for values in merged_values),
            "generation": generation,
            "last_access": 0,
        }
        self._joint_structural_unit_cache_access_generation += 1
        cache["last_access"] = int(self._joint_structural_unit_cache_access_generation)
        isotope_cache[signature] = cache
        self._trim_joint_structural_unit_transport_cache(
            isotope_key,
            protected_signature=signature,
        )
        self.last_joint_structural_unit_cache_hits += int(np.count_nonzero(hit))
        self.last_joint_structural_unit_cache_misses += int(np.count_nonzero(missing))
        return tuple(
            np.transpose(values[inverse], (1, 0, 2))
            for values in unique_component_values
        )

    @staticmethod
    def _joint_structural_unit_cache_shard_bytes(
        cache: Mapping[str, Any],
    ) -> int:
        """Return the exact NumPy storage owned by one transport-cache shard."""
        return int(
            np.asarray(cache["keys"]).nbytes
            + np.asarray(cache["recency"]).nbytes
            + sum(np.asarray(values).nbytes for values in cache["values"])
        )

    def _trim_joint_structural_unit_transport_cache(
        self,
        isotope: str,
        *,
        protected_signature: str,
    ) -> None:
        """Bound all station shards for one isotope with shard-level LRU."""
        (JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES,) = _runtime_facade_constants(
            "JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES",
        )
        isotope_cache = self._joint_structural_unit_transport_cache[str(isotope)]
        total_bytes = sum(
            self._joint_structural_unit_cache_shard_bytes(cache)
            for cache in isotope_cache.values()
        )
        while total_bytes > JOINT_STRUCTURAL_UNIT_CACHE_MAX_BYTES:
            candidates = [
                (int(cache["last_access"]), signature)
                for signature, cache in isotope_cache.items()
                if signature != protected_signature
            ]
            if not candidates:
                break
            _, signature = min(candidates)
            removed = isotope_cache.pop(signature)
            total_bytes -= self._joint_structural_unit_cache_shard_bytes(removed)

    @staticmethod
    def _joint_structural_station_geometry_shards(
        data: StructuralGeometryBatch,
    ) -> tuple[StructuralGeometryBatch, ...]:
        """Split contiguous immutable station rows without changing their order."""
        station_ids = np.asarray(
            data.station_sequence_ids,
            dtype=np.int64,
        ).reshape(-1)
        boundaries = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.flatnonzero(station_ids[1:] != station_ids[:-1]) + 1,
                np.asarray([station_ids.size], dtype=np.int64),
            )
        )
        shards = tuple(
            StructuralGeometryBatch(
                detector_positions=data.detector_positions[start:stop],
                fe_indices=data.fe_indices[start:stop],
                pb_indices=data.pb_indices[start:stop],
                live_times=data.live_times[start:stop],
                station_sequence_ids=data.station_sequence_ids[start:stop],
            )
            for start, stop in zip(
                boundaries[:-1].tolist(),
                boundaries[1:].tolist(),
                strict=True,
            )
        )
        observed_ids = [int(shard.station_sequence_ids[0]) for shard in shards]
        if len(set(observed_ids)) != len(observed_ids):
            raise ValueError(
                "Structural geometry station rows must form one contiguous "
                "block per station sequence."
            )
        return shards

    def _joint_cached_continuous_unit_components(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_s3: NDArray[np.float64],
        chart_ids_s: NDArray[np.int64],
        positive_line_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], ...]:
        """Return exact transport with one fused call for station-cache misses."""
        (
            JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
            JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES,
        ) = _runtime_facade_constants(
            "JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE",
            "JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES",
        )
        positions = np.asarray(positions_s3, dtype=np.float64).reshape(-1, 3)
        raw_chart_ids = np.asarray(chart_ids_s)
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64).reshape(-1)
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        if (
            not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.size != positions.shape[0]
            or positions.shape[0] != chart_ids.size
            or np.any(~np.isfinite(positions))
            or line_indices.size == 0
        ):
            raise ValueError(
                "Fused continuous transport requires aligned finite states "
                "and positive line indices."
            )
        station_shards = self._joint_structural_station_geometry_shards(data)
        if len(station_shards) == 1 or positions.shape[0] == 0:
            return self._joint_cached_continuous_unit_components_shard(
                filt=filt,
                data=station_shards[0],
                positions_s3=positions,
                chart_ids_s=chart_ids,
                positive_line_indices=line_indices,
            )

        request_keys = np.empty(
            positions.shape[0],
            dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
        )
        request_keys["chart"] = chart_ids
        request_keys["x"] = positions[:, 0]
        request_keys["y"] = positions[:, 1]
        request_keys["z"] = positions[:, 2]
        unique_keys, first_indices = np.unique(
            request_keys,
            return_index=True,
        )
        isotope_cache = self._joint_structural_unit_transport_cache.setdefault(
            str(filt.isotope),
            {},
        )
        active_shard_indices: list[int] = []
        union_missing = np.zeros(unique_keys.size, dtype=np.bool_)
        for shard_index, station_data in enumerate(station_shards):
            signature = self._joint_structural_unit_cache_signature(
                filt=filt,
                data=station_data,
                positive_line_indices=line_indices,
            )
            cache = isotope_cache.get(signature)
            if cache is None:
                missing = np.ones(unique_keys.size, dtype=np.bool_)
            else:
                cached_keys = np.asarray(
                    cache["keys"],
                    dtype=JOINT_STRUCTURAL_UNIT_CACHE_KEY_DTYPE,
                )
                lookup = np.searchsorted(cached_keys, unique_keys)
                if cached_keys.size:
                    safe_lookup = np.minimum(lookup, cached_keys.size - 1)
                    hit = (lookup < cached_keys.size) & (
                        cached_keys[safe_lookup] == unique_keys
                    )
                    missing = ~hit
                else:
                    missing = np.ones(unique_keys.size, dtype=np.bool_)
            if np.any(missing):
                active_shard_indices.append(shard_index)
                union_missing |= missing

        prefetched_keys: NDArray[Any] | None = None
        prefetched_by_shard: dict[
            int,
            tuple[NDArray[np.float64], ...],
        ] = {}
        if active_shard_indices:
            active_shards = [station_shards[index] for index in active_shard_indices]
            fused_geometry = StructuralGeometryBatch(
                detector_positions=np.concatenate(
                    [shard.detector_positions for shard in active_shards],
                    axis=0,
                ),
                fe_indices=np.concatenate(
                    [shard.fe_indices for shard in active_shards],
                ),
                pb_indices=np.concatenate(
                    [shard.pb_indices for shard in active_shards],
                ),
                live_times=np.concatenate(
                    [shard.live_times for shard in active_shards],
                ),
                station_sequence_ids=np.concatenate(
                    [shard.station_sequence_ids for shard in active_shards],
                ),
            )
            prefetched_keys = unique_keys[union_missing]
            prefetched_first_indices = first_indices[union_missing]
            evaluated = filt._continuous_rj_line_transport_component_columns(
                fused_geometry,
                positions[prefetched_first_indices],
                line_indices,
                chart_ids=chart_ids[prefetched_first_indices],
            )
            fused_values = tuple(
                np.transpose(
                    np.asarray(getattr(evaluated, name), dtype=np.float64),
                    (1, 0, 2),
                )
                for name in JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES
            )
            row_start = 0
            for shard_index, station_data in zip(
                active_shard_indices,
                active_shards,
                strict=True,
            ):
                row_stop = row_start + int(station_data.row_count)
                prefetched_by_shard[shard_index] = tuple(
                    values[:, row_start:row_stop, :] for values in fused_values
                )
                row_start = row_stop

        active_shard_set = set(active_shard_indices)
        processing_order = [
            index
            for index in range(len(station_shards))
            if index not in active_shard_set
        ] + active_shard_indices
        station_components_by_index: list[tuple[NDArray[np.float64], ...] | None] = [
            None
        ] * len(station_shards)
        for shard_index in processing_order:
            station_components_by_index[shard_index] = (
                self._joint_cached_continuous_unit_components_shard(
                    filt=filt,
                    data=station_shards[shard_index],
                    positions_s3=positions,
                    chart_ids_s=chart_ids,
                    positive_line_indices=line_indices,
                    prefetched_keys=(
                        prefetched_keys if shard_index in prefetched_by_shard else None
                    ),
                    prefetched_values=prefetched_by_shard.get(shard_index),
                )
            )
        if any(values is None for values in station_components_by_index):
            raise RuntimeError("Fused station transport assembly is incomplete.")
        station_components = [
            values for values in station_components_by_index if values is not None
        ]
        return tuple(
            np.concatenate(
                [components[component_index] for components in station_components],
                axis=0,
            )
            for component_index in range(len(JOINT_STRUCTURAL_UNIT_COMPONENT_NAMES))
        )

    @staticmethod
    def _joint_cached_state_match_torch(
        *,
        filt: IsotopeParticleFilter,
        reference: "torch.Tensor",
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Match proposals to immutable sweep-entry transport columns on Torch.

        The accepted numerical state is updated after every MH decision, while
        the transport tensor itself is refreshed only after an isotope sweep.
        Matching against the immutable sweep-entry mirror therefore prevents a
        moved source from being paired with a stale transport column.
        """
        import torch

        state = getattr(filt, "_structural_rj_device_state", None)
        if state is None and hasattr(
            filt,
            "_initialize_continuous_rj_device_state",
        ):
            filt._initialize_continuous_rj_device_state(reference)
            state = filt._structural_rj_device_state
        if state is None:
            packed = filt._packed_continuous_surface_state_arrays()
            state = {
                "cache_positions": torch.as_tensor(
                    packed[0], device=reference.device, dtype=reference.dtype
                ),
                "cache_strengths": torch.as_tensor(
                    packed[1], device=reference.device, dtype=reference.dtype
                ),
                "cache_mask": torch.as_tensor(
                    packed[2], device=reference.device, dtype=torch.bool
                ),
                "cache_chart_ids": torch.as_tensor(
                    packed[3], device=reference.device, dtype=torch.long
                ),
            }
        indices = torch.as_tensor(
            np.asarray(particle_indices, dtype=np.int64),
            device=reference.device,
            dtype=torch.long,
        )
        cached_positions = torch.index_select(
            state["cache_positions"],
            0,
            indices,
        )
        cached_strengths = torch.index_select(
            state["cache_strengths"],
            0,
            indices,
        )
        cached_mask = torch.index_select(
            state["cache_mask"],
            0,
            indices,
        )
        cached_charts = torch.index_select(
            state["cache_chart_ids"],
            0,
            indices,
        )
        positions = torch.as_tensor(
            positions_pks,
            device=reference.device,
            dtype=reference.dtype,
        )
        charts = torch.as_tensor(
            chart_ids_pk,
            device=reference.device,
            dtype=torch.long,
        )
        matches = (
            cached_mask[:, None, :]
            & (charts[:, :, None] == cached_charts[:, None, :])
            & torch.all(
                positions[:, :, None, :] == cached_positions[:, None, :, :],
                dim=3,
            )
        )
        match_counts = torch.sum(matches, dim=2, dtype=torch.long)
        matched = match_counts == 1
        matched_slots = torch.argmax(matches.to(torch.int8), dim=2)
        accepted_strength = torch.gather(
            cached_strengths,
            1,
            matched_slots,
        )
        invalid = torch.stack(
            (
                torch.any(match_counts > 1),
                torch.any(matched & (accepted_strength <= 0.0)),
            )
        ).any()
        if bool(invalid.item()):
            raise RuntimeError(
                "Cached source matches are ambiguous or have invalid strength."
            )
        return matched, matched_slots, accepted_strength

    def _joint_cuda_accepted_unit_cache_entry(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positive_line_indices: NDArray[np.int64],
        reference: "torch.Tensor",
    ) -> dict[str, object]:
        """Return a fixed-capacity cache for accepted sweep-local geometry."""
        import torch

        line_indices = np.ascontiguousarray(
            positive_line_indices,
            dtype=np.int64,
        )
        signature = (
            f"{id(data)}:{str(reference.device)}:"
            f"{hashlib.sha256(line_indices.tobytes()).hexdigest()}"
        )
        key = (str(filt.isotope), signature)
        cache_by_key = getattr(
            self,
            "_joint_cuda_accepted_unit_transport_cache",
            None,
        )
        if cache_by_key is None:
            cache_by_key = {}
            self._joint_cuda_accepted_unit_transport_cache = cache_by_key
        cached = cache_by_key.get(key)
        if cached is not None:
            return cached
        particle_count = int(reference.shape[0])
        view_count = int(data.row_count)
        maximum = int(filt.config.hard_max_sources or 0)
        line_count = int(np.asarray(positive_line_indices).size)
        feature_count = len(tuple(self._full_spectrum_model().transport_feature_order))
        cached = {
            "mask": torch.zeros(
                (particle_count, maximum),
                device=reference.device,
                dtype=torch.bool,
            ),
            "positions": torch.zeros(
                (particle_count, maximum, 3),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "chart_ids": torch.zeros(
                (particle_count, maximum),
                device=reference.device,
                dtype=torch.long,
            ),
            "total": torch.zeros(
                (particle_count, view_count, maximum, line_count),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "uncollided": torch.zeros(
                (particle_count, view_count, maximum, line_count),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "features": torch.zeros(
                (
                    particle_count,
                    view_count,
                    maximum,
                    line_count,
                    feature_count,
                ),
                device=reference.device,
                dtype=reference.dtype,
            ),
            "pending": None,
        }
        cache_by_key[key] = cached
        return cached

    @staticmethod
    def _joint_promote_pending_cuda_unit_transport(
        *,
        filt: IsotopeParticleFilter,
        cache: dict[str, object],
    ) -> None:
        """Commit pending unit columns only when their proposal was accepted."""
        import torch

        pending = cache.get("pending")
        state = getattr(filt, "_structural_rj_device_state", None)
        if not isinstance(pending, dict) or state is None:
            return
        indices = pending["particle_indices"]
        cardinality = int(pending["cardinality"])
        current_cardinality = torch.index_select(
            state["cardinalities"],
            0,
            indices,
        )
        accepted = current_cardinality == cardinality
        if cardinality:
            current_positions = torch.index_select(
                state["positions"],
                0,
                indices,
            )[:, :cardinality]
            current_charts = torch.index_select(
                state["chart_ids"],
                0,
                indices,
            )[:, :cardinality]
            accepted &= torch.all(
                current_positions == pending["positions"],
                dim=(1, 2),
            )
            accepted &= torch.all(
                current_charts == pending["chart_ids"],
                dim=1,
            )
        accepted_indices = indices[accepted]
        cache["mask"][accepted_indices] = False
        cache["total"][accepted_indices] = 0.0
        cache["uncollided"][accepted_indices] = 0.0
        cache["features"][accepted_indices] = 0.0
        if cardinality:
            cache["mask"][accepted_indices, :cardinality] = True
            cache["positions"][accepted_indices, :cardinality] = pending["positions"][
                accepted
            ]
            cache["chart_ids"][accepted_indices, :cardinality] = pending["chart_ids"][
                accepted
            ]
            cache["total"][accepted_indices, :, :cardinality] = pending["total"][
                accepted
            ]
            cache["uncollided"][accepted_indices, :, :cardinality] = pending[
                "uncollided"
            ][accepted]
            cache["features"][accepted_indices, :, :cardinality] = pending["features"][
                accepted
            ]
        cache["pending"] = None

    @staticmethod
    def _joint_match_cuda_accepted_unit_transport(
        *,
        cache: dict[str, object],
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        reference: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", tuple["torch.Tensor", ...]]:
        """Match candidates to accepted sweep-local unit columns on CUDA."""
        import torch

        indices = torch.as_tensor(
            particle_indices,
            device=reference.device,
            dtype=torch.long,
        )
        positions = torch.as_tensor(
            positions_pks,
            device=reference.device,
            dtype=reference.dtype,
        )
        charts = torch.as_tensor(
            chart_ids_pk,
            device=reference.device,
            dtype=torch.long,
        )
        cached_mask = torch.index_select(cache["mask"], 0, indices)
        cached_positions = torch.index_select(cache["positions"], 0, indices)
        cached_charts = torch.index_select(cache["chart_ids"], 0, indices)
        matches = (
            cached_mask[:, None, :]
            & (charts[:, :, None] == cached_charts[:, None, :])
            & torch.all(
                positions[:, :, None, :] == cached_positions[:, None, :, :],
                dim=3,
            )
        )
        matched = torch.any(matches, dim=2)
        matched_slots = torch.argmax(matches.to(torch.int8), dim=2)
        view_count = int(cache["total"].shape[1])
        line_count = int(cache["total"].shape[3])
        feature_count = int(cache["features"].shape[4])
        line_gather = matched_slots[:, None, :, None].expand(
            -1,
            view_count,
            -1,
            line_count,
        )
        feature_gather = line_gather[..., None].expand(
            -1,
            -1,
            -1,
            -1,
            feature_count,
        )
        selected_total = torch.index_select(cache["total"], 0, indices)
        selected_uncollided = torch.index_select(
            cache["uncollided"],
            0,
            indices,
        )
        selected_features = torch.index_select(cache["features"], 0, indices)
        gathered = (
            torch.gather(selected_total, 2, line_gather),
            torch.gather(selected_uncollided, 2, line_gather),
            torch.gather(selected_features, 2, feature_gather),
        )
        return matched, matched_slots, gathered

    @staticmethod
    def _joint_stage_cuda_unit_transport(
        *,
        cache: dict[str, object],
        particle_indices: NDArray[np.int64],
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        unit_total: "torch.Tensor",
        unit_uncollided: "torch.Tensor",
        unit_features: "torch.Tensor",
    ) -> None:
        """Stage exact proposal columns for acceptance detection next call."""
        import torch

        reference = unit_total
        cache["pending"] = {
            "particle_indices": torch.as_tensor(
                particle_indices,
                device=reference.device,
                dtype=torch.long,
            ),
            "cardinality": int(np.asarray(positions_pks).shape[1]),
            "positions": torch.as_tensor(
                positions_pks,
                device=reference.device,
                dtype=reference.dtype,
            ).clone(),
            "chart_ids": torch.as_tensor(
                chart_ids_pk,
                device=reference.device,
                dtype=torch.long,
            ).clone(),
            "total": unit_total.detach().clone(),
            "uncollided": unit_uncollided.detach().clone(),
            "features": unit_features.detach().clone(),
        }

    def _joint_structural_target_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        strengths_pk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
        tempering_start_row: int | None,
    ) -> NDArray[np.float64]:
        """Evaluate a conditional isotope proposal under the full joint target."""
        stations = self._active_joint_station_history
        cache = self._joint_structural_transport_cache
        if stations is None or cache is None:
            raise RuntimeError("Joint structural target is not active.")
        self._validate_joint_structural_geometry(data, stations)
        order = self.joint_isotope_order()
        if filt.isotope not in order:
            raise ValueError("Conditional RJ filter is not a joint isotope.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        positions = np.asarray(positions_pks, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids_pk)
        strengths = np.asarray(strengths_pk, dtype=np.float64)
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            positions.ndim != 3
            or positions.shape[0] != indices.size
            or positions.shape[2] != 3
            or strengths.shape != positions.shape[:2]
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strengths.shape
        ):
            raise ValueError(
                "Conditional isotope candidates must be aligned surface states."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        model = self._full_spectrum_model()
        cached_total, cached_uncollided, cached_features = cache
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.hard_max_sources)
        total_slot_count = slots_per_isotope * len(order)
        if (
            tuple(cached_total.shape[1:]) != (total_views, total_slot_count, line_count)
            or tuple(cached_uncollided.shape) != tuple(cached_total.shape)
            or tuple(cached_features.shape)
            != tuple(cached_total.shape) + (feature_count,)
            or np.any(indices < 0)
            or np.any(indices >= int(cached_total.shape[0]))
        ):
            raise RuntimeError("Joint structural transport cache is misaligned.")
        cache_is_torch = hasattr(cached_total, "detach")
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        local_shape = (
            total_views,
            indices.size,
            int(positions.shape[1]),
            int(local_indices.size),
        )
        isotope_index = order.index(str(filt.isotope))
        slot_start = isotope_index * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        if cache_is_torch:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=cached_total.device,
                dtype=torch.long,
            )
            total = torch.index_select(
                cached_total,
                0,
                index_tensor,
            ).clone()
            uncollided = torch.index_select(
                cached_uncollided,
                0,
                index_tensor,
            ).clone()
            features = torch.index_select(
                cached_features,
                0,
                index_tensor,
            ).clone()
            global_column_selection = torch.as_tensor(
                global_columns,
                device=total.device,
                dtype=torch.long,
            )
            cardinality = int(positions.shape[1])
            matched, matched_slot_tensor, accepted_strength_tensor = (
                self._joint_cached_state_match_torch(
                    filt=filt,
                    reference=cached_total,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                )
            )
            accepted_unit_cache = self._joint_cuda_accepted_unit_cache_entry(
                filt=filt,
                data=data,
                positive_line_indices=local_indices,
                reference=cached_total,
            )
            self._joint_promote_pending_cuda_unit_transport(
                filt=filt,
                cache=accepted_unit_cache,
            )
            accepted_matched, _, accepted_unit_components = (
                self._joint_match_cuda_accepted_unit_transport(
                    cache=accepted_unit_cache,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    reference=cached_total,
                )
            )
            accepted_matched &= ~matched
            local_line_count = int(local_indices.size)
            candidate_total = torch.zeros(
                (
                    indices.size,
                    total_views,
                    cardinality,
                    local_line_count,
                ),
                device=total.device,
                dtype=total.dtype,
            )
            candidate_uncollided = torch.zeros_like(candidate_total)
            candidate_features = torch.zeros(
                tuple(candidate_total.shape) + (feature_count,),
                device=total.device,
                dtype=total.dtype,
            )
            if cardinality:
                accepted_total = torch.index_select(
                    total[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_uncollided = torch.index_select(
                    uncollided[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_features = torch.index_select(
                    features[:, :, slot_start:slot_stop, :, :],
                    3,
                    global_column_selection,
                )
                line_gather = matched_slot_tensor[:, None, :, None].expand(
                    -1,
                    total_views,
                    -1,
                    local_line_count,
                )
                feature_gather = line_gather[..., None].expand(
                    -1,
                    -1,
                    -1,
                    -1,
                    feature_count,
                )
                gathered_total = torch.gather(
                    accepted_total,
                    2,
                    line_gather,
                )
                gathered_uncollided = torch.gather(
                    accepted_uncollided,
                    2,
                    line_gather,
                )
                gathered_features = torch.gather(
                    accepted_features,
                    2,
                    feature_gather,
                )
                proposed_strength_tensor = torch.as_tensor(
                    strengths,
                    device=total.device,
                    dtype=total.dtype,
                )
                ratio_tensor = torch.where(
                    matched,
                    proposed_strength_tensor
                    / torch.where(
                        matched,
                        accepted_strength_tensor,
                        torch.ones_like(accepted_strength_tensor),
                    ),
                    torch.ones_like(proposed_strength_tensor),
                )[:, None, :, None]
                matched_tensor = matched[:, None, :, None]
                candidate_total = torch.where(
                    matched_tensor,
                    gathered_total * ratio_tensor,
                    candidate_total,
                )
                candidate_uncollided = torch.where(
                    matched_tensor,
                    gathered_uncollided * ratio_tensor,
                    candidate_uncollided,
                )
                candidate_features = torch.where(
                    matched_tensor[..., None],
                    gathered_features,
                    candidate_features,
                )
            if cardinality:
                proposed_strength_tensor = torch.as_tensor(
                    strengths,
                    device=total.device,
                    dtype=total.dtype,
                )[:, None, :, None]
                accepted_tensor = accepted_matched[:, None, :, None]
                candidate_total = torch.where(
                    accepted_tensor,
                    accepted_unit_components[0] * proposed_strength_tensor,
                    candidate_total,
                )
                candidate_uncollided = torch.where(
                    accepted_tensor,
                    accepted_unit_components[1] * proposed_strength_tensor,
                    candidate_uncollided,
                )
                candidate_features = torch.where(
                    accepted_tensor[..., None],
                    accepted_unit_components[2],
                    candidate_features,
                )
            all_matched = matched | accepted_matched
            unmatched_index = torch.nonzero(~all_matched, as_tuple=False)
            unmatched_rows = (
                unmatched_index[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            unmatched_slots = (
                unmatched_index[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            if unmatched_rows.size:
                device_components = (
                    filt._continuous_rj_line_transport_component_columns(
                        data,
                        positions[unmatched_rows, unmatched_slots],
                        local_indices,
                        chart_ids=chart_ids[unmatched_rows, unmatched_slots],
                        device_resident=True,
                    )
                )
                if not hasattr(device_components.total_kernel, "detach"):
                    raise RuntimeError(
                        "CUDA structural transport returned host components."
                    )
                branch_tensor = torch.as_tensor(
                    branching_weights,
                    device=total.device,
                    dtype=total.dtype,
                )[None, None, :]
                strength_tensor = torch.as_tensor(
                    strengths[unmatched_rows, unmatched_slots],
                    device=total.device,
                    dtype=total.dtype,
                )[:, None, None]
                unmatched_total = (
                    device_components.total_kernel.permute(1, 0, 2)
                    * strength_tensor
                    * branch_tensor
                )
                unmatched_uncollided = (
                    device_components.uncollided_kernel.permute(1, 0, 2)
                    * strength_tensor
                    * branch_tensor
                )
                unmatched_features = torch.stack(
                    (
                        device_components.tau_fe,
                        device_components.tau_pb,
                        device_components.tau_obstacle,
                        device_components.distance_m,
                    ),
                    dim=-1,
                ).permute(1, 0, 2, 3)
                unmatched_row_tensor = torch.as_tensor(
                    unmatched_rows,
                    device=total.device,
                    dtype=torch.long,
                )
                unmatched_slot_tensor = torch.as_tensor(
                    unmatched_slots,
                    device=total.device,
                    dtype=torch.long,
                )
                candidate_total[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_total
                candidate_uncollided[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_uncollided
                candidate_features[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                    :,
                ] = unmatched_features
            station_count = len(self._joint_structural_station_geometry_shards(data))
            self.last_joint_structural_unit_cache_hits += int(
                torch.count_nonzero(all_matched).item() * station_count
            )
            self.last_joint_structural_unit_cache_misses += int(
                unmatched_rows.size * station_count
            )
            staged_strength = torch.as_tensor(
                strengths,
                device=total.device,
                dtype=total.dtype,
            )[:, None, :, None]
            self._joint_stage_cuda_unit_transport(
                cache=accepted_unit_cache,
                particle_indices=indices,
                positions_pks=positions,
                chart_ids_pk=chart_ids,
                unit_total=candidate_total / staged_strength,
                unit_uncollided=candidate_uncollided / staged_strength,
                unit_features=candidate_features,
            )
        else:
            (
                component_total,
                component_uncollided,
                component_tau_fe,
                component_tau_pb,
                component_tau_obstacle,
                component_distance,
            ) = self._joint_cached_continuous_unit_components(
                filt=filt,
                data=data,
                positions_s3=positions.reshape(-1, 3),
                chart_ids_s=chart_ids.reshape(-1),
                positive_line_indices=local_indices,
            )
            candidate_total_numpy = np.asarray(
                component_total,
                dtype=np.float64,
            ).reshape(local_shape)
            candidate_uncollided_numpy = np.asarray(
                component_uncollided,
                dtype=np.float64,
            ).reshape(local_shape)
            candidate_features_numpy = np.stack(
                (
                    np.asarray(component_tau_fe, dtype=np.float64),
                    np.asarray(component_tau_pb, dtype=np.float64),
                    np.asarray(component_tau_obstacle, dtype=np.float64),
                    np.asarray(component_distance, dtype=np.float64),
                ),
                axis=-1,
            ).reshape(local_shape + (feature_count,))
            total = np.asarray(
                cached_total[indices],
                dtype=np.float64,
            ).copy()
            uncollided = np.asarray(
                cached_uncollided[indices],
                dtype=np.float64,
            ).copy()
            features = np.asarray(
                cached_features[indices],
                dtype=np.float64,
            ).copy()
            scale = strengths[None, :, :, None] * branching_weights.reshape(1, 1, 1, -1)
            candidate_total = np.transpose(
                candidate_total_numpy * scale,
                (1, 0, 2, 3),
            )
            candidate_uncollided = np.transpose(
                candidate_uncollided_numpy * scale,
                (1, 0, 2, 3),
            )
            candidate_features = np.transpose(
                candidate_features_numpy,
                (1, 0, 2, 3, 4),
            )
            global_column_selection = global_columns
        total[:, :, slot_start:slot_stop, :] = 0.0
        uncollided[:, :, slot_start:slot_stop, :] = 0.0
        features[:, :, slot_start:slot_stop, :, :] = 0.0
        cardinality = int(positions.shape[1])
        if cardinality > slots_per_isotope:
            raise ValueError(
                "Conditional candidate cardinality exceeds its source slots."
            )
        if cardinality:
            target_slots = slice(slot_start, slot_start + cardinality)
            total_subset = total[:, :, target_slots, :]
            uncollided_subset = uncollided[:, :, target_slots, :]
            feature_subset = features[:, :, target_slots, :, :]
            total_subset[..., global_column_selection] = candidate_total
            uncollided_subset[..., global_column_selection] = candidate_uncollided
            feature_subset[..., global_column_selection, :] = candidate_features
            total[:, :, target_slots, :] = total_subset
            uncollided[:, :, target_slots, :] = uncollided_subset
            features[:, :, target_slots, :, :] = feature_subset
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint structural target_beta must lie in [0, 1].")
        expected_start_row = total_views - int(stations[-1].fe_indices.size)
        if (
            tempering_start_row is not None
            and int(tempering_start_row) != expected_start_row
        ):
            raise ValueError(
                "Joint structural tempering must begin at the newest station."
            )
        if data.row_count != total_views:
            raise ValueError(
                "Conditional isotope evidence geometry differs from joint history."
            )
        if cache_is_torch:
            result = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=total,
                uncollided_nvsl=uncollided,
                features_nvslf=features,
                target_beta=beta,
                newest_prefix_count=(self._active_joint_tempering_prefix_count),
            )
            return result.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=total,
            uncollided_nvsl=uncollided,
            features_nvslf=features,
            target_beta=beta,
            newest_prefix_count=self._active_joint_tempering_prefix_count,
        )

    def _joint_structural_strength_grid_target_evaluator(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        positions_pks: NDArray[np.float64],
        chart_ids_pk: NDArray[np.int64],
        strengths_pgk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
        tempering_start_row: int | None,
    ) -> NDArray[np.float64]:
        """Evaluate a fixed-geometry strength grid with one transport pass.

        Source transport is independent of source strength. The standard
        conditional proposal therefore computes each particle/source unit
        transport once, broadcasts it over the strength-grid axis, and scores
        the resulting exact joint targets in bounded CPU/GPU batches.
        """
        (JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE,) = (
            _runtime_facade_constants(
                "JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE",
            )
        )
        stations = self._active_joint_station_history
        cache = self._joint_structural_transport_cache
        if stations is None or cache is None:
            raise RuntimeError("Joint structural target is not active.")
        self._validate_joint_structural_geometry(data, stations)
        order = self.joint_isotope_order()
        if filt.isotope not in order:
            raise ValueError("Conditional RJ filter is not a joint isotope.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        positions = np.asarray(positions_pks, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids_pk)
        strengths = np.asarray(strengths_pgk, dtype=np.float64)
        if (
            positions.ndim != 3
            or positions.shape[0] != indices.size
            or positions.shape[2] != 3
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != positions.shape[:2]
            or strengths.ndim != 3
            or strengths.shape[0] != indices.size
            or strengths.shape[2] != positions.shape[1]
            or strengths.shape[1] < 1
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
        ):
            raise ValueError(
                "Conditional strength-grid states must share aligned "
                "particle, grid, and source axes."
            )
        chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        total_views = sum(int(station.fe_indices.size) for station in stations)
        expected_start_row = total_views - int(stations[-1].fe_indices.size)
        if (
            tempering_start_row is not None
            and int(tempering_start_row) != expected_start_row
        ):
            raise ValueError(
                "Joint structural tempering must begin at the newest station."
            )
        if data.row_count != total_views:
            raise ValueError(
                "Conditional isotope evidence geometry differs from joint history."
            )
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint structural target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.hard_max_sources)
        total_slot_count = slots_per_isotope * len(order)
        cached_total, cached_uncollided, cached_features = cache
        if (
            tuple(cached_total.shape[1:]) != (total_views, total_slot_count, line_count)
            or tuple(cached_uncollided.shape) != tuple(cached_total.shape)
            or tuple(cached_features.shape)
            != tuple(cached_total.shape) + (feature_count,)
            or np.any(indices < 0)
            or np.any(indices >= int(cached_total.shape[0]))
        ):
            raise RuntimeError("Joint structural transport cache is misaligned.")
        row_count = int(indices.size)
        grid_count = int(strengths.shape[1])
        if row_count == 0:
            return np.empty((0, grid_count), dtype=np.float64)
        configured_batch_size = int(self.pf_config.joint_strength_block_batch_size)
        batch_size = configured_batch_size
        maximum_batch_size = configured_batch_size
        cache_key: tuple[object, ...] | None = None
        cache_is_cuda = hasattr(cached_total, "detach") and bool(cached_total.is_cuda)
        if cache_is_cuda:
            import torch

            free_bytes, _ = torch.cuda.mem_get_info(cached_total.device)
            bytes_per_value = int(cached_total.element_size())
            expanded_values_per_row = (
                grid_count
                * total_views
                * total_slot_count
                * line_count
                * (2 + feature_count)
            )
            working_bytes_per_row = max(
                1,
                2 * expanded_values_per_row * bytes_per_value,
            )
            memory_budget = min(
                2 * 1024**3,
                max(64 * 1024**2, int(free_bytes) // 4),
            )
            maximum_batch_size = min(
                JOINT_STRENGTH_GRID_AUTOTUNE_MAX_BATCH_SIZE,
                max(1, memory_budget // working_bytes_per_row),
            )
            cache_key = self._joint_strength_grid_autotune_key(
                cache_tensor=cached_total,
                total_views=total_views,
                total_slot_count=total_slot_count,
                line_count=line_count,
                feature_count=feature_count,
                grid_count=grid_count,
                cardinality=int(positions.shape[1]),
            )
            batch_cache = getattr(
                self,
                "_joint_strength_grid_batch_size_cache",
                None,
            )
            if batch_cache is None:
                batch_cache = {}
                self._joint_strength_grid_batch_size_cache = batch_cache
            batch_size = batch_cache.get(
                cache_key,
                min(configured_batch_size, maximum_batch_size),
            )
        result = np.empty((row_count, grid_count), dtype=np.float64)

        def _evaluate(start: int, stop: int) -> None:
            """Evaluate and retain one exact contiguous candidate row slab."""
            result[start:stop] = self._joint_structural_strength_grid_target_batch(
                filt=filt,
                data=data,
                stations=stations,
                positions_bks=positions[start:stop],
                chart_ids_bk=chart_ids[start:stop],
                strengths_bgk=strengths[start:stop],
                particle_indices=indices[start:stop],
                target_beta=beta,
            )

        start = 0
        batch_cache = getattr(
            self,
            "_joint_strength_grid_batch_size_cache",
            {},
        )
        if cache_is_cuda and cache_key not in batch_cache:
            import torch

            trials: list[dict[str, float | int | str]] = []
            candidates = self._joint_strength_grid_autotune_candidates(
                configured_batch_size=configured_batch_size,
                maximum_batch_size=maximum_batch_size,
                row_count=row_count,
            )
            for candidate in candidates:
                if row_count - start < candidate:
                    break
                stop = start + candidate
                trial_start = time.perf_counter()
                try:
                    _evaluate(start, stop)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    trials.append(
                        {
                            "batch_size": int(candidate),
                            "rows_per_second": 0.0,
                            "elapsed_s": float("inf"),
                            "status": "cuda_oom",
                        }
                    )
                    break
                elapsed = time.perf_counter() - trial_start
                trials.append(
                    {
                        "batch_size": int(candidate),
                        "rows_per_second": float(candidate / max(elapsed, 1.0e-12)),
                        "elapsed_s": float(elapsed),
                        "status": "ok",
                    }
                )
                start = stop
            successful = [trial for trial in trials if trial["status"] == "ok"]
            if successful:
                selected_trial = max(
                    successful,
                    key=lambda trial: float(trial["rows_per_second"]),
                )
                batch_size = int(selected_trial["batch_size"])
            else:
                batch_size = max(1, min(configured_batch_size, maximum_batch_size))
            batch_cache[cache_key] = batch_size
            self._joint_strength_grid_batch_size_cache = batch_cache
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "empirical_cuda_autotune",
                "configured_batch_size": configured_batch_size,
                "memory_limited_max_batch_size": maximum_batch_size,
                "selected_batch_size": batch_size,
                "trials": trials,
                "shape_key": cache_key,
            }
            trial_summary = ",".join(
                f"{int(trial['batch_size'])}:{float(trial['rows_per_second']):.3g}row/s"
                for trial in successful
            )
            print(
                "[joint-smc] strength-grid-batch-autotune "
                f"trials={trial_summary or 'none'} "
                f"selected={batch_size}",
                flush=True,
            )
        elif cache_is_cuda:
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "cached_cuda_autotune",
                "configured_batch_size": configured_batch_size,
                "memory_limited_max_batch_size": maximum_batch_size,
                "selected_batch_size": batch_size,
                "shape_key": cache_key,
            }
        else:
            self.last_joint_strength_grid_batch_diagnostics = {
                "mode": "fixed_non_cuda",
                "configured_batch_size": configured_batch_size,
                "selected_batch_size": batch_size,
            }
        for start in range(start, row_count, batch_size):
            stop = min(start + batch_size, row_count)
            _evaluate(start, stop)
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise ValueError("Joint strength-grid target contains invalid values.")
        return result

    @staticmethod
    def _joint_strength_grid_autotune_candidates(
        *,
        configured_batch_size: int,
        maximum_batch_size: int,
        row_count: int,
    ) -> tuple[int, ...]:
        """Return ascending power-of-two batches for one real-work trial."""
        configured = max(1, int(configured_batch_size))
        maximum = max(1, min(int(maximum_batch_size), int(row_count)))
        first = min(configured, maximum)
        candidates = [first]
        while candidates[-1] < maximum:
            candidate = min(maximum, candidates[-1] * 2)
            if candidate == candidates[-1]:
                break
            candidates.append(candidate)
        return tuple(candidates)

    def _joint_strength_grid_autotune_key(
        self,
        *,
        cache_tensor: object,
        total_views: int,
        total_slot_count: int,
        line_count: int,
        feature_count: int,
        grid_count: int,
        cardinality: int,
    ) -> tuple[object, ...]:
        """Return the reusable performance-shape key for CUDA batch tuning."""
        return (
            str(getattr(cache_tensor, "device", "numpy")),
            str(getattr(cache_tensor, "dtype", "float64")),
            int(total_views),
            int(total_slot_count),
            int(line_count),
            int(feature_count),
            int(grid_count),
            int(cardinality),
        )

    def _joint_structural_strength_grid_target_batch(
        self,
        *,
        filt: IsotopeParticleFilter,
        data: StructuralGeometryBatch,
        stations: Sequence[JointStationObservation],
        positions_bks: NDArray[np.float64],
        chart_ids_bk: NDArray[np.int64],
        strengths_bgk: NDArray[np.float64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Score one bounded fixed-geometry strength-grid batch exactly."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError("Joint structural target cache is not active.")
        positions = np.asarray(positions_bks, dtype=np.float64)
        chart_ids = np.asarray(chart_ids_bk, dtype=np.int64)
        strengths = np.asarray(strengths_bgk, dtype=np.float64)
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        row_count, grid_count, cardinality = strengths.shape
        total_views = sum(int(station.fe_indices.size) for station in stations)
        order = self.joint_isotope_order()
        model = self._full_spectrum_model()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        slots_per_isotope = int(filt.config.hard_max_sources)
        isotope_index = order.index(str(filt.isotope))
        slot_start = isotope_index * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        layout = self._joint_line_layout()
        global_columns, local_indices, branching_weights = layout[str(filt.isotope)]
        local_line_count = int(local_indices.size)
        cached_total, cached_uncollided, cached_features = cache
        cache_is_torch = hasattr(cached_total, "detach")
        if cache_is_torch:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=cached_total.device,
                dtype=torch.long,
            )
            selected_total = torch.index_select(
                cached_total,
                0,
                index_tensor,
            )
            selected_uncollided = torch.index_select(
                cached_uncollided,
                0,
                index_tensor,
            )
            selected_features = torch.index_select(
                cached_features,
                0,
                index_tensor,
            )
            global_column_selection = torch.as_tensor(
                global_columns,
                device=cached_total.device,
                dtype=torch.long,
            )
            matched, matched_slot_tensor, accepted_strength_tensor = (
                self._joint_cached_state_match_torch(
                    filt=filt,
                    reference=cached_total,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                )
            )
            accepted_unit_cache = self._joint_cuda_accepted_unit_cache_entry(
                filt=filt,
                data=data,
                positive_line_indices=local_indices,
                reference=cached_total,
            )
            self._joint_promote_pending_cuda_unit_transport(
                filt=filt,
                cache=accepted_unit_cache,
            )
            accepted_matched, _, accepted_unit_components = (
                self._joint_match_cuda_accepted_unit_transport(
                    cache=accepted_unit_cache,
                    particle_indices=indices,
                    positions_pks=positions,
                    chart_ids_pk=chart_ids,
                    reference=cached_total,
                )
            )
            accepted_matched &= ~matched
            unit_total = torch.zeros(
                (
                    row_count,
                    total_views,
                    cardinality,
                    local_line_count,
                ),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            unit_uncollided = torch.zeros_like(unit_total)
            unit_features = torch.zeros(
                tuple(unit_total.shape) + (feature_count,),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            if cardinality:
                accepted_total = torch.index_select(
                    selected_total[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_uncollided = torch.index_select(
                    selected_uncollided[:, :, slot_start:slot_stop, :],
                    3,
                    global_column_selection,
                )
                accepted_features = torch.index_select(
                    selected_features[:, :, slot_start:slot_stop, :, :],
                    3,
                    global_column_selection,
                )
                line_gather = matched_slot_tensor[:, None, :, None].expand(
                    -1,
                    total_views,
                    -1,
                    local_line_count,
                )
                feature_gather = line_gather[..., None].expand(
                    -1,
                    -1,
                    -1,
                    -1,
                    feature_count,
                )
                gathered_total = torch.gather(
                    accepted_total,
                    2,
                    line_gather,
                )
                gathered_uncollided = torch.gather(
                    accepted_uncollided,
                    2,
                    line_gather,
                )
                gathered_features = torch.gather(
                    accepted_features,
                    2,
                    feature_gather,
                )
                safe_strength = torch.where(
                    matched,
                    accepted_strength_tensor,
                    torch.ones_like(accepted_strength_tensor),
                )[:, None, :, None]
                matched_tensor = matched[:, None, :, None]
                unit_total = torch.where(
                    matched_tensor,
                    gathered_total / safe_strength,
                    unit_total,
                )
                unit_uncollided = torch.where(
                    matched_tensor,
                    gathered_uncollided / safe_strength,
                    unit_uncollided,
                )
                unit_features = torch.where(
                    matched_tensor[..., None],
                    gathered_features,
                    unit_features,
                )
            if cardinality:
                accepted_tensor = accepted_matched[:, None, :, None]
                unit_total = torch.where(
                    accepted_tensor,
                    accepted_unit_components[0],
                    unit_total,
                )
                unit_uncollided = torch.where(
                    accepted_tensor,
                    accepted_unit_components[1],
                    unit_uncollided,
                )
                unit_features = torch.where(
                    accepted_tensor[..., None],
                    accepted_unit_components[2],
                    unit_features,
                )
            all_matched = matched | accepted_matched
            unmatched_index = torch.nonzero(~all_matched, as_tuple=False)
            unmatched_rows = (
                unmatched_index[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            unmatched_slots = (
                unmatched_index[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            if unmatched_rows.size:
                device_components = (
                    filt._continuous_rj_line_transport_component_columns(
                        data,
                        positions[unmatched_rows, unmatched_slots],
                        local_indices,
                        chart_ids=chart_ids[unmatched_rows, unmatched_slots],
                        device_resident=True,
                    )
                )
                if not hasattr(device_components.total_kernel, "detach"):
                    raise RuntimeError(
                        "CUDA structural transport returned host components."
                    )
                branch_tensor = torch.as_tensor(
                    branching_weights,
                    device=cached_total.device,
                    dtype=cached_total.dtype,
                )[None, None, :]
                unmatched_total = (
                    device_components.total_kernel.permute(1, 0, 2) * branch_tensor
                )
                unmatched_uncollided = (
                    device_components.uncollided_kernel.permute(1, 0, 2) * branch_tensor
                )
                unmatched_features = torch.stack(
                    (
                        device_components.tau_fe,
                        device_components.tau_pb,
                        device_components.tau_obstacle,
                        device_components.distance_m,
                    ),
                    dim=-1,
                ).permute(1, 0, 2, 3)
                unmatched_row_tensor = torch.as_tensor(
                    unmatched_rows,
                    device=cached_total.device,
                    dtype=torch.long,
                )
                unmatched_slot_tensor = torch.as_tensor(
                    unmatched_slots,
                    device=cached_total.device,
                    dtype=torch.long,
                )
                unit_total[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_total
                unit_uncollided[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                ] = unmatched_uncollided
                unit_features[
                    unmatched_row_tensor,
                    :,
                    unmatched_slot_tensor,
                    :,
                    :,
                ] = unmatched_features
            station_count = len(self._joint_structural_station_geometry_shards(data))
            self.last_joint_structural_unit_cache_hits += int(
                torch.count_nonzero(all_matched).item() * station_count
            )
            self.last_joint_structural_unit_cache_misses += int(
                unmatched_rows.size * station_count
            )
            strength_tensor = torch.as_tensor(
                strengths,
                device=cached_total.device,
                dtype=cached_total.dtype,
            )[:, :, None, :, None]
            column_selection = global_column_selection
        else:
            (
                component_total,
                component_uncollided,
                component_tau_fe,
                component_tau_pb,
                component_tau_obstacle,
                component_distance,
            ) = self._joint_cached_continuous_unit_components(
                filt=filt,
                data=data,
                positions_s3=positions.reshape(-1, 3),
                chart_ids_s=chart_ids.reshape(-1),
                positive_line_indices=local_indices,
            )
            local_shape = (
                total_views,
                row_count,
                cardinality,
                local_line_count,
            )
            unit_total_numpy = np.transpose(
                np.asarray(component_total, dtype=np.float64).reshape(local_shape),
                (1, 0, 2, 3),
            ) * branching_weights.reshape(1, 1, 1, -1)
            unit_uncollided_numpy = np.transpose(
                np.asarray(component_uncollided, dtype=np.float64).reshape(local_shape),
                (1, 0, 2, 3),
            ) * branching_weights.reshape(1, 1, 1, -1)
            unit_features_numpy = np.transpose(
                np.stack(
                    (
                        np.asarray(component_tau_fe, dtype=np.float64),
                        np.asarray(component_tau_pb, dtype=np.float64),
                        np.asarray(component_tau_obstacle, dtype=np.float64),
                        np.asarray(component_distance, dtype=np.float64),
                    ),
                    axis=-1,
                ).reshape(local_shape + (feature_count,)),
                (1, 0, 2, 3, 4),
            )
            selected_total = np.asarray(
                cached_total[indices],
                dtype=np.float64,
            )
            selected_uncollided = np.asarray(
                cached_uncollided[indices],
                dtype=np.float64,
            )
            selected_features = np.asarray(
                cached_features[indices],
                dtype=np.float64,
            )
            candidate_total = (
                unit_total_numpy[:, None] * strengths[:, :, None, :, None]
            ).reshape(
                row_count * grid_count,
                total_views,
                cardinality,
                local_line_count,
            )
            candidate_uncollided = (
                unit_uncollided_numpy[:, None] * strengths[:, :, None, :, None]
            ).reshape(candidate_total.shape)
            candidate_features = np.broadcast_to(
                unit_features_numpy[:, None],
                (
                    row_count,
                    grid_count,
                    total_views,
                    cardinality,
                    local_line_count,
                    feature_count,
                ),
            ).reshape(
                row_count * grid_count,
                total_views,
                cardinality,
                local_line_count,
                feature_count,
            )
            column_selection = global_columns
        if cardinality > slots_per_isotope:
            raise ValueError(
                "Conditional candidate cardinality exceeds its source slots."
            )
        if cache_is_torch:
            base_active = torch.any(
                (selected_total != 0.0) | (selected_uncollided != 0.0),
                dim=(0, 1, 3),
            )
            before_indices = torch.nonzero(
                base_active[:slot_start],
                as_tuple=False,
            ).reshape(-1)
            after_indices = (
                torch.nonzero(
                    base_active[slot_stop:],
                    as_tuple=False,
                ).reshape(-1)
                + slot_stop
            )

            before_count = int(before_indices.numel())
            after_count = int(after_indices.numel())
            compact_slot_count = before_count + cardinality + after_count
            total_view = torch.zeros(
                (
                    row_count,
                    grid_count,
                    total_views,
                    compact_slot_count,
                    line_count,
                ),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )
            uncollided_view = torch.zeros_like(total_view)
            features_view = torch.zeros(
                tuple(total_view.shape) + (feature_count,),
                device=cached_total.device,
                dtype=cached_total.dtype,
            )

            def _copy_base_block(
                destination: object,
                source: object,
                source_indices: object,
                destination_start: int,
            ) -> None:
                """Broadcast immutable accepted slots into one final tensor."""
                count = int(source_indices.numel())
                if count == 0:
                    return
                selected = torch.index_select(source, 2, source_indices)
                destination[
                    :,
                    :,
                    :,
                    destination_start : destination_start + count,
                    ...,
                ] = selected[:, None, ...]

            _copy_base_block(total_view, selected_total, before_indices, 0)
            _copy_base_block(
                uncollided_view,
                selected_uncollided,
                before_indices,
                0,
            )
            _copy_base_block(
                features_view,
                selected_features,
                before_indices,
                0,
            )
            after_start = before_count + cardinality
            _copy_base_block(
                total_view,
                selected_total,
                after_indices,
                after_start,
            )
            _copy_base_block(
                uncollided_view,
                selected_uncollided,
                after_indices,
                after_start,
            )
            _copy_base_block(
                features_view,
                selected_features,
                after_indices,
                after_start,
            )
            if cardinality:
                target_slice = slice(before_count, after_start)
                total_view[..., target_slice, :][..., column_selection] = (
                    unit_total[:, None] * strength_tensor
                )
                uncollided_view[..., target_slice, :][..., column_selection] = (
                    unit_uncollided[:, None] * strength_tensor
                )
                features_view[..., target_slice, :, :][..., column_selection, :] = (
                    unit_features[:, None]
                )
            total = total_view.reshape(
                row_count * grid_count,
                total_views,
                compact_slot_count,
                line_count,
            )
            uncollided = uncollided_view.reshape_as(total)
            features = features_view.reshape(tuple(total.shape) + (feature_count,))
            self.last_joint_strength_grid_source_slots_before = int(
                selected_total.shape[2]
            )
            self.last_joint_strength_grid_source_slots_after = int(total.shape[2])
            target = self._joint_history_log_likelihood_torch(
                filt=filt,
                stations=stations,
                total_nvsl=total,
                uncollided_nvsl=uncollided,
                features_nvslf=features,
                target_beta=float(target_beta),
                newest_prefix_count=(self._active_joint_tempering_prefix_count),
            )
            return (
                target.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
                .reshape(row_count, grid_count)
            )
        base_active = np.any(
            (selected_total != 0.0) | (selected_uncollided != 0.0),
            axis=(0, 1, 3),
        )
        before_indices = np.flatnonzero(base_active[:slot_start])
        after_indices = np.flatnonzero(base_active[slot_stop:]) + slot_stop

        def _expanded_base_numpy(
            values: NDArray[np.float64],
            slot_indices: NDArray[np.int64],
        ) -> NDArray[np.float64]:
            """Select active immutable slots and broadcast the grid axis."""
            selected = values[:, :, slot_indices, ...]
            return np.broadcast_to(
                selected[:, None],
                (row_count, grid_count) + selected.shape[1:],
            ).reshape(row_count * grid_count, *selected.shape[1:])

        target_total_numpy = np.zeros(
            (
                row_count * grid_count,
                total_views,
                cardinality,
                line_count,
            ),
            dtype=np.float64,
        )
        target_uncollided_numpy = np.zeros_like(target_total_numpy)
        target_features_numpy = np.zeros(
            target_total_numpy.shape + (feature_count,),
            dtype=np.float64,
        )
        if cardinality:
            target_total_numpy[..., column_selection] = candidate_total
            target_uncollided_numpy[..., column_selection] = candidate_uncollided
            target_features_numpy[..., column_selection, :] = candidate_features
        total_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_total, before_indices),
                target_total_numpy,
                _expanded_base_numpy(selected_total, after_indices),
            ),
            axis=2,
        )
        uncollided_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_uncollided, before_indices),
                target_uncollided_numpy,
                _expanded_base_numpy(selected_uncollided, after_indices),
            ),
            axis=2,
        )
        features_numpy = np.concatenate(
            (
                _expanded_base_numpy(selected_features, before_indices),
                target_features_numpy,
                _expanded_base_numpy(selected_features, after_indices),
            ),
            axis=2,
        )
        self.last_joint_strength_grid_source_slots_before = int(selected_total.shape[2])
        self.last_joint_strength_grid_source_slots_after = int(total_numpy.shape[2])
        return self._joint_history_log_likelihood_numpy(
            filt=filt,
            stations=stations,
            total_nvsl=total_numpy,
            uncollided_nvsl=uncollided_numpy,
            features_nvslf=features_numpy,
            target_beta=float(target_beta),
            newest_prefix_count=self._active_joint_tempering_prefix_count,
        ).reshape(row_count, grid_count)
