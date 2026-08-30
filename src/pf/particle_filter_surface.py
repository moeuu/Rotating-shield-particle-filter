"""Continuous-surface state and transport-kernel support for the PF."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import MeasurementGeometry, ShieldParams
from measurement.model import EnvironmentConfig
from measurement.shielding import generate_octant_orientations, resolve_mu_values
from measurement.source_boundary import surface_transport_positions
from measurement.surface_atlas import ContinuousSurfaceAtlas
from measurement.surface_charts import (
    build_surface_chart_geometry,
    surface_chart_geometry_sha256,
)
from pf.state import IsotopeState
from pf.structural_rj import (
    BirthDeathMoveProbabilities,
    CardinalityPrior,
    SplitMergeMoveProbabilities,
)


class ParticleSurfaceMixin:
    """Manage continuous-surface state and its immutable physics kernel."""

    def _initialize_structural_rj_surface_support(self) -> None:
        """Build the rectangular atlas used by continuous surface states."""
        chart_geometry = build_surface_chart_geometry(
            self._source_prior_environment(),
            self.obstacle_grid,
            max_edge_m=float(self.config.structural_rj_surface_chart_max_edge_m),
            obstacle_height_m=self.obstacle_height_m,
        )
        if not chart_geometry.obstacle_surfaces_available:
            warning = chart_geometry.obstacle_geometry_warning or (
                "Obstacle component surfaces are unavailable."
            )
            raise ValueError(
                f"rj_mh requires complete obstacle component geometry: {warning}"
            )
        self._structural_rj_surface_atlas = ContinuousSurfaceAtlas(chart_geometry)
        self._structural_rj_surface_atlas_sha256 = surface_chart_geometry_sha256(
            chart_geometry
        )
        max_sources = int(self.config.hard_max_sources or 0)
        self._structural_rj_cardinality_prior = CardinalityPrior(
            self._structural_rj_cardinality_prior_probs
        )
        self._structural_rj_move_probabilities = BirthDeathMoveProbabilities(
            max_cardinality=max_sources,
            birth_weight=float(self.config.structural_rj_birth_probability),
            death_weight=float(self.config.structural_rj_death_probability),
        )
        self._structural_rj_split_merge_probabilities = SplitMergeMoveProbabilities(
            max_cardinality=max_sources,
            split_weight=float(self.config.structural_rj_split_probability),
            merge_weight=float(self.config.structural_rj_merge_probability),
        )

    @property
    def structural_rj_surface_atlas_sha256(self) -> str:
        """Return the immutable continuous-surface atlas contract digest."""
        value = self._structural_rj_surface_atlas_sha256
        if value is None:
            raise RuntimeError("Continuous surface atlas digest is unavailable.")
        return value

    def _surface_coordinates_for_state(
        self,
        state: IsotopeState,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return one validated authoritative chart/UV state."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        source_count = int(state.num_sources)
        strengths = np.asarray(state.strengths, dtype=float).reshape(-1)
        if strengths.size != source_count:
            raise ValueError("Surface state arrays must match num_sources.")
        chart_ids, surface_uv = atlas.validate_coordinates(
            state.surface_chart_ids,
            state.surface_uv,
        )
        if chart_ids.shape != (source_count,):
            raise ValueError("surface_chart_ids must contain one value per source.")
        return chart_ids, surface_uv

    def validate_continuous_surface_states(self) -> None:
        """Fail if any authoritative chart/UV/strength state is invalid.

        Validation never projects, reconstructs, or otherwise repairs state.
        Cartesian positions are absent from state and are derived only after
        this check.
        """
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        if bool(
            getattr(
                self,
                "_structural_rj_device_state_authoritative",
                False,
            )
        ):
            self._validate_authoritative_continuous_surface_device_state()
            return
        states = [particle.state for particle in self.continuous_particles]
        if not states:
            return
        cardinalities = np.asarray(
            [int(state.num_sources) for state in states],
            dtype=np.int64,
        )
        maximum_cardinality = int(self.config.hard_max_sources or 0)
        if np.any(cardinalities < 0) or np.any(cardinalities > maximum_cardinality):
            raise ValueError(
                "PF state cardinalities must lie inside configured support."
            )
        for cardinality in np.unique(cardinalities).tolist():
            indices = np.flatnonzero(cardinalities == int(cardinality))
            selected = [states[int(index)] for index in indices]
            strengths = np.stack(
                [
                    np.asarray(state.strengths, dtype=np.float64).reshape(
                        int(cardinality)
                    )
                    for state in selected
                ],
                axis=0,
            )
            if np.any(
                ~np.asarray(
                    self._strength_prior.in_support(strengths),
                    dtype=bool,
                )
            ):
                raise ValueError(
                    "PF source strengths must lie inside configured prior support."
                )
            chart_ids = np.stack(
                [
                    np.asarray(
                        state.surface_chart_ids,
                        dtype=np.int64,
                    ).reshape(int(cardinality))
                    for state in selected
                ],
                axis=0,
            )
            surface_uv = np.stack(
                [
                    np.asarray(
                        state.surface_uv,
                        dtype=np.float64,
                    ).reshape(int(cardinality), 2)
                    for state in selected
                ],
                axis=0,
            )
            validated_ids, validated_uv = atlas.validate_coordinates(
                chart_ids,
                surface_uv,
            )
            if validated_ids.shape != (
                indices.size,
                int(cardinality),
            ) or validated_uv.shape != (indices.size, int(cardinality), 2):
                raise ValueError(
                    "PF chart/UV arrays do not match their source cardinality."
                )
            if int(cardinality) > 1:
                order = np.lexsort(
                    (
                        validated_uv[:, :, 1],
                        validated_uv[:, :, 0],
                        validated_ids,
                    ),
                    axis=1,
                )
                expected = np.broadcast_to(
                    np.arange(int(cardinality), dtype=np.int64),
                    order.shape,
                )
                if not np.array_equal(order, expected):
                    raise ValueError(
                        "PF source states must remain in canonical chart/UV order."
                    )

    def _validate_authoritative_continuous_surface_device_state(self) -> None:
        """Validate the station-authoritative fixed-capacity Torch state."""
        state = getattr(self, "_structural_rj_device_state", None)
        if state is None:
            raise RuntimeError(
                "Authoritative continuous state is missing its Torch tensors."
            )
        import torch

        particle_count = len(self.continuous_particles)
        slot_count = int(self.config.hard_max_sources or 0)
        expected_shapes = {
            "positions": (particle_count, slot_count, 3),
            "strengths": (particle_count, slot_count),
            "mask": (particle_count, slot_count),
            "chart_ids": (particle_count, slot_count),
            "surface_uv": (particle_count, slot_count, 2),
            "cardinalities": (particle_count,),
        }
        for name, shape in expected_shapes.items():
            value = state.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise RuntimeError(
                    f"Authoritative continuous state tensor {name!r} is invalid."
                )
        mask = state["mask"]
        cardinalities = state["cardinalities"]
        slot_ids = torch.arange(
            slot_count,
            device=mask.device,
            dtype=torch.long,
        )[None, :]
        expected_mask = slot_ids < cardinalities[:, None]
        chart_ids = state["chart_ids"]
        surface_uv = state["surface_uv"]
        strengths = state["strengths"]
        positions = state["positions"]
        chart_count = int(self._structural_rj_surface_atlas.chart_count)
        minimum = float(self._strength_prior.minimum)
        maximum = float(self._strength_prior.support_maximum)
        active_charts = torch.where(mask, chart_ids, torch.zeros_like(chart_ids))
        status = torch.stack(
            (
                torch.all(cardinalities >= 0),
                torch.all(cardinalities <= slot_count),
                torch.all(mask == expected_mask),
                torch.all(~mask | ((chart_ids >= 0) & (chart_ids < chart_count))),
                torch.all(~mask[..., None] | torch.isfinite(surface_uv)),
                torch.all(~mask[..., None] | ((surface_uv >= 0.0) & (surface_uv <= 1.0))),
                torch.all(~mask | torch.isfinite(strengths)),
                torch.all(~mask | ((strengths >= minimum) & (strengths <= maximum))),
                torch.all(~mask[..., None] | torch.isfinite(positions)),
                torch.all(active_charts >= 0),
            )
        )
        if not bool(torch.all(status).item()):
            raise ValueError(
                "Station-authoritative PF surface state violates configured support."
            )

    def continuous_state_positions(
        self,
        state: IsotopeState,
    ) -> NDArray[np.float64]:
        """Derive one state's Cartesian XYZ solely from authoritative chart/UV."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        chart_ids, surface_uv = self._surface_coordinates_for_state(state)
        return np.asarray(
            atlas.positions_xyz(chart_ids, surface_uv),
            dtype=np.float64,
        )

    def _surface_transport_positions(
        self,
        anchors_xyz: NDArray[np.float64],
        *,
        chart_ids: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Map exact PF surface anchors to the shared air-side physics XYZ."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        anchors = np.asarray(anchors_xyz, dtype=np.float64)
        if anchors.shape[-1:] != (3,):
            raise ValueError("Surface anchors must have final dimension three.")
        if chart_ids is None:
            resolved_chart_ids, _ = atlas.locate_positions(anchors)
        else:
            raw_chart_ids = np.asarray(chart_ids)
            if raw_chart_ids.shape != anchors.shape[:-1]:
                raise ValueError("chart_ids must align with surface anchors.")
            if not np.issubdtype(raw_chart_ids.dtype, np.integer):
                raise TypeError("chart_ids must contain integers.")
            resolved_chart_ids = np.asarray(raw_chart_ids, dtype=np.int64)
        normals = atlas.air_facing_normals_xyz(resolved_chart_ids)
        return surface_transport_positions(anchors, normals)

    def _packed_continuous_state_arrays(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """Pack Cartesian states into the configured fixed source-slot layout."""
        positions, strengths, mask, _, _ = (
            self._packed_continuous_surface_state_arrays()
        )
        return positions, strengths, mask

    def _packed_continuous_surface_state_arrays(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        """Pack states and authoritative chart/UV into fixed source slots.

        A fixed ``max_sources`` slot axis is part of the aligned joint-isotope
        transport contract.  Shrinking this axis to the largest currently
        represented cardinality would shift isotope slot boundaries after
        resampling and make conditional RJ overwrite another isotope's
        components.
        """
        if bool(
            getattr(
                self,
                "_structural_rj_device_state_authoritative",
                False,
            )
        ):
            state = getattr(self, "_structural_rj_device_state", None)
            if state is None:
                raise RuntimeError(
                    "Authoritative continuous state is missing its Torch tensors."
                )
            self.validate_continuous_surface_states()
            names = (
                "positions",
                "strengths",
                "mask",
                "chart_ids",
                "surface_uv",
            )
            arrays = tuple(
                state[name].detach().cpu().numpy() for name in names
            )
            diagnostics = self.last_structural_device_diagnostics
            diagnostics["host_snapshot_calls"] = int(
                diagnostics.get("host_snapshot_calls", 0)
            ) + 1
            return (
                np.asarray(arrays[0], dtype=np.float64),
                np.asarray(arrays[1], dtype=np.float64),
                np.asarray(arrays[2], dtype=np.bool_),
                np.asarray(arrays[3], dtype=np.int64),
                np.asarray(arrays[4], dtype=np.float64),
            )
        self.validate_continuous_surface_states()
        states = [particle.state for particle in self.continuous_particles]
        particle_count = len(states)
        slot_count = int(self.config.hard_max_sources or 0)
        chart_ids = np.zeros(
            (particle_count, slot_count),
            dtype=np.int64,
        )
        surface_uv = np.zeros(
            (particle_count, slot_count, 2),
            dtype=np.float64,
        )
        strengths = np.zeros(
            (particle_count, slot_count),
            dtype=np.float64,
        )
        mask = np.zeros(
            (particle_count, slot_count),
            dtype=bool,
        )
        for row, state in enumerate(states):
            cardinality = int(state.num_sources)
            if cardinality == 0:
                continue
            chart_ids[row, :cardinality] = state.surface_chart_ids
            surface_uv[row, :cardinality] = state.surface_uv
            strengths[row, :cardinality] = state.strengths
            mask[row, :cardinality] = True
        positions = np.zeros(
            (particle_count, slot_count, 3),
            dtype=np.float64,
        )
        if np.any(mask):
            positions[mask] = self._structural_rj_surface_atlas.positions_xyz(
                chart_ids[mask],
                surface_uv[mask],
            )
        return positions, strengths, mask, chart_ids, surface_uv

    def structural_surface_chart_coordinates(
        self,
        positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Resolve continuous physical-surface XYZ to chart identifiers and UV."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return atlas.locate_positions(positions)

    def structural_surface_kinds(
        self,
        positions: NDArray[np.float64],
        *,
        strict: bool = True,
    ) -> NDArray[np.object_]:
        """Return authoritative physical-surface kinds for continuous positions."""
        if not bool(strict):
            raise ValueError(
                "Continuous PF surface labels require strict on-surface positions."
            )
        chart_ids, _ = self.structural_surface_chart_coordinates(positions)
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        return np.asarray(atlas.geometry.kinds, dtype=object)[chart_ids]

    def _canonicalize_structural_rj_state(
        self,
        state: IsotopeState,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Sort one state by chart/UV and return its continuous coordinates."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        chart_ids, surface_uv = self._surface_coordinates_for_state(state)
        if chart_ids.size <= 1:
            return chart_ids, surface_uv
        order = atlas.canonical_order(chart_ids, surface_uv)
        if not np.array_equal(order, np.arange(chart_ids.size)):
            state.surface_chart_ids = chart_ids[order]
            state.surface_uv = surface_uv[order]
            state.strengths = np.asarray(state.strengths, dtype=float)[order]
            chart_ids = state.surface_chart_ids
            surface_uv = state.surface_uv
        return (
            np.asarray(chart_ids, dtype=np.int64),
            np.asarray(surface_uv, dtype=np.float64),
        )

    def _build_continuous_kernel(
        self,
        mu_by_isotope: dict[str, object] | None,
        shield_params: ShieldParams,
    ) -> ContinuousKernel:
        """Build the kernel with the filter's environment attenuation settings."""
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
            strict_catalog_line_contract=self.strict_catalog_line_contract,
            dry_air_total_attenuation_contract_id=(
                self.dry_air_total_attenuation_contract_id
            ),
            dry_air_total_attenuation_contract_sha256=(
                self.dry_air_total_attenuation_contract_sha256
            ),
            additive_scatter_response=self.additive_scatter_response,
            **kernel_kwargs,
        )

    def _incoming_kernel_physics_signature(
        self,
        kernel: MeasurementGeometry | None,
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

    def set_kernel(self, kernel: MeasurementGeometry) -> None:
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
        self._structural_rj_position_proposal = None
        self._last_structural_rj_position_proposal = None
        self._structural_rj_strength_proposal = None
        self._last_structural_rj_strength_proposal = None

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


__all__ = ["ParticleSurfaceMixin"]
