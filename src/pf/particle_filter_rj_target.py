"""Batched target evaluation and device-state support for exact-RJ PF."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import LineTransportComponents
from pf.exact_mh import ExactMHDecision, run_exact_mh_acceptance_torch
from pf.particle_types import StructuralGeometryBatch, TorchLineTransportComponents
from pf.state import IsotopeState


class StructuralRJTargetMixin:
    """Evaluate exact-RJ targets and commit accepted batched state rows."""

    def _continuous_rj_cardinalities_numpy(self) -> NDArray[np.int64]:
        """Return current cardinalities without reading stale Python states."""
        state = self._structural_rj_device_state
        if state is not None:
            return (
                state["cardinalities"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
        return np.fromiter(
            (
                int(particle.state.num_sources)
                for particle in self.continuous_particles
            ),
            dtype=np.int64,
            count=len(self.continuous_particles),
        )

    def _continuous_rj_line_transport_component_columns(
        self,
        data: StructuralGeometryBatch,
        positions: NDArray[np.float64],
        positive_line_indices: NDArray[np.int64],
        *,
        chart_ids: NDArray[np.int64] | None = None,
        device_resident: bool = False,
    ) -> LineTransportComponents | TorchLineTransportComponents:
        """Return view-by-source-by-line unit-strength rate components.

        The optional device-resident path changes only where CUDA results are
        stored. It preserves the same continuous transport kernel and row
        ordering while avoiding a GPU-to-host-to-GPU round trip before the
        joint Torch likelihood.
        """
        requested = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        requested_chart_ids: NDArray[np.int64] | None = None
        if chart_ids is not None:
            raw_chart_ids = np.asarray(chart_ids)
            if (
                not np.issubdtype(raw_chart_ids.dtype, np.integer)
                or raw_chart_ids.size != requested.shape[0]
            ):
                raise ValueError(
                    "Continuous line-component chart IDs must be integer and "
                    "align with source positions."
                )
            requested_chart_ids = np.asarray(
                raw_chart_ids,
                dtype=np.int64,
            ).reshape(-1)
        requested_transport = self._surface_transport_positions(
            requested,
            chart_ids=requested_chart_ids,
        )
        impact_edges = self.detector_impact_parameter_edges_fraction
        impact_phase_count = (
            0 if impact_edges is None else int(np.asarray(impact_edges).size - 1)
        )
        measurement_count = data.row_count
        line_indices = np.asarray(
            positive_line_indices,
            dtype=np.int64,
        ).reshape(-1)
        detector_positions = np.asarray(
            data.detector_positions,
            dtype=np.float64,
        )
        fe_indices = np.asarray(data.fe_indices, dtype=np.int64).reshape(-1)
        pb_indices = np.asarray(data.pb_indices, dtype=np.int64).reshape(-1)
        live_times = np.asarray(data.live_times, dtype=np.float64).reshape(-1)
        sequence_ids = np.asarray(
            data.station_sequence_ids,
            dtype=np.int64,
        ).reshape(-1)
        if (
            detector_positions.shape != (measurement_count, 3)
            or fe_indices.size != measurement_count
            or pb_indices.size != measurement_count
            or live_times.size != measurement_count
            or sequence_ids.size != measurement_count
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
        ):
            raise ValueError(
                "Line-component measurement geometry and live times are invalid."
            )
        if self._can_use_gpu():
            orientation_count = int(len(self.continuous_kernel.orientations))
            if (
                orientation_count <= 0
                or np.any(fe_indices < 0)
                or np.any(fe_indices >= orientation_count)
                or np.any(pb_indices < 0)
                or np.any(pb_indices >= orientation_count)
            ):
                raise ValueError(
                    "Continuous RJ shield indices lie outside the orientation support."
                )
            pair_indices = (fe_indices * orientation_count + pb_indices).astype(
                np.int64, copy=False
            )
            unique_sequences, sequence_inverse = np.unique(
                sequence_ids,
                return_inverse=True,
            )
            station_rows = [
                np.flatnonzero(sequence_inverse == station_index)
                for station_index in range(unique_sequences.size)
            ]
            view_counts = np.asarray(
                [rows.size for rows in station_rows],
                dtype=np.int64,
            )
            if np.any(view_counts <= 0):
                raise RuntimeError(
                    "Continuous RJ station grouping produced an empty program."
                )
            if np.unique(view_counts).size == 1:
                for rows in station_rows:
                    if not np.all(
                        detector_positions[rows] == detector_positions[int(rows[0])]
                    ):
                        raise ValueError(
                            "One station sequence contains multiple detector positions."
                        )
                station_detectors = np.stack(
                    [detector_positions[int(rows[0])] for rows in station_rows],
                    axis=0,
                )
                fe_program = np.stack(
                    [fe_indices[rows] for rows in station_rows],
                    axis=0,
                )
                pb_program = np.stack(
                    [pb_indices[rows] for rows in station_rows],
                    axis=0,
                )
                program_components = (
                    self.continuous_kernel
                    .line_transport_components_pair_program_for_detectors(
                        isotope=self.isotope,
                        detector_positions=station_detectors,
                        sources=requested_transport,
                        fe_indices=fe_program,
                        pb_indices=pb_program,
                        positive_line_indices=line_indices,
                        device_resident=device_resident,
                        impact_parameter_edges_fraction=impact_edges,
                    )
                )
                row_view_indices = np.empty(
                    measurement_count,
                    dtype=np.int64,
                )
                for rows in station_rows:
                    row_view_indices[rows] = np.arange(
                        rows.size,
                        dtype=np.int64,
                    )
                expected_program_shape = (
                    int(unique_sequences.size),
                    int(view_counts[0]),
                    int(requested_transport.shape[0]),
                    int(line_indices.size),
                )
                expected_impact_shape = expected_program_shape + (
                    impact_phase_count,
                )
                expected_selected_impact_shape = (
                    measurement_count,
                    int(requested_transport.shape[0]),
                    int(line_indices.size),
                    impact_phase_count,
                )

                if device_resident:
                    import torch

                    sequence_index = torch.as_tensor(
                        sequence_inverse,
                        device=program_components.total_kernel.device,
                        dtype=torch.long,
                    )
                    view_index = torch.as_tensor(
                        row_view_indices,
                        device=program_components.total_kernel.device,
                        dtype=torch.long,
                    )

                    def _selected_device_component(
                        field_name: str,
                    ) -> "torch.Tensor":
                        """Restore history rows without leaving the GPU."""
                        values = getattr(program_components, field_name)
                        if tuple(values.shape) != expected_program_shape:
                            raise RuntimeError(
                                "Pair-program continuous RJ component shape is invalid."
                            )
                        return values[sequence_index, view_index]

                    def _selected_device_impact() -> "torch.Tensor":
                        """Restore phase-resolved rows without leaving the GPU."""
                        values = program_components.uncollided_impact_fractions
                        if tuple(values.shape) != expected_impact_shape:
                            raise RuntimeError(
                                "Pair-program detector-impact shape is invalid."
                            )
                        return values[sequence_index, view_index]

                    components = TorchLineTransportComponents(
                        total_kernel=_selected_device_component("total_kernel"),
                        uncollided_kernel=_selected_device_component(
                            "uncollided_kernel"
                        ),
                        tau_fe=_selected_device_component("tau_fe"),
                        tau_pb=_selected_device_component("tau_pb"),
                        tau_obstacle=_selected_device_component("tau_obstacle"),
                        tau_obstacle_compton=_selected_device_component(
                            "tau_obstacle_compton"
                        ),
                        distance_m=_selected_device_component("distance_m"),
                        uncollided_impact_fractions=_selected_device_impact(),
                    )
                else:

                    def _selected_component(
                        field_name: str,
                    ) -> NDArray[np.float64]:
                        """Restore rows from one batched station program."""
                        values = np.asarray(
                            getattr(program_components, field_name),
                            dtype=np.float64,
                        )
                        if values.shape != expected_program_shape:
                            raise RuntimeError(
                                "Pair-program continuous RJ component shape is invalid."
                            )
                        return np.asarray(
                            values[sequence_inverse, row_view_indices],
                            dtype=np.float64,
                        )

                    components = LineTransportComponents(
                        total_kernel=_selected_component("total_kernel"),
                        unattenuated_kernel=_selected_component("unattenuated_kernel"),
                        uncollided_kernel=_selected_component("uncollided_kernel"),
                        tau_fe=_selected_component("tau_fe"),
                        tau_pb=_selected_component("tau_pb"),
                        tau_obstacle=_selected_component("tau_obstacle"),
                        tau_obstacle_compton=_selected_component(
                            "tau_obstacle_compton"
                        ),
                        distance_m=_selected_component("distance_m"),
                        uncollided_impact_fractions=np.asarray(
                            program_components.uncollided_impact_fractions[
                                sequence_inverse,
                                row_view_indices,
                            ],
                            dtype=np.float64,
                        ),
                    )
                    if (
                        components.uncollided_impact_fractions.shape
                        != expected_selected_impact_shape
                    ):
                        raise RuntimeError(
                            "Pair-program detector-impact component shape is invalid."
                        )
            else:
                if device_resident:
                    raise RuntimeError(
                        "Device-resident structural transport requires one "
                        "common shield-view count across stations."
                    )
                unique_detectors, detector_inverse = np.unique(
                    detector_positions,
                    axis=0,
                    return_inverse=True,
                )
                all_pair_components = (
                    self.continuous_kernel
                    .line_transport_components_all_pairs_for_detectors(
                        isotope=self.isotope,
                        detector_positions=unique_detectors,
                        sources=requested_transport,
                        positive_line_indices=line_indices,
                        impact_parameter_edges_fraction=impact_edges,
                    )
                )
                expected_all_pair_shape = (
                    int(unique_detectors.shape[0]),
                    orientation_count**2,
                    int(requested_transport.shape[0]),
                    int(line_indices.size),
                )

                def _selected_component(
                    field_name: str,
                ) -> NDArray[np.float64]:
                    """Select requested rows from all-pair GPU components."""
                    values = np.asarray(
                        getattr(all_pair_components, field_name),
                        dtype=np.float64,
                    )
                    if values.shape != expected_all_pair_shape:
                        raise RuntimeError(
                            "All-pair continuous RJ component shape is invalid."
                        )
                    return np.asarray(
                        values[detector_inverse, pair_indices],
                        dtype=np.float64,
                    )

                components = LineTransportComponents(
                    total_kernel=_selected_component("total_kernel"),
                    unattenuated_kernel=_selected_component("unattenuated_kernel"),
                    uncollided_kernel=_selected_component("uncollided_kernel"),
                    tau_fe=_selected_component("tau_fe"),
                    tau_pb=_selected_component("tau_pb"),
                    tau_obstacle=_selected_component("tau_obstacle"),
                    tau_obstacle_compton=_selected_component("tau_obstacle_compton"),
                    distance_m=_selected_component("distance_m"),
                    uncollided_impact_fractions=np.asarray(
                        all_pair_components.uncollided_impact_fractions[
                            detector_inverse,
                            pair_indices,
                        ],
                        dtype=np.float64,
                    ),
                )
        else:
            if device_resident:
                raise ValueError("Device-resident structural transport requires CUDA.")
            components = (
                self.continuous_kernel
                .line_transport_components_selected_pairs_for_detectors(
                    isotope=self.isotope,
                    detector_positions=detector_positions,
                    sources=requested_transport,
                    fe_indices=fe_indices,
                    pb_indices=pb_indices,
                    positive_line_indices=line_indices,
                    impact_parameter_edges_fraction=impact_edges,
                )
            )
        if isinstance(components, TorchLineTransportComponents):
            return components
        return LineTransportComponents(
            total_kernel=np.asarray(
                components.total_kernel,
                dtype=np.float64,
            ),
            unattenuated_kernel=np.asarray(
                components.unattenuated_kernel,
                dtype=np.float64,
            ),
            uncollided_kernel=np.asarray(
                components.uncollided_kernel,
                dtype=np.float64,
            ),
            tau_fe=np.asarray(components.tau_fe, dtype=np.float64),
            tau_pb=np.asarray(components.tau_pb, dtype=np.float64),
            tau_obstacle=np.asarray(
                components.tau_obstacle,
                dtype=np.float64,
            ),
            tau_obstacle_compton=np.asarray(
                components.tau_obstacle_compton,
                dtype=np.float64,
            ),
            distance_m=np.asarray(components.distance_m, dtype=np.float64),
            uncollided_impact_fractions=np.asarray(
                components.uncollided_impact_fractions,
                dtype=np.float64,
            ),
        )

    def _continuous_rj_group_arrays(
        self,
        particle_indices: NDArray[np.int64],
        cardinality: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return chart, UV, derived XYZ, and strength arrays for one K."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        source_count = int(cardinality)
        device_state = self._structural_rj_device_state
        if device_state is not None:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=device_state["strengths"].device,
                dtype=torch.long,
            )
            cardinalities = (
                torch.index_select(
                    device_state["cardinalities"],
                    0,
                    index_tensor,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
        else:
            states = [
                self.continuous_particles[int(particle_index)].state
                for particle_index in indices
            ]
            cardinalities = np.fromiter(
                (int(state.num_sources) for state in states),
                dtype=np.int64,
                count=indices.size,
            )
        if np.any(cardinalities != source_count):
            raise ValueError("Continuous RJ group mixes cardinalities.")
        if indices.size == 0 or source_count == 0:
            return (
                np.zeros((indices.size, source_count), dtype=np.int64),
                np.zeros((indices.size, source_count, 2), dtype=np.float64),
                np.zeros((indices.size, source_count, 3), dtype=np.float64),
                np.zeros((indices.size, source_count), dtype=np.float64),
            )
        if device_state is not None:
            selected = {
                name: (
                    torch.index_select(values, 0, index_tensor).detach().cpu().numpy()
                )
                for name, values in (
                    ("chart_ids", device_state["chart_ids"]),
                    ("surface_uv", device_state["surface_uv"]),
                    ("positions", device_state["positions"]),
                    ("strengths", device_state["strengths"]),
                )
            }
            charts = np.asarray(
                selected["chart_ids"][:, :source_count],
                dtype=np.int64,
            )
            surface_uv = np.asarray(
                selected["surface_uv"][:, :source_count],
                dtype=np.float64,
            )
            positions = np.asarray(
                selected["positions"][:, :source_count],
                dtype=np.float64,
            )
            strengths = np.asarray(
                selected["strengths"][:, :source_count],
                dtype=np.float64,
            )
            diagnostics = self.last_structural_device_diagnostics
            diagnostics["group_gather_calls"] = (
                int(diagnostics.get("group_gather_calls", 0)) + 1
            )
        else:
            charts = np.stack(
                [
                    np.asarray(state.surface_chart_ids, dtype=np.int64).reshape(
                        source_count
                    )
                    for state in states
                ],
                axis=0,
            )
            surface_uv = np.stack(
                [
                    np.asarray(state.surface_uv, dtype=np.float64).reshape(
                        source_count,
                        2,
                    )
                    for state in states
                ],
                axis=0,
            )
            strengths = np.stack(
                [
                    np.asarray(state.strengths, dtype=np.float64).reshape(source_count)
                    for state in states
                ],
                axis=0,
            )
            positions = self._structural_rj_surface_atlas.positions_xyz(
                charts,
                surface_uv,
            )
        canonical = self._continuous_rj_canonicalize_rows(
            charts,
            surface_uv,
            positions,
            strengths,
        )
        if not all(
            np.array_equal(actual, expected)
            for actual, expected in zip(
                (charts, surface_uv, positions, strengths),
                canonical,
            )
        ):
            raise RuntimeError("Continuous RJ state sources must already be canonical.")
        if not np.all(self._strength_prior.in_support(strengths)):
            raise ValueError("Continuous RJ strength lies outside its prior.")
        return (
            charts.astype(np.int64, copy=False),
            surface_uv.astype(np.float64, copy=False),
            positions.astype(np.float64, copy=False),
            strengths.astype(np.float64, copy=False),
        )

    def _continuous_rj_torch_enabled(self) -> bool:
        """Return whether the station-authoritative CUDA RJ path is active."""
        state = self._structural_rj_device_state
        return bool(
            self._structural_rj_device_state_authoritative
            and state is not None
            and bool(state["strengths"].is_cuda)
        )

    def _continuous_rj_torch_generator_required(self) -> object:
        """Return the sweep-local CUDA generator or fail closed."""
        generator = self._structural_rj_torch_generator
        if generator is None or not self._continuous_rj_torch_enabled():
            raise RuntimeError(
                "A station-authoritative CUDA RJ sweep has no Torch generator."
            )
        return generator

    def _continuous_rj_atlas_tensors(self) -> dict[str, object]:
        """Return immutable atlas geometry tensors on the RJ state device."""
        state = self._structural_rj_device_state
        atlas = self._structural_rj_surface_atlas
        if state is None or atlas is None:
            raise RuntimeError("CUDA RJ atlas tensors require active surface state.")
        import torch

        reference = state["strengths"]
        cached = self._structural_rj_device_constants
        cache_valid = bool(cached) and all(
            value.device == reference.device and value.dtype == reference.dtype
            for name, value in cached.items()
            if name not in {"chart_ids", "portal_neighbor_ids"}
        )
        if cache_valid:
            return cached
        vertices = torch.tensor(
            np.asarray(atlas.geometry.vertices_xyz, dtype=np.float64),
            device=reference.device,
            dtype=reference.dtype,
        )
        cached = {
            "origins": vertices[:, 0],
            "u_edges": vertices[:, 1] - vertices[:, 0],
            "v_edges": vertices[:, 3] - vertices[:, 0],
            "chart_probabilities": torch.tensor(
                atlas.chart_probabilities,
                device=reference.device,
                dtype=reference.dtype,
            ),
            "log_chart_probabilities": torch.tensor(
                atlas.log_chart_probabilities,
                device=reference.device,
                dtype=reference.dtype,
            ),
        }
        self._structural_rj_device_constants = cached
        return cached

    def _continuous_rj_positions_torch(
        self,
        chart_ids: object,
        surface_uv: object,
    ) -> object:
        """Map device chart/UV coordinates to XYZ without a host round trip."""
        import torch

        if not torch.is_tensor(chart_ids) or not torch.is_tensor(surface_uv):
            raise TypeError("CUDA surface coordinates must be Torch tensors.")
        constants = self._continuous_rj_atlas_tensors()
        flat_ids = chart_ids.reshape(-1)
        flat_uv = surface_uv.reshape(-1, 2)
        positions = (
            constants["origins"][flat_ids]
            + flat_uv[:, :1] * constants["u_edges"][flat_ids]
            + flat_uv[:, 1:] * constants["v_edges"][flat_ids]
        )
        return positions.reshape(tuple(chart_ids.shape) + (3,))

    def _continuous_rj_group_tensors(
        self,
        particle_indices: object,
        cardinality: int,
    ) -> tuple[object, object, object, object, object]:
        """Gather one equal-cardinality group from authoritative CUDA state."""
        import torch

        state = self._structural_rj_device_state
        if state is None or not self._continuous_rj_torch_enabled():
            raise RuntimeError("CUDA RJ group gather requires authoritative state.")
        indices = torch.as_tensor(
            particle_indices,
            device=state["strengths"].device,
            dtype=torch.long,
        ).reshape(-1)
        source_count = int(cardinality)
        selected_cardinalities = torch.index_select(
            state["cardinalities"],
            0,
            indices,
        )
        if bool(torch.any(selected_cardinalities != source_count).item()):
            raise ValueError("CUDA continuous RJ group mixes cardinalities.")
        charts = torch.index_select(state["chart_ids"], 0, indices)[
            :, :source_count
        ]
        uv = torch.index_select(state["surface_uv"], 0, indices)[
            :, :source_count
        ]
        positions = torch.index_select(state["positions"], 0, indices)[
            :, :source_count
        ]
        strengths = torch.index_select(state["strengths"], 0, indices)[
            :, :source_count
        ]
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["device_group_gather_calls"] = int(
            diagnostics.get("device_group_gather_calls", 0)
        ) + 1
        return indices, charts, uv, positions, strengths

    def _continuous_rj_canonicalize_tensors(
        self,
        chart_ids: object,
        surface_uv: object,
        positions: object,
        strengths: object,
    ) -> tuple[object, object, object, object]:
        """Canonicalize device source rows by chart, U, and V."""
        import torch

        if not all(
            torch.is_tensor(value)
            for value in (chart_ids, surface_uv, positions, strengths)
        ):
            raise TypeError("CUDA RJ canonical state values must be tensors.")
        if (
            chart_ids.ndim != 2
            or surface_uv.shape != tuple(chart_ids.shape) + (2,)
            or positions.shape != tuple(chart_ids.shape) + (3,)
            or strengths.shape != chart_ids.shape
        ):
            raise ValueError("CUDA RJ canonical arrays have invalid shapes.")
        derived = self._continuous_rj_positions_torch(chart_ids, surface_uv)
        if bool(
            torch.any(
                ~torch.isclose(
                    positions,
                    derived,
                    rtol=0.0,
                    atol=1.0e-10,
                )
            ).item()
        ):
            raise ValueError(
                "CUDA transient RJ XYZ must equal the chart/UV image."
            )
        if int(chart_ids.shape[1]) <= 1:
            return chart_ids, surface_uv, derived, strengths
        order = torch.argsort(surface_uv[:, :, 1], dim=1, stable=True)
        ordered_u = torch.gather(surface_uv[:, :, 0], 1, order)
        next_order = torch.argsort(ordered_u, dim=1, stable=True)
        order = torch.gather(order, 1, next_order)
        ordered_charts = torch.gather(chart_ids, 1, order)
        next_order = torch.argsort(ordered_charts, dim=1, stable=True)
        order = torch.gather(order, 1, next_order)
        return (
            torch.gather(chart_ids, 1, order),
            torch.gather(surface_uv, 1, order[..., None].expand(-1, -1, 2)),
            torch.gather(derived, 1, order[..., None].expand(-1, -1, 3)),
            torch.gather(strengths, 1, order),
        )

    def _continuous_rj_strength_support_torch(self, values: object) -> object:
        """Return the configured strength-prior support on Torch."""
        import torch

        if not torch.is_tensor(values):
            raise TypeError("CUDA RJ strengths must be a Torch tensor.")
        support = torch.isfinite(values) & (values >= self._strength_prior.minimum)
        maximum = float(self._strength_prior.support_maximum)
        if np.isfinite(maximum):
            support &= values <= maximum
        return support

    def _continuous_rj_strength_log_prior_torch(self, values: object) -> object:
        """Evaluate the normalized physical strength prior on Torch."""
        import torch

        if not torch.is_tensor(values):
            raise TypeError("CUDA RJ strengths must be a Torch tensor.")
        support = self._continuous_rj_strength_support_torch(values)
        if self._strength_prior.family == "bounded_uniform":
            density = -float(
                np.log(
                    self._strength_prior.maximum
                    - self._strength_prior.minimum
                )
            )
            return torch.where(
                support,
                torch.full_like(values, density),
                torch.full_like(values, float("-inf")),
            )
        shifted = values - self._strength_prior.minimum
        safe = torch.clamp(shifted, min=torch.finfo(values.dtype).tiny)
        log_density = (
            (self._strength_prior.gamma_shape - 1.0) * torch.log(safe)
            - shifted / self._strength_prior.gamma_scale
            - torch.lgamma(
                torch.as_tensor(
                    self._strength_prior.gamma_shape,
                    device=values.device,
                    dtype=values.dtype,
                )
            )
            - self._strength_prior.gamma_shape
            * np.log(self._strength_prior.gamma_scale)
        )
        positive = support & (shifted > 0.0)
        boundary_density = (
            -np.log(self._strength_prior.gamma_scale)
            if self._strength_prior.gamma_shape == 1.0
            else float("-inf")
        )
        result = torch.where(
            positive,
            log_density,
            torch.full_like(values, float("-inf")),
        )
        return torch.where(
            support & (shifted == 0.0),
            torch.full_like(values, boundary_density),
            result,
        )

    def _continuous_rj_sample_strength_prior_torch(
        self,
        shape: tuple[int, ...],
        *,
        generator: object | None = None,
    ) -> object:
        """Draw physical-prior strengths with the sweep-local CUDA generator."""
        import torch

        state = self._structural_rj_device_state
        if state is None:
            raise RuntimeError("CUDA strength sampling requires device state.")
        reference = state["strengths"]
        active_generator = (
            self._continuous_rj_torch_generator_required()
            if generator is None
            else generator
        )
        if not isinstance(active_generator, torch.Generator):
            raise TypeError("CUDA strength sampling requires a Torch generator.")
        if self._strength_prior.family == "bounded_uniform":
            unit = torch.rand(
                shape,
                device=reference.device,
                dtype=reference.dtype,
                generator=active_generator,
            )
            return self._strength_prior.minimum + unit * (
                self._strength_prior.maximum - self._strength_prior.minimum
            )
        concentration = torch.full(
            shape,
            self._strength_prior.gamma_shape,
            device=reference.device,
            dtype=reference.dtype,
        )
        gamma = torch._standard_gamma(
            concentration,
            generator=active_generator,
        )
        return (
            self._strength_prior.minimum
            + self._strength_prior.gamma_scale * gamma
        )

    def _continuous_rj_sample_surface_torch(
        self,
        sample_count: int,
        *,
        chart_probabilities: object | None = None,
    ) -> tuple[object, object, object]:
        """Draw chart IDs and UV values entirely on the RJ CUDA device."""
        import torch

        constants = self._continuous_rj_atlas_tensors()
        probabilities = (
            constants["chart_probabilities"]
            if chart_probabilities is None
            else torch.as_tensor(
                chart_probabilities,
                device=constants["chart_probabilities"].device,
                dtype=constants["chart_probabilities"].dtype,
            )
        )
        count = int(sample_count)
        if count < 0:
            raise ValueError("CUDA surface sample_count must be non-negative.")
        generator = self._continuous_rj_torch_generator_required()
        if count == 0:
            ids = torch.zeros(
                0,
                device=probabilities.device,
                dtype=torch.long,
            )
            uv = torch.zeros(
                (0, 2),
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            return ids, uv, self._continuous_rj_positions_torch(ids, uv)
        ids = torch.multinomial(
            probabilities,
            count,
            replacement=True,
            generator=generator,
        )
        uv = torch.rand(
            (count, 2),
            device=probabilities.device,
            dtype=probabilities.dtype,
            generator=generator,
        )
        return ids, uv, self._continuous_rj_positions_torch(ids, uv)

    def _continuous_rj_position_proposal_log_density_torch(
        self,
        chart_ids: object,
    ) -> object:
        """Evaluate the active full-support chart proposal on CUDA."""
        import torch

        if not torch.is_tensor(chart_ids):
            raise TypeError("CUDA proposal chart IDs must be a tensor.")
        proposal = self._active_continuous_rj_position_proposal()
        log_probabilities = torch.tensor(
            proposal.log_chart_probabilities,
            device=chart_ids.device,
            dtype=self._structural_rj_device_state["strengths"].dtype,
        )
        return log_probabilities[chart_ids]

    def _continuous_rj_strength_proposal_log_density_torch(
        self,
        chart_ids: object,
        strengths: object,
    ) -> object:
        """Evaluate the active chart-conditional strength proposal on CUDA."""
        import torch

        if not torch.is_tensor(chart_ids) or not torch.is_tensor(strengths):
            raise TypeError("CUDA strength proposal inputs must be tensors.")
        proposal = self._active_continuous_rj_strength_proposal()
        prior_log_density = self._continuous_rj_strength_log_prior_torch(strengths)
        support = self._continuous_rj_strength_support_torch(strengths)
        if (
            not proposal.data_informative
            or proposal.prior_component_probability >= 1.0
        ):
            return torch.where(
                support,
                prior_log_density,
                torch.full_like(strengths, float("-inf")),
            )
        locations_table = torch.tensor(
            proposal.data_locations_by_chart,
            device=strengths.device,
            dtype=strengths.dtype,
        )
        locations = locations_table[chart_ids]
        lower_z = (proposal.minimum - locations) / proposal.data_sigma
        if proposal.prior_family == "bounded_uniform":
            upper_z = (proposal.maximum - locations) / proposal.data_sigma
            upper_cdf = torch.special.ndtr(upper_z)
        else:
            upper_cdf = torch.ones_like(locations)
        lower_cdf = torch.special.ndtr(lower_z)
        normalization = upper_cdf - lower_cdf
        standardized = (strengths - locations) / proposal.data_sigma
        data_log_density = (
            -0.5 * standardized.square()
            - np.log(np.sqrt(2.0 * np.pi) * proposal.data_sigma)
            - torch.log(normalization)
        )
        mixture = torch.logaddexp(
            np.log(proposal.prior_component_probability) + prior_log_density,
            np.log1p(-proposal.prior_component_probability) + data_log_density,
        )
        return torch.where(
            support,
            mixture,
            torch.full_like(strengths, float("-inf")),
        )

    def _continuous_rj_sample_strength_proposal_torch(
        self,
        chart_ids: object,
    ) -> object:
        """Draw the active chart-conditional strength mixture on CUDA."""
        import torch

        if not torch.is_tensor(chart_ids):
            raise TypeError("CUDA strength-proposal chart IDs must be a tensor.")
        proposal = self._active_continuous_rj_strength_proposal()
        result = self._continuous_rj_sample_strength_prior_torch(
            tuple(chart_ids.shape)
        )
        if (
            not proposal.data_informative
            or proposal.prior_component_probability >= 1.0
        ):
            return result
        generator = self._continuous_rj_torch_generator_required()
        use_data = torch.rand(
            chart_ids.shape,
            device=chart_ids.device,
            dtype=result.dtype,
            generator=generator,
        ) >= proposal.prior_component_probability
        locations = torch.tensor(
            proposal.data_locations_by_chart,
            device=chart_ids.device,
            dtype=result.dtype,
        )[chart_ids]
        lower_cdf = torch.special.ndtr(
            (proposal.minimum - locations) / proposal.data_sigma
        )
        if proposal.prior_family == "bounded_uniform":
            upper_cdf = torch.special.ndtr(
                (proposal.maximum - locations) / proposal.data_sigma
            )
        else:
            upper_cdf = torch.ones_like(locations)
        uniforms = lower_cdf + torch.rand(
            chart_ids.shape,
            device=chart_ids.device,
            dtype=result.dtype,
            generator=generator,
        ) * (upper_cdf - lower_cdf)
        eps = torch.finfo(result.dtype).eps
        data_sample = locations + proposal.data_sigma * torch.special.ndtri(
            torch.clamp(uniforms, min=eps, max=1.0 - eps)
        )
        data_sample = torch.clamp(data_sample, min=proposal.minimum)
        if proposal.prior_family == "bounded_uniform":
            data_sample = torch.clamp(data_sample, max=proposal.maximum)
        return torch.where(use_data, data_sample, result)

    def _continuous_rj_group_log_likelihood(
        self,
        data: StructuralGeometryBatch,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        chart_ids: NDArray[np.int64],
        particle_indices: NDArray[np.int64] | None = None,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate a batched equal-K group at an optional intermediate target."""
        position_array = np.asarray(positions, dtype=np.float64)
        strength_array = np.asarray(strengths, dtype=np.float64)
        raw_chart_ids = np.asarray(chart_ids)
        active_tempering_start_row = self._structural_rj_tempering_start_row
        if (
            tempering_start_row is not None
            and active_tempering_start_row is not None
            and int(tempering_start_row) != int(active_tempering_start_row)
        ):
            raise ValueError(
                "Continuous RJ likelihood evaluation changed the active "
                "tempering station boundary."
            )
        resolved_tempering_start_row = (
            active_tempering_start_row
            if tempering_start_row is None
            else int(tempering_start_row)
        )
        if (
            position_array.ndim != 3
            or position_array.shape[2] != 3
            or strength_array.shape != position_array.shape[:2]
            or not np.issubdtype(raw_chart_ids.dtype, np.integer)
            or raw_chart_ids.shape != strength_array.shape
        ):
            raise ValueError(
                "Continuous RJ chart, position, and strength arrays must share "
                "particle/source axes."
            )
        chart_id_array = np.asarray(raw_chart_ids, dtype=np.int64)
        particle_count = int(strength_array.shape[0])
        if self._joint_target_evaluator is not None:
            if particle_indices is None:
                raise ValueError(
                    "Joint-target RJ evaluation requires aligned particle indices."
                )
            indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
            if (
                indices.size != particle_count
                or np.any(indices < 0)
                or np.any(indices >= len(self.continuous_particles))
            ):
                raise ValueError(
                    "Joint-target particle indices must match the candidate rows."
                )
            result = np.asarray(
                self._joint_target_evaluator(
                    filt=self,
                    data=data,
                    positions_pks=position_array,
                    chart_ids_pk=chart_id_array,
                    strengths_pk=strength_array,
                    particle_indices=indices,
                    target_beta=float(target_beta),
                    tempering_start_row=resolved_tempering_start_row,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                result.size != particle_count
                or np.any(np.isnan(result))
                or np.any(np.isposinf(result))
            ):
                raise ValueError(
                    "Joint-target evaluator must return one finite or "
                    "negative-infinity value per candidate particle."
                )
            return result
        raise RuntimeError(
            "Continuous exact-RJ moves require the estimator-owned full "
            "joint-isotope target evaluator."
        )

    def _continuous_rj_group_log_likelihood_torch(
        self,
        data: StructuralGeometryBatch,
        positions: object,
        strengths: object,
        *,
        chart_ids: object,
        particle_indices: object,
        target_beta: float = 1.0,
        tempering_start_row: int | None = None,
    ) -> object:
        """Evaluate one equal-cardinality candidate group on its CUDA device."""
        import torch

        if not all(
            torch.is_tensor(value)
            for value in (positions, strengths, chart_ids, particle_indices)
        ):
            raise TypeError("CUDA RJ target inputs must all be Torch tensors.")
        active_start = self._structural_rj_tempering_start_row
        if (
            tempering_start_row is not None
            and active_start is not None
            and int(tempering_start_row) != int(active_start)
        ):
            raise ValueError(
                "Continuous RJ likelihood evaluation changed the active "
                "tempering station boundary."
            )
        resolved_start = (
            active_start
            if tempering_start_row is None
            else int(tempering_start_row)
        )
        if (
            positions.ndim != 3
            or positions.shape[2] != 3
            or strengths.shape != positions.shape[:2]
            or chart_ids.shape != strengths.shape
            or chart_ids.dtype != torch.long
            or int(particle_indices.numel()) != int(positions.shape[0])
        ):
            raise ValueError(
                "CUDA RJ chart, position, strength, and row arrays are misaligned."
            )
        if self._joint_target_evaluator is None:
            raise RuntimeError(
                "Continuous exact-RJ moves require the estimator-owned full "
                "joint-isotope target evaluator."
            )
        result = self._joint_target_evaluator(
            filt=self,
            data=data,
            positions_pks=positions,
            chart_ids_pk=chart_ids,
            strengths_pk=strengths,
            particle_indices=particle_indices,
            target_beta=float(target_beta),
            tempering_start_row=resolved_start,
        )
        if not torch.is_tensor(result):
            raise RuntimeError("CUDA RJ target evaluator returned a host array.")
        result = result.reshape(-1)
        invalid = torch.isnan(result) | torch.isposinf(result)
        if int(result.numel()) != int(positions.shape[0]) or bool(
            torch.any(invalid).item()
        ):
            raise ValueError(
                "CUDA RJ target must return one finite or negative-infinity "
                "value per row."
            )
        return result

    def _continuous_rj_proposal_guide_log_target_torch(
        self,
        data: StructuralGeometryBatch,
        positions: object,
        strengths: object,
        *,
        chart_ids: object,
        particle_indices: object,
        target_beta: float,
    ) -> object:
        """Evaluate the full-history target used symmetrically by a proposal."""
        return self._continuous_rj_group_log_likelihood_torch(
            data,
            positions,
            strengths,
            chart_ids=chart_ids,
            particle_indices=particle_indices,
            target_beta=float(target_beta),
        )

    def _continuous_rj_current_log_likelihood(
        self,
        data: StructuralGeometryBatch,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
        *,
        chart_ids: NDArray[np.int64],
        particle_indices: NDArray[np.int64],
        target_beta: float,
    ) -> NDArray[np.float64]:
        """Return cached current-target values or evaluate an uncached group."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        device_cached = self._structural_rj_current_target_log_likelihood_device
        if device_cached is not None:
            import torch

            index_tensor = torch.as_tensor(
                indices,
                device=device_cached.device,
                dtype=torch.long,
            )
            result = (
                torch.index_select(device_cached, 0, index_tensor)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
                raise RuntimeError(
                    "Continuous RJ CUDA current-target cache is invalid."
                )
            return result
        cached = self._structural_rj_current_target_log_likelihood
        if cached is None:
            return self._continuous_rj_group_log_likelihood(
                data,
                positions,
                strengths,
                chart_ids=chart_ids,
                particle_indices=indices,
                target_beta=target_beta,
            )
        if (
            cached.shape != (len(self.continuous_particles),)
            or np.any(indices < 0)
            or np.any(indices >= cached.size)
        ):
            raise RuntimeError("Continuous RJ current-target cache is misaligned.")
        result = np.asarray(cached[indices], dtype=np.float64)
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Continuous RJ current-target cache is numerically invalid."
            )
        return result.copy()

    def _continuous_rj_current_log_likelihood_torch(
        self,
        data: StructuralGeometryBatch,
        positions: object,
        strengths: object,
        *,
        chart_ids: object,
        particle_indices: object,
        target_beta: float,
    ) -> object:
        """Return cached current-target values without leaving CUDA."""
        import torch

        if not torch.is_tensor(particle_indices):
            raise TypeError("CUDA current-target indices must be a tensor.")
        cached = self._structural_rj_current_target_log_likelihood_device
        if cached is None:
            return self._continuous_rj_group_log_likelihood_torch(
                data,
                positions,
                strengths,
                chart_ids=chart_ids,
                particle_indices=particle_indices,
                target_beta=target_beta,
            )
        if tuple(cached.shape) != (len(self.continuous_particles),):
            raise RuntimeError("CUDA RJ current-target cache is misaligned.")
        result = torch.index_select(cached, 0, particle_indices)
        if bool(torch.any(torch.isnan(result) | torch.isposinf(result)).item()):
            raise RuntimeError("CUDA RJ current-target cache is invalid.")
        return result

    def _update_continuous_rj_current_log_likelihood(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        proposed_log_likelihood: NDArray[np.float64],
    ) -> None:
        """Commit accepted candidate target values to the sweep-local cache."""
        cached = self._structural_rj_current_target_log_likelihood
        device_cached = self._structural_rj_current_target_log_likelihood_device
        if cached is None and device_cached is None:
            return
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
        proposed = np.asarray(
            proposed_log_likelihood,
            dtype=np.float64,
        ).reshape(-1)
        if (
            indices.size != acceptance.size
            or proposed.size != indices.size
            or np.any(indices < 0)
            or np.any(indices >= len(self.continuous_particles))
            or np.any(np.isnan(proposed[acceptance]))
            or np.any(np.isposinf(proposed[acceptance]))
        ):
            raise RuntimeError("Accepted continuous RJ target values are invalid.")
        if cached is not None:
            cached[indices[acceptance]] = proposed[acceptance]
        if device_cached is not None:
            import torch

            selected_indices = torch.as_tensor(
                indices[acceptance],
                device=device_cached.device,
                dtype=torch.long,
            )
            selected_values = torch.as_tensor(
                proposed[acceptance],
                device=device_cached.device,
                dtype=device_cached.dtype,
            )
            device_cached.index_copy_(0, selected_indices, selected_values)

    def _update_continuous_rj_current_log_likelihood_torch(
        self,
        particle_indices: object,
        accepted: object,
        proposed_log_likelihood: object,
        proposed_station_log_likelihood: object | None = None,
    ) -> None:
        """Commit accepted CUDA target and station values to sweep-local caches."""
        import torch

        cached = self._structural_rj_current_target_log_likelihood_device
        if cached is None:
            return
        if not all(
            torch.is_tensor(value)
            for value in (particle_indices, accepted, proposed_log_likelihood)
        ):
            raise TypeError("CUDA target-cache updates require Torch tensors.")
        indices = particle_indices.reshape(-1)
        acceptance = accepted.reshape(-1)
        proposed = proposed_log_likelihood.reshape(-1)
        if (
            int(indices.numel()) != int(acceptance.numel())
            or int(proposed.numel()) != int(indices.numel())
        ):
            raise RuntimeError("CUDA RJ target-cache update is misaligned.")
        selected_indices = indices[acceptance]
        selected_values = proposed[acceptance]
        if bool(
            torch.any(torch.isnan(selected_values) | torch.isposinf(selected_values)).item()
        ):
            raise RuntimeError("Accepted CUDA RJ target values are invalid.")
        cached.index_copy_(0, selected_indices, selected_values)
        station_cached = (
            self._structural_rj_current_station_log_likelihood_device
        )
        if station_cached is None:
            if proposed_station_log_likelihood is not None:
                raise RuntimeError(
                    "A per-station proposal exists without its current cache."
                )
            return
        if proposed_station_log_likelihood is None or not torch.is_tensor(
            proposed_station_log_likelihood
        ):
            raise RuntimeError("Accepted rows require per-station targets.")
        station_proposed = proposed_station_log_likelihood
        if (
            station_proposed.device != station_cached.device
            or station_proposed.dtype != station_cached.dtype
            or tuple(station_proposed.shape)
            != (int(indices.numel()), int(station_cached.shape[1]))
        ):
            raise RuntimeError("Per-station target update is misaligned.")
        selected_station = station_proposed[acceptance]
        if bool(torch.any(~torch.isfinite(selected_station)).item()):
            raise RuntimeError("Accepted per-station targets are invalid.")
        station_cached.index_copy_(0, selected_indices, selected_station)

    def _continuous_rj_exact_decision_torch(
        self,
        data: StructuralGeometryBatch,
        proposed_positions: object,
        proposed_strengths: object,
        *,
        proposed_chart_ids: object,
        particle_indices: object,
        base_log_likelihood: object,
        log_non_likelihood_ratio: object,
        support: object,
        target_beta: float,
        move_family: str,
    ) -> ExactMHDecision:
        """Evaluate one full-history target and make one exact MH decision."""
        import torch

        values = (
            proposed_positions,
            proposed_strengths,
            proposed_chart_ids,
            particle_indices,
            base_log_likelihood,
            log_non_likelihood_ratio,
            support,
        )
        if not all(torch.is_tensor(value) for value in values):
            raise TypeError("CUDA exact-MH decisions require Torch tensors.")
        base = base_log_likelihood.reshape(-1)
        non_likelihood = log_non_likelihood_ratio.reshape(-1)
        feasible = support.to(dtype=torch.bool).reshape(-1)
        row_count = int(base.numel())
        if (
            row_count <= 0
            or int(particle_indices.numel()) != row_count
            or int(proposed_positions.shape[0]) != row_count
            or tuple(non_likelihood.shape) != (row_count,)
            or tuple(feasible.shape) != (row_count,)
        ):
            raise ValueError("CUDA exact-MH decision arrays are not row aligned.")
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["mh_acceptance_calls"] = int(
            diagnostics.get("mh_acceptance_calls", 0)
        ) + 1
        diagnostics["mh_acceptance_rows"] = int(
            diagnostics.get("mh_acceptance_rows", 0)
        ) + row_count
        current_station = (
            self._structural_rj_current_station_log_likelihood_device
        )
        supported_rows = torch.nonzero(
            feasible,
            as_tuple=False,
        ).reshape(-1)
        proposed_target = torch.full_like(base, float("-inf"))
        if current_station is None:
            # Directly constructed filters are the explicit small test oracle.
            proposed_station = None
            if int(supported_rows.numel()):
                selected_target = self._continuous_rj_group_log_likelihood_torch(
                    data,
                    torch.index_select(proposed_positions, 0, supported_rows),
                    torch.index_select(proposed_strengths, 0, supported_rows),
                    chart_ids=torch.index_select(
                        proposed_chart_ids,
                        0,
                        supported_rows,
                    ),
                    particle_indices=torch.index_select(
                        particle_indices.reshape(-1),
                        0,
                        supported_rows,
                    ),
                    target_beta=target_beta,
                )
                proposed_target.index_copy_(
                    0,
                    supported_rows,
                    selected_target,
                )
        else:
            if self._joint_target_evaluator is None:
                raise RuntimeError(
                    "Production exact MH requires the joint target evaluator."
                )
            indices = particle_indices.reshape(-1)
            proposed_station = torch.index_select(
                current_station,
                0,
                indices,
            ).clone()
            if int(supported_rows.numel()):
                selected_target, selected_station = self._joint_target_evaluator(
                    filt=self,
                    data=data,
                    positions_pks=torch.index_select(
                        proposed_positions,
                        0,
                        supported_rows,
                    ),
                    chart_ids_pk=torch.index_select(
                        proposed_chart_ids,
                        0,
                        supported_rows,
                    ),
                    strengths_pk=torch.index_select(
                        proposed_strengths,
                        0,
                        supported_rows,
                    ),
                    particle_indices=torch.index_select(
                        indices,
                        0,
                        supported_rows,
                    ),
                    target_beta=float(target_beta),
                    tempering_start_row=self._structural_rj_tempering_start_row,
                    return_station_log_likelihood=True,
                    stage_unit_transport=True,
                )
                selected_target = torch.as_tensor(
                    selected_target,
                    device=base.device,
                    dtype=base.dtype,
                ).reshape(-1)
                selected_station = torch.as_tensor(
                    selected_station,
                    device=base.device,
                    dtype=base.dtype,
                )
                proposed_target.index_copy_(
                    0,
                    supported_rows,
                    selected_target,
                )
                proposed_station.index_copy_(
                    0,
                    supported_rows,
                    selected_station,
                )
            expected_station_shape = (row_count, int(current_station.shape[1]))
            if tuple(proposed_station.shape) != expected_station_shape:
                raise RuntimeError(
                    "Exact-MH proposal station targets are misaligned."
                )
        del move_family
        return run_exact_mh_acceptance_torch(
            current_target_log_likelihood=base,
            proposed_target_log_likelihood=proposed_target,
            proposed_station_log_likelihood=proposed_station,
            log_non_likelihood_ratio=non_likelihood,
            support=feasible,
            generator=self._continuous_rj_torch_generator_required(),
        )

    def set_joint_target_evaluator(
        self,
        evaluator: Callable[..., NDArray[np.float64]] | None,
    ) -> None:
        """Attach the estimator-owned aligned multi-isotope MH target."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError("Joint target evaluator must be callable or None.")
        self._joint_target_evaluator = evaluator

    def set_joint_strength_grid_target_evaluator(
        self,
        evaluator: Callable[..., NDArray[np.float64]] | None,
    ) -> None:
        """Attach the estimator-owned fixed-geometry strength-grid target."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError(
                "Joint strength-grid target evaluator must be callable or None."
            )
        self._joint_strength_grid_target_evaluator = evaluator

    def set_joint_proposal_evaluator(
        self,
        evaluator: Callable[
            ...,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                bool,
            ],
        ]
        | None,
    ) -> None:
        """Attach the estimator-owned full-spectrum residual proposal."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError("Joint proposal evaluator must be callable or None.")
        self._joint_proposal_evaluator = evaluator

    def _continuous_rj_canonicalize_rows(
        self,
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Canonicalize batched source rows by chart, U, then V."""
        atlas = self._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("The continuous surface atlas is unavailable.")
        charts, uv = atlas.validate_coordinates(chart_ids, surface_uv)
        xyz = np.asarray(positions, dtype=np.float64)
        q = np.asarray(strengths, dtype=np.float64)
        if (
            charts.ndim != 2
            or uv.shape != charts.shape + (2,)
            or xyz.shape != charts.shape + (3,)
            or q.shape != charts.shape
        ):
            raise ValueError("Continuous RJ canonical arrays have invalid shapes.")
        derived_xyz = atlas.positions_xyz(charts, uv)
        if not np.allclose(
            xyz,
            derived_xyz,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                "Transient RJ XYZ must equal the authoritative chart/UV image."
            )
        xyz = derived_xyz
        if charts.shape[1] <= 1:
            return charts, uv, xyz, q
        order = np.lexsort(
            (uv[:, :, 1], uv[:, :, 0], charts),
            axis=1,
        )
        return (
            np.take_along_axis(charts, order, axis=1),
            np.take_along_axis(uv, order[:, :, None], axis=1),
            np.take_along_axis(xyz, order[:, :, None], axis=1),
            np.take_along_axis(q, order, axis=1),
        )

    def _initialize_continuous_rj_device_state(
        self,
        reference: object,
    ) -> bool:
        """Mirror fixed-capacity RJ state on the active Torch compute device.

        Variable-length :class:`IsotopeState` objects remain the reporting and
        serialization contract. During one structural sweep, numerical state
        is also kept in fixed-capacity tensors so accepted rows can be updated
        with one device scatter and later proposals avoid repacking every
        Python particle.
        """
        if not hasattr(reference, "detach"):
            self._structural_rj_device_state = None
            self.last_structural_device_diagnostics = {
                "backend": "numpy",
                "mh_acceptance_calls": 0,
                "mh_acceptance_rows": 0,
                "state_scatter_calls": 0,
                "state_scatter_rows": 0,
                "group_gather_calls": 0,
            }
            return False
        import torch

        if not torch.is_tensor(reference):
            raise TypeError("RJ device-state reference must be a Torch tensor.")
        device = reference.device
        dtype = reference.dtype
        existing = self._structural_rj_device_state
        if (
            existing is not None
            and bool(self._structural_rj_device_state_authoritative)
        ):
            if (
                existing["strengths"].device != device
                or existing["strengths"].dtype != dtype
                or int(existing["strengths"].shape[0])
                != len(self.continuous_particles)
            ):
                raise RuntimeError(
                    "Station-authoritative RJ state changed device, dtype, or rows."
                )
            self._refresh_continuous_rj_device_cache_snapshot()
            diagnostics = self.last_structural_device_diagnostics
            diagnostics["state_reuse_calls"] = int(
                diagnostics.get("state_reuse_calls", 0)
            ) + 1
            return True
        positions, strengths, mask, chart_ids, surface_uv = (
            self._packed_continuous_surface_state_arrays()
        )
        current = {
            "positions": torch.as_tensor(
                positions,
                device=device,
                dtype=dtype,
            ).clone(),
            "strengths": torch.as_tensor(
                strengths,
                device=device,
                dtype=dtype,
            ).clone(),
            "mask": torch.as_tensor(
                mask,
                device=device,
                dtype=torch.bool,
            ).clone(),
            "chart_ids": torch.as_tensor(
                chart_ids,
                device=device,
                dtype=torch.long,
            ).clone(),
            "surface_uv": torch.as_tensor(
                surface_uv,
                device=device,
                dtype=dtype,
            ).clone(),
            "cardinalities": torch.as_tensor(
                np.sum(mask, axis=1, dtype=np.int64),
                device=device,
                dtype=torch.long,
            ).clone(),
        }
        # These immutable tensors describe the transport cache at sweep entry.
        # Matching against them is exact even after accepted states move.
        current.update(
            {
                f"cache_{name}": value.clone()
                for name, value in current.items()
                if name != "cardinalities"
            }
        )
        self._structural_rj_device_state = current
        self.last_structural_device_diagnostics = {
            "backend": str(device),
            "mh_acceptance_calls": 0,
            "mh_acceptance_rows": 0,
            "state_scatter_calls": 0,
            "state_scatter_rows": 0,
            "group_gather_calls": 0,
            "host_snapshot_calls": 0,
            "state_reuse_calls": 0,
            "deferred_clear_calls": 0,
            "materialization_calls": 0,
            "resample_reindex_calls": 0,
        }
        return True

    def _refresh_continuous_rj_device_cache_snapshot(self) -> None:
        """Freeze the accepted state used by one structural transport sweep."""
        state = self._structural_rj_device_state
        if state is None:
            raise RuntimeError("Cannot snapshot a missing RJ device state.")
        primary_names = (
            "positions",
            "strengths",
            "mask",
            "chart_ids",
            "surface_uv",
        )
        for name in tuple(state):
            if name.startswith("cache_"):
                del state[name]
        state.update(
            {f"cache_{name}": state[name].clone() for name in primary_names}
        )

    def _begin_continuous_rj_station_device_state(
        self,
        reference: object,
    ) -> bool:
        """Make a fixed-capacity CUDA state authoritative for one station."""
        if not hasattr(reference, "detach"):
            return False
        import torch

        if not torch.is_tensor(reference) or not bool(reference.is_cuda):
            return False
        if bool(self._structural_rj_device_state_authoritative):
            raise RuntimeError("RJ station device authority is already active.")
        initialized = self._initialize_continuous_rj_device_state(reference)
        if not initialized or self._structural_rj_device_state is None:
            raise RuntimeError("CUDA RJ state failed to initialize.")
        self._structural_rj_device_state_authoritative = True
        self._structural_rj_device_state_dirty = False
        self.last_structural_device_diagnostics["authority"] = "station"
        return True

    def _reindex_continuous_rj_device_state(
        self,
        indices: NDArray[np.int64],
    ) -> None:
        """Apply one joint-resampling ancestor vector to all device state rows."""
        if not bool(self._structural_rj_device_state_authoritative):
            return
        state = self._structural_rj_device_state
        if state is None:
            raise RuntimeError("Authoritative RJ device state is missing.")
        import torch

        raw = np.asarray(indices)
        particle_count = len(self.continuous_particles)
        if (
            raw.dtype != np.int64
            or raw.shape != (particle_count,)
            or np.any(raw < 0)
            or np.any(raw >= particle_count)
        ):
            raise RuntimeError("RJ device resampling indices are invalid.")
        index_tensor = torch.as_tensor(
            raw,
            device=state["strengths"].device,
            dtype=torch.long,
        )
        for name, value in tuple(state.items()):
            state[name] = torch.index_select(value, 0, index_tensor).contiguous()
        self._structural_rj_device_state_dirty = True
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["resample_reindex_calls"] = int(
            diagnostics.get("resample_reindex_calls", 0)
        ) + 1

    def _materialize_continuous_rj_device_state(self) -> None:
        """Convert the authoritative fixed-capacity state to Python particles."""
        if not bool(self._structural_rj_device_state_authoritative):
            return
        state = self._structural_rj_device_state
        if state is None:
            raise RuntimeError("Authoritative RJ device state is missing.")
        import torch

        self.validate_continuous_surface_states()
        dtype = state["strengths"].dtype
        payload = torch.cat(
            (
                state["positions"],
                state["strengths"][..., None],
                state["surface_uv"],
                state["chart_ids"][..., None].to(dtype=dtype),
                state["mask"][..., None].to(dtype=dtype),
            ),
            dim=2,
        ).detach().cpu().numpy()
        cardinalities = np.sum(payload[..., 7] != 0.0, axis=1, dtype=np.int64)
        for row, particle in enumerate(self.continuous_particles):
            cardinality = int(cardinalities[row])
            particle.state = IsotopeState(
                num_sources=cardinality,
                surface_chart_ids=np.asarray(
                    payload[row, :cardinality, 6],
                    dtype=np.int64,
                ),
                surface_uv=np.asarray(
                    payload[row, :cardinality, 4:6],
                    dtype=np.float64,
                ),
                strengths=np.asarray(
                    payload[row, :cardinality, 3],
                    dtype=np.float64,
                ),
            )
        self._structural_rj_device_state_dirty = False
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["materialization_calls"] = int(
            diagnostics.get("materialization_calls", 0)
        ) + 1

    def _end_continuous_rj_station_device_state(self) -> None:
        """Materialize and release station-authoritative CUDA state exactly once."""
        if not bool(self._structural_rj_device_state_authoritative):
            return
        try:
            self._materialize_continuous_rj_device_state()
            target = self.last_structural_target_log_likelihood_device
            if target is not None:
                self.last_structural_target_log_likelihood = (
                    target.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                    .copy()
                )
                self.last_structural_target_log_likelihood_device = None
        finally:
            self._structural_rj_device_state_authoritative = False
            self._structural_rj_device_state_dirty = False
            self._structural_rj_device_state = None

    def _clear_continuous_rj_device_state(self) -> None:
        """Release a sweep mirror or defer release of station authority."""
        if bool(self._structural_rj_device_state_authoritative):
            state = self._structural_rj_device_state
            if state is None:
                raise RuntimeError("Authoritative RJ device state is missing.")
            for name in tuple(state):
                if name.startswith("cache_"):
                    del state[name]
            diagnostics = self.last_structural_device_diagnostics
            diagnostics["deferred_clear_calls"] = int(
                diagnostics.get("deferred_clear_calls", 0)
            ) + 1
            return
        self._structural_rj_device_state = None

    def _continuous_rj_mh_acceptance_mask(
        self,
        log_ratio: NDArray[np.float64],
        *,
        support: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Draw one exact batched MH mask on the active compute device.

        Uniform draws and their logarithms stay on NumPy to preserve the
        established random stream bit-for-bit. Only the vectorized comparison
        and optional support mask move to Torch/CUDA.
        """
        ratios = np.asarray(log_ratio, dtype=np.float64).reshape(-1)
        uniforms = self._random_generator.random(ratios.size)
        with np.errstate(divide="ignore"):
            log_uniforms = np.log(uniforms)
        thresholds = np.minimum(ratios, 0.0)
        feasible = None
        if support is not None:
            feasible = np.asarray(support, dtype=np.bool_).reshape(-1)
            if feasible.size != ratios.size:
                raise ValueError("MH support must align with log_ratio.")
        state = self._structural_rj_device_state
        if state is None:
            accepted = log_uniforms < thresholds
            if feasible is not None:
                accepted &= feasible
            return np.asarray(accepted, dtype=np.bool_)
        import torch

        device = state["strengths"].device
        accepted_tensor = torch.as_tensor(
            log_uniforms,
            device=device,
            dtype=torch.float64,
        ) < torch.as_tensor(
            thresholds,
            device=device,
            dtype=torch.float64,
        )
        if feasible is not None:
            accepted_tensor &= torch.as_tensor(
                feasible,
                device=device,
                dtype=torch.bool,
            )
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["mh_acceptance_calls"] = (
            int(diagnostics.get("mh_acceptance_calls", 0)) + 1
        )
        diagnostics["mh_acceptance_rows"] = int(
            diagnostics.get("mh_acceptance_rows", 0)
        ) + int(ratios.size)
        return accepted_tensor.detach().cpu().numpy().astype(np.bool_, copy=False)

    def _commit_continuous_rj_states(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        chart_ids: NDArray[np.int64],
        surface_uv: NDArray[np.float64],
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> int:
        """Commit accepted continuous chart states without changing PF weights."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
        charts, uv, xyz, q = self._continuous_rj_canonicalize_rows(
            chart_ids,
            surface_uv,
            positions,
            strengths,
        )
        if acceptance.size != indices.size or charts.shape[0] != indices.size:
            raise ValueError("Continuous RJ commit arrays must share P.")
        accepted_rows = np.flatnonzero(acceptance)
        cardinality = int(charts.shape[1])
        center_cache = self._structural_rj_current_block_strength_centers
        cardinality_cache = self._structural_rj_current_block_strength_cardinalities
        if center_cache is not None and cardinality_cache is not None:
            changed_indices = indices[accepted_rows]
            center_cache[changed_indices] = float("nan")
            cardinality_cache[changed_indices] = -1
        device_state = self._structural_rj_device_state
        if device_state is not None and accepted_rows.size:
            import torch

            accepted_indices = indices[accepted_rows]
            maximum = int(self.config.hard_max_sources or 0)
            row_count = int(accepted_rows.size)
            device = device_state["strengths"].device
            dtype = device_state["strengths"].dtype
            padded_positions = torch.zeros(
                (row_count, maximum, 3),
                device=device,
                dtype=dtype,
            )
            padded_strengths = torch.zeros(
                (row_count, maximum),
                device=device,
                dtype=dtype,
            )
            padded_mask = torch.zeros(
                (row_count, maximum),
                device=device,
                dtype=torch.bool,
            )
            padded_charts = torch.zeros(
                (row_count, maximum),
                device=device,
                dtype=torch.long,
            )
            padded_uv = torch.zeros(
                (row_count, maximum, 2),
                device=device,
                dtype=dtype,
            )
            if cardinality:
                selected = accepted_rows
                padded_positions[:, :cardinality] = torch.as_tensor(
                    xyz[selected],
                    device=device,
                    dtype=dtype,
                )
                padded_strengths[:, :cardinality] = torch.as_tensor(
                    q[selected],
                    device=device,
                    dtype=dtype,
                )
                padded_mask[:, :cardinality] = True
                padded_charts[:, :cardinality] = torch.as_tensor(
                    charts[selected],
                    device=device,
                    dtype=torch.long,
                )
                padded_uv[:, :cardinality] = torch.as_tensor(
                    uv[selected],
                    device=device,
                    dtype=dtype,
                )
            index_tensor = torch.as_tensor(
                accepted_indices,
                device=device,
                dtype=torch.long,
            )
            for name, values in (
                ("positions", padded_positions),
                ("strengths", padded_strengths),
                ("mask", padded_mask),
                ("chart_ids", padded_charts),
                ("surface_uv", padded_uv),
            ):
                device_state[name].index_copy_(0, index_tensor, values)
            device_state["cardinalities"].index_fill_(
                0,
                index_tensor,
                cardinality,
            )
            diagnostics = self.last_structural_device_diagnostics
            diagnostics["state_scatter_calls"] = (
                int(diagnostics.get("state_scatter_calls", 0)) + 1
            )
            diagnostics["state_scatter_rows"] = (
                int(diagnostics.get("state_scatter_rows", 0)) + row_count
            )
            self._structural_rj_device_state_dirty = True
        # All numerical proposal and acceptance work is batched. This loop only
        # commits variable-length state objects for the accepted particle rows.
        if not bool(self._structural_rj_device_state_authoritative):
            for row in accepted_rows.tolist():
                self.continuous_particles[int(indices[row])].state = IsotopeState(
                    num_sources=cardinality,
                    surface_chart_ids=charts[row],
                    surface_uv=uv[row],
                    strengths=q[row],
                )
        return int(accepted_rows.size)

    def _commit_continuous_rj_state_tensors(
        self,
        particle_indices: object,
        accepted: object,
        chart_ids: object,
        surface_uv: object,
        positions: object,
        strengths: object,
    ) -> int:
        """Commit accepted fixed-cardinality candidate rows directly on CUDA."""
        import torch

        state = self._structural_rj_device_state
        if state is None or not self._continuous_rj_torch_enabled():
            raise RuntimeError("CUDA RJ commit requires authoritative device state.")
        if not all(
            torch.is_tensor(value)
            for value in (
                particle_indices,
                accepted,
                chart_ids,
                surface_uv,
                positions,
                strengths,
            )
        ):
            raise TypeError("CUDA RJ commit values must all be Torch tensors.")
        indices = particle_indices.reshape(-1)
        acceptance = accepted.to(dtype=torch.bool).reshape(-1)
        charts, uv, xyz, q = self._continuous_rj_canonicalize_tensors(
            chart_ids,
            surface_uv,
            positions,
            strengths,
        )
        if int(acceptance.numel()) != int(indices.numel()) or int(
            charts.shape[0]
        ) != int(indices.numel()):
            raise ValueError("CUDA RJ commit arrays must share particle rows.")
        accepted_rows = torch.nonzero(acceptance, as_tuple=False).reshape(-1)
        accepted_count = int(accepted_rows.numel())
        if accepted_count == 0:
            return 0
        accepted_indices = indices[accepted_rows]
        cardinality = int(charts.shape[1])
        maximum = int(self.config.hard_max_sources or 0)
        if cardinality > maximum:
            raise ValueError("CUDA RJ candidate exceeds fixed source capacity.")
        row_count = accepted_count
        device = state["strengths"].device
        dtype = state["strengths"].dtype
        padded_positions = torch.zeros(
            (row_count, maximum, 3),
            device=device,
            dtype=dtype,
        )
        padded_strengths = torch.zeros(
            (row_count, maximum),
            device=device,
            dtype=dtype,
        )
        padded_mask = torch.zeros(
            (row_count, maximum),
            device=device,
            dtype=torch.bool,
        )
        padded_charts = torch.zeros(
            (row_count, maximum),
            device=device,
            dtype=torch.long,
        )
        padded_uv = torch.zeros(
            (row_count, maximum, 2),
            device=device,
            dtype=dtype,
        )
        if cardinality:
            padded_positions[:, :cardinality] = xyz[accepted_rows]
            padded_strengths[:, :cardinality] = q[accepted_rows]
            padded_mask[:, :cardinality] = True
            padded_charts[:, :cardinality] = charts[accepted_rows]
            padded_uv[:, :cardinality] = uv[accepted_rows]
        for name, values in (
            ("positions", padded_positions),
            ("strengths", padded_strengths),
            ("mask", padded_mask),
            ("chart_ids", padded_charts),
            ("surface_uv", padded_uv),
        ):
            state[name].index_copy_(0, accepted_indices, values)
        state["cardinalities"].index_fill_(
            0,
            accepted_indices,
            cardinality,
        )
        center_cache = self._structural_rj_current_block_strength_centers
        cardinality_cache = self._structural_rj_current_block_strength_cardinalities
        if center_cache is not None and cardinality_cache is not None:
            changed = accepted_indices.detach().cpu().numpy()
            center_cache[changed] = float("nan")
            cardinality_cache[changed] = -1
        diagnostics = self.last_structural_device_diagnostics
        diagnostics["state_scatter_calls"] = int(
            diagnostics.get("state_scatter_calls", 0)
        ) + 1
        diagnostics["state_scatter_rows"] = int(
            diagnostics.get("state_scatter_rows", 0)
        ) + row_count
        self._structural_rj_device_state_dirty = True
        return accepted_count

    def _continuous_rj_transition_mass(
        self,
        name: str,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_] | None = None,
    ) -> None:
        """Accumulate attempted/accepted posterior weight mass diagnostics."""
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.continuous_weights, dtype=np.float64)
        if accepted is None:
            selected = indices
        else:
            acceptance = np.asarray(accepted, dtype=bool).reshape(-1)
            if acceptance.size != indices.size:
                raise ValueError("accepted must match particle_indices.")
            selected = indices[acceptance]
        key = f"{name}_weight_mass"
        mass = float(np.sum(weights[selected], dtype=np.float64))
        self._structural_rj_move_counts[key] = (
            float(self._structural_rj_move_counts.get(key, 0.0)) + mass
        )
        self.last_structural_transition_weight_mass[key] = (
            float(self.last_structural_transition_weight_mass.get(key, 0.0)) + mass
        )
        if name in {
            "birth_accepted",
            "global_position_accepted",
            "block_accepted",
        }:
            marker = np.asarray(
                self.last_structural_full_support_accepted_mask,
                dtype=np.bool_,
            ).reshape(-1)
            if marker.shape != weights.shape:
                raise RuntimeError(
                    "Full-support acceptance markers do not match PF rows."
                )
            marker[selected] = True
            self.last_structural_full_support_accepted_mask = marker

    def _continuous_rj_transition_mass_torch(
        self,
        name: str,
        particle_indices: object,
        accepted: object | None = None,
    ) -> None:
        """Accumulate transition mass after one compact device-to-host copy."""
        import torch

        if not torch.is_tensor(particle_indices):
            raise TypeError("CUDA transition indices must be a Torch tensor.")
        indices = particle_indices.detach().cpu().numpy().astype(
            np.int64,
            copy=False,
        )
        acceptance = None
        if accepted is not None:
            if not torch.is_tensor(accepted):
                raise TypeError("CUDA transition acceptance must be a tensor.")
            acceptance = accepted.detach().cpu().numpy().astype(
                np.bool_,
                copy=False,
            )
        self._continuous_rj_transition_mass(
            name,
            indices,
            acceptance,
        )

    def _record_structural_mh_components_torch(
        self,
        move: str,
        *,
        particle_indices: object,
        delta_log_likelihood: object,
        delta_log_prior: object,
        log_reverse_minus_forward: object,
        log_jacobian: object,
        support_feasible: object,
        accepted: object,
        current_cardinality: object = -1,
        proposed_cardinality: object = -1,
        geometry_support_feasible: object | None = None,
        strength_support_feasible: object | None = None,
        log_acceptance_ratio: object | None = None,
    ) -> None:
        """Transfer one fused diagnostic matrix after an exact CUDA MH decision."""
        import torch

        if not torch.is_tensor(delta_log_likelihood):
            raise TypeError("CUDA MH diagnostics require tensor likelihoods.")
        reference = delta_log_likelihood.reshape(-1)
        row_count = int(reference.numel())

        def _column(value: object, *, dtype: object) -> object:
            """Broadcast one diagnostic value on the reference device."""
            tensor = torch.as_tensor(
                value,
                device=reference.device,
                dtype=dtype,
            ).reshape(-1)
            if int(tensor.numel()) == 1 and row_count != 1:
                tensor = tensor.expand(row_count)
            if int(tensor.numel()) != row_count:
                raise ValueError("CUDA MH diagnostic columns must align.")
            return tensor

        geometry = (
            support_feasible
            if geometry_support_feasible is None
            else geometry_support_feasible
        )
        strength = (
            support_feasible
            if strength_support_feasible is None
            else strength_support_feasible
        )
        ratio = (
            reference
            + _column(delta_log_prior, dtype=reference.dtype)
            + _column(log_reverse_minus_forward, dtype=reference.dtype)
            + _column(log_jacobian, dtype=reference.dtype)
            if log_acceptance_ratio is None
            else _column(log_acceptance_ratio, dtype=reference.dtype)
        )
        matrix = torch.stack(
            (
                _column(particle_indices, dtype=reference.dtype),
                reference,
                _column(delta_log_prior, dtype=reference.dtype),
                _column(log_reverse_minus_forward, dtype=reference.dtype),
                _column(log_jacobian, dtype=reference.dtype),
                _column(support_feasible, dtype=reference.dtype),
                _column(accepted, dtype=reference.dtype),
                _column(current_cardinality, dtype=reference.dtype),
                _column(proposed_cardinality, dtype=reference.dtype),
                _column(geometry, dtype=reference.dtype),
                _column(strength, dtype=reference.dtype),
                ratio,
            ),
            dim=1,
        ).detach().cpu().numpy()
        self._record_structural_mh_components(
            move,
            particle_indices=matrix[:, 0].astype(np.int64),
            delta_log_likelihood=matrix[:, 1],
            delta_log_prior=matrix[:, 2],
            log_reverse_minus_forward=matrix[:, 3],
            log_jacobian=matrix[:, 4],
            support_feasible=matrix[:, 5].astype(np.bool_),
            accepted=matrix[:, 6].astype(np.bool_),
            current_cardinality=matrix[:, 7].astype(np.int64),
            proposed_cardinality=matrix[:, 8].astype(np.int64),
            geometry_support_feasible=matrix[:, 9].astype(np.bool_),
            strength_support_feasible=matrix[:, 10].astype(np.bool_),
            log_acceptance_ratio=matrix[:, 11],
        )

    def _record_source_events_torch(
        self,
        event: str,
        *,
        positions: object,
        strengths: object,
        source_columns: object,
        accepted: object,
        reason: str,
        extras: dict[str, object] | None = None,
    ) -> None:
        """Record accepted source events from CUDA values, not Python particles."""
        import torch

        if not all(
            torch.is_tensor(value)
            for value in (positions, strengths, source_columns, accepted)
        ):
            raise TypeError("CUDA source-event values must be tensors.")
        accepted_rows = torch.nonzero(accepted, as_tuple=False).reshape(-1)
        if int(accepted_rows.numel()) == 0:
            return
        columns = source_columns[accepted_rows].to(torch.long)
        selected_positions = positions[accepted_rows, columns]
        selected_strengths = strengths[accepted_rows, columns]
        payload = torch.cat(
            (
                columns[:, None].to(dtype=selected_positions.dtype),
                selected_positions,
                selected_strengths[:, None],
            ),
            dim=1,
        ).detach().cpu().numpy()
        extra_payload: dict[str, object] = {}
        for name, value in (extras or {}).items():
            if torch.is_tensor(value):
                extra_payload[name] = value[accepted_rows].detach().cpu().numpy()
            else:
                extra_payload[name] = value
        for row, values in enumerate(payload):
            record: dict[str, object] = {
                "event": str(event),
                "isotope": str(self.isotope),
                "reason": str(reason),
                "source_index": int(values[0]),
                "position": [float(value) for value in values[1:4]],
                "strength": float(values[4]),
            }
            for name, value in extra_payload.items():
                if isinstance(value, np.ndarray):
                    selected = value[row]
                    record[name] = (
                        selected.tolist()
                        if np.asarray(selected).ndim
                        else np.asarray(selected).item()
                    )
                else:
                    record[name] = value
            self.last_source_event_diagnostics.append(record)

    def _record_structural_mh_components(
        self,
        move: str,
        *,
        particle_indices: NDArray[np.int64],
        delta_log_likelihood: NDArray[np.float64],
        delta_log_prior: NDArray[np.float64],
        log_reverse_minus_forward: NDArray[np.float64],
        log_jacobian: NDArray[np.float64],
        support_feasible: NDArray[np.bool_],
        accepted: NDArray[np.bool_],
        current_cardinality: NDArray[np.int64] | int = -1,
        proposed_cardinality: NDArray[np.int64] | int = -1,
        geometry_support_feasible: NDArray[np.bool_] | bool | None = None,
        strength_support_feasible: NDArray[np.bool_] | bool | None = None,
        log_acceptance_ratio: NDArray[np.float64] | None = None,
    ) -> None:
        """Accumulate batched exact-MH terms for diagnosis."""
        likelihood = np.asarray(
            delta_log_likelihood,
            dtype=np.float64,
        ).reshape(-1)
        row_count = int(likelihood.size)

        def _broadcast(
            value: object,
            *,
            dtype: object,
            name: str,
        ) -> NDArray[np.generic]:
            """Broadcast one scalar or aligned diagnostic vector."""
            array = np.asarray(value, dtype=dtype)
            try:
                return np.broadcast_to(array, (row_count,)).copy()
            except ValueError as exc:
                raise ValueError(
                    f"Structural MH diagnostic {name} must align with rows."
                ) from exc

        prior = _broadcast(
            delta_log_prior,
            dtype=np.float64,
            name="delta_log_prior",
        )
        proposal = _broadcast(
            log_reverse_minus_forward,
            dtype=np.float64,
            name="log_reverse_minus_forward",
        )
        jacobian = _broadcast(
            log_jacobian,
            dtype=np.float64,
            name="log_jacobian",
        )
        support = _broadcast(
            support_feasible,
            dtype=np.bool_,
            name="support_feasible",
        )
        geometry_support = _broadcast(
            support if geometry_support_feasible is None else geometry_support_feasible,
            dtype=np.bool_,
            name="geometry_support_feasible",
        )
        strength_support = _broadcast(
            support if strength_support_feasible is None else strength_support_feasible,
            dtype=np.bool_,
            name="strength_support_feasible",
        )
        if log_acceptance_ratio is None:
            log_acceptance = likelihood + prior + proposal + jacobian
        else:
            log_acceptance = _broadcast(
                log_acceptance_ratio,
                dtype=np.float64,
                name="log_acceptance_ratio",
            )
        arrays = {
            "delta_log_likelihood": likelihood,
            "delta_log_prior": prior,
            "log_reverse_minus_forward": proposal,
            "log_jacobian": jacobian,
            "log_acceptance_ratio": log_acceptance,
            "support_feasible": support,
            "geometry_support_feasible": geometry_support,
            "strength_support_feasible": strength_support,
            "accepted": _broadcast(
                accepted,
                dtype=np.bool_,
                name="accepted",
            ),
            "current_cardinality": _broadcast(
                current_cardinality,
                dtype=np.int64,
                name="current_cardinality",
            ),
            "proposed_cardinality": _broadcast(
                proposed_cardinality,
                dtype=np.int64,
                name="proposed_cardinality",
            ),
        }
        lengths = {int(value.size) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Structural MH diagnostic arrays must align.")
        indices = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if indices.size != row_count:
            raise ValueError(
                "Structural MH particle_indices must align with proposal rows."
            )
        particle_count = len(self.continuous_particles)
        if np.any(indices < 0) or np.any(indices >= particle_count):
            raise ValueError("Structural MH particle_indices are out of range.")
        ordinary_maximum = int(self.config.max_sources or 0)
        current = np.asarray(arrays["current_cardinality"], dtype=np.int64)
        proposed = np.asarray(arrays["proposed_cardinality"], dtype=np.int64)
        inward_attempted = (
            (current >= ordinary_maximum)
            & (proposed >= 0)
            & (proposed < current)
        )
        inward_supported = inward_attempted & np.asarray(
            arrays["support_feasible"],
            dtype=np.bool_,
        )
        inward_finite = inward_supported & np.isfinite(
            np.asarray(arrays["log_acceptance_ratio"], dtype=np.float64)
        )
        inward_accepted = inward_supported & np.asarray(
            arrays["accepted"],
            dtype=np.bool_,
        )
        self._continuous_rj_transition_mass(
            "ordinary_boundary_inward_attempted",
            indices,
            inward_attempted,
        )
        self._continuous_rj_transition_mass(
            "ordinary_boundary_inward_supported",
            indices,
            inward_supported,
        )
        self._continuous_rj_transition_mass(
            "ordinary_boundary_inward_finite",
            indices,
            inward_finite,
        )
        self._continuous_rj_transition_mass(
            "ordinary_boundary_inward_accepted",
            indices,
            inward_accepted,
        )
        self._structural_mh_component_samples.setdefault(str(move), []).append(arrays)

    def _summarize_structural_mh_components(self) -> dict[str, object]:
        """Return compact rejection causes and MH terms by K transition."""
        result: dict[str, object] = {}
        quantile_levels = np.asarray(
            [0.0, 0.1, 0.5, 0.9, 1.0],
            dtype=np.float64,
        )
        for move, batches in self._structural_mh_component_samples.items():
            combined = {
                key: np.concatenate([np.asarray(batch[key]) for batch in batches])
                for key in batches[0]
            }
            numeric_names = (
                "delta_log_likelihood",
                "delta_log_prior",
                "log_reverse_minus_forward",
                "log_jacobian",
                "log_acceptance_ratio",
            )

            def _summarize_rows(mask: NDArray[np.bool_]) -> dict[str, object]:
                """Summarize one vectorized row subset."""
                feasible = np.asarray(
                    combined["support_feasible"],
                    dtype=bool,
                )[mask]
                geometry = np.asarray(
                    combined["geometry_support_feasible"],
                    dtype=bool,
                )[mask]
                strength = np.asarray(
                    combined["strength_support_feasible"],
                    dtype=bool,
                )[mask]
                accepted = np.asarray(
                    combined["accepted"],
                    dtype=bool,
                )[mask]
                finite_all = feasible.copy()
                quantiles: dict[str, dict[str, float | int] | None] = {}
                for name in numeric_names:
                    values = np.asarray(
                        combined[name],
                        dtype=np.float64,
                    )[mask]
                    finite = np.isfinite(values)
                    finite_all &= finite
                    if not np.any(finite):
                        quantiles[name] = None
                        continue
                    finite_values = values[finite]
                    resolved = np.quantile(finite_values, quantile_levels)
                    quantiles[name] = {
                        "finite_count": int(finite_values.size),
                        "mean": float(np.mean(finite_values)),
                        "std": float(np.std(finite_values)),
                        **{
                            label: float(value)
                            for label, value in zip(
                                ("min", "p10", "median", "p90", "max"),
                                resolved,
                                strict=True,
                            )
                        },
                    }
                return {
                    "attempted": int(feasible.size),
                    "accepted": int(np.count_nonzero(accepted)),
                    "support_rejected": int(np.count_nonzero(~feasible)),
                    "geometry_support_rejected": int(np.count_nonzero(~geometry)),
                    "strength_support_rejected": int(
                        np.count_nonzero(geometry & ~strength)
                    ),
                    "other_support_rejected": int(
                        np.count_nonzero(geometry & strength & ~feasible)
                    ),
                    "nonfinite_rejected": int(np.count_nonzero(feasible & ~finite_all)),
                    "mh_random_rejected": int(
                        np.count_nonzero(
                            feasible
                            & finite_all
                            & ~accepted
                        )
                    ),
                    "component_quantiles": quantiles,
                }

            all_rows = np.ones(
                np.asarray(combined["accepted"]).size,
                dtype=np.bool_,
            )
            move_summary = _summarize_rows(all_rows)
            current = np.asarray(
                combined["current_cardinality"],
                dtype=np.int64,
            )
            proposed = np.asarray(
                combined["proposed_cardinality"],
                dtype=np.int64,
            )
            cardinality_rows = (current >= 0) & (proposed >= 0)
            transition_summaries: dict[str, object] = {}
            # At most (hard_max_sources + 1)^2 transition labels exist. This
            # tiny loop only packages already-vectorized numeric summaries.
            if np.any(cardinality_rows):
                encoded = np.stack((current, proposed), axis=1)
                for source_count, destination_count in np.unique(
                    encoded[cardinality_rows],
                    axis=0,
                ).tolist():
                    transition_mask = (
                        cardinality_rows
                        & (current == int(source_count))
                        & (proposed == int(destination_count))
                    )
                    transition_summaries[
                        f"{int(source_count)}->{int(destination_count)}"
                    ] = _summarize_rows(transition_mask)
            move_summary["by_cardinality_transition"] = transition_summaries
            result[move] = move_summary
        return result


__all__ = ["StructuralRJTargetMixin"]
