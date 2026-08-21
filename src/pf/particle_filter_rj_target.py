"""Batched target evaluation and device-state support for exact-RJ PF."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import LineTransportComponents
from pf.particle_types import StructuralGeometryBatch, TorchLineTransportComponents
from pf.state import IsotopeState


class StructuralRJTargetMixin:
    """Evaluate exact-RJ targets and commit accepted batched state rows."""

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

                    components = TorchLineTransportComponents(
                        total_kernel=_selected_device_component("total_kernel"),
                        uncollided_kernel=_selected_device_component(
                            "uncollided_kernel"
                        ),
                        tau_fe=_selected_device_component("tau_fe"),
                        tau_pb=_selected_device_component("tau_pb"),
                        tau_obstacle=_selected_device_component("tau_obstacle"),
                        distance_m=_selected_device_component("distance_m"),
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

    def _update_continuous_rj_current_log_likelihood(
        self,
        particle_indices: NDArray[np.int64],
        accepted: NDArray[np.bool_],
        proposed_log_likelihood: NDArray[np.float64],
    ) -> None:
        """Commit accepted candidate target values to the sweep-local cache."""
        cached = self._structural_rj_current_target_log_likelihood
        if cached is None:
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
            or np.any(indices >= cached.size)
            or np.any(np.isnan(proposed[acceptance]))
            or np.any(np.isposinf(proposed[acceptance]))
        ):
            raise RuntimeError("Accepted continuous RJ target values are invalid.")
        cached[indices[acceptance]] = proposed[acceptance]

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
        positions, strengths, mask, chart_ids, surface_uv = (
            self._packed_continuous_surface_state_arrays()
        )
        device = reference.device
        dtype = reference.dtype
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
        }
        return True

    def _clear_continuous_rj_device_state(self) -> None:
        """Release the sweep-local fixed-capacity Torch state mirror."""
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
        # All numerical proposal and acceptance work is batched. This loop only
        # commits variable-length state objects for the accepted particle rows.
        for row in accepted_rows.tolist():
            self.continuous_particles[int(indices[row])].state = IsotopeState(
                num_sources=cardinality,
                surface_chart_ids=charts[row],
                surface_uv=uv[row],
                strengths=q[row],
            )
        return int(accepted_rows.size)

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

    def _record_structural_mh_components(
        self,
        move: str,
        *,
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
        """Accumulate batched exact-MH terms for rejection diagnosis."""
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
                        np.count_nonzero(feasible & finite_all & ~accepted)
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
