"""Joint full-spectrum likelihood algorithms for the PF estimator."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import validate_orientation_pair_indices
from pf.estimator_config import (
    _strict_config_number,
    _strict_nonnegative_integer,
)
from pf.estimator_types import JointStationObservation
from pf.full_spectrum import (
    FullSpectrumGenerativeModel,
    validate_observed_spectrum,
)
from pf.particle_filter import (
    IsotopeParticleFilter,
    StructuralGeometryBatch,
)

if TYPE_CHECKING:
    import torch


JOINT_HISTORY_STATION_ACTION_BATCH_SIZE = 4
JOINT_DEVICE_UNIT_TRANSPORT_CACHE_MAX_BYTES = 268_435_456


class JointLikelihoodMixin:
    """Provide batched joint-station construction and likelihood evaluation."""

    @staticmethod
    def _joint_device_unit_transport_state_signature(
        *,
        positions: NDArray[np.float64],
        active_mask: NDArray[np.bool_],
        chart_ids: NDArray[np.int64],
    ) -> str:
        """Hash the exact fixed-slot geometry represented by one PF state."""
        digest = hashlib.sha256(b"joint_device_unit_transport_state_v1\0")
        mask = np.ascontiguousarray(active_mask, dtype=np.bool_)
        active_positions = np.ascontiguousarray(
            np.asarray(positions, dtype=np.float64)[mask],
            dtype=np.float64,
        )
        active_chart_ids = np.ascontiguousarray(
            np.asarray(chart_ids, dtype=np.int64)[mask],
            dtype=np.int64,
        )
        for values in (mask, active_positions, active_chart_ids):
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _joint_device_unit_transport_entry_bytes(
        components: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
    ) -> int:
        """Return live tensor storage owned by one device mirror entry."""
        return int(
            sum(
                int(value.numel()) * int(value.element_size())
                for value in components
            )
        )

    @property
    def joint_device_unit_transport_cache_bytes(self) -> int:
        """Return live tensor bytes retained by the device unit mirror."""
        cache = getattr(self, "_joint_device_unit_transport_cache", {})
        return int(
            sum(
                self._joint_device_unit_transport_entry_bytes(components)
                for components in cache.values()
            )
        )

    def _joint_cached_device_unit_transport(
        self,
        cache_key: tuple[str, str, str, str, str],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"] | None:
        """Return one exact device mirror entry and refresh its own LRU age.

        The host sparse-unit cache and this dense CUDA mirror have independent
        eviction schedules.  A CUDA hit intentionally does not touch or copy
        host values; later eviction from both caches can cost recomputation but
        cannot change transport values or reuse stale geometry.
        """
        cache = getattr(self, "_joint_device_unit_transport_cache", None)
        if cache is None:
            cache = {}
            self._joint_device_unit_transport_cache = cache
        cached = cache.pop(cache_key, None)
        if cached is None:
            self.last_joint_device_unit_cache_misses = int(
                getattr(self, "last_joint_device_unit_cache_misses", 0)
            ) + 1
            return None
        cache[cache_key] = cached
        self.last_joint_device_unit_cache_hits = int(
            getattr(self, "last_joint_device_unit_cache_hits", 0)
        ) + 1
        return cached

    def _store_joint_device_unit_transport(
        self,
        cache_key: tuple[str, str, str, str, str],
        components: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Store one exact device mirror entry under a strict byte-bound LRU."""
        cache = getattr(self, "_joint_device_unit_transport_cache", None)
        if cache is None:
            cache = {}
            self._joint_device_unit_transport_cache = cache
        cache.pop(cache_key, None)
        entry_bytes = self._joint_device_unit_transport_entry_bytes(components)
        if entry_bytes > JOINT_DEVICE_UNIT_TRANSPORT_CACHE_MAX_BYTES:
            return components
        while (
            cache
            and self.joint_device_unit_transport_cache_bytes + entry_bytes
            > JOINT_DEVICE_UNIT_TRANSPORT_CACHE_MAX_BYTES
        ):
            oldest_key = next(iter(cache))
            cache.pop(oldest_key)
        cache[cache_key] = components
        if (
            self.joint_device_unit_transport_cache_bytes
            > JOINT_DEVICE_UNIT_TRANSPORT_CACHE_MAX_BYTES
        ):
            raise RuntimeError("Device unit-transport cache exceeded its byte bound.")
        return components

    @staticmethod
    def _joint_dense_unit_transport_numpy(
        unit_components: tuple[NDArray[np.float64], ...],
        *,
        active_mask: NDArray[np.bool_],
        particle_count: int,
        slot_count: int,
        view_count: int,
        local_line_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Expand sparse active-source units into the fixed-slot layout."""
        if len(unit_components) != 6:
            raise RuntimeError("Unit transport must contain six physical components.")
        dense = np.zeros(
            (
                particle_count,
                slot_count,
                view_count,
                local_line_count,
                6,
            ),
            dtype=np.float64,
        )
        sparse = np.stack(unit_components, axis=-1)
        dense[active_mask] = np.transpose(sparse, (1, 0, 2, 3))
        view_major = np.transpose(dense, (0, 2, 1, 3, 4))
        return view_major[..., 0], view_major[..., 1], view_major[..., 2:]

    @staticmethod
    def _validate_torch_transport_components(
        total: "torch.Tensor",
        uncollided: "torch.Tensor",
        features: "torch.Tensor",
        *,
        error_message: str,
    ) -> None:
        """Validate one device transport batch with a single host sync."""
        import torch

        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(total)),
                torch.any(~torch.isfinite(uncollided)),
                torch.any(~torch.isfinite(features)),
                torch.any(total < 0.0),
                torch.any(uncollided < 0.0),
            )
        ).any()
        if bool(invalid.item()):
            raise RuntimeError(error_message)

    def _full_spectrum_model(self) -> FullSpectrumGenerativeModel:
        """Return the required independently validated generative model."""
        return self.full_spectrum_generative_model

    def _joint_station_from_spectrum_records(
        self,
        records: Sequence[Sequence[object]],
        *,
        pose_idx: int,
        station_sequence_id: int,
        generative_contract_hash_sha256: str,
    ) -> JointStationObservation:
        """Build one strict view-major full-spectrum station observation."""
        if not records:
            raise ValueError("A joint station must contain at least one view.")
        model = self._full_spectrum_model()
        if not isinstance(generative_contract_hash_sha256, str):
            raise TypeError("generative_contract_hash_sha256 must be a JSON string.")
        supplied_hash = generative_contract_hash_sha256
        if supplied_hash != model.contract_hash_sha256:
            raise ValueError(
                "Station full-spectrum contract hash differs from the active "
                "generative model."
            )
        bin_count = int(np.asarray(model.energy_axis_keV).size)
        spectra: list[NDArray[np.float64]] = []
        raw_fe_indices: list[int] = []
        raw_pb_indices: list[int] = []
        live_times = np.empty(len(records), dtype=np.float64)
        for view_index, record in enumerate(records):
            if len(record) != 4:
                raise ValueError(
                    "Full-spectrum station records must have exactly four "
                    "fields: (spectrum, Fe, Pb, live time)."
                )
            spectrum, fe_index, pb_index, live_time_s = record
            raw_spectrum = np.asarray(spectrum)
            if raw_spectrum.ndim != 1:
                raise ValueError(
                    "Each full-spectrum station record must contain one "
                    "one-dimensional raw spectrum."
                )
            validated = validate_observed_spectrum(
                raw_spectrum[np.newaxis, :],
                expected_bin_count=bin_count,
            )
            spectra.append(validated[0])
            raw_fe_indices.append(
                _strict_nonnegative_integer(
                    fe_index,
                    name=f"records[{view_index}].fe_index",
                )
            )
            raw_pb_indices.append(
                _strict_nonnegative_integer(
                    pb_index,
                    name=f"records[{view_index}].pb_index",
                )
            )
            resolved_live_time = _strict_config_number(
                live_time_s,
                name=f"records[{view_index}].live_time_s",
            )
            if resolved_live_time <= 0.0:
                raise ValueError("Full-spectrum station live times must be positive.")
            live_times[view_index] = resolved_live_time
        fe_indices, pb_indices = validate_orientation_pair_indices(
            np.asarray(raw_fe_indices),
            np.asarray(raw_pb_indices),
            orientation_count=int(self.num_orientations),
            expected_count=len(records),
        )
        resolved_pose_idx = _strict_nonnegative_integer(
            pose_idx,
            name="pose_idx",
        )
        resolved_station_sequence_id = _strict_nonnegative_integer(
            station_sequence_id,
            name="station_sequence_id",
        )
        return JointStationObservation(
            spectrum_vb=np.ascontiguousarray(np.stack(spectra, axis=0)),
            energy_axis_keV=np.ascontiguousarray(
                np.asarray(model.energy_axis_keV, dtype=np.float64)
            ),
            generative_contract_hash_sha256=supplied_hash,
            pose_idx=resolved_pose_idx,
            detector_position_xyz_m=self._registered_detector_position_xyz(
                resolved_pose_idx
            ),
            fe_indices=np.ascontiguousarray(fe_indices),
            pb_indices=np.ascontiguousarray(pb_indices),
            live_times_s=np.ascontiguousarray(live_times),
            station_sequence_id=resolved_station_sequence_id,
        )

    def _joint_station_expected_means_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Return predicted spectra shaped particle x view x energy bin."""
        model = self._full_spectrum_model()
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
        )
        result = model.predict_mean_torch(
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        expected_shape = (
            len(self.filters[self.joint_isotope_order()[0]].continuous_particles),
            int(station.fe_indices.size),
            int(station.energy_axis_keV.size),
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "Joint expected-spectrum tensor has an invalid aligned shape."
            )
        return result

    def _joint_line_layout(
        self,
    ) -> dict[
        str,
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
    ]:
        """Return global columns, isotope line indices, and branching weights."""
        model = self._full_spectrum_model()
        line_identity = tuple(model.line_identity)
        layout: dict[
            str,
            tuple[
                NDArray[np.int64],
                NDArray[np.int64],
                NDArray[np.float64],
            ],
        ] = {}
        for isotope in self.joint_isotope_order():
            global_columns = np.asarray(
                [
                    column
                    for column, payload in enumerate(line_identity)
                    if str(payload["isotope"]) == isotope
                ],
                dtype=np.int64,
            )
            local_indices = np.asarray(
                [
                    int(line_identity[int(column)]["transport_line_index"])
                    for column in global_columns
                ],
                dtype=np.int64,
            )
            branching_weights = np.asarray(
                [
                    float(line_identity[int(column)]["branching_weight"])
                    for column in global_columns
                ],
                dtype=np.float64,
            )
            if (
                global_columns.size == 0
                or np.unique(local_indices).size != local_indices.size
                or np.any(local_indices < 0)
                or np.any(~np.isfinite(branching_weights))
                or np.any(branching_weights <= 0.0)
            ):
                raise RuntimeError(
                    f"Full-spectrum line layout is invalid for {isotope!r}."
                )
            configured_weights = self.filters[
                isotope
            ].continuous_kernel.line_branching_weights(
                isotope,
                local_indices,
            )
            if not np.allclose(
                configured_weights,
                branching_weights / float(np.sum(branching_weights)),
                rtol=1.0e-12,
                atol=1.0e-15,
            ):
                raise RuntimeError(
                    "Full-spectrum branching weights differ from the physical "
                    f"kernel for {isotope!r}."
                )
            layout[isotope] = (
                global_columns,
                local_indices,
                branching_weights,
            )
        covered = np.concatenate([value[0] for value in layout.values()])
        active_names = frozenset(self.joint_isotope_order())
        expected = np.asarray(
            [
                column
                for column, payload in enumerate(line_identity)
                if str(payload["isotope"]) in active_names
            ],
            dtype=np.int64,
        )
        if not np.array_equal(
            np.sort(covered),
            expected,
        ):
            raise RuntimeError(
                "Full-spectrum line layout does not cover every active-isotope "
                "global line."
            )
        return layout

    def _joint_isotope_station_transport_components_torch(
        self,
        station: JointStationObservation,
        isotope: str,
        *,
        particle_indices: NDArray[np.int64] | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return selected isotope rows in the global fixed-slot layout."""
        import torch

        self._assert_joint_particle_alignment()
        model = self._full_spectrum_model()
        layout = self._joint_line_layout()
        line_count = len(tuple(model.line_identity))
        feature_count = len(tuple(model.transport_feature_order))
        isotope_key = str(isotope)
        if isotope_key not in self.joint_isotope_order():
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        global_columns, local_indices, branching_weights = layout[isotope_key]
        filt = self.filters[isotope_key]
        (
            positions,
            strengths,
            active_mask,
            chart_ids,
            _surface_uv,
        ) = filt._packed_continuous_surface_state_arrays()
        if particle_indices is not None:
            raw_indices = np.asarray(particle_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(
                raw_indices.dtype, np.integer
            ):
                raise ValueError("particle_indices must be a 1-D integer array.")
            indices = np.asarray(raw_indices, dtype=np.int64)
            if (
                np.unique(indices).size != indices.size
                or np.any(indices < 0)
                or np.any(indices >= positions.shape[0])
            ):
                raise ValueError("particle_indices contain invalid PF rows.")
            positions = positions[indices]
            strengths = strengths[indices]
            active_mask = active_mask[indices]
            chart_ids = chart_ids[indices]
        view_count = int(station.fe_indices.size)
        station_geometry = StructuralGeometryBatch(
            detector_positions=np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                view_count,
                axis=0,
            ),
            fe_indices=station.fe_indices,
            pb_indices=station.pb_indices,
            live_times=station.live_times_s,
            station_sequence_ids=np.full(
                view_count,
                int(station.station_sequence_id),
                dtype=np.int64,
            ),
        )
        particle_count, slot_count = active_mask.shape
        local_line_count = int(local_indices.size)
        device = torch.device("cpu")
        if filt._can_use_gpu():
            from pf import gpu_utils

            device = gpu_utils.resolve_device(filt.config.gpu_device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float64
        cached_units = None
        cache_key = None
        if device.type == "cuda":
            geometry_signature = self._joint_structural_unit_cache_signature(
                filt=filt,
                data=station_geometry,
                positive_line_indices=local_indices,
            )
            state_signature = self._joint_device_unit_transport_state_signature(
                positions=positions,
                active_mask=active_mask,
                chart_ids=chart_ids,
            )
            cache_key = (
                isotope_key,
                geometry_signature,
                state_signature,
                str(device),
                str(dtype),
            )
            cached_units = self._joint_cached_device_unit_transport(cache_key)
        dense_units = None
        if cached_units is None:
            unit_components = self._joint_cached_continuous_unit_components(
                filt=filt,
                data=station_geometry,
                positions_s3=positions[active_mask],
                chart_ids_s=chart_ids[active_mask],
                positive_line_indices=local_indices,
            )
            dense_units = self._joint_dense_unit_transport_numpy(
                unit_components,
                active_mask=active_mask,
                particle_count=particle_count,
                slot_count=slot_count,
                view_count=view_count,
                local_line_count=local_line_count,
            )
            if cache_key is not None:
                cached_units = self._store_joint_device_unit_transport(
                    cache_key,
                    (
                        torch.as_tensor(
                            dense_units[0],
                            dtype=dtype,
                            device=device,
                        ),
                        torch.as_tensor(
                            dense_units[1],
                            dtype=dtype,
                            device=device,
                        ),
                        torch.as_tensor(
                            dense_units[2],
                            dtype=dtype,
                            device=device,
                        ),
                    ),
                )
        if cached_units is not None:
            unit_total, unit_uncollided, feature_local = cached_units
            strength_tensor = torch.as_tensor(
                strengths,
                dtype=dtype,
                device=device,
            )[:, None, :, None]
            branch_tensor = torch.as_tensor(
                branching_weights,
                dtype=dtype,
                device=device,
            ).reshape(1, 1, 1, -1)
            total_local = unit_total * strength_tensor * branch_tensor
            uncollided_local = unit_uncollided * strength_tensor * branch_tensor
        else:
            if dense_units is None:
                raise RuntimeError("Host unit-transport assembly is unavailable.")
            branch_numpy = branching_weights.reshape(1, 1, 1, -1)
            strength_numpy = strengths[:, None, :, None]
            total_local = torch.as_tensor(
                dense_units[0] * strength_numpy * branch_numpy,
                dtype=dtype,
                device=device,
            )
            uncollided_local = torch.as_tensor(
                dense_units[1] * strength_numpy * branch_numpy,
                dtype=dtype,
                device=device,
            )
            feature_local = torch.as_tensor(
                dense_units[2],
                dtype=dtype,
                device=device,
            )
        expected_local_shape = tuple(total_local.shape)
        expected_slots = self.pf_config.cardinality_capacity
        if (
            total_local.ndim != 4
            or int(total_local.shape[2]) != expected_slots
            or tuple(uncollided_local.shape) != expected_local_shape
            or tuple(feature_local.shape) != expected_local_shape + (feature_count,)
            or int(total_local.shape[-1]) != int(local_indices.size)
        ):
            raise RuntimeError(
                "Full-spectrum isotope transport must use the configured "
                "fixed source-slot layout."
            )
        global_total = torch.zeros(
            (*total_local.shape[:-1], line_count),
            dtype=torch.float64,
            device=total_local.device,
        )
        global_uncollided = torch.zeros_like(global_total)
        global_features = torch.zeros(
            (*total_local.shape[:-1], line_count, feature_count),
            dtype=torch.float64,
            device=total_local.device,
        )
        global_total[..., global_columns] = total_local
        global_uncollided[..., global_columns] = uncollided_local
        global_features[..., global_columns, :] = feature_local
        self._validate_torch_transport_components(
            global_total,
            global_uncollided,
            global_features,
            error_message=(
                "Full-spectrum transport components must be finite, "
                "nonnegative source-slot contributions."
            ),
        )
        return global_total, global_uncollided, global_features

    def _joint_station_transport_components_torch(
        self,
        station: JointStationObservation,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return source-resolved total, uncollided, and geometry features."""
        total_parts: list["torch.Tensor"] = []
        uncollided_parts: list["torch.Tensor"] = []
        feature_parts: list["torch.Tensor"] = []
        reference_device = None
        for isotope in self.joint_isotope_order():
            total_local, uncollided_local, feature_local = (
                self._joint_isotope_station_transport_components_torch(
                    station,
                    isotope,
                )
            )
            if reference_device is None:
                reference_device = total_local.device
            elif total_local.device != reference_device:
                total_local = total_local.to(device=reference_device)
                uncollided_local = uncollided_local.to(device=reference_device)
                feature_local = feature_local.to(device=reference_device)
            total_parts.append(total_local)
            uncollided_parts.append(uncollided_local)
            feature_parts.append(feature_local)
        if not total_parts:
            raise RuntimeError(
                "Joint transport components require configured isotopes."
            )
        import torch

        total = torch.cat(total_parts, dim=2)
        uncollided = torch.cat(uncollided_parts, dim=2)
        features = torch.cat(feature_parts, dim=2)
        self._validate_torch_transport_components(
            total,
            uncollided,
            features,
            error_message=(
                "Full-spectrum transport components must be finite, "
                "nonnegative source-slot contributions."
            ),
        )
        return total, uncollided, features

    def _joint_station_expected_means_np(
        self,
        station: JointStationObservation,
    ) -> NDArray[np.float64]:
        """Return the NumPy equivalent of aligned station particle means."""
        return (
            self._joint_station_expected_means_torch(station)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    def _joint_station_log_likelihood_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Evaluate the sole joint full-spectrum likelihood for all particles."""
        model = self._full_spectrum_model()
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
        )
        result = model.log_likelihood_torch(
            station.spectrum_vb,
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        if tuple(result.shape) != (int(total.shape[0]),):
            raise RuntimeError(
                "Full-spectrum likelihood must return one value per particle."
            )
        import torch

        status = (
            torch.stack(
                (
                    torch.any(torch.isnan(result)),
                    torch.any(torch.isinf(result) & (result > 0.0)),
                    torch.any(torch.isfinite(result)),
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        if bool(status[0]) or bool(status[1]):
            raise RuntimeError(
                "Full-spectrum likelihood contains NaN or positive infinity."
            )
        if not bool(status[2]):
            raise RuntimeError(
                "Full-spectrum likelihood is negative infinity for every "
                "particle; the observation is outside model support."
            )
        return result

    def _joint_station_prefix_log_likelihood_torch(
        self,
        station: JointStationObservation,
    ) -> "torch.Tensor":
        """Evaluate exact shared-latent likelihoods for all view prefixes."""
        model = self._full_spectrum_model()
        total, uncollided, features = self._joint_station_transport_components_torch(
            station
        )
        result = model.prefix_log_likelihood_torch(
            station.spectrum_vb,
            total,
            uncollided,
            features,
            station.live_times_s,
        )
        expected_shape = (
            int(station.fe_indices.size) + 1,
            int(total.shape[0]),
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "Full-spectrum prefix likelihood returned an invalid shape."
            )
        import torch

        status = (
            torch.stack(
                (
                    torch.any(torch.isnan(result)),
                    torch.any(torch.isinf(result) & (result > 0.0)),
                    torch.all(result[0] == 0.0),
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        if bool(status[0]) or bool(status[1]):
            raise RuntimeError(
                "Full-spectrum prefix likelihood is numerically invalid."
            )
        if not bool(status[2]):
            raise RuntimeError(
                "The empty full-spectrum prefix must have zero log likelihood."
            )
        return result

    def _joint_history_structural_geometry(
        self,
        isotope: str,
        stations: Sequence[JointStationObservation],
    ) -> StructuralGeometryBatch:
        """Build geometry-only evidence for exact conditional RJ proposals."""
        isotope_key = str(isotope)
        order = self.joint_isotope_order()
        if isotope_key not in order:
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        total_rows = sum(int(station.fe_indices.size) for station in stations)
        if total_rows <= 0:
            raise ValueError("Joint RJ history requires at least one station row.")
        detector_positions = np.concatenate(
            [
                np.repeat(
                    np.asarray(
                        station.detector_position_xyz_m,
                        dtype=np.float64,
                    ).reshape(1, 3),
                    int(station.fe_indices.size),
                    axis=0,
                )
                for station in stations
            ],
            axis=0,
        )
        fe_indices = np.concatenate(
            [np.asarray(station.fe_indices, dtype=np.int64) for station in stations]
        )
        pb_indices = np.concatenate(
            [np.asarray(station.pb_indices, dtype=np.int64) for station in stations]
        )
        live_times = np.concatenate(
            [np.asarray(station.live_times_s, dtype=np.float64) for station in stations]
        )
        sequence_ids = np.concatenate(
            [
                np.full(
                    int(station.fe_indices.size),
                    int(station.station_sequence_id),
                    dtype=np.int64,
                )
                for station in stations
            ]
        )
        return StructuralGeometryBatch(
            detector_positions=np.ascontiguousarray(detector_positions),
            fe_indices=np.ascontiguousarray(fe_indices),
            pb_indices=np.ascontiguousarray(pb_indices),
            live_times=np.ascontiguousarray(live_times),
            station_sequence_ids=np.ascontiguousarray(sequence_ids),
        )

    def _validate_joint_structural_geometry(
        self,
        data: StructuralGeometryBatch,
        stations: Sequence[JointStationObservation],
    ) -> None:
        """Require exact row-wise agreement with the active station history."""
        active_geometry = self._active_joint_structural_geometry
        if active_geometry is not None:
            if data is not active_geometry:
                raise ValueError(
                    "Conditional isotope evidence is not the immutable active "
                    "joint-history geometry."
                )
            return
        row_start = 0
        for station in stations:
            row_count = int(np.asarray(station.fe_indices).size)
            row_stop = row_start + row_count
            row_slice = slice(row_start, row_stop)
            expected_positions = np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                row_count,
                axis=0,
            )
            if not (
                np.array_equal(
                    data.detector_positions[row_slice],
                    expected_positions,
                )
                and np.array_equal(
                    data.fe_indices[row_slice],
                    np.asarray(station.fe_indices, dtype=np.int64),
                )
                and np.array_equal(
                    data.pb_indices[row_slice],
                    np.asarray(station.pb_indices, dtype=np.int64),
                )
                and np.array_equal(
                    data.live_times[row_slice],
                    np.asarray(station.live_times_s, dtype=np.float64),
                )
                and np.array_equal(
                    data.station_sequence_ids[row_slice],
                    np.full(
                        row_count,
                        int(station.station_sequence_id),
                        dtype=np.int64,
                    ),
                )
            ):
                raise ValueError(
                    "Conditional isotope evidence geometry differs from the "
                    "active joint station history."
                )
            row_start = row_stop
        if row_start != data.row_count:
            raise ValueError(
                "Conditional isotope evidence row count differs from the "
                "active joint station history."
            )

    def _refresh_joint_structural_transport_cache(
        self,
        stations: Sequence[JointStationObservation],
    ) -> None:
        """Cache source-resolved transport components for conditional RJ.

        CUDA runs retain the immutable station history on the device for the
        whole Gibbs sweep.  Candidate states are much smaller than this cache,
        so keeping the history resident removes repeated device-to-host and
        host-to-device copies without changing any transport or likelihood
        arithmetic.
        """
        for filt in self.filters.values():
            filt._clear_continuous_rj_device_state()
        active = tuple(stations)
        station_signature = tuple(
            self._joint_station_cache_signature(station) for station in active
        )
        state_sha256 = self._joint_structural_state_sha256()
        persistent = self._joint_persistent_structural_transport_cache
        persistent_signature = self._joint_persistent_structural_station_signature
        if (
            persistent is not None
            and self._joint_persistent_structural_state_sha256 == state_sha256
            and persistent_signature == station_signature
        ):
            self._joint_structural_transport_cache = persistent
            self.last_joint_persistent_cache_reuse_count += 1
            return
        can_append = (
            persistent is not None
            and self._joint_persistent_structural_state_sha256 == state_sha256
            and len(persistent_signature) < len(station_signature)
            and station_signature[: len(persistent_signature)] == persistent_signature
        )
        pending_stations = active[len(persistent_signature) :] if can_append else active
        station_components = [
            self._joint_station_transport_components_torch(station)
            for station in pending_stations
        ]
        if not station_components:
            raise RuntimeError("Structural cache refresh has no station data.")
        if self.pf_config.use_gpu:
            import torch

            appended = tuple(
                torch.cat(
                    [components[index] for components in station_components],
                    dim=1,
                ).contiguous()
                for index in range(3)
            )
        else:
            appended = tuple(
                np.concatenate(
                    [
                        components[index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                        for components in station_components
                    ],
                    axis=1,
                )
                for index in range(3)
            )
        if can_append:
            if hasattr(persistent[0], "detach"):
                import torch

                refreshed = tuple(
                    torch.cat((old, new), dim=1).contiguous()
                    for old, new in zip(persistent, appended, strict=True)
                )
            else:
                refreshed = tuple(
                    np.concatenate((old, new), axis=1)
                    for old, new in zip(persistent, appended, strict=True)
                )
            self.last_joint_persistent_cache_append_count += 1
        else:
            refreshed = appended
        self._joint_structural_transport_cache = refreshed
        self._joint_persistent_structural_transport_cache = refreshed
        self._joint_persistent_structural_station_signature = station_signature
        self._joint_persistent_structural_state_sha256 = state_sha256

    def _joint_structural_state_sha256(self) -> str:
        """Hash compact accepted chart/UV/strength state without transport."""
        digest = hashlib.sha256()
        digest.update(b"joint_structural_accepted_state_v1")
        for isotope in self.joint_isotope_order():
            filt = self.filters[isotope]
            _, strengths, mask, chart_ids, surface_uv = (
                filt._packed_continuous_surface_state_arrays()
            )
            digest.update(str(isotope).encode("utf-8"))
            for values in (strengths, mask, chart_ids, surface_uv):
                array = np.ascontiguousarray(values)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _joint_station_cache_signature(
        station: JointStationObservation,
    ) -> str:
        """Return the immutable geometry signature of one station cache slab."""
        digest = hashlib.sha256()
        digest.update(b"joint_station_transport_geometry_v1")
        digest.update(
            np.asarray(
                station.detector_position_xyz_m,
                dtype=np.float64,
            ).tobytes()
        )
        for values in (
            station.fe_indices,
            station.pb_indices,
            station.live_times_s,
        ):
            array = np.ascontiguousarray(values)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        digest.update(
            np.asarray(
                [int(station.station_sequence_id)],
                dtype=np.int64,
            ).tobytes()
        )
        return digest.hexdigest()

    @classmethod
    def _joint_station_cache_signatures(
        cls,
        stations: Sequence[JointStationObservation],
    ) -> tuple[str, ...] | None:
        """Return station signatures, or disable persistence for debug stubs."""
        try:
            return tuple(
                cls._joint_station_cache_signature(station) for station in stations
            )
        except AttributeError:
            return None

    def _refresh_joint_structural_transport_cache_isotope(
        self,
        stations: Sequence[JointStationObservation],
        isotope: str,
        *,
        particle_indices: NDArray[np.int64] | None = None,
    ) -> None:
        """Refresh moved rows of one isotope's accepted-state cache slice."""
        cache = self._joint_structural_transport_cache
        if cache is None:
            raise RuntimeError(
                "Incremental structural transport refresh requires a cache."
            )
        order = self.joint_isotope_order()
        isotope_key = str(isotope)
        if isotope_key not in order:
            raise KeyError(f"Unknown joint PF isotope: {isotope_key!r}.")
        if particle_indices is None:
            indices = np.arange(
                int(self.pf_config.num_particles),
                dtype=np.int64,
            )
        else:
            raw_indices = np.asarray(particle_indices)
            if raw_indices.ndim != 1 or not np.issubdtype(
                raw_indices.dtype, np.integer
            ):
                raise ValueError("particle_indices must be a 1-D integer array.")
            indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.size == 0:
            return
        station_components = [
            self._joint_isotope_station_transport_components_torch(
                station,
                isotope_key,
                particle_indices=indices,
            )
            for station in stations
        ]
        cache_is_torch = hasattr(cache[0], "detach")
        if cache_is_torch:
            import torch

            refreshed = tuple(
                torch.cat(
                    [components[index] for components in station_components],
                    dim=1,
                ).contiguous()
                for index in range(3)
            )
        else:
            refreshed = tuple(
                np.concatenate(
                    [
                        components[index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                        for components in station_components
                    ],
                    axis=1,
                )
                for index in range(3)
            )
        slots_per_isotope = self.pf_config.cardinality_capacity
        slot_start = order.index(isotope_key) * slots_per_isotope
        slot_stop = slot_start + slots_per_isotope
        mutable_cache = list(cache)
        for cached_values, refreshed_values in zip(
            mutable_cache, refreshed, strict=True
        ):
            if (
                int(cached_values.shape[0]) != int(self.pf_config.num_particles)
                or int(refreshed_values.shape[0]) != int(indices.size)
                or int(cached_values.shape[1]) != int(refreshed_values.shape[1])
                or int(cached_values.shape[2]) < slot_stop
                or int(refreshed_values.shape[2]) != slots_per_isotope
                or tuple(cached_values.shape[3:]) != tuple(refreshed_values.shape[3:])
            ):
                raise RuntimeError(
                    "Incremental isotope transport cache shapes disagree."
                )
            if cache_is_torch:
                import torch

                index_tensor = torch.as_tensor(
                    indices,
                    device=cached_values.device,
                    dtype=torch.long,
                )
                cached_values[:, :, slot_start:slot_stop, ...].index_copy_(
                    0, index_tensor, refreshed_values
                )
            else:
                cached_values[indices, :, slot_start:slot_stop, ...] = refreshed_values
        self._joint_structural_transport_cache = tuple(mutable_cache)
        self.filters[isotope_key]._clear_continuous_rj_device_state()
        station_signature = self._joint_station_cache_signatures(stations)
        if station_signature is None:
            self._joint_persistent_structural_transport_cache = None
            self._joint_persistent_structural_station_signature = ()
            self._joint_persistent_structural_state_sha256 = None
        else:
            self._joint_persistent_structural_transport_cache = (
                self._joint_structural_transport_cache
            )
            self._joint_persistent_structural_station_signature = station_signature
            self._joint_persistent_structural_state_sha256 = (
                self._joint_structural_state_sha256()
            )

    def _full_spectrum_log_likelihood_numpy(
        self,
        *,
        filt: IsotopeParticleFilter,
        station: JointStationObservation,
        total_nvsl: NDArray[np.float64],
        uncollided_nvsl: NDArray[np.float64],
        features_nvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one batched station on GPU when available, else NumPy."""
        model = self._full_spectrum_model()
        total = np.asarray(total_nvsl, dtype=np.float64)
        uncollided = np.asarray(uncollided_nvsl, dtype=np.float64)
        features = np.asarray(features_nvslf, dtype=np.float64)
        if filt._can_use_gpu():
            from pf import gpu_utils
            import torch

            device = gpu_utils.resolve_device(filt.config.gpu_device)
            result = (
                model.log_likelihood_torch(
                    station.spectrum_vb,
                    torch.as_tensor(total, dtype=torch.float64, device=device),
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
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        else:
            result = np.asarray(
                model.log_likelihood_numpy(
                    station.spectrum_vb,
                    total,
                    uncollided,
                    features,
                    station.live_times_s,
                ),
                dtype=np.float64,
            )
        expected_shape = (int(total.shape[0]),)
        if np.asarray(result).shape != expected_shape:
            raise RuntimeError(
                "Full-spectrum conditional likelihood must return one value "
                "per candidate row."
            )
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Full-spectrum conditional likelihood contains NaN or "
                "positive infinity."
            )
        return np.asarray(result, dtype=np.float64)

    def _joint_history_log_likelihood_numpy(
        self,
        *,
        filt: IsotopeParticleFilter,
        stations: Sequence[JointStationObservation],
        total_nvsl: NDArray[np.float64],
        uncollided_nvsl: NDArray[np.float64],
        features_nvslf: NDArray[np.float64],
        target_beta: float,
        newest_prefix_count: int | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate station-independent latent blocks on one batched action axis."""
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint history target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        total = np.asarray(total_nvsl, dtype=np.float64)
        uncollided = np.asarray(uncollided_nvsl, dtype=np.float64)
        features = np.asarray(features_nvslf, dtype=np.float64)
        feature_count = len(tuple(model.transport_feature_order))
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            total.ndim != 4
            or uncollided.shape != total.shape
            or features.shape != total.shape + (feature_count,)
            or total.shape[0] <= 0
            or total.shape[1] != total_views
        ):
            raise ValueError(
                "Joint-history transport arrays must align with every station "
                "view and configured transport feature."
            )
        particle_count = int(total.shape[0])
        result = np.zeros(particle_count, dtype=np.float64)
        prefix_count = None if newest_prefix_count is None else int(newest_prefix_count)
        newest_view_count = int(stations[-1].fe_indices.size)
        if prefix_count is not None and not (1 <= prefix_count <= newest_view_count):
            raise ValueError(
                "newest_prefix_count must identify a nonempty newest-station "
                "view prefix."
            )
        layout_key = (
            tuple(id(station) for station in stations),
            bool(prefix_count is not None),
        )
        cached_layout = self._joint_torch_history_layout_cache.get(layout_key)
        if cached_layout is None:
            newest_slice: slice | None = None
            grouped_lists: dict[
                tuple[int, bytes],
                list[tuple[JointStationObservation, int, int, bool]],
            ] = {}
            view_start = 0
            for station_index, station in enumerate(stations):
                view_count = int(station.fe_indices.size)
                view_stop = view_start + view_count
                if prefix_count is not None and station_index == len(stations) - 1:
                    newest_slice = slice(view_start, view_stop)
                    view_start = view_stop
                    continue
                live_times = np.ascontiguousarray(
                    station.live_times_s,
                    dtype=np.float64,
                )
                if live_times.shape != (view_count,):
                    raise ValueError(
                        "Joint-history station live times must align with views."
                    )
                key = (view_count, live_times.tobytes(order="C"))
                grouped_lists.setdefault(key, []).append(
                    (
                        station,
                        view_start,
                        view_stop,
                        station_index == len(stations) - 1,
                    )
                )
                view_start = view_stop
            grouped = tuple(tuple(entries) for entries in grouped_lists.values())
            cached_layout = (grouped, newest_slice, view_start)
            self._joint_torch_history_layout_cache[layout_key] = cached_layout
        grouped, newest_slice, view_start = cached_layout
        if view_start != total_views:
            raise ValueError(
                "Full-spectrum transport views differ from station history."
            )
        for grouped_entries in grouped:
            entries = tuple(
                entry
                for entry in grouped_entries
                if not (bool(entry[3]) and beta == 0.0)
            )
            if not entries:
                continue
            view_count = int(entries[0][2] - entries[0][1])
            first_start = int(entries[0][1])
            last_stop = int(entries[-1][2])
            contiguous = all(
                int(entry[1]) == first_start + index * view_count
                and int(entry[2]) == first_start + (index + 1) * view_count
                for index, entry in enumerate(entries)
            )

            def _station_action_axis(
                values: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                """Return station x particle x view without scalar station work."""
                trailing_shape = tuple(values.shape[2:])
                if contiguous:
                    block = values[:, first_start:last_stop, ...]
                    reshaped = block.reshape(
                        particle_count,
                        len(entries),
                        view_count,
                        *trailing_shape,
                    )
                    return np.moveaxis(reshaped, 1, 0)
                return np.stack(
                    [
                        values[:, int(entry[1]) : int(entry[2]), ...]
                        for entry in entries
                    ],
                    axis=0,
                )

            observed = np.stack(
                [
                    np.asarray(entry[0].spectrum_vb, dtype=np.float64)
                    for entry in entries
                ],
                axis=0,
            )[:, None, :, :]
            total_group = _station_action_axis(total)
            uncollided_group = _station_action_axis(uncollided)
            feature_group = _station_action_axis(features)
            live_times = np.asarray(
                entries[0][0].live_times_s,
                dtype=np.float64,
            )
            action_chunk_size = min(
                len(entries),
                JOINT_HISTORY_STATION_ACTION_BATCH_SIZE,
            )
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                group_ll = (
                    model.cross_log_likelihood_torch(
                        observed,
                        torch.as_tensor(
                            total_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            uncollided_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            feature_group,
                            dtype=torch.float64,
                            device=device,
                        ),
                        live_times,
                        action_chunk_size=action_chunk_size,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                group_ll = np.asarray(
                    model.cross_log_likelihood_numpy(
                        observed,
                        total_group,
                        uncollided_group,
                        feature_group,
                        live_times,
                        action_chunk_size=action_chunk_size,
                    ),
                    dtype=np.float64,
                )
            expected_shape = (len(entries), 1, particle_count)
            if (
                group_ll.shape != expected_shape
                or np.any(np.isnan(group_ll))
                or np.any(np.isposinf(group_ll))
            ):
                raise RuntimeError(
                    "Batched station-history likelihood returned invalid "
                    "action/sample/state values."
                )
            powers = np.asarray(
                [beta if bool(entry[3]) else 1.0 for entry in entries],
                dtype=np.float64,
            )
            result += np.sum(
                powers[:, None] * group_ll[:, 0, :],
                axis=0,
            )
        if prefix_count is not None:
            if newest_slice is None:
                raise RuntimeError("Newest-station prefix geometry was not selected.")
            station = stations[-1]
            if filt._can_use_gpu():
                from pf import gpu_utils
                import torch

                device = gpu_utils.resolve_device(filt.config.gpu_device)
                prefix_ll = (
                    model.prefix_log_likelihood_torch(
                        station.spectrum_vb,
                        torch.as_tensor(
                            total[:, newest_slice, ...],
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            uncollided[:, newest_slice, ...],
                            dtype=torch.float64,
                            device=device,
                        ),
                        torch.as_tensor(
                            features[:, newest_slice, ...],
                            dtype=torch.float64,
                            device=device,
                        ),
                        station.live_times_s,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
            else:
                prefix_ll = np.asarray(
                    model.prefix_log_likelihood_numpy(
                        station.spectrum_vb,
                        total[:, newest_slice, ...],
                        uncollided[:, newest_slice, ...],
                        features[:, newest_slice, ...],
                        station.live_times_s,
                    ),
                    dtype=np.float64,
                )
            expected_prefix_shape = (
                newest_view_count + 1,
                particle_count,
            )
            if prefix_ll.shape != expected_prefix_shape:
                raise RuntimeError("Newest-station prefix likelihood shape is invalid.")
            result += (1.0 - beta) * prefix_ll[prefix_count - 1] + beta * prefix_ll[
                prefix_count
            ]
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError(
                "Joint history conditional likelihood is numerically invalid."
            )
        return result

    def _joint_history_log_likelihood_torch(
        self,
        *,
        filt: IsotopeParticleFilter,
        stations: Sequence[JointStationObservation],
        total_nvsl: object,
        uncollided_nvsl: object,
        features_nvslf: object,
        target_beta: float,
        newest_prefix_count: int | None = None,
    ) -> object:
        """Evaluate the station history while keeping all state arrays on Torch.

        This is the device-resident equivalent of
        :meth:`_joint_history_log_likelihood_numpy`.  It preserves the same
        station grouping, target powers, model call, and summation order.
        """
        import torch

        if not filt._can_use_gpu():
            raise RuntimeError(
                "Device-resident joint likelihood requires the Torch backend."
            )
        beta = float(target_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Joint history target_beta must lie in [0, 1].")
        model = self._full_spectrum_model()
        total = torch.as_tensor(total_nvsl)
        uncollided = torch.as_tensor(
            uncollided_nvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            features_nvslf,
            device=total.device,
            dtype=total.dtype,
        )
        feature_count = len(tuple(model.transport_feature_order))
        total_views = sum(int(station.fe_indices.size) for station in stations)
        if (
            total.dtype != torch.float64
            or total.ndim != 4
            or tuple(uncollided.shape) != tuple(total.shape)
            or tuple(features.shape) != tuple(total.shape) + (feature_count,)
            or int(total.shape[0]) <= 0
            or int(total.shape[1]) != total_views
        ):
            raise ValueError(
                "Torch joint-history arrays must align with every station "
                "view and configured transport feature."
            )
        particle_count = int(total.shape[0])
        result = torch.zeros(
            particle_count,
            device=total.device,
            dtype=total.dtype,
        )
        prefix_count = None if newest_prefix_count is None else int(newest_prefix_count)
        newest_view_count = int(stations[-1].fe_indices.size)
        if prefix_count is not None and not (1 <= prefix_count <= newest_view_count):
            raise ValueError(
                "newest_prefix_count must identify a nonempty newest-station "
                "view prefix."
            )
        layout_key = (
            tuple(id(station) for station in stations),
            bool(prefix_count is not None),
        )
        cached_layout = self._joint_torch_history_layout_cache.get(layout_key)
        if cached_layout is None:
            newest_slice = None
            grouped_lists: dict[
                tuple[int, bytes],
                list[tuple[JointStationObservation, int, int, bool]],
            ] = {}
            view_start = 0
            for station_index, station in enumerate(stations):
                view_count = int(station.fe_indices.size)
                view_stop = view_start + view_count
                if prefix_count is not None and station_index == len(stations) - 1:
                    newest_slice = slice(view_start, view_stop)
                    view_start = view_stop
                    continue
                live_times = np.ascontiguousarray(
                    station.live_times_s,
                    dtype=np.float64,
                )
                if live_times.shape != (view_count,):
                    raise ValueError(
                        "Joint-history station live times must align with views."
                    )
                key = (view_count, live_times.tobytes(order="C"))
                grouped_lists.setdefault(key, []).append(
                    (
                        station,
                        view_start,
                        view_stop,
                        station_index == len(stations) - 1,
                    )
                )
                view_start = view_stop
            grouped = tuple(tuple(entries) for entries in grouped_lists.values())
            cached_layout = (grouped, newest_slice, view_start)
            self._joint_torch_history_layout_cache[layout_key] = cached_layout
        grouped, newest_slice, view_start = cached_layout
        if view_start != total_views:
            raise ValueError(
                "Full-spectrum transport views differ from station history."
            )
        for grouped_entries in grouped:
            entries = tuple(
                entry
                for entry in grouped_entries
                if not (bool(entry[3]) and beta == 0.0)
            )
            if not entries:
                continue
            view_count = int(entries[0][2] - entries[0][1])
            first_start = int(entries[0][1])
            last_stop = int(entries[-1][2])
            contiguous = all(
                int(entry[1]) == first_start + index * view_count
                and int(entry[2]) == first_start + (index + 1) * view_count
                for index, entry in enumerate(entries)
            )

            def _station_action_axis(values: object) -> object:
                """Return station x particle x view without leaving Torch."""
                tensor = torch.as_tensor(values)
                trailing_shape = tuple(int(value) for value in tensor.shape[2:])
                if contiguous:
                    block = tensor[:, first_start:last_stop, ...]
                    reshaped = block.reshape(
                        particle_count,
                        len(entries),
                        view_count,
                        *trailing_shape,
                    )
                    return torch.movedim(reshaped, 1, 0)
                return torch.stack(
                    [
                        tensor[:, int(entry[1]) : int(entry[2]), ...]
                        for entry in entries
                    ],
                    dim=0,
                )

            observation_key = (
                tuple(id(entry[0]) for entry in entries),
                str(total.device),
                str(total.dtype),
            )
            prepared_observation = self._joint_torch_observation_context_cache.get(
                observation_key
            )
            if prepared_observation is None:
                observed = torch.as_tensor(
                    np.stack(
                        [
                            np.asarray(
                                entry[0].spectrum_vb,
                                dtype=np.float64,
                            )
                            for entry in entries
                        ],
                        axis=0,
                    )[:, None, :, :],
                    device=total.device,
                    dtype=total.dtype,
                )
                prepared_observation = model.prepare_cross_observation_torch(
                    observed,
                    reference=total,
                )
                self._joint_torch_observation_context_cache[observation_key] = (
                    prepared_observation
                )
            else:
                observed = prepared_observation.observed_asvb
            group_ll = model.cross_log_likelihood_torch(
                observed,
                _station_action_axis(total),
                _station_action_axis(uncollided),
                _station_action_axis(features),
                entries[0][0].live_times_s,
                action_chunk_size=min(
                    len(entries),
                    JOINT_HISTORY_STATION_ACTION_BATCH_SIZE,
                ),
                prepared_observation=prepared_observation,
            )
            group_ll = torch.as_tensor(
                group_ll,
                device=total.device,
                dtype=total.dtype,
            )
            expected_shape = (len(entries), 1, particle_count)
            if tuple(group_ll.shape) != expected_shape:
                raise RuntimeError("Torch station-history likelihood shape is invalid.")
            powers = torch.as_tensor(
                [beta if bool(entry[3]) else 1.0 for entry in entries],
                device=total.device,
                dtype=total.dtype,
            )
            result = result + torch.sum(
                powers[:, None] * group_ll[:, 0, :],
                dim=0,
            )
        if prefix_count is not None:
            if newest_slice is None:
                raise RuntimeError("Newest-station prefix geometry was not selected.")
            station = stations[-1]
            prefix_ll = model.prefix_log_likelihood_torch(
                station.spectrum_vb,
                total[:, newest_slice, ...],
                uncollided[:, newest_slice, ...],
                features[:, newest_slice, ...],
                station.live_times_s,
            )
            prefix_ll = torch.as_tensor(
                prefix_ll,
                device=total.device,
                dtype=total.dtype,
            )
            expected_prefix_shape = (
                newest_view_count + 1,
                particle_count,
            )
            if tuple(prefix_ll.shape) != expected_prefix_shape:
                raise RuntimeError(
                    "Torch newest-station prefix likelihood shape is invalid."
                )
            result = result + (
                (1.0 - beta) * prefix_ll[prefix_count - 1]
                + beta * prefix_ll[prefix_count]
            )
        invalid_result = torch.stack(
            (
                torch.any(torch.isnan(result)),
                torch.any(torch.isinf(result) & (result > 0.0)),
            )
        ).any()
        if bool(invalid_result.item()):
            raise RuntimeError("Torch joint-history likelihood is numerically invalid.")
        return result
