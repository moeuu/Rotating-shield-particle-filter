"""Fixed-capacity source-resolved transport cache for joint exact RJ."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


JOINT_EXACT_MAX_STATIONS = 16
JOINT_EXACT_MAX_VIEWS = 128
JOINT_REINDEX_VIEW_CHUNK_SIZE = 8


@dataclass(frozen=True)
class JointSlotProposalOverlay:
    """Hold one immutable CUDA slot replacement across TPHT refinement."""

    replacement: tuple[object, object, object]
    particle_indices: object
    slot_start: int
    slot_stop: int
    active_slot_mask: object

    def __post_init__(self) -> None:
        """Validate row, slot, backend, and activity alignment once."""
        total, uncollided, features = self.replacement
        JointTransportCache._validate_component_triplet(
            total,
            uncollided,
            features,
        )
        if not _is_torch_tensor(total):
            raise TypeError("A TPHT proposal overlay must remain on Torch.")
        import torch

        indices = self.particle_indices
        if (
            not torch.is_tensor(indices)
            or indices.device != total.device
            or indices.dtype != torch.long
            or indices.ndim != 1
            or int(indices.numel()) != int(total.shape[0])
        ):
            raise ValueError("Proposal-overlay particle rows are misaligned.")
        if (
            type(self.slot_start) is not int
            or type(self.slot_stop) is not int
            or self.slot_start < 0
            or self.slot_stop <= self.slot_start
            or self.slot_stop - self.slot_start != int(total.shape[2])
        ):
            raise ValueError("Proposal-overlay slot bounds are invalid.")
        JointTransportCache.validate_replacement_slot_activity(
            self.replacement,
            active_slot_mask=self.active_slot_mask,
        )

    @property
    def row_count(self) -> int:
        """Return the number of proposal rows in the overlay."""
        return int(self.replacement[0].shape[0])

    @property
    def view_count(self) -> int:
        """Return the complete acquired-history view count."""
        return int(self.replacement[0].shape[1])

    def select(self, rows: object, view_slice: slice) -> "JointSlotProposalOverlay":
        """Return a device-resident row/view slice without host conversion."""
        import torch

        indices = torch.as_tensor(
            rows,
            device=self.replacement[0].device,
            dtype=torch.long,
        ).reshape(-1)
        start = 0 if view_slice.start is None else int(view_slice.start)
        stop = self.view_count if view_slice.stop is None else int(view_slice.stop)
        if start < 0 or stop <= start or stop > self.view_count:
            raise IndexError("Proposal-overlay view slice is invalid.")
        selected = tuple(
            torch.index_select(value[:, start:stop, ...], 0, indices).contiguous()
            for value in self.replacement
        )
        return JointSlotProposalOverlay(
            replacement=selected,
            particle_indices=torch.index_select(self.particle_indices, 0, indices),
            slot_start=self.slot_start,
            slot_stop=self.slot_stop,
            active_slot_mask=torch.index_select(
                self.active_slot_mask,
                0,
                indices,
            ),
        )


def _is_torch_tensor(value: object) -> bool:
    """Return whether *value* is a Torch tensor without importing Torch eagerly."""
    return hasattr(value, "detach") and hasattr(value, "device")


def _tensor_nbytes(value: object) -> int:
    """Return the exact storage bytes represented by one dense array."""
    if _is_torch_tensor(value):
        return int(value.numel()) * int(value.element_size())
    array = np.asarray(value)
    return int(array.size) * int(array.dtype.itemsize)


@dataclass
class JointTransportCache(Sequence[object]):
    """Own accepted source-resolved transport for at most 16 stations.

    Backing arrays are allocated once for 128 views.  Sequence behaviour is
    intentionally limited to the three active transport views so existing
    likelihood code can consume the cache without seeing uninitialized future
    capacity.  Metadata and station likelihoods remain part of the same source
    of truth and follow every accepted-state commit and ancestor reindex.
    """

    _storage: tuple[object, object, object]
    station_log_likelihood: object
    valid_view_count: int
    station_offsets: tuple[int, ...]
    station_signatures: tuple[str, ...]
    state_sha256: str
    row_generation: int | None
    max_views: int = JOINT_EXACT_MAX_VIEWS
    max_stations: int = JOINT_EXACT_MAX_STATIONS
    history_append_count: int = 0
    ancestor_reindex_count: int = 0
    slot_overlay_commit_count: int = 0
    _staged_ancestor_sha256: str | None = None
    _staged_ancestor_indices_n: NDArray[np.int64] | None = None

    @staticmethod
    def required_storage_bytes(
        *,
        particle_count: int,
        max_views: int,
        source_slots: int,
        line_count: int,
        feature_count: int,
        max_stations: int,
        dtype_bytes: int = 8,
    ) -> int:
        """Return bytes for the accepted fixed-capacity cache."""
        dimensions = (
            particle_count,
            max_views,
            source_slots,
            line_count,
            feature_count,
            max_stations,
            dtype_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError(
                "Joint transport cache dimensions must be positive integers."
            )
        line_elements = (
            particle_count * max_views * source_slots * line_count
        )
        transport_elements = line_elements * (2 + feature_count)
        station_elements = particle_count * max_stations
        return int((transport_elements + station_elements) * dtype_bytes)

    @staticmethod
    def reindex_scratch_bytes(
        *,
        particle_count: int,
        source_slots: int,
        line_count: int,
        feature_count: int,
        max_stations: int,
        dtype_bytes: int = 8,
    ) -> int:
        """Return peak scratch bytes for one exact chunked ancestor commit."""
        dimensions = (
            particle_count,
            source_slots,
            line_count,
            feature_count,
            max_stations,
            dtype_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError(
                "Joint transport reindex dimensions must be positive integers."
            )
        view_chunk_elements = (
            particle_count
            * source_slots
            * line_count
            * (2 + feature_count)
            * JOINT_REINDEX_VIEW_CHUNK_SIZE
        )
        station_elements = particle_count * max_stations
        return int((view_chunk_elements + station_elements) * dtype_bytes)

    @classmethod
    def allocate_empty_torch(
        cls,
        *,
        particle_count: int,
        source_slots: int,
        line_count: int,
        feature_count: int,
        device: object,
        dtype: object,
        state_sha256: str,
        row_generation: int | None,
        max_views: int = JOINT_EXACT_MAX_VIEWS,
        max_stations: int = JOINT_EXACT_MAX_STATIONS,
    ) -> "JointTransportCache":
        """Reserve the complete live CUDA cache before any acquisition."""
        import torch

        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or dtype != torch.float64:
            raise RuntimeError(
                "Live joint exact cache requires CUDA float64; fallback is "
                "not permitted."
            )
        required_bytes = cls.required_storage_bytes(
            particle_count=particle_count,
            max_views=max_views,
            source_slots=source_slots,
            line_count=line_count,
            feature_count=feature_count,
            max_stations=max_stations,
            dtype_bytes=torch.empty((), dtype=dtype).element_size(),
        )
        free_bytes, _ = torch.cuda.mem_get_info(resolved_device)
        if required_bytes > int(free_bytes):
            raise RuntimeError(
                "Fixed 16-station exact transport cache requires "
                f"{required_bytes} bytes but only {int(free_bytes)} CUDA "
                "bytes are free. CPU fallback is not permitted."
            )
        storage_shapes = (
            (particle_count, max_views, source_slots, line_count),
            (particle_count, max_views, source_slots, line_count),
            (
                particle_count,
                max_views,
                source_slots,
                line_count,
                feature_count,
            ),
        )
        try:
            storage = tuple(
                torch.empty(shape, device=resolved_device, dtype=dtype)
                for shape in storage_shapes
            )
            station_ll = torch.full(
                (particle_count, max_stations),
                float("nan"),
                device=resolved_device,
                dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Fixed joint exact cache allocation failed before acquisition; "
                "CPU fallback is not permitted."
            ) from exc
        return cls(
            _storage=storage,
            station_log_likelihood=station_ll,
            valid_view_count=0,
            station_offsets=(0,),
            station_signatures=(),
            state_sha256=state_sha256,
            row_generation=row_generation,
            max_views=max_views,
            max_stations=max_stations,
        )

    @classmethod
    def allocate(
        cls,
        station_components: tuple[object, object, object],
        *,
        station_signature: str,
        state_sha256: str,
        row_generation: int | None,
        max_views: int = JOINT_EXACT_MAX_VIEWS,
        max_stations: int = JOINT_EXACT_MAX_STATIONS,
    ) -> "JointTransportCache":
        """Allocate fixed backing storage and append the first station slab."""
        total, uncollided, features = station_components
        cls._validate_component_triplet(total, uncollided, features)
        particle_count, view_count, source_slots, line_count = (
            int(value) for value in total.shape
        )
        feature_count = int(features.shape[-1])
        capacity = int(max_views)
        station_capacity = int(max_stations)
        if capacity <= 0 or station_capacity <= 0:
            raise ValueError("Joint transport cache capacities must be positive.")
        if view_count > capacity:
            raise RuntimeError(
                "The first station exceeds the fixed exact-history view capacity."
            )
        if not isinstance(station_signature, str) or not station_signature:
            raise ValueError("A station cache signature must be a nonempty string.")
        if not isinstance(state_sha256, str) or not state_sha256:
            raise ValueError("An accepted-state digest must be a nonempty string.")
        if _is_torch_tensor(total):
            import torch

            if total.dtype != torch.float64:
                raise TypeError("Joint exact transport cache requires torch.float64.")
            storage_shapes = (
                (particle_count, capacity, source_slots, line_count),
                (particle_count, capacity, source_slots, line_count),
                (
                    particle_count,
                    capacity,
                    source_slots,
                    line_count,
                    feature_count,
                ),
            )
            required_bytes = cls.required_storage_bytes(
                particle_count=particle_count,
                max_views=capacity,
                source_slots=source_slots,
                line_count=line_count,
                feature_count=feature_count,
                max_stations=station_capacity,
                dtype_bytes=int(total.element_size()),
            )
            if bool(total.is_cuda):
                free_bytes, _ = torch.cuda.mem_get_info(total.device)
                if required_bytes > int(free_bytes):
                    raise RuntimeError(
                        "Fixed 16-station exact transport cache requires "
                        f"{required_bytes} bytes but only {int(free_bytes)} CUDA "
                        "bytes are free. CPU fallback is not permitted."
                    )
            storage = tuple(
                torch.empty(
                    shape,
                    device=total.device,
                    dtype=total.dtype,
                )
                for shape in storage_shapes
            )
            station_ll = torch.full(
                (particle_count, station_capacity),
                float("nan"),
                device=total.device,
                dtype=total.dtype,
            )
        else:
            arrays = tuple(np.asarray(value) for value in station_components)
            if any(value.dtype != np.float64 for value in arrays):
                raise TypeError("Joint exact transport cache requires numpy.float64.")
            storage = (
                np.empty(
                    (particle_count, capacity, source_slots, line_count),
                    dtype=np.float64,
                ),
                np.empty(
                    (particle_count, capacity, source_slots, line_count),
                    dtype=np.float64,
                ),
                np.empty(
                    (
                        particle_count,
                        capacity,
                        source_slots,
                        line_count,
                        feature_count,
                    ),
                    dtype=np.float64,
                ),
            )
            station_ll = np.full(
                (particle_count, station_capacity),
                np.nan,
                dtype=np.float64,
            )
        cache = cls(
            _storage=storage,
            station_log_likelihood=station_ll,
            valid_view_count=0,
            station_offsets=(0,),
            station_signatures=(),
            state_sha256=state_sha256,
            row_generation=row_generation,
            max_views=capacity,
            max_stations=station_capacity,
        )
        cache.append_station(
            station_components,
            station_signature=station_signature,
            count_as_history_append=False,
        )
        return cache

    @staticmethod
    def _validate_component_triplet(
        total: object,
        uncollided: object,
        features: object,
    ) -> None:
        """Validate one aligned source-resolved transport component triplet."""
        if not all(hasattr(value, "shape") for value in (total, uncollided, features)):
            raise TypeError("Transport cache components must be dense arrays.")
        if len(total.shape) != 4:
            raise ValueError("Transport total must be particle x view x slot x line.")
        if tuple(uncollided.shape) != tuple(total.shape):
            raise ValueError("Uncollided transport must match total transport.")
        if len(features.shape) != 5 or tuple(features.shape[:-1]) != tuple(total.shape):
            raise ValueError("Transport features must extend total transport.")
        if any(int(value) <= 0 for value in total.shape):
            raise ValueError("Transport cache dimensions must be positive.")
        torch_backed = _is_torch_tensor(total)
        if any(
            _is_torch_tensor(value) != torch_backed
            for value in (uncollided, features)
        ):
            raise TypeError("Transport cache components must use one array backend.")
        if torch_backed:
            if any(
                value.device != total.device or value.dtype != total.dtype
                for value in (uncollided, features)
            ):
                raise ValueError(
                    "Transport cache components must share Torch device and dtype."
                )
        else:
            arrays = tuple(np.asarray(value) for value in (total, uncollided, features))
            if any(value.dtype != arrays[0].dtype for value in arrays[1:]):
                raise ValueError("Transport cache components must share NumPy dtype.")

    @property
    def arrays(self) -> tuple[object, object, object]:
        """Return views limited to the immutable acquired-history prefix."""
        stop = int(self.valid_view_count)
        return tuple(value[:, :stop, ...] for value in self._storage)

    @property
    def backing_storage(self) -> tuple[object, object, object]:
        """Return fixed-capacity backing arrays for diagnostics and tests."""
        return self._storage

    @property
    def particle_count(self) -> int:
        """Return the number of aligned accepted PF rows."""
        return int(self._storage[0].shape[0])

    @property
    def station_count(self) -> int:
        """Return the number of appended station slabs."""
        return len(self.station_signatures)

    @property
    def allocated_bytes(self) -> int:
        """Return bytes owned by transport and per-station likelihood storage."""
        return int(
            sum(_tensor_nbytes(value) for value in self._storage)
            + _tensor_nbytes(self.station_log_likelihood)
        )

    def __len__(self) -> int:
        """Return the canonical transport component count."""
        return 3

    def __iter__(self) -> Iterator[object]:
        """Iterate over acquired-history component views."""
        return iter(self.arrays)

    def __getitem__(self, index: int | slice) -> Any:
        """Return one or more acquired-history component views."""
        return self.arrays[index]

    def append_station(
        self,
        station_components: tuple[object, object, object],
        *,
        station_signature: str,
        count_as_history_append: bool = True,
    ) -> None:
        """Write one new station slab without reallocating prior history."""
        total, uncollided, features = station_components
        self._validate_component_triplet(total, uncollided, features)
        if self.station_count >= int(self.max_stations):
            raise RuntimeError("The 16-station exact transport cache is full.")
        if station_signature in self.station_signatures:
            raise RuntimeError("A station signature cannot be appended twice.")
        view_count = int(total.shape[1])
        start = int(self.valid_view_count)
        stop = start + view_count
        if stop > int(self.max_views):
            raise RuntimeError("The 128-view exact transport cache is full.")
        expected_total_shape = (
            self.particle_count,
            view_count,
            int(self._storage[0].shape[2]),
            int(self._storage[0].shape[3]),
        )
        if (
            tuple(total.shape) != expected_total_shape
            or tuple(uncollided.shape) != expected_total_shape
            or tuple(features.shape)
            != expected_total_shape + (int(self._storage[2].shape[-1]),)
        ):
            raise RuntimeError("Appended station transport shape changed mid-run.")
        if _is_torch_tensor(self._storage[0]):
            if any(
                not _is_torch_tensor(value)
                or value.device != self._storage[0].device
                or value.dtype != self._storage[0].dtype
                for value in station_components
            ):
                raise RuntimeError(
                    "Appended station transport changed Torch device or dtype."
                )
            for destination, source in zip(
                self._storage,
                station_components,
                strict=True,
            ):
                destination[:, start:stop, ...].copy_(source)
        else:
            if any(_is_torch_tensor(value) for value in station_components):
                raise RuntimeError("Appended station transport changed array backend.")
            for destination, source in zip(
                self._storage,
                station_components,
                strict=True,
            ):
                destination[:, start:stop, ...] = np.asarray(source)
        self.valid_view_count = stop
        self.station_offsets = (*self.station_offsets, stop)
        self.station_signatures = (*self.station_signatures, station_signature)
        if count_as_history_append:
            self.history_append_count += 1

    def validate_identity(
        self,
        *,
        station_signatures: tuple[str, ...],
        state_sha256: str,
        row_generation: int | None,
    ) -> None:
        """Fail unless cache history and accepted CUDA state are identical."""
        if self.station_signatures != tuple(station_signatures):
            raise RuntimeError("Transport cache belongs to another station history.")
        if self.state_sha256 != state_sha256:
            raise RuntimeError("Transport cache belongs to another accepted PF state.")
        if self.row_generation != row_generation:
            raise RuntimeError("Transport cache row generation is stale.")

    def update_state_identity(
        self,
        *,
        state_sha256: str,
        row_generation: int | None,
    ) -> None:
        """Authenticate an already committed cache against its accepted state."""
        if not isinstance(state_sha256, str) or not state_sha256:
            raise ValueError("An accepted-state digest must be a nonempty string.")
        self.state_sha256 = state_sha256
        self.row_generation = row_generation

    def invalidate_station_likelihood(self, rows: object | None = None) -> None:
        """Mark cached per-station likelihoods stale for selected PF rows."""
        station_count = self.station_count
        if station_count == 0:
            return
        if rows is None:
            self.station_log_likelihood[:, :station_count] = float("nan")
            return
        if _is_torch_tensor(self.station_log_likelihood):
            import torch

            indices = torch.as_tensor(
                rows,
                device=self.station_log_likelihood.device,
                dtype=torch.long,
            ).reshape(-1)
            selected = torch.full(
                (int(indices.numel()), station_count),
                float("nan"),
                device=self.station_log_likelihood.device,
                dtype=self.station_log_likelihood.dtype,
            )
            self.station_log_likelihood[:, :station_count].index_copy_(
                0,
                indices,
                selected,
            )
        else:
            indices = np.asarray(rows, dtype=np.int64).reshape(-1)
            self.station_log_likelihood[indices, :station_count] = np.nan

    def set_station_likelihood(
        self,
        values_ns: object,
        *,
        rows: object | None = None,
    ) -> None:
        """Store exact untempered station likelihoods for accepted rows."""
        station_count = self.station_count
        if _is_torch_tensor(self.station_log_likelihood):
            import torch

            values = torch.as_tensor(
                values_ns,
                device=self.station_log_likelihood.device,
                dtype=self.station_log_likelihood.dtype,
            )
            if rows is None:
                expected_rows = self.particle_count
                indices = None
            else:
                indices = torch.as_tensor(
                    rows,
                    device=self.station_log_likelihood.device,
                    dtype=torch.long,
                ).reshape(-1)
                expected_rows = int(indices.numel())
            if tuple(values.shape) != (expected_rows, station_count):
                raise ValueError("Per-station likelihood shape is inconsistent.")
            if bool(torch.any(torch.isnan(values) | torch.isposinf(values)).item()):
                raise RuntimeError("Per-station likelihood contains invalid values.")
            if indices is None:
                self.station_log_likelihood[:, :station_count].copy_(values)
            else:
                self.station_log_likelihood[:, :station_count].index_copy_(
                    0,
                    indices,
                    values,
                )
        else:
            values = np.asarray(values_ns, dtype=np.float64)
            indices = (
                np.arange(self.particle_count, dtype=np.int64)
                if rows is None
                else np.asarray(rows, dtype=np.int64).reshape(-1)
            )
            if values.shape != (indices.size, station_count):
                raise ValueError("Per-station likelihood shape is inconsistent.")
            if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
                raise RuntimeError("Per-station likelihood contains invalid values.")
            self.station_log_likelihood[indices, :station_count] = values

    def set_station_likelihood_column(
        self,
        values_n: object,
        *,
        station_index: int,
        rows: object | None = None,
    ) -> None:
        """Store one exact untempered station likelihood column."""
        if type(station_index) is not int or not (
            0 <= station_index < self.station_count
        ):
            raise IndexError("Station likelihood column is outside cache history.")
        if _is_torch_tensor(self.station_log_likelihood):
            import torch

            values = torch.as_tensor(
                values_n,
                device=self.station_log_likelihood.device,
                dtype=self.station_log_likelihood.dtype,
            ).reshape(-1)
            indices = (
                torch.arange(
                    self.particle_count,
                    device=self.station_log_likelihood.device,
                    dtype=torch.long,
                )
                if rows is None
                else torch.as_tensor(
                    rows,
                    device=self.station_log_likelihood.device,
                    dtype=torch.long,
                ).reshape(-1)
            )
            if int(values.numel()) != int(indices.numel()):
                raise ValueError("Station likelihood column has the wrong row count.")
            if bool(torch.any(torch.isnan(values) | torch.isposinf(values)).item()):
                raise RuntimeError("Station likelihood column is invalid.")
            self.station_log_likelihood[:, station_index].index_copy_(
                0,
                indices,
                values,
            )
            return
        values = np.asarray(values_n, dtype=np.float64).reshape(-1)
        indices = (
            np.arange(self.particle_count, dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64).reshape(-1)
        )
        if values.size != indices.size:
            raise ValueError("Station likelihood column has the wrong row count.")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise RuntimeError("Station likelihood column is invalid.")
        self.station_log_likelihood[indices, station_index] = values

    def weighted_target(self, *, newest_station_beta: float) -> object:
        """Return the exact history target from cached station likelihoods."""
        beta = float(newest_station_beta)
        if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Newest-station beta must lie in [0, 1].")
        station_count = self.station_count
        if station_count == 0:
            raise RuntimeError("No station likelihood is cached.")
        values = self.station_log_likelihood[:, :station_count]
        if _is_torch_tensor(values):
            import torch

            if bool(torch.any(torch.isnan(values) | torch.isposinf(values)).item()):
                raise RuntimeError("Per-station likelihood cache is stale or invalid.")
            prior = (
                torch.sum(values[:, :-1], dim=1)
                if station_count > 1
                else torch.zeros(
                    self.particle_count,
                    device=values.device,
                    dtype=values.dtype,
                )
            )
            return prior if beta == 0.0 else prior + beta * values[:, -1]
        array = np.asarray(values, dtype=np.float64)
        if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
            raise RuntimeError("Per-station likelihood cache is stale or invalid.")
        prior = (
            np.sum(array[:, :-1], axis=1, dtype=np.float64)
            if station_count > 1
            else np.zeros(self.particle_count, dtype=np.float64)
        )
        return prior if beta == 0.0 else prior + beta * array[:, -1]

    def replace_slot_rows(
        self,
        *,
        rows: object,
        slot_start: int,
        slot_stop: int,
        replacement: tuple[object, object, object],
        active_slot_mask: object,
    ) -> None:
        """Atomically validate then commit one accepted source-slot block."""
        start = int(slot_start)
        stop = int(slot_stop)
        if start < 0 or stop <= start or stop > int(self._storage[0].shape[2]):
            raise ValueError("Committed slot bounds are outside cache capacity.")
        total, uncollided, features = replacement
        self._validate_component_triplet(total, uncollided, features)
        self.validate_replacement_slot_activity(
            replacement,
            active_slot_mask=active_slot_mask,
        )
        if _is_torch_tensor(self._storage[0]):
            import torch

            indices = torch.as_tensor(
                rows,
                device=self._storage[0].device,
                dtype=torch.long,
            ).reshape(-1)
            if int(torch.unique(indices).numel()) != int(indices.numel()):
                raise ValueError("Committed cache rows must be unique.")
        else:
            indices = np.asarray(rows, dtype=np.int64).reshape(-1)
            if np.unique(indices).size != indices.size:
                raise ValueError("Committed cache rows must be unique.")
        row_count = int(
            indices.size
            if isinstance(indices, np.ndarray)
            else indices.numel()
        )
        expected = (
            row_count,
            int(self.valid_view_count),
            stop - start,
            int(self._storage[0].shape[3]),
        )
        if (
            tuple(total.shape) != expected
            or tuple(uncollided.shape) != expected
            or tuple(features.shape)
            != expected + (int(self._storage[2].shape[-1]),)
        ):
            raise RuntimeError("Committed slot replacement shape is inconsistent.")
        if row_count == 0:
            return
        if _is_torch_tensor(self._storage[0]):
            if any(
                not _is_torch_tensor(value)
                or value.device != self._storage[0].device
                or value.dtype != self._storage[0].dtype
                for value in replacement
            ):
                raise RuntimeError(
                    "Committed slot replacement changed device or dtype."
                )
            for destination, source in zip(self._storage, replacement, strict=True):
                destination[:, : self.valid_view_count, start:stop, ...].index_copy_(
                    0,
                    indices,
                    source,
                )
        else:
            for destination, source in zip(self._storage, replacement, strict=True):
                destination[indices, : self.valid_view_count, start:stop, ...] = (
                    np.asarray(source)
                )
        self.invalidate_station_likelihood(indices)
        self.slot_overlay_commit_count += row_count

    @staticmethod
    def validate_replacement_slot_activity(
        replacement: tuple[object, object, object],
        *,
        active_slot_mask: object,
    ) -> None:
        """Reject any nonzero transport stored in an inactive source slot."""
        total, uncollided, features = replacement
        JointTransportCache._validate_component_triplet(
            total,
            uncollided,
            features,
        )
        expected_mask_shape = (int(total.shape[0]), int(total.shape[2]))
        if _is_torch_tensor(total):
            import torch

            if not torch.is_tensor(active_slot_mask):
                raise TypeError("Torch slot activity must be a Torch tensor.")
            if (
                active_slot_mask.device != total.device
                or active_slot_mask.dtype != torch.bool
                or tuple(active_slot_mask.shape) != expected_mask_shape
            ):
                raise ValueError(
                    "Torch slot activity must align by row and replacement slot."
                )
            inactive_line_mask = (~active_slot_mask)[:, None, :, None]
            inactive_feature_mask = inactive_line_mask[..., None]
            invalid = (
                torch.any((total != 0.0) & inactive_line_mask)
                | torch.any((uncollided != 0.0) & inactive_line_mask)
                | torch.any((features != 0.0) & inactive_feature_mask)
            )
            if bool(invalid.item()):
                raise RuntimeError(
                    "Inactive source slots contain nonzero transport."
                )
            return
        mask = np.asarray(active_slot_mask)
        if mask.dtype != np.bool_ or mask.shape != expected_mask_shape:
            raise ValueError(
                "NumPy slot activity must align by row and replacement slot."
            )
        inactive_line_mask = (~mask)[:, None, :, None]
        inactive_feature_mask = inactive_line_mask[..., None]
        if (
            np.any((np.asarray(total) != 0.0) & inactive_line_mask)
            or np.any((np.asarray(uncollided) != 0.0) & inactive_line_mask)
            or np.any((np.asarray(features) != 0.0) & inactive_feature_mask)
        ):
            raise RuntimeError("Inactive source slots contain nonzero transport.")

    def reindex_rows(self, indices_n: NDArray[np.int64]) -> None:
        """Atomically apply one ancestor vector to every cache-owned row."""
        self.stage_reindex_rows(indices_n)
        self.commit_staged_reindex()

    def stage_reindex_rows(self, indices_n: NDArray[np.int64]) -> None:
        """Validate and stage one ancestor vector without mutating the cache."""
        raw = np.asarray(indices_n)
        if self._staged_ancestor_sha256 is not None:
            raise RuntimeError("A cache ancestor reindex is already staged.")
        if raw.dtype != np.int64 or raw.shape != (self.particle_count,):
            raise ValueError("Ancestor indices must be one aligned int64 vector.")
        if np.any(raw < 0) or np.any(raw >= self.particle_count):
            raise IndexError("Ancestor index is outside cache rows.")
        self._staged_ancestor_indices_n = np.ascontiguousarray(raw).copy()
        self._staged_ancestor_sha256 = hashlib.sha256(
            np.ascontiguousarray(raw).tobytes(order="C")
        ).hexdigest()

    def commit_staged_reindex(self) -> None:
        """Commit a staged ancestor vector with bounded exact scratch memory."""
        raw = self._staged_ancestor_indices_n
        if self._staged_ancestor_sha256 is None or raw is None:
            raise RuntimeError("No complete cache ancestor reindex is staged.")
        if _is_torch_tensor(self._storage[0]):
            import torch

            indices = torch.as_tensor(
                raw,
                device=self._storage[0].device,
                dtype=torch.long,
            )
            scratch_view_count = max(
                1,
                min(JOINT_REINDEX_VIEW_CHUNK_SIZE, self.valid_view_count),
            )
            transport_scratch = tuple(
                torch.empty_like(source[:, :scratch_view_count, ...])
                for source in self._storage
            )
            station_scratch = torch.empty_like(
                self.station_log_likelihood[:, : self.station_count]
            )
            for source, scratch in zip(
                self._storage,
                transport_scratch,
                strict=True,
            ):
                for view_start in range(
                    0,
                    self.valid_view_count,
                    JOINT_REINDEX_VIEW_CHUNK_SIZE,
                ):
                    view_stop = min(
                        view_start + JOINT_REINDEX_VIEW_CHUNK_SIZE,
                        self.valid_view_count,
                    )
                    view = source[:, view_start:view_stop, ...]
                    active_scratch = scratch[:, : view_stop - view_start, ...]
                    torch.index_select(view, 0, indices, out=active_scratch)
                    view.copy_(active_scratch)
            torch.index_select(
                self.station_log_likelihood[:, : self.station_count],
                0,
                indices,
                out=station_scratch,
            )
            self.station_log_likelihood[:, : self.station_count].copy_(
                station_scratch
            )
            if bool(self._storage[0].is_cuda):
                torch.cuda.synchronize(self._storage[0].device)
        else:
            scratch_view_count = max(
                1,
                min(JOINT_REINDEX_VIEW_CHUNK_SIZE, self.valid_view_count),
            )
            transport_scratch = tuple(
                np.empty_like(source[:, :scratch_view_count, ...])
                for source in self._storage
            )
            station_scratch = np.empty_like(
                self.station_log_likelihood[:, : self.station_count]
            )
            for source, scratch in zip(
                self._storage,
                transport_scratch,
                strict=True,
            ):
                for view_start in range(
                    0,
                    self.valid_view_count,
                    JOINT_REINDEX_VIEW_CHUNK_SIZE,
                ):
                    view_stop = min(
                        view_start + JOINT_REINDEX_VIEW_CHUNK_SIZE,
                        self.valid_view_count,
                    )
                    view = source[:, view_start:view_stop, ...]
                    active_scratch = scratch[:, : view_stop - view_start, ...]
                    np.take(view, raw, axis=0, out=active_scratch)
                    view[...] = active_scratch
            np.take(
                self.station_log_likelihood[:, : self.station_count],
                raw,
                axis=0,
                out=station_scratch,
            )
            self.station_log_likelihood[:, : self.station_count] = station_scratch
        self._staged_ancestor_indices_n = None
        self._staged_ancestor_sha256 = None
        self.ancestor_reindex_count += 1
