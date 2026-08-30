"""Batched geometry and accelerator-array contracts for the particle filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class StructuralGeometryBatch:
    """Store only geometry needed by continuous structural PF proposals."""

    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]
    station_sequence_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate, copy, and freeze one aligned geometry batch."""
        detector_positions = np.array(
            self.detector_positions,
            dtype=np.float64,
            copy=True,
        )
        fe_indices = np.array(self.fe_indices, dtype=np.int64, copy=True).reshape(-1)
        pb_indices = np.array(self.pb_indices, dtype=np.int64, copy=True).reshape(-1)
        live_times = np.array(
            self.live_times,
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        station_ids = np.array(
            self.station_sequence_ids,
            dtype=np.int64,
            copy=True,
        ).reshape(-1)
        row_count = int(fe_indices.size)
        if (
            row_count == 0
            or detector_positions.shape != (row_count, 3)
            or pb_indices.size != row_count
            or live_times.size != row_count
            or station_ids.size != row_count
            or np.any(~np.isfinite(detector_positions))
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
            or np.any(fe_indices < 0)
            or np.any(pb_indices < 0)
            or np.any(station_ids < 0)
        ):
            raise ValueError(
                "Structural geometry must contain aligned finite detector, "
                "shield, positive-live-time, and station-ID rows."
            )
        for values in (
            detector_positions,
            fe_indices,
            pb_indices,
            live_times,
            station_ids,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "detector_positions", detector_positions)
        object.__setattr__(self, "fe_indices", fe_indices)
        object.__setattr__(self, "pb_indices", pb_indices)
        object.__setattr__(self, "live_times", live_times)
        object.__setattr__(self, "station_sequence_ids", station_ids)

    @property
    def row_count(self) -> int:
        """Return the number of aligned geometry rows."""
        return int(np.asarray(self.fe_indices).size)


@dataclass(frozen=True)
class TorchLineTransportComponents:
    """Store source-resolved line-rate components as Torch tensors."""

    total_kernel: torch.Tensor
    uncollided_kernel: torch.Tensor
    tau_fe: torch.Tensor
    tau_pb: torch.Tensor
    tau_obstacle: torch.Tensor
    tau_obstacle_compton: torch.Tensor
    distance_m: torch.Tensor
    uncollided_impact_fractions: torch.Tensor


__all__ = ["StructuralGeometryBatch", "TorchLineTransportComponents"]
