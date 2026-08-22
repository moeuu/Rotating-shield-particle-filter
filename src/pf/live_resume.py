"""Truth-free station-prefix reconstruction for adaptive PF resume."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from runtime.measurement_log import MeasurementLogRecord, MeasurementLogStationView


@dataclass(frozen=True, slots=True)
class LiveResumeState:
    """Store controller state reconstructed from complete logged stations."""

    stations: tuple[tuple[MeasurementLogRecord, ...], ...]
    next_station_id: int
    record_count: int
    current_pose: NDArray[np.float64]
    visited_poses: tuple[NDArray[np.float64], ...]
    current_pair_id: int
    elapsed_time_s: float


def records_by_station(
    records: Sequence[MeasurementLogRecord] | MeasurementLogStationView,
) -> tuple[tuple[MeasurementLogRecord, ...], ...]:
    """Group a causal prefix by contiguous zero-based station id."""
    if isinstance(records, MeasurementLogStationView):
        if not records.records:
            raise ValueError("Resume requires at least one MeasurementLog record.")
        incomplete = [
            station.station_id
            for station in records.stations
            if not station.marked_complete
        ]
        if incomplete:
            raise ValueError("Every resumed station must be durably complete.")
        return tuple(station.records for station in records.stations)
    if not records:
        raise ValueError("Resume requires at least one MeasurementLog record.")
    grouped: list[list[MeasurementLogRecord]] = []
    for record in records:
        station_id = int(record.station_id)
        if station_id == len(grouped):
            grouped.append([record])
        elif station_id == len(grouped) - 1:
            grouped[-1].append(record)
        else:
            raise ValueError(
                "Resume records require contiguous zero-based station identifiers."
            )
    for station in grouped:
        if station[-1].metadata.get("station_complete") is not True:
            raise ValueError("Every resumed station must be durably complete.")
    return tuple(tuple(station) for station in grouped)


def reconstruct_live_resume_state(
    records: Sequence[MeasurementLogRecord] | MeasurementLogStationView,
    *,
    next_station_id: int,
    expected_views_per_station: int,
) -> LiveResumeState:
    """Validate a runtime prefix and reconstruct PF controller counters."""
    if (
        isinstance(next_station_id, bool)
        or not isinstance(next_station_id, int)
        or next_station_id < 1
    ):
        raise ValueError("next_station_id must be a positive integer.")
    if (
        isinstance(expected_views_per_station, bool)
        or not isinstance(expected_views_per_station, int)
        or expected_views_per_station < 1
    ):
        raise ValueError("expected_views_per_station must be positive.")
    stations = records_by_station(records)
    record_rows = (
        records.records
        if isinstance(records, MeasurementLogStationView)
        else tuple(records)
    )
    if next_station_id != len(stations):
        raise ValueError("Resume next_station_id disagrees with station count.")
    incompatible = [
        index
        for index, station in enumerate(stations)
        if len(station) != expected_views_per_station
    ]
    if incompatible:
        raise ValueError(
            "Resumed stations differ from the configured PF program length: "
            f"{incompatible}."
        )
    if isinstance(records, MeasurementLogStationView):
        arrays = records.array_view()
        station_poses = tuple(
            arrays.detector_pose_xyz[station.stop_index - 1].copy()
            for station in records.stations
        )
        elapsed = float(
            np.sum(
                arrays.live_time_s
                + arrays.travel_time_s
                + arrays.shield_actuation_time_s,
                dtype=np.float64,
            )
        )
    else:
        station_poses = tuple(
            np.asarray(station[-1].detector_pose_xyz, dtype=np.float64).copy()
            for station in stations
        )
        elapsed = float(
            sum(
                float(record.live_time_s)
                + float(record.travel_time_s)
                + float(record.shield_actuation_time_s)
                for record in record_rows
            )
        )
    last = stations[-1][-1]
    return LiveResumeState(
        stations=stations,
        next_station_id=next_station_id,
        record_count=len(record_rows),
        current_pose=station_poses[-1].copy(),
        visited_poses=tuple(pose.copy() for pose in station_poses),
        current_pair_id=(
            int(last.fe_orientation_index) * 8 + int(last.pb_orientation_index)
        ),
        elapsed_time_s=elapsed,
    )


__all__ = [
    "LiveResumeState",
    "reconstruct_live_resume_state",
    "records_by_station",
]
