"""Tests for current shared-runtime station-prefix reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from pf.live_resume import reconstruct_live_resume_state, records_by_station
from runtime.measurement_log import load_measurement_log
from tests.pure_pf_test_support import make_measurement_log, records


def _complete_records(count: int, *, views_per_station: int) -> tuple[object, ...]:
    """Return test records with complete fixed-size station metadata."""
    prepared = []
    for index, record in enumerate(records(count)):
        station_id = index // views_per_station
        station_complete = (index + 1) % views_per_station == 0
        prepared.append(
            type(record)(
                step_id=record.step_id,
                action_id=record.action_id,
                station_id=station_id,
                detector_pose_xyz=(float(station_id), 0.5, 0.5),
                detector_quat_wxyz=record.detector_quat_wxyz,
                fe_orientation_index=record.fe_orientation_index,
                pb_orientation_index=record.pb_orientation_index,
                live_time_s=record.live_time_s,
                travel_time_s=record.travel_time_s,
                shield_actuation_time_s=record.shield_actuation_time_s,
                energy_bin_edges_keV=record.energy_bin_edges_keV,
                spectrum_counts=record.spectrum_counts,
                metadata={**record.metadata, "station_complete": station_complete},
            )
        )
    return tuple(prepared)


def test_reconstruct_live_resume_state_preserves_complete_prefix() -> None:
    """Resume should recover counts, pose history, pair, and physical time."""
    prefix = _complete_records(4, views_per_station=2)

    state = reconstruct_live_resume_state(
        prefix,
        next_station_id=2,
        expected_views_per_station=2,
    )

    assert len(state.stations) == 2
    assert state.record_count == 4
    assert state.next_station_id == 2
    np.testing.assert_array_equal(state.current_pose, [1.0, 0.5, 0.5])
    assert len(state.visited_poses) == 2
    expected_pair = int(prefix[-1].fe_orientation_index) * 8 + int(
        prefix[-1].pb_orientation_index
    )
    assert state.current_pair_id == expected_pair
    assert state.elapsed_time_s == pytest.approx(
        sum(
            record.live_time_s + record.travel_time_s + record.shield_actuation_time_s
            for record in prefix
        )
    )


def test_resume_rejects_partial_or_wrong_length_station() -> None:
    """PF resume must occur only at its configured station boundary."""
    partial = _complete_records(3, views_per_station=2)
    with pytest.raises(ValueError, match="durably complete"):
        records_by_station(partial)

    prefix = _complete_records(4, views_per_station=2)
    with pytest.raises(ValueError, match="program length"):
        reconstruct_live_resume_state(
            prefix,
            next_station_id=2,
            expected_views_per_station=4,
        )


def test_resume_uses_shared_station_and_array_views_when_available(tmp_path) -> None:
    """Finalized resume reconstruction should consume shared immutable views."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    station_view = log.station_view()
    arrays = station_view.array_view()

    state = reconstruct_live_resume_state(
        station_view,
        next_station_id=2,
        expected_views_per_station=2,
    )

    assert [tuple(record.step_id for record in station) for station in state.stations] == [
        tuple(record.step_id for record in station.records)
        for station in station_view.stations
    ]
    assert state.record_count == station_view.record_count
    assert state.elapsed_time_s == pytest.approx(
        float(
            np.sum(
                arrays.live_time_s
                + arrays.travel_time_s
                + arrays.shield_actuation_time_s
            )
        )
    )
