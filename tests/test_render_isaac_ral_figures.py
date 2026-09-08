"""Tests for deterministic selection in the Isaac manuscript renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.render_isaac_ral_figures import (
    CaptureInputs,
    _select_geant4_tracks,
    _select_spatially_legible_pairs,
)


def test_shield_panels_select_legible_recorded_pairs_in_acquisition_order() -> None:
    """The selected subset must remain recorded, legible, and chronological."""
    recorded = (1, 45, 27, 49, 60, 40, 57, 23)

    assert _select_spatially_legible_pairs(recorded) == (49, 60, 40, 57)


def _track(
    source_index: int,
    isotope: str,
    *,
    detector_entered: bool,
    history: int,
) -> dict[str, object]:
    """Return one minimal native-step trajectory fixture."""
    return {
        "mode": "detector_entering" if detector_entered else "isotropic_emission",
        "source_index": source_index,
        "isotope": isotope,
        "primary_history_index": history,
        "raw_step_count": 1,
        "detector_entered": detector_entered,
        "points_truncated": False,
        "points_xyz_m": [[float(source_index), 0.0, 0.0], [3.0, 0.0, 0.0]],
    }


def test_geant4_track_selection_uses_three_saved_tracks_per_source() -> None:
    """Display selection must retain three native tracks for every source."""
    inputs = CaptureInputs(
        run_id="run",
        run_dir=Path("run"),
        measurement_log_dir=Path("log"),
        truth_manifest_path=Path("truth.json"),
        environment={},
        truth={
            "sources": [
                {
                    "isotope": "Cs-137",
                    "transport_position": [0.0, 0.0, 0.0],
                },
                {
                    "isotope": "Co-60",
                    "transport_position": [1.0, 0.0, 0.0],
                },
            ]
        },
        station_positions_xyz=np.asarray([[0.0, 0.0, 1.0]]),
        station_yaw_rad=np.asarray([0.0]),
        station_pair_ids=((9,),),
        route_segments_xyz=(),
    )
    emitted = [
        _track(source_index, isotope, detector_entered=False, history=history)
        for source_index, isotope, first_history in (
            (0, "Cs-137", 10),
            (1, "Co-60", 20),
        )
        for history in range(first_history, first_history + 3)
    ]
    artifact = {
        "run_id": "run",
        "station_index": 0,
        "recorded_pair_id": 9,
        "detector_pose_xyz_m": [0.0, 0.0, 1.0],
        "modes": {
            "isotropic_emission": {"tracks": emitted},
        },
    }

    selected_emitted = _select_geant4_tracks(
        inputs,
        artifact,
        station_index=0,
    )

    assert len(selected_emitted) == 6
    assert {int(track["source_index"]) for track in selected_emitted} == {0, 1}
    assert all(track in emitted for track in selected_emitted)
    np.testing.assert_array_equal(
        selected_emitted[0]["points_xyz_m"],
        emitted[0]["points_xyz_m"],
    )
