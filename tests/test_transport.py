"""Tests for the shared pre-spectrum transport layer."""

from __future__ import annotations

import pytest

from measurement.model import PointSource
from sim.isaacsim_app.stage_backend import StageMaterialInfo
from sim.transport import build_source_transport_result, make_transport_segment


def test_build_source_transport_result_tracks_obstacle_and_scatter() -> None:
    """The shared transport result should expose path totals and line statistics."""
    source = PointSource(isotope="Cs-137", position=(4.0, 4.0, 1.0), intensity_cps_1m=1000.0)
    stage_segments = (
        make_transport_segment(StageMaterialInfo(name="concrete"), 15.0, is_obstacle=True),
    )
    result = build_source_transport_result(
        source=source,
        detector_position_xyz=(4.0, 1.0, 1.0),
        dwell_time_s=10.0,
        stage_segments=stage_segments,
        fe_segment=make_transport_segment(StageMaterialInfo(name="fe"), 5.0),
        pb_segment=make_transport_segment(StageMaterialInfo(name="pb"), 2.0),
        nuclide_lines=((662.0, 0.85),),
        scatter_gain=0.12,
    )

    assert result.total_obstacle_path_cm == pytest.approx(15.0)
    assert result.total_stage_path_cm == pytest.approx(15.0)
    assert result.total_fe_path_cm == pytest.approx(5.0)
    assert result.total_pb_path_cm == pytest.approx(2.0)
    assert len(result.lines) == 1
    assert result.lines[0].total_transmission <= 1.0
    assert result.lines[0].scatter_counts >= 0.0
