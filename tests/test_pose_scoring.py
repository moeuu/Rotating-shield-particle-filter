"""Tests for vectorized DSS pose-score composition."""

from __future__ import annotations

import numpy as np
import pytest

from planning.dss_types import DSSPPConfig
from planning.pose_scoring import compose_pose_scores


def test_pose_scoring_separates_mast_motion_weight() -> None:
    """Mast time must use its own weight in proxy and exact pose ranking."""
    config = DSSPPConfig(
        lambda_eig=1.0,
        lambda_distance=0.0,
        lambda_time=0.02,
        lambda_horizontal_time=0.02,
        lambda_mast_vertical_time=0.005,
        lambda_settling_time=0.02,
    )
    scores, distance_weight = compose_pose_scores(
        np.asarray([2.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([3.0, 3.0]),
        config=config,
        program_length=8,
        motion_times_p=np.asarray([5.0, 5.0]),
        motion_time_components_p=(
            np.asarray([3.0, 0.0]),
            np.asarray([0.0, 3.0]),
            np.asarray([2.0, 2.0]),
        ),
    )

    assert distance_weight == 0.0
    assert scores[1] - scores[0] == pytest.approx(0.045)


def test_pose_scoring_marks_unreachable_pose_as_negative_infinity() -> None:
    """An infinite path must remain outside every pose shortlist."""
    scores, _ = compose_pose_scores(
        np.asarray([1.0, 2.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, np.inf]),
        config=DSSPPConfig(lambda_distance=0.0),
        program_length=8,
    )

    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf
