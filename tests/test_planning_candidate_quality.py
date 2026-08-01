"""Tests for fail-closed global planning-candidate quality."""

from __future__ import annotations

import numpy as np
import pytest

from planning import candidate_generation


def test_candidate_masks_reject_truthy_string_flags() -> None:
    """String flags must not turn blocked or unreachable poses into actions."""
    candidates = np.asarray(
        [[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="exact boolean"):
        candidate_generation._filter_candidates(
            candidates,
            None,
            0.0,
            is_free_batch_fn=lambda _: np.asarray(["false", "true"]),
        )

    class InvalidReachabilityMap:
        """Return corrupt reachability flags for contract testing."""

        @staticmethod
        def is_motion_reachable_batch(
            current: np.ndarray,
            goals: np.ndarray,
        ) -> np.ndarray:
            """Return string flags with the requested row count."""
            del current, goals
            return np.asarray(["false", "true"])

    invalid_map = InvalidReachabilityMap()
    with pytest.raises(ValueError, match="exact boolean"):
        candidate_generation._filter_motion_reachable_candidates(
            candidates,
            current_pose_xyz=np.zeros(3, dtype=np.float64),
            map_api=invalid_map,
            enabled=True,
        )


def test_height_only_global_pool_is_not_rejected_or_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global sampling must not impose an ad-hoc horizontal quality gate."""
    attempted_distances: list[float] = []

    def fake_generate_candidate_poses(**kwargs: object) -> np.ndarray:
        """Return one valid global pool at a single XY location."""
        attempted_distances.append(
            float(kwargs["min_dist_from_visited"])
        )
        return np.column_stack(
            (
                np.ones(8),
                np.ones(8),
                np.linspace(0.0, 7.0, 8),
            )
        )

    monkeypatch.setattr(
        candidate_generation,
        "generate_candidate_poses",
        fake_generate_candidate_poses,
    )
    candidates, diagnostics = (
        candidate_generation.generate_planning_candidates(
            current_pose_xyz=np.zeros(3, dtype=float),
            map_api=None,
            n_candidates=8,
            min_dist_from_visited=4.0,
            visited_poses_xyz=np.zeros((1, 3), dtype=float),
            bounds_xyz=(
                np.zeros(3, dtype=float),
                np.asarray([10.0, 10.0, 8.0], dtype=float),
            ),
            rng=np.random.default_rng(7),
        )
    )

    assert attempted_distances == pytest.approx([4.0])
    assert candidates.shape == (8, 3)
    assert diagnostics["candidate_count"] == 8
    assert diagnostics["horizontal_quality_gate"] is False
    assert diagnostics["physical_separation_relaxed"] is False
    assert diagnostics["minimum_3d_separation_m"] == pytest.approx(4.0)


def test_empty_global_pool_fails_without_relaxing_physical_separation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty physical action set must fail rather than relax its contract."""
    attempted_distances: list[float] = []

    def empty_candidates(**kwargs: object) -> np.ndarray:
        """Return no physically admissible candidate."""
        attempted_distances.append(
            float(kwargs["min_dist_from_visited"])
        )
        return np.zeros((0, 3), dtype=np.float64)

    monkeypatch.setattr(
        candidate_generation,
        "generate_candidate_poses",
        empty_candidates,
    )
    with pytest.raises(RuntimeError, match="No globally sampled candidate"):
        candidate_generation.generate_planning_candidates(
            current_pose_xyz=np.zeros(3, dtype=float),
            map_api=None,
            n_candidates=8,
            min_dist_from_visited=4.0,
            visited_poses_xyz=np.zeros((1, 3), dtype=float),
            bounds_xyz=(
                np.zeros(3, dtype=float),
                np.asarray([10.0, 10.0, 8.0], dtype=float),
            ),
            rng=np.random.default_rng(9),
        )

    assert attempted_distances == pytest.approx([4.0])


def test_candidate_checkpoint_records_global_pool_contract() -> None:
    """Resume compatibility must pin the global physical pool contract."""
    parameters = candidate_generation.planning_candidate_checkpoint_parameters(
        pose_candidates=64,
        pose_min_dist=3.0,
        bounds_xyz=(
            np.zeros(3, dtype=float),
            np.ones(3, dtype=float) * 10.0,
        ),
        detector_heights_m=None,
    )

    assert parameters["candidate_pool_contract"] == (
        "global_reachable_3d_sobol_with_physical_separation_v1"
    )
    assert parameters["pose_min_dist_m"] == pytest.approx(3.0)
    obsolete = {
        "candidate_min_unique_xy",
        "candidate_min_horizontal_extent_fraction",
        "candidate_xy_merge_tolerance_m",
        "candidate_distance_relaxation_factor",
        "candidate_max_distance_retries",
    }
    assert obsolete.isdisjoint(parameters)
