"""Tests for MC-stability controlled DSS pose shortlisting."""

from __future__ import annotations

import numpy as np
import pytest

from planning.adaptive_shortlist import select_adaptive_pose_shortlist


def test_adaptive_shortlist_stops_at_stable_minimum() -> None:
    """A positive paired boundary gap should retain the minimum budget."""
    base = np.linspace(30.0, 1.0, 30, dtype=np.float64)
    replicas = np.vstack((base, base + 0.01, base - 0.01))

    result = select_adaptive_pose_shortlist(
        replicas,
        np.linspace(0.0, 1.0, 30, dtype=np.float64),
        minimum_pose_count=8,
        maximum_pose_count=16,
        pose_count_step=4,
        coverage_reserve_count=0,
        boundary_confidence=0.95,
        minimum_top_k_jaccard=0.75,
    )

    assert result.stop_reason == "stable_boundary"
    assert result.pose_indices.tolist() == list(range(8))
    assert len(result.boundary_diagnostics) == 1
    assert result.boundary_diagnostics[0].stable is True


def test_adaptive_shortlist_expands_to_maximum_when_boundary_is_unstable() -> None:
    """Replica rank reversals should consume the configured exact-pose budget."""
    replicas = np.vstack(
        (
            np.arange(24.0, 0.0, -1.0),
            np.roll(np.arange(24.0, 0.0, -1.0), 10),
            np.roll(np.arange(24.0, 0.0, -1.0), 18),
        )
    ).astype(np.float64)

    result = select_adaptive_pose_shortlist(
        replicas,
        np.zeros(24, dtype=np.float64),
        minimum_pose_count=8,
        maximum_pose_count=16,
        pose_count_step=4,
        coverage_reserve_count=0,
        boundary_confidence=0.95,
        minimum_top_k_jaccard=0.75,
    )

    assert result.stop_reason == "max_reached_unstable"
    assert result.pose_indices.size == 16
    assert [item.pose_count for item in result.boundary_diagnostics] == [8, 12, 16]


def test_adaptive_shortlist_accepts_refinement_only_for_top_pool() -> None:
    """NaN values may exclude non-refined poses from additional MC seeds."""
    base = np.linspace(40.0, 1.0, 40, dtype=np.float64)
    refined = np.full((2, 40), np.nan, dtype=np.float64)
    refined[:, :24] = np.vstack((base[:24] + 0.01, base[:24] - 0.01))
    replicas = np.vstack((base, refined))

    result = select_adaptive_pose_shortlist(
        replicas,
        np.zeros(40, dtype=np.float64),
        minimum_pose_count=8,
        maximum_pose_count=16,
        pose_count_step=4,
        coverage_reserve_count=0,
        boundary_confidence=0.95,
        minimum_top_k_jaccard=0.75,
    )

    assert result.stop_reason == "stable_boundary"
    assert result.pose_indices.tolist() == list(range(8))


def test_adaptive_shortlist_reserves_coverage_leader_without_growing() -> None:
    """The coverage leader should replace the lowest-ranked selected pose."""
    replicas = np.vstack(
        (
            np.linspace(20.0, 1.0, 20),
            np.linspace(20.0, 1.0, 20) + 0.01,
            np.linspace(20.0, 1.0, 20) - 0.01,
        )
    )
    coverage = np.zeros(20, dtype=np.float64)
    coverage[19] = 1.0

    result = select_adaptive_pose_shortlist(
        replicas,
        coverage,
        minimum_pose_count=8,
        maximum_pose_count=16,
        pose_count_step=4,
        coverage_reserve_count=1,
        boundary_confidence=0.95,
        minimum_top_k_jaccard=0.75,
    )

    assert result.pose_indices.size == 8
    assert 19 in result.pose_indices
    assert 7 not in result.pose_indices
    assert result.coverage_reserve_pose == 19


@pytest.mark.parametrize(
    ("minimum", "maximum", "step", "reserve"),
    ((0, 16, 4, 1), (8, 7, 4, 1), (8, 16, 0, 1), (8, 16, 4, 9)),
)
def test_adaptive_shortlist_rejects_invalid_integer_contract(
    minimum: int,
    maximum: int,
    step: int,
    reserve: int,
) -> None:
    """Invalid adaptive-budget contracts must fail before ranking."""
    with pytest.raises(ValueError):
        select_adaptive_pose_shortlist(
            np.ones((3, 20), dtype=np.float64),
            np.zeros(20, dtype=np.float64),
            minimum_pose_count=minimum,
            maximum_pose_count=maximum,
            pose_count_step=step,
            coverage_reserve_count=reserve,
            boundary_confidence=0.95,
            minimum_top_k_jaccard=0.75,
        )
