"""Test deterministic named random streams used by pure-PF planning."""

from __future__ import annotations

import numpy as np

from pf.randomness import (
    named_random_generator,
    named_rng_provenance,
    named_stream_seed,
)


def test_named_streams_are_reproducible_and_domain_separated() -> None:
    """Equal names reproduce exactly while distinct names use distinct streams."""
    first = named_random_generator(71, "planning", "candidate").random(16)
    second = named_random_generator(71, "planning", "candidate").random(16)
    different = named_random_generator(71, "planning", "eig").random(16)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert named_stream_seed(71, "planning", "candidate") == named_stream_seed(
        71,
        "planning",
        "candidate",
    )


def test_named_rng_provenance_binds_root_and_stream_names() -> None:
    """Planning provenance must bind every declared stream to its root seed."""
    provenance = named_rng_provenance(
        901,
        ("live_planning_candidate_dss_eig", "diagnostic"),
    )

    assert provenance["root_seed"] == 901
    assert provenance["bit_generator"] == "PCG64"
    assert set(provenance["streams"]) == {
        "diagnostic",
        "live_planning_candidate_dss_eig",
    }
    assert provenance["streams"]["diagnostic"]["domain"] == "diagnostic"
    assert provenance["streams"]["diagnostic"]["derived_seed_u64"] == (
        named_stream_seed(901, "diagnostic")
    )
    assert provenance != named_rng_provenance(
        902,
        ("live_planning_candidate_dss_eig", "diagnostic"),
    )


def test_truth_source_domain_separates_equal_obstacle_and_source_roots() -> None:
    """An explicit source root equal to the obstacle seed must not reuse draws."""
    root_seed = 2026072701
    obstacle_draws = np.random.default_rng(root_seed).random(32)
    source_draws = named_random_generator(
        root_seed,
        "truth_surface_sources",
    ).random(32)

    assert named_stream_seed(root_seed, "truth_surface_sources") != root_seed
    assert not np.array_equal(obstacle_draws, source_draws)
