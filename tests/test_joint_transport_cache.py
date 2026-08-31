"""Tests for the fixed-capacity exact joint transport cache."""

from __future__ import annotations

import numpy as np
import pytest

from pf.joint_transport_cache import JointTransportCache


def _numpy_station(
    value: float,
    *,
    particles: int = 3,
    views: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one deterministic source-resolved NumPy station slab."""
    total = np.full((particles, views, 4, 2), value, dtype=np.float64)
    uncollided = 0.75 * total
    features = np.full(total.shape + (3,), value + 1.0, dtype=np.float64)
    return total, uncollided, features


def test_fixed_cache_appends_without_reallocating_history() -> None:
    """Appending a station must write into the existing 128-view allocation."""
    first = _numpy_station(1.0)
    cache = JointTransportCache.allocate(
        first,
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=2,
    )
    storage_ids = tuple(id(value) for value in cache.backing_storage)

    cache.append_station(
        _numpy_station(2.0),
        station_signature="station-1",
    )

    assert tuple(id(value) for value in cache.backing_storage) == storage_ids
    assert cache.valid_view_count == 4
    assert cache.station_offsets == (0, 2, 4)
    assert cache.station_signatures == ("station-0", "station-1")
    assert cache[0].shape == (3, 4, 4, 2)
    assert cache.backing_storage[0].shape == (3, 128, 4, 2)
    np.testing.assert_array_equal(cache[0][:, :2], 1.0)
    np.testing.assert_array_equal(cache[0][:, 2:], 2.0)


def test_fixed_cache_replaces_only_selected_slot_rows() -> None:
    """A slot overlay commit must leave every other row and slot unchanged."""
    cache = JointTransportCache.allocate(
        _numpy_station(1.0),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    cache.set_station_likelihood(
        np.asarray([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
    )
    replacement_total = np.full((2, 2, 2, 2), 9.0, dtype=np.float64)
    replacement = (
        replacement_total,
        0.5 * replacement_total,
        np.full(replacement_total.shape + (3,), 7.0, dtype=np.float64),
    )

    cache.replace_slot_rows(
        rows=np.asarray([2, 0], dtype=np.int64),
        slot_start=1,
        slot_stop=3,
        replacement=replacement,
        active_slot_mask=np.ones((2, 2), dtype=np.bool_),
    )

    np.testing.assert_array_equal(cache[0][1], 1.0)
    np.testing.assert_array_equal(cache[0][[2, 0], :, 1:3], 9.0)
    np.testing.assert_array_equal(cache[0][[2, 0], :, [0, 3]], 1.0)
    assert np.isnan(cache.station_log_likelihood[[2, 0], 0]).all()
    assert cache.station_log_likelihood[1, 0] == -2.0
    assert cache.slot_overlay_commit_count == 2


def test_fixed_cache_reindexes_transport_and_station_likelihood_together() -> None:
    """One ancestor vector must reorder every cache-owned row identically."""
    station = _numpy_station(1.0)
    for row in range(3):
        station[0][row] = float(row)
        station[1][row] = float(row + 10)
        station[2][row] = float(row + 20)
    cache = JointTransportCache.allocate(
        station,
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    cache.set_station_likelihood(
        np.asarray([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
    )

    cache.reindex_rows(np.asarray([2, 1, 1], dtype=np.int64))

    np.testing.assert_array_equal(cache[0][:, 0, 0, 0], [2.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        cache[1][:, 0, 0, 0],
        [12.0, 11.0, 11.0],
    )
    np.testing.assert_array_equal(
        cache.station_log_likelihood[:, 0],
        [-3.0, -2.0, -2.0],
    )
    assert cache.ancestor_reindex_count == 1


def test_resampling_stays_unpublished_until_atomic_commit() -> None:
    """Staging every cache component must not mutate the accepted buffer."""
    station = _numpy_station(1.0)
    for row in range(3):
        station[0][row] = float(row)
    cache = JointTransportCache.allocate(
        station,
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    accepted_before = cache[0].copy()

    cache.stage_reindex_rows(np.asarray([2, 1, 1], dtype=np.int64))

    np.testing.assert_array_equal(cache[0], accepted_before)
    assert cache.ancestor_reindex_count == 0
    cache.commit_staged_reindex()
    np.testing.assert_array_equal(cache[0][:, 0, 0, 0], [2.0, 1.0, 1.0])
    assert cache.ancestor_reindex_count == 1


def test_fixed_cache_fails_closed_at_contract_capacity() -> None:
    """A seventeenth station or a 129th view must not trigger reallocation."""
    cache = JointTransportCache.allocate(
        _numpy_station(1.0, particles=1, views=8),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    for station_index in range(1, 16):
        cache.append_station(
            _numpy_station(
                float(station_index + 1),
                particles=1,
                views=8,
            ),
            station_signature=f"station-{station_index}",
        )

    assert cache.valid_view_count == 128
    with pytest.raises(RuntimeError, match="16-station"):
        cache.append_station(
            _numpy_station(17.0, particles=1, views=1),
            station_signature="station-16",
        )


def test_torch_cache_matches_numpy_commit_and_reindex() -> None:
    """The batched Torch cache must preserve the NumPy cache row semantics."""
    torch = pytest.importorskip("torch")
    numpy_cache = JointTransportCache.allocate(
        _numpy_station(1.0),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    torch_cache = JointTransportCache.allocate(
        tuple(torch.as_tensor(value) for value in _numpy_station(1.0)),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    values = np.asarray([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
    numpy_cache.set_station_likelihood(values)
    torch_cache.set_station_likelihood(torch.as_tensor(values))
    rows = np.asarray([2, 0], dtype=np.int64)
    replacement_total = np.full((2, 2, 2, 2), 9.0, dtype=np.float64)
    numpy_replacement = (
        replacement_total,
        0.5 * replacement_total,
        np.full(replacement_total.shape + (3,), 7.0, dtype=np.float64),
    )
    numpy_cache.replace_slot_rows(
        rows=rows,
        slot_start=1,
        slot_stop=3,
        replacement=numpy_replacement,
        active_slot_mask=np.ones((2, 2), dtype=np.bool_),
    )
    torch_cache.replace_slot_rows(
        rows=torch.as_tensor(rows),
        slot_start=1,
        slot_stop=3,
        replacement=tuple(torch.as_tensor(value) for value in numpy_replacement),
        active_slot_mask=torch.ones((2, 2), dtype=torch.bool),
    )
    ancestors = np.asarray([2, 1, 1], dtype=np.int64)
    numpy_cache.reindex_rows(ancestors)
    torch_cache.reindex_rows(ancestors)

    for actual, expected in zip(torch_cache, numpy_cache, strict=True):
        np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(
        torch_cache.station_log_likelihood.numpy(),
        numpy_cache.station_log_likelihood,
    )


def test_required_storage_excludes_bounded_reindex_scratch() -> None:
    """Capacity preflight separates persistent state from bounded scratch."""
    cache = JointTransportCache.allocate(
        _numpy_station(1.0),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )

    required = JointTransportCache.required_storage_bytes(
        particle_count=3,
        max_views=128,
        source_slots=4,
        line_count=2,
        feature_count=3,
        max_stations=16,
        dtype_bytes=8,
    )
    scratch = JointTransportCache.reindex_scratch_bytes(
        particle_count=3,
        source_slots=4,
        line_count=2,
        feature_count=3,
        max_stations=16,
        dtype_bytes=8,
    )

    assert cache.allocated_bytes == required
    assert scratch < required


def test_inactive_slot_nonzero_transport_fails_before_commit() -> None:
    """Inactive source slots must remain exactly zero in accepted transport."""
    cache = JointTransportCache.allocate(
        _numpy_station(1.0),
        station_signature="station-0",
        state_sha256="state-a",
        row_generation=0,
    )
    before = cache[0].copy()
    replacement_total = np.ones((1, 2, 2, 2), dtype=np.float64)
    replacement = (
        replacement_total,
        replacement_total.copy(),
        np.ones(replacement_total.shape + (3,), dtype=np.float64),
    )

    with pytest.raises(RuntimeError, match="Inactive source slots"):
        cache.replace_slot_rows(
            rows=np.asarray([0], dtype=np.int64),
            slot_start=1,
            slot_stop=3,
            replacement=replacement,
            active_slot_mask=np.asarray([[True, False]], dtype=np.bool_),
        )

    np.testing.assert_array_equal(cache[0], before)
