"""Tests for memory-aware conditional DSS pose scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pf.provenance import json_safe
from planning.conditional_memory import (
    AcceleratorMemorySnapshot,
    plan_conditional_pose_chunk,
)


@dataclass
class _MemoryModel:
    """Return a deterministic per-pose arbitrary-subset estimate."""

    estimate_bytes: int

    def estimate_subset_cross_likelihood_working_set_bytes(
        self,
        **_dimensions: object,
    ) -> int:
        """Return the configured estimate for one pose."""
        return int(self.estimate_bytes)


def _snapshot(effective_free_bytes: int) -> AcceleratorMemorySnapshot:
    """Return one deterministic 32-GiB accelerator snapshot."""
    total = 32 * 1024**3
    return AcceleratorMemorySnapshot(
        device="cuda:0",
        free_driver_bytes=int(effective_free_bytes),
        total_bytes=int(total),
        allocated_bytes=0,
        reserved_bytes=0,
        reclaimable_allocator_bytes=0,
        effective_free_bytes=int(effective_free_bytes),
    )


def _plan(
    *,
    workload: str,
    requested: int,
    estimate_bytes: int,
    total_budget_bytes: int,
    effective_free_bytes: int,
):
    """Build one compact deterministic chunk plan."""
    return plan_conditional_pose_chunk(
        _MemoryModel(estimate_bytes),
        workload=workload,
        requested_pose_count=requested,
        particle_count=512 if workload == "exact" else 16,
        sample_count=50 if workload == "exact" else 2,
        source_slot_count=24,
        pair_count=64,
        program_length=8,
        line_count=9,
        feature_count=4,
        maximum_subset_candidate_count=448 if workload == "exact" else 64,
        configured_total_budget_bytes=total_budget_bytes,
        minimum_response_scratch_budget_bytes=64 * 1024**2,
        use_gpu=True,
        gpu_device="cuda:0",
        memory_snapshot=_snapshot(effective_free_bytes),
    )


def test_exact_chunk_uses_total_budget_and_device_capacity() -> None:
    """Exact scheduling must remain in the nominal two-to-four pose range."""
    gib = 1024**3
    plan = _plan(
        workload="exact",
        requested=16,
        estimate_bytes=gib,
        total_budget_bytes=4 * gib,
        effective_free_bytes=24 * gib,
    )

    assert plan.pose_chunk_size == 4
    assert plan.budget_limited_pose_capacity == 4
    assert plan.single_pose_low_memory_fallback is False
    assert plan.configured_total_phase_budget_bytes == 4 * gib
    assert plan.response_persistent_total_bytes == (
        4 * plan.response_persistent_per_pose_bytes
    )
    assert plan.planner_response_destination_per_pose_bytes == (
        6 * plan.response_field_per_pose_bytes
    )
    assert plan.runtime_response_retained_per_pose_bytes == (
        8 * plan.response_field_per_pose_bytes
    )
    assert plan.response_persistent_per_pose_bytes == (
        14 * plan.response_field_per_pose_bytes
    )
    assert plan.response_materialization_peak_per_pose_bytes == (
        18 * plan.response_field_per_pose_bytes
    )
    assert plan.response_scratch_budget_bytes == (
        4 * gib - plan.response_persistent_total_bytes
    )
    assert plan.response_materialization_peak_total_bytes <= 4 * gib


def test_exact_chunk_has_explicit_single_pose_low_memory_fallback() -> None:
    """Full fidelity must fall back to one pose instead of overcommitting."""
    gib = 1024**3
    plan = _plan(
        workload="exact",
        requested=8,
        estimate_bytes=3 * gib,
        total_budget_bytes=4 * gib,
        effective_free_bytes=8 * gib,
    )

    assert plan.pose_chunk_size == 1
    assert plan.budget_limited_pose_capacity == 1
    assert plan.single_pose_low_memory_fallback is True


def test_proxy_chunk_preserves_batched_pose_cap() -> None:
    """A sufficient total phase budget must retain the 32-pose proxy batch."""
    mib = 1024**2
    gib = 1024**3
    plan = _plan(
        workload="proxy",
        requested=256,
        estimate_bytes=64 * mib,
        total_budget_bytes=4 * gib,
        effective_free_bytes=28 * gib,
    )

    assert plan.pose_chunk_size == 32
    assert plan.maximum_pose_chunk_size == 32
    assert json_safe(plan.diagnostics()) == plan.diagnostics()


def test_reclaimable_allocator_memory_is_included_in_snapshot_contract() -> None:
    """Diagnostics must preserve the allocator-aware effective-free value."""
    gib = 1024**3
    snapshot = AcceleratorMemorySnapshot(
        device="cuda:0",
        free_driver_bytes=8 * gib,
        total_bytes=32 * gib,
        allocated_bytes=2 * gib,
        reserved_bytes=6 * gib,
        reclaimable_allocator_bytes=4 * gib,
        effective_free_bytes=12 * gib,
    )
    plan = plan_conditional_pose_chunk(
        _MemoryModel(gib),
        workload="exact",
        requested_pose_count=4,
        particle_count=512,
        sample_count=50,
        source_slot_count=24,
        pair_count=64,
        program_length=8,
        line_count=9,
        feature_count=4,
        maximum_subset_candidate_count=448,
        configured_total_budget_bytes=4 * gib,
        minimum_response_scratch_budget_bytes=64 * 1024**2,
        use_gpu=True,
        gpu_device="cuda:0",
        memory_snapshot=snapshot,
    )

    assert plan.accelerator_memory == snapshot
    assert plan.schedulable_device_bytes == 12 * gib - int(3.2 * gib)
    assert plan.pose_chunk_size == 4


def test_response_resident_buffers_are_part_of_total_phase_budget() -> None:
    """A budget smaller than one response materialization must fail closed."""
    mib = 1024**2
    with pytest.raises(MemoryError, match="response and subset-search"):
        plan_conditional_pose_chunk(
            _MemoryModel(32 * mib),
            workload="exact",
            requested_pose_count=4,
            particle_count=512,
            sample_count=50,
            source_slot_count=24,
            pair_count=64,
            program_length=8,
            line_count=9,
            feature_count=4,
            maximum_subset_candidate_count=448,
            configured_total_budget_bytes=512 * mib,
            minimum_response_scratch_budget_bytes=64 * mib,
            use_gpu=False,
            gpu_device="cpu",
        )
