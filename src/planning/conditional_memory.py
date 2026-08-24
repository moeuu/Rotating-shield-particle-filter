"""Memory-aware scheduling for all-pair conditional DSS evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pf.full_spectrum import SubsetCrossLikelihoodMemoryModel


_MINIMUM_DEVICE_HEADROOM_BYTES = 512 * 1024**2
_DEVICE_HEADROOM_FRACTION = 0.10
_PROXY_MAXIMUM_POSE_CHUNK = 32
_EXACT_MINIMUM_POSE_CHUNK = 2
_EXACT_MAXIMUM_POSE_CHUNK = 4


@dataclass(frozen=True, slots=True)
class AcceleratorMemorySnapshot:
    """Store allocator-aware accelerator memory availability."""

    device: str
    free_driver_bytes: int
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    reclaimable_allocator_bytes: int
    effective_free_bytes: int


@dataclass(frozen=True, slots=True)
class ConditionalPoseChunkPlan:
    """Store one deterministic pose-chunk scheduling decision."""

    workload: str
    requested_pose_count: int
    pose_chunk_size: int
    nominal_minimum_pose_chunk_size: int
    maximum_pose_chunk_size: int
    configured_total_phase_budget_bytes: int
    response_field_per_pose_bytes: int
    planner_response_destination_per_pose_bytes: int
    runtime_response_retained_per_pose_bytes: int
    response_persistent_per_pose_bytes: int
    response_persistent_total_bytes: int
    response_materialization_peak_per_pose_bytes: int
    response_materialization_peak_total_bytes: int
    minimum_response_scratch_budget_bytes: int
    response_scratch_budget_bytes: int
    runtime_subset_per_pose_estimate_bytes: int
    response_budget_limited_pose_capacity: int
    subset_budget_limited_pose_capacity: int
    budget_limited_pose_capacity: int
    device_headroom_bytes: int | None
    schedulable_device_bytes: int | None
    memory_limited_pose_capacity: int | None
    single_pose_low_memory_fallback: bool
    accelerator_memory: AcceleratorMemorySnapshot | None

    def diagnostics(self) -> dict[str, object]:
        """Return a JSON-safe diagnostic mapping."""
        payload = asdict(self)
        return {str(key): value for key, value in payload.items()}

    def response_scratch_budget_for_pose_count(self, pose_count: int) -> int:
        """Return residual response scratch for one retry-sized pose chunk."""
        if (
            isinstance(pose_count, bool)
            or not isinstance(pose_count, (int, np.integer))
            or int(pose_count) <= 0
            or int(pose_count) > int(self.pose_chunk_size)
        ):
            raise ValueError("pose_count must fit the planned pose chunk.")
        persistent = int(pose_count) * int(
            self.response_persistent_per_pose_bytes
        )
        configured_residual = (
            int(self.configured_total_phase_budget_bytes) - persistent
        )
        device_residual = (
            configured_residual
            if self.schedulable_device_bytes is None
            else int(self.schedulable_device_bytes) - persistent
        )
        residual = min(configured_residual, device_residual)
        if residual < int(self.minimum_response_scratch_budget_bytes):
            raise MemoryError(
                "Retry-sized response chunk cannot hold one transport row."
            )
        return int(residual)


def accelerator_memory_snapshot(
    *,
    use_gpu: bool,
    gpu_device: str,
) -> AcceleratorMemorySnapshot | None:
    """Return effective CUDA free memory including reclaimable cache blocks."""
    if not bool(use_gpu):
        return None
    import torch

    device = torch.device(str(gpu_device))
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
    except (RuntimeError, TypeError):
        return None
    reclaimable = max(0, int(reserved_bytes) - int(allocated_bytes))
    effective_free = min(int(total_bytes), int(free_bytes) + reclaimable)
    return AcceleratorMemorySnapshot(
        device=str(device),
        free_driver_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        allocated_bytes=int(allocated_bytes),
        reserved_bytes=int(reserved_bytes),
        reclaimable_allocator_bytes=int(reclaimable),
        effective_free_bytes=int(effective_free),
    )


def plan_conditional_pose_chunk(
    model: object,
    *,
    workload: str,
    requested_pose_count: int,
    particle_count: int,
    sample_count: int,
    source_slot_count: int,
    pair_count: int,
    program_length: int,
    line_count: int,
    feature_count: int,
    maximum_subset_candidate_count: int,
    configured_total_budget_bytes: int,
    minimum_response_scratch_budget_bytes: int,
    use_gpu: bool,
    gpu_device: str,
    memory_snapshot: AcceleratorMemorySnapshot | None = None,
) -> ConditionalPoseChunkPlan:
    """Choose a pose chunk without changing response or likelihood meaning.

    The configured exact/proxy byte value remains a total phase budget. The
    Response-persistent component buffers are subtracted before its runtime
    scratch cap is passed down. The separate cache/search phase is bounded by
    the runtime's per-pose subset estimate. Accelerator availability applies a
    second cap to both phases. Candidate subsets remain fully batched inside
    each selected pose chunk.
    """
    integer_counts = {
        "requested_pose_count": requested_pose_count,
        "particle_count": particle_count,
        "sample_count": sample_count,
        "source_slot_count": source_slot_count,
        "pair_count": pair_count,
        "program_length": program_length,
        "line_count": line_count,
        "feature_count": feature_count,
        "maximum_subset_candidate_count": maximum_subset_candidate_count,
        "configured_total_budget_bytes": (
            configured_total_budget_bytes
        ),
        "minimum_response_scratch_budget_bytes": (
            minimum_response_scratch_budget_bytes
        ),
    }
    for name, value in integer_counts.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if workload not in {"proxy", "exact"}:
        raise ValueError("workload must be 'proxy' or 'exact'.")
    if int(program_length) > int(pair_count):
        raise ValueError("program_length cannot exceed pair_count.")
    if not isinstance(model, SubsetCrossLikelihoodMemoryModel):
        raise RuntimeError(
            "Conditional DSS requires the runtime arbitrary-subset memory "
            "estimate contract."
        )
    runtime_estimate = int(
        model.estimate_subset_cross_likelihood_working_set_bytes(
            num_actions=1,
            num_samples=int(sample_count),
            num_particles=int(particle_count),
            num_source_slots=int(source_slot_count),
            num_views=int(pair_count),
            num_candidates=int(maximum_subset_candidate_count),
            subset_size=int(program_length),
            action_chunk_size=1,
            sample_chunk_size=min(int(sample_count), 10),
            state_chunk_size=min(int(particle_count), 128),
            view_chunk_size=min(int(pair_count), 8),
            dtype_bytes=np.dtype(np.float64).itemsize,
        )
    )
    if runtime_estimate <= 0:
        raise RuntimeError(
            "Runtime returned an invalid arbitrary-subset memory estimate."
        )
    maximum_chunk = (
        _PROXY_MAXIMUM_POSE_CHUNK
        if workload == "proxy"
        else _EXACT_MAXIMUM_POSE_CHUNK
    )
    nominal_minimum = 1 if workload == "proxy" else _EXACT_MINIMUM_POSE_CHUNK
    configured_budget = int(configured_total_budget_bytes)
    response_field_per_pose = int(
        int(particle_count)
        * int(pair_count)
        * int(source_slot_count)
        * int(line_count)
        * np.dtype(np.float64).itemsize
    )
    planner_response_destination_per_pose = int(
        (2 + int(feature_count)) * response_field_per_pose
    )
    # Runtime streams exact transport into eight retained fields. The dense
    # device adapter returns the six planner-required fields by reference.
    runtime_response_retained_per_pose = int(8 * response_field_per_pose)
    response_persistent_per_pose = int(
        planner_response_destination_per_pose
        + runtime_response_retained_per_pose
    )
    response_assembly_peak_per_pose = int(
        # Planner destination, six referenced transport fields, and four
        # stacked transport features are conservatively bounded by three
        # planner-sized groups for the production four-feature contract.
        3 * (2 + int(feature_count)) * response_field_per_pose
    )
    response_materialization_peak_per_pose = int(
        max(
            response_persistent_per_pose,
            response_assembly_peak_per_pose,
            (
                planner_response_destination_per_pose
                + 16 * response_field_per_pose
                if not bool(use_gpu)
                else 0
            ),
        )
    )
    minimum_response_scratch = int(minimum_response_scratch_budget_bytes)
    response_capacity = min(
        int(
            configured_budget
            // max(1, response_materialization_peak_per_pose)
        ),
        int(
            max(0, configured_budget - minimum_response_scratch)
            // max(1, response_persistent_per_pose)
        ),
    )
    subset_capacity = int(configured_budget // max(1, runtime_estimate))
    budget_capacity = min(response_capacity, subset_capacity)
    if budget_capacity <= 0:
        raise MemoryError(
            "Conditional DSS configured phase budget cannot hold one "
            "full-fidelity response and subset-search pose."
        )
    snapshot = memory_snapshot
    if snapshot is None:
        snapshot = accelerator_memory_snapshot(
            use_gpu=bool(use_gpu),
            gpu_device=str(gpu_device),
        )
    headroom: int | None = None
    schedulable: int | None = None
    memory_capacity: int | None = None
    fallback = False
    if snapshot is None:
        chunk_size = min(
            int(requested_pose_count),
            int(maximum_chunk),
            int(budget_capacity),
        )
    else:
        headroom = max(
            _MINIMUM_DEVICE_HEADROOM_BYTES,
            int(snapshot.total_bytes * _DEVICE_HEADROOM_FRACTION),
        )
        schedulable = max(0, int(snapshot.effective_free_bytes) - headroom)
        response_memory_capacity = min(
            int(
                int(schedulable)
                // max(1, response_materialization_peak_per_pose)
            ),
            int(
                max(0, int(schedulable) - minimum_response_scratch)
                // max(1, response_persistent_per_pose)
            ),
        )
        subset_memory_capacity = int(
            int(schedulable) // max(1, int(runtime_estimate))
        )
        memory_capacity = min(
            response_memory_capacity,
            subset_memory_capacity,
        )
        if memory_capacity <= 0:
            raise MemoryError(
                "Conditional DSS cannot reserve one full-fidelity pose within "
                "the currently available accelerator memory."
            )
        chunk_size = min(
            int(requested_pose_count),
            int(maximum_chunk),
            int(budget_capacity),
            int(memory_capacity),
        )
    fallback = bool(
        workload == "exact"
        and int(requested_pose_count) >= _EXACT_MINIMUM_POSE_CHUNK
        and int(chunk_size) < _EXACT_MINIMUM_POSE_CHUNK
    )
    response_persistent_total = int(chunk_size) * response_persistent_per_pose
    configured_scratch = configured_budget - response_persistent_total
    device_scratch = (
        configured_scratch
        if schedulable is None
        else int(schedulable) - response_persistent_total
    )
    response_scratch_budget = min(configured_scratch, device_scratch)
    if response_scratch_budget < minimum_response_scratch:
        raise MemoryError(
            "Conditional DSS response buffers leave insufficient budget for "
            "one exact transport source row."
        )
    return ConditionalPoseChunkPlan(
        workload=str(workload),
        requested_pose_count=int(requested_pose_count),
        pose_chunk_size=int(chunk_size),
        nominal_minimum_pose_chunk_size=int(nominal_minimum),
        maximum_pose_chunk_size=int(maximum_chunk),
        configured_total_phase_budget_bytes=int(configured_total_budget_bytes),
        response_field_per_pose_bytes=int(response_field_per_pose),
        planner_response_destination_per_pose_bytes=int(
            planner_response_destination_per_pose
        ),
        runtime_response_retained_per_pose_bytes=int(
            runtime_response_retained_per_pose
        ),
        response_persistent_per_pose_bytes=int(response_persistent_per_pose),
        response_persistent_total_bytes=int(response_persistent_total),
        response_materialization_peak_per_pose_bytes=int(
            response_materialization_peak_per_pose
        ),
        response_materialization_peak_total_bytes=int(
            int(chunk_size) * response_materialization_peak_per_pose
        ),
        minimum_response_scratch_budget_bytes=int(minimum_response_scratch),
        response_scratch_budget_bytes=int(response_scratch_budget),
        runtime_subset_per_pose_estimate_bytes=int(runtime_estimate),
        response_budget_limited_pose_capacity=int(response_capacity),
        subset_budget_limited_pose_capacity=int(subset_capacity),
        budget_limited_pose_capacity=int(budget_capacity),
        device_headroom_bytes=headroom,
        schedulable_device_bytes=schedulable,
        memory_limited_pose_capacity=memory_capacity,
        single_pose_low_memory_fallback=bool(fallback),
        accelerator_memory=snapshot,
    )


__all__ = [
    "AcceleratorMemorySnapshot",
    "ConditionalPoseChunkPlan",
    "accelerator_memory_snapshot",
    "plan_conditional_pose_chunk",
]
