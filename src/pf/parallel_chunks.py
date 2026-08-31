"""Small ordered thread-pool helpers for exact read-only NumPy workloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import os
from typing import TypeVar


_ResultT = TypeVar("_ResultT")
MAX_EXACT_CPU_WORKERS = 16


def exact_cpu_worker_count(task_count: int) -> int:
    """Return the bounded physical-core worker count for independent chunks."""
    count = int(task_count)
    if count < 1:
        raise ValueError("task_count must be positive.")
    logical_cpus = os.cpu_count() or 1
    physical_core_bound = max(1, logical_cpus // 2)
    return min(count, physical_core_bound, MAX_EXACT_CPU_WORKERS)


def ordered_exact_parallel_map(
    function: Callable[[tuple[int, int]], _ResultT],
    chunks: Sequence[tuple[int, int]],
) -> list[_ResultT]:
    """Evaluate independent exact chunks concurrently while preserving order."""
    work = tuple((int(start), int(stop)) for start, stop in chunks)
    if not work:
        return []
    if any(start < 0 or stop <= start for start, stop in work):
        raise ValueError("Parallel chunks must be nonempty increasing intervals.")
    worker_count = exact_cpu_worker_count(len(work))
    if worker_count == 1:
        return [function(chunk) for chunk in work]
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pf-exact",
    ) as executor:
        return list(executor.map(function, work))
