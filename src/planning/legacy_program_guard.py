"""Optional adapter exposing the retired 48-program library as an EIG floor."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from planning.program_types import ShieldProgram


def legacy_program_guard_candidates(
    normals: NDArray[np.float64],
    *,
    program_length: int,
    max_programs: int,
) -> tuple[ShieldProgram, ...]:
    """Return legacy programs without making the new search depend on them.

    This narrow adapter is the only production conditional-greedy dependency
    on the retired predeclared library. Removing the non-regression guard later
    therefore requires deleting one optional provider call, not rewriting the
    all-pairs search.
    """
    from planning.shield_programs import build_shield_program_library

    return tuple(
        build_shield_program_library(
            normals,
            program_length=int(program_length),
            max_programs=int(max_programs),
        )
    )


def legacy_program_pair_matrix(
    programs: Sequence[ShieldProgram],
) -> NDArray[np.int64]:
    """Return a dense pair-index matrix for same-sample cache evaluation."""
    rows = tuple(
        np.asarray(program.pair_ids, dtype=np.int64).reshape(-1)
        for program in programs
    )
    if not rows:
        return np.zeros((0, 0), dtype=np.int64)
    view_count = int(rows[0].size)
    if view_count <= 0 or any(int(row.size) != view_count for row in rows):
        raise ValueError("Legacy guard programs must have equal nonzero lengths.")
    matrix = np.vstack(rows).astype(np.int64, copy=False)
    if any(np.unique(row).size != row.size for row in matrix):
        raise ValueError("Legacy guard programs must not repeat a shield pair.")
    return np.ascontiguousarray(matrix)


__all__ = [
    "legacy_program_guard_candidates",
    "legacy_program_pair_matrix",
]
