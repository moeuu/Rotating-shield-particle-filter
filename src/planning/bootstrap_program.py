"""Deterministic balanced shield program for the first PF station."""

from __future__ import annotations

import numpy as np

from planning.program_types import ShieldProgram


def build_balanced_bootstrap_program(
    *,
    num_orientations: int,
    program_length: int,
) -> ShieldProgram:
    """Return a deterministic unique-pair program from orientation geometry.

    Pair IDs advance by ``num_orientations + 1`` modulo the square Fe/Pb
    pair domain. This step is coprime to ``num_orientations ** 2``, so every
    prefix is repetition-free. For the production 8-by-8, eight-view setup,
    the result visits each Fe and Pb orientation exactly once.
    """
    for name, value in {
        "num_orientations": num_orientations,
        "program_length": program_length,
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    orientation_count = int(num_orientations)
    length = int(program_length)
    pair_count = orientation_count**2
    if length > pair_count:
        raise ValueError("program_length cannot exceed the Fe/Pb pair count.")
    step = orientation_count + 1
    pair_ids = tuple(
        int((index * step) % pair_count) for index in range(length)
    )
    if len(set(pair_ids)) != length:
        raise RuntimeError("Bootstrap shield traversal repeated a pair.")
    return ShieldProgram(
        name="prior_balanced_bootstrap",
        pair_ids=pair_ids,
        kind="prior_balanced_bootstrap",
    )


__all__ = ["build_balanced_bootstrap_program"]
