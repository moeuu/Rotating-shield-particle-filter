"""Construct balanced Fe/Pb posture programs for PF planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShieldProgram:
    """Represent a short sequence of Fe/Pb shield orientation pairs."""

    name: str
    pair_ids: tuple[int, ...]
    kind: str


def build_shield_program_library(
    normals: NDArray[np.float64],
    *,
    program_length: int = 2,
    max_programs: int = 40,
) -> list[ShieldProgram]:
    """Build the complete structured library within a declared capacity.

    ``max_programs`` is a safety capacity, not a requested output count. The
    canonical eight-orientation, eight-view construction has exactly 48
    programs even when the supplied capacity is larger.
    """
    normal_arr = np.asarray(normals, dtype=float)
    if (
        normal_arr.ndim != 2
        or normal_arr.shape[1] != 3
        or np.any(~np.isfinite(normal_arr))
    ):
        raise ValueError("normals must be shaped (N, 3).")
    num_orients = int(normal_arr.shape[0])
    if num_orients <= 0:
        raise ValueError("normals must contain at least one orientation.")
    if (
        isinstance(program_length, bool)
        or not isinstance(program_length, (int, np.integer))
        or int(program_length) <= 0
    ):
        raise ValueError("program_length must be a positive integer.")
    if (
        isinstance(max_programs, bool)
        or not isinstance(max_programs, (int, np.integer))
        or int(max_programs) <= 0
    ):
        raise ValueError("max_programs must be a positive integer.")
    length = int(program_length)
    pair_count = num_orients * num_orients
    if length > pair_count:
        raise ValueError(
            "program_length cannot exceed the number of unique Fe/Pb pairs."
        )
    if num_orients == 8 and length == 8:
        orientation_axis = np.arange(num_orients, dtype=np.int64)
        # The four units modulo eight generate bijective Pb tours for every
        # Fe orientation. Together with the fixed-Fe and fixed-Pb partitions,
        # this yields (4 + 2) * 8 = 48 structured programs. The number is not
        # a truncation of the 64 individual Fe/Pb posture pairs.
        slopes = orientation_axis[np.gcd(orientation_axis, num_orients) == 1]
        latin_pb = (
            slopes[:, None, None] * orientation_axis[None, None, :]
            + orientation_axis[None, :, None]
        ) % num_orients
        latin_pairs = (
            orientation_axis[None, None, :] * num_orients + latin_pb
        ).reshape(-1, num_orients)
        fixed_fe_pairs = (
            orientation_axis[:, None] * num_orients + orientation_axis[None, :]
        )
        fixed_pb_pairs = (
            orientation_axis[None, :] * num_orients + orientation_axis[:, None]
        )
        pair_matrix = np.concatenate(
            (latin_pairs, fixed_fe_pairs, fixed_pb_pairs), axis=0
        )
        partition_names = tuple(
            [f"latin_slope_{int(slope)}" for slope in slopes] + ["fixed_fe", "fixed_pb"]
        )
        required_programs = len(partition_names) * num_orients
        if int(max_programs) < required_programs:
            raise ValueError(
                "max_programs is too small for the canonical balanced "
                "multi-partition shield library "
                f"({int(max_programs)} < {required_programs})."
            )
        program_names = tuple(
            f"{partition_name}_{program_index:02d}"
            for partition_name in partition_names
            for program_index in range(num_orients)
        )
        programs = [
            ShieldProgram(
                name=name,
                pair_ids=tuple(int(pair_id) for pair_id in row),
                kind="all_pair_balanced_multi_partition",
            )
            for name, row in zip(program_names, pair_matrix, strict=True)
        ]
        pair_occurrences = np.bincount(pair_matrix.reshape(-1), minlength=pair_count)
        if (
            np.any(np.diff(np.sort(pair_matrix, axis=1), axis=1) == 0)
            or np.any(pair_occurrences <= 0)
            or int(np.max(pair_occurrences) - np.min(pair_occurrences)) != 0
        ):
            raise RuntimeError(
                "Canonical shield partitions must be repetition-free and "
                "pair-frequency balanced."
            )
        return programs
    required_programs = int(np.ceil(pair_count / float(length)))
    if int(max_programs) < required_programs:
        raise ValueError(
            "max_programs is too small to expose every Fe/Pb pair without "
            f"within-program repetition ({int(max_programs)} < "
            f"{required_programs})."
        )
    orientation_axis = np.arange(num_orients, dtype=np.int64)
    ordered_pairs = (
        orientation_axis[None, :] * num_orients
        + (orientation_axis[None, :] + orientation_axis[:, None]) % num_orients
    ).reshape(-1)
    pair_indices = np.arange(required_programs * length, dtype=np.int64) % pair_count
    pair_matrix = ordered_pairs[pair_indices].reshape(required_programs, length)
    if np.any(np.diff(np.sort(pair_matrix, axis=1), axis=1) == 0):
        raise RuntimeError("Balanced shield construction produced a repeated pair.")
    programs = [
        ShieldProgram(
            name=f"all_pair_balanced_{index:02d}",
            pair_ids=tuple(int(pair_id) for pair_id in row),
            kind="all_pair_balanced",
        )
        for index, row in enumerate(pair_matrix)
    ]
    occurrences = np.bincount(pair_matrix.reshape(-1), minlength=pair_count)
    if np.any(occurrences <= 0) or int(np.ptp(occurrences)) > 1:
        raise RuntimeError(
            "Balanced shield programs must cover every pair with frequencies "
            "differing by at most one."
        )
    return programs


__all__ = ["ShieldProgram", "build_shield_program_library"]
