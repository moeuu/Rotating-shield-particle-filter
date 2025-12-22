"""Manage post-resampling perturbations, birth/death moves, and particle-count adaptation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.kernels import KernelPrecomputer
from pf.state import ParticleState, jitter_state


def _sample_birth_index(
    kernel: KernelPrecomputer,
    existing: NDArray[np.int32],
) -> int | None:
    """
    Choose a new source index from candidate locations.

    Prefer unused grid locations; if none remain, fall back to uniform sampling.
    """
    candidates = np.setdiff1d(np.arange(kernel.num_sources, dtype=np.int32), existing, assume_unique=True)
    if candidates.size == 0:
        return None
    return int(np.random.choice(candidates))


def regularize_states(
    states: list[ParticleState],
    kernel: KernelPrecomputer,
    strength_sigma: float = 0.1,
    background_sigma: float = 0.1,
    min_strength: float = 0.01,
    p_birth: float = 0.05,
    max_sources: int | None = None,
) -> list[ParticleState]:
    """
    Apply post-resampling jitter plus birth/death moves.

    Death: remove sources with strength below min_strength.
    Birth: with probability p_birth, add a new source at an unused grid location
    and draw its initial strength from a log-normal prior.
    """
    regularized: list[ParticleState] = []
    for st in states:
        st = jitter_state(st, strength_sigma=strength_sigma, background_sigma=background_sigma)
        if st.strengths.size:
            keep = st.strengths >= min_strength
            st.source_indices = st.source_indices[keep]
            st.strengths = st.strengths[keep]
        # Birth move
        can_birth = max_sources is None or st.source_indices.size < max_sources
        if can_birth and np.random.rand() < p_birth:
            candidate_idx = _sample_birth_index(kernel, st.source_indices)
            if candidate_idx is not None:
                birth_strength = float(np.random.lognormal(mean=-2.0, sigma=0.5))
                st.source_indices = np.append(st.source_indices, np.int32(candidate_idx))
                st.strengths = np.append(st.strengths, birth_strength)
        regularized.append(st)
    return regularized
