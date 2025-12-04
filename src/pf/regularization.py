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
    候補点の中から新規源位置を選択する単純な事前分布。

    現在の粒子が未使用のグリッドを優先し、それでも存在しない場合は一様にサンプルする。
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
    max_sources: int = 5,
) -> list[ParticleState]:
    """
    リサンプリング後に小さな摂動を与え、弱い源の死亡と低確率の出生を適用する。

    Death: 強度がmin_strength未満の源を削除する（Sec. 3.4.2の可変源数モデルに沿う簡易版）。
    Birth: 確率p_birthで未使用グリッドから新規源を追加し、log-normal初期強度を与える。
    """
    regularized: list[ParticleState] = []
    for st in states:
        st = jitter_state(st, strength_sigma=strength_sigma, background_sigma=background_sigma)
        if st.strengths.size:
            keep = st.strengths >= min_strength
            st.source_indices = st.source_indices[keep]
            st.strengths = st.strengths[keep]
        # Birth move
        if st.source_indices.size < max_sources and np.random.rand() < p_birth:
            candidate_idx = _sample_birth_index(kernel, st.source_indices)
            if candidate_idx is not None:
                birth_strength = float(np.random.lognormal(mean=-2.0, sigma=0.5))
                st.source_indices = np.append(st.source_indices, np.int32(candidate_idx))
                st.strengths = np.append(st.strengths, birth_strength)
        regularized.append(st)
    return regularized
