"""Tests for batched all-pair conditional-greedy shield search."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from planning.conditional_greedy import (
    conditional_greedy_candidate_count,
    evaluate_subset_information_gain_torch,
    select_conditional_greedy_programs,
)
from spectrum.transport_spectral import PreparedTorchSubsetCrossLikelihood


class _AdditiveLikelihoodCache:
    """Provide a vectorized deterministic likelihood cache for unit tests."""

    def __init__(self, pair_scores_ap: np.ndarray, *, device: torch.device) -> None:
        """Store per-action pair scores on the requested Torch device."""
        scores = np.asarray(pair_scores_ap, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError("pair_scores_ap must be shaped (A, P).")
        self.action_count = int(scores.shape[0])
        self.view_count = int(scores.shape[1])
        self.device = device
        self.dtype = torch.float64
        self._scores = torch.as_tensor(scores, device=device, dtype=torch.float64)
        self.call_shapes: list[tuple[int, int, int]] = []

    def evaluate(self, subset_indices_ack: object) -> torch.Tensor:
        """Return a batched two-particle likelihood for every subset."""
        subsets = torch.as_tensor(
            subset_indices_ack,
            device=self.device,
            dtype=torch.long,
        )
        self.call_shapes.append(tuple(map(int, subsets.shape)))
        candidate_count = int(subsets.shape[1])
        action_scores = self._scores[:, None, :].expand(
            -1,
            candidate_count,
            -1,
        )
        subset_scores = torch.gather(action_scores, 2, subsets).sum(dim=2)
        per_particle = torch.stack(
            (subset_scores, torch.zeros_like(subset_scores)),
            dim=2,
        )
        return per_particle[:, :, None, :].expand(-1, -1, 3, -1)


class _MappedSubsetLikelihoodCache:
    """Provide explicitly scored small subsets for refinement tests."""

    def __init__(
        self,
        *,
        pair_count: int,
        score: Callable[[tuple[int, ...]], float],
        device: torch.device,
    ) -> None:
        """Store one small deterministic set-utility function."""
        self.action_count = 1
        self.view_count = int(pair_count)
        self.device = device
        self.dtype = torch.float64
        self._score = score
        self.call_shapes: list[tuple[int, int, int]] = []

    def evaluate(self, subset_indices_ack: object) -> torch.Tensor:
        """Map batched test subsets to deterministic likelihood contrasts."""
        subsets = torch.as_tensor(
            subset_indices_ack,
            device=self.device,
            dtype=torch.long,
        )
        self.call_shapes.append(tuple(map(int, subsets.shape)))
        subset_numpy = np.asarray(subsets.detach().cpu().numpy(), dtype=np.int64)
        values = np.asarray(
            [
                [
                    self._score(tuple(int(pair) for pair in subset))
                    for subset in action_subsets
                ]
                for action_subsets in subset_numpy
            ],
            dtype=np.float64,
        )
        scores = torch.as_tensor(values, device=self.device, dtype=torch.float64)
        per_particle = torch.stack(
            (scores, torch.zeros_like(scores)),
            dim=2,
        )
        return per_particle[:, :, None, :].expand(-1, -1, 2, -1)


def _test_device() -> torch.device:
    """Use CUDA when available while keeping CPU-only CI deterministic."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _numpy_information_gain(
    log_likelihood_qn: np.ndarray,
    weights_n: np.ndarray,
) -> float:
    """Return one serial NumPy mutual-information reference value."""
    log_likelihood = np.asarray(log_likelihood_qn, dtype=np.float64)
    weights = np.asarray(weights_n, dtype=np.float64)
    weights = weights / np.sum(weights)
    positive = weights > 0.0
    likelihood = log_likelihood[:, positive]
    prior = weights[positive]
    log_joint = likelihood + np.log(prior)[None, :]
    log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
    posterior = np.exp(log_joint - log_evidence)
    terms = np.zeros_like(posterior)
    np.multiply(
        posterior,
        likelihood - log_evidence,
        out=terms,
        where=posterior > 0.0,
    )
    return float(np.mean(np.sum(terms, axis=1)))


def _serial_additive_greedy(
    pair_scores_ap: np.ndarray,
    weights_n: np.ndarray,
    *,
    program_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select subsets with a scalar NumPy oracle used only by tests."""
    scores = np.asarray(pair_scores_ap, dtype=np.float64)
    selected_programs: list[list[int]] = []
    selected_gains: list[float] = []
    for action_scores in scores:
        selected: list[int] = []
        final_gain = 0.0
        for _ in range(int(program_length)):
            candidates: list[tuple[float, int]] = []
            for pair_id in range(int(action_scores.size)):
                if pair_id in selected:
                    continue
                subset = selected + [pair_id]
                contrast = float(np.sum(action_scores[subset]))
                log_likelihood = np.tile(
                    np.asarray([[contrast, 0.0]], dtype=np.float64),
                    (3, 1),
                )
                candidates.append(
                    (
                        _numpy_information_gain(log_likelihood, weights_n),
                        pair_id,
                    )
                )
            final_gain, selected_pair = max(
                candidates,
                key=lambda candidate: (candidate[0], -candidate[1]),
            )
            selected.append(selected_pair)
        selected_programs.append(selected)
        selected_gains.append(final_gain)
    return (
        np.asarray(selected_programs, dtype=np.int64),
        np.asarray(selected_gains, dtype=np.float64),
    )


def _nonadditive_score(subset: tuple[int, ...]) -> float:
    """Return a set utility whose greedy result has a better one-swap neighbor."""
    key = tuple(sorted(subset))
    explicit = {
        (0,): 0.80,
        (1,): 0.70,
        (2,): 0.60,
        (3,): 0.10,
        (0, 1): 0.81,
        (0, 2): 0.82,
        (0, 3): 0.805,
        (1, 2): 1.40,
        (1, 3): 2.00,
        (2, 3): 0.65,
    }
    return explicit[key]


def test_batched_torch_greedy_matches_small_numpy_serial_oracle() -> None:
    """The batched search must match a scalar NumPy reference exactly."""
    pair_scores = np.asarray(
        [
            [0.1, 0.8, 0.2, 1.2, 0.4, 0.3, 1.0, 0.7, 0.6],
            [1.1, 0.2, 0.9, 0.4, 0.5, 1.3, 0.6, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.35, 0.65], dtype=np.float64)
    cache = _AdditiveLikelihoodCache(pair_scores, device=_test_device())

    actual = select_conditional_greedy_programs(
        cache,
        weights,
        num_orientations=3,
        program_length=3,
        enable_one_swap=False,
    )
    expected_programs, expected_gains = _serial_additive_greedy(
        pair_scores,
        weights,
        program_length=3,
    )

    assert np.array_equal(actual.program_pair_ids_al, expected_programs)
    assert np.allclose(
        actual.information_gain_a,
        expected_gains,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert cache.call_shapes == [(2, 9, 1), (2, 8, 2), (2, 7, 3)]


def test_search_consumes_runtime_prepared_cache_without_an_adapter() -> None:
    """The standard runtime Torch cache must satisfy the core protocol."""
    device = _test_device()
    pair_scores = torch.as_tensor(
        [0.1, 0.9, 0.3, 0.7],
        device=device,
        dtype=torch.float64,
    )
    view_log = torch.zeros(
        (1, 2, 2, 1, 1, 4),
        device=device,
        dtype=torch.float64,
    )
    view_log[:, :, 0, 0, 0, :] = pair_scores
    cache = PreparedTorchSubsetCrossLikelihood(
        leading_shape=(1,),
        view_node_log_aqnjrv=view_log,
        latent_log_weights_jr=torch.zeros(
            (1, 1),
            device=device,
            dtype=torch.float64,
        ),
    )

    result = select_conditional_greedy_programs(
        cache,
        np.asarray([0.5, 0.5], dtype=np.float64),
        num_orientations=2,
        program_length=2,
        enable_one_swap=False,
    )

    assert np.array_equal(result.program_pair_ids_al, [[1, 3]])
    assert result.selection_source_a == ("greedy",)


def test_eight_of_sixty_four_scores_exactly_484_greedy_candidates() -> None:
    """Eight sequential all-pair stages must score 484 subsets per pose."""
    cache = _AdditiveLikelihoodCache(
        np.zeros((1, 64), dtype=np.float64),
        device=_test_device(),
    )

    result = select_conditional_greedy_programs(
        cache,
        np.asarray([0.5, 0.5], dtype=np.float64),
        num_orientations=8,
        program_length=8,
        enable_one_swap=True,
    )

    assert result.greedy_candidate_count_per_action == 484
    assert result.one_swap_candidate_count_per_action == 448
    assert conditional_greedy_candidate_count(8, 8) == 484
    assert [shape[1] for shape in cache.call_shapes[:8]] == list(
        range(64, 56, -1)
    )
    assert cache.call_shapes[8] == (1, 448, 8)
    assert np.array_equal(
        result.program_pair_ids_al,
        np.arange(8, dtype=np.int64)[None, :],
    )
    assert len(set(result.program_pair_ids_al[0])) == 8


def test_one_swap_is_one_batch_and_never_decreases_information_gain() -> None:
    """The full single-swap neighborhood must be batched and monotone."""
    cache = _MappedSubsetLikelihoodCache(
        pair_count=4,
        score=_nonadditive_score,
        device=_test_device(),
    )

    result = select_conditional_greedy_programs(
        cache,
        np.asarray([0.5, 0.5], dtype=np.float64),
        num_orientations=2,
        program_length=2,
        enable_one_swap=True,
    )

    assert np.array_equal(result.greedy_program_pair_ids_al, [[0, 2]])
    assert np.array_equal(result.one_swap_best_program_pair_ids_al, [[1, 2]])
    assert np.array_equal(result.program_pair_ids_al, [[1, 2]])
    assert result.one_swap_candidate_count_per_action == 4
    assert cache.call_shapes[-1] == (1, 4, 2)
    assert bool(result.one_swap_applied_a[0]) is True
    assert result.one_swap_removed_position_a[0] == 0
    assert result.one_swap_added_pair_id_a[0] == 1
    assert result.information_gain_a[0] > result.greedy_information_gain_a[0]


def test_optional_incumbent_library_is_a_same_cache_eig_floor() -> None:
    """An optional legacy library must floor the refined finite-MC EIG."""
    cache = _MappedSubsetLikelihoodCache(
        pair_count=4,
        score=_nonadditive_score,
        device=_test_device(),
    )
    incumbents = np.asarray([[0, 1], [1, 3]], dtype=np.int64)

    result = select_conditional_greedy_programs(
        cache,
        np.asarray([0.5, 0.5], dtype=np.float64),
        num_orientations=2,
        program_length=2,
        enable_one_swap=True,
        incumbent_subsets=incumbents,
    )

    assert np.array_equal(result.program_pair_ids_al, [[1, 3]])
    assert result.selection_source_a == ("incumbent",)
    assert result.incumbent_candidate_count_per_action == 2
    assert bool(result.incumbent_floor_applied_a[0]) is True
    assert result.incumbent_best_index_a[0] == 1
    assert np.array_equal(result.incumbent_best_program_pair_ids_al, [[1, 3]])
    assert result.information_gain_a[0] >= (
        result.one_swap_best_information_gain_a[0]
    )
    assert cache.call_shapes[-1] == (1, 2, 2)


def test_public_recheck_helper_retains_paired_kl_samples_on_device() -> None:
    """Ambiguous contenders must expose paired samples without new physics."""
    cache = _AdditiveLikelihoodCache(
        np.asarray([[0.1, 0.4, 0.2, 0.8]], dtype=np.float64),
        device=_test_device(),
    )
    subsets = torch.as_tensor(
        [[[0, 1], [2, 3]]],
        device=cache.device,
        dtype=torch.long,
    )

    information_gain, kl_samples = evaluate_subset_information_gain_torch(
        cache,
        subsets,
        np.asarray([0.4, 0.6], dtype=np.float64),
    )

    assert information_gain.device == subsets.device
    assert kl_samples.device == subsets.device
    assert tuple(information_gain.shape) == (1, 2)
    assert tuple(kl_samples.shape) == (1, 2, 3)
    assert torch.allclose(information_gain, torch.mean(kl_samples, dim=2))
    assert cache.call_shapes == [(1, 2, 2)]


def test_ties_choose_lowest_pair_and_do_not_replace_the_greedy_program() -> None:
    """Exact ties must have deterministic pair IDs and refinement precedence."""
    cache = _AdditiveLikelihoodCache(
        np.zeros((1, 9), dtype=np.float64),
        device=_test_device(),
    )

    result = select_conditional_greedy_programs(
        cache,
        np.asarray([0.5, 0.5], dtype=np.float64),
        num_orientations=3,
        program_length=3,
        enable_one_swap=True,
        incumbent_subsets=np.asarray([[8, 7, 6]], dtype=np.int64),
    )

    assert np.array_equal(result.program_pair_ids_al, [[0, 1, 2]])
    assert result.selection_source_a == ("greedy",)
    assert not bool(result.one_swap_applied_a[0])
    assert not bool(result.incumbent_floor_applied_a[0])


def test_pair_count_is_derived_from_orientation_contract() -> None:
    """A cache with a non-square physical pair count must be rejected."""
    cache = _AdditiveLikelihoodCache(
        np.zeros((1, 8), dtype=np.float64),
        device=_test_device(),
    )

    with pytest.raises(ValueError, match=r"num_orientations \*\* 2"):
        select_conditional_greedy_programs(
            cache,
            np.asarray([0.5, 0.5], dtype=np.float64),
            num_orientations=3,
            program_length=3,
        )


def test_incumbent_subsets_must_contain_unique_pairs() -> None:
    """No incumbent compatibility program may repeat a physical pair."""
    cache = _AdditiveLikelihoodCache(
        np.zeros((1, 4), dtype=np.float64),
        device=_test_device(),
    )

    with pytest.raises(ValueError, match="unique"):
        select_conditional_greedy_programs(
            cache,
            np.asarray([0.5, 0.5], dtype=np.float64),
            num_orientations=2,
            program_length=2,
            enable_one_swap=False,
            incumbent_subsets=np.asarray([[1, 1]], dtype=np.int64),
        )
