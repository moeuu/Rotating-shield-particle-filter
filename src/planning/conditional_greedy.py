"""GPU-batched conditional-greedy search over shield-pair subsets.

The shared runtime owns full-spectrum physics, predictive observations, and
nuisance integration.  This module owns only information-theoretic subset
search.  It therefore consumes an opaque prepared likelihood cache and never
imports the legacy shield-program library or runtime implementation classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PreparedSubsetLikelihoodCache(Protocol):
    """Describe the runtime-owned likelihood interface used by the search.

    ``evaluate`` receives pair indices shaped ``(A, C, K)`` for ``A`` poses,
    ``C`` candidate subsets, and ``K`` selected views.  It returns joint log
    likelihoods shaped ``(A, C, Q, N)`` for predictive samples ``Q`` and PF
    hypothesis particles ``N``.  All candidates in one call are evaluated in
    one device batch; the cache is responsible for exact shared-nuisance
    integration within each subset.
    """

    action_count: int
    view_count: int
    device: object
    dtype: object

    def evaluate(self, subset_indices_ack: object) -> object:
        """Return batched joint log likelihoods for the requested subsets."""


@dataclass(frozen=True)
class ConditionalGreedyStage:
    """Store one conditional-greedy stage for every candidate pose."""

    stage_index: int
    candidate_count: int
    selected_pair_ids_a: NDArray[np.int64]
    selected_information_gain_a: NDArray[np.float64]
    runner_up_information_gain_a: NDArray[np.float64]
    information_gain_gap_a: NDArray[np.float64]


@dataclass(frozen=True)
class ConditionalGreedyResult:
    """Store greedy, one-swap, and optional incumbent-floor results."""

    pair_count: int
    program_length: int
    program_pair_ids_al: NDArray[np.int64]
    information_gain_a: NDArray[np.float64]
    selection_source_a: tuple[str, ...]
    greedy_program_pair_ids_al: NDArray[np.int64]
    greedy_information_gain_a: NDArray[np.float64]
    stages: tuple[ConditionalGreedyStage, ...]
    greedy_candidate_count_per_action: int
    one_swap_candidate_count_per_action: int
    one_swap_applied_a: NDArray[np.bool_]
    one_swap_removed_position_a: NDArray[np.int64]
    one_swap_added_pair_id_a: NDArray[np.int64]
    one_swap_best_program_pair_ids_al: NDArray[np.int64]
    one_swap_best_information_gain_a: NDArray[np.float64]
    incumbent_candidate_count_per_action: int
    incumbent_floor_applied_a: NDArray[np.bool_]
    incumbent_best_index_a: NDArray[np.int64]
    incumbent_best_program_pair_ids_al: NDArray[np.int64]
    incumbent_best_information_gain_a: NDArray[np.float64]


def conditional_greedy_candidate_count(
    num_orientations: int,
    program_length: int,
) -> int:
    """Return the number of conditional subsets scored per pose.

    The pair count is derived from the physical orientation contract as
    ``num_orientations ** 2``.  For eight orientations and eight views the
    count is ``64 + 63 + ... + 57 = 484``.
    """
    orientation_count = _positive_integer(
        num_orientations,
        name="num_orientations",
    )
    pair_count = orientation_count**2
    length = _validated_program_length(program_length, pair_count=pair_count)
    return length * pair_count - length * (length - 1) // 2


def information_gain_from_log_likelihood_torch(
    log_likelihood_acqn: object,
    weights_n: NDArray[np.float64] | Sequence[float],
) -> object:
    """Return finite-sample mutual information for a device action batch.

    The input is shaped ``(A, C, Q, N)``.  Posterior weights have shape
    ``(N,)`` and are shared across poses because all poses are alternatives
    under one PF posterior.  Reduction over particles and Monte Carlo samples
    remains on the input Torch device; only the small final planner result is
    copied to the CPU by :func:`select_conditional_greedy_programs`.
    """
    import torch

    information_gain_samples = (
        information_gain_samples_from_log_likelihood_torch(
            log_likelihood_acqn,
            weights_n,
        )
    )
    information_gain = torch.mean(information_gain_samples, dim=2)
    numerical_tolerance = 1.0e-10
    invalid = torch.stack(
        (
            torch.any(~torch.isfinite(information_gain)),
            torch.any(information_gain < -numerical_tolerance),
        )
    ).any()
    if bool(invalid.item()):
        raise RuntimeError(
            "Batched subset likelihoods produced invalid mutual information."
        )
    return torch.clamp(information_gain, min=0.0)


def information_gain_samples_from_log_likelihood_torch(
    log_likelihood_acqn: object,
    weights_n: NDArray[np.float64] | Sequence[float],
) -> object:
    """Return paired posterior-KL samples shaped ``(A, C, Q)`` on device.

    Keeping the Monte Carlo axis permits paired confidence intervals between
    competing programs evaluated from the same prepared observations.  A
    caller can prepare a second cache with an independent seed and pass only
    ambiguous final contenders to recheck them without recomputing response
    physics or evaluating every greedy candidate again.
    """
    import torch

    if not torch.is_tensor(log_likelihood_acqn):
        raise TypeError("log_likelihood_acqn must be a Torch tensor.")
    log_likelihood = log_likelihood_acqn
    if log_likelihood.ndim != 4:
        raise ValueError(
            "log_likelihood_acqn must be shaped (action, candidate, sample, "
            "particle)."
        )
    if log_likelihood.dtype != torch.float64:
        raise TypeError("Subset log likelihoods must use torch.float64.")
    weights = _normalised_weights_torch(
        weights_n,
        particle_count=int(log_likelihood.shape[3]),
        device=log_likelihood.device,
    )
    positive_prior = weights > 0.0
    active_likelihood = log_likelihood[..., positive_prior]
    active_weights = weights[positive_prior]
    log_prior = torch.log(active_weights).reshape(1, 1, 1, -1)
    log_joint = active_likelihood + log_prior
    log_evidence = torch.logsumexp(log_joint, dim=3, keepdim=True)
    posterior = torch.exp(log_joint - log_evidence)
    kl_terms = torch.where(
        posterior > 0.0,
        posterior * (active_likelihood - log_evidence),
        torch.zeros_like(posterior),
    )
    information_gain_samples = torch.sum(kl_terms, dim=3)
    numerical_tolerance = 1.0e-10
    invalid = torch.stack(
        (
            torch.any(torch.isnan(log_likelihood)),
            torch.any(torch.isposinf(log_likelihood)),
            torch.any(~torch.isfinite(log_evidence)),
            torch.any(~torch.isfinite(information_gain_samples)),
            torch.any(information_gain_samples < -numerical_tolerance),
        )
    ).any()
    if bool(invalid.item()):
        raise RuntimeError(
            "Batched subset likelihoods produced invalid KL samples."
        )
    return torch.clamp(information_gain_samples, min=0.0)


def evaluate_subset_information_gain_torch(
    cache: PreparedSubsetLikelihoodCache,
    subset_indices_ack: object,
    weights_n: NDArray[np.float64] | Sequence[float],
) -> tuple[object, object]:
    """Return batched EIG and paired KL samples from one prepared cache call.

    This public helper is intended for an ambiguity recheck of a small set of
    final contenders.  It accepts the same ``(A, C, K)`` subset layout as the
    main search and performs no response or observation generation.
    """
    import torch

    if not torch.is_tensor(subset_indices_ack):
        raise TypeError("subset_indices_ack must be a Torch tensor.")
    subsets = subset_indices_ack
    if subsets.ndim != 3 or subsets.dtype != torch.long:
        raise TypeError("Subset indices must be an int64 tensor shaped (A, C, K).")
    log_likelihood = cache.evaluate(subsets)
    if not torch.is_tensor(log_likelihood):
        raise TypeError("Prepared cache evaluate() must return a Torch tensor.")
    expected_shape = (
        int(subsets.shape[0]),
        int(subsets.shape[1]),
    )
    if log_likelihood.ndim != 4 or tuple(log_likelihood.shape[:2]) != expected_shape:
        raise RuntimeError(
            "Prepared cache returned an invalid (A, C, Q, N) likelihood shape."
        )
    if log_likelihood.device != subsets.device:
        raise TypeError("Prepared cache returned likelihoods on the wrong device.")
    information_gain_samples = (
        information_gain_samples_from_log_likelihood_torch(
            log_likelihood,
            weights_n,
        )
    )
    information_gain = torch.mean(information_gain_samples, dim=2)
    return information_gain, information_gain_samples


def select_conditional_greedy_programs(
    cache: PreparedSubsetLikelihoodCache,
    weights_n: NDArray[np.float64] | Sequence[float],
    *,
    num_orientations: int,
    program_length: int,
    enable_one_swap: bool = True,
    incumbent_subsets: object | None = None,
) -> ConditionalGreedyResult:
    """Select an EIG-only shield program independently for every pose.

    The only Python loop is the mathematically sequential greedy depth, whose
    full-simulation bound is the runtime ``program_length`` (normally eight).
    At each depth all remaining pair candidates for all poses are evaluated in
    one GPU call.  The optional one-swap neighborhood and optional incumbent
    floor are each evaluated in one additional GPU call.  No spatial, travel,
    robot-turn, or shield-rotation score can enter this function.

    ``incumbent_subsets`` is deliberately generic and optional.  Passing the
    old 48-program library supplies a non-regression floor under the exact same
    prepared Monte Carlo observations, while omitting it leaves a complete
    legacy-free implementation.  Removing that compatibility policy therefore
    requires no change to this module.
    """
    import torch

    if not isinstance(enable_one_swap, bool):
        raise ValueError("enable_one_swap must be a boolean.")
    orientation_count = _positive_integer(
        num_orientations,
        name="num_orientations",
    )
    pair_count = orientation_count**2
    length = _validated_program_length(program_length, pair_count=pair_count)
    action_count = _cache_action_count(cache)
    cache_pair_count = _cache_pair_count(cache)
    if cache_pair_count != pair_count:
        raise ValueError(
            "Prepared cache view count must equal num_orientations ** 2."
        )
    device = torch.device(getattr(cache, "device"))
    cache_dtype = getattr(cache, "dtype", torch.float64)
    if cache_dtype != torch.float64:
        raise TypeError("Prepared subset likelihood caches must use float64.")

    selected = torch.empty(
        (action_count, 0),
        device=device,
        dtype=torch.long,
    )
    available = torch.ones(
        (action_count, pair_count),
        device=device,
        dtype=torch.bool,
    )
    all_pair_ids = torch.arange(pair_count, device=device, dtype=torch.long)
    selected_pair_stages: list[object] = []
    selected_gain_stages: list[object] = []
    runner_up_stages: list[object] = []
    gap_stages: list[object] = []

    for stage_index in range(length):
        candidate_count = pair_count - stage_index
        remaining = all_pair_ids.expand(action_count, -1)[available].reshape(
            action_count,
            candidate_count,
        )
        prefix = selected[:, None, :].expand(-1, candidate_count, -1)
        candidate_subsets = torch.cat((prefix, remaining[..., None]), dim=2)
        candidate_gains = _evaluate_subset_information_gain(
            cache,
            candidate_subsets,
            weights_n,
        )
        selected_candidate_index = torch.argmax(candidate_gains, dim=1)
        selected_pair = torch.gather(
            remaining,
            1,
            selected_candidate_index[:, None],
        )[:, 0]
        selected_gain = torch.gather(
            candidate_gains,
            1,
            selected_candidate_index[:, None],
        )[:, 0]
        if candidate_count > 1:
            alternatives = candidate_gains.clone()
            alternatives.scatter_(
                1,
                selected_candidate_index[:, None],
                -torch.inf,
            )
            runner_up = torch.max(alternatives, dim=1).values
            gap = selected_gain - runner_up
        else:
            runner_up = torch.full_like(selected_gain, torch.nan)
            gap = torch.full_like(selected_gain, torch.nan)
        selected = torch.cat((selected, selected_pair[:, None]), dim=1)
        available.scatter_(1, selected_pair[:, None], False)
        selected_pair_stages.append(selected_pair)
        selected_gain_stages.append(selected_gain)
        runner_up_stages.append(runner_up)
        gap_stages.append(gap)

    greedy_program = selected.clone()
    greedy_information_gain = selected_gain_stages[-1].clone()
    current_program = greedy_program.clone()
    current_information_gain = greedy_information_gain.clone()
    selection_code = torch.zeros(
        action_count,
        device=device,
        dtype=torch.int8,
    )

    swap_candidate_count = length * (pair_count - length)
    swap_applied = torch.zeros(
        action_count,
        device=device,
        dtype=torch.bool,
    )
    swap_removed_position = torch.full(
        (action_count,),
        -1,
        device=device,
        dtype=torch.long,
    )
    swap_added_pair = torch.full_like(swap_removed_position, -1)
    swap_best_information_gain = greedy_information_gain.clone()
    swap_best_program = greedy_program.clone()
    if enable_one_swap and swap_candidate_count > 0:
        swap_candidates, swap_additions = _one_swap_candidate_subsets(
            greedy_program,
            pair_count=pair_count,
        )
        swap_gains = _evaluate_subset_information_gain(
            cache,
            swap_candidates,
            weights_n,
        )
        best_swap_index = torch.argmax(swap_gains, dim=1)
        swap_best_information_gain = torch.gather(
            swap_gains,
            1,
            best_swap_index[:, None],
        )[:, 0]
        swap_best_program = torch.gather(
            swap_candidates,
            1,
            best_swap_index[:, None, None].expand(-1, 1, length),
        )[:, 0]
        removed_positions = torch.div(
            best_swap_index,
            pair_count - length,
            rounding_mode="floor",
        )
        added_pairs = torch.gather(
            swap_additions,
            1,
            best_swap_index[:, None],
        )[:, 0]
        swap_applied = swap_best_information_gain > greedy_information_gain
        current_program = torch.where(
            swap_applied[:, None],
            swap_best_program,
            current_program,
        )
        current_information_gain = torch.where(
            swap_applied,
            swap_best_information_gain,
            current_information_gain,
        )
        swap_removed_position = torch.where(
            swap_applied,
            removed_positions,
            swap_removed_position,
        )
        swap_added_pair = torch.where(
            swap_applied,
            added_pairs,
            swap_added_pair,
        )
        selection_code = torch.where(
            swap_applied,
            torch.ones_like(selection_code),
            selection_code,
        )
    elif not enable_one_swap:
        swap_candidate_count = 0

    incumbent_candidate_count = 0
    incumbent_floor_applied = torch.zeros_like(swap_applied)
    incumbent_best_index = torch.full_like(swap_removed_position, -1)
    incumbent_best_information_gain = torch.full_like(
        current_information_gain,
        torch.nan,
    )
    incumbent_best_program = torch.full_like(current_program, -1)
    if incumbent_subsets is not None:
        incumbents = _validated_incumbent_subsets(
            incumbent_subsets,
            action_count=action_count,
            pair_count=pair_count,
            program_length=length,
            device=device,
        )
        incumbent_candidate_count = int(incumbents.shape[1])
        incumbent_gains = _evaluate_subset_information_gain(
            cache,
            incumbents,
            weights_n,
        )
        best_incumbent_index = torch.argmax(incumbent_gains, dim=1)
        incumbent_best_information_gain = torch.gather(
            incumbent_gains,
            1,
            best_incumbent_index[:, None],
        )[:, 0]
        incumbent_best_program = torch.gather(
            incumbents,
            1,
            best_incumbent_index[:, None, None].expand(-1, 1, length),
        )[:, 0]
        incumbent_floor_applied = (
            incumbent_best_information_gain > current_information_gain
        )
        current_program = torch.where(
            incumbent_floor_applied[:, None],
            incumbent_best_program,
            current_program,
        )
        current_information_gain = torch.where(
            incumbent_floor_applied,
            incumbent_best_information_gain,
            current_information_gain,
        )
        incumbent_best_index = best_incumbent_index
        selection_code = torch.where(
            incumbent_floor_applied,
            torch.full_like(selection_code, 2),
            selection_code,
        )

    stage_pairs = torch.stack(selected_pair_stages, dim=0).detach().cpu().numpy()
    stage_gains = torch.stack(selected_gain_stages, dim=0).detach().cpu().numpy()
    stage_runners = torch.stack(runner_up_stages, dim=0).detach().cpu().numpy()
    stage_gaps = torch.stack(gap_stages, dim=0).detach().cpu().numpy()
    stages = tuple(
        ConditionalGreedyStage(
            stage_index=index,
            candidate_count=pair_count - index,
            selected_pair_ids_a=np.asarray(stage_pairs[index], dtype=np.int64),
            selected_information_gain_a=np.asarray(
                stage_gains[index],
                dtype=np.float64,
            ),
            runner_up_information_gain_a=np.asarray(
                stage_runners[index],
                dtype=np.float64,
            ),
            information_gain_gap_a=np.asarray(
                stage_gaps[index],
                dtype=np.float64,
            ),
        )
        for index in range(length)
    )
    source_names = ("greedy", "one_swap", "incumbent")
    source_indices = np.asarray(
        selection_code.detach().cpu().numpy(),
        dtype=np.int64,
    )
    return ConditionalGreedyResult(
        pair_count=pair_count,
        program_length=length,
        program_pair_ids_al=_numpy_int64(current_program),
        information_gain_a=_numpy_float64(current_information_gain),
        selection_source_a=tuple(source_names[index] for index in source_indices),
        greedy_program_pair_ids_al=_numpy_int64(greedy_program),
        greedy_information_gain_a=_numpy_float64(greedy_information_gain),
        stages=stages,
        greedy_candidate_count_per_action=conditional_greedy_candidate_count(
            orientation_count,
            length,
        ),
        one_swap_candidate_count_per_action=swap_candidate_count,
        one_swap_applied_a=_numpy_bool(swap_applied),
        one_swap_removed_position_a=_numpy_int64(swap_removed_position),
        one_swap_added_pair_id_a=_numpy_int64(swap_added_pair),
        one_swap_best_program_pair_ids_al=_numpy_int64(swap_best_program),
        one_swap_best_information_gain_a=_numpy_float64(
            swap_best_information_gain
        ),
        incumbent_candidate_count_per_action=incumbent_candidate_count,
        incumbent_floor_applied_a=_numpy_bool(incumbent_floor_applied),
        incumbent_best_index_a=_numpy_int64(incumbent_best_index),
        incumbent_best_program_pair_ids_al=_numpy_int64(
            incumbent_best_program
        ),
        incumbent_best_information_gain_a=_numpy_float64(
            incumbent_best_information_gain
        ),
    )


def _evaluate_subset_information_gain(
    cache: PreparedSubsetLikelihoodCache,
    subset_indices_ack: object,
    weights_n: NDArray[np.float64] | Sequence[float],
) -> object:
    """Evaluate every supplied subset and reduce its likelihood to EIG."""
    information_gain, _ = evaluate_subset_information_gain_torch(
        cache,
        subset_indices_ack,
        weights_n,
    )
    return information_gain


def _one_swap_candidate_subsets(
    selected_al: object,
    *,
    pair_count: int,
) -> tuple[object, object]:
    """Build every single replacement in one vectorized device operation."""
    import torch

    if not torch.is_tensor(selected_al) or selected_al.ndim != 2:
        raise TypeError("selected_al must be a two-dimensional Torch tensor.")
    action_count, program_length = map(int, selected_al.shape)
    remaining_count = int(pair_count) - program_length
    if remaining_count <= 0:
        raise ValueError("One-swap requires at least one unselected pair.")
    available = torch.ones(
        (action_count, int(pair_count)),
        device=selected_al.device,
        dtype=torch.bool,
    )
    available.scatter_(1, selected_al, False)
    all_pairs = torch.arange(
        int(pair_count),
        device=selected_al.device,
        dtype=torch.long,
    )
    remaining = all_pairs.expand(action_count, -1)[available].reshape(
        action_count,
        remaining_count,
    )
    candidates = selected_al[:, None, None, :].expand(
        -1,
        program_length,
        remaining_count,
        -1,
    ).clone()
    replacement_positions = torch.arange(
        program_length,
        device=selected_al.device,
        dtype=torch.long,
    )[None, :, None, None].expand(action_count, -1, remaining_count, 1)
    replacement_values = remaining[:, None, :, None].expand(
        -1,
        program_length,
        -1,
        1,
    )
    candidates.scatter_(3, replacement_positions, replacement_values)
    flattened_candidates = candidates.reshape(
        action_count,
        program_length * remaining_count,
        program_length,
    )
    flattened_additions = remaining[:, None, :].expand(
        -1,
        program_length,
        -1,
    ).reshape(action_count, program_length * remaining_count)
    return flattened_candidates, flattened_additions


def _validated_incumbent_subsets(
    incumbent_subsets: object,
    *,
    action_count: int,
    pair_count: int,
    program_length: int,
    device: object,
) -> object:
    """Return validated shared or per-action incumbent subsets on device."""
    import torch

    subsets = torch.as_tensor(incumbent_subsets, device=device)
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if subsets.dtype not in integer_dtypes:
        raise ValueError("incumbent_subsets must contain integer pair IDs.")
    if subsets.ndim == 2:
        if tuple(subsets.shape[1:]) != (program_length,):
            raise ValueError(
                "Shared incumbent subsets must be shaped (candidate, view)."
            )
        subsets = subsets[None, :, :].expand(action_count, -1, -1)
    elif subsets.ndim == 3:
        if tuple(subsets.shape[:1] + subsets.shape[2:]) != (
            action_count,
            program_length,
        ):
            raise ValueError(
                "Per-action incumbent subsets must be shaped (A, C, K)."
            )
    else:
        raise ValueError("incumbent_subsets must be shaped (C, K) or (A, C, K).")
    if int(subsets.shape[1]) <= 0:
        raise ValueError("incumbent_subsets must contain at least one candidate.")
    subsets = subsets.to(dtype=torch.long).contiguous()
    invalid_range = torch.any((subsets < 0) | (subsets >= int(pair_count)))
    sorted_subsets = torch.sort(subsets, dim=2).values
    duplicate = torch.any(sorted_subsets[:, :, 1:] == sorted_subsets[:, :, :-1])
    if bool((invalid_range | duplicate).item()):
        raise ValueError(
            "Every incumbent subset must contain unique in-range pair IDs."
        )
    return subsets


def _normalised_weights_torch(
    weights_n: NDArray[np.float64] | Sequence[float],
    *,
    particle_count: int,
    device: object,
) -> object:
    """Validate posterior weights and copy one normalized vector to device."""
    import torch

    weights_numpy = np.asarray(weights_n, dtype=np.float64)
    if weights_numpy.shape != (int(particle_count),):
        raise ValueError("weights_n must match the likelihood particle dimension.")
    if np.any(~np.isfinite(weights_numpy)) or np.any(weights_numpy < 0.0):
        raise ValueError("weights_n must be finite and nonnegative.")
    total = float(np.sum(weights_numpy))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights_n must contain positive posterior mass.")
    return torch.as_tensor(
        weights_numpy / total,
        device=device,
        dtype=torch.float64,
    )


def _cache_action_count(cache: PreparedSubsetLikelihoodCache) -> int:
    """Return the validated number of poses stored in a prepared cache."""
    return _positive_integer(getattr(cache, "action_count"), name="action_count")


def _cache_pair_count(cache: PreparedSubsetLikelihoodCache) -> int:
    """Return the validated runtime view count with a compatibility alias."""
    raw_count = getattr(cache, "view_count", None)
    if raw_count is None:
        raw_count = getattr(cache, "pair_count", None)
    return _positive_integer(raw_count, name="view_count")


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive exact integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _validated_program_length(value: object, *, pair_count: int) -> int:
    """Return a runtime program length within the unique-pair domain."""
    length = _positive_integer(value, name="program_length")
    if length > int(pair_count):
        raise ValueError("program_length cannot exceed the unique pair count.")
    return length


def _numpy_float64(value: object) -> NDArray[np.float64]:
    """Copy a device tensor to a float64 NumPy array."""
    return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)


def _numpy_int64(value: object) -> NDArray[np.int64]:
    """Copy a device tensor to an int64 NumPy array."""
    return np.asarray(value.detach().cpu().numpy(), dtype=np.int64)


def _numpy_bool(value: object) -> NDArray[np.bool_]:
    """Copy a device tensor to a boolean NumPy array."""
    return np.asarray(value.detach().cpu().numpy(), dtype=np.bool_)


__all__ = [
    "ConditionalGreedyResult",
    "ConditionalGreedyStage",
    "PreparedSubsetLikelihoodCache",
    "conditional_greedy_candidate_count",
    "evaluate_subset_information_gain_torch",
    "information_gain_from_log_likelihood_torch",
    "information_gain_samples_from_log_likelihood_torch",
    "select_conditional_greedy_programs",
]
