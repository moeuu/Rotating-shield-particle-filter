"""Certified dyadic history refinement for exact RJ/MH decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


TPHT_RECENT_EXACT_STATIONS = 2
TPHT_REFINEMENT_LEAF_STATIONS = 4
TPHT_LOG_PROBABILITY_UPPER_BOUND = 0.0
TPHT_STAGED_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class TPHTHistoryBlock:
    """Describe one contiguous station block in newest-first evaluation order."""

    station_start: int
    station_stop: int
    level: int
    recent_exact: bool

    def __post_init__(self) -> None:
        """Validate one nonempty dyadic station interval."""
        if (
            isinstance(self.station_start, bool)
            or not isinstance(self.station_start, int)
            or isinstance(self.station_stop, bool)
            or not isinstance(self.station_stop, int)
            or self.station_start < 0
            or self.station_stop <= self.station_start
        ):
            raise ValueError("TPHT station bounds must form a nonempty interval.")
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
        ):
            raise ValueError("TPHT block level must be a nonnegative integer.")
        if not isinstance(self.recent_exact, bool):
            raise TypeError("TPHT recent_exact must be a boolean.")
        size = self.station_count
        if self.recent_exact:
            if size != 1 or self.level != 0:
                raise ValueError("A recent exact TPHT block must contain one station.")
        elif size != 1 << self.level:
            raise ValueError("An old TPHT block must have a power-of-two size.")

    @property
    def station_count(self) -> int:
        """Return the number of stations represented by this block."""
        return int(self.station_stop - self.station_start)

    @property
    def station_indices(self) -> tuple[int, ...]:
        """Return the chronological station indices covered by this block."""
        return tuple(range(self.station_start, self.station_stop))

    def children(self) -> tuple["TPHTHistoryBlock", "TPHTHistoryBlock"]:
        """Split one old dyadic block into chronological exact children."""
        if self.recent_exact or self.level == 0:
            raise ValueError("A TPHT leaf cannot be split further.")
        midpoint = self.station_start + self.station_count // 2
        child_level = self.level - 1
        return (
            TPHTHistoryBlock(
                station_start=self.station_start,
                station_stop=midpoint,
                level=child_level,
                recent_exact=False,
            ),
            TPHTHistoryBlock(
                station_start=midpoint,
                station_stop=self.station_stop,
                level=child_level,
                recent_exact=False,
            ),
        )


@dataclass(frozen=True)
class TPHTProposalDecision:
    """Carry one exact-target hierarchical decision and its history audit."""

    accepted: object
    proposed_target_log_likelihood: object
    proposed_station_log_likelihood: object
    diagnostic_delta_log_likelihood: object
    diagnostic_log_acceptance_ratio: object
    likelihood_exact: object
    evaluated_station_count: object
    early_rejected: object
    block_evaluation_count: int
    maximum_block_level: int
    refinement_round_count: int
    refinement_bound_rejected: object
    exact_rejected: object
    staged_replay_row_count: int
    first_stage_station_count: object
    first_stage_rejected: object


def build_tpht_history_blocks(
    station_count: int,
    *,
    recent_exact_stations: int = TPHT_RECENT_EXACT_STATIONS,
) -> tuple[TPHTHistoryBlock, ...]:
    """Partition acquired history into recent leaves and an old dyadic forest.

    The returned blocks are disjoint and cover every station exactly once.
    Evaluation order is newest first.  The loop spans only the binary digits of
    the fixed live horizon, so it cannot become a particle- or view-scale
    scalar runtime loop.
    """
    for name, value in (
        ("station_count", station_count),
        ("recent_exact_stations", recent_exact_stations),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if station_count <= 0:
        raise ValueError("TPHT requires at least one acquired station.")
    if recent_exact_stations <= 0:
        raise ValueError("TPHT requires at least one recent exact station.")

    recent_count = min(station_count, recent_exact_stations)
    recent_start = station_count - recent_count
    recent = tuple(
        TPHTHistoryBlock(
            station_start=index,
            station_stop=index + 1,
            level=0,
            recent_exact=True,
        )
        for index in range(station_count - 1, recent_start - 1, -1)
    )

    old_blocks_chronological: list[TPHTHistoryBlock] = []
    cursor = 0
    remaining = recent_start
    level = remaining.bit_length() - 1
    while remaining:
        size = 1 << level
        if size <= remaining:
            old_blocks_chronological.append(
                TPHTHistoryBlock(
                    station_start=cursor,
                    station_stop=cursor + size,
                    level=level,
                    recent_exact=False,
                )
            )
            cursor += size
            remaining -= size
        level -= 1
    blocks = recent + tuple(reversed(old_blocks_chronological))
    covered = sorted(
        index
        for block in blocks
        for index in range(block.station_start, block.station_stop)
    )
    if covered != list(range(station_count)):
        raise RuntimeError("TPHT construction did not cover history exactly once.")
    return blocks


def tpht_block_count_upper_bound(station_count: int) -> int:
    """Return a conservative logarithmic bound on active TPHT blocks."""
    if isinstance(station_count, bool) or not isinstance(station_count, int):
        raise TypeError("station_count must be an integer.")
    if station_count <= 0:
        raise ValueError("station_count must be positive.")
    recent = min(station_count, TPHT_RECENT_EXACT_STATIONS)
    old = station_count - recent
    return int(recent + old.bit_count())


def build_tpht_refinement_leaves(
    station_count: int,
    *,
    recent_exact_stations: int = TPHT_RECENT_EXACT_STATIONS,
    maximum_leaf_stations: int = TPHT_REFINEMENT_LEAF_STATIONS,
) -> tuple[TPHTHistoryBlock, ...]:
    """Return disjoint exact leaves obtained by splitting every old root.

    The tree contains at most 16 stations in production.  Expanding its small
    set of metadata nodes on the host is therefore bounded independently of
    particle count; all likelihood evaluation over leaf rows remains batched.
    """
    if (
        isinstance(maximum_leaf_stations, bool)
        or not isinstance(maximum_leaf_stations, int)
        or maximum_leaf_stations <= 0
    ):
        raise ValueError("maximum_leaf_stations must be a positive integer.")
    roots = build_tpht_history_blocks(
        station_count,
        recent_exact_stations=recent_exact_stations,
    )
    leaves: list[TPHTHistoryBlock] = []
    pending = list(roots)
    while pending:
        block = pending.pop(0)
        if block.recent_exact or block.station_count <= maximum_leaf_stations:
            leaves.append(block)
            continue
        left, right = block.children()
        pending[0:0] = [right, left]
    covered = sorted(
        station
        for block in leaves
        for station in block.station_indices
    )
    if covered != list(range(station_count)):
        raise RuntimeError("TPHT refinement leaves do not cover exact history.")
    return tuple(leaves)


def run_tpht_hierarchical_exact_acceptance_torch(
    *,
    current_station_log_likelihood_ps: object,
    base_target_log_likelihood_p: object,
    log_non_likelihood_ratio_p: object,
    log_uniform_p: object,
    log_refinement_uniform_p: object,
    support_p: object,
    target_beta: float,
    evaluate_station_block: Callable[[object, int, int, bool], object],
    stage_accepted_rows: bool,
) -> TPHTProposalDecision:
    """Return an exact-target MH decision by refining dyadic history.

    The latest station is an exact, unweighted first-stage factor rather than
    a representative for older data.  Rows passing it descend through real
    dyadic child blocks.  The independent refinement test contains every older
    station, and unevaluated children use only the universal discrete-PMF bound
    ``log p(y | state) <= 0`` for certified early rejection.  The product of
    the two MH factors is the unchanged complete-history RJ ratio, so the
    resulting delayed-acceptance kernel is reversible for the exact target.

    No representative station, weighted pseudo-likelihood, or approximate
    posterior is used.
    """
    import math

    import torch

    if not callable(evaluate_station_block):
        raise TypeError("TPHT station evaluator must be callable.")
    tensors = (
        current_station_log_likelihood_ps,
        base_target_log_likelihood_p,
        log_non_likelihood_ratio_p,
        log_uniform_p,
        log_refinement_uniform_p,
        support_p,
    )
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("TPHT exact refinement requires Torch tensors.")
    current_station = current_station_log_likelihood_ps
    base_target = base_target_log_likelihood_p.reshape(-1)
    non_likelihood = log_non_likelihood_ratio_p.reshape(-1)
    log_uniform = log_uniform_p.reshape(-1)
    log_refinement_uniform = log_refinement_uniform_p.reshape(-1)
    support = support_p.to(dtype=torch.bool).reshape(-1)
    row_count = int(base_target.numel())
    if current_station.ndim != 2:
        raise ValueError("TPHT current station target must be a matrix.")
    station_count = int(current_station.shape[1])
    if (
        row_count <= 0
        or station_count <= 0
        or int(current_station.shape[0]) != row_count
        or any(
            tuple(value.reshape(-1).shape) != (row_count,)
            for value in (
                non_likelihood,
                log_uniform,
                log_refinement_uniform,
                support,
            )
        )
    ):
        raise ValueError("TPHT exact-refinement rows are not aligned.")
    reference = base_target
    if any(value.device != reference.device for value in tensors):
        raise ValueError("TPHT exact-refinement tensors changed device.")
    if (
        reference.dtype != torch.float64
        or current_station.dtype != reference.dtype
        or non_likelihood.dtype != reference.dtype
        or log_uniform.dtype != reference.dtype
        or log_refinement_uniform.dtype != reference.dtype
    ):
        raise TypeError("TPHT exact refinement requires float64 targets.")
    if bool(
        torch.any(~torch.isfinite(current_station)).item()
        or torch.any(~torch.isfinite(base_target)).item()
        or torch.any(support & ~torch.isfinite(non_likelihood)).item()
        or torch.any(~torch.isfinite(log_uniform)).item()
        or torch.any(~torch.isfinite(log_refinement_uniform)).item()
        or torch.any(log_uniform > 0.0).item()
        or torch.any(log_refinement_uniform > 0.0).item()
    ):
        raise RuntimeError("TPHT target or MH threshold is invalid.")
    if not isinstance(stage_accepted_rows, bool):
        raise TypeError("stage_accepted_rows must be a boolean.")
    non_likelihood = torch.where(
        support,
        non_likelihood,
        torch.full_like(non_likelihood, float("-inf")),
    )
    beta = float(target_beta)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("TPHT target_beta must lie in [0, 1].")
    station_powers = torch.ones(
        station_count,
        device=reference.device,
        dtype=reference.dtype,
    )
    station_powers[-1] = beta
    reconstructed = torch.sum(
        current_station * station_powers[None, :],
        dim=1,
    )
    if not torch.allclose(
        reconstructed,
        base_target,
        rtol=2.0e-12,
        atol=1.0e-8,
    ):
        maximum_error = float(
            torch.max(torch.abs(reconstructed - base_target)).item()
        )
        raise RuntimeError(
            "TPHT station cache differs from the exact target "
            f"(max error {maximum_error:.6g})."
        )

    if bool(
        torch.any(
            current_station
            > TPHT_LOG_PROBABILITY_UPPER_BOUND + 1.0e-8
        ).item()
    ):
        raise RuntimeError("TPHT current station target violates its PMF bound.")

    roots = build_tpht_history_blocks(station_count)
    blocks = build_tpht_refinement_leaves(station_count)
    proposed_station = torch.full_like(current_station, float("nan"))
    evaluated_delta = torch.zeros_like(base_target)
    refinement_delta = torch.zeros_like(base_target)
    unresolved_upper = torch.sum(
        (TPHT_LOG_PROBABILITY_UPPER_BOUND - current_station)
        * station_powers[None, :],
        dim=1,
    )
    evaluated_station_count = torch.zeros(
        row_count,
        device=reference.device,
        dtype=torch.long,
    )
    first_stage_station_count = torch.zeros_like(evaluated_station_count)
    first_stage_rejected = torch.zeros_like(support)
    refinement_bound_rejected = torch.zeros_like(support)
    block_evaluation_count = 0
    refinement_round_count = 0
    staged_replay_row_count = 0
    maximum_block_level = max(int(block.level) for block in roots)
    active = torch.zeros_like(support)
    first_block = blocks[0]
    supported_rows = torch.nonzero(support, as_tuple=False).reshape(-1)
    if int(supported_rows.numel()):
        first_values = evaluate_station_block(
            supported_rows,
            first_block.station_start,
            first_block.station_stop,
            False,
        )
        expected_shape = (int(supported_rows.numel()), first_block.station_count)
        if (
            not torch.is_tensor(first_values)
            or first_values.device != reference.device
            or first_values.dtype != reference.dtype
            or tuple(first_values.shape) != expected_shape
            or bool(
                torch.any(torch.isnan(first_values)).item()
                or torch.any(torch.isposinf(first_values)).item()
                or torch.any(
                    first_values
                    > TPHT_LOG_PROBABILITY_UPPER_BOUND + 1.0e-8
                ).item()
            )
        ):
            raise RuntimeError("TPHT exact first-stage likelihood is invalid.")
        first_slice = slice(first_block.station_start, first_block.station_stop)
        first_powers = station_powers[first_slice]
        first_current = current_station[supported_rows, first_slice]
        first_delta = torch.sum(
            (first_values - first_current) * first_powers[None, :],
            dim=1,
        )
        first_upper = torch.sum(
            (TPHT_LOG_PROBABILITY_UPPER_BOUND - first_current)
            * first_powers[None, :],
            dim=1,
        )
        proposed_station[supported_rows, first_slice] = first_values
        evaluated_delta[supported_rows] = first_delta
        unresolved_upper[supported_rows] -= first_upper
        evaluated_station_count[supported_rows] = first_block.station_count
        first_stage_station_count[supported_rows] = first_block.station_count
        active[supported_rows] = (
            log_uniform[supported_rows]
            < first_delta + non_likelihood[supported_rows]
        )
        block_evaluation_count += 1
        refinement_round_count += 1
    first_stage_rejected = support & ~active

    impossible = active & (log_refinement_uniform >= unresolved_upper)
    refinement_bound_rejected[impossible] = True
    active[impossible] = False

    for block in blocks[1:]:
        active_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
        if not int(active_rows.numel()):
            break
        values = evaluate_station_block(
            active_rows,
            block.station_start,
            block.station_stop,
            False,
        )
        expected_shape = (int(active_rows.numel()), block.station_count)
        if (
            not torch.is_tensor(values)
            or values.device != reference.device
            or values.dtype != reference.dtype
        ):
            raise RuntimeError(
                "TPHT exact child likelihood changed backend or dtype."
            )
        if tuple(values.shape) != expected_shape or bool(
            torch.any(torch.isnan(values)).item()
            or torch.any(torch.isposinf(values)).item()
            or torch.any(
                values
                > TPHT_LOG_PROBABILITY_UPPER_BOUND + 1.0e-8
            ).item()
        ):
            raise RuntimeError(
                "TPHT exact child likelihood violates its PMF contract."
            )
        block_slice = slice(block.station_start, block.station_stop)
        powers = station_powers[block_slice]
        current_block = current_station[active_rows, block_slice]
        block_delta = torch.sum(
            (values - current_block) * powers[None, :],
            dim=1,
        )
        block_upper = torch.sum(
            (TPHT_LOG_PROBABILITY_UPPER_BOUND - current_block)
            * powers[None, :],
            dim=1,
        )
        proposed_station[active_rows, block_slice] = values
        evaluated_delta[active_rows] += block_delta
        refinement_delta[active_rows] += block_delta
        unresolved_upper[active_rows] -= block_upper
        evaluated_station_count[active_rows] += block.station_count
        block_evaluation_count += 1
        refinement_round_count += 1
        ratio_upper = (
            refinement_delta[active_rows]
            + unresolved_upper[active_rows]
        )
        if bool(torch.any(torch.isnan(ratio_upper)).item()):
            raise RuntimeError("TPHT certified ratio upper bound is invalid.")
        unresolved_local = (
            evaluated_station_count[active_rows] < station_count
        )
        rejected_local = (
            unresolved_local
            & (log_refinement_uniform[active_rows] >= ratio_upper)
        )
        if bool(torch.any(rejected_local).item()):
            rejected_rows = active_rows[rejected_local]
            refinement_bound_rejected[rejected_rows] = True
            active[rejected_rows] = False

    likelihood_exact = support & (evaluated_station_count == station_count)
    if bool(torch.any(active & ~likelihood_exact).item()):
        raise RuntimeError("TPHT refinement ended with unresolved active rows.")
    accepted = active & likelihood_exact & (
        log_refinement_uniform < refinement_delta
    )
    early_rejected = first_stage_rejected | refinement_bound_rejected
    exact_rejected = likelihood_exact & ~accepted

    proposed_target = base_target.clone()
    proposed_target[likelihood_exact] = (
        base_target[likelihood_exact] + evaluated_delta[likelihood_exact]
    )
    diagnostic_delta = torch.where(
        likelihood_exact,
        evaluated_delta,
        evaluated_delta + unresolved_upper,
    )
    diagnostic_ratio = diagnostic_delta + non_likelihood
    diagnostic_delta = torch.where(
        support,
        diagnostic_delta,
        torch.full_like(diagnostic_delta, float("nan")),
    )
    diagnostic_ratio = torch.where(
        support,
        diagnostic_ratio,
        torch.full_like(diagnostic_ratio, float("-inf")),
    )

    if stage_accepted_rows:
        accepted_rows = torch.nonzero(accepted, as_tuple=False).reshape(-1)
        if int(accepted_rows.numel()):
            replay_station = evaluate_station_block(
                accepted_rows,
                0,
                station_count,
                True,
            )
            expected_shape = (int(accepted_rows.numel()), station_count)
            if (
                not torch.is_tensor(replay_station)
                or replay_station.device != reference.device
                or replay_station.dtype != reference.dtype
            ):
                raise RuntimeError(
                    "TPHT staged replay changed likelihood backend or dtype."
                )
            if tuple(replay_station.shape) != expected_shape or bool(
                torch.any(torch.isnan(replay_station)).item()
                or torch.any(torch.isposinf(replay_station)).item()
                or torch.any(
                    replay_station
                    > TPHT_LOG_PROBABILITY_UPPER_BOUND + 1.0e-8
                ).item()
            ):
                raise RuntimeError(
                    "TPHT staged replay violates its station PMF contract."
                )
            if not torch.allclose(
                replay_station,
                proposed_station[accepted_rows],
                rtol=2.0e-12,
                atol=TPHT_STAGED_REPLAY_ABSOLUTE_TOLERANCE,
            ):
                maximum_error = float(
                    torch.max(
                        torch.abs(
                            replay_station - proposed_station[accepted_rows]
                        )
                    ).item()
                )
                raise RuntimeError(
                    "TPHT staged replay differs from refined exact history "
                    f"(max error {maximum_error:.6g})."
                )
            block_evaluation_count += 1
            staged_replay_row_count = int(accepted_rows.numel())
    return TPHTProposalDecision(
        accepted=accepted,
        proposed_target_log_likelihood=proposed_target,
        proposed_station_log_likelihood=proposed_station,
        diagnostic_delta_log_likelihood=diagnostic_delta,
        diagnostic_log_acceptance_ratio=diagnostic_ratio,
        likelihood_exact=likelihood_exact,
        evaluated_station_count=evaluated_station_count,
        early_rejected=early_rejected,
        block_evaluation_count=block_evaluation_count,
        maximum_block_level=maximum_block_level,
        refinement_round_count=refinement_round_count,
        refinement_bound_rejected=refinement_bound_rejected,
        exact_rejected=exact_rejected,
        staged_replay_row_count=staged_replay_row_count,
        first_stage_station_count=first_stage_station_count,
        first_stage_rejected=first_stage_rejected,
    )


__all__ = [
    "TPHTHistoryBlock",
    "TPHTProposalDecision",
    "TPHT_LOG_PROBABILITY_UPPER_BOUND",
    "TPHT_RECENT_EXACT_STATIONS",
    "TPHT_REFINEMENT_LEAF_STATIONS",
    "TPHT_STAGED_REPLAY_ABSOLUTE_TOLERANCE",
    "build_tpht_history_blocks",
    "build_tpht_refinement_leaves",
    "run_tpht_hierarchical_exact_acceptance_torch",
    "tpht_block_count_upper_bound",
]
