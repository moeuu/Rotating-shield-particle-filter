"""Tests for transition-preserving dyadic history scheduling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pf.estimator_structural import EstimatorStructuralProposalMixin
from pf.history_tree import (
    build_tpht_history_blocks,
    build_tpht_refinement_leaves,
    tpht_block_count_upper_bound,
)
from pf.joint_transport_cache import JointTransportCache
from pf.particle_filter import StructuralGeometryBatch


class _TPHTEvaluatorHarness(EstimatorStructuralProposalMixin):
    """Provide deterministic station values to the production TPHT scheduler."""

    def __init__(
        self,
        candidate_station: object,
    ) -> None:
        """Initialize a minimal active history without constructing an estimator."""
        torch = pytest.importorskip("torch")
        self.candidate_station = torch.as_tensor(
            candidate_station,
            dtype=torch.float64,
        )
        self._joint_structural_transport_cache = object.__new__(
            JointTransportCache
        )
        self._active_joint_station_history = tuple(
            SimpleNamespace(fe_indices=np.asarray([0], dtype=np.int64))
            for _ in range(int(self.candidate_station.shape[1]))
        )
        self.evaluation_calls: list[tuple[int, int, bool]] = []
        self.staged_overlay_rows: list[int] = []

    def _joint_structural_target_evaluator(
        self,
        *,
        particle_indices: object,
        target_beta: float,
        station_start: int | None = None,
        station_stop: int | None = None,
        return_station_log_likelihood: bool = False,
        stage_unit_transport: bool = True,
        **_: object,
    ) -> object:
        """Return the configured exact station log-PMF for requested rows."""
        torch = pytest.importorskip("torch")
        start = 0 if station_start is None else int(station_start)
        stop = (
            int(self.candidate_station.shape[1])
            if station_stop is None
            else int(station_stop)
        )
        rows = torch.as_tensor(particle_indices, dtype=torch.long).reshape(-1)
        source = self.candidate_station
        station = torch.index_select(source, 0, rows)[
            :, start:stop
        ]
        powers = torch.ones(stop - start, dtype=torch.float64)
        if stop == int(self.candidate_station.shape[1]):
            powers[-1] = float(target_beta)
        target = torch.sum(station * powers[None, :], dim=1)
        self.evaluation_calls.append(
            (
                start,
                stop,
                bool(stage_unit_transport),
            )
        )
        if stage_unit_transport:
            self.staged_overlay_rows.extend(int(value) for value in rows.tolist())
        if return_station_log_likelihood:
            return target, station
        return target


@pytest.mark.parametrize("station_count", range(1, 17))
def test_tpht_blocks_cover_history_once_with_logarithmic_forest(
    station_count: int,
) -> None:
    """Every fixed-horizon station must occur in exactly one scheduled block."""
    blocks = build_tpht_history_blocks(station_count)
    covered = [
        index
        for block in blocks
        for index in block.station_indices
    ]
    assert sorted(covered) == list(range(station_count))
    assert len(covered) == len(set(covered))
    assert len(blocks) <= tpht_block_count_upper_bound(station_count)
    assert blocks[0].station_indices == (station_count - 1,)
    assert all(
        block.recent_exact or block.station_count == 1 << block.level
        for block in blocks
    )


def test_tpht_sixteen_station_schedule_has_five_blocks() -> None:
    """The live 16-station horizon should require only five active blocks."""
    blocks = build_tpht_history_blocks(16)
    assert [block.station_indices for block in blocks] == [
        (15,),
        (14,),
        (12, 13),
        (8, 9, 10, 11),
        tuple(range(8)),
    ]


def test_tpht_sixteen_station_roots_expand_to_exact_child_leaves() -> None:
    """Old roots must split into GPU-aligned four-station child blocks."""
    leaves = build_tpht_refinement_leaves(16)
    assert [block.station_indices for block in leaves] == [
        (15,),
        (14,),
        (12, 13),
        (8, 9, 10, 11),
        (4, 5, 6, 7),
        (0, 1, 2, 3),
    ]


def test_tpht_certified_rejection_and_exact_rows_are_audited() -> None:
    """Certified early rejection and exact rows must remain distinguishable."""
    torch = pytest.importorskip("torch")
    candidate = torch.as_tensor(
        [
            [0.0, 0.0, 0.0, -10.0],
            [0.0, 0.0, 0.0, 0.0],
            [-0.2, -0.2, -0.2, -0.2],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    harness = _TPHTEvaluatorHarness(candidate)
    current = torch.zeros_like(candidate)
    base = torch.sum(current, dim=1)
    non_likelihood = torch.as_tensor(
        [0.0, 0.0, 0.0, float("nan")],
        dtype=torch.float64,
    )
    log_uniform = torch.full((4,), -1.0, dtype=torch.float64)
    support = torch.as_tensor([True, True, True, False])
    decision = harness._joint_structural_history_tree_evaluator(
        filt=object(),
        data=StructuralGeometryBatch(
            detector_positions=np.zeros((4, 3), dtype=np.float64),
            fe_indices=np.zeros(4, dtype=np.int64),
            pb_indices=np.zeros(4, dtype=np.int64),
            live_times=np.ones(4, dtype=np.float64),
            station_sequence_ids=np.arange(4, dtype=np.int64),
        ),
        positions_pks=torch.zeros((4, 1, 3), dtype=torch.float64),
        chart_ids_pk=torch.zeros((4, 1), dtype=torch.long),
        strengths_pk=torch.ones((4, 1), dtype=torch.float64),
        particle_indices=torch.arange(4, dtype=torch.long),
        current_station_log_likelihood_ps=current,
        base_target_log_likelihood_p=base,
        log_non_likelihood_ratio_p=non_likelihood,
        log_uniform_p=log_uniform,
        log_refinement_uniform_p=log_uniform,
        support_p=support,
        target_beta=1.0,
        tempering_start_row=3,
        move_family="position_strength",
    )
    exact_ratio = torch.sum(candidate - current, dim=1) + non_likelihood
    expected = support & (log_uniform < exact_ratio)
    assert torch.equal(decision.accepted, expected)
    assert decision.accepted.tolist() == [False, True, True, False]
    assert decision.early_rejected.tolist() == [True, False, False, False]
    assert decision.likelihood_exact.tolist() == [False, True, True, False]
    assert decision.first_stage_rejected.tolist() == [True, False, False, False]
    assert decision.refinement_bound_rejected.tolist() == [
        False,
        False,
        False,
        False,
    ]
    assert decision.exact_rejected.tolist() == [False, False, False, False]
    assert decision.evaluated_station_count.tolist() == [1, 4, 4, 0]
    assert decision.staged_replay_row_count == 2
    assert harness.evaluation_calls[-1] == (0, 4, True)
    assert harness.staged_overlay_rows == [1, 2]


def test_tpht_rejects_nonfinite_ratio_on_supported_row() -> None:
    """Undefined MH terms may be masked only by an explicit false support."""
    torch = pytest.importorskip("torch")
    candidate = torch.zeros((1, 1), dtype=torch.float64)
    harness = _TPHTEvaluatorHarness(candidate)
    with pytest.raises(RuntimeError, match="target or MH threshold is invalid"):
        harness._joint_structural_history_tree_evaluator(
            filt=object(),
            data=StructuralGeometryBatch(
                detector_positions=np.zeros((1, 3), dtype=np.float64),
                fe_indices=np.zeros(1, dtype=np.int64),
                pb_indices=np.zeros(1, dtype=np.int64),
                live_times=np.ones(1, dtype=np.float64),
                station_sequence_ids=np.zeros(1, dtype=np.int64),
            ),
            positions_pks=torch.zeros((1, 1, 3), dtype=torch.float64),
            chart_ids_pk=torch.zeros((1, 1), dtype=torch.long),
            strengths_pk=torch.ones((1, 1), dtype=torch.float64),
            particle_indices=torch.zeros(1, dtype=torch.long),
            current_station_log_likelihood_ps=torch.zeros_like(candidate),
            base_target_log_likelihood_p=torch.zeros(1, dtype=torch.float64),
            log_non_likelihood_ratio_p=torch.full(
                (1,),
                float("nan"),
                dtype=torch.float64,
            ),
            log_uniform_p=torch.full((1,), -1.0, dtype=torch.float64),
            log_refinement_uniform_p=torch.full(
                (1,), -1.0, dtype=torch.float64
            ),
            support_p=torch.ones(1, dtype=torch.bool),
            target_beta=1.0,
            tempering_start_row=0,
            move_family="split",
        )


def test_tpht_refines_recently_ambiguous_row_to_exact_old_history() -> None:
    """A recent-window tie must expand until an old exact factor decides it."""
    torch = pytest.importorskip("torch")
    candidate = torch.as_tensor(
        [[-10.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    harness = _TPHTEvaluatorHarness(candidate)
    decision = harness._joint_structural_history_tree_evaluator(
        filt=object(),
        data=StructuralGeometryBatch(
            detector_positions=np.zeros((4, 3), dtype=np.float64),
            fe_indices=np.zeros(4, dtype=np.int64),
            pb_indices=np.zeros(4, dtype=np.int64),
            live_times=np.ones(4, dtype=np.float64),
            station_sequence_ids=np.arange(4, dtype=np.int64),
        ),
        positions_pks=torch.zeros((1, 1, 3), dtype=torch.float64),
        chart_ids_pk=torch.zeros((1, 1), dtype=torch.long),
        strengths_pk=torch.ones((1, 1), dtype=torch.float64),
        particle_indices=torch.zeros(1, dtype=torch.long),
        current_station_log_likelihood_ps=torch.zeros_like(candidate),
        base_target_log_likelihood_p=torch.zeros(1, dtype=torch.float64),
        log_non_likelihood_ratio_p=torch.zeros(1, dtype=torch.float64),
        log_uniform_p=torch.full((1,), -1.0, dtype=torch.float64),
        log_refinement_uniform_p=torch.full(
            (1,), -1.0, dtype=torch.float64
        ),
        support_p=torch.ones(1, dtype=torch.bool),
        target_beta=1.0,
        tempering_start_row=3,
        move_family="position_strength",
    )
    assert decision.likelihood_exact.tolist() == [True]
    assert decision.accepted.tolist() == [False]
    assert decision.exact_rejected.tolist() == [True]
    assert decision.diagnostic_log_acceptance_ratio.tolist() == [-10.0]
    assert decision.evaluated_station_count.tolist() == [4]
    assert not any(staged for _, _, staged in harness.evaluation_calls)


def test_tpht_sixteen_station_rejections_stop_at_first_exact_leaf() -> None:
    """Clearly poor recent proposals must stop after one certified station."""
    torch = pytest.importorskip("torch")
    row_count = 3
    candidate = torch.zeros((row_count, 16), dtype=torch.float64)
    candidate[:, -1] = -100.0
    harness = _TPHTEvaluatorHarness(candidate)
    decision = harness._joint_structural_history_tree_evaluator(
        filt=object(),
        data=StructuralGeometryBatch(
            detector_positions=np.zeros((16, 3), dtype=np.float64),
            fe_indices=np.zeros(16, dtype=np.int64),
            pb_indices=np.zeros(16, dtype=np.int64),
            live_times=np.ones(16, dtype=np.float64),
            station_sequence_ids=np.arange(16, dtype=np.int64),
        ),
        positions_pks=torch.zeros((row_count, 1, 3), dtype=torch.float64),
        chart_ids_pk=torch.zeros((row_count, 1), dtype=torch.long),
        strengths_pk=torch.ones((row_count, 1), dtype=torch.float64),
        particle_indices=torch.arange(row_count, dtype=torch.long),
        current_station_log_likelihood_ps=torch.zeros_like(candidate),
        base_target_log_likelihood_p=torch.zeros(
            row_count,
            dtype=torch.float64,
        ),
        log_non_likelihood_ratio_p=torch.zeros(
            row_count,
            dtype=torch.float64,
        ),
        log_uniform_p=torch.full(
            (row_count,),
            -1.0,
            dtype=torch.float64,
        ),
        log_refinement_uniform_p=torch.full(
            (row_count,), -1.0, dtype=torch.float64
        ),
        support_p=torch.ones(row_count, dtype=torch.bool),
        target_beta=1.0,
        tempering_start_row=15,
        move_family="position_strength",
    )
    assert decision.accepted.tolist() == [False, False, False]
    assert decision.evaluated_station_count.tolist() == [1, 1, 1]
    assert len(harness.evaluation_calls) == 1
    assert harness.evaluation_calls[0] == (15, 16, False)
    assert not any(staged for _, _, staged in harness.evaluation_calls)


def test_tpht_decision_matches_scalar_delayed_acceptance_exactly() -> None:
    """Batched refinement must equal scalar exact-target delayed acceptance."""
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(9071)
    row_count = 37
    station_count = 16
    current = -torch.rand(
        (row_count, station_count),
        dtype=torch.float64,
        generator=generator,
    ) * 20.0
    candidate = -torch.rand(
        (row_count, station_count),
        dtype=torch.float64,
        generator=generator,
    ) * 25.0
    non_likelihood = torch.randn(
        row_count,
        dtype=torch.float64,
        generator=generator,
    )
    log_uniform = torch.log(
        torch.rand(row_count, dtype=torch.float64, generator=generator)
    )
    log_refinement_uniform = torch.log(
        torch.rand(row_count, dtype=torch.float64, generator=generator)
    )
    support = torch.rand(
        row_count,
        dtype=torch.float64,
        generator=generator,
    ) > 0.15
    harness = _TPHTEvaluatorHarness(candidate)
    decision = harness._joint_structural_history_tree_evaluator(
        filt=object(),
        data=StructuralGeometryBatch(
            detector_positions=np.zeros((station_count, 3), dtype=np.float64),
            fe_indices=np.zeros(station_count, dtype=np.int64),
            pb_indices=np.zeros(station_count, dtype=np.int64),
            live_times=np.ones(station_count, dtype=np.float64),
            station_sequence_ids=np.arange(station_count, dtype=np.int64),
        ),
        positions_pks=torch.zeros((row_count, 1, 3), dtype=torch.float64),
        chart_ids_pk=torch.zeros((row_count, 1), dtype=torch.long),
        strengths_pk=torch.ones((row_count, 1), dtype=torch.float64),
        particle_indices=torch.arange(row_count, dtype=torch.long),
        current_station_log_likelihood_ps=current,
        base_target_log_likelihood_p=torch.sum(current, dim=1),
        log_non_likelihood_ratio_p=non_likelihood,
        log_uniform_p=log_uniform,
        log_refinement_uniform_p=log_refinement_uniform,
        support_p=support,
        target_beta=1.0,
        tempering_start_row=station_count - 1,
        move_family="position_strength",
    )
    newest_delta = candidate[:, -1] - current[:, -1]
    old_delta = torch.sum(candidate[:, :-1] - current[:, :-1], dim=1)
    expected = (
        support
        & (log_uniform < newest_delta + non_likelihood)
        & (log_refinement_uniform < old_delta)
    )
    assert torch.equal(decision.accepted, expected)
    assert not bool(torch.any(decision.accepted & ~decision.likelihood_exact))


def test_tpht_two_factor_acceptance_obeys_detailed_balance() -> None:
    """Latest and older exact factors must preserve the full target ratio."""
    log_target_x = -12.7
    latest_delta = 1.4
    older_delta = -0.8
    prior_delta = 0.3
    log_target_y = log_target_x + latest_delta + older_delta + prior_delta

    forward = min(1.0, np.exp(latest_delta + prior_delta)) * min(
        1.0,
        np.exp(older_delta),
    )
    reverse = min(1.0, np.exp(-latest_delta - prior_delta)) * min(
        1.0,
        np.exp(-older_delta),
    )

    assert np.exp(log_target_x) * forward == pytest.approx(
        np.exp(log_target_y) * reverse,
        rel=1.0e-14,
        abs=0.0,
    )


@pytest.mark.parametrize("station_count", (0, -1, True, 1.5))
def test_tpht_rejects_invalid_station_counts(station_count: object) -> None:
    """Malformed history sizes must fail instead of selecting another path."""
    with pytest.raises((TypeError, ValueError)):
        build_tpht_history_blocks(station_count)  # type: ignore[arg-type]
