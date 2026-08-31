"""Tests for the one-stage exact Metropolis-Hastings decision helper."""

from __future__ import annotations

import pytest

from pf.exact_mh import run_exact_mh_acceptance_torch


def test_exact_mh_matches_one_uniform_reference_and_consumes_one_draw() -> None:
    """The batched helper must implement the ordinary one-uniform MH rule."""
    torch = pytest.importorskip("torch")
    current = torch.tensor([-3.0, -2.0, -1.0], dtype=torch.float64)
    proposed = torch.tensor([-2.5, -3.5, -0.8], dtype=torch.float64)
    non_likelihood = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64)
    support = torch.tensor([True, True, False])
    station = torch.tensor(
        [[-0.7, -1.8], [-1.2, -2.3], [-0.4, -0.4]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(20260901)
    reference = torch.Generator().manual_seed(20260901)
    expected_log_uniform = torch.log(
        torch.rand(3, dtype=torch.float64, generator=reference)
    )
    expected = support & (
        expected_log_uniform < proposed - current + non_likelihood
    )

    decision = run_exact_mh_acceptance_torch(
        current_target_log_likelihood=current,
        proposed_target_log_likelihood=proposed,
        proposed_station_log_likelihood=station,
        log_non_likelihood_ratio=non_likelihood,
        support=support,
        generator=generator,
    )

    torch.testing.assert_close(decision.accepted, expected)
    torch.testing.assert_close(
        decision.diagnostic_delta_log_likelihood,
        proposed - current,
    )
    assert decision.diagnostic_log_acceptance_ratio[-1].item() == float("-inf")
    torch.testing.assert_close(
        torch.rand(4, dtype=torch.float64, generator=generator),
        torch.rand(4, dtype=torch.float64, generator=reference),
    )


def test_exact_mh_rejects_nonfinite_supported_kernel_ratio() -> None:
    """A malformed feasible proposal must fail closed before random acceptance."""
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="non-finite"):
        run_exact_mh_acceptance_torch(
            current_target_log_likelihood=torch.zeros(1, dtype=torch.float64),
            proposed_target_log_likelihood=torch.zeros(1, dtype=torch.float64),
            proposed_station_log_likelihood=torch.zeros(
                (1, 1), dtype=torch.float64
            ),
            log_non_likelihood_ratio=torch.tensor(
                [float("nan")], dtype=torch.float64
            ),
            support=torch.ones(1, dtype=torch.bool),
            generator=torch.Generator().manual_seed(7),
        )


def test_exact_mh_rejects_invalid_station_target() -> None:
    """Per-station values must remain finite and row aligned for atomic commit."""
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="per-station"):
        run_exact_mh_acceptance_torch(
            current_target_log_likelihood=torch.zeros(2, dtype=torch.float64),
            proposed_target_log_likelihood=torch.zeros(2, dtype=torch.float64),
            proposed_station_log_likelihood=torch.tensor(
                [[0.0], [float("nan")]], dtype=torch.float64
            ),
            log_non_likelihood_ratio=torch.zeros(2, dtype=torch.float64),
            support=torch.ones(2, dtype=torch.bool),
            generator=torch.Generator().manual_seed(9),
        )
