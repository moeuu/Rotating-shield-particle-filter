"""One-stage exact Metropolis-Hastings decisions for CUDA RJ proposals."""

from __future__ import annotations

from dataclasses import dataclass

from pf.particle_filter_math import extended_log_target_ratio_torch


@dataclass(frozen=True)
class ExactMHDecision:
    """Store one batched, one-stage exact Metropolis-Hastings decision."""

    accepted: object
    proposed_target_log_likelihood: object
    proposed_station_log_likelihood: object | None
    diagnostic_delta_log_likelihood: object
    diagnostic_log_acceptance_ratio: object


def run_exact_mh_acceptance_torch(
    *,
    current_target_log_likelihood: object,
    proposed_target_log_likelihood: object,
    proposed_station_log_likelihood: object | None,
    log_non_likelihood_ratio: object,
    support: object,
    generator: object,
) -> ExactMHDecision:
    """Evaluate one exact full-target MH ratio with one CUDA uniform draw."""
    import torch

    tensors = (
        current_target_log_likelihood,
        proposed_target_log_likelihood,
        log_non_likelihood_ratio,
        support,
    )
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("Exact MH inputs must be Torch tensors.")
    current = current_target_log_likelihood.reshape(-1)
    proposed = proposed_target_log_likelihood.reshape(-1)
    non_likelihood = log_non_likelihood_ratio.reshape(-1)
    feasible = support.to(dtype=torch.bool).reshape(-1)
    row_count = int(current.numel())
    if row_count <= 0 or any(
        tuple(value.shape) != (row_count,)
        for value in (proposed, non_likelihood, feasible)
    ):
        raise ValueError("Exact MH rows are not aligned.")
    if any(value.device != current.device for value in tensors):
        raise ValueError("Exact MH inputs changed device.")
    if (
        current.dtype != torch.float64
        or proposed.dtype != current.dtype
        or non_likelihood.dtype != current.dtype
    ):
        raise TypeError("Exact MH log targets must use float64.")
    if bool(torch.any(feasible & ~torch.isfinite(non_likelihood)).item()):
        raise RuntimeError(
            "A support-feasible exact MH proposal has a non-finite "
            "prior/proposal/Jacobian ratio."
        )
    station = proposed_station_log_likelihood
    if station is not None:
        if (
            not torch.is_tensor(station)
            or station.device != current.device
            or station.dtype != current.dtype
            or station.ndim != 2
            or int(station.shape[0]) != row_count
            or bool(torch.any(~torch.isfinite(station)).item())
        ):
            raise RuntimeError(
                "Exact MH per-station proposal targets are invalid."
            )
    delta = extended_log_target_ratio_torch(proposed, current)
    safe_non_likelihood = torch.where(
        feasible,
        non_likelihood,
        torch.zeros_like(non_likelihood),
    )
    log_ratio = delta + safe_non_likelihood
    if bool(torch.any(feasible & torch.isnan(log_ratio)).item()):
        raise RuntimeError("A support-feasible exact MH ratio is NaN.")
    log_uniform = torch.log(
        torch.rand(
            (row_count,),
            device=current.device,
            dtype=current.dtype,
            generator=generator,
        )
    )
    accepted = feasible & (log_uniform < log_ratio)
    diagnostic_ratio = torch.where(
        feasible,
        log_ratio,
        torch.full_like(log_ratio, float("-inf")),
    )
    return ExactMHDecision(
        accepted=accepted,
        proposed_target_log_likelihood=proposed,
        proposed_station_log_likelihood=station,
        diagnostic_delta_log_likelihood=delta,
        diagnostic_log_acceptance_ratio=diagnostic_ratio,
    )


__all__ = ["ExactMHDecision", "run_exact_mh_acceptance_torch"]
