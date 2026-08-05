"""Compare five- and nine-point conditional strength proposal grids."""

from __future__ import annotations

import json
import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
from scipy.stats import truncnorm


def _candidate_center(grid_size: int) -> NDArray[np.float64]:
    """Return the selected center for a smooth two-source exact target."""
    probabilities = np.linspace(0.005, 0.995, grid_size, dtype=np.float64)
    candidates = np.repeat((1.0 + 2.0 * probabilities)[:, None], 2, axis=1)
    target = -0.5 * np.sum(np.square((candidates - 1.82) / 0.25), axis=1)
    return candidates[int(np.argmax(target))]


def _proposal_log_density(
    values: NDArray[np.float64],
    *,
    center: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """Return the exact prior/data block-mixture proposal log density."""
    prior = np.full(values.shape[0], -2.0 * math.log(2.0), dtype=np.float64)
    normalization = ndtr((3.0 - center) / sigma) - ndtr((1.0 - center) / sigma)
    data = np.sum(
        -0.5 * np.square((values - center) / sigma)
        - np.log(sigma * np.sqrt(2.0 * np.pi) * normalization),
        axis=1,
    )
    return np.logaddexp(math.log(0.5) + prior, math.log(0.5) + data)


def _proposal_samples(
    sample_count: int,
    *,
    center: NDArray[np.float64],
    sigma: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw the same one-choice-per-block prior/data proposal mixture."""
    use_prior = rng.random(sample_count) < 0.5
    samples = np.empty((sample_count, 2), dtype=np.float64)
    samples[use_prior] = rng.uniform(1.0, 3.0, size=(np.sum(use_prior), 2))
    data_count = int(np.count_nonzero(~use_prior))
    if data_count:
        lower = (1.0 - center) / sigma
        upper = (3.0 - center) / sigma
        samples[~use_prior] = truncnorm.rvs(
            lower,
            upper,
            loc=center,
            scale=sigma,
            size=(data_count, 2),
            random_state=rng,
        )
    return samples


def _comparison_row(
    grid_size: int,
    *,
    stationary_states: NDArray[np.float64],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Return stationary independence-MH acceptance and ESJD diagnostics."""
    center = _candidate_center(grid_size)
    sigma = 0.3
    proposed = _proposal_samples(
        stationary_states.shape[0],
        center=center,
        sigma=sigma,
        rng=rng,
    )

    def _log_target(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the unchanged smooth exact target up to normalization."""
        return -0.5 * np.sum(np.square((values - 1.82) / 0.25), axis=1)

    log_acceptance_ratio = (
        _log_target(proposed)
        + _proposal_log_density(stationary_states, center=center, sigma=sigma)
        - _log_target(stationary_states)
        - _proposal_log_density(proposed, center=center, sigma=sigma)
    )
    acceptance = np.exp(np.minimum(log_acceptance_ratio, 0.0))
    squared_jump = np.sum(np.square(proposed - stationary_states), axis=1)
    return {
        "grid_size": int(grid_size),
        "exact_target_changed": False,
        "selected_center": center.tolist(),
        "mean_stationary_mh_acceptance": float(np.mean(acceptance)),
        "mean_stationary_esjd": float(np.mean(acceptance * squared_jump)),
    }


def main() -> int:
    """Run the deterministic generic mixing comparison and print JSON."""
    rng = np.random.default_rng(20260805)
    sample_count = 100_000
    lower = (1.0 - 1.82) / 0.25
    upper = (3.0 - 1.82) / 0.25
    stationary = truncnorm.rvs(
        lower,
        upper,
        loc=1.82,
        scale=0.25,
        size=(sample_count, 2),
        random_state=rng,
    )
    rows = [
        _comparison_row(grid_size, stationary_states=stationary, rng=rng)
        for grid_size in (5, 9)
    ]
    five, nine = rows
    payload = {
        "schema_version": 1,
        "benchmark": ("two_source_bounded_strength_stationary_independence_mh_v1"),
        "sample_count": sample_count,
        "rows": rows,
        "five_to_nine_acceptance_ratio": (
            five["mean_stationary_mh_acceptance"]
            / nine["mean_stationary_mh_acceptance"]
        ),
        "five_to_nine_esjd_ratio": (
            five["mean_stationary_esjd"] / nine["mean_stationary_esjd"]
        ),
        "grid_evaluation_ratio": 5.0 / 9.0,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
