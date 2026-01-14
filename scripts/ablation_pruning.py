"""Ablation script comparing pruning methods for shared-isotope sources."""

from __future__ import annotations

import numpy as np

from pf.mixing import prune_spurious_sources


def _simulate_trial(rng: np.random.Generator, num_meas: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate shared-isotope counts and per-source contributions."""
    base = rng.uniform(0.5, 1.5, size=num_meas)
    strength_true = np.array([8.0, 6.0], dtype=float)
    strength_spur = np.array([0.6], dtype=float)
    lambda_true = base[:, None] * strength_true[None, :]
    lambda_spur = rng.uniform(0.1, 0.4, size=(num_meas, 1)) * strength_spur
    lambda_m = np.hstack([lambda_true, lambda_spur])
    lambda_total = np.sum(lambda_m, axis=1)
    z_k = rng.poisson(lambda_total)
    return z_k.astype(float), lambda_m.astype(float)


def _evaluate_method(
    method: str,
    params: dict[str, float],
    trials: int,
    num_meas: int,
    seed: int,
) -> tuple[float, float]:
    """Return (miss_rate, false_positive_rate) for a pruning method."""
    rng = np.random.default_rng(seed)
    missed = 0
    false_pos = 0
    total_true = 0
    total_spur = 0
    for _ in range(trials):
        z_k, lambda_m = _simulate_trial(rng, num_meas)
        num_sources = lambda_m.shape[1]
        positions = np.zeros((num_sources, 3), dtype=float)
        strengths = np.ones(num_sources, dtype=float)

        def forward_model(_: np.ndarray, __: np.ndarray) -> np.ndarray:
            return lambda_m

        keep = prune_spurious_sources(
            z_k=z_k,
            live_times=np.ones(num_meas, dtype=float),
            positions=positions,
            strengths=strengths,
            background=0.0,
            forward_model=forward_model,
            method=method,
            params=params,
        )
        missed += int(np.sum(~keep[:2]))
        false_pos += int(np.sum(keep[2:]))
        total_true += 2
        total_spur += num_sources - 2
    miss_rate = missed / max(total_true, 1)
    false_rate = false_pos / max(total_spur, 1)
    return miss_rate, false_rate


def main() -> None:
    """Run the ablation and print summary metrics."""
    trials = 200
    num_meas = 5
    methods = {
        "deltaLL": {"deltaLL_min": 0.0, "penalty_d": 0.0},
        "bestcase": {"alpha": 0.7, "lambda_min": 1e-6, "lrt_threshold": 0.0},
        "legacy": {"tau_mix": 0.8},
    }
    for name, params in methods.items():
        miss, false = _evaluate_method(name, params, trials, num_meas, seed=123)
        print(f"{name}: miss_rate={miss:.3f} false_pos_rate={false:.3f}")


if __name__ == "__main__":
    main()
