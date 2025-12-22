"""Handle log-domain weight updates and stabilisation for Poisson observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def log_weight_update_poisson(
    log_w_prev: NDArray[np.float64],
    z_obs: float,
    lambda_exp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Log-weight update for Poisson observations (Sec. 3.4.3)."""
    # log-likelihood up to additive constant
    ll = z_obs * np.log(lambda_exp + 1e-12) - lambda_exp
    log_w = log_w_prev + ll
    # normalize in log-domain
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w /= w.sum()
    return np.log(w + 1e-20)


def effective_sample_size(log_w: NDArray[np.float64]) -> float:
    """Return the effective sample size."""
    w = np.exp(log_w)
    return float(1.0 / np.sum(w**2))
