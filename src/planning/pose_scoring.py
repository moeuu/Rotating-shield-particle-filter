"""Vectorized DSS pose utility composition independent of shield search."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from planning.dss_types import DSSPPConfig, estimate_lambda_cost


def compose_pose_scores(
    information_gains_p: NDArray[np.float64],
    static_scores_p: NDArray[np.float64],
    path_lengths_p: NDArray[np.float64],
    *,
    config: DSSPPConfig,
    motion_times_p: NDArray[np.float64] | None = None,
    motion_time_components_p: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Return EIG-plus-spatial-minus-motion scores for every pose at once."""
    gains = np.asarray(information_gains_p, dtype=np.float64).reshape(-1)
    static = np.asarray(static_scores_p, dtype=np.float64).reshape(-1)
    paths = np.asarray(path_lengths_p, dtype=np.float64).reshape(-1)
    if gains.shape != static.shape or paths.shape != gains.shape:
        raise ValueError("Pose EIG, static utility, and path arrays must align.")
    if (
        np.any(~np.isfinite(gains))
        or np.any(gains < 0.0)
        or np.any(~np.isfinite(static))
        or np.any(np.isnan(paths))
        or np.any(paths < 0.0)
    ):
        raise ValueError("Pose score inputs contain invalid values.")
    finite = np.isfinite(paths)
    if config.lambda_distance is None:
        distance_weight = estimate_lambda_cost(
            gains[finite],
            paths[finite],
            method="range",
        )
    else:
        distance_weight = float(config.lambda_distance)

    finite_paths = np.where(finite, paths, 0.0)
    component_weights = np.asarray(
        (
            config.lambda_horizontal_time,
            config.lambda_mast_vertical_time,
            config.lambda_settling_time,
        ),
        dtype=np.float64,
    )
    if (motion_times_p is None) is not (motion_time_components_p is None):
        raise ValueError(
            "Motion totals and all three components must be supplied together."
        )
    if motion_times_p is None:
        if np.any(component_weights != 0.0):
            raise ValueError(
                "Nonzero motion weights require runtime-authored motion components."
            )
        motion_penalty = np.zeros_like(gains)
    else:
        motion_times = np.asarray(motion_times_p, dtype=np.float64).reshape(-1)
        if (
            motion_times.shape != gains.shape
            or np.any(~np.isfinite(motion_times))
            or np.any(motion_times < 0.0)
        ):
            raise ValueError("motion_times_p must be finite and align with poses.")
        assert motion_time_components_p is not None
        if len(motion_time_components_p) != 3:
            raise ValueError("Motion scoring requires exactly three components.")
        components = tuple(
            np.asarray(values, dtype=np.float64).reshape(-1)
            for values in motion_time_components_p
        )
        if any(
            values.shape != gains.shape
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            for values in components
        ):
            raise ValueError(
                "Motion-time components must be finite and align with poses."
            )
        if not np.array_equal(
            np.sum(np.vstack(components), axis=0),
            motion_times,
        ):
            raise ValueError("Motion-time components must sum to motion times.")
        motion_penalty = np.einsum(
            "cp,c->p",
            np.vstack(components),
            component_weights,
            optimize=True,
        )
    scores = (
        static
        + float(config.lambda_eig) * gains
        - float(distance_weight) * finite_paths
        - motion_penalty
    )
    scores = np.where(finite, scores, -np.inf)
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise RuntimeError("Vectorized pose score composition produced invalid values.")
    return np.asarray(scores, dtype=np.float64), float(distance_weight)


__all__ = ["compose_pose_scores"]
