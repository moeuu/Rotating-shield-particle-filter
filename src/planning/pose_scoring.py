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
    program_length: int,
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
    if (
        isinstance(program_length, bool)
        or not isinstance(program_length, (int, np.integer))
        or int(program_length) <= 0
    ):
        raise ValueError("program_length must be a positive integer.")

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
    if motion_times_p is None:
        motion_times = finite_paths / float(config.robot_speed_m_s)
    else:
        motion_times = np.asarray(motion_times_p, dtype=np.float64).reshape(-1)
        if (
            motion_times.shape != gains.shape
            or np.any(~np.isfinite(motion_times))
            or np.any(motion_times < 0.0)
        ):
            raise ValueError("motion_times_p must be finite and align with poses.")

    if motion_time_components_p is None:
        motion_penalty = float(config.lambda_time) * motion_times
    else:
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
        if not np.allclose(
            np.sum(np.vstack(components), axis=0),
            motion_times,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Motion-time components must sum to motion times.")
        weights = (
            float(config.lambda_time)
            if config.lambda_horizontal_time is None
            else float(config.lambda_horizontal_time),
            float(config.lambda_time)
            if config.lambda_mast_vertical_time is None
            else float(config.lambda_mast_vertical_time),
            float(config.lambda_time)
            if config.lambda_settling_time is None
            else float(config.lambda_settling_time),
        )
        motion_penalty = np.einsum(
            "cp,c->p",
            np.vstack(components),
            np.asarray(weights, dtype=np.float64),
            optimize=True,
        )

    measurement_penalty = float(config.lambda_time) * int(program_length) * (
        float(config.rotation_overhead_s) + float(config.live_time_s)
    )
    scores = (
        static
        + float(config.lambda_eig) * gains
        - float(distance_weight) * finite_paths
        - motion_penalty
        - measurement_penalty
    )
    scores = np.where(finite, scores, -np.inf)
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise RuntimeError("Vectorized pose score composition produced invalid values.")
    return np.asarray(scores, dtype=np.float64), float(distance_weight)


__all__ = ["compose_pose_scores"]
