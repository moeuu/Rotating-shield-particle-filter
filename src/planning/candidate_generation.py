"""Generate candidate measurement poses for online exploration."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def _resolve_bounds(
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (lo, hi) bounds for candidate generation."""
    if bounds_xyz is None:
        raise ValueError(
            "Candidate generation requires explicit environment bounds."
        )
    lo = np.asarray(bounds_xyz[0], dtype=float)
    hi = np.asarray(bounds_xyz[1], dtype=float)
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or np.any(~np.isfinite(lo))
        or np.any(~np.isfinite(hi))
        or np.any(hi < lo)
    ):
        raise ValueError(
            "bounds_xyz must contain finite ordered (3,) arrays."
        )
    return lo, hi


def _resolve_free_space_batch_checker(
    map_api: object | None,
) -> Callable[[NDArray[np.float64]], NDArray[np.bool_]]:
    """Return a standard batched free-space checker when the map provides one."""
    if map_api is None:
        return lambda points: np.ones(np.asarray(points).shape[0], dtype=bool)
    for attr in ("is_free_batch", "is_free_space_batch"):
        function = getattr(map_api, attr, None)
        if callable(function):
            return function
    raise TypeError(
        "Production candidate generation requires is_free_batch or "
        "is_free_space_batch."
    )


def _map_free_cell_centers(
    map_api: object | None,
    *,
    z_value: float,
) -> NDArray[np.float64]:
    """Return deterministic free-cell centers without scalar cell callbacks."""
    if map_api is None:
        return np.zeros((0, 3), dtype=float)
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is None:
        return np.zeros((0, 3), dtype=float)
    raw_cells = np.asarray(tuple(traversable_cells))
    if raw_cells.size == 0:
        return np.zeros((0, 3), dtype=float)
    if (
        raw_cells.ndim != 2
        or raw_cells.shape[1] != 2
        or not np.issubdtype(raw_cells.dtype, np.integer)
        or np.any(raw_cells < 0)
    ):
        raise ValueError(
            "traversable_cells must be a finite N x 2 integer array."
        )
    cells = raw_cells.astype(np.int64, copy=False)
    center_batch = getattr(map_api, "cell_centers_batch", None)
    if callable(center_batch):
        xy_centers = np.asarray(center_batch(cells), dtype=np.float64)
        if (
            xy_centers.shape != (cells.shape[0], 2)
            or np.any(~np.isfinite(xy_centers))
        ):
            raise ValueError(
                "cell_centers_batch must return one finite xy center per cell."
            )
    else:
        if not hasattr(map_api, "origin") or not hasattr(map_api, "cell_size"):
            raise TypeError(
                "A planning grid without cell_centers_batch must define "
                "origin and cell_size for vectorized center construction."
            )
        origin = np.asarray(map_api.origin, dtype=np.float64)
        cell_size = getattr(map_api, "cell_size")
        if (
            origin.shape != (2,)
            or np.any(~np.isfinite(origin))
            or isinstance(cell_size, (bool, np.bool_))
            or not isinstance(
                cell_size,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(cell_size))
            or float(cell_size) <= 0.0
        ):
            raise ValueError("Map cell-center geometry is invalid.")
        xy_centers = (
            origin[None, :]
            + (cells.astype(np.float64) + 0.5) * float(cell_size)
        )
    if (
        isinstance(z_value, (bool, np.bool_))
        or not isinstance(z_value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(z_value))
    ):
        raise ValueError("Map cell-center height must be finite.")
    return np.column_stack(
        (
            xy_centers,
            np.full(cells.shape[0], float(z_value), dtype=np.float64),
        )
    )


def _filter_candidates(
    candidates: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    min_dist_from_visited: float,
    *,
    is_free_batch_fn: Callable[
        [NDArray[np.float64]],
        NDArray[np.bool_],
    ],
) -> NDArray[np.float64]:
    """Filter candidates with batched free-space and 3-D separation checks."""
    if candidates.size == 0:
        return candidates
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 3
        or np.any(~np.isfinite(candidates))
    ):
        raise ValueError("candidates must be finite and shaped (N, 3).")
    minimum_distance = float(min_dist_from_visited)
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError(
            "min_dist_from_visited must be finite and nonnegative."
        )
    mask = np.ones(candidates.shape[0], dtype=bool)
    if visited_poses_xyz is not None and visited_poses_xyz.size:
        visited = np.asarray(visited_poses_xyz, dtype=float).reshape(-1, 3)
        if np.any(~np.isfinite(visited)):
            raise ValueError("visited_poses_xyz must contain finite values.")
        distances = np.linalg.norm(
            candidates[:, None, :] - visited[None, :, :],
            axis=2,
        )
        separated = np.all(
            distances >= minimum_distance,
            axis=1,
        )
        mask &= separated
    if not callable(is_free_batch_fn):
        raise TypeError(
            "Candidate filtering requires a batched free-space checker."
        )
    free_mask = np.asarray(is_free_batch_fn(candidates))
    if free_mask.dtype != np.bool_ or free_mask.shape != (
        candidates.shape[0],
    ):
        raise ValueError(
            "Batched free-space checker must return one exact boolean "
            "per candidate."
        )
    mask &= free_mask
    return candidates[mask]


def _filter_motion_reachable_candidates(
    candidates_xyz: NDArray[np.float64],
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    enabled: bool,
) -> NDArray[np.float64]:
    """Filter one candidate batch with the map's native reachability API."""
    candidates = np.asarray(candidates_xyz, dtype=float).reshape(-1, 3)
    if not enabled or candidates.shape[0] == 0:
        return candidates
    checker = getattr(map_api, "is_motion_reachable_batch", None)
    if not callable(checker):
        raise TypeError(
            "Motion-reachable candidate generation requires "
            "is_motion_reachable_batch."
        )
    reachable = np.asarray(checker(current_pose_xyz, candidates))
    if reachable.dtype != np.bool_ or reachable.shape != (
        candidates.shape[0],
    ):
        raise ValueError(
            "is_motion_reachable_batch must return one exact boolean per "
            "candidate."
        )
    return candidates[reachable]


def resolve_detector_height_actions(
    detector_heights_m: Sequence[float] | None,
    *,
    default_height_m: float,
    bounds_z: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Return sorted unique detector-height actions after validation."""
    if detector_heights_m is None:
        values = np.asarray([default_height_m], dtype=float)
    else:
        values = np.asarray(tuple(detector_heights_m), dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("detector_heights_m must contain at least one height.")
    if not np.all(np.isfinite(values)):
        raise ValueError("detector_heights_m must contain only finite values.")
    values = np.unique(values)
    if bounds_z is not None:
        lower, upper = (float(bounds_z[0]), float(bounds_z[1]))
        if upper < lower:
            raise ValueError("bounds_z upper bound must be >= lower bound.")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("detector_heights_m must lie within bounds_z.")
    return values.astype(float)


def expand_candidate_height_actions(
    candidates_xyz: NDArray[np.float64],
    detector_heights_m: Sequence[float],
) -> NDArray[np.float64]:
    """Expand candidate xy stations across discrete detector heights in batch."""
    candidates = np.asarray(candidates_xyz, dtype=float)
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidates_xyz must be shape (N, 3).")
    heights = resolve_detector_height_actions(
        detector_heights_m,
        default_height_m=float(candidates[0, 2]),
    )
    expanded = np.repeat(candidates[:, None, :], heights.size, axis=1)
    expanded[:, :, 2] = heights[None, :]
    return expanded.reshape(-1, 3)


def _stable_unique_candidates(
    candidates_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Remove duplicate candidate rows while preserving their first-seen order."""
    candidates = np.asarray(candidates_xyz, dtype=float)
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=float)
    rounded = np.round(candidates, decimals=12)
    _, first_indices = np.unique(rounded, axis=0, return_index=True)
    return candidates[np.sort(first_indices)]


def _sample_uniform(
    rng: np.random.Generator,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_samples: int,
) -> NDArray[np.float64]:
    """Sample points uniformly within bounds."""
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
    ):
        raise TypeError("n_samples must be an integer.")
    if int(n_samples) < 0:
        raise ValueError("n_samples must be nonnegative.")
    if int(n_samples) == 0:
        return np.zeros((0, 3), dtype=float)
    span = hi - lo
    return lo + rng.random((n_samples, 3)) * span


def _sample_sobol(
    rng: np.random.Generator,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_samples: int,
) -> NDArray[np.float64]:
    """
    Sample points using the required scrambled Sobol implementation.

    Degenerate dimensions (lo == hi) are kept fixed at their bound value.
    """
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
    ):
        raise TypeError("n_samples must be an integer.")
    if int(n_samples) < 0:
        raise ValueError("n_samples must be nonnegative.")
    if int(n_samples) == 0:
        return np.zeros((0, 3), dtype=float)
    try:
        from scipy.stats import qmc
    except ImportError as error:
        raise RuntimeError(
            "Sobol candidate generation requires scipy.stats.qmc."
        ) from error
    active_dims = hi > lo
    if not np.any(active_dims):
        return np.repeat(lo[None, :], n_samples, axis=0)
    seed = int(rng.integers(0, 2**32 - 1))
    sampler = qmc.Sobol(d=int(np.sum(active_dims)), scramble=True, seed=seed)
    m = int(np.ceil(np.log2(int(n_samples))))
    sample = sampler.random_base2(m)
    sample = sample[:n_samples]
    scaled = qmc.scale(sample, lo[active_dims], hi[active_dims])
    out = np.repeat(lo[None, :], n_samples, axis=0)
    out[:, active_dims] = scaled
    return out


def sample_low_discrepancy_heights(
    rng: np.random.Generator,
    bounds_z: tuple[float, float],
    n_samples: int,
) -> NDArray[np.float64]:
    """Sample one-dimensional detector heights with a scrambled Sobol sequence."""
    lower = float(bounds_z[0])
    upper = float(bounds_z[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bounds_z must contain finite values.")
    if upper < lower:
        raise ValueError("bounds_z upper bound must be >= lower bound.")
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
    ):
        raise TypeError("n_samples must be an integer.")
    if int(n_samples) < 0:
        raise ValueError("n_samples must be nonnegative.")
    if int(n_samples) == 0:
        return np.zeros(0, dtype=float)
    lo = np.array([0.0, 0.0, lower], dtype=float)
    hi = np.array([0.0, 0.0, upper], dtype=float)
    return _sample_sobol(rng, lo, hi, int(n_samples))[:, 2]


def _generate_ring_candidates(
    current_pose_xyz: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_candidates: int,
    min_dist_from_visited: float,
) -> NDArray[np.float64]:
    """Generate candidates on concentric rings around the current pose."""
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, (int, np.integer))
    ):
        raise TypeError("n_candidates must be an integer.")
    if int(n_candidates) < 0:
        raise ValueError("n_candidates must be nonnegative.")
    if int(n_candidates) == 0:
        return np.zeros((0, 3), dtype=float)
    minimum_distance = float(min_dist_from_visited)
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError(
            "min_dist_from_visited must be finite and nonnegative."
        )
    max_dx = min(current_pose_xyz[0] - lo[0], hi[0] - current_pose_xyz[0])
    max_dy = min(current_pose_xyz[1] - lo[1], hi[1] - current_pose_xyz[1])
    max_radius = max(0.0, min(max_dx, max_dy))
    min_radius = max(0.1, minimum_distance)
    if max_radius < min_radius:
        max_radius = min_radius
    num_rings = max(1, int(np.sqrt(n_candidates)))
    num_angles = max(4, int(np.ceil(n_candidates / num_rings)))
    radii = np.linspace(min_radius, max_radius, num=num_rings)
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_angles, endpoint=False)
    radius_grid, angle_grid = np.meshgrid(radii, angles, indexing="ij")
    return np.column_stack(
        (
            current_pose_xyz[0]
            + radius_grid.ravel() * np.cos(angle_grid.ravel()),
            current_pose_xyz[1]
            + radius_grid.ravel() * np.sin(angle_grid.ravel()),
            np.full(radius_grid.size, current_pose_xyz[2], dtype=float),
        )
    )[: int(n_candidates)]


def generate_candidate_poses(
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None = None,
    n_candidates: int = 1024,
    strategy: str = "free_space_sobol",
    min_dist_from_visited: float = 1.0,
    visited_poses_xyz: NDArray[np.float64] | None = None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    rng: np.random.Generator | None = None,
    detector_heights_m: Sequence[float] | None = None,
    require_motion_reachable: bool = False,
) -> NDArray[np.float64]:
    """Return (L, 3) candidate poses in free space for the given strategy.

    When ``detector_heights_m`` is supplied, xy stations are sampled once and
    expanded over the discrete height actions. Continuous z values are sampled
    globally when discrete detector actions are not configured. Every generated
    batch uses the same 3-D separation and reachability contract.
    """
    if rng is None:
        raise ValueError("Candidate generation requires an explicit RNG.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if (
        isinstance(n_candidates, bool)
        or not isinstance(n_candidates, (int, np.integer))
        or int(n_candidates) <= 0
    ):
        raise ValueError("n_candidates must be a positive integer.")
    requested_count = int(n_candidates)
    minimum_distance = float(min_dist_from_visited)
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError(
            "min_dist_from_visited must be finite and nonnegative."
        )
    if not isinstance(require_motion_reachable, bool):
        raise TypeError("require_motion_reachable must be a boolean.")
    current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
    if current_pose_xyz.shape != (3,) or np.any(~np.isfinite(current_pose_xyz)):
        raise ValueError("current_pose_xyz must be finite and shape (3,).")
    visited = None
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if (
            visited.ndim != 2
            or visited.shape[1] != 3
            or np.any(~np.isfinite(visited))
        ):
            raise ValueError(
                "visited_poses_xyz must be finite and shape (N, 3)."
            )

    lo, hi = _resolve_bounds(bounds_xyz)
    is_free_batch_fn = _resolve_free_space_batch_checker(map_api)
    height_actions: NDArray[np.float64] | None = None
    sample_lo = lo.copy()
    sample_hi = hi.copy()
    base_candidate_count = requested_count
    if detector_heights_m is not None:
        height_actions = resolve_detector_height_actions(
            detector_heights_m,
            default_height_m=float(current_pose_xyz[2]),
            bounds_z=(float(lo[2]), float(hi[2])),
        )
        sample_lo[2] = float(current_pose_xyz[2])
        sample_hi[2] = float(current_pose_xyz[2])
        base_candidate_count = max(
            int(np.ceil(requested_count / height_actions.size)),
            1,
        )

    if strategy == "ring":
        raw = _generate_ring_candidates(
            current_pose_xyz=current_pose_xyz,
            lo=sample_lo,
            hi=sample_hi,
            n_candidates=base_candidate_count,
            min_dist_from_visited=minimum_distance,
        )
    elif strategy == "free_space_sobol":
        raw = _sample_sobol(
            rng,
            sample_lo,
            sample_hi,
            max(base_candidate_count * 3, base_candidate_count),
        )
    elif strategy == "gaussian":
        sample_count = max(base_candidate_count * 3, base_candidate_count)
        raw = rng.normal(
            loc=current_pose_xyz,
            scale=0.75,
            size=(sample_count, 3),
        )
        raw = np.clip(raw, sample_lo, sample_hi)
    else:
        raise ValueError(f"Unknown candidate generation strategy: {strategy}")

    if height_actions is not None:
        raw = expand_candidate_height_actions(raw, height_actions)

    filtered = _filter_candidates(
        raw,
        visited,
        minimum_distance,
        is_free_batch_fn=is_free_batch_fn,
    )
    filtered = _filter_motion_reachable_candidates(
        filtered,
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        enabled=require_motion_reachable,
    )
    if filtered.shape[0] < requested_count:
        map_centers = _map_free_cell_centers(
            map_api,
            z_value=float(current_pose_xyz[2]),
        )
        if map_centers.size:
            if height_actions is not None:
                map_centers = expand_candidate_height_actions(
                    map_centers,
                    height_actions,
                )
            if visited is not None and visited.size:
                distances = np.linalg.norm(
                    map_centers[:, None, :] - visited[None, :, :],
                    axis=2,
                )
                order = np.argsort(np.min(distances, axis=1))[::-1]
                map_centers = map_centers[order]
            map_centers = _filter_candidates(
                map_centers,
                visited,
                minimum_distance,
                is_free_batch_fn=is_free_batch_fn,
            )
            map_centers = _filter_motion_reachable_candidates(
                map_centers,
                current_pose_xyz=current_pose_xyz,
                map_api=map_api,
                enabled=require_motion_reachable,
            )
            if map_centers.size:
                filtered = np.vstack([filtered, map_centers])
    if filtered.shape[0] < requested_count:
        extra = _sample_uniform(
            rng,
            sample_lo,
            sample_hi,
            max(base_candidate_count, 1),
        )
        if height_actions is not None:
            extra = expand_candidate_height_actions(extra, height_actions)
        extra = _filter_candidates(
            extra,
            visited,
            minimum_distance,
            is_free_batch_fn=is_free_batch_fn,
        )
        extra = _filter_motion_reachable_candidates(
            extra,
            current_pose_xyz=current_pose_xyz,
            map_api=map_api,
            enabled=require_motion_reachable,
        )
        if extra.size:
            filtered = np.vstack([filtered, extra])
    return _stable_unique_candidates(filtered)[:requested_count]


def generate_planning_candidates(
    *,
    current_pose_xyz: NDArray[np.float64],
    map_api: object | None,
    n_candidates: int,
    min_dist_from_visited: float,
    visited_poses_xyz: NDArray[np.float64] | None,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Generate one globally sampled reachable 3-D PF planning action pool."""
    if rng is None:
        raise ValueError("Planning candidate generation requires an explicit RNG.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    bounds_lo = np.asarray(bounds_xyz[0], dtype=np.float64)
    bounds_hi = np.asarray(bounds_xyz[1], dtype=np.float64)
    if (
        bounds_lo.shape != (3,)
        or bounds_hi.shape != (3,)
        or np.any(~np.isfinite(bounds_lo))
        or np.any(~np.isfinite(bounds_hi))
        or np.any(bounds_hi < bounds_lo)
    ):
        raise ValueError("bounds_xyz must contain finite ordered 3-D bounds.")
    candidates = generate_candidate_poses(
        current_pose_xyz=current_pose_xyz,
        map_api=map_api,
        n_candidates=n_candidates,
        strategy="free_space_sobol",
        min_dist_from_visited=min_dist_from_visited,
        visited_poses_xyz=visited_poses_xyz,
        bounds_xyz=(bounds_lo, bounds_hi),
        detector_heights_m=detector_heights_m,
        require_motion_reachable=True,
        rng=rng,
    )
    candidates = np.asarray(candidates, dtype=np.float64)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 3
        or np.any(~np.isfinite(candidates))
    ):
        raise RuntimeError(
            "Global candidate generation returned an invalid 3-D action pool."
        )
    if candidates.shape[0] == 0:
        raise RuntimeError(
            "No globally sampled candidate satisfies bounds, free-space, "
            "reachability, and physical separation."
        )
    return candidates, {
        "contract": "global_reachable_3d_sobol_pool_v1",
        "candidate_count": int(candidates.shape[0]),
        "requested_candidate_count": int(n_candidates),
        "minimum_3d_separation_m": float(min_dist_from_visited),
        "physical_separation_relaxed": False,
        "horizontal_quality_gate": False,
        "bounds_lo_xyz_m": [float(value) for value in bounds_lo],
        "bounds_hi_xyz_m": [float(value) for value in bounds_hi],
        "detector_heights_m": (
            None
            if detector_heights_m is None
            else [float(value) for value in detector_heights_m]
        ),
    }


def planning_candidate_checkpoint_parameters(
    *,
    pose_candidates: int,
    pose_min_dist: float,
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]],
    detector_heights_m: Sequence[float] | None,
) -> dict[str, object]:
    """Return PF planning-pool parameters protected by checkpoint identity."""
    bounds_lo = np.asarray(bounds_xyz[0], dtype=float).reshape(3)
    bounds_hi = np.asarray(bounds_xyz[1], dtype=float).reshape(3)
    return {
        "pose_candidates": int(pose_candidates),
        "pose_min_dist_m": float(pose_min_dist),
        "bounds_lo_xyz_m": [float(value) for value in bounds_lo],
        "bounds_hi_xyz_m": [float(value) for value in bounds_hi],
        "detector_heights_m": (
            None
            if detector_heights_m is None
            else [float(value) for value in detector_heights_m]
        ),
        "candidate_pool_contract": (
            "global_reachable_3d_sobol_with_physical_separation_v1"
        ),
    }
