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
        lo = np.array([0.0, 0.0, 0.0], dtype=float)
        hi = np.array([10.0, 10.0, 10.0], dtype=float)
    else:
        lo = np.asarray(bounds_xyz[0], dtype=float)
        hi = np.asarray(bounds_xyz[1], dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds_xyz must contain two (3,) arrays.")
    hi = np.maximum(hi, lo)
    return lo, hi


def _resolve_free_space_checker(
    map_api: object | None,
) -> Callable[[NDArray[np.float64]], bool]:
    """Return the scalar compatibility checker for a planning map."""
    if map_api is None:
        return lambda _: True
    if callable(map_api):
        return map_api
    for attr in ("is_free", "is_free_space", "is_free_cell"):
        fn = getattr(map_api, attr, None)
        if callable(fn):
            return fn
    return lambda _: True


def _resolve_free_space_batch_checker(
    map_api: object | None,
) -> Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None:
    """Return a standard batched free-space checker when the map provides one."""
    if map_api is None:
        return lambda points: np.ones(np.asarray(points).shape[0], dtype=bool)
    for attr in ("is_free_batch", "is_free_space_batch"):
        function = getattr(map_api, attr, None)
        if callable(function):
            return function
    return None


def _cell_center_from_map(
    map_api: object,
    cell: tuple[int, int],
    z_value: float,
) -> NDArray[np.float64]:
    """Return a 3D cell-center point for a map cell."""
    center_fn = getattr(map_api, "cell_center", None)
    if callable(center_fn):
        x_value, y_value = center_fn(cell)
        return np.array([float(x_value), float(y_value), float(z_value)], dtype=float)
    origin = getattr(map_api, "origin", (0.0, 0.0))
    cell_size = float(getattr(map_api, "cell_size", 1.0))
    return np.array(
        [
            float(origin[0]) + (float(cell[0]) + 0.5) * cell_size,
            float(origin[1]) + (float(cell[1]) + 0.5) * cell_size,
            float(z_value),
        ],
        dtype=float,
    )


def _map_free_cell_centers(
    map_api: object | None,
    *,
    z_value: float,
) -> NDArray[np.float64]:
    """Return deterministic free-cell centers from the planning map."""
    if map_api is None:
        return np.zeros((0, 3), dtype=float)
    grid_shape = getattr(map_api, "grid_shape", None)
    if grid_shape is None:
        return np.zeros((0, 3), dtype=float)
    traversable_cells = getattr(map_api, "traversable_cells", None)
    if traversable_cells is not None:
        cells = [tuple(cell) for cell in traversable_cells]
    else:
        is_free_cell = getattr(map_api, "is_free_cell", None)
        if not callable(is_free_cell):
            is_free_cell = getattr(map_api, "is_cell_free", None)
        if not callable(is_free_cell):
            return np.zeros((0, 3), dtype=float)
        cells = [
            (ix, iy)
            for ix in range(int(grid_shape[0]))
            for iy in range(int(grid_shape[1]))
            if bool(is_free_cell((ix, iy)))
        ]
    if not cells:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(
        [_cell_center_from_map(map_api, tuple(cell), z_value) for cell in cells]
    ).astype(float)


def _filter_candidates(
    candidates: NDArray[np.float64],
    visited_poses_xyz: NDArray[np.float64] | None,
    min_dist_from_visited: float,
    is_free_fn: Callable[[NDArray[np.float64]], bool],
    *,
    is_free_batch_fn: (
        Callable[[NDArray[np.float64]], NDArray[np.bool_]] | None
    ) = None,
    allow_height_partners: bool = False,
    height_partner_reference_xyz: NDArray[np.float64] | None = None,
    height_partner_xy_tolerance_m: float = 1.0e-9,
    height_partner_z_tolerance_m: float = 1.0e-9,
    height_partner_min_z_separation_m: float = 0.0,
) -> NDArray[np.float64]:
    """Filter candidates while allowing only the current station's height mate."""
    if candidates.size == 0:
        return candidates
    mask = np.ones(candidates.shape[0], dtype=bool)
    if visited_poses_xyz is not None and visited_poses_xyz.size:
        visited = np.asarray(visited_poses_xyz, dtype=float).reshape(-1, 3)
        diffs = candidates[:, None, :] - visited[None, :, :]
        if allow_height_partners:
            xy_dist = np.linalg.norm(diffs[:, :, :2], axis=2)
            z_dist = np.abs(diffs[:, :, 2])
            xy_tolerance = max(float(height_partner_xy_tolerance_m), 0.0)
            z_tolerance = max(float(height_partner_z_tolerance_m), 0.0)
            min_z_separation = max(
                float(height_partner_min_z_separation_m),
                0.0,
            )
            exact_action_visited = np.any(
                (xy_dist <= xy_tolerance) & (z_dist <= z_tolerance),
                axis=1,
            )
            reference = (
                visited[-1]
                if height_partner_reference_xyz is None
                else np.asarray(height_partner_reference_xyz, dtype=float).reshape(3)
            )
            reference_xy_dist = np.linalg.norm(
                candidates[:, :2] - reference[None, :2],
                axis=1,
            )
            reference_z_dist = np.abs(candidates[:, 2] - reference[2])
            height_partner = (
                (reference_xy_dist <= xy_tolerance)
                & (reference_z_dist > z_tolerance)
                & (reference_z_dist >= min_z_separation)
                & ~exact_action_visited
            )
            separated = ~exact_action_visited
            if min_dist_from_visited > 0.0:
                separated &= (
                    np.all(xy_dist >= min_dist_from_visited, axis=1) | height_partner
                )
        else:
            distances = np.linalg.norm(diffs, axis=2)
            separated = np.all(
                distances >= max(float(min_dist_from_visited), 0.0),
                axis=1,
            )
        mask &= separated
    if is_free_batch_fn is not None:
        free_mask = np.asarray(is_free_batch_fn(candidates), dtype=bool).reshape(-1)
        if free_mask.size != candidates.shape[0]:
            raise ValueError("Batched free-space checker returned the wrong length.")
        mask &= free_mask
    elif is_free_fn is not None:
        # Compatibility fallback for third-party map APIs without a batch method.
        mask &= np.fromiter(
            (bool(is_free_fn(point)) for point in candidates),
            dtype=bool,
            count=candidates.shape[0],
        )
    return candidates[mask]


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
    if n_samples <= 0:
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
    Sample points using a Sobol sequence; fall back to uniform if unavailable.

    Degenerate dimensions (lo == hi) are kept fixed at their bound value.
    """
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=float)
    try:
        from scipy.stats import qmc
    except ImportError:
        return _sample_uniform(rng, lo, hi, n_samples)
    active_dims = hi > lo
    if not np.any(active_dims):
        return np.repeat(lo[None, :], n_samples, axis=0)
    seed = int(rng.integers(0, 2**32 - 1))
    sampler = qmc.Sobol(d=int(np.sum(active_dims)), scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_samples, 1))))
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
    if int(n_samples) <= 0:
        return np.zeros(0, dtype=float)
    lo = np.array([0.0, 0.0, lower], dtype=float)
    hi = np.array([0.0, 0.0, upper], dtype=float)
    return _sample_sobol(rng, lo, hi, int(n_samples))[:, 2]


def _sample_current_xy_height_anchors(
    rng: np.random.Generator,
    current_pose_xyz: NDArray[np.float64],
    bounds_z: tuple[float, float],
    n_samples: int,
) -> NDArray[np.float64]:
    """Return batched low-discrepancy height anchors at the current xy station."""
    heights = sample_low_discrepancy_heights(rng, bounds_z, n_samples)
    if heights.size == 0:
        return np.zeros((0, 3), dtype=float)
    anchors = np.repeat(
        np.asarray(current_pose_xyz, dtype=float).reshape(1, 3),
        heights.size,
        axis=0,
    )
    anchors[:, 2] = heights
    return anchors


def _generate_ring_candidates(
    current_pose_xyz: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    n_candidates: int,
    min_dist_from_visited: float,
) -> NDArray[np.float64]:
    """Generate candidates on concentric rings around the current pose."""
    if n_candidates <= 0:
        return np.zeros((0, 3), dtype=float)
    max_dx = min(current_pose_xyz[0] - lo[0], hi[0] - current_pose_xyz[0])
    max_dy = min(current_pose_xyz[1] - lo[1], hi[1] - current_pose_xyz[1])
    max_radius = max(0.0, min(max_dx, max_dy))
    min_radius = max(0.1, min_dist_from_visited)
    if max_radius < min_radius:
        max_radius = min_radius
    num_rings = max(1, int(np.sqrt(n_candidates)))
    num_angles = max(4, int(np.ceil(n_candidates / num_rings)))
    radii = np.linspace(min_radius, max_radius, num=num_rings)
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_angles, endpoint=False)
    points = []
    for r in radii:
        for theta in angles:
            x = current_pose_xyz[0] + r * np.cos(theta)
            y = current_pose_xyz[1] + r * np.sin(theta)
            z = current_pose_xyz[2]
            points.append([x, y, z])
    points_arr = np.array(points, dtype=float)
    points_arr = points_arr[:n_candidates]
    return points_arr


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
    include_current_xy_height_actions: bool = False,
    continuous_height_anchor_count: int = 8,
    allow_height_partners: bool | None = None,
    height_partner_xy_tolerance_m: float = 1.0e-9,
    height_partner_z_tolerance_m: float = 1.0e-9,
    height_partner_min_z_separation_m: float = 0.0,
) -> NDArray[np.float64]:
    """Return (L, 3) candidate poses in free space for the given strategy.

    When ``detector_heights_m`` is supplied, xy stations are sampled once and
    expanded over the discrete height actions. Continuous z values are never
    sampled in that mode. Without discrete actions, requesting current-xy
    height actions adds low-discrepancy continuous anchors over the z bounds.
    """
    rng = np.random.default_rng() if rng is None else rng
    current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
    if current_pose_xyz.shape != (3,):
        raise ValueError("current_pose_xyz must be shape (3,).")
    visited = None
    if visited_poses_xyz is not None:
        visited = np.asarray(visited_poses_xyz, dtype=float)
        if visited.ndim == 1 and visited.size == 3:
            visited = visited.reshape(1, 3)
        if visited.ndim != 2 or visited.shape[1] != 3:
            raise ValueError("visited_poses_xyz must be shape (N, 3).")

    lo, hi = _resolve_bounds(bounds_xyz)
    is_free_fn = _resolve_free_space_checker(map_api)
    is_free_batch_fn = _resolve_free_space_batch_checker(map_api)
    height_actions: NDArray[np.float64] | None = None
    sample_lo = lo.copy()
    sample_hi = hi.copy()
    base_candidate_count = max(int(n_candidates), 1)
    if detector_heights_m is not None:
        height_actions = resolve_detector_height_actions(
            detector_heights_m,
            default_height_m=float(current_pose_xyz[2]),
            bounds_z=(float(lo[2]), float(hi[2])),
        )
        sample_lo[2] = float(current_pose_xyz[2])
        sample_hi[2] = float(current_pose_xyz[2])
        base_candidate_count = max(
            int(np.ceil(max(int(n_candidates), 1) / height_actions.size)),
            1,
        )

    if strategy == "ring":
        raw = _generate_ring_candidates(
            current_pose_xyz=current_pose_xyz,
            lo=sample_lo,
            hi=sample_hi,
            n_candidates=base_candidate_count,
            min_dist_from_visited=min_dist_from_visited,
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
        if include_current_xy_height_actions:
            current_height_actions = expand_candidate_height_actions(
                current_pose_xyz.reshape(1, 3),
                height_actions,
            )
            raw = np.vstack([current_height_actions, raw])
    elif include_current_xy_height_actions:
        current_height_anchors = _sample_current_xy_height_anchors(
            rng,
            current_pose_xyz,
            (float(lo[2]), float(hi[2])),
            max(int(continuous_height_anchor_count), 0),
        )
        if current_height_anchors.size:
            raw = np.vstack([current_height_anchors, raw])

    height_partners_enabled = (
        height_actions is not None
        if allow_height_partners is None
        else bool(allow_height_partners)
    )

    filtered = _filter_candidates(
        raw,
        visited,
        min_dist_from_visited,
        is_free_fn,
        is_free_batch_fn=is_free_batch_fn,
        allow_height_partners=height_partners_enabled,
        height_partner_reference_xyz=current_pose_xyz,
        height_partner_xy_tolerance_m=height_partner_xy_tolerance_m,
        height_partner_z_tolerance_m=height_partner_z_tolerance_m,
        height_partner_min_z_separation_m=height_partner_min_z_separation_m,
    )
    if filtered.shape[0] < n_candidates:
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
                    map_centers[:, None, :2] - visited[None, :, :2],
                    axis=2,
                )
                order = np.argsort(np.min(distances, axis=1))[::-1]
                map_centers = map_centers[order]
            map_centers = _filter_candidates(
                map_centers,
                visited,
                min_dist_from_visited,
                is_free_fn,
                is_free_batch_fn=is_free_batch_fn,
                allow_height_partners=height_partners_enabled,
                height_partner_reference_xyz=current_pose_xyz,
                height_partner_xy_tolerance_m=height_partner_xy_tolerance_m,
                height_partner_z_tolerance_m=height_partner_z_tolerance_m,
                height_partner_min_z_separation_m=height_partner_min_z_separation_m,
            )
            if map_centers.size:
                filtered = np.vstack([filtered, map_centers])
    if filtered.shape[0] < n_candidates:
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
            min_dist_from_visited,
            is_free_fn,
            is_free_batch_fn=is_free_batch_fn,
            allow_height_partners=height_partners_enabled,
            height_partner_reference_xyz=current_pose_xyz,
            height_partner_xy_tolerance_m=height_partner_xy_tolerance_m,
            height_partner_z_tolerance_m=height_partner_z_tolerance_m,
            height_partner_min_z_separation_m=height_partner_min_z_separation_m,
        )
        if extra.size:
            filtered = np.vstack([filtered, extra])
    return _stable_unique_candidates(filtered)[:n_candidates]
