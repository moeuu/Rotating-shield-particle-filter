"""Posterior source-mode extraction algorithms for DSS-PP."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from pf.estimator import JointPlanningParticles, RotatingShieldPFEstimator
from pf.full_spectrum import validate_full_spectrum_model
from pf.posterior import (
    validated_probability,
    validated_probability_distribution,
)
from planning.dss_types import DSSPPConfig, SignatureMode


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized weights and fail on an invalid posterior contract."""
    arr = np.asarray(weights, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Posterior weights must be finite and nonnegative.")
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Posterior weights must contain positive mass.")
    return arr / total


def _posterior_mode_weights(
    weights: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """Return marginal existence probabilities without erasing source count."""
    arr = np.asarray(weights, dtype=object).reshape(-1)
    return np.fromiter(
        (
            validated_probability(
                value,
                name=f"Mode existence probability[{index}]",
            )
            for index, value in enumerate(arr)
        ),
        dtype=np.float64,
        count=arr.size,
    )


def _isotope_presence_probability(
    modes: Sequence[SignatureMode],
) -> float | None:
    """Return the shared PF probability that an isotope has at least one source."""
    if not modes:
        return None
    configured = [
        mode.isotope_presence_probability
        for mode in modes
        if mode.isotope_presence_probability is not None
    ]
    if not configured:
        return validated_probability(
            sum(
                validated_probability(
                    mode.weight,
                    name=(f"Implicit signature source-mode probability[{index}]"),
                )
                for index, mode in enumerate(modes)
            ),
            name="Implicit signature-isotope presence probability",
        )
    if len(configured) != len(modes):
        raise ValueError(
            "Modes for one isotope cannot mix explicit and implicit presence."
        )
    values = np.asarray(configured, dtype=object).reshape(-1)
    values = np.fromiter(
        (
            validated_probability(
                value,
                name=f"Signature isotope presence probability[{index}]",
            )
            for index, value in enumerate(values)
        ),
        dtype=np.float64,
        count=values.size,
    )
    if not np.allclose(values, values[:1], rtol=0.0, atol=1.0e-12):
        raise ValueError("Modes for one isotope must share one presence probability.")
    return float(values[0])


def _flattened_posterior_mode_weights(
    modes_by_isotope: dict[str, list[SignatureMode]],
) -> NDArray[np.float64]:
    """Return concatenated mode masses while preserving each isotope's K=0 mass."""
    blocks = [
        _posterior_mode_weights([mode.weight for mode in modes])
        for modes in modes_by_isotope.values()
        if modes
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)


def _presence_weighted_rows(
    rows: Sequence[NDArray[np.float64]],
    presence_masses: Sequence[float],
    *,
    population_size: int,
) -> NDArray[np.float64]:
    """Average feature rows without renormalizing away absent-isotope mass."""
    if not rows:
        return np.zeros(0, dtype=float)
    stacked = np.vstack([np.asarray(row, dtype=float) for row in rows])
    raw_masses = np.asarray(presence_masses, dtype=object)
    masses = np.fromiter(
        (
            validated_probability(
                value,
                name=f"Feature-row presence probability[{index}]",
            )
            for index, value in enumerate(raw_masses)
        ),
        dtype=np.float64,
        count=raw_masses.size,
    )
    if masses.shape != (stacked.shape[0],):
        raise ValueError("Presence masses must match feature rows.")
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, (int, np.integer))
        or int(population_size) <= 0
    ):
        raise ValueError("population_size must be a positive integer.")
    denominator = int(population_size)
    return np.sum(stacked * masses[:, None], axis=0) / float(denominator)


def _planning_rng(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    """Return the caller-owned persistent planning generator."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("DSS planning requires an explicit persistent RNG.")
    return rng


def _validate_mode_capacity(
    estimator: RotatingShieldPFEstimator,
    config: DSSPPConfig,
) -> int:
    """Require the planner mode capacity to cover every PF source slot."""
    try:
        pf_config = estimator.pf_config
        configured_capacity = getattr(
            pf_config,
            "cardinality_capacity",
            getattr(
                pf_config,
                "hard_max_sources",
                getattr(pf_config, "max_sources", None),
            ),
        )
    except AttributeError as error:
        raise TypeError(
            "DSS planning requires an estimator with an explicit PF config."
        ) from error
    if configured_capacity is None:
        raise ValueError("DSS planning requires a finite PF cardinality capacity.")
    pf_cardinality_capacity = int(configured_capacity)
    if pf_cardinality_capacity <= 0:
        raise ValueError("PF cardinality capacity must be a positive integer.")
    configured = int(config.max_modes_per_isotope)
    if configured < pf_cardinality_capacity:
        raise ValueError(
            "max_modes_per_isotope must be at least the PF cardinality "
            f"capacity ({configured} < {pf_cardinality_capacity})."
        )
    return pf_cardinality_capacity


def _validate_eig_likelihood_contract(
    estimator: RotatingShieldPFEstimator,
    config: DSSPPConfig,
) -> None:
    """Require the exact full-spectrum model used by the joint PF."""
    if float(config.lambda_eig) <= 0.0:
        return
    if config.shield_program_search_policy in {
        "conditional_greedy_shadow",
        "conditional_greedy_all_pairs",
    }:
        exact_samples = getattr(estimator.pf_config, "planning_eig_samples", None)
        if (
            isinstance(exact_samples, bool)
            or not isinstance(exact_samples, (int, np.integer))
            or int(exact_samples) < 2
        ):
            raise ValueError(
                "Conditional-greedy shield search requires "
                "planning_eig_samples >= 2."
            )
    model = validate_full_spectrum_model(estimator.full_spectrum_generative_model)
    if not callable(getattr(model, "cross_log_likelihood_numpy", None)):
        raise RuntimeError(
            "DSS EIG requires vectorized full-spectrum cross likelihoods."
        )
    if not callable(getattr(estimator, "planning_joint_particles", None)):
        raise RuntimeError("DSS EIG requires aligned joint PF particles.")


def _continuous_kernel_for_estimator(
    estimator: RotatingShieldPFEstimator,
    *,
    detector_aperture_samples: int | None = None,
) -> ContinuousKernel:
    """Build a ContinuousKernel matching the estimator."""
    return estimator.continuous_kernel(
        detector_aperture_samples=detector_aperture_samples,
    )


def _weighted_surface_medoid_index(
    positions_xyz: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    surface_path_distance: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ]
    | None,
    surface_chart_ids: NDArray[np.int64] | None = None,
    surface_uv: NDArray[np.float64] | None = None,
    surface_coordinate_path_distance: Callable[
        [
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
    ]
    | None = None,
    row_chunk_size: int = 256,
) -> int:
    """Return the weighted medoid index using intrinsic surface distance."""
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    sample_weights = _normalise_weights(np.asarray(weights, dtype=float).reshape(-1))
    if positions.shape[0] != sample_weights.size or positions.shape[0] == 0:
        raise ValueError("Surface medoid inputs must contain matching samples.")
    if positions.shape[0] == 1:
        return 0
    coordinate_inputs = (
        surface_chart_ids,
        surface_uv,
        surface_coordinate_path_distance,
    )
    coordinates_active = all(value is not None for value in coordinate_inputs)
    if any(value is not None for value in coordinate_inputs) and not coordinates_active:
        raise ValueError(
            "Surface medoids require chart IDs, UV, and the coordinate-distance "
            "function together."
        )
    chart_ids: NDArray[np.int64] | None = None
    uv: NDArray[np.float64] | None = None
    if coordinates_active:
        raw_chart_ids = np.asarray(surface_chart_ids)
        if not np.issubdtype(raw_chart_ids.dtype, np.integer):
            raise TypeError("surface_chart_ids must contain integers.")
        chart_ids = raw_chart_ids.astype(np.int64, copy=False).reshape(-1)
        uv = np.asarray(surface_uv, dtype=np.float64)
        if (
            chart_ids.shape != (positions.shape[0],)
            or uv.shape != (positions.shape[0], 2)
            or np.any(~np.isfinite(uv))
        ):
            raise ValueError(
                "Surface medoid coordinates must match the position samples."
            )
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, (int, np.integer))
        or int(row_chunk_size) <= 0
    ):
        raise ValueError("row_chunk_size must be a positive integer.")
    medoid_costs = np.empty(positions.shape[0], dtype=float)
    chunk_size = int(row_chunk_size)
    for start in range(0, positions.shape[0], chunk_size):
        stop = min(start + chunk_size, positions.shape[0])
        if coordinates_active:
            assert chart_ids is not None
            assert uv is not None
            assert surface_coordinate_path_distance is not None
            distance_matrix = np.asarray(
                surface_coordinate_path_distance(
                    chart_ids[start:stop, None],
                    uv[start:stop, None, :],
                    chart_ids[None, :],
                    uv[None, :, :],
                ),
                dtype=float,
            )
        else:
            left = np.repeat(positions[start:stop], positions.shape[0], axis=0)
            right = np.tile(positions, (stop - start, 1))
            if surface_path_distance is None:
                distance_rows = np.linalg.norm(left - right, axis=1)
            else:
                distance_rows = np.asarray(
                    surface_path_distance(left, right),
                    dtype=float,
                ).reshape(-1)
            distance_matrix = distance_rows.reshape(
                stop - start,
                positions.shape[0],
            )
        if (
            distance_matrix.shape != (stop - start, positions.shape[0])
            or np.any(np.isnan(distance_matrix))
            or np.any(distance_matrix < 0.0)
        ):
            raise RuntimeError("Surface medoid calculation returned invalid distances.")
        medoid_costs[start:stop] = distance_matrix @ sample_weights
    minimum_cost = float(np.min(medoid_costs))
    tied = np.flatnonzero(
        np.isclose(medoid_costs, minimum_cost, rtol=0.0, atol=1.0e-12)
    )
    return int(tied[np.argmax(sample_weights[tied])])


def _cluster_source_samples(
    isotope: str,
    positions: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    strengths: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    *,
    radius_m: float,
    max_modes: int,
    particle_ids: Sequence[int] | None = None,
    isotope_presence_probability: float | None = None,
    surface_path_distance: Callable[
        [NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ]
    | None = None,
    surface_chart_ids: Sequence[int] | NDArray[np.int64] | None = None,
    surface_uv: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
    surface_coordinate_path_distance: Callable[
        [
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
    ]
    | None = None,
) -> list[SignatureMode]:
    """Cluster source samples into marginal-existence surface modes.

    Samples from the same PF particle contribute to a cluster's existence only
    once and may never occupy the same cluster. Thus simultaneous nearby
    sources remain distinct rather than being collapsed into one strong mode.
    Production callers use intrinsic surface-path distances, preventing
    ambient-near points on disconnected or folded surfaces from being merged.
    """
    if len(positions) == 0:
        return []
    pos_arr = np.asarray(positions, dtype=float)
    str_arr = np.asarray(strengths, dtype=float)
    w_arr = np.asarray(weights, dtype=float).reshape(-1)
    if (
        pos_arr.ndim != 2
        or pos_arr.shape[1] != 3
        or str_arr.shape != (pos_arr.shape[0],)
        or w_arr.shape != (pos_arr.shape[0],)
    ):
        raise ValueError(
            "Mode samples must have matching position, strength, and weight."
        )
    if not np.all(np.isfinite(pos_arr)) or not np.all(np.isfinite(str_arr)):
        raise ValueError("Mode sample positions and strengths must be finite.")
    if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
        raise ValueError("Mode sample weights must be finite and nonnegative.")
    resolved_radius = float(radius_m)
    if not np.isfinite(resolved_radius) or resolved_radius <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    if (
        isinstance(max_modes, bool)
        or not isinstance(max_modes, (int, np.integer))
        or int(max_modes) <= 0
    ):
        raise ValueError("max_modes must be a positive integer.")
    # ``max_modes`` is the PF's simultaneous source-slot capacity.  Marginal
    # posterior clustering can legitimately produce more spatial modes than
    # any one particle contains, so it must never be used as a truncation cap.
    _ = int(max_modes)
    coordinate_inputs = (
        surface_chart_ids,
        surface_uv,
        surface_coordinate_path_distance,
    )
    coordinates_active = all(value is not None for value in coordinate_inputs)
    if any(value is not None for value in coordinate_inputs) and not coordinates_active:
        raise ValueError(
            "Surface clustering requires chart IDs, UV, and the coordinate-"
            "distance function together."
        )
    chart_id_arr: NDArray[np.int64] | None = None
    surface_uv_arr: NDArray[np.float64] | None = None
    if coordinates_active:
        raw_chart_ids = np.asarray(surface_chart_ids)
        if not np.issubdtype(raw_chart_ids.dtype, np.integer):
            raise TypeError("surface_chart_ids must contain integers.")
        chart_id_arr = raw_chart_ids.astype(np.int64, copy=False).reshape(-1)
        surface_uv_arr = np.asarray(surface_uv, dtype=np.float64)
        if (
            chart_id_arr.shape != (pos_arr.shape[0],)
            or surface_uv_arr.shape != (pos_arr.shape[0], 2)
            or np.any(chart_id_arr < 0)
            or np.any(~np.isfinite(surface_uv_arr))
            or np.any(surface_uv_arr < 0.0)
            or np.any(surface_uv_arr > 1.0)
        ):
            raise ValueError(
                "Surface chart coordinates must match every source sample."
            )
    if particle_ids is None:
        particle_id_arr = np.arange(pos_arr.shape[0], dtype=np.int64)
    else:
        particle_id_arr = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
        if particle_id_arr.shape != (pos_arr.shape[0],):
            raise ValueError("particle_ids must contain one ID per source sample.")
        if np.any(particle_id_arr < 0):
            raise ValueError("particle_ids must be nonnegative.")
    if float(np.sum(w_arr)) <= 0.0:
        return []
    if isotope_presence_probability is None:
        resolved_presence = validated_probability(
            float(np.sum(w_arr)),
            name="Implicit clustered-isotope presence probability",
        )
    else:
        resolved_presence = validated_probability(
            isotope_presence_probability,
            name="Clustered-isotope presence probability",
        )
    order = np.argsort(w_arr)[::-1]
    clusters: list[list[int]] = []
    centers: list[NDArray[np.float64]] = []
    center_chart_ids: list[int] = []
    center_surface_uv: list[NDArray[np.float64]] = []
    cluster_particle_ids: list[set[int]] = []
    for idx in order:
        pos = pos_arr[int(idx)]
        particle_id = int(particle_id_arr[int(idx)])
        assigned = False
        if centers:
            if coordinates_active:
                assert chart_id_arr is not None
                assert surface_uv_arr is not None
                assert surface_coordinate_path_distance is not None
                center_distances = np.asarray(
                    surface_coordinate_path_distance(
                        np.full(
                            len(centers),
                            chart_id_arr[int(idx)],
                            dtype=np.int64,
                        ),
                        np.broadcast_to(
                            surface_uv_arr[int(idx)],
                            (len(centers), 2),
                        ),
                        np.asarray(center_chart_ids, dtype=np.int64),
                        np.asarray(center_surface_uv, dtype=np.float64),
                    ),
                    dtype=float,
                ).reshape(-1)
            elif surface_path_distance is None:
                center_array = np.asarray(centers, dtype=float).reshape(-1, 3)
                center_distances = np.linalg.norm(
                    center_array - pos[None, :],
                    axis=1,
                )
            else:
                center_array = np.asarray(centers, dtype=float).reshape(-1, 3)
                center_distances = np.asarray(
                    surface_path_distance(
                        np.broadcast_to(pos, center_array.shape),
                        center_array,
                    ),
                    dtype=float,
                ).reshape(-1)
                if (
                    center_distances.shape != (len(centers),)
                    or np.any(np.isnan(center_distances))
                    or np.any(center_distances < 0.0)
                ):
                    raise RuntimeError(
                        "Surface mode clustering returned invalid path distances."
                    )
        else:
            center_distances = np.zeros(0, dtype=float)
        for cluster_idx, distance in enumerate(center_distances):
            if particle_id in cluster_particle_ids[cluster_idx]:
                continue
            if float(distance) <= resolved_radius:
                clusters[cluster_idx].append(int(idx))
                cluster_particle_ids[cluster_idx].add(particle_id)
                assigned = True
                break
        if not assigned:
            clusters.append([int(idx)])
            centers.append(pos.copy())
            if coordinates_active:
                assert chart_id_arr is not None
                assert surface_uv_arr is not None
                center_chart_ids.append(int(chart_id_arr[int(idx)]))
                center_surface_uv.append(surface_uv_arr[int(idx)].copy())
            cluster_particle_ids.append({particle_id})
    modes: list[SignatureMode] = []
    for cluster in clusters:
        cluster_weights = w_arr[cluster]
        cluster_particle_ids = particle_id_arr[cluster]
        particle_order = np.argsort(cluster_particle_ids, kind="stable")
        sorted_particle_ids = cluster_particle_ids[particle_order]
        group_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.asarray([True]),
                    sorted_particle_ids[1:] != sorted_particle_ids[:-1],
                )
            )
        )
        sorted_particle_weights = cluster_weights[particle_order]
        particle_weight_max = np.maximum.reduceat(
            sorted_particle_weights,
            group_starts,
        )
        particle_weight_min = np.minimum.reduceat(
            sorted_particle_weights,
            group_starts,
        )
        if not np.allclose(
            particle_weight_max,
            particle_weight_min,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                "All source samples from one PF particle must share its weight."
            )
        cluster_existence = float(np.sum(particle_weight_max))
        if cluster_existence <= 0.0:
            continue
        cluster_existence = validated_probability(
            cluster_existence,
            name="Cluster existence probability",
        )
        representative_local_index = _weighted_surface_medoid_index(
            pos_arr[cluster],
            cluster_weights,
            surface_path_distance=surface_path_distance,
            surface_chart_ids=(None if chart_id_arr is None else chart_id_arr[cluster]),
            surface_uv=(None if surface_uv_arr is None else surface_uv_arr[cluster]),
            surface_coordinate_path_distance=surface_coordinate_path_distance,
        )
        representative = pos_arr[cluster[representative_local_index]].copy()
        strength = float(np.sum(str_arr[cluster] * cluster_weights) / cluster_existence)
        if not np.isfinite(strength) or strength <= 0.0:
            raise RuntimeError(
                "A supported posterior source mode must have positive strength."
            )
        if coordinates_active:
            assert chart_id_arr is not None
            assert surface_uv_arr is not None
            assert surface_coordinate_path_distance is not None
            representative_sample_index = int(cluster[representative_local_index])
            representative_distances = np.asarray(
                surface_coordinate_path_distance(
                    np.full(
                        len(cluster),
                        chart_id_arr[representative_sample_index],
                        dtype=np.int64,
                    ),
                    np.broadcast_to(
                        surface_uv_arr[representative_sample_index],
                        (len(cluster), 2),
                    ),
                    chart_id_arr[cluster],
                    surface_uv_arr[cluster],
                ),
                dtype=float,
            ).reshape(-1)
        elif surface_path_distance is None:
            representative_distances = np.linalg.norm(
                pos_arr[cluster] - representative[None, :],
                axis=1,
            )
        else:
            representative_distances = np.asarray(
                surface_path_distance(
                    np.broadcast_to(representative, (len(cluster), 3)),
                    pos_arr[cluster],
                ),
                dtype=float,
            ).reshape(-1)
        if (
            representative_distances.shape != (len(cluster),)
            or np.any(np.isnan(representative_distances))
            or np.any(representative_distances < 0.0)
        ):
            raise RuntimeError("Surface mode spread returned invalid path distances.")
        spread = float(
            np.sqrt(
                np.average(
                    representative_distances * representative_distances,
                    weights=cluster_weights,
                )
            )
        )
        modes.append(
            SignatureMode(
                isotope=isotope,
                position_xyz=representative.astype(float),
                strength_cps_1m=strength,
                weight=cluster_existence,
                spread_m=spread,
                isotope_presence_probability=resolved_presence,
                surface_chart_id=(
                    None
                    if chart_id_arr is None
                    else int(chart_id_arr[int(cluster[representative_local_index])])
                ),
                surface_uv=(
                    None
                    if surface_uv_arr is None
                    else tuple(
                        float(value)
                        for value in surface_uv_arr[
                            int(cluster[representative_local_index])
                        ]
                    )
                ),
            )
        )
    modes.sort(key=lambda mode: mode.weight, reverse=True)
    represented_source_mass = float(
        np.sum([float(mode.weight) for mode in modes], dtype=np.float64)
    )
    expected_source_mass = float(np.sum(w_arr, dtype=np.float64))
    if not np.isclose(
        represented_source_mass,
        expected_source_mass,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Surface-mode clustering failed to preserve posterior expected "
            "source-count mass."
        )
    return modes


def extract_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_particles: int | None = None,
    method: str | None = None,
    mode_cluster_radius_m: float = 1.5,
    max_modes_per_isotope: int = 5,
    rng: np.random.Generator | None = None,
    particles_by_isotope: dict[
        str,
        tuple[Sequence[object], NDArray[np.float64]],
    ]
    | None = None,
    joint_particles: JointPlanningParticles | None = None,
) -> dict[str, list[SignatureMode]]:
    """Extract isotope-wise modes while preserving native joint-PF mass.

    Production planning passes ``joint_particles`` so particle/source slots are
    validated and packed with array operations. ``particles_by_isotope`` is
    retained only for small deterministic oracles that expose state objects.
    """
    planning_rng = _planning_rng(rng)
    if particles_by_isotope is not None and joint_particles is not None:
        raise ValueError(
            "Provide either particles_by_isotope or joint_particles, not both."
        )
    if particles_by_isotope is None and joint_particles is None:
        particles = estimator.planning_particles(
            max_particles=max_particles,
            method=method,
            rng=planning_rng,
        )
    else:
        particles = particles_by_isotope
    if joint_particles is not None:
        if tuple(str(value) for value in joint_particles.isotope_order) != tuple(
            str(value) for value in estimator.isotopes
        ):
            raise ValueError(
                "Joint mode snapshot isotope order differs from the estimator."
            )
        joint_weights = _normalise_weights(
            np.asarray(joint_particles.weights_n, dtype=np.float64)
        )
        joint_particle_count = int(joint_weights.size)
        joint_particle_ids_n1 = np.arange(
            joint_particle_count,
            dtype=np.int64,
        )[:, None]
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    eps = 1e-12
    for isotope in estimator.isotopes:
        if joint_particles is not None:
            isotope_key = str(isotope)
            try:
                packed_positions_nk3 = np.asarray(
                    joint_particles.positions_nk3_by_isotope[isotope_key],
                    dtype=np.float64,
                )
                packed_strengths_nk = np.asarray(
                    joint_particles.strengths_nk_by_isotope[isotope_key],
                    dtype=np.float64,
                )
                packed_mask_nk = np.asarray(
                    joint_particles.source_mask_nk_by_isotope[isotope_key],
                    dtype=bool,
                )
                packed_chart_ids_nk = np.asarray(
                    joint_particles.surface_chart_ids_nk_by_isotope[isotope_key]
                )
                packed_surface_uv_nk2 = np.asarray(
                    joint_particles.surface_uv_nk2_by_isotope[isotope_key],
                    dtype=np.float64,
                )
            except KeyError as error:
                raise ValueError(
                    "Joint mode snapshot is missing an estimator isotope."
                ) from error
            expected_slots = packed_mask_nk.shape
            if (
                packed_mask_nk.ndim != 2
                or packed_positions_nk3.shape != expected_slots + (3,)
                or packed_strengths_nk.shape != expected_slots
                or packed_chart_ids_nk.shape != expected_slots
                or not np.issubdtype(
                    packed_chart_ids_nk.dtype,
                    np.integer,
                )
                or packed_surface_uv_nk2.shape != expected_slots + (2,)
                or expected_slots[0] != joint_particle_count
                or np.any(~np.isfinite(packed_positions_nk3))
                or np.any(~np.isfinite(packed_strengths_nk))
                or np.any(~np.isfinite(packed_surface_uv_nk2))
                or np.any(packed_strengths_nk[packed_mask_nk] <= 0.0)
                or np.any(packed_strengths_nk[~packed_mask_nk] != 0.0)
                or np.any(packed_chart_ids_nk[packed_mask_nk] < 0)
                or np.any(packed_surface_uv_nk2[packed_mask_nk] < 0.0)
                or np.any(packed_surface_uv_nk2[packed_mask_nk] > 1.0)
            ):
                raise ValueError(
                    "Joint mode snapshot contains invalid packed source arrays."
                )
            positions = packed_positions_nk3[packed_mask_nk]
            strengths = packed_strengths_nk[packed_mask_nk]
            sample_chart_ids = packed_chart_ids_nk[packed_mask_nk].astype(
                np.int64, copy=False
            )
            sample_surface_uv = packed_surface_uv_nk2[packed_mask_nk]
            broadcast_weights_nk = np.broadcast_to(
                joint_weights[:, None],
                expected_slots,
            )
            sample_weights = broadcast_weights_nk[packed_mask_nk]
            broadcast_particle_ids_nk = np.broadcast_to(
                joint_particle_ids_n1,
                expected_slots,
            )
            sample_particle_ids = broadcast_particle_ids_nk[packed_mask_nk]
            isotope_presence_probability = float(
                np.sum(
                    joint_weights[np.any(packed_mask_nk, axis=1)],
                    dtype=np.float64,
                )
            )
        else:
            positions_list: list[NDArray[np.float64]] = []
            strengths_list: list[float] = []
            sample_weights_list: list[float] = []
            sample_particle_ids_list: list[int] = []
            sample_chart_ids_list: list[int] = []
            sample_surface_uv_list: list[NDArray[np.float64]] = []
            isotope_presence_probability = 0.0
            if particles is None or isotope not in particles:
                positions = np.zeros((0, 3), dtype=np.float64)
                strengths = np.zeros(0, dtype=np.float64)
                sample_weights = np.zeros(0, dtype=np.float64)
                sample_particle_ids = np.zeros(0, dtype=np.int64)
                sample_chart_ids = None
                sample_surface_uv = None
                modes_by_isotope[isotope] = []
                continue
            states, weights = particles[isotope]
            norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
            for particle_index, (state, particle_weight) in enumerate(
                zip(states, norm_weights)
            ):
                num_sources = int(state.num_sources)
                if num_sources <= 0:
                    continue
                isotope_presence_probability += float(particle_weight)
                state_strengths = np.asarray(
                    state.strengths[:num_sources],
                    dtype=float,
                )
                if (
                    state_strengths.shape != (num_sources,)
                    or np.any(~np.isfinite(state_strengths))
                    or np.any(state_strengths <= 0.0)
                ):
                    raise ValueError(
                        "A positive-cardinality PF state must contain one "
                        "finite positive strength per source."
                    )
                total_strength = float(np.sum(state_strengths))
                if total_strength <= eps:
                    raise ValueError(
                        "A positive-cardinality PF state must contain positive "
                        "source strength."
                    )
                particle_filter = estimator.filters[str(isotope)]
                state_positions = np.asarray(
                    particle_filter.continuous_state_positions(state),
                    dtype=float,
                )
                if state_positions.shape != (num_sources, 3):
                    raise ValueError(
                        "A positive-cardinality PF state must resolve to one "
                        "continuous surface position per source."
                    )
                state_chart_ids = np.asarray(
                    state.surface_chart_ids,
                )
                state_surface_uv = np.asarray(
                    state.surface_uv,
                    dtype=np.float64,
                )
                if (
                    not np.issubdtype(state_chart_ids.dtype, np.integer)
                    or state_chart_ids.shape != (num_sources,)
                    or state_surface_uv.shape != (num_sources, 2)
                    or np.any(state_chart_ids < 0)
                    or np.any(~np.isfinite(state_surface_uv))
                    or np.any(state_surface_uv < 0.0)
                    or np.any(state_surface_uv > 1.0)
                ):
                    raise ValueError(
                        "A positive-cardinality PF state must retain one "
                        "authoritative chart/UV coordinate per source."
                    )
                for pos, strength, chart_id, source_uv in zip(
                    state_positions,
                    state_strengths,
                    state_chart_ids,
                    state_surface_uv,
                ):
                    positions_list.append(np.asarray(pos, dtype=float))
                    strengths_list.append(float(strength))
                    sample_weights_list.append(float(particle_weight))
                    sample_particle_ids_list.append(int(particle_index))
                    sample_chart_ids_list.append(int(chart_id))
                    sample_surface_uv_list.append(
                        np.asarray(source_uv, dtype=np.float64)
                    )
            positions = np.asarray(positions_list, dtype=np.float64).reshape(-1, 3)
            strengths = np.asarray(strengths_list, dtype=np.float64)
            sample_weights = np.asarray(sample_weights_list, dtype=np.float64)
            sample_particle_ids = np.asarray(
                sample_particle_ids_list,
                dtype=np.int64,
            )
            sample_chart_ids = np.asarray(
                sample_chart_ids_list,
                dtype=np.int64,
            )
            sample_surface_uv = np.asarray(
                sample_surface_uv_list,
                dtype=np.float64,
            ).reshape(-1, 2)
        coordinate_distance = None
        if isinstance(estimator, RotatingShieldPFEstimator):
            atlas = estimator.filters[str(isotope)]._structural_rj_surface_atlas
            if atlas is None:
                raise RuntimeError(
                    "Production mode extraction requires a continuous surface atlas."
                )
            if sample_chart_ids is None or sample_surface_uv is None:
                raise RuntimeError(
                    "Production mode extraction requires authoritative chart "
                    "coordinates for every source sample."
                )
            decoded_positions = np.asarray(
                atlas.positions_xyz(
                    sample_chart_ids,
                    sample_surface_uv,
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if decoded_positions.shape != positions.shape or not np.allclose(
                decoded_positions,
                positions,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(
                    "Planner XYZ positions differ from their authoritative "
                    "continuous surface chart coordinates."
                )
            coordinate_distance = atlas.surface_coordinate_path_distance_upper_bound_m
        modes = _cluster_source_samples(
            isotope,
            positions,
            strengths,
            sample_weights,
            radius_m=mode_cluster_radius_m,
            max_modes=max_modes_per_isotope,
            particle_ids=np.asarray(sample_particle_ids, dtype=np.int64),
            isotope_presence_probability=isotope_presence_probability,
            surface_path_distance=(
                None
                if not isinstance(estimator, RotatingShieldPFEstimator)
                else estimator.filters[
                    str(isotope)
                ]._structural_rj_surface_atlas.surface_path_distance_upper_bound_m
            ),
            surface_chart_ids=(
                None if coordinate_distance is None else sample_chart_ids
            ),
            surface_uv=(None if coordinate_distance is None else sample_surface_uv),
            surface_coordinate_path_distance=coordinate_distance,
        )
        modes_by_isotope[isotope] = modes
    return modes_by_isotope


def _official_signature_modes(
    estimator: RotatingShieldPFEstimator,
    *,
    max_modes_per_isotope: int,
) -> tuple[dict[str, list[SignatureMode]], dict[str, object]]:
    """Return the official point projection for read-only consistency checks.

    Candidate geometry and exact EIG use the full aligned joint posterior with
    unconditional mass. These joint-MAP modes are diagnostic only and must
    never restrict candidate generation or erase the K=0 posterior mass.
    """
    point_estimates = estimator.posterior_point_estimate()
    isotope_order_getter = getattr(estimator, "joint_isotope_order", None)
    isotope_order = (
        tuple(str(value) for value in isotope_order_getter())
        if callable(isotope_order_getter)
        else tuple(sorted(str(value) for value in estimator.isotopes))
    )
    if set(point_estimates) != set(isotope_order):
        raise RuntimeError("Official PF point estimates do not match planner isotopes.")
    modes_by_isotope: dict[str, list[SignatureMode]] = {}
    cardinality_vector: list[int] = []
    stratum_masses: list[float] = []
    medoids_by_isotope: dict[str, list[list[float]]] = {}
    for isotope in isotope_order:
        point_estimate = point_estimates[isotope]
        raw_map_cardinality = point_estimate.map_cardinality
        if isinstance(raw_map_cardinality, (bool, np.bool_)) or not isinstance(
            raw_map_cardinality,
            (int, np.integer),
        ):
            raise ValueError("Official PF cardinality must be an integer.")
        map_cardinality = int(raw_map_cardinality)
        if map_cardinality < 0:
            raise ValueError("Official PF cardinality cannot be negative.")
        if map_cardinality > int(max_modes_per_isotope):
            raise ValueError("Official PF cardinality exceeds planner mode capacity.")
        if len(point_estimate.modes) != map_cardinality:
            raise RuntimeError(
                "Official PF mode count differs from its MAP cardinality."
            )
        cardinality_vector.append(map_cardinality)
        selected_mass = validated_probability(
            point_estimate.selected_stratum_mass,
            name=f"Official joint-MAP stratum mass[{isotope}]",
        )
        stratum_masses.append(selected_mass)
        distribution: dict[int, object] = {}
        for raw_cardinality, mass in point_estimate.cardinality_distribution.items():
            if isinstance(raw_cardinality, (bool, np.bool_)) or not isinstance(
                raw_cardinality,
                (int, np.integer),
            ):
                raise ValueError(
                    "Official PF cardinality-distribution keys must be integers."
                )
            cardinality = int(raw_cardinality)
            if cardinality < 0:
                raise ValueError(
                    "Official PF cardinality-distribution keys must be nonnegative."
                )
            distribution[cardinality] = mass
        distribution_values = validated_probability_distribution(
            [distribution[cardinality] for cardinality in sorted(distribution)],
            name=f"Official cardinality distribution[{isotope}]",
        )
        distribution_keys = sorted(distribution)
        presence_probability = validated_probability(
            float(
                np.sum(
                    distribution_values[
                        np.asarray(distribution_keys, dtype=np.int64) > 0
                    ]
                )
            ),
            name=f"Official isotope presence probability[{isotope}]",
        )
        isotope_modes: list[SignatureMode] = []
        medoid_rows: list[list[float]] = []
        for mode in point_estimate.modes:
            medoid = np.asarray(
                mode.position_medoid_xyz,
                dtype=np.float64,
            ).reshape(3)
            if np.any(~np.isfinite(medoid)):
                raise ValueError("Official PF surface medoid must be finite.")
            strength = float(mode.strength_representative_cps_1m)
            if not np.isfinite(strength) or strength <= 0.0:
                raise ValueError(
                    "Official PF source strength must be finite and positive."
                )
            mode_mass = validated_probability(
                mode.posterior_mass,
                name=f"Official source-mode mass[{isotope}]",
            )
            if not np.isclose(
                mode_mass,
                selected_mass,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "Official source mode and joint-MAP stratum masses differ."
                )
            surface_radius = mode.credible_surface_path_radius_95_m
            spread = (
                float(mode.credible_radius_95_m)
                if surface_radius is None
                else float(surface_radius)
            )
            if not np.isfinite(spread) or spread < 0.0:
                raise ValueError("Official PF source spread must be nonnegative.")
            isotope_modes.append(
                SignatureMode(
                    isotope=isotope,
                    position_xyz=medoid.copy(),
                    strength_cps_1m=strength,
                    weight=mode_mass,
                    spread_m=spread,
                    isotope_presence_probability=presence_probability,
                    surface_chart_id=(
                        None
                        if mode.surface_chart_id is None
                        else int(mode.surface_chart_id)
                    ),
                    surface_uv=(
                        None
                        if mode.surface_uv is None
                        else tuple(float(value) for value in mode.surface_uv)
                    ),
                )
            )
            medoid_rows.append([float(value) for value in medoid])
        modes_by_isotope[isotope] = isotope_modes
        medoids_by_isotope[isotope] = medoid_rows
    if stratum_masses and not np.allclose(
        stratum_masses,
        stratum_masses[:1],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Official isotope reports do not share one joint-MAP stratum mass."
        )
    joint_distribution_getter = getattr(
        estimator,
        "posterior_joint_cardinality_distribution",
        None,
    )
    joint_distribution: dict[tuple[int, ...], float] | None = None
    if callable(joint_distribution_getter):
        raw_joint_distribution = joint_distribution_getter()
        if not isinstance(raw_joint_distribution, Mapping) or not (
            raw_joint_distribution
        ):
            raise RuntimeError("Official joint cardinality distribution is empty.")
        joint_entries: list[tuple[tuple[int, ...], object]] = []
        for raw_vector, mass in raw_joint_distribution.items():
            if (
                not isinstance(raw_vector, tuple)
                or len(raw_vector) != len(isotope_order)
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    for value in raw_vector
                )
            ):
                raise ValueError(
                    "Official joint cardinality keys must be nonnegative "
                    "integer tuples matching the isotope order."
                )
            joint_entries.append(
                (
                    tuple(int(value) for value in raw_vector),
                    mass,
                )
            )
        joint_masses = validated_probability_distribution(
            [mass for _, mass in joint_entries],
            name="Official joint cardinality distribution",
        )
        joint_distribution = {
            vector: float(mass)
            for (vector, _), mass in zip(
                joint_entries,
                joint_masses,
                strict=True,
            )
        }
        maximum_mass = max(joint_distribution.values())
        tied_vectors = sorted(
            vector
            for vector, mass in joint_distribution.items()
            if np.isclose(
                mass,
                maximum_mass,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        official_vector = tuple(cardinality_vector)
        if not tied_vectors or official_vector != tied_vectors[0]:
            raise RuntimeError(
                "Planner cardinality vector differs from the official joint MAP."
            )
        if stratum_masses and not np.isclose(
            stratum_masses[0],
            maximum_mass,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Planner stratum mass differs from the official joint MAP."
            )
    diagnostics: dict[str, object] = {
        "source": "official_joint_map_posterior_projection",
        "isotope_order": list(isotope_order),
        "joint_map_cardinality_vector": list(cardinality_vector),
        "joint_map_stratum_mass": (float(stratum_masses[0]) if stratum_masses else 0.0),
        "position_representative": "common_joint_particle_surface_medoid",
        "medoids_by_isotope": medoids_by_isotope,
        "verified_against_joint_cardinality_distribution": bool(
            joint_distribution is not None
        ),
    }
    return modes_by_isotope, diagnostics
