"""Truth-free conditional one-source versus two-component diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.strength_prior import STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY


@dataclass(frozen=True)
class _RepresentativeState:
    """Store one coherent posterior representative in flat source-slot order."""

    isotope_order: tuple[str, ...]
    positions_s3: NDArray[np.float64]
    chart_ids_s: NDArray[np.int64]
    surface_uv_s2: NDArray[np.float64]
    strengths_s: NDArray[np.float64]
    isotope_indices_s: NDArray[np.int64]
    slot_offsets_i: NDArray[np.int64]
    source_counts_i: NDArray[np.int64]


@dataclass(frozen=True)
class _PairBatch:
    """Store eligible adjacent-cardinality diagnostic component pairs."""

    isotope_indices_m: NDArray[np.int64]
    component_indices_m2: NDArray[np.int64]
    source_slots_m2: NDArray[np.int64]
    chart_ids_m2: NDArray[np.int64]
    surface_uv_m22: NDArray[np.float64]
    positions_m23: NDArray[np.float64]
    strengths_m2: NDArray[np.float64]
    face_ids_m2: NDArray[np.object_]
    surface_distances_m: NDArray[np.float64]

    @property
    def count(self) -> int:
        """Return the number of eligible component pairs."""
        return int(self.isotope_indices_m.size)


@dataclass(frozen=True)
class _PositionBatch:
    """Store ragged per-pair surface positions in one flat batch."""

    pair_indices_r: NDArray[np.int64]
    chart_ids_r: NDArray[np.int64]
    surface_uv_r2: NDArray[np.float64]
    positions_r3: NDArray[np.float64]

    @property
    def count(self) -> int:
        """Return the number of position rows."""
        return int(self.pair_indices_r.size)


@dataclass(frozen=True)
class _UnitComponents:
    """Store unit-strength global-line transport for flat source positions."""

    total_rvl: NDArray[np.float64]
    uncollided_rvl: NDArray[np.float64]
    features_rvlf: NDArray[np.float64]


@dataclass(frozen=True)
class _ReferenceComponents:
    """Store the representative full-spectrum source-resolved components."""

    total_1vsl: NDArray[np.float64]
    uncollided_1vsl: NDArray[np.float64]
    features_1vslf: NDArray[np.float64]


def _representative_state(estimator: Any) -> _RepresentativeState:
    """Return the coherent posterior-report state in flat source-slot order."""
    estimates = estimator.posterior_point_estimate()
    isotope_order = tuple(str(value) for value in estimator.joint_isotope_order())
    if not isinstance(estimates, Mapping) or set(estimates) != set(isotope_order):
        raise RuntimeError(
            "Conditional source-count diagnostics require every isotope report."
        )
    position_parts: list[NDArray[np.float64]] = []
    chart_parts: list[NDArray[np.int64]] = []
    uv_parts: list[NDArray[np.float64]] = []
    strength_parts: list[NDArray[np.float64]] = []
    isotope_parts: list[NDArray[np.int64]] = []
    counts = np.empty(len(isotope_order), dtype=np.int64)
    for isotope_index, isotope in enumerate(isotope_order):
        modes = tuple(estimates[isotope].modes)
        counts[isotope_index] = len(modes)
        if not modes:
            continue
        positions = np.asarray(
            [mode.position_medoid_xyz for mode in modes],
            dtype=np.float64,
        )
        strengths = np.asarray(
            [mode.strength_representative_cps_1m for mode in modes],
            dtype=np.float64,
        )
        raw_charts = [mode.surface_chart_id for mode in modes]
        raw_uv = [mode.surface_uv for mode in modes]
        if any(value is None for value in raw_charts) or any(
            value is None for value in raw_uv
        ):
            raise RuntimeError(
                "Conditional source-count diagnostics require chart/UV modes."
            )
        charts = np.asarray(raw_charts, dtype=np.int64)
        surface_uv = np.asarray(raw_uv, dtype=np.float64)
        atlas = estimator.filters[isotope]._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Conditional diagnostics require a surface atlas.")
        expected_positions = np.asarray(
            atlas.positions_xyz(charts, surface_uv),
            dtype=np.float64,
        )
        if (
            positions.shape != (len(modes), 3)
            or surface_uv.shape != (len(modes), 2)
            or strengths.shape != (len(modes),)
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(strengths))
            or np.any(strengths <= 0.0)
            or not np.allclose(
                positions,
                expected_positions,
                rtol=0.0,
                atol=1.0e-10,
            )
        ):
            raise RuntimeError(
                "Posterior representative modes are not valid surface states."
            )
        position_parts.append(positions)
        chart_parts.append(charts)
        uv_parts.append(surface_uv)
        strength_parts.append(strengths)
        isotope_parts.append(np.full(len(modes), isotope_index, dtype=np.int64))
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    return _RepresentativeState(
        isotope_order=isotope_order,
        positions_s3=(
            np.concatenate(position_parts, axis=0)
            if position_parts
            else np.zeros((0, 3), dtype=np.float64)
        ),
        chart_ids_s=(
            np.concatenate(chart_parts) if chart_parts else np.zeros(0, dtype=np.int64)
        ),
        surface_uv_s2=(
            np.concatenate(uv_parts, axis=0)
            if uv_parts
            else np.zeros((0, 2), dtype=np.float64)
        ),
        strengths_s=(
            np.concatenate(strength_parts)
            if strength_parts
            else np.zeros(0, dtype=np.float64)
        ),
        isotope_indices_s=(
            np.concatenate(isotope_parts)
            if isotope_parts
            else np.zeros(0, dtype=np.int64)
        ),
        slot_offsets_i=offsets,
        source_counts_i=counts,
    )


def _eligible_pairs(
    estimator: Any,
    state: _RepresentativeState,
    *,
    maximum_surface_distance_m: float,
) -> _PairBatch:
    """Return same- or adjacent-face pairs inside a fixed surface distance."""
    isotope_parts: list[NDArray[np.int64]] = []
    component_parts: list[NDArray[np.int64]] = []
    slot_parts: list[NDArray[np.int64]] = []
    chart_parts: list[NDArray[np.int64]] = []
    uv_parts: list[NDArray[np.float64]] = []
    position_parts: list[NDArray[np.float64]] = []
    strength_parts: list[NDArray[np.float64]] = []
    face_parts: list[NDArray[np.object_]] = []
    distance_parts: list[NDArray[np.float64]] = []
    for isotope_index, isotope in enumerate(state.isotope_order):
        source_count = int(state.source_counts_i[isotope_index])
        if source_count < 2:
            continue
        slot_start = int(state.slot_offsets_i[isotope_index])
        slots = np.arange(slot_start, slot_start + source_count, dtype=np.int64)
        first, second = np.triu_indices(source_count, k=1)
        first_slots = slots[first]
        second_slots = slots[second]
        atlas = estimator.filters[isotope]._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Conditional diagnostics require a surface atlas.")
        geometry = atlas.geometry
        face_labels = np.asarray(geometry.face_ids, dtype=object)
        unique_faces, chart_face_codes = np.unique(
            face_labels,
            return_inverse=True,
        )
        face_adjacency = np.eye(unique_faces.size, dtype=np.bool_)
        edges = np.asarray(geometry.adjacency_edges, dtype=np.int64).reshape(-1, 2)
        if edges.size:
            edge_faces = chart_face_codes[edges]
            face_adjacency[edge_faces[:, 0], edge_faces[:, 1]] = True
            face_adjacency[edge_faces[:, 1], edge_faces[:, 0]] = True
        first_charts = state.chart_ids_s[first_slots]
        second_charts = state.chart_ids_s[second_slots]
        adjacent = face_adjacency[
            chart_face_codes[first_charts],
            chart_face_codes[second_charts],
        ]
        distances = np.asarray(
            atlas.surface_coordinate_path_distance_upper_bound_m(
                first_charts,
                state.surface_uv_s2[first_slots],
                second_charts,
                state.surface_uv_s2[second_slots],
            ),
            dtype=np.float64,
        )
        eligible = (
            adjacent
            & np.isfinite(distances)
            & (distances <= maximum_surface_distance_m)
        )
        if not np.any(eligible):
            continue
        selected_first = first_slots[eligible]
        selected_second = second_slots[eligible]
        selected_slots = np.column_stack((selected_first, selected_second))
        isotope_parts.append(
            np.full(selected_first.size, isotope_index, dtype=np.int64)
        )
        component_parts.append(
            np.column_stack((first[eligible], second[eligible])).astype(
                np.int64,
                copy=False,
            )
        )
        slot_parts.append(selected_slots)
        chart_parts.append(state.chart_ids_s[selected_slots])
        uv_parts.append(state.surface_uv_s2[selected_slots])
        position_parts.append(state.positions_s3[selected_slots])
        strength_parts.append(state.strengths_s[selected_slots])
        face_parts.append(face_labels[state.chart_ids_s[selected_slots]])
        distance_parts.append(distances[eligible])
    if not isotope_parts:
        return _PairBatch(
            isotope_indices_m=np.zeros(0, dtype=np.int64),
            component_indices_m2=np.zeros((0, 2), dtype=np.int64),
            source_slots_m2=np.zeros((0, 2), dtype=np.int64),
            chart_ids_m2=np.zeros((0, 2), dtype=np.int64),
            surface_uv_m22=np.zeros((0, 2, 2), dtype=np.float64),
            positions_m23=np.zeros((0, 2, 3), dtype=np.float64),
            strengths_m2=np.zeros((0, 2), dtype=np.float64),
            face_ids_m2=np.zeros((0, 2), dtype=object),
            surface_distances_m=np.zeros(0, dtype=np.float64),
        )
    return _PairBatch(
        isotope_indices_m=np.concatenate(isotope_parts),
        component_indices_m2=np.concatenate(component_parts, axis=0),
        source_slots_m2=np.concatenate(slot_parts, axis=0),
        chart_ids_m2=np.concatenate(chart_parts, axis=0),
        surface_uv_m22=np.concatenate(uv_parts, axis=0),
        positions_m23=np.concatenate(position_parts, axis=0),
        strengths_m2=np.concatenate(strength_parts, axis=0),
        face_ids_m2=np.concatenate(face_parts, axis=0),
        surface_distances_m=np.concatenate(distance_parts),
    )


def _coarse_position_batch(
    estimator: Any,
    state: _RepresentativeState,
    pairs: _PairBatch,
) -> _PositionBatch:
    """Return all union-face chart centers plus both observed pair positions."""
    pair_parts: list[NDArray[np.int64]] = []
    chart_parts: list[NDArray[np.int64]] = []
    uv_parts: list[NDArray[np.float64]] = []
    position_parts: list[NDArray[np.float64]] = []
    for isotope_index, isotope in enumerate(state.isotope_order):
        pair_ids = np.flatnonzero(pairs.isotope_indices_m == isotope_index)
        if pair_ids.size == 0:
            continue
        atlas = estimator.filters[isotope]._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Conditional diagnostics require a surface atlas.")
        chart_faces = np.asarray(atlas.geometry.face_ids, dtype=object)
        pair_faces = pairs.face_ids_m2[pair_ids]
        membership = (chart_faces[None, :] == pair_faces[:, :1]) | (
            chart_faces[None, :] == pair_faces[:, 1:]
        )
        local_pairs, chart_ids = np.nonzero(membership)
        selected_pairs = pair_ids[local_pairs]
        surface_uv = np.full((chart_ids.size, 2), 0.5, dtype=np.float64)
        pair_parts.append(selected_pairs)
        chart_parts.append(chart_ids.astype(np.int64, copy=False))
        uv_parts.append(surface_uv)
        position_parts.append(
            np.asarray(atlas.positions_xyz(chart_ids, surface_uv), dtype=np.float64)
        )
        exact_pairs = np.repeat(pair_ids, 2)
        exact_charts = pairs.chart_ids_m2[pair_ids].reshape(-1)
        exact_uv = pairs.surface_uv_m22[pair_ids].reshape(-1, 2)
        pair_parts.append(exact_pairs)
        chart_parts.append(exact_charts)
        uv_parts.append(exact_uv)
        position_parts.append(
            np.asarray(atlas.positions_xyz(exact_charts, exact_uv), dtype=np.float64)
        )
    return _PositionBatch(
        pair_indices_r=np.concatenate(pair_parts),
        chart_ids_r=np.concatenate(chart_parts),
        surface_uv_r2=np.concatenate(uv_parts, axis=0),
        positions_r3=np.concatenate(position_parts, axis=0),
    )


def _local_position_batch(
    estimator: Any,
    state: _RepresentativeState,
    pairs: _PairBatch,
    best_chart_ids_m: NDArray[np.int64],
    *,
    grid_size: int,
) -> _PositionBatch:
    """Return one deterministic UV refinement grid on each coarse-best chart."""
    axis = (np.arange(grid_size, dtype=np.float64) + 0.5) / float(grid_size)
    u_values, v_values = np.meshgrid(axis, axis, indexing="ij")
    template_uv = np.column_stack((u_values.reshape(-1), v_values.reshape(-1)))
    grid_count = int(template_uv.shape[0])
    pair_indices = np.repeat(np.arange(pairs.count, dtype=np.int64), grid_count)
    chart_ids = np.repeat(
        np.asarray(best_chart_ids_m, dtype=np.int64),
        grid_count,
    )
    surface_uv = np.tile(template_uv, (pairs.count, 1))
    positions = np.empty((pair_indices.size, 3), dtype=np.float64)
    for isotope_index, isotope in enumerate(state.isotope_order):
        rows = np.flatnonzero(pairs.isotope_indices_m[pair_indices] == isotope_index)
        if rows.size == 0:
            continue
        atlas = estimator.filters[isotope]._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Conditional diagnostics require a surface atlas.")
        positions[rows] = atlas.positions_xyz(
            chart_ids[rows],
            surface_uv[rows],
        )
    return _PositionBatch(
        pair_indices_r=pair_indices,
        chart_ids_r=chart_ids,
        surface_uv_r2=surface_uv,
        positions_r3=positions,
    )


def _strength_grids(
    estimator: Any,
    pairs: _PairBatch,
    *,
    grid_size: int,
) -> NDArray[np.float64]:
    """Return fixed-size per-pair conditional strength-search grids."""
    prior = estimator.pf_config.build_strength_prior()
    minimum = float(prior.minimum)
    prior_upper = float(
        prior.finite_upper_quantile(STRENGTH_PROPOSAL_UPPER_QUANTILE_PROBABILITY)
    )
    pair_sums = np.sum(pairs.strengths_m2, axis=1, dtype=np.float64)
    upper = np.maximum(prior_upper, 2.0 * pair_sums)
    support_maximum = float(prior.support_maximum)
    if np.isfinite(support_maximum):
        upper = np.minimum(upper, support_maximum)
    if np.any(~np.isfinite(upper)) or np.any(upper < minimum):
        raise RuntimeError("Conditional diagnostic strength support is invalid.")
    fractions = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    regular = minimum + (upper - minimum)[:, None] * fractions[None, :]
    exact_sum = np.minimum(np.maximum(pair_sums, minimum), upper)
    return np.concatenate((regular, exact_sum[:, None]), axis=1)


def _global_unit_components(
    estimator: Any,
    state: _RepresentativeState,
    data: Any,
    *,
    isotope_indices_r: NDArray[np.int64],
    positions_r3: NDArray[np.float64],
    chart_ids_r: NDArray[np.int64],
) -> _UnitComponents:
    """Evaluate arbitrary source positions in one batched pass per isotope."""
    isotope_indices = np.asarray(isotope_indices_r, dtype=np.int64).reshape(-1)
    positions = np.asarray(positions_r3, dtype=np.float64).reshape(-1, 3)
    chart_ids = np.asarray(chart_ids_r, dtype=np.int64).reshape(-1)
    if (
        isotope_indices.size != positions.shape[0]
        or chart_ids.size != positions.shape[0]
    ):
        raise ValueError("Unit-component source arrays are misaligned.")
    model = estimator._full_spectrum_model()
    line_count = len(tuple(model.line_identity))
    feature_count = len(tuple(model.transport_feature_order))
    view_count = int(data.row_count)
    total = np.zeros(
        (positions.shape[0], view_count, line_count),
        dtype=np.float64,
    )
    uncollided = np.zeros_like(total)
    features = np.zeros(
        (positions.shape[0], view_count, line_count, feature_count),
        dtype=np.float64,
    )
    layout = estimator._joint_line_layout()
    for isotope_index, isotope in enumerate(state.isotope_order):
        rows = np.flatnonzero(isotope_indices == isotope_index)
        if rows.size == 0:
            continue
        filt = estimator.filters[isotope]
        global_columns, local_indices, branching_weights = layout[isotope]
        components = estimator._joint_cached_continuous_unit_components(
            filt=filt,
            data=data,
            positions_s3=positions[rows],
            chart_ids_s=chart_ids[rows],
            positive_line_indices=local_indices,
        )
        local_total = np.transpose(components[0], (1, 0, 2))
        local_uncollided = np.transpose(components[1], (1, 0, 2))
        local_features = np.transpose(
            np.stack(components[2:], axis=-1),
            (1, 0, 2, 3),
        )
        expected_shape = (rows.size, view_count, int(local_indices.size))
        if (
            local_total.shape != expected_shape
            or local_uncollided.shape != expected_shape
            or local_features.shape != expected_shape + (feature_count,)
        ):
            raise RuntimeError("Conditional unit transport has an invalid shape.")
        total_block = np.zeros(
            (rows.size, view_count, line_count),
            dtype=np.float64,
        )
        uncollided_block = np.zeros_like(total_block)
        feature_block = np.zeros(
            (rows.size, view_count, line_count, feature_count),
            dtype=np.float64,
        )
        total_block[..., global_columns] = (
            local_total * branching_weights[None, None, :]
        )
        uncollided_block[..., global_columns] = (
            local_uncollided * branching_weights[None, None, :]
        )
        feature_block[..., global_columns, :] = local_features
        total[rows] = total_block
        uncollided[rows] = uncollided_block
        features[rows] = feature_block
    if (
        np.any(~np.isfinite(total))
        or np.any(~np.isfinite(uncollided))
        or np.any(~np.isfinite(features))
        or np.any(total < 0.0)
        or np.any(uncollided < 0.0)
    ):
        raise RuntimeError("Conditional unit transport is invalid.")
    return _UnitComponents(
        total_rvl=total,
        uncollided_rvl=uncollided,
        features_rvlf=features,
    )


def _reference_components(
    estimator: Any,
    state: _RepresentativeState,
    data: Any,
) -> _ReferenceComponents:
    """Return source-resolved components for the posterior representative."""
    units = _global_unit_components(
        estimator,
        state,
        data,
        isotope_indices_r=state.isotope_indices_s,
        positions_r3=state.positions_s3,
        chart_ids_r=state.chart_ids_s,
    )
    scale = state.strengths_s[:, None, None]
    return _ReferenceComponents(
        total_1vsl=(units.total_rvl * scale).transpose(1, 0, 2)[None, ...],
        uncollided_1vsl=(units.uncollided_rvl * scale).transpose(1, 0, 2)[None, ...],
        features_1vslf=units.features_rvlf.transpose(1, 0, 2, 3)[None, ...],
    )


def _candidate_rows(
    positions: _PositionBatch,
    strength_grids_mg: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Expand ragged positions against fixed-size strength grids in batch."""
    grid = np.asarray(strength_grids_mg, dtype=np.float64)
    strength_count = int(grid.shape[1])
    position_indices = np.repeat(
        np.arange(positions.count, dtype=np.int64),
        strength_count,
    )
    pair_indices = positions.pair_indices_r[position_indices]
    strength_columns = np.tile(
        np.arange(strength_count, dtype=np.int64),
        positions.count,
    )
    strengths = grid[pair_indices, strength_columns]
    return pair_indices, position_indices, strengths


def _candidate_components(
    reference: _ReferenceComponents,
    pairs: _PairBatch,
    position_units: _UnitComponents,
    *,
    candidate_pair_indices_n: NDArray[np.int64],
    candidate_position_indices_n: NDArray[np.int64],
    candidate_strengths_n: NDArray[np.float64],
) -> _ReferenceComponents:
    """Replace each selected two-source block with one candidate source."""
    pair_indices = np.asarray(candidate_pair_indices_n, dtype=np.int64).reshape(-1)
    position_indices = np.asarray(
        candidate_position_indices_n,
        dtype=np.int64,
    ).reshape(-1)
    strengths = np.asarray(candidate_strengths_n, dtype=np.float64).reshape(-1)
    row_count = int(pair_indices.size)
    if (
        position_indices.size != row_count
        or strengths.size != row_count
        or np.any(pair_indices < 0)
        or np.any(pair_indices >= pairs.count)
        or np.any(position_indices < 0)
        or np.any(position_indices >= position_units.total_rvl.shape[0])
        or np.any(~np.isfinite(strengths))
        or np.any(strengths <= 0.0)
    ):
        raise ValueError("Conditional candidate rows are invalid.")
    output_shape = (row_count, *reference.total_1vsl.shape[1:])
    total = np.broadcast_to(reference.total_1vsl, output_shape).copy()
    uncollided = np.broadcast_to(
        reference.uncollided_1vsl,
        output_shape,
    ).copy()
    feature_shape = (row_count, *reference.features_1vslf.shape[1:])
    features = np.broadcast_to(reference.features_1vslf, feature_shape).copy()
    rows = np.arange(row_count, dtype=np.int64)
    slots = pairs.source_slots_m2[pair_indices]
    first_slots = slots[:, 0]
    second_slots = slots[:, 1]
    total[rows, :, first_slots, :] = 0.0
    total[rows, :, second_slots, :] = 0.0
    uncollided[rows, :, first_slots, :] = 0.0
    uncollided[rows, :, second_slots, :] = 0.0
    features[rows, :, first_slots, :, :] = 0.0
    features[rows, :, second_slots, :, :] = 0.0
    scale = strengths[:, None, None]
    total[rows, :, first_slots, :] = position_units.total_rvl[position_indices] * scale
    uncollided[rows, :, first_slots, :] = (
        position_units.uncollided_rvl[position_indices] * scale
    )
    features[rows, :, first_slots, :, :] = position_units.features_rvlf[
        position_indices
    ]
    return _ReferenceComponents(
        total_1vsl=total,
        uncollided_1vsl=uncollided,
        features_1vslf=features,
    )


def _score_candidate_rows(
    estimator: Any,
    stations: Sequence[Any],
    reference: _ReferenceComponents,
    pairs: _PairBatch,
    position_units: _UnitComponents,
    *,
    candidate_pair_indices_n: NDArray[np.int64],
    candidate_position_indices_n: NDArray[np.int64],
    candidate_strengths_n: NDArray[np.float64],
    batch_size: int,
) -> NDArray[np.float64]:
    """Score every conditional candidate in bounded vectorized batches."""
    pair_indices = np.asarray(candidate_pair_indices_n, dtype=np.int64).reshape(-1)
    position_indices = np.asarray(
        candidate_position_indices_n,
        dtype=np.int64,
    ).reshape(-1)
    strengths = np.asarray(candidate_strengths_n, dtype=np.float64).reshape(-1)
    if batch_size < 1:
        raise ValueError("Conditional diagnostic batch_size must be positive.")
    result = np.empty(pair_indices.size, dtype=np.float64)
    reference_filter = estimator.filters[estimator.joint_isotope_order()[0]]
    for start in range(0, pair_indices.size, batch_size):
        stop = min(start + batch_size, pair_indices.size)
        components = _candidate_components(
            reference,
            pairs,
            position_units,
            candidate_pair_indices_n=pair_indices[start:stop],
            candidate_position_indices_n=position_indices[start:stop],
            candidate_strengths_n=strengths[start:stop],
        )
        result[start:stop] = estimator._joint_history_log_likelihood_numpy(
            filt=reference_filter,
            stations=stations,
            total_nvsl=components.total_1vsl,
            uncollided_nvsl=components.uncollided_1vsl,
            features_nvslf=components.features_1vslf,
            target_beta=1.0,
        )
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise RuntimeError("Conditional source-count scores are invalid.")
    return result


def _group_argmax(
    values_n: NDArray[np.float64],
    groups_n: NDArray[np.int64],
    *,
    group_count: int,
) -> NDArray[np.int64]:
    """Return one stable maximum-value row index for every integer group."""
    values = np.asarray(values_n, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups_n, dtype=np.int64).reshape(-1)
    if (
        values.size != groups.size
        or np.any(np.isnan(values))
        or np.any(groups < 0)
        or np.any(groups >= group_count)
    ):
        raise ValueError("Grouped diagnostic scores are invalid.")
    row_indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((row_indices, -values, groups))
    ordered_groups = groups[order]
    first = np.ones(order.size, dtype=np.bool_)
    first[1:] = ordered_groups[1:] != ordered_groups[:-1]
    selected = order[first]
    if selected.size != group_count or not np.array_equal(
        groups[selected],
        np.arange(group_count, dtype=np.int64),
    ):
        raise RuntimeError("Every diagnostic pair must have candidate support.")
    return selected


def _predict_means(
    estimator: Any,
    stations: Sequence[Any],
    components: _ReferenceComponents,
) -> NDArray[np.float64]:
    """Return exact model means for all candidates and acquired views."""
    model = estimator._full_spectrum_model()
    live_times = np.concatenate(
        [np.asarray(station.live_times_s, dtype=np.float64) for station in stations]
    )
    means = np.asarray(
        model.predict_mean_numpy(
            components.total_1vsl,
            components.uncollided_1vsl,
            components.features_1vslf,
            live_times,
        ),
        dtype=np.float64,
    )
    expected_shape = (
        int(components.total_1vsl.shape[0]),
        int(live_times.size),
        int(np.asarray(model.energy_axis_keV).size),
    )
    if means.shape != expected_shape or np.any(~np.isfinite(means)):
        raise RuntimeError("Conditional predictive means have an invalid shape.")
    return means


def _residual_metrics(
    observed_vb: NDArray[np.float64],
    predicted_nvb: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Return vectorized raw full-spectrum residual summaries per model/view."""
    observed = np.asarray(observed_vb, dtype=np.float64)
    predicted = np.asarray(predicted_nvb, dtype=np.float64)
    if predicted.ndim != 3 or observed.shape != predicted.shape[1:]:
        raise ValueError("Residual observations and predictions are misaligned.")
    residual = observed[None, :, :] - predicted
    observed_total = np.sum(observed, axis=1, dtype=np.float64)
    spectral_l1 = np.sum(np.abs(residual), axis=2, dtype=np.float64)
    return {
        "signed_total_count_residual": np.sum(
            residual,
            axis=2,
            dtype=np.float64,
        ),
        "spectral_l1_count_residual": spectral_l1,
        "spectral_l1_fraction_of_observed": spectral_l1
        / np.maximum(observed_total[None, :], 1.0),
        "spectral_rmse_count_per_bin": np.sqrt(np.mean(np.square(residual), axis=2)),
        "maximum_abs_bin_residual_count": np.max(np.abs(residual), axis=2),
    }


def _overall_residual_summary(
    observed_vb: NDArray[np.float64],
    predicted_vb: NDArray[np.float64],
) -> dict[str, float]:
    """Return one raw full-spectrum residual summary across all views/bins."""
    residual = np.asarray(observed_vb, dtype=np.float64) - np.asarray(
        predicted_vb,
        dtype=np.float64,
    )
    return {
        "signed_total_count_residual": float(np.sum(residual)),
        "spectral_l1_count_residual": float(np.sum(np.abs(residual))),
        "spectral_l1_fraction_of_observed": float(
            np.sum(np.abs(residual))
            / max(float(np.sum(observed_vb, dtype=np.float64)), 1.0)
        ),
        "spectral_rmse_count_per_bin": float(np.sqrt(np.mean(np.square(residual)))),
        "maximum_abs_bin_residual_count": float(np.max(np.abs(residual))),
    }


def _figure_energy_metadata(stations: Sequence[Any]) -> dict[str, object]:
    """Return validated energy-bin coordinates and model provenance for plots."""
    axis = np.asarray(stations[0].energy_axis_keV, dtype=np.float64)
    contract_hash = stations[0].generative_contract_hash_sha256
    if (
        axis.ndim != 1
        or axis.size < 2
        or np.any(~np.isfinite(axis))
        or not isinstance(contract_hash, str)
        or not contract_hash
    ):
        raise RuntimeError("Diagnostic energy metadata is invalid.")
    widths = np.diff(axis)
    bin_width = float(widths[0])
    if bin_width <= 0.0 or not np.allclose(
        widths,
        bin_width,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Diagnostic energy bins must be uniformly spaced.")
    for station in stations[1:]:
        if (
            station.generative_contract_hash_sha256 != contract_hash
            or not np.array_equal(
                np.asarray(station.energy_axis_keV, dtype=np.float64),
                axis,
            )
        ):
            raise RuntimeError(
                "Diagnostic stations do not share one energy/model contract."
            )
    edges = np.concatenate(
        (axis, np.asarray([axis[-1] + bin_width], dtype=np.float64))
    )
    return {
        "generative_contract_hash_sha256": contract_hash,
        "energy_axis_keV": axis.tolist(),
        "energy_bin_left_edges_keV": axis.tolist(),
        "energy_bin_edges_keV": edges.tolist(),
        "energy_bin_width_keV": bin_width,
    }


def conditional_source_count_residual_diagnostics(
    estimator: Any,
    *,
    maximum_surface_distance_m: float = 2.0,
    strength_grid_size: int = 33,
    local_uv_grid_size: int = 5,
    candidate_batch_size: int | None = None,
) -> dict[str, object]:
    """Compare conditional one-source fits with reported two-component states.

    Candidate pairs, surface support, observations, and all fixed source states
    come only from the PF and its authenticated MeasurementLog history. The
    routine is a terminal reporting diagnostic: it does not alter particles,
    weights, proposal probabilities, stopping, planning, or the PF target.
    """
    maximum_distance = float(maximum_surface_distance_m)
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError("maximum_surface_distance_m must be positive and finite.")
    if (
        isinstance(strength_grid_size, (bool, np.bool_))
        or not isinstance(strength_grid_size, Integral)
        or int(strength_grid_size) < 2
    ):
        raise ValueError("strength_grid_size must be at least two.")
    if (
        isinstance(local_uv_grid_size, (bool, np.bool_))
        or not isinstance(local_uv_grid_size, Integral)
        or int(local_uv_grid_size) < 2
    ):
        raise ValueError("local_uv_grid_size must be at least two.")
    if candidate_batch_size is not None and (
        isinstance(candidate_batch_size, (bool, np.bool_))
        or not isinstance(candidate_batch_size, Integral)
        or int(candidate_batch_size) < 1
    ):
        raise ValueError("candidate_batch_size must be a positive integer.")
    stations = tuple(getattr(estimator, "_joint_station_history", ()))
    if not stations:
        return {
            "available": False,
            "reason": "no_assimilated_station_history",
            "truth_used": False,
            "changes_inference": False,
            "candidate_pairs": [],
        }
    energy_metadata = _figure_energy_metadata(stations)
    state = _representative_state(estimator)
    pairs = _eligible_pairs(
        estimator,
        state,
        maximum_surface_distance_m=maximum_distance,
    )
    batch_size = (
        int(estimator.pf_config.joint_strength_block_batch_size)
        if candidate_batch_size is None
        else int(candidate_batch_size)
    )
    base_payload: dict[str, object] = {
        "available": True,
        "truth_used": False,
        "changes_inference": False,
        **energy_metadata,
        "pair_selection": {
            "same_or_physically_adjacent_face": True,
            "maximum_surface_path_distance_m": maximum_distance,
        },
        "fit_semantics": (
            "maximum_full_history_likelihood_over_union_face_chart_centers_"
            "plus_local_uv_refinement_and_strength_grid_with_all_other_"
            "posterior_representative_sources_fixed"
        ),
        "residual_semantics": (
            "raw_observed_minus_model_mean_over_every_analysis_spectrum_bin"
        ),
        "comparison_role": ("diagnostic_only_not_model_evidence_or_an_accuracy_gate"),
        "search_parameters": {
            "regular_strength_grid_size": int(strength_grid_size),
            "current_pair_strength_sum_included": True,
            "local_uv_grid_size_per_axis": int(local_uv_grid_size),
            "candidate_batch_size": batch_size,
        },
        "candidate_pairs": [],
        "candidate_pair_count": int(pairs.count),
        "representative_source_count": int(state.strengths_s.size),
        "station_count": int(len(stations)),
        "view_count": int(sum(int(station.fe_indices.size) for station in stations)),
    }
    if pairs.count == 0:
        return base_payload
    base_payload["figure_reconstruction"] = {
        "schema_version": 1,
        "scope": "conditional_one_source_versus_two_component_diagnostic",
        "raw_observation_storage": (
            "two_component_reference.views[].observed_spectrum_count_by_bin"
        ),
        "two_component_prediction_storage": (
            "two_component_reference.views[].predicted_mean_count_by_bin"
        ),
        "one_source_prediction_storage": (
            "candidate_pairs[].views[].one_source.predicted_mean_count_by_bin"
        ),
        "energy_coordinate_semantics": "left_bin_edge_keV",
        "count_semantics": "unit_weight_detected_event_count_per_energy_bin",
        "predicted_mean_semantics": (
            "full_spectrum_generative_model_expected_count_per_energy_bin"
        ),
        "residual_formula": "observed_count_minus_predicted_mean_count",
        "view_join_key": [
            "station_sequence_id",
            "view_index",
        ],
        "transformations": {
            "energy_rebinning": "none",
            "count_normalization": "none",
            "smoothing": "none",
            "energy_bin_exclusion": "none",
            "view_exclusion": "none",
        },
        "missing_value_semantics": "no_missing_values",
        "randomness": "none_terminal_deterministic_diagnostic",
    }
    data = estimator._joint_history_structural_geometry(
        state.isotope_order[0],
        stations,
    )
    reference = _reference_components(estimator, state, data)
    coarse_positions = _coarse_position_batch(estimator, state, pairs)
    strength_grids = _strength_grids(
        estimator,
        pairs,
        grid_size=int(strength_grid_size),
    )
    coarse_units = _global_unit_components(
        estimator,
        state,
        data,
        isotope_indices_r=pairs.isotope_indices_m[coarse_positions.pair_indices_r],
        positions_r3=coarse_positions.positions_r3,
        chart_ids_r=coarse_positions.chart_ids_r,
    )
    coarse_pair_rows, coarse_position_rows, coarse_strengths = _candidate_rows(
        coarse_positions,
        strength_grids,
    )
    coarse_scores = _score_candidate_rows(
        estimator,
        stations,
        reference,
        pairs,
        coarse_units,
        candidate_pair_indices_n=coarse_pair_rows,
        candidate_position_indices_n=coarse_position_rows,
        candidate_strengths_n=coarse_strengths,
        batch_size=batch_size,
    )
    coarse_best_rows = _group_argmax(
        coarse_scores,
        coarse_pair_rows,
        group_count=pairs.count,
    )
    coarse_best_position_rows = coarse_position_rows[coarse_best_rows]
    local_positions = _local_position_batch(
        estimator,
        state,
        pairs,
        coarse_positions.chart_ids_r[coarse_best_position_rows],
        grid_size=int(local_uv_grid_size),
    )
    local_units = _global_unit_components(
        estimator,
        state,
        data,
        isotope_indices_r=pairs.isotope_indices_m[local_positions.pair_indices_r],
        positions_r3=local_positions.positions_r3,
        chart_ids_r=local_positions.chart_ids_r,
    )
    local_pair_rows, local_position_rows, local_strengths = _candidate_rows(
        local_positions,
        strength_grids,
    )
    local_scores = _score_candidate_rows(
        estimator,
        stations,
        reference,
        pairs,
        local_units,
        candidate_pair_indices_n=local_pair_rows,
        candidate_position_indices_n=local_position_rows,
        candidate_strengths_n=local_strengths,
        batch_size=batch_size,
    )
    local_best_rows = _group_argmax(
        local_scores,
        local_pair_rows,
        group_count=pairs.count,
    )
    use_local = local_scores[local_best_rows] > coarse_scores[coarse_best_rows]
    final_chart_ids = np.where(
        use_local,
        local_positions.chart_ids_r[local_position_rows[local_best_rows]],
        coarse_positions.chart_ids_r[coarse_position_rows[coarse_best_rows]],
    )
    final_surface_uv = np.where(
        use_local[:, None],
        local_positions.surface_uv_r2[local_position_rows[local_best_rows]],
        coarse_positions.surface_uv_r2[coarse_position_rows[coarse_best_rows]],
    )
    final_positions = np.where(
        use_local[:, None],
        local_positions.positions_r3[local_position_rows[local_best_rows]],
        coarse_positions.positions_r3[coarse_position_rows[coarse_best_rows]],
    )
    final_strengths = np.where(
        use_local,
        local_strengths[local_best_rows],
        coarse_strengths[coarse_best_rows],
    )
    final_scores = np.where(
        use_local,
        local_scores[local_best_rows],
        coarse_scores[coarse_best_rows],
    )
    final_position_batch = _PositionBatch(
        pair_indices_r=np.arange(pairs.count, dtype=np.int64),
        chart_ids_r=np.asarray(final_chart_ids, dtype=np.int64),
        surface_uv_r2=np.asarray(final_surface_uv, dtype=np.float64),
        positions_r3=np.asarray(final_positions, dtype=np.float64),
    )
    final_units = _global_unit_components(
        estimator,
        state,
        data,
        isotope_indices_r=pairs.isotope_indices_m,
        positions_r3=final_position_batch.positions_r3,
        chart_ids_r=final_position_batch.chart_ids_r,
    )
    final_components = _candidate_components(
        reference,
        pairs,
        final_units,
        candidate_pair_indices_n=np.arange(pairs.count, dtype=np.int64),
        candidate_position_indices_n=np.arange(pairs.count, dtype=np.int64),
        candidate_strengths_n=final_strengths,
    )
    reference_filter = estimator.filters[state.isotope_order[0]]
    reference_score = float(
        estimator._joint_history_log_likelihood_numpy(
            filt=reference_filter,
            stations=stations,
            total_nvsl=reference.total_1vsl,
            uncollided_nvsl=reference.uncollided_1vsl,
            features_nvslf=reference.features_1vslf,
            target_beta=1.0,
        )[0]
    )
    reference_mean = _predict_means(estimator, stations, reference)[0]
    final_means = _predict_means(estimator, stations, final_components)
    observed = np.concatenate(
        [np.asarray(station.spectrum_vb, dtype=np.float64) for station in stations],
        axis=0,
    )
    two_metrics = _residual_metrics(observed, reference_mean[None, ...])
    one_metrics = _residual_metrics(observed, final_means)
    two_residual = observed - reference_mean
    one_residual = observed[None, :, :] - final_means
    view_station_ids = np.concatenate(
        [
            np.full(
                int(station.fe_indices.size),
                int(station.station_sequence_id),
                dtype=np.int64,
            )
            for station in stations
        ]
    )
    view_local_indices = np.concatenate(
        [
            np.arange(int(station.fe_indices.size), dtype=np.int64)
            for station in stations
        ]
    )
    pose_indices = np.concatenate(
        [
            np.full(
                int(station.fe_indices.size),
                int(station.pose_idx),
                dtype=np.int64,
            )
            for station in stations
        ]
    )
    detector_positions = np.concatenate(
        [
            np.repeat(
                np.asarray(
                    station.detector_position_xyz_m,
                    dtype=np.float64,
                ).reshape(1, 3),
                int(station.fe_indices.size),
                axis=0,
            )
            for station in stations
        ],
        axis=0,
    )
    fe_indices = np.concatenate(
        [np.asarray(station.fe_indices, dtype=np.int64) for station in stations]
    )
    pb_indices = np.concatenate(
        [np.asarray(station.pb_indices, dtype=np.int64) for station in stations]
    )
    live_times = np.concatenate(
        [np.asarray(station.live_times_s, dtype=np.float64) for station in stations]
    )
    observed_totals = np.sum(observed, axis=1, dtype=np.float64)
    coarse_counts = np.bincount(
        coarse_positions.pair_indices_r,
        minlength=pairs.count,
    ) * int(strength_grids.shape[1])
    local_counts = np.bincount(
        local_positions.pair_indices_r,
        minlength=pairs.count,
    ) * int(strength_grids.shape[1])
    reference_view_rows: list[dict[str, object]] = []
    for view_index in range(observed.shape[0]):
        reference_view = {
            name: float(values[0, view_index]) for name, values in two_metrics.items()
        }
        reference_view["full_spectrum_residual_count_by_bin"] = two_residual[
            view_index
        ].tolist()
        reference_view_rows.append(
            {
                "station_sequence_id": int(view_station_ids[view_index]),
                "pose_idx": int(pose_indices[view_index]),
                "view_index": int(view_local_indices[view_index]),
                "detector_position_xyz_m": detector_positions[view_index].tolist(),
                "fe_orientation_index": int(fe_indices[view_index]),
                "pb_orientation_index": int(pb_indices[view_index]),
                "live_time_s": float(live_times[view_index]),
                "observed_total_count": float(observed_totals[view_index]),
                "observed_spectrum_count_by_bin": observed[
                    view_index
                ].astype(np.int64).tolist(),
                "predicted_mean_count_by_bin": reference_mean[
                    view_index
                ].tolist(),
                "residual": reference_view,
            }
        )
    base_payload["two_component_reference"] = {
        "reference_id": "posterior_representative",
        "full_history_log_likelihood": reference_score,
        "overall_residual": _overall_residual_summary(
            observed,
            reference_mean,
        ),
        "views": reference_view_rows,
    }
    pair_rows: list[dict[str, object]] = []
    # This bounded loop only serializes at most Kmax choose two summaries;
    # all transport, candidate scoring, spectrum prediction, and residual math
    # above are batched over pairs, positions, strengths, views, and bins.
    for pair_index in range(pairs.count):
        isotope = state.isotope_order[int(pairs.isotope_indices_m[pair_index])]
        view_rows: list[dict[str, object]] = []
        for view_index in range(observed.shape[0]):
            one_view = {
                name: float(values[pair_index, view_index])
                for name, values in one_metrics.items()
            }
            one_view["full_spectrum_residual_count_by_bin"] = one_residual[
                pair_index, view_index
            ].tolist()
            one_view["predicted_mean_count_by_bin"] = final_means[
                pair_index, view_index
            ].tolist()
            view_rows.append(
                {
                    "two_component_reference_view_index": int(view_index),
                    "station_sequence_id": int(view_station_ids[view_index]),
                    "view_index": int(view_local_indices[view_index]),
                    "one_source": one_view,
                    "one_minus_two_spectral_l1_count_residual": float(
                        one_view["spectral_l1_count_residual"]
                        - two_metrics["spectral_l1_count_residual"][
                            0,
                            view_index,
                        ]
                    ),
                }
            )
        pair_rows.append(
            {
                "isotope": isotope,
                "component_indices": [
                    int(value) for value in pairs.component_indices_m2[pair_index]
                ],
                "component_surface_chart_ids": [
                    int(value) for value in pairs.chart_ids_m2[pair_index]
                ],
                "component_face_ids": [
                    str(value) for value in pairs.face_ids_m2[pair_index]
                ],
                "component_surface_path_distance_m": float(
                    pairs.surface_distances_m[pair_index]
                ),
                "component_positions_xyz_m": pairs.positions_m23[pair_index].tolist(),
                "component_strengths_cps_1m": pairs.strengths_m2[pair_index].tolist(),
                "two_component_reference_id": "posterior_representative",
                "conditional_one_source_fit": {
                    "surface_chart_id": int(final_chart_ids[pair_index]),
                    "surface_uv": final_surface_uv[pair_index].tolist(),
                    "position_xyz_m": final_positions[pair_index].tolist(),
                    "strength_cps_1m": float(final_strengths[pair_index]),
                    "full_history_log_likelihood": float(final_scores[pair_index]),
                    "one_minus_two_full_history_log_likelihood": float(
                        final_scores[pair_index] - reference_score
                    ),
                    "candidate_state_count": int(
                        coarse_counts[pair_index] + local_counts[pair_index]
                    ),
                    "position_search": (
                        "union_face_chart_centers_plus_pair_positions_then_"
                        "coarse_best_chart_uv_grid"
                    ),
                    "strength_grid_count": int(strength_grids.shape[1]),
                    "strength_grid_minimum_cps_1m": float(
                        np.min(strength_grids[pair_index])
                    ),
                    "strength_grid_maximum_cps_1m": float(
                        np.max(strength_grids[pair_index])
                    ),
                    "overall_residual": _overall_residual_summary(
                        observed,
                        final_means[pair_index],
                    ),
                },
                "views": view_rows,
            }
        )
    base_payload["candidate_pairs"] = pair_rows
    return base_payload


__all__ = ["conditional_source_count_residual_diagnostics"]
