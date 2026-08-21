"""Generate deterministic PF forward-response conformance results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacle_assets import material_mu_cm_inv
from measurement.obstacles import ObstacleGrid
from runtime.measurement_log import _write_deterministic_npz


_REQUIRED_CASE_ORDER = (
    "isotope",
    "detector_pose",
    "fe_orientation",
    "pb_orientation",
    "source_point",
    "obstacle",
)


class ForwardResponseFixtureError(ValueError):
    """Report an invalid provider-neutral conformance fixture."""


def _object_list(payload: Mapping[str, Any], name: str) -> tuple[dict[str, Any], ...]:
    """Return a non-empty tuple of JSON objects."""
    raw = payload.get(name)
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, dict) for item in raw)
    ):
        raise ForwardResponseFixtureError(f"{name} must be a non-empty object list.")
    return tuple(dict(item) for item in raw)


def _xyz(payload: Mapping[str, Any], *, field: str, owner: str) -> NDArray[np.float64]:
    """Return one finite XYZ vector from a fixture object."""
    value = np.asarray(payload.get(field), dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ForwardResponseFixtureError(f"{owner}.{field} must be finite XYZ.")
    return value


def _obstacle_grid(
    obstacle: Mapping[str, Any],
    *,
    isotopes: Sequence[str],
) -> ObstacleGrid | None:
    """Translate fixture boxes into the existing ContinuousKernel transport grid."""
    raw_boxes = obstacle.get("boxes")
    if not isinstance(raw_boxes, list):
        raise ForwardResponseFixtureError("obstacles[].boxes must be a list.")
    if not raw_boxes:
        return None
    boxes: list[tuple[float, float, float, float, float, float]] = []
    materials: list[str] = []
    for index, raw_box in enumerate(raw_boxes):
        if not isinstance(raw_box, Mapping):
            raise ForwardResponseFixtureError(
                f"obstacles[].boxes[{index}] must be an object."
            )
        lower = _xyz(raw_box, field="min_xyz", owner=f"boxes[{index}]")
        upper = _xyz(raw_box, field="max_xyz", owner=f"boxes[{index}]")
        if np.any(upper < lower):
            raise ForwardResponseFixtureError(
                f"obstacles[].boxes[{index}] has inverted bounds."
            )
        boxes.append(tuple(float(value) for value in np.concatenate((lower, upper))))
        material = str(raw_box.get("material", "")).strip()
        if not material:
            raise ForwardResponseFixtureError(
                f"obstacles[].boxes[{index}].material is required."
            )
        materials.append(material)
    transport_mu = {
        str(isotope): tuple(
            float(material_mu_cm_inv(material, str(isotope))) for material in materials
        )
        for isotope in isotopes
    }
    return ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(0, 0),
        blocked_cells=(),
        transport_boxes_m=tuple(boxes),
        transport_mu_by_isotope=transport_mu,
        collision_boxes_m=tuple(boxes),
    )


def _detector_pose_table(
    poses: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], NDArray[np.float64], NDArray[np.float64]]:
    """Return validated pose IDs, detector positions, and live times."""
    pose_ids: list[str] = []
    detector_positions = np.empty((len(poses), 3), dtype=np.float64)
    live_times = np.empty(len(poses), dtype=np.float64)
    for index, pose in enumerate(poses):
        pose_id = str(pose.get("pose_id", "")).strip()
        live_time_s = float(pose.get("live_time_s", np.nan))
        if not pose_id or not np.isfinite(live_time_s) or live_time_s <= 0.0:
            raise ForwardResponseFixtureError(
                "Each detector pose needs a pose_id and positive live_time_s."
            )
        pose_ids.append(pose_id)
        detector_positions[index] = _xyz(
            pose,
            field="xyz",
            owner=f"pose={pose_id}",
        )
        live_times[index] = live_time_s
    return tuple(pose_ids), detector_positions, live_times


def _source_point_table(
    source_points: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Return validated source IDs and aligned continuous XYZ positions."""
    source_ids: list[str] = []
    positions = np.empty((len(source_points), 3), dtype=np.float64)
    for index, source in enumerate(source_points):
        source_id = str(source.get("source_id", "")).strip()
        if not source_id:
            raise ForwardResponseFixtureError(
                "Each source point needs a source_id."
            )
        source_ids.append(source_id)
        positions[index] = _xyz(
            source,
            field="xyz",
            owner=f"source={source_id}",
        )
    return tuple(source_ids), positions


def _case_ids_for_isotope(
    *,
    isotope: str,
    pose_ids: Sequence[str],
    fe_pair_indices: NDArray[np.int64],
    pb_pair_indices: NDArray[np.int64],
    source_ids: Sequence[str],
    obstacle_ids: Sequence[str],
) -> NDArray[np.str_]:
    """Materialize case metadata in the provider-neutral required order."""
    # This loop creates strings only. All physical pose, pair, source, line,
    # and obstacle-component transport is evaluated by batched kernels.
    return np.asarray(
        [
            f"{isotope}|pose={pose_id}|fe={int(fe_index):02d}|"
            f"pb={int(pb_index):02d}|source={source_id}|"
            f"obstacle={obstacle_id}"
            for pose_id in pose_ids
            for fe_index, pb_index in zip(
                fe_pair_indices,
                pb_pair_indices,
                strict=True,
            )
            for source_id in source_ids
            for obstacle_id in obstacle_ids
        ],
        dtype=np.str_,
    )


def evaluate_forward_response_fixture(
    payload: Mapping[str, Any],
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Evaluate all cases with the production PF observation model and kernel."""
    if int(payload.get("schema_version", -1)) != 1:
        raise ForwardResponseFixtureError("schema_version must be 1.")
    if tuple(payload.get("required_case_order", ())) != _REQUIRED_CASE_ORDER:
        raise ForwardResponseFixtureError(
            "required_case_order does not match the conformance-v1 contract."
        )
    units = payload.get("units")
    if units != {
        "distance": "m",
        "live_time": "s",
        "source_strength": "detector_cps_1m",
    }:
        raise ForwardResponseFixtureError("Fixture units are incompatible.")
    raw_isotopes = payload.get("isotopes")
    if not isinstance(raw_isotopes, list) or not raw_isotopes:
        raise ForwardResponseFixtureError("isotopes must be a non-empty list.")
    isotopes = tuple(str(value).strip() for value in raw_isotopes)
    if any(not value for value in isotopes) or len(set(isotopes)) != len(isotopes):
        raise ForwardResponseFixtureError("isotopes must be unique and non-empty.")
    poses = _object_list(payload, "detector_poses")
    source_points = _object_list(payload, "source_points")
    obstacles = _object_list(payload, "obstacles")
    shield_program = payload.get("shield_program")
    if not isinstance(shield_program, Mapping):
        raise ForwardResponseFixtureError("shield_program must be an object.")
    if shield_program.get("pairing") != "cartesian_product":
        raise ForwardResponseFixtureError(
            "shield_program.pairing must be cartesian_product."
        )
    fe_indices = tuple(
        int(value) for value in shield_program.get("fe_orientation_indices", ())
    )
    pb_indices = tuple(
        int(value) for value in shield_program.get("pb_orientation_indices", ())
    )
    if (
        not fe_indices
        or not pb_indices
        or any(value < 0 or value > 7 for value in (*fe_indices, *pb_indices))
    ):
        raise ForwardResponseFixtureError(
            "Fe/Pb indices must be non-empty and in 0..7."
        )

    runtime_config = {
        "source_rate_model": "detector_cps_1m",
        "pf_line_resolved_shield_attenuation": True,
    }
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=isotopes,
    )
    grids = {
        str(obstacle.get("obstacle_id", "")): _obstacle_grid(
            obstacle,
            isotopes=isotopes,
        )
        for obstacle in obstacles
    }
    if any(not key for key in grids) or len(grids) != len(obstacles):
        raise ForwardResponseFixtureError(
            "obstacle_id values must be unique/non-empty."
        )
    kernels = {
        obstacle_id: continuous_kernel_from_observation_model(
            observation_model,
            obstacle_grid=grid,
            use_gpu=False,
        )
        for obstacle_id, grid in grids.items()
    }

    pose_ids, detector_positions, live_times = _detector_pose_table(poses)
    source_ids, source_positions = _source_point_table(source_points)
    obstacle_ids = tuple(str(obstacle["obstacle_id"]) for obstacle in obstacles)
    fe_pair_indices = np.repeat(
        np.asarray(fe_indices, dtype=np.int64),
        len(pb_indices),
    )
    pb_pair_indices = np.tile(
        np.asarray(pb_indices, dtype=np.int64),
        len(fe_indices),
    )
    fe_program = np.broadcast_to(
        fe_pair_indices,
        (detector_positions.shape[0], fe_pair_indices.size),
    ).copy()
    pb_program = np.broadcast_to(
        pb_pair_indices,
        (detector_positions.shape[0], pb_pair_indices.size),
    ).copy()

    case_id_parts: list[NDArray[np.str_]] = []
    response_parts: list[NDArray[np.float64]] = []
    for isotope in isotopes:
        reference_kernel = kernels[obstacle_ids[0]]
        line_indices = reference_kernel.positive_line_indices(isotope)
        branching_weights = reference_kernel.line_branching_weights(
            isotope,
            line_indices,
        )
        obstacle_responses: list[NDArray[np.float64]] = []
        # Each fixture obstacle entry is an alternative complete environment,
        # not one obstacle component. ContinuousKernel batches every component
        # inside that environment across all poses, pairs, sources, and lines.
        for obstacle_id in obstacle_ids:
            kernel = kernels[obstacle_id]
            if not np.array_equal(
                kernel.line_branching_weights(isotope, line_indices),
                branching_weights,
            ):
                raise RuntimeError(
                    "Conformance obstacle variants changed isotope line weights."
                )
            components = kernel.line_transport_components_pair_program_for_detectors(
                isotope=isotope,
                detector_positions=detector_positions,
                sources=source_positions,
                fe_indices=fe_program,
                pb_indices=pb_program,
                positive_line_indices=line_indices,
            )
            total_kernel = np.asarray(
                components.total_kernel,
                dtype=np.float64,
            )
            expected_shape = (
                detector_positions.shape[0],
                fe_pair_indices.size,
                source_positions.shape[0],
                line_indices.size,
            )
            if total_kernel.shape != expected_shape:
                raise RuntimeError(
                    "Batched PF conformance transport returned an invalid shape."
                )
            response = np.einsum(
                "pvsl,l->pvs",
                total_kernel,
                branching_weights,
                optimize=True,
            ) * live_times[:, None, None]
            obstacle_responses.append(response)
        ordered_response = np.stack(obstacle_responses, axis=-1)
        if np.any(~np.isfinite(ordered_response)) or np.any(ordered_response < 0.0):
            raise RuntimeError(
                f"PF kernel returned an invalid response for isotope {isotope!r}."
            )
        case_id_parts.append(
            _case_ids_for_isotope(
                isotope=isotope,
                pose_ids=pose_ids,
                fe_pair_indices=fe_pair_indices,
                pb_pair_indices=pb_pair_indices,
                source_ids=source_ids,
                obstacle_ids=obstacle_ids,
            )
        )
        response_parts.append(
            np.ascontiguousarray(ordered_response.reshape(-1), dtype=np.float64)
        )
    return np.concatenate(case_id_parts), np.concatenate(response_parts)


def write_forward_response_conformance(
    fixture_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load a provider-neutral fixture and atomically publish deterministic NPZ."""
    fixture = Path(fixture_path)
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ForwardResponseFixtureError("Fixture root must be an object.")
    case_ids, response = evaluate_forward_response_fixture(payload)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to replace conformance output {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary conformance output exists: {temporary}.")
    try:
        _write_deterministic_npz(
            temporary,
            {"case_ids": case_ids, "unit_response": response},
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PF forward-response conformance adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    write_forward_response_conformance(args.fixture, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
