"""Generate deterministic PF forward-response conformance results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import (
    build_nonproduction_observation_model,
    continuous_kernel_from_observation_model,
)
from runtime.forward_conformance import (
    FORWARD_CONFORMANCE_CASE_ORDER,
    ForwardConformanceFixture,
    ForwardConformanceFixtureError,
)
from runtime.measurement_log import write_deterministic_npz


_REQUIRED_CASE_ORDER = FORWARD_CONFORMANCE_CASE_ORDER
ForwardResponseFixtureError = ForwardConformanceFixtureError


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
    payload: Mapping[str, object] | ForwardConformanceFixture,
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Evaluate cases through the explicit nonproduction conformance kernel."""
    fixture = (
        payload
        if isinstance(payload, ForwardConformanceFixture)
        else ForwardConformanceFixture.from_payload(payload)
    )
    isotopes = fixture.isotopes
    fe_indices = fixture.fe_orientation_indices
    pb_indices = fixture.pb_orientation_indices

    runtime_config = {
        "source_rate_model": "detector_cps_1m",
        "line_resolved_shield_attenuation": True,
    }
    observation_model = build_nonproduction_observation_model(
        runtime_config,
        isotopes=isotopes,
    )
    grids = {
        obstacle.obstacle_id: fixture.obstacle_grid(obstacle.obstacle_id)
        for obstacle in fixture.obstacles
    }
    kernels = {
        obstacle_id: continuous_kernel_from_observation_model(
            observation_model,
            obstacle_grid=grid,
            use_gpu=False,
        )
        for obstacle_id, grid in grids.items()
    }

    pose_ids = tuple(pose.pose_id for pose in fixture.detector_poses)
    detector_positions = np.asarray(
        [pose.xyz for pose in fixture.detector_poses],
        dtype=np.float64,
    )
    live_times = np.asarray(
        [pose.live_time_s for pose in fixture.detector_poses],
        dtype=np.float64,
    )
    source_ids = tuple(source.source_id for source in fixture.source_points)
    source_positions = np.asarray(
        [source.xyz for source in fixture.source_points],
        dtype=np.float64,
    )
    obstacle_ids = tuple(obstacle.obstacle_id for obstacle in fixture.obstacles)
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
    fixture = ForwardConformanceFixture.from_path(fixture_path)
    case_ids, response = evaluate_forward_response_fixture(fixture)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to replace conformance output {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary conformance output exists: {temporary}.")
    try:
        write_deterministic_npz(
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
