"""Batched forward-response conformance provider tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.observation_model import (
    build_nonproduction_observation_model,
    continuous_kernel_from_observation_model,
)
from runtime.forward_conformance import ForwardConformanceFixture
from pf.forward_response_conformance import (
    _REQUIRED_CASE_ORDER,
    evaluate_forward_response_fixture,
)


def _fixture() -> dict[str, Any]:
    """Return a small fixture spanning every batched physical axis."""
    return {
        "schema_version": 1,
        "required_case_order": list(_REQUIRED_CASE_ORDER),
        "units": {
            "distance": "m",
            "live_time": "s",
            "source_strength": "detector_cps_1m",
        },
        "isotopes": ["Cs-137", "Co-60"],
        "detector_poses": [
            {"pose_id": "east", "xyz": [2.0, 0.0, 0.5], "live_time_s": 2.0},
            {"pose_id": "north", "xyz": [0.0, 2.0, 1.0], "live_time_s": 3.0},
        ],
        "source_points": [
            {"source_id": "floor", "xyz": [0.0, 0.0, 0.0]},
            {"source_id": "wall", "xyz": [1.0, 1.0, 0.5]},
        ],
        "obstacles": [
            {"obstacle_id": "empty", "boxes": []},
            {
                "obstacle_id": "steel",
                "boxes": [
                    {
                        "min_xyz": [0.8, -0.2, -0.2],
                        "max_xyz": [1.2, 0.2, 1.2],
                        "material": "steel",
                    }
                ],
            },
        ],
        "shield_program": {
            "pairing": "cartesian_product",
            "fe_orientation_indices": [0, 3],
            "pb_orientation_indices": [1, 6],
        },
    }


def _scalar_response_oracle(payload: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the former scalar case ordering as a test-only oracle."""
    isotopes = tuple(str(value) for value in payload["isotopes"])
    fixture = ForwardConformanceFixture.from_payload(payload)
    model = build_nonproduction_observation_model(
        {
            "source_rate_model": "detector_cps_1m",
            "line_resolved_shield_attenuation": True,
        },
        isotopes=isotopes,
    )
    obstacles = fixture.obstacles
    kernels = {
        obstacle.obstacle_id: continuous_kernel_from_observation_model(
            model,
            obstacle_grid=fixture.obstacle_grid(obstacle.obstacle_id),
            use_gpu=False,
        )
        for obstacle in obstacles
    }
    program = payload["shield_program"]
    values: list[float] = []
    for isotope in isotopes:
        for pose in payload["detector_poses"]:
            for fe_index in program["fe_orientation_indices"]:
                for pb_index in program["pb_orientation_indices"]:
                    for source in payload["source_points"]:
                        for obstacle in obstacles:
                            values.append(
                                kernels[obstacle.obstacle_id].expected_counts_pair(
                                    isotope=isotope,
                                    detector_pos=np.asarray(pose["xyz"], dtype=np.float64),
                                    sources=np.asarray(
                                        source["xyz"],
                                        dtype=np.float64,
                                    ).reshape(1, 3),
                                    strengths=np.ones(1, dtype=np.float64),
                                    fe_index=int(fe_index),
                                    pb_index=int(pb_index),
                                    live_time_s=float(pose["live_time_s"]),
                                    background=0.0,
                                )
                            )
    return np.asarray(values, dtype=np.float64)


def test_batched_conformance_matches_scalar_physics_and_case_order() -> None:
    """The batched provider must reproduce every former scalar case."""
    payload = _fixture()
    expected_response = _scalar_response_oracle(payload)

    case_ids, actual_response = evaluate_forward_response_fixture(payload)

    assert case_ids.shape == expected_response.shape == actual_response.shape
    assert case_ids[0] == (
        "Cs-137|pose=east|fe=00|pb=01|source=floor|obstacle=empty"
    )
    assert case_ids[-1] == (
        "Co-60|pose=north|fe=03|pb=06|source=wall|obstacle=steel"
    )
    np.testing.assert_allclose(
        actual_response,
        expected_response,
        rtol=2.0e-14,
        atol=1.0e-15,
    )


def test_conformance_runtime_never_selects_scalar_pair_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every numeric pose/pair/source axis must use pair-program batches."""
    batch_calls = 0
    original = ContinuousKernel.line_transport_components_pair_program_for_detectors

    def counted_batch(self: ContinuousKernel, *args: object, **kwargs: object) -> object:
        """Count physical batch calls while preserving their exact output."""
        nonlocal batch_calls
        batch_calls += 1
        return original(self, *args, **kwargs)

    def reject_scalar(*_args: object, **_kwargs: object) -> float:
        """Fail if the deleted scalar conformance route is selected."""
        raise AssertionError("scalar expected_counts_pair route was selected")

    monkeypatch.setattr(
        ContinuousKernel,
        "line_transport_components_pair_program_for_detectors",
        counted_batch,
    )
    monkeypatch.setattr(ContinuousKernel, "expected_counts_pair", reject_scalar)

    evaluate_forward_response_fixture(_fixture())

    assert batch_calls == 4
