"""Tests for PF control over the estimator-neutral runtime protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from measurement.shielding import generate_octant_orientations

from pf.closed_loop import PFClosedLoopResult, PFControlBudget, run_pf_closed_loop
from planning.configuration import dss_config_from_pf_settings


def _context_payload() -> dict[str, object]:
    """Return one minimal truth-free adaptive runtime context."""
    return {
        "repository_commit": "a" * 40,
        "runtime_config": {
            "full_spectrum_contract_hash_sha256": "b" * 64,
        },
        "environment": {
            "size_x": 2.0,
            "size_y": 2.0,
            "size_z": 2.0,
            "detector_position": [0.5, 0.5, 0.5],
        },
        "sim_backend": "test",
        "spectrum_count_method": "joint_full_spectrum_generative",
        "isotopes": ["Cs-137"],
        "obstacle_layout_path": None,
        "source_rate_model": "detector_cps_1m",
        "metadata": {},
        "run_id": "pf-live-test",
        "source_rate_semantics": {},
        "forward_model_manifest": {},
        "runtime_config_sha256": "c" * 64,
        "schema_version": 2,
    }


class _FakeRuntimeClient:
    """Return one station while capturing every PF-selected action."""

    instance: "_FakeRuntimeClient | None" = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a deterministic one-pose runtime session."""
        del args, kwargs
        type(self).instance = self
        self.requests: list[dict[str, object]] = []
        self.overlay_requests: list[bool] = []
        self.candidates = {
            "candidate_poses_xyz": [[0.5, 0.5, 0.5]],
            "travel_costs": [0.0],
            "allowed_pair_ids": list(range(64)),
            "current_pair_id": 63,
        }

    def read_event(self) -> dict[str, object]:
        """Return a bootstrap pair that PF is not required to execute."""
        return {
            "type": "ready",
            "schema_version": 1,
            "context": _context_payload(),
            "candidates": self.candidates,
            "bootstrap": {
                "candidate_index": 0,
                "fe_orientation_index": 7,
                "pb_orientation_index": 7,
            },
        }

    def request(self, payload: dict[str, object]) -> dict[str, object]:
        """Return an exact integer raw spectrum for the chosen PF action."""
        self.requests.append(dict(payload))
        pair_id = int(payload["fe_orientation_index"]) * 8 + int(
            payload["pb_orientation_index"]
        )
        self.candidates["current_pair_id"] = pair_id
        return {
            "type": "record",
            "record": {
                "step_id": 0,
                "action_id": 0,
                "station_id": int(payload["station_id"]),
                "detector_pose_xyz": [0.5, 0.5, 0.5],
                "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "fe_orientation_index": int(payload["fe_orientation_index"]),
                "pb_orientation_index": int(payload["pb_orientation_index"]),
                "live_time_s": float(payload["dwell_time_s"]),
                "travel_time_s": 0.0,
                "shield_actuation_time_s": 0.0,
                "energy_bin_edges_keV": [0.0, 1.0, 2.0],
                "spectrum_counts": [2, 3],
                "metadata": {
                    "full_spectrum_contract_hash_sha256": "b" * 64,
                    "station_complete": bool(payload["station_complete"]),
                    "travel_waypoints_xyz": [
                        [0.25, 0.5, 0.5],
                        [0.5, 0.5, 0.5],
                    ],
                },
            },
            "candidates": self.candidates,
        }

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Return private truth for the CUI without capturing a PF action."""
        self.overlay_requests.append(bool(include_truth))
        return {
            "type": "cui_overlay",
            "schema_version": 1,
            "truth": {
                "schema_version": 1,
                "semantics": "evaluation_cui_overlay_only_not_estimator_input",
                "true_sources": {"Cs-137": [[1.0, 1.0, 1.0]]},
                "true_strengths": {"Cs-137": [300000.0]},
            },
        }

    def finalize(self) -> dict[str, object]:
        """Return the fake immutable log path."""
        return {
            "type": "published",
            "path": "/tmp/pf-live-log",
            "record_count": len(self.requests),
        }

    def abort(self) -> None:
        """Expose the runtime cleanup method."""


class _FakeEstimator:
    """Expose only PF operations needed for the one-station controller test."""

    def __init__(self) -> None:
        """Initialize one pose and an empty record history."""
        self.normals = np.asarray(generate_octant_orientations(), dtype=float)
        self.poses = [np.asarray([0.5, 0.5, 0.5])]
        self.measurements: list[object] = []
        self.kernel_cache = None
        self.pf_config = SimpleNamespace(
            num_particles=2000,
            target_ess_ratio=0.4,
        )

    def update_spectrum_station(
        self,
        records: tuple[object, ...],
        **kwargs: object,
    ) -> None:
        """Capture the raw station inputs supplied by the controller."""
        del kwargs
        self.measurements.extend(records)

    def step_diagnostics(self, **kwargs: object) -> dict[str, object]:
        """Return minimal particle-adequacy evidence."""
        del kwargs
        return {
            "Cs-137": {
                "particle_count": 2000,
                "current_ess": 1000.0,
                "current_ess_ratio": 0.5,
            }
        }

    def posterior_point_estimate(self) -> dict[str, SimpleNamespace]:
        """Return one truth-free point estimate for the station trace."""
        return {
            "Cs-137": SimpleNamespace(
                to_dict=lambda: {"map_cardinality": 1},
            )
        }


def test_pf_budget_requires_one_complete_estimator_station() -> None:
    """The runtime must never receive a truncated PF likelihood block."""
    settings = {
        "orientation_k": 8,
        "measurement_budget_max_steps": 7,
        "dss_pp": {
            "program_length": 8,
            "planning_method": "resample",
        },
    }
    planner = dss_config_from_pf_settings(
        settings,
        runtime_owned_candidates=True,
    )

    with pytest.raises(ValueError, match="complete station"):
        PFControlBudget.from_settings(settings, planner)


def test_pf_closed_loop_owns_budget_and_shield_program(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Runtime must execute PF choices without supplying a fixed action plan."""
    from pf import closed_loop

    config = tmp_path / "pf.json"
    config.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "mission_stop_max_poses": 1,
                "measurement_budget_max_steps": 1,
                "orientation_k": 1,
                "measurement_live_time_s": 30.0,
                "cui_split_view": False,
                "dss_pp": {
                    "program_length": 1,
                    "max_programs": 64,
                    "augment_candidates": False,
                },
            }
        ),
        encoding="utf-8",
    )
    estimator = _FakeEstimator()
    fake_log = SimpleNamespace(
        path=Path("/tmp/pf-live-log"),
        run_id="pf-live-test",
        records=(SimpleNamespace(station_id=0),),
    )
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(
        closed_loop,
        "build_live_estimator",
        lambda *args, **kwargs: estimator,
    )
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)
    monkeypatch.setattr(
        closed_loop,
        "bind_finalized_measurement_log",
        lambda estimator, log: None,
    )
    monkeypatch.setattr(
        closed_loop,
        "_write_final_outputs",
        lambda *args, **kwargs: None,
    )

    result = run_pf_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
    )

    client = _FakeRuntimeClient.instance
    assert isinstance(result, PFClosedLoopResult)
    assert client is not None
    assert len(client.requests) == 1
    assert "actions" not in client.requests[0]
    assert client.requests[0]["station_complete"] is True
    assert (
        int(client.requests[0]["fe_orientation_index"]) * 8
        + int(client.requests[0]["pb_orientation_index"])
        != 63
    )
    audit = json.loads(
        (tmp_path / "output" / "planner_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert audit["selection_mode"] == "pf_prior_balanced_bootstrap"
    assert audit["total_action_count"] == 0
    assert audit["selected_information_gain"] is None
    assert audit["mc_seed_rank_stability"]["status"] == (
        "not_applicable_before_first_observation"
    )
    assert len(estimator.measurements) == 1
    assert result.station_count == 1
    station_trace = json.loads(
        (tmp_path / "output" / "pf_station_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert station_trace["pf_update_elapsed_s"] >= 0.0
    assert (
        station_trace["posterior_snapshot"]["isotopes"]["Cs-137"]["map_cardinality"]
        == 1
    )
    assert station_trace["posterior_snapshot"]["publishable"] is False


def test_detected_isotope_gate_builds_only_active_pf(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A detected isotope must trigger a fresh active-only full-history PF."""
    from pf import closed_loop

    config = tmp_path / "pf.json"
    config.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "mission_stop_max_poses": 1,
                "measurement_budget_max_steps": 1,
                "orientation_k": 1,
                "measurement_live_time_s": 30.0,
                "cui_split_view": False,
                "pf_detected_isotopes_only": True,
                "detected_isotope_false_activation_probability": 0.001,
                "dss_pp": {
                    "program_length": 1,
                    "max_programs": 64,
                    "augment_candidates": False,
                },
            }
        ),
        encoding="utf-8",
    )

    class _DetectionEstimator(_FakeEstimator):
        """Return decisive truth-free full-spectrum scores without PF updates."""

        def full_spectrum_isotope_detection_score_grids(
            self,
            records: tuple[object, ...],
            **kwargs: object,
        ) -> dict[str, np.ndarray]:
            """Return one candidate score grid above the corrected threshold."""
            del records, kwargs
            return {"Cs-137": np.full((2, 5), 30.0, dtype=np.float64)}

    detector = _DetectionEstimator()
    active_estimator = _FakeEstimator()
    build_calls: list[dict[str, object]] = []

    def _build(*args: object, **kwargs: object) -> _FakeEstimator:
        """Return the detector first and the active PF after activation."""
        build_calls.append({"args": args, "kwargs": dict(kwargs)})
        return detector if len(build_calls) == 1 else active_estimator

    fake_log = SimpleNamespace(
        path=Path("/tmp/pf-live-log"),
        run_id="pf-live-test",
        records=(SimpleNamespace(station_id=0),),
    )
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "build_live_estimator", _build)
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)
    monkeypatch.setattr(
        closed_loop,
        "bind_finalized_measurement_log",
        lambda estimator, log: None,
    )
    monkeypatch.setattr(
        closed_loop,
        "_write_final_outputs",
        lambda *args, **kwargs: None,
    )

    run_pf_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
    )

    assert len(build_calls) == 2
    assert build_calls[0]["args"][1]["num_particles"] == 1
    assert build_calls[0]["args"][1]["variable_cardinality"] is False
    assert build_calls[0]["args"][1]["init_num_sources"] == (0, 0)
    assert build_calls[1]["kwargs"]["inference_isotopes"] == ("Cs-137",)
    assert detector.measurements == []
    assert len(active_estimator.measurements) == 1
    trace = json.loads(
        (tmp_path / "output" / "pf_station_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert trace["detected_isotope_gate"]["active_isotopes"] == ["Cs-137"]
    assert trace["detected_isotope_gate"]["truth_used"] is False


def test_pf_closed_loop_starts_truth_free_cui_and_publishes_frames(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The closed-loop entry point must route CUI settings to a sidecar."""
    from pf import closed_loop

    config = tmp_path / "pf.json"
    config.write_text(
        json.dumps(
            {
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
                "mission_stop_max_poses": 1,
                "measurement_budget_max_steps": 1,
                "orientation_k": 1,
                "measurement_live_time_s": 30.0,
                "cui_split_view": True,
                "cui_split_view_dir": (tmp_path / "cui").as_posix(),
                "cui_truth_display_mode": "evaluation_live",
                "dss_pp": {
                    "program_length": 1,
                    "max_programs": 64,
                    "augment_candidates": False,
                },
            }
        ),
        encoding="utf-8",
    )
    estimator = _FakeEstimator()
    fake_log = SimpleNamespace(
        path=Path("/tmp/pf-live-log"),
        run_id="pf-live-test",
        records=(SimpleNamespace(station_id=0),),
    )
    frames: list[object] = []
    truth_updates: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []

    class _FakeCUI:
        """Capture CUI frames without spawning a renderer process."""

        index_path = tmp_path / "cui" / "index.html"

        def __init__(self, **kwargs: object) -> None:
            """Record construction arguments for the CUI sidecar."""
            self.kwargs = kwargs

        def update(self, frame: object) -> None:
            """Record one CUI frame."""
            frames.append(frame)

        def set_truth(
            self,
            true_sources: dict[str, np.ndarray],
            true_strengths: dict[str, np.ndarray],
        ) -> None:
            """Record one CUI-only truth update."""
            truth_updates.append((true_sources, true_strengths))

        def close(self) -> None:
            """Provide the production CUI lifecycle interface."""

    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(
        closed_loop,
        "build_live_estimator",
        lambda *args, **kwargs: estimator,
    )
    monkeypatch.setattr(closed_loop, "load_measurement_log", lambda path: fake_log)
    monkeypatch.setattr(
        closed_loop,
        "bind_finalized_measurement_log",
        lambda estimator, log: None,
    )
    monkeypatch.setattr(
        closed_loop,
        "_write_final_outputs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(closed_loop, "AsyncCUISplitPFVisualizer", _FakeCUI)
    monkeypatch.setattr(
        closed_loop,
        "build_frame_from_pf",
        lambda *args, **kwargs: SimpleNamespace(record_measurement=True),
    )
    monkeypatch.setattr(
        closed_loop,
        "ensure_cui_view_server",
        lambda *args, **kwargs: "http://example.test:8877/index.html",
    )

    run_pf_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
    )

    assert len(frames) == 2
    assert frames[0].record_measurement is True
    assert frames[1].record_measurement is False
    assert np.asarray(frames[0].path_waypoints_xyz).shape == (2, 3)
    assert len(truth_updates) == 1
    assert truth_updates[0][0]["Cs-137"].shape == (1, 3)
    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.overlay_requests == [True]
