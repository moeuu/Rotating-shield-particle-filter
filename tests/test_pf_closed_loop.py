"""Tests for PF control over the estimator-neutral runtime protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from measurement.shielding import generate_octant_orientations
from runtime.adaptive_client import (
    AdaptiveCandidateSnapshot,
    AdaptiveCandidatesEvent,
    AdaptivePublishedEvent,
    AdaptiveReadyEvent,
    AdaptiveRecordEvent,
    AdaptiveRefineRequest,
    AdaptiveStepRequest,
)

from pf.closed_loop import (
    PFClosedLoopResult,
    PFControlBudget,
    _cui_truth_display_mode,
    run_pf_closed_loop,
)
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


@pytest.mark.parametrize("mode", ("evaluation_live", "post_run"))
def test_estimator_owned_cui_rejects_truth_modes(mode: str) -> None:
    """Realized truth must be rendered only by a separate evaluator."""
    with pytest.raises(ValueError, match="separate post-estimation evaluator"):
        _cui_truth_display_mode({"cui_truth_display_mode": mode})

    assert _cui_truth_display_mode({}) == "hidden"


class _FakeRuntimeClient:
    """Return one station while capturing every PF-selected action."""

    instance: "_FakeRuntimeClient | None" = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a deterministic one-pose runtime session."""
        del args, kwargs
        type(self).instance = self
        self.closed = False
        self.requests: list[dict[str, object]] = []
        self.refinement_requests: list[dict[str, object]] = []
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

    def read_ready_event(self) -> AdaptiveReadyEvent:
        """Return the typed form of the fake runtime handshake."""
        return AdaptiveReadyEvent.from_payload(self.read_event())

    def handshake(self) -> AdaptiveReadyEvent:
        """Return the handshake through the concise runtime API."""
        return self.read_ready_event()

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

    def request_step(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Return the typed form of one fake runtime record response."""
        return AdaptiveRecordEvent.from_payload(self.request(request.to_payload()))

    def acquire(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Acquire one record through the concise runtime API."""
        return self.request_step(request)

    def request_refinement(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Return unchanged typed candidates for a fake refinement request."""
        self.refinement_requests.append(request.to_payload())
        return AdaptiveCandidatesEvent.from_payload(
            {"type": "candidates", "candidates": self.candidates}
        )

    def refine_candidates(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Refine candidates through the concise runtime API."""
        return self.request_refinement(request)

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Fail if an estimator-owned controller requests realized truth."""
        self.overlay_requests.append(bool(include_truth))
        raise AssertionError("PF closed loop must not request a truth overlay.")

    def finalize(self) -> dict[str, object]:
        """Return the fake immutable log path."""
        return {
            "type": "published",
            "path": "/tmp/pf-live-log",
            "record_count": len(self.requests),
        }

    def finalize_event(self) -> AdaptivePublishedEvent:
        """Return the typed form of the fake publication response."""
        return AdaptivePublishedEvent.from_payload(self.finalize())

    def finalize_log(self) -> AdaptivePublishedEvent:
        """Finalize the log through the concise runtime API."""
        return self.finalize_event()

    def close(self) -> None:
        """Record deterministic client cleanup."""
        self.closed = True

    def abort(self) -> None:
        """Expose the runtime cleanup method."""


class _FakeResumeRuntimeClient(_FakeRuntimeClient):
    """Return one complete prefix and no new acquisition actions."""

    def read_event(self) -> dict[str, object]:
        """Return the schema-v2 resume handshake."""
        record = {
            "step_id": 0,
            "action_id": 0,
            "station_id": 0,
            "detector_pose_xyz": [0.5, 0.5, 0.5],
            "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
            "live_time_s": 30.0,
            "travel_time_s": 0.0,
            "shield_actuation_time_s": 0.0,
            "energy_bin_edges_keV": [0.0, 1.0, 2.0],
            "spectrum_counts": [2, 3],
            "metadata": {
                "full_spectrum_contract_hash_sha256": "b" * 64,
                "station_complete": True,
            },
        }
        return {
            "type": "ready",
            "schema_version": 2,
            "context": _context_payload(),
            "candidates": self.candidates,
            "resume": {
                "record_count": 1,
                "records": [record],
                "next_station_id": 1,
            },
        }

    def finalize(self) -> dict[str, object]:
        """Publish the unchanged one-record prefix."""
        return {
            "type": "published",
            "path": "/tmp/pf-live-log",
            "record_count": 1,
        }


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


class _FakeLog:
    """Expose finalized-log fields and the shared station-view surface."""

    path = Path("/tmp/pf-live-log")
    run_id = "pf-live-test"
    records = (SimpleNamespace(station_id=0),)

    def station_view(self) -> SimpleNamespace:
        """Return the station count used by closed-loop result publication."""
        return SimpleNamespace(station_count=1)


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


def test_closed_loop_applies_declared_passive_path_and_fixed_shield() -> None:
    """RA-L passive policy must bypass PF EIG for both action dimensions."""
    from pf import closed_loop

    estimator = SimpleNamespace(
        normals=np.asarray(generate_octant_orientations(), dtype=float)
    )
    settings = {
        "baseline_path_policy": {"name": "passive_serpentine", "row_count": 2},
        "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
        "dss_pp": {"program_length": 2},
    }
    planner = dss_config_from_pf_settings(settings, runtime_owned_candidates=True)
    candidates = AdaptiveCandidateSnapshot(
        candidate_poses_xyz=((0.0, 0.0, 0.5), (2.0, 2.0, 0.5)),
        travel_costs=(1.0, 1.0),
        allowed_pair_ids=tuple(range(64)),
        current_pair_id=63,
    )

    result = closed_loop._plan(
        estimator,
        candidates,
        current_pose=np.asarray([1.0, 1.0, 0.5]),
        visited_poses=[],
        obstacle_grid=None,
        room_bounds=(np.asarray([0.0, 0.0, 0.5]), np.asarray([2.0, 2.0, 0.5])),
        height_bounds=None,
        planner=planner,
        rng=np.random.default_rng(7),
        settings=settings,
        station_index=0,
    )

    assert result.next_pose_index == 0
    assert result.shield_program.pair_ids == (0, 0)
    assert result.sequence == ()
    assert result.diagnostics["selection_mode"] == "ral_baseline_path"


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
    fake_log = _FakeLog()
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
    assert client.closed is True
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


def test_pf_closed_loop_replays_runtime_resume_prefix(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Schema-v2 resume should rebuild PF state before final publication."""
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
                "dss_pp": {"program_length": 1, "max_programs": 64},
            }
        ),
        encoding="utf-8",
    )
    estimator = _FakeEstimator()
    fake_log = _FakeLog()
    redraws: list[tuple[int, bool, int]] = []

    class _ResumeCUI:
        """Provide the minimal lifecycle for a resume-only redraw."""

        def close(self) -> None:
            """Close the fake renderer."""

    resume_cui = _ResumeCUI()

    def capture_resume_frame(
        visualizer: object,
        estimator: object,
        record: object,
        route_records: list[object],
        *,
        elapsed_time_s: float,
        record_measurement: bool,
    ) -> None:
        """Record the posterior-only redraw of the resumed prefix."""
        del estimator, elapsed_time_s
        assert visualizer is resume_cui
        redraws.append((int(record.step_id), record_measurement, len(route_records)))

    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _FakeResumeRuntimeClient,
    )
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
    monkeypatch.setattr(
        closed_loop,
        "_start_cui_split_view",
        lambda *args, **kwargs: resume_cui,
    )
    monkeypatch.setattr(closed_loop, "_publish_cui_frame", capture_resume_frame)

    result = run_pf_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        pf_config_path=config,
        output_dir=tmp_path / "output",
        resume_stage_path=tmp_path / "stage",
    )

    client = _FakeResumeRuntimeClient.instance
    assert client is not None
    assert client.requests == []
    assert len(estimator.measurements) == 1
    assert result.record_count == 1
    assert result.station_count == 1
    assert result.stop_reason == "maximum_measurement_budget"
    assert redraws == [(0, False, 1)]


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

    fake_log = _FakeLog()
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
                "cui_truth_display_mode": "hidden",
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
    fake_log = _FakeLog()
    frames: list[object] = []
    truth_updates: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    output_messages: list[str] = []

    class _FakeCUI:
        """Capture CUI frames without spawning a renderer process."""

        index_path = tmp_path / "cui" / "index.html"

        def __init__(self, **kwargs: object) -> None:
            """Record construction arguments for the CUI sidecar."""
            self.kwargs = kwargs
            output_dir = Path(str(kwargs["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            self.latest_overview_path = (
                output_dir / "latest_experiment_overview.png"
            )
            self.latest_robot_path = output_dir / "latest_robot_2d.png"
            self.latest_pf_path = output_dir / "latest_pf_3d.png"
            self.latest_pf_labeled_path = output_dir / "latest_pf_3d_labeled.png"
            self.latest_spectrum_path = output_dir / "latest_spectrum.png"
            for path in (
                self.latest_overview_path,
                self.latest_robot_path,
                self.latest_pf_path,
                self.latest_pf_labeled_path,
                self.latest_spectrum_path,
            ):
                path.write_bytes(path.name.encode())

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
        output_hook=output_messages.append,
    )

    assert len(frames) == 2
    assert frames[0].record_measurement is True
    assert frames[1].record_measurement is False
    assert np.asarray(frames[0].path_waypoints_xyz).shape == (2, 3)
    np.testing.assert_array_equal(
        frames[0].cui_route.measurement_visit_counts,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        frames[1].cui_route.measurement_visit_counts,
        np.asarray([1], dtype=np.int64),
    )
    assert truth_updates == []
    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.overlay_requests == []
    assert (
        "CUI split visualization URL: "
        "http://example.test:8877/index.html"
    ) in output_messages
    enabled_message = next(
        message
        for message in output_messages
        if message.startswith("CUI split visualization enabled:")
    )
    for filename in (
        "latest_experiment_overview.png",
        "latest_robot_2d.png",
        "latest_pf_3d.png",
        "latest_pf_3d_labeled.png",
        "latest_spectrum.png",
    ):
        assert filename in enabled_message
    assert (tmp_path / "output" / "final_experiment_overview.png").is_file()
    assert (tmp_path / "output" / "final_robot_2d.png").is_file()
    assert (tmp_path / "output" / "final_pf_3d.png").is_file()
    assert (tmp_path / "output" / "final_pf_3d_labeled.png").is_file()
    assert (tmp_path / "output" / "final_spectrum.png").is_file()
