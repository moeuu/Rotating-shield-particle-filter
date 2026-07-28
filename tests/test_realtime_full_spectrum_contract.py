"""Live-ingestion tests for the sole raw full-spectrum PF contract."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from realtime_demo import (
    _acquire_spectrum_observation,
    _analysis_spectrum_array,
    _atomic_write_json,
    _full_spectrum_model_diagnostics,
    _render_optional_outputs_after_artifacts,
    _resolve_candidate_isotopes,
    _resolve_random_source_isotopes,
    run_live_pf,
)
from sim.protocol import SimulationObservation
from tests.pure_pf_test_support import approved_full_spectrum_model


def test_live_runtime_requires_an_explicit_simulation_backend() -> None:
    """Programmatic callers must never fall back silently to analytic transport."""
    with pytest.raises(ValueError, match="requires an explicit sim_backend"):
        run_live_pf()


def test_live_runtime_rejects_backend_config_mismatch(tmp_path: Path) -> None:
    """Programmatic callers cannot run analytic physics under a Geant4 label."""
    config_path = tmp_path / "analytic.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "analytic",
                "pure_pf_schema_version": 1,
                "estimator_profile": "pf_strict",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the resolved config"):
        run_live_pf(
            sim_backend="geant4",
            sim_config_path=config_path.as_posix(),
        )


def _native_observation() -> SimulationObservation:
    """Return one exact native-contract observation."""
    model = approved_full_spectrum_model()
    axis = model.energy_axis_keV
    width = float(axis[1] - axis[0])
    edges = np.concatenate((axis, [axis[-1] + width]))
    counts = np.zeros(axis.shape, dtype=np.int64)
    counts[:3] = np.asarray([7, 2, 1], dtype=np.int64)
    return SimulationObservation(
        step_id=0,
        detector_pose_xyz=(0.5, 0.5, 0.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=1,
        pb_orientation_index=2,
        spectrum_counts=counts.tolist(),
        energy_bin_edges_keV=edges.tolist(),
        metadata={
            "detector_scoring_mode": "incident_gamma_energy",
            "detector_response_sampling_mode": (
                "multinomial_marking_with_nonparalyzable_event_time"
            ),
            "detector_response_sampling_model": (
                "native_incident_gamma_response_v1"
            ),
            "detector_response_sampling_contract_sha256": (
                model.manifest_payload()[
                    "detector_response_contract_sha256"
                ]
            ),
            "intensity_cps_1m_definition": (
                "pre_dead_time_detector_pulse_rate_at_1m"
            ),
            "transport_history_mode": "full_unit_weight",
            "dead_time_tau_s": model.dead_time_tau_s,
            "background_cps": model.background_rate_cps,
            "fe_orientation_index": 1,
            "pb_orientation_index": 2,
            "shield_num_orientations": 8,
            "shield_pair_id": 10,
            "dwell_time_s": 30.0,
        },
    )


def test_optional_plot_failure_preserves_published_scientific_artifacts(
    tmp_path: Path,
) -> None:
    """A renderer failure must not remove the log, posterior, or summary."""
    measurement_log = tmp_path / "measurement_log"
    measurement_log.mkdir()
    posterior_path = tmp_path / "pf_posterior.json"
    summary_path = tmp_path / "result_summary.json"
    _atomic_write_json(posterior_path, {"schema_version": 1})
    _atomic_write_json(summary_path, {"measurements_completed": 160})
    renderer_observations: list[tuple[int, int]] = []

    def fail_plot() -> None:
        """Verify publication order, then emulate one Matplotlib failure."""
        posterior = json.loads(posterior_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        renderer_observations.append(
            (
                int(posterior["schema_version"]),
                int(summary["measurements_completed"]),
            )
        )
        raise RuntimeError("synthetic plot failure")

    failures = _render_optional_outputs_after_artifacts(
        required_artifacts=(
            measurement_log,
            posterior_path,
            summary_path,
        ),
        renderers=(("final_plot", fail_plot),),
    )

    assert renderer_observations == [(1, 160)]
    assert failures == (
        {
            "label": "final_plot",
            "error_type": "RuntimeError",
            "error": "synthetic plot failure",
        },
    )
    assert posterior_path.is_file()
    assert summary_path.is_file()
    assert measurement_log.is_dir()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_live_ingestion_accepts_only_exact_native_axis_and_event_counts() -> None:
    """A valid native histogram is returned byte-for-byte as int64."""
    model = approved_full_spectrum_model()
    observation = _native_observation()

    spectrum = _analysis_spectrum_array(
        observation,
        model,
        require_native_contract=True,
    )

    assert spectrum.dtype == np.int64
    np.testing.assert_array_equal(
        spectrum,
        np.asarray(observation.spectrum_counts, dtype=np.int64),
    )


@pytest.mark.parametrize("mutation", ("fractional", "shifted_axis", "metadata"))
def test_live_ingestion_fails_closed_on_contract_mismatch(mutation: str) -> None:
    """No truncation, fractional weighting, or stale native bridge is accepted."""
    model = approved_full_spectrum_model()
    observation = _native_observation()
    if mutation == "fractional":
        counts = list(observation.spectrum_counts)
        counts[0] = 0.5
        with pytest.raises(TypeError, match="integer event counts"):
            replace(observation, spectrum_counts=counts)
        return
    elif mutation == "shifted_axis":
        observation = replace(
            observation,
            energy_bin_edges_keV=[
                value + 0.25 for value in observation.energy_bin_edges_keV
            ],
        )
    else:
        metadata = dict(observation.metadata)
        metadata.pop("detector_response_sampling_contract_sha256")
        observation = replace(observation, metadata=metadata)

    with pytest.raises(ValueError):
        _analysis_spectrum_array(
            observation,
            model,
            require_native_contract=True,
        )


class _Runtime:
    """Return one fixed observation and record submitted commands."""

    def __init__(self, observation: SimulationObservation) -> None:
        """Store the output observation."""
        self.observation = observation
        self.commands: list[object] = []

    def step(self, command: object) -> SimulationObservation:
        """Record and return one observation."""
        self.commands.append(command)
        return self.observation


def test_acquisition_uses_one_fixed_dwell_without_count_based_extension() -> None:
    """One action performs exactly one transport call for the requested dwell."""
    model = approved_full_spectrum_model()
    runtime = _Runtime(_native_observation())

    _, live_time_s, spectrum, reason, chunk_count = (
        _acquire_spectrum_observation(
            simulation_runtime=runtime,
            full_spectrum_model=model,
            step_id=0,
            pose_xyz=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
            fe_idx=1,
            pb_idx=2,
            live_time_s=30.0,
            travel_time_s=0.0,
            shield_actuation_time_s=0.0,
            require_native_contract=True,
        )
    )

    assert len(runtime.commands) == 1
    assert runtime.commands[0].dwell_time_s == 30.0
    assert live_time_s == 30.0
    assert reason == "fixed_dwell"
    assert chunk_count == 1
    assert spectrum.dtype == np.int64


@pytest.mark.parametrize(
    "mutation",
    (
        "step_id",
        "top_level_pose",
        "top_level_quaternion",
        "top_level_orientation",
        "metadata_orientation_mismatch",
        "metadata_orientation_string",
        "metadata_orientation_fractional",
        "metadata_orientation_count",
        "metadata_pair",
        "metadata_dwell_mismatch",
        "metadata_dwell_string",
    ),
)
def test_acquisition_rejects_stale_or_coerced_action_binding(
    mutation: str,
) -> None:
    """A returned spectrum must authenticate the exact requested action row."""
    model = approved_full_spectrum_model()
    observation = _native_observation()
    if mutation == "step_id":
        observation = replace(observation, step_id=1)
    elif mutation == "top_level_pose":
        observation = replace(
            observation,
            detector_pose_xyz=(0.5, 0.5, 0.75),
        )
    elif mutation == "top_level_quaternion":
        observation = replace(
            observation,
            detector_quat_wxyz=(0.0, 0.0, 0.0, 1.0),
        )
    elif mutation == "top_level_orientation":
        observation = replace(observation, pb_orientation_index=3)
    else:
        metadata = dict(observation.metadata)
        if mutation == "metadata_orientation_mismatch":
            metadata["fe_orientation_index"] = 0
        elif mutation == "metadata_orientation_string":
            metadata["fe_orientation_index"] = "1"
        elif mutation == "metadata_orientation_fractional":
            metadata["pb_orientation_index"] = 2.5
        elif mutation == "metadata_orientation_count":
            metadata["shield_num_orientations"] = 7
        elif mutation == "metadata_pair":
            metadata["shield_pair_id"] = 11
        elif mutation == "metadata_dwell_mismatch":
            metadata["dwell_time_s"] = 30.5
        else:
            metadata["dwell_time_s"] = "30.0"
        observation = replace(observation, metadata=metadata)
    runtime = _Runtime(observation)

    with pytest.raises((RuntimeError, ValueError)):
        _acquire_spectrum_observation(
            simulation_runtime=runtime,
            full_spectrum_model=model,
            step_id=0,
            pose_xyz=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
            fe_idx=1,
            pb_idx=2,
            live_time_s=30.0,
            travel_time_s=0.0,
            shield_actuation_time_s=0.0,
            require_native_contract=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("step_id", "0"),
        ("step_id", 0.5),
        ("fe_idx", "1"),
        ("fe_idx", 1.5),
        ("pb_idx", "2"),
        ("pb_idx", 2.5),
        ("live_time_s", "30.0"),
        ("live_time_s", float("nan")),
        ("require_native_contract", 1),
        (
            "pose_xyz",
            np.asarray(["0.5", 0.5, 0.5], dtype=object),
        ),
        (
            "pose_xyz",
            np.asarray([float("inf"), 0.5, 0.5], dtype=np.float64),
        ),
    ),
)
def test_acquisition_rejects_coerced_command_fields_before_transport(
    field: str,
    value: object,
) -> None:
    """Invalid command fields must fail before a simulator action is submitted."""
    model = approved_full_spectrum_model()
    runtime = _Runtime(_native_observation())
    arguments: dict[str, object] = {
        "simulation_runtime": runtime,
        "full_spectrum_model": model,
        "step_id": 0,
        "pose_xyz": np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
        "fe_idx": 1,
        "pb_idx": 2,
        "live_time_s": 30.0,
        "travel_time_s": 0.0,
        "shield_actuation_time_s": 0.0,
        "require_native_contract": True,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        _acquire_spectrum_observation(**arguments)

    assert runtime.commands == []


def test_diagnostics_declare_one_likelihood_and_one_background() -> None:
    """Runtime provenance must make removed count/contrast routes explicit."""
    diagnostics = _full_spectrum_model_diagnostics(
        approved_full_spectrum_model(),
        obstacle_attenuation_active=True,
    )["observation_likelihood"]

    assert diagnostics["background_owned_once_by_generative_model"] is True
    assert diagnostics["projected_isotope_counts"] is False
    assert diagnostics["contrast_term"] is False
    assert diagnostics["view_ratio_term"] is False


@pytest.mark.parametrize(
    "resolver, arguments",
    (
        (
            _resolve_candidate_isotopes,
            ({"candidate_isotopes": ["Cs-137", "Cs-137"]}, ["Cs-137"]),
        ),
        (
            _resolve_random_source_isotopes,
            (
                ["Cs-137", "Cs-137"],
                {},
                ["Cs-137"],
            ),
        ),
    ),
)
def test_duplicate_isotope_configuration_fails_closed(
    resolver: object,
    arguments: tuple[object, ...],
) -> None:
    """Duplicate isotope entries cannot silently alias one PF state block."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        resolver(*arguments)
