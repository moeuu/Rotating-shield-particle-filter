"""Integration tests for mandatory completed-run cluster evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.ral_ablation import session_runner
from evaluation.completed_run import evaluate_completed_pf_run


def _write_json(path: Path, payload: object) -> None:
    """Write one compact JSON fixture below a pytest temporary directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_run_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Write one exactly joined completed-run evaluation fixture."""
    digest = "a" * 64
    paths = {
        "result": tmp_path / "closed_loop_result.json",
        "posterior": tmp_path / "pf_posterior.json",
        "input": tmp_path / "pf_post_run_evaluation_input.json",
        "truth": tmp_path / "truth_manifest.json",
    }
    _write_json(
        paths["result"],
        {
            "schema_version": 2,
            "execution_status": "complete",
            "sampler_quality_status": "pass",
            "run_id": "run-1",
        },
    )
    _write_json(
        paths["posterior"],
        {
            "provenance": {"measurement_log_sha256": digest},
            "isotopes": {
                "Cs-137": {
                    "cardinality_distribution": {"1": 0.99, "8": 0.01},
                    "modes": [
                        {
                            "position_medoid_xyz": [0.1, 0.0, 0.0],
                            "strength_representative_cps_1m": 105.0,
                        }
                    ],
                }
            },
        },
    )
    _write_json(
        paths["input"],
        {
            "schema_version": 1,
            "artifact_family": "pf_post_run_cluster_evaluation_input",
            "source_run_id": "run-1",
            "measurement_log_sha256": digest,
            "hard_max_sources_per_isotope": 8,
            "response_signature_semantics": (
                "normalized_same_isotope_expected_count_by_completed_measurement"
            ),
            "truth_read": False,
            "isotopes": {
                "Cs-137": {
                    "mode_label_indices": [0],
                    "mode_positions_xyz_m": [[0.1, 0.0, 0.0]],
                    "mode_strengths_cps_1m": [105.0],
                    "normalized_response_signatures_measurement_by_mode": [[1.0]],
                }
            },
        },
    )
    _write_json(
        paths["truth"],
        {
            "schema_version": 1,
            "run_id": "run-1",
            "experiment_profile_id": "profile",
            "scene_variant_id": "scene",
            "scene_seed": 123,
            "scene_rng_provenance": {"algorithm": "PCG64"},
            "sources": [
                {
                    "isotope": "Cs-137",
                    "position": [0.0, 0.0, 0.0],
                    "intensity_cps_1m": 100.0,
                }
            ],
        },
    )
    return paths


def test_completed_run_evaluation_joins_exact_truth_and_reports_each_source(
    tmp_path: Path,
) -> None:
    """The reusable evaluator must emit per-source position and strength errors."""
    paths = _completed_run_artifacts(tmp_path)

    result = evaluate_completed_pf_run(
        result_path=paths["result"],
        posterior_path=paths["posterior"],
        evaluation_input_path=paths["input"],
        truth_manifest_path=paths["truth"],
    )

    assert result["execution_status"] == "complete"
    assert result["sampler_quality_status"] == "pass"
    assert result["accuracy_status"] == "pass"
    assert result["schema_version"] == 3
    assert result["run_identity"]["run_id"] == "run-1"
    source = result["isotopes"]["Cs-137"]["truth_sources"][0]
    assert source["merged_centroid_position_error_m"] == pytest.approx(0.1)
    assert source["strength_weighted_rms_position_error_m"] == pytest.approx(
        0.1
    )
    assert source["combined_relative_strength_error"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("sampler_status", "hard_cap_mass"),
    (("pass", 0.2), ("failed", 0.01)),
)
def test_completed_run_rejects_sampler_hard_cap_contradiction(
    tmp_path: Path,
    sampler_status: str,
    hard_cap_mass: float,
) -> None:
    """Published quality and posterior hard-cap evidence must agree exactly."""
    paths = _completed_run_artifacts(tmp_path)
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    result["sampler_quality_status"] = sampler_status
    _write_json(paths["result"], result)
    posterior = json.loads(paths["posterior"].read_text(encoding="utf-8"))
    posterior["isotopes"]["Cs-137"]["cardinality_distribution"] = {
        "1": 1.0 - hard_cap_mass,
        "8": hard_cap_mass,
    }
    _write_json(paths["posterior"], posterior)

    with pytest.raises(ValueError, match="contradicts hard-cap"):
        evaluate_completed_pf_run(
            result_path=paths["result"],
            posterior_path=paths["posterior"],
            evaluation_input_path=paths["input"],
            truth_manifest_path=paths["truth"],
        )


def test_private_session_runner_always_publishes_post_run_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successful private session must invoke the standard evaluator once."""
    runtime_root = tmp_path / "runtime"
    private_root = runtime_root / "private_runs" / "ral_ablation"
    scenario = private_root / "scenarios" / "run.json"
    truth = private_root / "truth_manifests" / "run.json"
    pf_output = tmp_path / "pf-output"
    _write_json(scenario, {"private": True})
    _write_json(truth, {"private": True})
    pf_output.mkdir()
    calls: list[dict[str, Path]] = []
    runtime_commands: list[list[str]] = []
    controller_commands: list[list[str]] = []

    class _RuntimeProcess:
        """Expose the successful process methods used by the orchestrator."""

        def poll(self) -> int:
            """Report a completed runtime process."""
            return 0

        def wait(self, timeout: float | None = None) -> int:
            """Return a successful runtime exit code."""
            del timeout
            return 0

    monkeypatch.setattr(
        session_runner.subprocess,
        "Popen",
        lambda command, **kwargs: (
            runtime_commands.append(list(command)) or _RuntimeProcess()
        ),
    )
    monkeypatch.setattr(
        session_runner.subprocess,
        "run",
        lambda command, **kwargs: (
            controller_commands.append(list(command))
            or SimpleNamespace(returncode=0)
        ),
    )

    def _evaluate(**kwargs: Path) -> dict[str, object]:
        """Record the private post-run join without reading fake artifacts."""
        calls.append(dict(kwargs))
        return {
            "schema_version": 3,
            "execution_status": "complete",
            "sampler_quality_status": "pass",
            "accuracy_status": "pass",
        }

    monkeypatch.setattr(session_runner, "evaluate_completed_pf_run", _evaluate)

    return_code = session_runner.run_isolated_ral_session(
        runtime_root=runtime_root,
        scenario_path=scenario,
        truth_manifest_path=truth,
        pf_config_path=tmp_path / "pf.json",
        control_policy_path=tmp_path / "policy.json",
        expected_control_policy_sha256="b" * 64,
        pf_output_dir=pf_output,
        pf_seed=17,
    )

    assert return_code == 0
    assert len(calls) == 1
    assert calls[0]["truth_manifest_path"] == truth.resolve()
    runtime_overlay_index = runtime_commands[0].index(
        "--cui-truth-overlay-socket-path"
    )
    controller_overlay_index = controller_commands[0].index(
        "--cui-truth-overlay-socket"
    )
    assert (
        runtime_commands[0][runtime_overlay_index + 1]
        == controller_commands[0][controller_overlay_index + 1]
    )
    evaluation_path = private_root / "evaluations" / "run.json"
    assert json.loads(evaluation_path.read_text(encoding="utf-8")) == {
        "accuracy_status": "pass",
        "execution_status": "complete",
        "sampler_quality_status": "pass",
        "schema_version": 3,
    }
