"""Tests for the truth-isolated pure-PF replay evaluation CLI."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pf.provenance import canonical_json_bytes
from scripts import evaluate_pure_pf_replay
from tests.pure_pf_test_support import make_measurement_log, replay_config


class _PosteriorStub:
    """Return a deterministic JSON-safe posterior fixture."""

    def to_dict(self) -> dict[str, Any]:
        """Return the posterior fixture payload."""
        return {"schema_version": 1, "final_estimate_source": "pf_posterior"}


class _EstimatorStub:
    """Expose the canonical reporting surface used by the evaluation wrapper."""

    estimator_variant = "pf_strict"
    resolved_config_hash = "b" * 64

    def __init__(self, *, config_hash: str) -> None:
        """Store deterministic state and method-call diagnostics."""
        self.config_hash = config_hash
        self.estimates_result = {
            "Cs-137": (
                np.asarray([[0.25, 0.25, 0.0]], dtype=float),
                np.asarray([123.0], dtype=float),
            )
        }
        self.uncertainty_calls: list[dict[str, Any]] = []
        self.posterior_calls = 0

    def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return the canonical point estimate fixture."""
        return self.estimates_result

    def structural_surface_kinds(
        self,
        isotope: str,
        positions: np.ndarray,
        *,
        strict: bool,
    ) -> np.ndarray:
        """Return authoritative surface labels for the estimate fixture."""
        assert isotope == "Cs-137"
        assert strict is True
        assert positions.shape == (1, 3)
        return np.asarray(["floor"], dtype=object)

    def posterior_source_uncertainty(
        self,
        reported_estimates: dict[str, tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Record and return the canonical uncertainty fixture."""
        assert reported_estimates is self.estimates_result
        self.uncertainty_calls.append(dict(kwargs))
        return {
            "Cs-137": [
                {
                    "mode_index": 0,
                    "existence_mass": 0.75,
                    "surface_kind_posterior": {"floor": 1.0},
                }
            ]
        }

    def posterior_snapshot(self) -> _PosteriorStub:
        """Return the canonical posterior fixture."""
        self.posterior_calls += 1
        return _PosteriorStub()

    def continuous_surface_atlas(self) -> None:
        """Return the absent atlas marker accepted by the metric stub."""
        return None

    def serialized_state(self) -> bytes:
        """Return deterministic final-state bytes."""
        return b"pure-pf-final-state"


def _write_truth(path: Path) -> Path:
    """Write one explicit source-truth fixture."""
    path.write_bytes(
        canonical_json_bytes(
            {
                "name": "external-truth",
                "sources": [
                    {
                        "isotope": "Cs-137",
                        "position": [0.25, 0.25, 0.0],
                        "intensity_cps_1m": 100.0,
                    }
                ],
            }
        )
    )
    return path


def test_evaluation_keeps_truth_out_of_replay_and_writes_required_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truth must enter only after the complete canonical PF replay."""
    log_path = make_measurement_log(tmp_path / "measurement-log", record_count=2)
    config_path = tmp_path / "replay.json"
    config_path.write_bytes(canonical_json_bytes(replay_config()))
    truth_path = _write_truth(tmp_path / "truth.json")
    output_path = tmp_path / "nested" / "evaluation.json"
    estimator = _EstimatorStub(config_hash=sha256(config_path.read_bytes()).hexdigest())
    replay_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    metric_calls: list[dict[str, Any]] = []

    def replay_stub(
        *args: Any, **kwargs: Any
    ) -> tuple[_EstimatorStub, tuple[dict, ...]]:
        """Capture the replay boundary without accepting evaluation truth."""
        replay_calls.append((args, kwargs))
        return estimator, ({"record_index": 0}, {"record_index": 1})

    def metrics_stub(
        truth_by_isotope: dict[str, list[dict[str, Any]]],
        estimates_by_isotope: dict[str, list[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture the post-hoc evaluation inputs."""
        metric_calls.append(
            {
                "truth": truth_by_isotope,
                "estimates": estimates_by_isotope,
                "kwargs": kwargs,
            }
        )
        return {"global": {"tp": 1}}

    monkeypatch.setattr(
        evaluate_pure_pf_replay,
        "replay_measurement_log",
        replay_stub,
    )
    monkeypatch.setattr(evaluate_pure_pf_replay, "compute_metrics", metrics_stub)

    result_path = evaluate_pure_pf_replay.evaluate_and_write(
        measurement_log=log_path,
        config=config_path,
        truth_source_json=truth_path,
        output=output_path,
        seed=19,
        match_radius_m=0.75,
    )

    assert result_path == output_path.resolve()
    assert len(replay_calls) == 1
    replay_args, replay_kwargs = replay_calls[0]
    assert replay_args == (log_path.resolve(), config_path.resolve())
    assert replay_kwargs == {"profile": "pf_strict", "seed": 19}
    assert truth_path.resolve() not in replay_args
    assert set(replay_kwargs) == {"profile", "seed"}
    assert len(metric_calls) == 1
    assert metric_calls[0]["truth"]["Cs-137"][0]["surface_kind"] == "floor"
    assert metric_calls[0]["estimates"]["Cs-137"][0] == {
        "pos": [0.25, 0.25, 0.0],
        "strength": 123.0,
        "surface_kind": "floor",
    }
    assert metric_calls[0]["kwargs"]["match_radius_m"] == 0.75
    assert (
        metric_calls[0]["kwargs"]["uncertainty_by_iso"]["Cs-137"][0]["existence_mass"]
        == 0.75
    )
    assert estimator.uncertainty_calls == [{"surface_tolerance_m": 1.0e-5}]
    assert estimator.posterior_calls == 1

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert {
        "provenance",
        "posterior",
        "estimates",
        "uncertainty",
        "metrics",
    } <= payload.keys()
    assert payload["artifact_type"] == ("pure_pf_measurement_log_replay_evaluation")
    assert payload["provenance"]["record_count"] == 2
    assert payload["provenance"]["replayed_record_count"] == 2
    assert payload["provenance"]["truth_passed_to_pf_replay"] is False
    assert payload["provenance"]["truth_scope"] == "posthoc_evaluation_only"
    assert (
        payload["provenance"]["truth_source_json_sha256"]
        == sha256(truth_path.read_bytes()).hexdigest()
    )
    assert payload["posterior"]["final_estimate_source"] == "pf_posterior"
    assert payload["metrics"] == {"global": {"tp": 1}}
    assert list(output_path.parent.glob(f".{output_path.name}.tmp-*")) == []


def test_atomic_writer_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    """An evaluation artifact must not overwrite an existing result."""
    output = tmp_path / "evaluation.json"
    output.write_text("keep-me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        evaluate_pure_pf_replay.write_evaluation_json(
            output,
            {"schema_version": 1},
        )

    assert output.read_text(encoding="utf-8") == "keep-me"
    assert list(tmp_path.glob(".evaluation.json.tmp-*")) == []


@pytest.mark.parametrize(
    "source",
    [
        {
            "isotope": "Cs-137",
            "position": [0.0, 0.0],
            "intensity_cps_1m": 1.0,
        },
        {
            "isotope": "Cs-137",
            "position": [0.0, 0.0, 0.0],
            "intensity_cps_1m": float("inf"),
        },
        {
            "isotope": "",
            "position": [0.0, 0.0, 0.0],
            "intensity_cps_1m": 1.0,
        },
    ],
)
def test_truth_loader_rejects_invalid_physical_sources(
    tmp_path: Path,
    source: dict[str, Any],
) -> None:
    """Malformed or non-finite truth must fail before replay begins."""
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps({"sources": [source]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Truth source|truth source"):
        evaluate_pure_pf_replay._load_truth_sources(truth_path)


@pytest.mark.parametrize(
    "legacy_field",
    ["pos", "strength_cps_1m", "strength", "intensity"],
)
def test_truth_loader_rejects_legacy_source_fields(
    tmp_path: Path,
    legacy_field: str,
) -> None:
    """Post-hoc evaluation must use the same canonical physical source schema."""
    source = {
        "isotope": "Cs-137",
        "position": [0.0, 0.0, 0.0],
        "intensity_cps_1m": 1.0,
    }
    source[legacy_field] = (
        [0.0, 0.0, 0.0] if legacy_field == "pos" else 1.0
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps({"sources": [source]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed field"):
        evaluate_pure_pf_replay._load_truth_sources(truth_path)


def test_truth_loader_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    """Duplicate truth fields must not be silently resolved by parser order."""
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        (
            '{"sources":[{"isotope":"Cs-137","position":[0,0,0],'
            '"intensity_cps_1m":1,"intensity_cps_1m":2}]}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate field"):
        evaluate_pure_pf_replay._load_truth_sources(truth_path)
