"""Tests for the live PF runtime-ingestion boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from runtime.measurement_log import load_measurement_log

from pf.live_session import (
    PFLiveSessionError,
    bind_published_measurement_log,
    build_live_estimator,
    measurement_record_to_station_input,
)
from pf.pure_estimator import RotatingShieldPFConfig
from tests.pure_pf_test_support import make_measurement_log


def test_live_builder_uses_run_context_without_synthetic_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live construction resolves its forward model directly from context."""
    from pf import live_session

    context = SimpleNamespace(
        schema_version=2,
        runtime_config_sha256="a" * 64,
    )
    forward = object()
    estimator = object()
    calls: dict[str, object] = {}

    class _Resolver:
        """Capture direct shared-runtime context resolution."""

        @classmethod
        def from_run_context(
            cls,
            actual_context: object,
            *,
            run_root: object,
        ) -> object:
            """Return the sentinel authenticated forward context."""
            del cls
            calls["context"] = actual_context
            calls["run_root"] = run_root
            return forward

    def build_from_forward(
        actual_forward: object,
        config: object,
        **kwargs: object,
    ) -> object:
        """Capture the thin PF-specific construction adapter."""
        calls["forward"] = actual_forward
        calls["config"] = config
        calls["kwargs"] = kwargs
        return estimator

    monkeypatch.setattr(live_session, "ResolvedForwardContext", _Resolver)
    monkeypatch.setattr(
        live_session,
        "_build_live_estimator_from_forward_context",
        build_from_forward,
    )

    actual = build_live_estimator(
        context,  # type: ignore[arg-type]
        {"pure_pf_schema_version": 1},
        profile="pf_strict",
        seed=9,
        runtime_root=tmp_path,
    )

    assert actual is estimator
    assert calls["context"] is context
    assert calls["run_root"] == tmp_path.resolve()
    assert calls["forward"] is forward
    assert calls["kwargs"] == {
        "profile": "pf_strict",
        "seed": 9,
        "measurement_log_schema_version": 2,
        "measurement_runtime_config_sha256": "a" * 64,
        "config_hash": None,
        "inference_isotopes": None,
    }


def test_live_record_forwards_only_raw_spectrum_and_action_geometry(
    tmp_path: Path,
) -> None:
    """Live ingestion forwards the raw spectrum and selected shield action."""
    log = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )

    station_input = measurement_record_to_station_input(log.records[0])

    assert len(station_input) == 4
    spectrum, fe_index, pb_index, live_time_s = station_input
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.dtype == np.int64
    np.testing.assert_array_equal(spectrum, log.records[0].spectrum_counts)
    assert (fe_index, pb_index, live_time_s) == (0, 0, 1.0)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("spectrum_counts", np.asarray([1.0, 0.5], dtype=np.float64)),
        ("spectrum_counts", np.asarray([1, 2], dtype=np.int32)),
        ("fe_orientation_index", True),
        ("pb_orientation_index", "0"),
        ("live_time_s", True),
        ("live_time_s", "1.0"),
    ),
)
def test_live_record_rejects_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """The live adapter preserves raw observation types without coercion."""
    values: dict[str, Any] = {
        "spectrum_counts": np.asarray([1, 2], dtype=np.int64),
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "live_time_s": 1.0,
    }
    values[field_name] = invalid

    with pytest.raises(PFLiveSessionError):
        measurement_record_to_station_input(SimpleNamespace(**values))


def test_published_log_binding_requires_the_live_record_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final provenance binding accepts only the assimilated live session."""
    from pf import live_session

    published = load_measurement_log(
        make_measurement_log(tmp_path / "measurement-log", record_count=1)
    )
    isotopes = tuple(published.run_manifest["isotopes"])
    estimator = SimpleNamespace(
        candidate_isotopes=isotopes,
        joint_isotope_order=lambda: isotopes,
        measurements=[object()],
        pf_config=RotatingShieldPFConfig(),
        random_seed=7,
        measurement_log_sha256="unavailable",
        resolved_config_hash="unavailable",
    )
    validated: list[object] = []
    monkeypatch.setattr(
        live_session,
        "_validate_published_forward_context",
        validated.append,
    )

    bind_published_measurement_log(
        estimator,
        published,
        live_records=published.records,
    )

    assert validated == [published]
    assert estimator.measurement_log_sha256 == published.log_sha256
    assert len(estimator.resolved_config_hash) == 64

    changed_record = replace(published.records[0], live_time_s=2.0)
    changed_published = replace(published, records=(changed_record,))
    with pytest.raises(PFLiveSessionError, match="ordered live records"):
        bind_published_measurement_log(
            estimator,
            changed_published,
            live_records=published.records,
        )
