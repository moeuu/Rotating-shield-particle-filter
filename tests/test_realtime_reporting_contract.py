"""Regression tests for strict live-run reporting contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import realtime_demo


def _single_source_history() -> list[
    dict[str, tuple[np.ndarray, np.ndarray]]
]:
    """Return one valid single-isotope online estimate frame."""
    return [
        {
            "Cs-137": (
                np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
                np.asarray([100.0], dtype=np.float64),
            )
        }
    ]


def _single_source_truth() -> dict[str, list[dict[str, Any]]]:
    """Return one truth source aligned with the valid online estimate."""
    return {
        "Cs-137": [
            {
                "pos": [0.0, 0.0, 0.0],
                "strength": 100.0,
            }
        ]
    }


def test_final_online_error_does_not_reuse_last_evaluable_frame() -> None:
    """An unmatched final frame must remain unavailable instead of looking stale."""
    history = [
        *_single_source_history(),
        {
            "Cs-137": (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        },
    ]

    summary = realtime_demo._online_estimate_metric_summary(
        history,
        _single_source_truth(),
        match_radius_m=1.0,
        surface_atlas=None,
    )["Cs-137"]

    assert summary["final_online_source_count_error"] == -1
    assert summary["final_online_surface_path_error_m"] is None
    assert summary["final_online_surface_path_error_available"] is False
    assert (
        summary["final_online_surface_path_error_unavailable_reason"]
        == "no_gated_localization_match"
    )
    assert summary["last_evaluable_online_surface_path_error_m"] == pytest.approx(
        0.0
    )
    assert summary["last_evaluable_online_surface_path_error_step"] == 0


@pytest.mark.parametrize(
    ("positions", "strengths"),
    (
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            np.asarray([100.0], dtype=np.float64),
        ),
        (
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([100.0, 200.0], dtype=np.float64),
        ),
    ),
)
def test_online_metric_sources_reject_position_strength_count_mismatch(
    positions: np.ndarray,
    strengths: np.ndarray,
) -> None:
    """Metric conversion must not truncate an inconsistent source estimate."""
    with pytest.raises(ValueError, match="same number of sources"):
        realtime_demo._estimate_map_to_metric_sources(
            {"Cs-137": (positions, strengths)}
        )


@pytest.mark.parametrize(
    ("malformed_metrics", "error_match"),
    (
        ({}, "required 'isotopes' mapping; legacy metric schemas"),
        (
            {"isotopes": {}},
            "missing required isotope .*legacy metric schemas",
        ),
        (
            {
                "isotopes": {
                    "Cs-137": {
                        "surface_path_error": {"mean": None},
                    }
                }
            },
            r"counts\.source_count_error; legacy metric schemas",
        ),
        (
            {
                "isotopes": {
                    "Cs-137": {
                        "counts": {},
                        "surface_path_error": {"mean": None},
                    }
                }
            },
            r"counts\.source_count_error; legacy metric schemas",
        ),
    ),
)
def test_online_summary_rejects_missing_metric_contract_fields(
    monkeypatch: pytest.MonkeyPatch,
    malformed_metrics: dict[str, Any],
    error_match: str,
) -> None:
    """Missing isotope and count fields must never imply zero count error."""

    def _malformed_compute_metrics(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, Any]:
        """Return one intentionally incomplete metric result."""
        return malformed_metrics

    monkeypatch.setattr(
        realtime_demo,
        "compute_metrics",
        _malformed_compute_metrics,
    )

    with pytest.raises(RuntimeError, match=error_match):
        realtime_demo._online_estimate_metric_summary(
            _single_source_history(),
            _single_source_truth(),
            match_radius_m=1.0,
            surface_atlas=None,
        )


@pytest.mark.parametrize(
    "nonfinite",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float32(np.nan),
    ),
)
def test_strict_json_payload_rejects_nonfinite_scientific_values(
    nonfinite: float,
) -> None:
    """Canonical summaries must fail at the exact nonfinite payload path."""
    payload = {
        "evaluation_metrics": {
            "accuracy": {
                "surface_path_error_m": nonfinite,
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            r"\$\['evaluation_metrics'\]\['accuracy'\]"
            r"\['surface_path_error_m'\]"
        ),
    ):
        realtime_demo._sanitize_json_payload(payload)


def test_strict_json_payload_preserves_explicit_unavailable_reason() -> None:
    """A deliberate null remains valid when its unavailable reason is explicit."""
    payload = {
        "surface_path_error_m": None,
        "surface_path_error_unavailable_reason": "no_gated_localization_match",
    }

    assert realtime_demo._sanitize_json_payload(payload) == payload


def test_notification_spectrum_rejects_axis_count_truncation() -> None:
    """Notification rendering must not conceal an energy-axis contract error."""
    with pytest.raises(ValueError, match="same nonempty shape"):
        realtime_demo._thin_spectrum_for_notification(
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([3.0], dtype=np.float64),
            16,
        )


def test_logged_sources_reject_position_strength_truncation() -> None:
    """Console formatting must not conceal an inconsistent source estimate."""
    with pytest.raises(ValueError, match="same count"):
        realtime_demo._fmt_sources(
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )


@pytest.mark.parametrize(
    "robot_position",
    (
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.asarray([1.0, np.nan, 3.0], dtype=np.float64),
    ),
)
def test_intermediate_trace_rejects_invalid_robot_position(
    robot_position: np.ndarray,
) -> None:
    """A malformed robot pose must not be padded or truncated in reports."""
    frame = {
        "robot_position": robot_position,
        "estimated_sources": {},
        "estimated_strengths": {},
    }
    with pytest.raises(ValueError, match="robot_position"):
        realtime_demo._build_intermediate_estimate_trace_payload(frame)


class _EstimateStub:
    """Expose only the strict posterior-estimate interface used by reports."""

    def __init__(self, estimates: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Store one deterministic estimate mapping."""
        self._estimates = estimates

    def estimates(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return the configured estimate mapping."""
        return self._estimates


def test_posterior_trace_rejects_missing_configured_isotope() -> None:
    """A missing filter output must not be reported as an empty K=0 state."""
    estimator = _EstimateStub(
        {
            "Cs-137": (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        }
    )
    with pytest.raises(RuntimeError, match="exactly every configured isotope"):
        realtime_demo._current_pf_posterior_estimate_trace_frame(
            estimator,
            ("Cs-137", "Co-60"),
            {"robot_position": np.zeros(3, dtype=np.float64)},
            step_index=0,
            elapsed_s=0.0,
        )


def test_primary_estimates_reject_missing_configured_isotope() -> None:
    """Primary scientific output must not turn a missing isotope into K=0."""
    estimator = _EstimateStub(
        {
            "Cs-137": (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        }
    )
    estimator.profile_capabilities = object()
    estimator.pf_config = SimpleNamespace(estimator_profile="pf_strict")
    with pytest.raises(RuntimeError, match="exactly every configured isotope"):
        realtime_demo._pure_pf_primary_estimates(
            estimator,
            ("Cs-137", "Co-60"),
        )
