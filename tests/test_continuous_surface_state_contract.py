"""Contracts for the authoritative continuous-surface PF state."""

from __future__ import annotations

from dataclasses import asdict, fields

import numpy as np
import pytest

from pf.estimator_config import RotatingShieldPFConfig
from pf.state import IsotopeState


def test_surface_state_has_no_cartesian_or_background_storage() -> None:
    """Only chart/UV coordinates and positive strengths may define a source."""
    state = IsotopeState(
        num_sources=1,
        strengths=np.asarray([3.0], dtype=np.float64),
        surface_chart_ids=np.asarray([4], dtype=np.int64),
        surface_uv=np.asarray([[0.25, 0.75]], dtype=np.float64),
    )

    assert {item.name for item in fields(IsotopeState)} == {
        "num_sources",
        "strengths",
        "surface_chart_ids",
        "surface_uv",
    }
    assert set(asdict(state)) == {
        "num_sources",
        "strengths",
        "surface_chart_ids",
        "surface_uv",
    }
    with pytest.raises(AttributeError):
        _ = state.positions
    with pytest.raises(AttributeError):
        _ = state.background


def test_surface_state_rejects_zero_strength_and_legacy_xyz_keywords() -> None:
    """A represented source cannot be a zero-strength or cached-XYZ ghost."""
    with pytest.raises(ValueError, match="positive finite strengths"):
        IsotopeState(
            num_sources=1,
            strengths=np.asarray([0.0]),
            surface_chart_ids=np.asarray([0], dtype=np.int64),
            surface_uv=np.asarray([[0.5, 0.5]]),
        )
    with pytest.raises(TypeError):
        IsotopeState(
            num_sources=1,
            strengths=np.asarray([1.0]),
            surface_chart_ids=np.asarray([0], dtype=np.int64),
            surface_uv=np.asarray([[0.5, 0.5]]),
            positions=np.asarray([[1.0, 2.0, 3.0]]),
        )


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_surface_state_rejects_noninteger_cardinality(invalid: object) -> None:
    """State cardinality must never be silently truncated or coerced."""
    with pytest.raises(TypeError, match="num_sources must be an integer"):
        IsotopeState(
            num_sources=invalid,
            strengths=np.asarray([1.0]),
            surface_chart_ids=np.asarray([0], dtype=np.int64),
            surface_uv=np.asarray([[0.5, 0.5]]),
        )


def test_pf_config_requires_positive_strengths_and_float64() -> None:
    """Production state and GPU arithmetic must share strict numeric support."""
    with pytest.raises(ValueError, match="positive"):
        RotatingShieldPFConfig(
            strength_prior={
                "kind": "shifted_gamma",
                "minimum_cps_1m": 0.0,
                "shape": 2.0,
                "scale_cps_1m": 1.0,
            }
        )
    with pytest.raises(ValueError, match="gpu_dtype='float64'"):
        RotatingShieldPFConfig(gpu_dtype="float32")
