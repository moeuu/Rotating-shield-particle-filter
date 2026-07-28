"""Shared detector geometry helpers for count and shield models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

DEFAULT_CRYSTAL_RADIUS_M = 0.038
DEFAULT_HOUSING_THICKNESS_M = 0.0015
DEFAULT_PF_DETECTOR_APERTURE_SAMPLES = 121


@dataclass(frozen=True)
class DetectorObservationGeometry:
    """Describe detector geometry used by PF and DSS-PP observation kernels."""

    count_radius_m: float
    aperture_radius_m: float
    aperture_samples: int
    aperture_sampling: str = "solid_angle_cone"


def _configuration_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Return a strict optional JSON-object configuration payload."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None.")
    return value


def _nonnegative_finite_real(value: object, *, field_name: str) -> float:
    """Return a strict nonnegative finite real configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative.")
    return parsed


def _positive_integer(value: object, *, field_name: str) -> int:
    """Return a strict positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def normalize_detector_aperture_sampling(value: object | None) -> str:
    """Return a canonical detector-aperture sampling mode."""
    if value is None:
        text = "solid_angle_cone"
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("pf_detector_aperture_sampling must be a string.")
    if text not in {"solid_angle_cone", "disk"}:
        raise ValueError(f"Unsupported detector aperture sampling mode: {value!r}")
    return text


def detector_active_radius_m(
    detector_model: Mapping[str, Any] | None,
    *,
    default_radius_m: float = 0.0,
) -> float:
    """Return the active crystal radius used by detector-cps count geometry."""
    payload = _configuration_mapping(detector_model, field_name="detector_model")
    default_radius = _nonnegative_finite_real(
        default_radius_m,
        field_name="default_radius_m",
    )
    return _nonnegative_finite_real(
        payload.get("crystal_radius_m", default_radius),
        field_name="detector_model.crystal_radius_m",
    )


def detector_outer_radius_cm(
    detector_model: Mapping[str, Any] | None,
    *,
    default_crystal_radius_m: float = DEFAULT_CRYSTAL_RADIUS_M,
    default_housing_thickness_m: float = DEFAULT_HOUSING_THICKNESS_M,
) -> float:
    """Return the crystal-plus-housing radius used by shield contact geometry."""
    payload = _configuration_mapping(detector_model, field_name="detector_model")
    crystal_radius_m = _nonnegative_finite_real(
        payload.get("crystal_radius_m", default_crystal_radius_m),
        field_name="detector_model.crystal_radius_m",
    )
    housing_thickness_m = _nonnegative_finite_real(
        payload.get("housing_thickness_m", default_housing_thickness_m),
        field_name="detector_model.housing_thickness_m",
    )
    return 100.0 * (crystal_radius_m + housing_thickness_m)


def detector_outer_radius_m(
    detector_model: Mapping[str, Any] | None,
    *,
    default_crystal_radius_m: float = DEFAULT_CRYSTAL_RADIUS_M,
    default_housing_thickness_m: float = DEFAULT_HOUSING_THICKNESS_M,
) -> float:
    """Return the crystal-plus-housing radius in meters."""
    return detector_outer_radius_cm(
        detector_model,
        default_crystal_radius_m=default_crystal_radius_m,
        default_housing_thickness_m=default_housing_thickness_m,
    ) / 100.0


def detector_observation_geometry_from_runtime_config(
    runtime_config: Mapping[str, Any] | None,
    *,
    default_aperture_samples: int = DEFAULT_PF_DETECTOR_APERTURE_SAMPLES,
) -> DetectorObservationGeometry:
    """
    Resolve detector geometry shared by PF likelihoods and DSS-PP scoring.

    ``count_radius_m`` follows the active crystal radius used by the Geant4
    detector-cps@1m source-rate normalization.  ``aperture_radius_m`` follows
    the source-to-detector target radius used for ray-level shield/obstacle
    sampling; by default this includes the housing because Geant4 directs
    detector-cone primaries to the detector outer radius.
    """
    payload = _configuration_mapping(
        runtime_config,
        field_name="runtime_config",
    )
    detector_model = payload.get("detector_model", {})
    if not isinstance(detector_model, Mapping):
        raise TypeError("detector_model must be a mapping.")
    count_radius = detector_active_radius_m(detector_model)
    if "pf_detector_count_radius_m" in payload:
        count_radius = _nonnegative_finite_real(
            payload["pf_detector_count_radius_m"],
            field_name="pf_detector_count_radius_m",
        )
    aperture_radius = detector_outer_radius_m(detector_model)
    if "pf_detector_aperture_radius_m" in payload:
        aperture_radius = _nonnegative_finite_real(
            payload["pf_detector_aperture_radius_m"],
            field_name="pf_detector_aperture_radius_m",
        )
    samples = _positive_integer(
        payload.get("pf_detector_aperture_samples", default_aperture_samples),
        field_name="pf_detector_aperture_samples",
    )
    sampling = normalize_detector_aperture_sampling(
        payload.get("pf_detector_aperture_sampling", "solid_angle_cone")
    )
    return DetectorObservationGeometry(
        count_radius_m=count_radius,
        aperture_radius_m=aperture_radius,
        aperture_samples=samples,
        aperture_sampling=sampling,
    )
