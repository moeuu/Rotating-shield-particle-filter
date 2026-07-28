"""Shared runtime observation-model construction for PF and DSS-PP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

from measurement.continuous_kernels import ContinuousKernel
from measurement.detector_geometry import (
    DetectorObservationGeometry,
    detector_observation_geometry_from_runtime_config,
    detector_outer_radius_cm,
)
from measurement.kernels import ShieldParams
from measurement.obstacle_assets import material_mu_cm_inv
from measurement.obstacles import ObstacleGrid
from measurement.shielding import (
    HVL_TVL_TABLE_MM,
    line_resolved_shield_mu_by_isotope,
    mu_by_isotope_from_tvl_mm,
)
from sim.shield_geometry import nested_shield_inner_radii_cm
from sim.shield_geometry import resolve_shield_thickness_config
from spectrum.additive_scatter import AdditiveNoncollidedTransportResponse
from spectrum.transport_spectral import (
    geometry_conditioned_model_from_runtime_config,
)


@dataclass(frozen=True)
class RuntimeObservationModel:
    """Collect the shared PF/DSS-PP observation-kernel parameters."""

    detector_geometry: DetectorObservationGeometry
    shield_params: ShieldParams
    mu_by_isotope: dict[str, object]
    line_mu_by_isotope: dict[str, tuple[dict[str, float], ...]] | None
    additive_scatter_response: AdditiveNoncollidedTransportResponse | None
    obstacle_mu_by_isotope: dict[str, float] | None
    obstacle_height_m: float
    obstacle_buildup_coeff: float
    source_extent_radius_m: float
    source_extent_samples: int


def _buildup_runtime_config(runtime_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested PF buildup configuration payload."""
    if "pf_buildup" not in runtime_config:
        return {}
    payload = runtime_config["pf_buildup"]
    if not isinstance(payload, Mapping):
        raise TypeError("pf_buildup must be a mapping.")
    unknown = sorted(
        set(str(key) for key in payload)
        - {"fe_coeff", "pb_coeff", "obstacle_coeff"}
    )
    if unknown:
        raise ValueError(f"pf_buildup contains unknown keys: {unknown}.")
    return payload


def _nonnegative_finite_float(value: object, *, field_name: str) -> float:
    """Return one strict nonnegative finite real configuration value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative.")
    return parsed


def _positive_integer(value: object, *, field_name: str) -> int:
    """Return one strict positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def _strict_boolean(value: object, *, field_name: str) -> bool:
    """Return one exact JSON boolean configuration value."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _nonempty_string(value: object, *, field_name: str) -> str:
    """Return one exact nonempty string configuration value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must be nonempty.")
    return value


def _explicit_obstacle_mu_by_isotope(
    payload: Mapping[str, Any],
    *,
    isotopes: Sequence[str],
) -> dict[str, float] | None:
    """Return an explicitly configured isotope obstacle-mu override."""
    if "obstacle_mu_by_isotope" in payload:
        raise ValueError(
            "obstacle_mu_by_isotope was removed; use the canonical "
            "pf_obstacle_mu_by_isotope field."
        )
    if "pf_obstacle_mu_by_isotope" not in payload:
        return None
    raw = payload["pf_obstacle_mu_by_isotope"]
    if not isinstance(raw, Mapping):
        raise TypeError(
            "pf_obstacle_mu_by_isotope must map isotope names to attenuation "
            "coefficients."
        )
    parsed = {
        _nonempty_string(
            isotope,
            field_name="pf_obstacle_mu_by_isotope key",
        ): _nonnegative_finite_float(
            value,
            field_name=f"pf_obstacle_mu_by_isotope[{isotope!s}]",
        )
        for isotope, value in raw.items()
    }
    expected = {str(isotope) for isotope in isotopes}
    if not expected or set(parsed) != expected or any(not key for key in parsed):
        raise ValueError(
            "pf_obstacle_mu_by_isotope keys must exactly equal the configured "
            "isotope set."
        )
    return parsed


def _obstacle_mu_by_isotope_from_runtime_config(
    payload: Mapping[str, Any],
    *,
    isotopes: Sequence[str],
) -> dict[str, float] | None:
    """Resolve obstacle attenuation coefficients for the runtime material."""
    explicit = _explicit_obstacle_mu_by_isotope(
        payload,
        isotopes=isotopes,
    )
    if explicit is not None:
        return explicit
    material = _nonempty_string(
        payload.get(
            "pf_obstacle_material",
            payload.get("obstacle_material", "concrete"),
        ),
        field_name="pf_obstacle_material",
    )
    return {
        str(isotope): _nonnegative_finite_float(
            material_mu_cm_inv(material, str(isotope)),
            field_name=f"material obstacle mu for {isotope!s}",
        )
        for isotope in isotopes
    }


def build_runtime_observation_model(
    runtime_config: Mapping[str, Any] | None,
    *,
    isotopes: Sequence[str],
) -> RuntimeObservationModel:
    """Build the shared observation model used by PF, planning, and validation."""
    if runtime_config is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(runtime_config, Mapping):
        payload = runtime_config
    else:
        raise TypeError("runtime_config must be a mapping or None.")
    detector_model = payload.get("detector_model", {})
    if not isinstance(detector_model, Mapping):
        raise TypeError("detector_model must be a mapping.")
    isotope_order = tuple(
        _nonempty_string(isotope, field_name="isotope")
        for isotope in isotopes
    )
    if (
        not isotope_order
        or any(not isotope for isotope in isotope_order)
        or len(set(isotope_order)) != len(isotope_order)
    ):
        raise ValueError("isotopes must contain unique nonempty names.")
    detector_geometry = detector_observation_geometry_from_runtime_config(payload)
    if detector_geometry.aperture_sampling != "solid_angle_cone":
        raise ValueError(
            "Production PF/DSS must use the native Geant4 "
            "solid_angle_cone detector-aperture geometry."
        )
    detector_outer_radius_cm_value = detector_outer_radius_cm(detector_model)
    shield_thickness = resolve_shield_thickness_config(dict(payload))
    inner_radius_fe_cm, inner_radius_pb_cm = nested_shield_inner_radii_cm(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        detector_outer_radius_cm=detector_outer_radius_cm_value,
    )
    buildup = _buildup_runtime_config(payload)
    buildup_fe = _nonnegative_finite_float(
        buildup.get("fe_coeff", 0.0),
        field_name="pf_buildup.fe_coeff",
    )
    buildup_pb = _nonnegative_finite_float(
        buildup.get("pb_coeff", 0.0),
        field_name="pf_buildup.pb_coeff",
    )
    buildup_obstacle = _nonnegative_finite_float(
        buildup.get("obstacle_coeff", 0.0),
        field_name="pf_buildup.obstacle_coeff",
    )
    shield_params = ShieldParams(
        thickness_fe_cm=float(shield_thickness.thickness_fe_cm),
        thickness_pb_cm=float(shield_thickness.thickness_pb_cm),
        inner_radius_fe_cm=inner_radius_fe_cm,
        inner_radius_pb_cm=inner_radius_pb_cm,
        buildup_fe_coeff=buildup_fe,
        buildup_pb_coeff=buildup_pb,
    )
    mu_by_isotope = mu_by_isotope_from_tvl_mm(
        HVL_TVL_TABLE_MM,
        isotopes=isotope_order,
    )
    if not mu_by_isotope:
        mu_by_isotope = {
            str(isotope): {"fe": shield_params.mu_fe, "pb": shield_params.mu_pb}
            for isotope in isotope_order
        }
    line_mu_by_isotope = None
    source_rate_model = _nonempty_string(
        payload.get("source_rate_model", "detector_cps_1m"),
        field_name="source_rate_model",
    )
    if source_rate_model != "detector_cps_1m":
        raise ValueError(
            "Production observation model requires "
            "source_rate_model='detector_cps_1m'."
        )
    if _strict_boolean(
        payload.get("pf_line_resolved_shield_attenuation", True),
        field_name="pf_line_resolved_shield_attenuation",
    ):
        line_mu_by_isotope = line_resolved_shield_mu_by_isotope(
            isotopes=isotope_order,
            normalize_line_intensities=True,
        )
    full_spectrum_model = (
        geometry_conditioned_model_from_runtime_config(payload)
        if (
            "full_spectrum_generative_model" in payload
            or "full_spectrum_generative_model_path" in payload
        )
        else None
    )
    obstacle_mu_by_isotope = _obstacle_mu_by_isotope_from_runtime_config(
        payload,
        isotopes=isotope_order,
    )
    obstacle_height_m = _nonnegative_finite_float(
        payload.get("obstacle_height_m", 2.0),
        field_name="obstacle_height_m",
    )
    removed_extent_keys = sorted(
        key
        for key in ("pf_source_extent_radius_m", "pf_source_extent_samples")
        if key in payload
    )
    if removed_extent_keys:
        raise ValueError(
            "Legacy PF source-extent keys were removed; use "
            "pf_obstacle_source_extent_radius_m and "
            f"pf_obstacle_source_extent_samples: {removed_extent_keys}."
        )
    source_extent_radius_m = _nonnegative_finite_float(
        payload.get("pf_obstacle_source_extent_radius_m", 0.0),
        field_name="pf_obstacle_source_extent_radius_m",
    )
    source_extent_samples = _positive_integer(
        payload.get("pf_obstacle_source_extent_samples", 1),
        field_name="pf_obstacle_source_extent_samples",
    )
    if (source_extent_radius_m == 0.0) != (source_extent_samples == 1):
        raise ValueError(
            "Source extent must use radius=0 with samples=1, or a positive "
            "radius with at least two samples."
        )
    return RuntimeObservationModel(
        detector_geometry=detector_geometry,
        shield_params=shield_params,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope=line_mu_by_isotope,
        additive_scatter_response=(
            None
            if full_spectrum_model is None
            else full_spectrum_model.additive_scatter_response
        ),
        obstacle_mu_by_isotope=obstacle_mu_by_isotope,
        obstacle_height_m=obstacle_height_m,
        obstacle_buildup_coeff=buildup_obstacle,
        source_extent_radius_m=source_extent_radius_m,
        source_extent_samples=source_extent_samples,
    )


def continuous_kernel_from_observation_model(
    model: RuntimeObservationModel,
    *,
    obstacle_grid: ObstacleGrid | None,
    use_gpu: bool,
) -> ContinuousKernel:
    """Build a ContinuousKernel from the shared runtime observation model."""
    if not isinstance(use_gpu, bool):
        raise TypeError("use_gpu must be a boolean.")
    return ContinuousKernel(
        mu_by_isotope=model.mu_by_isotope,
        shield_params=model.shield_params,
        obstacle_grid=obstacle_grid,
        obstacle_height_m=model.obstacle_height_m,
        obstacle_mu_by_isotope=model.obstacle_mu_by_isotope,
        obstacle_buildup_coeff=(
            model.obstacle_buildup_coeff if obstacle_grid is not None else 0.0
        ),
        detector_radius_m=model.detector_geometry.count_radius_m,
        detector_aperture_radius_m=model.detector_geometry.aperture_radius_m,
        detector_aperture_samples=model.detector_geometry.aperture_samples,
        detector_aperture_sampling=model.detector_geometry.aperture_sampling,
        source_extent_radius_m=model.source_extent_radius_m,
        source_extent_samples=model.source_extent_samples,
        line_mu_by_isotope=model.line_mu_by_isotope,
        additive_scatter_response=model.additive_scatter_response,
        use_gpu=use_gpu,
    )
