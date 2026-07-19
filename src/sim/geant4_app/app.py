"""Geant4 sidecar application entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.geant4_app.engine import (
    Geant4EngineConfig,
    Geant4StepRequest,
    build_geant4_engine,
)
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
    ExportedDetectorModel,
    export_scene_for_geant4,
)
from sim.isaacsim_app.app import IsaacAssetGeometry, StageMaterialRule
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.scene_builder import SceneBuilder, SceneDescription
from sim.isaacsim_app.stage_backend import (
    FakeStageBackend,
    IsaacSimStageBackend,
    StageBackend,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.radiation_visualization import RadiationVisualizationConfig
from sim.shield_geometry import ShieldThicknessConfig, resolve_shield_thickness_config
from spectrum.pipeline import SpectralDecomposer


_MANAGED_GEANT4_EXECUTABLE_OPTIONS = frozenset(
    {
        "--dead-time-tau-s",
        "--detector-scoring-mode",
        "--physics-profile",
        "--persistent",
        "--primary-sampling-fraction",
        "--target-sampled-primaries",
        "--request",
        "--response",
        "--scene",
        "--secondary-transport-mode",
        "--source-bias-cone-half-angle-deg",
        "--source-bias-isotropic-fraction",
        "--source-bias-mode",
        "--source-rate-model",
        "--threads",
    }
)

_MIN_PRIMARY_SAMPLING_FRACTION = 1.0e-6


def require_primary_sampling_fraction(
    value: object,
    *,
    accelerated_weighted_transport_enable: bool = False,
    target_sampled_primaries: int | None = None,
) -> float:
    """Validate primary sampling and require an explicit weighted-mode opt-in."""
    fraction = float(value)
    if (
        not np.isfinite(fraction)
        or fraction < _MIN_PRIMARY_SAMPLING_FRACTION
        or fraction > 1.0
    ):
        raise ValueError("primary_sampling_fraction must be in the interval [1e-6, 1].")
    weighted_requested = fraction < 1.0
    if weighted_requested and not accelerated_weighted_transport_enable:
        raise ValueError(
            "Geant4 runtime requires primary_sampling_fraction=1.0; "
            "weighted history thinning requires the explicit "
            "accelerated_weighted_transport_enable=true opt-in."
        )
    if (
        accelerated_weighted_transport_enable
        and not weighted_requested
        and target_sampled_primaries is None
    ):
        raise ValueError(
            "accelerated_weighted_transport_enable=true requires "
            "primary_sampling_fraction<1.0 or target_sampled_primaries."
        )
    return fraction


def require_target_sampled_primaries(value: object) -> int | None:
    """Return a positive integer primary budget or the disabled sentinel."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("target_sampled_primaries must be a positive JSON integer.")
    if value <= 0:
        raise ValueError("target_sampled_primaries must be a positive JSON integer.")
    return int(value)


def require_full_history_primary_sampling_fraction(value: object) -> float:
    """Return a full-history fraction or reject weighted history thinning."""
    return require_primary_sampling_fraction(value)


def resolve_primary_sampling_fraction(
    maximum_fraction: float,
    target_sampled_primaries: int | None,
    expected_unthinned_primaries: float,
) -> tuple[float, str]:
    """Resolve an observation-specific sampling fraction and provenance label."""
    if target_sampled_primaries is None:
        return float(maximum_fraction), "fixed_fraction"
    if (
        not np.isfinite(expected_unthinned_primaries)
        or expected_unthinned_primaries < 0.0
    ):
        raise RuntimeError("Native Geant4 expected-primary provenance is invalid.")
    budget_fraction = (
        float(target_sampled_primaries) / expected_unthinned_primaries
        if expected_unthinned_primaries > 0.0
        else np.inf
    )
    if budget_fraction < maximum_fraction:
        return (
            float(
                np.clip(
                    budget_fraction,
                    _MIN_PRIMARY_SAMPLING_FRACTION,
                    1.0,
                )
            ),
            "target_budget_limited",
        )
    return float(maximum_fraction), "maximum_fraction_limited"


def validate_geant4_executable_args(values: tuple[str, ...]) -> tuple[str, ...]:
    """Reject executable arguments that override engine-managed fidelity options."""
    for value in values:
        option = str(value).split("=", maxsplit=1)[0]
        if option in _MANAGED_GEANT4_EXECUTABLE_OPTIONS:
            raise ValueError(
                f"executable_args cannot override managed Geant4 option {option}."
            )
    return values


def geant4_executable_float_option(
    values: tuple[str, ...],
    option: str,
    *,
    default: float,
) -> float:
    """Return the last numeric value supplied for one native option."""
    result = float(default)
    for index, value in enumerate(values):
        token = str(value)
        if token == option:
            if index + 1 >= len(values):
                raise ValueError(f"{option} requires a numeric value.")
            try:
                result = float(values[index + 1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{option} requires a numeric value.") from exc
        elif token.startswith(f"{option}="):
            try:
                result = float(token.split("=", maxsplit=1)[1])
            except ValueError as exc:
                raise ValueError(f"{option} requires a numeric value.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{option} requires a finite numeric value.")
    return result


def _required_metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    """Return a strict boolean provenance value or fail closed."""
    if key not in metadata:
        raise RuntimeError(f"Native Geant4 response is missing {key} provenance.")
    value = metadata[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise RuntimeError(f"Native Geant4 response has invalid boolean {key}={value!r}.")


def _validate_native_weighted_response(
    spectrum_counts: object,
    metadata: dict[str, Any],
) -> None:
    """Validate weighted native spectrum and per-bin sum-w2 consistency."""
    try:
        spectrum = np.asarray(spectrum_counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Native Geant4 weighted spectrum is not numeric.") from exc
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise RuntimeError(
            "Native Geant4 weighted spectrum must be a nonempty one-dimensional array."
        )
    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("Native Geant4 weighted spectrum is not finite.")
    if np.any(spectrum < 0.0):
        raise RuntimeError("Native Geant4 weighted spectrum contains negative counts.")

    try:
        reported_spectrum_total = float(metadata["total_spectrum_counts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing total_spectrum_counts."
        ) from exc
    spectrum_total = float(np.sum(spectrum, dtype=float))
    if not np.isfinite(reported_spectrum_total) or reported_spectrum_total < 0.0:
        raise RuntimeError("Native Geant4 weighted total_spectrum_counts is invalid.")
    if not np.isclose(
        spectrum_total,
        reported_spectrum_total,
        rtol=1.0e-9,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "Native Geant4 weighted spectrum sum is inconsistent with "
            "total_spectrum_counts."
        )

    try:
        spectrum_variance = np.asarray(
            metadata["spectrum_count_variance"],
            dtype=float,
        )
    except KeyError as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing per-bin spectrum variance."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is not numeric."
        ) from exc
    if spectrum_variance.shape != spectrum.shape:
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance shape does not match "
            "the spectrum."
        )
    if not np.all(np.isfinite(spectrum_variance)):
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is not finite."
        )
    if np.any(spectrum_variance < 0.0):
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is negative."
        )

    variance_total = float(np.sum(spectrum_variance, dtype=float))
    try:
        parsed_variance_total = float(metadata["spectrum_count_variance_total"])
        reported_variance_total = float(metadata["weighted_spectrum_sumw2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing aggregate sum-w2 provenance."
        ) from exc
    if not np.isfinite(parsed_variance_total) or parsed_variance_total < 0.0:
        raise RuntimeError("Native Geant4 spectrum_count_variance_total is invalid.")
    for reported_total, label in (
        (parsed_variance_total, "spectrum_count_variance_total"),
        (reported_variance_total, "weighted_spectrum_sumw2"),
    ):
        if not np.isfinite(reported_total) or reported_total < 0.0:
            raise RuntimeError(f"Native Geant4 {label} is invalid.")
        if not np.isclose(
            variance_total,
            reported_total,
            rtol=1.0e-9,
            atol=1.0e-8,
        ):
            raise RuntimeError(
                "Native Geant4 weighted per-bin spectrum variance sum is "
                f"inconsistent with {label}."
            )


def validate_transport_metadata(
    metadata: dict[str, Any],
    *,
    expected_primary_sampling_fraction: float = 1.0,
    expected_target_sampled_primaries: int | None = None,
    accelerated_weighted_transport_enable: bool = False,
    expected_source_rate_model: str | None = None,
    expected_thread_count: int | None = None,
    expected_physics_profile: str | None = None,
    expected_detector_scoring_mode: str | None = None,
    expected_secondary_transport_mode: str | None = None,
    expected_source_bias_mode: str | None = None,
    expected_background_cps: float | None = None,
    expected_dead_time_tau_s: float | None = None,
) -> None:
    """Fail when native transport provenance disagrees with configured semantics."""
    configured_fraction = require_primary_sampling_fraction(
        expected_primary_sampling_fraction,
        accelerated_weighted_transport_enable=(accelerated_weighted_transport_enable),
        target_sampled_primaries=expected_target_sampled_primaries,
    )
    expected_target = require_target_sampled_primaries(
        expected_target_sampled_primaries
    )
    if expected_target is not None and not accelerated_weighted_transport_enable:
        raise ValueError(
            "expected_target_sampled_primaries requires "
            "accelerated_weighted_transport_enable=true."
        )
    try:
        observed_fraction = float(metadata["primary_sampling_fraction"])
        observed_history_weight = float(metadata["primary_history_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 response is missing sampling-fraction provenance."
        ) from exc
    if (
        not np.isfinite(observed_fraction)
        or observed_fraction < _MIN_PRIMARY_SAMPLING_FRACTION
        or observed_fraction > 1.0
    ):
        raise RuntimeError("Native Geant4 primary sampling fraction is invalid.")
    if expected_target is None and not np.isclose(
        observed_fraction,
        configured_fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 response requires "
            f"primary_sampling_fraction={configured_fraction}, "
            f"got {observed_fraction}."
        )
    expected_observed_weight = 1.0 / observed_fraction
    if not np.isfinite(observed_history_weight) or not np.isclose(
        observed_history_weight,
        expected_observed_weight,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        if expected_target is None:
            raise RuntimeError(
                "Native Geant4 response requires "
                f"primary_history_weight={1.0 / configured_fraction}, "
                f"got {observed_history_weight}."
            )
        raise RuntimeError(
            "Native Geant4 primary sampling fraction and history weight are "
            "invalid or inconsistent."
        )

    if str(metadata.get("backend", "")) != "geant4":
        raise RuntimeError("Native Geant4 response has invalid backend provenance.")
    if str(metadata.get("engine_mode", "")) != "external":
        raise RuntimeError("Native Geant4 response has invalid engine_mode provenance.")

    source_rate_model = str(metadata.get("source_rate_model", ""))
    if expected_source_rate_model is not None and source_rate_model != str(
        expected_source_rate_model
    ):
        raise RuntimeError(
            "Native Geant4 source-rate semantics disagree with runtime config: "
            f"expected {expected_source_rate_model}, got {source_rate_model or 'missing'}."
        )
    if source_rate_model not in {
        "detector_cps_1m",
        "isotropic_emission_equivalent",
    }:
        raise RuntimeError(
            "Native Geant4 response has invalid source_rate_model provenance."
        )
    history_thinning_enabled = observed_fraction < 1.0
    if history_thinning_enabled and source_rate_model != "detector_cps_1m":
        raise RuntimeError(
            "Accelerated weighted history thinning is currently restricted to "
            "source_rate_model=detector_cps_1m."
        )
    if str(metadata.get("intensity_cps_1m_definition", "")) != (
        "net_detector_count_rate_at_1m"
    ):
        raise RuntimeError(
            "Native Geant4 response has invalid intensity_cps_1m semantics."
        )

    physics_profile = str(metadata.get("physics_profile", ""))
    if expected_physics_profile is not None and physics_profile != str(
        expected_physics_profile
    ):
        raise RuntimeError(
            "Native Geant4 physics profile disagrees with runtime config: "
            f"expected {expected_physics_profile}, got {physics_profile or 'missing'}."
        )
    detector_scoring_mode = str(metadata.get("detector_scoring_mode", ""))
    if expected_detector_scoring_mode is not None and detector_scoring_mode != str(
        expected_detector_scoring_mode
    ):
        raise RuntimeError(
            "Native Geant4 detector scoring disagrees with runtime config: "
            f"expected {expected_detector_scoring_mode}, "
            f"got {detector_scoring_mode or 'missing'}."
        )
    detector_response_applied = _required_metadata_bool(
        metadata,
        "detector_response_applied_in_native",
    )
    if detector_response_applied != (detector_scoring_mode != "incident_gamma_energy"):
        raise RuntimeError(
            "Native Geant4 detector-response provenance is inconsistent with "
            "detector_scoring_mode."
        )

    secondary_transport_mode = str(metadata.get("secondary_transport_mode", ""))
    if (
        expected_secondary_transport_mode is not None
        and secondary_transport_mode != str(expected_secondary_transport_mode)
    ):
        raise RuntimeError(
            "Native Geant4 secondary transport disagrees with runtime config: "
            f"expected {expected_secondary_transport_mode}, "
            f"got {secondary_transport_mode or 'missing'}."
        )
    gamma_only = _required_metadata_bool(
        metadata,
        "gamma_only_secondary_transport",
    )
    if gamma_only != (secondary_transport_mode == "gamma_only"):
        raise RuntimeError(
            "Native Geant4 gamma-only provenance is inconsistent with "
            "secondary_transport_mode."
        )
    if _required_metadata_bool(metadata, "theory_tvl_attenuation"):
        raise RuntimeError("Native Geant4 runtime must not use theory-TVL attenuation.")
    if not _required_metadata_bool(metadata, "poisson_background"):
        raise RuntimeError(
            "Native Geant4 runtime must use Poisson background sampling."
        )
    try:
        background_cps = float(metadata["background_cps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 response is missing valid background_cps provenance."
        ) from exc
    if not np.isfinite(background_cps) or background_cps < 0.0:
        raise RuntimeError("Native Geant4 background_cps provenance is invalid.")
    if expected_background_cps is not None and not np.isclose(
        background_cps,
        float(expected_background_cps),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 background rate disagrees with runtime config: "
            f"expected {expected_background_cps}, got {background_cps}."
        )

    try:
        requested_threads = int(metadata["requested_threads"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 response is missing requested_threads provenance."
        ) from exc
    if expected_thread_count is not None and requested_threads != int(
        expected_thread_count
    ):
        raise RuntimeError(
            "Native Geant4 thread count disagrees with runtime config: "
            f"expected {expected_thread_count}, got {requested_threads}."
        )
    multithreaded = _required_metadata_bool(
        metadata,
        "multithreaded_run_manager",
    )
    if requested_threads > 1 and not multithreaded:
        raise RuntimeError(
            "Native Geant4 did not use a multithreaded run manager for a "
            f"{requested_threads}-thread runtime request."
        )

    if source_rate_model == "detector_cps_1m":
        expected_key = "expected_detector_equivalent_primaries"
        expected_semantics = "detector_equivalent_histories"
        expected_emission_model = "detector_equivalent_cone"
        expected_line_normalization = True
        resolved_source_bias_mode = "detector_cone"
        expected_source_bias_weighting = False
    else:
        expected_key = "expected_physical_primaries"
        expected_semantics = "isotropic_physical_histories"
        resolved_source_bias_mode = str(
            expected_source_bias_mode
            if expected_source_bias_mode is not None
            else metadata.get("source_bias_mode", "")
        )
        expected_source_bias_weighting = resolved_source_bias_mode == "detector_cone"
        expected_emission_model = (
            "weighted_isotropic" if expected_source_bias_weighting else "isotropic"
        )
        expected_line_normalization = False
    if (
        expected_source_bias_mode is not None
        and str(metadata.get("source_bias_mode", "")) != resolved_source_bias_mode
    ):
        raise RuntimeError(
            "Native Geant4 source bias disagrees with runtime config: "
            f"expected {resolved_source_bias_mode}, "
            f"got {metadata.get('source_bias_mode', 'missing')}."
        )
    if str(metadata.get("emission_model", "")) != expected_emission_model:
        raise RuntimeError(
            "Native Geant4 emission-model provenance disagrees with source-rate "
            "semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "line_intensities_normalized",
        )
        != expected_line_normalization
    ):
        raise RuntimeError(
            "Native Geant4 line-intensity normalization disagrees with "
            "source-rate semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "source_bias_weighted_transport",
        )
        != expected_source_bias_weighting
    ):
        raise RuntimeError(
            "Native Geant4 source-bias weighting disagrees with source-rate semantics."
        )
    expected_tally_weighted = history_thinning_enabled or expected_source_bias_weighting
    if (
        _required_metadata_bool(
            metadata,
            "weighted_transport",
        )
        != expected_tally_weighted
    ):
        raise RuntimeError(
            "Native Geant4 aggregate weighting provenance disagrees with "
            "configured transport semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "transport_tally_weighted",
        )
        != expected_tally_weighted
    ):
        raise RuntimeError(
            "Native Geant4 tally-weighting provenance disagrees with configured "
            "transport semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "history_thinning_enabled",
        )
        != history_thinning_enabled
    ):
        raise RuntimeError(
            "Native Geant4 history-thinning provenance disagrees with configured "
            "transport semantics."
        )
    expected_history_mode = (
        "weighted_thinning" if history_thinning_enabled else "full_unit_weight"
    )
    if str(metadata.get("transport_history_mode", "")) != expected_history_mode:
        raise RuntimeError(
            "Native Geant4 transport_history_mode disagrees with configured "
            f"transport semantics: expected {expected_history_mode}."
        )
    if str(metadata.get("spectrum_variance_semantics", "")) != (
        "compound_poisson_sumw2_includes_counting"
    ):
        raise RuntimeError(
            "Native Geant4 response is missing weighted sumw2 variance semantics."
        )
    if str(metadata.get("spectrum_variance_dead_time_propagation", "")) != (
        "fixed_observed_scale"
    ):
        raise RuntimeError(
            "Native Geant4 response has invalid dead-time variance provenance."
        )
    try:
        dead_time_tau_s = float(metadata["dead_time_tau_s"])
        dead_time_observed_scale = float(metadata["dead_time_observed_scale"])
        dwell_time_s = float(metadata["dwell_time_s"])
        pre_dead_time_counts = float(metadata["pre_dead_time_total_spectrum_counts"])
        pre_dead_time_sumw2 = float(metadata["pre_dead_time_weighted_spectrum_sumw2"])
        post_dead_time_sumw2 = float(metadata["weighted_spectrum_sumw2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 response is missing valid dead-time variance provenance."
        ) from exc
    dead_time_values = (
        dead_time_tau_s,
        dead_time_observed_scale,
        dwell_time_s,
        pre_dead_time_counts,
        pre_dead_time_sumw2,
        post_dead_time_sumw2,
    )
    if not all(np.isfinite(value) for value in dead_time_values):
        raise RuntimeError("Native Geant4 dead-time variance provenance is not finite.")
    if (
        dead_time_tau_s < 0.0
        or dead_time_observed_scale <= 0.0
        or dead_time_observed_scale > 1.0
        or dwell_time_s <= 0.0
        or pre_dead_time_counts < 0.0
        or pre_dead_time_sumw2 < 0.0
        or post_dead_time_sumw2 < 0.0
    ):
        raise RuntimeError("Native Geant4 dead-time variance provenance is invalid.")
    if expected_dead_time_tau_s is not None and not np.isclose(
        dead_time_tau_s,
        float(expected_dead_time_tau_s),
        rtol=1.0e-12,
        atol=1.0e-18,
    ):
        raise RuntimeError(
            "Native Geant4 dead-time constant disagrees with runtime config: "
            f"expected {expected_dead_time_tau_s}, got {dead_time_tau_s}."
        )
    expected_dead_time_scale = 1.0 / (
        1.0 + pre_dead_time_counts * dead_time_tau_s / dwell_time_s
    )
    if not np.isclose(
        dead_time_observed_scale,
        expected_dead_time_scale,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "Native Geant4 dead-time scale is inconsistent with its count-rate "
            "provenance."
        )
    if not np.isclose(
        post_dead_time_sumw2,
        pre_dead_time_sumw2 * dead_time_observed_scale**2,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Native Geant4 post-dead-time sumw2 is inconsistent with its raw "
            "variance provenance."
        )
    try:
        expected_primaries = float(metadata[expected_key])
        expected_unthinned_primaries = float(metadata["expected_unthinned_primaries"])
        sampled_primaries = float(metadata["expected_sampled_primaries"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 response is missing expected-primary provenance."
        ) from exc
    if (
        not np.isfinite(expected_primaries)
        or not np.isfinite(expected_unthinned_primaries)
        or not np.isfinite(sampled_primaries)
    ):
        raise RuntimeError("Native Geant4 expected-primary provenance is not finite.")
    if not np.isclose(
        expected_unthinned_primaries,
        expected_primaries,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Native Geant4 generic and source-model-specific unthinned primary "
            "expectations disagree."
        )
    resolved_fraction, expected_resolution = resolve_primary_sampling_fraction(
        configured_fraction,
        expected_target,
        expected_unthinned_primaries,
    )
    if not np.isclose(
        observed_fraction,
        resolved_fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 resolved primary sampling fraction disagrees with "
            f"the configured budget: expected {resolved_fraction}, "
            f"got {observed_fraction}."
        )
    budget_keys_present = any(
        key in metadata
        for key in (
            "requested_primary_sampling_fraction",
            "target_sampled_primaries",
            "primary_sampling_budget_enabled",
            "primary_sampling_fraction_resolution",
        )
    )
    if expected_target is not None or budget_keys_present:
        try:
            reported_requested_fraction = float(
                metadata["requested_primary_sampling_fraction"]
            )
            reported_target_raw = metadata["target_sampled_primaries"]
            if isinstance(reported_target_raw, bool):
                raise ValueError("boolean target")
            reported_target_value = float(reported_target_raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Native Geant4 response is missing primary-budget provenance."
            ) from exc
        if (
            not np.isfinite(reported_target_value)
            or reported_target_value < 0.0
            or not reported_target_value.is_integer()
        ):
            raise RuntimeError(
                "Native Geant4 target sampled-primary provenance is invalid."
            )
        reported_target = int(reported_target_value)
        if not np.isclose(
            reported_requested_fraction,
            configured_fraction,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Native Geant4 requested sampling fraction disagrees with runtime "
                "configuration."
            )
        expected_reported_target = int(expected_target or 0)
        if reported_target != expected_reported_target:
            raise RuntimeError(
                "Native Geant4 target sampled-primary provenance disagrees with "
                "runtime configuration."
            )
        if _required_metadata_bool(
            metadata,
            "primary_sampling_budget_enabled",
        ) != (expected_target is not None):
            raise RuntimeError(
                "Native Geant4 primary-budget enable provenance disagrees with "
                "runtime configuration."
            )
        if str(metadata.get("primary_sampling_fraction_resolution", "")) != (
            expected_resolution
        ):
            raise RuntimeError(
                "Native Geant4 sampling-fraction resolution provenance disagrees "
                f"with runtime configuration: expected {expected_resolution}."
            )
    expected_sampled = expected_primaries * observed_fraction
    if not np.isclose(
        sampled_primaries,
        expected_sampled,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Native Geant4 sampled-primary expectation does not match the "
            "configured sampling fraction."
        )
    semantics = str(metadata.get("expected_primary_semantics", ""))
    if semantics != expected_semantics:
        raise RuntimeError(
            "Native Geant4 expected-primary semantics disagree with source-rate "
            f"model: expected {expected_semantics}, got {semantics or 'missing'}."
        )
    if source_rate_model == "detector_cps_1m" and metadata.get(
        "expected_physical_primaries"
    ) not in (None, ""):
        raise RuntimeError(
            "Detector-equivalent history metadata must not be labelled as "
            "physical isotropic primaries."
        )


def validate_full_history_transport_metadata(
    metadata: dict[str, Any],
    *,
    expected_source_rate_model: str | None = None,
    expected_thread_count: int | None = None,
    expected_physics_profile: str | None = None,
    expected_detector_scoring_mode: str | None = None,
    expected_secondary_transport_mode: str | None = None,
    expected_source_bias_mode: str | None = None,
    expected_background_cps: float | None = None,
    expected_dead_time_tau_s: float | None = None,
) -> None:
    """Validate standard full-history native transport provenance."""
    validate_transport_metadata(
        metadata,
        expected_primary_sampling_fraction=1.0,
        accelerated_weighted_transport_enable=False,
        expected_source_rate_model=expected_source_rate_model,
        expected_thread_count=expected_thread_count,
        expected_physics_profile=expected_physics_profile,
        expected_detector_scoring_mode=expected_detector_scoring_mode,
        expected_secondary_transport_mode=expected_secondary_transport_mode,
        expected_source_bias_mode=expected_source_bias_mode,
        expected_background_cps=expected_background_cps,
        expected_dead_time_tau_s=expected_dead_time_tau_s,
    )


@dataclass(frozen=True)
class Geant4AppConfig:
    """Collect sidecar configuration relevant to the Geant4 app."""

    use_mock_stage: bool = True
    headless: bool = True
    renderer: str = "RayTracedLighting"
    usd_path: str | None = None
    detector_height_m: float = 0.5
    robot_ground_z_m: float = 0.0
    obstacle_height_m: float = 2.0
    author_obstacle_prims: bool | None = None
    author_room_boundary_prims: bool | None = None
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    stage_material_rules: tuple[StageMaterialRule, ...] = field(default_factory=tuple)
    engine_mode: str = "external"
    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.0
    executable_path: str | None = "build/geant4_sidecar"
    executable_args: tuple[str, ...] = field(default_factory=tuple)
    timeout_s: float = 120.0
    persistent_process: bool = False
    source_rate_model: str = "detector_cps_1m"
    source_bias_mode: str = "detector_cone"
    source_bias_cone_half_angle_deg: float = 0.0
    source_bias_isotropic_fraction: float = 0.1
    detector_scoring_mode: str = "full_transport"
    secondary_transport_mode: str = "full_transport"
    primary_sampling_fraction: float = 1.0
    target_sampled_primaries: int | None = None
    accelerated_weighted_transport_enable: bool = False
    background_cps: float = 0.0
    detector_model: ExportedDetectorModel = field(default_factory=ExportedDetectorModel)
    shield_thickness: ShieldThicknessConfig = field(
        default_factory=resolve_shield_thickness_config
    )
    absorbing_transport_groups: tuple[str, ...] = field(default_factory=tuple)
    absorbing_path_prefixes: tuple[str, ...] = field(default_factory=tuple)
    radiation_visualization: RadiationVisualizationConfig = field(
        default_factory=RadiationVisualizationConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Geant4AppConfig":
        """Normalize a JSON config payload into a strongly typed object."""
        payload = {} if data is None else dict(data)
        stage_material_rules_payload = payload.get("stage_material_rules", ())
        if not isinstance(stage_material_rules_payload, (list, tuple)):
            raise ValueError("stage_material_rules must be a list of objects.")
        detector_payload = payload.get("detector_model", {})
        if not isinstance(detector_payload, dict):
            raise ValueError("detector_model must be a JSON object.")
        visualization_payload = payload.get("radiation_visualization", {})
        if not isinstance(visualization_payload, dict):
            raise ValueError("radiation_visualization must be a JSON object.")
        executable_args = payload.get("executable_args", ())
        if not isinstance(executable_args, (list, tuple)):
            raise ValueError("executable_args must be a list of strings.")
        normalized_executable_args = validate_geant4_executable_args(
            tuple(str(value) for value in executable_args)
        )
        absorbing_transport_groups = payload.get("absorbing_transport_groups", ())
        if not isinstance(absorbing_transport_groups, (list, tuple)):
            raise ValueError("absorbing_transport_groups must be a list of strings.")
        absorbing_path_prefixes = payload.get("absorbing_path_prefixes", ())
        if not isinstance(absorbing_path_prefixes, (list, tuple)):
            raise ValueError("absorbing_path_prefixes must be a list of strings.")
        accelerated_weighted_transport_enable = payload.get(
            "accelerated_weighted_transport_enable",
            False,
        )
        if not isinstance(accelerated_weighted_transport_enable, bool):
            raise ValueError(
                "accelerated_weighted_transport_enable must be a JSON boolean."
            )
        target_sampled_primaries = require_target_sampled_primaries(
            payload.get("target_sampled_primaries")
        )
        if (
            target_sampled_primaries is not None
            and not accelerated_weighted_transport_enable
        ):
            raise ValueError(
                "target_sampled_primaries requires "
                "accelerated_weighted_transport_enable=true."
            )
        source_rate_model = str(payload.get("source_rate_model", "detector_cps_1m"))
        primary_sampling_fraction = require_primary_sampling_fraction(
            payload.get("primary_sampling_fraction", 1.0),
            accelerated_weighted_transport_enable=(
                accelerated_weighted_transport_enable
            ),
            target_sampled_primaries=target_sampled_primaries,
        )
        if (
            accelerated_weighted_transport_enable
            and source_rate_model != "detector_cps_1m"
        ):
            raise ValueError(
                "Accelerated weighted history thinning currently requires "
                "source_rate_model=detector_cps_1m."
            )
        return cls(
            use_mock_stage=bool(payload.get("use_mock_stage", True)),
            headless=bool(payload.get("headless", True)),
            renderer=str(payload.get("renderer", "RayTracedLighting")),
            usd_path=None
            if payload.get("usd_path") in (None, "")
            else str(payload["usd_path"]),
            detector_height_m=float(payload.get("detector_height_m", 0.5)),
            robot_ground_z_m=float(payload.get("robot_ground_z_m", 0.0)),
            obstacle_height_m=float(payload.get("obstacle_height_m", 2.0)),
            author_obstacle_prims=(
                None
                if payload.get("author_obstacle_prims") is None
                else bool(payload.get("author_obstacle_prims"))
            ),
            author_room_boundary_prims=(
                None
                if payload.get("author_room_boundary_prims") is None
                else bool(payload.get("author_room_boundary_prims"))
            ),
            fe_shield_size_xyz=tuple(
                float(v) for v in payload.get("fe_shield_size_xyz", (0.25, 0.08, 0.25))
            ),
            pb_shield_size_xyz=tuple(
                float(v) for v in payload.get("pb_shield_size_xyz", (0.25, 0.08, 0.25))
            ),
            stage_material_rules=tuple(
                StageMaterialRule(
                    path_prefix=str(entry["path_prefix"]),
                    material=str(entry["material"]),
                )
                for entry in stage_material_rules_payload
            ),
            engine_mode=str(payload.get("engine_mode", "external")),
            physics_profile=str(payload.get("physics_profile", "balanced")),
            thread_count=int(payload.get("thread_count", 1)),
            random_seed_base=int(payload.get("random_seed_base", 123)),
            dead_time_tau_s=float(payload.get("dead_time_tau_s", 5.813e-9)),
            scatter_gain=float(payload.get("scatter_gain", 0.0)),
            executable_path=(
                "build/geant4_sidecar"
                if payload.get("executable_path") in (None, "")
                else str(payload.get("executable_path"))
            ),
            executable_args=normalized_executable_args,
            timeout_s=float(payload.get("timeout_s", 120.0)),
            persistent_process=bool(payload.get("persistent_process", False)),
            source_rate_model=source_rate_model,
            source_bias_mode=str(payload.get("source_bias_mode", "detector_cone")),
            source_bias_cone_half_angle_deg=float(
                payload.get("source_bias_cone_half_angle_deg", 0.0)
            ),
            source_bias_isotropic_fraction=float(
                payload.get("source_bias_isotropic_fraction", 0.1)
            ),
            detector_scoring_mode=str(
                payload.get("detector_scoring_mode", "full_transport")
            ),
            secondary_transport_mode=str(
                payload.get("secondary_transport_mode", "full_transport")
            ),
            primary_sampling_fraction=primary_sampling_fraction,
            target_sampled_primaries=target_sampled_primaries,
            accelerated_weighted_transport_enable=(
                accelerated_weighted_transport_enable
            ),
            background_cps=max(
                geant4_executable_float_option(
                    normalized_executable_args,
                    "--background-cps",
                    default=0.0,
                ),
                0.0,
            ),
            detector_model=ExportedDetectorModel(
                crystal_radius_m=float(
                    detector_payload.get(
                        "crystal_radius_m", DEFAULT_DETECTOR_CRYSTAL_RADIUS_M
                    )
                ),
                crystal_length_m=float(
                    detector_payload.get(
                        "crystal_length_m", DEFAULT_DETECTOR_CRYSTAL_LENGTH_M
                    )
                ),
                housing_thickness_m=float(
                    detector_payload.get(
                        "housing_thickness_m",
                        DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
                    )
                ),
                crystal_shape=str(detector_payload.get("crystal_shape", "sphere")),
                crystal_material=str(detector_payload.get("crystal_material", "cebr3")),
                housing_material=str(
                    detector_payload.get("housing_material", "aluminum")
                ),
            ),
            shield_thickness=resolve_shield_thickness_config(payload),
            absorbing_transport_groups=tuple(
                str(v) for v in absorbing_transport_groups
            ),
            absorbing_path_prefixes=tuple(str(v) for v in absorbing_path_prefixes),
            radiation_visualization=RadiationVisualizationConfig.from_dict(
                visualization_payload
            ),
        )


class Geant4Application:
    """Wrap Geant4 sidecar scene handling and spectrum generation."""

    def __init__(
        self,
        *,
        app_config: dict[str, Any] | None = None,
        stage_backend: StageBackend | None = None,
    ) -> None:
        """Create the application and initialize the requested stage backend."""
        self.config = Geant4AppConfig.from_dict(app_config)
        self.scene = SceneDescription()
        self.asset_geometry = IsaacAssetGeometry(
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        backend = stage_backend
        if backend is None:
            if self.config.use_mock_stage:
                backend = FakeStageBackend()
            else:
                try:
                    backend = IsaacSimStageBackend(
                        headless=self.config.headless,
                        renderer=self.config.renderer,
                    )
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Geant4 use_mock_stage=false requires Isaac Sim Python modules. "
                        "Run the bridge with Isaac Sim's python.sh or set "
                        "ISAACSIM_PYTHON=/path/to/isaacsim/python.sh for auto-start."
                    ) from exc
        self._stage_backend = backend
        self.scene_builder = SceneBuilder(
            backend,
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        self.robot_controller = RobotController(
            backend,
            self.scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.engine = build_geant4_engine(
            Geant4EngineConfig(
                physics_profile=self.config.physics_profile,
                thread_count=self.config.thread_count,
                random_seed_base=self.config.random_seed_base,
                dead_time_tau_s=self.config.dead_time_tau_s,
                scatter_gain=self.config.scatter_gain,
                executable_path=self.config.executable_path,
                executable_args=self.config.executable_args,
                timeout_s=self.config.timeout_s,
                persistent_process=self.config.persistent_process,
                source_rate_model=self.config.source_rate_model,
                source_bias_mode=self.config.source_bias_mode,
                source_bias_cone_half_angle_deg=self.config.source_bias_cone_half_angle_deg,
                source_bias_isotropic_fraction=self.config.source_bias_isotropic_fraction,
                detector_scoring_mode=self.config.detector_scoring_mode,
                secondary_transport_mode=self.config.secondary_transport_mode,
                primary_sampling_fraction=self.config.primary_sampling_fraction,
                target_sampled_primaries=self.config.target_sampled_primaries,
                radiation_visualization=self.config.radiation_visualization,
            ),
            engine_mode=self.config.engine_mode,
        )
        self._last_cache_hit = False
        self._decomposer = SpectralDecomposer()

    def reset(self, scene: SceneDescription) -> None:
        """Load a new scene description and rebuild or reuse the Geant4 world."""
        if (
            scene.usd_path is None
            and scene.use_config_usd_fallback
            and self.config.usd_path is not None
        ):
            scene.usd_path = self.config.usd_path
        if self.config.author_obstacle_prims is not None:
            scene.author_obstacle_prims = self.config.author_obstacle_prims
        if self.config.author_room_boundary_prims is not None:
            scene.author_room_boundary_prims = self.config.author_room_boundary_prims
        self.scene = scene
        self.scene_builder.load_scene(scene, usd_path_override=None)
        self.robot_controller = RobotController(
            self._stage_backend,
            scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.robot_controller.reset()
        exported_scene = export_scene_for_geant4(
            scene,
            stage_backend=self._stage_backend,
            asset_geometry=self.asset_geometry,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
            stage_material_rules=self.config.stage_material_rules,
            absorbing_transport_groups=self.config.absorbing_transport_groups,
            absorbing_path_prefixes=self.config.absorbing_path_prefixes,
        )
        self._last_cache_hit = bool(self.engine.load_scene(exported_scene))

    def runtime_fidelity_metadata(self) -> dict[str, object]:
        """Return configured transport semantics for TCP reset handshakes."""
        weighted = bool(self.config.accelerated_weighted_transport_enable)
        budget_enabled = self.config.target_sampled_primaries is not None
        metadata: dict[str, object] = {
            "primary_sampling_fraction": float(self.config.primary_sampling_fraction),
            "primary_history_weight": float(
                1.0 / self.config.primary_sampling_fraction
            ),
            "accelerated_weighted_transport_enable": weighted,
            "requested_primary_sampling_fraction": float(
                self.config.primary_sampling_fraction
            ),
            "target_sampled_primaries": int(self.config.target_sampled_primaries or 0),
            "primary_sampling_budget_enabled": budget_enabled,
            "primary_sampling_fraction_resolution": (
                "per_observation_pending" if budget_enabled else "fixed_fraction"
            ),
            "dead_time_tau_s": float(self.config.dead_time_tau_s),
            "source_rate_model": str(self.config.source_rate_model),
            "requested_threads": int(self.config.thread_count),
            "physics_profile": str(self.config.physics_profile),
            "detector_scoring_mode": str(self.config.detector_scoring_mode),
            "secondary_transport_mode": str(self.config.secondary_transport_mode),
            "source_bias_mode": str(self.config.source_bias_mode),
            "background_cps": float(self.config.background_cps),
        }
        if budget_enabled:
            metadata["history_thinning_resolution"] = "per_observation_pending"
        else:
            history_thinning_enabled = self.config.primary_sampling_fraction < 1.0
            metadata["history_thinning_enabled"] = history_thinning_enabled
            metadata["transport_history_mode"] = (
                "weighted_thinning" if history_thinning_enabled else "full_unit_weight"
            )
        return metadata

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Apply a command and return the resulting Geant4-backed observation."""
        self.robot_controller.apply_command(command)
        detector_pose = self.robot_controller.detector_world_pose()
        fe_pose = self._stage_backend.get_world_pose(
            self.scene.prim_paths.fe_shield_path
        )
        pb_pose = self._stage_backend.get_world_pose(
            self.scene.prim_paths.pb_shield_path
        )
        spectrum, metadata = self.engine.simulate(
            Geant4StepRequest(
                step_id=command.step_id,
                dwell_time_s=float(command.dwell_time_s),
                seed=int(self.config.random_seed_base + int(command.step_id)),
                detector_pose_xyz=detector_pose.translation_xyz,
                detector_quat_wxyz=detector_pose.orientation_wxyz,
                fe_shield_pose_xyz=fe_pose.translation_xyz,
                fe_shield_quat_wxyz=fe_pose.orientation_wxyz,
                pb_shield_pose_xyz=pb_pose.translation_xyz,
                pb_shield_quat_wxyz=pb_pose.orientation_wxyz,
            )
        )
        metadata = dict(metadata)
        validate_transport_metadata(
            metadata,
            expected_primary_sampling_fraction=(self.config.primary_sampling_fraction),
            expected_target_sampled_primaries=(self.config.target_sampled_primaries),
            accelerated_weighted_transport_enable=(
                self.config.accelerated_weighted_transport_enable
            ),
            expected_source_rate_model=self.config.source_rate_model,
            expected_thread_count=self.config.thread_count,
            expected_physics_profile=self.config.physics_profile,
            expected_detector_scoring_mode=self.config.detector_scoring_mode,
            expected_secondary_transport_mode=self.config.secondary_transport_mode,
            expected_source_bias_mode=self.config.source_bias_mode,
            expected_background_cps=self.config.background_cps,
            expected_dead_time_tau_s=self.config.dead_time_tau_s,
        )
        if self.config.accelerated_weighted_transport_enable:
            _validate_native_weighted_response(spectrum, metadata)
        metadata["accelerated_weighted_transport_enable"] = bool(
            self.config.accelerated_weighted_transport_enable
        )
        metadata.setdefault("cache_hit", self._last_cache_hit)
        metadata.setdefault("fe_orientation_index", int(command.fe_orientation_index))
        metadata.setdefault("pb_orientation_index", int(command.pb_orientation_index))
        metadata.setdefault("shield_num_orientations", 8)
        metadata.setdefault(
            "shield_pair_id",
            int(command.fe_orientation_index) * 8 + int(command.pb_orientation_index),
        )
        metadata.setdefault(
            "shield_thickness_scale",
            float(self.config.shield_thickness.thickness_scale),
        )
        metadata.setdefault(
            "shield_thickness_fe_cm",
            float(self.config.shield_thickness.thickness_fe_cm),
        )
        metadata.setdefault(
            "shield_thickness_pb_cm",
            float(self.config.shield_thickness.thickness_pb_cm),
        )
        energy = self._decomposer.energy_axis
        bin_width_keV = float(self._decomposer.config.bin_width_keV)
        edges = list(energy) + [float(energy[-1] + bin_width_keV)]
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=detector_pose.translation_xyz,
            detector_quat_wxyz=detector_pose.orientation_wxyz,
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.asarray(spectrum, dtype=float).tolist(),
            energy_bin_edges_keV=[float(v) for v in edges],
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the underlying engine and stage backend."""
        self.engine.close()
        self._stage_backend.close()
