"""Geometry-conditioned full-spectrum model for the pure particle filter.

The model keeps source contributions separate until their physical direct and
scattered incident-gamma spectra have been formed.  Detector-response marking
is applied once, background is added once, and nonparalyzable detector dead
time is represented by a renewal total-count law with conditional multinomial
energy marks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from measurement.source_boundary import surface_emission_policy_sha256
from measurement.shielding import line_resolved_shield_mu_by_isotope
from runtime.forward_model_manifest import resolve_file_backed_model_asset
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    AdditiveNoncollidedTransportResponse,
)
from spectrum.library import default_library
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    build_native_geant4_detector_response_matrix,
    native_geant4_background_shape,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)


ELECTRON_REST_ENERGY_KEV = 510.99895
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
AIR_DENSITY_G_CM3 = 1.225e-3
AIR_EFFECTIVE_Z_OVER_A = 0.49919
IRON_DENSITY_G_CM3 = 7.874
IRON_Z_OVER_A = 26.0 / 55.845
LEAD_DENSITY_G_CM3 = 11.34
LEAD_Z_OVER_A = 82.0 / 207.2
TRANSPORT_FEATURE_ORDER = (
    "tau_fe",
    "tau_pb",
    "tau_obstacle",
    "distance_m",
)
BIRTH_PROPOSAL_WORKING_SET_BYTES = 64 * 1024 * 1024
CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE = 1
CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE = 64
CROSS_LIKELIHOOD_STATE_CHUNK_SIZE = 8
CROSS_LIKELIHOOD_BIN_CHUNK_SIZE = 128
RENEWAL_LOG_GAMMA_MAX_ITERATIONS = 2_048
DESIGNATED_TRAINING_SCENE_SEEDS = (2026072701, 2026072702, 2026072703)
DESIGNATED_HOLDOUT_SCENE_SEEDS = (2026072791, 2026072792)
RATE_SCALE_HALF_WIDTH_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
RATE_SCALE_MIXTURE_WEIGHTS = (0.25, 0.50, 0.25)
MARK_CONCENTRATION_GRID = (
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    100_000.0,
)
VALIDATION_SCENARIO_IDS = (
    "background_only",
    "single_line_source_resolved",
    "dominant_plus_absent_isotope",
    "multi_isotope_superposition",
    "continuous_surface_perturbation_ranking",
)
ACCEPTANCE_METRIC_CONTRACT = MappingProxyType(
    {
        "native_response_max_abs_error": ("le", 1.0e-12),
        "native_deadtime_mean_abs_z": ("le", 4.0),
        "native_deadtime_fano_relative_error": ("le", 0.05),
        "cpu_torch_mean_max_abs_error": ("le", 1.0e-8),
        "cpu_torch_log_likelihood_max_abs_error": ("le", 1.0e-6),
        "background_pairwise_95_coverage_fraction": ("ge", 0.85),
        "background_k_positive_decision_rate_at_p0p95": ("le", 0.05),
        "single_source_pairwise_95_coverage_fraction": ("ge", 0.80),
        "dominant_absent_pairwise_95_coverage_fraction": ("ge", 0.80),
        "absent_isotope_k_positive_decision_rate_at_p0p95": ("le", 0.05),
        "superposition_pairwise_95_coverage_fraction": ("ge", 0.80),
        "truth_vs_perturbed_ranking_fraction": ("ge", 0.80),
        "pairwise_standardized_total_abs_q95": ("le", 3.0),
        "pairwise_mark_tail_ge_0p01_fraction": ("ge", 0.80),
        "renewal_total_randomized_pit_ks_pvalue": ("ge", 0.01),
        "conditional_mark_randomized_pit_ks_pvalue": ("ge", 0.01),
        "line_count_conservation_max_relative_error": ("le", 1.0e-12),
        "validation_label_production_influence_max_abs": ("le", 0.0),
    }
)


def full_spectrum_acceptance_contract_payload() -> Mapping[str, object]:
    """Return the predeclared validation split, scenarios, and thresholds."""
    return {
        "schema_version": 1,
        "contract_id": "geometry_conditioned_full_spectrum_acceptance_v1",
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "holdout_scene_seeds": list(DESIGNATED_HOLDOUT_SCENE_SEEDS),
        "shield_pair_ids": list(range(64)),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "metrics": {
            metric_id: {
                "comparison": comparison,
                "threshold": float(threshold),
            }
            for metric_id, (comparison, threshold) in (
                ACCEPTANCE_METRIC_CONTRACT.items()
            )
        },
        "training_only_discrepancy_selection": {
            "rate_scale_family": (
                "station_shared_three_node_symmetric_mean_one"
            ),
            "rate_scale_half_width_grid": list(
                RATE_SCALE_HALF_WIDTH_GRID
            ),
            "rate_scale_weights": list(RATE_SCALE_MIXTURE_WEIGHTS),
            "mark_family": (
                "source_fraction_dirichlet_multinomial"
            ),
            "mark_concentration_source_grid": list(
                MARK_CONCENTRATION_GRID
            ),
            "objective": (
                "maximum_joint_training_log_predictive_density"
            ),
            "tie_break": (
                "smallest_rate_half_width_then_largest_mark_concentration"
            ),
            "scope": (
                "one_global_parameter_pair_for_all_scenes_pairs_isotopes"
            ),
        },
        "selection_policy": (
            "thresholds_fixed_before_holdout_no_holdout_tuning"
        ),
    }


def _freeze_json_value(value: object) -> object:
    """Return an immutable recursively copied JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Validation and line manifests must be JSON-compatible.")


def _thaw_json_value(value: object) -> object:
    """Return a detached mutable JSON-compatible copy."""
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON-compatible value."""
    return hashlib.sha256(
        json.dumps(
            _thaw_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_nonfinite_json_constant(value: str) -> None:
    """Reject non-standard non-finite constants in model JSON."""
    raise ValueError(
        "Full-spectrum model JSON must contain finite standard-JSON numbers; "
        f"found {value}."
    )


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one model JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Full-spectrum model JSON contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _strict_json_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _strict_json_number_sequence(
    value: object,
    *,
    field_name: str,
) -> tuple[float, ...]:
    """Return one nonempty finite JSON-number sequence without coercion."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise TypeError(f"{field_name} must be a nonempty JSON array.")
    return tuple(
        _strict_json_number(
            item,
            field_name=f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )


FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256 = _canonical_json_sha256(
    full_spectrum_acceptance_contract_payload()
)


def rate_scale_mixture_for_half_width(
    half_width: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the predeclared symmetric positive mean-one scale mixture."""
    width = float(half_width)
    if width not in RATE_SCALE_HALF_WIDTH_GRID:
        raise ValueError(
            "Rate-scale half width must belong to the predeclared training grid."
        )
    return (
        (1.0 - width, 1.0, 1.0 + width),
        RATE_SCALE_MIXTURE_WEIGHTS,
    )


def _is_sha256(value: object) -> bool:
    """Return whether one value is a lowercase hexadecimal SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_digest(array: NDArray[np.float64]) -> bytes:
    """Return shape-sensitive bytes for one immutable float64 array."""
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.digest()


def _logdiffexp_numpy(
    log_large: NDArray[np.float64],
    log_small: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``log(exp(log_large) - exp(log_small))`` stably."""
    large = np.asarray(log_large, dtype=np.float64)
    small = np.asarray(log_small, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.minimum(small - large, 0.0)
        result = large + np.log(-np.expm1(delta))
    return np.where(small < large, result, -np.inf)


def _logdiffexp_torch(log_large: object, log_small: object) -> object:
    """Return the Torch stable logarithm of a positive exponential difference."""
    import torch

    delta = torch.minimum(log_small - log_large, torch.zeros_like(log_large))
    result = log_large + torch.log(-torch.expm1(delta))
    return torch.where(log_small < log_large, result, -torch.inf)


def _renewal_positive_decomposition_numpy(
    counts: NDArray[np.float64],
    first_arguments: NDArray[np.float64],
    second_arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the positive-term renewal PMF decomposition in log space.

    For integer ``m >= 1`` and ``a >= b >= 0``, the renewal probability is

    ``[P(m, a) - P(m, b)] + exp(-b) b**m / Gamma(m + 1)``.

    Both terms are nonnegative.  The first is evaluated from log regularized
    gamma tails, using a convergent positive series below the mode and a
    continued fraction above it.  This path is used only when the faster
    ordinary regularized-gamma CDF/SF subtraction has underflowed.
    """
    m, first, second = np.broadcast_arrays(
        np.asarray(counts, dtype=np.float64),
        np.asarray(first_arguments, dtype=np.float64),
        np.asarray(second_arguments, dtype=np.float64),
    )
    log_interval = np.full(m.shape, -np.inf, dtype=np.float64)
    lower = first < m + 1.0
    if np.any(lower):
        log_interval[lower] = _logdiffexp_numpy(
            _regularized_gamma_lower_log_numpy(
                m[lower],
                first[lower],
            ),
            _regularized_gamma_lower_log_numpy(
                m[lower],
                second[lower],
            ),
        )
    upper = second >= m + 1.0
    if np.any(upper):
        log_interval[upper] = _logdiffexp_numpy(
            _regularized_gamma_upper_log_numpy(
                m[upper],
                second[upper],
            ),
            _regularized_gamma_upper_log_numpy(
                m[upper],
                first[upper],
            ),
        )
    central = ~(lower | upper)
    if np.any(central):
        central_probability = (
            special.gammainc(m[central], first[central])
            - special.gammainc(m[central], second[central])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_interval[central] = np.log(central_probability)
    log_boundary = (
        special.xlogy(m, second)
        - second
        - special.gammaln(m + 1.0)
    )
    return np.asarray(
        np.logaddexp(log_interval, log_boundary),
        dtype=np.float64,
    )


def _regularized_gamma_lower_log_numpy(
    shape: NDArray[np.float64],
    argument: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return log regularized lower gamma by its positive convergent series."""
    a, x = np.broadcast_arrays(
        np.asarray(shape, dtype=np.float64),
        np.asarray(argument, dtype=np.float64),
    )
    term = np.ones(a.shape, dtype=np.float64)
    series = np.ones(a.shape, dtype=np.float64)
    tolerance = 8.0 * np.finfo(np.float64).eps
    converged = np.zeros(a.shape, dtype=np.bool_)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        term *= x / (a + float(iteration))
        series += term
        converged = term <= tolerance * series
        if np.all(converged):
            break
    if not np.all(converged):
        raise RuntimeError(
            "Lower regularized-gamma log series did not converge."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            special.xlogy(a, x)
            - x
            - special.gammaln(a + 1.0)
            + np.log(series)
        )
    return np.where(x > 0.0, result, -np.inf)


def _regularized_gamma_upper_log_numpy(
    shape: NDArray[np.float64],
    argument: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return log regularized upper gamma by a continued fraction."""
    a, x = np.broadcast_arrays(
        np.asarray(shape, dtype=np.float64),
        np.asarray(argument, dtype=np.float64),
    )
    tolerance = 8.0 * np.finfo(np.float64).eps
    tiny = np.finfo(np.float64).tiny / np.finfo(np.float64).eps
    denominator = x + 1.0 - a
    c_value = np.full(a.shape, 1.0 / tiny, dtype=np.float64)
    d_value = 1.0 / denominator
    fraction = d_value.copy()
    converged = np.zeros(a.shape, dtype=np.bool_)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        coefficient = -float(iteration) * (float(iteration) - a)
        denominator += 2.0
        d_value = coefficient * d_value + denominator
        d_value = np.where(
            np.abs(d_value) < tiny,
            np.copysign(tiny, d_value),
            d_value,
        )
        c_value = denominator + coefficient / c_value
        c_value = np.where(
            np.abs(c_value) < tiny,
            np.copysign(tiny, c_value),
            c_value,
        )
        d_value = 1.0 / d_value
        delta = d_value * c_value
        fraction *= delta
        converged = np.abs(delta - 1.0) <= tolerance
        if np.all(converged):
            break
    if (
        not np.all(converged)
        or np.any(~np.isfinite(fraction))
        or np.any(fraction <= 0.0)
    ):
        raise RuntimeError(
            "Upper regularized-gamma continued fraction did not converge."
        )
    return np.asarray(
        -x
        + special.xlogy(a, x)
        - special.gammaln(a)
        + np.log(fraction),
        dtype=np.float64,
    )


def _renewal_positive_decomposition_torch(
    counts: object,
    first_arguments: object,
    second_arguments: object,
) -> object:
    """Return the Torch positive-term renewal PMF decomposition."""
    import torch

    m, first, second = torch.broadcast_tensors(
        counts,
        first_arguments,
        second_arguments,
    )
    log_interval = torch.full_like(m, -torch.inf)
    lower = first < m + 1.0
    if bool(torch.any(lower)):
        lower_interval = _logdiffexp_torch(
            _regularized_gamma_lower_log_torch(
                m[lower],
                first[lower],
            ),
            _regularized_gamma_lower_log_torch(
                m[lower],
                second[lower],
            ),
        )
        log_interval = log_interval.masked_scatter(lower, lower_interval)
    upper = second >= m + 1.0
    if bool(torch.any(upper)):
        upper_interval = _logdiffexp_torch(
            _regularized_gamma_upper_log_torch(
                m[upper],
                second[upper],
            ),
            _regularized_gamma_upper_log_torch(
                m[upper],
                first[upper],
            ),
        )
        log_interval = log_interval.masked_scatter(upper, upper_interval)
    central = ~(lower | upper)
    if bool(torch.any(central)):
        central_probability = (
            torch.special.gammainc(m[central], first[central])
            - torch.special.gammainc(m[central], second[central])
        )
        central_interval = torch.log(central_probability)
        log_interval = log_interval.masked_scatter(
            central,
            central_interval,
        )
    log_boundary = (
        torch.xlogy(m, second)
        - second
        - torch.lgamma(m + 1.0)
    )
    return torch.logaddexp(log_interval, log_boundary)


def _regularized_gamma_lower_log_torch(
    shape: object,
    argument: object,
) -> object:
    """Return the Torch log regularized lower gamma positive series."""
    import torch

    a, x = torch.broadcast_tensors(shape, argument)
    term = torch.ones_like(a)
    series = torch.ones_like(a)
    tolerance = 8.0 * torch.finfo(a.dtype).eps
    converged = torch.zeros_like(a, dtype=torch.bool)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        term = term * x / (a + float(iteration))
        series = series + term
        converged = term <= tolerance * series
        if bool(torch.all(converged)):
            break
    if not bool(torch.all(converged)):
        raise RuntimeError(
            "Torch lower regularized-gamma log series did not converge."
        )
    result = (
        torch.xlogy(a, x)
        - x
        - torch.lgamma(a + 1.0)
        + torch.log(series)
    )
    return torch.where(x > 0.0, result, -torch.inf)


def _regularized_gamma_upper_log_torch(
    shape: object,
    argument: object,
) -> object:
    """Return the Torch log regularized upper gamma continued fraction."""
    import torch

    a, x = torch.broadcast_tensors(shape, argument)
    tolerance = 8.0 * torch.finfo(a.dtype).eps
    tiny = torch.finfo(a.dtype).tiny / torch.finfo(a.dtype).eps
    denominator = x + 1.0 - a
    c_value = torch.full_like(a, 1.0 / tiny)
    d_value = 1.0 / denominator
    fraction = d_value
    converged = torch.zeros_like(a, dtype=torch.bool)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        coefficient = -float(iteration) * (float(iteration) - a)
        denominator = denominator + 2.0
        d_value = coefficient * d_value + denominator
        d_value = torch.where(
            torch.abs(d_value) < tiny,
            torch.where(d_value >= 0.0, tiny, -tiny),
            d_value,
        )
        c_value = denominator + coefficient / c_value
        c_value = torch.where(
            torch.abs(c_value) < tiny,
            torch.where(c_value >= 0.0, tiny, -tiny),
            c_value,
        )
        d_value = 1.0 / d_value
        delta = d_value * c_value
        fraction = fraction * delta
        converged = torch.abs(delta - 1.0) <= tolerance
        if bool(torch.all(converged)):
            break
    if (
        not bool(torch.all(converged))
        or bool(torch.any(~torch.isfinite(fraction)))
        or bool(torch.any(fraction <= 0.0))
    ):
        raise RuntimeError(
            "Torch upper regularized-gamma continued fraction did not converge."
        )
    return (
        -x
        + torch.xlogy(a, x)
        - torch.lgamma(a)
        + torch.log(fraction)
    )


def nonparalyzable_count_log_probability_numpy(
    observed_counts: NDArray[np.float64],
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
) -> NDArray[np.float64]:
    """Return exact nonparalyzable renewal-count log probabilities.

    The detector starts live.  For ``m >= 1`` the probability is

    ``F_Gamma(m, rate; T-(m-1)tau) - F_Gamma(m+1, rate; T-m tau)``.

    The implementation switches between lower and upper regularized gamma
    tails before applying ``logdiffexp``.  At exactly zero dead time it uses
    the algebraically equivalent Poisson law directly.
    """
    counts, rates, live_times = np.broadcast_arrays(
        np.asarray(observed_counts, dtype=np.float64),
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    tau = float(dead_time_tau_s)
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(counts != np.floor(counts))
        or np.any(~np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(~np.isfinite(live_times))
        or np.any(live_times <= 0.0)
        or not np.isfinite(tau)
        or tau < 0.0
    ):
        raise ValueError(
            "Renewal counts/rates/live times must be finite with exact "
            "nonnegative integer counts, nonnegative rates, and positive times."
        )
    mean = rates * live_times
    if tau == 0.0:
        return np.asarray(
            special.xlogy(counts, mean)
            - mean
            - special.gammaln(counts + 1.0),
            dtype=np.float64,
        )
    result = np.full(counts.shape, -np.inf, dtype=np.float64)
    zero = counts == 0.0
    result[zero] = -mean[zero]
    positive = ~zero
    if not np.any(positive):
        return result
    m = counts[positive]
    rate = rates[positive]
    live = live_times[positive]
    raw_second_window = live - m * tau
    first_window = np.maximum(raw_second_window + tau, 0.0)
    second_window = np.maximum(raw_second_window, 0.0)
    first_argument = rate * first_window
    second_argument = rate * second_window
    lower_first = special.gammainc(m, first_argument)
    lower_second = special.gammainc(m + 1.0, second_argument)
    upper_first = special.gammaincc(m, first_argument)
    upper_second = special.gammaincc(m + 1.0, second_argument)
    with np.errstate(divide="ignore"):
        lower_log = _logdiffexp_numpy(
            np.log(lower_first),
            np.log(lower_second),
        )
        upper_log = _logdiffexp_numpy(
            np.log(upper_second),
            np.log(upper_first),
        )
    use_lower = lower_first <= 0.5
    selected = np.where(use_lower, lower_log, upper_log)
    fallback = ~np.isfinite(selected)
    if np.any(fallback):
        recovered = _renewal_positive_decomposition_numpy(
            m[fallback],
            first_argument[fallback],
            second_argument[fallback],
        )
        if np.any(np.isnan(recovered)) or np.any(np.isposinf(recovered)):
            raise RuntimeError(
                "Positive-term renewal likelihood recovery was invalid."
            )
        selected[fallback] = recovered
    result[positive] = selected
    return result


def nonparalyzable_count_log_probability_torch(
    observed_counts: object,
    incident_rates_cps: object,
    live_times_s: object,
    *,
    dead_time_tau_s: float,
) -> object:
    """Return the Torch equivalent renewal-count log probabilities."""
    import torch

    rates = torch.as_tensor(incident_rates_cps)
    if rates.dtype != torch.float64:
        raise TypeError("Production renewal likelihood requires torch.float64.")
    dtype = rates.dtype
    device = rates.device
    counts = torch.as_tensor(
        observed_counts,
        dtype=dtype,
        device=device,
    )
    live_times = torch.as_tensor(
        live_times_s,
        dtype=dtype,
        device=device,
    )
    counts, rates, live_times = torch.broadcast_tensors(
        counts,
        rates,
        live_times,
    )
    tau = float(dead_time_tau_s)
    invalid = (
        torch.any(~torch.isfinite(counts))
        or torch.any(counts < 0.0)
        or torch.any(counts != torch.floor(counts))
        or torch.any(~torch.isfinite(rates))
        or torch.any(rates < 0.0)
        or torch.any(~torch.isfinite(live_times))
        or torch.any(live_times <= 0.0)
        or not np.isfinite(tau)
        or tau < 0.0
    )
    if bool(invalid):
        raise ValueError("Torch renewal-count inputs are invalid.")
    mean = rates * live_times
    if tau == 0.0:
        safe_mean = torch.clamp(mean, min=torch.finfo(dtype).tiny)
        poisson = (
            torch.xlogy(counts, safe_mean)
            - mean
            - torch.lgamma(counts + 1.0)
        )
        return torch.where(
            (mean == 0.0) & (counts == 0.0),
            torch.zeros_like(poisson),
            torch.where(
                (mean == 0.0) & (counts > 0.0),
                torch.full_like(poisson, -torch.inf),
                poisson,
            ),
        )
    positive = counts > 0.0
    m = torch.where(positive, counts, torch.ones_like(counts))
    raw_second_window = live_times - m * tau
    first_window = torch.clamp(raw_second_window + tau, min=0.0)
    second_window = torch.clamp(raw_second_window, min=0.0)
    first_argument = rates * first_window
    second_argument = rates * second_window
    lower_first = torch.special.gammainc(m, first_argument)
    lower_second = torch.special.gammainc(m + 1.0, second_argument)
    upper_first = torch.special.gammaincc(m, first_argument)
    upper_second = torch.special.gammaincc(m + 1.0, second_argument)
    lower_log = _logdiffexp_torch(
        torch.log(lower_first),
        torch.log(lower_second),
    )
    upper_log = _logdiffexp_torch(
        torch.log(upper_second),
        torch.log(upper_first),
    )
    selected = torch.where(lower_first <= 0.5, lower_log, upper_log)
    fallback = positive & ~torch.isfinite(selected)
    if bool(torch.any(fallback)):
        recovered = _renewal_positive_decomposition_torch(
            m[fallback],
            first_argument[fallback],
            second_argument[fallback],
        )
        if bool(
            torch.any(torch.isnan(recovered))
            or torch.any(torch.isposinf(recovered))
        ):
            raise RuntimeError(
                "Torch positive-term renewal likelihood recovery was invalid."
            )
        selected = selected.masked_scatter(fallback, recovered)
    return torch.where(positive, selected, -mean)


def nonparalyzable_count_cdf_numpy(
    count_threshold: NDArray[np.int64],
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
) -> NDArray[np.float64]:
    """Return ``P(M <= m)`` for a nonparalyzable detector starting live."""
    threshold, rates, live_times = np.broadcast_arrays(
        np.asarray(count_threshold, dtype=np.int64),
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    tau = float(dead_time_tau_s)
    remaining = live_times - threshold.astype(np.float64) * tau
    argument = rates * np.maximum(remaining, 0.0)
    cdf = special.gammaincc(threshold.astype(np.float64) + 1.0, argument)
    cdf = np.where(threshold < 0, 0.0, cdf)
    cdf = np.where(remaining <= 0.0, 1.0, cdf)
    return np.asarray(np.clip(cdf, 0.0, 1.0), dtype=np.float64)


def sample_nonparalyzable_counts_numpy(
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
    sample_count: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw renewal totals by vectorized inverse-CDF integer bisection."""
    rates, live_times = np.broadcast_arrays(
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive.")
    if (
        np.any(~np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(~np.isfinite(live_times))
        or np.any(live_times <= 0.0)
    ):
        raise ValueError("Renewal sampling rates/times are invalid.")
    sample_shape = rates.shape + (int(sample_count),)
    expanded_rates = np.broadcast_to(rates[..., np.newaxis], sample_shape)
    expanded_times = np.broadcast_to(
        live_times[..., np.newaxis],
        sample_shape,
    )
    if float(dead_time_tau_s) == 0.0:
        return np.asarray(
            rng.poisson(expanded_rates * expanded_times),
            dtype=np.int64,
        )
    uniform = rng.random(sample_shape)
    poisson_mean = expanded_rates * expanded_times
    initial_high = np.ceil(
        poisson_mean + 10.0 * np.sqrt(poisson_mean + 1.0) + 10.0
    )
    if np.any(initial_high >= float(np.iinfo(np.int64).max // 2)):
        raise OverflowError("Renewal count support exceeds int64.")
    high = np.asarray(np.maximum(initial_high, 0.0), dtype=np.int64)
    support_maximum = np.asarray(
        np.floor(expanded_times / float(dead_time_tau_s)) + 1.0,
        dtype=np.int64,
    )
    high = np.minimum(high, support_maximum)
    high_cdf = nonparalyzable_count_cdf_numpy(
        high,
        expanded_rates,
        expanded_times,
        dead_time_tau_s=float(dead_time_tau_s),
    )
    unresolved = high_cdf < uniform
    while bool(np.any(unresolved)):
        expanded_high = np.minimum(
            np.maximum(2 * high + 1, 1),
            support_maximum,
        )
        if np.array_equal(expanded_high[unresolved], high[unresolved]):
            raise RuntimeError(
                "Renewal inverse-CDF upper support failed to bracket a draw."
            )
        high = np.where(unresolved, expanded_high, high)
        high_cdf = nonparalyzable_count_cdf_numpy(
            high,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=float(dead_time_tau_s),
        )
        unresolved = high_cdf < uniform
    low = np.full(sample_shape, -1, dtype=np.int64)
    while bool(np.any(high - low > 1)):
        midpoint = low + (high - low) // 2
        cdf = nonparalyzable_count_cdf_numpy(
            midpoint,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=float(dead_time_tau_s),
        )
        move_high = cdf >= uniform
        high = np.where(move_high, midpoint, high)
        low = np.where(move_high, low, midpoint)
    return high


def _klein_nishina_total_cross_section_cm2(
    energy_keV: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return total Klein-Nishina cross section per electron."""
    energy = np.asarray(energy_keV, dtype=np.float64)
    alpha = np.maximum(energy / ELECTRON_REST_ENERGY_KEV, 1.0e-12)
    log_term = np.log1p(2.0 * alpha)
    bracket = (
        (1.0 + alpha)
        / np.square(alpha)
        * (
            2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)
            - log_term / alpha
        )
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / np.square(1.0 + 2.0 * alpha)
    )
    return (
        2.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * np.maximum(bracket, 0.0)
    )


def _klein_nishina_transition_matrix(
    energy_axis_keV: NDArray[np.float64],
    *,
    quadrature_order: int,
) -> NDArray[np.float64]:
    """Build a column-stochastic single-Compton-scatter energy operator."""
    axis = np.asarray(energy_axis_keV, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("Klein-Nishina transition requires an increasing axis.")
    mu, quadrature_weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    incident = axis[:, np.newaxis]
    alpha = incident / ELECTRON_REST_ENERGY_KEV
    ratio = 1.0 / (1.0 + alpha * (1.0 - mu[np.newaxis, :]))
    scattered = incident * ratio
    differential = np.square(ratio) * (
        ratio
        + 1.0 / np.maximum(ratio, np.finfo(np.float64).tiny)
        - (1.0 - np.square(mu))[np.newaxis, :]
    )
    weights = np.maximum(
        differential * quadrature_weights[np.newaxis, :],
        0.0,
    )
    weights /= np.maximum(
        np.sum(weights, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    bin_width = float(axis[1] - axis[0])
    fractional = (scattered - float(axis[0])) / bin_width
    lower = np.floor(fractional).astype(np.int64)
    upper_fraction = fractional - lower
    lower = np.clip(lower, 0, axis.size - 1)
    upper = np.clip(lower + 1, 0, axis.size - 1)
    input_indices = np.broadcast_to(
        np.arange(axis.size, dtype=np.int64)[:, np.newaxis],
        lower.shape,
    )
    transition = np.zeros((axis.size, axis.size), dtype=np.float64)
    np.add.at(
        transition,
        (lower.reshape(-1), input_indices.reshape(-1)),
        (weights * (1.0 - upper_fraction)).reshape(-1),
    )
    np.add.at(
        transition,
        (upper.reshape(-1), input_indices.reshape(-1)),
        (weights * upper_fraction).reshape(-1),
    )
    column_sums = np.sum(transition, axis=0)
    zero_energy = axis <= 0.0
    transition[:, zero_energy] = 0.0
    transition[0, zero_energy] = 1.0
    column_sums = np.sum(transition, axis=0)
    transition /= np.maximum(
        column_sums[np.newaxis, :],
        np.finfo(np.float64).tiny,
    )
    return transition


def _line_order_shapes(
    energy_axis_keV: NDArray[np.float64],
    raw_bin_indices: NDArray[np.int64],
    *,
    maximum_scatter_order: int,
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return direct and successive Klein-Nishina line shapes."""
    axis = np.asarray(energy_axis_keV, dtype=np.float64)
    indices = np.asarray(raw_bin_indices, dtype=np.int64)
    direct = np.zeros((indices.size, axis.size), dtype=np.float64)
    direct[np.arange(indices.size), indices] = 1.0
    transition = _klein_nishina_transition_matrix(
        axis,
        quadrature_order=int(quadrature_order),
    )
    current = direct.T
    orders: list[NDArray[np.float64]] = []
    for _ in range(int(maximum_scatter_order)):
        current = transition @ current
        current /= np.maximum(
            np.sum(current, axis=0, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        orders.append(current.T.copy())
    return direct, np.stack(orders, axis=1)


@dataclass
class GeometryConditionedSpectralModel:
    """Represent the shared source-resolved PF/DSS spectrum distribution."""

    _energy_axis_keV: NDArray[np.float64]
    _line_identity: tuple[Mapping[str, object], ...]
    response_operator_br: NDArray[np.float64]
    background_shape_b: NDArray[np.float64]
    dead_time_tau_s: float
    background_rate_cps: float
    maximum_scatter_order: int = 5
    klein_nishina_quadrature_order: int = 64
    rate_scale_nodes_j: tuple[float, ...] = (1.0,)
    rate_scale_weights_j: tuple[float, ...] = (1.0,)
    mark_concentration_source: float | None = None
    discrepancy_training_manifest: Mapping[str, object] | None = None
    validation_manifest: Mapping[str, object] | None = None
    additive_scatter_response: AdditiveNoncollidedTransportResponse | None = None
    _torch_cache: dict[tuple[str, str], tuple[object, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _proposal_basis_cache: dict[bytes, NDArray[np.float64]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the physical model arrays."""
        self._line_identity = tuple(
            _freeze_json_value(dict(item))
            for item in tuple(self._line_identity)
        )
        if not all(isinstance(item, Mapping) for item in self._line_identity):
            raise TypeError("Line identity rows must be mappings.")
        self.validation_manifest = (
            None
            if self.validation_manifest is None
            else _freeze_json_value(dict(self.validation_manifest))
        )
        self.discrepancy_training_manifest = (
            None
            if self.discrepancy_training_manifest is None
            else _freeze_json_value(dict(self.discrepancy_training_manifest))
        )
        if (
            self.additive_scatter_response is not None
            and not isinstance(
                self.additive_scatter_response,
                AdditiveNoncollidedTransportResponse,
            )
        ):
            raise TypeError(
                "additive_scatter_response must use the authenticated additive "
                "noncollided schema."
            )
        self._discrepancy_training_manifest_sha256 = (
            None
            if self.discrepancy_training_manifest is None
            else _canonical_json_sha256(self.discrepancy_training_manifest)
        )
        self._validation_manifest_sha256 = (
            None
            if self.validation_manifest is None
            else _canonical_json_sha256(self.validation_manifest)
        )
        self._energy_axis_keV = np.ascontiguousarray(
            self._energy_axis_keV,
            dtype=np.float64,
        )
        self.response_operator_br = np.ascontiguousarray(
            self.response_operator_br,
            dtype=np.float64,
        )
        self.background_shape_b = np.ascontiguousarray(
            self.background_shape_b,
            dtype=np.float64,
        )
        line_count = len(self._line_identity)
        bin_count = int(self._energy_axis_keV.size)
        if (
            bin_count < 2
            or line_count == 0
            or self.response_operator_br.shape != (bin_count, bin_count)
            or self.background_shape_b.shape != (bin_count,)
            or np.any(~np.isfinite(self._energy_axis_keV))
            or np.any(np.diff(self._energy_axis_keV) <= 0.0)
            or np.any(~np.isfinite(self.response_operator_br))
            or np.any(self.response_operator_br < 0.0)
            or np.any(~np.isfinite(self.background_shape_b))
            or np.any(self.background_shape_b < 0.0)
        ):
            raise ValueError("Geometry-conditioned spectrum arrays are invalid.")
        if not np.allclose(
            np.sum(self.response_operator_br, axis=0),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Detector-response columns must preserve counts.")
        if not np.isclose(
            np.sum(self.background_shape_b),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Background mark probabilities must sum to one.")
        if (
            not np.isfinite(self.dead_time_tau_s)
            or self.dead_time_tau_s < 0.0
            or not np.isfinite(self.background_rate_cps)
            or self.background_rate_cps < 0.0
            or int(self.maximum_scatter_order) < 1
            or int(self.klein_nishina_quadrature_order) < 8
        ):
            raise ValueError("Spectrum scalar physical parameters are invalid.")
        nodes = np.asarray(self.rate_scale_nodes_j, dtype=np.float64)
        weights = np.asarray(self.rate_scale_weights_j, dtype=np.float64)
        if (
            nodes.ndim != 1
            or nodes.size == 0
            or weights.shape != nodes.shape
            or np.any(~np.isfinite(nodes))
            or np.any(nodes <= 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
            or not np.isclose(
                np.sum(nodes * weights),
                1.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "Rate-scale mixture must be positive, normalized, and mean one."
            )
        concentration = self.mark_concentration_source
        if concentration is not None and (
            not np.isfinite(concentration) or float(concentration) <= 0.0
        ):
            raise ValueError(
                "mark_concentration_source must be positive when configured."
            )
        self._rate_scale_nodes_j = np.ascontiguousarray(nodes)
        self._rate_scale_weights_j = np.ascontiguousarray(weights)
        raw_indices = np.asarray(
            [int(item["raw_bin_index"]) for item in self._line_identity],
            dtype=np.int64,
        )
        if np.any(raw_indices < 0) or np.any(raw_indices >= bin_count):
            raise ValueError("Transport-line raw bins are outside the energy axis.")
        direct, scatter = _line_order_shapes(
            self._energy_axis_keV,
            raw_indices,
            maximum_scatter_order=int(self.maximum_scatter_order),
            quadrature_order=int(self.klein_nishina_quadrature_order),
        )
        self._direct_line_shapes_lb = direct
        self._scatter_order_shapes_lob = scatter
        self._marked_direct_line_shapes_lb = np.einsum(
            "br,lr->lb",
            self.response_operator_br,
            direct,
            optimize=True,
        )
        self._marked_scatter_order_shapes_lob = np.einsum(
            "br,lor->lob",
            self.response_operator_br,
            scatter,
            optimize=True,
        )
        energies = np.asarray(
            [float(item["energy_keV"]) for item in self._line_identity],
            dtype=np.float64,
        )
        sigma_kn = _klein_nishina_total_cross_section_cm2(energies)
        self._air_mu_compton_l = (
            AIR_DENSITY_G_CM3
            * AVOGADRO_CONSTANT_MOL_INV
            * AIR_EFFECTIVE_Z_OVER_A
            * sigma_kn
        )
        self._fe_compton_fraction_l = self._material_compton_fraction(
            sigma_kn,
            density_g_cm3=IRON_DENSITY_G_CM3,
            z_over_a=IRON_Z_OVER_A,
            material_key="mu_fe_cm_inv",
        )
        self._pb_compton_fraction_l = self._material_compton_fraction(
            sigma_kn,
            density_g_cm3=LEAD_DENSITY_G_CM3,
            z_over_a=LEAD_Z_OVER_A,
            material_key="mu_pb_cm_inv",
        )
        self._obstacle_compton_fraction_l = np.ones(
            line_count,
            dtype=np.float64,
        )
        for array in (
            self._energy_axis_keV,
            self.response_operator_br,
            self.background_shape_b,
            self._direct_line_shapes_lb,
            self._scatter_order_shapes_lob,
            self._marked_direct_line_shapes_lb,
            self._marked_scatter_order_shapes_lob,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
            self._rate_scale_nodes_j,
            self._rate_scale_weights_j,
        ):
            array.setflags(write=False)
        self._contract_hash_sha256 = self._build_contract_hash()

    @classmethod
    def standard_native(
        cls,
        isotopes: Sequence[str],
        *,
        dead_time_tau_s: float,
        background_rate_cps: float,
        rate_scale_nodes_j: Sequence[float] = (1.0,),
        rate_scale_weights_j: Sequence[float] = (1.0,),
        mark_concentration_source: float | None = None,
        discrepancy_training_manifest: Mapping[str, object] | None = None,
        validation_manifest: Mapping[str, object] | None = None,
        additive_scatter_response: (
            AdditiveNoncollidedTransportResponse | None
        ) = None,
    ) -> GeometryConditionedSpectralModel:
        """Build the native model directly from axis and physical line data."""
        isotope_order = tuple(sorted(str(value) for value in isotopes))
        if not isotope_order or len(set(isotope_order)) != len(isotope_order):
            raise ValueError("Spectrum model isotopes must be nonempty and unique.")
        bin_width = float(NATIVE_GEANT4_BIN_WIDTH_KEV)
        energy_axis = (
            np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
            * bin_width
        )
        library = default_library()
        shield_lines = line_resolved_shield_mu_by_isotope(
            isotope_order,
            normalize_line_intensities=True,
        )
        line_identity: list[dict[str, object]] = []
        for isotope in isotope_order:
            nuclide = library.get(isotope)
            if nuclide is None:
                raise KeyError(f"Missing physical line library for {isotope!r}.")
            positive_lines = [
                line for line in nuclide.lines if float(line.intensity) > 0.0
            ]
            isotope_shield_lines = shield_lines.get(isotope, ())
            if len(isotope_shield_lines) != len(positive_lines):
                raise RuntimeError(
                    "Shield and spectrum line libraries disagree for "
                    f"{isotope!r}."
                )
            total_weight = sum(float(line.intensity) for line in positive_lines)
            for local_index, line in enumerate(positive_lines):
                shield_entry = isotope_shield_lines[local_index]
                raw_bin = int(
                    np.floor(
                        (float(line.energy_keV) - float(energy_axis[0]))
                        / bin_width
                    )
                )
                line_identity.append(
                    {
                        "isotope": isotope,
                        "transport_line_index": int(local_index),
                        "energy_keV": float(line.energy_keV),
                        "branching_weight": (
                            float(line.intensity) / float(total_weight)
                        ),
                        "raw_bin_index": raw_bin,
                        "raw_bin_energy_keV": float(energy_axis[raw_bin]),
                        "mu_fe_cm_inv": float(shield_entry["fe"]),
                        "mu_pb_cm_inv": float(shield_entry["pb"]),
                    }
                )
        response_operator = build_native_geant4_detector_response_matrix(
            energy_axis,
            bin_width,
        )
        background_shape = native_geant4_background_shape(
            energy_axis,
            bin_width,
        )
        return cls(
            _energy_axis_keV=energy_axis,
            _line_identity=tuple(line_identity),
            response_operator_br=response_operator,
            background_shape_b=background_shape,
            dead_time_tau_s=float(dead_time_tau_s),
            background_rate_cps=float(background_rate_cps),
            rate_scale_nodes_j=tuple(
                float(value) for value in rate_scale_nodes_j
            ),
            rate_scale_weights_j=tuple(
                float(value) for value in rate_scale_weights_j
            ),
            mark_concentration_source=(
                None
                if mark_concentration_source is None
                else float(mark_concentration_source)
            ),
            discrepancy_training_manifest=discrepancy_training_manifest,
            validation_manifest=validation_manifest,
            additive_scatter_response=additive_scatter_response,
        )

    @classmethod
    def from_manifest_payload(
        cls,
        payload: Mapping[str, object],
    ) -> GeometryConditionedSpectralModel:
        """Reconstruct and authenticate one approved schema-v2 model."""
        if not isinstance(payload, Mapping):
            raise TypeError("Full-spectrum model manifest must be a mapping.")
        if (
            payload.get("schema_version") != 3
            or payload.get("model")
            != "geometry_conditioned_full_spectrum"
        ):
            raise ValueError(
                "Runtime requires a geometry-conditioned schema-v3 spectrum "
                "manifest."
            )
        line_rows = payload.get("line_identity")
        mixture = payload.get("rate_scale_mixture")
        if (
            not isinstance(line_rows, Sequence)
            or isinstance(line_rows, (str, bytes))
            or not line_rows
            or not all(isinstance(row, Mapping) for row in line_rows)
            or not isinstance(mixture, Mapping)
            or set(mixture) != {"scope", "nodes", "weights", "weighted_mean"}
            or mixture.get("scope") != "station_shared_source_only"
        ):
            raise ValueError(
                "Full-spectrum manifest line or discrepancy identity is invalid."
            )
        raw_isotopes = tuple(row.get("isotope") for row in line_rows)
        if any(
            not isinstance(value, str) or not value
            for value in raw_isotopes
        ):
            raise ValueError(
                "Full-spectrum manifest requires nonempty line isotopes."
            )
        isotope_order = tuple(sorted(set(raw_isotopes)))
        additive_payload = payload.get(
            "additive_noncollided_transport_response"
        )
        if not isinstance(additive_payload, Mapping):
            raise ValueError(
                "Schema-v3 full-spectrum manifests require the authenticated "
                "additive noncollided transport response."
            )
        mixture_nodes = _strict_json_number_sequence(
            mixture.get("nodes"),
            field_name="rate_scale_mixture.nodes",
        )
        mixture_weights = _strict_json_number_sequence(
            mixture.get("weights"),
            field_name="rate_scale_mixture.weights",
        )
        dead_time_tau_s = _strict_json_number(
            payload.get("dead_time_tau_s"),
            field_name="dead_time_tau_s",
        )
        background_rate_cps = _strict_json_number(
            payload.get("background_rate_cps"),
            field_name="background_rate_cps",
        )
        raw_concentration = payload.get("mark_concentration_source")
        mark_concentration_source = (
            None
            if raw_concentration is None
            else _strict_json_number(
                raw_concentration,
                field_name="mark_concentration_source",
            )
        )
        model = cls.standard_native(
            isotope_order,
            dead_time_tau_s=dead_time_tau_s,
            background_rate_cps=background_rate_cps,
            rate_scale_nodes_j=mixture_nodes,
            rate_scale_weights_j=mixture_weights,
            mark_concentration_source=mark_concentration_source,
            discrepancy_training_manifest=(
                payload.get("discrepancy_training")
                if isinstance(
                    payload.get("discrepancy_training"),
                    Mapping,
                )
                else None
            ),
            validation_manifest=(
                payload.get("validation")
                if isinstance(payload.get("validation"), Mapping)
                else None
            ),
            additive_scatter_response=(
                AdditiveNoncollidedTransportResponse.from_payload(
                    additive_payload
                )
            ),
        )
        reconstructed = model.manifest_payload()
        supplied = _thaw_json_value(_freeze_json_value(dict(payload)))
        if reconstructed != supplied:
            raise ValueError(
                "Full-spectrum manifest does not exactly reconstruct the "
                "declared physical and statistical contract."
            )
        model.require_production_ready()
        return model

    def _material_compton_fraction(
        self,
        sigma_kn_l: NDArray[np.float64],
        *,
        density_g_cm3: float,
        z_over_a: float,
        material_key: str,
    ) -> NDArray[np.float64]:
        """Return Compton/total attenuation fractions from line provenance."""
        compton_mu = (
            float(density_g_cm3)
            * AVOGADRO_CONSTANT_MOL_INV
            * float(z_over_a)
            * np.asarray(sigma_kn_l, dtype=np.float64)
        )
        total_mu = np.asarray(
            [
                float(item.get(material_key, compton_mu[index]))
                for index, item in enumerate(self._line_identity)
            ],
            dtype=np.float64,
        )
        return np.clip(
            np.divide(
                compton_mu,
                np.maximum(total_mu, np.finfo(np.float64).tiny),
            ),
            0.0,
            1.0,
        )

    def _build_contract_hash(self) -> str:
        """Return the physical model digest independent of validation results."""
        digest = hashlib.sha256()
        digest.update(b"geometry_conditioned_spectral_model_v3")
        digest.update(
            json.dumps(
                {
                    "line_identity": [dict(item) for item in self._line_identity],
                    "source_rate_semantics": (
                        "pre_dead_time_detector_pulse_rate_at_1m"
                    ),
                    "dead_time_tau_s": float(self.dead_time_tau_s),
                    "background_rate_cps": float(self.background_rate_cps),
                    "maximum_scatter_order": int(self.maximum_scatter_order),
                    "klein_nishina_quadrature_order": int(
                        self.klein_nishina_quadrature_order
                    ),
                    "transport_feature_order": TRANSPORT_FEATURE_ORDER,
                    "detector_response_model": (
                        "native_incident_gamma_response_v1"
                    ),
                    "detector_response_contract_sha256": (
                        NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
                    ),
                    "dead_time_model": (
                        "nonparalyzable_renewal_total_conditional_multinomial"
                    ),
                    "birth_proposal_score_method": (
                        "background_whitened_non_target_line_subspace_"
                        "matched_filter_v1"
                    ),
                    "birth_proposal_background_regularization_counts": 1.0,
                    "rate_scale_mixture": "station_shared_finite_positive",
                    "mark_discrepancy": (
                        "source_fraction_dirichlet_multinomial"
                        if self.mark_concentration_source is not None
                        else "exact_multinomial_diagnostic_only"
                    ),
                    "mark_concentration_source": (
                        None
                        if self.mark_concentration_source is None
                        else float(self.mark_concentration_source)
                    ),
                    "discrepancy_training_manifest_sha256": (
                        self._discrepancy_training_manifest_sha256
                    ),
                    "acceptance_contract_sha256": (
                        FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
                    ),
                    "additive_scatter_contract_sha256": (
                        None
                        if self.additive_scatter_response is None
                        else self.additive_scatter_response.contract_hash_sha256
                    ),
                    "transport_training_label_semantics": (
                        ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for array in (
            self._energy_axis_keV,
            self.response_operator_br,
            self.background_shape_b,
            self._direct_line_shapes_lb,
            self._scatter_order_shapes_lob,
            self._marked_direct_line_shapes_lb,
            self._marked_scatter_order_shapes_lob,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
            self._rate_scale_nodes_j,
            self._rate_scale_weights_j,
        ):
            digest.update(_array_digest(array))
        return digest.hexdigest()

    @property
    def discrepancy_training_ready(self) -> bool:
        """Return whether global discrepancy parameters used training only."""
        manifest = self.discrepancy_training_manifest
        if not isinstance(manifest, Mapping):
            return False
        expected_keys = {
            "schema_version",
            "acceptance_contract_sha256",
            "training_scene_seeds",
            "scenario_ids",
            "pair_ids_by_scene",
            "artifact_sha256_by_scene",
            "rate_scale_family",
            "mark_family",
            "selection_objective",
            "selected_rate_scale_half_width",
            "selected_mark_concentration_source",
            "candidate_count",
            "selected_training_log_predictive_density",
            "selection_artifact_sha256",
            "selection_completed",
        }
        if set(manifest) != expected_keys:
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("acceptance_contract_sha256")
            != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            or tuple(manifest.get("training_scene_seeds", ()))
            != DESIGNATED_TRAINING_SCENE_SEEDS
            or tuple(manifest.get("scenario_ids", ()))
            != VALIDATION_SCENARIO_IDS
            or manifest.get("rate_scale_family")
            != "station_shared_three_node_symmetric_mean_one"
            or manifest.get("mark_family")
            != "source_fraction_dirichlet_multinomial"
            or manifest.get("selection_objective")
            != "maximum_joint_training_log_predictive_density"
            or manifest.get("selection_completed") is not True
            or manifest.get("candidate_count")
            != len(RATE_SCALE_HALF_WIDTH_GRID)
            * len(MARK_CONCENTRATION_GRID)
            or not _is_sha256(manifest.get("selection_artifact_sha256"))
        ):
            return False
        pair_ids = manifest.get("pair_ids_by_scene")
        artifact_hashes = manifest.get("artifact_sha256_by_scene")
        expected_seed_keys = {
            str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS
        }
        if (
            not isinstance(pair_ids, Mapping)
            or set(pair_ids) != expected_seed_keys
            or any(
                tuple(pair_ids[str(seed)]) != tuple(range(64))
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            )
            or not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes) != expected_seed_keys
            or any(
                not _is_sha256(artifact_hashes[str(seed)])
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            )
        ):
            return False
        try:
            width = float(manifest["selected_rate_scale_half_width"])
            concentration = float(
                manifest["selected_mark_concentration_source"]
            )
            selected_score = float(
                manifest["selected_training_log_predictive_density"]
            )
        except (TypeError, ValueError):
            return False
        if (
            width not in RATE_SCALE_HALF_WIDTH_GRID
            or concentration not in MARK_CONCENTRATION_GRID
            or not np.isfinite(selected_score)
            or self.mark_concentration_source is None
            or float(self.mark_concentration_source) != concentration
        ):
            return False
        expected_nodes, expected_weights = rate_scale_mixture_for_half_width(
            width
        )
        return bool(
            np.array_equal(
                self._rate_scale_nodes_j,
                np.asarray(expected_nodes, dtype=np.float64),
            )
            and np.array_equal(
                self._rate_scale_weights_j,
                np.asarray(expected_weights, dtype=np.float64),
            )
        )

    @property
    def production_ready(self) -> bool:
        """Return whether a fixed independent all-64 holdout approved the model."""
        if not self.discrepancy_training_ready:
            return False
        if (
            self.additive_scatter_response is None
            or not self.additive_scatter_response.training_ready
        ):
            return False
        manifest = self.validation_manifest
        if not isinstance(manifest, Mapping):
            return False
        expected_keys = {
            "schema_version",
            "validation_contract_sha256",
            "approved_model_contract_sha256",
            "native_response_contract_sha256",
            "additive_scatter_contract_sha256",
            "surface_emission_policy_sha256",
            "training_scene_seeds",
            "holdout_scene_seeds",
            "training_selection_scene_seeds",
            "metric_scene_seeds",
            "metric_split",
            "metric_aggregation",
            "scenario_ids",
            "pair_ids_by_scene",
            "artifact_sha256_by_scene",
            "scene_hash_by_scene_and_scenario",
            "surface_source_contract_sha256_by_scene_and_scenario",
            "metrics",
            "all_passed",
        }
        if set(manifest) != expected_keys:
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("validation_contract_sha256")
            != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            or manifest.get("approved_model_contract_sha256")
            != self.contract_hash_sha256
            or manifest.get("native_response_contract_sha256")
            != NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            or manifest.get("additive_scatter_contract_sha256")
            != self.additive_scatter_response.contract_hash_sha256
            or manifest.get("surface_emission_policy_sha256")
            != surface_emission_policy_sha256()
            or tuple(manifest.get("training_scene_seeds", ()))
            != DESIGNATED_TRAINING_SCENE_SEEDS
            or tuple(manifest.get("holdout_scene_seeds", ()))
            != DESIGNATED_HOLDOUT_SCENE_SEEDS
            or tuple(manifest.get("training_selection_scene_seeds", ()))
            != DESIGNATED_TRAINING_SCENE_SEEDS
            or tuple(manifest.get("metric_scene_seeds", ()))
            != DESIGNATED_HOLDOUT_SCENE_SEEDS
            or manifest.get("metric_split") != "holdout_only"
            or manifest.get("metric_aggregation")
            != "holdout_scene_conservative_worst_case"
            or tuple(manifest.get("scenario_ids", ()))
            != VALIDATION_SCENARIO_IDS
            or manifest.get("all_passed") is not True
        ):
            return False
        all_seeds = (
            DESIGNATED_TRAINING_SCENE_SEEDS
            + DESIGNATED_HOLDOUT_SCENE_SEEDS
        )
        pair_ids = manifest.get("pair_ids_by_scene")
        artifact_hashes = manifest.get("artifact_sha256_by_scene")
        scene_hashes = manifest.get(
            "scene_hash_by_scene_and_scenario"
        )
        source_hashes = manifest.get(
            "surface_source_contract_sha256_by_scene_and_scenario"
        )
        expected_seed_keys = {str(seed) for seed in all_seeds}
        if (
            not isinstance(pair_ids, Mapping)
            or set(pair_ids) != expected_seed_keys
            or any(
                tuple(pair_ids[str(seed)]) != tuple(range(64))
                for seed in all_seeds
            )
            or not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes) != expected_seed_keys
            or any(
                not _is_sha256(artifact_hashes[str(seed)])
                for seed in all_seeds
            )
            or not isinstance(scene_hashes, Mapping)
            or set(scene_hashes) != expected_seed_keys
            or any(
                not isinstance(scene_hashes[str(seed)], Mapping)
                or set(scene_hashes[str(seed)])
                != set(VALIDATION_SCENARIO_IDS)
                or any(
                    not _is_sha256(
                        scene_hashes[str(seed)][scenario]
                    )
                    for scenario in VALIDATION_SCENARIO_IDS
                )
                for seed in all_seeds
            )
            or not isinstance(source_hashes, Mapping)
            or set(source_hashes) != expected_seed_keys
            or any(
                not isinstance(source_hashes[str(seed)], Mapping)
                or set(source_hashes[str(seed)])
                != set(VALIDATION_SCENARIO_IDS)
                or any(
                    not _is_sha256(
                        source_hashes[str(seed)][scenario]
                    )
                    for scenario in VALIDATION_SCENARIO_IDS
                )
                for seed in all_seeds
            )
        ):
            return False
        metrics = manifest.get("metrics")
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != set(ACCEPTANCE_METRIC_CONTRACT)
        ):
            return False
        for metric_id, (comparison, threshold) in (
            ACCEPTANCE_METRIC_CONTRACT.items()
        ):
            result = metrics.get(metric_id)
            if not isinstance(result, Mapping) or set(result) != {
                "value",
                "comparison",
                "threshold",
                "passed",
            }:
                return False
            raw_value = result["value"]
            raw_threshold = result["threshold"]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or isinstance(raw_threshold, bool)
                or not isinstance(raw_threshold, (int, float))
            ):
                return False
            value = float(raw_value)
            reported_threshold = float(raw_threshold)
            if (
                not np.isfinite(value)
                or reported_threshold != float(threshold)
                or result["comparison"] != comparison
                or result["passed"] is not True
            ):
                return False
            expected_pass = (
                value <= float(threshold)
                if comparison == "le"
                else value >= float(threshold)
            )
            if not expected_pass:
                return False
        return True

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable physical model hash."""
        return self._contract_hash_sha256

    @property
    def energy_axis_keV(self) -> NDArray[np.float64]:
        """Return a defensive copy of the native analysis axis."""
        return self._energy_axis_keV.copy()

    @property
    def line_identity(self) -> tuple[Mapping[str, object], ...]:
        """Return the global positive transport-line order."""
        return tuple(dict(item) for item in self._line_identity)

    @property
    def transport_feature_order(self) -> tuple[str, ...]:
        """Return the canonical geometry feature order."""
        return TRANSPORT_FEATURE_ORDER

    def require_production_ready(self) -> None:
        """Fail closed until independent validation approves this exact hash."""
        if not self.production_ready:
            raise RuntimeError(
                "Geometry-conditioned spectrum model has not passed the fixed "
                "independent all-64 holdout gate for this exact contract hash."
            )

    def _validated_numpy_inputs(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return aligned finite source-resolved NumPy transport inputs."""
        total = np.asarray(total_line_contributions_xvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_xvsl,
            dtype=np.float64,
        )
        features = np.asarray(transport_features_xvslf, dtype=np.float64)
        live_times = np.asarray(live_times_s_v, dtype=np.float64)
        line_count = len(self._line_identity)
        if (
            total.ndim < 3
            or total.shape[-1] != line_count
            or uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or live_times.shape != (total.shape[-3],)
            or np.any(~np.isfinite(total))
            or np.any(total < 0.0)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
            or np.any(~np.isfinite(features))
            or np.any(features < 0.0)
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
        ):
            raise ValueError(
                "Spectrum transport inputs must be finite nonnegative "
                "...view/source/line arrays with positive view live times."
            )
        tolerance = (
            128.0
            * np.finfo(np.float64).eps
            * np.maximum(
                1.0,
                np.maximum(np.abs(total), np.abs(uncollided)),
            )
        )
        if np.any(uncollided > total + tolerance):
            raise ValueError(
                "Uncollided line contributions cannot exceed total incident "
                "line contributions."
            )
        uncollided = np.minimum(uncollided, total)
        return total, uncollided, features, live_times

    def _interaction_order_weights_numpy(
        self,
        features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return conditional positive-interaction order probabilities."""
        features = np.asarray(features_xvslf, dtype=np.float64)
        tau = (
            features[..., 0] * self._fe_compton_fraction_l
            + features[..., 1] * self._pb_compton_fraction_l
            + features[..., 2] * self._obstacle_compton_fraction_l
            + features[..., 3]
            * 100.0
            * self._air_mu_compton_l
        )
        tau = np.maximum(tau, 0.0)
        exact_orders = np.arange(
            1,
            int(self.maximum_scatter_order),
            dtype=np.float64,
        )
        denominator = -np.expm1(-tau)
        safe_tau = np.maximum(tau, np.finfo(np.float64).tiny)
        log_exact = (
            -tau[..., np.newaxis]
            + np.log(safe_tau)[..., np.newaxis] * exact_orders
            - special.gammaln(exact_orders + 1.0)
            - np.log(
                np.maximum(denominator, np.finfo(np.float64).tiny)
            )[..., np.newaxis]
        )
        exact = np.exp(log_exact)
        exact = np.where(tau[..., np.newaxis] > 0.0, exact, 0.0)
        tail = np.maximum(1.0 - np.sum(exact, axis=-1), 0.0)
        weights = np.concatenate(
            (exact, tail[..., np.newaxis]),
            axis=-1,
        )
        zero_tau = tau <= 0.0
        weights[..., 0] = np.where(zero_tau, 1.0, weights[..., 0])
        weights[..., 1:] = np.where(
            zero_tau[..., np.newaxis],
            0.0,
            weights[..., 1:],
        )
        weights /= np.maximum(
            np.sum(weights, axis=-1, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        return weights

    def _pre_dead_time_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        return_components: bool = False,
    ) -> (
        NDArray[np.float64]
        | tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Return expected marked spectra before detector dead time."""
        total, uncollided, features, live_times = self._validated_numpy_inputs(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
        )
        live_scale = live_times.reshape(
            (1,) * (total.ndim - 3)
            + (int(total.shape[-3]), 1, 1)
        )
        total_counts = total * live_scale
        uncollided_counts = uncollided * live_scale
        direct = np.minimum(total_counts, uncollided_counts)
        scatter = total_counts - direct
        order_weights = self._interaction_order_weights_numpy(features)
        direct_by_line = np.sum(direct, axis=-2)
        scatter_by_line_order = np.sum(
            scatter[..., np.newaxis] * order_weights,
            axis=-3,
        )
        marked_source = (
            np.einsum(
                "...vl,lb->...vb",
                direct_by_line,
                self._marked_direct_line_shapes_lb,
                optimize=True,
            )
            + np.einsum(
                "...vlo,lob->...vb",
                scatter_by_line_order,
                self._marked_scatter_order_shapes_lob,
                optimize=True,
            )
        )
        background = (
            float(self.background_rate_cps)
            * live_times[:, np.newaxis]
            * self.background_shape_b[np.newaxis, :]
        )
        background = np.broadcast_to(
            background,
            marked_source.shape,
        ).copy()
        mean = marked_source + background
        expected_total = np.sum(total_counts, axis=(-2, -1))
        if not np.allclose(
            np.sum(marked_source, axis=-1),
            expected_total,
            rtol=1.0e-11,
            atol=1.0e-8,
        ):
            raise RuntimeError(
                "Source-resolved spectral transport failed line-count "
                "conservation."
            )
        if return_components:
            return np.maximum(marked_source, 0.0), background
        return np.maximum(mean, 0.0)

    def predict_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return asymptotic renewal means with exact conditional mark means."""
        source_mean, background_mean = self._pre_dead_time_mean_numpy(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        live_times = np.asarray(live_times_s_v, dtype=np.float64)
        pre_mean = (
            background_mean[..., np.newaxis, :, :]
            + source_mean[..., np.newaxis, :, :]
            * self._rate_scale_nodes_j.reshape(
                (1,) * (source_mean.ndim - 2) + (-1, 1, 1)
            )
        )
        pre_total = np.sum(pre_mean, axis=-1)
        rates = pre_total / live_times
        expected_total = pre_total / (
            1.0 + rates * float(self.dead_time_tau_s)
        )
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        node_means = probabilities * expected_total[..., np.newaxis]
        return np.sum(
            node_means
            * self._rate_scale_weights_j.reshape(
                (1,) * (source_mean.ndim - 2) + (-1, 1, 1)
            ),
            axis=-3,
        )

    def pre_dead_time_components_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return source and background marked means before detector dead time."""
        source, background = self._pre_dead_time_mean_numpy(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        return (
            np.asarray(source, dtype=np.float64).copy(),
            np.asarray(background, dtype=np.float64).copy(),
        )

    @property
    def rate_scale_nodes(self) -> NDArray[np.float64]:
        """Return a defensive copy of the shared source-rate mixture nodes."""
        return self._rate_scale_nodes_j.copy()

    @property
    def rate_scale_weights(self) -> NDArray[np.float64]:
        """Return a defensive copy of the shared source-rate mixture weights."""
        return self._rate_scale_weights_j.copy()

    def _torch_constants(self, reference: object) -> tuple[object, ...]:
        """Return cached immutable constants matching a reference Torch tensor."""
        import torch

        tensor = torch.as_tensor(reference)
        key = (str(tensor.device), str(tensor.dtype))
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        arrays = (
            self.background_shape_b,
            self._marked_direct_line_shapes_lb,
            self._marked_scatter_order_shapes_lob,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
        )
        cached = tuple(
            torch.as_tensor(
                np.array(value, dtype=np.float64, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for value in arrays
        )
        self._torch_cache[key] = cached
        return cached

    def _pre_dead_time_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
        *,
        return_components: bool = False,
    ) -> object:
        """Return the Torch pre-dead-time marked spectral mean."""
        import torch

        total = torch.as_tensor(total_line_contributions_xvsl)
        if total.dtype != torch.float64:
            raise TypeError(
                "Production full-spectrum inference requires torch.float64."
            )
        uncollided = torch.as_tensor(
            uncollided_line_contributions_xvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            transport_features_xvslf,
            device=total.device,
            dtype=total.dtype,
        )
        live_times = torch.as_tensor(
            live_times_s_v,
            device=total.device,
            dtype=total.dtype,
        )
        if (
            total.ndim < 3
            or total.shape[-1] != len(self._line_identity)
            or uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or tuple(live_times.shape) != (int(total.shape[-3]),)
            or bool(torch.any(~torch.isfinite(total)))
            or bool(torch.any(total < 0.0))
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(uncollided < 0.0))
            or bool(torch.any(~torch.isfinite(features)))
            or bool(torch.any(features < 0.0))
            or bool(torch.any(~torch.isfinite(live_times)))
            or bool(torch.any(live_times <= 0.0))
        ):
            raise ValueError("Torch spectrum transport inputs are invalid.")
        tolerance = (
            128.0
            * torch.finfo(total.dtype).eps
            * torch.maximum(
                torch.ones((), device=total.device, dtype=total.dtype),
                torch.maximum(torch.abs(total), torch.abs(uncollided)),
            )
        )
        if bool(torch.any(uncollided > total + tolerance)):
            raise ValueError(
                "Uncollided line contributions cannot exceed total incident "
                "line contributions."
            )
        uncollided = torch.minimum(uncollided, total)
        (
            background_shape,
            direct_shapes,
            scatter_shapes,
            air_mu,
            fe_fraction,
            pb_fraction,
            obstacle_fraction,
        ) = self._torch_constants(total)
        live_scale = live_times.reshape(
            (1,) * (total.ndim - 3)
            + (int(total.shape[-3]), 1, 1)
        )
        total_counts = total * live_scale
        uncollided_counts = uncollided * live_scale
        direct = uncollided_counts
        scatter = total_counts - direct
        tau = (
            features[..., 0] * fe_fraction
            + features[..., 1] * pb_fraction
            + features[..., 2] * obstacle_fraction
            + features[..., 3] * 100.0 * air_mu
        )
        tau = torch.clamp(tau, min=0.0)
        exact_orders = torch.arange(
            1,
            int(self.maximum_scatter_order),
            device=total.device,
            dtype=total.dtype,
        )
        denominator = -torch.expm1(-tau)
        tiny = torch.finfo(total.dtype).tiny
        log_exact = (
            -tau.unsqueeze(-1)
            + torch.log(torch.clamp(tau, min=tiny)).unsqueeze(-1)
            * exact_orders
            - torch.lgamma(exact_orders + 1.0)
            - torch.log(torch.clamp(denominator, min=tiny)).unsqueeze(-1)
        )
        exact = torch.where(
            tau.unsqueeze(-1) > 0.0,
            torch.exp(log_exact),
            torch.zeros_like(log_exact),
        )
        tail = torch.clamp(1.0 - torch.sum(exact, dim=-1), min=0.0)
        order_weights = torch.cat((exact, tail.unsqueeze(-1)), dim=-1)
        zero_tau = tau <= 0.0
        first = torch.where(
            zero_tau,
            torch.ones_like(order_weights[..., 0]),
            order_weights[..., 0],
        )
        rest = torch.where(
            zero_tau.unsqueeze(-1),
            torch.zeros_like(order_weights[..., 1:]),
            order_weights[..., 1:],
        )
        order_weights = torch.cat((first.unsqueeze(-1), rest), dim=-1)
        order_weights = order_weights / torch.clamp(
            torch.sum(order_weights, dim=-1, keepdim=True),
            min=tiny,
        )
        direct_by_line = torch.sum(direct, dim=-2)
        scatter_by_line_order = torch.sum(
            scatter.unsqueeze(-1) * order_weights,
            dim=-3,
        )
        marked_source = (
            torch.einsum("...vl,lb->...vb", direct_by_line, direct_shapes)
            + torch.einsum(
                "...vlo,lob->...vb",
                scatter_by_line_order,
                scatter_shapes,
            )
        )
        background = (
            float(self.background_rate_cps)
            * live_times[:, None]
            * background_shape[None, :]
        )
        background = torch.broadcast_to(
            background,
            marked_source.shape,
        )
        expected_total = torch.sum(total_counts, dim=(-2, -1))
        if not bool(
            torch.allclose(
                torch.sum(marked_source, dim=-1),
                expected_total,
                rtol=1.0e-10,
                atol=1.0e-7,
            )
        ):
            raise RuntimeError("Torch spectral count conservation failed.")
        if return_components:
            return torch.clamp(marked_source, min=0.0), background
        return torch.clamp(marked_source + background, min=0.0)

    def predict_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch renewal/mark predictive mean."""
        import torch

        source_mean, background_mean = self._pre_dead_time_mean_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        live_times = torch.as_tensor(
            live_times_s_v,
            device=source_mean.device,
            dtype=source_mean.dtype,
        )
        nodes = torch.as_tensor(
            np.array(self._rate_scale_nodes_j, copy=True),
            device=source_mean.device,
            dtype=torch.float64,
        )
        node_weights = torch.as_tensor(
            np.array(self._rate_scale_weights_j, copy=True),
            device=source_mean.device,
            dtype=torch.float64,
        )
        node_shape = (
            (1,) * (source_mean.ndim - 2)
            + (int(nodes.numel()), 1, 1)
        )
        pre_mean = (
            background_mean.unsqueeze(-3)
            + source_mean.unsqueeze(-3) * nodes.reshape(node_shape)
        )
        pre_total = torch.sum(pre_mean, dim=-1)
        expected_total = pre_total / (
            1.0
            + pre_total
            / live_times
            * float(self.dead_time_tau_s)
        )
        probabilities = torch.where(
            pre_total.unsqueeze(-1) > 0.0,
            pre_mean
            / torch.clamp(
                pre_total.unsqueeze(-1),
                min=torch.finfo(pre_mean.dtype).tiny,
            ),
            torch.zeros_like(pre_mean),
        )
        node_means = probabilities * expected_total.unsqueeze(-1)
        return torch.sum(
            node_means * node_weights.reshape(node_shape),
            dim=-3,
        )

    def _cross_log_likelihood_numpy_unchunked(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return vectorized sample-by-state renewal/multinomial likelihoods."""
        observed = np.asarray(observed_spectra_xqvb, dtype=np.float64)
        source_mean, background_mean = self._pre_dead_time_mean_numpy(
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
            transport_features_xnvslf,
            live_times_s_v,
            return_components=True,
        )
        if (
            observed.ndim < 3
            or observed.shape[:-3] != source_mean.shape[:-3]
            or observed.shape[-2:] != source_mean.shape[-2:]
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
            or np.any(observed != np.floor(observed))
        ):
            raise ValueError(
                "Cross spectra must be exact nonnegative counts shaped "
                "...sample/view/bin with common model leading axes."
            )
        node_shape = (
            (1,) * (source_mean.ndim - 3)
            + (1, int(self._rate_scale_nodes_j.size), 1, 1)
        )
        node_source = (
            source_mean[..., :, np.newaxis, :, :]
            * self._rate_scale_nodes_j.reshape(node_shape)
        )
        pre_mean = background_mean[..., :, np.newaxis, :, :] + node_source
        observed_total = np.sum(observed, axis=-1)
        pre_total = np.sum(pre_mean, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        count_log = nonparalyzable_count_log_probability_numpy(
            observed_total[..., :, np.newaxis, np.newaxis, :],
            pre_total[..., np.newaxis, :, :, :] / live,
            live,
            dead_time_tau_s=float(self.dead_time_tau_s),
        )
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        log_probabilities = np.log(
            np.maximum(probabilities, np.finfo(np.float64).tiny)
        )
        multinomial_log = (
            special.gammaln(observed_total + 1.0)[
                ..., :, np.newaxis, np.newaxis, :
            ]
            - np.sum(
                special.gammaln(observed + 1.0),
                axis=-1,
            )[..., :, np.newaxis, np.newaxis, :]
            + np.einsum(
                "...qvb,...njvb->...qnjv",
                observed,
                log_probabilities,
                optimize=True,
            )
        )
        impossible_marks = np.einsum(
            "...qvb,...njvb->...qnjv",
            observed,
            probabilities <= 0.0,
            optimize=True,
        ) > 0.0
        multinomial_log = np.where(
            impossible_marks,
            -np.inf,
            multinomial_log,
        )
        mark_log = multinomial_log
        if self.mark_concentration_source is not None:
            source_total = np.sum(node_source, axis=-1)
            source_fraction = np.divide(
                source_total,
                pre_total,
                out=np.zeros_like(source_total),
                where=pre_total > 0.0,
            )
            concentration = float(self.mark_concentration_source) / np.maximum(
                np.square(source_fraction),
                1.0e-12,
            )
            alpha = probabilities * concentration[..., np.newaxis]
            dirichlet_sum = np.zeros_like(multinomial_log)
            for start in range(
                0,
                observed.shape[-1],
                CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
            ):
                stop = min(
                    start + CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
                    observed.shape[-1],
                )
                observed_chunk = observed[..., start:stop]
                alpha_chunk = alpha[..., start:stop]
                expanded_alpha = alpha_chunk[
                    ..., np.newaxis, :, :, :, :
                ]
                expanded_observed = observed_chunk[
                    ..., :, np.newaxis, np.newaxis, :, :
                ]
                active_increment = (
                    (expanded_alpha > 0.0)
                    & (expanded_observed > 0.0)
                )
                safe_alpha = np.where(
                    active_increment,
                    expanded_alpha,
                    1.0,
                )
                safe_observed = np.where(
                    active_increment,
                    expanded_observed,
                    1.0,
                )
                dirichlet_sum += np.sum(
                    np.where(
                        active_increment,
                        np.log(safe_alpha)
                        + special.gammaln(
                            safe_alpha + safe_observed
                        )
                        - special.gammaln(safe_alpha + 1.0),
                        0.0,
                    ),
                    axis=-1,
                )
            dirichlet_log = (
                special.gammaln(observed_total + 1.0)[
                    ..., :, np.newaxis, np.newaxis, :
                ]
                - np.sum(
                    special.gammaln(observed + 1.0),
                    axis=-1,
                )[..., :, np.newaxis, np.newaxis, :]
                + special.gammaln(concentration)[
                    ..., np.newaxis, :, :, :
                ]
                - special.gammaln(
                    concentration[..., np.newaxis, :, :, :]
                    + observed_total[
                        ..., :, np.newaxis, np.newaxis, :
                    ]
                )
                + dirichlet_sum
            )
            dirichlet_log = np.where(
                impossible_marks,
                -np.inf,
                dirichlet_log,
            )
            mark_log = np.where(
                source_fraction[..., np.newaxis, :, :, :] > 0.0,
                dirichlet_log,
                multinomial_log,
            )
        zero_mark_total = (
            observed_total[..., :, np.newaxis, np.newaxis, :] == 0.0
        )
        mark_log = np.where(zero_mark_total, 0.0, mark_log)
        node_log = np.sum(count_log + mark_log, axis=-1)
        return special.logsumexp(
            node_log
            + np.log(self._rate_scale_weights_j).reshape(
                (1,) * (node_log.ndim - 1) + (-1,)
            ),
            axis=-1,
        )

    def _cross_log_likelihood_torch_unchunked(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch vectorized sample-by-state likelihood matrix."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        observed = torch.as_tensor(
            observed_spectra_xqvb,
            device=total.device,
            dtype=total.dtype,
        )
        source_mean, background_mean = self._pre_dead_time_mean_torch(
            total,
            uncollided_line_contributions_xnvsl,
            transport_features_xnvslf,
            live_times_s_v,
            return_components=True,
        )
        if (
            observed.ndim < 3
            or tuple(observed.shape[:-3]) != tuple(source_mean.shape[:-3])
            or tuple(observed.shape[-2:]) != tuple(source_mean.shape[-2:])
            or bool(torch.any(~torch.isfinite(observed)))
            or bool(torch.any(observed < 0.0))
            or bool(torch.any(observed != torch.floor(observed)))
        ):
            raise ValueError("Torch cross-spectrum observations are invalid.")
        nodes = torch.as_tensor(
            np.array(self._rate_scale_nodes_j, copy=True),
            device=total.device,
            dtype=torch.float64,
        )
        node_weights = torch.as_tensor(
            np.array(self._rate_scale_weights_j, copy=True),
            device=total.device,
            dtype=torch.float64,
        )
        node_shape = (
            (1,) * (source_mean.ndim - 3)
            + (1, int(nodes.numel()), 1, 1)
        )
        node_source = (
            source_mean.unsqueeze(-3) * nodes.reshape(node_shape)
        )
        pre_mean = background_mean.unsqueeze(-3) + node_source
        observed_total = torch.sum(observed, dim=-1)
        pre_total = torch.sum(pre_mean, dim=-1)
        live = torch.as_tensor(
            live_times_s_v,
            device=total.device,
            dtype=total.dtype,
        )
        count_log = nonparalyzable_count_log_probability_torch(
            observed_total.unsqueeze(-2).unsqueeze(-2),
            pre_total.unsqueeze(-4) / live,
            live,
            dead_time_tau_s=float(self.dead_time_tau_s),
        )
        tiny = torch.finfo(total.dtype).tiny
        probabilities = torch.where(
            pre_total.unsqueeze(-1) > 0.0,
            pre_mean / torch.clamp(pre_total.unsqueeze(-1), min=tiny),
            torch.zeros_like(pre_mean),
        )
        log_probabilities = torch.log(torch.clamp(probabilities, min=tiny))
        multinomial_log = (
            torch.lgamma(observed_total + 1.0).unsqueeze(-2).unsqueeze(-2)
            - torch.sum(
                torch.lgamma(observed + 1.0),
                dim=-1,
            ).unsqueeze(-2).unsqueeze(-2)
            + torch.einsum(
                "...qvb,...njvb->...qnjv",
                observed,
                log_probabilities,
            )
        )
        impossible = torch.einsum(
            "...qvb,...njvb->...qnjv",
            observed,
            (probabilities <= 0.0).to(dtype=observed.dtype),
        ) > 0.0
        multinomial_log = torch.where(
            impossible,
            -torch.inf,
            multinomial_log,
        )
        mark_log = multinomial_log
        if self.mark_concentration_source is not None:
            source_total = torch.sum(node_source, dim=-1)
            source_fraction = torch.where(
                pre_total > 0.0,
                source_total / torch.clamp(pre_total, min=tiny),
                torch.zeros_like(source_total),
            )
            concentration = float(self.mark_concentration_source) / torch.clamp(
                torch.square(source_fraction),
                min=1.0e-12,
            )
            alpha = probabilities * concentration.unsqueeze(-1)
            dirichlet_sum = torch.zeros_like(multinomial_log)
            for start in range(
                0,
                int(observed.shape[-1]),
                CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
            ):
                stop = min(
                    start + CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
                    int(observed.shape[-1]),
                )
                observed_chunk = observed[..., start:stop]
                alpha_chunk = alpha[..., start:stop]
                expanded_alpha = alpha_chunk.unsqueeze(-5)
                expanded_observed = (
                    observed_chunk.unsqueeze(-3).unsqueeze(-3)
                )
                active_increment = (
                    (expanded_alpha > 0.0)
                    & (expanded_observed > 0.0)
                )
                safe_alpha = torch.where(
                    active_increment,
                    expanded_alpha,
                    torch.ones_like(expanded_alpha),
                )
                safe_observed = torch.where(
                    active_increment,
                    expanded_observed,
                    torch.ones_like(expanded_observed),
                )
                dirichlet_sum = dirichlet_sum + torch.sum(
                    torch.where(
                        active_increment,
                        torch.log(safe_alpha)
                        + torch.lgamma(safe_alpha + safe_observed)
                        - torch.lgamma(safe_alpha + 1.0),
                        torch.zeros_like(safe_alpha),
                    ),
                    dim=-1,
                )
            dirichlet_log = (
                torch.lgamma(observed_total + 1.0)
                .unsqueeze(-2)
                .unsqueeze(-2)
                - torch.sum(
                    torch.lgamma(observed + 1.0),
                    dim=-1,
                )
                .unsqueeze(-2)
                .unsqueeze(-2)
                + torch.lgamma(concentration).unsqueeze(-4)
                - torch.lgamma(
                    concentration.unsqueeze(-4)
                    + observed_total.unsqueeze(-2).unsqueeze(-2)
                )
                + dirichlet_sum
            )
            dirichlet_log = torch.where(
                impossible,
                -torch.inf,
                dirichlet_log,
            )
            mark_log = torch.where(
                source_fraction.unsqueeze(-4) > 0.0,
                dirichlet_log,
                multinomial_log,
            )
        zero_mark_total = (
            observed_total.unsqueeze(-2).unsqueeze(-2) == 0.0
        )
        mark_log = torch.where(
            zero_mark_total,
            torch.zeros_like(mark_log),
            mark_log,
        )
        node_log = torch.sum(count_log + mark_log, dim=-1)
        weight_shape = (1,) * (node_log.ndim - 1) + (
            int(node_weights.numel()),
        )
        return torch.logsumexp(
            node_log + torch.log(node_weights).reshape(weight_shape),
            dim=-1,
        )

    @staticmethod
    def _resolved_cross_chunk_size(
        value: int | None,
        *,
        total: int,
        default: int,
        label: str,
    ) -> int:
        """Return a positive cross-likelihood chunk size bounded by its axis."""
        if int(total) <= 0:
            raise ValueError(f"{label} axis must be nonempty.")
        if value is None:
            resolved = int(default)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{label} chunk size must be a positive integer.")
            resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{label} chunk size must be a positive integer.")
        return min(resolved, int(total))

    def estimate_cross_likelihood_working_set_bytes(
        self,
        *,
        num_actions: int,
        num_samples: int,
        num_particles: int,
        num_isotopes: int,
        num_views: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        dtype_bytes: int = 8,
    ) -> int:
        """Conservatively estimate one chunk of exact likelihood workspace."""
        counts = {
            "num_actions": num_actions,
            "num_samples": num_samples,
            "num_particles": num_particles,
            "num_isotopes": num_isotopes,
            "num_views": num_views,
            "dtype_bytes": dtype_bytes,
        }
        for label, raw_value in counts.items():
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, np.integer))
                or int(raw_value) <= 0
            ):
                raise ValueError(f"{label} must be a positive integer.")
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=int(num_actions),
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=int(num_samples),
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=int(num_particles),
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        bin_count = int(np.asarray(self.energy_axis_keV).size)
        bin_chunk = min(CROSS_LIKELIHOOD_BIN_CHUNK_SIZE, bin_count)
        node_count = int(self._rate_scale_nodes_j.size)
        line_count = len(self._line_identity)
        expanded = (
            action_chunk
            * sample_chunk
            * state_chunk
            * node_count
            * int(num_views)
            * bin_chunk
        )
        # lgamma/where evaluation can hold several simultaneous dense
        # temporaries.  Eight copies is deliberately conservative for both
        # NumPy and Torch allocator behaviour.
        dirichlet_temporaries = 8 * expanded
        marked_state = (
            action_chunk
            * state_chunk
            * int(num_views)
            * bin_count
        )
        observed = (
            action_chunk
            * sample_chunk
            * int(num_views)
            * bin_count
        )
        transport_inputs = (
            action_chunk
            * state_chunk
            * int(num_views)
            * int(num_isotopes)
            * line_count
            * (2 + len(TRANSPORT_FEATURE_ORDER))
        )
        output_and_reductions = (
            6
            * action_chunk
            * sample_chunk
            * state_chunk
            * node_count
            * int(num_views)
        )
        total_elements = (
            dirichlet_temporaries
            + 10 * marked_state
            + 3 * observed
            + transport_inputs
            + output_and_reductions
        )
        return int(total_elements * int(dtype_bytes))

    def cross_log_likelihood_numpy(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return exact cross likelihoods with bounded action/sample/state memory."""
        observed = np.asarray(observed_spectra_xqvb, dtype=np.float64)
        total = np.asarray(total_line_contributions_xnvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_xnvsl,
            dtype=np.float64,
        )
        features = np.asarray(transport_features_xnvslf, dtype=np.float64)
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Cross-likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError(
                "Cross spectra and transport states require identical action axes."
            )
        if (
            uncollided.shape != total.shape
            or features.shape
            != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1])
            != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Cross-likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        observed_flat = observed.reshape(
            (action_count,) + tuple(observed.shape[-3:])
        )
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape(
            (action_count,) + tuple(features.shape[-5:])
        )
        result = np.empty(
            (action_count, sample_count, state_count),
            dtype=np.float64,
        )
        for action_start in range(0, action_count, action_chunk):
            action_stop = min(action_start + action_chunk, action_count)
            for state_start in range(0, state_count, state_chunk):
                state_stop = min(state_start + state_chunk, state_count)
                total_block = total_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                uncollided_block = uncollided_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                features_block = features_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    result[
                        action_start:action_stop,
                        sample_start:sample_stop,
                        state_start:state_stop,
                    ] = self._cross_log_likelihood_numpy_unchunked(
                        observed_flat[
                            action_start:action_stop,
                            sample_start:sample_stop,
                        ],
                        total_block,
                        uncollided_block,
                        features_block,
                        live_times_s_v,
                    )
        return result.reshape(leading_shape + (sample_count, state_count))

    def cross_log_likelihood_torch(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> object:
        """Return Torch cross likelihoods with bounded deterministic workspace."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        observed = torch.as_tensor(
            observed_spectra_xqvb,
            device=total.device,
            dtype=total.dtype,
        )
        uncollided = torch.as_tensor(
            uncollided_line_contributions_xnvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            transport_features_xnvslf,
            device=total.device,
            dtype=total.dtype,
        )
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Torch cross-likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError(
                "Torch spectra and states require identical action axes."
            )
        if (
            tuple(uncollided.shape) != tuple(total.shape)
            or tuple(features.shape)
            != tuple(total.shape) + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1])
            != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Torch cross-likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        observed_flat = observed.reshape(
            (action_count,) + tuple(observed.shape[-3:])
        )
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape(
            (action_count,) + tuple(features.shape[-5:])
        )
        result = torch.empty(
            (action_count, sample_count, state_count),
            device=total.device,
            dtype=total.dtype,
        )
        for action_start in range(0, action_count, action_chunk):
            action_stop = min(action_start + action_chunk, action_count)
            for state_start in range(0, state_count, state_chunk):
                state_stop = min(state_start + state_chunk, state_count)
                total_block = total_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                uncollided_block = uncollided_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                features_block = features_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    result[
                        action_start:action_stop,
                        sample_start:sample_stop,
                        state_start:state_stop,
                    ] = self._cross_log_likelihood_torch_unchunked(
                        observed_flat[
                            action_start:action_stop,
                            sample_start:sample_stop,
                        ],
                        total_block,
                        uncollided_block,
                        features_block,
                        live_times_s_v,
                    )
        return result.reshape(leading_shape + (sample_count, state_count))

    def log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one joint full-spectrum log likelihood per particle."""
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        if observed.ndim != 2:
            raise ValueError("One station observation must be view x bin.")
        return self.cross_log_likelihood_numpy(
            observed[np.newaxis, ...],
            total_line_contributions_nvsl,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )[0]

    def log_likelihood_torch(
        self,
        observed_spectrum_vb: object,
        total_line_contributions_nvsl: object,
        uncollided_line_contributions_nvsl: object,
        transport_features_nvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch station likelihood for every particle."""
        import torch

        total = torch.as_tensor(total_line_contributions_nvsl)
        observed = torch.as_tensor(
            observed_spectrum_vb,
            device=total.device,
            dtype=total.dtype,
        )
        if observed.ndim != 2:
            raise ValueError("One Torch station observation must be view x bin.")
        return self.cross_log_likelihood_torch(
            observed.unsqueeze(0),
            total,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )[0]

    def sample_predictive_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        sample_count: int,
        rng: np.random.Generator,
        action_seeds_a: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Draw shared-scale renewal totals and calibrated energy marks."""
        if int(sample_count) <= 0:
            raise ValueError("sample_count must be positive.")
        if action_seeds_a is not None:
            total = np.asarray(
                total_line_contributions_xvsl,
                dtype=np.float64,
            )
            uncollided = np.asarray(
                uncollided_line_contributions_xvsl,
                dtype=np.float64,
            )
            features = np.asarray(
                transport_features_xvslf,
                dtype=np.float64,
            )
            seeds = np.asarray(action_seeds_a)
            if (
                total.ndim < 4
                or seeds.ndim != 1
                or seeds.shape != (int(total.shape[0]),)
                or not np.issubdtype(seeds.dtype, np.integer)
                or uncollided.shape != total.shape
                or features.shape
                != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            ):
                raise ValueError(
                    "action_seeds_a must provide one integer seed for the "
                    "leading action axis."
                )
            action_samples = []
            for action_index, raw_seed in enumerate(seeds):
                seed = int(raw_seed) & ((1 << 64) - 1)
                action_rng = np.random.Generator(np.random.Philox(seed))
                action_samples.append(
                    self.sample_predictive_numpy(
                        total[action_index],
                        uncollided[action_index],
                        features[action_index],
                        live_times_s_v,
                        sample_count=int(sample_count),
                        rng=action_rng,
                    )
                )
            return np.stack(action_samples, axis=0).astype(
                np.int64,
                copy=False,
            )
        source_mean, background_mean = self._pre_dead_time_mean_numpy(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        live = np.asarray(live_times_s_v, dtype=np.float64)
        leading_shape = source_mean.shape[:-2]
        node_indices = rng.choice(
            self._rate_scale_nodes_j.size,
            size=leading_shape + (int(sample_count),),
            p=self._rate_scale_weights_j,
        )
        sampled_scale = self._rate_scale_nodes_j[node_indices]
        node_source = (
            source_mean[..., np.newaxis, :, :]
            * sampled_scale[..., np.newaxis, np.newaxis]
        )
        pre_mean = background_mean[..., np.newaxis, :, :] + node_source
        pre_total = np.sum(pre_mean, axis=-1)
        rates = pre_total / live
        totals = sample_nonparalyzable_counts_numpy(
            rates,
            np.broadcast_to(live, rates.shape),
            dead_time_tau_s=float(self.dead_time_tau_s),
            sample_count=1,
            rng=rng,
        )[..., 0]
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        zero_rate = pre_total <= 0.0
        if np.any(zero_rate):
            fallback = np.zeros_like(probabilities)
            fallback[..., 0] = 1.0
            probabilities = np.where(
                zero_rate[..., np.newaxis],
                fallback,
                probabilities,
            )
        if self.mark_concentration_source is not None:
            source_total = np.sum(node_source, axis=-1)
            source_fraction = np.divide(
                source_total,
                pre_total,
                out=np.zeros_like(source_total),
                where=pre_total > 0.0,
            )
            concentration = float(
                self.mark_concentration_source
            ) / np.maximum(np.square(source_fraction), 1.0e-12)
            alpha = probabilities * concentration[..., np.newaxis]
            positive_alpha = alpha > 0.0
            gamma_draws = rng.gamma(
                shape=np.where(positive_alpha, alpha, 1.0),
            )
            gamma_draws = np.where(
                positive_alpha,
                gamma_draws,
                0.0,
            )
            random_probabilities = np.divide(
                gamma_draws,
                np.sum(gamma_draws, axis=-1, keepdims=True),
                out=probabilities.copy(),
                where=np.sum(
                    gamma_draws,
                    axis=-1,
                    keepdims=True,
                )
                > 0.0,
            )
            probabilities = np.where(
                source_fraction[..., np.newaxis] > 0.0,
                random_probabilities,
                probabilities,
            )
        samples = rng.multinomial(
            totals,
            probabilities,
        )
        return np.asarray(samples, dtype=np.int64)

    def _birth_proposal_nuisance_basis_numpy(
        self,
        target_line_mask_l: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return a fixed whitened orthonormal non-target line subspace."""
        mask = np.asarray(target_line_mask_l, dtype=np.bool_)
        if mask.shape != (len(self._line_identity),) or not np.any(mask):
            raise ValueError(
                "target_line_mask_l must select at least one global line."
            )
        key = np.ascontiguousarray(mask).tobytes()
        cached = self._proposal_basis_cache.get(key)
        if cached is not None:
            return cached
        nuisance_direct = self._marked_direct_line_shapes_lb[~mask]
        nuisance_scatter = self._marked_scatter_order_shapes_lob[
            ~mask
        ].reshape(-1, self._energy_axis_keV.size)
        nuisance = np.concatenate(
            (nuisance_direct, nuisance_scatter),
            axis=0,
        )
        whitening = 1.0 / np.sqrt(
            self.background_shape_b
            + 1.0 / float(self._energy_axis_keV.size)
        )
        whitened = nuisance * whitening[np.newaxis, :]
        if whitened.shape[0] == 0:
            basis = np.zeros(
                (self._energy_axis_keV.size, 0),
                dtype=np.float64,
            )
        else:
            basis, _ = np.linalg.qr(whitened.T, mode="reduced")
        basis = np.ascontiguousarray(basis, dtype=np.float64)
        basis.setflags(write=False)
        self._proposal_basis_cache[key] = basis
        return basis

    def _birth_proposal_candidate_chunk_size(self, view_count: int) -> int:
        """Return a conservative candidate chunk under the memory cap."""
        values_per_candidate = (
            int(view_count)
            * int(self._energy_axis_keV.size)
            * (8 + 3 * int(self._rate_scale_nodes_j.size))
        )
        bytes_per_candidate = max(values_per_candidate * 8, 1)
        return max(
            1,
            int(BIRTH_PROPOSAL_WORKING_SET_BYTES // bytes_per_candidate),
        )

    def birth_proposal_log_scores_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        candidate_total_line_contributions_gvsl: NDArray[np.float64],
        candidate_uncollided_line_contributions_gvsl: NDArray[np.float64],
        candidate_transport_features_gvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        target_line_mask_l: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return deterministic proposal-only chart-by-strength log scores."""
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        total = np.asarray(
            candidate_total_line_contributions_gvsl,
            dtype=np.float64,
        )
        mask = np.asarray(target_line_mask_l, dtype=np.bool_)
        if (
            observed.shape != (
                int(total.shape[-3]),
                int(self._energy_axis_keV.size),
            )
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
            or np.any(observed != np.floor(observed))
            or mask.shape != (len(self._line_identity),)
            or np.any(total[..., ~mask] != 0.0)
        ):
            raise ValueError(
                "Birth proposal candidates must contain exact observed counts "
                "and target-isotope line rates only."
            )
        uncollided = np.asarray(
            candidate_uncollided_line_contributions_gvsl,
            dtype=np.float64,
        )
        features = np.asarray(
            candidate_transport_features_gvslf,
            dtype=np.float64,
        )
        zero_total = np.zeros(
            (1,) + total.shape[-3:],
            dtype=np.float64,
        )
        zero_features = np.zeros(
            zero_total.shape + (len(TRANSPORT_FEATURE_ORDER),),
            dtype=np.float64,
        )
        baseline = self.predict_mean_numpy(
            zero_total,
            zero_total,
            zero_features,
            live_times_s_v,
        )[0]
        whitening = 1.0 / np.sqrt(
            baseline + 1.0
        )
        residual = (observed - baseline) * whitening
        basis = self._birth_proposal_nuisance_basis_numpy(mask)
        if basis.shape[1] > 0:
            residual = residual - (residual @ basis) @ basis.T
        scores = np.empty(int(total.shape[0]), dtype=np.float64)
        chunk_size = self._birth_proposal_candidate_chunk_size(
            int(total.shape[-3])
        )
        for start in range(0, int(total.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(total.shape[0]))
            candidate_mean = self.predict_mean_numpy(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times_s_v,
            )
            templates = (
                candidate_mean - baseline[np.newaxis, ...]
            ) * whitening
            if basis.shape[1] > 0:
                coefficients = np.einsum(
                    "gvb,bj->gvj",
                    templates,
                    basis,
                    optimize=True,
                )
                templates = templates - np.einsum(
                    "gvj,bj->gvb",
                    coefficients,
                    basis,
                    optimize=True,
                )
            correlation = np.einsum(
                "vb,gvb->g",
                residual,
                templates,
                optimize=True,
            )
            energy = np.einsum(
                "gvb,gvb->g",
                templates,
                templates,
                optimize=True,
            )
            scores[start:stop] = correlation - 0.5 * energy
        if scores.shape != (int(total.shape[0]),) or np.any(
            ~np.isfinite(scores)
        ):
            raise RuntimeError("Birth proposal score is not finite and aligned.")
        return np.asarray(scores, dtype=np.float64)

    def birth_proposal_log_scores_torch(
        self,
        observed_spectrum_vb: object,
        candidate_total_line_contributions_gvsl: object,
        candidate_uncollided_line_contributions_gvsl: object,
        candidate_transport_features_gvslf: object,
        live_times_s_v: object,
        *,
        target_line_mask_l: object,
    ) -> object:
        """Return the Torch-equivalent proposal-only matched-filter scores."""
        import torch

        total = torch.as_tensor(candidate_total_line_contributions_gvsl)
        if total.dtype != torch.float64:
            raise TypeError("Birth proposal scoring requires torch.float64.")
        observed = torch.as_tensor(
            observed_spectrum_vb,
            device=total.device,
            dtype=torch.float64,
        )
        mask = torch.as_tensor(
            target_line_mask_l,
            device=total.device,
            dtype=torch.bool,
        )
        if (
            observed.shape
            != (int(total.shape[-3]), int(self._energy_axis_keV.size))
            or tuple(mask.shape) != (len(self._line_identity),)
            or bool(torch.any(~torch.isfinite(observed)))
            or bool(torch.any(observed < 0.0))
            or bool(torch.any(observed != torch.floor(observed)))
            or bool(torch.any(total[..., ~mask] != 0.0))
        ):
            raise ValueError("Torch birth proposal inputs are invalid.")
        uncollided = torch.as_tensor(
            candidate_uncollided_line_contributions_gvsl,
            device=total.device,
            dtype=torch.float64,
        )
        features = torch.as_tensor(
            candidate_transport_features_gvslf,
            device=total.device,
            dtype=torch.float64,
        )
        zero_total = torch.zeros(
            (1,) + tuple(total.shape[-3:]),
            device=total.device,
            dtype=torch.float64,
        )
        zero_features = torch.zeros(
            tuple(zero_total.shape) + (len(TRANSPORT_FEATURE_ORDER),),
            device=total.device,
            dtype=torch.float64,
        )
        baseline = self.predict_mean_torch(
            zero_total,
            zero_total,
            zero_features,
            live_times_s_v,
        )[0]
        whitening = torch.rsqrt(baseline + 1.0)
        residual = (observed - baseline) * whitening
        basis = torch.as_tensor(
            np.array(
                self._birth_proposal_nuisance_basis_numpy(
                    mask.detach().cpu().numpy(),
                ),
                copy=True,
            ),
            device=total.device,
            dtype=torch.float64,
        )
        if int(basis.shape[1]) > 0:
            residual = residual - (residual @ basis) @ basis.T
        scores = torch.empty(
            int(total.shape[0]),
            device=total.device,
            dtype=torch.float64,
        )
        chunk_size = self._birth_proposal_candidate_chunk_size(
            int(total.shape[-3])
        )
        for start in range(0, int(total.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(total.shape[0]))
            candidate_mean = self.predict_mean_torch(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times_s_v,
            )
            templates = (
                candidate_mean - baseline.unsqueeze(0)
            ) * whitening
            if int(basis.shape[1]) > 0:
                coefficients = torch.einsum(
                    "gvb,bj->gvj",
                    templates,
                    basis,
                )
                templates = templates - torch.einsum(
                    "gvj,bj->gvb",
                    coefficients,
                    basis,
                )
            correlation = torch.einsum(
                "vb,gvb->g",
                residual,
                templates,
            )
            energy = torch.einsum(
                "gvb,gvb->g",
                templates,
                templates,
            )
            scores[start:stop] = correlation - 0.5 * energy
        if tuple(scores.shape) != (int(total.shape[0]),) or bool(
            torch.any(~torch.isfinite(scores))
        ):
            raise RuntimeError("Torch birth proposal scores are invalid.")
        return scores

    def posterior_predictive_innovation_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        particle_weights_n: NDArray[np.float64],
        *,
        confidence: float,
    ) -> Mapping[str, float | int | bool | None]:
        """Return renewal-total and conditional-mark posterior diagnostics."""
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        weights = np.asarray(particle_weights_n, dtype=np.float64)
        means = self.predict_mean_numpy(
            total_line_contributions_nvsl,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )
        if (
            weights.shape != (means.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("Posterior innovation weights are invalid.")
        normalized = weights / float(np.sum(weights))
        posterior_mean = np.einsum(
            "n,nvb->vb",
            normalized,
            means,
            optimize=True,
        )
        observed_total = np.sum(observed, axis=-1)
        predicted_total = np.sum(posterior_mean, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        incident_rate = predicted_total / np.maximum(
            live - predicted_total * float(self.dead_time_tau_s),
            np.finfo(np.float64).tiny,
        )
        renewal_variance = (
            incident_rate
            * live
            / np.power(
                1.0 + incident_rate * float(self.dead_time_tau_s),
                3.0,
            )
        )
        total_z = (observed_total - predicted_total) / np.sqrt(
            np.maximum(renewal_variance, 1.0)
        )
        probabilities = np.divide(
            posterior_mean,
            predicted_total[:, np.newaxis],
            out=np.zeros_like(posterior_mean),
            where=predicted_total[:, np.newaxis] > 0.0,
        )
        expected_marks = observed_total[:, np.newaxis] * probabilities
        mark_pearson = float(
            np.sum(
                np.square(observed - expected_marks)
                / np.maximum(expected_marks, 1.0)
            )
        )
        degrees = int(
            np.sum(expected_marks >= 1.0)
            - observed.shape[0]
        )
        mark_tail_probability = (
            float(stats.chi2.sf(mark_pearson, degrees))
            if degrees > 0
            else None
        )
        threshold = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
        maximum_total_z = float(np.max(np.abs(total_z)))
        return {
            "renewal_total_max_abs_z": maximum_total_z,
            "renewal_total_within_confidence": maximum_total_z <= threshold,
            "conditional_mark_pearson": mark_pearson,
            "conditional_mark_degrees_of_freedom": degrees,
            "conditional_mark_tail_probability": mark_tail_probability,
            "confidence": float(confidence),
        }

    def manifest_payload(self) -> Mapping[str, object]:
        """Return immutable physics and validation provenance."""
        bin_width = float(self._energy_axis_keV[1] - self._energy_axis_keV[0])
        mark_model = (
            "source_fraction_dirichlet_multinomial"
            if self.mark_concentration_source is not None
            else "exact_multinomial_diagnostic_only"
        )
        return {
            "schema_version": 3,
            "model": "geometry_conditioned_full_spectrum",
            "contract_hash_sha256": self.contract_hash_sha256,
            "production_ready": self.production_ready,
            "energy_bin_count": int(self._energy_axis_keV.size),
            "energy_min_keV": float(self._energy_axis_keV[0]),
            "energy_max_keV": float(self._energy_axis_keV[-1]),
            "bin_width_keV": bin_width,
            "transport_feature_order": list(TRANSPORT_FEATURE_ORDER),
            "additive_noncollided_transport_response": (
                None
                if self.additive_scatter_response is None
                else self.additive_scatter_response.to_payload()
            ),
            "line_identity": [dict(item) for item in self._line_identity],
            "source_rate_semantics": (
                "pre_dead_time_detector_pulse_rate_at_1m"
            ),
            "direct_partition": "minimum_of_total_and_uncollided",
            "scatter_partition": "total_minus_direct",
            "scatter_shape": "klein_nishina_optical_depth_orders",
            "maximum_scatter_order": int(self.maximum_scatter_order),
            "detector_response_sampling": (
                "multinomial_marking_with_nonparalyzable_event_time"
            ),
            "detector_response_contract_sha256": (
                NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            ),
            "dead_time_model": (
                "nonparalyzable_event_time_renewal_total"
            ),
            "dead_time_tau_s": float(self.dead_time_tau_s),
            "dead_time_application_count": 1,
            "background_rate_cps": float(self.background_rate_cps),
            "background_model": (
                "native_geant4_background_shape_v1_bin_centres"
            ),
            "background_semantics": (
                "independent_pre_dead_time_pulse_rate_added_once"
            ),
            "rate_scale_mixture": {
                "scope": "station_shared_source_only",
                "nodes": self._rate_scale_nodes_j.tolist(),
                "weights": self._rate_scale_weights_j.tolist(),
                "weighted_mean": float(
                    np.sum(
                        self._rate_scale_nodes_j
                        * self._rate_scale_weights_j
                    )
                ),
            },
            "mark_model": mark_model,
            "mark_concentration_source": (
                None
                if self.mark_concentration_source is None
                else float(self.mark_concentration_source)
            ),
            "discrepancy_training_ready": self.discrepancy_training_ready,
            "discrepancy_training": (
                None
                if self.discrepancy_training_manifest is None
                else _thaw_json_value(self.discrepancy_training_manifest)
            ),
            "discrepancy_training_manifest_sha256": (
                self._discrepancy_training_manifest_sha256
            ),
            "validation": (
                None
                if self.validation_manifest is None
                else _thaw_json_value(self.validation_manifest)
            ),
            "validation_manifest_sha256": self._validation_manifest_sha256,
            "acceptance_contract_sha256": (
                FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            ),
        }


def geometry_conditioned_model_from_runtime_config(
    runtime_config: Mapping[str, object],
    *,
    run_root: str | Path | None = None,
) -> GeometryConditionedSpectralModel:
    """Reconstruct and verify the sole live/replay spectrum contract."""
    if not isinstance(runtime_config, Mapping):
        raise TypeError("Resolved runtime configuration must be a mapping.")
    inline_payload = runtime_config.get("full_spectrum_generative_model")
    path_value = runtime_config.get("full_spectrum_generative_model_path")
    if inline_payload is not None and path_value is not None:
        raise ValueError(
            "Full-spectrum runtime must select exactly one inline or "
            "file-backed generative model."
        )
    if inline_payload is None and path_value is None:
        raise ValueError(
            "Resolved runtime requires one full-spectrum generative model."
        )
    if path_value is None:
        if not isinstance(inline_payload, Mapping):
            raise ValueError(
                "full_spectrum_generative_model must be a mapping."
            )
        if "full_spectrum_generative_model_file_sha256" in runtime_config:
            raise ValueError(
                "Inline full-spectrum models cannot declare a file digest."
            )
        payload = inline_payload
    else:
        declared_file_hash = runtime_config.get(
            "full_spectrum_generative_model_file_sha256"
        )
        if not _is_sha256(declared_file_hash):
            raise ValueError(
                "File-backed full-spectrum models require an exact SHA-256."
            )
        resolved_path = resolve_file_backed_model_asset(
            path_value,
            field_name="full_spectrum_generative_model_path",
            run_root=run_root,
        )
        raw_bytes = resolved_path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != declared_file_hash:
            raise ValueError(
                "Full-spectrum model file SHA-256 does not match the "
                "configured digest."
            )
        try:
            decoded_payload = json.loads(
                raw_bytes,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Full-spectrum model asset must be canonical JSON."
            ) from exc
        if not isinstance(decoded_payload, Mapping):
            raise ValueError(
                "Full-spectrum model asset must contain one manifest mapping."
            )
        payload = decoded_payload
    model = GeometryConditionedSpectralModel.from_manifest_payload(payload)
    declared_hash = runtime_config.get(
        "full_spectrum_contract_hash_sha256"
    )
    if declared_hash != model.contract_hash_sha256:
        raise ValueError(
            "Resolved runtime full-spectrum hash does not match its model."
        )
    expected_numeric = {
        "energy_min_keV": float(model.energy_axis_keV[0]),
        "energy_max_keV": float(model.energy_axis_keV[-1]),
        "bin_width_keV": float(
            model.energy_axis_keV[1] - model.energy_axis_keV[0]
        ),
        "energy_bin_count": int(model.energy_axis_keV.size),
        "background_rate_cps": float(model.background_rate_cps),
        "dead_time_tau_s": float(model.dead_time_tau_s),
    }
    for key, expected in expected_numeric.items():
        value = runtime_config.get(key)
        if key == "energy_bin_count":
            valid = (
                not isinstance(value, bool)
                and isinstance(value, int)
                and value == expected
            )
        else:
            valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                and np.isclose(
                    float(value),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
        if not valid:
            raise ValueError(
                f"Resolved runtime {key} disagrees with the full-spectrum model."
            )
    if runtime_config.get("source_rate_model") != "detector_cps_1m":
        raise ValueError(
            "Full-spectrum runtime requires source_rate_model=detector_cps_1m."
        )
    return model
