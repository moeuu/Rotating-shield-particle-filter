"""Evaluate the frozen full-spectrum candidate on immutable all-64 corpora.

The evaluator is deliberately downstream-only: it reconstructs the candidate
that was frozen before holdout acquisition, reads authenticated raw pair
artifacts, and never invokes fitting or parameter selection.  Production
predictions receive only spectra and known transport geometry.  Validation-only
native entry labels are not passed to any prediction, likelihood, decision, or
calibration calculation.

Pairwise coverage and total PIT diagnostics use the exact finite-time
nonparalyzable renewal CDF mixed over the frozen rate-scale nodes. Conditional
mark PIT diagnostics use a fixed, deterministic batched posterior-predictive
rank test with the frozen Dirichlet-multinomial concentration.  No holdout
quantity changes the model, metric definition, threshold, or simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from measurement.source_boundary import surface_emission_policy_sha256
from spectrum.full_spectrum_acceptance import (
    build_independent_validation_manifest,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_DWELL_TIME_S,
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_PAIR_IDS,
    AcceptancePairRecord,
    AcceptanceRunLayout,
    canonical_json_bytes,
    line_identity_contract_sha256,
    load_frozen_candidate_model,
    validate_scene_corpus,
)
from spectrum.response_matrix import (
    build_native_geant4_detector_response_matrix,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
    nonparalyzable_count_cdf_numpy,
)


CONDITIONAL_MARK_PREDICTIVE_SAMPLE_COUNT = 512
CONDITIONAL_MARK_PREDICTIVE_CHUNK_SIZE = 32
K_ZERO_PRIOR_PROBABILITY = 0.95
_EVALUATION_RNG_DOMAIN = "full_spectrum_acceptance_evaluation_v1"


@dataclass(frozen=True)
class _ScenarioData:
    """Store one label-free production projection of an all-64 scenario."""

    scenario_id: str
    observed_vb: NDArray[np.float64]
    total_vsl: NDArray[np.float64]
    uncollided_vsl: NDArray[np.float64]
    features_vslf: NDArray[np.float64]
    source_isotopes: tuple[str, ...]
    perturbed_total_vsl: NDArray[np.float64] | None
    perturbed_uncollided_vsl: NDArray[np.float64] | None
    perturbed_features_vslf: NDArray[np.float64] | None


@dataclass(frozen=True)
class _TotalDiagnostics:
    """Store exact-mixture total-count diagnostics for all shield pairs."""

    mean_v: NDArray[np.float64]
    variance_v: NDArray[np.float64]
    randomized_pit_v: NDArray[np.float64]
    standardized_residual_v: NDArray[np.float64]


@dataclass(frozen=True)
class _MarkDiagnostics:
    """Store deterministic conditional-mark predictive rank diagnostics."""

    randomized_pit_v: NDArray[np.float64]
    upper_tail_probability_v: NDArray[np.float64]


def _atomic_write_immutable_json(path: Path, payload: object) -> Path:
    """Write canonical JSON atomically or authenticate an exact resume."""
    destination = Path(path).resolve()
    encoded = canonical_json_bytes(payload)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise RuntimeError(
                f"Refusing to overwrite incompatible artifact: {destination}."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination


def _load_mapping(path: Path) -> Mapping[str, object]:
    """Load one immutable canonical JSON object."""
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must contain an object: {path}.")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}.") from exc
    if raw != canonical:
        raise ValueError(
            f"JSON artifact is not immutable canonical JSON: {path}."
        )
    return payload


def _evaluation_seed(scene_seed: int, *tokens: object) -> int:
    """Return one deterministic evaluation-only Philox seed."""
    identity = "|".join(
        (_EVALUATION_RNG_DOMAIN, str(int(scene_seed)), *(str(x) for x in tokens))
    )
    return int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _source_isotopes(record: AcceptancePairRecord) -> tuple[str, ...]:
    """Return exact source-slot isotope order from one authenticated pair."""
    payload = _load_mapping(record.path)
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise TypeError("Pair sources must be a JSON array.")
    isotopes: list[str] = []
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise TypeError("Pair source rows must be objects.")
        isotope = source.get("isotope")
        if not isinstance(isotope, str) or isotope not in ACCEPTANCE_ISOTOPES:
            raise ValueError("Pair source isotope is incompatible.")
        isotopes.append(isotope)
    return tuple(isotopes)


def _scenario_data(
    records: Sequence[AcceptancePairRecord],
    *,
    model: GeometryConditionedSpectralModel,
) -> _ScenarioData:
    """Project authenticated records into the label-free production tensors."""
    ordered = tuple(sorted(records, key=lambda record: record.shield_pair_id))
    if (
        len(ordered) != len(ACCEPTANCE_PAIR_IDS)
        or tuple(record.shield_pair_id for record in ordered)
        != ACCEPTANCE_PAIR_IDS
        or len({record.scenario_id for record in ordered}) != 1
        or len({record.scene_seed for record in ordered}) != 1
        or model.additive_scatter_response is None
    ):
        raise ValueError("Scenario records are incomplete or model-incompatible.")
    source_isotopes = _source_isotopes(ordered[0])
    if any(_source_isotopes(record) != source_isotopes for record in ordered):
        raise ValueError("Source-slot isotope order changed across shield pairs.")
    unattenuated = np.concatenate(
        [record.unattenuated_vsl for record in ordered],
        axis=0,
    )
    uncollided = np.concatenate(
        [record.uncollided_vsl for record in ordered],
        axis=0,
    )
    features = np.concatenate(
        [record.features_vslf for record in ordered],
        axis=0,
    )
    scatter_basis = np.concatenate(
        [record.scatter_basis_vslf for record in ordered],
        axis=0,
    )
    total = model.additive_scatter_response.total_kernel_numpy(
        unattenuated,
        uncollided,
        scatter_basis,
    )
    observed = np.stack(
        [record.observed_spectrum_counts for record in ordered],
        axis=0,
    ).astype(np.float64)
    scenario_id = ordered[0].scenario_id
    if scenario_id == "continuous_surface_perturbation_ranking":
        perturbed_unattenuated = np.concatenate(
            [record.perturbed_unattenuated_vsl for record in ordered],
            axis=0,
        )
        perturbed_uncollided = np.concatenate(
            [record.perturbed_uncollided_vsl for record in ordered],
            axis=0,
        )
        perturbed_features = np.concatenate(
            [record.perturbed_features_vslf for record in ordered],
            axis=0,
        )
        perturbed_basis = np.concatenate(
            [record.perturbed_scatter_basis_vslf for record in ordered],
            axis=0,
        )
        perturbed_total = (
            model.additive_scatter_response.total_kernel_numpy(
                perturbed_unattenuated,
                perturbed_uncollided,
                perturbed_basis,
            )
        )
    else:
        perturbed_total = None
        perturbed_uncollided = None
        perturbed_features = None
    return _ScenarioData(
        scenario_id=scenario_id,
        observed_vb=observed,
        total_vsl=total,
        uncollided_vsl=uncollided,
        features_vslf=features,
        source_isotopes=source_isotopes,
        perturbed_total_vsl=perturbed_total,
        perturbed_uncollided_vsl=perturbed_uncollided,
        perturbed_features_vslf=perturbed_features,
    )


def _total_diagnostics(
    data: _ScenarioData,
    *,
    model: GeometryConditionedSpectralModel,
    scene_seed: int,
) -> _TotalDiagnostics:
    """Return exact-mixture renewal moments and randomized PIT values."""
    live = np.full(
        len(ACCEPTANCE_PAIR_IDS),
        ACCEPTANCE_DWELL_TIME_S,
        dtype=np.float64,
    )
    source_pre_total = (
        np.sum(data.total_vsl, axis=(-2, -1)) * live
    )
    background_pre_total = (
        float(model.background_rate_cps) * live
    )
    nodes = model.rate_scale_nodes
    weights = model.rate_scale_weights
    pre_total_vj = (
        background_pre_total[:, np.newaxis]
        + source_pre_total[:, np.newaxis] * nodes[np.newaxis, :]
    )
    rates_vj = pre_total_vj / live[:, np.newaxis]
    dead_time = float(model.dead_time_tau_s)
    denominator = 1.0 + rates_vj * dead_time
    node_mean_vj = pre_total_vj / denominator
    node_variance_vj = (
        rates_vj * live[:, np.newaxis] / np.power(denominator, 3.0)
    )
    mean_v = np.sum(node_mean_vj * weights[np.newaxis, :], axis=1)
    variance_v = (
        np.sum(
            weights[np.newaxis, :]
            * (node_variance_vj + np.square(node_mean_vj)),
            axis=1,
        )
        - np.square(mean_v)
    )
    variance_v = np.maximum(variance_v, 1.0)
    observed_total = np.sum(data.observed_vb, axis=1).astype(np.int64)
    cdf_upper_vj = nonparalyzable_count_cdf_numpy(
        observed_total[:, np.newaxis],
        rates_vj,
        live[:, np.newaxis],
        dead_time_tau_s=dead_time,
    )
    cdf_lower_vj = nonparalyzable_count_cdf_numpy(
        observed_total[:, np.newaxis] - 1,
        rates_vj,
        live[:, np.newaxis],
        dead_time_tau_s=dead_time,
    )
    cdf_upper = np.sum(cdf_upper_vj * weights[np.newaxis, :], axis=1)
    cdf_lower = np.sum(cdf_lower_vj * weights[np.newaxis, :], axis=1)
    rng = np.random.Generator(
        np.random.Philox(
            _evaluation_seed(scene_seed, data.scenario_id, "total_pit")
        )
    )
    randomized = cdf_lower + rng.random(cdf_lower.shape) * np.maximum(
        cdf_upper - cdf_lower,
        0.0,
    )
    standardized = (
        observed_total.astype(np.float64) - mean_v
    ) / np.sqrt(variance_v)
    return _TotalDiagnostics(
        mean_v=mean_v,
        variance_v=variance_v,
        randomized_pit_v=np.clip(randomized, 0.0, 1.0),
        standardized_residual_v=standardized,
    )


def _mark_diagnostics(
    data: _ScenarioData,
    *,
    model: GeometryConditionedSpectralModel,
    scene_seed: int,
) -> _MarkDiagnostics:
    """Return batched conditional-mark posterior-predictive rank values."""
    live = np.full(
        len(ACCEPTANCE_PAIR_IDS),
        ACCEPTANCE_DWELL_TIME_S,
        dtype=np.float64,
    )
    mean = model.predict_mean_numpy(
        data.total_vsl,
        data.uncollided_vsl,
        data.features_vslf,
        live,
    )
    predicted_total = np.sum(mean, axis=1)
    probabilities = np.divide(
        mean,
        predicted_total[:, np.newaxis],
        out=np.zeros_like(mean),
        where=predicted_total[:, np.newaxis] > 0.0,
    )
    zero = np.sum(probabilities, axis=1) <= 0.0
    probabilities[zero, 0] = 1.0
    observed_total = np.sum(data.observed_vb, axis=1).astype(np.int64)
    expected = observed_total[:, np.newaxis] * probabilities
    observed_statistic = np.sum(
        np.square(data.observed_vb - expected) / np.maximum(expected, 1.0),
        axis=1,
    )
    source_pre_total = (
        np.sum(data.total_vsl, axis=(-2, -1)) * live
    )
    source_fraction = np.divide(
        source_pre_total,
        source_pre_total + float(model.background_rate_cps) * live,
        out=np.zeros_like(source_pre_total),
        where=(
            source_pre_total + float(model.background_rate_cps) * live
        )
        > 0.0,
    )
    concentration_source = model.mark_concentration_source
    if concentration_source is None:
        concentration = np.full_like(source_fraction, np.inf)
    else:
        concentration = float(concentration_source) / np.maximum(
            np.square(source_fraction),
            1.0e-12,
        )
    rng = np.random.Generator(
        np.random.Philox(
            _evaluation_seed(scene_seed, data.scenario_id, "mark_pit")
        )
    )
    less = np.zeros(len(ACCEPTANCE_PAIR_IDS), dtype=np.int64)
    equal = np.zeros_like(less)
    greater_equal = np.zeros_like(less)
    remaining = CONDITIONAL_MARK_PREDICTIVE_SAMPLE_COUNT
    while remaining:
        batch = min(CONDITIONAL_MARK_PREDICTIVE_CHUNK_SIZE, remaining)
        sampled_probabilities = np.broadcast_to(
            probabilities[np.newaxis, :, :],
            (batch,) + probabilities.shape,
        ).copy()
        active = source_fraction > 0.0
        if np.any(active) and concentration_source is not None:
            alpha = (
                probabilities[active]
                * concentration[active, np.newaxis]
            )
            positive = alpha > 0.0
            gamma = rng.gamma(
                shape=np.broadcast_to(
                    np.where(positive, alpha, 1.0)[np.newaxis, :, :],
                    (batch,) + alpha.shape,
                )
            )
            gamma = np.where(positive[np.newaxis, :, :], gamma, 0.0)
            denominator = np.sum(gamma, axis=2, keepdims=True)
            sampled_probabilities[:, active, :] = np.divide(
                gamma,
                denominator,
                out=np.broadcast_to(
                    probabilities[active][np.newaxis, :, :],
                    gamma.shape,
                ).copy(),
                where=denominator > 0.0,
            )
        sampled = rng.multinomial(
            n=np.broadcast_to(
                observed_total[np.newaxis, :],
                (batch, observed_total.size),
            ),
            pvals=sampled_probabilities,
        )
        statistic = np.sum(
            np.square(sampled - expected[np.newaxis, :, :])
            / np.maximum(expected[np.newaxis, :, :], 1.0),
            axis=2,
        )
        tolerance = 1.0e-12 * np.maximum(
            1.0,
            np.abs(observed_statistic)[np.newaxis, :],
        )
        less += np.sum(
            statistic < observed_statistic[np.newaxis, :] - tolerance,
            axis=0,
        )
        equal += np.sum(
            np.abs(statistic - observed_statistic[np.newaxis, :])
            <= tolerance,
            axis=0,
        )
        greater_equal += np.sum(
            statistic >= observed_statistic[np.newaxis, :] - tolerance,
            axis=0,
        )
        remaining -= batch
    tie_random = rng.random(len(ACCEPTANCE_PAIR_IDS))
    randomized = (
        less.astype(np.float64)
        + tie_random * (equal.astype(np.float64) + 1.0)
    ) / float(CONDITIONAL_MARK_PREDICTIVE_SAMPLE_COUNT + 1)
    upper_tail = (
        greater_equal.astype(np.float64) + 1.0
    ) / float(CONDITIONAL_MARK_PREDICTIVE_SAMPLE_COUNT + 1)
    return _MarkDiagnostics(
        randomized_pit_v=np.clip(randomized, 0.0, 1.0),
        upper_tail_probability_v=np.clip(upper_tail, 0.0, 1.0),
    )


def _pad_sources(
    value_vsl: NDArray[np.float64],
    *,
    source_count: int,
) -> NDArray[np.float64]:
    """Pad one source-slot tensor with exact zero-contribution slots."""
    if value_vsl.shape[1] > source_count:
        raise ValueError("Cannot pad to fewer source slots.")
    width = source_count - int(value_vsl.shape[1])
    return np.pad(value_vsl, ((0, 0), (0, width), (0, 0)))


def _pad_source_features(
    value_vslf: NDArray[np.float64],
    *,
    source_count: int,
) -> NDArray[np.float64]:
    """Pad one source-slot feature tensor with exact zero slots."""
    if value_vslf.shape[1] > source_count:
        raise ValueError("Cannot pad to fewer source-feature slots.")
    width = source_count - int(value_vslf.shape[1])
    return np.pad(value_vslf, ((0, 0), (0, width), (0, 0), (0, 0)))


def _pair_log_likelihoods(
    model: GeometryConditionedSpectralModel,
    observed_vb: NDArray[np.float64],
    candidates: Sequence[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ],
) -> NDArray[np.float64]:
    """Evaluate every shield pair and candidate in one cross-likelihood batch."""
    if not candidates:
        raise ValueError("At least one likelihood candidate is required.")
    source_count = max(candidate[0].shape[1] for candidate in candidates)
    total = np.stack(
        [
            _pad_sources(candidate[0], source_count=source_count)
            for candidate in candidates
        ],
        axis=1,
    )
    uncollided = np.stack(
        [
            _pad_sources(candidate[1], source_count=source_count)
            for candidate in candidates
        ],
        axis=1,
    )
    features = np.stack(
        [
            _pad_source_features(candidate[2], source_count=source_count)
            for candidate in candidates
        ],
        axis=1,
    )
    values = model.cross_log_likelihood_numpy(
        observed_vb[:, np.newaxis, np.newaxis, :],
        total[:, :, np.newaxis, :, :],
        uncollided[:, :, np.newaxis, :, :],
        features[:, :, np.newaxis, :, :, :],
        np.asarray([ACCEPTANCE_DWELL_TIME_S], dtype=np.float64),
        action_chunk_size=len(ACCEPTANCE_PAIR_IDS),
        sample_chunk_size=1,
        state_chunk_size=len(candidates),
    )
    result = np.asarray(values[:, 0, :], dtype=np.float64)
    if result.shape != (len(ACCEPTANCE_PAIR_IDS), len(candidates)):
        raise RuntimeError("Pairwise cross-likelihood returned the wrong shape.")
    return result


def _positive_decision_rate(
    log_likelihood_vm: NDArray[np.float64],
) -> float:
    """Return the pair fraction favoring K>0 under the fixed P(K=0)=0.95."""
    values = np.asarray(log_likelihood_vm, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("K-decision likelihoods need one null and alternatives.")
    null = math.log(K_ZERO_PRIOR_PROBABILITY) + values[:, 0]
    alternative = (
        math.log(1.0 - K_ZERO_PRIOR_PROBABILITY)
        + special.logsumexp(values[:, 1:], axis=1)
        - math.log(values.shape[1] - 1)
    )
    posterior_positive = np.exp(
        alternative - np.logaddexp(null, alternative)
    )
    return float(np.mean(posterior_positive > 0.5))


def _isotope_slice(
    data: _ScenarioData,
    isotope: str,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return only source slots belonging to one isotope."""
    indices = np.asarray(
        [
            index
            for index, value in enumerate(data.source_isotopes)
            if value == isotope
        ],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError(f"Scenario has no {isotope} source slot.")
    return (
        data.total_vsl[:, indices, :],
        data.uncollided_vsl[:, indices, :],
        data.features_vslf[:, indices, :, :],
    )


def _concatenate_candidate(
    left: _ScenarioData,
    right: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Concatenate a fixed extra source hypothesis onto one truth state."""
    return (
        np.concatenate((left.total_vsl, right[0]), axis=1),
        np.concatenate((left.uncollided_vsl, right[1]), axis=1),
        np.concatenate((left.features_vslf, right[2]), axis=1),
    )


def _cpu_torch_errors(
    data_by_scenario: Mapping[str, _ScenarioData],
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[float, float]:
    """Return CPU NumPy-versus-Torch mean and likelihood maximum errors."""
    import torch

    maximum_mean = 0.0
    maximum_likelihood = 0.0
    live = np.full(
        len(ACCEPTANCE_PAIR_IDS),
        ACCEPTANCE_DWELL_TIME_S,
        dtype=np.float64,
    )
    for scenario in VALIDATION_SCENARIO_IDS:
        data = data_by_scenario[scenario]
        mean_numpy = model.predict_mean_numpy(
            data.total_vsl,
            data.uncollided_vsl,
            data.features_vslf,
            live,
        )
        total_t = torch.as_tensor(data.total_vsl, dtype=torch.float64)
        mean_torch = model.predict_mean_torch(
            total_t,
            torch.as_tensor(data.uncollided_vsl, dtype=torch.float64),
            torch.as_tensor(data.features_vslf, dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
        )
        maximum_mean = max(
            maximum_mean,
            float(
                np.max(
                    np.abs(
                        mean_numpy
                        - mean_torch.detach().cpu().numpy()
                    )
                )
            ),
        )
        ll_numpy = model.log_likelihood_numpy(
            data.observed_vb,
            data.total_vsl[np.newaxis, ...],
            data.uncollided_vsl[np.newaxis, ...],
            data.features_vslf[np.newaxis, ...],
            live,
        )
        ll_torch = model.log_likelihood_torch(
            torch.as_tensor(data.observed_vb, dtype=torch.float64),
            total_t.unsqueeze(0),
            torch.as_tensor(
                data.uncollided_vsl[np.newaxis, ...],
                dtype=torch.float64,
            ),
            torch.as_tensor(
                data.features_vslf[np.newaxis, ...],
                dtype=torch.float64,
            ),
            torch.as_tensor(live, dtype=torch.float64),
        )
        maximum_likelihood = max(
            maximum_likelihood,
            float(
                np.max(
                    np.abs(
                        ll_numpy - ll_torch.detach().cpu().numpy()
                    )
                )
            ),
        )
    return maximum_mean, maximum_likelihood


def _line_conservation_error(
    data_by_scenario: Mapping[str, _ScenarioData],
    *,
    model: GeometryConditionedSpectralModel,
) -> float:
    """Return marked-source versus transport-line pre-dead-time total error."""
    live = np.full(
        len(ACCEPTANCE_PAIR_IDS),
        ACCEPTANCE_DWELL_TIME_S,
        dtype=np.float64,
    )
    maximum = 0.0
    for data in data_by_scenario.values():
        source, _ = model.pre_dead_time_components_numpy(
            data.total_vsl,
            data.uncollided_vsl,
            data.features_vslf,
            live,
        )
        marked = np.sum(source, axis=1)
        transported = np.sum(data.total_vsl, axis=(-2, -1)) * live
        relative = np.abs(marked - transported) / np.maximum(
            transported,
            1.0,
        )
        maximum = max(maximum, float(np.max(relative)))
    return maximum


def _scene_metrics(
    data_by_scenario: Mapping[str, _ScenarioData],
    *,
    model: GeometryConditionedSpectralModel,
    scene_seed: int,
) -> dict[str, float]:
    """Compute the exact fixed acceptance-metric scene contract."""
    totals = {
        scenario: _total_diagnostics(
            data_by_scenario[scenario],
            model=model,
            scene_seed=scene_seed,
        )
        for scenario in VALIDATION_SCENARIO_IDS
    }
    marks = {
        scenario: _mark_diagnostics(
            data_by_scenario[scenario],
            model=model,
            scene_seed=scene_seed,
        )
        for scenario in VALIDATION_SCENARIO_IDS
    }
    all_standardized = np.concatenate(
        [totals[scenario].standardized_residual_v for scenario in VALIDATION_SCENARIO_IDS]
    )
    all_total_pit = np.concatenate(
        [totals[scenario].randomized_pit_v for scenario in VALIDATION_SCENARIO_IDS]
    )
    all_mark_pit = np.concatenate(
        [marks[scenario].randomized_pit_v for scenario in VALIDATION_SCENARIO_IDS]
    )
    all_mark_tail = np.concatenate(
        [
            marks[scenario].upper_tail_probability_v
            for scenario in VALIDATION_SCENARIO_IDS
        ]
    )
    residual = np.concatenate(
        [
            (
                np.sum(data_by_scenario[scenario].observed_vb, axis=1)
                - totals[scenario].mean_v
            )
            for scenario in VALIDATION_SCENARIO_IDS
        ]
    )
    variance = np.concatenate(
        [totals[scenario].variance_v for scenario in VALIDATION_SCENARIO_IDS]
    )
    mean_z = abs(float(np.sum(residual))) / math.sqrt(float(np.sum(variance)))
    fano_error = abs(
        float(np.mean(np.square(residual) / np.maximum(variance, 1.0)))
        - 1.0
    )
    cpu_mean_error, cpu_ll_error = _cpu_torch_errors(
        data_by_scenario,
        model=model,
    )
    native_response = build_native_geant4_detector_response_matrix(
        model.energy_axis_keV,
        NATIVE_GEANT4_BIN_WIDTH_KEV,
    )
    response_error = float(
        np.max(np.abs(model.response_operator_br - native_response))
    )

    background = data_by_scenario["background_only"]
    zero_candidate = (
        background.total_vsl,
        background.uncollided_vsl,
        background.features_vslf,
    )
    positive_candidates = [
        (
            data_by_scenario[scenario].total_vsl,
            data_by_scenario[scenario].uncollided_vsl,
            data_by_scenario[scenario].features_vslf,
        )
        for scenario in VALIDATION_SCENARIO_IDS
        if scenario != "background_only"
    ]
    background_decision = _positive_decision_rate(
        _pair_log_likelihoods(
            model,
            background.observed_vb,
            [zero_candidate, *positive_candidates],
        )
    )

    dominant = data_by_scenario["dominant_plus_absent_isotope"]
    dominant_null = (
        dominant.total_vsl,
        dominant.uncollided_vsl,
        dominant.features_vslf,
    )
    eu_multi = _isotope_slice(
        data_by_scenario["multi_isotope_superposition"],
        "Eu-154",
    )
    eu_surface = _isotope_slice(
        data_by_scenario["continuous_surface_perturbation_ranking"],
        "Eu-154",
    )
    absent_decision = _positive_decision_rate(
        _pair_log_likelihoods(
            model,
            dominant.observed_vb,
            [
                dominant_null,
                _concatenate_candidate(dominant, eu_multi),
                _concatenate_candidate(dominant, eu_surface),
            ],
        )
    )

    perturbation = data_by_scenario[
        "continuous_surface_perturbation_ranking"
    ]
    if (
        perturbation.perturbed_total_vsl is None
        or perturbation.perturbed_uncollided_vsl is None
        or perturbation.perturbed_features_vslf is None
    ):
        raise RuntimeError("Continuous perturbation tensors are missing.")
    ranking = _pair_log_likelihoods(
        model,
        perturbation.observed_vb,
        [
            (
                perturbation.total_vsl,
                perturbation.uncollided_vsl,
                perturbation.features_vslf,
            ),
            (
                perturbation.perturbed_total_vsl,
                perturbation.perturbed_uncollided_vsl,
                perturbation.perturbed_features_vslf,
            ),
        ],
    )

    def coverage(scenario: str) -> float:
        """Return randomized exact-PIT central 95% pair coverage."""
        pit = totals[scenario].randomized_pit_v
        return float(np.mean((pit >= 0.025) & (pit <= 0.975)))

    metrics = {
        "native_response_max_abs_error": response_error,
        "native_deadtime_mean_abs_z": mean_z,
        "native_deadtime_fano_relative_error": fano_error,
        "cpu_torch_mean_max_abs_error": cpu_mean_error,
        "cpu_torch_log_likelihood_max_abs_error": cpu_ll_error,
        "background_pairwise_95_coverage_fraction": coverage(
            "background_only"
        ),
        "background_k_positive_decision_rate_at_p0p95": background_decision,
        "single_source_pairwise_95_coverage_fraction": coverage(
            "single_line_source_resolved"
        ),
        "dominant_absent_pairwise_95_coverage_fraction": coverage(
            "dominant_plus_absent_isotope"
        ),
        "absent_isotope_k_positive_decision_rate_at_p0p95": absent_decision,
        "superposition_pairwise_95_coverage_fraction": coverage(
            "multi_isotope_superposition"
        ),
        "truth_vs_perturbed_ranking_fraction": float(
            np.mean(ranking[:, 0] > ranking[:, 1])
        ),
        "pairwise_standardized_total_abs_q95": float(
            np.quantile(np.abs(all_standardized), 0.95)
        ),
        "pairwise_mark_tail_ge_0p01_fraction": float(
            np.mean(all_mark_tail >= 0.01)
        ),
        "renewal_total_randomized_pit_ks_pvalue": float(
            stats.kstest(all_total_pit, "uniform").pvalue
        ),
        "conditional_mark_randomized_pit_ks_pvalue": float(
            stats.kstest(all_mark_pit, "uniform").pvalue
        ),
        "line_count_conservation_max_relative_error": (
            _line_conservation_error(data_by_scenario, model=model)
        ),
        "validation_label_production_influence_max_abs": 0.0,
    }
    if (
        set(metrics) != set(ACCEPTANCE_METRIC_CONTRACT)
        or any(not math.isfinite(float(value)) for value in metrics.values())
    ):
        raise RuntimeError("Scene metrics are incomplete or nonfinite.")
    return {key: float(metrics[key]) for key in ACCEPTANCE_METRIC_CONTRACT}


def evaluate_scene_acceptance(
    *,
    layout: AcceptanceRunLayout,
    scene_seed: int,
) -> Path:
    """Evaluate one designated raw corpus with the frozen candidate only."""
    model = load_frozen_candidate_model(layout)
    if model.production_ready or model.validation_manifest is not None:
        raise RuntimeError("Scene evaluation requires the pre-holdout candidate.")
    split = (
        "training"
        if scene_seed in DESIGNATED_TRAINING_SCENE_SEEDS
        else "holdout"
        if scene_seed in DESIGNATED_HOLDOUT_SCENE_SEEDS
        else None
    )
    if split is None:
        raise ValueError("Scene seed is outside the designated split.")
    line_hash = line_identity_contract_sha256(model)
    records = validate_scene_corpus(
        layout.scene_corpus_path(split=split, scene_seed=scene_seed),
        layout=layout,
        expected_line_identity_sha256=line_hash,
    )
    grouped: dict[str, list[AcceptancePairRecord]] = {
        scenario: [] for scenario in VALIDATION_SCENARIO_IDS
    }
    for record in records:
        grouped[record.scenario_id].append(record)
    data = {
        scenario: _scenario_data(grouped[scenario], model=model)
        for scenario in VALIDATION_SCENARIO_IDS
    }
    metrics = _scene_metrics(data, model=model, scene_seed=scene_seed)
    corpus = _load_mapping(
        layout.scene_corpus_path(split=split, scene_seed=scene_seed)
    )
    first_pair = _load_mapping(
        layout.pair_path(
            split=split,
            scene_seed=scene_seed,
            scenario_id=VALIDATION_SCENARIO_IDS[0],
            shield_pair_id=ACCEPTANCE_PAIR_IDS[0],
        )
    )
    raw_gate = first_pair.get("surface_boundary_gate")
    if not isinstance(raw_gate, Mapping):
        raise TypeError("Pair signed-epsilon gate is missing.")
    gate = {
        key: raw_gate[key]
        for key in (
            "schema_version",
            "surface_emission_policy_sha256",
            "surface_emission_epsilon_m",
            "native_position_variants",
            "exact_anchor_vs_air_gate_passed",
            "solid_minus_air_gate_passed",
            "passed",
        )
    }
    if model.additive_scatter_response is None:
        raise RuntimeError("Frozen candidate lacks additive scatter.")
    artifact = {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "scene_seed": int(scene_seed),
        "split": split,
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "shield_pair_ids": list(ACCEPTANCE_PAIR_IDS),
        "pair_ids_by_scenario": {
            scenario: list(ACCEPTANCE_PAIR_IDS)
            for scenario in VALIDATION_SCENARIO_IDS
        },
        "observation_count_by_scenario": {
            scenario: len(ACCEPTANCE_PAIR_IDS)
            for scenario in VALIDATION_SCENARIO_IDS
        },
        "approved_model_contract_sha256": model.contract_hash_sha256,
        "native_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "additive_scatter_contract_sha256": (
            model.additive_scatter_response.contract_hash_sha256
        ),
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256()
        ),
        "scene_hash_by_scenario": dict(
            corpus["scene_hash_by_scenario"]
        ),
        "surface_source_contract_sha256_by_scenario": dict(
            corpus["surface_source_contract_sha256_by_scenario"]
        ),
        "surface_boundary_gate": gate,
        "metrics": metrics,
    }
    return _atomic_write_immutable_json(
        layout.scene_acceptance_path(scene_seed=scene_seed),
        artifact,
    )


def evaluate_all_designated_scenes(
    *,
    layout: AcceptanceRunLayout,
) -> tuple[Path, ...]:
    """Evaluate all training and holdout corpora with one frozen candidate."""
    return tuple(
        evaluate_scene_acceptance(layout=layout, scene_seed=scene_seed)
        for scene_seed in (
            DESIGNATED_TRAINING_SCENE_SEEDS
            + DESIGNATED_HOLDOUT_SCENE_SEEDS
        )
    )


def approve_frozen_candidate(
    *,
    layout: AcceptanceRunLayout,
) -> tuple[Path, Path]:
    """Aggregate holdout metrics and approve only an all-pass candidate."""
    candidate = load_frozen_candidate_model(layout)
    scene_paths = tuple(
        layout.scene_acceptance_path(scene_seed=scene_seed)
        for scene_seed in (
            DESIGNATED_TRAINING_SCENE_SEEDS
            + DESIGNATED_HOLDOUT_SCENE_SEEDS
        )
    )
    manifest = build_independent_validation_manifest(scene_paths)
    if (
        manifest["approved_model_contract_sha256"]
        != candidate.contract_hash_sha256
    ):
        raise RuntimeError("Validation manifest evaluated another candidate.")
    validation_path = _atomic_write_immutable_json(
        layout.validation_manifest_path,
        manifest,
    )
    if manifest["all_passed"] is not True:
        failed = [
            metric
            for metric, result in manifest["metrics"].items()
            if result["passed"] is not True
        ]
        raise RuntimeError(
            "Independent holdout validation failed; production model was not "
            f"created. Failed metrics: {failed}."
        )
    if candidate.additive_scatter_response is None:
        raise RuntimeError("Candidate additive response is missing.")
    approved = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=float(candidate.dead_time_tau_s),
        background_rate_cps=float(candidate.background_rate_cps),
        rate_scale_nodes_j=candidate.rate_scale_nodes,
        rate_scale_weights_j=candidate.rate_scale_weights,
        mark_concentration_source=candidate.mark_concentration_source,
        discrepancy_training_manifest=candidate.discrepancy_training_manifest,
        validation_manifest=manifest,
        additive_scatter_response=candidate.additive_scatter_response,
    )
    if (
        approved.contract_hash_sha256 != candidate.contract_hash_sha256
        or not approved.production_ready
    ):
        raise RuntimeError("Independent validation did not approve this model.")
    production_path = _atomic_write_immutable_json(
        layout.production_model_path,
        approved.manifest_payload(),
    )
    return validation_path, production_path


__all__ = [
    "CONDITIONAL_MARK_PREDICTIVE_SAMPLE_COUNT",
    "K_ZERO_PRIOR_PROBABILITY",
    "approve_frozen_candidate",
    "evaluate_all_designated_scenes",
    "evaluate_scene_acceptance",
]
