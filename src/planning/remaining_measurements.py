"""Estimate remaining station windows from PF ambiguity and DSS-PP gain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from pf.likelihood import expected_counts_per_source
from planning.dss_pp import DSSPPNode, extract_signature_modes


@dataclass(frozen=True)
class RemainingMeasurementConfig:
    """Configuration for online remaining-measurement estimation."""

    enabled: bool = True
    mode_cluster_radius_m: float = 1.5
    max_modes_per_isotope: int = 4
    max_particles: int | None = None
    planning_method: str | None = None
    target_position_spread_m: float = 1.0
    target_strength_cv: float = 0.5
    target_cardinality_confidence: float = 0.9
    pairwise_separation_threshold: float = 9.0
    residual_chi2_threshold: float = 9.0
    count_variance_floor: float = 1.0
    stop_budget: float = 0.0
    eta_default: float = 0.7
    eta_min: float = 0.3
    eta_max: float = 1.0
    gain_epsilon: float = 1.0e-6
    max_reported_stations: int = 99
    uncertainty_weight: float = 1.0
    cardinality_weight: float = 1.0
    separation_weight: float = 1.5
    verification_weight: float = 1.0
    residual_weight: float = 1.0
    dss_information_gain_weight: float = 1.0
    dss_count_utility_weight: float = 0.25
    range_scale: float = 1.35


@dataclass(frozen=True)
class RemainingMeasurementEstimate:
    """Summarize the predicted remaining station and spectrum budget."""

    current_station_count: int
    estimated_remaining_stations: int
    estimated_remaining_station_low: int
    estimated_remaining_station_high: int
    estimated_remaining_spectra_low: int
    estimated_remaining_spectra_high: int
    program_length: int
    current_budget: float
    stop_budget: float
    predicted_gain: float
    empirical_eta: float
    bottleneck: str
    unresolved_factors: tuple[str, ...]
    components: dict[str, float] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)
    isotope_details: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable estimate payload."""
        payload = _json_safe(asdict(self))
        payload["unresolved_factors"] = list(self.unresolved_factors)
        return payload


@dataclass(frozen=True)
class _PairwiseSignatureStats:
    """Store pairwise response-separation statistics."""

    deficit: float
    min_separation: float
    unresolved_pairs: int
    weighted_increment: float


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe copy with nonfinite floats converted to null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalized nonnegative weights with a uniform fallback."""
    arr = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if arr.size == 0:
        return arr
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(arr.size, 1.0 / float(arr.size), dtype=float)
    return arr / total


def _pair_indices(pair_id: int, num_orients: int) -> tuple[int, int]:
    """Return Fe and Pb orientation indices from a flattened pair id."""
    n = max(1, int(num_orients))
    return int(pair_id) // n, int(pair_id) % n


def _pairwise_signature_stats_batched(
    response_by_measurement_mode: NDArray[np.float64],
    variance_by_measurement: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    *,
    threshold: float,
) -> _PairwiseSignatureStats:
    """Return batched same-isotope pairwise signature deficits."""
    response = np.asarray(response_by_measurement_mode, dtype=float)
    if response.ndim != 2 or response.shape[1] <= 1:
        return _PairwiseSignatureStats(0.0, float("inf"), 0, 0.0)
    variance = np.maximum(
        np.asarray(variance_by_measurement, dtype=float).reshape(-1),
        1.0e-12,
    )
    if variance.size == 1 and response.shape[0] != 1:
        variance = np.full(response.shape[0], float(variance[0]), dtype=float)
    if variance.size != response.shape[0]:
        raise ValueError("variance_by_measurement must match response rows.")
    weights = _normalise_weights(np.asarray(mode_weights, dtype=float))
    if weights.size != response.shape[1]:
        weights = np.ones(response.shape[1], dtype=float) / float(response.shape[1])
    diff = response[:, :, None] - response[:, None, :]
    d_matrix = np.sum((diff * diff) / variance[:, None, None], axis=0)
    upper = np.triu_indices(response.shape[1], k=1)
    distances = np.asarray(d_matrix[upper], dtype=float)
    if distances.size == 0:
        return _PairwiseSignatureStats(0.0, float("inf"), 0, 0.0)
    pair_weights = weights[upper[0]] * weights[upper[1]]
    pair_weight_sum = float(np.sum(pair_weights))
    if pair_weight_sum <= 0.0:
        pair_weights = np.full(distances.size, 1.0 / float(distances.size))
    else:
        pair_weights = pair_weights / pair_weight_sum
    deficit = np.maximum(float(threshold) - distances, 0.0)
    unresolved = distances < float(threshold)
    return _PairwiseSignatureStats(
        deficit=float(np.sum(pair_weights * deficit)),
        min_separation=float(np.min(distances)),
        unresolved_pairs=int(np.count_nonzero(unresolved)),
        weighted_increment=float(np.sum(pair_weights * distances)),
    )


def _weighted_cardinality_stats(
    source_counts: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> tuple[float, float, int, float]:
    """Return entropy, MAP confidence, MAP count, and weighted variance."""
    counts = np.asarray(source_counts, dtype=int).reshape(-1)
    norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
    if counts.size == 0 or norm_weights.size != counts.size:
        return 0.0, 1.0, 0, 0.0
    unique, inverse = np.unique(counts, return_inverse=True)
    probs = np.zeros(unique.size, dtype=float)
    np.add.at(probs, inverse, norm_weights)
    probs = _normalise_weights(probs)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12))))
    best_idx = int(np.argmax(probs))
    mean = float(np.sum(norm_weights * counts))
    variance = float(np.sum(norm_weights * (counts - mean) ** 2))
    return entropy, float(probs[best_idx]), int(unique[best_idx]), variance


def _weighted_strength_cv(
    strengths_by_particle: list[float],
    weights: NDArray[np.float64],
) -> float:
    """Return weighted coefficient of variation for total isotope strength."""
    values = np.asarray(strengths_by_particle, dtype=float).reshape(-1)
    norm_weights = _normalise_weights(np.asarray(weights, dtype=float))
    if values.size == 0 or norm_weights.size != values.size:
        return 0.0
    mean = float(np.sum(norm_weights * values))
    if mean <= 1.0e-12:
        return 0.0
    variance = float(np.sum(norm_weights * (values - mean) ** 2))
    return float(np.sqrt(max(variance, 0.0)) / mean)


def _mode_response_matrix(
    estimator: RotatingShieldPFEstimator,
    isotope: str,
    detector_positions: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
    live_times: NDArray[np.float64],
    mode_positions: NDArray[np.float64],
    mode_strengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return expected counts for source modes over measurement rows."""
    if mode_positions.size == 0:
        return np.zeros((np.asarray(detector_positions).shape[0], 0), dtype=float)
    filt = estimator.filters[isotope]
    return np.maximum(
        expected_counts_per_source(
            kernel=filt.continuous_kernel,
            isotope=isotope,
            detector_positions=np.asarray(detector_positions, dtype=float),
            sources=np.asarray(mode_positions, dtype=float),
            strengths=np.asarray(mode_strengths, dtype=float),
            live_times=np.asarray(live_times, dtype=float),
            fe_indices=np.asarray(fe_indices, dtype=int),
            pb_indices=np.asarray(pb_indices, dtype=int),
            source_scale=estimator.response_scale_for_isotope(isotope),
        ),
        0.0,
    )


def _state_budget_components(
    estimator: RotatingShieldPFEstimator,
    config: RemainingMeasurementConfig,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
]:
    """Return current ambiguity components and cached mode arrays."""
    modes_by_iso = extract_signature_modes(
        estimator,
        max_particles=config.max_particles,
        method=config.planning_method,
        mode_cluster_radius_m=float(config.mode_cluster_radius_m),
        max_modes_per_isotope=int(config.max_modes_per_isotope),
        tentative_weight_multiplier=1.5,
    )
    target_spread = max(float(config.target_position_spread_m), 1.0e-12)
    target_cv = max(float(config.target_strength_cv), 1.0e-12)
    target_cardinality = float(np.clip(config.target_cardinality_confidence, 0.0, 1.0))
    threshold = max(float(config.pairwise_separation_threshold), 0.0)
    residual_threshold = max(float(config.residual_chi2_threshold), 1.0e-12)
    variance_floor = max(float(config.count_variance_floor), 1.0e-12)
    components = {
        "uncertainty": 0.0,
        "cardinality": 0.0,
        "same_isotope_separation": 0.0,
        "pseudo_source_verification": 0.0,
        "residual": 0.0,
    }
    isotope_details: dict[str, dict[str, float | int]] = {}
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ] = {}
    for isotope, filt in estimator.filters.items():
        particles = filt.continuous_particles
        if not particles:
            isotope_details[isotope] = {}
            continue
        weights = _normalise_weights(np.asarray(filt.continuous_weights, dtype=float))
        source_counts = np.asarray(
            [int(particle.state.num_sources) for particle in particles],
            dtype=int,
        )
        entropy, confidence, map_count, cardinality_var = _weighted_cardinality_stats(
            source_counts,
            weights,
        )
        strength_totals = [
            float(
                np.sum(
                    np.maximum(
                        particle.state.strengths[: particle.state.num_sources],
                        0.0,
                    )
                )
            )
            for particle in particles
        ]
        strength_cv = _weighted_strength_cv(strength_totals, weights)
        modes = modes_by_iso.get(isotope, [])
        if modes:
            spread_budget = float(
                np.sum(
                    [
                        max(float(mode.spread_m) / target_spread - 1.0, 0.0)
                        * max(float(mode.weight), 0.0)
                        for mode in modes
                    ]
                )
            )
            mode_positions = np.vstack([mode.position_xyz for mode in modes])
            mode_strengths = np.asarray(
                [max(float(mode.strength_cps_1m), 0.0) for mode in modes],
                dtype=float,
            )
            mode_weights = _normalise_weights(
                np.asarray([mode.weight for mode in modes], dtype=float)
            )
            mode_arrays[isotope] = [(mode_positions, mode_strengths, mode_weights)]
        else:
            spread_budget = 0.0
            mode_arrays[isotope] = []
        strength_budget = max(strength_cv / target_cv - 1.0, 0.0)
        cardinality_budget = entropy + max(target_cardinality - confidence, 0.0)
        tentative_expected = _weighted_tentative_source_count(particles, weights)
        verification_views = int(getattr(filt, "last_birth_residual_distinct_poses", 0))
        required_views = max(1, int(filt.config.pseudo_source_min_distinct_views))
        verification_budget = tentative_expected * max(
            required_views - min(verification_views, required_views),
            0,
        ) / float(required_views)
        verification_budget += float(
            max(int(getattr(filt, "last_pseudo_source_quarantine_active", 0)), 0)
        )
        separation_budget = 0.0
        min_separation = float("inf")
        unresolved_pairs = 0
        residual_budget = 0.0
        data = estimator._measurement_data_for_iso(isotope, window=None)
        if data is not None and data.z_k.size and mode_arrays[isotope]:
            mode_positions, mode_strengths, mode_weights = mode_arrays[isotope][0]
            response = _mode_response_matrix(
                estimator,
                isotope,
                data.detector_positions,
                data.fe_indices,
                data.pb_indices,
                data.live_times,
                mode_positions,
                mode_strengths,
            )
            variance = np.maximum(data.observation_variances, variance_floor)
            stats = _pairwise_signature_stats_batched(
                response,
                variance,
                mode_weights,
                threshold=threshold,
            )
            separation_budget = stats.deficit
            min_separation = stats.min_separation
            unresolved_pairs = stats.unresolved_pairs
            background_rate = (
                float(filt.best_particle().state.background)
                if filt.continuous_particles
                else 0.0
            )
            prediction = background_rate * data.live_times + np.sum(response, axis=1)
            residual = np.maximum(np.asarray(data.z_k, dtype=float) - prediction, 0.0)
            residual_chi2 = float(np.sum((residual * residual) / variance))
            residual_budget = max(residual_chi2 / residual_threshold - 1.0, 0.0)
        else:
            residual_chi2 = 0.0
        components["uncertainty"] += spread_budget + strength_budget
        components["cardinality"] += cardinality_budget
        components["same_isotope_separation"] += separation_budget
        components["pseudo_source_verification"] += verification_budget
        components["residual"] += residual_budget
        isotope_details[isotope] = {
            "mode_count": int(len(modes)),
            "map_source_count": int(map_count),
            "cardinality_confidence": float(confidence),
            "cardinality_entropy": float(entropy),
            "cardinality_variance": float(cardinality_var),
            "strength_cv": float(strength_cv),
            "tentative_source_expectation": float(tentative_expected),
            "verification_views": int(verification_views),
            "required_verification_views": int(required_views),
            "min_pairwise_separation": (
                0.0 if not np.isfinite(min_separation) else float(min_separation)
            ),
            "unresolved_pair_count": int(unresolved_pairs),
            "residual_chi2": float(residual_chi2),
        }
    return components, isotope_details, mode_arrays


def _weighted_tentative_source_count(
    particles: Sequence[object],
    weights: NDArray[np.float64],
) -> float:
    """Return the weighted number of tentative or failed source slots."""
    total = 0.0
    for particle, weight in zip(particles, weights):
        state = particle.state
        count = max(0, int(state.num_sources))
        if count <= 0:
            continue
        tentative_raw = getattr(state, "tentative_sources", None)
        tentative = (
            np.zeros(count, dtype=bool)
            if tentative_raw is None
            else np.asarray(tentative_raw, dtype=bool)[:count]
        )
        failed_raw = getattr(state, "verification_fail_streaks", None)
        failed = (
            np.zeros(count, dtype=int)
            if failed_raw is None
            else np.asarray(failed_raw, dtype=int)[:count]
        )
        if tentative.size != count:
            padded = np.zeros(count, dtype=bool)
            padded[: min(tentative.size, count)] = tentative[:count]
            tentative = padded
        if failed.size != count:
            padded = np.zeros(count, dtype=int)
            padded[: min(failed.size, count)] = failed[:count]
            failed = padded
        total += float(weight) * float(np.count_nonzero(tentative | (failed > 0)))
    return float(total)


def _prediction_gain_components(
    estimator: RotatingShieldPFEstimator,
    next_pose_xyz: NDArray[np.float64] | None,
    shield_program_pair_ids: Sequence[int] | None,
    live_time_s: float,
    mode_arrays: dict[
        str,
        list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    ],
    config: RemainingMeasurementConfig,
    dss_node: DSSPPNode | None,
    dss_diagnostics: dict[str, float | int | str] | None,
) -> tuple[dict[str, float], int]:
    """Return predicted one-station ambiguity reduction components."""
    program = tuple(int(pair_id) for pair_id in (shield_program_pair_ids or ()))
    if not program:
        program = (0,)
    program_length = max(1, len(program))
    gains = {
        "uncertainty": 0.0,
        "same_isotope_separation": 0.0,
        "pseudo_source_verification": 0.0,
        "residual": 0.0,
        "dss_information": 0.0,
    }
    if next_pose_xyz is not None:
        pose = np.asarray(next_pose_xyz, dtype=float).reshape(3)
        fe = []
        pb = []
        for pair_id in program:
            fe_idx, pb_idx = _pair_indices(pair_id, estimator.num_orientations)
            fe.append(fe_idx)
            pb.append(pb_idx)
        detector_positions = np.repeat(pose[None, :], program_length, axis=0)
        live_times = np.full(program_length, max(float(live_time_s), 0.0), dtype=float)
        fe_indices = np.asarray(fe, dtype=int)
        pb_indices = np.asarray(pb, dtype=int)
        threshold = max(float(config.pairwise_separation_threshold), 0.0)
        floor = max(float(config.count_variance_floor), 1.0e-12)
        for isotope, arrays in mode_arrays.items():
            if not arrays:
                continue
            mode_positions, mode_strengths, mode_weights = arrays[0]
            response = _mode_response_matrix(
                estimator,
                isotope,
                detector_positions,
                fe_indices,
                pb_indices,
                live_times,
                mode_positions,
                mode_strengths,
            )
            row_variance = np.maximum(np.mean(response, axis=1), floor)
            stats = _pairwise_signature_stats_batched(
                response,
                row_variance,
                mode_weights,
                threshold=threshold,
            )
            gains["same_isotope_separation"] += stats.weighted_increment
    if dss_node is not None:
        gains["uncertainty"] += max(float(dss_node.information_gain), 0.0)
        gains["dss_information"] += max(float(dss_node.information_gain), 0.0)
        gains["same_isotope_separation"] += max(
            float(dss_node.signature_score)
            + float(dss_node.temporal_separation_score)
            + float(dss_node.elevation_signature_score),
            0.0,
        )
        gains["residual"] += max(float(dss_node.count_utility), 0.0)
    if dss_diagnostics:
        for key in ("best_information_gain", "information_gain", "eig"):
            if key in dss_diagnostics:
                gains["dss_information"] += max(float(dss_diagnostics[key]), 0.0)
                break
    gains["pseudo_source_verification"] += float(program_length)
    return gains, program_length


def _empirical_eta(
    estimator: RotatingShieldPFEstimator,
    current_budget: float,
    predicted_gain: float,
    config: RemainingMeasurementConfig,
    *,
    update_history: bool = True,
) -> float:
    """Update and return the empirical predicted-vs-realized gain correction."""
    history = getattr(estimator, "_remaining_measurement_budget_history", [])
    if not isinstance(history, list):
        history = []
    ratios = getattr(estimator, "_remaining_measurement_eta_ratios", [])
    if not isinstance(ratios, list):
        ratios = []
    if update_history and history:
        previous = history[-1]
        prev_budget = float(previous.get("budget", current_budget))
        prev_gain = max(float(previous.get("predicted_gain", 0.0)), 1.0e-12)
        realized = max(prev_budget - float(current_budget), 0.0)
        ratios.append(realized / prev_gain)
        ratios = ratios[-8:]
    if update_history:
        history.append(
            {
                "budget": float(current_budget),
                "predicted_gain": float(predicted_gain),
            }
        )
        setattr(estimator, "_remaining_measurement_budget_history", history[-8:])
        setattr(estimator, "_remaining_measurement_eta_ratios", ratios)
    if ratios:
        eta = float(np.median(np.asarray(ratios, dtype=float)))
    else:
        eta = float(config.eta_default)
    return float(np.clip(eta, float(config.eta_min), float(config.eta_max)))


def estimate_remaining_measurement_budget(
    estimator: RotatingShieldPFEstimator,
    *,
    next_pose_xyz: NDArray[np.float64] | None = None,
    shield_program_pair_ids: Sequence[int] | None = None,
    live_time_s: float = 1.0,
    dss_node: DSSPPNode | None = None,
    dss_diagnostics: dict[str, float | int | str] | None = None,
    config: RemainingMeasurementConfig | None = None,
    current_station_count: int | None = None,
    update_history: bool = True,
) -> RemainingMeasurementEstimate:
    """Estimate remaining station windows from current PF ambiguity."""
    cfg = config or RemainingMeasurementConfig()
    components, isotope_details, mode_arrays = _state_budget_components(
        estimator,
        cfg,
    )
    gains, program_length = _prediction_gain_components(
        estimator,
        next_pose_xyz,
        shield_program_pair_ids,
        live_time_s,
        mode_arrays,
        cfg,
        dss_node,
        dss_diagnostics,
    )
    weighted_budget = (
        float(cfg.uncertainty_weight) * components["uncertainty"]
        + float(cfg.cardinality_weight) * components["cardinality"]
        + float(cfg.separation_weight) * components["same_isotope_separation"]
        + float(cfg.verification_weight) * components["pseudo_source_verification"]
        + float(cfg.residual_weight) * components["residual"]
    )
    weighted_gain = (
        float(cfg.uncertainty_weight) * gains["uncertainty"]
        + float(cfg.separation_weight) * gains["same_isotope_separation"]
        + float(cfg.verification_weight) * gains["pseudo_source_verification"]
        + float(cfg.residual_weight) * gains["residual"]
        + float(cfg.dss_information_gain_weight) * gains["dss_information"]
        + float(cfg.dss_count_utility_weight) * gains["residual"]
    )
    eta = _empirical_eta(
        estimator,
        weighted_budget,
        weighted_gain,
        cfg,
        update_history=update_history,
    )
    remaining_budget = max(weighted_budget - float(cfg.stop_budget), 0.0)
    denom = max(eta * weighted_gain, float(cfg.gain_epsilon))
    estimate = int(np.ceil(remaining_budget / denom)) if remaining_budget > 0.0 else 0
    estimate = min(max(estimate, 0), max(0, int(cfg.max_reported_stations)))
    if estimate > 0:
        low = max(1, int(np.floor(float(estimate) / max(float(cfg.range_scale), 1.0))))
    else:
        low = 0
    high = min(
        max(0, int(cfg.max_reported_stations)),
        max(
            estimate,
            int(np.ceil(float(estimate) * max(float(cfg.range_scale), 1.0))),
        ),
    )
    unresolved = tuple(
        key for key, value in sorted(components.items()) if float(value) > 1.0e-9
    )
    bottleneck = (
        "none"
        if not unresolved
        else max(components, key=lambda key: float(components[key]))
    )
    station_count = (
        int(current_station_count)
        if current_station_count is not None
        else int(len({record.pose_idx for record in estimator.measurements}))
    )
    return RemainingMeasurementEstimate(
        current_station_count=station_count,
        estimated_remaining_stations=estimate,
        estimated_remaining_station_low=low,
        estimated_remaining_station_high=high,
        estimated_remaining_spectra_low=low * program_length,
        estimated_remaining_spectra_high=high * program_length,
        program_length=program_length,
        current_budget=float(weighted_budget),
        stop_budget=float(cfg.stop_budget),
        predicted_gain=float(weighted_gain),
        empirical_eta=float(eta),
        bottleneck=str(bottleneck),
        unresolved_factors=unresolved,
        components={key: float(value) for key, value in components.items()},
        gains={key: float(value) for key, value in gains.items()},
        isotope_details=isotope_details,
    )


def format_remaining_measurement_estimate(
    estimate: RemainingMeasurementEstimate,
) -> str:
    """Return a compact log line for a remaining-measurement estimate."""
    factors = ",".join(estimate.unresolved_factors) or "none"
    return (
        "Remaining measurement estimate: "
        f"stations={estimate.estimated_remaining_station_low}-"
        f"{estimate.estimated_remaining_station_high} "
        f"spectra={estimate.estimated_remaining_spectra_low}-"
        f"{estimate.estimated_remaining_spectra_high} "
        f"bottleneck={estimate.bottleneck} "
        f"budget={estimate.current_budget:.3g} "
        f"gain={estimate.predicted_gain:.3g} "
        f"eta={estimate.empirical_eta:.2f} "
        f"unresolved={factors}"
    )
