"""Replay diagnostic line/photopeak-channel counts from saved Geant4 spectra."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectrum.pipeline import PhotopeakChannelEstimate, SpectralDecomposer
from spectrum.runtime_config import spectrum_config_from_runtime_config

ISOTOPES = ("Cs-137", "Co-60", "Eu-154")
RESPONSE_VS_TRUTH_GATE = (0.04, 0.12)


def _float_value(value: object, default: float = float("nan")) -> float:
    """Return a float parsed from a JSON/CSV-like value."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_error(count: float, truth: float) -> float:
    """Return absolute relative error for a positive truth count."""
    if truth <= 0.0:
        return float("nan")
    return abs(float(count) - float(truth)) / float(truth)


def _signed_relative_error(count: float, truth: float) -> float:
    """Return signed relative error for a positive truth count."""
    if truth <= 0.0:
        return float("nan")
    return (float(count) - float(truth)) / float(truth)


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Return standard summary statistics for finite values."""
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _signed_summary(values: Sequence[float]) -> dict[str, float]:
    """Return mean and quantiles for signed finite residuals."""
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _quality_gate(
    summary: dict[str, float],
    *,
    mean_threshold: float,
    p95_threshold: float,
) -> dict[str, object]:
    """Return a mean/p95 gate while treating max as a tail report."""
    mean = float(summary.get("mean", float("inf")))
    p95 = float(summary.get("p95", float("inf")))
    return {
        "mean_threshold": float(mean_threshold),
        "p95_threshold": float(p95_threshold),
        "mean": mean,
        "p95": p95,
        "max_tail_report_only": float(summary.get("max", float("nan"))),
        "passed": bool(mean <= mean_threshold and p95 <= p95_threshold),
    }


def _energy_token_to_float(token: str) -> float:
    """Convert a Geant4 metadata energy token such as 1274p5 to keV."""
    return float(str(token).replace("p", "."))


def _metadata_line_counts(
    metadata: dict[str, Any],
    *,
    prefix: str,
    isotopes: Sequence[str],
) -> dict[tuple[str, float], float]:
    """Return line counts keyed by isotope and line energy."""
    allowed = {str(isotope) for isotope in isotopes}
    pattern = re.compile(
        rf"^{re.escape(prefix)}_src\d+_(?P<isotope>[^_]+)_e"
        r"(?P<energy>\d+(?:p\d+)?)$"
    )
    totals: dict[tuple[str, float], float] = defaultdict(float)
    for key, value in metadata.items():
        match = pattern.match(str(key))
        if match is None:
            continue
        isotope = str(match.group("isotope"))
        if isotope not in allowed:
            continue
        energy = _energy_token_to_float(str(match.group("energy")))
        totals[(isotope, energy)] += max(_float_value(value, 0.0), 0.0)
    return dict(totals)


def _nearest_line_truth(
    line_truth: dict[tuple[str, float], float],
    *,
    isotope: str,
    energy_keV: float,
    tolerance_keV: float = 2.0,
) -> tuple[float, float]:
    """Return nearest truth energy and count for one estimated channel."""
    candidates = [
        (abs(float(truth_energy) - float(energy_keV)), float(truth_energy), count)
        for (truth_isotope, truth_energy), count in line_truth.items()
        if str(truth_isotope) == str(isotope)
    ]
    if not candidates:
        return float("nan"), float("nan")
    distance, truth_energy, count = min(candidates, key=lambda item: item[0])
    if distance > float(tolerance_keV):
        return float("nan"), float("nan")
    return truth_energy, float(count)


def _combine_channel_source_counts(
    channels: Sequence[PhotopeakChannelEstimate],
) -> tuple[float, float]:
    """Combine line-channel source-equivalent counts by SNR-aware inverse variance."""
    finite = [
        channel
        for channel in channels
        if np.isfinite(channel.source_equivalent_variance)
        and channel.source_equivalent_variance > 0.0
    ]
    if not finite:
        return 0.0, 1.0
    counts = np.asarray(
        [channel.source_equivalent_counts for channel in finite],
        dtype=float,
    )
    variances = np.asarray(
        [channel.source_equivalent_variance for channel in finite],
        dtype=float,
    )
    snr = np.asarray(
        [max(float(channel.signal_to_noise), 0.0) for channel in finite],
        dtype=float,
    )
    snr_weight = np.clip((snr - 1.0) / 3.0, 0.0, 1.0) ** 2
    weights = snr_weight / np.maximum(variances, 1e-12)
    if float(np.sum(weights)) <= 0.0:
        weights = 1.0 / np.maximum(variances, 1e-12)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        value = max(float(np.mean(counts)), 0.0)
        return value, max(value, 1.0)
    value = max(float(np.sum(weights * counts) / weight_sum), 0.0)
    variance = max(float(1.0 / weight_sum), value, 1.0)
    return value, variance


def _scaled_line_sum_source_counts(
    channels: Sequence[PhotopeakChannelEstimate],
) -> float:
    """Return line-split sum scaled back to the included line-weight mass."""
    weight_sum = float(np.sum([max(channel.line_weight, 0.0) for channel in channels]))
    if weight_sum <= 0.0:
        return 0.0
    line_sum = float(np.sum([channel.line_equivalent_counts for channel in channels]))
    return max(line_sum / weight_sum, 0.0)


def _load_results_by_case(results_path: Path) -> dict[str, dict[str, Any]]:
    """Load validation result entries keyed by case name."""
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("results.json must contain a list of validation entries")
    results: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        case = item.get("case", {})
        if not isinstance(case, dict):
            continue
        name = str(case.get("name", ""))
        if name:
            results[name] = item
    return results


def _case_live_time(result: dict[str, Any]) -> float:
    """Return dwell time for one saved validation case."""
    case = result.get("case", {})
    if isinstance(case, dict):
        value = _float_value(case.get("dwell_time_s"))
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return 1.0


def _case_in_accuracy_summary(result: dict[str, Any]) -> bool:
    """Return whether a saved validation case should contribute to metrics."""
    case = result.get("case", {})
    if not isinstance(case, dict):
        return True
    return bool(case.get("include_in_accuracy_summary", True))


def _isotope_truth_count(metadata: dict[str, Any], isotope: str) -> float:
    """Return Geant4 isotope-labeled detector truth count for one isotope."""
    for key in (
        f"transport_detected_counts_{isotope}",
        f"transport_uncollided_primary_counts_{isotope}",
    ):
        value = _float_value(metadata.get(key))
        if np.isfinite(value):
            return max(float(value), 0.0)
    return float("nan")


def _append_grouped(
    grouped: dict[str, list[float]],
    key: str,
    value: float,
) -> None:
    """Append a metric value to one grouped metric bucket."""
    if np.isfinite(value):
        grouped[str(key)].append(float(value))


def _summarize_groups(grouped: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Return summaries for grouped metric buckets."""
    return {key: _summary(values) for key, values in sorted(grouped.items())}


def replay_photopeak_channels(
    *,
    config_path: Path,
    results_path: Path,
    spectra_path: Path,
    isotopes: Sequence[str],
    truth_min: float,
    line_truth_min: float,
    top: int,
    mean_threshold: float,
    p95_threshold: float,
) -> dict[str, Any]:
    """Replay current response counts and diagnostic photopeak channels."""
    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    decomposer = SpectralDecomposer(spectrum_config_from_runtime_config(runtime_config))
    requested = [str(isotope) for isotope in isotopes]
    results_by_case = _load_results_by_case(results_path)

    channel_line_errors: list[float] = []
    channel_line_signed_errors: list[float] = []
    channel_aggregate_errors: list[float] = []
    channel_scaled_sum_errors: list[float] = []
    response_errors: list[float] = []
    response_vs_channel_errors: list[float] = []
    signed_channel_aggregate_errors: list[float] = []
    signed_response_errors: list[float] = []
    grouped_channel_line_by_isotope: dict[str, list[float]] = defaultdict(list)
    grouped_channel_line_by_label: dict[str, list[float]] = defaultdict(list)
    grouped_channel_aggregate_by_isotope: dict[str, list[float]] = defaultdict(list)
    grouped_response_by_isotope: dict[str, list[float]] = defaultdict(list)
    worst_channel_rows: list[dict[str, Any]] = []
    worst_aggregate_rows: list[dict[str, Any]] = []
    processed_cases = 0

    with np.load(spectra_path) as spectra:
        for case_name in sorted(spectra.files):
            result = results_by_case.get(str(case_name))
            if result is None or not _case_in_accuracy_summary(result):
                continue
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            spectrum = np.asarray(spectra[case_name], dtype=float)
            live_time_s = _case_live_time(result)
            response_estimates = decomposer.compute_response_poisson_estimates(
                spectrum,
                isotopes=requested,
                live_time_s=live_time_s,
            )
            channels = decomposer.compute_photopeak_channel_estimates(
                spectrum,
                isotopes=requested,
                live_time_s=live_time_s,
            )
            processed_cases += 1
            line_truth = _metadata_line_counts(
                metadata,
                prefix="transport_detected_counts",
                isotopes=requested,
            )
            isotope_truths = {
                isotope: _isotope_truth_count(metadata, isotope)
                for isotope in requested
            }
            channels_by_isotope: dict[str, list[PhotopeakChannelEstimate]] = {
                isotope: [] for isotope in requested
            }
            for channel in channels:
                channels_by_isotope.setdefault(channel.isotope, []).append(channel)
                isotope_truth = isotope_truths.get(channel.isotope, float("nan"))
                if not np.isfinite(isotope_truth) or isotope_truth < truth_min:
                    continue
                truth_energy, truth = _nearest_line_truth(
                    line_truth,
                    isotope=channel.isotope,
                    energy_keV=channel.energy_keV,
                )
                if not np.isfinite(truth) or truth < line_truth_min:
                    continue
                error = _relative_error(channel.line_equivalent_counts, truth)
                signed = _signed_relative_error(channel.line_equivalent_counts, truth)
                channel_line_errors.append(error)
                channel_line_signed_errors.append(signed)
                _append_grouped(
                    grouped_channel_line_by_isotope,
                    channel.isotope,
                    error,
                )
                _append_grouped(grouped_channel_line_by_label, channel.label, error)
                worst_channel_rows.append(
                    {
                        "case": str(case_name),
                        "isotope": channel.isotope,
                        "energy_keV": float(channel.energy_keV),
                        "truth_energy_keV": float(truth_energy),
                        "line_truth": float(truth),
                        "line_equivalent_count": float(
                            channel.line_equivalent_counts
                        ),
                        "source_equivalent_count": float(
                            channel.source_equivalent_counts
                        ),
                        "line_weight": float(channel.line_weight),
                        "relative_error": float(error),
                        "signed_relative_error": float(signed),
                        "snr": float(channel.signal_to_noise),
                        "reduced_chi2": float(channel.reduced_chi2),
                    }
                )
            for isotope in requested:
                truth = isotope_truths.get(isotope, float("nan"))
                if not np.isfinite(truth) or truth < truth_min:
                    continue
                response_count = float(
                    response_estimates.get(isotope).counts
                    if isotope in response_estimates
                    else 0.0
                )
                channel_count, channel_variance = _combine_channel_source_counts(
                    channels_by_isotope.get(isotope, []),
                )
                scaled_sum_count = _scaled_line_sum_source_counts(
                    channels_by_isotope.get(isotope, []),
                )
                response_error = _relative_error(response_count, truth)
                channel_error = _relative_error(channel_count, truth)
                scaled_sum_error = _relative_error(scaled_sum_count, truth)
                response_channel_error = _relative_error(channel_count, response_count)
                response_errors.append(response_error)
                channel_aggregate_errors.append(channel_error)
                channel_scaled_sum_errors.append(scaled_sum_error)
                response_vs_channel_errors.append(response_channel_error)
                signed_response_errors.append(
                    _signed_relative_error(response_count, truth)
                )
                signed_channel_aggregate_errors.append(
                    _signed_relative_error(channel_count, truth)
                )
                _append_grouped(
                    grouped_response_by_isotope,
                    isotope,
                    response_error,
                )
                _append_grouped(
                    grouped_channel_aggregate_by_isotope,
                    isotope,
                    channel_error,
                )
                worst_aggregate_rows.append(
                    {
                        "case": str(case_name),
                        "isotope": isotope,
                        "truth": float(truth),
                        "response_poisson_count": float(response_count),
                        "photopeak_channel_count": float(channel_count),
                        "photopeak_channel_variance": float(channel_variance),
                        "scaled_line_sum_count": float(scaled_sum_count),
                        "response_relative_error": float(response_error),
                        "photopeak_channel_relative_error": float(channel_error),
                        "scaled_line_sum_relative_error": float(scaled_sum_error),
                        "channel_count": int(
                            len(channels_by_isotope.get(isotope, []))
                        ),
                    }
                )

    worst_channel_rows.sort(
        key=lambda row: float(row.get("relative_error", 0.0)),
        reverse=True,
    )
    worst_aggregate_rows.sort(
        key=lambda row: float(row.get("photopeak_channel_relative_error", 0.0)),
        reverse=True,
    )
    channel_summary = _summary(channel_aggregate_errors)
    response_summary = _summary(response_errors)
    recommendation = _recommendation(
        response_summary=response_summary,
        channel_summary=channel_summary,
        gate=_quality_gate(
            channel_summary,
            mean_threshold=mean_threshold,
            p95_threshold=p95_threshold,
        ),
    )
    return {
        "inputs": {
            "config_path": str(config_path),
            "results_path": str(results_path),
            "spectra_path": str(spectra_path),
            "isotopes": list(requested),
            "truth_min": float(truth_min),
            "line_truth_min": float(line_truth_min),
            "processed_cases": int(processed_cases),
        },
        "response_poisson_vs_truth": response_summary,
        "photopeak_channel_source_aggregate_vs_truth": channel_summary,
        "photopeak_channel_scaled_line_sum_vs_truth": _summary(
            channel_scaled_sum_errors
        ),
        "photopeak_channel_line_equivalent_vs_line_truth": _summary(
            channel_line_errors
        ),
        "response_vs_photopeak_channel_aggregate": _summary(
            response_vs_channel_errors
        ),
        "signed_response_poisson_vs_truth": _signed_summary(signed_response_errors),
        "signed_photopeak_channel_source_aggregate_vs_truth": _signed_summary(
            signed_channel_aggregate_errors
        ),
        "signed_photopeak_channel_line_equivalent_vs_line_truth": _signed_summary(
            channel_line_signed_errors
        ),
        "quality_gates": {
            "response_poisson_vs_truth": _quality_gate(
                response_summary,
                mean_threshold=mean_threshold,
                p95_threshold=p95_threshold,
            ),
            "photopeak_channel_source_aggregate_vs_truth": _quality_gate(
                channel_summary,
                mean_threshold=mean_threshold,
                p95_threshold=p95_threshold,
            ),
        },
        "by_isotope": {
            "response_poisson_vs_truth": _summarize_groups(
                grouped_response_by_isotope
            ),
            "photopeak_channel_source_aggregate_vs_truth": _summarize_groups(
                grouped_channel_aggregate_by_isotope
            ),
            "photopeak_channel_line_equivalent_vs_line_truth": _summarize_groups(
                grouped_channel_line_by_isotope
            ),
        },
        "by_channel": _summarize_groups(grouped_channel_line_by_label),
        "worst_photopeak_channel_rows": worst_channel_rows[: max(int(top), 0)],
        "worst_photopeak_channel_aggregate_rows": worst_aggregate_rows[
            : max(int(top), 0)
        ],
        "recommendation": recommendation,
    }


def _recommendation(
    *,
    response_summary: dict[str, float],
    channel_summary: dict[str, float],
    gate: dict[str, object],
) -> dict[str, object]:
    """Return a replacement recommendation from current replay metrics."""
    response_mean = float(response_summary.get("mean", float("inf")))
    response_p95 = float(response_summary.get("p95", float("inf")))
    channel_mean = float(channel_summary.get("mean", float("inf")))
    channel_p95 = float(channel_summary.get("p95", float("inf")))
    mean_ratio = channel_mean / max(response_mean, 1e-12)
    p95_ratio = channel_p95 / max(response_p95, 1e-12)
    if bool(gate.get("passed")) and mean_ratio <= 0.9 and p95_ratio <= 0.9:
        decision = "candidate_for_pf_observation_ablation"
        reason = "line channels pass the gate and improve both mean and p95"
    elif mean_ratio < 1.0 or p95_ratio < 1.0:
        decision = "diagnostic_only_until_likelihood_ablation"
        reason = "line channels improve one metric but are not a drop-in win"
    else:
        decision = "do_not_replace_current_method"
        reason = "line-channel aggregate is worse than current response_poisson"
    return {
        "decision": decision,
        "reason": reason,
        "channel_to_response_mean_ratio": float(mean_ratio),
        "channel_to_response_p95_ratio": float(p95_ratio),
    }


def _default_validation_dir() -> Path:
    """Return the current saved new-seed validation directory."""
    return (
        ROOT
        / "results"
        / "spectrum_validation"
        / "response_truth_generalization_newseed_20260608_211129"
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=_default_validation_dir(),
        help="Directory containing saved spectra.npz and results.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json",
        help="Runtime Geant4 spectrum configuration.",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    parser.add_argument("--spectra", type=Path, default=None)
    parser.add_argument("--truth-min", type=float, default=100.0)
    parser.add_argument("--line-truth-min", type=float, default=100.0)
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=RESPONSE_VS_TRUTH_GATE[0],
        help="Mean relative-error threshold for diagnostic quality gates.",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=RESPONSE_VS_TRUTH_GATE[1],
        help="P95 relative-error threshold for diagnostic quality gates.",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def _print_summary(payload: dict[str, Any]) -> None:
    """Print the main replay metrics in percent units."""
    for key in (
        "response_poisson_vs_truth",
        "photopeak_channel_source_aggregate_vs_truth",
        "photopeak_channel_line_equivalent_vs_line_truth",
    ):
        summary = payload[key]
        print(
            f"{key}: mean {100.0 * summary.get('mean', float('nan')):.3f}%, "
            f"p95 {100.0 * summary.get('p95', float('nan')):.3f}%, "
            f"max {100.0 * summary.get('max', float('nan')):.3f}%"
        )
    recommendation = payload["recommendation"]
    print(
        "recommendation: "
        f"{recommendation['decision']} ({recommendation['reason']})"
    )


def main() -> None:
    """Replay photopeak-channel counts and optionally write a JSON report."""
    args = _parser().parse_args()
    validation_dir = Path(args.validation_dir)
    results_path = Path(args.results_json or validation_dir / "results.json")
    spectra_path = Path(args.spectra or validation_dir / "spectra.npz")
    payload = replay_photopeak_channels(
        config_path=Path(args.config),
        results_path=results_path,
        spectra_path=spectra_path,
        isotopes=ISOTOPES,
        truth_min=float(args.truth_min),
        line_truth_min=float(args.line_truth_min),
        top=int(args.top),
        mean_threshold=float(args.mean_threshold),
        p95_threshold=float(args.p95_threshold),
    )
    _print_summary(payload)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if bool(args.pretty) else None
        output_path.write_text(
            json.dumps(payload, indent=indent, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
