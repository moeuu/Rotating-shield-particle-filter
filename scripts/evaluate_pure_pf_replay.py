"""Evaluate a completed pure-PF MeasurementLog replay against external truth."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from measurement.source_surfaces import source_surface_kinds
from runtime.measurement_log import MeasurementLog, load_measurement_log

from evaluation_metrics import compute_metrics
from pf.atomic_io import atomic_write_bytes
from pf.provenance import canonical_json_bytes
from pf.pure_estimator import PurePFEstimator
from pf.replay import replay_measurement_log

ARTIFACT_TYPE = "pure_pf_measurement_log_replay_evaluation"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without interpreting its contents."""
    return sha256(path.read_bytes()).hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    """Return one finite float while rejecting booleans and invalid values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _reject_duplicate_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate field names."""
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Truth source JSON contains duplicate field {key!r}.")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject non-standard NaN and infinity constants in truth JSON."""
    raise ValueError(
        f"Truth source JSON contains non-finite constant {value!r}."
    )


def _load_truth_sources(path: Path) -> list[dict[str, Any]]:
    """Load and normalize an explicit source-truth JSON document."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read truth source JSON {path}: {exc}") from exc
    entries = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Truth source JSON must be a list or contain a sources list.")

    sources: list[dict[str, Any]] = []
    for source_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Truth source entry {source_index} must be an object.")
        legacy_fields = sorted(
            field
            for field in ("pos", "strength_cps_1m", "strength", "intensity")
            if field in entry
        )
        if legacy_fields:
            raise ValueError(
                f"Truth source entry {source_index} uses removed field(s) "
                f"{legacy_fields}; use exactly isotope, position, and "
                "intensity_cps_1m."
            )
        required_fields = {"isotope", "position", "intensity_cps_1m"}
        unexpected_fields = sorted(set(entry) - required_fields)
        if unexpected_fields:
            raise ValueError(
                f"Truth source entry {source_index} contains unsupported field(s) "
                f"{unexpected_fields}."
            )
        isotope_raw = entry.get("isotope")
        if not isinstance(isotope_raw, str) or not isotope_raw.strip():
            raise ValueError(
                f"Truth source entry {source_index} requires a non-empty isotope."
            )
        isotope = isotope_raw.strip()
        position_raw = entry.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) != 3:
            raise ValueError(
                f"Truth source entry {source_index} position must have length 3."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in position_raw
        ):
            raise ValueError(
                f"Truth source entry {source_index} position must contain "
                "three JSON numbers."
            )
        position = [
            _finite_float(
                value,
                name=f"truth source {source_index} position coordinate",
            )
            for value in position_raw
        ]
        strength_raw = entry.get("intensity_cps_1m")
        if strength_raw is None:
            raise ValueError(
                f"Truth source entry {source_index} requires intensity_cps_1m."
            )
        if isinstance(strength_raw, bool) or not isinstance(
            strength_raw,
            (int, float),
        ):
            raise ValueError(
                f"Truth source entry {source_index} intensity_cps_1m must be "
                "a JSON number."
            )
        strength = _finite_float(
            strength_raw,
            name=f"truth source {source_index} intensity_cps_1m",
        )
        if strength <= 0.0:
            raise ValueError(
                f"Truth source entry {source_index} intensity must be positive."
            )
        sources.append(
            {
                "isotope": isotope,
                "position": position,
                "intensity_cps_1m": strength,
            }
        )
    return sources


def _environment_from_log(log: MeasurementLog) -> EnvironmentConfig:
    """Build the physical room declared by a validated MeasurementLog."""
    environment = log.environment
    detector_raw = environment.get("detector_position")
    detector_position = (
        None
        if detector_raw is None
        else tuple(
            _finite_float(value, name="environment detector coordinate")
            for value in detector_raw
        )
    )
    if detector_position is not None and len(detector_position) != 3:
        raise ValueError("MeasurementLog detector_position must have length 3.")
    return EnvironmentConfig(
        size_x=_finite_float(environment.get("size_x"), name="environment size_x"),
        size_y=_finite_float(environment.get("size_y"), name="environment size_y"),
        size_z=_finite_float(environment.get("size_z"), name="environment size_z"),
        detector_position=detector_position,
    )


def _obstacle_grid_from_log(log: MeasurementLog) -> ObstacleGrid | None:
    """Return the obstacle grid embedded in a validated MeasurementLog."""
    raw_grid = log.environment.get("obstacle_grid")
    if raw_grid is None:
        return None
    if not isinstance(raw_grid, dict):
        raise ValueError("MeasurementLog obstacle_grid must be an object or null.")
    return ObstacleGrid.from_dict(dict(raw_grid))


def _truth_by_isotope(
    sources: Sequence[Mapping[str, Any]],
    log: MeasurementLog,
) -> dict[str, list[dict[str, Any]]]:
    """Annotate external truth with physical surface kinds for evaluation."""
    if not sources:
        return {}
    positions = np.asarray(
        [source["position"] for source in sources],
        dtype=float,
    ).reshape(-1, 3)
    tolerance_m = max(
        0.0,
        _finite_float(
            log.runtime_config.get("posterior_surface_tolerance_m", 1.0e-5),
            name="posterior_surface_tolerance_m",
        ),
    )
    obstacle_height_m = _finite_float(
        log.runtime_config.get("obstacle_height_m", 2.0),
        name="obstacle_height_m",
    )
    kinds = source_surface_kinds(
        positions,
        _environment_from_log(log),
        _obstacle_grid_from_log(log),
        obstacle_height_m=obstacle_height_m,
        tolerance_m=tolerance_m,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for source, kind in zip(sources, kinds, strict=True):
        isotope = str(source["isotope"])
        result.setdefault(isotope, []).append(
            {
                "pos": [float(value) for value in source["position"]],
                "strength": float(source["intensity_cps_1m"]),
                "surface_kind": "off_surface" if kind is None else str(kind),
            }
        )
    return result


def _estimate_payload(
    estimator: PurePFEstimator,
    estimates: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert canonical estimator arrays into JSON-safe evaluation sources."""
    result: dict[str, list[dict[str, Any]]] = {}
    for isotope, estimate in sorted(estimates.items()):
        positions = np.asarray(estimate[0], dtype=float).reshape(-1, 3)
        strengths = np.asarray(estimate[1], dtype=float).reshape(-1)
        if strengths.size != positions.shape[0]:
            raise ValueError(
                f"Estimate {isotope} must have one strength per source position."
            )
        if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(strengths)):
            raise ValueError(f"Estimate {isotope} contains non-finite values.")
        kinds = estimator.structural_surface_kinds(
            isotope,
            positions,
            strict=True,
        )
        result[str(isotope)] = [
            {
                "pos": [float(value) for value in position],
                "strength": float(strength),
                "surface_kind": str(kind),
            }
            for position, strength, kind in zip(
                positions,
                strengths,
                kinds,
                strict=True,
            )
        ]
    return result


def build_evaluation_payload(
    *,
    measurement_log: Path,
    config: Path,
    truth_source_json: Path,
    profile: str = "pf_strict",
    seed: int = 0,
    match_radius_m: float = 0.5,
) -> dict[str, Any]:
    """Replay all observations and build a post-hoc truth evaluation payload."""
    log_path = measurement_log.expanduser().resolve()
    config_path = config.expanduser().resolve()
    truth_path = truth_source_json.expanduser().resolve()
    truth_sources = _load_truth_sources(truth_path)
    log = load_measurement_log(log_path)
    radius = _finite_float(match_radius_m, name="match_radius_m")
    if radius < 0.0:
        raise ValueError("match_radius_m must be non-negative.")

    estimator, trace = replay_measurement_log(
        log_path,
        config_path,
        profile=profile,
        seed=int(seed),
    )
    if len(trace) != len(log.records):
        raise RuntimeError(
            "Evaluation requires a complete replay of every MeasurementLog record."
        )

    estimates = estimator.estimates()
    uncertainty_kwargs: dict[str, float] = {}
    uncertainty_radius = log.runtime_config.get("posterior_uncertainty_match_radius_m")
    if uncertainty_radius is not None:
        uncertainty_kwargs["match_radius_m"] = max(
            0.0,
            _finite_float(
                uncertainty_radius,
                name="posterior_uncertainty_match_radius_m",
            ),
        )
    surface_tolerance_m = max(
        0.0,
        _finite_float(
            log.runtime_config.get("posterior_surface_tolerance_m", 1.0e-5),
            name="posterior_surface_tolerance_m",
        ),
    )
    uncertainty = estimator.posterior_source_uncertainty(
        estimates,
        surface_tolerance_m=surface_tolerance_m,
        **uncertainty_kwargs,
    )
    estimates_by_isotope = _estimate_payload(estimator, estimates)
    truth_by_isotope = _truth_by_isotope(truth_sources, log)
    close_pair_distance_m = max(
        0.0,
        _finite_float(
            log.runtime_config.get("evaluation_close_pair_distance_m", 2.0),
            name="evaluation_close_pair_distance_m",
        ),
    )
    close_pair_minimum_m = max(
        0.0,
        _finite_float(
            log.runtime_config.get(
                "evaluation_close_pair_min_estimated_separation_m",
                0.5,
            ),
            name="evaluation_close_pair_min_estimated_separation_m",
        ),
    )
    metrics = compute_metrics(
        truth_by_isotope,
        estimates_by_isotope,
        match_radius_m=radius,
        surface_atlas=estimator.continuous_surface_atlas(),
        close_pair_distance_m=close_pair_distance_m,
        close_pair_min_estimated_separation_m=close_pair_minimum_m,
        uncertainty_by_iso=uncertainty,
    )
    posterior = estimator.posterior_snapshot().to_dict()
    config_digest = _sha256_file(config_path)
    if str(estimator.config_hash) != config_digest:
        raise RuntimeError(
            "Replay estimator config hash does not match the input file."
        )

    return {
        "schema_version": 1,
        "pure_pf_schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "provenance": {
            "measurement_log_path": str(log_path),
            "measurement_log_schema_version": int(log.schema_version),
            "measurement_log_sha256": str(log.log_sha256),
            "measurement_log_run_id": str(log.run_manifest["run_id"]),
            "measurement_log_repository_commit": str(
                log.run_manifest["repository_commit"]
            ),
            "measurement_log_resolved_config_sha256": str(log.resolved_config_sha256),
            "record_count": len(log.records),
            "replayed_record_count": len(trace),
            "replay_config_path": str(config_path),
            "replay_config_sha256": config_digest,
            "replay_resolved_config_sha256": str(estimator.resolved_config_hash),
            "replay_profile": str(profile),
            "replay_seed": int(seed),
            "estimator_variant": str(estimator.estimator_variant),
            "final_state_sha256": sha256(estimator.serialized_state()).hexdigest(),
            "truth_source_json_path": str(truth_path),
            "truth_source_json_sha256": _sha256_file(truth_path),
            "truth_source_count": len(truth_sources),
            "truth_isotopes": sorted(truth_by_isotope),
            "truth_scope": "posthoc_evaluation_only",
            "truth_passed_to_pf_replay": False,
            "match_radius_m": radius,
            "posterior_uncertainty_match_radius_m": uncertainty_kwargs.get(
                "match_radius_m",
                0.8,
            ),
            "posterior_surface_tolerance_m": surface_tolerance_m,
            "close_pair_distance_m": close_pair_distance_m,
            "close_pair_min_estimated_separation_m": close_pair_minimum_m,
        },
        "posterior": posterior,
        "truth": truth_by_isotope,
        "estimates": estimates_by_isotope,
        "uncertainty": uncertainty,
        "metrics": metrics,
    }


def write_evaluation_json(output: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish one canonical evaluation JSON without overwriting."""
    target = output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace evaluation output {target}.")
    return atomic_write_bytes(target, canonical_json_bytes(dict(payload)))


def evaluate_and_write(
    *,
    measurement_log: Path,
    config: Path,
    truth_source_json: Path,
    output: Path,
    profile: str = "pf_strict",
    seed: int = 0,
    match_radius_m: float = 0.5,
) -> Path:
    """Build and atomically publish one completed replay evaluation."""
    payload = build_evaluation_payload(
        measurement_log=measurement_log,
        config=config,
        truth_source_json=truth_source_json,
        profile=profile,
        seed=seed,
        match_radius_m=match_radius_m,
    )
    return write_evaluation_json(output, payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the standalone pure-PF replay evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-source-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("pf_strict",), default="pf_strict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-radius-m", type=float, default=0.5)
    args = parser.parse_args(None if argv is None else list(argv))
    output = evaluate_and_write(
        measurement_log=args.measurement_log,
        config=args.config,
        truth_source_json=args.truth_source_json,
        output=args.output,
        profile=args.profile,
        seed=args.seed,
        match_radius_m=args.match_radius_m,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
