"""Compare old and candidate full-spectrum models on one saved PF run.

The diagnostic never fits parameters and never re-runs the particle filter.
It evaluates the physical truth, the saved final state, and a mixed
Co/Cs-final plus Eu-truth counterfactual on the immutable raw spectra.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pf.particle_filter import IsotopeParticle
from pf.replay import build_replay_estimator
from pf.state import IsotopeState
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    build_forward_model_manifest,
    load_measurement_log,
)
from spectrum.transport_spectral import GeometryConditionedSpectralModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _upgrade_diagnostic_log(log: MeasurementLog) -> MeasurementLog:
    """Remove superseded model-selection fields from an embedded-model log.

    The interrupted acquisition predated the rule that a resolved log may
    contain either registry selection or an embedded model, but not both.  The
    authenticated embedded model is the one identified by every observation,
    so only the now-ambiguous selection metadata is removed in memory.
    """
    runtime = json.loads(json.dumps(dict(log.runtime_config)))
    for key in (
        "isotope_experiment_profile",
        "full_spectrum_model_registry_path",
        "full_spectrum_model_registry_file_sha256",
        "full_spectrum_profile_calibration_status",
    ):
        runtime.pop(key, None)
    forward = build_forward_model_manifest(
        runtime_config=runtime,
        environment=log.environment,
        obstacle_layout_path=log.run_manifest.get("obstacle_layout_path"),
        isotopes=tuple(log.run_manifest["isotopes"]),
        repository_commit=log.run_manifest["repository_commit"],
        resolved_config_sha256=log.run_manifest["resolved_config_sha256"],
        run_root=log.path,
    )
    return replace(
        log,
        runtime_config=runtime,
        forward_model_manifest=forward,
    )


def _parse_args() -> argparse.Namespace:
    """Parse immutable inputs and one new output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-log", required=True, type=Path)
    parser.add_argument("--truth-source-config", required=True, type=Path)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--posterior", type=Path)
    state_group.add_argument("--estimate-trace", type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _truth_states(path: Path) -> dict[str, list[dict[str, object]]]:
    """Load evaluation-only truth states without exposing them to inference."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, object]]] = {}
    for source in payload["sources"]:
        isotope = str(source["isotope"])
        result.setdefault(isotope, []).append(
            {
                "surface_chart_id": int(source["surface_chart_id"]),
                "surface_uv": [float(value) for value in source["surface_uv"]],
                "strength_cps_1m": float(source["intensity_cps_1m"]),
            }
        )
    return result


def _posterior_states(path: Path) -> dict[str, list[dict[str, object]]]:
    """Load canonical saved modes as a fixed counterfactual state."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, object]]] = {}
    for isotope, isotope_payload in payload["isotopes"].items():
        result[str(isotope)] = [
            {
                "surface_chart_id": int(mode["surface_chart_id"]),
                "surface_uv": [float(value) for value in mode["surface_uv"]],
                "strength_cps_1m": float(
                    mode["strength_representative_cps_1m"]
                ),
            }
            for mode in isotope_payload["modes"]
        ]
    return result


def _trace_states(path: Path, estimator: Any) -> dict[str, list[dict[str, object]]]:
    """Resolve the final saved canonical XYZ trace onto exact surface charts."""
    final_row: Mapping[str, object] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid estimate-trace JSON on line {line_number}."
                ) from exc
            if not isinstance(candidate, Mapping):
                raise ValueError("Each estimate-trace row must be an object.")
            final_row = candidate
    if final_row is None:
        raise ValueError("Estimate trace is empty.")
    estimates = final_row.get("estimates")
    if not isinstance(estimates, list):
        raise ValueError("Final estimate-trace row has no estimates list.")
    grouped: dict[str, list[Mapping[str, object]]] = {
        isotope: [] for isotope in estimator.joint_isotope_order()
    }
    for estimate in estimates:
        if not isinstance(estimate, Mapping):
            raise ValueError("Estimate-trace entries must be objects.")
        isotope = str(estimate.get("isotope"))
        if isotope not in grouped:
            raise ValueError(f"Unexpected isotope in estimate trace: {isotope}.")
        grouped[isotope].append(estimate)
    result: dict[str, list[dict[str, object]]] = {}
    for isotope, isotope_estimates in grouped.items():
        if not isotope_estimates:
            result[isotope] = []
            continue
        positions = np.asarray(
            [estimate["pos"] for estimate in isotope_estimates],
            dtype=np.float64,
        )
        atlas = estimator.filters[isotope]._structural_rj_surface_atlas
        if atlas is None:
            raise RuntimeError("Continuous surface atlas is unavailable.")
        chart_ids, surface_uv = atlas.locate_positions(positions)
        result[isotope] = [
            {
                "surface_chart_id": int(chart_ids[index]),
                "surface_uv": [
                    float(surface_uv[index, 0]),
                    float(surface_uv[index, 1]),
                ],
                "strength_cps_1m": float(estimate["strength"]),
            }
            for index, estimate in enumerate(isotope_estimates)
        ]
    return result


def _state(rows: Sequence[Mapping[str, object]]) -> IsotopeState:
    """Convert one serialized isotope source list into continuous PF state."""
    return IsotopeState(
        num_sources=len(rows),
        strengths=np.asarray(
            [float(row["strength_cps_1m"]) for row in rows],
            dtype=np.float64,
        ),
        surface_chart_ids=np.asarray(
            [int(row["surface_chart_id"]) for row in rows],
            dtype=np.int64,
        ),
        surface_uv=np.asarray(
            [row["surface_uv"] for row in rows],
            dtype=np.float64,
        ).reshape(len(rows), 2),
    )


def _install_joint_state(
    estimator: Any,
    rows_by_isotope: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Install one aligned diagnostic particle without changing its identity."""
    for isotope in estimator.joint_isotope_order():
        filt = estimator.filters[isotope]
        identity = filt.continuous_particles[0].joint_row_identity
        state = _state(rows_by_isotope.get(isotope, ()))
        filt._canonicalize_structural_rj_state(state)
        filt.continuous_particles = [
            IsotopeParticle(
                state=state,
                log_weight=0.0,
                joint_row_identity=identity,
            )
        ]
        filt.N = 1
        filt.config.num_particles = 1
    estimator._assert_joint_particle_alignment()


def _station_groups(
    records: Sequence[MeasurementLogRecord],
) -> list[list[MeasurementLogRecord]]:
    """Group the immutable records into contiguous completed stations."""
    groups: list[list[MeasurementLogRecord]] = []
    current: list[MeasurementLogRecord] = []
    current_id: int | None = None
    for record in records:
        if current_id is None:
            current_id = int(record.station_id)
        if int(record.station_id) != current_id:
            raise RuntimeError("Measurement log station groups are not contiguous.")
        current.append(record)
        if record.metadata.get("station_complete", False) is True:
            groups.append(current)
            current = []
            current_id = None
    if current:
        raise RuntimeError("Final measurement station is incomplete.")
    return groups


def _build_stations(estimator: Any, records: Sequence[MeasurementLogRecord]) -> list[Any]:
    """Build strict station objects without assimilating any observation."""
    estimator.poses = []
    estimator.kernel_cache = None
    stations = []
    contract_hash = estimator._full_spectrum_model().contract_hash_sha256
    for sequence_id, group in enumerate(_station_groups(records)):
        pose = np.asarray(group[0].detector_pose_xyz, dtype=np.float64)
        estimator.add_measurement_pose(pose, reset_filters=False)
        station_records = tuple(
            (
                np.asarray(record.spectrum_counts, dtype=np.int64),
                int(record.fe_orientation_index),
                int(record.pb_orientation_index),
                float(record.live_time_s),
            )
            for record in group
        )
        stations.append(
            estimator._joint_station_from_spectrum_records(
                station_records,
                pose_idx=sequence_id,
                station_sequence_id=sequence_id,
                generative_contract_hash_sha256=contract_hash,
            )
        )
    return stations


def _evaluate_state(
    estimator: Any,
    stations: Sequence[Any],
    rows_by_isotope: Mapping[str, Sequence[Mapping[str, object]]],
    models: Mapping[str, GeometryConditionedSpectralModel],
) -> dict[str, object]:
    """Evaluate one fixed state under models sharing one physical mean kernel."""
    _install_joint_state(estimator, rows_by_isotope)
    result: dict[str, object] = {
        "cardinality": {
            isotope: len(rows_by_isotope.get(isotope, ()))
            for isotope in estimator.joint_isotope_order()
        },
        "models": {},
    }
    model_metrics: dict[str, object] = {}
    for model_name, model in models.items():
        station_log_likelihoods: list[float] = []
        station_count_log_likelihoods: list[float] = []
        station_total_z: list[float] = []
        station_mark_tail: list[float] = []
        predicted_total = 0.0
        observed_total = 0.0
        for station in stations:
            components = tuple(
                value.detach().cpu().numpy().astype(np.float64, copy=False)
                for value in estimator._joint_station_transport_components_torch(
                    station
                )
            )
            station_log_likelihoods.append(
                float(
                    model.log_likelihood_numpy(
                        station.spectrum_vb,
                        components[0],
                        components[1],
                        components[2],
                        station.live_times_s,
                    )[0]
                )
            )
            station_count_log_likelihoods.append(
                float(
                    model.count_log_likelihood_numpy(
                        station.spectrum_vb,
                        components[0],
                        components[1],
                        components[2],
                        station.live_times_s,
                    )[0]
                )
            )
            predicted = model.predict_mean_numpy(
                components[0],
                components[1],
                components[2],
                station.live_times_s,
            )[0]
            predicted_total += float(np.sum(predicted))
            observed_total += float(np.sum(station.spectrum_vb))
            innovation = model.posterior_predictive_innovation_numpy(
                station.spectrum_vb,
                components[0],
                components[1],
                components[2],
                station.live_times_s,
                np.ones(1, dtype=np.float64),
                confidence=0.99,
            )
            station_total_z.append(
                float(innovation["renewal_total_max_abs_z"])
            )
            mark_tail = innovation["conditional_mark_tail_probability"]
            station_mark_tail.append(
                float(mark_tail) if mark_tail is not None else float("nan")
            )
        finite_tails = np.asarray(station_mark_tail, dtype=np.float64)
        finite_tails = finite_tails[np.isfinite(finite_tails)]
        model_metrics[model_name] = {
            "log_likelihood": float(np.sum(station_log_likelihoods)),
            "station_log_likelihoods": station_log_likelihoods,
            "count_log_likelihood": float(
                np.sum(station_count_log_likelihoods)
            ),
            "station_count_log_likelihoods": (
                station_count_log_likelihoods
            ),
            "predicted_to_observed_total_ratio": (
                predicted_total / observed_total
            ),
            "maximum_station_total_abs_z": float(np.max(station_total_z)),
            "median_station_total_abs_z": float(np.median(station_total_z)),
            "minimum_station_mark_tail_probability": (
                None if finite_tails.size == 0 else float(np.min(finite_tails))
            ),
            "station_count": len(stations),
        }
    result["models"] = model_metrics
    return result


def main() -> int:
    """Run the read-only causal comparison and save one deterministic report."""
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    log = _upgrade_diagnostic_log(load_measurement_log(args.measurement_log))
    estimator = build_replay_estimator(
        log,
        {"pure_pf_schema_version": 1, "estimator_profile": "pf_strict"},
        profile="pf_strict",
        seed=0,
    )
    estimator._ensure_kernel_cache()
    estimator._configure_joint_particle_filters()
    stations = _build_stations(estimator, log.records)
    old_model = estimator._full_spectrum_model()
    candidate_model = GeometryConditionedSpectralModel.from_manifest_payload(
        json.loads(args.candidate_model.read_text(encoding="utf-8"))
    )
    candidate_model.require_runtime_ready()
    if tuple(candidate_model.line_identity) != tuple(old_model.line_identity):
        raise RuntimeError("Candidate and logged models use different line layouts.")
    truth = _truth_states(args.truth_source_config)
    if args.posterior is not None:
        final = _posterior_states(args.posterior)
        saved_state_kind = "canonical_posterior_modes"
    else:
        final = _trace_states(args.estimate_trace, estimator)
        saved_state_kind = "final_canonical_estimate_trace"
    mixed = {
        isotope: (truth[isotope] if isotope == "Eu-154" else final[isotope])
        for isotope in estimator.joint_isotope_order()
    }
    models = {"logged_exact": old_model, "candidate_discrepancy": candidate_model}
    states = {
        "physical_truth": truth,
        "saved_final_split_ghost": final,
        "saved_final_co_cs_plus_truth_eu": mixed,
    }
    evaluations = {
        name: _evaluate_state(estimator, stations, rows, models)
        for name, rows in states.items()
    }
    truth_ll = evaluations["physical_truth"]["models"]
    comparisons = {}
    for model_name in models:
        truth_value = float(truth_ll[model_name]["log_likelihood"])
        comparisons[model_name] = {
            f"{state_name}_minus_truth_log_likelihood": float(
                evaluation["models"][model_name]["log_likelihood"]
            )
            - truth_value
            for state_name, evaluation in evaluations.items()
            if state_name != "physical_truth"
        }
        truth_count_value = float(
            truth_ll[model_name]["count_log_likelihood"]
        )
        comparisons[model_name].update(
            {
                f"{state_name}_minus_truth_count_log_likelihood": float(
                    evaluation["models"][model_name][
                        "count_log_likelihood"
                    ]
                )
                - truth_count_value
                for state_name, evaluation in evaluations.items()
                if state_name != "physical_truth"
            }
        )
    payload = {
        "schema_version": 1,
        "diagnostic": "saved_full_spectrum_truth_split_ghost_causality",
        "fit_or_tuning_performed": False,
        "saved_state_kind": saved_state_kind,
        "acceptance_use": "diagnostic_only_legacy_environment",
        "candidate_environment_applicability": (
            "not_asserted_legacy_log_has_no_authenticated_geometry_family"
        ),
        "measurement_log": str(args.measurement_log.resolve()),
        "record_count": len(log.records),
        "candidate_model": str(args.candidate_model.resolve()),
        "candidate_runtime_ready": candidate_model.runtime_ready,
        "candidate_production_ready": candidate_model.production_ready,
        "evaluations": evaluations,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
