"""Fit and validate Geant4 spectrum-net response factors for the PF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.continuous_kernels import ContinuousKernel, geometric_term
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from realtime_demo import load_sources_from_json
from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from spectrum.net_response import NetResponseCalibration, fit_net_response_calibration
from spectrum.nnls import nnls_solve
from spectrum.pipeline import SpectralDecomposer


def _default_protocol(dwell_time_s: float) -> list[dict[str, Any]]:
    """Return a fit/validation protocol that exercises free and blocked octants."""
    return [
        {
            "shot_id": "fit_free_near",
            "role": "fit",
            "pose": [1.0, 1.0, 0.5],
            "fe_index": 0,
            "pb_index": 0,
            "dwell_time_s": dwell_time_s,
        },
        {
            "shot_id": "fit_free_shifted",
            "role": "fit",
            "pose": [2.0, 1.0, 0.5],
            "fe_index": 0,
            "pb_index": 0,
            "dwell_time_s": dwell_time_s,
        },
        {
            "shot_id": "validate_fe_only",
            "role": "validate",
            "pose": [1.0, 1.0, 0.5],
            "fe_index": 7,
            "pb_index": 0,
            "dwell_time_s": dwell_time_s,
        },
        {
            "shot_id": "validate_pb_only",
            "role": "validate",
            "pose": [1.0, 1.0, 0.5],
            "fe_index": 0,
            "pb_index": 7,
            "dwell_time_s": dwell_time_s,
        },
        {
            "shot_id": "validate_fe_pb",
            "role": "validate",
            "pose": [1.0, 1.0, 0.5],
            "fe_index": 7,
            "pb_index": 7,
            "dwell_time_s": dwell_time_s,
        },
    ]


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _load_protocol(path: Path | None, dwell_time_s: float) -> list[dict[str, Any]]:
    """Load a shot protocol or return the built-in shield-validation protocol."""
    if path is None:
        return _default_protocol(dwell_time_s)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        shots = payload.get("shots", [])
    else:
        shots = payload
    if not isinstance(shots, list):
        raise ValueError("Protocol JSON must be a list or contain a 'shots' list.")
    return [dict(shot) for shot in shots]


def _build_scene(sources: Iterable[PointSource], usd_path: str | None) -> SceneDescription:
    """Build a minimal Geant4 scene without static obstacles."""
    return SceneDescription(
        room_size_xyz=(10.0, 20.0, 10.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[],
        author_obstacle_prims=False,
        sources=[
            SourceDescription(
                isotope=source.isotope,
                position_xyz=tuple(float(v) for v in source.position),
                intensity_cps_1m=float(source.intensity_cps_1m),
            )
            for source in sources
        ],
        usd_path=usd_path,
    )


def _scale_sources(sources: Iterable[PointSource], scale: float) -> list[PointSource]:
    """Return sources with intensities multiplied for Monte Carlo history scaling."""
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("source scale must be positive.")
    return [
        PointSource(
            isotope=source.isotope,
            position=source.position,
            intensity_cps_1m=float(source.intensity_cps_1m) * factor,
        )
        for source in sources
    ]


def _ideal_counts(
    sources: Iterable[PointSource],
    isotopes: Iterable[str],
    *,
    detector_pos: np.ndarray,
    fe_index: int,
    pb_index: int,
    live_time_s: float,
    kernel: ContinuousKernel,
) -> dict[str, float]:
    """Compute ideal inverse-square plus shield counts for one shot."""
    counts = {str(isotope): 0.0 for isotope in isotopes}
    for source in sources:
        source_pos = source.position_array()
        attenuation = kernel.attenuation_factor_pair(
            source.isotope,
            source_pos,
            detector_pos,
            fe_index,
            pb_index,
        )
        counts[source.isotope] += (
            float(live_time_s)
            * float(source.intensity_cps_1m)
            * geometric_term(detector_pos, source_pos)
            * float(attenuation)
        )
    return counts


def _run_observation(
    app: Geant4Application,
    shot: dict[str, Any],
    step_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Execute one simulator shot and return spectrum, detector pose, and metadata."""
    pose = tuple(float(v) for v in shot["pose"])
    observation = app.step(
        SimulationCommand(
            step_id=step_id,
            target_pose_xyz=pose,
            target_base_yaw_rad=0.0,
            fe_orientation_index=int(shot["fe_index"]),
            pb_orientation_index=int(shot["pb_index"]),
            dwell_time_s=float(shot["dwell_time_s"]),
        )
    )
    return (
        np.asarray(observation.spectrum_counts, dtype=float),
        np.asarray(observation.detector_pose_xyz, dtype=float),
        dict(observation.metadata),
    )


def _run_replicated_observation(
    app: Geant4Application,
    shot: dict[str, Any],
    step_id: int,
    *,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the average spectrum from repeated independent shots."""
    num_replicates = max(int(replicates), 1)
    spectra: list[np.ndarray] = []
    metadata_items: list[dict[str, Any]] = []
    detector_pos = np.zeros(3, dtype=float)
    for rep_idx in range(num_replicates):
        spectrum, detector_pos, metadata = _run_observation(
            app,
            shot,
            step_id + rep_idx,
        )
        spectra.append(spectrum)
        metadata_items.append(metadata)
    averaged = np.mean(np.stack(spectra, axis=0), axis=0)
    merged_metadata = dict(metadata_items[-1])
    primaries = [
        float(item.get("num_primaries", 0.0))
        for item in metadata_items
    ]
    merged_metadata["num_primaries"] = float(np.mean(primaries)) if primaries else 0.0
    tally_keys = {
        key
        for item in metadata_items
        for key in item
        if str(key).startswith("source_equivalent_counts_")
    }
    for key in tally_keys:
        values = [float(item.get(key, 0.0)) for item in metadata_items]
        merged_metadata[key] = float(np.mean(values)) if values else 0.0
    merged_metadata["measurement_replicates"] = num_replicates
    return averaged, detector_pos, merged_metadata


def _build_geant4_response_templates(
    runtime_config: dict[str, Any],
    sources: list[PointSource],
    kernel: ContinuousKernel,
    protocol: list[dict[str, Any]],
    *,
    source_config_usd_path: str | None,
    intensity_scale: float,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Build shot-specific Geant4 response templates normalized to theory counts."""
    if intensity_scale <= 0.0:
        raise ValueError("intensity_scale must be positive.")
    templates: dict[str, tuple[list[str], np.ndarray]] = {}
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    step_id = 100000
    try:
        for shot in protocol:
            names: list[str] = []
            columns: list[np.ndarray] = []
            for source in sources:
                scaled_source = PointSource(
                    isotope=source.isotope,
                    position=source.position,
                    intensity_cps_1m=float(source.intensity_cps_1m) * float(intensity_scale),
                )
                app.reset(_build_scene([scaled_source], usd_path=source_config_usd_path))
                spectrum, detector_pos, _ = _run_observation(app, shot, step_id)
                step_id += 1
                theory = _ideal_counts(
                    [scaled_source],
                    [scaled_source.isotope],
                    detector_pos=detector_pos,
                    fe_index=int(shot["fe_index"]),
                    pb_index=int(shot["pb_index"]),
                    live_time_s=float(shot["dwell_time_s"]),
                    kernel=kernel,
                )[scaled_source.isotope]
                if theory <= 0.0:
                    continue
                names.append(scaled_source.isotope)
                columns.append(np.asarray(spectrum, dtype=float) / float(theory))
            if columns:
                templates[str(shot["shot_id"])] = (names, np.column_stack(columns))
    finally:
        app.close()
    return templates


def _fit_geant4_response_counts(
    spectrum: np.ndarray,
    isotopes: list[str],
    shot_id: str,
    response_templates: dict[str, tuple[list[str], np.ndarray]] | None,
) -> dict[str, float]:
    """Fit a spectrum with shot-specific Geant4 response templates."""
    counts = {isotope: 0.0 for isotope in isotopes}
    if response_templates is None:
        return counts
    template = response_templates.get(str(shot_id))
    if template is None:
        return counts
    names, design = template
    coeffs = nnls_solve(design, np.asarray(spectrum, dtype=float))
    for name, value in zip(names, coeffs):
        counts[name] = counts.get(name, 0.0) + max(float(value), 0.0)
    return counts


def _source_tally_counts(
    metadata: dict[str, Any],
    isotopes: list[str],
    *,
    history_scale: float,
) -> dict[str, float]:
    """Read source-equivalent native Geant4 tallies from response metadata."""
    scale = max(float(history_scale), 1.0e-12)
    counts = {isotope: 0.0 for isotope in isotopes}
    for isotope in isotopes:
        key = f"source_equivalent_counts_{isotope}"
        counts[isotope] = float(metadata.get(key, 0.0)) / scale
    return counts


def _run_shot(
    app: Geant4Application,
    decomposer: SpectralDecomposer,
    sources: list[PointSource],
    isotopes: list[str],
    kernel: ContinuousKernel,
    shot: dict[str, Any],
    step_id: int,
    count_method: str,
    response_templates: dict[str, tuple[list[str], np.ndarray]] | None = None,
    measurement_replicates: int = 1,
    history_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Execute one Geant4 shot and return per-isotope calibration records."""
    pose = tuple(float(v) for v in shot["pose"])
    fe_index = int(shot["fe_index"])
    pb_index = int(shot["pb_index"])
    live_time_s = float(shot["dwell_time_s"])
    spectrum, detector_pos, metadata = _run_replicated_observation(
        app,
        shot,
        step_id,
        replicates=measurement_replicates,
    )
    spectrum = spectrum / float(history_scale)
    peak_window_counts = decomposer.compute_isotope_counts_thesis(
        spectrum,
        live_time_s=live_time_s,
        isotopes=isotopes,
    )
    response_matrix_counts = decomposer.compute_response_model_counts(
        spectrum,
        isotopes=isotopes,
    )
    geant4_response_counts = _fit_geant4_response_counts(
        spectrum,
        isotopes,
        str(shot["shot_id"]),
        response_templates,
    )
    geant4_source_tally_counts = _source_tally_counts(
        metadata,
        isotopes,
        history_scale=history_scale,
    )
    if count_method == "response_matrix":
        net_counts = response_matrix_counts
    elif count_method == "geant4_response_matrix":
        net_counts = geant4_response_counts
    elif count_method == "geant4_source_tally":
        net_counts = geant4_source_tally_counts
    elif count_method == "peak_window":
        net_counts = peak_window_counts
    else:
        raise ValueError(f"Unknown count_method: {count_method}")
    theory_counts = _ideal_counts(
        sources,
        isotopes,
        detector_pos=detector_pos,
        fe_index=fe_index,
        pb_index=pb_index,
        live_time_s=live_time_s,
        kernel=kernel,
    )
    records: list[dict[str, Any]] = []
    for isotope in isotopes:
        theory = float(theory_counts.get(isotope, 0.0))
        net = float(net_counts.get(isotope, 0.0))
        records.append(
            {
                "shot_id": str(shot["shot_id"]),
                "role": str(shot["role"]),
                "isotope": isotope,
                "pose_x": pose[0],
                "pose_y": pose[1],
                "pose_z": pose[2],
                "fe_index": fe_index,
                "pb_index": pb_index,
                "live_time_s": live_time_s,
                "count_method": count_method,
                "theory_counts": theory,
                "net_counts": net,
                "peak_window_counts": float(peak_window_counts.get(isotope, 0.0)),
                "response_matrix_counts": float(response_matrix_counts.get(isotope, 0.0)),
                "geant4_response_counts": float(geant4_response_counts.get(isotope, 0.0)),
                "geant4_source_tally_counts": float(geant4_source_tally_counts.get(isotope, 0.0)),
                "raw_total_counts": float(np.sum(spectrum)),
                "num_primaries": float(metadata.get("num_primaries", 0.0)),
                "measurement_replicates": int(metadata.get("measurement_replicates", 1)),
                "history_scale": float(history_scale),
            }
        )
    return records


def _add_calibrated_columns(
    records: list[dict[str, Any]],
    calibration: NetResponseCalibration,
) -> list[dict[str, Any]]:
    """Add calibrated predictions and residuals to each report record."""
    enriched: list[dict[str, Any]] = []
    baseline_by_isotope = {
        str(record["isotope"]): record
        for record in records
        if record.get("shot_id") == "fit_free_near"
    }
    for record in records:
        item = dict(record)
        scale = calibration.response_scale(str(item["isotope"]))
        prediction = scale * float(item["theory_counts"])
        net = float(item["net_counts"])
        item["response_scale"] = scale
        item["calibrated_prediction"] = prediction
        item["residual_counts"] = net - prediction
        item["relative_residual"] = (net - prediction) / max(abs(net), 1.0)
        baseline = baseline_by_isotope.get(str(item["isotope"]))
        if baseline is not None:
            item["theory_attenuation_ratio"] = float(item["theory_counts"]) / max(
                float(baseline["theory_counts"]),
                1e-12,
            )
            item["net_attenuation_ratio"] = net / max(float(baseline["net_counts"]), 1e-12)
        else:
            item["theory_attenuation_ratio"] = 1.0
            item["net_attenuation_ratio"] = 1.0
        enriched.append(item)
    return enriched


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write calibration records as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geant4/shield_validation_scene.json")
    parser.add_argument("--source-config", default="source_layouts/shield_validation.json")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--dwell-time-s", type=float, default=30.0)
    parser.add_argument(
        "--count-method",
        choices=("peak_window", "response_matrix", "geant4_response_matrix", "geant4_source_tally"),
        default=None,
        help="Spectrum-to-isotope count extraction used for net_counts.",
    )
    parser.add_argument(
        "--template-intensity-scale",
        type=float,
        default=5.0,
        help="Source intensity multiplier for Geant4 response-template shots.",
    )
    parser.add_argument(
        "--measurement-replicates",
        type=int,
        default=1,
        help="Independent native shots to average for each reported 30 s measurement.",
    )
    parser.add_argument(
        "--history-scale",
        type=float,
        default=1.0,
        help="Monte Carlo source-intensity multiplier for reported shots; spectra are divided back by this factor.",
    )
    parser.add_argument("--calibration-output", default="configs/geant4/net_response_calibration.json")
    parser.add_argument("--report-csv", default="results/geant4_net_response_validation.csv")
    parser.add_argument("--report-json", default="results/geant4_net_response_validation.json")
    return parser.parse_args()


def main() -> None:
    """Run the Geant4 net-response calibration workflow."""
    args = parse_args()
    config_path = _resolve_path(args.config)
    source_path = _resolve_path(args.source_config)
    protocol_path = None if args.protocol in (None, "") else _resolve_path(args.protocol)
    output_path = _resolve_path(args.calibration_output)
    report_csv_path = _resolve_path(args.report_csv)
    report_json_path = _resolve_path(args.report_json)

    runtime_config = load_runtime_config(config_path.as_posix())
    count_method = str(
        args.count_method
        or runtime_config.get("calibration_count_method")
        or runtime_config.get("spectrum_count_method")
        or "peak_window"
    ).strip().lower()
    if count_method not in {"peak_window", "response_matrix", "geant4_response_matrix", "geant4_source_tally"}:
        raise ValueError(f"Unknown count method: {count_method}")
    template_intensity_scale = float(
        runtime_config.get("calibration_template_intensity_scale", args.template_intensity_scale)
    )
    measurement_replicates = max(
        int(runtime_config.get("calibration_measurement_replicates", args.measurement_replicates)),
        1,
    )
    history_scale = float(runtime_config.get("calibration_history_scale", args.history_scale))
    executable_path = runtime_config.get("executable_path")
    if executable_path:
        resolved_executable = _resolve_path(str(executable_path))
        runtime_config["executable_path"] = resolved_executable.as_posix()
        if not resolved_executable.exists():
            raise FileNotFoundError(f"Geant4 executable not found: {resolved_executable}")
    sources = load_sources_from_json(source_path)
    isotopes = sorted({source.isotope for source in sources})
    decomposer = SpectralDecomposer()
    kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes),
        shield_params=ShieldParams(),
        use_gpu=False,
    )
    scene = _build_scene(_scale_sources(sources, history_scale), usd_path=runtime_config.get("usd_path"))
    protocol = _load_protocol(protocol_path, float(args.dwell_time_s))
    response_templates = None
    if count_method == "geant4_response_matrix":
        response_templates = _build_geant4_response_templates(
            runtime_config,
            sources,
            kernel,
            protocol,
            source_config_usd_path=runtime_config.get("usd_path"),
            intensity_scale=template_intensity_scale,
        )
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    try:
        app.reset(scene)
        records: list[dict[str, Any]] = []
        for step_id, shot in enumerate(protocol):
            records.extend(
                _run_shot(
                    app,
                    decomposer,
                    sources,
                    isotopes,
                    kernel,
                    shot,
                    step_id,
                    count_method,
                    response_templates,
                    measurement_replicates=measurement_replicates,
                    history_scale=history_scale,
                )
            )
    finally:
        app.close()

    fit_records = [record for record in records if record["role"] == "fit"]
    calibration = fit_net_response_calibration(
        fit_records,
        isotopes=isotopes,
        metadata={
            "config": config_path.as_posix(),
            "source_config": source_path.as_posix(),
            "protocol": "built-in" if protocol_path is None else protocol_path.as_posix(),
            "fit_role": "fit",
            "validation_role": "validate",
            "count_method": count_method,
            "template_intensity_scale": template_intensity_scale,
            "measurement_replicates": measurement_replicates,
            "history_scale": history_scale,
        },
    )
    calibration.save(output_path)
    report_records = _add_calibrated_columns(records, calibration)
    _write_csv(report_csv_path, report_records)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(
            {
                "calibration": calibration.to_dict(),
                "records": report_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote calibration: {output_path}")
    print(f"Wrote validation CSV: {report_csv_path}")
    print(f"Wrote validation JSON: {report_json_path}")


if __name__ == "__main__":
    main()
