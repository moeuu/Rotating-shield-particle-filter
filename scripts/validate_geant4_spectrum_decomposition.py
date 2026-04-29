"""Validate Geant4 spectrum decomposition across multi-source cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.model import PointSource
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm
from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import SceneDescription, SourceDescription
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.runtime import load_runtime_config
from spectrum.pipeline import SpectralDecomposer, SpectrumConfig


ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


@dataclass(frozen=True)
class ValidationSource:
    """Describe a source used by a validation case."""

    isotope: str
    position_xyz: tuple[float, float, float]
    intensity_cps_1m: float

    def to_point_source(self) -> PointSource:
        """Convert this source to the measurement-model representation."""
        return PointSource(
            isotope=self.isotope,
            position=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )

    def to_scene_source(self) -> SourceDescription:
        """Convert this source to the Geant4 scene representation."""
        return SourceDescription(
            isotope=self.isotope,
            position_xyz=self.position_xyz,
            intensity_cps_1m=self.intensity_cps_1m,
        )


@dataclass(frozen=True)
class ValidationCase:
    """Describe one Geant4 spectrum-decomposition validation case."""

    name: str
    description: str
    detector_pose_xyz: tuple[float, float, float]
    sources: tuple[ValidationSource, ...]
    fe_index: int = 0
    pb_index: int = 0
    dwell_time_s: float = 30.0
    obstacle_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    include_in_accuracy_summary: bool = True


def default_cases() -> list[ValidationCase]:
    """Return a broad set of Geant4 validation cases."""
    detector = (1.0, 1.0, 0.5)
    near_x = (2.0, 1.0, 0.5)
    near_y = (1.0, 2.0, 0.5)
    near_z = (1.0, 1.0, 1.5)
    far_y = (1.0, 3.0, 0.5)
    blocked_octant = (2.0, 2.0, 1.5)
    free_other_octant = (0.0, 2.0, 1.5)
    return [
        ValidationCase(
            name="single_cs_free",
            description="Single Cs-137 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Cs-137", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_co_free",
            description="Single Co-60 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Co-60", near_x, 100.0),),
        ),
        ValidationCase(
            name="single_eu_free",
            description="Single Eu-154 source at 1 m without shield blocking.",
            detector_pose_xyz=detector,
            sources=(ValidationSource("Eu-154", near_x, 100.0),),
        ),
        ValidationCase(
            name="two_cs_free",
            description="Two Cs-137 sources at different distances.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 70.0),
                ValidationSource("Cs-137", far_y, 120.0),
            ),
        ),
        ValidationCase(
            name="cs_co_free",
            description="Two-isotope mixture with Cs-137 and Co-60.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Co-60", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_eu_free",
            description="Two-isotope mixture with Cs-137 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="co_eu_free",
            description="Two-isotope mixture with Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Co-60", near_x, 80.0),
                ValidationSource("Eu-154", near_y, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_balanced_free",
            description="Balanced three-isotope mixture at three directions.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 60.0),
                ValidationSource("Co-60", near_y, 60.0),
                ValidationSource("Eu-154", near_z, 60.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_imbalanced_free",
            description="Three-isotope mixture with weak Co-60 and Eu-154.",
            detector_pose_xyz=detector,
            sources=(
                ValidationSource("Cs-137", near_x, 120.0),
                ValidationSource("Co-60", near_y, 35.0),
                ValidationSource("Eu-154", near_z, 25.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_fe_blocked",
            description="Three-isotope mixture through the iron shield octant.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Co-60", blocked_octant, 80.0),
                ValidationSource("Eu-154", blocked_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="cs_two_sources_one_fe_blocked",
            description="Two Cs-137 sources with one direction blocked by Fe.",
            detector_pose_xyz=detector,
            fe_index=7,
            pb_index=0,
            sources=(
                ValidationSource("Cs-137", blocked_octant, 80.0),
                ValidationSource("Cs-137", free_other_octant, 80.0),
            ),
        ),
        ValidationCase(
            name="three_isotope_obstacle_stress",
            description=(
                "Three-isotope mixture with a concrete obstacle; excluded from "
                "accuracy summary because the PF target omits obstacle transport."
            ),
            detector_pose_xyz=(1.0, 1.5, 0.5),
            dwell_time_s=20.0,
            obstacle_cells=((2, 1),),
            include_in_accuracy_summary=False,
            sources=(
                ValidationSource("Cs-137", (4.0, 1.5, 0.5), 80.0),
                ValidationSource("Co-60", (4.0, 1.5, 1.5), 80.0),
                ValidationSource("Eu-154", (4.0, 1.5, 0.8), 80.0),
            ),
        ),
    ]


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def build_scene(case: ValidationCase, usd_path: str | None) -> SceneDescription:
    """Build a Geant4 scene for one validation case."""
    return SceneDescription(
        room_size_xyz=(10.0, 20.0, 10.0),
        obstacle_origin_xy=(0.0, 0.0),
        obstacle_cell_size_m=1.0,
        obstacle_grid_shape=(10, 20),
        obstacle_cells=[tuple(cell) for cell in case.obstacle_cells],
        author_obstacle_prims=True,
        sources=[source.to_scene_source() for source in case.sources],
        usd_path=usd_path,
    )


def expected_pf_counts(
    case: ValidationCase,
    kernel: ContinuousKernel,
) -> dict[str, float]:
    """Compute the PF-model target counts for one case."""
    detector = np.asarray(case.detector_pose_xyz, dtype=float)
    counts = {isotope: 0.0 for isotope in ISOTOPES}
    for source in case.sources:
        point_source = source.to_point_source()
        source_pos = point_source.position_array()
        counts[source.isotope] += (
            float(case.dwell_time_s)
            * float(source.intensity_cps_1m)
            * kernel.kernel_value_pair(
                source.isotope,
                detector,
                source_pos,
                int(case.fe_index),
                int(case.pb_index),
            )
        )
    return counts


def source_tally_counts(metadata: dict[str, Any]) -> dict[str, float]:
    """Read native Geant4 source-equivalent tally counts from metadata."""
    return {
        isotope: float(metadata.get(f"source_equivalent_counts_{isotope}", 0.0))
        for isotope in ISOTOPES
    }


def relative_error(value: float, target: float, min_target: float) -> float | None:
    """Return relative error when the target is large enough."""
    if abs(float(target)) < float(min_target):
        return None
    return (float(value) - float(target)) / float(target)


def case_to_dict(case: ValidationCase) -> dict[str, Any]:
    """Return a JSON-compatible representation of a validation case."""
    return {
        "name": case.name,
        "description": case.description,
        "detector_pose_xyz": list(case.detector_pose_xyz),
        "fe_index": int(case.fe_index),
        "pb_index": int(case.pb_index),
        "dwell_time_s": float(case.dwell_time_s),
        "obstacle_cells": [list(cell) for cell in case.obstacle_cells],
        "include_in_accuracy_summary": bool(case.include_in_accuracy_summary),
        "sources": [
            {
                "isotope": source.isotope,
                "position_xyz": list(source.position_xyz),
                "intensity_cps_1m": float(source.intensity_cps_1m),
            }
            for source in case.sources
        ],
    }


def run_case(
    app: Geant4Application,
    decomposer: SpectralDecomposer,
    case: ValidationCase,
    step_id: int,
    runtime_config: dict[str, Any],
    kernel: ContinuousKernel,
    min_target: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one Geant4 validation case and return metrics plus spectrum."""
    scene = build_scene(case, usd_path=runtime_config.get("usd_path"))
    app.reset(scene)
    start = time.time()
    observation = app.step(
        SimulationCommand(
            step_id=step_id,
            target_pose_xyz=case.detector_pose_xyz,
            target_base_yaw_rad=0.0,
            fe_orientation_index=int(case.fe_index),
            pb_orientation_index=int(case.pb_index),
            dwell_time_s=float(case.dwell_time_s),
        )
    )
    runtime_s = time.time() - start
    spectrum = np.asarray(observation.spectrum_counts, dtype=float)
    photopeak_counts = decomposer.compute_photopeak_nnls_counts(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    response_counts = decomposer.compute_response_model_counts(
        spectrum,
        isotopes=ISOTOPES,
    )
    peak_window_counts = decomposer.compute_isotope_counts_thesis(
        spectrum,
        live_time_s=float(case.dwell_time_s),
        isotopes=ISOTOPES,
    )
    target_counts = expected_pf_counts(case, kernel)
    tally_counts = source_tally_counts(dict(observation.metadata))
    methods = {
        "photopeak_nnls": photopeak_counts,
        "response_matrix": response_counts,
        "peak_window": peak_window_counts,
    }
    per_isotope: dict[str, dict[str, Any]] = {}
    for isotope in ISOTOPES:
        target = float(target_counts.get(isotope, 0.0))
        per_isotope[isotope] = {
            "target_pf_counts": target,
            "source_tally_counts": float(tally_counts.get(isotope, 0.0)),
            "method_counts": {
                method: float(values.get(isotope, 0.0))
                for method, values in methods.items()
            },
            "relative_errors": {
                method: relative_error(
                    float(values.get(isotope, 0.0)),
                    target,
                    min_target,
                )
                for method, values in methods.items()
            },
        }
    result = {
        "case": case_to_dict(case),
        "runtime_s": float(runtime_s),
        "total_spectrum_counts": float(np.sum(spectrum)),
        "num_primaries": float(observation.metadata.get("num_primaries", 0.0)),
        "metadata": {
            key: observation.metadata[key]
            for key in sorted(observation.metadata)
            if str(key).startswith(
                (
                    "backend",
                    "engine_mode",
                    "emission_model",
                    "physics_profile",
                    "source_equivalent",
                    "num_primaries",
                    "runtime_s",
                    "thread_count",
                )
            )
        },
        "per_isotope": per_isotope,
    }
    return result, spectrum


def flatten_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested validation results into CSV records."""
    rows: list[dict[str, Any]] = []
    for result in results:
        case = result["case"]
        for isotope, item in result["per_isotope"].items():
            for method, value in item["method_counts"].items():
                rel_err = item["relative_errors"][method]
                rows.append(
                    {
                        "case": case["name"],
                        "description": case["description"],
                        "include_in_accuracy_summary": case["include_in_accuracy_summary"],
                        "isotope": isotope,
                        "method": method,
                        "target_pf_counts": item["target_pf_counts"],
                        "source_tally_counts": item["source_tally_counts"],
                        "estimated_counts": value,
                        "relative_error": "" if rel_err is None else rel_err,
                        "abs_relative_error": "" if rel_err is None else abs(rel_err),
                        "total_spectrum_counts": result["total_spectrum_counts"],
                        "num_primaries": result["num_primaries"],
                        "runtime_s": result["runtime_s"],
                        "fe_index": case["fe_index"],
                        "pb_index": case["pb_index"],
                        "dwell_time_s": case["dwell_time_s"],
                    }
                )
    return rows


def summarize_accuracy(
    results: list[dict[str, Any]],
    min_target: float,
) -> dict[str, dict[str, float]]:
    """Summarize accuracy metrics by decomposition method."""
    values_by_method: dict[str, list[float]] = {
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    false_positive_by_method: dict[str, list[float]] = {
        "photopeak_nnls": [],
        "response_matrix": [],
        "peak_window": [],
    }
    for result in results:
        if not bool(result["case"]["include_in_accuracy_summary"]):
            continue
        for item in result["per_isotope"].values():
            target = float(item["target_pf_counts"])
            for method, estimate in item["method_counts"].items():
                if target >= float(min_target):
                    err = relative_error(float(estimate), target, min_target)
                    if err is not None:
                        values_by_method[method].append(abs(err))
                else:
                    false_positive_by_method[method].append(float(estimate))

    summary: dict[str, dict[str, float]] = {}
    for method, values in values_by_method.items():
        arr = np.asarray(values, dtype=float)
        fp = np.asarray(false_positive_by_method[method], dtype=float)
        summary[method] = {
            "num_accuracy_points": float(arr.size),
            "mean_abs_relative_error": float(np.mean(arr)) if arr.size else float("nan"),
            "median_abs_relative_error": float(np.median(arr)) if arr.size else float("nan"),
            "max_abs_relative_error": float(np.max(arr)) if arr.size else float("nan"),
            "num_absent_isotope_points": float(fp.size),
            "max_absent_isotope_counts": float(np.max(fp)) if fp.size else 0.0,
            "mean_absent_isotope_counts": float(np.mean(fp)) if fp.size else 0.0,
        }
    return summary


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    spectra: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    """Write validation outputs to JSON, CSV, and NPZ files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = flatten_records(results)
    csv_path = output_dir / "records.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(output_dir / "spectra.npz", **spectra)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geant4/high_fidelity_external_no_isaac.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case", action="append", default=None, help="Run only the named case; repeatable.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--min-target-counts", type=float, default=25.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--thread-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the Geant4 spectrum-decomposition validation sweep."""
    args = parse_args()
    config_path = resolve_path(args.config)
    runtime_config = load_runtime_config(config_path.as_posix())
    executable_path = runtime_config.get("executable_path", "build/geant4_sidecar")
    runtime_config["executable_path"] = resolve_path(str(executable_path)).as_posix()
    runtime_config["timeout_s"] = float(args.timeout_s)
    if args.thread_count is not None:
        runtime_config["thread_count"] = int(args.thread_count)
    runtime_config["engine_mode"] = "external"
    runtime_config["physics_profile"] = "balanced"
    runtime_config["scatter_gain"] = 0.0

    all_cases = default_cases()
    if args.case:
        selected = set(args.case)
        cases = [case for case in all_cases if case.name in selected]
        missing = selected.difference({case.name for case in cases})
        if missing:
            raise ValueError(f"Unknown case names: {sorted(missing)}")
    else:
        cases = all_cases
    if args.max_cases is not None:
        cases = cases[: max(int(args.max_cases), 0)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "spectrum_validation" / f"geant4_photopeak_nnls_sweep_{timestamp}"
    )

    mu_by_isotope = mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=ISOTOPES)
    kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope,
        shield_params=ShieldParams(),
        use_gpu=False,
    )
    decomposer = SpectralDecomposer(SpectrumConfig(dead_time_tau_s=0.0))
    results: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    sweep_start = time.time()
    app = Geant4Application(app_config=runtime_config, stage_backend=FakeStageBackend())
    try:
        for step_id, case in enumerate(cases):
            print(f"[{step_id + 1}/{len(cases)}] running {case.name}: {case.description}", flush=True)
            result, spectrum = run_case(
                app,
                decomposer,
                case,
                step_id,
                runtime_config,
                kernel,
                float(args.min_target_counts),
            )
            results.append(result)
            spectra[case.name] = spectrum
            photo = {
                isotope: result["per_isotope"][isotope]["method_counts"]["photopeak_nnls"]
                for isotope in ISOTOPES
            }
            target = {
                isotope: result["per_isotope"][isotope]["target_pf_counts"]
                for isotope in ISOTOPES
            }
            print(
                f"  primaries={result['num_primaries']:.0f} "
                f"runtime={result['runtime_s']:.1f}s target={target} photopeak={photo}",
                flush=True,
            )
    finally:
        app.close()

    summary = {
        "config": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_cases": len(results),
        "elapsed_s": float(time.time() - sweep_start),
        "min_target_counts": float(args.min_target_counts),
        "accuracy_summary": summarize_accuracy(results, float(args.min_target_counts)),
        "cases": [case_to_dict(case) for case in cases],
    }
    write_outputs(output_dir, results, spectra, summary)
    print(json.dumps(summary["accuracy_summary"], indent=2, sort_keys=True))
    print(f"Wrote validation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
