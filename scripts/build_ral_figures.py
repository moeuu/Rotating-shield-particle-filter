"""Build the current RA-L concept and completed-run diagnostic figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.optimize import linear_sum_assignment

try:
    from scripts.ral_figure_common import (
        EXPERIMENT_FIG_PATH,
        FIG1_PATH,
        FIG2_PATH,
        FIG_LABEL_SIZE,
        FIG_TICK_SIZE,
        FIG_TITLE_SIZE,
        ISAAC_DETECTOR_RENDER,
        ISAAC_PROBLEM_RENDER,
        ISOTOPE_COLORS,
        REVIEW_DIR,
        read_json,
        save_figure,
        write_review_images,
    )
except ModuleNotFoundError:
    from ral_figure_common import (
        EXPERIMENT_FIG_PATH,
        FIG1_PATH,
        FIG2_PATH,
        FIG_LABEL_SIZE,
        FIG_TICK_SIZE,
        FIG_TITLE_SIZE,
        ISAAC_DETECTOR_RENDER,
        ISAAC_PROBLEM_RENDER,
        ISOTOPE_COLORS,
        REVIEW_DIR,
        read_json,
        save_figure,
        write_review_images,
    )


POSITION_THRESHOLD_M = 0.5
STRENGTH_THRESHOLD_FRACTION = 0.25
HARD_CAP = 8
HARD_CAP_MASS_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Represent one truth source or one posterior mode."""

    isotope: str
    index: int
    position_xyz: np.ndarray
    strength_cps_1m: float


@dataclass(frozen=True, slots=True)
class SourceMatch:
    """Represent one isotope-preserving one-to-one diagnostic match."""

    truth: SourceRecord
    estimate: SourceRecord
    position_error_m: float
    relative_strength_error: float


@dataclass(frozen=True, slots=True)
class CompletedRunBundle:
    """Contain the verified data needed for one completed-run figure."""

    root: Path
    run_id: str
    estimator_commit: str
    predecessor_code: bool
    room_xyz_m: tuple[float, float, float]
    environment: dict[str, Any]
    station_positions_xyz: np.ndarray
    pair_ids: np.ndarray
    live_time_s: float
    truth_sources: tuple[SourceRecord, ...]
    estimated_sources: tuple[SourceRecord, ...]
    matches: tuple[SourceMatch, ...]
    posterior_support: dict[str, np.ndarray]
    station_indices: np.ndarray
    map_cardinality: dict[str, np.ndarray]
    hard_cap_mass: dict[str, np.ndarray]


def _as_position(value: object, *, name: str) -> np.ndarray:
    """Return one finite 3-D position."""
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or np.any(~np.isfinite(position)):
        raise ValueError(f"{name} must be one finite 3-D position.")
    return position


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite positive floating-point value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    """Load a nonempty JSONL artifact as objects."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object.")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} must contain at least one JSON object.")
    return rows


def _source_records(
    truth_payload: dict[str, Any],
    posterior_payload: dict[str, Any],
) -> tuple[tuple[SourceRecord, ...], tuple[SourceRecord, ...]]:
    """Parse truth sources and posterior modes with stable isotope indices."""
    raw_truth = truth_payload.get("sources")
    if not isinstance(raw_truth, list) or not raw_truth:
        raise ValueError("Truth manifest must contain a nonempty sources list.")
    truth_counts: dict[str, int] = {}
    truth_sources: list[SourceRecord] = []
    for raw in raw_truth:
        if not isinstance(raw, dict):
            raise TypeError("Every truth source must be a JSON object.")
        isotope = str(raw.get("isotope", ""))
        if not isotope:
            raise ValueError("Every truth source must declare an isotope.")
        truth_counts[isotope] = truth_counts.get(isotope, 0) + 1
        truth_sources.append(
            SourceRecord(
                isotope=isotope,
                index=truth_counts[isotope],
                position_xyz=_as_position(
                    raw.get("position"),
                    name=f"truth source {isotope} position",
                ),
                strength_cps_1m=_positive_float(
                    raw.get("intensity_cps_1m"),
                    name=f"truth source {isotope} strength",
                ),
            )
        )

    isotope_payload = posterior_payload.get("isotopes")
    if not isinstance(isotope_payload, dict) or not isotope_payload:
        raise ValueError("PF posterior must contain isotope reports.")
    estimated_sources: list[SourceRecord] = []
    for isotope in sorted(isotope_payload):
        report = isotope_payload[isotope]
        if not isinstance(report, dict):
            raise TypeError("Every PF isotope report must be a JSON object.")
        modes = report.get("modes")
        if not isinstance(modes, list):
            raise TypeError(f"PF modes for {isotope} must be a list.")
        for mode_index, mode in enumerate(modes, start=1):
            if not isinstance(mode, dict):
                raise TypeError("Every posterior mode must be a JSON object.")
            label_index = mode.get("label_index", mode_index - 1)
            if isinstance(label_index, bool) or not isinstance(label_index, int):
                raise TypeError("Posterior label_index must be an integer.")
            estimated_sources.append(
                SourceRecord(
                    isotope=str(isotope),
                    index=int(label_index) + 1,
                    position_xyz=_as_position(
                        mode.get("position_medoid_xyz"),
                        name=f"posterior mode {isotope} position",
                    ),
                    strength_cps_1m=_positive_float(
                        mode.get("strength_representative_cps_1m"),
                        name=f"posterior mode {isotope} strength",
                    ),
                )
            )
    return tuple(truth_sources), tuple(estimated_sources)


def _match_sources(
    truth_sources: tuple[SourceRecord, ...],
    estimated_sources: tuple[SourceRecord, ...],
) -> tuple[SourceMatch, ...]:
    """Match truth to modes by isotope and minimum total 3-D distance."""
    matches: list[SourceMatch] = []
    isotopes = sorted({source.isotope for source in truth_sources})
    for isotope in isotopes:
        truths = [source for source in truth_sources if source.isotope == isotope]
        estimates = [
            source for source in estimated_sources if source.isotope == isotope
        ]
        if len(estimates) < len(truths):
            raise ValueError(
                f"Diagnostic matching requires at least one mode per {isotope} truth."
            )
        truth_xyz = np.stack([source.position_xyz for source in truths])
        estimate_xyz = np.stack([source.position_xyz for source in estimates])
        distances = np.linalg.norm(
            truth_xyz[:, None, :] - estimate_xyz[None, :, :],
            axis=2,
        )
        truth_indices, estimate_indices = linear_sum_assignment(distances)
        for truth_index, estimate_index in zip(
            truth_indices.tolist(),
            estimate_indices.tolist(),
        ):
            truth = truths[truth_index]
            estimate = estimates[estimate_index]
            matches.append(
                SourceMatch(
                    truth=truth,
                    estimate=estimate,
                    position_error_m=float(distances[truth_index, estimate_index]),
                    relative_strength_error=abs(
                        estimate.strength_cps_1m - truth.strength_cps_1m
                    )
                    / truth.strength_cps_1m,
                )
            )
    return tuple(sorted(matches, key=lambda match: (match.truth.isotope, match.truth.index)))


def _posterior_support(path: Path, *, sample_count: int = 192) -> dict[str, np.ndarray]:
    """Return a deterministic weighted particle sample for visual context."""
    with np.load(path, allow_pickle=False) as payload:
        names = tuple(str(value) for value in payload["isotope_names"].tolist())
        weights = np.asarray(payload["weights_n"], dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0 or np.any(weights < 0.0):
            raise ValueError("PF particle weights are invalid.")
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("PF particle weights must have positive finite mass.")
        normalized = weights / total
        cdf = np.cumsum(normalized)
        targets = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
        row_indices = np.searchsorted(cdf, targets, side="left")
        support: dict[str, np.ndarray] = {}
        for isotope_index, isotope in enumerate(names):
            prefix = f"isotope_{isotope_index:03d}"
            positions = np.asarray(payload[f"{prefix}_positions_nk3"], dtype=np.float64)
            mask = np.asarray(payload[f"{prefix}_source_mask_nk"], dtype=bool)
            if positions.shape[:2] != mask.shape or positions.shape[2:] != (3,):
                raise ValueError("PF particle positions and masks are misaligned.")
            sampled_positions = positions[row_indices]
            sampled_mask = mask[row_indices]
            active = sampled_positions[sampled_mask]
            if active.size and np.any(~np.isfinite(active)):
                raise ValueError("PF particle support contains nonfinite positions.")
            support[isotope] = active.reshape((-1, 3))
        return support


def _cardinality_trace(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract MAP cardinality and hard-cap mass from the durable station trace."""
    station_indices = np.asarray([int(row["station_id"]) + 1 for row in rows])
    if not np.array_equal(station_indices, np.arange(1, len(rows) + 1)):
        raise ValueError("PF station trace must be ordered and contiguous.")
    first = rows[0].get("posterior_snapshot", {}).get("isotopes", {})
    if not isinstance(first, dict) or not first:
        raise ValueError("PF station trace lacks isotope posterior snapshots.")
    map_cardinality: dict[str, np.ndarray] = {}
    hard_cap_mass: dict[str, np.ndarray] = {}
    for isotope in sorted(first):
        map_values: list[int] = []
        cap_values: list[float] = []
        for row in rows:
            report = row.get("posterior_snapshot", {}).get("isotopes", {}).get(isotope)
            if not isinstance(report, dict):
                raise ValueError(f"Station trace lacks {isotope} posterior data.")
            map_values.append(int(report["map_cardinality"]))
            distribution = report.get("cardinality_distribution", {})
            if not isinstance(distribution, dict):
                raise TypeError("Cardinality distribution must be an object.")
            cap_values.append(float(distribution.get(str(HARD_CAP), 0.0)))
        map_cardinality[isotope] = np.asarray(map_values, dtype=np.int64)
        hard_cap_mass[isotope] = np.asarray(cap_values, dtype=np.float64)
    return station_indices, map_cardinality, hard_cap_mass


def load_completed_run(run_dir: Path) -> CompletedRunBundle:
    """Load and cross-check one durable completed full-simulation bundle."""
    root = Path(run_dir).expanduser().resolve()
    result = read_json(root / "pf_output" / "closed_loop_result.json")
    truth = read_json(root / "truth_manifest.json")
    environment = read_json(root / "measurement_log" / "environment.json")
    posterior = read_json(root / "pf_output" / "pf_posterior.json")
    if result.get("status") != "complete":
        raise ValueError("The requested full-simulation result is not complete.")
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Completed result must contain one nonempty run_id.")
    if truth.get("run_id") != run_id:
        raise ValueError("Completed result and truth manifest run_id values differ.")

    with np.load(root / "measurement_log" / "observations.npz", allow_pickle=False) as obs:
        station_ids = np.asarray(obs["station_id"], dtype=np.int64)
        poses = np.asarray(obs["detector_pose_xyz"], dtype=np.float64)
        fe = np.asarray(obs["fe_orientation_index"], dtype=np.int64)
        pb = np.asarray(obs["pb_orientation_index"], dtype=np.int64)
        live_times = np.asarray(obs["live_time_s"], dtype=np.float64)
    record_count = int(result.get("record_count", -1))
    station_count = int(result.get("station_count", -1))
    if station_ids.shape != (record_count,) or poses.shape != (record_count, 3):
        raise ValueError("Observation rows differ from closed-loop record_count.")
    expected_stations = np.arange(station_count, dtype=np.int64)
    if not np.array_equal(np.unique(station_ids), expected_stations):
        raise ValueError("Observation station IDs differ from closed-loop station_count.")
    if np.any(~np.isfinite(poses)) or np.any((fe < 0) | (fe >= 8)) or np.any(
        (pb < 0) | (pb >= 8)
    ):
        raise ValueError("Observation pose or Fe/Pb orientation data are invalid.")
    if live_times.shape != (record_count,) or not np.allclose(
        live_times,
        live_times[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Completed-run figure requires one fixed live time per view.")
    first_rows = np.asarray(
        [int(np.flatnonzero(station_ids == station)[0]) for station in expected_stations]
    )
    station_positions = poses[first_rows]

    truth_sources, estimated_sources = _source_records(truth, posterior)
    matches = _match_sources(truth_sources, estimated_sources)
    trace_rows = _load_json_lines(root / "pf_output" / "pf_station_trace.jsonl")
    if len(trace_rows) != station_count:
        raise ValueError("PF station trace length differs from completed station_count.")
    station_indices, map_cardinality, hard_cap_mass = _cardinality_trace(trace_rows)
    provenance = posterior.get("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("Posterior provenance must be a JSON object.")
    estimator_commit = str(provenance.get("estimator_commit", "unknown"))
    room_xyz = tuple(
        _positive_float(environment.get(field), name=field)
        for field in ("size_x", "size_y", "size_z")
    )
    return CompletedRunBundle(
        root=root,
        run_id=run_id,
        estimator_commit=estimator_commit,
        predecessor_code=True,
        room_xyz_m=room_xyz,
        environment=environment,
        station_positions_xyz=station_positions,
        pair_ids=fe * 8 + pb,
        live_time_s=float(live_times[0]),
        truth_sources=truth_sources,
        estimated_sources=estimated_sources,
        matches=matches,
        posterior_support=_posterior_support(root / "pf_output" / "pf_particles.npz"),
        station_indices=station_indices,
        map_cardinality=map_cardinality,
        hard_cap_mass=hard_cap_mass,
    )


def completed_run_metrics(bundle: CompletedRunBundle) -> dict[str, object]:
    """Return the fixed nearest-mode diagnostic metrics for manuscript text."""
    position_passes = [
        match.position_error_m <= POSITION_THRESHOLD_M for match in bundle.matches
    ]
    joint_passes = [
        position_pass
        and match.relative_strength_error <= STRENGTH_THRESHOLD_FRACTION
        for position_pass, match in zip(position_passes, bundle.matches)
    ]
    final_cap_mass = {
        isotope: float(values[-1]) for isotope, values in bundle.hard_cap_mass.items()
    }
    return {
        "schema_version": 1,
        "evidence_status": "completed_predecessor_code_diagnostic",
        "run_id": bundle.run_id,
        "source_count": len(bundle.matches),
        "position_pass_count": int(sum(position_passes)),
        "joint_position_strength_pass_count": int(sum(joint_passes)),
        "position_threshold_m": POSITION_THRESHOLD_M,
        "strength_threshold_fraction": STRENGTH_THRESHOLD_FRACTION,
        "final_hard_cap_mass": final_cap_mass,
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one provenance input or output file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    """Build one path, size, and digest record for a generated artifact."""
    resolved = Path(path).expanduser().resolve()
    return {
        "path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def write_figure_provenance(
    generated: list[Path],
    output_path: Path,
    *,
    completed_run_dir: Path | None,
) -> Path:
    """Write a machine-readable source and transformation manifest."""
    inputs = [ISAAC_PROBLEM_RENDER, ISAAC_DETECTOR_RENDER]
    bundle: CompletedRunBundle | None = None
    if completed_run_dir is not None:
        bundle = load_completed_run(completed_run_dir)
        inputs.extend(
            (
                bundle.root / "truth_manifest.json",
                bundle.root / "measurement_log" / "environment.json",
                bundle.root / "measurement_log" / "observations.npz",
                bundle.root / "pf_output" / "closed_loop_result.json",
                bundle.root / "pf_output" / "pf_posterior.json",
                bundle.root / "pf_output" / "pf_particles.npz",
                bundle.root / "pf_output" / "pf_station_trace.jsonl",
            )
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "IEEE Robotics and Automation Letters, initial submission",
        "source_files": [_artifact_record(path) for path in inputs],
        "outputs": [_artifact_record(path) for path in generated],
        "transformations": {
            "concept_figures": (
                "Direct raster crops plus vector annotations; no synthetic "
                "measurement or response data."
            ),
            "completed_run_figure": (
                "Isotope-preserving Hungarian nearest-mode matching in 3-D; "
                "systematic posterior-particle resampling for visual support; "
                "errors normalized only for display by fixed 0.5 m and 25% "
                "acceptance thresholds."
            ),
            "randomness": "none",
        },
    }
    if bundle is not None:
        payload["completed_run"] = completed_run_metrics(bundle)
        payload["completed_run"]["estimator_commit"] = bundle.estimator_commit
    resolved_output = Path(output_path).expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_suffix(resolved_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved_output)
    return resolved_output


def _draw_image(ax: Axes, path: Path, *, title: str) -> None:
    """Draw one cropped scientific-render panel with a readable title."""
    image = plt.imread(Path(path).as_posix())
    height, width = image.shape[:2]
    crop = image[int(0.06 * height) : int(0.92 * height), int(0.05 * width) : int(0.95 * width)]
    ax.imshow(crop)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.90, "pad": 1.5},
    )


def _draw_pair_alphabet(ax: Axes) -> None:
    """Draw the complete 64-pair alphabet with one example eight-pair code."""
    selected = {(0, 0), (0, 4), (2, 1), (2, 5), (4, 2), (4, 6), (6, 3), (6, 7)}
    values = np.zeros((8, 8), dtype=np.float64)
    for fe_index, pb_index in selected:
        values[fe_index, pb_index] = 1.0
    ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="none")
    ax.set_xticks(np.arange(8), labels=[str(value) for value in range(8)])
    ax.set_yticks(np.arange(8), labels=[str(value) for value in range(8)])
    ax.set_xlabel("Pb octant orientation", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("Fe octant orientation", fontsize=FIG_LABEL_SIZE)
    ax.tick_params(labelsize=FIG_TICK_SIZE, length=0)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="#c7c7c7", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(
        0.5,
        0.98,
        "(c) One 8-of-64 physical code",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.90, "pad": 1.2},
    )


def render_problem_setting(output_path: Path = FIG1_PATH) -> Path:
    """Render the physical task and Fe/Pb attenuation-code alphabet."""
    fig = plt.figure(figsize=(7.15, 2.10))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.05, 1.12), wspace=0.16)
    _draw_image(
        fig.add_subplot(grid[0, 0]),
        ISAAC_PROBLEM_RENDER,
        title="(a) Surface-source search",
    )
    _draw_image(
        fig.add_subplot(grid[0, 1]),
        ISAAC_DETECTOR_RENDER,
        title="(b) CeBr$_3$ + Fe/Pb octants",
    )
    _draw_pair_alphabet(fig.add_subplot(grid[0, 2]))
    fig.text(
        0.5,
        0.01,
        "Robot motion changes geometry; the selected Fe/Pb pair sequence adds a controlled attenuation code.",
        ha="center",
        va="bottom",
        fontsize=FIG_LABEL_SIZE,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.18)
    return save_figure(fig, output_path)


def _flow_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    *,
    facecolor: str,
) -> None:
    """Draw one readable method-flow box."""
    x_value, y_value = xy
    ax.add_patch(
        FancyBboxPatch(
            (x_value, y_value),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor="#333333",
            linewidth=0.8,
        )
    )
    ax.text(
        x_value + 0.12,
        y_value + height - 0.16,
        title,
        ha="left",
        va="top",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
    )
    ax.text(
        x_value + 0.12,
        y_value + height - 0.50,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=FIG_LABEL_SIZE,
        linespacing=1.16,
    )


def _flow_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
) -> None:
    """Draw one method-flow arrow and optional label."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#353535",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0 + 0.10,
            label,
            ha="center",
            va="bottom",
            fontsize=FIG_TICK_SIZE,
            bbox={"fc": "white", "ec": "none", "alpha": 0.92, "pad": 0.8},
        )


def render_method_overview(output_path: Path = FIG2_PATH) -> Path:
    """Render the coupled code-design, inference, and exact-history loop."""
    fig, ax = plt.subplots(figsize=(7.15, 2.82))
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 4.7)
    ax.axis("off")

    box_width = 2.42
    box_height = 1.30
    top_y = 3.12
    bottom_y = 0.45
    top_x = (0.20, 3.28, 6.36, 9.44)
    bottom_x = (9.44, 6.36, 3.28, 0.20)
    _flow_box(
        ax,
        (top_x[0], top_y),
        box_width,
        box_height,
        "Joint posterior",
        (r"state: $\{K_i,\mathbf{s}_{ij},a_{ij}\}_i$", "shared weight + ancestry"),
        facecolor="#eaf2fb",
    )
    _flow_box(
        ax,
        (top_x[1], top_y),
        box_width,
        box_height,
        "64 pair views",
        ("all Fe/Pb orientations", "same spectral model"),
        facecolor="#edf7ed",
    )
    _flow_box(
        ax,
        (top_x[2], top_y),
        box_width,
        box_height,
        "Design 8-view code",
        ("conditional greedy", "448 one-swap checks"),
        facecolor="#fff4df",
    )
    _flow_box(
        ax,
        (top_x[3], top_y),
        box_width,
        box_height,
        "Acquire station",
        ("one robot pose", "8 spectra × 20 s"),
        facecolor="#fcebec",
    )
    _flow_box(
        ax,
        (bottom_x[0], bottom_y),
        box_width,
        box_height,
        "Full-station SMC",
        (r"joint $\beta:0\rightarrow1$", "one station target"),
        facecolor="#fcebec",
    )
    _flow_box(
        ax,
        (bottom_x[1], bottom_y),
        box_width,
        box_height,
        "Shield-aware RJ",
        ("birth, death, merge", "multiscale pose + rate"),
        facecolor="#fff4df",
    )
    _flow_box(
        ax,
        (bottom_x[2], bottom_y),
        box_width,
        box_height,
        "TPHT scheduler",
        ("recent factor first", "survivor exact replay"),
        facecolor="#edf7ed",
    )
    _flow_box(
        ax,
        (bottom_x[3], bottom_y),
        box_width,
        box_height,
        "Updated posterior",
        ("unknown isotope-wise $K$", "surface pose + rate"),
        facecolor="#eaf2fb",
    )

    for left in top_x[:-1]:
        _flow_arrow(
            ax,
            (left + box_width, top_y + box_height / 2.0),
            (left + 3.08, top_y + box_height / 2.0),
        )
    _flow_arrow(
        ax,
        (top_x[-1] + box_width / 2.0, top_y),
        (bottom_x[0] + box_width / 2.0, bottom_y + box_height),
        label="shield-conditioned likelihood",
    )
    for right in bottom_x[:-1]:
        _flow_arrow(
            ax,
            (right, bottom_y + box_height / 2.0),
            (right - 0.66, bottom_y + box_height / 2.0),
        )
    _flow_arrow(
        ax,
        (bottom_x[-1] + box_width / 2.0, bottom_y + box_height),
        (top_x[0] + box_width / 2.0, top_y),
        label="posterior-adaptive redesign",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    return save_figure(fig, output_path)


def _isotope_short_name(isotope: str) -> str:
    """Return a compact source label prefix."""
    return {"Cs-137": "Cs", "Co-60": "Co", "Eu-154": "Eu"}.get(
        isotope,
        isotope,
    )


def _truth_marker(isotope: str) -> str:
    """Return a color-independent marker for one truth isotope."""
    return {"Cs-137": "*", "Co-60": "P", "Eu-154": "^"}.get(isotope, "o")


def _projection(position: np.ndarray, projection: str) -> tuple[float, float]:
    """Project one 3-D point into the requested metric plane."""
    if projection == "xy":
        return float(position[0]), float(position[1])
    if projection == "yz":
        return float(position[1]), float(position[2])
    raise ValueError(f"Unsupported projection {projection!r}.")


def _draw_obstacles(ax: Axes, bundle: CompletedRunBundle, projection: str) -> None:
    """Draw the authenticated obstacle geometry in one metric projection."""
    obstacle_grid = bundle.environment.get("obstacle_grid", {})
    if not isinstance(obstacle_grid, dict):
        return
    if projection == "xy":
        cell_size = float(obstacle_grid.get("cell_size", 1.0))
        origin = obstacle_grid.get("origin", [0.0, 0.0])
        if not isinstance(origin, list) or len(origin) < 2:
            origin = [0.0, 0.0]
        for cell in obstacle_grid.get("blocked_cells", []):
            if not isinstance(cell, list) or len(cell) != 2:
                continue
            ax.add_patch(
                Rectangle(
                    (
                        float(origin[0]) + int(cell[0]) * cell_size,
                        float(origin[1]) + int(cell[1]) * cell_size,
                    ),
                    cell_size,
                    cell_size,
                    facecolor="#b8bdc3",
                    edgecolor="#777d84",
                    linewidth=0.25,
                    alpha=0.62,
                    zorder=0,
                )
            )
        return
    boxes = obstacle_grid.get("transport_boxes_m", [])
    seen: set[tuple[float, float, float, float]] = set()
    for raw in boxes:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (6,) or np.any(~np.isfinite(values)):
            continue
        rectangle = (
            round(float(values[1]), 3),
            round(float(values[2]), 3),
            round(float(values[4] - values[1]), 3),
            round(float(values[5] - values[2]), 3),
        )
        if rectangle in seen or rectangle[2] <= 0.0 or rectangle[3] <= 0.0:
            continue
        seen.add(rectangle)
        ax.add_patch(
            Rectangle(
                rectangle[:2],
                rectangle[2],
                rectangle[3],
                facecolor="#b8bdc3",
                edgecolor="#777d84",
                linewidth=0.15,
                alpha=0.12,
                zorder=0,
            )
        )


def _plot_projection(
    ax: Axes,
    bundle: CompletedRunBundle,
    *,
    projection: str,
    title: str,
) -> None:
    """Plot truth, modes, posterior support, stations, and authenticated obstacles."""
    room_x, room_y, room_z = bundle.room_xyz_m
    limits = (room_x, room_y) if projection == "xy" else (room_y, room_z)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            limits[0],
            limits[1],
            facecolor="#fbfbfb",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=-2,
        )
    )
    _draw_obstacles(ax, bundle, projection)
    for isotope, support in bundle.posterior_support.items():
        if support.size == 0:
            continue
        projected = np.asarray([_projection(position, projection) for position in support])
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            s=3.0,
            color=ISOTOPE_COLORS.get(isotope, "#666666"),
            alpha=0.055,
            linewidths=0.0,
            zorder=1,
        )
    station_projection = np.asarray(
        [_projection(position, projection) for position in bundle.station_positions_xyz]
    )
    ax.scatter(
        station_projection[:, 0],
        station_projection[:, 1],
        s=11,
        marker="o",
        facecolor="#222222",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.72,
        zorder=3,
    )

    matched_estimate_ids = {
        (match.estimate.isotope, match.estimate.index): match.truth
        for match in bundle.matches
    }
    for source in bundle.truth_sources:
        x_value, y_value = _projection(source.position_xyz, projection)
        color = ISOTOPE_COLORS.get(source.isotope, "#555555")
        ax.scatter(
            x_value,
            y_value,
            marker=_truth_marker(source.isotope),
            s=74 if source.isotope == "Cs-137" else 48,
            facecolor=color,
            edgecolor="#111111",
            linewidth=0.55,
            zorder=7,
        )
        x_inward = x_value > 0.86 * limits[0]
        y_inward = y_value > 0.86 * limits[1]
        ax.annotate(
            str(source.index),
            xy=(x_value, y_value),
            xytext=(-3 if x_inward else 3, -3 if y_inward else 3),
            textcoords="offset points",
            ha="right" if x_inward else "left",
            va="top" if y_inward else "bottom",
            fontsize=FIG_TICK_SIZE,
            color=color,
            fontweight="bold",
            bbox={"fc": "white", "ec": "none", "alpha": 0.72, "pad": 0.2},
            zorder=8,
        )
    for estimate in bundle.estimated_sources:
        x_value, y_value = _projection(estimate.position_xyz, projection)
        color = ISOTOPE_COLORS.get(estimate.isotope, "#555555")
        matched_truth = matched_estimate_ids.get((estimate.isotope, estimate.index))
        marker = "X" if matched_truth is not None else "D"
        ax.scatter(
            x_value,
            y_value,
            marker=marker,
            s=38 if matched_truth is not None else 24,
            facecolor="none" if matched_truth is None else color,
            edgecolor=color,
            linewidth=1.0,
            zorder=6,
        )
    ax.set_xlim(-0.15, limits[0] + 0.25)
    ax.set_ylim(-0.15, limits[1] + (0.60 if projection == "xy" else 0.35))
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0.0, limits[0] + 0.1, 2.0))
    ax.set_yticks(np.arange(0.0, limits[1] + 0.1, 2.0))
    ax.grid(True, linewidth=0.25, alpha=0.34)
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    ax.set_xlabel("x [m]" if projection == "xy" else "y [m]", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("y [m]" if projection == "xy" else "z [m]", fontsize=FIG_LABEL_SIZE)
    ax.set_title(title, fontsize=FIG_TITLE_SIZE, fontweight="bold", pad=3)


def _plot_cardinality(ax: Axes, bundle: CompletedRunBundle) -> None:
    """Plot online cardinality evolution and hard-cap warning evidence."""
    for isotope in sorted(bundle.map_cardinality):
        color = ISOTOPE_COLORS.get(isotope, "#666666")
        values = bundle.map_cardinality[isotope]
        ax.step(
            bundle.station_indices,
            values,
            where="post",
            color=color,
            linewidth=1.55,
            marker="o",
            markersize=2.8,
            label=f"{isotope} MAP $K$",
        )
        cap = bundle.hard_cap_mass[isotope]
        warning = cap > HARD_CAP_MASS_THRESHOLD
        if np.any(warning):
            ax.scatter(
                bundle.station_indices[warning],
                np.full(np.count_nonzero(warning), HARD_CAP),
                marker="^",
                s=28,
                facecolor=color,
                edgecolor="#111111",
                linewidth=0.35,
                zorder=4,
            )
    ax.axhline(HARD_CAP, color="#333333", linestyle="--", linewidth=0.8)
    ax.text(
        1.1,
        HARD_CAP - 0.25,
        "hard capacity",
        ha="left",
        va="top",
        fontsize=FIG_TICK_SIZE,
    )
    ax.set_xlim(1, int(bundle.station_indices[-1]))
    ax.set_ylim(0, HARD_CAP + 0.55)
    ax.set_xticks([1, 4, 8, 12, int(bundle.station_indices[-1])])
    ax.set_yticks(np.arange(0, HARD_CAP + 1, 2))
    ax.set_xlabel("completed station", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("MAP cardinality", fontsize=FIG_LABEL_SIZE, labelpad=1.0)
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.legend(fontsize=FIG_TICK_SIZE, loc="lower right", framealpha=0.94)
    ax.set_title("(d) Online structural diagnostic", fontsize=FIG_TITLE_SIZE, fontweight="bold")


def _plot_truth_coordinates(ax: Axes, bundle: CompletedRunBundle) -> None:
    """Draw a compact coordinate key linked to truth labels in both projections."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        0.98,
        "(c) Truth coordinates [m]",
        ha="left",
        va="top",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
    )
    ax.text(
        0.25,
        0.86,
        "ID       x      y      z",
        ha="left",
        va="top",
        fontsize=FIG_TICK_SIZE,
        family="monospace",
        fontweight="bold",
    )
    for row_index, source in enumerate(bundle.truth_sources):
        y_value = 0.76 - 0.105 * row_index
        color = ISOTOPE_COLORS.get(source.isotope, "#555555")
        ax.scatter(
            0.05,
            y_value + 0.014,
            marker=_truth_marker(source.isotope),
            s=45 if source.isotope == "Cs-137" else 34,
            facecolor=color,
            edgecolor="#111111",
            linewidth=0.4,
        )
        label = f"{_isotope_short_name(source.isotope)}{source.index}"
        x_pos, y_pos, z_pos = source.position_xyz
        ax.text(
            0.25,
            y_value,
            f"{label:<4} {x_pos:4.1f}  {y_pos:4.1f}  {z_pos:4.1f}",
            ha="left",
            va="center",
            fontsize=FIG_TICK_SIZE,
            family="monospace",
            color=color,
        )


def _plot_errors(ax: Axes, bundle: CompletedRunBundle) -> None:
    """Plot errors normalized by their prespecified acceptance thresholds."""
    matches = bundle.matches
    x_values = np.arange(len(matches), dtype=np.float64)
    labels = [
        f"{_isotope_short_name(match.truth.isotope)}{match.truth.index}"
        for match in matches
    ]
    colors = [ISOTOPE_COLORS.get(match.truth.isotope, "#666666") for match in matches]
    position = np.asarray(
        [match.position_error_m / POSITION_THRESHOLD_M for match in matches]
    )
    strength = np.asarray(
        [
            match.relative_strength_error / STRENGTH_THRESHOLD_FRACTION
            for match in matches
        ]
    )
    ax.bar(
        x_values,
        position,
        color=colors,
        alpha=0.70,
        width=0.66,
        label="position / 0.5 m",
    )
    ax.axhline(
        1.0,
        color="#222222",
        linestyle="--",
        linewidth=0.85,
        label="acceptance threshold",
    )
    ax.plot(
        x_values,
        strength,
        color="#111111",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markersize=4.2,
        linewidth=1.0,
        label="strength / 25%",
    )
    ax.set_ylabel("threshold-normalized error", fontsize=FIG_LABEL_SIZE, labelpad=1.0)
    ax.set_xticks(x_values, labels=labels, rotation=25, ha="right")
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    ax.grid(True, axis="y", linewidth=0.25, alpha=0.35)
    ax.set_ylim(0.0, max(3.4, float(np.max(np.concatenate((position, strength)))) * 1.18))
    metrics = completed_run_metrics(bundle)
    ax.legend(
        fontsize=FIG_TICK_SIZE,
        loc="upper left",
        ncol=1,
        framealpha=0.92,
    )
    ax.set_title(
        "(e) Accuracy: "
        f"{metrics['position_pass_count']}/{metrics['source_count']} position; "
        f"{metrics['joint_position_strength_pass_count']}/{metrics['source_count']} joint",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
    )


def render_completed_run_summary(
    run_dir: Path,
    output_path: Path = EXPERIMENT_FIG_PATH,
) -> Path:
    """Render one auditable result figure from a verified completed run."""
    bundle = load_completed_run(run_dir)
    fig = plt.figure(figsize=(7.15, 4.62))
    grid = fig.add_gridspec(
        2,
        8,
        height_ratios=(1.18, 0.82),
        width_ratios=(1.0,) * 8,
        hspace=0.34,
        wspace=0.46,
    )
    _plot_projection(
        fig.add_subplot(grid[0, 0:2]),
        bundle,
        projection="xy",
        title="(a) Floor projection",
    )
    _plot_projection(
        fig.add_subplot(grid[0, 2:5]),
        bundle,
        projection="yz",
        title="(b) Depth--height projection",
    )
    _plot_truth_coordinates(fig.add_subplot(grid[0, 5:8]), bundle)
    _plot_cardinality(fig.add_subplot(grid[1, 0:4]), bundle)
    _plot_errors(fig.add_subplot(grid[1, 4:8]), bundle)

    legend = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=6,
            label="matched PF mode",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#666666",
            markersize=5,
            label="unmatched PF mode",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markersize=5,
            label="measurement station",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        fontsize=FIG_TICK_SIZE,
        frameon=True,
        borderpad=0.25,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.text(
        0.5,
        0.055,
        "Completed predecessor-code diagnostic; nearest-mode matching is descriptive, not current four-variant evidence.",
        ha="center",
        va="bottom",
        fontsize=FIG_LABEL_SIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.060, right=0.985, top=0.94, bottom=0.16)
    return save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for deterministic figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-concepts",
        action="store_true",
        help="Do not regenerate the problem and method figures.",
    )
    parser.add_argument(
        "--completed-run-dir",
        type=Path,
        help="Durable completed full-simulation directory for the diagnostic figure.",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Do not regenerate the completed-run diagnostic figure.",
    )
    parser.add_argument(
        "--experiment-output",
        type=Path,
        default=EXPERIMENT_FIG_PATH,
        help="Output PDF for the completed-run diagnostic figure.",
    )
    parser.add_argument(
        "--review-output-dir",
        type=Path,
        default=REVIEW_DIR,
        help="Directory for raster review copies used for visual QA.",
    )
    parser.add_argument(
        "--no-review-images",
        action="store_true",
        help="Do not write raster review copies.",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=REVIEW_DIR / "figure_provenance.json",
        help="Machine-readable source and transformation manifest.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the selected current RA-L figures and their review copies."""
    args = parse_args()
    generated: list[Path] = []
    if not args.skip_concepts:
        generated.extend((render_problem_setting(), render_method_overview()))
    if not args.skip_experiment:
        if args.completed_run_dir is None:
            raise ValueError(
                "--completed-run-dir is required unless --skip-experiment is used."
            )
        generated.append(
            render_completed_run_summary(
                args.completed_run_dir,
                args.experiment_output,
            )
        )
    for output in generated:
        print(f"Wrote {output}")
    if generated and not args.no_review_images:
        for review in write_review_images(generated, args.review_output_dir):
            print(f"Wrote review image {review}")
    if generated:
        provenance = write_figure_provenance(
            generated,
            args.provenance_output,
            completed_run_dir=(
                None if args.skip_experiment else args.completed_run_dir
            ),
        )
        print(f"Wrote figure provenance {provenance}")


if __name__ == "__main__":
    main()
