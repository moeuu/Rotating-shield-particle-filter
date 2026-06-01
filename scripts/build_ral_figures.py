"""Build manuscript-ready RA-L figures from schematics and run summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
LATEX_ROOT = ROOT.parent / "latex" / "projects" / "ieee-ra-l-letter"
FIG1_PATH = LATEX_ROOT / "sections/01_introduction/figures/ral_problem_shield_code.pdf"
FIG2_PATH = LATEX_ROOT / "sections/03_system_model/figures/ral_method_loop.pdf"
EXPERIMENT_FIG_PATH = (
    LATEX_ROOT / "sections/05_experiments/figures/ral_result_overview.png"
)
REVIEW_DIR = ROOT / "results" / "ral_figure_review"
ISAAC_FIGURE_DIR = ROOT / "results" / "ral_isaac_figures"
ISAAC_PROBLEM_RENDER = ISAAC_FIGURE_DIR / "capture_problem_setting" / "rgb_0000.png"
ISAAC_DETECTOR_RENDER = ISAAC_FIGURE_DIR / "capture_detector_module" / "rgb_0000.png"
ISAAC_STATION_RENDER = ISAAC_FIGURE_DIR / "capture_simulation_environment" / "rgb_0000.png"
ISAAC_SHIELD_PROGRAM_RENDERS = (
    ISAAC_FIGURE_DIR / "capture_shield_selection_00" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_01" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_02" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_03" / "rgb_0000.png",
)
FIG_TITLE_SIZE = 8.2
FIG_LABEL_SIZE = 7.2
FIG_TICK_SIZE = 7.0
FIG_PANEL_SIZE = 9.0
ISOTOPE_COLORS = {
    "Cs-137": "#d62728",
    "Co-60": "#1f77b4",
    "Eu-154": "#2ca02c",
}


@dataclass(frozen=True)
class SummaryBundle:
    """Parsed result summary and its filesystem context."""

    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class AblationRow:
    """Compact metrics for one ablation variant."""

    label: str
    spectra: int
    true_positive: int
    false_positive: int
    false_negative: int
    mean_position_error_m: float
    mean_strength_error_pct: float


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a matplotlib figure to disk with deterministic layout settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def _write_review_image(figure_path: Path, review_dir: Path) -> Path | None:
    """Write a raster review copy for visual inspection when possible."""
    figure_path = Path(figure_path)
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    output_path = review_dir / f"{figure_path.stem}.png"
    if figure_path.suffix.lower() == ".png":
        shutil.copyfile(figure_path, output_path)
        return output_path
    if figure_path.suffix.lower() != ".pdf":
        return None
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        return None
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-singlefile",
            "-r",
            "220",
            figure_path.as_posix(),
            output_path.with_suffix("").as_posix(),
        ],
        check=True,
    )
    return output_path


def _write_review_images(figure_paths: Iterable[Path], review_dir: Path) -> list[Path]:
    """Write review images for all generated figures."""
    outputs: list[Path] = []
    for figure_path in figure_paths:
        review_image = _write_review_image(figure_path, review_dir)
        if review_image is not None:
            outputs.append(review_image)
    return outputs


def _panel_label(ax: Axes, label: str) -> None:
    """Place a bold panel label outside the data area."""
    if not label:
        return
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FIG_PANEL_SIZE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
        clip_on=False,
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    mutation_scale: float = 10.0,
    lw: float = 1.2,
) -> None:
    """Draw one arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            mutation_scale=mutation_scale,
            lw=lw,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _draw_schematic_obstacles(ax: Axes) -> None:
    """Draw a deterministic obstacle layout for the problem schematic."""
    obstacles = [
        (1.2, 2.0, 1.1, 3.7, 1.6),
        (3.2, 10.0, 1.3, 5.0, 2.2),
        (6.2, 4.1, 1.4, 4.6, 1.2),
        (6.9, 12.5, 1.2, 4.8, 2.0),
        (1.1, 14.5, 1.6, 1.9, 0.9),
    ]
    patches: list[Rectangle] = []
    colors: list[float] = []
    for x0, y0, width, height, obstacle_height in obstacles:
        patches.append(Rectangle((x0, y0), width, height))
        colors.append(obstacle_height)
    collection = PatchCollection(
        patches,
        cmap="Greys",
        edgecolor="#333333",
        linewidth=0.7,
        alpha=0.82,
    )
    collection.set_array(np.asarray(colors))
    collection.set_clim(0.5, 2.5)
    ax.add_collection(collection)


def _draw_problem_scene(ax: Axes, *, compact: bool = False) -> None:
    """Draw a compact multi-isotope surface-source scene for Fig. 1."""
    ax.add_patch(
        Rectangle((0.0, 0.0), 10.0, 20.0, facecolor="#fbfbfb",
                  edgecolor="#222222", lw=0.6, zorder=0)
    )
    _draw_schematic_obstacles(ax)
    route = np.asarray(
        [[1.0, 1.0], [1.3, 3.8], [3.0, 6.2], [4.7, 10.0], [7.6, 15.4]],
        dtype=float,
    )
    ax.plot(route[:, 0], route[:, 1], color="#005bbb", lw=0.9, alpha=0.85, zorder=3)
    ax.scatter(route[:, 0], route[:, 1], s=12, color="#005bbb", zorder=4)
    ax.scatter(route[0, 0], route[0, 1], s=30, marker="s", color="#005bbb", zorder=5)
    sources = [
        ("Cs-137", (2.1, 1.0), "Cs1"),
        ("Cs-137", (7.9, 9.2), "Cs2"),
        ("Co-60", (9.0, 5.0), "Co1"),
        ("Co-60", (8.2, 14.0), "Co2"),
        ("Eu-154", (0.0, 2.4), "Eu"),
    ]
    for isotope, xy, label in sources:
        color = ISOTOPE_COLORS[isotope]
        ax.scatter(xy[0], xy[1], marker="*", s=68, color=color,
                   edgecolor="#222222", lw=0.35, zorder=6, clip_on=False)
        if not compact:
            ax.text(xy[0] + 0.18, xy[1] + 0.18, label, fontsize=5.8,
                    color=color, ha="left", va="bottom", zorder=7)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 20.3)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.25, alpha=0.35)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    else:
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 10, 20])
        ax.tick_params(labelsize=5.8, length=1.5)
        ax.set_xlabel("x [m]", fontsize=6.1, labelpad=0)
        ax.set_ylabel("y [m]", fontsize=6.1, labelpad=0)


def _draw_detector_posture(
    ax: Axes,
    center: tuple[float, float],
    fe_angle: float,
    pb_angle: float,
    label: str,
) -> None:
    """Draw one top-down Fe/Pb shield posture around a detector."""
    cx, cy = center
    ax.add_patch(Circle((cx, cy), 0.16, facecolor="#33c4d8", edgecolor="#1d6f7a", lw=0.8))
    ax.add_patch(Wedge((cx, cy), 0.44, fe_angle, fe_angle + 85, width=0.13,
                       facecolor="#e28b2d", edgecolor="#8a4d00", lw=0.8, alpha=0.95))
    ax.add_patch(Wedge((cx, cy), 0.62, pb_angle, pb_angle + 85, width=0.14,
                       facecolor="#8d96a8", edgecolor="#4c5463", lw=0.8, alpha=0.95))
    ax.text(cx, cy - 0.52, label, ha="center", va="top", fontsize=6.5)
    ax.annotate(
        "",
        xy=(cx + 0.54, cy + 0.20),
        xytext=(cx + 0.24, cy + 0.55),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#444444",
                    "connectionstyle": "arc3,rad=-0.45"},
    )


def _read_image(path: Path) -> np.ndarray:
    """Read an image as an RGB or RGBA array."""
    image = plt.imread(Path(path).as_posix())
    if image.ndim == 2:
        return np.dstack([image, image, image])
    return image


def _draw_image_panel(
    ax: Axes,
    image_path: Path,
    *,
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> None:
    """Draw a cropped raster image panel without axes."""
    image = _read_image(image_path)
    height, width = image.shape[:2]
    left, bottom, right, top = crop
    x0 = int(max(0.0, min(1.0, left)) * width)
    x1 = int(max(0.0, min(1.0, right)) * width)
    y0 = int((1.0 - max(0.0, min(1.0, top))) * height)
    y1 = int((1.0 - max(0.0, min(1.0, bottom))) * height)
    cropped = image[y0:y1, x0:x1]
    ax.imshow(cropped)
    ax.set_axis_off()


def render_problem_setting(output_path: Path = FIG1_PATH) -> Path:
    """Render Fig. 1 for a one-column manuscript width."""
    fig = plt.figure(figsize=(3.45, 4.06))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.32, 1.0), hspace=0.32)

    ax_render = fig.add_subplot(grid[0, 0])
    _draw_image_panel(ax_render, ISAAC_PROBLEM_RENDER, crop=(0.04, 0.05, 0.96, 0.90))
    ax_render.text(
        0.02,
        0.96,
        "(a) 3-D robot survey scene",
        transform=ax_render.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 1.2},
    )
    inset = ax_render.inset_axes([0.60, 0.06, 0.37, 0.52])
    _draw_problem_scene(inset, compact=True)
    inset.text(
        0.03,
        0.97,
        "MIX-9",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        bbox={"fc": "white", "ec": "none", "alpha": 0.76, "pad": 0.6},
    )

    lower = grid[1, 0].subgridspec(1, 2, width_ratios=(0.92, 1.16), wspace=0.44)

    ax_detector = fig.add_subplot(lower[0, 0])
    _draw_image_panel(ax_detector, ISAAC_DETECTOR_RENDER, crop=(0.18, 0.18, 0.84, 0.85))
    ax_detector.text(
        0.03,
        0.95,
        "(b)",
        transform=ax_detector.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.86, "pad": 1.2},
    )
    ax_detector.text(
        0.04,
        0.08,
        "CeBr$_3$\nFe/Pb octants",
        transform=ax_detector.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.9,
        bbox={"fc": "white", "ec": "#777777", "alpha": 0.86, "pad": 1.1},
    )

    ax_sig = fig.add_subplot(lower[0, 1])
    ax_sig.set_title("(c) Same-isotope shield-time signatures", fontsize=7.2, pad=2)
    postures = np.arange(1, 9)
    signatures = np.asarray(
        [
            [1.00, 0.82, 0.43, 0.36, 0.68, 0.94, 0.74, 0.41],
            [0.51, 0.63, 0.88, 1.00, 0.77, 0.46, 0.34, 0.59],
            [0.77, 0.48, 0.35, 0.55, 0.96, 0.79, 0.47, 0.31],
        ]
    )
    colors = ["#d62728", "#8c1d18", "#f28e2b"]
    for index, row in enumerate(signatures):
        ax_sig.plot(
            postures,
            row,
            marker="o",
            ms=2.4,
            lw=0.95,
            color=colors[index],
            label=f"Cs cand. {index + 1}",
        )
    ax_sig.set_xlim(1, 8)
    ax_sig.set_ylim(0.2, 1.08)
    ax_sig.set_xlabel("posture", fontsize=6)
    ax_sig.set_ylabel("normalized count", fontsize=6, labelpad=1)
    ax_sig.tick_params(labelsize=6)
    ax_sig.grid(True, lw=0.35, alpha=0.45)
    ax_sig.legend(fontsize=5.0, loc="upper right", frameon=True, handlelength=1.4)
    return _save_figure(fig, output_path)


def _draw_box(
    ax: Axes,
    xy: tuple[float, float],
    text: str,
    *,
    width: float = 1.52,
    fontsize: float = 7.0,
    height: float = 0.48,
    facecolor: str = "#f6f7f8",
) -> None:
    """Draw a labeled rounded-looking process box using a rectangle."""
    x0, y0 = xy
    ax.add_patch(
        Rectangle((x0, y0), width, height, facecolor=facecolor, edgecolor="#333333", lw=0.8)
    )
    ax.text(x0 + width / 2.0, y0 + height / 2.0, text, ha="center", va="center", fontsize=fontsize)


def render_method_overview(output_path: Path = FIG2_PATH) -> Path:
    """Render Fig. 2 as a 3-D shield view plus station-level inference loop."""
    fig = plt.figure(figsize=(7.15, 3.42))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.90, 1.54), wspace=0.12)

    ax_render = fig.add_subplot(grid[0, 0])
    _draw_image_panel(ax_render, ISAAC_DETECTOR_RENDER, crop=(0.12, 0.12, 0.88, 0.88))
    ax_render.text(
        0.03,
        0.96,
        "(a) 3-D Fe/Pb shielded spectrometer",
        transform=ax_render.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.88, "pad": 1.1},
    )
    ax_render.text(
        0.05,
        0.10,
        "CeBr$_3$ core\nrotating Fe/Pb octants",
        transform=ax_render.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        linespacing=0.94,
        bbox={"fc": "white", "ec": "#777777", "alpha": 0.88, "pad": 1.2},
    )

    right = grid[0, 1].subgridspec(
        3,
        1,
        height_ratios=(1.22, 0.78, 0.95),
        hspace=0.18,
    )

    posture_grid = right[0].subgridspec(1, 4, wspace=0.04)
    for index, image_path in enumerate(ISAAC_SHIELD_PROGRAM_RENDERS, start=1):
        ax_posture = fig.add_subplot(posture_grid[0, index - 1])
        _draw_image_panel(ax_posture, image_path, crop=(0.16, 0.16, 0.86, 0.84))
        ax_posture.text(
            0.06,
            0.92,
            f"$a_{index}$",
            transform=ax_posture.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            bbox={"fc": "white", "ec": "none", "alpha": 0.86, "pad": 0.8},
        )
        if index == 1:
            ax_posture.text(
                0.00,
                1.14,
                "(b) one station $p_i$: rendered shield program",
                transform=ax_posture.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                clip_on=False,
            )

    flow_grid = right[1].subgridspec(1, 3, wspace=0.28)
    ax_spectrum = fig.add_subplot(flow_grid[0, 0])
    ax_spectrum.set_title("shielded spectra", fontsize=6.8, pad=1.5)
    energy = np.linspace(0.0, 1.0, 120)
    spectrum = (
        0.10
        + 0.35 * np.exp(-((energy - 0.24) / 0.055) ** 2)
        + 0.48 * np.exp(-((energy - 0.50) / 0.070) ** 2)
        + 0.28 * np.exp(-((energy - 0.76) / 0.050) ** 2)
    )
    ax_spectrum.plot(energy, spectrum, color="#303030", lw=0.9)
    for center, isotope in zip((0.24, 0.50, 0.76), ("Cs-137", "Co-60", "Eu-154")):
        ax_spectrum.axvspan(center - 0.025, center + 0.025,
                            color=ISOTOPE_COLORS[isotope], alpha=0.22, lw=0)
    ax_spectrum.set_xticks([])
    ax_spectrum.set_yticks([])
    for spine in ax_spectrum.spines.values():
        spine.set_linewidth(0.65)

    ax_pf = fig.add_subplot(flow_grid[0, 1])
    ax_pf.set_title("isotope-wise PF support", fontsize=6.8, pad=1.5)
    ax_pf.add_patch(Rectangle((0.10, 0.10), 0.80, 0.80, facecolor="#fbfbfb",
                              edgecolor="#4e555b", lw=0.55))
    ax_pf.add_patch(Rectangle((0.44, 0.10), 0.10, 0.80, facecolor="#d5d8dc",
                              edgecolor="#8b8f94", lw=0.35))
    particle_points = {
        "Cs-137": ([0.28, 0.37, 0.63, 0.73], [0.28, 0.36, 0.30, 0.35]),
        "Co-60": ([0.28, 0.37, 0.63, 0.73], [0.50, 0.58, 0.51, 0.60]),
        "Eu-154": ([0.28, 0.37, 0.63, 0.73], [0.74, 0.82, 0.73, 0.80]),
    }
    for isotope, (xs, ys) in particle_points.items():
        ax_pf.scatter(xs, ys, s=[10, 26, 13, 22], color=ISOTOPE_COLORS[isotope],
                      alpha=0.72, edgecolor="none")
    ax_pf.set_xlim(0.0, 1.0)
    ax_pf.set_ylim(0.0, 1.0)
    ax_pf.set_xticks([])
    ax_pf.set_yticks([])
    for spine in ax_pf.spines.values():
        spine.set_linewidth(0.65)

    ax_birth = fig.add_subplot(flow_grid[0, 2])
    ax_birth.set_title("residual birth / verify", fontsize=6.8, pad=1.5)
    bars = np.asarray([0.20, 0.48, 0.34, 0.72, 0.26])
    ax_birth.bar(np.arange(len(bars)), bars, color="#d95f5f", width=0.58, alpha=0.82)
    ax_birth.scatter([3.7], [0.74], marker="*", s=58, color="#f5a623",
                     edgecolor="#8a5a00", lw=0.45, zorder=4)
    ax_birth.plot([3.48, 3.60, 3.92], [0.42, 0.30, 0.55],
                  color="#2ca02c", lw=1.2, solid_capstyle="round")
    ax_birth.set_xlim(-0.6, 4.6)
    ax_birth.set_ylim(0.0, 0.95)
    ax_birth.set_xticks([])
    ax_birth.set_yticks([])
    for spine in ax_birth.spines.values():
        spine.set_linewidth(0.65)

    decision_grid = right[2].subgridspec(1, 3, wspace=0.28)
    ax_corr = fig.add_subplot(decision_grid[0, 0])
    ax_corr.set_title("response correlation", fontsize=6.8, pad=1.5)
    corr = np.asarray([[1.0, 0.88, 0.38], [0.88, 1.0, 0.44], [0.38, 0.44, 1.0]])
    ax_corr.imshow(corr, cmap="Reds", vmin=0.0, vmax=1.0)
    for row in range(3):
        for col in range(3):
            ax_corr.text(col, row, f"{corr[row, col]:.1f}", ha="center", va="center",
                         fontsize=5.4, color="#202020")
    ax_corr.set_xticks([0, 1, 2], labels=["m1", "m2", "m3"], fontsize=5.4)
    ax_corr.set_yticks([0, 1, 2], labels=["m1", "m2", "m3"], fontsize=5.4)
    ax_corr.tick_params(length=0, pad=0.5)
    for spine in ax_corr.spines.values():
        spine.set_linewidth(0.65)

    ax_score = fig.add_subplot(decision_grid[0, 1])
    ax_score.set_title("DSS-PP terms", fontsize=6.8, pad=1.5)
    labels = ["E", "S", "O", "Z"]
    scores = [0.46, 0.92, 0.64, 0.58]
    colors = ["#4e79a7", "#d62728", "#7f7f7f", "#59a14f"]
    ax_score.bar(np.arange(4), scores, color=colors, width=0.62, alpha=0.86)
    ax_score.set_ylim(0.0, 1.0)
    ax_score.set_xticks(np.arange(4), labels=labels, fontsize=5.8)
    ax_score.set_yticks([])
    ax_score.axhline(0.80, color="#d62728", lw=0.7, ls="--", alpha=0.55)
    for spine in ax_score.spines.values():
        spine.set_linewidth(0.65)

    ax_next = fig.add_subplot(decision_grid[0, 2])
    ax_next.set_title("next station/program", fontsize=6.8, pad=1.5)
    ax_next.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, facecolor="#fbfbfb",
                                edgecolor="#4e555b", lw=0.55))
    ax_next.add_patch(Rectangle((0.36, 0.10), 0.16, 0.62, facecolor="#c6cbd1",
                                edgecolor="#686e75", lw=0.35))
    ax_next.plot([0.18, 0.42, 0.78], [0.18, 0.36, 0.72],
                 color="#005bbb", lw=1.0, marker="o", ms=2.8)
    ax_next.scatter([0.78], [0.72], marker="*", s=64, color="#005bbb", zorder=5)
    ax_next.text(0.79, 0.20, "$g_{i+1}$", ha="center", va="center", fontsize=6.0,
                 bbox={"fc": "white", "ec": "#777777", "alpha": 0.88, "pad": 0.8})
    ax_next.set_xlim(0.0, 1.0)
    ax_next.set_ylim(0.0, 1.0)
    ax_next.set_xticks([])
    ax_next.set_yticks([])
    for spine in ax_next.spines.values():
        spine.set_linewidth(0.65)

    return _save_figure(fig, output_path)


def _summary_tag(summary_path: Path) -> str:
    """Return the output tag encoded in a result summary filename."""
    name = Path(summary_path).stem
    prefix = "result_summary_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _candidate_manifest_paths(summary: SummaryBundle) -> Iterable[Path]:
    """Yield plausible environment manifest paths for a summary."""
    tag = _summary_tag(summary.path)
    direct_config = Path(str(summary.payload.get("sim_config_path", ""))).expanduser()
    candidates: list[Path] = []
    if direct_config.is_file():
        try:
            config = _read_json(direct_config)
        except json.JSONDecodeError:
            config = {}
        usd_path = Path(str(config.get("usd_path", ""))).expanduser()
        if usd_path.suffix == ".usda":
            candidates.append(usd_path.with_suffix(".manifest.json"))
    candidates.extend(sorted((ROOT / "results/blender_environments").glob(f"*{tag}*.manifest.json")))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            yield candidate


def _find_environment_manifest(summary: SummaryBundle) -> dict[str, Any] | None:
    """Find and read the environment manifest associated with a summary."""
    for candidate in _candidate_manifest_paths(summary):
        try:
            return _read_json(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _draw_manifest_obstacles(ax: Axes, manifest: dict[str, Any]) -> None:
    """Draw metric obstacle footprints from an environment manifest."""
    patches: list[Rectangle] = []
    cell_size = float(manifest.get("obstacle_cell_size_m", 1.0))
    origin = manifest.get("obstacle_origin_xy", [0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    for cell in manifest.get("obstacle_cells", []):
        ix, iy = int(cell[0]), int(cell[1])
        patches.append(Rectangle((ox + ix * cell_size, oy + iy * cell_size), cell_size, cell_size))
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor="#c6cbd1",
                edgecolor="#686e75",
                linewidth=0.28,
                alpha=0.92,
                zorder=2,
            )
        )
    component_patches: list[Rectangle] = []
    for instance in manifest.get("obstacle_instances", []):
        for component in instance.get("components", []):
            center = component.get("center_xyz", [0.0, 0.0, 0.0])
            size = component.get("size_xyz", [0.0, 0.0, 0.0])
            width, height = float(size[0]), float(size[1])
            if width <= 0.0 or height <= 0.0:
                continue
            component_patches.append(
                Rectangle((float(center[0]) - width / 2.0, float(center[1]) - height / 2.0),
                          width, height)
            )
    if component_patches:
        ax.add_collection(
            PatchCollection(
                component_patches,
                facecolor="#8b929a",
                edgecolor="#4b4f54",
                linewidth=0.22,
                alpha=0.62,
                zorder=2.2,
            )
        )


def _room_size(summary: SummaryBundle, manifest: dict[str, Any] | None) -> tuple[float, float, float]:
    """Return room dimensions from the manifest or source coordinates."""
    if manifest is not None and "room_size_xyz" in manifest:
        room = manifest["room_size_xyz"]
        return float(room[0]), float(room[1]), float(room[2])
    all_positions: list[list[float]] = []
    for collection_name in ("ground_truth_sources", "estimated_sources"):
        for sources in summary.payload.get(collection_name, {}).values():
            all_positions.extend(source["pos"] for source in sources)
    if not all_positions:
        return 10.0, 20.0, 10.0
    arr = np.asarray(all_positions, dtype=float)
    return (
        max(10.0, float(np.nanmax(arr[:, 0])) + 1.0),
        max(10.0, float(np.nanmax(arr[:, 1])) + 1.0),
        max(4.0, float(np.nanmax(arr[:, 2])) + 1.0),
    )


def _load_trace_positions(summary: SummaryBundle) -> np.ndarray:
    """Load unique robot station positions from an intermediate trace JSONL."""
    trace_path = summary.payload.get("output_paths", {}).get("intermediate_estimate_trace_jsonl")
    if not trace_path:
        return np.zeros((0, 3), dtype=float)
    path = Path(trace_path)
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    stations: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_position = payload.get("robot_position")
            if raw_position is None or len(raw_position) < 3:
                continue
            key = tuple(round(float(value), 3) for value in raw_position[:3])
            if key in seen:
                continue
            seen.add(key)
            stations.append(key)
    return np.asarray(stations, dtype=float) if stations else np.zeros((0, 3), dtype=float)


def _load_path_waypoints(summary: SummaryBundle) -> list[np.ndarray]:
    """Load saved obstacle-aware path waypoint polylines from a summary."""
    segments = summary.payload.get("mission_metrics", {}).get("path_segments", [])
    if not isinstance(segments, list):
        return []
    output: list[np.ndarray] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        raw_waypoints = segment.get("waypoints_xyz", [])
        waypoints = np.asarray(raw_waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 3:
            continue
        output.append(waypoints[:, :3])
    return output


def _load_final_particle_cloud(summary: SummaryBundle) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load final PF particle source-slot positions and weights from a summary."""
    cloud = summary.payload.get("final_particle_cloud", {})
    if not isinstance(cloud, dict):
        return {}
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for isotope, payload in cloud.items():
        if not isinstance(payload, dict):
            continue
        positions = np.asarray(payload.get("positions", []), dtype=float)
        if positions.ndim != 2 or positions.shape[1] < 3:
            continue
        positions = positions[:, :3]
        weights = np.asarray(payload.get("weights", []), dtype=float).reshape(-1)
        if weights.size != positions.shape[0]:
            weights = np.ones(positions.shape[0], dtype=float)
        output[str(isotope)] = (positions, weights)
    return output


def _particle_sizes(weights: np.ndarray, *, base: float = 3.0, scale: float = 16.0) -> np.ndarray:
    """Return readable marker sizes for weighted PF particles."""
    if weights.size == 0:
        return np.zeros(0, dtype=float)
    weights = np.asarray(weights, dtype=float)
    max_weight = float(np.nanmax(weights)) if weights.size else 0.0
    if max_weight <= 0.0 or not np.isfinite(max_weight):
        return np.full(weights.shape, base, dtype=float)
    return base + scale * np.sqrt(np.clip(weights / max_weight, 0.0, 1.0))


def _plot_sources(ax: Axes, summary: SummaryBundle, *, elevation: bool = False) -> None:
    """Plot ground-truth and estimated sources on one axis."""
    for isotope, sources in summary.payload.get("ground_truth_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for source in sources:
            pos = np.asarray(source["pos"], dtype=float)
            xy = (pos[0], pos[2]) if elevation else (pos[0], pos[1])
            ax.scatter(*xy, marker="x", s=55, color=color, lw=1.6, zorder=6, clip_on=False)
    for isotope, sources in summary.payload.get("estimated_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for source in sources:
            pos = np.asarray(source["pos"], dtype=float)
            xy = (pos[0], pos[2]) if elevation else (pos[0], pos[1])
            ax.scatter(
                *xy,
                marker="o",
                s=38,
                facecolor="white",
                edgecolor=color,
                lw=1.4,
                zorder=7,
                clip_on=False,
            )


def _source_axis_position(pos: np.ndarray, projection: str) -> tuple[float, float]:
    """Return the 2-D coordinates for a source or station projection."""
    if projection == "xy":
        return float(pos[0]), float(pos[1])
    if projection == "yz":
        return float(pos[1]), float(pos[2])
    if projection == "xz":
        return float(pos[0]), float(pos[2])
    raise ValueError(f"Unsupported projection: {projection}")


def _iter_match_pairs(
    summary: SummaryBundle,
) -> Iterable[tuple[str, np.ndarray, np.ndarray, float]]:
    """Yield isotope, truth position, estimate position, and 3-D error pairs."""
    metrics = summary.payload.get("match_metrics", {}).get("isotopes", {})
    truth_sources = summary.payload.get("ground_truth_sources", {})
    estimated_sources = summary.payload.get("estimated_sources", {})
    for isotope, isotope_metrics in metrics.items():
        truth = truth_sources.get(isotope, [])
        estimates = estimated_sources.get(isotope, [])
        for match in isotope_metrics.get("matches", []):
            gt_index = int(match.get("gt_index", -1))
            est_index = int(match.get("est_index", -1))
            if gt_index < 0 or gt_index >= len(truth):
                continue
            if est_index < 0 or est_index >= len(estimates):
                continue
            truth_pos = np.asarray(truth[gt_index]["pos"], dtype=float)
            estimate_pos = np.asarray(estimates[est_index]["pos"], dtype=float)
            distance = float(match.get("distance", np.linalg.norm(truth_pos - estimate_pos)))
            yield isotope, truth_pos, estimate_pos, distance


def _matched_source_ids(summary: SummaryBundle) -> set[tuple[str, int]]:
    """Return the estimated source indices that have been matched to truth."""
    matched: set[tuple[str, int]] = set()
    metrics = summary.payload.get("match_metrics", {}).get("isotopes", {})
    for isotope, isotope_metrics in metrics.items():
        for match in isotope_metrics.get("matches", []):
            est_index = int(match.get("est_index", -1))
            if est_index >= 0:
                matched.add((isotope, est_index))
    return matched


def _cuboid_faces(
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
) -> list[list[tuple[float, float, float]]]:
    """Return six polygon faces for one axis-aligned cuboid."""
    x0, y0, z0 = origin
    dx, dy, dz = size
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _draw_room_box_3d(ax: Axes, room: tuple[float, float, float]) -> None:
    """Draw a light 3-D room frame on an axes."""
    room_x, room_y, room_z = room
    floor = Poly3DCollection(
        [[(0.0, 0.0, 0.0), (room_x, 0.0, 0.0), (room_x, room_y, 0.0), (0.0, room_y, 0.0)]],
        facecolors="#f5f6f7",
        edgecolors="none",
        alpha=0.18,
    )
    ax.add_collection3d(floor)
    edges = [
        ((0, 0, 0), (room_x, 0, 0)),
        ((0, room_y, 0), (room_x, room_y, 0)),
        ((0, 0, room_z), (room_x, 0, room_z)),
        ((0, room_y, room_z), (room_x, room_y, room_z)),
        ((0, 0, 0), (0, room_y, 0)),
        ((room_x, 0, 0), (room_x, room_y, 0)),
        ((0, 0, room_z), (0, room_y, room_z)),
        ((room_x, 0, room_z), (room_x, room_y, room_z)),
        ((0, 0, 0), (0, 0, room_z)),
        ((room_x, 0, 0), (room_x, 0, room_z)),
        ((0, room_y, 0), (0, room_y, room_z)),
        ((room_x, room_y, 0), (room_x, room_y, room_z)),
    ]
    for start, end in edges:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#c4c7c5",
            lw=0.45,
            alpha=0.8,
        )


def _draw_manifest_obstacles_3d(ax: Axes, manifest: dict[str, Any] | None) -> None:
    """Draw obstacle cells as translucent 3-D blocks."""
    if manifest is None:
        return
    cell_size = float(manifest.get("obstacle_cell_size_m", 1.0))
    origin = manifest.get("obstacle_origin_xy", [0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    obstacle_height = float(manifest.get("obstacle_height_m", 2.0) or 2.0)
    faces: list[list[tuple[float, float, float]]] = []
    for cell in manifest.get("obstacle_cells", []):
        ix, iy = int(cell[0]), int(cell[1])
        faces.extend(
            _cuboid_faces(
                (ox + ix * cell_size, oy + iy * cell_size, 0.0),
                (cell_size, cell_size, obstacle_height),
            )
        )
    for instance in manifest.get("obstacle_instances", []):
        for component in instance.get("components", []):
            center = component.get("center_xyz", [0.0, 0.0, 0.0])
            size = component.get("size_xyz", [0.0, 0.0, 0.0])
            sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
            if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
                continue
            origin_xyz = (
                float(center[0]) - sx / 2.0,
                float(center[1]) - sy / 2.0,
                max(0.0, float(center[2]) - sz / 2.0),
            )
            faces.extend(_cuboid_faces(origin_xyz, (sx, sy, sz)))
    if not faces:
        return
    collection = Poly3DCollection(
        faces,
        facecolors="#9aa0a6",
        edgecolors="#6b7075",
        linewidths=0.12,
        alpha=0.14,
    )
    ax.add_collection3d(collection)


def _plot_match_segments_3d(ax: Axes, summary: SummaryBundle) -> None:
    """Draw truth-estimate match segments in 3-D."""
    for isotope, truth_pos, estimate_pos, _distance in _iter_match_pairs(summary):
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        ax.plot(
            [truth_pos[0], estimate_pos[0]],
            [truth_pos[1], estimate_pos[1]],
            [truth_pos[2], estimate_pos[2]],
            color=color,
            lw=1.0,
            alpha=0.78,
            ls="-",
            zorder=9,
        )


def _plot_sources_3d(ax: Axes, summary: SummaryBundle) -> None:
    """Plot 3-D ground-truth and estimated source markers."""
    matched_estimates = _matched_source_ids(summary)
    for isotope, truth_pos, estimate_pos, _distance in _iter_match_pairs(summary):
        for pos in (truth_pos, estimate_pos):
            ax.plot(
                [pos[0], pos[0]],
                [pos[1], pos[1]],
                [0.0, pos[2]],
                color=ISOTOPE_COLORS.get(isotope, "#111111"),
                lw=0.45,
                alpha=0.32,
                ls=":",
                zorder=4,
            )
    for isotope, sources in summary.payload.get("ground_truth_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for source in sources:
            pos = np.asarray(source["pos"], dtype=float)
            ax.scatter(pos[0], pos[1], pos[2], marker="x", s=48, color=color,
                       linewidths=1.5, depthshade=False, zorder=10)
    for isotope, sources in summary.payload.get("estimated_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for index, source in enumerate(sources):
            pos = np.asarray(source["pos"], dtype=float)
            alpha = 1.0 if (isotope, index) in matched_estimates else 0.68
            ax.scatter(
                pos[0],
                pos[1],
                pos[2],
                marker="o",
                s=44,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                depthshade=False,
                alpha=alpha,
                zorder=11,
            )
    _plot_match_segments_3d(ax, summary)


def _plot_particle_cloud_3d(ax: Axes, summary: SummaryBundle) -> None:
    """Plot final PF particle support in 3-D."""
    for isotope, (positions, weights) in _load_final_particle_cloud(summary).items():
        if positions.size == 0:
            continue
        color = ISOTOPE_COLORS.get(isotope, "#555555")
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=_particle_sizes(weights, base=2.0, scale=10.0),
            color=color,
            alpha=0.13,
            marker=".",
            depthshade=False,
            zorder=3,
        )


def _plot_path_3d(ax: Axes, summary: SummaryBundle) -> None:
    """Plot saved obstacle-aware robot route segments in 3-D."""
    for waypoints in _load_path_waypoints(summary):
        ax.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            color="#005bbb",
            lw=0.75,
            alpha=0.58,
            zorder=6,
        )


def _set_projection_ticks(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply metric axis styling with equal 2-D aspect and 2 m ticks."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    x_tick_start = 2.0 * np.ceil(xlim[0] / 2.0)
    y_tick_start = 2.0 * np.ceil(ylim[0] / 2.0)
    ax.set_xticks(np.arange(x_tick_start, xlim[1] + 0.1, 2.0))
    ax.set_yticks(np.arange(y_tick_start, ylim[1] + 0.1, 2.0))
    ax.grid(True, lw=0.25, alpha=0.38)
    ax.set_xlabel(xlabel, fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=FIG_LABEL_SIZE)
    ax.tick_params(labelsize=FIG_TICK_SIZE, pad=1.0, length=2.0)


def _draw_projection_obstacles(
    ax: Axes,
    manifest: dict[str, Any] | None,
    projection: str,
) -> None:
    """Draw known obstacle geometry in a 2-D projection."""
    if manifest is None:
        return
    if projection == "xy":
        _draw_manifest_obstacles(ax, manifest)
        return
    if projection != "yz":
        return
    patches: list[Rectangle] = []
    cell_size = float(manifest.get("obstacle_cell_size_m", 1.0))
    origin = manifest.get("obstacle_origin_xy", [0.0, 0.0])
    oy = float(origin[1])
    obstacle_height = float(manifest.get("obstacle_height_m", 2.0) or 2.0)
    for cell in manifest.get("obstacle_cells", []):
        iy = int(cell[1])
        patches.append(Rectangle((oy + iy * cell_size, 0.0), cell_size, obstacle_height))
    for instance in manifest.get("obstacle_instances", []):
        for component in instance.get("components", []):
            center = component.get("center_xyz", [0.0, 0.0, 0.0])
            size = component.get("size_xyz", [0.0, 0.0, 0.0])
            width = float(size[1])
            height = float(size[2])
            if width <= 0.0 or height <= 0.0:
                continue
            patches.append(
                Rectangle(
                    (float(center[1]) - width / 2.0, max(0.0, float(center[2]) - height / 2.0)),
                    width,
                    height,
                )
            )
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor="#c0c6cc",
                edgecolor="#687078",
                linewidth=0.20,
                alpha=0.82,
                zorder=2,
            )
        )


def _plot_match_segments_projection(
    ax: Axes,
    summary: SummaryBundle,
    projection: str,
) -> None:
    """Draw truth-estimate match segments in one 2-D projection."""
    for isotope, truth_pos, estimate_pos, _distance in _iter_match_pairs(summary):
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        p_truth = _source_axis_position(truth_pos, projection)
        p_estimate = _source_axis_position(estimate_pos, projection)
        ax.plot(
            [p_truth[0], p_estimate[0]],
            [p_truth[1], p_estimate[1]],
            color=color,
            lw=0.9,
            alpha=0.72,
            zorder=5,
        )


def _plot_sources_projection(
    ax: Axes,
    summary: SummaryBundle,
    projection: str,
) -> None:
    """Plot truth and estimate source markers in one 2-D projection."""
    matched_estimates = _matched_source_ids(summary)
    _plot_match_segments_projection(ax, summary, projection)
    for isotope, sources in summary.payload.get("ground_truth_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for source in sources:
            pos = np.asarray(source["pos"], dtype=float)
            ax.scatter(
                *_source_axis_position(pos, projection),
                marker="x",
                s=44,
                color=color,
                lw=1.45,
                zorder=7,
                clip_on=False,
            )
    for isotope, sources in summary.payload.get("estimated_sources", {}).items():
        color = ISOTOPE_COLORS.get(isotope, "#111111")
        for index, source in enumerate(sources):
            pos = np.asarray(source["pos"], dtype=float)
            alpha = 1.0 if (isotope, index) in matched_estimates else 0.62
            ax.scatter(
                *_source_axis_position(pos, projection),
                marker="o",
                s=32,
                facecolor="white",
                edgecolor=color,
                lw=1.25,
                alpha=alpha,
                zorder=8,
                clip_on=False,
            )


def _plot_particle_cloud_projection(
    ax: Axes,
    summary: SummaryBundle,
    projection: str,
) -> None:
    """Plot final PF particle support in one projection."""
    for isotope, (positions, weights) in _load_final_particle_cloud(summary).items():
        if positions.size == 0:
            continue
        color = ISOTOPE_COLORS.get(isotope, "#555555")
        coords = np.asarray(
            [_source_axis_position(position, projection) for position in positions],
            dtype=float,
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=_particle_sizes(weights, base=2.4, scale=10.0),
            color=color,
            alpha=0.13,
            marker=".",
            zorder=3,
        )


def _plot_path_projection(
    ax: Axes,
    summary: SummaryBundle,
    projection: str,
) -> None:
    """Plot saved robot route segments in one projection."""
    for waypoints in _load_path_waypoints(summary):
        coords = np.asarray(
            [_source_axis_position(waypoint, projection) for waypoint in waypoints],
            dtype=float,
        )
        ax.plot(coords[:, 0], coords[:, 1], color="#005bbb", lw=0.75, alpha=0.58, zorder=4)


def _plot_result_projection(
    ax: Axes,
    summary: SummaryBundle,
    manifest: dict[str, Any] | None,
    *,
    projection: str,
    title: str,
    panel: str,
) -> None:
    """Plot a metric floor or elevation projection for a result summary."""
    room_x, room_y, room_z = _room_size(summary, manifest)
    if projection == "xy":
        ax.add_patch(
            Rectangle((0.0, 0.0), room_x, room_y, facecolor="#fbfbfb",
                      edgecolor="none", lw=0.0, zorder=-10)
        )
        stations = _load_trace_positions(summary)
        if stations.size:
            ax.scatter(stations[:, 0], stations[:, 1], s=7.0, color="#005bbb",
                       alpha=0.58, zorder=4)
            ax.scatter(stations[0, 0], stations[0, 1], s=20, color="#005bbb",
                       marker="s", zorder=5)
            ax.scatter(stations[-1, 0], stations[-1, 1], s=25, color="#111111",
                       marker="*", zorder=5)
        _plot_path_projection(ax, summary, "xy")
        _draw_projection_obstacles(ax, manifest, "xy")
        _plot_particle_cloud_projection(ax, summary, "xy")
        _plot_sources_projection(ax, summary, "xy")
        _set_projection_ticks(
            ax,
            (-0.35, room_x + 0.35),
            (-0.35, room_y + 0.85),
            xlabel="x [m]",
            ylabel="y [m]",
        )
    elif projection == "yz":
        ax.add_patch(
            Rectangle((0.0, 0.0), room_y, room_z, facecolor="#fbfbfb",
                      edgecolor="none", lw=0.0, zorder=-10)
        )
        stations = _load_trace_positions(summary)
        if stations.size:
            ax.scatter(stations[:, 1], stations[:, 2], s=7.0, color="#005bbb",
                       alpha=0.42, zorder=4)
        _plot_path_projection(ax, summary, "yz")
        _draw_projection_obstacles(ax, manifest, "yz")
        _plot_particle_cloud_projection(ax, summary, "yz")
        _plot_sources_projection(ax, summary, "yz")
        _set_projection_ticks(
            ax,
            (-0.35, room_y + 0.35),
            (-0.25, room_z + 0.85),
            xlabel="y [m]",
            ylabel="z [m]",
        )
    else:
        raise ValueError(f"Unsupported result projection: {projection}")
    ax.set_title(title, fontsize=FIG_TITLE_SIZE, pad=2.0)
    _panel_label(ax, panel)


def _plot_spatial_result_3d(
    ax: Axes,
    summary: SummaryBundle,
    manifest: dict[str, Any] | None,
    *,
    title: str,
    panel: str,
) -> None:
    """Plot a 3-D result map with room, obstacles, stations, truth, and estimates."""
    room = _room_size(summary, manifest)
    room_x, room_y, room_z = room
    _draw_room_box_3d(ax, room)
    _draw_manifest_obstacles_3d(ax, manifest)
    stations = _load_trace_positions(summary)
    if stations.size:
        ax.scatter(stations[:, 0], stations[:, 1], stations[:, 2], s=10,
                   color="#005bbb", alpha=0.72, depthshade=False, zorder=8)
        ax.scatter(stations[0, 0], stations[0, 1], stations[0, 2], s=28,
                   color="#005bbb", marker="s", depthshade=False, zorder=9)
        ax.scatter(stations[-1, 0], stations[-1, 1], stations[-1, 2], s=34,
                   color="#111111", marker="*", depthshade=False, zorder=9)
    _plot_path_3d(ax, summary)
    _plot_particle_cloud_3d(ax, summary)
    _plot_sources_3d(ax, summary)
    ax.set_xlim(0, room_x)
    ax.set_ylim(0, room_y)
    ax.set_zlim(0, room_z)
    ax.set_box_aspect((room_x, room_y, room_z))
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks(np.arange(0, room_x + 0.1, 2.0))
    ax.set_yticks(np.arange(0, room_y + 0.1, 2.0))
    ax.set_zticks(np.arange(0, room_z + 0.1, 2.0))
    ax.set_xlabel("x [m]", fontsize=FIG_LABEL_SIZE, labelpad=-5)
    ax.set_ylabel("y [m]", fontsize=FIG_LABEL_SIZE, labelpad=-5)
    ax.set_zlabel("z [m]", fontsize=FIG_LABEL_SIZE, labelpad=-4)
    ax.tick_params(labelsize=FIG_TICK_SIZE, pad=-3, length=1.5)
    ax.set_title(title, fontsize=FIG_TITLE_SIZE, pad=1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03)
        axis._axinfo["grid"]["linewidth"] = 0.25
        axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.35)
    if panel:
        ax.text2D(
            -0.08,
            1.02,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FIG_PANEL_SIZE,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.9},
            clip_on=False,
        )


def _map_legend_handles(summary: SummaryBundle) -> list[Line2D]:
    """Return compact legend handles for map source and station symbols."""
    handles: list[Line2D] = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#005bbb",
               markeredgecolor="#005bbb", markersize=4.5, label="station"),
        Line2D([0, 1], [0, 0], color="#005bbb", lw=0.9, label="saved route"),
        Line2D([0], [0], marker=".", color="#777777", linestyle="none",
               markersize=5.0, alpha=0.55, label="PF particle"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#005bbb",
               markeredgecolor="#005bbb", markersize=5.0, label="first station"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111",
               markeredgecolor="#111111", markersize=6.0, label="last station"),
        Line2D([0], [0], marker="x", color="#111111", linestyle="none",
               markersize=6.0, markeredgewidth=1.2, label="truth"),
        Line2D([0], [0], marker="o", color="#111111", linestyle="none",
               markerfacecolor="white", markersize=5.5, markeredgewidth=1.2,
               label="estimate"),
    ]
    isotopes = [
        isotope for isotope in ("Cs-137", "Co-60", "Eu-154")
        if isotope in summary.payload.get("ground_truth_sources", {})
        or isotope in summary.payload.get("estimated_sources", {})
    ]
    for isotope in isotopes:
        handles.append(
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=ISOTOPE_COLORS.get(isotope, "#111111"),
                   markeredgecolor=ISOTOPE_COLORS.get(isotope, "#111111"),
                   markersize=4.5, label=isotope)
        )
    return handles[:9]


def _combined_legend_handles(summaries: Iterable[SummaryBundle]) -> list[Line2D]:
    """Return one compact legend for all result-map panels."""
    summary_list = list(summaries)
    if not summary_list:
        return []
    handles = _map_legend_handles(summary_list[0])
    all_isotopes = {
        isotope
        for summary in summary_list
        for isotope in set(summary.payload.get("ground_truth_sources", {}))
        | set(summary.payload.get("estimated_sources", {}))
    }
    labels = [handle.get_label() for handle in handles]
    for isotope in ("Cs-137", "Co-60", "Eu-154"):
        if isotope not in all_isotopes or isotope in labels:
            continue
        handles.append(
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=ISOTOPE_COLORS[isotope],
                   markeredgecolor=ISOTOPE_COLORS[isotope],
                   markersize=4.5, label=isotope)
        )
    handles.append(
        Line2D([0, 1], [0, 0], color="#555555", lw=0.9, label="match segment")
    )
    return handles


def _extract_ablation_row(summary: SummaryBundle) -> AblationRow:
    """Extract compact ablation metrics from one result summary."""
    tag = _summary_tag(summary.path)
    label = tag
    for prefix in (
        "mix9_multi_isotope_cardinality_",
    ):
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = label.replace("_seed_2026050901", "")
    label = label.replace("baseline_passive_equal_time_no_shield", "passive equal-time")
    label = label.replace("baseline_passive_no_shield", "passive no-shield")
    label = label.replace("round_robin_shield", "round-robin")
    label = label.replace("one_step_path", "one-step")
    label = label.replace("eig_only_path", "EIG-only")
    label = label.replace("_", " ")
    isotope_metrics = summary.payload.get("match_metrics", {}).get("isotopes", {})
    counts = {"assigned": 0, "fp": 0, "fn": 0}
    position_errors: list[float] = []
    strength_errors: list[float] = []
    for metrics in isotope_metrics.values():
        item_counts = metrics.get("counts", {})
        counts["assigned"] += int(item_counts.get("assigned", item_counts.get("tp", 0)) or 0)
        counts["fp"] += int(item_counts.get("fp", 0) or 0)
        counts["fn"] += int(item_counts.get("fn", 0) or 0)
        position = metrics.get("position_error", {}).get("mean")
        strength = metrics.get("intensity_rel_error_pct", {}).get("mean")
        if position is not None:
            position_errors.append(float(position))
        if strength is not None:
            strength_errors.append(float(strength))
    return AblationRow(
        label=label,
        spectra=int(summary.payload.get("measurements_completed", 0) or 0),
        true_positive=counts["assigned"],
        false_positive=counts["fp"],
        false_negative=counts["fn"],
        mean_position_error_m=float(np.mean(position_errors)) if position_errors else float("nan"),
        mean_strength_error_pct=float(np.mean(strength_errors)) if strength_errors else float("nan"),
    )


def _plot_spatial_result(
    ax: Axes,
    summary: SummaryBundle,
    manifest: dict[str, Any] | None,
    *,
    title: str = "Metric map: obstacles, stations, truth, estimates",
    panel: str = "(a)",
    show_legend: bool = False,
) -> None:
    """Plot the metric top-down result map."""
    room_x, room_y, _ = _room_size(summary, manifest)
    ax.add_patch(Rectangle((0.0, 0.0), room_x, room_y, facecolor="#f9fafb",
                           edgecolor="#222222", lw=0.9, zorder=0))
    if manifest is not None:
        _draw_manifest_obstacles(ax, manifest)
    stations = _load_trace_positions(summary)
    if stations.size:
        ax.scatter(stations[:, 0], stations[:, 1], s=8, color="#005bbb",
                   alpha=0.68, label="stations", zorder=4)
        ax.scatter(stations[0, 0], stations[0, 1], s=22, color="#005bbb",
                   marker="s", zorder=5)
        ax.scatter(stations[-1, 0], stations[-1, 1], s=26, color="#111111",
                   marker="*", zorder=5)
    _plot_sources(ax, summary, elevation=False)
    ax.set_xlim(0, room_x)
    ax.set_ylim(0, room_y)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, room_x + 0.1, 2.0))
    ax.set_yticks(np.arange(0, room_y + 0.1, 2.0))
    ax.grid(True, lw=0.25, alpha=0.35)
    ax.set_xlabel("x [m]", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("y [m]", fontsize=FIG_LABEL_SIZE)
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    ax.set_title(title, fontsize=FIG_TITLE_SIZE)
    if show_legend:
        ax.legend(
            handles=_map_legend_handles(summary),
            loc="upper right",
            fontsize=FIG_TICK_SIZE,
            frameon=True,
            framealpha=0.92,
            borderpad=0.25,
            handletextpad=0.3,
            labelspacing=0.2,
            borderaxespad=0.25,
            ncol=1,
        )
    _panel_label(ax, panel)


def _plot_elevation(ax: Axes, summary: SummaryBundle, manifest: dict[str, Any] | None) -> None:
    """Plot an x-z elevation projection with metric scale."""
    room_x, _, room_z = _room_size(summary, manifest)
    ax.add_patch(Rectangle((0.0, 0.0), room_x, room_z, facecolor="#fbfbfb",
                           edgecolor="#222222", lw=0.8, zorder=0))
    if manifest is not None:
        obstacle_height = float(manifest.get("obstacle_height_m", 0.0))
        for cell in manifest.get("obstacle_cells", []):
            cell_size = float(manifest.get("obstacle_cell_size_m", 1.0))
            origin = manifest.get("obstacle_origin_xy", [0.0, 0.0])
            x0 = float(origin[0]) + int(cell[0]) * cell_size
            ax.add_patch(Rectangle((x0, 0.0), cell_size, obstacle_height,
                                   facecolor="#d5d8dc", edgecolor="#9aa0a6",
                                   lw=0.15, alpha=0.18, zorder=1))
    _plot_sources(ax, summary, elevation=True)
    ax.set_xlim(0, room_x)
    ax.set_ylim(0, room_z)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, room_x + 0.1, 2.0))
    ax.set_yticks(np.arange(0, room_z + 0.1, 2.0))
    ax.grid(True, lw=0.25, alpha=0.35)
    ax.set_xlabel("x [m]", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("z [m]", fontsize=FIG_LABEL_SIZE)
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    ax.set_title("Elevation projection", fontsize=FIG_TITLE_SIZE)
    _panel_label(ax, "(b)")


def _plot_convergence(
    ax: Axes,
    summary: SummaryBundle,
    *,
    panel: str = "(c)",
) -> None:
    """Plot source-count and remaining-view traces from summary diagnostics."""
    estimates = summary.payload.get("remaining_measurement_estimates", [])
    stations: list[int] = []
    remaining: list[float] = []
    by_isotope: dict[str, list[float]] = {iso: [] for iso in ("Cs-137", "Co-60", "Eu-154")}
    for item in estimates:
        stations.append(int(item.get("current_station_count", len(stations) + 1)))
        details = item.get("isotope_details", {})
        for isotope in by_isotope:
            isotope_details = details.get(isotope, {})
            by_isotope[isotope].append(
                float(isotope_details.get("map_source_count", np.nan))
            )
        remaining.append(float(item.get("estimated_remaining_stations", np.nan)))
    if stations:
        for isotope, values in by_isotope.items():
            if not np.any(np.isfinite(values)):
                continue
            ax.plot(
                stations,
                values,
                marker="o",
                ms=2.6,
                lw=1.0,
                color=ISOTOPE_COLORS.get(isotope, "#555555"),
                label=f"$\\hat r$ {isotope}",
            )
        ax.plot(
            stations,
            remaining,
            marker="s",
            ms=2.6,
            lw=0.9,
            color="#444444",
            label="remaining stations",
        )
    truth_counts = {
        isotope: len(sources)
        for isotope, sources in summary.payload.get("ground_truth_sources", {}).items()
    }
    for isotope, count in truth_counts.items():
        ax.axhline(
            count,
            color=ISOTOPE_COLORS.get(isotope, "#555555"),
            lw=0.7,
            ls="--",
            alpha=0.55,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "MIX-9 run pending",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FIG_TITLE_SIZE,
            color="#555555",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max(1.0, float(max(truth_counts.values(), default=1)) + 1.0))
    ax.set_xlabel("station", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("count", fontsize=FIG_LABEL_SIZE)
    ax.set_title("Online isotope-wise cardinality", fontsize=FIG_TITLE_SIZE)
    ax.grid(True, lw=0.25, alpha=0.45)
    ax.tick_params(labelsize=FIG_TICK_SIZE)
    if stations:
        ax.legend(fontsize=FIG_TICK_SIZE, frameon=True, loc="best")
    _panel_label(ax, panel)


def _extract_signature_heatmap(
    summary: SummaryBundle,
) -> tuple[np.ndarray | None, list[str], list[str], str, str | None]:
    """Return a shield-time signature matrix from a flexible summary schema."""
    candidate_keys = (
        "shield_time_signature_heatmap",
        "shield_signature_heatmap",
        "selected_shield_signature_heatmap",
        "signature_heatmap",
    )
    payload: Any = None
    for key in candidate_keys:
        if key in summary.payload:
            payload = summary.payload[key]
            break
    if payload is None:
        diagnostics = summary.payload.get("signature_diagnostics", {})
        if isinstance(diagnostics, dict):
            payload = diagnostics.get("shield_time_heatmap")
    if payload is None:
        return None, [], [], "Shield-time response matrix", None
    if isinstance(payload, dict):
        raw_matrix = payload.get("matrix", payload.get("values"))
        row_labels = [str(value) for value in payload.get("row_labels", [])]
        col_labels = [str(value) for value in payload.get("column_labels", [])]
        title = str(payload.get("title", "Shield-time response matrix"))
        before = payload.get("rho_before", payload.get("correlation_before"))
        after = payload.get("rho_after", payload.get("correlation_after"))
        rho_text = None
        if before is not None and after is not None:
            rho_text = f"$\\rho_{{max}}$: {float(before):.2f} -> {float(after):.2f}"
    else:
        raw_matrix = payload
        row_labels = []
        col_labels = []
        title = "Shield-time response matrix"
        rho_text = None
    matrix = np.asarray(raw_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return None, [], [], title, rho_text
    if not row_labels:
        row_labels = [f"$a_{index + 1}$" for index in range(matrix.shape[0])]
    if not col_labels:
        col_labels = [f"H{index + 1}" for index in range(matrix.shape[1])]
    return matrix, row_labels, col_labels, title, rho_text


def _plot_signature_heatmap(
    ax: Axes,
    summary: SummaryBundle,
    *,
    panel: str = "(d)",
) -> None:
    """Plot the selected shield-time response matrix for a confusable pair."""
    matrix, row_labels, col_labels, title, rho_text = _extract_signature_heatmap(summary)
    ax.set_title(title, fontsize=FIG_TITLE_SIZE)
    if matrix is None:
        ax.text(
            0.5,
            0.54,
            "shield-time matrix\npending full run",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FIG_TITLE_SIZE,
            color="#555555",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        _panel_label(ax, panel)
        return
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    vmin = float(np.nanmin(matrix)) if np.isfinite(matrix).any() else 0.0
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel("source hypothesis", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("posture", fontsize=FIG_LABEL_SIZE)
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(labelsize=FIG_TICK_SIZE, length=1.8)
    cbar = plt.colorbar(image, ax=ax, fraction=0.052, pad=0.025)
    cbar.set_label("normalized count", fontsize=FIG_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=FIG_TICK_SIZE, length=1.5)
    if rho_text:
        ax.text(
            0.98,
            0.04,
            rho_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FIG_TICK_SIZE,
            color="#111111",
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 1.2},
        )
    _panel_label(ax, panel)


def _plot_ablation(ax: Axes, summaries: list[SummaryBundle], *, panel: str = "(d)") -> None:
    """Plot a compact ablation comparison from result summaries."""
    rows = [_extract_ablation_row(summary) for summary in summaries]
    labels = [row.label for row in rows]
    values = [row.mean_position_error_m for row in rows]
    colors = ["#1b9e77" if "proposed" in row.label else "#7570b3" for row in rows]
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors, alpha=0.86)
    for index, row in enumerate(rows):
        ax.text(
            index,
            values[index] + 0.03,
            f"{row.true_positive}/{row.false_positive}/{row.false_negative}\n{row.spectra} sp.",
            ha="center",
            va="bottom",
            fontsize=FIG_TICK_SIZE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=FIG_TICK_SIZE)
    ax.set_ylabel("mean 3-D error [m]", fontsize=FIG_LABEL_SIZE)
    ax.set_title("Ablation summary: TP/FP/FN and spectra", fontsize=FIG_TITLE_SIZE)
    finite_values = [value for value in values if np.isfinite(value)]
    if finite_values:
        ax.set_ylim(0.0, max(finite_values) * 1.28 + 0.12)
    ax.grid(True, axis="y", lw=0.25, alpha=0.45)
    ax.tick_params(axis="y", labelsize=FIG_TICK_SIZE)
    _panel_label(ax, panel)


def _task_name(summary: SummaryBundle) -> str:
    """Return the paper task name encoded by a summary path."""
    tag = _summary_tag(summary.path)
    if "mix9_multi_isotope_cardinality" in tag:
        return "mix9"
    return "unknown"


def _select_summary(
    summaries: list[SummaryBundle],
    task_name: str,
    *,
    proposed_only: bool = True,
) -> SummaryBundle | None:
    """Select one summary for a task."""
    matches = [summary for summary in summaries if _task_name(summary) == task_name]
    if proposed_only:
        proposed = [summary for summary in matches if "proposed" in _summary_tag(summary.path)]
        if proposed:
            return proposed[0]
    return matches[0] if matches else None


def _plot_mixed_isotope_errors(ax: Axes, summary: SummaryBundle | None) -> None:
    """Plot isotope-wise mixed-cardinality errors and TP/FP/FN labels."""
    ax.set_title("Proposed mixed-cardinality: isotope-wise report",
                 fontsize=FIG_TITLE_SIZE)
    if summary is None:
        ax.text(0.5, 0.5, "mixed-cardinality\nsummary not available",
                transform=ax.transAxes, ha="center", va="center", fontsize=FIG_TITLE_SIZE)
        ax.axis("off")
        _panel_label(ax, "(d)")
        return

    isotopes = [iso for iso in ("Cs-137", "Co-60", "Eu-154")
                if iso in summary.payload.get("match_metrics", {}).get("isotopes", {})]
    metrics = summary.payload.get("match_metrics", {}).get("isotopes", {})
    values = [float(metrics[iso].get("position_error", {}).get("mean", np.nan)) for iso in isotopes]
    colors = [ISOTOPE_COLORS.get(iso, "#666666") for iso in isotopes]
    x = np.arange(len(isotopes))
    ax.bar(x, values, color=colors, alpha=0.86)
    for index, iso in enumerate(isotopes):
        counts = metrics[iso].get("counts", {})
        label = (
            f"{int(counts.get('assigned', 0))}/"
            f"{int(counts.get('fp', 0))}/"
            f"{int(counts.get('fn', 0))}"
        )
        ax.text(index, values[index] + 0.05, label, ha="center", va="bottom",
                fontsize=FIG_TICK_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(isotopes, fontsize=FIG_TICK_SIZE)
    ax.set_ylabel("mean 3-D error [m]", fontsize=FIG_LABEL_SIZE)
    finite_values = [value for value in values if np.isfinite(value)]
    if finite_values:
        ax.set_ylim(0.0, max(finite_values) * 1.28 + 0.2)
    ax.grid(True, axis="y", lw=0.25, alpha=0.45)
    ax.tick_params(axis="y", labelsize=FIG_TICK_SIZE)
    _panel_label(ax, "(d)")


def _plot_map_symbol_legend(ax: Axes, summaries: list[SummaryBundle]) -> None:
    """Draw a standalone legend for map symbols and isotope colors."""
    ax.axis("off")
    reference = summaries[0]
    handles = _map_legend_handles(reference)
    all_isotopes = {
        isotope
        for summary in summaries
        for isotope in set(summary.payload.get("ground_truth_sources", {}))
        | set(summary.payload.get("estimated_sources", {}))
    }
    handles = [
        handle for handle in handles
        if handle.get_label() not in ISOTOPE_COLORS or handle.get_label() in all_isotopes
    ]
    for isotope in ("Co-60", "Eu-154"):
        if isotope in all_isotopes and isotope not in [handle.get_label() for handle in handles]:
            handles.append(
                Line2D([0], [0], marker="s", color="none",
                       markerfacecolor=ISOTOPE_COLORS[isotope],
                       markeredgecolor=ISOTOPE_COLORS[isotope],
                       markersize=4.5, label=isotope)
            )
    ax.legend(
        handles=handles,
        loc="center",
        fontsize=FIG_TICK_SIZE,
        frameon=True,
        framealpha=0.94,
        borderpad=0.35,
        handletextpad=0.35,
        labelspacing=0.24,
    )
    ax.text(0.5, 0.98, "map legend", transform=ax.transAxes,
            ha="center", va="top", fontsize=FIG_LABEL_SIZE)


def render_experiment_summary(
    summary_paths: list[Path],
    output_path: Path = EXPERIMENT_FIG_PATH,
) -> Path:
    """Render the main proposed-method PF result figure from summaries."""
    if not summary_paths:
        raise ValueError("At least one summary JSON is required.")
    bundles = [SummaryBundle(path=Path(path), payload=_read_json(Path(path))) for path in summary_paths]
    mix9 = _select_summary(bundles, "mix9")
    if mix9 is not None:
        manifest = _find_environment_manifest(mix9)
        fig = plt.figure(figsize=(7.15, 5.12))
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(1.05, 0.96),
            width_ratios=(1.08, 1.0),
            hspace=0.31,
            wspace=0.26,
        )
        _plot_result_projection(
            fig.add_subplot(grid[0, 0]),
            mix9,
            manifest,
            projection="xy",
            title="(a) proposed MIX-9: floor audit",
            panel="",
        )
        _plot_result_projection(
            fig.add_subplot(grid[0, 1]),
            mix9,
            manifest,
            projection="yz",
            title="(b) depth-height projection",
            panel="",
        )
        _plot_convergence(fig.add_subplot(grid[1, 0]), mix9, panel="(c)")
        _plot_signature_heatmap(fig.add_subplot(grid[1, 1]), mix9, panel="(d)")
        handles = _combined_legend_handles([mix9])
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.012),
            ncol=min(6, len(handles)),
            fontsize=6.6,
            frameon=True,
            framealpha=0.96,
            borderpad=0.24,
            handletextpad=0.30,
            columnspacing=0.72,
        )
        fig.subplots_adjust(left=0.055, right=0.992, top=0.965, bottom=0.155)
        return _save_figure(fig, output_path)
    raise ValueError("The RA-L experiment figure requires a MIX-9 summary JSON.")


def default_paper_summaries() -> list[Path]:
    """Return the current paper-scope result summaries if they are available."""
    names = [
        "result_summary_mix9_multi_isotope_cardinality_proposed_seed_2026050901.json",
        "result_summary_mix9_multi_isotope_cardinality_baseline_passive_equal_time_no_shield_seed_2026050901.json",
        "result_summary_mix9_multi_isotope_cardinality_round_robin_shield_seed_2026050901.json",
        "result_summary_mix9_multi_isotope_cardinality_eig_only_path_seed_2026050901.json",
    ]
    return [ROOT / "results" / name for name in names if (ROOT / "results" / name).exists()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-concepts",
        action="store_true",
        help="Do not regenerate Fig. 1 and Fig. 2 concept figures.",
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        type=Path,
        default=[],
        help="Result summary JSON for the experiment figure. May be repeated.",
    )
    parser.add_argument(
        "--experiment-output",
        type=Path,
        default=EXPERIMENT_FIG_PATH,
        help="Output path for the experiment result figure.",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Do not render the experiment result figure.",
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
        help="Do not write raster review copies of generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Build requested RA-L figures."""
    args = parse_args()
    generated: list[Path] = []
    if not args.skip_concepts:
        fig1 = render_problem_setting()
        fig2 = render_method_overview()
        generated.extend([fig1, fig2])
        print(f"Wrote {fig1}")
        print(f"Wrote {fig2}")
    if not args.skip_experiment:
        summary_paths = list(args.summary_json) if args.summary_json else default_paper_summaries()
        if not summary_paths:
            raise FileNotFoundError(
                "No summary JSON files were provided and the default MIX-9 summaries were not found."
            )
        experiment = render_experiment_summary(summary_paths, args.experiment_output)
        generated.append(experiment)
        print(f"Wrote {experiment}")
    if generated and not args.no_review_images:
        review_images = _write_review_images(generated, args.review_output_dir)
        for review_image in review_images:
            print(f"Wrote review image {review_image}")


if __name__ == "__main__":
    main()
