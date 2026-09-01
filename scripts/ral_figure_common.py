"""Common paths and I/O helpers for RA-L manuscript figure builders."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
LATEX_ROOT = ROOT.parent / "ai-latex-workspace" / "projects" / "ieee-ra-l-letter"
FIG1_PATH = LATEX_ROOT / "sections/01_introduction/figures/ral_problem_shield_code.pdf"
FIG2_PATH = LATEX_ROOT / "sections/03_system_model/figures/ral_method_loop.pdf"
REVIEW_DIR = ROOT / "results" / "ral_figure_review"
EXPERIMENT_FIG_PATH = REVIEW_DIR / "ral_result_case_audit.pdf"
ISAAC_FIGURE_DIR = ROOT / "results" / "ral_isaac_figures"
ISAAC_PROBLEM_RENDER = ISAAC_FIGURE_DIR / "capture_problem_setting" / "rgb_0000.png"
ISAAC_DETECTOR_RENDER = ISAAC_FIGURE_DIR / "capture_detector_module" / "rgb_0000.png"
ISAAC_STATION_RENDER = (
    ISAAC_FIGURE_DIR / "capture_simulation_environment" / "rgb_0000.png"
)
ISAAC_SHIELD_PROGRAM_RENDERS = (
    ISAAC_FIGURE_DIR / "capture_shield_selection_00" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_01" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_02" / "rgb_0000.png",
    ISAAC_FIGURE_DIR / "capture_shield_selection_03" / "rgb_0000.png",
)
FIG_TITLE_SIZE = 8.6
FIG_LABEL_SIZE = 7.8
FIG_TICK_SIZE = 7.2
FIG_PANEL_SIZE = 9.2
ISOTOPE_COLORS = {
    "Cs-137": "#d62728",
    "Co-60": "#1f77b4",
    "Eu-154": "#2ca02c",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a matplotlib figure to disk with deterministic layout settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches=None,
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return output_path


def write_review_image(figure_path: Path, review_dir: Path) -> Path | None:
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


def write_review_images(figure_paths: Iterable[Path], review_dir: Path) -> list[Path]:
    """Write review images for all generated figures."""
    outputs: list[Path] = []
    for figure_path in figure_paths:
        review_image = write_review_image(figure_path, review_dir)
        if review_image is not None:
            outputs.append(review_image)
    return outputs
