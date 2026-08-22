"""Publish stable visualization artifacts for completed PF runs."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from visualization.frame import PFFrame


def prepare_final_visualization_frame(
    frame: PFFrame,
    *,
    step_index: int,
    elapsed_s: float,
    final_estimates: Mapping[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
) -> PFFrame:
    """Return a final-estimate frame while preserving the travel segment.

    ``frame.path_waypoints_xyz`` contains the obstacle-aware path that reached
    the final station. Replacing it with measurement poses would draw false
    straight-line links through obstacles. A new frame is returned so callers
    do not accidentally mutate a frame that may still be rendered elsewhere.
    """
    estimated_sources = {
        str(isotope): np.asarray(positions, dtype=np.float64).reshape((-1, 3)).copy()
        for isotope, (positions, _) in final_estimates.items()
    }
    estimated_strengths = {
        str(isotope): np.asarray(strengths, dtype=np.float64).reshape(-1).copy()
        for isotope, (_, strengths) in final_estimates.items()
    }
    return replace(
        frame,
        step_index=max(0, int(step_index)),
        time=float(elapsed_s),
        estimated_sources=estimated_sources,
        estimated_strengths=estimated_strengths,
    )


def publish_final_cui_split_views(
    *,
    source_robot_path: Path,
    source_pf_path: Path,
    source_pf_labeled_path: Path,
    final_robot_path: Path,
    final_pf_path: Path,
    final_pf_labeled_path: Path,
    source_overview_path: Path | None = None,
    source_spectrum_path: Path | None = None,
    final_overview_path: Path | None = None,
    final_spectrum_path: Path | None = None,
) -> None:
    """Publish completed CUI split views as stable result artifacts.

    Every source is validated and copied to a temporary file before any final
    path is replaced. A missing or unreadable source therefore leaves all
    existing result artifacts untouched. Overview and spectrum paths are
    optional for compatibility with callers that publish the original three
    views; each optional source and target must be supplied together.
    """
    optional_pairs = (
        (source_overview_path, final_overview_path, "overview"),
        (source_spectrum_path, final_spectrum_path, "spectrum"),
    )
    for source, target, name in optional_pairs:
        if (source is None) != (target is None):
            raise ValueError(
                f"Final CUI {name} source and target paths must be supplied together."
            )
    source_target_pairs = [
        (Path(source_robot_path), Path(final_robot_path)),
        (Path(source_pf_path), Path(final_pf_path)),
        (Path(source_pf_labeled_path), Path(final_pf_labeled_path)),
    ]
    source_target_pairs.extend(
        (Path(source), Path(target))
        for source, target, _ in optional_pairs
        if source is not None and target is not None
    )
    missing = [source for source, _ in source_target_pairs if not source.is_file()]
    if missing:
        raise RuntimeError(
            "Final CUI split views are missing: "
            + ", ".join(path.as_posix() for path in missing)
        )

    staged_paths: list[tuple[Path, Path]] = []
    try:
        for source, target in source_target_pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.tmp-",
                dir=target.parent,
                delete=False,
            ) as handle:
                staged_path = Path(handle.name)
            try:
                shutil.copyfile(source, staged_path)
            except Exception:
                staged_path.unlink(missing_ok=True)
                raise
            staged_paths.append((staged_path, target))

        for staged_path, target in staged_paths:
            os.replace(staged_path, target)
    finally:
        for staged_path, _ in staged_paths:
            staged_path.unlink(missing_ok=True)


__all__ = [
    "prepare_final_visualization_frame",
    "publish_final_cui_split_views",
]
