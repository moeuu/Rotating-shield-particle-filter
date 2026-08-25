"""Publish stable visualization artifacts for completed PF runs."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def publish_final_cui_split_views(
    *,
    source_overview_path: Path,
    source_robot_path: Path,
    source_pf_path: Path,
    source_pf_labeled_path: Path,
    source_spectrum_path: Path,
    final_overview_path: Path,
    final_robot_path: Path,
    final_pf_path: Path,
    final_pf_labeled_path: Path,
    final_spectrum_path: Path,
) -> None:
    """Publish the complete five-panel CUI result as stable artifacts.

    Every source is validated and copied to a temporary file before any final
    path is replaced. A missing or unreadable source therefore leaves all
    existing result artifacts untouched.
    """
    source_target_pairs = [
        (Path(source_overview_path), Path(final_overview_path)),
        (Path(source_robot_path), Path(final_robot_path)),
        (Path(source_pf_path), Path(final_pf_path)),
        (Path(source_pf_labeled_path), Path(final_pf_labeled_path)),
        (Path(source_spectrum_path), Path(final_spectrum_path)),
    ]
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
    "publish_final_cui_split_views",
]
