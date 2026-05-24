"""Build the RA-L paper ablation subset from the exhaustive manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_MANIFEST = ROOT / "results" / "ral_ablation" / "manifest.csv"
DEFAULT_SUBSET_MANIFEST = (
    ROOT / "results" / "ral_ablation" / "ral_paper_subset_manifest.csv"
)
DEFAULT_RUN_SCRIPT = ROOT / "results" / "ral_ablation" / "run_paper_subset.sh"
DEFAULT_SEED = "2026050901"
CORE_VARIANTS = (
    "proposed",
    "baseline_passive_no_shield",
    "round_robin_shield",
    "one_step_path",
)
CASE_EXTRA_VARIANTS = {
    "case03_mixed_cardinality": ("no_residual_birth",),
}
MANIFEST_FIELDS = ("case", "variant", "seed", "config_path", "source_path", "command")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the compact RA-L paper ablation manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_FULL_MANIFEST,
        help="Path to the exhaustive RA-L ablation manifest.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_SUBSET_MANIFEST,
        help="Path for the compact paper-subset manifest.",
    )
    parser.add_argument(
        "--output-script",
        type=Path,
        default=DEFAULT_RUN_SCRIPT,
        help="Path for the compact paper-subset run script.",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help="Single seed to use for the RA-L paper subset.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read an ablation manifest CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        missing = [field for field in MANIFEST_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Manifest row is missing fields: {missing}")
    return [{field: str(row[field]) for field in MANIFEST_FIELDS} for row in rows]


def selected_variants_for_case(case: str) -> tuple[str, ...]:
    """Return the compact RA-L paper variants for one case."""
    return CORE_VARIANTS + tuple(CASE_EXTRA_VARIANTS.get(str(case), ()))


def select_paper_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: str = DEFAULT_SEED,
) -> list[dict[str, str]]:
    """Select the compact paper subset while preserving manifest order."""
    seed = str(seed)
    cases = tuple(dict.fromkeys(row["case"] for row in rows if row["seed"] == seed))
    wanted = {
        (case, variant)
        for case in cases
        for variant in selected_variants_for_case(case)
    }
    selected = [
        {field: str(row[field]) for field in MANIFEST_FIELDS}
        for row in rows
        if row["seed"] == seed and (row["case"], row["variant"]) in wanted
    ]
    found = {(row["case"], row["variant"]) for row in selected}
    missing = sorted(wanted - found)
    if missing:
        formatted = ", ".join(f"{case}:{variant}" for case, variant in missing)
        raise ValueError(f"Full manifest is missing paper-subset entries: {formatted}")
    return selected


def write_manifest(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write a deterministic subset manifest CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})


def write_run_script(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write a shell script for the selected paper-subset commands."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(row["command"] for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_subset(
    manifest_path: Path,
    subset_manifest_path: Path,
    run_script_path: Path,
    *,
    seed: str = DEFAULT_SEED,
) -> list[dict[str, str]]:
    """Build and write the compact RA-L paper subset."""
    rows = _read_manifest(manifest_path)
    selected = select_paper_subset(rows, seed=seed)
    write_manifest(subset_manifest_path, selected)
    write_run_script(run_script_path, selected)
    return selected


def main() -> None:
    """Run the paper-subset manifest builder."""
    args = _parse_args()
    selected = build_subset(
        args.manifest,
        args.output_manifest,
        args.output_script,
        seed=str(args.seed),
    )
    print(f"Wrote {len(selected)} RA-L paper-subset trials.")
    print(f"Manifest: {args.output_manifest}")
    print(f"Run script: {args.output_script}")


if __name__ == "__main__":
    main()
