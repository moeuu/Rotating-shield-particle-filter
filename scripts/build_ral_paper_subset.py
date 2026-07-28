"""Build the RA-L paper ablation subset from the exhaustive manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from pathlib import Path
import shlex
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.runtime import load_runtime_config  # noqa: E402

DEFAULT_FULL_MANIFEST = ROOT / "results" / "ral_ablation" / "manifest.csv"
DEFAULT_SUBSET_MANIFEST = (
    ROOT / "results" / "ral_ablation" / "ral_paper_subset_manifest.csv"
)
DEFAULT_RUN_SCRIPT = ROOT / "results" / "ral_ablation" / "run_paper_subset.sh"
DEFAULT_SEED = "2026050901"
PAPER_CASES = ("mix9_multi_isotope_cardinality",)
CORE_VARIANTS = (
    "proposed",
    "baseline_passive_equal_time_no_shield",
    "round_robin_shield",
    "eig_only_path",
)
MANIFEST_FIELDS = ("case", "variant", "seed", "config_path", "source_path", "command")
_MODE_FLAGS = frozenset(
    {
        "--mode",
        "--gui",
        "--cui",
        "--python-gui",
        "--geant4-isaacsim-gui",
        "--python-cui",
        "--geant4-cui",
        "--standard-geant4-full",
        "--sim-backend",
    }
)


def _required_option_value(tokens: Sequence[str], option: str) -> str:
    """Return one required, uniquely specified CLI option value."""
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"RA-L command {option} requires one value.")
            values.append(tokens[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"RA-L command requires exactly one {option}; got {len(values)}."
        )
    return values[0]


def _validated_full_simulation_command(row: Mapping[str, str]) -> str:
    """Return one shell-safe canonical RA-L Geant4 command or fail closed."""
    command = str(row["command"])
    if "\n" in command or "\r" in command:
        raise ValueError("RA-L command must occupy exactly one line.")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"RA-L command is not valid shell syntax: {exc}") from exc
    if tokens[:4] != ["uv", "run", "python", "main.py"]:
        raise ValueError(
            "RA-L command must start with 'uv run python main.py'."
        )
    if tokens.count("--full-simulation") != 1:
        raise ValueError(
            "RA-L paper commands require exactly one --full-simulation flag."
        )
    conflicting_flags = sorted(
        {
            token.split("=", 1)[0]
            for token in tokens[4:]
            if token.split("=", 1)[0] in _MODE_FLAGS
        }
    )
    if conflicting_flags:
        raise ValueError(
            "RA-L full-simulation command contains conflicting mode/backend "
            f"flags: {conflicting_flags}."
        )
    expected_values = {
        "--sim-config": str(row["config_path"]),
        "--source-config": str(row["source_path"]),
        "--output-tag": (
            f"{row['case']}_{row['variant']}_seed_{row['seed']}"
        ),
    }
    for option, expected in expected_values.items():
        actual = _required_option_value(tokens, option)
        if Path(actual).as_posix() != Path(expected).as_posix():
            raise ValueError(
                f"RA-L command {option}={actual!r} does not match manifest "
                f"value {expected!r}."
            )
    return shlex.join(tokens)


def _validate_geant4_config(row: Mapping[str, str]) -> None:
    """Require the selected paper config to resolve to native external Geant4."""
    config_path = Path(str(row["config_path"])).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ValueError(f"RA-L config does not exist: {config_path}.")
    config = load_runtime_config(config_path)
    if (
        str(config.get("backend", "")).strip().lower() != "geant4"
        or str(config.get("engine_mode", "")).strip().lower() != "external"
    ):
        raise ValueError(
            "RA-L paper configs require backend='geant4' and "
            f"engine_mode='external': {config_path}."
        )


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
    if str(case) not in PAPER_CASES:
        return ()
    return CORE_VARIANTS


def select_paper_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: str = DEFAULT_SEED,
) -> list[dict[str, str]]:
    """Select the compact paper subset while preserving manifest order."""
    seed = str(seed)
    cases = tuple(
        case
        for case in PAPER_CASES
        if any(row["case"] == case and row["seed"] == seed for row in rows)
    )
    if not cases:
        formatted_cases = ", ".join(PAPER_CASES)
        raise ValueError(
            f"Full manifest has no paper cases for seed {seed}: {formatted_cases}"
        )
    wanted = {
        (case, variant)
        for case in cases
        for variant in selected_variants_for_case(case)
    }
    selected_unsorted = [
        {field: str(row[field]) for field in MANIFEST_FIELDS}
        for row in rows
        if row["seed"] == seed and (row["case"], row["variant"]) in wanted
    ]
    order = {
        (case, variant): index
        for case in cases
        for index, variant in enumerate(selected_variants_for_case(case))
    }
    selected = sorted(
        selected_unsorted,
        key=lambda row: order[(row["case"], row["variant"])],
    )
    found = {(row["case"], row["variant"]) for row in selected}
    missing = sorted(wanted - found)
    if missing:
        formatted = ", ".join(f"{case}:{variant}" for case, variant in missing)
        raise ValueError(f"Full manifest is missing paper-subset entries: {formatted}")
    for row in selected:
        row["command"] = _validated_full_simulation_command(row)
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
    lines.extend(_validated_full_simulation_command(row) for row in rows)
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
    for row in selected:
        _validate_geant4_config(row)
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
