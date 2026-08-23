"""Build the four-run RA-L paper subset from the exhaustive manifest."""

# ruff: noqa: E402  # Repository-local imports require the src path bootstrap.

from __future__ import annotations

import argparse
import csv
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.runtime import load_runtime_config

from baselines.ral_ablation.config_factory import (
    DEFAULT_PRIVATE_ROOT,
    MANIFEST_FIELDS,
)
from baselines.ral_ablation.control_policy import load_ral_control_policy
from pf.atomic_io import atomic_write_text
from pf.configuration import load_pf_config
from pf.profiles import enforce_pure_runtime_settings

DEFAULT_FULL_MANIFEST = DEFAULT_PRIVATE_ROOT / "manifest.csv"
DEFAULT_SUBSET_MANIFEST = DEFAULT_PRIVATE_ROOT / "ral_paper_subset_manifest.csv"
DEFAULT_RUN_SCRIPT = DEFAULT_PRIVATE_ROOT / "run_paper_subset.sh"
PAPER_CASES = ("mix9_multi_isotope_cardinality",)
CORE_VARIANTS = (
    "proposed",
    "baseline_passive_equal_time_no_shield",
    "round_robin_shield",
    "eig_only_path",
)


def _required_option(tokens: Sequence[str], option: str) -> str:
    """Return one uniquely declared command-line option value."""
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"RA-L command {option} requires a value.")
            values.append(tokens[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"RA-L command requires exactly one {option}.")
    return values[0]


def _tokens(command: str, *, name: str) -> list[str]:
    """Parse one single-line shell command or fail closed."""
    if "\n" in command or "\r" in command:
        raise ValueError(f"{name} must occupy one line.")
    try:
        return shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"{name} has invalid shell syntax: {exc}") from exc


def _same_path(actual: str, expected: str) -> bool:
    """Return whether two manifest path spellings identify the same path."""
    return Path(actual).expanduser().resolve() == Path(expected).expanduser().resolve()


def _validated_scenario_command(row: Mapping[str, str]) -> str:
    """Validate and canonicalize one shared-runtime scenario command."""
    tokens = _tokens(str(row["scenario_command"]), name="scenario_command")
    if len(tokens) < 7 or tokens[:2] != ["uv", "run"]:
        raise ValueError("RA-L scenario command must start with 'uv run'.")
    if "rotating-shield-sim" not in tokens:
        raise ValueError("RA-L scenario command must use rotating-shield-sim.")
    executable_index = tokens.index("rotating-shield-sim")
    expected_prefix = ["rotating-shield-sim", "generate-scenario"]
    if tokens[executable_index : executable_index + 2] != expected_prefix:
        raise ValueError("RA-L scenario command must generate a private scenario.")
    scenario_index = executable_index + 2
    if scenario_index >= len(tokens) or not _same_path(
        tokens[scenario_index], row["scenario_path"]
    ):
        raise ValueError("scenario_command path differs from the manifest.")
    expected_options = {
        "--truth-manifest-output": row["truth_manifest_path"],
        "--measurement-log-output": row["measurement_log_path"],
        "--runtime-config": row["runtime_config_path"],
        "--scene-seed": row["scene_seed"],
    }
    for option, expected in expected_options.items():
        actual = _required_option(tokens, option)
        equal = actual == expected
        if option in {
            "--truth-manifest-output",
            "--measurement-log-output",
            "--runtime-config",
        }:
            equal = _same_path(actual, expected)
        if not equal:
            raise ValueError(f"scenario_command {option} differs from the manifest.")
    return shlex.join(tokens)


def _validated_session_command(row: Mapping[str, str]) -> str:
    """Validate one adapter command while keeping PF inputs truth-free."""
    tokens = _tokens(str(row["session_command"]), name="session_command")
    expected_module = "baselines.ral_ablation.session_runner"
    if tokens[:2] != ["uv", "run"] or expected_module not in tokens:
        raise ValueError("RA-L session command must use the isolated adapter runner.")
    scenario_tokens = _tokens(
        str(row["scenario_command"]),
        name="scenario_command",
    )
    runtime_root = _required_option(scenario_tokens, "--directory")
    expected_options = {
        "--scenario": row["scenario_path"],
        "--runtime-root": runtime_root,
        "--pf-config": row["pf_config_path"],
        "--control-policy": row["control_policy_path"],
        "--pf-output-dir": row["pf_output_dir"],
        "--pf-seed": row["pf_seed"],
    }
    for option, expected in expected_options.items():
        actual = _required_option(tokens, option)
        equal = actual == expected
        if option in {
            "--scenario",
            "--runtime-root",
            "--pf-config",
            "--control-policy",
            "--pf-output-dir",
        }:
            equal = _same_path(actual, expected)
        if not equal:
            raise ValueError(f"session_command {option} differs from the manifest.")
    forbidden = {
        "--private-scene-profile",
        "--scene-seed",
        "--scene-variant",
        "--truth-manifest-output",
    }
    leaked = sorted(forbidden.intersection(tokens))
    if leaked:
        raise ValueError(f"session_command exposes private truth inputs: {leaked}")
    return shlex.join(tokens)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the compact RA-L paper ablation manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FULL_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--output-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument(
        "--seed",
        default=None,
        help="Select one recorded scene seed from a multi-batch manifest.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the current shared-runtime RA-L manifest schema."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in MANIFEST_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest is missing current fields: {missing}")
        return [{field: str(row[field]) for field in MANIFEST_FIELDS} for row in reader]


def selected_variants_for_case(case: str) -> tuple[str, ...]:
    """Return the compact paper variants for one declared case."""
    return CORE_VARIANTS if str(case) in PAPER_CASES else ()


def select_paper_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: str | None = None,
) -> list[dict[str, str]]:
    """Select and validate the four causal sessions for one scene seed."""
    available_seeds = sorted(
        {str(row["scene_seed"]) for row in rows if str(row["case"]) in PAPER_CASES}
    )
    if seed is None:
        if len(available_seeds) != 1:
            raise ValueError(
                "Paper manifest must contain exactly one scene seed when "
                f"--seed is omitted; found {available_seeds}."
            )
        seed = available_seeds[0]
    wanted_order = tuple(
        (case, variant)
        for case in PAPER_CASES
        for variant in selected_variants_for_case(case)
    )
    wanted = set(wanted_order)
    selected = [
        {field: str(row[field]) for field in MANIFEST_FIELDS}
        for row in rows
        if str(row["scene_seed"]) == str(seed)
        and (str(row["case"]), str(row["variant"])) in wanted
    ]
    order = {pair: index for index, pair in enumerate(wanted_order)}
    selected.sort(key=lambda row: order[(row["case"], row["variant"])])
    found = {(row["case"], row["variant"]) for row in selected}
    missing = sorted(wanted - found)
    duplicates = len(selected) != len(found)
    if missing or duplicates:
        raise ValueError(
            "Paper subset requires exactly one row per case/variant; "
            f"missing={missing}, duplicates={duplicates}."
        )
    for row in selected:
        row["scenario_command"] = _validated_scenario_command(row)
        row["session_command"] = _validated_session_command(row)
    return selected


def _validate_configs(row: Mapping[str, str]) -> None:
    """Require native Geant4 physics and one strict pure-PF configuration."""
    runtime = load_runtime_config(Path(row["runtime_config_path"]))
    if runtime.get("backend") != "geant4" or runtime.get("engine_mode") != "external":
        raise ValueError("RA-L runtime config must use external Geant4.")
    if runtime.get("primary_sampling_fraction", 1.0) != 1.0:
        raise ValueError("RA-L runtime config must use all native histories.")
    if runtime.get("accelerated_weighted_transport_enable", False) is not False:
        raise ValueError("RA-L runtime config must not weight transport histories.")
    if runtime.get("target_sampled_primaries") is not None:
        raise ValueError("RA-L runtime config must not cap transport histories.")
    transport_seed = int(row["transport_seed"])
    if runtime.get("random_seed_base") != transport_seed:
        raise ValueError("RA-L transport seed differs from the private manifest.")
    if transport_seed in {int(row["scene_seed"]), int(row["pf_seed"])}:
        raise ValueError("RA-L transport seed must be independent from scene/PF.")
    pf_config, _ = load_pf_config(Path(row["pf_config_path"]))
    enforce_pure_runtime_settings(pf_config, profile="pf_strict")
    load_ral_control_policy(Path(row["control_policy_path"]))


def write_manifest(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Atomically write a deterministic subset manifest."""
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
    atomic_write_text(path, buffer.getvalue())
    Path(path).chmod(0o600)


def write_run_script(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Write scenario-authoring and PF-control commands for each trial."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in rows:
        lines.append(_validated_scenario_command(row))
        lines.append(_validated_session_command(row))
    atomic_write_text(path, "\n".join(lines) + "\n")
    mode = Path(path).stat().st_mode
    Path(path).chmod((mode | stat.S_IXUSR) & ~stat.S_IRWXG & ~stat.S_IRWXO)


def build_subset(
    manifest_path: Path,
    subset_manifest_path: Path,
    run_script_path: Path,
    *,
    seed: str | None = None,
) -> list[dict[str, str]]:
    """Build and write the compact RA-L paper subset."""
    selected = select_paper_subset(_read_manifest(manifest_path), seed=seed)
    for row in selected:
        _validate_configs(row)
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
        seed=None if args.seed is None else str(args.seed),
    )
    print(f"Wrote {len(selected)} RA-L paper-subset trials.")
    print(f"Manifest: {args.output_manifest}")
    print(f"Run script: {args.output_script}")


if __name__ == "__main__":
    main()
