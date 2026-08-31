"""Build the four-run RA-L paper subset from the exhaustive manifest."""

# ruff: noqa: E402  # Repository-local imports require the src path bootstrap.

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim.runtime import load_production_runtime_config

from baselines.ral_ablation.config_factory import (
    AblationVariant,
    DEFAULT_ABLATION_VARIANTS,
    DEFAULT_PRIVATE_ROOT,
    MANIFEST_FIELDS,
    MAX_FRESH_ABLATION_SEED,
    RAL_CASE_NAME,
    RAL_EXPERIMENT_PROFILE_ID,
    RAL_SCENE_VARIANT_ID,
)
from baselines.ral_ablation.control_policy import (
    load_ral_control_policy_document,
    validate_ral_control_policy_pf_settings,
)
from runtime.artifacts import atomic_write_text
from pf.configuration import load_pf_config
from pf.profiles import enforce_pure_runtime_settings

DEFAULT_FULL_MANIFEST = DEFAULT_PRIVATE_ROOT / "manifest.csv"
DEFAULT_SUBSET_MANIFEST = DEFAULT_PRIVATE_ROOT / "ral_paper_subset_manifest.csv"
DEFAULT_RUN_SCRIPT = DEFAULT_PRIVATE_ROOT / "run_paper_subset.sh"
CORE_VARIANTS = tuple(variant.name for variant in DEFAULT_ABLATION_VARIANTS)


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


def _positive_decimal_integer(value: str, *, name: str) -> int:
    """Parse one canonical positive decimal integer without coercion."""
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a canonical positive decimal integer.")
    parsed = int(value)
    if parsed < 1 or parsed > MAX_FRESH_ABLATION_SEED or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical positive decimal integer.")
    return parsed


def _safe_identifier(value: str, *, name: str) -> str:
    """Validate one nonempty identifier used to bind private artifacts."""
    if (
        not value
        or value.strip() != value
        or any(
            not (character.isascii() and (character.isalnum() or character in "-_"))
            for character in value
        )
    ):
        raise ValueError(f"{name} may contain only ASCII letters, digits, '-' and '_'.")
    return value


def _sha256_digest(value: str, *, name: str) -> str:
    """Validate one exact lowercase SHA-256 manifest field."""
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _declared_variant(name: str) -> AblationVariant:
    """Return one exact production variant contract by name."""
    matches = [variant for variant in DEFAULT_ABLATION_VARIANTS if variant.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate RA-L variant contract {name!r}.")
    return matches[0]


def _deep_update(
    base: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Apply one declared nested variant delta to a copied configuration."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
        "--run-id": row["run_id"],
        "--runtime-config": row["runtime_config_path"],
        "--scene-seed": row["scene_seed"],
        "--experiment-profile": row["experiment_profile_id"],
        "--scene-variant": row["scene_variant_id"],
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
    """Validate the private runner while keeping its PF child truth-free."""
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
        "--truth-manifest": row["truth_manifest_path"],
        "--runtime-root": runtime_root,
        "--pf-config": row["pf_config_path"],
        "--control-policy": row["control_policy_path"],
        "--expected-control-policy-sha256": row["control_policy_sha256"],
        "--pf-output-dir": row["pf_output_dir"],
        "--pf-seed": row["pf_seed"],
    }
    for option, expected in expected_options.items():
        actual = _required_option(tokens, option)
        equal = actual == expected
        if option in {
            "--scenario",
            "--truth-manifest",
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
        "--batch-id",
        default=None,
        help="Select one opaque recorded batch from a multi-batch manifest.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the current shared-runtime RA-L manifest schema."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(
                "Manifest header must exactly match the current schema in order."
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest must contain at least one trial row.")
    if any(
        None in row or any(row[field] is None for field in MANIFEST_FIELDS)
        for row in rows
    ):
        raise ValueError("Manifest contains a malformed row.")
    return [{field: str(row[field]) for field in MANIFEST_FIELDS} for row in rows]


def selected_variants_for_case(case: str) -> tuple[str, ...]:
    """Return the compact paper variants for one declared case."""
    return CORE_VARIANTS if str(case) == RAL_CASE_NAME else ()


def select_paper_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    batch_id: str | None = None,
) -> list[dict[str, str]]:
    """Select and validate four causal sessions from one exact batch."""
    available_batch_ids = sorted(
        {str(row["batch_id"]) for row in rows if str(row["case"]) == RAL_CASE_NAME}
    )
    if batch_id is None:
        if len(available_batch_ids) != 1:
            raise ValueError(
                "Paper manifest must contain exactly one batch_id when "
                f"--batch-id is omitted; found {available_batch_ids}."
            )
        batch_id = available_batch_ids[0]
    elif batch_id not in available_batch_ids:
        raise ValueError(f"Unknown RA-L batch_id {batch_id!r}.")
    _safe_identifier(batch_id, name="batch_id")
    wanted_order = tuple(
        (case, variant)
        for case in (RAL_CASE_NAME,)
        for variant in selected_variants_for_case(case)
    )
    wanted = set(wanted_order)
    selected = [
        {field: str(row[field]) for field in MANIFEST_FIELDS}
        for row in rows
        if str(row["batch_id"]) == batch_id
    ]
    unexpected = sorted(
        {
            (row["case"], row["variant"])
            for row in selected
            if (row["case"], row["variant"]) not in wanted
        }
    )
    if unexpected:
        raise ValueError(
            f"Paper batch contains undeclared case/variant rows: {unexpected}."
        )
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
    if not selected:
        raise ValueError("Paper subset selection is empty.")
    shared_fields = (
        "batch_id",
        "case",
        "experiment_profile_id",
        "scene_variant_id",
        "scene_seed",
        "pf_seed",
        "transport_seed",
        "seed_policy",
    )
    for field in shared_fields:
        values = {row[field] for row in selected}
        if len(values) != 1:
            raise ValueError(
                f"Paper variants must share one exact {field}; found {sorted(values)}."
            )
    if selected[0]["case"] != RAL_CASE_NAME:
        raise ValueError("Paper batch must use the declared MIX-9 case.")
    if selected[0]["experiment_profile_id"] != RAL_EXPERIMENT_PROFILE_ID:
        raise ValueError("Paper batch has the wrong experiment profile.")
    if selected[0]["scene_variant_id"] != RAL_SCENE_VARIANT_ID:
        raise ValueError("Paper batch has the wrong private scene variant.")
    if selected[0]["seed_policy"] not in {
        "fresh_per_batch",
        "explicit_live_repeat",
    }:
        raise ValueError("Paper batch has an unknown seed_policy.")
    scene_seed = _positive_decimal_integer(
        selected[0]["scene_seed"],
        name="scene_seed",
    )
    pf_seed = _positive_decimal_integer(selected[0]["pf_seed"], name="pf_seed")
    transport_seed = _positive_decimal_integer(
        selected[0]["transport_seed"],
        name="transport_seed",
    )
    if len({scene_seed, pf_seed, transport_seed}) != 3:
        raise ValueError("Scene, PF, and transport seeds must be pairwise independent.")
    for path_field in (
        "scenario_path",
        "truth_manifest_path",
        "measurement_log_path",
        "pf_output_dir",
    ):
        values = [Path(row[path_field]).expanduser().resolve() for row in selected]
        if len(set(values)) != len(values):
            raise ValueError(f"Paper variants must use unique {path_field} values.")
    run_ids = [row["run_id"] for row in selected]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Paper variants must use unique run_id values.")
    for run_id in run_ids:
        _safe_identifier(run_id, name="run_id")
    for row in selected:
        _sha256_digest(
            row["control_policy_sha256"],
            name="control_policy_sha256",
        )
        row["scenario_command"] = _validated_scenario_command(row)
        row["session_command"] = _validated_session_command(row)
    return selected


def _validate_configs(
    row: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Require native Geant4 physics and one strict pure-PF configuration."""
    variant = _declared_variant(row["variant"])
    runtime_path = Path(row["runtime_config_path"]).expanduser().resolve()
    raw_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(raw_runtime, dict):
        raise TypeError("RA-L generated runtime config must be a JSON object.")
    runtime = load_production_runtime_config(runtime_path)
    for field, value in variant.runtime_overrides.items():
        if runtime[field] != value:
            raise ValueError(
                f"RA-L variant {row['variant']!r} has the wrong runtime "
                f"intervention for {field}."
            )
    if runtime.get("backend") != "geant4" or runtime.get("engine_mode") != "external":
        raise ValueError("RA-L runtime config must use external Geant4.")
    if runtime["primary_sampling_fraction"] != 1.0:
        raise ValueError("RA-L runtime config must use all native histories.")
    transport_seed = int(row["transport_seed"])
    if runtime.get("random_seed_base") != transport_seed:
        raise ValueError("RA-L transport seed differs from the private manifest.")
    if transport_seed in {int(row["scene_seed"]), int(row["pf_seed"])}:
        raise ValueError("RA-L transport seed must be independent from scene/PF.")
    pf_config = load_pf_config(Path(row["pf_config_path"])).config()
    enforce_pure_runtime_settings(pf_config, profile="pf_strict")
    control_path = Path(row["control_policy_path"]).expanduser().resolve()
    control_document = load_ral_control_policy_document(
        control_path,
        expected_source_sha256=row["control_policy_sha256"],
    )
    expected_control = {
        "schema_version": 2,
        "variant": variant.name,
        "shield_policy": (
            None if variant.shield_policy is None else dict(variant.shield_policy)
        ),
    }
    if control_document.payload() != expected_control:
        raise ValueError(
            f"RA-L variant {row['variant']!r} has the wrong control policy."
        )
    validate_ral_control_policy_pf_settings(control_document.policy(), pf_config)
    return runtime, pf_config


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


def write_run_script(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    manifest_path: Path,
) -> None:
    """Write scenario-authoring and PF-control commands for each trial."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in rows:
        lines.append(_validated_scenario_command(row))
    batch_ids = tuple(dict.fromkeys(row["batch_id"] for row in rows))
    if len(batch_ids) != 1:
        raise ValueError("RA-L paper run script requires exactly one batch_id.")
    contract_path = (
        Path(path).expanduser().resolve().parent
        / "batch_contracts"
        / f"{batch_ids[0]}.json"
    )
    lines.extend(
        (
            "",
            shlex.join(
                (
                    "uv",
                    "run",
                    "--directory",
                    ROOT.as_posix(),
                    "python",
                    "-m",
                    "baselines.ral_ablation.batch_contract",
                    "--manifest",
                    Path(manifest_path).expanduser().resolve().as_posix(),
                    "--batch-id",
                    batch_ids[0],
                    "--output",
                    contract_path.as_posix(),
                )
            ),
            "",
        )
    )
    for row in rows:
        lines.append(_validated_session_command(row))
    atomic_write_text(path, "\n".join(lines) + "\n")
    mode = Path(path).stat().st_mode
    Path(path).chmod((mode | stat.S_IXUSR) & ~stat.S_IRWXG & ~stat.S_IRWXO)


def build_subset(
    manifest_path: Path,
    subset_manifest_path: Path,
    run_script_path: Path,
    *,
    batch_id: str | None = None,
) -> list[dict[str, str]]:
    """Build and write the compact RA-L paper subset."""
    selected = select_paper_subset(_read_manifest(manifest_path), batch_id=batch_id)
    validated = {row["variant"]: _validate_configs(row) for row in selected}
    proposed_runtime = validated["proposed"][0]
    for variant in DEFAULT_ABLATION_VARIANTS:
        expected_runtime = _deep_update(
            proposed_runtime,
            variant.runtime_overrides,
        )
        if variant.runtime_overrides and expected_runtime == proposed_runtime:
            raise ValueError(
                f"RA-L variant {variant.name!r} has a no-op runtime intervention."
            )
        if validated[variant.name][0] != expected_runtime:
            raise ValueError(
                f"RA-L variant {variant.name!r} has an undeclared runtime "
                "difference."
            )
    proposed_config = validated["proposed"][1]
    for variant in DEFAULT_ABLATION_VARIANTS:
        expected = _deep_update(proposed_config, variant.pf_overrides)
        if variant.pf_overrides and expected == proposed_config:
            raise ValueError(
                f"RA-L variant {variant.name!r} has a no-op PF intervention."
            )
        if validated[variant.name][1] != expected:
            raise ValueError(
                f"RA-L variant {variant.name!r} has an undeclared PF difference."
            )
    write_manifest(subset_manifest_path, selected)
    write_run_script(
        run_script_path,
        selected,
        manifest_path=subset_manifest_path,
    )
    return selected


def main() -> None:
    """Run the paper-subset manifest builder."""
    args = _parse_args()
    selected = build_subset(
        args.manifest,
        args.output_manifest,
        args.output_script,
        batch_id=None if args.batch_id is None else str(args.batch_id),
    )
    print(f"Wrote {len(selected)} RA-L paper-subset trials.")
    print(f"Manifest: {args.output_manifest}")
    print(f"Run script: {args.output_script}")


if __name__ == "__main__":
    main()
