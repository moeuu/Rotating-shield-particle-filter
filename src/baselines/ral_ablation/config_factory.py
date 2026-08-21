"""Build pure-PF RA-L trials over the shared adaptive runtime boundary."""

from __future__ import annotations

import csv
import json
import secrets
import shlex
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pf.atomic_io import atomic_write_json, atomic_write_text
from pf.profiles import enforce_pure_runtime_settings

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_ROOT = ROOT.parent / "Rotating-shield-simulation-runtime"
DEFAULT_RUNTIME_CONFIG = (
    DEFAULT_RUNTIME_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
DEFAULT_PF_CONFIG = ROOT / "configs" / "pf" / "pf_strict_3d.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ral_ablation"
DEFAULT_PRIVATE_ROOT = DEFAULT_RUNTIME_ROOT / "private_runs" / "ral_ablation"
DEFAULT_SOURCE_PROFILE = "ral-mix9"
MAX_FRESH_ABLATION_SEED = (1 << 48) - 18


def generate_fresh_ablation_seed() -> int:
    """Return a fresh JSON-safe seed for one independent RA-L batch."""
    return 1 + secrets.randbelow(MAX_FRESH_ABLATION_SEED)


def _json_integer(value: object, *, name: str, minimum: int = 0) -> int:
    """Return an exact JSON integer satisfying an inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _nonempty_string(value: object, *, name: str) -> str:
    """Return one nonempty string without accepting implicit conversion."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _safe_suffix(value: str) -> str:
    """Validate and normalize an optional output tag suffix."""
    if not isinstance(value, str):
        raise TypeError("output_tag_suffix must be a string.")
    normalized = value.strip().strip("_")
    if normalized and any(
        not (character.isalnum() or character in {"-", "_"}) for character in normalized
    ):
        raise ValueError(
            "output_tag_suffix may contain only letters, digits, '-' and '_'."
        )
    return normalized


def resolve_ablation_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    """Resolve replay seeds or create one fresh comparison-scene seed."""
    if seeds is None:
        return (generate_fresh_ablation_seed(),)
    resolved = tuple(_json_integer(seed, name="seed") for seed in seeds)
    if not resolved:
        raise ValueError("seeds must contain at least one seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("seeds must not contain duplicate scene seeds.")
    return resolved


@dataclass(frozen=True, slots=True)
class AblationCase:
    """Describe one runtime-authored RA-L source profile."""

    name: str
    description: str
    source_profile: str
    isotope_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Validate one case declaration before plan generation."""
        _nonempty_string(self.name, name="AblationCase.name")
        _nonempty_string(self.description, name="AblationCase.description")
        _nonempty_string(self.source_profile, name="AblationCase.source_profile")
        if not self.isotope_counts:
            raise ValueError("AblationCase.isotope_counts must not be empty.")
        names: list[str] = []
        for isotope, count in self.isotope_counts:
            names.append(_nonempty_string(isotope, name="isotope"))
            _json_integer(count, name="isotope count", minimum=1)
        if len(set(names)) != len(names):
            raise ValueError("AblationCase isotope names must be unique.")


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """Describe PF-policy and physical-runtime overrides for one variant."""

    name: str
    description: str
    pf_overrides: Mapping[str, Any]
    runtime_overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate one variant declaration."""
        _nonempty_string(self.name, name="AblationVariant.name")
        _nonempty_string(self.description, name="AblationVariant.description")
        if not isinstance(self.pf_overrides, Mapping):
            raise TypeError("AblationVariant.pf_overrides must be a mapping.")
        if not isinstance(self.runtime_overrides, Mapping):
            raise TypeError("AblationVariant.runtime_overrides must be a mapping.")


@dataclass(frozen=True, slots=True)
class AblationPlanEntry:
    """Store one causal acquisition and PF-control trial."""

    case: str
    variant: str
    seed: int
    pf_seed: int
    seed_policy: str
    source_profile: str
    pf_config_path: Path
    runtime_config_path: Path
    scenario_path: Path
    measurement_log_path: Path
    pf_output_dir: Path
    scenario_command: tuple[str, ...]
    pf_command: tuple[str, ...]


DEFAULT_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        name="mix9_multi_isotope_cardinality",
        description="4 Cs-137, 3 Co-60, and 2 Eu-154 surface sources.",
        source_profile=DEFAULT_SOURCE_PROFILE,
        isotope_counts=(("Cs-137", 4), ("Co-60", 3), ("Eu-154", 2)),
    ),
)

DEFAULT_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        name="proposed",
        description="Full proposed temporal shield program and DSS-PP.",
        pf_overrides={},
        runtime_overrides={},
    ),
    AblationVariant(
        name="baseline_passive_equal_time_no_shield",
        description="Passive equal-time path with physically absent shields.",
        pf_overrides={
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
        },
        runtime_overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
        },
    ),
    AblationVariant(
        name="round_robin_shield",
        description="Cycle Fe/Pb pairs without posterior-dependent selection.",
        pf_overrides={
            "baseline_shield_policy": {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": True,
            },
        },
        runtime_overrides={},
    ),
    AblationVariant(
        name="eig_only_path",
        description="Retain EIG while removing optional route geometry terms.",
        pf_overrides={
            "dss_pp": {
                "coverage_weight": 0.0,
                "bearing_diversity_weight": 0.0,
                "frontier_weight": 0.0,
                "local_orbit_weight": 0.0,
                "elevation_condition_weight": 0.0,
                "revisit_penalty_weight": 0.0,
                "turn_smoothness_weight": 0.0,
            },
        },
        runtime_overrides={},
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object used as a generated-config base."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _deep_update(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a recursive mapping merge without mutating either input."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _pf_config(
    base: Mapping[str, Any],
    *,
    case: AblationCase,
    variant: AblationVariant,
    seed: int,
    seed_policy: str,
) -> dict[str, Any]:
    """Return one pure-PF policy config with reproducibility metadata."""
    config = _deep_update(base, variant.pf_overrides)
    metadata = config.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("PF metadata must be an object when present.")
    config["metadata"] = {
        **metadata,
        "ral_ablation_case": case.name,
        "ral_ablation_variant": variant.name,
        "ral_scene_seed": seed,
        "ral_scene_seed_policy": seed_policy,
        "ral_source_profile": case.source_profile,
    }
    config = enforce_pure_runtime_settings(config, profile="pf_strict")
    if config.get("variable_cardinality") is not True:
        raise ValueError("RA-L requires variable_cardinality=true.")
    return config


def _runtime_config(
    base_path: Path,
    *,
    variant: AblationVariant,
    seed: int,
) -> dict[str, Any]:
    """Return a private runtime override that inherits canonical physics."""
    payload: dict[str, Any] = {
        "extends": base_path.resolve().as_posix(),
        "random_seed_base": seed,
        "primary_sampling_fraction": 1.0,
        "accelerated_weighted_transport_enable": False,
        "target_sampled_primaries": None,
    }
    payload.update(variant.runtime_overrides)
    return payload


def _scenario_command(
    *,
    runtime_root: Path,
    scenario_path: Path,
    measurement_log_path: Path,
    run_id: str,
    runtime_config_path: Path,
    scene_seed: int,
    source_profile: str,
) -> tuple[str, ...]:
    """Return the shared-runtime private-scenario authoring command."""
    return (
        "uv",
        "run",
        "--directory",
        runtime_root.as_posix(),
        "rotating-shield-sim",
        "generate-ral-scenario",
        scenario_path.as_posix(),
        "--measurement-log-output",
        measurement_log_path.as_posix(),
        "--run-id",
        run_id,
        "--runtime-config",
        runtime_config_path.as_posix(),
        "--scene-seed",
        str(scene_seed),
        "--source-profile",
        source_profile,
    )


def _pf_command(
    *,
    scenario_path: Path,
    runtime_root: Path,
    pf_config_path: Path,
    pf_output_dir: Path,
    pf_seed: int,
    source_profile: str,
) -> tuple[str, ...]:
    """Return the PF-owned causal acquisition command."""
    return (
        "uv",
        "run",
        "--directory",
        ROOT.as_posix(),
        "rotating-shield-pf-live",
        "--scenario",
        scenario_path.as_posix(),
        "--runtime-root",
        runtime_root.as_posix(),
        "--config",
        pf_config_path.as_posix(),
        "--output-dir",
        pf_output_dir.as_posix(),
        "--profile",
        "pf_strict",
        "--seed",
        str(pf_seed),
        "--private-scene-profile",
        source_profile,
    )


def build_ablation_plan(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG,
    pf_config_path: Path = DEFAULT_PF_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    seeds: Sequence[int] | None = None,
    cases: Sequence[AblationCase] = DEFAULT_ABLATION_CASES,
    variants: Sequence[AblationVariant] = DEFAULT_ABLATION_VARIANTS,
    output_tag_suffix: str = "",
) -> list[AblationPlanEntry]:
    """Build isolated causal sessions using one scene seed per batch."""
    runtime_root = Path(runtime_root).expanduser().resolve()
    runtime_config_path = Path(runtime_config_path).expanduser().resolve()
    pf_config_path = Path(pf_config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    private_root = Path(private_root).expanduser().resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"Shared runtime root does not exist: {runtime_root}")
    if not runtime_config_path.is_file():
        raise FileNotFoundError(
            f"Shared runtime config does not exist: {runtime_config_path}"
        )
    pf_base = _load_json(pf_config_path)
    seed_policy = "fresh_per_batch" if seeds is None else "explicit_replay"
    suffix = _safe_suffix(output_tag_suffix)
    entries: list[AblationPlanEntry] = []
    resolved_seeds = resolve_ablation_seeds(seeds)
    for case in cases:
        for seed in resolved_seeds:
            for variant in variants:
                tag = f"{case.name}_{variant.name}_seed_{seed}"
                if suffix:
                    tag = f"{tag}_{suffix}"
                generated_pf_path = output_dir / "configs" / f"{tag}.json"
                generated_runtime_path = (
                    private_root / "runtime_configs" / f"{tag}.json"
                )
                scenario_path = private_root / "scenarios" / f"{tag}.json"
                log_path = output_dir / "measurement_logs" / tag
                pf_output = output_dir / "runs" / tag
                atomic_write_json(
                    generated_pf_path,
                    _pf_config(
                        pf_base,
                        case=case,
                        variant=variant,
                        seed=seed,
                        seed_policy=seed_policy,
                    ),
                )
                atomic_write_json(
                    generated_runtime_path,
                    _runtime_config(
                        runtime_config_path,
                        variant=variant,
                        seed=seed,
                    ),
                )
                scenario_command = _scenario_command(
                    runtime_root=runtime_root,
                    scenario_path=scenario_path,
                    measurement_log_path=log_path,
                    run_id=tag,
                    runtime_config_path=generated_runtime_path,
                    scene_seed=seed,
                    source_profile=case.source_profile,
                )
                pf_seed = seed
                entries.append(
                    AblationPlanEntry(
                        case=case.name,
                        variant=variant.name,
                        seed=seed,
                        pf_seed=pf_seed,
                        seed_policy=seed_policy,
                        source_profile=case.source_profile,
                        pf_config_path=generated_pf_path,
                        runtime_config_path=generated_runtime_path,
                        scenario_path=scenario_path,
                        measurement_log_path=log_path,
                        pf_output_dir=pf_output,
                        scenario_command=scenario_command,
                        pf_command=_pf_command(
                            scenario_path=scenario_path,
                            runtime_root=runtime_root,
                            pf_config_path=generated_pf_path,
                            pf_output_dir=pf_output,
                            pf_seed=pf_seed,
                            source_profile=case.source_profile,
                        ),
                    )
                )
    return entries


MANIFEST_FIELDS = (
    "case",
    "variant",
    "seed",
    "pf_seed",
    "seed_policy",
    "source_profile",
    "pf_config_path",
    "runtime_config_path",
    "scenario_path",
    "measurement_log_path",
    "pf_output_dir",
    "scenario_command",
    "pf_command",
)


def _entry_row(entry: AblationPlanEntry) -> dict[str, object]:
    """Return one CSV-safe manifest row."""
    return {
        "case": entry.case,
        "variant": entry.variant,
        "seed": entry.seed,
        "pf_seed": entry.pf_seed,
        "seed_policy": entry.seed_policy,
        "source_profile": entry.source_profile,
        "pf_config_path": entry.pf_config_path.as_posix(),
        "runtime_config_path": entry.runtime_config_path.as_posix(),
        "scenario_path": entry.scenario_path.as_posix(),
        "measurement_log_path": entry.measurement_log_path.as_posix(),
        "pf_output_dir": entry.pf_output_dir.as_posix(),
        "scenario_command": shlex.join(entry.scenario_command),
        "pf_command": shlex.join(entry.pf_command),
    }


def write_ablation_plan(
    entries: Sequence[AblationPlanEntry],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Atomically write the exhaustive CSV manifest and run script."""
    import io

    output_dir = Path(output_dir).expanduser().resolve()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow(_entry_row(entry))
    manifest_path = output_dir / "manifest.csv"
    atomic_write_text(manifest_path, buffer.getvalue())
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for entry in entries:
        lines.append(shlex.join(entry.scenario_command))
        lines.append(shlex.join(entry.pf_command))
    script_path = output_dir / "run_all.sh"
    atomic_write_text(script_path, "\n".join(lines) + "\n")
    mode = script_path.stat().st_mode
    script_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return manifest_path, script_path


__all__ = [
    "DEFAULT_ABLATION_CASES",
    "DEFAULT_ABLATION_VARIANTS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PF_CONFIG",
    "DEFAULT_PRIVATE_ROOT",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_RUNTIME_ROOT",
    "MANIFEST_FIELDS",
    "AblationCase",
    "AblationPlanEntry",
    "AblationVariant",
    "build_ablation_plan",
    "generate_fresh_ablation_seed",
    "resolve_ablation_seeds",
    "write_ablation_plan",
]
