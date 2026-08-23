"""Build isolated RA-L trials without exposing private truth to generic PF."""

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
    """Return a fresh private scene seed for one independent RA-L batch."""
    return 1 + secrets.randbelow(MAX_FRESH_ABLATION_SEED)


def generate_fresh_pf_seed() -> int:
    """Return a PF seed drawn independently from private scene generation."""
    return 1 + secrets.randbelow(MAX_FRESH_ABLATION_SEED)


def generate_fresh_transport_seed() -> int:
    """Return a transport seed independent from scene and PF randomness."""
    return 1 + secrets.randbelow(MAX_FRESH_ABLATION_SEED)


def generate_fresh_batch_id() -> str:
    """Return an opaque identifier that reveals no scene-generation input."""
    return secrets.token_hex(8)


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


def _safe_batch_id(value: object) -> str:
    """Validate one opaque batch identifier for use in artifact names."""
    normalized = _nonempty_string(value, name="batch_id")
    if any(
        not (character.isalnum() or character in {"-", "_"}) for character in normalized
    ):
        raise ValueError("batch_id may contain only letters, digits, '-' and '_'.")
    return normalized


def resolve_ablation_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    """Resolve private live seeds or create one fresh comparison seed."""
    if seeds is None:
        return (generate_fresh_ablation_seed(),)
    resolved = tuple(_json_integer(seed, name="seed") for seed in seeds)
    if not resolved:
        raise ValueError("seeds must contain at least one seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("seeds must not contain duplicate scene seeds.")
    return resolved


def resolve_pf_seeds(
    scene_seeds: Sequence[int],
    pf_seeds: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve one independent PF seed for each private scene seed."""
    scenes = tuple(scene_seeds)
    if pf_seeds is None:
        resolved: list[int] = []
        for scene_seed in scenes:
            pf_seed = generate_fresh_pf_seed()
            while pf_seed == scene_seed or pf_seed in resolved:
                pf_seed = generate_fresh_pf_seed()
            resolved.append(pf_seed)
        return tuple(resolved)
    resolved = tuple(_json_integer(seed, name="PF seed") for seed in pf_seeds)
    if len(resolved) != len(scenes):
        raise ValueError("pf_seeds must contain one value per scene seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("pf_seeds must not contain duplicates.")
    if any(pf_seed == scene_seed for pf_seed, scene_seed in zip(resolved, scenes)):
        raise ValueError("PF seeds must be independent from private scene seeds.")
    return resolved


def resolve_transport_seeds(
    scene_seeds: Sequence[int],
    pf_seeds: Sequence[int],
    transport_seeds: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve transport seeds that cannot reconstruct private scene truth."""
    scenes = tuple(scene_seeds)
    estimators = tuple(pf_seeds)
    if len(scenes) != len(estimators):
        raise ValueError("scene_seeds and pf_seeds must have equal length.")
    if transport_seeds is None:
        resolved: list[int] = []
        for scene_seed, pf_seed in zip(scenes, estimators):
            transport_seed = generate_fresh_transport_seed()
            while transport_seed in {scene_seed, pf_seed, *resolved}:
                transport_seed = generate_fresh_transport_seed()
            resolved.append(transport_seed)
        return tuple(resolved)
    resolved = tuple(
        _json_integer(seed, name="transport seed") for seed in transport_seeds
    )
    if len(resolved) != len(scenes):
        raise ValueError("transport_seeds must contain one value per scene seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("transport_seeds must not contain duplicates.")
    if any(
        transport_seed in {scene_seed, pf_seed}
        for transport_seed, scene_seed, pf_seed in zip(
            resolved,
            scenes,
            estimators,
        )
    ):
        raise ValueError("Transport seeds must be independent from scene/PF seeds.")
    return resolved


def resolve_batch_ids(
    count: int,
    batch_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve opaque artifact identifiers for private comparison batches."""
    if count < 1:
        raise ValueError("count must be positive.")
    if batch_ids is None:
        resolved: list[str] = []
        while len(resolved) < count:
            candidate = generate_fresh_batch_id()
            if candidate not in resolved:
                resolved.append(candidate)
        return tuple(resolved)
    resolved = tuple(_safe_batch_id(value) for value in batch_ids)
    if len(resolved) != count:
        raise ValueError("batch_ids must contain one value per scene seed and case.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("batch_ids must not contain duplicates.")
    return resolved


@dataclass(frozen=True, slots=True)
class AblationCase:
    """Describe one private runtime-authored RA-L source profile."""

    name: str
    description: str
    source_profile: str
    isotope_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Validate one private case declaration before plan generation."""
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
    """Describe separated PF, adapter, and physical-runtime interventions."""

    name: str
    description: str
    pf_overrides: Mapping[str, Any]
    runtime_overrides: Mapping[str, Any]
    path_policy: Mapping[str, Any] | None = None
    shield_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate one variant declaration and its separation boundary."""
        _nonempty_string(self.name, name="AblationVariant.name")
        _nonempty_string(self.description, name="AblationVariant.description")
        if not isinstance(self.pf_overrides, Mapping):
            raise TypeError("AblationVariant.pf_overrides must be a mapping.")
        if not isinstance(self.runtime_overrides, Mapping):
            raise TypeError("AblationVariant.runtime_overrides must be a mapping.")
        if self.path_policy is not None and not isinstance(self.path_policy, Mapping):
            raise TypeError("AblationVariant.path_policy must be a mapping or null.")
        if self.shield_policy is not None and not isinstance(
            self.shield_policy, Mapping
        ):
            raise TypeError("AblationVariant.shield_policy must be a mapping or null.")
        if self.path_policy is not None and self.shield_policy is None:
            raise ValueError("A fixed path variant requires an explicit shield policy.")


@dataclass(frozen=True, slots=True)
class AblationPlanEntry:
    """Store one private experiment session and truth-free PF invocation."""

    case: str
    variant: str
    batch_id: str
    scene_seed: int
    pf_seed: int
    transport_seed: int
    seed_policy: str
    source_profile: str
    pf_config_path: Path
    control_policy_path: Path
    runtime_config_path: Path
    scenario_path: Path
    truth_manifest_path: Path
    measurement_log_path: Path
    pf_output_dir: Path
    scenario_command: tuple[str, ...]
    session_command: tuple[str, ...]


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
        pf_overrides={},
        runtime_overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
        },
        path_policy={"name": "passive_serpentine", "row_count": 8},
        shield_policy={"name": "fixed", "fixed_pair_id": 0},
    ),
    AblationVariant(
        name="round_robin_shield",
        description="Cycle Fe/Pb pairs without posterior-dependent selection.",
        pf_overrides={},
        runtime_overrides={},
        shield_policy={
            "name": "round_robin",
            "start_pair_id": 0,
            "advance_by_pose": True,
        },
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
    variant: AblationVariant,
) -> dict[str, Any]:
    """Return a truth-free PF config containing no experiment metadata."""
    config = enforce_pure_runtime_settings(
        _deep_update(base, variant.pf_overrides),
        profile="pf_strict",
    )
    if config.get("variable_cardinality") is not True:
        raise ValueError("RA-L requires variable_cardinality=true.")
    return config


def _control_policy(variant: AblationVariant) -> dict[str, object]:
    """Return the separate RA-L adapter policy for one experiment variant."""
    return {
        "schema_version": 1,
        "path_policy": (
            None if variant.path_policy is None else dict(variant.path_policy)
        ),
        "shield_policy": (
            None if variant.shield_policy is None else dict(variant.shield_policy)
        ),
    }


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
    truth_manifest_path: Path,
    measurement_log_path: Path,
    run_id: str,
    runtime_config_path: Path,
    scene_seed: int,
    source_profile: str,
) -> tuple[str, ...]:
    """Return the private runtime scenario and truth-manifest command."""
    return (
        "uv",
        "run",
        "--directory",
        runtime_root.as_posix(),
        "rotating-shield-sim",
        "generate-ral-scenario",
        scenario_path.as_posix(),
        "--truth-manifest-output",
        truth_manifest_path.as_posix(),
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


def _session_command(
    *,
    scenario_path: Path,
    runtime_root: Path,
    pf_config_path: Path,
    control_policy_path: Path,
    pf_output_dir: Path,
    pf_seed: int,
) -> tuple[str, ...]:
    """Return the RA-L adapter command that isolates PF behind a socket."""
    return (
        "uv",
        "run",
        "--directory",
        ROOT.as_posix(),
        "python",
        "-m",
        "baselines.ral_ablation.session_runner",
        "--runtime-root",
        runtime_root.as_posix(),
        "--scenario",
        scenario_path.as_posix(),
        "--pf-config",
        pf_config_path.as_posix(),
        "--control-policy",
        control_policy_path.as_posix(),
        "--pf-output-dir",
        pf_output_dir.as_posix(),
        "--pf-seed",
        str(pf_seed),
    )


def build_ablation_plan(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG,
    pf_config_path: Path = DEFAULT_PF_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    seeds: Sequence[int] | None = None,
    pf_seeds: Sequence[int] | None = None,
    transport_seeds: Sequence[int] | None = None,
    batch_ids: Sequence[str] | None = None,
    cases: Sequence[AblationCase] = DEFAULT_ABLATION_CASES,
    variants: Sequence[AblationVariant] = DEFAULT_ABLATION_VARIANTS,
    output_tag_suffix: str = "",
) -> list[AblationPlanEntry]:
    """Build sessions whose PF-facing artifacts contain no private truth."""
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
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root.chmod(0o700)
    pf_base = _load_json(pf_config_path)
    seed_policy = "fresh_per_batch" if seeds is None else "explicit_live_repeat"
    suffix = _safe_suffix(output_tag_suffix)
    resolved_scene_seeds = resolve_ablation_seeds(seeds)
    resolved_pf_seeds = resolve_pf_seeds(resolved_scene_seeds, pf_seeds)
    resolved_transport_seeds = resolve_transport_seeds(
        resolved_scene_seeds,
        resolved_pf_seeds,
        transport_seeds,
    )
    batch_count = len(cases) * len(resolved_scene_seeds)
    resolved_batch_ids = iter(resolve_batch_ids(batch_count, batch_ids))
    entries: list[AblationPlanEntry] = []
    for case in cases:
        for scene_seed, pf_seed, transport_seed in zip(
            resolved_scene_seeds,
            resolved_pf_seeds,
            resolved_transport_seeds,
        ):
            batch_id = next(resolved_batch_ids)
            for variant in variants:
                tag = f"ral_{batch_id}_{variant.name}"
                if suffix:
                    tag = f"{tag}_{suffix}"
                generated_pf_path = output_dir / "configs" / f"{tag}.json"
                control_policy_path = output_dir / "control_policies" / f"{tag}.json"
                generated_runtime_path = (
                    private_root / "runtime_configs" / f"{tag}.json"
                )
                scenario_path = private_root / "scenarios" / f"{tag}.json"
                truth_manifest_path = private_root / "truth_manifests" / f"{tag}.json"
                log_path = output_dir / "measurement_logs" / tag
                pf_output = output_dir / "runs" / tag
                atomic_write_json(
                    generated_pf_path,
                    _pf_config(pf_base, variant=variant),
                )
                atomic_write_json(control_policy_path, _control_policy(variant))
                atomic_write_json(
                    generated_runtime_path,
                    _runtime_config(
                        runtime_config_path,
                        variant=variant,
                        seed=transport_seed,
                    ),
                )
                scenario_command = _scenario_command(
                    runtime_root=runtime_root,
                    scenario_path=scenario_path,
                    truth_manifest_path=truth_manifest_path,
                    measurement_log_path=log_path,
                    run_id=tag,
                    runtime_config_path=generated_runtime_path,
                    scene_seed=scene_seed,
                    source_profile=case.source_profile,
                )
                entries.append(
                    AblationPlanEntry(
                        case=case.name,
                        variant=variant.name,
                        batch_id=batch_id,
                        scene_seed=scene_seed,
                        pf_seed=pf_seed,
                        transport_seed=transport_seed,
                        seed_policy=seed_policy,
                        source_profile=case.source_profile,
                        pf_config_path=generated_pf_path,
                        control_policy_path=control_policy_path,
                        runtime_config_path=generated_runtime_path,
                        scenario_path=scenario_path,
                        truth_manifest_path=truth_manifest_path,
                        measurement_log_path=log_path,
                        pf_output_dir=pf_output,
                        scenario_command=scenario_command,
                        session_command=_session_command(
                            scenario_path=scenario_path,
                            runtime_root=runtime_root,
                            pf_config_path=generated_pf_path,
                            control_policy_path=control_policy_path,
                            pf_output_dir=pf_output,
                            pf_seed=pf_seed,
                        ),
                    )
                )
    return entries


MANIFEST_FIELDS = (
    "case",
    "variant",
    "batch_id",
    "scene_seed",
    "pf_seed",
    "transport_seed",
    "seed_policy",
    "source_profile",
    "pf_config_path",
    "control_policy_path",
    "runtime_config_path",
    "scenario_path",
    "truth_manifest_path",
    "measurement_log_path",
    "pf_output_dir",
    "scenario_command",
    "session_command",
)


def _entry_row(entry: AblationPlanEntry) -> dict[str, object]:
    """Return one private-manifest CSV row."""
    return {
        "case": entry.case,
        "variant": entry.variant,
        "batch_id": entry.batch_id,
        "scene_seed": entry.scene_seed,
        "pf_seed": entry.pf_seed,
        "transport_seed": entry.transport_seed,
        "seed_policy": entry.seed_policy,
        "source_profile": entry.source_profile,
        "pf_config_path": entry.pf_config_path.as_posix(),
        "control_policy_path": entry.control_policy_path.as_posix(),
        "runtime_config_path": entry.runtime_config_path.as_posix(),
        "scenario_path": entry.scenario_path.as_posix(),
        "truth_manifest_path": entry.truth_manifest_path.as_posix(),
        "measurement_log_path": entry.measurement_log_path.as_posix(),
        "pf_output_dir": entry.pf_output_dir.as_posix(),
        "scenario_command": shlex.join(entry.scenario_command),
        "session_command": shlex.join(entry.session_command),
    }


def write_ablation_plan(
    entries: Sequence[AblationPlanEntry],
    *,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
) -> tuple[Path, Path]:
    """Write the truth-bearing manifest and run script under runtime privacy."""
    import io

    private_root = Path(private_root).expanduser().resolve()
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root.chmod(0o700)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow(_entry_row(entry))
    manifest_path = private_root / "manifest.csv"
    atomic_write_text(manifest_path, buffer.getvalue())
    manifest_path.chmod(0o600)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for entry in entries:
        lines.append(shlex.join(entry.scenario_command))
        lines.append(shlex.join(entry.session_command))
    script_path = private_root / "run_all.sh"
    atomic_write_text(script_path, "\n".join(lines) + "\n")
    mode = script_path.stat().st_mode
    script_path.chmod((mode | stat.S_IXUSR) & ~stat.S_IRWXG & ~stat.S_IRWXO)
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
    "generate_fresh_batch_id",
    "generate_fresh_pf_seed",
    "generate_fresh_transport_seed",
    "resolve_ablation_seeds",
    "resolve_batch_ids",
    "resolve_pf_seeds",
    "resolve_transport_seeds",
    "write_ablation_plan",
]
