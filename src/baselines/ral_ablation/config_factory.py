"""Build isolated RA-L trials without exposing private truth to generic PF."""

from __future__ import annotations

import copy
import csv
import secrets
import shlex
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.experiment_profiles import MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE
from sim.runtime import (
    load_production_runtime_config,
    validate_production_runtime_config,
)

from runtime.artifacts import atomic_write_json, atomic_write_text
from baselines.ral_ablation.control_policy import (
    RALControlPolicy,
    load_ral_control_policy_document,
    validate_ral_control_policy_payload,
    validate_ral_control_policy_pf_settings,
)
from pf.configuration import load_pf_config
from pf.profiles import enforce_pure_runtime_settings

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_ROOT = ROOT.parent / "Rotating-shield-simulation-runtime"
DEFAULT_RUNTIME_CONFIG = (
    DEFAULT_RUNTIME_ROOT
    / MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.runtime_config_relative_path
)
DEFAULT_PF_CONFIG = ROOT / "configs" / "pf" / "pf_strict_3d.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ral_ablation"
DEFAULT_PRIVATE_ROOT = DEFAULT_RUNTIME_ROOT / "private_runs" / "ral_ablation"
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


def _json_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Return an exact JSON integer inside inclusive declared bounds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be an integer of at least {minimum}{upper}.")
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
    resolved = tuple(
        _json_integer(
            seed,
            name="seed",
            minimum=1,
            maximum=MAX_FRESH_ABLATION_SEED,
        )
        for seed in seeds
    )
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
        forbidden_scene_seeds = set(scenes)
        for _scene_seed in scenes:
            pf_seed = generate_fresh_pf_seed()
            while pf_seed in forbidden_scene_seeds or pf_seed in resolved:
                pf_seed = generate_fresh_pf_seed()
            resolved.append(pf_seed)
        return tuple(resolved)
    resolved = tuple(
        _json_integer(
            seed,
            name="PF seed",
            minimum=1,
            maximum=MAX_FRESH_ABLATION_SEED,
        )
        for seed in pf_seeds
    )
    if len(resolved) != len(scenes):
        raise ValueError("pf_seeds must contain one value per scene seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("pf_seeds must not contain duplicates.")
    if set(resolved).intersection(scenes):
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
        forbidden_seed_streams = {*scenes, *estimators}
        for _scene_seed, _pf_seed in zip(scenes, estimators):
            transport_seed = generate_fresh_transport_seed()
            while (
                transport_seed in forbidden_seed_streams or transport_seed in resolved
            ):
                transport_seed = generate_fresh_transport_seed()
            resolved.append(transport_seed)
        return tuple(resolved)
    resolved = tuple(
        _json_integer(
            seed,
            name="transport seed",
            minimum=1,
            maximum=MAX_FRESH_ABLATION_SEED,
        )
        for seed in transport_seeds
    )
    if len(resolved) != len(scenes):
        raise ValueError("transport_seeds must contain one value per scene seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("transport_seeds must not contain duplicates.")
    if set(resolved).intersection({*scenes, *estimators}):
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
        raise ValueError("batch_ids must contain one value per scene seed.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("batch_ids must not contain duplicates.")
    return resolved


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
        validate_ral_control_policy_payload(
            {
                "schema_version": 1,
                "path_policy": self.path_policy,
                "shield_policy": self.shield_policy,
            }
        )
        unknown_runtime = sorted(
            set(self.runtime_overrides).difference(RAL_RUNTIME_INTERVENTION_FIELDS)
        )
        if unknown_runtime:
            raise ValueError(
                "RA-L runtime overrides may change only declared shield "
                f"interventions; unknown={unknown_runtime}."
            )


@dataclass(frozen=True, slots=True)
class AblationPlanEntry:
    """Store one private experiment session and truth-free PF invocation."""

    case: str
    experiment_profile_id: str
    scene_variant_id: str
    variant: str
    batch_id: str
    scene_seed: int
    pf_seed: int
    transport_seed: int
    seed_policy: str
    run_id: str
    pf_config_path: Path
    control_policy_path: Path
    control_policy_sha256: str
    runtime_config_path: Path
    scenario_path: Path
    truth_manifest_path: Path
    measurement_log_path: Path
    pf_output_dir: Path
    scenario_command: tuple[str, ...]
    session_command: tuple[str, ...]


RAL_CASE_NAME = "mix9_multi_isotope_cardinality"
RAL_EXPERIMENT_PROFILE_ID = MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.profile_id
RAL_SCENE_VARIANT_ID = "mix9"
RAL_RUNTIME_INTERVENTION_FIELDS = frozenset({"shield_transmission_target"})

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
            "dss_pp": None,
            "planning_eig_samples": None,
            "runtime_candidate_refinement_top_k": 0,
            "planner_audit_top_k": 0,
        },
        runtime_overrides={
            "shield_transmission_target": 1.0,
        },
        path_policy={"name": "passive_serpentine", "row_count": 8},
        shield_policy={"name": "fixed", "fixed_pair_id": 0},
    ),
    AblationVariant(
        name="round_robin_shield",
        description="Cycle Fe/Pb pairs without posterior-dependent selection.",
        pf_overrides={
            "dss_pp": {
                "shield_view_count_shadow_enabled": False,
                "conditional_greedy_one_swap": False,
            },
        },
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
                "coverage_floor_quantile": 0.0,
                "coverage_floor_weight": 0.0,
                "coverage_surface_max_hausdorff_m": None,
                "coverage_surface_quadrature_max_points": None,
                "exact_eig_coverage_reserve": 0,
                "bearing_diversity_weight": 0.0,
                "frontier_weight": 0.0,
                "local_orbit_weight": 0.0,
                "local_orbit_ring_radii_m": [],
                "local_orbit_sigma_m": None,
                "elevation_condition_weight": 0.0,
                "elevation_pair_xy_scale_m": None,
                "elevation_pair_z_scale_m": None,
                "elevation_angle_threshold_deg": None,
                "revisit_penalty_weight": 0.0,
                "turn_smoothness_weight": 0.0,
            },
        },
        runtime_overrides={},
    ),
)


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
    policy = RALControlPolicy(
        path_policy=variant.path_policy,
        shield_policy=variant.shield_policy,
    )
    validate_ral_control_policy_pf_settings(policy, config)
    return config


def _control_policy(variant: AblationVariant) -> dict[str, object]:
    """Return the separate RA-L adapter policy for one experiment variant."""
    return validate_ral_control_policy_payload(
        {
            "schema_version": 1,
            "path_policy": (
                None if variant.path_policy is None else dict(variant.path_policy)
            ),
            "shield_policy": (
                None if variant.shield_policy is None else dict(variant.shield_policy)
            ),
        }
    )


def _runtime_config(
    base: Mapping[str, Any],
    *,
    variant: AblationVariant,
    seed: int,
) -> dict[str, Any]:
    """Return one self-contained strict private runtime configuration."""
    payload = copy.deepcopy(dict(base))
    payload["random_seed_base"] = seed
    payload.update(variant.runtime_overrides)
    validate_production_runtime_config(payload)
    return payload


def _require_effective_variant_deltas(
    pf_base: Mapping[str, Any],
    runtime_base: Mapping[str, Any],
    variants: Sequence[AblationVariant],
) -> None:
    """Reject declared ablations that do not change their shared base."""
    for variant in variants:
        if variant.pf_overrides and _deep_update(
            pf_base,
            variant.pf_overrides,
        ) == dict(pf_base):
            raise ValueError(
                f"RA-L variant {variant.name!r} has a no-op PF intervention."
            )
        no_op_runtime = sorted(
            field
            for field, value in variant.runtime_overrides.items()
            if runtime_base.get(field) == value
        )
        if no_op_runtime:
            raise ValueError(
                f"RA-L variant {variant.name!r} has no-op runtime interventions: "
                f"{no_op_runtime}."
            )


def _scenario_command(
    *,
    runtime_root: Path,
    scenario_path: Path,
    truth_manifest_path: Path,
    measurement_log_path: Path,
    run_id: str,
    runtime_config_path: Path,
    scene_seed: int,
    experiment_profile_id: str,
    scene_variant_id: str,
) -> tuple[str, ...]:
    """Return the private runtime scenario and truth-manifest command."""
    return (
        "uv",
        "run",
        "--directory",
        runtime_root.as_posix(),
        "rotating-shield-sim",
        "generate-scenario",
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
        "--experiment-profile",
        experiment_profile_id,
        "--scene-variant",
        scene_variant_id,
    )


def _session_command(
    *,
    scenario_path: Path,
    truth_manifest_path: Path,
    runtime_root: Path,
    pf_config_path: Path,
    control_policy_path: Path,
    control_policy_sha256: str,
    pf_output_dir: Path,
    pf_seed: int,
) -> tuple[str, ...]:
    """Return the private runner command that isolates its PF child by socket."""
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
        "--truth-manifest",
        truth_manifest_path.as_posix(),
        "--pf-config",
        pf_config_path.as_posix(),
        "--control-policy",
        control_policy_path.as_posix(),
        "--expected-control-policy-sha256",
        control_policy_sha256,
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
    pf_base = load_pf_config(pf_config_path).config()
    runtime_base = load_production_runtime_config(runtime_config_path)
    resolved_variants = DEFAULT_ABLATION_VARIANTS
    _require_effective_variant_deltas(pf_base, runtime_base, resolved_variants)
    if seeds is None:
        if pf_seeds is not None or transport_seeds is not None:
            raise ValueError(
                "Fresh RA-L batches generate scene, PF, and transport seeds "
                "together; omit every seed option."
            )
        seed_policy = "fresh_per_batch"
    else:
        if pf_seeds is None or transport_seeds is None or batch_ids is None:
            raise ValueError(
                "An explicit live repeat requires recorded scene, PF, transport, "
                "and batch identifiers."
            )
        seed_policy = "explicit_live_repeat"
    suffix = _safe_suffix(output_tag_suffix)
    resolved_scene_seeds = resolve_ablation_seeds(seeds)
    resolved_pf_seeds = resolve_pf_seeds(resolved_scene_seeds, pf_seeds)
    resolved_transport_seeds = resolve_transport_seeds(
        resolved_scene_seeds,
        resolved_pf_seeds,
        transport_seeds,
    )
    batch_count = len(resolved_scene_seeds)
    resolved_batch_ids = iter(resolve_batch_ids(batch_count, batch_ids))
    entries: list[AblationPlanEntry] = []
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root.chmod(0o700)
    for scene_seed, pf_seed, transport_seed in zip(
        resolved_scene_seeds,
        resolved_pf_seeds,
        resolved_transport_seeds,
    ):
        batch_id = next(resolved_batch_ids)
        for variant in resolved_variants:
            tag = f"ral_{batch_id}_{variant.name}"
            if suffix:
                tag = f"{tag}_{suffix}"
            generated_pf_path = output_dir / "configs" / f"{tag}.json"
            control_policy_path = output_dir / "control_policies" / f"{tag}.json"
            generated_runtime_path = private_root / "runtime_configs" / f"{tag}.json"
            scenario_path = private_root / "scenarios" / f"{tag}.json"
            truth_manifest_path = private_root / "truth_manifests" / f"{tag}.json"
            log_path = output_dir / "measurement_logs" / tag
            pf_output = output_dir / "runs" / tag
            atomic_write_json(
                generated_pf_path,
                _pf_config(pf_base, variant=variant),
            )
            atomic_write_json(control_policy_path, _control_policy(variant))
            control_document = load_ral_control_policy_document(control_policy_path)
            if control_document.source_sha256 != control_document.canonical_sha256:
                raise ValueError(
                    "Generated RA-L control policy must use exact canonical bytes."
                )
            atomic_write_json(
                generated_runtime_path,
                _runtime_config(
                    runtime_base,
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
                experiment_profile_id=RAL_EXPERIMENT_PROFILE_ID,
                scene_variant_id=RAL_SCENE_VARIANT_ID,
            )
            entries.append(
                AblationPlanEntry(
                    case=RAL_CASE_NAME,
                    experiment_profile_id=RAL_EXPERIMENT_PROFILE_ID,
                    scene_variant_id=RAL_SCENE_VARIANT_ID,
                    variant=variant.name,
                    batch_id=batch_id,
                    scene_seed=scene_seed,
                    pf_seed=pf_seed,
                    transport_seed=transport_seed,
                    seed_policy=seed_policy,
                    run_id=tag,
                    pf_config_path=generated_pf_path,
                    control_policy_path=control_policy_path,
                    control_policy_sha256=control_document.source_sha256,
                    runtime_config_path=generated_runtime_path,
                    scenario_path=scenario_path,
                    truth_manifest_path=truth_manifest_path,
                    measurement_log_path=log_path,
                    pf_output_dir=pf_output,
                    scenario_command=scenario_command,
                    session_command=_session_command(
                        scenario_path=scenario_path,
                        truth_manifest_path=truth_manifest_path,
                        runtime_root=runtime_root,
                        pf_config_path=generated_pf_path,
                        control_policy_path=control_policy_path,
                        control_policy_sha256=control_document.source_sha256,
                        pf_output_dir=pf_output,
                        pf_seed=pf_seed,
                    ),
                )
            )
    return entries


MANIFEST_FIELDS = (
    "case",
    "experiment_profile_id",
    "scene_variant_id",
    "variant",
    "batch_id",
    "scene_seed",
    "pf_seed",
    "transport_seed",
    "seed_policy",
    "run_id",
    "pf_config_path",
    "control_policy_path",
    "control_policy_sha256",
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
        "experiment_profile_id": entry.experiment_profile_id,
        "scene_variant_id": entry.scene_variant_id,
        "variant": entry.variant,
        "batch_id": entry.batch_id,
        "scene_seed": entry.scene_seed,
        "pf_seed": entry.pf_seed,
        "transport_seed": entry.transport_seed,
        "seed_policy": entry.seed_policy,
        "run_id": entry.run_id,
        "pf_config_path": entry.pf_config_path.as_posix(),
        "control_policy_path": entry.control_policy_path.as_posix(),
        "control_policy_sha256": entry.control_policy_sha256,
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
    "DEFAULT_ABLATION_VARIANTS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PF_CONFIG",
    "DEFAULT_PRIVATE_ROOT",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_RUNTIME_ROOT",
    "MANIFEST_FIELDS",
    "MAX_FRESH_ABLATION_SEED",
    "AblationPlanEntry",
    "AblationVariant",
    "RAL_CASE_NAME",
    "RAL_EXPERIMENT_PROFILE_ID",
    "RAL_RUNTIME_INTERVENTION_FIELDS",
    "RAL_SCENE_VARIANT_ID",
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
