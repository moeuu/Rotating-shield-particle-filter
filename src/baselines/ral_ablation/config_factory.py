"""Generate RA-L ablation configurations without mixing baseline logic into DSS-PP."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from measurement.model import EnvironmentConfig
from measurement.obstacles import build_obstacle_grid
from measurement.source_boundary import (
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from measurement.source_surfaces import (
    generate_surface_sources,
    validate_area_uniform_source_config,
)
from measurement.surface_charts import (
    build_surface_chart_geometry,
    surface_chart_geometry_sha256,
)
from pf.profiles import enforce_pure_runtime_settings
from pf.randomness import (
    named_random_generator,
    named_rng_provenance,
    named_stream_seed,
)
from runtime_defaults import (
    DEFAULT_CUI_SPLIT_VIEW_DIR,
    DEFAULT_MEASUREMENT_TIME_S,
    DEFAULT_NO_ROTATION_OVERHEAD_S,
    DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
)
from runtime_environment import attach_random_manchester_transport_geometry
from sim.runtime import load_runtime_config

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = (
    ROOT / "configs" / "geant4" / "variance_reduction_external_no_isaac_32threads.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ral_ablation"
DEFAULT_MEASUREMENT_LOG_ROOT = Path("results") / "ral_ablation" / "measurement_logs"
DEFAULT_ISOTOPES = ("Cs-137", "Co-60", "Eu-154")
TRUTH_SURFACE_SOURCE_RNG_DOMAIN = "truth_surface_sources"


def _json_boolean(value: object, *, field_name: str) -> bool:
    """Return an exact JSON boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON boolean.")
    return value


def _json_integer(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    """Return an exact JSON integer satisfying an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a JSON integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return value


def _finite_json_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return a finite JSON number satisfying its physical domain."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _nonempty_string(value: object, *, field_name: str) -> str:
    """Return an exact nonempty string without case or whitespace aliases."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string.")
    return value


@dataclass(frozen=True)
class AblationCase:
    """Describe a fixed-source-cardinality RA-L ablation case."""

    name: str
    description: str
    isotopes: tuple[str, ...]
    source_count: int
    isotope_counts: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        """Validate one paper-case declaration before it reaches generation."""
        _nonempty_string(self.name, field_name="AblationCase.name")
        _nonempty_string(self.description, field_name="AblationCase.description")
        if not isinstance(self.isotopes, tuple) or not self.isotopes:
            raise ValueError("AblationCase.isotopes must be a nonempty tuple.")
        isotope_names = tuple(
            _nonempty_string(value, field_name="AblationCase.isotopes entry")
            for value in self.isotopes
        )
        if len(set(isotope_names)) != len(isotope_names):
            raise ValueError("AblationCase.isotopes must not contain duplicates.")
        source_count = _json_integer(
            self.source_count,
            field_name="AblationCase.source_count",
            minimum=1,
        )
        if self.isotope_counts is None:
            if len(isotope_names) != source_count:
                raise ValueError(
                    "Without isotope_counts, AblationCase.isotopes must list "
                    "one isotope per source."
                )
            return
        if not isinstance(self.isotope_counts, tuple) or not self.isotope_counts:
            raise ValueError(
                "AblationCase.isotope_counts must be a nonempty tuple or null."
            )
        declared: list[str] = []
        count_total = 0
        for isotope, count in self.isotope_counts:
            declared.append(
                _nonempty_string(
                    isotope,
                    field_name="AblationCase.isotope_counts isotope",
                )
            )
            count_total += _json_integer(
                count,
                field_name="AblationCase.isotope_counts count",
                minimum=1,
            )
        if len(set(declared)) != len(declared):
            raise ValueError("AblationCase.isotope_counts must be unique by isotope.")
        if set(declared) != set(isotope_names) or count_total != source_count:
            raise ValueError(
                "AblationCase isotope_counts must cover isotopes and sum to "
                "source_count."
            )


@dataclass(frozen=True)
class AblationVariant:
    """Describe one module-ablation variant."""

    name: str
    description: str
    overrides: Mapping[str, Any]
    cli_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate one paper-variant declaration."""
        _nonempty_string(self.name, field_name="AblationVariant.name")
        _nonempty_string(self.description, field_name="AblationVariant.description")
        if not isinstance(self.overrides, Mapping):
            raise ValueError("AblationVariant.overrides must be a mapping.")
        if not isinstance(self.cli_args, tuple) or any(
            not isinstance(value, str) or not value for value in self.cli_args
        ):
            raise ValueError(
                "AblationVariant.cli_args must be a tuple of nonempty strings."
            )


@dataclass(frozen=True)
class AblationPlanEntry:
    """Store one executable ablation trial."""

    case: str
    variant: str
    seed: int
    config_path: Path
    source_path: Path
    command: tuple[str, ...]


DEFAULT_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        name="mix9_multi_isotope_cardinality",
        description=(
            "Main RA-L task: 4 Cs-137, 3 Co-60, and 2 Eu-154 sources with "
            "same-isotope ambiguity inside a multi-isotope STE problem."
        ),
        isotopes=("Cs-137", "Co-60", "Eu-154"),
        source_count=9,
        isotope_counts=(("Cs-137", 4), ("Co-60", 3), ("Eu-154", 2)),
    ),
)

DEFAULT_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        name="proposed",
        description="Full proposed temporal shield program and DSS-PP.",
        overrides={},
    ),
    AblationVariant(
        name="baseline_passive_equal_time_no_shield",
        description=(
            "Passive no-shield baseline with the same per-station physical "
            "live-time budget as the proposed shield program."
        ),
        overrides={
            "shield_transmission_target": 1.0,
            "shield_thickness_scale": 0.0,
            "baseline_shield_policy": {"name": "fixed", "fixed_pair_id": 0},
            "baseline_path_policy": {"name": "passive_serpentine", "row_count": 8},
        },
        cli_args=("--rotation-overhead-s", f"{DEFAULT_NO_ROTATION_OVERHEAD_S:g}"),
    ),
    AblationVariant(
        name="round_robin_shield",
        description="Cycle Fe/Pb posture pairs without posterior-dependent selection.",
        overrides={
            "baseline_shield_policy": {
                "name": "round_robin",
                "start_pair_id": 0,
                "advance_by_pose": True,
            },
            "strict_planned_shield_program": True,
        },
    ),
    AblationVariant(
        name="eig_only_path",
        description=(
            "Keep exact joint full-spectrum generative EIG but remove optional "
            "route and coverage geometry terms from DSS-PP."
        ),
        overrides={
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
    ),
)


def _parallel_runtime_overrides(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-fidelity-changing compute settings for generated trials."""
    worker_value = (
        base_config["python_worker_count"]
        if "python_worker_count" in base_config
        else base_config.get("cpu_worker_count", 32)
    )
    workers = _json_integer(
        worker_value,
        field_name="python_worker_count",
        minimum=1,
    )
    thread_count = _json_integer(
        base_config.get("thread_count", workers),
        field_name="thread_count",
        minimum=1,
    )
    pose_workers = _json_integer(
        base_config.get("pose_selection_workers", workers),
        field_name="pose_selection_workers",
        minimum=1,
    )
    return {
        "thread_count": thread_count,
        "python_worker_count": workers,
        "pose_selection_workers": pose_workers,
    }


def _deep_update(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a recursive dictionary merge of base and overrides."""
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    """Load a runtime JSON object, including an inherited parent config."""
    payload = load_runtime_config(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON object to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _resolve_base_config_path(value: object, *, base_config_path: Path) -> str | None:
    """Resolve a config-relative path so generated configs remain relocatable."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Runtime asset paths must be nonempty strings or null.")
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.as_posix()
    return (base_config_path.parent / raw_path).resolve().as_posix()


def _case_isotope_sequence(case: AblationCase) -> tuple[str, ...]:
    """Return the exact isotope sequence used for source generation."""
    if case.isotope_counts is None:
        return case.isotopes
    expanded: list[str] = []
    for isotope, count in case.isotope_counts:
        expanded.extend([isotope] * count)
    if len(expanded) != case.source_count:
        raise ValueError(
            f"Case {case.name} isotope_counts expand to {len(expanded)} sources, "
            f"but source_count is {case.source_count}."
        )
    return tuple(expanded)


def _case_isotope_count_metadata(case: AblationCase) -> dict[str, int]:
    """Return isotope-count metadata for generated source layouts."""
    counts: dict[str, int] = {}
    if case.isotope_counts is not None:
        for isotope, count in case.isotope_counts:
            counts[isotope] = count
        return counts
    for idx in range(case.source_count):
        isotope = case.isotopes[idx]
        counts[isotope] = counts.get(isotope, 0) + 1
    return counts


def _source_generation_options(base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return physical geometry options for area-uniform truth generation."""
    validate_area_uniform_source_config(base_config)
    return {
        "obstacle_height_m": _finite_json_number(
            base_config.get("obstacle_height_m", 2.0),
            field_name="obstacle_height_m",
            strictly_positive=True,
        ),
        "include_room_boundaries": _json_boolean(
            base_config.get("author_room_boundary_prims", False),
            field_name="author_room_boundary_prims",
        ),
        "room_boundary_thickness_m": _finite_json_number(
            base_config.get("room_boundary_thickness_m", 0.1),
            field_name="room_boundary_thickness_m",
            strictly_positive=True,
        ),
        "structural_rj_surface_chart_max_edge_m": _finite_json_number(
            base_config.get("structural_rj_surface_chart_max_edge_m", 1.0),
            field_name="structural_rj_surface_chart_max_edge_m",
            strictly_positive=True,
        ),
    }


def _case_source_layout(
    case: AblationCase,
    *,
    obstacle_seed: int,
    source_seed: int,
    intensity_cps_1m: float | Sequence[float],
    source_generation_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a surface-constrained source layout for one case and seed."""
    env = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=10.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=env.size_x,
        size_y=env.size_y,
        cell_size=1.0,
        blocked_fraction=0.4,
        rng_seed=obstacle_seed,
        keep_free_points=[(env.detector_position[0], env.detector_position[1])],
        passage_width_m=1.0,
    )
    rng = named_random_generator(
        _json_integer(source_seed, field_name="source_seed", minimum=0),
        TRUTH_SURFACE_SOURCE_RNG_DOMAIN,
    )
    options = dict(source_generation_options or {})
    obstacle_seed = _json_integer(
        obstacle_seed,
        field_name="obstacle_seed",
        minimum=0,
    )
    obstacle_height_m = _finite_json_number(
        options.get("obstacle_height_m", 2.0),
        field_name="obstacle_height_m",
        strictly_positive=True,
    )
    include_room_boundaries = _json_boolean(
        options.get("include_room_boundaries", False),
        field_name="include_room_boundaries",
    )
    room_boundary_thickness_m = _finite_json_number(
        options.get("room_boundary_thickness_m", 0.1),
        field_name="room_boundary_thickness_m",
        strictly_positive=True,
    )
    chart_max_edge_m = _finite_json_number(
        options.get("structural_rj_surface_chart_max_edge_m", 1.0),
        field_name="structural_rj_surface_chart_max_edge_m",
        strictly_positive=True,
    )
    _intensity_sampling_metadata(intensity_cps_1m)
    grid, _ = attach_random_manchester_transport_geometry(
        grid,
        room_size_xyz=(env.size_x, env.size_y, env.size_z),
        obstacle_height_m=obstacle_height_m,
        rng_seed=obstacle_seed,
        include_room_boundaries=include_room_boundaries,
        room_boundary_thickness_m=room_boundary_thickness_m,
    )
    isotope_sequence = _case_isotope_sequence(case)
    sources = generate_surface_sources(
        env=env,
        obstacle_grid=grid,
        isotopes=isotope_sequence,
        intensity_cps_1m=intensity_cps_1m,
        rng=rng,
        count=case.source_count,
        obstacle_height_m=obstacle_height_m,
        chart_max_edge_m=chart_max_edge_m,
    )
    surface_geometry = build_surface_chart_geometry(
        env,
        grid,
        max_edge_m=chart_max_edge_m,
        obstacle_height_m=obstacle_height_m,
    )
    surface_atlas_sha256 = surface_chart_geometry_sha256(surface_geometry)
    source_entries = [
        {
            "isotope": source.isotope,
            "position": [float(value) for value in source.position],
            "transport_position": [
                float(value) for value in source.transport_position
            ],
            "intensity_cps_1m": float(source.intensity_cps_1m),
            "surface_chart_id": int(source.surface_chart_id),
            "surface_uv": [float(value) for value in source.surface_uv],
            "surface_normal": [
                float(value) for value in source.surface_normal
            ],
            "surface_emission_policy_sha256": str(
                source.surface_emission_policy_sha256
            ),
        }
        for source in sources
    ]
    source_contract_sha256 = surface_source_runtime_contract_sha256(
        source_entries
    )
    return {
        "name": f"ral_ablation_{case.name}_seed_{source_seed}",
        "metadata": {
            "case": case.name,
            "description": case.description,
            "isotope_counts": _case_isotope_count_metadata(case),
            "source_seed": source_seed,
            "source_rng_provenance": named_rng_provenance(
                source_seed,
                (TRUTH_SURFACE_SOURCE_RNG_DOMAIN,),
            ),
            "source_derived_seed": named_stream_seed(
                source_seed,
                TRUTH_SURFACE_SOURCE_RNG_DOMAIN,
            ),
            "obstacle_seed": obstacle_seed,
            "source_surface_sampling_schema_version": 3,
            "sampling": "continuous area-uniform physical-surface placement",
            "sampling_measure": "continuous_area_uniform",
            "surface_geometry": "runtime_transport_component_union",
            "selection_conditioning": "none_physical_area_only",
            "obstacle_height_m": obstacle_height_m,
            "surface_chart_max_edge_m": chart_max_edge_m,
            "surface_atlas_contract_sha256": surface_atlas_sha256,
            "surface_emission_policy_sha256": (
                surface_emission_policy_sha256()
            ),
            "surface_source_runtime_contract_sha256": (
                source_contract_sha256
            ),
            "include_room_boundaries": include_room_boundaries,
            "room_boundary_thickness_m": room_boundary_thickness_m,
            "intensity_model": (
                "intensity_cps_1m is expected pre-dead-time detector pulse "
                "rate at 1 m"
            ),
            "intensity_sampling": _intensity_sampling_metadata(intensity_cps_1m),
        },
        "sources": source_entries,
    }


def _intensity_sampling_metadata(
    intensity_cps_1m: float | Sequence[float],
) -> dict[str, float | str]:
    """Return metadata describing source-strength sampling for a case."""
    if isinstance(intensity_cps_1m, Sequence) and not isinstance(
        intensity_cps_1m,
        (str, bytes),
    ):
        if len(intensity_cps_1m) != 2:
            raise ValueError("intensity range must contain exactly two values.")
        lo = _finite_json_number(
            intensity_cps_1m[0],
            field_name="intensity range minimum",
            strictly_positive=True,
        )
        hi = _finite_json_number(
            intensity_cps_1m[1],
            field_name="intensity range maximum",
            strictly_positive=True,
        )
        if hi < lo:
            raise ValueError("intensity range maximum must not be below minimum.")
        return {"mode": "uniform", "min_cps_1m": lo, "max_cps_1m": hi}
    return {
        "mode": "fixed",
        "cps_1m": _finite_json_number(
            intensity_cps_1m,
            field_name="intensity_cps_1m",
            strictly_positive=True,
        ),
    }


def _variant_config(
    base_config: Mapping[str, Any],
    *,
    base_config_path: Path,
    case: AblationCase,
    variant: AblationVariant,
    seed: int,
    output_tag: str,
) -> dict[str, Any]:
    """Return the runtime config for one ablation variant."""
    config = _deep_update(base_config, _parallel_runtime_overrides(base_config))
    config = _deep_update(config, variant.overrides)
    config = enforce_pure_runtime_settings(config)
    if config.get("backend") != "geant4" or config.get("engine_mode") != "external":
        raise ValueError(
            "RA-L full simulations require backend='geant4' and "
            "engine_mode='external'; analytic or in-process transport is not "
            "an ablation backend."
        )
    if config.get("variable_cardinality") is not True:
        raise ValueError(
            "RA-L ablations require variable_cardinality=true for exact "
            "reversible-jump PF."
        )
    strength_min = _finite_json_number(
        config.get("pf_strength_prior_min_cps_1m"),
        field_name="pf_strength_prior_min_cps_1m",
        minimum=0.0,
    )
    strength_max = _finite_json_number(
        config.get("pf_strength_prior_max_cps_1m"),
        field_name="pf_strength_prior_max_cps_1m",
        strictly_positive=True,
    )
    if strength_max <= strength_min:
        raise ValueError(
            "RA-L ablations require finite ordered "
            "pf_strength_prior_min_cps_1m and "
            "pf_strength_prior_max_cps_1m bounds."
        )
    transport_history_mode = _validate_ral_transport_sampling(config)
    config["primary_sampling_fraction"] = 1.0
    config["accelerated_weighted_transport_enable"] = False
    config["target_sampled_primaries"] = None
    thread_count = _json_integer(
        config.get("thread_count", 1),
        field_name="thread_count",
        minimum=1,
    )
    if thread_count <= 1:
        raise ValueError(
            "RA-L full simulations require a multithreaded Geant4 runtime; "
            "thread_count must be greater than one."
        )
    seed = _json_integer(seed, field_name="seed", minimum=0)
    output_tag = _nonempty_string(output_tag, field_name="output_tag")
    config["random_seed_base"] = seed
    config["measurement_log_output_dir"] = (
        DEFAULT_MEASUREMENT_LOG_ROOT / output_tag
    ).as_posix()
    config["measurement_log_run_id"] = output_tag
    # Keep the browser progress page stable across ablation runs. The final
    # result files still use output_tag, so only the live progress view is shared.
    config["cui_split_view_dir"] = DEFAULT_CUI_SPLIT_VIEW_DIR
    for path_key in ("usd_path", "random_environment_base_usd_path"):
        resolved_path = _resolve_base_config_path(
            config.get(path_key),
            base_config_path=base_config_path,
        )
        if resolved_path is not None:
            config[path_key] = resolved_path
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("RA-L runtime metadata must be a JSON object.")
    metadata.update(
        {
            "ral_ablation_case": case.name,
            "ral_ablation_variant": variant.name,
            "ral_ablation_seed": seed,
            "ral_transport_history_mode": transport_history_mode,
            "ral_accelerated_transport": False,
            "ral_primary_sampling_fraction": config["primary_sampling_fraction"],
            "ral_primary_history_weight": 1.0,
            "ral_target_sampled_primaries": None,
        }
    )
    config["metadata"] = metadata
    return config


def _validate_ral_transport_sampling(config: Mapping[str, Any]) -> str:
    """Require full, unit-weight native histories for every RA-L variant."""
    fraction_raw = config.get("primary_sampling_fraction", 1.0)
    if isinstance(fraction_raw, bool) or not isinstance(
        fraction_raw,
        (int, float),
    ):
        raise ValueError("primary_sampling_fraction must be the JSON number 1.0.")
    fraction = float(fraction_raw)
    if not np.isfinite(fraction) or fraction != 1.0:
        raise ValueError(
            "RA-L full simulations require primary_sampling_fraction=1.0."
        )
    accelerated = config.get("accelerated_weighted_transport_enable", False)
    if not isinstance(accelerated, bool) or accelerated:
        raise ValueError(
            "RA-L full simulations require "
            "accelerated_weighted_transport_enable=false."
        )
    if config.get("target_sampled_primaries") is not None:
        raise ValueError(
            "RA-L full simulations require target_sampled_primaries=null."
        )
    source_rate_model = config.get("source_rate_model", "detector_cps_1m")
    if not isinstance(source_rate_model, str) or (
        source_rate_model != "detector_cps_1m"
    ):
        raise ValueError(
            "RA-L full simulations require source_rate_model=detector_cps_1m."
        )
    return "full_unit_weight"


def build_ablation_plan(
    *,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: Sequence[int] = (2026050901, 2026050902, 2026050903),
    cases: Sequence[AblationCase] = DEFAULT_ABLATION_CASES,
    variants: Sequence[AblationVariant] = DEFAULT_ABLATION_VARIANTS,
    intensity_cps_1m: float | Sequence[float] = DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
    output_tag_suffix: str = "",
) -> list[AblationPlanEntry]:
    """Build and write config/source files for RA-L ablation trials."""
    base_config_path = Path(base_config_path).expanduser().resolve()
    base_config = _load_json(base_config_path)
    source_options = _source_generation_options(base_config)
    entries: list[AblationPlanEntry] = []
    if not isinstance(output_tag_suffix, str):
        raise ValueError("output_tag_suffix must be a string.")
    normalized_suffix = output_tag_suffix.strip().strip("_")
    if normalized_suffix and any(
        not (character.isalnum() or character in {"-", "_"})
        for character in normalized_suffix
    ):
        raise ValueError(
            "output_tag_suffix may contain only letters, digits, '-' and '_'."
        )
    config_dir = Path(output_dir) / "configs"
    source_dir = Path(output_dir) / "sources"
    for case in cases:
        for seed_raw in seeds:
            seed = _json_integer(seed_raw, field_name="seed", minimum=0)
            source_seed = seed + 17
            source_payload = _case_source_layout(
                case,
                obstacle_seed=seed,
                source_seed=source_seed,
                intensity_cps_1m=intensity_cps_1m,
                source_generation_options=source_options,
            )
            source_path = source_dir / f"{case.name}_seed_{seed}.json"
            _write_json(source_path, source_payload)
            for variant in variants:
                tag = f"{case.name}_{variant.name}_seed_{seed}"
                if normalized_suffix:
                    tag = f"{tag}_{normalized_suffix}"
                config = _variant_config(
                    base_config,
                    base_config_path=base_config_path,
                    case=case,
                    variant=variant,
                    seed=seed,
                    output_tag=tag,
                )
                config_path = config_dir / f"{tag}.json"
                _write_json(config_path, config)
                command = _trial_command(
                    config_path=config_path,
                    source_path=source_path,
                    obstacle_seed=seed,
                    output_tag=tag,
                    extra_args=variant.cli_args,
                )
                entries.append(
                    AblationPlanEntry(
                        case=case.name,
                        variant=variant.name,
                        seed=seed,
                        config_path=config_path,
                        source_path=source_path,
                        command=command,
                    )
                )
    return entries


def _trial_command(
    *,
    config_path: Path,
    source_path: Path,
    obstacle_seed: int,
    output_tag: str,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the standard full-simulation command for one ablation trial."""
    return (
        "uv",
        "run",
        "python",
        "main.py",
        "--full-simulation",
        "--sim-config",
        config_path.as_posix(),
        "--environment-mode",
        "random",
        "--obstacle-seed",
        str(_json_integer(obstacle_seed, field_name="obstacle_seed", minimum=0)),
        "--source-config",
        source_path.as_posix(),
        "--measurement-time-s",
        f"{DEFAULT_MEASUREMENT_TIME_S:g}",
        "--output-tag",
        output_tag,
        *tuple(extra_args),
    )


def write_ablation_plan(
    entries: Sequence[AblationPlanEntry],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write a CSV manifest and shell command file for ablation entries."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    script_path = out / "run_all.sh"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "variant",
                "seed",
                "config_path",
                "source_path",
                "command",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "case": entry.case,
                    "variant": entry.variant,
                    "seed": entry.seed,
                    "config_path": entry.config_path.as_posix(),
                    "source_path": entry.source_path.as_posix(),
                    "command": " ".join(entry.command),
                }
            )
    with script_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for entry in entries:
            handle.write(" ".join(entry.command))
            handle.write("\n")
    script_path.chmod(0o755)
    return manifest_path, script_path
