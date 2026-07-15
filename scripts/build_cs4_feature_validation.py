"""Build full-simulation feature-validation runs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "src").as_posix())

from baselines.ral_ablation.config_factory import (
    AblationCase,
    AblationVariant,
    build_ablation_plan,
    write_ablation_plan,
)
from runtime_defaults import DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M

DEFAULT_OUTPUT_DIR = ROOT / "results" / "cs4_feature_validation"
DEFAULT_MIX9_OUTPUT_DIR = ROOT / "results" / "mix9_feature_validation"
DEFAULT_SEED = 2026051001

CS4_CASE = AblationCase(
    name="cs4_same_isotope_cardinality",
    description=(
        "Feature-validation task with four Cs-137 surface sources in a random "
        "obstacle environment."
    ),
    isotopes=("Cs-137",),
    source_count=4,
    isotope_counts=(("Cs-137", 4),),
)

MIX9_CASE = AblationCase(
    name="mix9_multi_isotope_feature_validation",
    description=(
        "Feature-validation task with four Cs-137, three Co-60, and two "
        "Eu-154 surface sources in a random obstacle environment."
    ),
    isotopes=("Cs-137", "Co-60", "Eu-154"),
    source_count=9,
    isotope_counts=(("Cs-137", 4), ("Co-60", 3), ("Eu-154", 2)),
)


def _all_on_overrides(candidate_isotopes: tuple[str, ...]) -> dict[str, object]:
    """Return feature-enabled overrides for one candidate-isotope set."""
    return {
        "candidate_isotopes": list(candidate_isotopes),
        "birth_orthogonalize_residual_candidates": True,
        "birth_orthogonal_candidate_corr_max": 0.98,
        "mode_preserving_dynamic_cardinality_allocation": True,
        "mode_preserving_dynamic_cardinality_extra_particles": 4,
        "mode_preserving_dynamic_cardinality_min_mass": 0.01,
        "mode_preserving_dynamic_cardinality_entropy_min": 0.25,
    }


def _feature_variants(candidate_isotopes: tuple[str, ...]) -> tuple[AblationVariant, ...]:
    """Return the feature-toggle variant set for one isotope candidate set."""
    all_on_overrides = _all_on_overrides(candidate_isotopes)
    return (
        AblationVariant(
            name="feature_all_on",
            description=(
                "Current proposed method plus dynamic cardinality allocation and "
                "orthogonalized residual birth."
            ),
            overrides=all_on_overrides,
        ),
        AblationVariant(
            name="no_dynamic_particle_allocation",
            description=(
                "Disable only entropy-driven cardinality allocation while keeping "
                "other validation features enabled."
            ),
            overrides={
                **all_on_overrides,
                "mode_preserving_dynamic_cardinality_allocation": False,
                "mode_preserving_dynamic_cardinality_extra_particles": 0,
            },
        ),
        AblationVariant(
            name="no_condition_planning",
            description=(
                "Disable response-matrix condition-number planning terms while "
                "keeping other validation features enabled."
            ),
            overrides={
                **all_on_overrides,
                "dss_pp": {
                    "station_condition_weight": 0.0,
                    "elevation_condition_weight": 0.0,
                },
            },
        ),
        AblationVariant(
            name="no_recovery_verification_modes",
            description=(
                "Disable DSS-PP runtime rescue/global recovery modes and remaining "
                "verification pressure while keeping estimator verification enabled."
            ),
            overrides={
                **all_on_overrides,
                "dss_pp": {
                    "include_runtime_rescue_modes": False,
                    "include_global_surface_rescue_modes": False,
                    "runtime_rescue_mode_weight": 0.0,
                    "global_surface_rescue_mode_weight": 0.0,
                },
                "remaining_measurement_estimate": {
                    "verification_weight": 0.0,
                    "report_response_correlation_weight": 0.0,
                },
            },
        ),
        AblationVariant(
            name="no_orthogonal_birth",
            description=(
                "Disable only orthogonalized residual-birth candidate ranking while "
                "keeping dynamic allocation and planner modes enabled."
            ),
            overrides={
                **all_on_overrides,
                "birth_orthogonalize_residual_candidates": False,
            },
        ),
    )


CS4_FEATURE_VARIANTS = _feature_variants(("Cs-137",))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate full-simulation feature-validation configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for manifest, run script, configs, and source layout.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[DEFAULT_SEED],
        help="One or more random seeds. Each seed reuses one source layout across variants.",
    )
    parser.add_argument(
        "--case",
        choices=("cs4", "mix9"),
        default="cs4",
        help="Validation case. Use mix9 when a feature needs multiple isotopes.",
    )
    return parser.parse_args()


def build_cs4_feature_validation_plan(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] = (DEFAULT_SEED,),
) -> tuple[Path, Path]:
    """Build the Cs4 feature-validation manifest and run script."""
    entries = build_ablation_plan(
        output_dir=Path(output_dir),
        seeds=tuple(int(seed) for seed in seeds),
        cases=(CS4_CASE,),
        variants=CS4_FEATURE_VARIANTS,
        intensity_cps_1m=DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
    )
    return write_ablation_plan(entries, output_dir=Path(output_dir))


def build_feature_validation_plan(
    *,
    case_name: str = "cs4",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = (DEFAULT_SEED,),
) -> tuple[Path, Path]:
    """Build a Cs4 or multi-isotope feature-validation plan."""
    normalized = str(case_name).strip().lower()
    if normalized == "cs4":
        return build_cs4_feature_validation_plan(
            output_dir=DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir),
            seeds=seeds,
        )
    if normalized != "mix9":
        raise ValueError("case_name must be 'cs4' or 'mix9'.")
    case = MIX9_CASE
    entries = build_ablation_plan(
        output_dir=DEFAULT_MIX9_OUTPUT_DIR if output_dir is None else Path(output_dir),
        seeds=tuple(int(seed) for seed in seeds),
        cases=(case,),
        variants=_feature_variants(tuple(case.isotopes)),
        intensity_cps_1m=DEFAULT_SOURCE_INTENSITY_RANGE_CPS_1M,
    )
    return write_ablation_plan(
        entries,
        output_dir=DEFAULT_MIX9_OUTPUT_DIR if output_dir is None else Path(output_dir),
    )


def main() -> None:
    """Run the Cs4 feature-validation plan generator."""
    args = _parse_args()
    output_dir = (
        DEFAULT_MIX9_OUTPUT_DIR
        if args.case == "mix9" and args.output_dir == DEFAULT_OUTPUT_DIR
        else args.output_dir
    )
    manifest_path, script_path = build_feature_validation_plan(
        case_name=args.case,
        output_dir=output_dir,
        seeds=tuple(int(seed) for seed in args.seeds),
    )
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote run script: {script_path}")


if __name__ == "__main__":
    main()
