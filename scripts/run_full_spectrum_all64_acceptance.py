"""Run the resumable external-Geant4 all-64 acceptance acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from spectrum.full_spectrum_acceptance import (
    write_independent_validation_manifest,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_PAIR_IDS,
    AcceptanceRunLayout,
    acquire_designated_split,
    build_acceptance_run_contract,
    build_complete_training_manifest,
    canonical_json_bytes,
    fit_training_additive_scatter,
    freeze_candidate_model,
    line_identity_contract_sha256,
    load_acceptance_pair,
    select_training_discrepancy,
)
from spectrum.full_spectrum_acceptance_evaluator import (
    approve_frozen_candidate,
    evaluate_all_designated_scenes,
)
from spectrum.geant4_acceptance_backend import (
    ExternalGeant4AcceptanceBackend,
)
from spectrum.transport_spectral import (
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    GeometryConditionedSpectralModel,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_OUTPUT = (
    _REPOSITORY_ROOT / "results" / "full_spectrum_all64_acceptance"
)
_IMPLEMENTATION_STATIC_PATHS = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("configs/validation/full_spectrum_acceptance_v1.json"),
    Path("scripts/run_full_spectrum_all64_acceptance.py"),
)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    """Add the shared native acquisition arguments to one subparser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=(
            "Standard external-Geant4 runtime config "
            f"(default: {_DEFAULT_CONFIG})."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Immutable/resumable artifact root "
            f"(default: {_DEFAULT_OUTPUT})."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    """Return the fail-closed phase-oriented command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the fixed full-spectrum training/holdout corpus with "
            "real external Geant4. Pair files are immutable checkpoints, so "
            "an interrupted phase can be rerun safely."
        )
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    status = subparsers.add_parser(
        "status",
        help="Report immutable pair/corpus/model checkpoint counts.",
    )
    _common_parser(status)

    smoke = subparsers.add_parser(
        "smoke",
        help=(
            "Acquire one real 30 s background pair plus the native signed-"
            "epsilon gate; the checkpoint is reusable by the training phase."
        ),
    )
    _common_parser(smoke)
    smoke.add_argument(
        "--scene-seed",
        type=int,
        choices=DESIGNATED_TRAINING_SCENE_SEEDS,
        default=DESIGNATED_TRAINING_SCENE_SEEDS[0],
    )
    smoke.add_argument(
        "--pair-id",
        type=int,
        choices=ACCEPTANCE_PAIR_IDS,
        default=0,
    )

    training = subparsers.add_parser(
        "training",
        help="Acquire all designated training scenes, scenarios, and 64 pairs.",
    )
    _common_parser(training)

    fit = subparsers.add_parser(
        "fit-freeze",
        help=(
            "Require the complete training corpus, fit/select on training "
            "only, and immutably freeze the pre-holdout candidate."
        ),
    )
    _common_parser(fit)

    holdout = subparsers.add_parser(
        "holdout",
        help=(
            "Require a frozen candidate, then acquire every designated "
            "holdout scene/scenario/pair without feedback or tuning."
        ),
    )
    _common_parser(holdout)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate the same frozen candidate on all five immutable raw "
            "scene corpora and write one fixed-metric artifact per seed."
        ),
    )
    _common_parser(evaluate)

    approve = subparsers.add_parser(
        "approve",
        help=(
            "Conservatively aggregate holdout-only metrics and create a "
            "production model only when every fixed threshold passes."
        ),
    )
    _common_parser(approve)

    all_phases = subparsers.add_parser(
        "all",
        help=(
            "Run training acquisition, training-only fit/freeze, then "
            "holdout acquisition, fixed evaluation, and approval in order."
        ),
    )
    _common_parser(all_phases)

    aggregate = subparsers.add_parser(
        "aggregate",
        help=(
            "Aggregate exactly five already evaluated scene-acceptance "
            "artifacts into the independent validation manifest."
        ),
    )
    aggregate.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=Path,
        help=(
            "One evaluated scene artifact; pass exactly the three training "
            "and two holdout artifacts."
        ),
    )
    aggregate.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination for the deterministic validation manifest.",
    )
    return parser


def _write_immutable_json(path: Path, payload: object) -> Path:
    """Atomically write one canonical artifact or verify exact resumption."""
    destination = path.resolve()
    encoded = canonical_json_bytes(payload)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise RuntimeError(
                f"Refusing to overwrite incompatible artifact: {destination}."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination


def _progress(message: str) -> None:
    """Print one timestamp-free line suitable for a persistent log."""
    print(message, flush=True)


def _implementation_bundle_sha256(repository_root: Path) -> str:
    """Hash every Python implementation input shared by acceptance phases."""
    root = repository_root.resolve()
    relative_paths = set(_IMPLEMENTATION_STATIC_PATHS)
    relative_paths.update(
        path.relative_to(root)
        for path in (root / "src").rglob("*.py")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for relative_path in sorted(
        relative_paths,
        key=lambda value: value.as_posix(),
    ):
        source = root / relative_path
        if not source.is_file():
            raise FileNotFoundError(
                f"Acceptance implementation input is missing: {source}."
            )
        encoded_path = relative_path.as_posix().encode("utf-8")
        raw = source.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _native_context(
    arguments: argparse.Namespace,
) -> tuple[
    AcceptanceRunLayout,
    ExternalGeant4AcceptanceBackend,
    GeometryConditionedSpectralModel,
    str,
]:
    """Build and authenticate the shared immutable acquisition context."""
    layout = AcceptanceRunLayout(Path(arguments.output_root).resolve())
    backend = ExternalGeant4AcceptanceBackend(
        runtime_config_path=Path(arguments.config),
        repository_root=_REPOSITORY_ROOT,
    )
    run_contract = build_acceptance_run_contract(
        runtime_config_sha256=backend.runtime_config_sha256,
        native_executable_sha256=backend.native_executable_sha256,
        implementation_bundle_sha256=_implementation_bundle_sha256(
            _REPOSITORY_ROOT
        ),
    )
    _write_immutable_json(layout.run_contract_path, run_contract)
    base_model = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=float(backend.app_config.dead_time_tau_s),
        background_rate_cps=float(backend.app_config.background_cps),
    )
    line_hash = line_identity_contract_sha256(base_model)
    return layout, backend, base_model, line_hash


def _run_training(
    *,
    layout: AcceptanceRunLayout,
    backend: ExternalGeant4AcceptanceBackend,
    line_hash: str,
) -> None:
    """Acquire all training checkpoints and freeze the completion manifest."""
    acquire_designated_split(
        layout=layout,
        backend=backend,
        seeds=DESIGNATED_TRAINING_SCENE_SEEDS,
        split="training",
        line_identity_sha256=line_hash,
        progress=_progress,
    )
    completion = build_complete_training_manifest(
        layout=layout,
        line_identity_sha256=line_hash,
    )
    _write_immutable_json(layout.training_complete_path, completion)


def _fit_and_freeze(
    *,
    layout: AcceptanceRunLayout,
    base_model: GeometryConditionedSpectralModel,
) -> None:
    """Fit physical training components and freeze one candidate contract."""
    additive_response = fit_training_additive_scatter(
        layout=layout,
        model=base_model,
    )
    _write_immutable_json(
        layout.additive_model_path,
        additive_response.to_payload(),
    )
    selection, candidate = select_training_discrepancy(
        layout=layout,
        base_model=base_model,
        additive_response=additive_response,
    )
    _write_immutable_json(layout.discrepancy_selection_path, selection)
    freeze_candidate_model(layout=layout, model=candidate)
    _progress(
        "candidate frozen before holdout: "
        f"{candidate.contract_hash_sha256}"
    )


def _run_holdout(
    *,
    layout: AcceptanceRunLayout,
    backend: ExternalGeant4AcceptanceBackend,
    line_hash: str,
) -> None:
    """Acquire all holdout checkpoints only after candidate freezing."""
    acquire_designated_split(
        layout=layout,
        backend=backend,
        seeds=DESIGNATED_HOLDOUT_SCENE_SEEDS,
        split="holdout",
        line_identity_sha256=line_hash,
        progress=_progress,
    )


def _status_payload(layout: AcceptanceRunLayout) -> Mapping[str, object]:
    """Return a compact, deterministic checkpoint summary."""
    split_payload: dict[str, object] = {}
    for split, seeds in (
        ("training", DESIGNATED_TRAINING_SCENE_SEEDS),
        ("holdout", DESIGNATED_HOLDOUT_SCENE_SEEDS),
    ):
        pair_count = len(tuple((layout.root / split).glob("scene_*/*/pair_*.json")))
        corpus_count = sum(
            layout.scene_corpus_path(split=split, scene_seed=seed).is_file()
            for seed in seeds
        )
        split_payload[split] = {
            "pair_checkpoint_count": pair_count,
            "expected_pair_checkpoint_count": len(seeds) * 5 * 64,
            "scene_corpus_manifest_count": corpus_count,
            "expected_scene_corpus_manifest_count": len(seeds),
        }
    return {
        "output_root": layout.root.as_posix(),
        "run_contract": layout.run_contract_path.is_file(),
        "training_complete": layout.training_complete_path.is_file(),
        "additive_model": layout.additive_model_path.is_file(),
        "discrepancy_selection": layout.discrepancy_selection_path.is_file(),
        "candidate_model": layout.candidate_model_path.is_file(),
        "independent_validation": layout.validation_manifest_path.is_file(),
        "production_model": layout.production_model_path.is_file(),
        "splits": split_payload,
    }


def _run_smoke(
    *,
    layout: AcceptanceRunLayout,
    backend: ExternalGeant4AcceptanceBackend,
    line_hash: str,
    scene_seed: int,
    pair_id: int,
) -> None:
    """Acquire one resumable real-native background observation."""
    destination = layout.pair_path(
        split="training",
        scene_seed=scene_seed,
        scenario_id="background_only",
        shield_pair_id=pair_id,
    )
    if not destination.exists():
        with backend.open_scenario(
            scene_seed=scene_seed,
            split="training",
            scenario_id="background_only",
            line_identity_sha256=line_hash,
        ) as session:
            payload = session.acquire_pair(pair_id)
        _write_immutable_json(destination, payload)
    record = load_acceptance_pair(
        destination,
        expected_line_identity_sha256=line_hash,
    )
    _progress(
        "real-native smoke strict-loader passed: "
        f"{destination} total_counts="
        f"{int(np.sum(record.observed_spectrum_counts, dtype=np.int64))}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicitly ordered resumable acceptance phase."""
    arguments = _parser().parse_args(argv)
    if arguments.phase == "aggregate":
        write_independent_validation_manifest(
            arguments.artifact,
            arguments.output,
        )
        return 0

    layout, backend, base_model, line_hash = _native_context(arguments)
    if arguments.phase == "status":
        print(
            json.dumps(
                _status_payload(layout),
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0
    if arguments.phase == "smoke":
        _run_smoke(
            layout=layout,
            backend=backend,
            line_hash=line_hash,
            scene_seed=int(arguments.scene_seed),
            pair_id=int(arguments.pair_id),
        )
        return 0
    if arguments.phase in {"training", "all"}:
        _run_training(
            layout=layout,
            backend=backend,
            line_hash=line_hash,
        )
    if arguments.phase in {"fit-freeze", "all"}:
        _fit_and_freeze(layout=layout, base_model=base_model)
    if arguments.phase in {"holdout", "all"}:
        _run_holdout(
            layout=layout,
            backend=backend,
            line_hash=line_hash,
        )
    if arguments.phase in {"evaluate", "all"}:
        for path in evaluate_all_designated_scenes(layout=layout):
            _progress(f"scene acceptance: {path}")
    if arguments.phase in {"approve", "all"}:
        validation_path, production_path = approve_frozen_candidate(
            layout=layout
        )
        _progress(f"independent validation: {validation_path}")
        _progress(f"approved production model: {production_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
