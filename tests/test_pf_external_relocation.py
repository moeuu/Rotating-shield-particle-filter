"""Tests for the opt-in target-preserving external PF relocation boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from pf.external_relocation import (
    ExternalRelocationError,
    GaussianCandidate,
    IsotopeCandidateMixture,
    aggregate_candidate_outcomes,
    batched_count_log_targets_for_relocations,
    covered_records_sha256,
    directive_from_mapping,
    metropolis_log_acceptance,
    scalar_count_log_targets_for_relocations,
)
from pf.hybrid_replay import replay_with_external_relocations
from pf.profiles import ProposalOrigin, enforce_pure_runtime_settings
from pf.provenance import sha256_json
from pf.replay import (
    build_replay_estimator,
    replay_measurement_log,
    replay_records,
)
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    load_measurement_log,
)
from tests.pure_pf_test_support import make_measurement_log, replay_config


def _native_directive(log: object, estimator: object) -> dict[str, object]:
    """Return one exact cutoff-bound native relocation directive."""
    return {
        "schema_version": 1,
        "directive_id": "directive-test-001",
        "directive_kind": "fixed_cardinality_position_relocation",
        "proposal_source": "external_surface_estimator_fixture",
        "source_run_id": log.run_id,  # type: ignore[attr-defined]
        "covered_records_sha256": covered_records_sha256(log, 2),  # type: ignore[arg-type]
        "source_measurement_log_sha256": log.log_sha256,  # type: ignore[attr-defined]
        "pf_resolved_config_sha256": estimator.resolved_config_hash,  # type: ignore[attr-defined]
        "apply_after_record_index": 1,
        "data_cutoff_step": 1,
        "data_cutoff_station": 0,
        "covered_step_ids": [0, 1],
        "defensive_weight": 0.25,
        "defensive_sigma_xyz_m": [0.18, 0.18, 0.18],
        "isotopes": {
            "Cs-137": {
                "candidates": [
                    {
                        "mean_xyz": [1.55, 1.4, 0.8],
                        "sigma_xyz_m": [0.22, 0.22, 0.18],
                        "weight": 1.0,
                        "proposal_id": "proposal-cs-001",
                        "snapshot_candidate_id": "cluster-cs-001",
                    }
                ]
            }
        },
    }


def _fixture_context(tmp_path: Path) -> tuple[object, dict[str, object], object]:
    """Build a marked log, resolved config, and matching estimator."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    config = enforce_pure_runtime_settings(replay_config(), profile="pf_strict")
    estimator = build_replay_estimator(
        log,
        config,
        profile="pf_strict",
        seed=37,
        config_hash=sha256_json(replay_config()),
        resolved_config_hash=None,
    )
    return log, config, estimator


def _shared_contract_prefix_log() -> MeasurementLog:
    """Recreate the first three neutral rows of the shared cross-repo fixture."""
    isotope_order = ("Cs-137", "Co-60", "Eu-154")
    isotope_counts = np.asarray(
        [[260.0, 95.0, 150.0], [215.0, 120.0, 135.0], [175.0, 145.0, 118.0]],
        dtype=float,
    )
    edges = np.linspace(0.0, 1600.0, 17, dtype=float)
    background = np.asarray(
        [8, 7, 6, 5, 5, 5, 6, 5, 5, 5, 6, 7, 7, 6, 6, 5],
        dtype=float,
    )
    spectra = np.broadcast_to(background, (3, 16)).copy()
    line_models = (
        ((662.0, 0.85),),
        ((1173.0, 0.5), (1332.0, 0.5)),
        (
            (723.3, 0.25),
            (873.2, 0.14),
            (996.3, 0.14),
            (1274.5, 0.45),
            (1494.0, 0.01),
            (1596.5, 0.02),
        ),
    )
    for isotope_index, lines in enumerate(line_models):
        normalization = sum(weight for _energy, weight in lines)
        for energy, weight in lines:
            bin_index = int(np.searchsorted(edges, energy, side="right") - 1)
            normalized_weight = weight / normalization
            spectra[:, bin_index] += (
                normalized_weight * isotope_counts[:, isotope_index]
            )
    covariance = np.zeros((3, 3, 3), dtype=float)
    for row_index in range(3):
        diagonal = isotope_counts[row_index] * 1.2
        covariance[row_index] = np.diag(diagonal)
        for left in range(3):
            for right in range(left + 1, 3):
                value = 0.03 * float(np.sqrt(diagonal[left] * diagonal[right]))
                covariance[row_index, left, right] = value
                covariance[row_index, right, left] = value
    records = tuple(
        MeasurementLogRecord(
            step_id=index,
            action_id=index,
            station_id=0,
            detector_pose_xyz=(0.8, 0.8, 0.45),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=(0, 2, 5)[index],
            pb_orientation_index=(0, 4, 7)[index],
            live_time_s=10.0,
            travel_time_s=0.0,
            shield_actuation_time_s=(0.0, 0.6, 0.6)[index],
            energy_bin_edges_keV=edges,
            spectrum_counts=spectra[index],
            spectrum_variance=spectra[index] + 1.0,
            isotope_counts={
                isotope: float(isotope_counts[index, isotope_index])
                for isotope_index, isotope in enumerate(isotope_order)
            },
            isotope_count_covariance=covariance[index],
            metadata={
                "acquisition": "analytic_shared_fixture",
                "shield_pair": ("0:0", "2:4", "5:7")[index],
            },
        )
        for index in range(3)
    )
    return MeasurementLog(
        run_manifest={
            "run_id": "shared-small-run-v1",
            "schema_version": 1,
            "resolved_config_sha256": "a" * 64,
            "isotopes": list(isotope_order),
        },
        runtime_config={},
        environment={},
        forward_model_manifest={},
        records=records,
    )


def test_covered_record_digest_matches_shared_cross_repo_fixture() -> None:
    """The PF prefix digest must equal the orchestrator/MLE fixture constant."""
    assert covered_records_sha256(_shared_contract_prefix_log(), 3) == (
        "f57c5e5cc83689dfed4b12310e3b63d27e3e95d0c5d53e0904763879f7430efb"
    )


def test_strict_estimator_still_refuses_external_mle_origin(tmp_path: Path) -> None:
    """The opt-in wrapper must not widen the PurePFEstimator boundary."""
    _log, _config, estimator = _fixture_context(tmp_path)
    assert estimator.accepts_proposal_origin(ProposalOrigin.EXTERNAL_MLE) is False


def test_empty_schedule_preserves_standard_replay_outputs_byte_for_byte(
    tmp_path: Path,
) -> None:
    """Predictive instrumentation and an empty wrapper must be a pure no-op."""
    log_path = make_measurement_log(
        tmp_path / "measurement-log",
        record_count=4,
        station_complete_markers=True,
    )
    pure_output = tmp_path / "pure-output"
    hybrid_output = tmp_path / "hybrid-output"
    pure_estimator, pure_trace = replay_measurement_log(
        log_path,
        replay_config(),
        profile="pf_strict",
        seed=37,
        output_dir=pure_output,
    )
    hybrid_estimator, hybrid_trace, schedule, predictions = (
        replay_with_external_relocations(
            log_path,
            replay_config(),
            {"schema_version": 1, "directives": []},
            profile="pf_strict",
            seed=37,
            output_dir=hybrid_output,
        )
    )
    assert hybrid_estimator.serialized_state() == pure_estimator.serialized_state()
    assert hybrid_trace == pure_trace
    assert schedule.receipts == []
    assert len(predictions) == 4
    assert set(predictions[0]["isotopes"]) == {"Cs-137", "Co-60", "Eu-154"}
    for filename in ("pf_posterior.json", "pf_trace.jsonl", "pf_diagnostics.json"):
        assert (hybrid_output / filename).read_bytes() == (
            pure_output / filename
        ).read_bytes()


def test_duplicate_directive_id_is_idempotent_and_deterministic(tmp_path: Path) -> None:
    """An identical repeated directive ID is applied once with one RNG stream."""
    log, config, estimator = _fixture_context(tmp_path)
    directive = _native_directive(log, estimator)
    single = {"schema_version": 1, "directives": [directive]}
    duplicate = {"schema_version": 1, "directives": [directive, directive]}
    first, _trace_a, first_schedule, _pred_a = replay_with_external_relocations(
        log.path,  # type: ignore[attr-defined]
        config,
        single,
        seed=37,
        relocation_seed=101,
    )
    second, _trace_b, second_schedule, _pred_b = replay_with_external_relocations(
        log.path,  # type: ignore[attr-defined]
        config,
        duplicate,
        seed=37,
        relocation_seed=101,
    )
    assert first.serialized_state() == second.serialized_state()
    assert len(first_schedule.receipts) == len(second_schedule.receipts) == 1
    assert first_schedule.receipts == second_schedule.receipts
    assert first_schedule.applied_directive_ids == ("directive-test-001",)


def test_future_directive_remains_pending_and_cannot_change_prefix(
    tmp_path: Path,
) -> None:
    """A schedule item after stop_after must not consume RNG or alter the prefix."""
    log, config, estimator = _fixture_context(tmp_path)
    schedule_payload = {
        "schema_version": 1,
        "directives": [_native_directive(log, estimator)],
    }
    hybrid, _trace, schedule, _predictions = replay_with_external_relocations(
        log.path,  # type: ignore[attr-defined]
        config,
        schedule_payload,
        seed=37,
        stop_after=1,
    )
    pure = build_replay_estimator(log, config, profile="pf_strict", seed=37)
    replay_records(log, pure, stop_after=1)
    assert hybrid.serialized_state() == pure.serialized_state()
    assert schedule.applied_directive_ids == ()
    assert schedule.pending_directive_ids == ("directive-test-001",)


def test_unseen_log_suffix_does_not_change_cutoff_state_or_relocation_rng(
    tmp_path: Path,
) -> None:
    """Prefix hashing and relocation RNG must be independent of a future suffix."""
    short_log = load_measurement_log(
        make_measurement_log(
            tmp_path / "short-log",
            record_count=4,
            station_complete_markers=True,
        )
    )
    long_log = load_measurement_log(
        make_measurement_log(
            tmp_path / "long-log",
            record_count=6,
            station_complete_markers=True,
        )
    )
    config = enforce_pure_runtime_settings(replay_config(), profile="pf_strict")
    short_builder = build_replay_estimator(
        short_log, config, profile="pf_strict", seed=37
    )
    long_builder = build_replay_estimator(
        long_log, config, profile="pf_strict", seed=37
    )
    short_directive = _native_directive(short_log, short_builder)
    long_directive = _native_directive(long_log, long_builder)
    assert short_log.log_sha256 != long_log.log_sha256
    assert (
        short_directive["covered_records_sha256"]
        == long_directive["covered_records_sha256"]
    )
    short_estimator, short_trace, _schedule_a, short_predictions = (
        replay_with_external_relocations(
            short_log.path,
            config,
            {"schema_version": 1, "directives": [short_directive]},
            seed=37,
            relocation_seed=101,
            stop_after=2,
        )
    )
    long_estimator, long_trace, _schedule_b, long_predictions = (
        replay_with_external_relocations(
            long_log.path,
            config,
            {"schema_version": 1, "directives": [long_directive]},
            seed=37,
            relocation_seed=101,
            stop_after=2,
        )
    )
    assert short_estimator.serialized_state() == long_estimator.serialized_state()
    assert short_trace[-1]["state_sha256"] == long_trace[-1]["state_sha256"]
    assert short_predictions == long_predictions


def test_directive_rejects_future_coverage_and_non_boundary_cutoff(
    tmp_path: Path,
) -> None:
    """Coverage and cutoff markers are checked before replay mutation."""
    log, config, estimator = _fixture_context(tmp_path)
    future = _native_directive(log, estimator)
    future["covered_step_ids"] = [0, 1, 2]
    with pytest.raises(ExternalRelocationError, match="exact log prefix"):
        replay_with_external_relocations(
            log.path,  # type: ignore[attr-defined]
            config,
            {"schema_version": 1, "directives": [future]},
            seed=37,
        )
    non_boundary = dict(_native_directive(log, estimator))
    non_boundary.update(
        {
            "apply_after_record_index": 0,
            "data_cutoff_step": 0,
            "covered_step_ids": [0],
            "covered_records_sha256": covered_records_sha256(log, 1),
        }
    )
    with pytest.raises(ExternalRelocationError, match="station boundaries"):
        replay_with_external_relocations(
            log.path,  # type: ignore[attr-defined]
            config,
            {"schema_version": 1, "directives": [non_boundary]},
            seed=37,
        )


def test_batched_relocation_target_matches_scalar_oracle(tmp_path: Path) -> None:
    """The default batched target equals a serial state-by-state oracle."""
    log, config, estimator = _fixture_context(tmp_path)
    replay_records(log, estimator, stop_after=2)
    filt = estimator.filters["Cs-137"]
    data = estimator._measurement_data_for_iso("Cs-137", None)
    assert data is not None
    indices = np.arange(min(5, len(filt.continuous_particles)), dtype=np.int64)
    slots = np.zeros(indices.size, dtype=np.int64)
    proposed = np.asarray(
        [
            np.clip(
                filt.continuous_particles[int(index)].state.positions[0]
                + np.asarray([0.07, -0.03, 0.04]),
                filt.config.position_min,
                filt.config.position_max,
            )
            for index in indices
        ],
        dtype=float,
    )
    batched = batched_count_log_targets_for_relocations(
        filt, data, indices, slots, proposed
    )
    scalar = scalar_count_log_targets_for_relocations(
        filt, data, indices, slots, proposed
    )
    np.testing.assert_allclose(batched[0], scalar[0], rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(batched[1], scalar[1], rtol=1.0e-11, atol=1.0e-11)


def test_metropolis_acceptance_has_exact_target_and_q_correction() -> None:
    """The analytic MH rule clips only after adding the reverse/forward q term."""
    target_delta = np.asarray([1.0, -0.5, -3.0])
    q_correction = np.asarray([-0.25, 0.75, -0.2])
    np.testing.assert_array_equal(
        metropolis_log_acceptance(target_delta, q_correction),
        np.asarray([0.0, 0.0, -3.2]),
    )


def test_candidate_outcomes_report_all_particle_results_without_representative() -> (
    None
):
    """Mixed candidate draws must retain honest aggregate MH disposition counts."""
    mixture = IsotopeCandidateMixture(
        candidates=(
            GaussianCandidate(
                mean_xyz=(1.0, 1.0, 1.0),
                sigma_xyz_m=(0.2, 0.2, 0.2),
                weight=1.0,
                proposal_id="proposal-mixed",
            ),
            GaussianCandidate(
                mean_xyz=(2.0, 2.0, 2.0),
                sigma_xyz_m=(0.2, 0.2, 0.2),
                weight=1.0,
                proposal_id="proposal-singleton",
            ),
            GaussianCandidate(
                mean_xyz=(3.0, 3.0, 3.0),
                sigma_xyz_m=(0.2, 0.2, 0.2),
                weight=1.0,
                proposal_id="proposal-unsampled",
            ),
        )
    )
    outcomes = aggregate_candidate_outcomes(
        mixture,
        {
            "eligible_particle_count": 5,
            "proposal_components": [0, 0, 1, -1, 0],
            "accepted": [True, False, True, False, True],
            "log_acceptance_ratio": [-0.1, -2.0, 0.0, -0.5, -0.2],
            "log_uniform_draws": [-0.2, -0.3, -0.7, -0.4, -0.4],
        },
    )
    assert outcomes == [
        {
            "proposal_id": "proposal-mixed",
            "outcome": "mh_mixed",
            "mh_attempt_count": 3,
            "mh_accepted_count": 2,
            "mh_rejected_count": 1,
            "not_sampled_count": 2,
            "eligible_particle_count": 5,
            "mh_log_acceptance_ratio": None,
            "mh_log_uniform_draw": None,
        },
        {
            "proposal_id": "proposal-singleton",
            "outcome": "mh_accepted",
            "mh_attempt_count": 1,
            "mh_accepted_count": 1,
            "mh_rejected_count": 0,
            "not_sampled_count": 4,
            "eligible_particle_count": 5,
            "mh_log_acceptance_ratio": 0.0,
            "mh_log_uniform_draw": -0.7,
        },
        {
            "proposal_id": "proposal-unsampled",
            "outcome": "not_applied",
            "mh_attempt_count": 0,
            "mh_accepted_count": 0,
            "mh_rejected_count": 0,
            "not_sampled_count": 5,
            "eligible_particle_count": 5,
            "mh_log_acceptance_ratio": None,
            "mh_log_uniform_draw": None,
        },
    ]


def test_orchestrator_directive_translates_without_strength_mutation(
    tmp_path: Path,
) -> None:
    """PFDirective v1 maps its candidate to the position-only internal kernel."""
    log, _config, estimator = _fixture_context(tmp_path)
    payload = {
        "schema_version": 1,
        "directive_id": "directive-orchestrator-001",
        "directive_kind": "proposal_only_mh",
        "snapshot_id": "snapshot-001",
        "snapshot_sha256": "b" * 64,
        "source_run_id": log.run_id,
        "prefix_measurement_log_sha256": "c" * 64,
        "covered_records_sha256": covered_records_sha256(log, 2),
        "covered_station_boundaries_sha256": "d" * 64,
        "pf_resolved_config_sha256": estimator.resolved_config_hash,
        "data_cutoff_step": 1,
        "data_cutoff_station": 0,
        "cutoff_station_complete": True,
        "covered_step_ids": [0, 1],
        "apply_after_step": 1,
        "corroboration_min_step": 2,
        "proposals": [
            {
                "proposal_id": "proposal-cs-001",
                "snapshot_candidate_id": "cluster-cs-001",
                "isotope": "Cs-137",
                "candidate_mean_xyz": [1.5, 1.25, 0.7],
                "snapshot_strength_cps_1m_metadata": 1234.0,
                "proposal_kernel": {
                    "family": "defensive_truncated_gaussian_position",
                    "position_sigma_xyz_m": [0.2, 0.2, 0.2],
                    "defensive_weight": 1.0,
                    "candidate_weight": 1.0,
                },
            }
        ],
        "safety_policy": {
            "direct_mle_objective_reweight": False,
            "hard_prune_authorized": False,
            "future_only_corroboration": True,
            "once_only_application": True,
            "requires_target_preserving_mh": True,
        },
        "provenance": {},
    }
    directive = directive_from_mapping(payload)
    candidate = directive.isotopes["Cs-137"].candidates[0]
    assert directive.directive_kind == "proposal_only_mh"
    assert directive.apply_after_record_index == 1
    assert candidate.mean_xyz == (1.5, 1.25, 0.7)
    assert candidate.proposal_id == "proposal-cs-001"
    assert not hasattr(candidate, "strength_cps_1m")

    pure = build_replay_estimator(log, replay_config(), profile="pf_strict", seed=37)
    replay_records(log, pure, stop_after=2)
    hybrid, _trace, schedule, _predictions = replay_with_external_relocations(
        log.path,
        replay_config(),
        {"schema_version": 1, "directives": [payload]},
        seed=37,
        relocation_seed=103,
        stop_after=2,
        output_dir=tmp_path / "orchestrator-hybrid-output",
    )
    pure_filter = pure.filters["Cs-137"]
    hybrid_filter = hybrid.filters["Cs-137"]
    np.testing.assert_array_equal(
        [particle.log_weight for particle in hybrid_filter.continuous_particles],
        [particle.log_weight for particle in pure_filter.continuous_particles],
    )
    for pure_particle, hybrid_particle in zip(
        pure_filter.continuous_particles,
        hybrid_filter.continuous_particles,
    ):
        np.testing.assert_array_equal(
            np.sort(hybrid_particle.state.strengths),
            np.sort(pure_particle.state.strengths),
        )
        assert hybrid_particle.state.num_sources == pure_particle.state.num_sources
        assert hybrid_particle.state.background == pure_particle.state.background
    assert len(schedule.receipts) == 1
    contract = schedule.receipts[0]["contract_receipt"]
    assert contract["directive_sha256"] == sha256_json(payload)
    assert contract["safety_evidence"] == {
        "direct_mle_objective_reweight_performed": False,
        "hard_prune_performed": False,
        "target_preserving_mh_performed": True,
        "reweighted_observation_step_ids": [],
        "next_observation_min_step": 2,
    }
    eligible_count = schedule.receipts[0]["isotopes"][0]["eligible_particle_count"]
    assert contract["candidate_outcomes"] == [
        {
            "proposal_id": "proposal-cs-001",
            "outcome": "not_applied",
            "mh_attempt_count": 0,
            "mh_accepted_count": 0,
            "mh_rejected_count": 0,
            "not_sampled_count": eligible_count,
            "eligible_particle_count": eligible_count,
            "mh_log_acceptance_ratio": None,
            "mh_log_uniform_draw": None,
        }
    ]
    receipt_files = tuple(
        (tmp_path / "orchestrator-hybrid-output" / "pf_directive_receipts").glob(
            "*.json"
        )
    )
    assert len(receipt_files) == 1
    assert json.loads(receipt_files[0].read_text(encoding="utf-8")) == contract


def test_external_boundary_has_no_estimator_package_import() -> None:
    """The PF-side boundary must remain estimator-neutral at import time."""
    source = (
        Path(__file__).parents[1] / "src" / "pf" / "external_relocation.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("three_d_estimation" in name for name in imported)
    assert not any(name.startswith("orchestrator") for name in imported)
