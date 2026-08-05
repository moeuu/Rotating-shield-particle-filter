"""Integration contracts for the separated online particle-filter controller."""

from __future__ import annotations

import inspect
import json

from pf.closed_loop import run_pf_closed_loop
from pf.online_config import (
    DEFAULT_PF_CONFIG,
    _validated_provided_source_provenance,
    load_online_runtime_configs,
)
from runtime.assets import simulation_runtime_root, standard_geant4_config_path
from runtime.adaptive_client import AdaptiveRuntimeClient
from runtime.session import estimator_neutral_runtime_config
from sim.protocol import encode_message
from sim.runtime import load_runtime_config
from spectrum.transport_spectral import (
    geometry_conditioned_model_from_runtime_config,
)


def test_standard_online_config_preserves_original_pf_contract() -> None:
    """The separated default must retain the production online PF settings."""
    physical, online = load_online_runtime_configs(
        standard_geant4_config_path(),
        DEFAULT_PF_CONFIG,
    )

    assert "pure_pf_schema_version" not in physical
    assert online["pure_pf_schema_version"] == 1
    assert online["estimator_profile"] == "pf_strict"
    assert online["num_particles"] == 2000
    assert online["variable_cardinality"] is True
    assert online["pf_max_sources"] == 5
    assert online["pf_strength_prior_min_cps_1m"] == 300_000.0
    assert online["pf_strength_prior_family"] == "shifted_gamma"
    assert online["pf_strength_prior_gamma_shape"] == 2.0
    assert online["joint_strength_block_probability"] > 0.0
    assert online["joint_strength_block_batch_size"] == 128
    assert online["joint_cross_isotope_transfer_probability"] == 0.0
    assert online["structural_rj_multi_component_max_group_size"] == 4
    assert online["joint_guided_initialization"] is True
    assert online["target_ess_ratio"] == 0.4


def test_legacy_combined_config_is_split_before_simulation(tmp_path) -> None:
    """A generated combined trial must keep PF controls estimator-side only."""
    combined = json.loads(DEFAULT_PF_CONFIG.read_text(encoding="utf-8"))
    standard_physical = load_runtime_config(standard_geant4_config_path())
    combined.update(standard_physical)
    combined["cui_truth_display_mode"] = "evaluation_live"
    combined["num_particles"] = 2000
    config_path = tmp_path / "combined.json"
    config_path.write_text(json.dumps(combined), encoding="utf-8")

    resolved_physical, online = load_online_runtime_configs(config_path, None)

    assert resolved_physical["backend"] == "geant4"
    assert "cui_truth_display_mode" not in resolved_physical
    assert "num_particles" not in resolved_physical
    assert online["cui_truth_display_mode"] == "evaluation_live"
    assert online["num_particles"] == 2000


def test_standard_measurement_log_config_remains_estimator_neutral() -> None:
    """Raw full spectra must be reusable without embedding PF configuration."""
    physical, _ = load_online_runtime_configs(
        standard_geant4_config_path(),
        DEFAULT_PF_CONFIG,
    )
    model = geometry_conditioned_model_from_runtime_config(physical)
    isotopes = tuple(
        sorted({str(row["isotope"]) for row in model.line_identity})
    )

    logged = estimator_neutral_runtime_config(
        physical,
        backend="geant4",
        isotopes=isotopes,
        run_root=simulation_runtime_root(),
    )

    assert logged["simulation_runtime_schema_version"] == 1
    assert logged["candidate_isotopes"] == list(isotopes)
    assert logged["full_spectrum_contract_hash_sha256"] == (
        model.contract_hash_sha256
    )
    assert not [
        key
        for key in logged
        if key.startswith(("pf_", "structural_rj_", "dss_"))
    ]
    assert "pure_pf_schema_version" not in logged
    assert "estimator_profile" not in logged


def test_closed_loop_receives_runtime_record_before_pf_updates() -> None:
    """Acquired spectra must cross the runtime boundary before PF ingestion."""
    source = inspect.getsource(run_pf_closed_loop)

    request_offset = source.index("event = client.request(")
    parse_offset = source.index("record = parse_adaptive_record(")
    station_offset = source.index("station_records.append(record)")
    update_offset = source.index("_assimilate_station(")

    assert request_offset < parse_offset < station_offset < update_offset


def test_closed_loop_uses_runtime_client_not_simulator_in_process() -> None:
    """Estimator settings must stay outside direct simulator construction."""
    source = inspect.getsource(run_pf_closed_loop)
    client_source = inspect.getsource(AdaptiveRuntimeClient.__init__)

    assert "AdaptiveRuntimeClient(" in source
    assert "create_simulation_runtime" not in source
    assert "rotating-shield-sim" in client_source
    assert "run-adaptive-session" in client_source


def test_closed_loop_binds_final_log_after_live_assimilation() -> None:
    """Live posterior publication must wait for immutable MeasurementLog binding."""
    source = inspect.getsource(run_pf_closed_loop)

    update_offset = source.index("_assimilate_station(")
    finalize_offset = source.index("published = client.finalize()")
    bind_offset = source.index("bind_finalized_measurement_log(estimator, log)")
    write_offset = source.index("_write_final_outputs(")

    assert update_offset < finalize_offset < bind_offset < write_offset


def test_source_provenance_preserves_uint64_seed_across_json_protocol() -> None:
    """Unsigned seed provenance must cross JSON without numeric rounding."""
    derived_seed = 2**63 + 17
    provenance = _validated_provided_source_provenance(
        {
            "provided_file_path": "sources.json",
            "provided_file_path_kind": "repository_relative",
            "provided_file_bytes_sha256": "a" * 64,
            "provided_file_declared_metadata": {
                "source_derived_seed": derived_seed,
                "source_rng_provenance": {
                    "streams": {
                        "truth": {"derived_seed_u64": derived_seed}
                    }
                },
            },
        }
    )

    declared = provenance["provided_file_declared_metadata"]
    assert declared["source_derived_seed"] == str(derived_seed)
    assert declared["source_rng_provenance"]["streams"]["truth"][
        "derived_seed_u64"
    ] == str(derived_seed)
    encode_message("reset", {"source_sampling": provenance})
