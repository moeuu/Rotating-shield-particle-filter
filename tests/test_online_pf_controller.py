"""Integration contracts for the separated online particle-filter controller."""

from __future__ import annotations

import inspect

from realtime_demo import (
    DEFAULT_PF_CONFIG,
    load_online_runtime_configs,
    run_live_pf,
)
from runtime.assets import simulation_runtime_root, standard_geant4_config_path
from runtime.session import estimator_neutral_runtime_config
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
    assert online["structural_rj_multi_component_max_group_size"] == 4
    assert online["joint_guided_initialization"] is True
    assert online["target_ess_ratio"] == 0.4


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


def test_online_controller_stages_raw_spectra_before_pf_updates() -> None:
    """Acquired spectra must become durable before the estimator consumes them."""
    source = inspect.getsource(run_live_pf)

    append_offset = source.index("measurement_log_writer.append_before_update(")
    station_offset = source.index(
        "measurement_log_writer.mark_station_complete_before_update("
    )
    update_offset = source.index("estimator.update_spectrum_station(")

    assert append_offset < station_offset < update_offset


def test_online_controller_passes_only_physical_config_to_simulator() -> None:
    """Estimator settings must not leak across the simulation boundary."""
    source = inspect.getsource(run_live_pf)

    call_start = source.index("simulation_runtime = create_simulation_runtime(")
    call_end = source.index("\n    )", call_start)
    call_source = source[call_start:call_end]

    assert "runtime_config=physical_runtime_config" in call_source
    assert "runtime_config=runtime_config" not in call_source
