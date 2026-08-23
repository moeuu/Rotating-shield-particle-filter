"""Integration contracts for the separated online particle-filter controller."""

from __future__ import annotations

import inspect

from pf.closed_loop import _refine_and_replan, run_pf_closed_loop
from runtime.assets import simulation_runtime_root, standard_geant4_config_path
from runtime.adaptive_client import AdaptiveRuntimeClient
from runtime.session import estimator_neutral_runtime_config
from sim.runtime import load_runtime_config
from spectrum.transport_spectral import (
    geometry_conditioned_model_from_runtime_config,
)


def test_standard_measurement_log_config_remains_estimator_neutral() -> None:
    """Raw full spectra must be reusable without embedding PF configuration."""
    physical = load_runtime_config(standard_geant4_config_path())
    model = geometry_conditioned_model_from_runtime_config(physical)
    isotopes = tuple(sorted({str(row["isotope"]) for row in model.line_identity}))

    logged = estimator_neutral_runtime_config(
        physical,
        backend="geant4",
        isotopes=isotopes,
        run_root=simulation_runtime_root(),
    )

    assert logged["simulation_runtime_schema_version"] == 1
    assert logged["candidate_isotopes"] == list(isotopes)
    assert logged["full_spectrum_contract_hash_sha256"] == (model.contract_hash_sha256)
    assert not [
        key for key in logged if key.startswith(("pf_", "structural_rj_", "dss_"))
    ]
    assert "pure_pf_schema_version" not in logged
    assert "estimator_profile" not in logged


def test_closed_loop_receives_runtime_record_before_pf_updates() -> None:
    """Acquired spectra must cross the runtime boundary before PF ingestion."""
    source = inspect.getsource(run_pf_closed_loop)

    request_offset = source.index("event = client.acquire(")
    parse_offset = source.index("record = event.record")
    station_offset = source.index("station_records.append(record)")
    update_offset = source.index("_assimilate_station(", station_offset)

    assert request_offset < parse_offset < station_offset < update_offset


def test_closed_loop_uses_runtime_client_not_simulator_in_process() -> None:
    """Estimator settings must stay outside direct simulator construction."""
    source = inspect.getsource(run_pf_closed_loop)
    client_source = inspect.getsource(AdaptiveRuntimeClient.connect)

    assert "AdaptiveRuntimeClient.connect(" in source
    assert "create_simulation_runtime" not in source
    assert "scenario_path" not in source
    assert "private_scene_profile" not in source
    assert "AF_UNIX" in client_source


def test_generic_pf_has_no_ral_baseline_import() -> None:
    """RA-L policy implementations must remain outside the generic PF package."""
    source = inspect.getsource(__import__("pf.closed_loop", fromlist=["*"]))

    assert "baselines.ral_ablation" not in source
    assert "ral-mix9" not in source


def test_closed_loop_uses_typed_adaptive_lifecycle_and_refinement() -> None:
    """PF control must delegate adaptive envelope validation to runtime DTOs."""
    source = inspect.getsource(run_pf_closed_loop)
    refinement_source = inspect.getsource(_refine_and_replan)

    assert "client.handshake()" in source
    assert "client.acquire(" in source
    assert "client.finalize_log()" in source
    assert "client.refine_candidates(" in refinement_source
    assert "AdaptiveRefineRequest.from_indices(" in refinement_source
    assert "parse_adaptive_" not in source
    assert "_strict_fields(" not in source


def test_closed_loop_binds_final_log_after_live_assimilation() -> None:
    """Live posterior publication must wait for immutable MeasurementLog binding."""
    source = inspect.getsource(run_pf_closed_loop)

    update_offset = source.index("_assimilate_station(")
    finalize_offset = source.index("published = client.finalize_log()")
    records_offset = source.index("live_records = tuple(")
    bind_offset = source.index("bind_published_measurement_log(")
    write_offset = source.index("_write_final_outputs(")

    assert "for station_records in station_history" in source
    assert update_offset < finalize_offset < records_offset < bind_offset < write_offset
