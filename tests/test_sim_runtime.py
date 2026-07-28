"""Focused runtime tests for native full-spectrum fidelity handshakes."""

from __future__ import annotations

import json

import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from sim.runtime import (
    Geant4TCPClientRuntime,
    _config_bool,
    _config_integer,
    _config_number,
    _config_string,
    load_runtime_config,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_BACKGROUND_MODEL_ID,
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    NATIVE_GEANT4_ENERGY_MAX_KEV,
    NATIVE_GEANT4_ENERGY_MIN_KEV,
)


def _full_spectrum_handshake() -> dict[str, object]:
    """Return a unit-history native reset handshake."""
    return {
        "runtime_fidelity": {
            "primary_sampling_fraction": 1.0,
            "requested_primary_sampling_fraction": 1.0,
            "primary_history_weight": 1.0,
            "target_sampled_primaries": 0,
            "primary_sampling_budget_enabled": False,
            "primary_sampling_fraction_resolution": "fixed_fraction",
            "accelerated_weighted_transport_enable": False,
            "sample_detector_response": True,
            "detector_response_sampling_contract_sha256": (
                NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            ),
            "background_spectrum_model_id": NATIVE_GEANT4_BACKGROUND_MODEL_ID,
            "spectrum_energy_min_keV": NATIVE_GEANT4_ENERGY_MIN_KEV,
            "spectrum_energy_max_keV": NATIVE_GEANT4_ENERGY_MAX_KEV,
            "spectrum_bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
            "spectrum_bin_count": NATIVE_GEANT4_BIN_COUNT,
            "requested_threads": 32,
            "source_position_semantics": "air_side_native_emission_xyz",
            "source_anchor_semantics": (
                "exact_surface_chart_uv_evaluation_truth"
            ),
            "all_sources_surface_bound": True,
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "surface_emission_policy_sha256": (
                surface_emission_policy_sha256()
            ),
            "surface_source_contract_sha256": "b" * 64,
            "scene_hash": "c" * 64,
            "intensity_cps_1m_definition": (
                "pre_dead_time_detector_pulse_rate_at_1m"
            ),
        }
    }


def _client(
    *,
    expected_thread_count: int | None = None,
) -> Geant4TCPClientRuntime:
    """Return an unconnected client configured for production response sampling."""
    return Geant4TCPClientRuntime(
        "127.0.0.1",
        65530,
        expected_detector_response_sampling=True,
        expected_thread_count=expected_thread_count,
    )


@pytest.mark.parametrize("value", (0, 1, "false", "true", None))
def test_sidecar_switches_require_exact_json_booleans(value: object) -> None:
    """A truthy string must never enable a mock or alternate sidecar path."""
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        _config_bool({"sidecar_mock_stage": value}, "sidecar_mock_stage", False)


@pytest.mark.parametrize("value", (True, 5556.0, "5556", None))
def test_sidecar_ports_require_exact_json_integers(value: object) -> None:
    """Port and restart counts must not be silently coerced."""
    with pytest.raises(ValueError, match="must be a JSON integer"):
        _config_integer(
            {"port": value},
            "port",
            5556,
            minimum=1,
            maximum=65535,
        )


@pytest.mark.parametrize("value", (True, "120", None, float("nan")))
def test_sidecar_timeouts_require_finite_json_numbers(value: object) -> None:
    """Invalid timeouts must stop before starting an unmonitored transport."""
    with pytest.raises(ValueError, match="must be|finite"):
        _config_number(
            {"timeout_s": value},
            "timeout_s",
            120.0,
            minimum=0.0,
            strict_minimum=True,
        )


@pytest.mark.parametrize("value", (True, 127, None, " "))
def test_sidecar_strings_reject_stringification(value: object) -> None:
    """Hosts, backends, and material IDs must be exact nonempty strings."""
    with pytest.raises(ValueError, match="JSON string"):
        _config_string({"host": value}, "host", "127.0.0.1")


def test_runtime_config_rejects_duplicate_json_keys(tmp_path) -> None:
    """Duplicate physics settings must not be resolved by last-key wins."""
    config_path = tmp_path / "duplicate.json"
    config_path.write_text(
        '{"background_cps": 12.0, "background_cps": 0.0}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'background_cps'"):
        load_runtime_config(config_path)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_runtime_config_rejects_nonfinite_json_numbers(
    tmp_path,
    constant: str,
) -> None:
    """Non-standard non-finite JSON cannot alter a scientific runtime."""
    config_path = tmp_path / "nonfinite.json"
    config_path.write_text(
        f'{{"background_cps": {constant}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite standard-JSON numbers"):
        load_runtime_config(config_path)


@pytest.mark.parametrize("parent_ref", ("", 1, False, ["parent.json"]))
def test_runtime_config_extends_requires_nonempty_string(
    tmp_path,
    parent_ref: object,
) -> None:
    """Inheritance must not stringify ambiguous JSON values into paths."""
    config_path = tmp_path / "invalid_extends.json"
    config_path.write_text(
        json.dumps({"extends": parent_ref}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonempty JSON string"):
        load_runtime_config(config_path)


def test_reset_handshake_accepts_exact_native_full_spectrum_contract() -> None:
    """The reset boundary authenticates response sampling before any action."""
    _client()._validate_fidelity_handshake(_full_spectrum_handshake())


@pytest.mark.parametrize(
    "key",
    (
        "sample_detector_response",
        "detector_response_sampling_contract_sha256",
        "background_spectrum_model_id",
        "spectrum_energy_min_keV",
        "spectrum_energy_max_keV",
        "spectrum_bin_width_keV",
        "spectrum_bin_count",
        "source_position_semantics",
        "source_anchor_semantics",
        "all_sources_surface_bound",
        "surface_emission_epsilon_m",
        "surface_emission_policy_sha256",
        "surface_source_contract_sha256",
        "scene_hash",
        "intensity_cps_1m_definition",
    ),
)
def test_reset_handshake_rejects_missing_response_contract_field(key: str) -> None:
    """A stale native process cannot survive reset compatibility checks."""
    handshake = _full_spectrum_handshake()
    del handshake["runtime_fidelity"][key]

    with pytest.raises(RuntimeError, match="handshake|response sampling"):
        _client()._validate_fidelity_handshake(handshake)


def test_reset_handshake_rejects_disabled_response_sampling() -> None:
    """Incident-energy histograms without native detector marking are invalid."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["sample_detector_response"] = False

    with pytest.raises(RuntimeError, match="response sampling mismatch"):
        _client()._validate_fidelity_handshake(handshake)


@pytest.mark.parametrize(
    "value",
    (
        float(NATIVE_GEANT4_BIN_COUNT),
        NATIVE_GEANT4_BIN_COUNT + 0.5,
        str(NATIVE_GEANT4_BIN_COUNT),
    ),
)
def test_reset_handshake_rejects_coerced_spectrum_bin_count(
    value: object,
) -> None:
    """A fractional-capable or string bin count cannot authenticate the axis."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["spectrum_bin_count"] = value

    with pytest.raises(RuntimeError, match="spectrum_bin_count"):
        _client()._validate_fidelity_handshake(handshake)


@pytest.mark.parametrize("value", (32.0, 32.5, "32"))
def test_reset_handshake_rejects_coerced_thread_count(value: object) -> None:
    """Native thread-count provenance must remain an exact JSON integer."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["requested_threads"] = value

    with pytest.raises(RuntimeError, match="requested_threads"):
        _client(expected_thread_count=32)._validate_fidelity_handshake(
            handshake
        )


@pytest.mark.parametrize("value", (True, 0, 32.0, "32"))
def test_client_rejects_coerced_expected_thread_count(value: object) -> None:
    """The local thread contract cannot be weakened before reset validation."""
    with pytest.raises(ValueError, match="expected_thread_count"):
        _client(expected_thread_count=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("primary_sampling_fraction", "1.0"),
        ("primary_sampling_fraction", True),
        (
            "surface_emission_epsilon_m",
            str(SURFACE_EMISSION_EPSILON_M),
        ),
        (
            "spectrum_bin_width_keV",
            str(NATIVE_GEANT4_BIN_WIDTH_KEV),
        ),
    ),
)
def test_reset_handshake_rejects_coerced_numeric_provenance(
    key: str,
    value: object,
) -> None:
    """Wire numeric provenance cannot be accepted through float coercion."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"][key] = value

    with pytest.raises(RuntimeError, match=key):
        _client()._validate_fidelity_handshake(handshake)


def test_reset_handshake_binds_source_strength_and_transport_sha() -> None:
    """The native reset must authenticate the exact source payload it received."""
    source = {
        "isotope": "Cs-137",
        "position": [0.0, 0.5, 0.5],
        "transport_position": [
            SURFACE_EMISSION_EPSILON_M,
            0.5,
            0.5,
        ],
        "intensity_cps_1m": 300_000.0,
        "surface_chart_id": 0,
        "surface_uv": [0.5, 0.5],
        "surface_normal": [1.0, 0.0, 0.0],
        "surface_emission_policy_sha256": surface_emission_policy_sha256(),
    }
    expected_hash = surface_source_runtime_contract_sha256([source])
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"][
        "surface_source_contract_sha256"
    ] = expected_hash
    client = _client()
    client._round_trip = lambda *_args, **_kwargs: handshake  # type: ignore[method-assign]

    client.reset({"sources": [source]})

    assert client.expected_surface_source_contract_sha256 == expected_hash
    assert client.expected_scene_hash == "c" * 64

    stale = _full_spectrum_handshake()
    stale["runtime_fidelity"]["surface_source_contract_sha256"] = "d" * 64
    client = _client()
    client._round_trip = lambda *_args, **_kwargs: stale  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="contract hash differs"):
        client.reset({"sources": [source]})
