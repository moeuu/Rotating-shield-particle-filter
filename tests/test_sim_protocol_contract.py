"""Fail-closed tests for simulator command and observation wire payloads."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sim.protocol import (
    SimulationCommand,
    SimulationObservation,
    decode_message,
    encode_message,
)


def _command_payload() -> dict[str, object]:
    """Return one valid command wire payload."""
    return {
        "step_id": 0,
        "target_pose_xyz": [1.0, 2.0, 0.5],
        "target_base_yaw_rad": 0.0,
        "fe_orientation_index": 3,
        "pb_orientation_index": 7,
        "dwell_time_s": 30.0,
        "travel_time_s": 1.0,
        "shield_actuation_time_s": 0.5,
        "travel_waypoints_xyz": [[1.0, 1.0, 0.5]],
    }


def _observation_payload() -> dict[str, object]:
    """Return one valid observation wire payload."""
    return {
        "step_id": 0,
        "detector_pose_xyz": [1.0, 2.0, 0.5],
        "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fe_orientation_index": 3,
        "pb_orientation_index": 7,
        "spectrum_counts": [1.0, 2.0],
        "energy_bin_edges_keV": [0.0, 5.0, 10.0],
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("key", "invalid"),
    (
        ("step_id", "0"),
        ("step_id", True),
        ("fe_orientation_index", "3"),
        ("fe_orientation_index", -1),
        ("fe_orientation_index", 8),
        ("pb_orientation_index", True),
        ("dwell_time_s", "30.0"),
        ("dwell_time_s", 0.0),
        ("travel_time_s", -1.0),
        ("shield_actuation_time_s", float("nan")),
    ),
)
def test_command_rejects_wire_coercion_and_invalid_physics(
    key: str,
    invalid: object,
) -> None:
    """A corrupt command must not be reinterpreted as another physical step."""
    payload = _command_payload()
    payload[key] = invalid

    with pytest.raises((TypeError, ValueError)):
        SimulationCommand.from_dict(payload)


@pytest.mark.parametrize(
    ("key", "invalid"),
    (
        ("step_id", "0"),
        ("fe_orientation_index", 8),
        ("pb_orientation_index", -1),
        ("spectrum_counts", [1.0, float("nan")]),
        ("spectrum_counts", [1.0, -1.0]),
        ("energy_bin_edges_keV", [0.0, 10.0]),
        ("energy_bin_edges_keV", [0.0, 5.0, 5.0]),
        ("detector_quat_wxyz", [0.0, 0.0, 0.0, 0.0]),
    ),
)
def test_observation_rejects_corrupt_wire_values(
    key: str,
    invalid: object,
) -> None:
    """A corrupt response must fail before it can influence the PF."""
    payload = _observation_payload()
    payload[key] = invalid

    with pytest.raises((TypeError, ValueError)):
        SimulationObservation.from_dict(payload)


def test_protocol_round_trip_preserves_valid_payloads() -> None:
    """Validated wire payloads must retain their exact physical meaning."""
    command = SimulationCommand.from_dict(_command_payload())
    observation = SimulationObservation.from_dict(_observation_payload())

    assert SimulationCommand.from_dict(command.to_dict()) == command
    assert SimulationObservation.from_dict(observation.to_dict()) == observation


@pytest.mark.parametrize(
    ("factory", "mutation"),
    (
        (_command_payload, ("remove", "travel_time_s")),
        (_command_payload, ("add", "unknown_field")),
        (_observation_payload, ("remove", "metadata")),
        (_observation_payload, ("add", "unknown_field")),
    ),
)
def test_versioned_payloads_reject_missing_or_unknown_fields(
    factory: Callable[[], dict[str, object]],
    mutation: tuple[str, str],
) -> None:
    """A schema mismatch must not be interpreted as the current protocol."""
    payload = factory()
    operation, field_name = mutation
    if operation == "remove":
        del payload[field_name]
    else:
        payload[field_name] = 1

    parser = (
        SimulationCommand.from_dict
        if factory is _command_payload
        else SimulationObservation.from_dict
    )
    with pytest.raises(ValueError, match="wire schema"):
        parser(payload)


def test_observation_rejects_mapping_coercion_from_pair_sequence() -> None:
    """A list of pairs must not masquerade as a metadata JSON object."""
    payload = _observation_payload()
    payload["metadata"] = [["mode", "native"]]

    with pytest.raises(TypeError, match="metadata wire value"):
        SimulationObservation.from_dict(payload)


def test_sampled_event_spectrum_round_trip_preserves_integer_counts() -> None:
    """Native sampled detector events must remain JSON integers."""
    payload = _observation_payload()
    payload["spectrum_counts"] = [1, 2]
    payload["metadata"] = {
        "detector_response_sampling_mode": (
            "multinomial_marking_with_nonparalyzable_event_time"
        ),
        "transport_history_mode": "full_unit_weight",
    }

    observation = SimulationObservation.from_dict(payload)
    encoded = observation.to_dict()

    assert observation.spectrum_counts == [1, 2]
    assert encoded["spectrum_counts"] == [1, 2]
    assert all(
        isinstance(value, int)
        for value in encoded["spectrum_counts"]
    )
    assert SimulationObservation.from_dict(encoded) == observation


def test_sampled_event_spectrum_rejects_float_wire_counts() -> None:
    """A float wire payload must not masquerade as sampled event counts."""
    payload = _observation_payload()
    payload["metadata"] = {
        "detector_response_sampling_mode": (
            "multinomial_marking_with_nonparalyzable_event_time"
        ),
        "transport_history_mode": "full_unit_weight",
    }

    with pytest.raises(TypeError, match="integer event counts"):
        SimulationObservation.from_dict(payload)


def test_message_envelope_round_trip_preserves_json_payload() -> None:
    """A valid envelope must round-trip without type coercion."""
    payload = {"counts": [1, 2], "metadata": {"valid": True}}

    encoded = encode_message("step", payload)

    assert decode_message(encoded.strip()) == ("step", payload)


@pytest.mark.parametrize(
    ("message_type", "payload"),
    (
        (1, {}),
        ("", {}),
        ("step", []),
        ("step", {1: "coerced-key"}),
        ("step", {"value": float("nan")}),
        ("step", {"value": 2**53 + 1}),
        ("step", {"value": (1, 2)}),
    ),
)
def test_encode_message_rejects_lossy_wire_values(
    message_type: object,
    payload: object,
) -> None:
    """Encoding must reject values JSON would silently reinterpret."""
    with pytest.raises((TypeError, ValueError)):
        encode_message(message_type, payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "encoded",
    (
        b"[]",
        b'{"type":"step"}',
        b'{"type":"step","payload":{},"extra":1}',
        b'{"type":1,"payload":{}}',
        b'{"type":"step","payload":[]}',
        b'{"type":"step","type":"reset","payload":{}}',
        b'{"type":"step","payload":{"value":NaN}}',
        b'{"type":"step","payload":{"value":1e999}}',
    ),
)
def test_decode_message_rejects_ambiguous_or_nonfinite_envelopes(
    encoded: bytes,
) -> None:
    """Decoding must fail rather than reinterpret a corrupt envelope."""
    with pytest.raises((TypeError, ValueError)):
        decode_message(encoded)
