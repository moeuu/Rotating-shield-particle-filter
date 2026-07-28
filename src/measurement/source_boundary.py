"""Bind exact surface anchors to deterministic air-side transport positions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from measurement.surface_charts import SurfaceChartGeometry


SURFACE_EMISSION_POLICY_SCHEMA_VERSION = 1
SURFACE_EMISSION_POLICY_ID = "exact_anchor_air_normal_epsilon_v1"
SURFACE_EMISSION_EPSILON_M = 1.0e-6
SURFACE_SOURCE_RUNTIME_KEYS = frozenset(
    {
        "isotope",
        "position",
        "transport_position",
        "intensity_cps_1m",
        "surface_chart_id",
        "surface_uv",
        "surface_normal",
        "surface_emission_policy_sha256",
    }
)


def surface_emission_policy_payload() -> dict[str, object]:
    """Return the immutable surface-anchor/native-transport contract."""
    return {
        "schema_version": SURFACE_EMISSION_POLICY_SCHEMA_VERSION,
        "policy_id": SURFACE_EMISSION_POLICY_ID,
        "anchor_semantics": "exact_surface_chart_uv_evaluation_truth",
        "transport_semantics": "anchor_plus_air_facing_normal_times_epsilon",
        "epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "normal_convention": {
            "room_floor": "positive_z_into_room_air",
            "room_ceiling": "negative_z_into_room_air",
            "room_walls": "inward_into_room_air",
            "transport_components": "outward_from_transport_solid",
        },
        "native_validation_requirement": (
            "exact_and_signed_epsilon_pre_dead_time_entry_spectrum_gate"
        ),
    }


def surface_emission_policy_sha256() -> str:
    """Return the canonical policy digest stored with every source/run."""
    payload = json.dumps(
        surface_emission_policy_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_real_vector(
    value: object,
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    """Return a strict fixed-length finite real JSON array."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an array of real numbers.")
    if len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values.")
    parsed: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(f"{field_name} must contain only real numbers.")
        numeric = float(component)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must contain only finite values.")
        parsed.append(numeric)
    return tuple(parsed)


def canonical_surface_source_runtime_payload(
    entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return strict normalized source truth/transport handshake entries."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("Surface source runtime payload must be a sequence.")
    from measurement.model import PointSource

    normalized: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"Surface source entry {index} must be a mapping.")
        if set(entry) != SURFACE_SOURCE_RUNTIME_KEYS:
            raise ValueError(
                f"Surface source entry {index} must contain exactly "
                f"{sorted(SURFACE_SOURCE_RUNTIME_KEYS)}."
            )
        isotope = entry["isotope"]
        intensity = entry["intensity_cps_1m"]
        chart_id = entry["surface_chart_id"]
        if not isinstance(isotope, str) or not isotope:
            raise ValueError("Surface source isotope must be a nonempty string.")
        if isinstance(intensity, bool) or not isinstance(intensity, Real):
            raise TypeError("Surface source intensity must be a real number.")
        intensity_value = float(intensity)
        if not math.isfinite(intensity_value) or intensity_value <= 0.0:
            raise ValueError(
                "Surface source intensity must be finite and positive."
            )
        if isinstance(chart_id, bool) or not isinstance(chart_id, Integral):
            raise TypeError("Surface source chart ID must be an integer.")
        policy_hash = entry["surface_emission_policy_sha256"]
        if not isinstance(policy_hash, str):
            raise TypeError(
                "Surface source emission policy digest must be a string."
            )
        point_source = PointSource(
            isotope=isotope,
            position=_finite_real_vector(
                entry["position"],
                length=3,
                field_name="Surface source position",
            ),
            intensity_cps_1m=intensity_value,
            surface_chart_id=int(chart_id),
            surface_uv=_finite_real_vector(
                entry["surface_uv"],
                length=2,
                field_name="Surface source UV",
            ),
            surface_normal=_finite_real_vector(
                entry["surface_normal"],
                length=3,
                field_name="Surface source normal",
            ),
            transport_position=_finite_real_vector(
                entry["transport_position"],
                length=3,
                field_name="Surface source transport position",
            ),
            surface_emission_policy_sha256=policy_hash,
        )
        anchor = point_source.position_array()
        transport = point_source.transport_position_array()
        normal = np.asarray(point_source.surface_normal, dtype=np.float64)
        expected_transport = surface_transport_positions(
            anchor.reshape(1, 3),
            normal.reshape(1, 3),
        )[0]
        if (
            point_source.surface_emission_policy_sha256
            != surface_emission_policy_sha256()
            or not np.array_equal(transport, expected_transport)
        ):
            raise ValueError(
                "Surface source surface-emission position violates the signed "
                "air-side epsilon contract."
            )
        normalized.append(
            {
                "isotope": point_source.isotope,
                "position": [float(value) for value in anchor],
                "transport_position": [
                    float(value) for value in transport
                ],
                "intensity_cps_1m": float(point_source.intensity_cps_1m),
                "surface_chart_id": int(point_source.surface_chart_id),
                "surface_uv": [
                    float(value) for value in point_source.surface_uv
                ],
                "surface_normal": [float(value) for value in normal],
                "surface_emission_policy_sha256": str(
                    point_source.surface_emission_policy_sha256
                ),
            }
        )
    return normalized


def surface_source_runtime_contract_sha256(
    entries: Sequence[Mapping[str, object]],
) -> str:
    """Hash source strengths, anchors, chart identity, and emission positions."""
    payload = canonical_surface_source_runtime_payload(entries)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_air_facing_normal(
    *,
    kind: str,
    face_id: str,
) -> tuple[float, float, float]:
    """Return the semantic air-facing normal for one atlas face identifier."""
    face = str(face_id)
    surface_kind = str(kind)
    room_normals = {
        "room_floor": (0.0, 0.0, 1.0),
        "room_ceiling": (0.0, 0.0, -1.0),
        "room_wall_x0": (1.0, 0.0, 0.0),
        "room_wall_x1": (-1.0, 0.0, 0.0),
        "room_wall_y0": (0.0, 1.0, 0.0),
        "room_wall_y1": (0.0, -1.0, 0.0),
    }
    if face in room_normals:
        expected_kind = {
            "room_floor": "floor",
            "room_ceiling": "ceiling",
        }.get(face, "wall")
        if surface_kind != expected_kind:
            raise ValueError(
                f"Surface kind {surface_kind!r} conflicts with face {face!r}."
            )
        return room_normals[face]
    suffix_normals = {
        "_x0": (-1.0, 0.0, 0.0),
        "_x1": (1.0, 0.0, 0.0),
        "_y0": (0.0, -1.0, 0.0),
        "_y1": (0.0, 1.0, 0.0),
        "_z0": (0.0, 0.0, -1.0),
        "_z1": (0.0, 0.0, 1.0),
    }
    if not face.startswith("transport_component_"):
        raise ValueError(f"Unknown surface face identifier {face!r}.")
    for suffix, normal in suffix_normals.items():
        if face.endswith(suffix):
            expected_kind = (
                "obstacle_bottom"
                if suffix == "_z0"
                else "obstacle_top"
                if suffix == "_z1"
                else "obstacle_side"
            )
            if surface_kind != expected_kind:
                raise ValueError(
                    f"Surface kind {surface_kind!r} conflicts with face {face!r}."
                )
            return normal
    raise ValueError(f"Unknown transport-component face {face!r}.")


def validate_air_facing_surface_normals(
    geometry: SurfaceChartGeometry,
) -> None:
    """Fail if stored atlas normals disagree with semantic solid/air sides."""
    expected = np.asarray(
        [
            expected_air_facing_normal(kind=kind, face_id=face_id)
            for kind, face_id in zip(geometry.kinds, geometry.face_ids)
        ],
        dtype=np.float64,
    )
    actual = np.asarray(geometry.normals_xyz, dtype=np.float64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            "Surface atlas normals do not follow the air-facing boundary "
            "contract."
        )


def surface_transport_positions(
    anchors_xyz: NDArray[np.float64] | Sequence[Sequence[float]],
    normals_xyz: NDArray[np.float64] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    """Shift exact anchors by the fixed physically negligible air epsilon."""
    anchors = np.asarray(anchors_xyz, dtype=np.float64)
    normals = np.asarray(normals_xyz, dtype=np.float64)
    if anchors.shape != normals.shape or anchors.shape[-1:] != (3,):
        raise ValueError("Surface anchors and normals must share shape (..., 3).")
    norm = np.linalg.norm(normals, axis=-1)
    if (
        np.any(~np.isfinite(anchors))
        or np.any(~np.isfinite(normals))
        or not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Surface anchors and air-facing normals are invalid.")
    return anchors + SURFACE_EMISSION_EPSILON_M * normals


__all__ = [
    "SURFACE_EMISSION_EPSILON_M",
    "SURFACE_EMISSION_POLICY_ID",
    "SURFACE_EMISSION_POLICY_SCHEMA_VERSION",
    "SURFACE_SOURCE_RUNTIME_KEYS",
    "canonical_surface_source_runtime_payload",
    "expected_air_facing_normal",
    "surface_emission_policy_payload",
    "surface_emission_policy_sha256",
    "surface_transport_positions",
    "surface_source_runtime_contract_sha256",
    "validate_air_facing_surface_normals",
]
