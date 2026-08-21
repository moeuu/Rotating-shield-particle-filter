"""Array contracts shared by PF estimation, replay, and planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MeasurementRecord:
    """Store one full-spectrum shield-view measurement and provenance."""

    spectrum_counts_b: NDArray[np.float64]
    pose_idx: int
    live_time_s: float
    fe_index: int
    pb_index: int
    detector_position_xyz_m: tuple[float, float, float]
    station_sequence_id: int
    station_view_index: int
    generative_contract_hash_sha256: str


@dataclass(frozen=True)
class JointStationObservation:
    """Store one joint view-major full-spectrum station observation."""

    spectrum_vb: NDArray[np.float64]
    energy_axis_keV: NDArray[np.float64]
    generative_contract_hash_sha256: str
    pose_idx: int
    detector_position_xyz_m: tuple[float, float, float]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    station_sequence_id: int


@dataclass(frozen=True)
class JointPlanningParticles:
    """Expose one aligned joint-particle subset as padded numeric arrays."""

    isotope_order: tuple[str, ...]
    weights_n: NDArray[np.float64]
    positions_nk3_by_isotope: dict[str, NDArray[np.float64]]
    surface_chart_ids_nk_by_isotope: dict[str, NDArray[np.int64]]
    surface_uv_nk2_by_isotope: dict[str, NDArray[np.float64]]
    strengths_nk_by_isotope: dict[str, NDArray[np.float64]]
    source_mask_nk_by_isotope: dict[str, NDArray[np.bool_]]
    original_particle_indices: NDArray[np.int64]


__all__ = ["JointPlanningParticles", "JointStationObservation", "MeasurementRecord"]
