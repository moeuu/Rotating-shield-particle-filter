"""Runtime measurement record shared by the pure PF and visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    """Store isotope-wise counts and shield geometry for one PF update."""

    counts_by_isotope: dict[str, float]
    pose_idx: int
    orient_idx: int
    live_time_s: float = 1.0
    fe_index: int | None = None
    pb_index: int | None = None
    RFe: np.ndarray | None = None
    RPb: np.ndarray | None = None
    detector_position: np.ndarray | None = None
    count_variance_by_isotope: dict[str, float] | None = None
