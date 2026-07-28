"""Physical continuous-kernel response utilities outside the PF likelihood."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel


def expected_counts_per_source(
    kernel: ContinuousKernel,
    isotope: str,
    detector_positions: NDArray[np.float64],
    sources: NDArray[np.float64],
    strengths: NDArray[np.float64],
    live_times: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return unmodified physical counts for each measurement and source."""
    detector_arr = np.asarray(detector_positions, dtype=np.float64)
    source_arr = np.asarray(sources, dtype=np.float64)
    strength_arr = np.asarray(strengths, dtype=np.float64).reshape(-1)
    live_arr = np.asarray(live_times, dtype=np.float64).reshape(-1)
    fe_arr = np.asarray(fe_indices, dtype=np.int64).reshape(-1)
    pb_arr = np.asarray(pb_indices, dtype=np.int64).reshape(-1)
    measurement_count = int(live_arr.size)
    if (
        detector_arr.shape != (measurement_count, 3)
        or fe_arr.size != measurement_count
        or pb_arr.size != measurement_count
        or np.any(~np.isfinite(detector_arr))
        or np.any(~np.isfinite(live_arr))
        or np.any(live_arr <= 0.0)
    ):
        raise ValueError("Measurement geometry and live times are invalid.")
    if source_arr.size == 0:
        return np.zeros((measurement_count, 0), dtype=np.float64)
    source_arr = source_arr.reshape(-1, 3)
    if (
        source_arr.shape[0] != strength_arr.shape[0]
        or np.any(~np.isfinite(source_arr))
        or np.any(~np.isfinite(strength_arr))
        or np.any(strength_arr <= 0.0)
    ):
        raise ValueError(
            "Sources and positive strengths must be finite and aligned."
        )
    source_count = int(source_arr.shape[0])
    kernel_values = np.asarray(
        kernel.kernel_values_selected_pairs_for_detectors(
            isotope=isotope,
            detector_positions=detector_arr,
            sources=source_arr,
            fe_indices=fe_arr,
            pb_indices=pb_arr,
        ),
        dtype=np.float64,
    )
    if (
        kernel_values.shape != (measurement_count, source_count)
        or np.any(~np.isfinite(kernel_values))
        or np.any(kernel_values < 0.0)
    ):
        raise ValueError(
            "Selected-pair physical kernel returned invalid response values."
        )
    return (
        live_arr[:, None]
        * kernel_values
        * strength_arr[None, :]
    )
