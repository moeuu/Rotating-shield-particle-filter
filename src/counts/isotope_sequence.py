"""Generate isotope-wise count sequences from unfolded spectra and apply weighted aggregation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from spectrum.baseline import asymmetric_least_squares
from spectrum.dead_time import non_paralyzable_correction
from spectrum.library import Nuclide
from spectrum.smoothing import gaussian_smooth


def _dead_time_scale(total_counts: float, live_time_s: float, dead_time_s: float) -> float:
    """Return a global scale factor using a non-paralyzable dead-time model."""
    if dead_time_s <= 0.0 or live_time_s <= 0.0:
        return 1.0
    m_rate = total_counts / live_time_s
    return 1.0 / max(1.0 - m_rate * dead_time_s, 1e-9)


def _apply_preprocess(
    spectrum: NDArray[np.float64],
    live_time_s: float,
    dead_time_s: float,
    smooth_sigma_bins: float | None,
    subtract_baseline: bool,
) -> NDArray[np.float64]:
    """Apply dead-time correction, smoothing, and baseline subtraction."""
    corrected = spectrum.astype(float)
    scale = _dead_time_scale(corrected.sum(), live_time_s, dead_time_s)
    corrected *= scale
    if smooth_sigma_bins is not None and smooth_sigma_bins > 0.0:
        corrected = gaussian_smooth(corrected, sigma_bins=smooth_sigma_bins)
    if subtract_baseline:
        base = asymmetric_least_squares(corrected, lam=1e4, p=0.01, niter=10)
        corrected = np.clip(corrected - base, a_min=0.0, a_max=None)
    return corrected


def build_isotope_count_sequence(
    spectra: Iterable[NDArray[np.float64]],
    energy_axis_keV: NDArray[np.float64],
    library: Dict[str, Nuclide],
    live_time_s: float | Sequence[float],
    dead_time_s: float = 0.0,
    window_keV: float = 5.0,
    smooth_sigma_bins: float | None = None,
    subtract_baseline: bool = True,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Build isotope-wise count sequences z_k from short-time spectra.

    Args:
        spectra: Time-series spectra (iterable of channel-count arrays).
        energy_axis_keV: Energy axis per channel (keV).
        library: Nuclide library.
        live_time_s: Live time (single value or per-spectrum list).
        dead_time_s: Dead time in seconds.
        window_keV: Integration window (±window_keV) around each line.
        smooth_sigma_bins: Smoothing sigma in bins (None or 0 disables).
        subtract_baseline: Whether to subtract the baseline.

    Returns:
        (isotope_names, counts_matrix) where counts_matrix shape = (T, H)
    """
    energy_axis_keV = np.asarray(energy_axis_keV, dtype=float)
    iso_names = list(library.keys())
    live_times = (
        [float(live_time_s)] * len(list(spectra))
        if isinstance(live_time_s, (int, float))
        else [float(v) for v in live_time_s]
    )
    spectra_list = list(spectra)
    if len(spectra_list) != len(live_times):
        raise ValueError("Number of spectra and live_time_s entries must match")

    counts_matrix = np.zeros((len(spectra_list), len(iso_names)), dtype=float)
    # Pre-compute total line intensities.
    total_intensity: Dict[str, float] = {
        name: sum(line.intensity for line in nuclide.lines) for name, nuclide in library.items()
    }

    for idx, (spec, lt) in enumerate(zip(spectra_list, live_times)):
        processed = _apply_preprocess(
            np.asarray(spec, dtype=float), live_time_s=lt, dead_time_s=dead_time_s,
            smooth_sigma_bins=smooth_sigma_bins, subtract_baseline=subtract_baseline
        )
        for j, iso in enumerate(iso_names):
            nuclide = library[iso]
            total_int = total_intensity.get(iso, 0.0)
            if total_int <= 0.0:
                continue
            z_val = 0.0
            for line in nuclide.lines:
                weight = line.intensity / total_int
                mask = np.abs(energy_axis_keV - line.energy_keV) <= window_keV
                if not np.any(mask):
                    continue
                y_hp = processed[mask].sum()
                z_val += weight * y_hp
            counts_matrix[idx, j] = z_val
    return iso_names, counts_matrix
