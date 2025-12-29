"""Provide the spectrum simulation and unfolding pipeline used in Chapter 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource, inverse_square_scale
from measurement.shielding import OctantShield, octant_index_from_normal
from measurement.kernels import ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from spectrum.library import Nuclide, default_library
from spectrum.response_matrix import (
    build_response_matrix,
    default_background_shape,
    default_resolution,
)
from spectrum.smoothing import gaussian_smooth
from spectrum.baseline import asymmetric_least_squares
from spectrum.dead_time import non_paralyzable_correction
from spectrum.activity_estimation import estimate_activities
from spectrum.decomposition import Peak, strip_overlaps
from spectrum.peak_detection import detect_peaks

# Background intensity (counts/s).
BACKGROUND_RATE_CPS = 3.0
# Backward-compatible alias.
BACKGROUND_COUNTS_PER_SECOND = BACKGROUND_RATE_CPS
# Default ALS baseline parameters.
BASELINE_LAM = 1e5
BASELINE_P = 0.01
BASELINE_NITER = 10

@dataclass
class SpectrumConfig:
    """Configuration for spectrum simulation and unfolding."""

    energy_min_keV: float = 0.0
    energy_max_keV: float = 1500.0
    bin_width_keV: float = 2.0
    resolution_a: float = 0.8
    resolution_b: float = 1.5

    def energy_axis(self) -> NDArray[np.float64]:
        """Return the energy axis in keV."""
        return np.arange(self.energy_min_keV, self.energy_max_keV + self.bin_width_keV, self.bin_width_keV)


class SpectralDecomposer:
    """Peak-based spectrum decomposer following the Chapter 2 pipeline."""

    def __init__(
        self,
        spectrum_config: SpectrumConfig | None = None,
        library: Dict[str, Nuclide] | None = None,
    ) -> None:
        """Initialize the response matrix and pipeline configuration."""
        self.config = spectrum_config or SpectrumConfig()
        self.library = library or default_library()
        self.energy_axis = self.config.energy_axis()
        self.resolution_fn = default_resolution()
        # Energy-dependent efficiency model (CeBr3 assumption).
        from spectrum.response_matrix import cebr3_efficiency

        self.efficiency_fn = cebr3_efficiency
        self._background_shape = default_background_shape(self.energy_axis)
        self.response_matrix = build_response_matrix(
            self.energy_axis,
            self.library,
            resolution_fn=self.resolution_fn,
            efficiency_fn=self.efficiency_fn,
            bin_width_keV=self.config.bin_width_keV,
        )
        self.isotope_names = list(self.library.keys())

    def simulate_spectrum(
        self,
        sources: Iterable[PointSource],
        environment: EnvironmentConfig | None = None,
        acquisition_time: float = 1.0,
        rng: np.random.Generator | None = None,
        dead_time_s: float = 0.0,
        shield_orientation: NDArray[np.float64] | None = None,
        octant_shield: OctantShield | None = None,
        fe_shield_orientation: NDArray[np.float64] | None = None,
        pb_shield_orientation: NDArray[np.float64] | None = None,
        mu_by_isotope: Dict[str, object] | None = None,
        shield_params: ShieldParams | None = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, float]]:
        """
        Simulate a spectrum from point sources and the environment.

        Returns the spectrum array and the effective source strengths after
        geometric attenuation.

        Shielding (Sec. 3.4–3.5): if shield orientations are provided, the line-of-sight
        is tested and an exponential attenuation factor exp(-mu * L) is applied to each
        source contribution to reflect attenuated photopeaks.
        """
        env = environment or EnvironmentConfig()
        detector = env.detector()
        kernel = ContinuousKernel(mu_by_isotope=mu_by_isotope, shield_params=shield_params or ShieldParams())
        expected = np.zeros_like(self.energy_axis, dtype=float)
        effective_strengths: Dict[str, float] = {name: 0.0 for name in self.isotope_names}
        for source in sources:
            if source.isotope not in self.library:
                continue
            geom = inverse_square_scale(detector, source)
            effective_strength = source.intensity_cps_1m * geom
            atten = 1.0
            if fe_shield_orientation is not None or pb_shield_orientation is not None:
                fe_idx = octant_index_from_normal(np.asarray(fe_shield_orientation)) if fe_shield_orientation is not None else None
                pb_idx = octant_index_from_normal(np.asarray(pb_shield_orientation)) if pb_shield_orientation is not None else None
                if fe_idx is not None and pb_idx is not None:
                    atten = kernel.attenuation_factor_pair(
                        isotope=source.isotope,
                        source_pos=source.position_array(),
                        detector_pos=detector,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                    )
                else:
                    orient_idx = fe_idx if fe_idx is not None else pb_idx
                    atten = kernel.attenuation_factor(
                        isotope=source.isotope,
                        source_pos=source.position_array(),
                        detector_pos=detector,
                        orient_idx=int(orient_idx),
                    )
            elif octant_shield is not None and shield_orientation is not None:
                oct_idx = octant_index_from_normal(np.asarray(shield_orientation))
                if octant_shield.blocks_ray(
                    detector_position=detector,
                    source_position=source.position_array(),
                    octant_index=oct_idx,
                ):
                    atten = kernel.attenuation_factor(
                        isotope=source.isotope,
                        source_pos=source.position_array(),
                        detector_pos=detector,
                        orient_idx=oct_idx,
                    )
            col_idx = self.isotope_names.index(source.isotope)
            contribution = acquisition_time * effective_strength
            expected += atten * contribution * self.response_matrix[:, col_idx]
            effective_strengths[source.isotope] += atten * contribution

        # Add background, resolving the alias consistently.
        background_rate = BACKGROUND_RATE_CPS
        if BACKGROUND_COUNTS_PER_SECOND != BACKGROUND_RATE_CPS:
            background_rate = BACKGROUND_COUNTS_PER_SECOND
        if background_rate > 0.0:
            total_bg_counts = background_rate * acquisition_time
            expected += self._background_shape * total_bg_counts

        noisy = rng.poisson(expected) if rng is not None else expected
        corrected = non_paralyzable_correction(noisy, dead_time_s=dead_time_s)
        return corrected, effective_strengths

    def preprocess(self, spectrum: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply smoothing and baseline correction to stabilise peak detection."""
        smoothed = gaussian_smooth(spectrum, sigma_bins=2.0)
        baseline = asymmetric_least_squares(
            smoothed,
            lam=BASELINE_LAM,
            p=BASELINE_P,
            niter=BASELINE_NITER,
        )
        corrected = np.clip(smoothed - baseline, a_min=0.0, a_max=None)
        return corrected

    def decompose(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """Decompose a spectrum by NNLS and return isotope-wise activities."""
        return self.decompose_subset(spectrum, isotopes=None)

    def decompose_subset(
        self,
        spectrum: NDArray[np.float64],
        isotopes: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        """
        Decompose a spectrum by NNLS for a subset of isotopes.

        Args:
            spectrum: Observed spectrum.
            isotopes: Optional subset of isotopes to fit; None fits all.
        """
        if isotopes is None:
            return estimate_activities(self.response_matrix, spectrum, self.isotope_names)
        indices = [self.isotope_names.index(iso) for iso in isotopes if iso in self.isotope_names]
        if not indices:
            return {iso: 0.0 for iso in isotopes}
        design = self.response_matrix[:, indices]
        iso_names = [self.isotope_names[i] for i in indices]
        return estimate_activities(design, spectrum, iso_names)

    def isotope_counts(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """Return isotope-wise counts suitable for PF updates."""
        counts, _ = self.isotope_counts_with_detection(spectrum)
        return counts

    def isotope_counts_with_detection(
        self,
        spectrum: NDArray[np.float64],
        *,
        detect_isotopes: bool = True,
        detect_threshold_abs: float = 0.1,
        detect_threshold_rel: float = 0.2,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
        peak_tolerance_keV: float = 10.0,
        min_peaks_multi: int = 2,
    ) -> Tuple[Dict[str, float], set[str]]:
        """
        Return isotope-wise counts plus detected isotopes.

        Detection uses peak matching and minimum peak counts before fitting NNLS
        on the subset of candidate isotopes.
        """
        if not detect_isotopes:
            return self.decompose(spectrum), set(self.isotope_names)
        detected, _ = self.detect_isotopes(
            spectrum,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance,
            peak_tolerance_keV=peak_tolerance_keV,
            min_peaks_multi=min_peaks_multi,
        )
        counts_partial = self.decompose_subset(spectrum, isotopes=sorted(detected)) if detected else {}
        counts_full = {iso: float(counts_partial.get(iso, 0.0)) for iso in self.isotope_names}
        if detected:
            peak_counts = self.peak_window_counts(
                spectrum,
                isotopes=sorted(detected),
                window_keV=None,
                window_sigma=3.0,
                apply_stripping=True,
                peak_tolerance_keV=peak_tolerance_keV,
                peak_prominence=peak_prominence,
                peak_distance=peak_distance,
            )
            for iso in detected:
                counts_full[iso] = max(counts_full.get(iso, 0.0), peak_counts.get(iso, 0.0))
        return counts_full, detected

    def detect_isotopes(
        self,
        spectrum: NDArray[np.float64],
        *,
        detect_threshold_abs: float = 0.1,
        detect_threshold_rel: float = 0.2,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
        peak_tolerance_keV: float = 10.0,
        min_peaks_multi: int = 2,
    ) -> Tuple[set[str], Dict[str, list[int]]]:
        """
        Detect isotopes using peak assignments and area thresholds.

        Returns:
            (detected_isotopes, peaks_by_iso)
        """
        corrected = self.preprocess(spectrum)
        work = corrected
        if float(np.max(work)) <= 0.0:
            work = gaussian_smooth(np.asarray(spectrum, dtype=float), sigma_bins=2.0)
        peak_indices = detect_peaks(work, prominence=peak_prominence, distance=peak_distance)
        peaks_by_iso, _ = self._assign_peak_indices(
            self.energy_axis,
            peak_indices,
            self.library,
            tolerance_keV=peak_tolerance_keV,
        )
        min_peaks = self._min_peak_count_by_isotope(self.library, min_peaks_multi=min_peaks_multi)
        line_counts = self.peak_line_counts(
            spectrum,
            isotopes=list(self.isotope_names),
            window_keV=None,
            window_sigma=3.0,
            smooth_sigma_bins=2.0,
            subtract_baseline=True,
            apply_stripping=False,
            peak_tolerance_keV=peak_tolerance_keV,
        )
        detected: set[str] = set()
        max_line_overall = 0.0
        best_iso: str | None = None
        for iso, counts in line_counts.items():
            if not counts:
                continue
            max_line = float(max(counts))
            if max_line > max_line_overall:
                max_line_overall = max_line
                best_iso = iso
            if max_line <= 0.0:
                continue
            threshold = max(detect_threshold_abs, detect_threshold_rel * max_line)
            num_above = sum(val >= threshold for val in counts)
            min_required = min_peaks.get(iso, 1)
            if num_above >= min_required:
                detected.add(iso)
                continue
            if min_required > 1 and num_above >= 1:
                other_peak_ok = any(val >= detect_threshold_abs for val in counts if val < threshold)
                if other_peak_ok:
                    detected.add(iso)
        if not detected and best_iso is not None and max_line_overall > 0.0:
            detected = {best_iso}
        return detected, peaks_by_iso

    def peak_line_counts(
        self,
        spectrum: NDArray[np.float64],
        *,
        isotopes: Sequence[str] | None = None,
        window_keV: float | None = 5.0,
        window_sigma: float = 3.0,
        smooth_sigma_bins: float = 2.0,
        subtract_baseline: bool = True,
        apply_stripping: bool = True,
        peak_tolerance_keV: float = 10.0,
    ) -> Dict[str, list[float]]:
        """
        Compute per-line peak areas for each isotope using energy windows.

        This follows Eq. (20) style window integration and can optionally apply
        spectral stripping to separate overlapping peaks. If window_keV is None,
        use ±window_sigma * sigma(E) based on the detector resolution model.
        """
        corrected = np.asarray(spectrum, dtype=float)
        if smooth_sigma_bins > 0.0:
            corrected = gaussian_smooth(corrected, sigma_bins=smooth_sigma_bins)
        if subtract_baseline:
            base = asymmetric_least_squares(
                corrected,
                lam=BASELINE_LAM,
                p=BASELINE_P,
                niter=BASELINE_NITER,
            )
            corrected = np.clip(corrected - base, a_min=0.0, a_max=None)
        energy_axis = self.energy_axis
        iso_names = list(isotopes) if isotopes is not None else list(self.isotope_names)
        library_subset = {iso: self.library[iso] for iso in iso_names if iso in self.library}
        peaks: list[Peak] = []
        for iso, nuclide in library_subset.items():
            for line in nuclide.lines:
                half_width = window_keV
                if half_width is None:
                    sigma = float(self.resolution_fn(line.energy_keV))
                    half_width = max(window_sigma * sigma, 1e-6)
                mask = np.abs(energy_axis - line.energy_keV) <= float(half_width)
                if not np.any(mask):
                    continue
                peaks.append(Peak(energy_keV=float(line.energy_keV), area=float(np.sum(corrected[mask]))))
        if apply_stripping and peaks:
            _, stripped_peaks = strip_overlaps(
                peaks,
                library_subset,
                tolerance_keV=peak_tolerance_keV,
                efficiency_fn=self.efficiency_fn,
            )
        else:
            stripped_peaks = peaks

        def _closest_peak(energy: float) -> Peak | None:
            best: Peak | None = None
            min_diff = peak_tolerance_keV
            for pk in stripped_peaks:
                diff = abs(pk.energy_keV - energy)
                if diff <= min_diff:
                    min_diff = diff
                    best = pk
            return best

        line_counts: Dict[str, list[float]] = {}
        for iso, nuclide in library_subset.items():
            counts: list[float] = []
            for line in nuclide.lines:
                match = _closest_peak(float(line.energy_keV))
                counts.append(float(match.area) if match is not None else 0.0)
            line_counts[iso] = counts
        return line_counts

    def peak_window_counts(
        self,
        spectrum: NDArray[np.float64],
        *,
        isotopes: Sequence[str] | None = None,
        window_keV: float | None = 5.0,
        window_sigma: float = 3.0,
        smooth_sigma_bins: float = 2.0,
        subtract_baseline: bool = True,
        apply_stripping: bool = True,
        peak_tolerance_keV: float = 10.0,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
    ) -> Dict[str, float]:
        """
        Compute isotope-wise counts by integrating windows around line energies.

        Uses preprocessing to stabilize low-count spectra and weights line windows
        by their relative intensities within each nuclide. When apply_stripping is
        enabled, peak areas are first corrected using spectral stripping.
        """
        iso_names = list(isotopes) if isotopes is not None else list(self.isotope_names)
        library_subset = {iso: self.library[iso] for iso in iso_names if iso in self.library}
        total_intensity = {
            name: sum(line.intensity for line in nuclide.lines) for name, nuclide in library_subset.items()
        }
        line_counts = self.peak_line_counts(
            spectrum,
            isotopes=iso_names,
            window_keV=window_keV,
            window_sigma=window_sigma,
            smooth_sigma_bins=smooth_sigma_bins,
            subtract_baseline=subtract_baseline,
            apply_stripping=apply_stripping,
            peak_tolerance_keV=peak_tolerance_keV,
        )
        counts: Dict[str, float] = {}
        for iso in iso_names:
            nuclide = library_subset.get(iso)
            if nuclide is None:
                continue
            total_int = total_intensity.get(iso, 0.0)
            if total_int <= 0.0:
                counts[iso] = 0.0
                continue
            z_val = 0.0
            per_line = line_counts.get(iso, [])
            for line, line_val in zip(nuclide.lines, per_line):
                weight = line.intensity / total_int
                z_val += weight * float(line_val)
            counts[iso] = float(z_val)
        return counts

    @staticmethod
    def debug_baseline(
        energy_axis: NDArray[np.float64],
        raw: NDArray[np.float64],
        smoothed: NDArray[np.float64],
        baseline: NDArray[np.float64],
        corrected: NDArray[np.float64],
        title: str = "Baseline Debug",
    ) -> None:
        """Plot baseline estimation diagnostics."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(energy_axis, raw, label="Raw")
        plt.plot(energy_axis, smoothed, label="Smoothed")
        plt.plot(energy_axis, baseline, label="Baseline")
        plt.plot(energy_axis, corrected, label="Corrected")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.title(title)
        plt.legend()
        plt.show()

    def identify_by_peaks(
        self,
        spectrum: NDArray[np.float64],
        tolerance_keV: float = 5.0,
    ) -> Dict[str, float]:
        """
        Estimate isotope-wise reference peak areas via detection and stripping.

        Use this in low-count scenarios where peak-based identification is preferred.
        """
        corrected = self.preprocess(spectrum)
        peak_indices = detect_peaks(corrected, prominence=0.05, distance=5)
        peaks: list[Peak] = []
        for idx in peak_indices:
            energy = self.energy_axis[idx]
            area = corrected[idx]
            peaks.append(Peak(energy_keV=float(energy), area=float(area)))
        ref_areas, _ = strip_overlaps(peaks, self.library, tolerance_keV=tolerance_keV)
        return ref_areas

    @staticmethod
    def _assign_peak_indices(
        energy_axis: NDArray[np.float64],
        peak_indices: NDArray[np.int64],
        library: Dict[str, Nuclide],
        tolerance_keV: float,
    ) -> Tuple[Dict[str, list[int]], list[int]]:
        """Assign detected peak indices to isotopes based on closest library lines."""
        peaks_by_iso: Dict[str, list[int]] = {iso: [] for iso in library}
        unassigned: list[int] = []
        line_energies = {
            iso: np.array([line.energy_keV for line in nuclide.lines], dtype=float)
            for iso, nuclide in library.items()
        }
        for idx in peak_indices:
            energy = float(energy_axis[int(idx)])
            best_iso = None
            best_diff = float("inf")
            for iso, energies in line_energies.items():
                if energies.size == 0:
                    continue
                diff = float(np.min(np.abs(energies - energy)))
                if diff < best_diff:
                    best_diff = diff
                    best_iso = iso
            if best_iso is not None and best_diff <= tolerance_keV:
                peaks_by_iso[best_iso].append(int(idx))
            else:
                unassigned.append(int(idx))
        return peaks_by_iso, unassigned

    @staticmethod
    def _min_peak_count_by_isotope(
        library: Dict[str, Nuclide],
        min_peaks_multi: int = 2,
    ) -> Dict[str, int]:
        """Return minimum peak counts required to accept each isotope."""
        min_counts: Dict[str, int] = {}
        for iso, nuclide in library.items():
            line_count = len(nuclide.lines)
            min_counts[iso] = 1 if line_count <= 1 else max(int(min_peaks_multi), 1)
        return min_counts
