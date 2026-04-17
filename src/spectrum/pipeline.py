"""Provide the spectrum simulation and unfolding pipeline used in Chapter 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig, PointSource, inverse_square_scale
from measurement.shielding import OctantShield, octant_index_from_normal
from measurement.kernels import ShieldParams
from measurement.continuous_kernels import ContinuousKernel
from spectrum.library import (
    ANALYSIS_ISOTOPES,
    Nuclide,
    default_library,
    get_analysis_lines_with_intensity,
    get_detection_lines_keV,
)
from spectrum.response_matrix import (
    build_response_matrix,
    default_background_shape,
)
from spectrum.baseline import baseline_als
from spectrum.dead_time import non_paralyzable_correction
from spectrum.activity_estimation import estimate_activities
from spectrum.decomposition import Peak, strip_overlaps
from spectrum.peak_detection import detect_peaks, gaussian_smooth, has_peak_near, line_window_evidence, sigma_E_keV
from spectrum.nnls import nnls_solve

# Background intensity (counts/s).
BACKGROUND_RATE_CPS = 3.0
# Backward-compatible alias.
BACKGROUND_COUNTS_PER_SECOND = BACKGROUND_RATE_CPS
# Default ALS baseline parameters.
BASELINE_LAM = 1e5
BASELINE_P = 0.01
BASELINE_NITER = 10

# Module logger for optional detection debugging.
logger = logging.getLogger(__name__)

@dataclass
class SpectrumConfig:
    """Configuration for spectrum simulation and unfolding."""

    energy_min_keV: float = 0.0
    energy_max_keV: float = 1500.0
    bin_width_keV: float = 2.0
    smooth_sigma_bins: float = 2.0
    baseline_lam: float = BASELINE_LAM
    baseline_p: float = BASELINE_P
    baseline_niter: int = BASELINE_NITER
    als_lambda: float | None = None
    als_p: float | None = None
    als_niter: int | None = None
    resolution_a: float = 0.8
    resolution_b: float = 1.5
    peak_window_sigma: float = 3.0
    dead_time_tau_s: float = 5.813e-9
    analysis_peak_tolerance_keV: float = 10.0
    analysis_overlap_tolerance_keV: float = 5.0
    analysis_peak_prominence: float = 0.05
    analysis_peak_distance: int = 5
    detect_half_window_keV: float = 12.0
    detect_sideband_keV: float = 40.0
    detect_net_abs_cps: float = 1.0
    detect_snr_threshold: float = 4.0
    detect_strong_snr: float = 8.0
    detect_min_lines_by_iso: dict[str, int] = field(
        default_factory=lambda: {"Cs-137": 1, "Co-60": 2, "Eu-154": 2}
    )
    detect_hit_hysteresis: int = 5
    detect_miss_hysteresis: int = 8
    detect_use_residual: bool = True
    detect_debug: bool = False

    def __post_init__(self) -> None:
        """Synchronize legacy ALS fields with baseline parameters."""
        if self.als_lambda is None:
            self.als_lambda = float(self.baseline_lam)
        else:
            self.baseline_lam = float(self.als_lambda)
        if self.als_p is None:
            self.als_p = float(self.baseline_p)
        else:
            self.baseline_p = float(self.als_p)
        if self.als_niter is None:
            self.als_niter = int(self.baseline_niter)
        else:
            self.baseline_niter = int(self.als_niter)

    def energy_axis(self) -> NDArray[np.float64]:
        """Return the energy axis in keV."""
        return np.arange(self.energy_min_keV, self.energy_max_keV + self.bin_width_keV, self.bin_width_keV)


@dataclass
class DetectionState:
    """Track per-isotope hit/miss streaks and active detections."""

    hit_streak: dict[str, int] = field(default_factory=dict)
    miss_streak: dict[str, int] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class AnalysisLineObservation:
    """Hold analysis-line metadata and integrated peak area."""

    isotope: str
    energy_keV: float
    intensity: float
    center_keV: float
    half_width_keV: float
    area: float


class SpectralDecomposer:
    """Peak-based spectrum decomposer following the Chapter 2 pipeline."""

    def __init__(
        self,
        spectrum_config: SpectrumConfig | None = None,
        library: Dict[str, Nuclide] | None = None,
        *,
        use_gpu: bool | None = None,
        gpu_device: str = "cuda",
        gpu_dtype: str = "float32",
    ) -> None:
        """
        Initialize the response matrix and pipeline configuration.

        GPU acceleration for response/smoothing is enabled when use_gpu=True
        and torch supports the requested device.
        """
        self.config = spectrum_config or SpectrumConfig()
        self.library = library or default_library()
        self.energy_axis = self.config.energy_axis()
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.gpu_dtype = gpu_dtype
        self.resolution_fn = lambda energy_keV: sigma_E_keV(
            energy_keV,
            a=self.config.resolution_a,
            b=self.config.resolution_b,
        )
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
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        self.isotope_names = list(self.library.keys())
        self.last_peak_window_debug: Dict[str, Dict[str, object]] = {}

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

    def efficiency(self, energy_keV: float) -> float:
        """Return the detector full-energy peak efficiency ε(E)."""
        if self.efficiency_fn is None:
            return 1.0
        return float(self.efficiency_fn(float(energy_keV)))

    def preprocess(self, spectrum: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply smoothing and baseline correction to stabilise peak detection."""
        cfg = self.config
        smoothed = gaussian_smooth(
            spectrum,
            sigma_bins=cfg.smooth_sigma_bins,
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        baseline = baseline_als(
            smoothed,
            lam=cfg.baseline_lam,
            p=cfg.baseline_p,
            niter=cfg.baseline_niter,
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

    def decompose_subset_with_fit(
        self,
        spectrum: NDArray[np.float64],
        isotopes: Sequence[str] | None = None,
    ) -> Tuple[Dict[str, float], NDArray[np.float64]]:
        """
        Decompose a spectrum by NNLS and return activities plus fitted spectrum.

        Args:
            spectrum: Observed spectrum.
            isotopes: Optional subset of isotopes to fit; None fits all.

        Returns:
            (activities, fitted_spectrum)
        """
        if isotopes is None:
            indices = list(range(len(self.isotope_names)))
            iso_names = list(self.isotope_names)
        else:
            indices = [self.isotope_names.index(iso) for iso in isotopes if iso in self.isotope_names]
            iso_names = [self.isotope_names[i] for i in indices]
        if not indices:
            return ({iso: 0.0 for iso in (isotopes or [])}, np.zeros_like(spectrum, dtype=float))
        design = self.response_matrix[:, indices]
        from scipy.optimize import nnls

        coeffs, _ = nnls(design, spectrum)
        fitted = design @ coeffs
        return {name: float(val) for name, val in zip(iso_names, coeffs)}, fitted

    def compute_response_model_counts(
        self,
        spectrum: NDArray[np.float64],
        *,
        isotopes: Sequence[str],
        include_background: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate isotope counts by fitting the full detector response matrix.

        The peak-window method is useful for line-level diagnostics, but it is not
        a conservative unmixing operator when multiple isotopes contribute
        overlapping continua or neighboring photopeaks. This method fits the raw
        spectrum to the same response columns used by ``simulate_spectrum`` and
        therefore returns source-equivalent counts on the transport-model scale.
        """
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        observed = np.asarray(spectrum, dtype=float)
        requested = [str(isotope) for isotope in isotopes]
        counts: Dict[str, float] = {isotope: 0.0 for isotope in requested}
        if observed.size == 0 or energy_axis.size == 0:
            return counts
        if observed.size != energy_axis.size:
            min_len = min(observed.size, energy_axis.size)
            logger.warning(
                "Spectrum length (%d) != energy axis length (%d); truncating to %d",
                observed.size,
                energy_axis.size,
                min_len,
            )
            observed = observed[:min_len]
            response_matrix = self.response_matrix[:min_len, :]
            background_shape = self._background_shape[:min_len]
        else:
            response_matrix = self.response_matrix
            background_shape = self._background_shape

        indices = [
            self.isotope_names.index(isotope)
            for isotope in requested
            if isotope in self.isotope_names
        ]
        if not indices:
            return counts
        design_columns = [response_matrix[:, index] for index in indices]
        fit_names = [self.isotope_names[index] for index in indices]
        if include_background:
            design_columns.append(np.asarray(background_shape, dtype=float))
        design = np.column_stack(design_columns)
        coeffs = nnls_solve(design, observed)
        for name, value in zip(fit_names, coeffs[: len(fit_names)]):
            counts[name] = max(float(value), 0.0)
        return counts

    def isotope_counts(self, spectrum: NDArray[np.float64]) -> Dict[str, float]:
        """Return isotope-wise counts suitable for PF updates."""
        return self.compute_isotope_counts_thesis(
            spectrum,
            live_time_s=1.0,
            isotopes=self._analysis_isotopes(),
        )

    def _analysis_isotopes(self) -> list[str]:
        """Return the fixed set of candidate isotopes for analysis."""
        return list(ANALYSIS_ISOTOPES)

    def _analysis_lines(self, max_energy_keV: float) -> list[tuple[str, float, float]]:
        """
        Return analysis lines as (isotope, energy_keV, intensity) tuples.
        """
        lines: list[tuple[str, float, float]] = []
        for iso in self._analysis_isotopes():
            for energy, intensity in get_analysis_lines_with_intensity(
                iso,
                self.library,
                max_energy_keV=max_energy_keV,
            ):
                lines.append((iso, float(energy), float(intensity)))
        return lines

    def _dead_time_scale(self, spectrum: NDArray[np.float64], live_time_s: float) -> float:
        """Return the dead-time correction scale factor."""
        tau = float(self.config.dead_time_tau_s)
        if live_time_s <= 0.0 or tau <= 0.0:
            return 1.0
        m_tot = float(np.sum(spectrum)) / float(live_time_s)
        denom = 1.0 - m_tot * tau
        if denom <= 0.0:
            logger.warning("Dead-time correction saturated (denom=%.3e); clamping scale.", denom)
            denom = 1e-9
        return 1.0 / denom

    def _window_indices(
        self,
        center_keV: float,
        half_width_keV: float,
        energy_axis: NDArray[np.float64],
    ) -> tuple[int, int]:
        """
        Convert a window in keV to inclusive bin indices on the energy axis.
        """
        if energy_axis.size == 0:
            return 0, -1
        if energy_axis.size == 1:
            return 0, 0
        bin_width = float(np.median(np.diff(energy_axis)))
        energy_min = float(energy_axis[0])
        start = int(np.floor((center_keV - half_width_keV - energy_min) / bin_width))
        end = int(np.ceil((center_keV + half_width_keV - energy_min) / bin_width))
        start = max(start, 0)
        end = min(end, int(energy_axis.size) - 1)
        return start, end

    def _closest_peak_center(
        self,
        line_energy: float,
        peak_energies: NDArray[np.float64],
        tolerance_keV: float,
    ) -> float:
        """Return the nearest detected peak energy within tolerance, or the line energy."""
        if peak_energies.size == 0:
            return float(line_energy)
        diffs = np.abs(peak_energies - float(line_energy))
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) <= float(tolerance_keV):
            return float(peak_energies[idx])
        return float(line_energy)

    def _group_overlapping_lines(
        self,
        lines: list[AnalysisLineObservation],
        overlap_tol_keV: float,
    ) -> list[list[int]]:
        """Group overlapping lines by window intersection or proximity."""
        num_lines = len(lines)
        parent = list(range(num_lines))

        def _find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def _union(i: int, j: int) -> None:
            ri = _find(i)
            rj = _find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                li = lines[i]
                lj = lines[j]
                lo_i = li.center_keV - li.half_width_keV
                hi_i = li.center_keV + li.half_width_keV
                lo_j = lj.center_keV - lj.half_width_keV
                hi_j = lj.center_keV + lj.half_width_keV
                overlap = (lo_i <= hi_j) and (lo_j <= hi_i)
                close = abs(li.center_keV - lj.center_keV) <= float(overlap_tol_keV)
                if overlap or close:
                    _union(i, j)
        groups: dict[int, list[int]] = {}
        for idx in range(num_lines):
            root = _find(idx)
            groups.setdefault(root, []).append(idx)
        return list(groups.values())

    def _strip_overlap_groups(
        self,
        lines: list[AnalysisLineObservation],
        overlap_tol_keV: float,
    ) -> list[AnalysisLineObservation]:
        """
        Apply small-system NNLS stripping for overlapping line groups.
        """
        if not lines:
            return []
        by_iso: dict[str, list[AnalysisLineObservation]] = {}
        for line in lines:
            by_iso.setdefault(line.isotope, []).append(line)
        ref_weight: dict[str, float] = {}
        for iso, iso_lines in by_iso.items():
            weights = []
            for entry in iso_lines:
                eff = float(self.efficiency_fn(entry.energy_keV)) if self.efficiency_fn is not None else 1.0
                weights.append(entry.intensity * eff)
            ref_weight[iso] = max(weights) if weights else 0.0

        line_ratio: list[float] = []
        for line in lines:
            eff = float(self.efficiency_fn(line.energy_keV)) if self.efficiency_fn is not None else 1.0
            denom = ref_weight.get(line.isotope, 0.0)
            ratio = (line.intensity * eff / denom) if denom > 0.0 else 0.0
            line_ratio.append(float(ratio))

        groups = self._group_overlapping_lines(lines, overlap_tol_keV=overlap_tol_keV)
        updated = list(lines)
        for group in groups:
            if len(group) <= 1:
                continue
            isotopes = sorted({lines[idx].isotope for idx in group})
            iso_index = {iso: j for j, iso in enumerate(isotopes)}
            num_rows = len(group)
            num_cols = len(isotopes)
            S = np.zeros((num_rows, num_cols), dtype=float)
            N_obs = np.zeros(num_rows, dtype=float)
            for row_idx, line_idx in enumerate(group):
                row_line = lines[line_idx]
                N_obs[row_idx] = row_line.area
                for other_idx in group:
                    other_line = lines[other_idx]
                    distance = abs(row_line.center_keV - other_line.center_keV)
                    if distance <= row_line.half_width_keV + float(overlap_tol_keV):
                        col = iso_index[other_line.isotope]
                        S[row_idx, col] += line_ratio[other_idx]
            if np.allclose(S, 0.0):
                continue
            theta = nnls_solve(S, N_obs)
            for line_idx in group:
                iso = lines[line_idx].isotope
                col = iso_index[iso]
                updated_area = line_ratio[line_idx] * float(theta[col])
                updated[line_idx] = AnalysisLineObservation(
                    isotope=lines[line_idx].isotope,
                    energy_keV=lines[line_idx].energy_keV,
                    intensity=lines[line_idx].intensity,
                    center_keV=lines[line_idx].center_keV,
                    half_width_keV=lines[line_idx].half_width_keV,
                    area=max(float(updated_area), 0.0),
                )
        return updated

    def compute_isotope_counts_thesis(
        self,
        spectrum: NDArray[np.float64],
        *,
        live_time_s: float,
        isotopes: Sequence[str],
        debug_lines: bool = False,
    ) -> Dict[str, float]:
        """
        Compute isotope-wise counts using smoothing, ALS baseline, and peak windows.

        This follows thesis Sec. 2.5.7: apply dead-time correction, smooth, estimate
        ALS baseline, integrate net peak areas within +/- 3 sigma(E), apply overlap
        stripping, and convert to a strength scale using beta * efficiency.
        """
        cfg = self.config
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        all_isotopes = self._analysis_isotopes()
        counts: Dict[str, float] = {iso: 0.0 for iso in all_isotopes}
        if spectrum.size == 0 or energy_axis.size == 0:
            return counts
        if spectrum.size != energy_axis.size:
            min_len = min(spectrum.size, energy_axis.size)
            logger.warning(
                "Spectrum length (%d) != energy axis length (%d); truncating to %d",
                spectrum.size,
                energy_axis.size,
                min_len,
            )
            spectrum = spectrum[:min_len]
            energy_axis = energy_axis[:min_len]

        f_dt = self._dead_time_scale(spectrum, live_time_s)
        y_sm = gaussian_smooth(
            spectrum,
            sigma_bins=cfg.smooth_sigma_bins,
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        baseline = baseline_als(
            y_sm,
            lam=cfg.baseline_lam,
            p=cfg.baseline_p,
            niter=cfg.baseline_niter,
        )
        y_net = np.maximum(y_sm - baseline, 0.0)
        if f_dt != 1.0:
            y_net = y_net * float(f_dt)

        peak_indices = detect_peaks(
            y_net,
            prominence=float(cfg.analysis_peak_prominence),
            distance=int(cfg.analysis_peak_distance),
        )
        peak_energies = energy_axis[peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)

        max_energy = float(np.max(energy_axis))
        line_specs = self._analysis_lines(max_energy_keV=max_energy)
        observations: list[AnalysisLineObservation] = []
        for iso, energy, intensity in line_specs:
            sigma = sigma_E_keV(energy, a=cfg.resolution_a, b=cfg.resolution_b)
            half_width = max(float(cfg.peak_window_sigma) * sigma, 1e-6)
            center = self._closest_peak_center(
                line_energy=energy,
                peak_energies=peak_energies,
                tolerance_keV=float(cfg.analysis_peak_tolerance_keV),
            )
            start, end = self._window_indices(center, half_width, energy_axis)
            if start > end:
                area = 0.0
            else:
                area = float(np.sum(y_net[start : end + 1]))
            observations.append(
                AnalysisLineObservation(
                    isotope=iso,
                    energy_keV=float(energy),
                    intensity=float(intensity),
                    center_keV=float(center),
                    half_width_keV=float(half_width),
                    area=max(float(area), 0.0),
                )
            )

        stripped = self._strip_overlap_groups(
            observations,
            overlap_tol_keV=float(cfg.analysis_overlap_tolerance_keV),
        )

        debug: Dict[str, Dict[str, object]] = {}
        for iso in all_isotopes:
            iso_lines = [line for line in stripped if line.isotope == iso]
            if not iso_lines:
                counts[iso] = 0.0
                debug[iso] = {
                    "line_energies": [],
                    "per_line_peak_areas": [],
                    "per_line_beta_eff": [],
                    "per_line_strength_estimates": [],
                    "denom_sum": 0.0,
                    "dead_time_scale": float(f_dt),
                }
                continue
            per_line_peak = [float(line.area) for line in iso_lines]
            per_line_beta_eff = []
            for line in iso_lines:
                eff = float(self.efficiency_fn(line.energy_keV)) if self.efficiency_fn is not None else 1.0
                per_line_beta_eff.append(float(line.intensity) * eff)
            denom_sum = float(np.sum(per_line_beta_eff))
            num_sum = float(np.sum(per_line_peak))
            if denom_sum > 0.0:
                counts[iso] = max(num_sum / denom_sum, 0.0)
            else:
                counts[iso] = 0.0
            per_line_strength = [
                (peak / be if be > 0.0 else 0.0) for peak, be in zip(per_line_peak, per_line_beta_eff)
            ]
            debug[iso] = {
                "line_energies": [float(line.energy_keV) for line in iso_lines],
                "per_line_peak_areas": per_line_peak,
                "per_line_beta_eff": per_line_beta_eff,
                "per_line_strength_estimates": per_line_strength,
                "denom_sum": denom_sum,
                "dead_time_scale": float(f_dt),
            }
            if debug_lines and iso in {"Eu-154", "Co-60", "Cs-137"}:
                for line, peak, be in zip(iso_lines, per_line_peak, per_line_beta_eff):
                    logger.info(
                        "counts line %s E=%.1f keV net=%.3f beta_eff=%.4f",
                        iso,
                        line.energy_keV,
                        peak,
                        be,
                    )
        self.last_peak_window_debug = debug
        return counts

    def isotope_counts_with_detection(
        self,
        spectrum: NDArray[np.float64],
        *,
        live_time_s: float = 1.0,
        count_method: str = "peak_window",
        detect_isotopes: bool = True,
        detect_threshold_abs: float = 0.1,
        detect_threshold_rel: float = 0.2,
        detect_threshold_rel_by_isotope: Dict[str, float] | None = None,
        detect_snr_threshold: float = 4.0,
        detect_strong_snr_threshold: float = 8.0,
        detect_window_keV: float | None = None,
        detect_window_sigma: float = 3.0,
        detect_sideband_factor: float = 2.0,
        detect_min_line_intensity: float | None = 0.05,
        detect_key_lines_max: int | None = None,
        detect_require_peak_shape: bool = True,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
        peak_tolerance_keV: float = 10.0,
        min_peaks_multi: int = 2,
        min_peaks_by_isotope: Dict[str, int] | None = None,
        spectrum_for_detection: NDArray[np.float64] | None = None,
        detection_state: DetectionState | None = None,
        active_isotopes: Sequence[str] | None = None,
        debug_detection: bool | None = None,
    ) -> Tuple[Dict[str, float], set[str]]:
        """
        Return isotope-wise counts plus detected isotopes.

        Detection uses peak matching and minimum peak counts. The returned
        isotope-wise counts follow ``count_method``:

        - ``peak_window`` uses the thesis pipeline: smoothing, ALS baseline,
          net peak integration within ±3 sigma(E), and branching-ratio weighting.
        - ``response_matrix`` fits the full detector response matrix by NNLS and
          returns source-equivalent counts on the transport-model scale.

        Use min_peaks_by_isotope to override the required peak count for specific
        isotopes. Use detect_threshold_rel_by_isotope to override the relative
        threshold for specific isotopes.
        """
        normalized_count_method = str(count_method).strip().lower()
        if normalized_count_method not in {"peak_window", "response_matrix"}:
            raise ValueError(f"Unknown count_method: {count_method}")
        if not detect_isotopes:
            if normalized_count_method == "response_matrix":
                counts = self.compute_response_model_counts(
                    spectrum,
                    isotopes=self._analysis_isotopes(),
                )
            else:
                counts = self.compute_isotope_counts_thesis(
                    spectrum,
                    live_time_s=live_time_s,
                    isotopes=self._analysis_isotopes(),
                )
            return counts, set(self._analysis_isotopes())
        cfg = self.config
        active_set = set(active_isotopes or [])
        if not active_set and detection_state is not None and detection_state.active:
            active_set = set(detection_state.active)
        if cfg.detect_use_residual and active_set:
            _, fitted_active = self.decompose_subset_with_fit(spectrum, isotopes=sorted(active_set))
        else:
            fitted_active = np.zeros_like(spectrum, dtype=float)
        if cfg.detect_use_residual:
            residual = spectrum - fitted_active
            residual_pos = np.clip(residual, a_min=0.0, a_max=None)
            detect_spec = residual_pos
            candidates = [iso for iso in self.isotope_names if iso not in active_set]
        else:
            detect_spec = spectrum if spectrum_for_detection is None else spectrum_for_detection
            candidates = None
        detected, _ = self.detect_isotopes(
            spectrum,
            live_time_s=live_time_s,
            candidate_isotopes=candidates,
            spectrum_for_detection=detect_spec,
            detection_state=detection_state,
            debug_detection=debug_detection,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            detect_snr_threshold=detect_snr_threshold,
            detect_strong_snr_threshold=detect_strong_snr_threshold,
            detect_window_keV=detect_window_keV,
            detect_window_sigma=detect_window_sigma,
            detect_sideband_factor=detect_sideband_factor,
            detect_min_line_intensity=detect_min_line_intensity,
            detect_key_lines_max=detect_key_lines_max,
            detect_require_peak_shape=detect_require_peak_shape,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance,
            peak_tolerance_keV=peak_tolerance_keV,
            min_peaks_multi=min_peaks_multi,
            min_peaks_by_isotope=min_peaks_by_isotope,
        )
        if detection_state is None:
            new_active = set(detected)
        else:
            new_active = set(detected)
        debug_lines = cfg.detect_debug if debug_detection is None else debug_detection
        if normalized_count_method == "response_matrix":
            counts_full = self.compute_response_model_counts(
                spectrum,
                isotopes=self._analysis_isotopes(),
            )
        else:
            counts_full = self.compute_isotope_counts_thesis(
                spectrum,
                live_time_s=live_time_s,
                isotopes=self._analysis_isotopes(),
                debug_lines=bool(debug_lines),
            )
        if "Eu-154" in new_active and counts_full.get("Eu-154", 0.0) > 0.0:
            active_wo_eu = [iso for iso in new_active if iso != "Eu-154"]
            if active_wo_eu:
                _, fitted_wo_eu = self.decompose_subset_with_fit(spectrum, isotopes=sorted(active_wo_eu))
            else:
                fitted_wo_eu = np.zeros_like(spectrum, dtype=float)
            residual_wo = np.clip(spectrum - fitted_wo_eu, a_min=0.0, a_max=None)
            if not self._validate_isotope_on_residual(
                "Eu-154",
                residual_wo,
                live_time_s=live_time_s,
                detect_threshold_abs=detect_threshold_abs,
                detect_snr_threshold=detect_snr_threshold,
                detect_strong_snr_threshold=detect_strong_snr_threshold,
                detect_window_keV=detect_window_keV,
                detect_window_sigma=detect_window_sigma,
                detect_sideband_factor=detect_sideband_factor,
                detect_require_peak_shape=detect_require_peak_shape,
                detect_min_line_intensity=detect_min_line_intensity,
                peak_prominence=peak_prominence,
                peak_distance=peak_distance,
                peak_tolerance_keV=peak_tolerance_keV,
            ):
                new_active = set(active_wo_eu)
        return counts_full, new_active

    def detect_isotopes(
        self,
        spectrum: NDArray[np.float64],
        *,
        live_time_s: float = 1.0,
        candidate_isotopes: Sequence[str] | None = None,
        spectrum_for_detection: NDArray[np.float64] | None = None,
        detection_state: DetectionState | None = None,
        debug_detection: bool | None = None,
        detect_threshold_abs: float = 0.1,
        detect_threshold_rel: float = 0.2,
        detect_threshold_rel_by_isotope: Dict[str, float] | None = None,
        detect_snr_threshold: float | None = None,
        detect_strong_snr_threshold: float | None = None,
        detect_window_keV: float | None = None,
        detect_window_sigma: float | None = None,
        detect_sideband_factor: float | None = None,
        detect_min_line_intensity: float | None = None,
        detect_key_lines_max: int | None = None,
        detect_require_peak_shape: bool | None = None,
        peak_prominence: float = 0.05,
        peak_distance: int = 5,
        peak_tolerance_keV: float = 10.0,
        min_peaks_multi: int = 2,
        min_peaks_by_isotope: Dict[str, int] | None = None,
    ) -> Tuple[set[str], Dict[str, list[int]]]:
        """
        Detect isotopes using peak assignments and window evidence thresholds.

        Returns:
            (detected_isotopes, peaks_by_iso)
        """
        cfg = self.config
        detect_spec = spectrum if spectrum_for_detection is None else spectrum_for_detection
        corrected = self.preprocess(detect_spec)
        work = corrected
        if float(np.max(work)) <= 0.0:
            work = gaussian_smooth(
                np.asarray(detect_spec, dtype=float),
                sigma_bins=cfg.smooth_sigma_bins,
                use_gpu=self.use_gpu,
                gpu_device=self.gpu_device,
                gpu_dtype=self.gpu_dtype,
            )
        peak_indices = detect_peaks(work, prominence=peak_prominence, distance=peak_distance)
        peaks_by_iso, _ = self._assign_peak_indices(
            self.energy_axis,
            peak_indices,
            self.library,
            tolerance_keV=peak_tolerance_keV,
        )
        peak_energies = self.energy_axis[peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)
        min_lines = min_peaks_by_isotope if min_peaks_by_isotope is not None else cfg.detect_min_lines_by_iso
        snr_threshold = cfg.detect_snr_threshold if detect_snr_threshold is None else detect_snr_threshold
        strong_snr = cfg.detect_strong_snr if detect_strong_snr_threshold is None else detect_strong_snr_threshold
        half_window = cfg.detect_half_window_keV if detect_window_keV is None else detect_window_keV
        sideband_keV = cfg.detect_sideband_keV
        if detect_sideband_factor is not None:
            sideband_keV = max(float(half_window) * float(detect_sideband_factor), 1e-6)
        require_peak_shape = cfg.detect_require_peak_shape if detect_require_peak_shape is None else detect_require_peak_shape
        debug_enabled = cfg.detect_debug if debug_detection is None else debug_detection
        smoothed_raw = gaussian_smooth(
            np.asarray(detect_spec, dtype=float),
            sigma_bins=cfg.smooth_sigma_bins,
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        iso_names = (
            list(self.isotope_names)
            if candidate_isotopes is None
            else [iso for iso in candidate_isotopes if iso in self.library]
        )
        detected_hits: set[str] = set()
        best_snr_overall = -np.inf
        best_iso: str | None = None
        net_abs_counts = float(cfg.detect_net_abs_cps * live_time_s)
        if detect_threshold_abs is not None and float(detect_threshold_abs) >= 1.0:
            net_abs_counts = float(detect_threshold_abs)
        for iso in iso_names:
            key_lines = get_detection_lines_keV(iso)
            if not key_lines:
                nuclide = self.library.get(iso)
                if nuclide is None:
                    continue
                key_lines = [float(line.energy_keV) for line in nuclide.lines]
            best_snr = -np.inf
            good = 0
            evidences = []
            for line_keV in key_lines:
                if detect_min_line_intensity is not None:
                    nuclide = self.library.get(iso)
                    if nuclide is not None:
                        keep = any(
                            (abs(float(line.energy_keV) - float(line_keV)) <= peak_tolerance_keV)
                            and (line.intensity >= float(detect_min_line_intensity))
                            for line in nuclide.lines
                        )
                        if not keep:
                            continue
                ev = line_window_evidence(
                    self.energy_axis,
                    smoothed_raw,
                    line_keV=float(line_keV),
                    half_window_keV=float(half_window),
                    sideband_keV=float(sideband_keV),
                )
                evidences.append((line_keV, ev))
                best_snr = max(best_snr, ev.snr)
                if ev.net < net_abs_counts or ev.snr < float(snr_threshold):
                    continue
                if require_peak_shape and not has_peak_near(
                    peak_energies,
                    float(line_keV),
                    peak_tolerance_keV,
                ):
                    continue
                good += 1
            if best_snr > best_snr_overall:
                best_snr_overall = best_snr
                best_iso = iso
            min_required = int(min_lines.get(iso, 1))
            hit = (good >= min_required) or (best_snr >= float(strong_snr))
            if debug_enabled and iso == "Eu-154":
                for line_keV, ev in evidences:
                    near = (ev.net >= 0.8 * net_abs_counts) or (ev.snr >= 0.8 * float(snr_threshold))
                    if near:
                        logger.debug(
                            "detect %s line=%.2f gross=%.2f bg=%.2f net=%.2f snr=%.2f",
                            iso,
                            line_keV,
                            ev.gross,
                            ev.background,
                            ev.net,
                            ev.snr,
                        )
            if detection_state is None:
                if hit:
                    detected_hits.add(iso)
                continue
            hits = detection_state.hit_streak.get(iso, 0)
            misses = detection_state.miss_streak.get(iso, 0)
            if hit:
                hits += 1
                misses = 0
            else:
                hits = 0
                misses += 1
            detection_state.hit_streak[iso] = hits
            detection_state.miss_streak[iso] = misses
            if hits >= int(cfg.detect_hit_hysteresis):
                detection_state.active.add(iso)
            if iso in detection_state.active and misses >= int(cfg.detect_miss_hysteresis):
                detection_state.active.remove(iso)
        if detection_state is None:
            detected = detected_hits
            if not detected and best_iso is not None and best_snr_overall >= float(strong_snr):
                detected = {best_iso}
            return detected, peaks_by_iso
        return set(detection_state.active), peaks_by_iso

    def _validate_isotope_on_residual(
        self,
        isotope: str,
        residual_spectrum: NDArray[np.float64],
        *,
        live_time_s: float,
        detect_threshold_abs: float | None,
        detect_snr_threshold: float | None,
        detect_strong_snr_threshold: float | None,
        detect_window_keV: float | None,
        detect_window_sigma: float | None,
        detect_sideband_factor: float | None,
        detect_require_peak_shape: bool | None,
        detect_min_line_intensity: float | None,
        peak_prominence: float,
        peak_distance: int,
        peak_tolerance_keV: float,
    ) -> bool:
        """Validate isotope presence on a residual spectrum using net+SNR criteria."""
        cfg = self.config
        if isotope not in self.library:
            return False
        corrected = self.preprocess(residual_spectrum)
        work = corrected
        if float(np.max(work)) <= 0.0:
            work = gaussian_smooth(
                np.asarray(residual_spectrum, dtype=float),
                sigma_bins=cfg.smooth_sigma_bins,
                use_gpu=self.use_gpu,
                gpu_device=self.gpu_device,
                gpu_dtype=self.gpu_dtype,
            )
        peak_indices = detect_peaks(work, prominence=peak_prominence, distance=peak_distance)
        peak_energies = (
            self.energy_axis[peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)
        )
        snr_threshold = cfg.detect_snr_threshold if detect_snr_threshold is None else detect_snr_threshold
        strong_snr = cfg.detect_strong_snr if detect_strong_snr_threshold is None else detect_strong_snr_threshold
        half_window = cfg.detect_half_window_keV if detect_window_keV is None else detect_window_keV
        sideband_keV = cfg.detect_sideband_keV
        if detect_sideband_factor is not None:
            sideband_keV = max(float(half_window) * float(detect_sideband_factor), 1e-6)
        require_peak_shape = cfg.detect_require_peak_shape if detect_require_peak_shape is None else detect_require_peak_shape
        key_lines = get_detection_lines_keV(isotope)
        if not key_lines:
            nuclide = self.library.get(isotope)
            if nuclide is None:
                return False
            key_lines = [float(line.energy_keV) for line in nuclide.lines]
        smoothed_residual = gaussian_smooth(
            np.asarray(residual_spectrum, dtype=float),
            sigma_bins=cfg.smooth_sigma_bins,
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        net_abs_counts = float(cfg.detect_net_abs_cps * live_time_s)
        if detect_threshold_abs is not None and float(detect_threshold_abs) >= 1.0:
            net_abs_counts = float(detect_threshold_abs)
        min_required = int(cfg.detect_min_lines_by_iso.get(isotope, 1))
        good = 0
        best_snr = -np.inf
        for line_keV in key_lines:
            if detect_min_line_intensity is not None:
                nuclide = self.library.get(isotope)
                if nuclide is not None:
                    keep = any(
                        (abs(float(line.energy_keV) - float(line_keV)) <= peak_tolerance_keV)
                        and (line.intensity >= float(detect_min_line_intensity))
                        for line in nuclide.lines
                    )
                    if not keep:
                        continue
            ev = line_window_evidence(
                self.energy_axis,
                smoothed_residual,
                line_keV=float(line_keV),
                half_window_keV=float(half_window),
                sideband_keV=float(sideband_keV),
            )
            best_snr = max(best_snr, ev.snr)
            if ev.net < net_abs_counts or ev.snr < float(snr_threshold):
                continue
            if require_peak_shape and not has_peak_near(
                peak_energies,
                float(line_keV),
                peak_tolerance_keV,
            ):
                continue
            good += 1
        return (good >= min_required) or (best_snr >= float(strong_snr))

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
        peak_prominence: float | None = None,
        peak_distance: int | None = None,
    ) -> Dict[str, list[float]]:
        """
        Compute per-line peak areas for analysis lines using energy windows.

        This follows Eq. (20) style window integration and can optionally apply
        spectral stripping to separate overlapping peaks. If window_keV is None,
        use ±window_sigma * sigma(E) based on the detector resolution model.
        """
        cfg = self.config
        corrected = np.asarray(spectrum, dtype=float)
        if smooth_sigma_bins > 0.0:
            corrected = gaussian_smooth(
                corrected,
                sigma_bins=smooth_sigma_bins,
                use_gpu=self.use_gpu,
                gpu_device=self.gpu_device,
                gpu_dtype=self.gpu_dtype,
            )
        if subtract_baseline:
            base = baseline_als(
                corrected,
                lam=cfg.baseline_lam,
                p=cfg.baseline_p,
                niter=cfg.baseline_niter,
            )
            corrected = np.clip(corrected - base, a_min=0.0, a_max=None)
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        iso_names = list(isotopes) if isotopes is not None else self._analysis_isotopes()
        iso_names = [iso for iso in iso_names if iso in self._analysis_isotopes()]
        max_energy = float(np.max(energy_axis)) if energy_axis.size else 0.0

        prominence = cfg.analysis_peak_prominence if peak_prominence is None else peak_prominence
        distance = cfg.analysis_peak_distance if peak_distance is None else peak_distance
        peak_indices = detect_peaks(
            corrected,
            prominence=float(prominence),
            distance=int(distance),
        )
        peak_energies = energy_axis[peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)

        line_specs: list[tuple[str, float, float]] = []
        line_indices_by_iso: Dict[str, list[int]] = {iso: [] for iso in iso_names}
        for iso in iso_names:
            lines = get_analysis_lines_with_intensity(
                iso,
                self.library,
                max_energy_keV=max_energy,
            )
            for energy, intensity in lines:
                line_indices_by_iso[iso].append(len(line_specs))
                line_specs.append((iso, float(energy), float(intensity)))

        observations: list[AnalysisLineObservation] = []
        for iso, energy, intensity in line_specs:
            half_width = window_keV
            if half_width is None:
                sigma = float(self.resolution_fn(energy))
                half_width = max(float(window_sigma) * sigma, 1e-6)
            center = self._closest_peak_center(
                line_energy=energy,
                peak_energies=peak_energies,
                tolerance_keV=float(peak_tolerance_keV),
            )
            start, end = self._window_indices(center, float(half_width), energy_axis)
            if start > end:
                area = 0.0
            else:
                area = float(np.sum(corrected[start : end + 1]))
            observations.append(
                AnalysisLineObservation(
                    isotope=iso,
                    energy_keV=float(energy),
                    intensity=float(intensity),
                    center_keV=float(center),
                    half_width_keV=float(half_width),
                    area=max(float(area), 0.0),
                )
            )

        if apply_stripping and observations:
            stripped = self._strip_overlap_groups(observations, overlap_tol_keV=float(peak_tolerance_keV))
        else:
            stripped = observations

        line_counts: Dict[str, list[float]] = {}
        for iso in iso_names:
            indices = line_indices_by_iso.get(iso, [])
            line_counts[iso] = [float(stripped[idx].area) for idx in indices]
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
        by beta * efficiency within each nuclide to return counts in the strength
        scale. When apply_stripping is enabled, peak areas are corrected using
        spectral stripping.
        """
        iso_names = list(isotopes) if isotopes is not None else self._analysis_isotopes()
        iso_names = [iso for iso in iso_names if iso in self._analysis_isotopes()]
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        max_energy = float(np.max(energy_axis)) if energy_axis.size else 0.0
        line_specs: Dict[str, list[tuple[float, float]]] = {}
        for iso in iso_names:
            line_specs[iso] = get_analysis_lines_with_intensity(
                iso,
                self.library,
                max_energy_keV=max_energy,
            )
        line_counts = self.peak_line_counts(
            spectrum,
            isotopes=iso_names,
            window_keV=window_keV,
            window_sigma=window_sigma,
            smooth_sigma_bins=smooth_sigma_bins,
            subtract_baseline=subtract_baseline,
            apply_stripping=apply_stripping,
            peak_tolerance_keV=peak_tolerance_keV,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance,
        )
        counts: Dict[str, float] = {iso: 0.0 for iso in self._analysis_isotopes()}
        debug: Dict[str, Dict[str, object]] = {}
        for iso in iso_names:
            lines = line_specs.get(iso, [])
            per_line = line_counts.get(iso, [])
            per_line_peak = [float(val) for val in per_line]
            per_line_beta_eff = []
            for (energy, intensity) in lines:
                eff = float(self.efficiency_fn(energy)) if self.efficiency_fn is not None else 1.0
                per_line_beta_eff.append(float(intensity) * eff)
            denom_sum = float(np.sum(per_line_beta_eff))
            num_sum = float(np.sum(per_line_peak))
            if denom_sum > 0.0:
                counts[iso] = max(num_sum / denom_sum, 0.0)
            else:
                counts[iso] = 0.0
            per_line_strength = [
                (peak / be if be > 0.0 else 0.0) for peak, be in zip(per_line_peak, per_line_beta_eff)
            ]
            debug[iso] = {
                "line_energies": [float(energy) for energy, _ in lines],
                "per_line_peak_areas": per_line_peak,
                "per_line_beta_eff": per_line_beta_eff,
                "per_line_strength_estimates": per_line_strength,
                "denom_sum": denom_sum,
            }
        for iso in self._analysis_isotopes():
            if iso in debug:
                continue
            debug[iso] = {
                "line_energies": [],
                "per_line_peak_areas": [],
                "per_line_beta_eff": [],
                "per_line_strength_estimates": [],
                "denom_sum": 0.0,
            }
        self.last_peak_window_debug = debug
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

    def debug_peak_windows(
        self,
        spectrum: NDArray[np.float64],
        *,
        live_time_s: float = 1.0,
        isotopes: Sequence[str] | None = None,
        output_path: str | None = None,
    ) -> None:
        """
        Plot raw/smoothed/baseline spectra and shaded integration windows.
        """
        import matplotlib.pyplot as plt

        cfg = self.config
        energy_axis = np.asarray(self.energy_axis, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.size != energy_axis.size:
            min_len = min(spectrum.size, energy_axis.size)
            spectrum = spectrum[:min_len]
            energy_axis = energy_axis[:min_len]
        f_dt = self._dead_time_scale(spectrum, live_time_s)
        y_sm = gaussian_smooth(
            spectrum,
            sigma_bins=cfg.smooth_sigma_bins,
            use_gpu=self.use_gpu,
            gpu_device=self.gpu_device,
            gpu_dtype=self.gpu_dtype,
        )
        baseline = baseline_als(
            y_sm,
            lam=cfg.baseline_lam,
            p=cfg.baseline_p,
            niter=cfg.baseline_niter,
        )
        y_net = np.maximum(y_sm - baseline, 0.0)
        if f_dt != 1.0:
            y_net = y_net * float(f_dt)

        peak_indices = detect_peaks(
            y_net,
            prominence=float(cfg.analysis_peak_prominence),
            distance=int(cfg.analysis_peak_distance),
        )
        peak_energies = energy_axis[peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)

        max_energy = float(np.max(energy_axis)) if energy_axis.size else 0.0
        lines = self._analysis_lines(max_energy_keV=max_energy)
        if isotopes is not None:
            iso_set = set(isotopes)
            lines = [entry for entry in lines if entry[0] in iso_set]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(energy_axis, spectrum, label="Raw", alpha=0.5)
        ax.plot(energy_axis, y_sm, label="Smoothed")
        ax.plot(energy_axis, baseline, label="Baseline")
        ax.plot(energy_axis, y_net, label="Net", alpha=0.8)

        colors = {"Cs-137": "tab:blue", "Co-60": "tab:orange", "Eu-154": "tab:green"}
        for iso, energy, _ in lines:
            sigma = sigma_E_keV(energy, a=cfg.resolution_a, b=cfg.resolution_b)
            half_width = max(float(cfg.peak_window_sigma) * sigma, 1e-6)
            center = self._closest_peak_center(
                line_energy=energy,
                peak_energies=peak_energies,
                tolerance_keV=float(cfg.analysis_peak_tolerance_keV),
            )
            color = colors.get(iso, "gray")
            ax.axvspan(
                center - half_width,
                center + half_width,
                alpha=0.2,
                color=color,
            )
            ax.axvline(center, color=color, linestyle="--", linewidth=0.8)

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")
        ax.set_title("Peak Windows and Baseline")
        ax.legend()
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=150)
            plt.close(fig)

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
        line_energies: Dict[str, NDArray[np.float64]] | None = None,
    ) -> Tuple[Dict[str, list[int]], list[int]]:
        """Assign detected peak indices to isotopes using closest line energies."""
        peaks_by_iso: Dict[str, list[int]] = {iso: [] for iso in library}
        unassigned: list[int] = []
        if line_energies is None:
            line_energy_map = {
                iso: np.array([line.energy_keV for line in nuclide.lines], dtype=float)
                for iso, nuclide in library.items()
            }
        else:
            line_energy_map = {}
            for iso, nuclide in library.items():
                if iso in line_energies:
                    line_energy_map[iso] = np.asarray(line_energies[iso], dtype=float)
                else:
                    line_energy_map[iso] = np.array([line.energy_keV for line in nuclide.lines], dtype=float)
        for idx in peak_indices:
            energy = float(energy_axis[int(idx)])
            best_iso = None
            best_diff = float("inf")
            for iso, energies in line_energy_map.items():
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
        overrides: Dict[str, int] | None = None,
    ) -> Dict[str, int]:
        """Return minimum peak counts required to accept each isotope."""
        min_counts: Dict[str, int] = {}
        for iso, nuclide in library.items():
            line_count = len(nuclide.lines)
            min_counts[iso] = 1 if line_count <= 1 else max(int(min_peaks_multi), 1)
        if overrides:
            for iso, value in overrides.items():
                if iso not in min_counts:
                    continue
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    continue
                if count < 1:
                    count = 1
                min_counts[iso] = count
        return min_counts
