"""Runtime isotope-count extraction for PF observation ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from spectrum.pipeline import SpectralDecomposer


@dataclass(frozen=True)
class RuntimeCountResult:
    """Container for isotope counts, variances, and detection labels."""

    counts: dict[str, float]
    variances: dict[str, float]
    detected: set[str]


class RuntimeCountExtractor:
    """Extract PF-ready isotope counts from spectra through the runtime path."""

    STANDARD_METHOD = "response_poisson"
    DIAGNOSTIC_METHODS = frozenset({"photopeak_nnls"})
    REJECTED_RUNTIME_METHODS = frozenset({"peak_window", "response_matrix"})

    def __init__(
        self,
        decomposer: SpectralDecomposer,
        *,
        count_method: str = STANDARD_METHOD,
        allow_diagnostic_methods: bool = False,
    ) -> None:
        """Create a runtime count extractor for one spectral decomposer."""
        self.decomposer = decomposer
        self.count_method = self.validate_count_method(
            count_method,
            allow_diagnostic_methods=allow_diagnostic_methods,
        )

    @classmethod
    def validate_count_method(
        cls,
        count_method: str,
        *,
        allow_diagnostic_methods: bool = False,
    ) -> str:
        """Return a normalized count method or raise for runtime-invalid modes."""
        normalized = str(count_method).strip().lower()
        allowed_methods = {cls.STANDARD_METHOD}
        if allow_diagnostic_methods:
            allowed_methods.update(cls.DIAGNOSTIC_METHODS)
        if normalized in allowed_methods:
            return normalized
        diagnostic = ", ".join(sorted(cls.DIAGNOSTIC_METHODS))
        raise ValueError(
            "Runtime spectrum_count_method must be 'response_poisson'. "
            f"Diagnostic-only methods ({diagnostic}) must stay outside the "
            "runtime PF observation-ingestion path. "
            "peak_window and response_matrix are not runtime count-ingestion "
            "methods."
        )

    def extract(
        self,
        spectrum: NDArray[np.float64],
        *,
        live_time_s: float,
        detect_threshold_abs: float,
        detect_threshold_rel: float,
        detect_threshold_rel_by_isotope: dict[str, float],
        min_peaks_by_isotope: dict[str, int] | None,
        spectrum_variance: NDArray[np.float64] | None = None,
        transport_metadata: Mapping[str, object] | None = None,
    ) -> RuntimeCountResult:
        """Extract isotope counts and conservative variances for PF updates."""
        counts, detected = self.decomposer.isotope_counts_with_detection(
            spectrum,
            live_time_s=live_time_s,
            count_method=self.count_method,
            active_isotopes=None,
            detect_threshold_abs=detect_threshold_abs,
            detect_threshold_rel=detect_threshold_rel,
            detect_threshold_rel_by_isotope=detect_threshold_rel_by_isotope,
            min_peaks_by_isotope=min_peaks_by_isotope,
        )
        counts_out = {iso: float(val) for iso, val in counts.items()}
        variances = {
            iso: float(
                max(
                    self.decomposer.last_count_variances.get(iso, max(val, 1.0)),
                    1.0,
                )
            )
            for iso, val in counts_out.items()
        }
        variances = self._apply_spectrum_variance_floor(
            counts_out,
            variances,
            spectrum_variance=spectrum_variance,
        )
        variances = self._apply_effective_entries_floor(
            counts_out,
            variances,
            spectrum=spectrum,
            spectrum_variance=spectrum_variance,
            transport_metadata=transport_metadata,
        )
        return RuntimeCountResult(counts_out, variances, set(detected))

    def _apply_spectrum_variance_floor(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
        *,
        spectrum_variance: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        """Apply count-variance floors propagated from weighted spectra."""
        if spectrum_variance is None:
            return variances
        variance_floor = self.decomposer.estimate_count_variances_from_spectrum_variance(
            spectrum_variance,
            isotopes=list(counts.keys()),
        )
        return {
            iso: float(max(variances.get(iso, 1.0), variance_floor.get(iso, 1.0)))
            for iso in counts
        }

    def _apply_effective_entries_floor(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
        *,
        spectrum: NDArray[np.float64],
        spectrum_variance: NDArray[np.float64] | None,
        transport_metadata: Mapping[str, object] | None,
    ) -> dict[str, float]:
        """Apply a variance floor from weighted effective spectrum entries."""
        effective_entries = self._effective_entries_from_spectrum(
            spectrum,
            spectrum_variance,
        )
        metadata_effective_entries = (
            None
            if transport_metadata is None
            else self._metadata_float(
                transport_metadata,
                "weighted_spectrum_effective_entries",
            )
        )
        if effective_entries is None:
            effective_entries = metadata_effective_entries
        if effective_entries is None or effective_entries <= 0.0:
            return variances
        return {
            iso: float(
                max(
                    variances.get(iso, 1.0),
                    (max(float(count), 0.0) ** 2) / max(float(effective_entries), 1.0),
                )
            )
            for iso, count in counts.items()
        }

    @staticmethod
    def _effective_entries_from_spectrum(
        spectrum: NDArray[np.float64],
        spectrum_variance: NDArray[np.float64] | None,
    ) -> float | None:
        """Estimate effective entries from summed weighted spectrum variance."""
        if spectrum_variance is None:
            return None
        total_sum = float(np.sum(np.clip(spectrum, a_min=0.0, a_max=None)))
        total_variance = float(
            np.sum(np.clip(spectrum_variance, a_min=0.0, a_max=None))
        )
        if total_sum <= 0.0 or total_variance <= 0.0:
            return None
        return float((total_sum * total_sum) / total_variance)

    @staticmethod
    def _metadata_float(metadata: Mapping[str, object], key: str) -> float | None:
        """Return a finite metadata value as float when available."""
        value = metadata.get(key)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result):
            return None
        return result
