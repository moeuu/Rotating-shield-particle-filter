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
        variances = self._apply_response_diagnostics_floor(counts_out, variances)
        variances = self._apply_shield_systematic_variance_floor(
            counts_out,
            variances,
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
        variance_floor = (
            self.decomposer.estimate_count_variances_from_spectrum_variance(
                spectrum_variance,
                isotopes=list(counts.keys()),
            )
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
                    (max(float(count), 0.0) ** 2)
                    / max(float(effective_entries), 1.0),
                )
            )
            for iso, count in counts.items()
        }

    def _apply_response_diagnostics_floor(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
    ) -> dict[str, float]:
        """Soften PF updates when response regression diagnostics are unreliable."""
        config = getattr(self.decomposer, "config", None)
        if config is None or not bool(
            getattr(config, "response_poisson_diagnostic_variance_enable", True)
        ):
            return variances
        diagnostics = dict(
            getattr(self.decomposer, "last_response_poisson_diagnostics", {})
        )
        if not diagnostics:
            return variances

        rel_sigma = self._diagnostic_relative_sigma(diagnostics)
        rel_sigma_by_isotope = self._diagnostic_relative_sigma_by_isotope(
            diagnostics
        )
        max_count = max(
            (max(float(value), 0.0) for value in counts.values()),
            default=0.0,
        )
        floors: dict[str, float] = {}
        inflated: dict[str, float] = {}
        for isotope, count in counts.items():
            iso_rel_sigma = max(
                float(rel_sigma),
                float(rel_sigma_by_isotope.get(isotope, 0.0)),
            )
            if iso_rel_sigma <= 0.0:
                inflated[isotope] = float(max(variances.get(isotope, 1.0), 1.0))
                continue
            reference_count = max(float(count), 0.05 * max_count, 1.0)
            floor = float((iso_rel_sigma * reference_count) ** 2)
            floors[isotope] = floor
            inflated[isotope] = float(max(variances.get(isotope, 1.0), floor, 1.0))

        if floors:
            diagnostics["runtime_diagnostic_variance_floor"] = {
                isotope: float(value) for isotope, value in sorted(floors.items())
            }
            self.decomposer.last_response_poisson_diagnostics = diagnostics
        return inflated

    def _apply_shield_systematic_variance_floor(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
        *,
        transport_metadata: Mapping[str, object] | None,
    ) -> dict[str, float]:
        """
        Inflate count variances for shielded spectra that carry model bias risk.

        Rotating-shield postures are valuable temporal codes for localization,
        but their spectra are more sensitive to geometry and scattering mismatch
        than weakly shielded anchor views. This floor keeps those postures in
        the PF likelihood while preventing them from dominating strength refits.
        """
        config = getattr(self.decomposer, "config", None)
        if config is None or not bool(
            getattr(
                config,
                "response_poisson_shield_systematic_variance_enable",
                False,
            )
        ):
            return variances
        if transport_metadata is None:
            return variances
        thickness_scale = self._metadata_float(
            transport_metadata,
            "shield_thickness_scale",
        )
        if thickness_scale is None:
            return variances
        zero_threshold = max(
            float(
                getattr(
                    config,
                    "response_poisson_shield_systematic_zero_thickness_threshold",
                    1.0e-9,
                )
            ),
            0.0,
        )
        if thickness_scale is not None and thickness_scale <= zero_threshold:
            return variances

        pair_id = self._shield_pair_id_from_metadata(transport_metadata)
        anchor_pair_ids = self._anchor_pair_ids_from_config()
        if pair_id is not None and pair_id in anchor_pair_ids:
            rel_sigma = max(
                float(
                    getattr(
                        config,
                        "response_poisson_shield_systematic_anchor_rel_sigma",
                        0.0,
                    )
                ),
                0.0,
            )
        else:
            rel_sigma = max(
                float(
                    getattr(
                        config,
                        "response_poisson_shield_systematic_rel_sigma",
                        0.0,
                    )
                ),
                0.0,
            )
        if rel_sigma <= 0.0:
            return variances

        max_count = max(
            (max(float(value), 0.0) for value in counts.values()),
            default=0.0,
        )
        min_fraction = max(
            float(
                getattr(
                    config,
                    "response_poisson_shield_systematic_min_count_fraction",
                    0.05,
                )
            ),
            0.0,
        )
        floors: dict[str, float] = {}
        inflated: dict[str, float] = {}
        for isotope, count in counts.items():
            reference = max(float(count), min_fraction * max_count, 1.0)
            floor = float((rel_sigma * reference) ** 2)
            floors[isotope] = floor
            inflated[isotope] = float(max(variances.get(isotope, 1.0), floor, 1.0))

        diagnostics = dict(
            getattr(self.decomposer, "last_response_poisson_diagnostics", {})
        )
        diagnostics["runtime_shield_systematic_variance_floor"] = {
            isotope: float(value) for isotope, value in sorted(floors.items())
        }
        diagnostics["runtime_shield_systematic_pair_id"] = (
            None if pair_id is None else int(pair_id)
        )
        diagnostics["runtime_shield_systematic_rel_sigma"] = float(rel_sigma)
        diagnostics["runtime_shield_systematic_anchor_pair"] = bool(
            pair_id is not None and pair_id in anchor_pair_ids
        )
        self.decomposer.last_response_poisson_diagnostics = diagnostics
        return inflated

    def _diagnostic_relative_sigma(self, diagnostics: Mapping[str, object]) -> float:
        """Return a global relative sigma implied by response-fit diagnostics."""
        config = self.decomposer.config
        rel_sigma = 0.0
        reduced_chi2 = self._mapping_float(diagnostics, "reduced_chi2")
        chi2_threshold = max(
            float(
                getattr(
                    config,
                    "response_poisson_diagnostic_reduced_chi2_threshold",
                    2.0,
                )
            ),
            1.0e-12,
        )
        if reduced_chi2 is not None and reduced_chi2 > chi2_threshold:
            rel_sigma = max(
                rel_sigma,
                float(
                    getattr(
                        config,
                        "response_poisson_diagnostic_reduced_chi2_scale",
                        0.5,
                    )
                )
                * np.sqrt(reduced_chi2 / chi2_threshold - 1.0),
            )

        condition = max(
            self._mapping_float(diagnostics, "design_condition_number") or 0.0,
            self._mapping_float(diagnostics, "fisher_condition_number") or 0.0,
        )
        condition_threshold = max(
            float(
                getattr(
                    config,
                    "response_poisson_diagnostic_condition_threshold",
                    1.0e4,
                )
            ),
            1.0,
        )
        if np.isfinite(condition) and condition > condition_threshold:
            rel_sigma = max(
                rel_sigma,
                float(
                    getattr(
                        config,
                        "response_poisson_diagnostic_condition_scale",
                        0.25,
                    )
                )
                * np.log10(condition / condition_threshold),
            )
        return float(max(rel_sigma, 0.0))

    def _diagnostic_relative_sigma_by_isotope(
        self,
        diagnostics: Mapping[str, object],
    ) -> dict[str, float]:
        """Return per-isotope relative sigma from crosstalk and low-SNR diagnostics."""
        config = self.decomposer.config
        rel_by_isotope: dict[str, float] = {}
        corr_map = diagnostics.get("coefficient_correlation_by_isotope", {})
        if isinstance(corr_map, Mapping):
            threshold = min(
                max(
                    float(
                        getattr(
                            config,
                            "response_poisson_crosstalk_corr_threshold",
                            0.85,
                        )
                    ),
                    0.0,
                ),
                0.999,
            )
            scale = max(
                float(
                    getattr(
                        config,
                        "response_poisson_crosstalk_variance_scale",
                        1.0,
                    )
                ),
                0.0,
            )
            min_rel = max(
                float(
                    getattr(
                        config,
                        "response_poisson_crosstalk_min_rel_sigma",
                        0.25,
                    )
                ),
                0.0,
            )
            for isotope, value in corr_map.items():
                try:
                    corr = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(corr) and corr > threshold:
                    excess = (corr - threshold) / max(1.0 - threshold, 1.0e-6)
                    rel_by_isotope[str(isotope)] = max(
                        rel_by_isotope.get(str(isotope), 0.0),
                        min_rel,
                        scale * excess,
                    )

        low_snr = diagnostics.get(
            "low_snr_photopeak_suppression",
            diagnostics.get("low_snr_suppression", {}),
        )
        if isinstance(low_snr, Mapping):
            for isotope, payload in low_snr.items():
                if not isinstance(payload, Mapping):
                    continue
                reason = str(payload.get("reason", ""))
                if reason:
                    rel_by_isotope[str(isotope)] = max(
                        rel_by_isotope.get(str(isotope), 0.0),
                        0.5,
                    )
        return rel_by_isotope

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

    def _metadata_int(self, metadata: Mapping[str, object], key: str) -> int | None:
        """Return a finite metadata value as int when available."""
        value = self._metadata_float(metadata, key)
        if value is None:
            return None
        return int(value)

    def _shield_pair_id_from_metadata(
        self,
        metadata: Mapping[str, object],
    ) -> int | None:
        """Return the flattened shield pair id from observation metadata."""
        direct = self._metadata_int(metadata, "shield_pair_id")
        if direct is not None:
            return direct
        fe_index = self._metadata_int(metadata, "fe_orientation_index")
        pb_index = self._metadata_int(metadata, "pb_orientation_index")
        if fe_index is None or pb_index is None:
            return None
        num_orientations = self._metadata_int(metadata, "shield_num_orientations")
        if num_orientations is None or num_orientations <= 0:
            num_orientations = 8
        return int(fe_index) * int(num_orientations) + int(pb_index)

    def _anchor_pair_ids_from_config(self) -> set[int]:
        """Return configured weakly shielded anchor pair ids."""
        config = self.decomposer.config
        raw_value = getattr(
            config,
            "response_poisson_shield_systematic_anchor_pair_ids",
            (),
        )
        if raw_value is None:
            return set()
        if isinstance(raw_value, (str, bytes)):
            values = [item for item in str(raw_value).split(",") if item.strip()]
        else:
            values = list(raw_value)
        anchors: set[int] = set()
        for value in values:
            try:
                anchors.add(int(value))
            except (TypeError, ValueError):
                continue
        return anchors

    @staticmethod
    def _mapping_float(metadata: Mapping[str, object], key: str) -> float | None:
        """Return a finite mapping value as float when available."""
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
