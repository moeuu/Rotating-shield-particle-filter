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
    covariance: dict[str, dict[str, float]] | None = None


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
        variance_stages = {"formal_input": dict(variances)}
        # The configured ceiling is a guard for the formal regression
        # covariance.  Statistical and model-discrepancy components are
        # applied afterwards, so they remain visible instead of being folded
        # into an opaque, order-dependent exception to the ceiling.
        variances = self._apply_response_poisson_variance_ceiling(
            counts_out,
            variances,
        )
        variance_stages["formal_after_ceiling"] = dict(variances)
        variances = self._apply_spectrum_variance_floor(
            counts_out,
            variances,
            spectrum_variance=spectrum_variance,
        )
        variance_stages["spectrum_statistical"] = dict(variances)
        variances = self._apply_effective_entries_floor(
            counts_out,
            variances,
            spectrum=spectrum,
            spectrum_variance=spectrum_variance,
            transport_metadata=transport_metadata,
        )
        variance_stages["transport_statistical"] = dict(variances)
        variances = self._apply_response_diagnostics_floor(counts_out, variances)
        variance_stages["isotope_diagnostic"] = dict(variances)
        variances = self._apply_shield_systematic_variance_floor(
            counts_out,
            variances,
            transport_metadata=transport_metadata,
        )
        variance_stages["shield_systematic"] = dict(variances)
        self._record_runtime_variance_components(counts_out, variance_stages)
        covariance = self._count_covariance_with_runtime_variances(
            counts_out,
            variances,
            formal_variances=variance_stages["formal_after_ceiling"],
        )
        return RuntimeCountResult(counts_out, variances, set(detected), covariance)

    def _count_covariance_with_runtime_variances(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
        *,
        formal_variances: Mapping[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Return isotope count covariance after runtime variance guards."""
        isotopes = [str(isotope) for isotope in counts]
        source_covariance = getattr(self.decomposer, "last_count_covariance", {})
        if not isinstance(source_covariance, Mapping):
            source_covariance = {}
        runtime_variances = {
            isotope: max(
                float(variances.get(isotope, counts.get(isotope, 1.0))),
                1.0,
            )
            for isotope in isotopes
        }
        covariance = {
            row_iso: {
                col_iso: (
                    float(runtime_variances[row_iso])
                    if row_iso == col_iso
                    else 0.0
                )
                for col_iso in isotopes
            }
            for row_iso in isotopes
        }
        for row_index, row_iso in enumerate(isotopes):
            source_row = source_covariance.get(row_iso, {})
            if not isinstance(source_row, Mapping):
                source_row = {}
            source_row_var = self._mapping_float(source_row, row_iso)
            for col_iso in isotopes[row_index + 1 :]:
                source_col_row = source_covariance.get(col_iso, {})
                if not isinstance(source_col_row, Mapping):
                    source_col_row = {}
                source_col_var = self._mapping_float(source_col_row, col_iso)
                if (
                    source_row_var is None
                    or source_col_var is None
                    or source_row_var <= 0.0
                    or source_col_var <= 0.0
                ):
                    continue
                reciprocal_values = [
                    value
                    for value in (
                        self._mapping_float(source_row, col_iso),
                        self._mapping_float(source_col_row, row_iso),
                    )
                    if value is not None
                ]
                if not reciprocal_values:
                    continue
                source_offdiag = float(np.mean(reciprocal_values))
                if formal_variances is not None:
                    # The ceiling acts on the complete formal covariance, not
                    # only its diagonal.  Congruence scaling preserves the
                    # fitted correlation and positive-semidefinite structure.
                    row_formal = max(
                        float(formal_variances.get(row_iso, source_row_var)),
                        0.0,
                    )
                    col_formal = max(
                        float(formal_variances.get(col_iso, source_col_var)),
                        0.0,
                    )
                    row_scale = np.sqrt(min(row_formal / source_row_var, 1.0))
                    col_scale = np.sqrt(min(col_formal / source_col_var, 1.0))
                    source_offdiag *= float(row_scale * col_scale)
                # Extra runtime floors represent independent statistical or
                # model-discrepancy components.  Preserve the formal off-
                # diagonal covariance itself; rescaling its correlation to the
                # enlarged runtime diagonal would incorrectly correlate those
                # independent components and make the later conservative
                # projection grow a second time.
                runtime_bound = float(
                    np.sqrt(
                        runtime_variances[row_iso] * runtime_variances[col_iso]
                    )
                )
                formal_offdiag = float(
                    np.clip(source_offdiag, -runtime_bound, runtime_bound)
                )
                covariance[row_iso][col_iso] = formal_offdiag
                covariance[col_iso][row_iso] = formal_offdiag
        return covariance

    def _record_runtime_variance_components(
        self,
        counts: Mapping[str, float],
        stages: Mapping[str, Mapping[str, float]],
    ) -> None:
        """Record the sequential contribution of each runtime variance stage."""
        diagnostics = dict(
            getattr(self.decomposer, "last_response_poisson_diagnostics", {})
        )
        components: dict[str, dict[str, float]] = {}
        for isotope in counts:
            previous = 0.0
            payload: dict[str, float] = {}
            for stage_name, stage_values in stages.items():
                value = max(float(stage_values.get(isotope, previous)), 1.0)
                payload[f"{stage_name}_variance"] = value
                payload[f"{stage_name}_increment"] = max(value - previous, 0.0)
                previous = value
            payload["final_variance"] = previous
            components[str(isotope)] = payload
        diagnostics["runtime_variance_components"] = components
        self.decomposer.last_response_poisson_diagnostics = diagnostics

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
            getattr(config, "response_poisson_diagnostic_variance_enable", False)
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
            # Each diagnostic sigma is relative to this isotope's extracted
            # count. Borrowing a fixed fraction of the strongest other isotope
            # transfers channel magnitude without a probabilistic crosstalk
            # model and can make a weak isotope's covariance arbitrarily large.
            reference_count = max(float(count), 1.0)
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

    def _apply_response_poisson_variance_ceiling(
        self,
        counts: dict[str, float],
        variances: dict[str, float],
    ) -> dict[str, float]:
        """
        Cap formal response-Poisson covariance before PF observation ingestion.

        Ill-conditioned full-spectrum regressions can produce enormous formal
        coefficient covariance even when independent replay shows the retained
        count is accurate in ordinary high-count cases. Runtime configurations
        should apply diagnostic and guard floors in the later explicit
        diagnostic stage. The preserve flags remain only for compatibility with
        legacy callers that already embedded those floors in the formal input.
        """
        config = getattr(self.decomposer, "config", None)
        if config is None or not bool(
            getattr(config, "response_poisson_count_variance_ceiling_enable", True)
        ):
            return variances
        rel_sigma = max(
            float(
                getattr(
                    config,
                    "response_poisson_count_variance_max_rel_sigma",
                    0.15,
                )
            ),
            0.0,
        )
        abs_sigma = max(
            float(
                getattr(
                    config,
                    "response_poisson_count_variance_max_abs_sigma",
                    40.0,
                )
            ),
            0.0,
        )
        if rel_sigma <= 0.0 and abs_sigma <= 0.0:
            return {iso: float(max(var, 1.0)) for iso, var in variances.items()}

        diagnostics = dict(
            getattr(self.decomposer, "last_response_poisson_diagnostics", {})
        )
        capped: dict[str, float] = {}
        cap_debug: dict[str, dict[str, float]] = {}
        preserved_debug: dict[str, dict[str, float]] = {}
        for isotope, count in counts.items():
            count_value = max(float(count), 0.0)
            base_variance = max(
                float(variances.get(isotope, max(count_value, 1.0))),
                1.0,
            )
            reference_count = max(count_value, 1.0)
            variance_ceiling = max(
                count_value,
                (rel_sigma * reference_count) ** 2,
                abs_sigma**2,
                1.0,
            )
            if base_variance > variance_ceiling:
                preserved_floor = self._response_poisson_preserved_variance_floor(
                    isotope,
                    diagnostics,
                    base_variance=base_variance,
                )
                output_variance = max(
                    float(variance_ceiling),
                    min(float(base_variance), float(preserved_floor)),
                )
                capped[isotope] = float(output_variance)
                cap_debug[isotope] = {
                    "input_variance": float(base_variance),
                    "capped_variance": float(output_variance),
                    "uncertainty_ceiling": float(variance_ceiling),
                    "count": float(count_value),
                    "max_rel_sigma": float(rel_sigma),
                    "max_abs_sigma": float(abs_sigma),
                }
                if output_variance > variance_ceiling:
                    cap_debug[isotope]["preserved_variance_floor"] = float(
                        output_variance
                    )
                    preserved_debug[isotope] = {
                        "input_variance": float(base_variance),
                        "uncertainty_ceiling": float(variance_ceiling),
                        "preserved_variance_floor": float(output_variance),
                    }
            else:
                capped[isotope] = float(base_variance)

        if cap_debug:
            diagnostics["runtime_variance_ceiling"] = cap_debug
            if preserved_debug:
                diagnostics["runtime_variance_ceiling_preserved"] = preserved_debug
            self.decomposer.last_response_poisson_diagnostics = diagnostics
        return capped

    def _response_poisson_preserved_variance_floor(
        self,
        isotope: str,
        diagnostics: Mapping[str, object],
        *,
        base_variance: float,
    ) -> float:
        """Return diagnostic variance that must survive the formal covariance cap."""
        config = getattr(self.decomposer, "config", None)
        preserve_diagnostic = False
        preserve_guard = False
        if config is not None:
            preserve_diagnostic = bool(
                getattr(
                    config,
                    "response_poisson_count_variance_preserve_diagnostic_floors",
                    False,
                )
            )
            preserve_guard = bool(
                getattr(
                    config,
                    "response_poisson_count_variance_preserve_guard_floors",
                    False,
                )
            )
        preserved = 0.0
        isotope_key = str(isotope)
        if preserve_diagnostic:
            for floor_key in (
                "runtime_diagnostic_variance_floor",
                "runtime_shield_systematic_variance_floor",
            ):
                floor_map = diagnostics.get(floor_key, {})
                if not isinstance(floor_map, Mapping):
                    continue
                value = self._payload_float(floor_map, isotope_key)
                if value is not None:
                    preserved = max(preserved, value)
        if preserve_guard:
            guard_map = diagnostics.get("crosstalk_count_guard", {})
            if isinstance(guard_map, Mapping):
                guard_payload = guard_map.get(isotope_key, {})
                if isinstance(guard_payload, Mapping):
                    preserved = max(
                        preserved,
                        self._guard_payload_variance_floor(guard_payload),
                    )
            low_snr_map = diagnostics.get(
                "low_snr_photopeak_suppression",
                diagnostics.get("low_snr_suppression", {}),
            )
            if isinstance(low_snr_map, Mapping):
                low_snr_payload = low_snr_map.get(isotope_key, {})
                if isinstance(low_snr_payload, Mapping):
                    preserved = max(
                        preserved,
                        self._low_snr_payload_variance_floor(
                            low_snr_payload,
                            base_variance=base_variance,
                        ),
                    )
        return float(max(preserved, 0.0))

    def _guard_payload_variance_floor(self, payload: Mapping[str, object]) -> float:
        """Return the variance implied by a response/photopeak guard payload."""
        preserved = 0.0
        guarded_variance = self._payload_float(payload, "guarded_variance")
        if guarded_variance is not None:
            preserved = max(preserved, guarded_variance)
        preserved = max(preserved, self._count_disagreement_variance_floor(payload))
        return float(max(preserved, 0.0))

    def _low_snr_payload_variance_floor(
        self,
        payload: Mapping[str, object],
        *,
        base_variance: float,
    ) -> float:
        """Return variance that preserves low-SNR threshold uncertainty."""
        reason = str(payload.get("reason", ""))
        if not reason:
            return 0.0
        return float(
            max(
                self._count_disagreement_variance_floor(payload),
                float(base_variance),
                0.0,
            )
        )

    def _count_disagreement_variance_floor(
        self,
        payload: Mapping[str, object],
    ) -> float:
        """Return a count variance floor from response/photopeak disagreement."""
        reason = str(payload.get("reason", ""))
        if not reason:
            return 0.0
        response_count = self._payload_float(payload, "poisson_count")
        photopeak_count = self._payload_float(payload, "photopeak_count")
        if photopeak_count is None:
            photopeak_count = self._payload_float(payload, "photo_count")
        if response_count is None or photopeak_count is None:
            return 0.0
        return float(max((response_count - photopeak_count) ** 2, 0.0))

    def _diagnostic_relative_sigma(self, diagnostics: Mapping[str, object]) -> float:
        """Return a global relative sigma implied by response-fit diagnostics."""
        config = self.decomposer.config
        # A full-spectrum goodness-of-fit statistic is not an isotope-count
        # calibration.  Applying it to every isotope made one continuum/tail
        # mismatch flatten all PF channels.  Keep this legacy diagnostic path
        # opt-in until an independent Geant4 calibration establishes a mapping
        # from global fit residuals to isotope-count error.
        if not bool(
            getattr(
                config,
                "response_poisson_global_diagnostic_variance_enable",
                False,
            )
        ):
            return 0.0
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
                # A retained Poisson estimate was not suppressed. Its fused
                # variance already entered the formal covariance before the
                # ceiling, so treating any diagnostic reason as another 50%
                # floor would count the same heuristic twice.
                if bool(payload.get("suppressed", False)):
                    rel_by_isotope[str(isotope)] = max(
                        rel_by_isotope.get(str(isotope), 0.0),
                        0.5,
                    )
        guard_map = diagnostics.get("crosstalk_count_guard", {})
        if isinstance(guard_map, Mapping):
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
            max_rel = max(
                float(
                    getattr(
                        config,
                        "response_poisson_crosstalk_max_rel_sigma",
                        1.0,
                    )
                ),
                min_rel,
            )
            for isotope, payload in guard_map.items():
                if not isinstance(payload, Mapping):
                    continue
                guard_rel = self._guard_relative_sigma(
                    payload,
                    min_rel=min_rel,
                    scale=scale,
                    max_rel=max_rel,
                )
                if guard_rel <= 0.0:
                    continue
                rel_by_isotope[str(isotope)] = max(
                    rel_by_isotope.get(str(isotope), 0.0),
                    guard_rel,
                )
        return rel_by_isotope

    def _guard_relative_sigma(
        self,
        payload: Mapping[str, object],
        *,
        min_rel: float,
        scale: float,
        max_rel: float,
    ) -> float:
        """Return variance inflation implied by photopeak/response disagreement."""
        reason = str(payload.get("reason", ""))
        if not reason:
            return 0.0
        disagreement = self._payload_disagreement_fraction(payload)
        if disagreement <= 0.0:
            return 0.0
        evidence = max(
            self._payload_float(payload, "blend_weight") or 0.0,
            self._payload_float(payload, "combined_crosstalk_weight") or 0.0,
            self._payload_float(payload, "high_chi2_ratio_boost") or 0.0,
            self._payload_float(payload, "dominance_blend_weight") or 0.0,
            self._payload_float(payload, "chi2_mismatch_weight") or 0.0,
            self._payload_float(payload, "dominance_weight") or 0.0,
        )
        if not bool(payload.get("adjust_count", False)):
            evidence = max(evidence, 0.5)
        elif evidence <= 0.0:
            evidence = 1.0
        rel_sigma = max(float(min_rel), float(scale) * disagreement * evidence)
        return float(min(max(rel_sigma, 0.0), max(float(max_rel), 0.0)))

    def _payload_disagreement_fraction(
        self,
        payload: Mapping[str, object],
    ) -> float:
        """Return the response-photopeak count disagreement fraction."""
        direct = self._payload_float(payload, "disagreement_fraction")
        if direct is not None and direct >= 0.0:
            return float(direct)
        response_count = self._payload_float(payload, "poisson_count")
        photopeak_count = self._payload_float(payload, "photopeak_count")
        if response_count is None or photopeak_count is None:
            return 0.0
        reference = max(abs(response_count), abs(photopeak_count), 1.0)
        return float(abs(response_count - photopeak_count) / reference)

    @staticmethod
    def _payload_float(payload: Mapping[str, object], key: str) -> float | None:
        """Return a finite payload value as float when available."""
        value = payload.get(key)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result):
            return None
        return result

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
