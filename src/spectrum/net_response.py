"""Calibrate spectrum-derived net counts onto the PF measurement scale."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class NetResponseCalibration:
    """Store isotope-wise response factors from ideal counts to spectrum net counts."""

    scale_by_isotope: dict[str, float]
    fit_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def response_scale(self, isotope: str, default: float = 1.0) -> float:
        """Return the response scale for one isotope with a unity fallback."""
        value = self.scale_by_isotope.get(isotope, default)
        return max(float(value), 0.0)

    def apply_expected_counts(self, counts: Mapping[str, float]) -> dict[str, float]:
        """Map ideal inverse-square/shield counts into calibrated net-count space."""
        return {
            str(isotope): float(value) * self.response_scale(str(isotope))
            for isotope, value in counts.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration payload."""
        return {
            "measurement_space": "spectrum_net_counts",
            "model": "net_counts = response_scale[isotope] * ideal_inverse_square_shield_counts",
            "scale_by_isotope": {
                str(isotope): float(scale) for isotope, scale in self.scale_by_isotope.items()
            },
            "fit_statistics": self.fit_statistics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetResponseCalibration":
        """Build a calibration object from a JSON-like mapping."""
        scales_payload = payload.get("scale_by_isotope", payload.get("isotope_scales", {}))
        if not isinstance(scales_payload, Mapping):
            raise ValueError("scale_by_isotope must be a JSON object.")
        stats_payload = payload.get("fit_statistics", {})
        metadata_payload = payload.get("metadata", {})
        if not isinstance(stats_payload, Mapping):
            raise ValueError("fit_statistics must be a JSON object.")
        if not isinstance(metadata_payload, Mapping):
            raise ValueError("metadata must be a JSON object.")
        return cls(
            scale_by_isotope={str(key): float(value) for key, value in scales_payload.items()},
            fit_statistics={
                str(key): {str(k): float(v) for k, v in dict(value).items()}
                for key, value in stats_payload.items()
                if isinstance(value, Mapping)
            },
            metadata=dict(metadata_payload),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NetResponseCalibration":
        """Read a calibration JSON file."""
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("Calibration JSON root must be an object.")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        """Write the calibration JSON file with deterministic formatting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def fit_net_response_calibration(
    records: Iterable[Mapping[str, Any]],
    *,
    isotopes: Iterable[str] | None = None,
    min_theory_counts: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> NetResponseCalibration:
    """
    Fit isotope-wise net response factors by weighted through-origin regression.

    Each record must contain ``isotope``, ``theory_counts``, and ``net_counts``.
    A record may optionally provide ``weight``. Without an explicit weight, the
    fit uses inverse Poisson variance, ``1 / max(net_counts, 1)``.
    """
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    requested = {str(iso) for iso in isotopes} if isotopes is not None else None
    for record in records:
        isotope = str(record["isotope"])
        if requested is not None and isotope not in requested:
            continue
        theory = float(record["theory_counts"])
        net = float(record["net_counts"])
        if theory < float(min_theory_counts):
            continue
        weight = float(record.get("weight", 1.0 / max(net, 1.0)))
        if weight <= 0.0:
            continue
        grouped.setdefault(isotope, []).append((theory, net, weight))

    scale_by_isotope: dict[str, float] = {}
    fit_statistics: dict[str, dict[str, float]] = {}
    for isotope, values in grouped.items():
        x = np.asarray([entry[0] for entry in values], dtype=float)
        y = np.asarray([entry[1] for entry in values], dtype=float)
        w = np.asarray([entry[2] for entry in values], dtype=float)
        denom = float(np.sum(w * x * x))
        if denom <= 0.0:
            continue
        scale = float(np.sum(w * x * y) / denom)
        prediction = scale * x
        residual = y - prediction
        dof = max(int(x.size) - 1, 1)
        weighted_sse = float(np.sum(w * residual * residual))
        sigma2 = weighted_sse / float(dof)
        standard_error = float(np.sqrt(max(sigma2, 0.0) / denom))
        relative_error = np.divide(
            residual,
            np.maximum(np.abs(y), 1.0),
            out=np.zeros_like(residual),
            where=np.maximum(np.abs(y), 1.0) > 0.0,
        )
        scale_by_isotope[isotope] = max(scale, 0.0)
        fit_statistics[isotope] = {
            "num_fit_points": float(x.size),
            "scale": max(scale, 0.0),
            "standard_error": standard_error,
            "relative_standard_error": standard_error / max(abs(scale), 1e-12),
            "mean_relative_residual": float(np.mean(relative_error)) if relative_error.size else 0.0,
            "max_abs_relative_residual": float(np.max(np.abs(relative_error))) if relative_error.size else 0.0,
            "theory_counts_sum": float(np.sum(x)),
            "net_counts_sum": float(np.sum(y)),
        }

    return NetResponseCalibration(
        scale_by_isotope=scale_by_isotope,
        fit_statistics=fit_statistics,
        metadata=dict(metadata or {}),
    )
