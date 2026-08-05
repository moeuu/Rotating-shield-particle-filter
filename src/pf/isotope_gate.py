"""Truth-free full-spectrum isotope activation for the exact-RJ PF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import NDArray


@dataclass
class FullSpectrumIsotopeGate:
    """Accumulate model-native evidence and activate candidate isotopes once."""

    candidate_isotopes: Sequence[str]
    false_activation_probability: float = 1.0e-3
    station_count: int = 0
    active_isotopes: set[str] = field(default_factory=set)
    _cumulative_score_grids: dict[str, NDArray[np.float64]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate the predeclared candidate set and family-wise error rate."""
        candidates = tuple(self.candidate_isotopes)
        if (
            not candidates
            or any(not isinstance(value, str) or not value for value in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("candidate_isotopes must be unique nonempty strings.")
        probability = float(self.false_activation_probability)
        if not np.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError(
                "false_activation_probability must lie strictly between zero and one."
            )
        self.candidate_isotopes = candidates
        self.false_activation_probability = probability

    def update(
        self,
        score_grids: Mapping[str, NDArray[np.float64]],
    ) -> dict[str, object]:
        """Add one station of full-spectrum scores and return gate diagnostics."""
        if set(score_grids) != set(self.candidate_isotopes):
            raise ValueError(
                "Detection score grids must cover every candidate isotope exactly."
            )
        validated: dict[str, NDArray[np.float64]] = {}
        for isotope in self.candidate_isotopes:
            values = np.asarray(score_grids[isotope], dtype=np.float64)
            if values.ndim != 2 or values.size == 0 or np.any(~np.isfinite(values)):
                raise ValueError(f"Detection score grid is invalid for {isotope!r}.")
            previous = self._cumulative_score_grids.get(isotope)
            if previous is not None and previous.shape != values.shape:
                raise ValueError(f"Detection score grid shape changed for {isotope!r}.")
            validated[isotope] = np.ascontiguousarray(values)

        self.station_count += 1
        for isotope, values in validated.items():
            previous = self._cumulative_score_grids.get(isotope)
            self._cumulative_score_grids[isotope] = (
                values.copy() if previous is None else previous + values
            )

        hypothesis_count = sum(
            int(values.size) for values in self._cumulative_score_grids.values()
        )
        sequential_probability = (
            self.false_activation_probability
            * 6.0
            / (math.pi**2 * float(self.station_count**2))
        )
        threshold = math.log(float(hypothesis_count) / sequential_probability)
        maximum_scores = {
            isotope: float(np.max(self._cumulative_score_grids[isotope]))
            for isotope in self.candidate_isotopes
        }
        newly_active = {
            isotope
            for isotope, score in maximum_scores.items()
            if score >= threshold and isotope not in self.active_isotopes
        }
        self.active_isotopes.update(newly_active)
        return {
            "station_count": int(self.station_count),
            "hypothesis_count": int(hypothesis_count),
            "familywise_false_activation_probability": float(
                self.false_activation_probability
            ),
            "sequential_false_activation_probability": float(sequential_probability),
            "activation_log_score_threshold": float(threshold),
            "cumulative_maximum_log_scores": maximum_scores,
            "newly_active_isotopes": sorted(newly_active),
            "active_isotopes": sorted(self.active_isotopes),
            "score_semantics": (
                "background_whitened_non_target_line_subspace_full_spectrum_"
                "candidate_log_score"
            ),
            "truth_used": False,
        }
