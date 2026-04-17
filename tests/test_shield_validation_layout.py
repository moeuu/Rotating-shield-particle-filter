"""Tests for the Geant4 shield-validation source layout."""

import json
from pathlib import Path

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.shielding import HVL_TVL_TABLE_MM, mu_by_isotope_from_tvl_mm


def test_shield_validation_sources_are_blocked_by_octant_7() -> None:
    """The validation layout should make octant 7 a strong shielded condition."""
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "source_layouts" / "shield_validation.json").read_text())
    detector = np.array([1.0, 1.0, 0.5], dtype=float)
    isotopes = [str(entry["isotope"]) for entry in payload["sources"]]
    kernel = ContinuousKernel(
        mu_by_isotope=mu_by_isotope_from_tvl_mm(HVL_TVL_TABLE_MM, isotopes=isotopes),
        shield_params=ShieldParams(),
        use_gpu=False,
    )

    for entry in payload["sources"]:
        source_pos = np.asarray(entry["position"], dtype=float)
        unblocked = kernel.attenuation_factor_pair(
            str(entry["isotope"]),
            source_pos,
            detector,
            fe_index=0,
            pb_index=0,
        )
        blocked = kernel.attenuation_factor_pair(
            str(entry["isotope"]),
            source_pos,
            detector,
            fe_index=7,
            pb_index=7,
        )

        assert unblocked == pytest.approx(1.0)
        assert blocked < 0.1
