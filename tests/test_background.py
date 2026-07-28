"""Background-shape tests for the sole raw full-spectrum contract."""

import numpy as np

from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_ENERGY_MAX_KEV,
    NATIVE_GEANT4_ENERGY_MIN_KEV,
    default_background_shape,
    native_geant4_background_shape,
)


def test_background_shape_basic() -> None:
    """The diagnostic background shape is normalized and low-energy weighted."""
    energy_axis = np.linspace(0.0, 1500.0, 1501)
    bg = default_background_shape(energy_axis)
    assert bg.shape == energy_axis.shape
    assert float(bg[energy_axis < 30.0].max()) == 0.0

    low_band = bg[(energy_axis >= 80.0) & (energy_axis <= 200.0)].mean()
    high_band = bg[(energy_axis >= 800.0) & (energy_axis <= 1500.0)].mean()
    assert low_band > high_band


def test_native_background_matches_the_production_axis() -> None:
    """The native background law is a probability vector on the exact axis."""
    axis = np.arange(
        NATIVE_GEANT4_ENERGY_MIN_KEV,
        NATIVE_GEANT4_ENERGY_MAX_KEV + NATIVE_GEANT4_BIN_WIDTH_KEV,
        NATIVE_GEANT4_BIN_WIDTH_KEV,
        dtype=np.float64,
    )
    shape = native_geant4_background_shape(
        axis,
        NATIVE_GEANT4_BIN_WIDTH_KEV,
    )

    assert shape.dtype == np.float64
    assert shape.shape == axis.shape
    assert np.all(shape >= 0.0)
    assert np.isclose(float(np.sum(shape)), 1.0)
