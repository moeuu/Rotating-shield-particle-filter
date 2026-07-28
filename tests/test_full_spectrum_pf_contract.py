"""Fail-closed contract tests for production full-spectrum PF observations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from pf.full_spectrum import (
    validate_full_spectrum_model,
    validate_observed_spectrum,
)
from pure_pf_test_support import approved_full_spectrum_model


def _station_contract_estimator() -> RotatingShieldPFEstimator:
    """Return a compact estimator with one registered detector pose."""
    estimator = RotatingShieldPFEstimator(
        isotopes=("Cs-137",),
        surface_diagnostic_points=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        shield_normals=None,
        mu_by_isotope={"Cs-137": 0.0},
        pf_config=RotatingShieldPFConfig(
            num_particles=2,
            max_sources=1,
            variable_cardinality=False,
            init_num_sources=(0, 0),
            use_gpu=False,
            position_max=(2.0, 2.0, 2.0),
        ),
        full_spectrum_generative_model=approved_full_spectrum_model(),
    )
    estimator.add_measurement_pose(
        np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    )
    return estimator


def _zero_spectrum_record(
    estimator: RotatingShieldPFEstimator,
    *,
    fe_index: object = 0,
    pb_index: object = 0,
) -> tuple[np.ndarray, object, object, float]:
    """Return one integer-valued zero-spectrum station record."""
    bin_count = int(
        np.asarray(
            estimator._full_spectrum_model().energy_axis_keV,
            dtype=np.float64,
        ).size
    )
    return (
        np.zeros(bin_count, dtype=np.float64),
        fe_index,
        pb_index,
        1.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"detector_aperture_samples": 0}, ValueError),
        ({"detector_aperture_samples": 1.5}, TypeError),
        ({"detector_aperture_samples": "5"}, TypeError),
        ({"use_gpu": 1}, TypeError),
        ({"use_gpu": "false"}, TypeError),
    ],
)
def test_shared_kernel_override_rejects_runtime_coercion(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Diagnostic kernel overrides must preserve the production contract."""
    estimator = _station_contract_estimator()
    with pytest.raises(error_type):
        estimator.continuous_kernel(**kwargs)


def test_observed_full_spectrum_accepts_unit_weight_integer_counts() -> None:
    """Raw histogram counts may be represented as float64 exact integers."""
    observed = np.asarray(
        [[0.0, 1.0, 7.0], [3.0, 0.0, 11.0]],
        dtype=np.float64,
    )

    validated = validate_observed_spectrum(
        observed,
        expected_bin_count=3,
    )

    np.testing.assert_array_equal(validated, observed)
    assert validated.flags.c_contiguous


def test_model_validator_rejects_truthy_string_production_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile protocol object cannot use ``"false"`` as approval."""
    model = approved_full_spectrum_model()
    monkeypatch.setattr(
        type(model),
        "production_ready",
        property(lambda _self: "false"),
    )

    with pytest.raises(RuntimeError, match="production_ready=False"):
        validate_full_spectrum_model(model)


def test_model_validator_rejects_non_string_contract_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numeric value must not be string-coerced into a contract digest."""
    model = approved_full_spectrum_model()
    monkeypatch.setattr(
        type(model),
        "contract_hash_sha256",
        property(lambda _self: 1),
    )
    monkeypatch.setattr(
        type(model),
        "production_ready",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        type(model),
        "require_production_ready",
        lambda _self: None,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        validate_full_spectrum_model(model)


def test_model_validator_rejects_coerced_feature_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric feature IDs cannot be normalized through ``str``."""
    model = approved_full_spectrum_model()
    monkeypatch.setattr(
        type(model),
        "transport_feature_order",
        property(
            lambda _self: (
                "tau_fe",
                "tau_pb",
                "tau_obstacle",
                3,
            )
        ),
    )

    with pytest.raises(ValueError, match="canonical order"):
        validate_full_spectrum_model(model)


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([[0.0, 1.25, 2.0]], dtype=np.float64),
        np.asarray([[0.0, 0.5, 2.0]], dtype=np.float64),
        np.asarray([[0.0, float(2**53 + 2), 2.0]], dtype=np.float64),
    ],
)
def test_observed_full_spectrum_rejects_non_event_count_semantics(
    invalid: np.ndarray,
) -> None:
    """Weighted, corrected, fractional, or inexact counts must fail closed."""
    with pytest.raises(ValueError, match="unit-weight integer"):
        validate_observed_spectrum(
            invalid,
            expected_bin_count=3,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([["0", "1", "2"]]),
        np.asarray([[False, True, False]]),
        np.asarray([[0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j]]),
    ],
)
def test_observed_full_spectrum_rejects_numeric_coercion(
    invalid: np.ndarray,
) -> None:
    """Strings, booleans, and complex values cannot become event counts."""
    with pytest.raises(TypeError, match="JSON numbers"):
        validate_observed_spectrum(
            invalid,
            expected_bin_count=3,
        )


@pytest.mark.parametrize("invalid", [True, 3.0, "3"])
def test_observed_full_spectrum_rejects_coerced_bin_count(
    invalid: object,
) -> None:
    """The model bin count must remain an exact integer contract."""
    with pytest.raises(TypeError, match="expected_bin_count"):
        validate_observed_spectrum(
            np.zeros((1, 3), dtype=np.float64),
            expected_bin_count=invalid,  # type: ignore[arg-type]
        )


def test_birth_proposal_score_cache_is_immutable_and_memory_bounded() -> None:
    """The proposal cache may retain score grids, never unbounded spectra."""
    estimator = object.__new__(RotatingShieldPFEstimator)
    estimator.pf_config = SimpleNamespace(
        structural_rj_proposal_score_cache_max_bytes=64
    )
    estimator._joint_birth_proposal_station_score_cache = {}
    estimator._joint_birth_proposal_station_score_cache_order = []
    first_key = ("Cs-137", "a" * 64)
    second_key = ("Cs-137", "b" * 64)

    first = estimator._store_joint_birth_proposal_station_scores(
        first_key,
        np.arange(8, dtype=np.float64).reshape(2, 4),
    )
    assert first.flags.writeable is False
    assert estimator.joint_birth_proposal_cache_bytes == 64

    estimator._store_joint_birth_proposal_station_scores(
        second_key,
        np.arange(8, dtype=np.float64).reshape(4, 2),
    )
    assert first_key not in estimator._joint_birth_proposal_station_score_cache
    assert second_key in estimator._joint_birth_proposal_station_score_cache
    assert estimator.joint_birth_proposal_cache_bytes == 64

    with pytest.raises(MemoryError, match="score grid exceeds"):
        estimator._store_joint_birth_proposal_station_scores(
            ("Co-60", "c" * 64),
            np.zeros((3, 3), dtype=np.float64),
        )


@pytest.mark.parametrize("field", ["fe", "pb"])
@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_station_rejects_coerced_orientation_indices(
    field: str,
    invalid: object,
) -> None:
    """Corrupt posture indices must not be truncated into a physical view."""
    estimator = _station_contract_estimator()
    record = _zero_spectrum_record(
        estimator,
        fe_index=invalid if field == "fe" else 0,
        pb_index=invalid if field == "pb" else 0,
    )

    with pytest.raises(TypeError, match="must be an integer"):
        estimator._joint_station_from_spectrum_records(
            (record,),
            pose_idx=0,
            station_sequence_id=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


@pytest.mark.parametrize("invalid", [True, 0.5, "0"])
def test_station_rejects_coerced_pose_index(invalid: object) -> None:
    """A pose identifier must be an exact integer before array indexing."""
    estimator = _station_contract_estimator()
    record = _zero_spectrum_record(estimator)

    with pytest.raises(TypeError, match="pose_idx must be an integer"):
        estimator._joint_station_from_spectrum_records(
            (record,),
            pose_idx=invalid,
            station_sequence_id=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


@pytest.mark.parametrize("invalid", [True, "1.0"])
def test_station_rejects_coerced_live_time(invalid: object) -> None:
    """A truthy string or boolean cannot become physical acquisition time."""
    estimator = _station_contract_estimator()
    spectrum, fe_index, pb_index, _ = _zero_spectrum_record(estimator)

    with pytest.raises(TypeError, match="live_time_s must be numeric"):
        estimator._joint_station_from_spectrum_records(
            ((spectrum, fe_index, pb_index, invalid),),
            pose_idx=0,
            station_sequence_id=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    (
        ("wrong_rank", ValueError, "one-dimensional raw spectrum"),
        ("numeric_strings", TypeError, "JSON numbers"),
        ("nonfinite", ValueError, "finite"),
        ("fractional", ValueError, "unit-weight integer"),
    ),
)
def test_station_rejects_spectrum_coercion_before_assimilation(
    mutation: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Station ingestion must validate raw rank and count semantics before casting."""
    estimator = _station_contract_estimator()
    spectrum, fe_index, pb_index, live_time_s = _zero_spectrum_record(estimator)
    if mutation == "wrong_rank":
        invalid_spectrum = spectrum.reshape(1, -1)
    elif mutation == "numeric_strings":
        invalid_spectrum = spectrum.astype(str)
    else:
        invalid_spectrum = spectrum.copy()
        invalid_spectrum[0] = (
            np.nan if mutation == "nonfinite" else 0.5
        )

    with pytest.raises(error_type, match=message):
        estimator._joint_station_from_spectrum_records(
            ((invalid_spectrum, fe_index, pb_index, live_time_s),),
            pose_idx=0,
            station_sequence_id=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


def test_station_rejects_coerced_contract_hash() -> None:
    """A non-string object cannot authenticate a spectrum model contract."""
    estimator = _station_contract_estimator()

    with pytest.raises(TypeError, match="must be a JSON string"):
        estimator._joint_station_from_spectrum_records(
            (_zero_spectrum_record(estimator),),
            pose_idx=0,
            station_sequence_id=0,
            generative_contract_hash_sha256=123,
        )


@pytest.mark.parametrize("invalid", [-1, 1])
def test_station_rejects_unregistered_pose_index(invalid: int) -> None:
    """Negative and out-of-range poses must not wrap or read another pose."""
    estimator = _station_contract_estimator()
    record = _zero_spectrum_record(estimator)
    expected_exception = ValueError if invalid < 0 else IndexError

    with pytest.raises(expected_exception):
        estimator._joint_station_from_spectrum_records(
            (record,),
            pose_idx=invalid,
            station_sequence_id=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


def test_empty_station_fails_instead_of_skipping_posterior_update() -> None:
    """A causal station boundary with no views is a runtime contract error."""
    estimator = _station_contract_estimator()

    with pytest.raises(ValueError, match="at least one shield view"):
        estimator.update_spectrum_station(
            (),
            pose_idx=0,
            generative_contract_hash_sha256=(
                estimator._full_spectrum_model().contract_hash_sha256
            ),
        )


def test_eight_raw_views_enter_one_joint_station_update() -> None:
    """A shield program is assimilated once, not once per derived statistic."""
    estimator = _station_contract_estimator()
    captured: list[object] = []

    def _capture_station(station: object) -> None:
        """Record the sole station update without running numerical SMC."""
        captured.append(station)
        estimator._joint_station_history.append(station)

    estimator._joint_tempered_station_update = _capture_station  # type: ignore[method-assign]
    records = tuple(
        _zero_spectrum_record(
            estimator,
            fe_index=index // 8,
            pb_index=index % 8,
        )
        for index in range(8)
    )

    estimator.update_spectrum_station(
        records,
        pose_idx=0,
        generative_contract_hash_sha256=(
            estimator._full_spectrum_model().contract_hash_sha256
        ),
    )

    assert len(captured) == 1
    station = captured[0]
    assert station.spectrum_vb.shape[0] == 8
    assert len(estimator.measurements) == 8
    assert {
        record.station_sequence_id for record in estimator.measurements
    } == {0}
