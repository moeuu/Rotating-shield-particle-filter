"""Fail-closed checks for removed runtime configuration generations."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import textwrap

import pytest

from measurement.obstacles import ObstacleGrid
from planning.dss_pp import DSSPPConfig
import realtime_demo
from realtime_demo import (
    _full_git_commit,
    _optional_runtime_bool,
    _pf_obstacle_attenuation_enabled,
    _pf_obstacle_grid_for_runtime,
    _resolve_candidate_isotopes,
    _resolve_detector_height_planning_config,
    _resolve_pf_strength_prior_bounds,
    _resolve_random_source_isotopes,
    _runtime_bool,
    _runtime_float,
    _strict_json_integer,
    _strict_json_number,
    _strict_json_string,
    _transport_detector_budget_radius_m,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
OBSOLETE_ADAPTIVE_KEYS = frozenset(
    {
        "adaptive_allow_low_signal_stop",
        "adaptive_cardinality_min_live_s",
        "adaptive_low_signal_count_fraction",
        "adaptive_low_signal_min_live_s",
        "adaptive_low_signal_projected_live_factor",
        "adaptive_low_signal_upper_sigma",
        "adaptive_ready_allow_informative_low",
        "apply_incident_gamma_detector_response",
        "coverage_grid_max_cells",
        "candidate_distance_relaxation_factor",
        "candidate_max_distance_retries",
        "candidate_min_horizontal_extent_fraction",
        "candidate_min_unique_xy",
        "candidate_xy_merge_tolerance_m",
        "response_backscatter_fraction",
        "response_continuum_to_peak",
        "response_efficiency_model",
    }
)


def test_runtime_configs_deny_removed_adaptive_count_settings() -> None:
    """No runtime config may resurrect the removed count/dwell controller."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        present = sorted(OBSOLETE_ADAPTIVE_KEYS.intersection(payload))
        if present:
            offenders[path.relative_to(ROOT).as_posix()] = present

    assert offenders == {}


def test_runtime_configs_do_not_pin_noop_scatter_gain() -> None:
    """A zero-valued visualization scatter default must not mimic physics."""
    offenders: list[str] = []
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            "scatter_gain" in payload
            and float(payload["scatter_gain"]) == 0.0
        ):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_standard_exact_eig_predictive_schedule_is_bounded() -> None:
    """The only per-action predictive scheduling must remain at most 32."""
    path = (
        CONFIG_ROOT
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    dss_config = payload["dss_pp"]

    assert int(dss_config["exact_eig_action_limit"]) <= 32
    assert int(dss_config["exact_eig_action_limit"]) > 0


def test_guarded_full_launcher_forces_the_canonical_full_mode() -> None:
    """The guarded launcher must not act as an alternate analytic entry point."""
    launcher = (ROOT / "scripts" / "run_guarded_full_sim.sh").read_text(
        encoding="utf-8"
    )

    assert 'MAIN_ARGS=(--full-simulation "${MAIN_ARGS[@]}")' in launcher
    assert 'has_arg "--full-simulation"' in launcher
    assert 'has_arg "--sim-config"' in launcher


@pytest.mark.parametrize(
    "value",
    (None, 0, 1, "false", "true", "disabled"),
)
def test_pf_obstacle_attenuation_rejects_non_boolean_values(
    value: object,
) -> None:
    """Ambiguous values must not silently change the PF physics model."""
    with pytest.raises(
        ValueError,
        match="pf_obstacle_attenuation must be a boolean",
    ):
        _pf_obstacle_attenuation_enabled(
            {"pf_obstacle_attenuation": value}
        )


def test_pf_cannot_drop_active_environment_obstacles() -> None:
    """Geant4 obstacles must remain present in every PF hypothesis."""
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )

    with pytest.raises(
        ValueError,
        match="invalid when physical environment obstacles are active",
    ):
        _pf_obstacle_grid_for_runtime(
            obstacle_grid,
            {"pf_obstacle_attenuation": False},
        )


@pytest.mark.parametrize("value", (0, 1, "false", "true"))
def test_optional_runtime_boole_reject_type_coercion(value: object) -> None:
    """Scientific runtime switches must use explicit JSON booleans."""
    with pytest.raises(ValueError, match="must be a boolean"):
        _optional_runtime_bool({"use_gpu": value}, "use_gpu")


@pytest.mark.parametrize("value", (0, 1, "false", "true", None))
def test_general_runtime_boole_reject_truthy_coercion(value: object) -> None:
    """Mission and planning switches must use exact JSON booleans."""
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        _runtime_bool({"adaptive_mission_stop": value}, "adaptive_mission_stop", False)


@pytest.mark.parametrize("value", (True, 1.0, "4", 0, -1))
def test_adaptive_stop_rejects_coerced_minimum_pose_count(
    value: object,
) -> None:
    """The stop gate must not reinterpret an invalid minimum mission length."""
    with pytest.raises(ValueError, match="min_poses"):
        realtime_demo._adaptive_mission_stop_reason(
            object(),  # type: ignore[arg-type]
            visited_poses_xyz=(),
            min_poses=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", (0, 1, "false", "true", None))
def test_adaptive_stop_rejects_truthy_cardinality_switch(
    value: object,
) -> None:
    """The stop gate must require an exact boolean cardinality contract."""
    with pytest.raises(ValueError, match="JSON boolean"):
        realtime_demo._adaptive_mission_stop_reason(
            object(),  # type: ignore[arg-type]
            visited_poses_xyz=(),
            min_poses=1,
            require_pf_cardinality_ready=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_runtime_integer_contract_rejects_numeric_coercion(value: object) -> None:
    """Discrete scientific settings must be represented as JSON integers."""
    with pytest.raises(ValueError, match="must be a JSON integer"):
        _strict_json_integer(value, name="orientation_k", minimum=1)


@pytest.mark.parametrize("value", (True, "0.5", None, float("nan"), float("inf")))
def test_runtime_number_contract_rejects_non_json_finite_values(
    value: object,
) -> None:
    """Continuous scientific settings must be finite JSON numbers."""
    with pytest.raises(ValueError, match="must be|finite"):
        _strict_json_number(value, name="coverage_weight", minimum=0.0)


@pytest.mark.parametrize("value", (True, 1, None, " "))
def test_runtime_string_contract_rejects_stringification(
    value: object,
) -> None:
    """Identifiers and enum values must be represented as nonempty strings."""
    with pytest.raises(ValueError, match="JSON string"):
        _strict_json_string(value, name="path_planner")


def test_runtime_float_rejects_numeric_strings() -> None:
    """Physics scalars must not acquire a second string-valued contract."""
    with pytest.raises(ValueError, match="JSON number"):
        _runtime_float({"primary_sampling_fraction": "1.0"}, "primary_sampling_fraction", 1.0)


def test_scientific_trace_rejects_station_pending_posterior() -> None:
    """A stale view-level posterior must never enter the official trace."""
    with pytest.raises(ValueError, match="after the joint station PF update"):
        realtime_demo._emit_intermediate_estimate_trace(
            object(),
            (),
            {},
            step_index=0,
            elapsed_s=0.0,
            trace_path=None,
            log_enabled=False,
            log_every=1,
            max_log_records=1,
            estimate_source="current_pf_posterior",
        )


@pytest.mark.parametrize(
    "resolver",
    (_resolve_candidate_isotopes, _resolve_random_source_isotopes),
)
def test_isotope_lists_reject_non_string_entries(resolver: object) -> None:
    """A malformed isotope list must not be stringified into a model key."""
    if resolver is _resolve_candidate_isotopes:
        with pytest.raises(TypeError, match="JSON strings"):
            resolver({"candidate_isotopes": ["Cs-137", 137]}, ("Cs-137",))
    else:
        with pytest.raises(TypeError, match="JSON strings"):
            resolver(
                None,
                {"random_source_isotopes": ["Cs-137", 137]},
                ("Cs-137",),
            )


def test_strength_prior_rejects_numeric_strings() -> None:
    """State-space bounds must remain exact JSON numbers."""
    with pytest.raises(ValueError, match="JSON number"):
        _resolve_pf_strength_prior_bounds(
            {
                "pf_strength_prior_min_cps_1m": "300000",
                "pf_strength_prior_max_cps_1m": 2_000_000.0,
            }
        )


def test_detector_model_rejects_non_object_payload() -> None:
    """Invalid detector geometry must not fall back to a default crystal."""
    with pytest.raises(TypeError, match="detector_model"):
        _transport_detector_budget_radius_m({"detector_model": []})


@pytest.mark.parametrize("value", (1234567890123456789012345678901234567890, None))
def test_repository_commit_identity_requires_a_string(value: object) -> None:
    """Resume provenance must not stringify external manifest values."""
    assert not _full_git_commit(value)


@pytest.mark.parametrize("mode", ("discrete", "sobol", "continuous_sobol", None))
def test_runtime_rejects_retired_detector_height_modes(
    mode: object,
) -> None:
    """Production planning must sample the full continuous height interval."""
    with pytest.raises(ValueError, match="continuous"):
        _resolve_detector_height_planning_config(
            {"detector_height_sampling_mode": mode},
            room_height_m=10.0,
        )


@pytest.mark.parametrize(
    "payload",
    (
        {
            "detector_height_sampling_mode": "continuous",
            "detector_height_actions_m": [0.5],
        },
        {
            "detector_height_sampling_mode": "continuous",
            "detector_heights_m": [0.5],
        },
    ),
)
def test_runtime_rejects_discrete_detector_height_keys(
    payload: dict[str, object],
) -> None:
    """Retired discrete action arrays must not narrow the 3-D search."""
    with pytest.raises(ValueError, match="Discrete detector-height"):
        _resolve_detector_height_planning_config(
            payload,
            room_height_m=10.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("robot_ground_z_m", -1.0),
        ("detector_height_min_m", -1.0),
        ("detector_height_max_m", 11.0),
        ("detector_height_m", "0.5"),
    ),
)
def test_detector_height_bounds_fail_instead_of_clamping(
    field: str,
    value: object,
) -> None:
    """Malformed mast bounds must stop instead of changing the search volume."""
    with pytest.raises(ValueError):
        _resolve_detector_height_planning_config(
            {
                "detector_height_sampling_mode": "continuous",
                field: value,
            },
            room_height_m=10.0,
        )


def test_live_runtime_has_no_direct_config_type_coercion() -> None:
    """Runtime config values must pass strict parsers before type conversion."""
    tree = ast.parse(Path(realtime_demo.__file__).read_text(encoding="utf-8"))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in {
            "bool",
            "int",
            "float",
            "str",
        }:
            continue
        has_runtime_get = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "runtime_config"
            for child in ast.walk(node)
        )
        if has_runtime_get:
            offenders.append((node.lineno, ast.unparse(node)))

    assert offenders == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_programs", 0),
        ("program_length", True),
        ("live_time_s", 0.0),
        ("lambda_coverage", -1.0),
        ("augment_candidates", "false"),
        ("coverage_floor_quantile", 1.1),
        ("planning_particles", 1),
    ),
)
def test_dss_configuration_fails_instead_of_clamping(
    field: str,
    value: object,
) -> None:
    """Invalid planning settings must stop before they bias measurement geometry."""
    with pytest.raises(ValueError):
        DSSPPConfig(**{field: value})


def test_every_structural_rj_runtime_setting_is_wired_to_pf_config() -> None:
    """A declared exact-RJ setting must not be accepted and then ignored."""
    source = textwrap.dedent(inspect.getsource(realtime_demo.run_live_pf))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RotatingShieldPFConfig"
    ]
    assert len(calls) == 1
    configured_keywords = {
        keyword.arg for keyword in calls[0].keywords if keyword.arg is not None
    }
    required = {
        "structural_rj_surface_chart_max_edge_m",
        "structural_rj_move_probability",
        "structural_rj_birth_probability",
        "structural_rj_death_probability",
        "structural_rj_position_move_probability",
        "structural_rj_position_proposal_prior_weight",
        "structural_rj_strength_proposal_prior_weight",
        "structural_rj_strength_proposal_sigma_fraction",
        "structural_rj_strength_proposal_grid_size",
        "structural_rj_proposal_chart_batch_size",
        "structural_rj_proposal_score_cache_max_bytes",
        "structural_rj_local_position_move_probability",
        "structural_rj_local_position_sigma_m",
        "structural_rj_strength_move_probability",
        "structural_rj_split_merge_probability",
        "structural_rj_split_probability",
        "structural_rj_merge_probability",
        "structural_cardinality_prior_policy",
        "structural_cardinality_prior_probs",
        "structural_cardinality_prior_mean",
    }

    assert required <= configured_keywords
    for key in required:
        assert f'"{key}"' in source
