"""Tests for known Manchester-style obstacle assets."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacle_assets import (
    KnownObstacleInstance,
    ObstacleComponent,
    _aluminum_equipment_frame_components,
    _concrete_jersey_barrier_components,
    _pipe_rack_components,
    _steel_cabinet_components,
    _water_drum_pair_components,
    environment_transport_model,
    generate_manchester_obstacle_instances,
    known_obstacle_line_transport_model,
    known_obstacle_transport_model,
    known_obstacle_traversability_rects,
    material_mu_cm_inv,
    obstacle_instances_from_dicts,
    obstacle_instances_to_dicts,
    room_boundary_transport_components,
)
from measurement.obstacles import ObstacleGrid


def _box_overlap_volume_m3(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> float:
    """Return positive overlap volume for two axis-aligned boxes."""
    dx = max(0.0, min(first[3], second[3]) - max(first[0], second[0]))
    dy = max(0.0, min(first[4], second[4]) - max(first[1], second[1]))
    dz = max(0.0, min(first[5], second[5]) - max(first[2], second[2]))
    return float(dx * dy * dz)


def test_manchester_assets_provide_hollow_transport_components() -> None:
    """Generated obstacle assets should separate motion footprints and transport boxes."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(2, 2),
        blocked_cells=((0, 0), (1, 1)),
    )
    instances = generate_manchester_obstacle_instances(
        grid,
        room_size_xyz=(2.0, 2.0, 3.0),
        obstacle_height_m=2.0,
        rng_seed=4,
    )
    boxes_m, mu_by_isotope = known_obstacle_transport_model(instances)
    rects = known_obstacle_traversability_rects(instances)

    assert len(instances) == len(grid.blocked_cells)
    assert len(boxes_m) > len(grid.blocked_cells)
    assert len(rects) == len(grid.blocked_cells)
    assert set(mu_by_isotope) >= {"Cs-137", "Co-60", "Eu-154"}
    assert len(mu_by_isotope["Cs-137"]) == len(boxes_m)


def _valid_obstacle_instance_payload() -> dict[str, object]:
    """Return one complete serialized known-obstacle payload."""
    instance = KnownObstacleInstance(
        name="Obstacle_0",
        template="test_box",
        footprint_xy=(0.0, 1.0, 0.0, 1.0),
        footprint_cells=((0, 0),),
        components=(
            ObstacleComponent(
                name="solid",
                center_xyz=(0.5, 0.5, 0.5),
                size_xyz=(0.5, 0.5, 1.0),
                material="concrete",
            ),
        ),
    )
    return obstacle_instances_to_dicts((instance,))[0]


def test_known_obstacle_manifest_roundtrip_is_exact() -> None:
    """Known obstacle manifests should preserve collision and transport geometry."""
    payload = _valid_obstacle_instance_payload()

    instances = obstacle_instances_from_dicts([payload])

    assert obstacle_instances_to_dicts(instances) == [payload]


@pytest.mark.parametrize(
    ("path", "invalid"),
    (
        (("name",), 1),
        (("template",), ""),
        (("footprint_xy",), ("0.0", 1.0, 0.0, 1.0)),
        (("footprint_xy",), (0.0, 0.0, 0.0, 1.0)),
        (("footprint_cells",), []),
        (("footprint_cells",), [[0.0, 0]]),
        (("components",), []),
        (("components", 0, "name"), 1),
        (("components", 0, "center_xyz"), [0.5, float("nan"), 0.5]),
        (("components", 0, "size_xyz"), [0.0, 0.5, 1.0]),
        (("components", 0, "material"), True),
    ),
)
def test_known_obstacle_manifest_rejects_fail_open_values(
    path: tuple[object, ...],
    invalid: object,
) -> None:
    """Malformed instance geometry must not suppress all authored obstacles."""
    payload = _valid_obstacle_instance_payload()
    target: object = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid

    with pytest.raises(ValueError):
        obstacle_instances_from_dicts([payload])


def test_known_obstacle_manifest_rejects_unknown_fields_and_names() -> None:
    """Manifest aliases and duplicate prim paths must fail before scene authoring."""
    payload = _valid_obstacle_instance_payload()
    unknown = dict(payload)
    unknown["component"] = unknown.pop("components")
    with pytest.raises(ValueError, match="schema mismatch"):
        obstacle_instances_from_dicts([unknown])

    duplicate = _valid_obstacle_instance_payload()
    with pytest.raises(ValueError, match="names must be unique"):
        obstacle_instances_from_dicts([payload, duplicate])


def test_known_obstacle_component_must_stay_inside_footprint() -> None:
    """Planner footprint and transport geometry must describe the same object."""
    payload = _valid_obstacle_instance_payload()
    payload["components"][0]["center_xyz"] = [1.0, 0.5, 0.5]

    with pytest.raises(ValueError, match="outside footprint"):
        obstacle_instances_from_dicts([payload])


def test_manchester_assets_provide_line_transport_components() -> None:
    """Generated obstacle assets should expose gamma-line transport rows."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    instances = generate_manchester_obstacle_instances(
        grid,
        room_size_xyz=(1.0, 1.0, 3.0),
        obstacle_height_m=2.0,
        rng_seed=5,
    )
    boxes_m, mu_by_isotope = known_obstacle_transport_model(instances)
    line_mu_by_isotope = known_obstacle_line_transport_model(instances)
    grid_with_transport = grid.with_transport_model(
        boxes_m=boxes_m,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope=line_mu_by_isotope,
    )

    assert set(line_mu_by_isotope) >= {"Cs-137", "Co-60", "Eu-154"}
    assert len(line_mu_by_isotope["Cs-137"]) >= 1
    assert len(line_mu_by_isotope["Cs-137"][0]) == len(boxes_m)
    assert grid_with_transport.transport_line_mu_values("Cs-137") is not None


def test_room_boundary_transport_components_match_authored_room() -> None:
    """Room boundary transport should provide floor, four walls, and ceiling boxes."""
    components = room_boundary_transport_components(
        (10.0, 20.0, 10.0),
        thickness_m=0.1,
    )
    boxes = tuple(component.box_m for component in components)

    assert len(components) == 6
    assert any(box[2] < 0.0 and box[5] <= 0.0 for box in boxes)
    assert any(box[2] >= 10.0 and box[5] > 10.0 for box in boxes)
    assert any(box[1] < 0.0 and box[4] <= 0.0 for box in boxes)
    assert any(box[1] >= 20.0 and box[4] > 20.0 for box in boxes)
    assert any(box[0] < 0.0 and box[3] <= 0.0 for box in boxes)
    assert any(box[0] >= 10.0 and box[3] > 10.0 for box in boxes)


def test_environment_transport_model_can_include_room_boundaries() -> None:
    """Environment transport should append room boundaries to obstacle components."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    )
    instances = generate_manchester_obstacle_instances(
        grid,
        room_size_xyz=(1.0, 1.0, 3.0),
        obstacle_height_m=2.0,
        rng_seed=6,
    )
    obstacle_boxes, _ = known_obstacle_transport_model(instances)
    (
        boxes_m,
        mu_by_isotope,
        line_mu_by_isotope,
        line_compton_mu_by_isotope,
    ) = environment_transport_model(
        instances,
        room_size_xyz=(1.0, 1.0, 3.0),
        include_room_boundaries=True,
    )

    assert len(boxes_m) == len(obstacle_boxes) + 6
    assert set(mu_by_isotope) >= {"Cs-137", "Co-60", "Eu-154"}
    assert len(mu_by_isotope["Cs-137"]) == len(boxes_m)
    assert len(line_mu_by_isotope["Cs-137"][0]) == len(boxes_m)
    assert len(line_compton_mu_by_isotope["Cs-137"][0]) == len(boxes_m)
    assert np.all(
        np.asarray(line_compton_mu_by_isotope["Cs-137"])
        <= np.asarray(line_mu_by_isotope["Cs-137"]) * (1.0 + 1.0e-12)
    )


def test_known_obstacle_templates_have_non_overlapping_transport_boxes() -> None:
    """Known obstacle components should not double-count the same material volume."""
    factories = (
        _steel_cabinet_components,
        _pipe_rack_components,
        _water_drum_pair_components,
        _concrete_jersey_barrier_components,
        _aluminum_equipment_frame_components,
    )
    bounds_xy = (0.0, 1.0, 0.0, 1.0)
    rng = np.random.default_rng(17)

    for factory in factories:
        components, template_name = factory(
            name_prefix="ObstacleTest",
            bounds_xy=bounds_xy,
            max_height_m=2.0,
            rng=rng,
        )
        boxes = [component.box_m for component in components]
        for first_index, first_box in enumerate(boxes):
            for second_index, second_box in enumerate(
                boxes[first_index + 1 :],
                start=first_index + 1,
            ):
                assert _box_overlap_volume_m3(first_box, second_box) <= 1.0e-12, (
                    template_name,
                    first_index,
                    second_index,
                )


def test_material_mu_uses_known_material_presets() -> None:
    """Material attenuation coefficients should depend on known material identity."""
    steel_mu = material_mu_cm_inv("steel", "Cs-137")
    water_mu = material_mu_cm_inv("water", "Cs-137")

    assert steel_mu > water_mu
    assert steel_mu > 0.0
    assert water_mu > 0.0


def test_continuous_kernel_uses_known_transport_components() -> None:
    """PF expected-count attenuation should use per-component obstacle materials."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=((0, 0),),
    ).with_transport_model(
        boxes_m=((0.4, 0.0, 0.0, 0.6, 1.0, 1.0),),
        mu_by_isotope={"Cs-137": (0.5,)},
    )
    kernel = ContinuousKernel(obstacle_grid=grid)

    attenuation = kernel._obstacle_attenuation_factor(
        "Cs-137",
        np.asarray((0.5, -1.0, 0.5), dtype=float),
        np.asarray((0.5, 2.0, 0.5), dtype=float),
    )

    assert attenuation < 1.0
    expected = float(np.exp(-50.0))
    assert abs(attenuation / expected - 1.0) < 1.0e-6
