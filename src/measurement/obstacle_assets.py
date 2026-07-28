"""Known composite obstacle assets for random Manchester-style scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Sequence

import numpy as np

from measurement.obstacles import ObstacleGrid
from sim.isaacsim_app.materials import (
    composition_mass_attenuation,
    composition_mass_attenuation_at_energy,
    normalize_material_name,
    resolve_material_preset,
)
from sim.transport import DEFAULT_MATERIAL_MU_CM_INV
from spectrum.additive_scatter import material_compton_mu_cm_inv_numpy
from spectrum.library import default_library


DEFAULT_TRANSPORT_ISOTOPES = ("Cs-137", "Co-60", "Eu-154")


@dataclass(frozen=True)
class ObstacleComponent:
    """Describe one transport-relevant box component of a known obstacle."""

    name: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    material: str

    def __post_init__(self) -> None:
        """Validate one physical component without implicit coercion."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Obstacle component name must be a nonempty string.")
        center = _as_float_tuple(
            self.center_xyz,
            3,
            "ObstacleComponent.center_xyz",
        )
        size = _as_float_tuple(
            self.size_xyz,
            3,
            "ObstacleComponent.size_xyz",
        )
        if any(value <= 0.0 for value in size):
            raise ValueError("Obstacle component size_xyz must be positive.")
        if not isinstance(self.material, str) or not self.material:
            raise ValueError("Obstacle component material must be a nonempty string.")
        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "size_xyz", size)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable component payload."""
        return {
            "name": self.name,
            "center_xyz": [float(value) for value in self.center_xyz],
            "size_xyz": [float(value) for value in self.size_xyz],
            "material": str(self.material),
        }

    @property
    def box_m(self) -> tuple[float, float, float, float, float, float]:
        """Return the component as an axis-aligned box."""
        center = np.asarray(self.center_xyz, dtype=float)
        size = np.asarray(self.size_xyz, dtype=float)
        lower = center - 0.5 * size
        upper = center + 0.5 * size
        return (
            float(lower[0]),
            float(lower[1]),
            float(lower[2]),
            float(upper[0]),
            float(upper[1]),
            float(upper[2]),
        )


@dataclass(frozen=True)
class KnownObstacleInstance:
    """Describe a composite obstacle with separate transport and motion models."""

    name: str
    template: str
    footprint_xy: tuple[float, float, float, float]
    footprint_cells: tuple[tuple[int, int], ...]
    components: tuple[ObstacleComponent, ...]

    def __post_init__(self) -> None:
        """Validate planning and transport geometry for one known obstacle."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Known obstacle name must be a nonempty string.")
        if not isinstance(self.template, str) or not self.template:
            raise ValueError("Known obstacle template must be a nonempty string.")
        footprint = _as_float_tuple(
            self.footprint_xy,
            4,
            "KnownObstacleInstance.footprint_xy",
        )
        if footprint[1] <= footprint[0] or footprint[3] <= footprint[2]:
            raise ValueError(
                "Known obstacle footprint_xy must have positive XY extent."
            )
        if not isinstance(self.footprint_cells, (list, tuple)):
            raise ValueError("Known obstacle footprint_cells must be a list or tuple.")
        cells: list[tuple[int, int]] = []
        for index, cell in enumerate(self.footprint_cells):
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError(
                    f"Known obstacle footprint_cells[{index}] must contain two "
                    "integers."
                )
            cells.append(
                (
                    _exact_integer(
                        cell[0],
                        field_name=f"footprint_cells[{index}][0]",
                    ),
                    _exact_integer(
                        cell[1],
                        field_name=f"footprint_cells[{index}][1]",
                    ),
                )
            )
        if not cells:
            raise ValueError("Known obstacle footprint_cells must not be empty.")
        if len(set(cells)) != len(cells):
            raise ValueError("Known obstacle footprint_cells must not contain duplicates.")
        if not isinstance(self.components, (list, tuple)) or not self.components:
            raise ValueError("Known obstacle components must not be empty.")
        components = tuple(self.components)
        if any(not isinstance(component, ObstacleComponent) for component in components):
            raise ValueError(
                "Known obstacle components must contain ObstacleComponent values."
            )
        component_names = [component.name for component in components]
        if len(set(component_names)) != len(component_names):
            raise ValueError("Known obstacle component names must be unique.")
        for component in components:
            box = component.box_m
            if (
                box[0] < footprint[0] - 1.0e-12
                or box[3] > footprint[1] + 1.0e-12
                or box[1] < footprint[2] - 1.0e-12
                or box[4] > footprint[3] + 1.0e-12
            ):
                raise ValueError(
                    "Known obstacle component lies outside footprint_xy."
                )
        object.__setattr__(self, "footprint_xy", footprint)
        object.__setattr__(self, "footprint_cells", tuple(cells))
        object.__setattr__(self, "components", components)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable known-obstacle payload."""
        return {
            "name": self.name,
            "template": self.template,
            "footprint_xy": [float(value) for value in self.footprint_xy],
            "footprint_cells": [list(cell) for cell in self.footprint_cells],
            "components": [component.to_dict() for component in self.components],
        }


def obstacle_instances_to_dicts(
    instances: Iterable[KnownObstacleInstance],
) -> list[dict[str, Any]]:
    """Return JSON-serializable payloads for known obstacle instances."""
    return [instance.to_dict() for instance in instances]


def obstacle_instances_from_dicts(
    payloads: Iterable[dict[str, Any]],
) -> tuple[KnownObstacleInstance, ...]:
    """Parse known obstacle instances from manifest dictionaries."""
    if not isinstance(payloads, (list, tuple)):
        raise ValueError("obstacle_instances must be a list or tuple.")
    instances: list[KnownObstacleInstance] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ValueError(f"obstacle_instances[{index}] must be a mapping.")
        required = {
            "name",
            "template",
            "footprint_xy",
            "footprint_cells",
            "components",
        }
        if set(payload) != required:
            raise ValueError(
                f"obstacle_instances[{index}] schema mismatch: "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}."
            )
        raw_components = payload["components"]
        if not isinstance(raw_components, list):
            raise ValueError(
                f"obstacle_instances[{index}].components must be a list."
            )
        components: list[ObstacleComponent] = []
        for comp_index, comp in enumerate(raw_components):
            if not isinstance(comp, dict):
                raise ValueError(
                    f"obstacle_instances[{index}].components[{comp_index}] "
                    "must be a mapping."
                )
            component_fields = {"name", "center_xyz", "size_xyz", "material"}
            if set(comp) != component_fields:
                raise ValueError(
                    f"obstacle_instances[{index}].components[{comp_index}] "
                    "schema mismatch."
                )
            center = _as_float_tuple(
                comp["center_xyz"],
                3,
                f"obstacle_instances[{index}].components[{comp_index}].center_xyz",
            )
            size = _as_float_tuple(
                comp["size_xyz"],
                3,
                f"obstacle_instances[{index}].components[{comp_index}].size_xyz",
            )
            components.append(
                ObstacleComponent(
                    name=_nonempty_string(
                        comp["name"],
                        field_name=(
                            f"obstacle_instances[{index}].components"
                            f"[{comp_index}].name"
                        ),
                    ),
                    center_xyz=(center[0], center[1], center[2]),
                    size_xyz=(size[0], size[1], size[2]),
                    material=_nonempty_string(
                        comp["material"],
                        field_name=(
                            f"obstacle_instances[{index}].components"
                            f"[{comp_index}].material"
                        ),
                    ),
                )
            )
        footprint = _as_float_tuple(
            payload["footprint_xy"],
            4,
            f"obstacle_instances[{index}].footprint_xy",
        )
        raw_cells = payload["footprint_cells"]
        if not isinstance(raw_cells, list):
            raise ValueError(
                f"obstacle_instances[{index}].footprint_cells must be a list."
            )
        parsed_cells: list[tuple[int, int]] = []
        for cell_index, cell in enumerate(raw_cells):
            _validate_cell_shape(
                cell,
                field_name=(
                    f"obstacle_instances[{index}].footprint_cells[{cell_index}]"
                ),
            )
            parsed_cells.append(
                (
                    _exact_integer(
                        cell[0],
                        field_name=(
                            f"obstacle_instances[{index}].footprint_cells"
                            f"[{cell_index}][0]"
                        ),
                    ),
                    _exact_integer(
                        cell[1],
                        field_name=(
                            f"obstacle_instances[{index}].footprint_cells"
                            f"[{cell_index}][1]"
                        ),
                    ),
                )
            )
        cells = tuple(parsed_cells)
        instances.append(
            KnownObstacleInstance(
                name=_nonempty_string(
                    payload["name"],
                    field_name=f"obstacle_instances[{index}].name",
                ),
                template=_nonempty_string(
                    payload["template"],
                    field_name=f"obstacle_instances[{index}].template",
                ),
                footprint_xy=(footprint[0], footprint[1], footprint[2], footprint[3]),
                footprint_cells=cells,
                components=tuple(components),
            )
        )
    names = [instance.name for instance in instances]
    if len(set(names)) != len(names):
        raise ValueError("obstacle_instances names must be unique.")
    return tuple(instances)


def known_obstacle_transport_model(
    instances: Iterable[KnownObstacleInstance],
    *,
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> tuple[
    tuple[tuple[float, float, float, float, float, float], ...],
    dict[str, tuple[float, ...]],
]:
    """Return component boxes and per-isotope linear attenuation values."""
    return transport_model_from_components(
        _components_from_instances(instances),
        isotopes=isotopes,
    )


def known_obstacle_line_transport_model(
    instances: Iterable[KnownObstacleInstance],
    *,
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """
    Return gamma-line-resolved obstacle attenuation values.

    The row order follows ``spectrum.library.default_library`` for each isotope
    and therefore matches ``line_resolved_shield_mu_by_isotope``.  Each row
    contains per-transport-box linear attenuation coefficients in 1/cm.
    """
    return line_transport_model_from_components(
        _components_from_instances(instances),
        isotopes=isotopes,
    )


def _components_from_instances(
    instances: Iterable[KnownObstacleInstance],
) -> tuple[ObstacleComponent, ...]:
    """Return transport components from known obstacle instances."""
    components: list[ObstacleComponent] = []
    for instance in instances:
        components.extend(instance.components)
    return tuple(components)


def transport_model_from_components(
    components: Iterable[ObstacleComponent],
    *,
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> tuple[
    tuple[tuple[float, float, float, float, float, float], ...],
    dict[str, tuple[float, ...]],
]:
    """Return boxes and isotope attenuation values for transport components."""
    component_tuple = tuple(components)
    boxes = tuple(component.box_m for component in component_tuple)
    component_materials = tuple(component.material for component in component_tuple)
    mu_by_isotope: dict[str, tuple[float, ...]] = {}
    for isotope in isotopes:
        mu_by_isotope[str(isotope)] = tuple(
            material_mu_cm_inv(material, str(isotope))
            for material in component_materials
        )
    return boxes, mu_by_isotope


def line_transport_model_from_components(
    components: Iterable[ObstacleComponent],
    *,
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Return gamma-line-resolved attenuation values for transport components."""
    component_tuple = tuple(components)
    component_materials = tuple(component.material for component in component_tuple)
    if not component_materials:
        return {}
    library = default_library()
    line_mu_by_isotope: dict[str, tuple[tuple[float, ...], ...]] = {}
    for isotope in isotopes:
        nuclide = _lookup_nuclide(library, str(isotope))
        if nuclide is None:
            continue
        rows: list[tuple[float, ...]] = []
        for line in nuclide.lines:
            if max(float(line.intensity), 0.0) <= 0.0:
                continue
            rows.append(
                tuple(
                    material_mu_cm_inv_at_energy(
                        material,
                        float(line.energy_keV),
                        isotope=str(isotope),
                    )
                    for material in component_materials
                )
            )
        if rows:
            line_mu_by_isotope[str(isotope)] = tuple(rows)
    return line_mu_by_isotope


def line_compton_transport_model_from_components(
    components: Iterable[ObstacleComponent],
    *,
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Return per-line, per-component physical Compton attenuation values."""
    component_tuple = tuple(components)
    if not component_tuple:
        return {}
    material_presets = []
    for component in component_tuple:
        normalized = normalize_material_name(component.material)
        preset = resolve_material_preset(normalized)
        if (
            preset is None
            or preset.density_g_cm3 is None
            or not preset.composition_by_mass
        ):
            raise ValueError(
                "Obstacle scatter requires a known material composition for "
                f"{component.material!r}."
            )
        material_presets.append(preset)
    library = default_library()
    by_isotope: dict[str, tuple[tuple[float, ...], ...]] = {}
    for isotope in isotopes:
        nuclide = _lookup_nuclide(library, str(isotope))
        if nuclide is None:
            continue
        rows: list[tuple[float, ...]] = []
        for line in nuclide.lines:
            if max(float(line.intensity), 0.0) <= 0.0:
                continue
            rows.append(
                tuple(
                    float(
                        material_compton_mu_cm_inv_numpy(
                            float(line.energy_keV),
                            density_g_cm3=float(preset.density_g_cm3),
                            composition_by_mass=preset.composition_by_mass,
                        )
                    )
                    for preset in material_presets
                )
            )
        if rows:
            by_isotope[str(isotope)] = tuple(rows)
    return by_isotope


def room_boundary_transport_components(
    room_size_xyz: tuple[float, float, float],
    *,
    thickness_m: float = 0.1,
    material: str = "concrete",
) -> tuple[ObstacleComponent, ...]:
    """Return transport components for the authored room floor, walls, and ceiling."""
    size_x, size_y, size_z = (float(value) for value in room_size_xyz)
    wall_height = max(0.1, size_z)
    t = max(float(thickness_m), 0.0)
    if t <= 0.0:
        return ()
    return (
        _component(
            "RoomBoundary_floor",
            center_xy=(0.5 * size_x, 0.5 * size_y),
            z_center=-0.5 * t,
            size_xyz=(size_x, size_y, t),
            material=material,
        ),
        _component(
            "RoomBoundary_north_wall",
            center_xy=(0.5 * size_x, size_y + 0.5 * t),
            z_center=0.5 * wall_height,
            size_xyz=(size_x, t, wall_height),
            material=material,
        ),
        _component(
            "RoomBoundary_south_wall",
            center_xy=(0.5 * size_x, -0.5 * t),
            z_center=0.5 * wall_height,
            size_xyz=(size_x, t, wall_height),
            material=material,
        ),
        _component(
            "RoomBoundary_east_wall",
            center_xy=(size_x + 0.5 * t, 0.5 * size_y),
            z_center=0.5 * wall_height,
            size_xyz=(t, size_y, wall_height),
            material=material,
        ),
        _component(
            "RoomBoundary_west_wall",
            center_xy=(-0.5 * t, 0.5 * size_y),
            z_center=0.5 * wall_height,
            size_xyz=(t, size_y, wall_height),
            material=material,
        ),
        _component(
            "RoomBoundary_ceiling",
            center_xy=(0.5 * size_x, 0.5 * size_y),
            z_center=size_z + 0.5 * t,
            size_xyz=(size_x, size_y, t),
            material=material,
        ),
    )


def environment_transport_model(
    instances: Iterable[KnownObstacleInstance],
    *,
    room_size_xyz: tuple[float, float, float] | None = None,
    include_room_boundaries: bool = False,
    room_boundary_thickness_m: float = 0.1,
    room_boundary_material: str = "concrete",
    isotopes: Sequence[str] = DEFAULT_TRANSPORT_ISOTOPES,
) -> tuple[
    tuple[tuple[float, float, float, float, float, float], ...],
    dict[str, tuple[float, ...]],
    dict[str, tuple[tuple[float, ...], ...]],
    dict[str, tuple[tuple[float, ...], ...]],
]:
    """Return a complete PF/Geant4 environment transport model."""
    components = list(_components_from_instances(instances))
    if include_room_boundaries:
        if room_size_xyz is None:
            raise ValueError("room_size_xyz is required for room boundary transport.")
        components.extend(
            room_boundary_transport_components(
                room_size_xyz,
                thickness_m=room_boundary_thickness_m,
                material=room_boundary_material,
            )
        )
    boxes_m, mu_by_isotope = transport_model_from_components(
        components,
        isotopes=isotopes,
    )
    line_mu_by_isotope = line_transport_model_from_components(
        components,
        isotopes=isotopes,
    )
    line_compton_mu_by_isotope = line_compton_transport_model_from_components(
        components,
        isotopes=isotopes,
    )
    return (
        boxes_m,
        mu_by_isotope,
        line_mu_by_isotope,
        line_compton_mu_by_isotope,
    )


def known_obstacle_traversability_rects(
    instances: Iterable[KnownObstacleInstance],
) -> tuple[tuple[float, float, float, float], ...]:
    """Return footprint rectangles that constrain robot motion."""
    return tuple(instance.footprint_xy for instance in instances)


def material_mu_cm_inv(material: str, isotope: str) -> float:
    """Return an effective linear attenuation coefficient for a material."""
    normalized = normalize_material_name(str(material))
    preset = resolve_material_preset(normalized)
    if preset is not None and preset.density_g_cm3 is not None:
        mass_att = _line_weighted_mass_attenuation(preset.composition_by_mass, isotope)
        if mass_att is None:
            mass_att = composition_mass_attenuation(preset.composition_by_mass, isotope)
        if mass_att is not None:
            return float(preset.density_g_cm3) * float(mass_att)
    fallback = DEFAULT_MATERIAL_MU_CM_INV.get(normalized, {})
    if isotope in fallback:
        return float(fallback[isotope])
    concrete = DEFAULT_MATERIAL_MU_CM_INV.get("concrete", {})
    return float(concrete.get(isotope, 0.0))


def material_mu_cm_inv_at_energy(
    material: str,
    energy_keV: float,
    *,
    isotope: str,
) -> float:
    """Return material linear attenuation at a gamma-line energy in 1/cm."""
    normalized = normalize_material_name(str(material))
    preset = resolve_material_preset(normalized)
    if preset is not None and preset.density_g_cm3 is not None:
        mass_att = composition_mass_attenuation_at_energy(
            preset.composition_by_mass,
            float(energy_keV),
        )
        if mass_att is not None:
            return float(preset.density_g_cm3) * float(mass_att)
    return material_mu_cm_inv(normalized, isotope)


def _lookup_nuclide(library: dict[str, object], isotope: str) -> object | None:
    """Return a nuclide entry using tolerant isotope-name matching."""
    nuclide = library.get(str(isotope))
    if nuclide is not None:
        return nuclide
    normalized = "".join(ch for ch in str(isotope).upper() if ch.isalnum())
    for name, candidate in library.items():
        candidate_key = "".join(ch for ch in str(name).upper() if ch.isalnum())
        if candidate_key == normalized:
            return candidate
    return None


def _line_weighted_mass_attenuation(
    composition_by_mass: dict[str, float],
    isotope: str,
) -> float | None:
    """Return gamma-line-weighted mass attenuation for a nuclide."""
    library = default_library()
    nuclide = _lookup_nuclide(library, str(isotope))
    if nuclide is None:
        return None
    if len(nuclide.lines) < 2:
        return None
    total_weight = 0.0
    weighted_mu = 0.0
    for line in nuclide.lines:
        weight = max(float(line.intensity), 0.0)
        if weight <= 0.0:
            continue
        mass_att = composition_mass_attenuation_at_energy(
            composition_by_mass,
            float(line.energy_keV),
        )
        if mass_att is None:
            continue
        total_weight += weight
        weighted_mu += weight * float(mass_att)
    if total_weight <= 0.0:
        return None
    return float(weighted_mu / total_weight)


def generate_manchester_obstacle_instances(
    grid: ObstacleGrid,
    *,
    room_size_xyz: tuple[float, float, float],
    obstacle_height_m: float,
    rng_seed: int | None = None,
) -> tuple[KnownObstacleInstance, ...]:
    """
    Generate known composite obstacles for the blocked grid cells.

    The cell grid remains the robot traversability model, while each cell is
    replaced by a known object made from thin shells, racks, drums, or barriers.
    This gives Manchester-like clutter without treating every obstacle as a
    fully solid concrete block.
    """
    rng = np.random.default_rng(rng_seed)
    room_z = max(float(room_size_xyz[2]), 0.1)
    max_height = min(max(float(obstacle_height_m), 0.2), room_z)
    templates = (
        _steel_cabinet_components,
        _pipe_rack_components,
        _water_drum_pair_components,
        _concrete_jersey_barrier_components,
        _aluminum_equipment_frame_components,
    )
    instances: list[KnownObstacleInstance] = []
    for index, cell in enumerate(grid.blocked_cells):
        bounds = _cell_bounds(grid, cell)
        template_fn = templates[int(rng.integers(0, len(templates)))]
        components, template_name = template_fn(
            name_prefix=f"Obstacle_{index:04d}",
            bounds_xy=bounds,
            max_height_m=max_height,
            rng=rng,
        )
        instances.append(
            KnownObstacleInstance(
                name=f"KnownObstacle_{index:04d}",
                template=template_name,
                footprint_xy=bounds,
                footprint_cells=(cell,),
                components=tuple(components),
            )
        )
    return tuple(instances)


def _as_float_tuple(
    values: Any,
    expected_len: int,
    field_name: str,
) -> tuple[float, ...]:
    """Return finite real values without accepting strings or booleans."""
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"{field_name} must be a {expected_len}-element list.")
    parsed: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name}[{index}] must be a real number.")
        parsed_value = float(value)
        if not math.isfinite(parsed_value):
            raise ValueError(f"{field_name}[{index}] must be finite.")
        parsed.append(parsed_value)
    return tuple(parsed)


def _exact_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    """Return a non-negative exact JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a JSON integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _nonempty_string(value: object, *, field_name: str) -> str:
    """Return an exact nonempty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string.")
    return value


def _validate_cell_shape(value: object, *, field_name: str) -> None:
    """Validate a serialized grid-cell pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two integers.")


def _cell_bounds(
    grid: ObstacleGrid,
    cell: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Return world XY bounds for one obstacle cell."""
    ix, iy = cell
    x0 = grid.origin[0] + float(ix) * grid.cell_size
    y0 = grid.origin[1] + float(iy) * grid.cell_size
    return (x0, x0 + grid.cell_size, y0, y0 + grid.cell_size)


def _footprint_center_size(
    bounds_xy: tuple[float, float, float, float],
    *,
    fill_fraction: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a jittered footprint center and size inside a cell."""
    x0, x1, y0, y1 = bounds_xy
    cell_x = x1 - x0
    cell_y = y1 - y0
    size_x = float(cell_x * fill_fraction)
    size_y = float(cell_y * fill_fraction)
    margin_x = max(0.0, 0.5 * (cell_x - size_x))
    margin_y = max(0.0, 0.5 * (cell_y - size_y))
    cx = 0.5 * (x0 + x1) + float(rng.uniform(-0.35, 0.35)) * margin_x
    cy = 0.5 * (y0 + y1) + float(rng.uniform(-0.35, 0.35)) * margin_y
    return (cx, cy), (size_x, size_y)


def _component(
    name: str,
    *,
    center_xy: tuple[float, float],
    z_center: float,
    size_xyz: tuple[float, float, float],
    material: str,
) -> ObstacleComponent:
    """Create one axis-aligned obstacle component."""
    return ObstacleComponent(
        name=name,
        center_xyz=(center_xy[0], center_xy[1], z_center),
        size_xyz=size_xyz,
        material=material,
    )


def _shell_components(
    *,
    name_prefix: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    height_m: float,
    thickness_m: float,
    material: str,
) -> list[ObstacleComponent]:
    """Return thin wall components for a hollow rectangular shell."""
    cx, cy = center_xy
    sx, sy = size_xy
    h = float(height_m)
    t = min(float(thickness_m), 0.45 * min(sx, sy, h))
    inner_y = max(sy - 2.0 * t, 1.0e-3)
    inner_h = max(h - 2.0 * t, 1.0e-3)
    components = [
        _component(
            f"{name_prefix}_west_panel",
            center_xy=(cx - 0.5 * sx + 0.5 * t, cy),
            z_center=0.5 * h,
            size_xyz=(t, inner_y, inner_h),
            material=material,
        ),
        _component(
            f"{name_prefix}_east_panel",
            center_xy=(cx + 0.5 * sx - 0.5 * t, cy),
            z_center=0.5 * h,
            size_xyz=(t, inner_y, inner_h),
            material=material,
        ),
        _component(
            f"{name_prefix}_south_panel",
            center_xy=(cx, cy - 0.5 * sy + 0.5 * t),
            z_center=0.5 * h,
            size_xyz=(sx, t, inner_h),
            material=material,
        ),
        _component(
            f"{name_prefix}_north_panel",
            center_xy=(cx, cy + 0.5 * sy - 0.5 * t),
            z_center=0.5 * h,
            size_xyz=(sx, t, inner_h),
            material=material,
        ),
        _component(
            f"{name_prefix}_top_panel",
            center_xy=(cx, cy),
            z_center=h - 0.5 * t,
            size_xyz=(sx, sy, t),
            material=material,
        ),
        _component(
            f"{name_prefix}_bottom_panel",
            center_xy=(cx, cy),
            z_center=0.5 * t,
            size_xyz=(sx, sy, t),
            material=material,
        ),
    ]
    return components


def _steel_cabinet_components(
    *,
    name_prefix: str,
    bounds_xy: tuple[float, float, float, float],
    max_height_m: float,
    rng: np.random.Generator,
) -> tuple[list[ObstacleComponent], str]:
    """Return a hollow steel equipment-cabinet obstacle."""
    center, size = _footprint_center_size(bounds_xy, fill_fraction=0.82, rng=rng)
    height = min(max_height_m, float(rng.uniform(1.4, 1.9)))
    return (
        _shell_components(
            name_prefix=name_prefix,
            center_xy=center,
            size_xy=size,
            height_m=height,
            thickness_m=0.035,
            material="steel",
        ),
        "steel_cabinet_hollow",
    )


def _pipe_rack_components(
    *,
    name_prefix: str,
    bounds_xy: tuple[float, float, float, float],
    max_height_m: float,
    rng: np.random.Generator,
) -> tuple[list[ObstacleComponent], str]:
    """Return a sparse steel pipe-rack obstacle."""
    center, size = _footprint_center_size(bounds_xy, fill_fraction=0.9, rng=rng)
    cx, cy = center
    sx, sy = size
    height = min(max_height_m, float(rng.uniform(1.2, 1.8)))
    beam = 0.055
    xs = (cx - 0.42 * sx, cx + 0.42 * sx)
    ys = (cy - 0.42 * sy, cy + 0.42 * sy)
    span_x = max(xs[1] - xs[0] - beam, 1.0e-3)
    span_y = max(ys[1] - ys[0] - beam, 1.0e-3)
    components: list[ObstacleComponent] = []
    for leg_idx, x in enumerate(xs):
        for y in ys:
            components.append(
                _component(
                    f"{name_prefix}_leg_{leg_idx}_{len(components)}",
                    center_xy=(x, y),
                    z_center=0.5 * height,
                    size_xyz=(beam, beam, height),
                    material="steel",
                )
            )
    for z_center in (0.25 * height, 0.78 * height):
        components.append(
            _component(
                f"{name_prefix}_rail_x_{len(components)}",
                center_xy=(cx, ys[0]),
                z_center=z_center,
                size_xyz=(span_x, beam, beam),
                material="steel",
            )
        )
        components.append(
            _component(
                f"{name_prefix}_rail_x_{len(components)}",
                center_xy=(cx, ys[1]),
                z_center=z_center,
                size_xyz=(span_x, beam, beam),
                material="steel",
            )
        )
        components.append(
            _component(
                f"{name_prefix}_rail_y_{len(components)}",
                center_xy=(xs[0], cy),
                z_center=z_center,
                size_xyz=(beam, span_y, beam),
                material="steel",
            )
        )
        components.append(
            _component(
                f"{name_prefix}_rail_y_{len(components)}",
                center_xy=(xs[1], cy),
                z_center=z_center,
                size_xyz=(beam, span_y, beam),
                material="steel",
            )
        )
    return components, "steel_pipe_rack_sparse"


def _water_drum_pair_components(
    *,
    name_prefix: str,
    bounds_xy: tuple[float, float, float, float],
    max_height_m: float,
    rng: np.random.Generator,
) -> tuple[list[ObstacleComponent], str]:
    """Return two partially filled drum-like components."""
    center, size = _footprint_center_size(bounds_xy, fill_fraction=0.86, rng=rng)
    cx, cy = center
    sx, sy = size
    drum_w = min(0.38 * sx, 0.44 * sy)
    height = min(max_height_m, float(rng.uniform(0.75, 1.15)))
    offset = 0.23 * sx
    components = []
    for idx, x in enumerate((cx - offset, cx + offset)):
        components.extend(
            _shell_components(
                name_prefix=f"{name_prefix}_drum_{idx}",
                center_xy=(x, cy),
                size_xy=(drum_w, drum_w),
                height_m=height,
                thickness_m=0.025,
                material="steel",
            )
        )
        components.append(
            _component(
                f"{name_prefix}_drum_{idx}_water_fill",
                center_xy=(x, cy),
                z_center=0.38 * height,
                size_xyz=(0.72 * drum_w, 0.72 * drum_w, 0.55 * height),
                material="water",
            )
        )
    return components, "partially_filled_steel_drums"


def _concrete_jersey_barrier_components(
    *,
    name_prefix: str,
    bounds_xy: tuple[float, float, float, float],
    max_height_m: float,
    rng: np.random.Generator,
) -> tuple[list[ObstacleComponent], str]:
    """Return a compact concrete barrier that does not fill the whole cell."""
    center, size = _footprint_center_size(bounds_xy, fill_fraction=0.92, rng=rng)
    height = min(max_height_m, float(rng.uniform(0.75, 1.15)))
    components = [
        _component(
            f"{name_prefix}_concrete_base",
            center_xy=center,
            z_center=0.22 * height,
            size_xyz=(size[0], 0.46 * size[1], 0.44 * height),
            material="concrete",
        ),
        _component(
            f"{name_prefix}_concrete_cap",
            center_xy=center,
            z_center=0.67 * height,
            size_xyz=(0.72 * size[0], 0.30 * size[1], 0.46 * height),
            material="concrete",
        ),
    ]
    return components, "concrete_barrier_partial"


def _aluminum_equipment_frame_components(
    *,
    name_prefix: str,
    bounds_xy: tuple[float, float, float, float],
    max_height_m: float,
    rng: np.random.Generator,
) -> tuple[list[ObstacleComponent], str]:
    """Return a hollow aluminum instrument frame with a small steel insert."""
    center, size = _footprint_center_size(bounds_xy, fill_fraction=0.84, rng=rng)
    height = min(max_height_m, float(rng.uniform(1.0, 1.5)))
    components = _shell_components(
        name_prefix=name_prefix,
        center_xy=center,
        size_xy=size,
        height_m=height,
        thickness_m=0.03,
        material="aluminum",
    )
    components.append(
        _component(
            f"{name_prefix}_steel_inner_box",
            center_xy=center,
            z_center=0.42 * height,
            size_xyz=(0.35 * size[0], 0.35 * size[1], 0.35 * height),
            material="steel",
        )
    )
    return components, "aluminum_equipment_frame_hollow"
