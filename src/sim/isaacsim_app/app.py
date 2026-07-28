"""Isaac Sim sidecar application entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

from sim.isaacsim_app.observation_model import (
    IsaacAssetGeometry,
    IsaacSimObservationModel,
    MockObservationModel,
    ObservationModel,
    StageMaterialRule,
)
from sim.isaacsim_app.pf_visualizer import (
    PFSceneVisualizationConfig,
    PFSceneVisualizer,
)
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.radiation_visualizer import RadiationSceneVisualizer
from sim.isaacsim_app.scene_builder import SceneBuilder, SceneDescription
from sim.isaacsim_app.stage_backend import IsaacSimStageBackend, StageBackend
from sim.protocol import SimulationCommand, SimulationObservation
from sim.shield_geometry import ShieldThicknessConfig, resolve_shield_thickness_config


@dataclass(frozen=True)
class InitialCameraConfig:
    """Describe the initial Isaac Sim viewport camera."""

    eye_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    focal_length_mm: float = 24.0


@dataclass(frozen=True)
class StageLightingConfig:
    """Describe helper lights authored for GUI visibility."""

    dome_intensity: float = 0.0
    interior_light_position_xyz: tuple[float, float, float] | None = None
    interior_light_intensity: float = 0.0
    interior_light_radius_m: float = 1.0
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interior_lights: tuple["StageSphereLightConfig", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StageSphereLightConfig:
    """Describe one local sphere light used for GUI fill lighting."""

    position_xyz: tuple[float, float, float]
    intensity: float
    radius_m: float = 0.03
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class StageVisualRule:
    """Describe a visual-only material override for stage prims."""

    path_prefix: str
    color_rgb: tuple[float, float, float]
    opacity: float = 1.0
    roughness: float = 0.8
    emissive_scale: float = 0.0


def _json_boolean(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool | None,
) -> bool | None:
    """Return an exact JSON boolean or an explicitly nullable value."""
    value = payload.get(key, default)
    if value is None and default is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON boolean.")
    return value


def _json_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    """Return a bounded exact JSON integer."""
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a JSON integer.")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def _finite_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return a finite JSON number satisfying its physical domain."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _numeric_vector3(
    value: object,
    *,
    field_name: str,
    strictly_positive: bool = False,
) -> tuple[float, float, float]:
    """Return a finite three-number JSON array."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-element array.")
    parsed = tuple(
        _finite_number(
            component,
            field_name=field_name,
            strictly_positive=strictly_positive,
        )
        for component in value
    )
    return parsed


def _nonempty_string(value: object, *, field_name: str) -> str:
    """Return an exact nonempty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string.")
    return value


def _detector_model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict detector model without implicit scalar coercion."""
    raw = payload.get("detector_model", {})
    if not isinstance(raw, Mapping):
        raise ValueError("detector_model must be a JSON object.")
    allowed = {
        "crystal_radius_m",
        "crystal_length_m",
        "housing_thickness_m",
        "crystal_shape",
        "crystal_material",
        "housing_material",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported detector_model settings: "
            + ", ".join(str(key) for key in unknown)
        )
    result = dict(raw)
    for key in ("crystal_radius_m", "crystal_length_m"):
        if key in result:
            result[key] = _finite_number(
                result[key],
                field_name=f"detector_model.{key}",
                strictly_positive=True,
            )
    if "housing_thickness_m" in result:
        result["housing_thickness_m"] = _finite_number(
            result["housing_thickness_m"],
            field_name="detector_model.housing_thickness_m",
            minimum=0.0,
        )
    if "crystal_shape" in result:
        shape = _nonempty_string(
            result["crystal_shape"],
            field_name="detector_model.crystal_shape",
        )
        if shape != "sphere":
            raise ValueError(
                "detector_model.crystal_shape must be 'sphere'; the runtime "
                "detector geometry is spherical."
            )
    for key in ("crystal_material", "housing_material"):
        if key in result:
            result[key] = _nonempty_string(
                result[key],
                field_name=f"detector_model.{key}",
            )
    return result


@dataclass(frozen=True)
class IsaacSimAppConfig:
    """Collect sidecar configuration relevant to the Isaac Sim app."""

    headless: bool = True
    renderer: str = "RayTracedLighting"
    usd_path: str | None = None
    detector_height_m: float = 0.5
    obstacle_height_m: float = 2.0
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    scatter_gain: float = 0.03
    detector_model: dict[str, Any] = field(default_factory=dict)
    shield_thickness: ShieldThicknessConfig = field(default_factory=resolve_shield_thickness_config)
    robot_motion_speed_m_s: float = 0.5
    robot_ground_z_m: float = 0.0
    robot_animation_dt_s: float = 0.2
    robot_animation_time_scale: float = 0.0
    robot_max_animation_steps: int = 200
    camera_gesture_bindings: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    initial_camera: InitialCameraConfig | None = None
    lighting: StageLightingConfig | None = None
    preserve_viewport_on_reset: bool = False
    pf_visualization: PFSceneVisualizationConfig = field(
        default_factory=PFSceneVisualizationConfig
    )
    author_obstacle_prims: bool | None = None
    stage_visual_rules: tuple[StageVisualRule, ...] = field(default_factory=tuple)
    stage_material_rules: tuple[StageMaterialRule, ...] = field(default_factory=tuple)
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IsaacSimAppConfig":
        """Normalize a JSON config payload into a strongly typed object."""
        if data is None:
            payload: dict[str, Any] = {}
        elif isinstance(data, Mapping):
            payload = dict(data)
        else:
            raise TypeError("Isaac Sim app config must be a mapping or None.")
        extra_args = payload.get("extra_args", ())
        if not isinstance(extra_args, (list, tuple)):
            raise ValueError("extra_args must be a list of strings.")
        if any(not isinstance(value, str) or not value for value in extra_args):
            raise ValueError("extra_args entries must be nonempty strings.")
        fe_shield_size = payload.get("fe_shield_size_xyz", (0.25, 0.08, 0.25))
        pb_shield_size = payload.get("pb_shield_size_xyz", (0.25, 0.08, 0.25))
        stage_material_rules_payload = payload.get("stage_material_rules", ())
        if not isinstance(stage_material_rules_payload, (list, tuple)):
            raise ValueError("stage_material_rules must be a list of objects.")
        stage_material_rules_list: list[StageMaterialRule] = []
        seen_material_prefixes: set[str] = set()
        for index, entry in enumerate(stage_material_rules_payload):
            if not isinstance(entry, Mapping) or set(entry) != {
                "path_prefix",
                "material",
            }:
                raise ValueError(
                    f"stage_material_rules[{index}] must contain exactly "
                    "path_prefix and material."
                )
            path_prefix = _nonempty_string(
                entry["path_prefix"],
                field_name=f"stage_material_rules[{index}].path_prefix",
            )
            material = _nonempty_string(
                entry["material"],
                field_name=f"stage_material_rules[{index}].material",
            )
            if path_prefix in seen_material_prefixes:
                raise ValueError("stage_material_rules path prefixes must be unique.")
            seen_material_prefixes.add(path_prefix)
            stage_material_rules_list.append(
                StageMaterialRule(
                    path_prefix=path_prefix,
                    material=material,
                )
            )
        stage_material_rules = tuple(stage_material_rules_list)
        renderer = _nonempty_string(
            payload.get("renderer", "RayTracedLighting"),
            field_name="renderer",
        )
        usd_path = payload.get("usd_path")
        if usd_path is not None:
            usd_path = _nonempty_string(usd_path, field_name="usd_path")
        return cls(
            headless=_json_boolean(payload, "headless", default=True),
            renderer=renderer,
            usd_path=usd_path,
            detector_height_m=_finite_number(
                payload.get("detector_height_m", 0.5),
                field_name="detector_height_m",
                strictly_positive=True,
            ),
            obstacle_height_m=_finite_number(
                payload.get("obstacle_height_m", 2.0),
                field_name="obstacle_height_m",
                strictly_positive=True,
            ),
            fe_shield_size_xyz=_numeric_vector3(
                fe_shield_size,
                field_name="fe_shield_size_xyz",
                strictly_positive=True,
            ),
            pb_shield_size_xyz=_numeric_vector3(
                pb_shield_size,
                field_name="pb_shield_size_xyz",
                strictly_positive=True,
            ),
            scatter_gain=_finite_number(
                payload.get("scatter_gain", 0.03),
                field_name="scatter_gain",
                minimum=0.0,
            ),
            detector_model=_detector_model_payload(payload),
            shield_thickness=resolve_shield_thickness_config(payload),
            robot_motion_speed_m_s=_finite_number(
                payload.get("robot_motion_speed_m_s", 0.5),
                field_name="robot_motion_speed_m_s",
                strictly_positive=True,
            ),
            robot_ground_z_m=_finite_number(
                payload.get("robot_ground_z_m", 0.0),
                field_name="robot_ground_z_m",
            ),
            robot_animation_dt_s=_finite_number(
                payload.get("robot_animation_dt_s", 0.2),
                field_name="robot_animation_dt_s",
                strictly_positive=True,
            ),
            robot_animation_time_scale=_finite_number(
                payload.get("robot_animation_time_scale", 0.0),
                field_name="robot_animation_time_scale",
                minimum=0.0,
            ),
            robot_max_animation_steps=_json_integer(
                payload,
                "robot_max_animation_steps",
                default=200,
                minimum=1,
            ),
            camera_gesture_bindings=_parse_camera_gesture_bindings(payload),
            initial_camera=_parse_initial_camera_config(payload),
            lighting=_parse_lighting_config(payload),
            preserve_viewport_on_reset=_json_boolean(
                payload,
                "preserve_viewport_on_reset",
                default=False,
            ),
            pf_visualization=PFSceneVisualizationConfig.from_mapping(payload),
            author_obstacle_prims=_json_boolean(
                payload,
                "author_obstacle_prims",
                default=None,
            ),
            stage_visual_rules=_parse_stage_visual_rules(payload),
            stage_material_rules=stage_material_rules,
            extra_args=tuple(extra_args),
        )


def _parse_vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    """Parse a three-element numeric vector from a JSON payload."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a 3-element list.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_camera_gesture_bindings(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Normalize optional Isaac Sim viewport camera gesture bindings."""
    raw_bindings = payload.get("camera_gesture_bindings", {})
    if raw_bindings in (None, ""):
        return ()
    if not isinstance(raw_bindings, Mapping):
        raise ValueError("camera_gesture_bindings must be an object.")
    return tuple((str(gesture), str(binding)) for gesture, binding in raw_bindings.items())


def _parse_initial_camera_config(payload: Mapping[str, Any]) -> InitialCameraConfig | None:
    """Normalize optional initial camera placement config."""
    raw_camera = payload.get("initial_camera")
    if raw_camera in (None, ""):
        return None
    if not isinstance(raw_camera, Mapping):
        raise ValueError("initial_camera must be an object.")
    return InitialCameraConfig(
        eye_xyz=_parse_vector3(raw_camera.get("eye_xyz"), "initial_camera.eye_xyz"),
        target_xyz=_parse_vector3(raw_camera.get("target_xyz"), "initial_camera.target_xyz"),
        focal_length_mm=float(raw_camera.get("focal_length_mm", 24.0)),
    )


def _parse_lighting_config(payload: Mapping[str, Any]) -> StageLightingConfig | None:
    """Normalize optional GUI lighting config."""
    raw_lighting = payload.get("lighting")
    if raw_lighting in (None, ""):
        return None
    if not isinstance(raw_lighting, Mapping):
        raise ValueError("lighting must be an object.")
    interior_position = raw_lighting.get("interior_light_position_xyz")
    color = raw_lighting.get("color_rgb", (1.0, 1.0, 1.0))
    interior_lights = _parse_sphere_lights(raw_lighting)
    return StageLightingConfig(
        dome_intensity=float(raw_lighting.get("dome_intensity", 0.0)),
        interior_light_position_xyz=(
            None
            if interior_position in (None, "")
            else _parse_vector3(interior_position, "lighting.interior_light_position_xyz")
        ),
        interior_light_intensity=float(raw_lighting.get("interior_light_intensity", 0.0)),
        interior_light_radius_m=float(raw_lighting.get("interior_light_radius_m", 1.0)),
        color_rgb=_parse_vector3(color, "lighting.color_rgb"),
        interior_lights=interior_lights,
    )


def _parse_sphere_lights(raw_lighting: Mapping[str, Any]) -> tuple[StageSphereLightConfig, ...]:
    """Normalize optional extra local sphere lights."""
    raw_lights = raw_lighting.get("interior_lights", ())
    if raw_lights in (None, ""):
        return ()
    if not isinstance(raw_lights, (list, tuple)):
        raise ValueError("lighting.interior_lights must be a list of objects.")
    lights: list[StageSphereLightConfig] = []
    default_color = _parse_vector3(
        raw_lighting.get("color_rgb", (1.0, 1.0, 1.0)),
        "lighting.color_rgb",
    )
    default_radius = float(raw_lighting.get("interior_light_radius_m", 1.0))
    for index, entry in enumerate(raw_lights):
        if not isinstance(entry, Mapping):
            raise ValueError(f"lighting.interior_lights[{index}] must be an object.")
        color = entry.get("color_rgb", default_color)
        lights.append(
            StageSphereLightConfig(
                position_xyz=_parse_vector3(
                    entry.get("position_xyz"),
                    f"lighting.interior_lights[{index}].position_xyz",
                ),
                intensity=float(
                    entry.get("intensity", raw_lighting.get("interior_light_intensity", 0.0))
                ),
                radius_m=float(entry.get("radius_m", default_radius)),
                color_rgb=_parse_vector3(color, f"lighting.interior_lights[{index}].color_rgb"),
            )
        )
    return tuple(lights)


def _parse_stage_visual_rules(payload: Mapping[str, Any]) -> tuple[StageVisualRule, ...]:
    """Normalize optional visual material override rules."""
    raw_rules = payload.get("stage_visual_rules", ())
    if raw_rules in (None, ""):
        return ()
    if not isinstance(raw_rules, (list, tuple)):
        raise ValueError("stage_visual_rules must be a list of objects.")
    rules: list[StageVisualRule] = []
    for index, entry in enumerate(raw_rules):
        if not isinstance(entry, Mapping):
            raise ValueError(f"stage_visual_rules[{index}] must be an object.")
        rules.append(
            StageVisualRule(
                path_prefix=str(entry["path_prefix"]),
                color_rgb=_parse_vector3(
                    entry.get("color_rgb"),
                    f"stage_visual_rules[{index}].color_rgb",
                ),
                opacity=float(entry.get("opacity", 1.0)),
                roughness=float(entry.get("roughness", 0.8)),
                emissive_scale=float(entry.get("emissive_scale", 0.0)),
            )
        )
    return tuple(rules)


class IsaacSimApplication:
    """Wrap the sidecar simulation app in mock or real mode."""

    def __init__(
        self,
        *,
        use_mock: bool = True,
        app_config: dict[str, Any] | None = None,
        stage_backend: StageBackend | None = None,
    ) -> None:
        """Create the application and initialize the requested backend."""
        if not isinstance(use_mock, bool):
            raise ValueError("use_mock must be a boolean.")
        self.use_mock = use_mock
        self.config = IsaacSimAppConfig.from_dict(app_config)
        self.scene = SceneDescription()
        self.scene_builder: SceneBuilder | None = None
        self.robot_controller: RobotController | None = None
        self.radiation_visualizer: RadiationSceneVisualizer | None = None
        self.pf_visualizer: PFSceneVisualizer | None = None
        self.observation_model: ObservationModel
        self._stage_backend = stage_backend
        self._loaded_scene_usd_path: str | None = None
        self._initial_camera_applied = False
        self.asset_geometry = IsaacAssetGeometry(
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        self.stage_material_rules = self.config.stage_material_rules
        if self.use_mock:
            self.observation_model = MockObservationModel(
            asset_geometry=self.asset_geometry,
            scatter_gain=self.config.scatter_gain,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
        )
            return
        backend = stage_backend
        if backend is None:
            backend = IsaacSimStageBackend(
                headless=self.config.headless,
                renderer=self.config.renderer,
                camera_gesture_bindings=dict(self.config.camera_gesture_bindings),
            )
        self._stage_backend = backend
        self.radiation_visualizer = RadiationSceneVisualizer(backend)
        self.pf_visualizer = PFSceneVisualizer(
            backend,
            config=self.config.pf_visualization,
        )
        self.scene_builder = SceneBuilder(
            backend,
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
            shield_thickness=self.config.shield_thickness,
        )
        self.robot_controller = RobotController(
            backend,
            self.scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
            motion_speed_m_s=self.config.robot_motion_speed_m_s,
            animation_dt_s=self.config.robot_animation_dt_s,
            animation_time_scale=self.config.robot_animation_time_scale,
            max_animation_steps=self.config.robot_max_animation_steps,
        )
        self.observation_model = IsaacSimObservationModel(
            self.robot_controller,
            usd_path=self.config.usd_path,
            asset_geometry=self.asset_geometry,
            stage_material_rules=self.stage_material_rules,
            scatter_gain=self.config.scatter_gain,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
        )

    def reset(self, scene: SceneDescription) -> None:
        """Load a new scene description and reset robot state."""
        if (
            scene.usd_path is None
            and scene.use_config_usd_fallback
            and self.config.usd_path is not None
        ):
            scene.usd_path = self.config.usd_path
        if self.config.author_obstacle_prims is not None:
            scene.author_obstacle_prims = self.config.author_obstacle_prims
        self.scene = scene
        if self.use_mock:
            self.observation_model.reset(scene)
            return
        if self.scene_builder is None or self.robot_controller is None:
            raise RuntimeError("Real Isaac Sim mode was not initialized correctly.")
        scene_usd_path = None if scene.usd_path is None else str(scene.usd_path)
        reuse_loaded_stage = (
            self.config.preserve_viewport_on_reset
            and self._loaded_scene_usd_path is not None
            and self._loaded_scene_usd_path == scene_usd_path
        )
        self.scene_builder.load_scene(
            scene,
            usd_path_override=None,
            reopen_stage=not reuse_loaded_stage,
        )
        self._loaded_scene_usd_path = scene_usd_path
        self.robot_controller = RobotController(
            self._stage_backend,
            scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
            motion_speed_m_s=self.config.robot_motion_speed_m_s,
            animation_dt_s=self.config.robot_animation_dt_s,
            animation_time_scale=self.config.robot_animation_time_scale,
            max_animation_steps=self.config.robot_max_animation_steps,
        )
        self.robot_controller.reset()
        self.observation_model = IsaacSimObservationModel(
            self.robot_controller,
            usd_path=scene.usd_path,
            asset_geometry=self.asset_geometry,
            stage_material_rules=self.stage_material_rules,
            scatter_gain=self.config.scatter_gain,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
        )
        self.observation_model.reset(scene)
        self._configure_stage_view_helpers()

    def _configure_stage_view_helpers(self) -> None:
        """Author optional lights and initial camera for live GUI inspection."""
        if self._stage_backend is None:
            return
        self._stage_backend.ensure_xform("/World/SimBridge/View")
        for rule in self.config.stage_visual_rules:
            self._stage_backend.apply_visual_material(
                rule.path_prefix,
                color_rgb=rule.color_rgb,
                opacity=rule.opacity,
                roughness=rule.roughness,
                emissive_scale=rule.emissive_scale,
            )
        lighting = self.config.lighting
        if lighting is not None:
            if lighting.dome_intensity > 0.0:
                self._stage_backend.ensure_dome_light(
                    "/World/SimBridge/View/DomeLight",
                    intensity=lighting.dome_intensity,
                    color_rgb=lighting.color_rgb,
                )
            if (
                lighting.interior_light_position_xyz is not None
                and lighting.interior_light_intensity > 0.0
            ):
                self._stage_backend.ensure_sphere_light(
                    "/World/SimBridge/View/InteriorLight",
                    intensity=lighting.interior_light_intensity,
                    radius_m=lighting.interior_light_radius_m,
                    translation_xyz=lighting.interior_light_position_xyz,
                    color_rgb=lighting.color_rgb,
                )
            for index, fill_light in enumerate(lighting.interior_lights):
                if fill_light.intensity <= 0.0:
                    continue
                self._stage_backend.ensure_sphere_light(
                    f"/World/SimBridge/View/InteriorLight_{index:02d}",
                    intensity=fill_light.intensity,
                    radius_m=fill_light.radius_m,
                    translation_xyz=fill_light.position_xyz,
                    color_rgb=fill_light.color_rgb,
                )
        camera = self.config.initial_camera
        should_apply_initial_camera = (
            camera is not None
            and (
                not self.config.preserve_viewport_on_reset
                or not self._initial_camera_applied
            )
        )
        if camera is not None and should_apply_initial_camera:
            self._stage_backend.set_camera_view(
                "/World/SimBridge/View/InitialCamera",
                eye_xyz=camera.eye_xyz,
                target_xyz=camera.target_xyz,
                focal_length_mm=camera.focal_length_mm,
            )
            self._initial_camera_applied = True
        self._stage_backend.step()

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Apply a command and return the resulting observation."""
        if self.use_mock:
            return self.observation_model.observe(command)
        if self.robot_controller is None:
            raise RuntimeError("Robot controller is not available in real mode.")
        self.robot_controller.apply_command(command)
        return self.observation_model.observe(command)

    def visualize_observation(self, observation: SimulationObservation) -> None:
        """Render simulator-provided observation metadata in Isaac Sim."""
        if self.use_mock:
            return
        if self.radiation_visualizer is None:
            return
        self.radiation_visualizer.update_from_observation(observation)

    def visualize_pf_state(self, payload: dict[str, Any]) -> None:
        """Render PF particles and estimates as visual-only Isaac Sim markers."""
        if self.use_mock:
            return
        if self.pf_visualizer is None:
            return
        self.pf_visualizer.update_from_payload(payload)

    def update(self) -> None:
        """Pump the simulator event loop once when a real backend is active."""
        backend = self._stage_backend
        if backend is not None:
            backend.step()

    def close(self) -> None:
        """Close the underlying simulator backend if present."""
        backend = self._stage_backend
        if backend is not None:
            backend.close()
