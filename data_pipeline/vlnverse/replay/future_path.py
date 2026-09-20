"""Occluded 3D future-pose markers for previews, never training observations.

Ground is established from the unmodified imported scene before creating markers.
Ground must be in an identified floor group, or a flat structural ground cap
independently supported by floor geometry. Missing ground and furniture fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FuturePathConfig:
    horizon_steps: int = 30
    line_width_m: float = .04
    point_diameter_m: float = .05
    ground_lift_m: float = .012
    ray_start_above_source_m: float = .3
    max_ground_source_delta_m: float = .08
    minimum_up_normal_z: float = .98
    structural_cap_tolerance_m: float = .002

    def __post_init__(self):
        if (isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, int)
                or self.horizon_steps <= 0):
            raise ValueError("future horizon_steps must be a positive integer")
        positive = (self.line_width_m, self.point_diameter_m, self.ground_lift_m,
                    self.ray_start_above_source_m, self.max_ground_source_delta_m,
                    self.structural_cap_tolerance_m)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("future path dimensions and ground tolerances must be positive and finite")
        if (self.ray_start_above_source_m <= self.max_ground_source_delta_m
                or not 0 < self.minimum_up_normal_z <= 1):
            raise ValueError("invalid ground ray settings")


def future_indices(current_index: int, pose_count: int, horizon_steps: int) -> list[int]:
    """The current pose connects the line but is not one of the future points."""
    if not 0 <= current_index < pose_count or horizon_steps < 0:
        raise ValueError("invalid future path index or horizon")
    return list(range(current_index + 1, min(pose_count, current_index + horizon_steps + 1)))


def camera_forward(yaw_rad: float, pitch_down_deg: float) -> np.ndarray:
    """World direction for +X yaw zero, +Z up and positive downward pitch."""
    if not np.isfinite([yaw_rad, pitch_down_deg]).all() or not -89 < pitch_down_deg < 89:
        raise ValueError("camera yaw/pitch must be finite and pitch between -89 and 89 degrees")
    pitch = np.deg2rad(pitch_down_deg)
    return np.array([np.cos(pitch) * np.cos(yaw_rad), np.cos(pitch) * np.sin(yaw_rad), -np.sin(pitch)])


def validate_structural_cap(location, source_z, bounds_z, floor_hit, config):
    """A wall-named horizontal ground cap needs a separately verified floor below."""
    location, bounds = np.asarray(location), np.asarray(bounds_z)
    tolerance = config.structural_cap_tolerance_m
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or np.ptp(bounds) > tolerance or np.max(np.abs(bounds-source_z)) > tolerance):
        raise ValueError("structural wall surface is not a flat cap at source ground height")
    if floor_hit is None:
        raise ValueError("structural ground cap has no identified floor beneath it")
    floor_location, normal, path = floor_hit
    floor_location, normal = np.asarray(floor_location), np.asarray(normal)
    if (floor_location.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(floor_location).all() or not np.isfinite(normal).all()
            or np.linalg.norm(normal) < 1e-8
            or normal[2] / np.linalg.norm(normal) < config.minimum_up_normal_z
            or "floor" not in str(path).lower().split("/")
            or not np.allclose(floor_location[:2], location[:2], atol=1e-5, rtol=0)):
        raise ValueError("structural ground cap has an invalid supporting floor")
    depth = float(location[2]-floor_location[2])
    if not -tolerance <= depth <= config.max_ground_source_delta_m:
        raise ValueError("supporting floor is too far below the structural ground cap")
    return dict(classification="horizontal_structural_ground_cap", mesh_z_bounds_m=bounds.tolist(),
                supporting_floor_path=str(path), supporting_floor_z_m=float(floor_location[2]),
                supporting_floor_depth_m=depth)


def project_to_ground(positions_xyz: np.ndarray, ray_cast: Callable,
                      config: FuturePathConfig, structural_cap_check=None) -> tuple[np.ndarray, list[dict]]:
    """Use the first scene hit, supplied as (location, normal, USD path), or None.

    The callback receives origin, downward unit direction and maximum distance.
    It must query only scene geometry; all rays precede overlay creation.
    """
    positions = np.asarray(positions_xyz, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not len(positions)
            or not np.isfinite(positions).all()):
        raise ValueError("ground projection requires finite positions_xyz[N,3]")
    grounded, hits = positions.copy(), []
    for index, position in enumerate(positions):
        origin = position + [0, 0, config.ray_start_above_source_m]
        result = ray_cast(origin, np.array([0., 0., -1.]),
                          config.ray_start_above_source_m + config.max_ground_source_delta_m)
        if result is None:
            raise ValueError(f"no scene ground hit at pose {index}; refusing a floating preview overlay")
        location, normal, usd_path = result
        location, normal = np.asarray(location, dtype=float), np.asarray(normal, dtype=float)
        if (location.shape != (3,) or normal.shape != (3,)
                or not np.isfinite(location).all() or not np.isfinite(normal).all()):
            raise ValueError(f"invalid scene ground hit at pose {index}")
        if not np.allclose(location[:2], position[:2], atol=1e-5, rtol=0):
            raise ValueError(f"ground ray moved horizontally at pose {index}")
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-8 or normal[2] / normal_length < config.minimum_up_normal_z:
            raise ValueError(f"scene hit is not an upward floor at pose {index}: {usd_path}")
        if abs(location[2] - position[2]) > config.max_ground_source_delta_m:
            raise ValueError(f"scene ground height differs from source Z at pose {index}: {usd_path}")
        cap_evidence = None
        groups = str(usd_path).lower().split("/")
        if "floor" not in groups:
            if "wall" not in groups or structural_cap_check is None:
                raise ValueError(f"scene hit is not in an identified USD floor group at pose {index}: {usd_path}")
            cap_evidence = structural_cap_check(usd_path, location, position[2])
        grounded[index, 2] = location[2]
        hits.append(dict(pose_index=index, usd_prim_path=str(usd_path),
                         height_m=float(location[2]), normal_xyz=normal.tolist(),
                         source_z_delta_m=float(location[2] - position[2]),
                         structural_cap_evidence=cap_evidence))
    return grounded, hits


def line_mesh(ground_points: np.ndarray, width_m: float, lift_m: float) -> tuple[list, list]:
    """Flat 3D ribbon quads; in-place turns keep their indices but add no segment."""
    vertices, faces = [], []
    for start, end in zip(ground_points[:-1], ground_points[1:]):
        delta = end[:2] - start[:2]
        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            continue
        sideways = np.array([-delta[1], delta[0], 0.]) * (width_m / (2 * distance))
        start, end = start + [0, 0, lift_m], end + [0, 0, lift_m]
        offset = len(vertices)
        vertices.extend([start - sideways, end - sideways, end + sideways, start + sideways])
        faces.append(tuple(range(offset, offset + 4)))
    return [v.tolist() for v in vertices], faces


def create_future_path(positions_xyz: np.ndarray, config: FuturePathConfig):
    """Validate every ground ray before creating any bpy overlay geometry."""
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    if any(obj.get("vlnverse_preview_overlay", False) for obj in scene.objects):
        raise ValueError("ground rays require a scene without existing preview overlays")
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = {obj.get("usd_prim_path", ""): obj for obj in scene.objects if obj.type == "MESH"}
    floor_meshes = [obj.evaluated_get(depsgraph) for path, obj in meshes.items()
                    if "floor" in path.lower().split("/")]

    def ray_cast(origin, direction, distance):
        hit, location, normal, _, obj, _ = scene.ray_cast(
            depsgraph, Vector(origin), Vector(direction), distance=distance)
        if not hit:
            return None
        return tuple(location), tuple(normal), obj.get("usd_prim_path", "")

    def structural_cap_check(path, location, source_z):
        obj = meshes[path].evaluated_get(depsgraph)
        world_z = [(obj.matrix_world @ Vector(point)).z for point in obj.bound_box]
        # Query only identified floor meshes, independently of the first scene hit.
        origin = Vector(location + [0, 0, config.structural_cap_tolerance_m])
        endpoint = origin - Vector((0, 0, config.max_ground_source_delta_m + 2*config.structural_cap_tolerance_m))
        floor_hit = None
        for floor in floor_meshes:
            inverse = floor.matrix_world.inverted()
            local_origin = inverse @ origin
            direction = inverse @ endpoint - local_origin
            hit, point, normal, _ = floor.ray_cast(local_origin, direction.normalized(), distance=direction.length)
            if hit:
                point = floor.matrix_world @ point
                normal = (inverse.transposed().to_3x3() @ normal).normalized()
                if floor_hit is None or point.z > floor_hit[0][2]:
                    floor_hit = (tuple(point), tuple(normal), floor.get("usd_prim_path", ""))
        return validate_structural_cap(location, source_z, [min(world_z), max(world_z)], floor_hit, config)

    positions, hits = project_to_ground(positions_xyz, ray_cast, config, structural_cap_check)
    return BlenderFuturePath(positions, hits, config)


class BlenderFuturePath:
    """Reusable ground geometry; one point object per future pose, plus a ribbon."""

    def __init__(self, ground_positions: np.ndarray, ground_hits: list[dict], config: FuturePathConfig):
        import bpy

        self.positions, self.config = ground_positions, config
        collection = bpy.data.collections.new("PreviewFuturePath")
        bpy.context.scene.collection.children.link(collection)

        def material(name, srgb):
            value = np.asarray(srgb, dtype=float) / 255
            linear = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*linear, 1.)
            emission.inputs["Strength"].default_value = 1.
            output = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
            return mat

        line_material = material("PreviewFutureLineCyan", (40, 200, 255))
        point_material = material("PreviewFuturePointsGreen", (105, 255, 195))

        def object_for_mesh(name, mesh, mat):
            obj = bpy.data.objects.new(name, mesh)
            obj["vlnverse_preview_overlay"] = True
            collection.objects.link(obj)
            if mat.name not in mesh.materials:
                mesh.materials.append(mat)
            # Geometry is camera-visible and naturally occluded, but cannot light
            # the room, alter reflections or cast shadows into source previews.
            for visibility in ("visible_diffuse", "visible_glossy", "visible_transmission",
                               "visible_volume_scatter", "visible_shadow"):
                setattr(obj, visibility, False)
            return obj

        self.line = object_for_mesh("PreviewFutureLine", bpy.data.meshes.new("PreviewFutureLine"), line_material)
        angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        circle = bpy.data.meshes.new("PreviewFuturePointDisk")
        circle.from_pydata([(config.point_diameter_m / 2 * np.cos(a),
                             config.point_diameter_m / 2 * np.sin(a), 0.) for a in angle],
                           [], [tuple(range(len(angle)))])
        circle.materials.append(point_material)
        self.points = []
        for index in range(min(config.horizon_steps, max(0, len(ground_positions) - 1))):
            obj = object_for_mesh(f"PreviewFuturePoint{index:03d}", circle, point_material)
            self.points.append(obj)
        self.manifest = dict(enabled=True, horizon_steps=config.horizon_steps,
            preview_only=True, training_observations=False, config=asdict(config),
            line_color_srgb=[40, 200, 255], point_color_srgb=[105, 255, 195],
            geometry="ground ribbon and disks, naturally occluded by scene geometry",
            ground_validation="first downward scene hit, upward and near source Z; identified floor or verified flat structural ground cap with identified floor beneath",
            ground_rays_before_overlay_creation=True, ground_positions_xyz=ground_positions.tolist(),
            ground_hits=ground_hits, current_point_counts_toward_horizon=False,
            horizon_units="stored future pose indices, not seconds or fixed-rate control steps", frames=[])

    def update(self, current_index: int) -> dict:
        future = future_indices(current_index, len(self.positions), self.config.horizon_steps)
        vertices, faces = line_mesh(self.positions[[current_index, *future]],
                                    self.config.line_width_m, self.config.ground_lift_m)
        self.line.data.clear_geometry()
        self.line.data.from_pydata(vertices, [], faces)
        self.line.data.update()
        self.line.hide_render = not bool(faces)
        for offset, obj in enumerate(self.points):
            obj.hide_render = offset >= len(future)
            if not obj.hide_render:
                obj.location = self.positions[future[offset]] + [0, 0, self.config.ground_lift_m + .002]
        audit = dict(frame_index=current_index, current_index=current_index, future_indices=future,
                     future_ground_heights_m=self.positions[future, 2].tolist(),
                     current_ground_height_m=float(self.positions[current_index, 2]),
                     line_segment_count=len(faces), future_point_objects=len(future))
        self.manifest["frames"].append(audit)
        return audit
