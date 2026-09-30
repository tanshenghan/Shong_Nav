"""World-space ground markers for predicted metric trajectories."""

import mujoco
import numpy as np

from .mpc import project_body_to_world


def ground_path(waypoints, capture_pose, ground_z=0.012):
    """Keep the model's metric scale; anchor at the pose of its input image."""
    points = project_body_to_world(waypoints, capture_pose)
    if not np.isfinite(points).all():
        raise ValueError("trajectory must be finite")
    return np.column_stack((points[:, :2], np.full(len(points), ground_z)))


def add_ground_path(scene, points):
    """Thin depth-tested beads and connectors, seen by either real camera."""
    color = np.array([0.15, 0.65, 1.0, 1.0], dtype=np.float32)
    for index, point in enumerate(points):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.025, 0, 0]), point, np.eye(3).ravel(), color)
        geom.emission = 0.6
        scene.ngeom += 1
        if index == 0 or scene.ngeom >= scene.maxgeom:
            continue
        if np.linalg.norm(point - points[index - 1]) < 1e-6:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008,
                            points[index - 1], point)
        geom.emission = 0.6
        scene.ngeom += 1
