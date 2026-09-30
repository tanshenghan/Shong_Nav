"""Adapt VLNVerse world poses to the shared lightnav.viz review graphics.

The ribbon is a display annotation, including the shared visual forward offset.
It does not certify scene occlusion or change the saved reference trajectory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def future_local_ribbon_points(positions, yaws, index, horizon=30):
    """Future absolute positions expressed once in the current robot frame.

    The third column is zero because the ribbon only uses position. This avoids
    the shared projector's action-oriented near-zero/yaw-only heuristic moving
    an actual near-camera GT position onto the preceding point.
    """
    positions, yaws = np.asarray(positions, dtype=float), np.asarray(yaws, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3 or yaws.shape != (len(positions),)
            or not np.isfinite(positions).all() or not np.isfinite(yaws).all()
            or not 0 <= index < len(positions) or horizon < 1):
        raise ValueError("invalid world poses or future horizon")
    stop = min(index + horizon + 1, len(positions))
    delta = positions[index + 1:stop] - positions[index]
    if len(delta) and np.max(np.abs(delta[:, 2])) > 1e-5:
        raise ValueError("shared LightNav projection requires a planar trajectory")
    c, s = np.cos(yaws[index]), np.sin(yaws[index])
    points = np.zeros((len(delta), 3), dtype=float)
    points[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    points[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    return points


def reference_body_velocity(world_velocity, yaw, yaw_rate):
    """Rotate the saved analytic reference derivative; video FPS is not a dt."""
    vx, vy = np.asarray(world_velocity)[:2]
    c, s = np.cos(yaw), np.sin(yaw)
    return float(c*vx + s*vy), float(-s*vx + c*vy), float(yaw_rate)


def validate_camera(camera, render_manifests):
    if abs(float(camera.get("pitch_down_deg", 0.))) > 1e-8:
        raise ValueError("lightnav.viz uses a horizontal camera; render fresh images with pitch 0")
    if any(m.get("future_path", {}).get("enabled", False) for m in render_manifests):
        raise ValueError("lightnav.viz requires clean scene RGB; disable the 3D future-path overlay")


def style_provenance(camera, horizon):
    from lightnav.viz import projection, render

    width, height = camera["width"], camera["height"]
    return dict(
        style="lightnav.viz", future_horizon=horizon,
        functions=["lightnav.viz.render.draw_traj_ribbon", "lightnav.viz.render.draw_scifi_hud"],
        source_files={str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                      for m in (render, projection)},
        future_coordinates="world future positions relative to current body; forward +X, lateral +left; no cumulative sum",
        ribbon_yaw_column="zero for position-only GT projection; actual yaw retained in trajectory NPZ",
        forward_offset_mode="auto, shared bottom-edge visibility offset",
        visual_forward_offset_m=projection.bottom_edge_depth(height, width, camera["height_above_source_z_m"], camera["hfov_deg"]),
        width_frac=render.TRAJ_WIDTH_FRAC, alpha=render.TRAJ_FILL_ALPHA,
        depth_occlusion=False, physically_exact_ground_placement=False,
        hud_velocity="saved synthetic reference derivative in body frame; unavailable for original GT",
        hud_step_and_rate="video frame number and playback FPS, not measured source observation rate",
        original_source_rgb_overlay=False, review_only=True, training_observations=False)


def draw_review(rgb, *, label, video_frame, fps, stop, camera, waypoints=None, velocity=None):
    from lightnav.viz.render import draw_scifi_hud, draw_traj_ribbon

    out = np.asarray(rgb, dtype=np.uint8).copy()
    if waypoints is not None:
        out = draw_traj_ribbon(out, waypoints, hfov_deg=camera["hfov_deg"],
                               cam_height=camera["height_above_source_z_m"], forward_offset=None)
    return draw_scifi_hud(out, instruction=label, step=video_frame, fps=fps,
                         vel=velocity, stop=stop)
