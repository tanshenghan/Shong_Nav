"""Pinhole projection of robot-local ground-plane waypoints into image pixels."""

from __future__ import annotations

import math

import numpy as np


def project_waypoints_to_image(
    waypoints: np.ndarray,
    image_size: tuple[int, int],
    hfov_deg: float,
    cam_height: float,
    min_depth: float = 0.05,
    forward_offset: float = 0.0,
) -> np.ndarray:
    """Project robot-local ground-plane waypoints ``[forward, lateral, yaw]`` to ``(u, v)``.

    ``image_size`` is ``(height, width)``. The camera is a pinhole at ``cam_height``
    metres above the ground, looking straight ahead with horizontal field of view
    ``hfov_deg``; ``fx == fy`` is derived from the frame width.

    Returns an ``(H, 2)`` float64 array. Rows whose depth is ``<= min_depth`` are NaN
    so the caller can skip them; nothing is clipped to the image bounds. A pure-yaw
    row (no displacement) inherits the previous row's position, so an in-place turn
    does not collapse onto the camera. ``forward_offset`` pushes every row that many
    metres away from the camera before projecting.
    """
    arr = np.asarray(waypoints, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(
            f"waypoints must have shape (H, 3) with H > 0, got shape {arr.shape}"
        )
    if not math.isfinite(forward_offset) or forward_offset < 0.0:
        raise ValueError(f"forward_offset must be a non-negative finite float, got {forward_offset}")

    height, width = image_size
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    forward = arr[:, 0].copy()
    lateral = arr[:, 1].copy()
    yaw = arr[:, 2]

    yaw_only = (np.abs(forward) <= min_depth) & (np.abs(lateral) <= min_depth) & (np.abs(yaw) > 1e-8)
    for i in np.flatnonzero(yaw_only):
        if i == 0:
            continue
        forward[i] = forward[i - 1]
        lateral[i] = lateral[i - 1]

    if forward_offset > 0.0:
        forward = forward + forward_offset

    valid = forward > min_depth
    safe_forward = np.where(valid, forward, 1.0)

    u = cx + fx * (-lateral / safe_forward)
    v = cy + fy * (cam_height / safe_forward)

    out = np.stack([u, v], axis=1)
    out[~valid] = np.nan
    return out


def bottom_edge_depth(h: int, w: int, cam_height: float, hfov_deg: float) -> float:
    """Ground depth (metres) that projects onto the frame's bottom edge.

    From ``v = cy + fy * cam_height / depth`` with ``cy = h / 2`` and
    ``fy = (w / 2) / tan(hfov / 2)``, setting ``v = h`` gives
    ``depth = 2 * fy * cam_height / h``. Anything nearer than this lies below the
    picture and cannot be drawn; it is the smallest forward offset that keeps a
    zero-displacement waypoint inside the frame.
    """
    if h <= 0:
        return 0.0
    fy = (float(w) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return 2.0 * fy * float(cam_height) / float(h)
