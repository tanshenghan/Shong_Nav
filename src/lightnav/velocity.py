"""Convert a predicted waypoint into a Habitat ``velocity_control`` action dict."""

from __future__ import annotations

import math

import numpy as np


def _normalize_to_raw(value: float, vmin: float, vmax: float) -> float:
    """Inverse of habitat's scaled = vmin + (raw+1)/2 * (vmax - vmin); clipped to [-1, 1]."""
    span = vmax - vmin
    if not math.isfinite(span) or span == 0.0:
        raise ValueError(f"velocity range must have non-zero finite span, got ({vmin}, {vmax})")
    raw = 2.0 * (value - vmin) / span - 1.0
    return float(np.clip(raw, -1.0, 1.0))


def first_waypoint_to_velocity_cmd(
    centroid_first_step: np.ndarray,
    dt: float,
    lin_vel_range: tuple[float, float],
    ang_vel_range: tuple[float, float],
) -> dict:
    """Map a single (forward_m, lateral_m, yaw_rad) waypoint to a Habitat velocity_control dict.

    Decoupled 2-DoF mapping (unicycle, lateral component dropped):
      v = forward_m / dt           m/s
      w = degrees(yaw_rad) / dt    deg/s
    Habitat's velocity_control then clips into [lin_vel_range, ang_vel_range], which acts
    as the per-step motion cap (e.g. lin_vel_range=[0, 2.5] + dt=0.1 => <= 0.25 m/step).
    """
    arr = np.asarray(centroid_first_step)
    if arr.shape != (3,):
        raise ValueError(f"waypoint must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"waypoint must be finite, got {arr}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite float, got {dt}")

    forward_m = float(arr[0])
    yaw_rad = float(arr[2])

    lin_mps = forward_m / dt
    ang_dps = math.degrees(yaw_rad) / dt

    lin_raw = _normalize_to_raw(lin_mps, lin_vel_range[0], lin_vel_range[1])
    ang_raw = _normalize_to_raw(ang_dps, ang_vel_range[0], ang_vel_range[1])

    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": lin_raw,
            "angular_velocity": ang_raw,
        },
    }


def is_stop_centroid(centroid: np.ndarray, atol: float = 1e-6) -> bool:
    """True iff every element of the (horizon, 3) centroid is within `atol` of zero."""
    arr = np.asarray(centroid)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"centroid must have shape (horizon, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("centroid must be finite")
    return bool(np.all(np.abs(arr) <= atol))
