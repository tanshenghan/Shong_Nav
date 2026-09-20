from __future__ import annotations

import math
from collections.abc import Sequence


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def waypoint_command(
    waypoints: Sequence[Sequence[float]],
    *,
    max_linear: float = 0.35,
    max_angular: float = 1.2,
) -> tuple[float, float]:
    """Turn a body-frame VLN path into a conservative differential-drive command."""
    if not waypoints:
        return 0.0, 0.0
    valid = [
        point
        for point in waypoints
        if len(point) >= 3 and all(math.isfinite(float(value)) for value in point[:3])
    ]
    if not valid:
        return 0.0, 0.0

    target = next(
        (point for point in valid if math.hypot(float(point[0]), float(point[1])) >= 0.35),
        valid[-1],
    )
    forward, lateral, target_yaw = (float(value) for value in target[:3])
    distance = math.hypot(forward, lateral)
    bearing = math.atan2(lateral, forward)
    angular = clamp(1.8 * bearing + 0.25 * target_yaw, -max_angular, max_angular)
    alignment = max(0.0, math.cos(bearing))
    linear = clamp(0.75 * distance * alignment, 0.0, max_linear)
    if abs(bearing) > math.radians(65):
        linear = 0.0
    return linear, angular
