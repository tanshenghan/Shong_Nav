"""Pure command-selection helpers shared by the adapter and its tests."""

from __future__ import annotations

import math
from typing import Optional

Twist2D = tuple[float, float]


def finite_twist(linear: float, angular: float) -> bool:
    return math.isfinite(linear) and math.isfinite(angular)


def select_command(
    *,
    source: str,
    robot_mode: str,
    manual_command: Twist2D,
    manual_age_s: float,
    auto_command: Twist2D,
    auto_age_s: float,
    watchdog_s: float,
) -> Optional[Twist2D]:
    """Return the safe active command, or None when output must stop."""
    if robot_mode != "WALK":
        return None
    if source == "manual" and manual_age_s <= watchdog_s:
        if not finite_twist(*manual_command):
            return None
        return manual_command
    if source == "auto" and auto_age_s <= watchdog_s:
        if not finite_twist(*auto_command):
            return None
        return auto_command
    return None
