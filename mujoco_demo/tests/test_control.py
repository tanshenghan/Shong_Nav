import math

import pytest

from vln_mujoco.control import waypoint_command


def test_waypoint_ahead_drives_forward() -> None:
    linear, angular = waypoint_command([(0.8, 0.0, 0.0)])
    assert linear == pytest.approx(0.35)
    assert angular == pytest.approx(0.0)


def test_waypoint_to_the_left_turns_left() -> None:
    linear, angular = waypoint_command([(0.5, 0.5, 0.0)])
    assert linear > 0.0
    assert angular > 0.0


def test_waypoint_behind_rotates_without_forward_motion() -> None:
    linear, angular = waypoint_command([(-0.4, 0.1, math.pi)])
    assert linear == 0.0
    assert angular > 0.0


def test_empty_path_stops() -> None:
    assert waypoint_command([]) == (0.0, 0.0)
