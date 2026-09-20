"""first_waypoint_to_velocity_cmd and is_stop_centroid: waypoint -> Habitat velocity_control."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lightnav.velocity import (
    _normalize_to_raw,
    first_waypoint_to_velocity_cmd,
    is_stop_centroid,
)


def test_zero_waypoint_maps_to_lin_min_and_zero_ang() -> None:
    out = first_waypoint_to_velocity_cmd(
        np.zeros(3, dtype=np.float32),
        dt=1.0,
        lin_vel_range=(0.0, 1.5),
        ang_vel_range=(-45.0, 45.0),
    )
    assert out["action"] == "velocity_control"
    # lin_mps=0 with range (0, 1.5): raw = 2*(0-0)/1.5 - 1 = -1.
    assert out["action_args"]["linear_velocity"] == pytest.approx(-1.0)
    # ang_dps=0 with symmetric range: raw = 2*(0+45)/90 - 1 = 0.
    assert out["action_args"]["angular_velocity"] == pytest.approx(0.0)
    assert set(out) == {"action", "action_args"}
    assert set(out["action_args"]) == {"linear_velocity", "angular_velocity"}
    assert isinstance(out["action_args"]["linear_velocity"], float)


def test_pure_forward_quarter_meter() -> None:
    out = first_waypoint_to_velocity_cmd(
        np.array([0.25, 0.0, 0.0]),
        dt=1.0,
        lin_vel_range=(0.0, 1.0),
        ang_vel_range=(-45.0, 45.0),
    )
    # lin_mps=0.25 in (0,1): raw = 2*0.25 - 1 = -0.5
    assert out["action_args"]["linear_velocity"] == pytest.approx(-0.5)
    assert out["action_args"]["angular_velocity"] == pytest.approx(0.0)


def test_pure_left_turn() -> None:
    out = first_waypoint_to_velocity_cmd(
        np.array([0.0, 0.0, math.pi / 4]),
        dt=1.0,
        lin_vel_range=(0.0, 1.0),
        ang_vel_range=(-90.0, 90.0),
    )
    # ang_dps=45 in (-90, 90): raw = 2*(45+90)/180 - 1 = 0.5
    assert out["action_args"]["angular_velocity"] == pytest.approx(0.5)
    # lin_mps=0 in (0,1): raw = -1
    assert out["action_args"]["linear_velocity"] == pytest.approx(-1.0)


def test_dt_scales_the_commanded_speeds() -> None:
    # 0.1 m per 0.1 s step = 1 m/s; range (0, 2) -> raw 0.
    out = first_waypoint_to_velocity_cmd(
        np.array([0.1, 0.0, math.radians(3.0)]),
        dt=0.1,
        lin_vel_range=(0.0, 2.0),
        ang_vel_range=(-60.0, 60.0),
    )
    assert out["action_args"]["linear_velocity"] == pytest.approx(0.0)
    # 3 deg per 0.1 s = 30 deg/s in (-60, 60) -> raw 0.5
    assert out["action_args"]["angular_velocity"] == pytest.approx(0.5)


def test_lateral_component_is_ignored() -> None:
    a = first_waypoint_to_velocity_cmd(
        np.array([0.2, 0.0, 0.0]), dt=1.0, lin_vel_range=(0.0, 1.0), ang_vel_range=(-45.0, 45.0)
    )
    b = first_waypoint_to_velocity_cmd(
        np.array([0.2, 0.7, 0.0]), dt=1.0, lin_vel_range=(0.0, 1.0), ang_vel_range=(-45.0, 45.0)
    )
    assert a == b


def test_forward_saturation_clips_to_one() -> None:
    out = first_waypoint_to_velocity_cmd(
        np.array([10.0, 0.0, 0.0]),
        dt=1.0,
        lin_vel_range=(0.0, 1.0),
        ang_vel_range=(-45.0, 45.0),
    )
    assert out["action_args"]["linear_velocity"] == pytest.approx(1.0)


def test_backward_angular_saturation_clips_to_minus_one() -> None:
    # Large negative yaw -> ang_dps very negative -> raw clamps to -1.
    out = first_waypoint_to_velocity_cmd(
        np.array([0.0, 0.0, -math.pi]),
        dt=1.0,
        lin_vel_range=(0.0, 1.0),
        ang_vel_range=(-45.0, 45.0),
    )
    assert out["action_args"]["angular_velocity"] == pytest.approx(-1.0)


def test_normalize_to_raw_is_the_inverse_of_habitats_scaling() -> None:
    vmin, vmax = 0.0, 0.25
    for raw in (-1.0, -0.5, 0.0, 0.3, 1.0):
        scaled = vmin + (raw + 1.0) / 2.0 * (vmax - vmin)
        assert _normalize_to_raw(scaled, vmin, vmax) == pytest.approx(raw)
    with pytest.raises(ValueError):
        _normalize_to_raw(0.1, 1.0, 1.0)


def test_is_stop_centroid_variants() -> None:
    horizon = 10
    assert is_stop_centroid(np.zeros((horizon, 3))) is True

    noise = np.full((horizon, 3), 1e-9)
    assert is_stop_centroid(noise, atol=1e-6) is True

    nonzero = np.zeros((horizon, 3))
    nonzero[0, 0] = 0.1
    assert is_stop_centroid(nonzero) is False

    near = np.full((horizon, 3), 1e-3)
    assert is_stop_centroid(near) is False
    assert is_stop_centroid(near, atol=5e-3) is True


def test_malformed_waypoint_shape_raises() -> None:
    with pytest.raises(ValueError):
        first_waypoint_to_velocity_cmd(
            np.zeros(4),
            dt=1.0,
            lin_vel_range=(0.0, 1.0),
            ang_vel_range=(-45.0, 45.0),
        )


def test_non_finite_waypoint_and_bad_dt_raise() -> None:
    with pytest.raises(ValueError):
        first_waypoint_to_velocity_cmd(
            np.array([np.nan, 0.0, 0.0]),
            dt=1.0,
            lin_vel_range=(0.0, 1.0),
            ang_vel_range=(-45.0, 45.0),
        )
    with pytest.raises(ValueError):
        first_waypoint_to_velocity_cmd(
            np.zeros(3), dt=0.0, lin_vel_range=(0.0, 1.0), ang_vel_range=(-45.0, 45.0)
        )


def test_malformed_centroid_shape_raises() -> None:
    with pytest.raises(ValueError):
        is_stop_centroid(np.zeros(10))
    with pytest.raises(ValueError):
        is_stop_centroid(np.full((10, 3), np.inf))
