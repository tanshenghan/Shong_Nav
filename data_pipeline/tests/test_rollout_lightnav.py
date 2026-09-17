"""Prediction sentinels and reference controller compatibility, without GPU weights."""

import math

import numpy as np
import pytest
from PIL import Image

from data_pipeline.rollout_lightnav import (
    physical_command,
    predicted_points,
    render_prediction_frame,
)
from lightnav.habitat.policy import select_action_waypoint
from lightnav.velocity import first_waypoint_to_velocity_cmd

CONTROL = dict(lin_vel_range=[0., 2.5], ang_vel_range=[-300., 300.], time_step=.1,
               min_abs_lin_speed=.01, min_abs_ang_speed=.5)


def test_controller_round_trip_caps_and_skips_origin():
    index, chosen = select_action_waypoint(np.array([[0., 0., 0.], [.5, .2, -math.pi / 4]]))
    assert index == 1
    action = first_waypoint_to_velocity_cmd(chosen, .1, (0., 2.5), (-300., 300.))
    result = physical_command(action, CONTROL)
    assert result["linear_m_s"] == 2.5
    assert result["angular_deg_s"] == -300
    assert not result["stop"]


def test_zero_velocity_stops_and_turn_in_place_does_not():
    for waypoint, stopped in (([0., 0., 0.], True), ([0., 0., .2], False)):
        action = first_waypoint_to_velocity_cmd(np.array(waypoint), .1, (0., 2.5), (-300., 300.))
        assert physical_command(action, CONTROL)["stop"] is stopped
    with pytest.raises(ValueError):
        physical_command({"action_args": {"linear_velocity": float("nan")}}, CONTROL)


def test_predicted_grid_centers_and_not_visible_are_distinct():
    points = predicted_points("<apos_1178><opos_0><act_l0_6>", 470, 280)
    assert points["apos"]["grid"] == dict(row=24, col=25, cell_id=1178)
    np.testing.assert_allclose(points["apos"]["pixel"], [25.5 / 48 * 470, 24.5 / 27 * 280])
    assert points["opos"]["state"] == "not_visible"
    assert points["opos"]["pixel"] is None
    assert points["opos"]["grid"] is None
    assert predicted_points("", 470, 280)["opos"]["state"] == "none"


def test_model_directives_are_preserved_without_invented_pixels():
    result = predicted_points("<apos_1297><opos_1>", 470, 280)
    assert result["apos"]["state"] == "rot_left"
    assert result["apos"]["pixel"] is None
    assert result["apos"]["grid"] is None
    assert result["opos"]["clamped"]


def test_prediction_and_terminal_frames_keep_identical_video_dimensions(tmp_path):
    Image.new("RGB", (470, 280), (110, 100, 90)).save(tmp_path / "rgb.png")
    cfg = dict(episode_id="test", instruction="Walk down the steps. Turn right and stop inside the door.",
               camera=dict(width=470, height=280, hfov_deg=120, position_body_native=[0, .88, 0]))
    frame = dict(frame_id=0, rgb="rgb.png", goal_distance_m=2.0)
    prediction = dict(waypoints_ego=[[.2 * i, 0., 0.] for i in range(1, 11)],
                      executed_command=dict(stop=False), response=dict(latency_ms=10),
                      rvq_codes=[1, 2, 3], raw_text="<apos_1178><opos_0>",
                      **predicted_points("<apos_1178><opos_0>", 470, 280))
    actual = render_prediction_frame(tmp_path, cfg, frame, prediction)
    terminal = render_prediction_frame(tmp_path, cfg, frame, None)
    assert actual.size == terminal.size
    assert actual.width % 2 == actual.height % 2 == 0
    assert not np.array_equal(actual, terminal)
