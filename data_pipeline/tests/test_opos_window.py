"""Frame-window boundaries and independent depth/visibility supervision."""

import numpy as np
import pytest

from data_pipeline.pointing import PointingConfig, attach_pixel_grid, terminal_ground_point


@pytest.mark.parametrize("count,start", [(1, 0), (2, 1), (3, 2), (4, 2), (59, 39), (60, 40), (61, 40)])
def test_last_third_rounds_up_the_number_of_allowed_frames(count, start):
    assert PointingConfig(opos_visibility_window="last_third").opos_visibility_start_frame(count) == start
    assert PointingConfig(opos_visibility_window="all").opos_visibility_start_frame(count) == 0


def scene():
    positions = np.zeros((60, 3))
    positions[-1, 2] = -2.
    camera = np.diag([1., -1., -1., 1.])
    camera[1, 3] = .88
    K = np.array([[50., 0., 50.], [0., 50., 50.], [0., 0., 1.]])
    return positions, camera, K, np.full((100, 100), 3.), np.ones((100, 100), bool)


def test_visible_goal_is_suppressed_until_boundary_with_its_true_target_retained():
    args = scene()
    historical = PointingConfig.from_metadata({"opos_mode": "terminal_ground", "tail_policy": "terminal",
                                               "unknown_token_id": 0})
    original = terminal_ground_point(*args, historical, frame_index=29)
    assert original["state"] == "point"
    early = attach_pixel_grid(terminal_ground_point(*args, PointingConfig(), frame_index=29), 100, 100)
    assert early["state"] == "not_visible" and early["token_id"] == 0 and early["loss_valid"]
    assert early["reason"] == "before_opos_visibility_window"
    assert early["pixel"] is early["pixel_rc"] is early["grid"] is early["world_point"] is None
    assert early["target_world_point"] == original["target_world_point"] == [0., 0., -2.]
    assert early["target_frame_id"] == 59
    allowed = terminal_ground_point(*args, PointingConfig(), frame_index=30)
    assert allowed == original


@pytest.mark.parametrize("failure,reason,valid", [
    ("occluded", "stop_location_occluded", True),
    ("behind", "stop_location_outside_view", True),
    ("invalid_depth", "stop_location_depth_unknown", False),
])
def test_last_half_still_requires_actual_geometric_visibility(failure, reason, valid):
    positions, camera, K, depth, depth_valid = scene()
    if failure == "occluded":
        depth[:] = 1.
    elif failure == "behind":
        camera[2, 3] = -3.
    else:
        depth_valid[:] = False
    point = terminal_ground_point(positions, camera, K, depth, depth_valid, PointingConfig(), frame_index=30)
    assert point["reason"] == reason and point["loss_valid"] == valid
    assert point["token_id"] == 0 and point["pixel"] is None
    assert point["target_world_point"] == [0., 0., -2.]


@pytest.mark.parametrize("frame_index", [-1, 60, 40.5, True, None])
def test_invalid_frame_index_cannot_bypass_temporal_restriction(frame_index):
    with pytest.raises(ValueError, match="frame_index"):
        terminal_ground_point(*scene(), PointingConfig(), frame_index=frame_index)


@pytest.mark.parametrize("count,start", [(1, 0), (2, 1), (3, 1), (4, 2), (59, 29), (60, 30), (61, 30)])
def test_default_last_half_rounds_up_allowed_frames(count, start):
    assert PointingConfig().opos_visibility_start_frame(count) == start
