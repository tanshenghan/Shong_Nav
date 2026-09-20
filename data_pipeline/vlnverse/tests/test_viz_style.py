"""World GT positions must not be mistaken for per-step displacement tokens."""

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.viz_style import (
    future_local_ribbon_points,
    reference_body_velocity,
    validate_camera,
)


def test_future_points_are_current_body_offsets_without_cumulative_sum():
    points = future_local_ribbon_points([[4, 7, 0], [5, 7, 0], [6, 8, 0]], [0, 0, 0], 0)
    np.testing.assert_allclose(points, [[1, 0, 0], [2, 1, 0]])


def test_world_translation_and_rotation_preserve_local_ribbon():
    positions = np.array([[1, 2, 0], [2, 2, 0], [3, 3, 0]], dtype=float)
    expected = future_local_ribbon_points(positions, [0, 0, 0], 0)
    rotated = positions.copy()
    rotated[:, 0], rotated[:, 1] = -positions[:, 1] + 5, positions[:, 0] - 8
    actual = future_local_ribbon_points(rotated, [np.pi/2]*3, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_future_tail_is_not_padded_and_yaw_only_does_not_move_positions():
    positions = [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    yaws = [0, np.pi/2, np.pi/2]
    np.testing.assert_allclose(future_local_ribbon_points(positions, yaws, 0), [[0, 0, 0], [0, 1, 0]])
    assert future_local_ribbon_points(positions, yaws, 0, 1).shape == (1, 3)
    assert future_local_ribbon_points(positions, yaws, 2).shape == (0, 3)


def test_nonplanar_reference_is_rejected_by_flat_camera_projection():
    with pytest.raises(ValueError, match="planar"):
        future_local_ribbon_points([[0, 0, 0], [1, 0, .1]], [0, 0], 0)


def test_reference_velocity_uses_body_rotation_not_playback_fps():
    np.testing.assert_allclose(reference_body_velocity([-.1, .4, 0], np.pi/2, .3), [.4, .1, .3])


def test_shared_viz_rejects_pitched_or_already_annotated_rgb():
    with pytest.raises(ValueError, match="horizontal"):
        validate_camera({"pitch_down_deg": 20}, [{}])
    with pytest.raises(ValueError, match="clean"):
        validate_camera({"pitch_down_deg": 0}, [{"future_path": {"enabled": True}}])
    validate_camera({"pitch_down_deg": 0}, [{"future_path": {"enabled": False}}])
