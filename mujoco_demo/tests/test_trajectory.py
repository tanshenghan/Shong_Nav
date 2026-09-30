import math

import mujoco
import numpy as np
import pytest
from vln_mujoco.trajectory import add_ground_path, ground_path


def test_ground_path_uses_capture_pose_and_metric_scale():
    points = ground_path([(1, 0, 0), (2, 1, 0)], (5, 7, math.pi / 2))
    np.testing.assert_allclose(points, [[5, 8, .012], [4, 9, .012]])
    assert ground_path([], (0, 0, 0)).shape == (0, 3)
    with pytest.raises(ValueError, match='finite'):
        ground_path([(float('nan'), 0, 0)], (0, 0, 0))


def test_visual_geometries_are_world_ground_positions():
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    scene = mujoco.MjvScene(model, maxgeom=10)
    points = ground_path([(1, 0, 0), (2, 0, 0)], (5, 7, 0))
    add_ground_path(scene, points)
    assert scene.ngeom == 3  # two waypoints, one connector
    np.testing.assert_allclose(scene.geoms[0].pos, [6, 7, .012])
    np.testing.assert_allclose(scene.geoms[1].pos, [7, 7, .012])
