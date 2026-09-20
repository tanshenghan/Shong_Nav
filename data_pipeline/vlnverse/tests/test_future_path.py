"""CPU-only ground validation, future-index and camera-projection contracts."""

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.future_path import (
    FuturePathConfig,
    camera_forward,
    future_indices,
    line_mesh,
    project_to_ground,
    validate_structural_cap,
)


def test_future_horizon_excludes_current_and_shortens_at_end():
    assert future_indices(5, 111, 30) == list(range(6, 36))
    assert future_indices(108, 111, 30) == [109, 110]
    assert future_indices(110, 111, 30) == []
    assert future_indices(0, 111, 0) == []


@pytest.mark.parametrize("current,count,horizon", [(-1, 111, 30), (111, 111, 30), (0, 0, 30), (0, 1, -1)])
def test_future_horizon_rejects_invalid_indices(current, count, horizon):
    with pytest.raises(ValueError):
        future_indices(current, count, horizon)


def test_positive_camera_pitch_points_down_in_world_and_preserves_yaw():
    expected = [0., np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20))]
    np.testing.assert_allclose(camera_forward(np.pi / 2, 20), expected, atol=1e-12)
    np.testing.assert_allclose(camera_forward(0, 0), [1, 0, 0])
    assert np.linalg.norm(camera_forward(-1.65, 20)) == pytest.approx(1)


@pytest.mark.parametrize("yaw,pitch", [(np.nan, 0), (0, np.inf), (0, 90), (0, -89)])
def test_invalid_camera_pitch_rejected(yaw, pitch):
    with pytest.raises(ValueError):
        camera_forward(yaw, pitch)


def test_ground_rays_preserve_xy_record_actual_heights_and_all_indices():
    config = FuturePathConfig()
    poses = np.array([[1., 2, 0], [2., 2, 0]])
    calls = []

    def ray(origin, direction, distance):
        calls.append((origin, direction, distance))
        return [origin[0], origin[1], .03], [0, 0, 1], "/Root/Meshes/floor/floor_0000/Mesh"

    ground, hits = project_to_ground(poses, ray, config)
    np.testing.assert_array_equal(ground[:, :2], poses[:, :2])
    np.testing.assert_array_equal(ground[:, 2], [.03, .03])
    assert [hit["pose_index"] for hit in hits] == [0, 1]
    assert [hit["source_z_delta_m"] for hit in hits] == [.03, .03]
    for index, (origin, direction, distance) in enumerate(calls):
        np.testing.assert_array_equal(origin, poses[index] + [0, 0, .3])
        np.testing.assert_array_equal(direction, [0, 0, -1])
        assert distance == pytest.approx(.38)


@pytest.mark.parametrize("hit,message", [
    (None, "no scene ground"),
    (([0, 0, .2], [0, 0, 1], "/Root/Meshes/floor/Mesh"), "height differs"),
    (([0, 0, 0], [1, 0, 0], "/Root/Meshes/floor/Mesh"), "not an upward floor"),
    (([0, 0, 0], [0, 0, -1], "/Root/Meshes/floor/Mesh"), "not an upward floor"),
    (([0, 0, .03], [0, 0, 1], "/Root/Meshes/decor/Mesh"), "identified USD floor"),
    (([0, 0, np.nan], [0, 0, 1], "/Root/Meshes/floor/Mesh"), "invalid scene ground"),
    (([0, .1, 0], [0, 0, 1], "/Root/Meshes/floor/Mesh"), "moved horizontally"),
])
def test_ground_rejects_missing_nonfloor_or_unreliable_first_hit(hit, message):
    with pytest.raises(ValueError, match=message):
        project_to_ground([[0., 0, 0]], lambda *_: hit, FuturePathConfig())


def test_ribbon_connects_current_to_future_without_in_place_degenerate_faces():
    points = np.array([[0., 0, .02], [1., 0, .02], [1., 0, .02], [1., 1, .03]])
    vertices, faces = line_mesh(points, .04, .012)
    assert faces == [(0, 1, 2, 3), (4, 5, 6, 7)]
    np.testing.assert_allclose(vertices[:4], [[0, -.02, .032], [1, -.02, .032],
                                             [1, .02, .032], [0, .02, .032]])
    assert np.isfinite(vertices).all()
    assert line_mesh(points[-1:], .04, .012) == ([], [])


def test_flat_wall_named_ground_cap_requires_identified_floor_below():
    evidence = validate_structural_cap([1, 2, -.000005], 0, [-.000005, -.000005],
        ([1, 2, -.05], [0, 0, 1], "/Root/Meshes/floor/slab"), FuturePathConfig())
    assert evidence["supporting_floor_depth_m"] == pytest.approx(.049995)


@pytest.mark.parametrize("bounds,hit", [
    ([0, .2], ([0, 0, -.05], [0, 0, 1], "/Root/Meshes/floor/slab")),
    ([.03, .03], ([0, 0, -.05], [0, 0, 1], "/Root/Meshes/floor/slab")),
    ([0, 0], None),
    ([0, 0], ([0, 0, -.2], [0, 0, 1], "/Root/Meshes/floor/slab")),
    ([0, 0], ([0, 0, -.05], [0, 0, 1], "/Root/Meshes/furniture/top")),
])
def test_wall_ground_cap_rejects_vertical_raised_unsupported_or_furniture_surface(bounds, hit):
    with pytest.raises(ValueError):
        validate_structural_cap([0, 0, 0], 0, bounds, hit, FuturePathConfig())


@pytest.mark.parametrize("values", [dict(horizon_steps=0), dict(horizon_steps=True),
                                     dict(horizon_steps=2.5), dict(line_width_m=np.nan),
                                     dict(max_ground_source_delta_m=.5)])
def test_invalid_overlay_settings_rejected(values):
    with pytest.raises(ValueError):
        FuturePathConfig(**values)
