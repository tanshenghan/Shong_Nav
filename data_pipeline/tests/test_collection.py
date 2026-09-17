"""Coordinate/action alignment checks independent of Habitat and GPU rendering."""

import numpy as np
import pytest

from data_pipeline.geometry import (
    encode_rvq,
    future_targets,
    pinhole_intrinsics,
    route_deviation,
)
from lightnav.traj_vocab import RVQBundle


def body_pose(x, z, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pose[:3, 3] = [x, 0, z]
    return pose


def test_forward_then_stationary_left_turn_then_forward():
    poses = np.array([body_pose(0, 0, 0), body_pose(0, -1, 0),
                      body_pose(0, -1, np.pi / 2), body_pose(-1, -1, np.pi / 2)])
    result = future_targets(poses, 3)
    np.testing.assert_allclose(result["waypoints"][0],
                               [[1, 0, 0], [1, 0, np.pi / 2], [1, 1, np.pi / 2]], atol=1e-6)
    np.testing.assert_allclose(result["se2_diff"][0],
                               [[1, 0, 0], [0, 0, np.pi / 2], [1, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(result["waypoints"][2, 0], [1, 0, 0], atol=1e-6)
    assert result["valid"].tolist() == [[True, True, True], [True, True, False],
                                      [True, False, False], [False, False, False]]


def test_right_turn_and_yaw_wrap():
    poses = np.array([body_pose(0, 0, np.deg2rad(-175)),
                      body_pose(0, 0, np.deg2rad(170))])
    target = future_targets(poses, 30)
    assert target["se2_diff"][0, 0, 2] == pytest.approx(np.deg2rad(-15))
    assert target["valid"].sum() == 1


def test_30_horizon_and_10_horizon_share_exact_prefix():
    poses = np.array([body_pose(0, -i * .25, 0) for i in range(52)])
    long = future_targets(poses, 30)
    short = future_targets(poses, 10)
    assert long["valid"].all(axis=1).sum() == 22
    assert short["valid"].all(axis=1).sum() == 42
    np.testing.assert_array_equal(long["se2_diff"][:, :10], short["se2_diff"])
    assert long["frame_ids"][30, :21].tolist() == list(range(31, 52))
    assert (long["frame_ids"][30, 21:] == -1).all()


def test_world_rotation_and_translation_do_not_change_targets():
    poses = np.array([body_pose(0, 0, 0), body_pose(0, -1, .2), body_pose(-.2, -2, .4)])
    transform = body_pose(7, 18, 1.3)
    a = future_targets(poses, 2)
    b = future_targets(transform[None] @ poses, 2)
    np.testing.assert_allclose(a["waypoints"], b["waypoints"], atol=1e-6)


def test_camera_geometry_for_requested_resolution():
    k = pinhole_intrinsics(470, 280, 120)
    np.testing.assert_allclose(k[:2, 2], [235, 140])
    assert k[0, 0] == pytest.approx(470 / (2 * np.sqrt(3)))
    assert k[0, 0] == k[1, 1]
    with pytest.raises(ValueError):
        pinhole_intrinsics(470, 280, 180)


def test_encoder_excludes_stop_and_respects_weighted_features(tmp_path):
    h = 10
    book = np.zeros((3, h * 3), np.float32)
    book[1].reshape(h, 3)[:, 0] = .5  # deweighted by 2 -> 0.25 m increments
    book[2].reshape(h, 3)[:, 2] = -.2
    bundle = RVQBundle(
        path=tmp_path, manifest={}, levels=[3, 3, 3], horizon=h,
        representation="se2_diff", objective="ade_v1",
        encode={"type": "ade", "heading_weight": .3}, feature_space="weighted_diff",
        codebooks=[book, np.zeros_like(book), np.zeros_like(book)],
        jacobian_weights=np.full(h * 3, 2), alpha={}, stop_l0=0,
    )
    target = np.zeros((h, 3), np.float32)
    target[:, 0] = np.arange(1, h + 1) * .25
    codes, reconstruction, error = encode_rvq(target, bundle)
    assert codes[0] == 1
    assert error < 1e-6
    np.testing.assert_allclose(reconstruction, target)
    zero_codes, _, _ = encode_rvq(np.zeros_like(target), bundle)
    assert zero_codes[0] != bundle.stop_l0


def test_polyline_diagnostic_uses_segments_and_vertical_distance():
    distance = route_deviation([[.5, 0, 0], [.5, .3, 0], [2, 0, 0]],
                               [[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(distance, [0, .3, 1])
