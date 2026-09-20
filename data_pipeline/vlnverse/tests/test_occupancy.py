"""Whole-segment collision guarantees and explicit map coordinate contracts."""

import numpy as np
import pytest

from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid


def test_wall_blocks_segment_with_free_endpoints():
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[:, 5] = True
    grid = OccupancyGrid(occupied, 0.1, (0.0, 0.0))
    start, end = [0.2, 0.45], [0.8, 0.45]
    assert grid.path_is_free([start], 0.01)
    assert grid.path_is_free([end], 0.01)
    assert not grid.is_segment_free(start, end, 0.01)
    assert not grid.is_segment_free(end, start, 0.01)


def test_clearance_and_map_exterior_include_robot_radius():
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[5, 5] = True
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    assert grid.is_segment_free([2.0, 4.5], [8.0, 4.5], 0.49)
    assert not grid.is_segment_free([2.0, 4.5], [8.0, 4.5], 0.5)
    assert not grid.is_segment_free([0.2, 2.0], [3.0, 2.0], 0.3)
    assert not grid.path_is_free([[10.0, 2.0]], 0.0)
    assert not grid.path_is_free([[-0.01, 2.0]], 0.0)
    assert not grid.path_is_free([[0.0, 2.0]], 0.0)
    assert not grid.path_is_free([[5.0, 2.0]], 6.0)


def test_rotated_translated_map_uses_rows_for_local_y():
    occupied = np.zeros((4, 7), dtype=bool)
    occupied[1, 3] = True
    grid = OccupancyGrid(occupied, 0.5, (10.0, -2.0), np.pi / 2)
    # local x -> world +y; local y -> world -x.
    assert not grid.is_segment_free([9.25, -1.5], [9.25, 1.0], 0.01)
    assert grid.is_segment_free([8.75, -1.5], [8.75, 1.0], 0.01)
    assert not grid.path_is_free([[9.25, -0.25]], 0.0)


def test_point_on_obstacle_edge_and_corner_are_blocked():
    occupied = np.zeros((6, 6), dtype=bool)
    occupied[2, 2] = True
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    assert not grid.path_is_free([[3.0, 2.5]], 0.0)
    assert not grid.path_is_free([[3.0, 3.0]], 0.0)
    assert not grid.path_is_free([[2.5, 2.5]], 0.0)
    assert grid.path_is_free([[3.1, 3.1]], 0.0)
    # Euclidean corner clearance is sqrt(2) * .09 > .1, unlike square inflation.
    assert grid.path_is_free([[3.09, 3.09]], 0.1)
    tangent = 3.0 + 0.1 / np.sqrt(2)
    assert not grid.path_is_free([[tangent, tangent]], 0.1)
    assert grid.path_is_free([[tangent + 1e-7, tangent + 1e-7]], 0.1)


@pytest.mark.parametrize("yaw", [0.0, 0.37, np.pi / 2, -2.4])
def test_swept_disc_checks_interior_corner_clearance_in_both_directions(yaw):
    occupied = np.zeros((8, 8), dtype=bool)
    occupied[3, 3] = True
    origin = np.array([10.0, -7.0])
    grid = OccupancyGrid(occupied, 1.0, tuple(origin), yaw)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    # Line x+y=5.7 passes below the corner (3, 3); minimum clearance .3/sqrt(2)
    # occurs halfway along the segment, far from its individually safe endpoints.
    start, end = np.array([[2.0, 3.7], [3.7, 2.0]]) @ rotation.T + origin
    tangent_radius = 0.3 / np.sqrt(2)
    assert grid.path_is_free([start], 0.3)
    assert grid.path_is_free([end], 0.3)
    for first, second in ((start, end), (end, start)):
        assert grid.is_segment_free(first, second, 0.2)
        assert not grid.is_segment_free(first, second, tangent_radius)
        assert not grid.is_segment_free(first, second, 0.3)


def test_segment_endpoint_is_closest_to_rectangle_edge():
    occupied = np.zeros((8, 8), dtype=bool)
    occupied[3, 3] = True
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    # The nearest obstacle point is inside its left edge, not a cell vertex.
    assert grid.is_segment_free([1.0, 2.5], [2.75, 3.5], 0.249)
    assert not grid.is_segment_free([1.0, 2.5], [2.75, 3.5], 0.25)


def test_diagonal_segment_crosses_obstacle_without_nearby_endpoints():
    occupied = np.zeros((8, 8), dtype=bool)
    occupied[3, 3] = True
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    assert not grid.is_segment_free([1.0, 1.0], [7.0, 7.0], 0.0)
    # A segment tangent to the closed corner also collides at zero radius.
    assert not grid.is_segment_free([2.0, 4.0], [4.0, 2.0], 0.0)


def test_polyline_checks_all_segments_and_does_not_check_only_bounding_box():
    occupied = np.zeros((8, 8), dtype=bool)
    occupied[2, 5] = True
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    # Obstacle is inside the overall AABB but well off the diagonal.
    assert grid.is_segment_free([1.0, 1.0], [7.0, 7.0], 0.1)
    assert not grid.path_is_free([[1.0, 1.0], [1.0, 2.5], [7.0, 2.5]], 0.1)


def test_constructor_copies_map_and_npz_preserves_contract(tmp_path):
    occupied = np.zeros((6, 6), dtype=bool)
    grid = OccupancyGrid(occupied, 0.5, (2.0, 3.0))
    occupied[:] = True
    assert not grid.occupied.any()
    assert not grid.occupied.flags.writeable
    path = tmp_path / "canonical.npz"
    np.savez(path, occupied=occupied, resolution_m=0.5, origin_xy_m=[2.0, 3.0])
    loaded = OccupancyGrid.from_npz(path)
    assert loaded.grid_yaw_rad == 0.0
    assert loaded.origin_xy_m == (2.0, 3.0)
    assert not loaded.path_is_free([[3.0, 4.0]], 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"occupied": np.zeros((0, 2), dtype=bool)},
        {"occupied": np.zeros((2, 2), dtype=np.uint8)},
        {"occupied": np.zeros((2, 2, 2), dtype=bool)},
        {"resolution_m": 0.0},
        {"resolution_m": np.nan},
        {"resolution_m": [1.0]},
        {"resolution_m": 1e308},
        {"origin_xy_m": (0.0, np.inf)},
        {"origin_xy_m": (0.0,)},
        {"grid_yaw_rad": np.inf},
    ],
)
def test_malformed_grid_is_rejected(kwargs):
    arguments = {
        "occupied": np.zeros((3, 3), dtype=bool),
        "resolution_m": 1.0,
        "origin_xy_m": (0.0, 0.0),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        OccupancyGrid(**arguments)


@pytest.mark.parametrize("radius", [-0.1, np.nan, np.inf, [0.1]])
def test_invalid_radius_is_rejected(radius):
    grid = OccupancyGrid(np.zeros((4, 4), dtype=bool), 1.0, (0.0, 0.0))
    with pytest.raises(ValueError):
        grid.is_segment_free([1.0, 1.0], [2.0, 2.0], radius)


@pytest.mark.parametrize("path", [[], [[1, 2, 3]], [[1, np.nan]], [1, 2]])
def test_malformed_path_is_rejected(path):
    grid = OccupancyGrid(np.zeros((4, 4), dtype=bool), 1.0, (0.0, 0.0))
    with pytest.raises(ValueError):
        grid.path_is_free(path, 0.1)


def test_npz_requires_explicit_transform_fields(tmp_path):
    path = tmp_path / "uncalibrated.npz"
    np.savez(path, occupied=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="missing fields"):
        OccupancyGrid.from_npz(path)
