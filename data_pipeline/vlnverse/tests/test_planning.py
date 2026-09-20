"""Spatial planning certificates, endpoint headings, and route continuity."""

import math

import numpy as np
import pytest

from data_pipeline.vlnverse.planning import PlanningConfig, certify_bezier, plan_route
from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid


def _empty_grid():
    return OccupancyGrid(np.zeros((50, 70), dtype=bool), .1, (0., 0.))


def _evaluate(control, u):
    weights = np.array([math.comb(5, i) * u**i * (1 - u)**(5 - i) for i in range(6)])
    return weights @ control


def test_open_route_is_curved_c4_and_keeps_ordered_checkpoints_and_headings():
    source = np.array([[1., 1.], [2., 1.], [3., 2.], [4., 3.], [5., 3.]])
    result = plan_route(_empty_grid(), source, 0., 0., waypoint_indices=[0, 2, 4])
    np.testing.assert_allclose(result.bezier_xy[0, 0], source[0], atol=1e-10)
    np.testing.assert_allclose(result.bezier_xy[-1, -1], source[-1], atol=1e-10)
    assert np.linalg.norm(result.guide_xy - source[2], axis=1).min() < 1e-10
    h = np.diff(result.parameter_knots)
    for order in range(1, 5):
        derivative = np.diff(result.bezier_xy, n=order, axis=1) * math.factorial(5) / math.factorial(5 - order)
        derivative /= h[:, None, None]**order
        np.testing.assert_allclose(derivative[:-1, -1], derivative[1:, 0], atol=2e-8)
        if order == 1:
            np.testing.assert_allclose(derivative[[0, -1], [0, -1]], [[1., 0.], [1., 0.]], atol=1e-10)
        if order == 2:
            np.testing.assert_allclose(derivative[[0, -1], [0, -1]], 0., atol=1e-9)
    midpoint = _evaluate(result.bezier_xy[0], .5)
    chord_midpoint = result.bezier_xy[0, [0, -1]].mean(axis=0)
    assert np.linalg.norm(midpoint - chord_midpoint) > .01
    assert result.metadata["certificate"]["certified"]
    assert result.metadata["certificate"]["min_parameter_speed_lower_bound"] > 0


def test_wall_cannot_be_crossed_by_grid_edges_or_a_spline_shortcut():
    occupied = np.zeros((50, 70), dtype=bool)
    occupied[:25, 30] = True
    grid = OccupancyGrid(occupied, .1, (0., 0.))
    source = np.array([[1., 1.], [2., 3.3], [4., 3.3], [5., 1.]])
    result = plan_route(grid, source, np.pi / 2, -np.pi / 2, waypoint_indices=[0, 1, 2, 3])
    assert grid.path_is_free(result.astar_xy, .30)
    assert result.astar_xy[:, 1].max() > 2.8
    sampled = np.vstack([[_evaluate(piece, u) for u in np.linspace(0, 1, 101)] for piece in result.bezier_xy])
    assert grid.path_is_free(sampled, .30)  # Independent smoke check, not the certificate.


def test_narrow_source_corridor_does_not_allow_an_unrelated_detour():
    occupied = np.zeros((30, 70), dtype=bool)
    occupied[:, 30] = True
    grid = OccupancyGrid(occupied, .1, (0., 0.))
    with pytest.raises(ValueError, match="no collision-free A\\* route"):
        plan_route(grid, [[1., 1.5], [5., 1.5]], 0., 0., waypoint_indices=[0, 1])


def test_curve_certificate_checks_between_safe_endpoints():
    occupied = np.zeros((50, 70), dtype=bool)
    occupied[23:27, 28:32] = True
    grid = OccupancyGrid(occupied, .1, (0., 0.))
    control = np.linspace([1., 2.5], [5., 2.5], 6)[None]
    assert grid.path_is_free([control[0, 0]], .30)
    assert grid.path_is_free([control[0, -1]], .30)
    with pytest.raises(ValueError, match="cannot certify"):
        certify_bezier(grid, control, [0., 4.], [[1., 2.5], [5., 2.5]], max_depth=12)


def test_curve_certificate_rejects_corridor_excursion_and_cusp():
    control = np.array([[1., 1.], [2., 1.], [2., 4.], [4., 4.], [4., 1.], [5., 1.]])[None]
    with pytest.raises(ValueError, match="cannot certify"):
        certify_bezier(_empty_grid(), control, [0., 4.], [[1., 1.], [5., 1.]], max_depth=12)
    constant = np.repeat(np.array([[[2., 2.]]]), 6, axis=1)
    with pytest.raises(ValueError, match="cannot certify"):
        certify_bezier(_empty_grid(), constant, [0., 1.], [[1., 2.], [3., 2.]], max_depth=10)


def test_certificate_rejects_a_jump_between_individually_free_pieces():
    controls = np.array([np.linspace([1., 2.], [2., 2.], 6), np.linspace([3., 2.], [4., 2.], 6)])
    with pytest.raises(ValueError, match="continuous C4"):
        certify_bezier(_empty_grid(), controls, [0., 1., 2.], [[1., 2.], [4., 2.]])


def test_world_rotation_and_nonzero_endpoint_headings():
    yaw, origin = .53, np.array([-4., 8.])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    source = np.array([[1., 1.], [3., 1.], [5., 1.]]) @ rotation.T + origin
    grid = OccupancyGrid(np.zeros((30, 70), dtype=bool), .1, tuple(origin), yaw)
    result = plan_route(grid, source, yaw, yaw, waypoint_indices=[0, 1, 2])
    assert grid.path_is_free(result.astar_xy, .30)
    tangent = 5 * np.diff(result.bezier_xy, axis=1) / np.diff(result.parameter_knots)[:, None, None]
    np.testing.assert_allclose(tangent[0, 0], rotation[:, 0], atol=1e-10)
    np.testing.assert_allclose(tangent[-1, -1], rotation[:, 0], atol=1e-10)


@pytest.mark.parametrize("indices", [[1, 4], [0, 4, 2], [0, 2], [0., 4.], [0, 2, 2, 4],
                                      np.array([0, 3, 2, 4], dtype=np.uint64)])
def test_invalid_checkpoint_indices_are_rejected(indices):
    with pytest.raises(ValueError, match="waypoint_indices"):
        plan_route(_empty_grid(), [[1., 1.], [2., 1.], [3., 1.], [4., 1.], [5., 1.]],
                   0., 0., waypoint_indices=indices)


@pytest.mark.parametrize("kwargs", [
    {"footprint_radius_m": .29}, {"corridor_half_width_m": 0.},
    {"clearance_cost_weight": np.nan}, {"max_refinements": 0}, {"certificate_max_depth": True},
])
def test_invalid_planning_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PlanningConfig(**kwargs)
