"""Cost-aware grid search and certified C4 quintic route interpolation.

The reference corridor and ordered checkpoints constrain route geometry; they
are not an automatic proof of language-instruction agreement. Collision proofs
cover the supplied planar raster and circular footprint, not scene dynamics.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass

import numpy as np

from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid


@dataclass(frozen=True)
class PlanningConfig:
    footprint_radius_m: float = 0.30
    corridor_half_width_m: float = 0.50
    clearance_cost_weight: float = 2.0
    clearance_cost_range_m: float = 0.80
    max_guide_spacing_m: float = 2.0
    max_refinements: int = 5
    certificate_max_depth: int = 18
    max_expanded_nodes: int = 100_000

    def __post_init__(self):
        for name in (
            "footprint_radius_m", "corridor_half_width_m", "clearance_cost_weight",
            "clearance_cost_range_m", "max_guide_spacing_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if value < 0 or (value == 0 and name != "clearance_cost_weight"):
                raise ValueError(f"{name} must be positive")
        if self.footprint_radius_m < 0.30:
            raise ValueError("the planning footprint must retain at least the 0.30 m envelope")
        for name in ("max_refinements", "certificate_max_depth", "max_expanded_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class PlannedRoute:
    bezier_xy: np.ndarray
    parameter_knots: np.ndarray
    astar_xy: np.ndarray
    guide_xy: np.ndarray
    metadata: dict


def _points(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (2,) or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [N, 2] array with N >= 2")
    return array


def _reference_segments(reference):
    delta = np.diff(reference, axis=0)
    keep = np.linalg.norm(delta, axis=1) > 1e-10
    if not keep.any():
        raise ValueError("each checkpoint interval must contain nonzero translation")
    return reference[:-1][keep], delta[keep]


def _distances_to_segments(points, starts, delta):
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.clip(np.sum(relative * delta, axis=2) / np.sum(delta * delta, axis=1), 0., 1.)
    difference = relative - fractions[:, :, None] * delta
    return np.linalg.norm(difference, axis=2)


def _hull_in_corridor(points, reference, width):
    starts, delta = reference
    # Every control point inside ONE segment's convex capsule implies its entire
    # convex hull is there. Membership in a union of capsules alone is not enough.
    return bool(np.any(np.max(_distances_to_segments(points, starts, delta), axis=0) <= width - 1e-10))


def _line_in_corridor(start, end, reference, width, depth=0):
    if _hull_in_corridor(np.array([start, end]), reference, width):
        return True
    if depth >= 16:
        return False
    midpoint = (start + end) / 2
    return (_line_in_corridor(start, midpoint, reference, width, depth + 1)
            and _line_in_corridor(midpoint, end, reference, width, depth + 1))


def _split_bezier(control):
    levels = [control]
    while len(levels[-1]) > 1:
        levels.append((levels[-1][:-1] + levels[-1][1:]) / 2)
    return np.array([level[0] for level in levels]), np.array([level[-1] for level in levels[::-1]])


def certify_bezier(
    grid, bezier_xy, parameter_knots, reference_xy, *, radius_m=0.30,
    corridor_half_width_m=0.50, max_depth=18,
):
    """Certify complete curves with recursive convex-hull enclosures.

    Each accepted leaf is contained in its endpoint chord's capsule of radius
    ``max_control_to_chord_distance``. Checking that entire chord with this extra
    radius proves the full curved swept footprint. A separate convex-capsule
    proof bounds route deviation, and derivative hulls prove nonzero speed.
    Failure to certify at the depth limit raises; sparse samples never substitute
    for any of these three certificates.
    """
    control = np.asarray(bezier_xy, dtype=float)
    knots = np.asarray(parameter_knots, dtype=float)
    if (control.ndim != 3 or control.shape[1:] != (6, 2) or not len(control)
            or not np.isfinite(control).all() or knots.shape != (len(control) + 1,)
            or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all()):
        raise ValueError("invalid quintic Bezier controls or parameter knots")
    for name, value in (("radius_m", radius_m), ("corridor_half_width_m", corridor_half_width_m)):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if radius_m < .30:
        raise ValueError("the certificate must retain at least the 0.30 m envelope")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    for order in range(5):
        derivative = np.diff(control, n=order, axis=1) * math.factorial(5) / math.factorial(5 - order)
        derivative /= np.diff(knots)[:, None, None]**order
        if not np.allclose(derivative[:-1, -1], derivative[1:, 0], rtol=1e-7, atol=1e-8):
            raise ValueError("Bezier pieces do not form a continuous C4 curve in the global parameter")
    reference = _reference_segments(_points(reference_xy, "reference_xy"))
    leaf_count, deepest, maximum_enclosure = 0, 0, 0.
    minimum_derivative = np.inf
    stack = [(piece, float(duration), 0) for piece, duration in zip(control, np.diff(knots), strict=True)]
    while stack:
        piece, duration, depth = stack.pop()
        start, end = piece[0], piece[-1]
        chord = end - start
        length = np.linalg.norm(chord)
        enclosure = float(_distances_to_segments(piece, start[None, :], chord[None, :]).max()) \
            if length > 1e-12 else float(np.linalg.norm(piece - start, axis=1).max())
        derivative_projection = float(np.min(5 * np.diff(piece, axis=0) @ (chord / length) / duration)) \
            if length > 1e-12 else 0.
        if (derivative_projection > 1e-8
                and _hull_in_corridor(piece, reference, corridor_half_width_m)
                and grid.is_segment_free(start, end, radius_m + enclosure)):
            leaf_count += 1
            deepest = max(deepest, depth)
            maximum_enclosure = max(maximum_enclosure, enclosure)
            minimum_derivative = min(minimum_derivative, derivative_projection)
            continue
        if depth >= max_depth:
            raise ValueError("cannot certify curved footprint, route corridor, or nonzero tangent")
        left, right = _split_bezier(piece)
        stack.extend(((right, duration / 2, depth + 1), (left, duration / 2, depth + 1)))
    return dict(
        certified=True, method="recursive_bezier_hull_capsules",
        footprint_radius_m=float(radius_m), corridor_half_width_m=float(corridor_half_width_m),
        leaves=leaf_count, max_depth=deepest, max_leaf_enclosure_radius_m=maximum_enclosure,
        min_parameter_speed_lower_bound=float(minimum_derivative),
        planar_raster_only=True,
    )


class _GridSearch:
    def __init__(self, grid, config):
        self.grid, self.config = grid, config
        angle = grid.grid_yaw_rad
        self.rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.origin = np.asarray(grid.origin_xy_m)
        self.shape = grid.occupied.shape
        self.clearance_cache = {}
        self.edge_cache = {}

    def world(self, node):
        row, col = divmod(node, self.shape[1])
        local = (np.array([col, row]) + .5) * self.grid.resolution_m
        return local @ self.rotation.T + self.origin

    def clearance(self, node):
        if node in self.clearance_cache:
            return self.clearance_cache[node]
        row, col = divmod(node, self.shape[1])
        resolution = self.grid.resolution_m
        point = (np.array([col, row]) + .5) * resolution
        extent = np.array(self.shape[::-1]) * resolution
        distance_cap = max(self.config.clearance_cost_range_m, self.config.footprint_radius_m + resolution)
        reach = int(np.ceil(distance_cap / resolution)) + 1
        r0, r1 = max(0, row - reach), min(self.shape[0], row + reach + 1)
        c0, c1 = max(0, col - reach), min(self.shape[1], col + reach + 1)
        rows, cols = np.nonzero(self.grid.occupied[r0:r1, c0:c1])
        value = min(float(np.min(np.r_[point, extent - point])), distance_cap)
        if len(rows):
            lower = np.column_stack([cols + c0, rows + r0]) * resolution
            difference = np.maximum(np.maximum(lower - point, point - lower - resolution), 0.)
            value = min(value, float(np.linalg.norm(difference, axis=1).min()))
        self.clearance_cache[node] = value
        return value

    def allowed_nodes(self, source):
        local = (source - self.origin) @ self.rotation
        width, resolution = self.config.corridor_half_width_m, self.grid.resolution_m
        low = np.maximum(np.floor((local.min(axis=0) - width) / resolution).astype(int), 0)
        high = np.minimum(np.ceil((local.max(axis=0) + width) / resolution).astype(int), self.shape[::-1])
        rows, cols = np.mgrid[low[1]:high[1], low[0]:high[0]]
        nodes = (rows * self.shape[1] + cols).ravel()
        world = (np.column_stack([cols.ravel(), rows.ravel()]) + .5) * resolution @ self.rotation.T + self.origin
        reference = _reference_segments(source)
        allowed = np.zeros(np.prod(self.shape), dtype=bool)
        for begin in range(0, len(nodes), 2048):
            part = nodes[begin:begin + 2048]
            distance = _distances_to_segments(world[begin:begin + 2048], *reference).min(axis=1)
            allowed[part] = distance < width - 1e-10
        allowed &= ~self.grid.occupied.ravel()
        return allowed, reference

    def connection_nodes(self, point, allowed, reference):
        col, row = np.floor(((point - self.origin) @ self.rotation) / self.grid.resolution_m).astype(int)
        result = {}
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row + dr, col + dc
                if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
                    continue
                node = r * self.shape[1] + c
                if allowed[node] and self.clearance(node) > self.config.footprint_radius_m:
                    target = self.world(node)
                    if (self.grid.is_segment_free(point, target, self.config.footprint_radius_m)
                            and _line_in_corridor(point, target, reference, self.config.corridor_half_width_m)):
                        result[node] = float(np.linalg.norm(target - point))
        if not result:
            raise ValueError("a checkpoint cannot connect to the collision-free search grid")
        return result

    def cost_multiplier(self, node):
        distance = self.clearance(node)
        ratio = max(0., 1 - distance / self.config.clearance_cost_range_m)
        return 1 + self.config.clearance_cost_weight * ratio**2

    def search(self, source):
        allowed, reference = self.allowed_nodes(source)
        start, goal = source[0], source[-1]
        starts = self.connection_nodes(start, allowed, reference)
        goals = self.connection_nodes(goal, allowed, reference)
        distance, parent, queue = {}, {}, []
        for node, length in starts.items():
            cost = length * self.cost_multiplier(node)
            distance[node], parent[node] = cost, None
            heapq.heappush(queue, (cost + np.linalg.norm(self.world(node) - goal), cost, node))
        best_goal, goal_node, expanded = np.inf, None, 0
        while queue:
            estimate, cost, node = heapq.heappop(queue)
            if estimate >= best_goal:
                break
            if cost != distance.get(node):
                continue
            expanded += 1
            if expanded > self.config.max_expanded_nodes:
                raise ValueError("A* exceeded max_expanded_nodes")
            if node in goals:
                candidate = cost + goals[node] * self.cost_multiplier(node)
                if candidate < best_goal:
                    best_goal, goal_node = candidate, node
            row, col = divmod(node, self.shape[1])
            first = self.world(node)
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                r, c = row + dr, col + dc
                if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
                    continue
                target = r * self.shape[1] + c
                if not allowed[target] or self.clearance(target) <= self.config.footprint_radius_m:
                    continue
                second = self.world(target)
                edge = tuple(sorted((node, target)))
                if edge not in self.edge_cache:
                    self.edge_cache[edge] = self.grid.is_segment_free(first, second, self.config.footprint_radius_m)
                if (not self.edge_cache[edge]
                        or not _line_in_corridor(first, second, reference, self.config.corridor_half_width_m)):
                    continue
                length = self.grid.resolution_m * np.hypot(dr, dc)
                candidate = cost + length * (self.cost_multiplier(node) + self.cost_multiplier(target)) / 2
                if candidate < distance.get(target, np.inf):
                    distance[target], parent[target] = candidate, node
                    heapq.heappush(queue, (candidate + np.linalg.norm(second - goal), candidate, target))
        if goal_node is None:
            raise ValueError("no collision-free A* route exists inside the ordered source corridor")
        nodes = []
        while goal_node is not None:
            nodes.append(goal_node)
            goal_node = parent[goal_node]
        points = np.vstack([start, *(self.world(node) for node in nodes[::-1]), goal])
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
        return points, dict(expanded_nodes=expanded, search_cost=float(best_goal))


def _simplify(points, grid, reference, config):
    selected, current = [points[0]], 0
    corridor = _reference_segments(reference)
    while current < len(points) - 1:
        for target in range(len(points) - 1, current, -1):
            if (grid.is_segment_free(points[current], points[target], config.footprint_radius_m)
                    and _line_in_corridor(points[current], points[target], corridor, config.corridor_half_width_m)):
                break
        else:
            raise ValueError("A* route has no safe simplification edge")
        selected.append(points[target])
        current = target
    return np.asarray(selected)


def _quintic_coefficients(points, knots, velocity, acceleration):
    h = np.diff(knots)[:, None]
    c0, c1, c2 = points[:-1], h * velocity[:-1], .5 * h**2 * acceleration[:-1]
    delta = points[1:] - c0 - c1 - c2
    v = h * velocity[1:] - c1 - 2 * c2
    a = h**2 * acceleration[1:] - 2 * c2
    return np.stack([c0, c1, c2, 10*delta - 4*v + .5*a,
                     -15*delta + 7*v - a, 6*delta - 3*v + .5*a], axis=1)


def _quintic_spline(points, start_yaw, end_yaw):
    knots = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    velocity, acceleration = np.zeros_like(points), np.zeros_like(points)
    velocity[[0, -1]] = np.array([[np.cos(start_yaw), np.sin(start_yaw)],
                                  [np.cos(end_yaw), np.sin(end_yaw)]])

    def continuity(coefficients):
        h = np.diff(knots)[:, None]
        third_end = (6*coefficients[:, 3] + 24*coefficients[:, 4] + 60*coefficients[:, 5]) / h**3
        fourth_end = (24*coefficients[:, 4] + 120*coefficients[:, 5]) / h**4
        third_start = 6*coefficients[:, 3] / h**3
        fourth_start = 24*coefficients[:, 4] / h**4
        return np.stack([third_end[:-1] - third_start[1:], fourth_end[:-1] - fourth_start[1:]], axis=1).reshape(-1, 2)

    residual = continuity(_quintic_coefficients(points, knots, velocity, acceleration))
    count = len(residual)
    if count:
        system = np.empty((count, count))
        for column in range(count):
            v, a = np.zeros_like(points), np.zeros_like(points)
            (v if column % 2 == 0 else a)[column // 2 + 1, 0] = 1.
            system[:, column] = continuity(_quintic_coefficients(np.zeros_like(points), knots, v, a))[:, 0]
        # Row equilibration keeps the third/fourth derivative equations comparable.
        scale = np.max(np.abs(system), axis=1)
        solution = np.linalg.solve(system / scale[:, None], -residual / scale[:, None])
        velocity[1:-1], acceleration[1:-1] = solution[::2], solution[1::2]
    coefficients = _quintic_coefficients(points, knots, velocity, acceleration)
    control = np.zeros_like(coefficients)
    for index in range(6):
        for power in range(index + 1):
            control[:, index] += coefficients[:, power] * math.comb(index, power) / math.comb(5, power)
    return control, knots


def plan_route(
    grid: OccupancyGrid, source_xy, start_yaw_rad, end_yaw_rad, *, waypoint_indices,
    config: PlanningConfig | None = None,
) -> PlannedRoute:
    """Plan through ordered source checkpoints, then certify a C4 spatial curve.

    Global parameter knots are cumulative guide chord lengths in metres, NOT
    time or the exact curved arc length. End tangents are unit vectors along the
    supplied yaw; endpoint second derivatives vanish. No time profile is made.
    Exhausted search/refinement raises instead of falling back to official GT.
    """
    config = PlanningConfig() if config is None else config
    source = _points(source_xy, "source_xy")
    indices = np.asarray(waypoint_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 2
            or (indices >= len(source)).any() or (indices < 0).any()
            or indices[0] != 0 or indices[-1] != len(source) - 1
            or not (np.diff(indices.astype(np.int64)) > 0).all()):
        raise ValueError("waypoint_indices must increase from source index 0 to its last index")
    for value in (start_yaw_rad, end_yaw_rad):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("endpoint yaw must be a finite scalar")
    if not all(grid.path_is_free([point], config.footprint_radius_m) for point in source[indices]):
        raise ValueError("a required checkpoint footprint is blocked")
    search = _GridSearch(grid, config)
    all_astar, all_simplified, simplified_leg, legs = [], [], [], []
    for start, end in zip(indices[:-1], indices[1:], strict=True):
        reference = source[start:end + 1]
        astar, audit = search.search(reference)
        simplified = _simplify(astar, grid, reference, config)
        all_astar.append(astar if not all_astar else astar[1:])
        all_simplified.append(simplified if not all_simplified else simplified[1:])
        simplified_leg.extend([len(legs)] * (len(simplified) - 1))
        legs.append(dict(source_start=int(start), source_end=int(end), astar_points=len(astar),
                         simplified_points=len(simplified), **audit))
    astar, simplified = np.vstack(all_astar), np.vstack(all_simplified)
    attempts = []
    for refinement in range(config.max_refinements):
        spacing = config.max_guide_spacing_m / 2**refinement
        intervals = np.maximum(1, np.ceil(np.linalg.norm(np.diff(simplified, axis=0), axis=1) / spacing).astype(int))
        guide = np.vstack([
            *(np.linspace(a, b, count + 1)[:-1]
              for a, b, count in zip(simplified[:-1], simplified[1:], intervals, strict=True)), simplified[-1:]
        ])
        curve_legs = np.repeat(simplified_leg, intervals)
        if len(guide) > 600:
            raise ValueError("curve refinement would exceed 600 guide points")
        try:
            control, knots = _quintic_spline(guide, float(start_yaw_rad), float(end_yaw_rad))
            certificate = certify_bezier(
                grid, control, knots, source, radius_m=config.footprint_radius_m,
                corridor_half_width_m=config.corridor_half_width_m, max_depth=config.certificate_max_depth,
            )
            # Certify each checkpoint-to-checkpoint interval against its own
            # reference subroute as well; another nearby leg cannot supply a
            # spurious corridor shortcut when a source route crosses itself.
            interval_certificates = []
            for leg_index, leg in enumerate(legs):
                selected = np.flatnonzero(curve_legs == leg_index)
                first, last = int(selected[0]), int(selected[-1]) + 1
                interval_certificates.append(certify_bezier(
                    grid, control[first:last], knots[first:last + 1],
                    source[leg["source_start"]:leg["source_end"] + 1],
                    radius_m=config.footprint_radius_m, corridor_half_width_m=config.corridor_half_width_m,
                    max_depth=config.certificate_max_depth,
                ))
        except (ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(refinement=refinement, guide_points=len(guide), accepted=False, reason=str(error)))
            continue
        attempts.append(dict(refinement=refinement, guide_points=len(guide), accepted=True))
        metadata = dict(
            planner="cost_aware_8_connected_astar_with_ordered_source_corridors",
            curve="clamped_C4_quintic_interpolating_spline",
            parameter="cumulative_guide_chord_length_m_not_time_or_curve_arc_length",
            config=asdict(config), waypoint_indices=indices.tolist(), waypoint_xy=source[indices].tolist(),
            source_path_length_m=float(np.linalg.norm(np.diff(source, axis=0), axis=1).sum()),
            guide_chord_length_m=float(knots[-1]), legs=legs, refinement_attempts=attempts,
            simplified_xy=simplified.tolist(), certificate=certificate,
            curve_checkpoint_leg_indices=curve_legs.tolist(), checkpoint_interval_certificates=interval_certificates,
            simplification="greedy collision/corridor-safe line of sight; A* clearance cost is not preserved",
            route_semantics="reviewed ordered checkpoints and source corridor; language agreement not automatically proven",
        )
        return PlannedRoute(control, knots, astar, guide, metadata)
    raise ValueError(f"no certified smooth route after {len(attempts)} refinement attempts: {attempts}")
