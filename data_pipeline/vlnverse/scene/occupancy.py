"""Conservative continuous collision checks in an explicitly calibrated 2-D map.

This module does not interpret a native VLNVerse image. A scene adapter must first
resolve its axes, pixel convention, occupied/unknown values, and world transform.
Unknown space must be marked occupied in the canonical grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _scalar(value: object, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xy(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite XY coordinates")
    return array


def _segment_rectangle_distances(
    start: np.ndarray, end: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Euclidean distance from a closed segment to each closed axis-aligned box.

    Intersecting pairs have distance zero. Otherwise, a closest pair includes
    either a segment endpoint or a rectangle vertex, including parallel edges.
    Checking both sets therefore covers the complete segment without sampling.
    """
    delta = end - start
    enter, leave = np.zeros(len(lower)), np.ones(len(lower))
    for axis in range(2):
        if delta[axis] == 0.0:
            outside = (start[axis] < lower[:, axis]) | (start[axis] > upper[:, axis])
            leave[outside] = -np.inf
        else:
            with np.errstate(over="ignore"):
                first = (lower[:, axis] - start[axis]) / delta[axis]
                second = (upper[:, axis] - start[axis]) / delta[axis]
            enter = np.maximum(enter, np.minimum(first, second))
            leave = np.minimum(leave, np.maximum(first, second))

    def endpoint_distance(point):
        displacement = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return np.hypot(displacement[:, 0], displacement[:, 1])

    distance = np.minimum(endpoint_distance(start), endpoint_distance(end))
    with np.errstate(over="ignore"):
        length = np.hypot(delta[0], delta[1])
    if not np.isfinite(length):
        # Numerical overflow cannot certify a free segment, even with finite inputs.
        return np.zeros(len(lower))
    if length > 0.0:
        direction = delta / length
        for corner in (
            lower, upper,
            np.column_stack([lower[:, 0], upper[:, 1]]),
            np.column_stack([upper[:, 0], lower[:, 1]]),
        ):
            relative = corner - start
            with np.errstate(over="ignore", invalid="ignore"):
                projection = np.clip(relative @ direction, 0.0, length)
                displacement = relative - projection[:, None] * direction
                candidate = np.hypot(displacement[:, 0], displacement[:, 1])
            # A nonfinite result fails closed for that rectangle.
            distance = np.minimum(distance, np.where(np.isfinite(candidate), candidate, 0.0))
    distance[enter <= leave] = 0.0
    return distance


@dataclass(frozen=True)
class OccupancyGrid:
    """Metric map with rows along local +Y and columns along local +X.

    ``origin_xy_m`` is the outer corner of cell (row=0, column=0), transformed
    into world coordinates. Rotate local coordinates by ``grid_yaw_rad`` about
    +Z, then add the origin to get world XY. True cells and map exterior are
    blocked. Touching an obstacle or map boundary is considered a collision.

    Collision checking uses the exact Euclidean distance between the complete
    centerline segment and each closed occupied cell, with a small conservative
    floating-point tolerance. It validates the swept disc independently of
    output sampling frequency. Use a radius that includes both robot footprint
    and the desired safety margin. The supplied raster may itself conservatively
    approximate scene geometry; this check is exact for its cell rectangles.
    """

    occupied: np.ndarray
    resolution_m: float
    origin_xy_m: tuple[float, float]
    grid_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        occupied = np.asarray(self.occupied)
        if occupied.ndim != 2 or 0 in occupied.shape or occupied.dtype != np.bool_:
            raise ValueError("occupied must be a nonempty 2-D boolean array")
        resolution = _scalar(self.resolution_m, "resolution_m")
        if resolution <= 0:
            raise ValueError("resolution_m must be positive")
        with np.errstate(over="ignore"):
            extent = np.array(occupied.shape[::-1], dtype=np.float64) * resolution
        if not np.all(np.isfinite(extent)):
            raise ValueError("map extent must be finite")
        origin = _xy(self.origin_xy_m, "origin_xy_m")
        yaw = _scalar(self.grid_yaw_rad, "grid_yaw_rad")
        occupied = occupied.copy()
        occupied.setflags(write=False)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "resolution_m", resolution)
        object.__setattr__(self, "origin_xy_m", tuple(float(value) for value in origin))
        object.__setattr__(self, "grid_yaw_rad", yaw)

    @classmethod
    def from_npz(cls, path: str | Path) -> OccupancyGrid:
        """Read canonical fields only; never infer a native PNG transform.

        Required fields are ``occupied`` (bool [H, W]), ``resolution_m``
        (scalar), and ``origin_xy_m`` ([2]); ``grid_yaw_rad`` is optional.
        """
        with np.load(path, allow_pickle=False) as fields:
            missing = {"occupied", "resolution_m", "origin_xy_m"} - set(fields.files)
            if missing:
                raise ValueError(f"canonical occupancy map is missing fields: {sorted(missing)}")
            return cls(
                occupied=fields["occupied"],
                resolution_m=fields["resolution_m"],
                origin_xy_m=fields["origin_xy_m"],
                grid_yaw_rad=fields["grid_yaw_rad"] if "grid_yaw_rad" in fields else 0.0,
            )

    def is_segment_free(
        self, start_xy: object, end_xy: object, radius_m: float
    ) -> bool:
        """Check an entire swept disc, including endpoints and zero-length moves."""
        start_world = _xy(start_xy, "start_xy")
        end_world = _xy(end_xy, "end_xy")
        radius = _scalar(radius_m, "radius_m")
        if radius < 0:
            raise ValueError("radius_m must be nonnegative")

        cosine, sine = np.cos(self.grid_yaw_rad), np.sin(self.grid_yaw_rad)
        # Row vectors multiplied by R produce the inverse rotation of columns.
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        with np.errstate(over="ignore", invalid="ignore"):
            endpoints = (
                np.stack([start_world, end_world]) - np.asarray(self.origin_xy_m)
            ) @ rotation
        if not np.all(np.isfinite(endpoints)):
            return False
        extent = np.array(self.occupied.shape[::-1]) * self.resolution_m
        # Inflate slightly to keep floating-point boundary ambiguity conservative.
        scale = max(
            1.0, float(np.max(extent)), radius,
            float(np.max(np.abs([start_world, end_world, self.origin_xy_m]))),
        )
        inflation = radius + 32 * np.finfo(np.float64).eps * scale
        if np.any(endpoints <= inflation) or np.any(endpoints >= extent - inflation):
            return False

        # A rectangle is convex, so checking both ends above proves the entire
        # centerline stays within the shrunken map rectangle.
        lower = np.floor((endpoints.min(axis=0) - inflation) / self.resolution_m)
        upper = np.floor((endpoints.max(axis=0) + inflation) / self.resolution_m)
        lower = np.maximum(lower.astype(np.int64) - 1, 0)
        upper = np.minimum(upper.astype(np.int64) + 1, self.occupied.shape[::-1])
        row, col = np.nonzero(self.occupied[lower[1]:upper[1], lower[0]:upper[0]])
        if not len(row):
            return True
        cells = np.column_stack([col + lower[0], row + lower[1]])
        box_lower = cells * self.resolution_m
        box_upper = (cells + 1) * self.resolution_m
        distance = _segment_rectangle_distances(*endpoints, box_lower, box_upper)
        return not bool(np.any(distance <= inflation))

    def path_is_free(self, xy: object, radius_m: float) -> bool:
        """Check a nonempty polyline; singleton paths check the footprint in place."""
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            raise ValueError("xy must be a nonempty [N, 2] array")
        if not np.all(np.isfinite(points)):
            raise ValueError("xy must contain only finite coordinates")
        if len(points) == 1:
            return self.is_segment_free(points[0], points[0], radius_m)
        return all(
            self.is_segment_free(start, end, radius_m)
            for start, end in zip(points[:-1], points[1:], strict=True)
        )
