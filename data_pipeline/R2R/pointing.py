"""CPU geometry for apos-10, fixed terminal opos, and image/grid coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lightnav.vln_utils import POINT_GRID_H, POINT_GRID_W, encode_point_pixel


@dataclass(frozen=True)
class PointingConfig:
    horizon: int = 10
    tail_policy: str = "terminal"
    path_spacing_m: float = .02
    depth_tolerance_m: float = .05
    depth_relative_tolerance: float = .01
    opos_mode: str = "terminal_ground"
    opos_visibility_window: str = "last_half"
    unknown_token_id: int = 0
    # Read-only compatibility with historical metadata; not emitted in new configs.
    min_object_pixels: int = 12
    min_box_coverage: float = .6
    min_detection_confidence: float = .6
    opos_search_frames: int = 8

    def __post_init__(self):
        if type(self.unknown_token_id) is not int or self.unknown_token_id not in (-1, 0):
            raise ValueError("unknown_token_id must be 0 (current) or -1 (historical)")
        if self.opos_mode not in ("terminal_ground", "doubao"):
            raise ValueError("opos_mode must be terminal_ground or doubao")
        if self.opos_visibility_window not in ("last_half", "last_third", "all"):
            raise ValueError("opos_visibility_window must be last_half, last_third or all")
        if self.tail_policy not in ("terminal", "mask"):
            raise ValueError("tail_policy must be terminal or mask")
        if self.horizon != 10 or not 0 < self.path_spacing_m <= 1:
            raise ValueError("apos requires horizon=10 and spacing in (0, 1] metres")
        if not 0 <= self.depth_tolerance_m <= 1 or not 0 <= self.depth_relative_tolerance <= .1:
            raise ValueError("invalid depth tolerance")
        if self.min_object_pixels < 1:
            raise ValueError("min_object_pixels must be positive")
        if type(self.opos_search_frames) is not int or self.opos_search_frames < 1:
            raise ValueError("opos_search_frames must be a positive integer")
        if not 0 < self.min_box_coverage <= 1 or not 0 <= self.min_detection_confidence <= 1:
            raise ValueError("invalid object binding thresholds")

    def metadata(self):
        result = asdict(self)
        if self.opos_mode == "terminal_ground":
            for key in ("min_object_pixels", "min_box_coverage", "min_detection_confidence", "opos_search_frames"):
                result.pop(key)
        return result

    def require_current(self):
        """Old definitions can be validated, but are no longer generation options."""
        if self.opos_mode != "terminal_ground" or self.tail_policy != "terminal" or self.unknown_token_id != 0:
            raise ValueError("final GT generation requires opos_mode=terminal_ground, tail_policy=terminal "
                             "and unknown_token_id=0")

    @classmethod
    def from_metadata(cls, metadata):
        """Preserve historical visibility, tail and sentinel rules when validating."""
        return cls(**{"tail_policy": "mask", "opos_search_frames": 1, "opos_mode": "doubao",
                      "unknown_token_id": -1, "opos_visibility_window": "all", **metadata})

    def opos_visibility_start_frame(self, frame_count):
        """Zero-based inclusive start; round up the number of allowed tail frames."""
        if type(frame_count) is not int or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if self.opos_visibility_window == "last_half":
            return frame_count // 2
        return (2 * frame_count) // 3 if self.opos_visibility_window == "last_third" else 0


def project_world(points, world_from_camera, K):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera_from_world = np.linalg.inv(np.asarray(world_from_camera))
    xyz = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    projected = xyz @ np.asarray(K).T
    uv = np.full((len(points), 2), np.nan)
    front = xyz[:, 2] > 1e-6
    uv[front] = projected[front, :2] / projected[front, 2:]
    return uv, xyz[:, 2]


def backproject_pixels(uv, depth, world_from_camera, K):
    """Image-edge coordinates: integer array pixel (row,col) has centre +0.5."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    xyz = rays * np.asarray(depth).reshape(-1, 1)
    transform = np.asarray(world_from_camera)
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def unknown_point(reason, token_id=0):
    return {"state": "unknown", "pixel": None, "world_point": None,
            "token_id": token_id, "loss_valid": False, "reason": reason}


def pointing_token(channel, point):
    """Serialize current zero placeholders independently of their supervision mask.

    Negative IDs are supported only when reading historical masked labels.
    """
    tid = point["token_id"]
    return f"<{channel}_{tid}>" if tid >= 0 else None


def pixel_grid_metadata(width, height):
    return {"version": 1, "image_size_wh": [int(width), int(height)],
            "grid_shape_rc": [POINT_GRID_H, POINT_GRID_W], "index_base": 0,
            "pixel_order": "[u, v] = [x, y]", "pixel_rc_order": "[row, col]",
            "grid_rc_order": "[row, col]", "cell_id_formula": "1 + row * 48 + col",
            "missing_array_index": -1, "reference": "original RGB, before video scaling or padding"}


def attach_pixel_grid(point, width, height):
    """Expose both original-image pixel indices and the existing model grid cell.

    Does not select a point or change geometry/tokens. Missing spatial labels,
    including the valid opos=0 not-visible class, have no pixel or grid cell.
    """
    result = {**point, "pixel_rc": None, "grid": None}
    if point["state"] != "point" or not point["loss_valid"]:
        return result
    pixel = np.asarray(point["pixel"], dtype=float)
    if (width <= 0 or height <= 0 or pixel.shape != (2,) or not np.isfinite(pixel).all()
            or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
        raise ValueError("cannot assign a grid cell to an invalid/out-of-image pixel")
    tid = encode_point_pixel(*pixel, width, height)
    if point["token_id"] != tid:
        raise ValueError("point token disagrees with its pixel/grid cell")
    col, row = np.floor(pixel).astype(int)
    result["pixel_rc"] = [int(row), int(col)]
    result["grid"] = {"row": (tid - 1) // POINT_GRID_W, "col": (tid - 1) % POINT_GRID_W,
                      "cell_id": tid}
    return result


def pixel_grid_arrays(points, channel):
    return {
        channel + "_pixel_rc": np.asarray([p["pixel_rc"] if p["pixel_rc"] is not None
                                          else [-1, -1] for p in points], dtype=np.int32),
        channel + "_grid_rc": np.asarray([[p["grid"]["row"], p["grid"]["col"]]
                                         if p["grid"] is not None else [-1, -1]
                                         for p in points], dtype=np.int32),
        channel + "_grid_id": np.asarray([p["grid"]["cell_id"] if p["grid"] is not None
                                         else -1 for p in points], dtype=np.int32),
    }


def ground_target_arrays(points, channel="apos"):
    """Intended 3D targets are independent of selected visible-point labels."""
    if not all("target_world_point" in p for p in points):
        return {}
    return {
        channel + "_target_world": np.asarray([p["target_world_point"] for p in points], dtype=np.float64),
        channel + "_target_frame_id": np.asarray([p["target_frame_id"] for p in points], dtype=np.int32),
    }


def ground_point(positions, t, world_from_camera, K, depth, depth_valid, config):
    """Prefer exactly t+10; otherwise walk backwards along that future polyline.

    The first visible candidate is nearest in *path arclength* to t+10, to within
    path_spacing_m. Body origins are the recorded feet/navmesh ground positions.
    When t+10 is unavailable, use the recorded terminal ground position. Keep
    this 3D target even when no visible pixel exists; pixel loss stays masked.
    """
    positions = np.asarray(positions, dtype=np.float64)
    end = t + config.horizon
    if end >= len(positions):
        if config.tail_policy == "mask":
            return unknown_point("missing_t_plus_10", config.unknown_token_id)
        end = len(positions) - 1
    target = ({"target_frame_id": end, "target_world_point": positions[end].tolist()}
              if config.tail_policy == "terminal" else {})
    exact_reason = "terminal_ground_point" if t + config.horizon >= len(positions) else "t_plus_10"
    candidates, indices, backtracks = [positions[end]], [float(end)], [0.]
    travelled = 0.
    for j in range(end, t, -1):
        length = float(np.linalg.norm(positions[j] - positions[j - 1]))
        if length <= 1e-9:
            continue
        count = int(np.ceil(length / config.path_spacing_m))
        for k in range(1, count + 1):
            fraction = k / count
            candidates.append(positions[j] * (1 - fraction) + positions[j - 1] * fraction)
            indices.append(j - fraction)
            backtracks.append(travelled + length * fraction)
        travelled += length
    candidates = np.asarray(candidates)
    uv, z = project_world(candidates, world_from_camera, K)
    height, width = depth.shape
    inside = (np.isfinite(uv).all(1) & (z > 1e-6) & (uv[:, 0] >= 0)
              & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    for i in np.flatnonzero(inside):
        col, row = np.floor(uv[i]).astype(int)
        observed = float(depth[row, col])
        if not depth_valid[row, col] or not np.isfinite(observed) or observed <= 0:
            continue
        tolerance = config.depth_tolerance_m + config.depth_relative_tolerance * observed
        if z[i] > observed + tolerance:
            continue
        return {**target, "state": "point", "pixel": uv[i].tolist(),
                "world_point": candidates[i].tolist(),
                "token_id": encode_point_pixel(*uv[i], width, height), "loss_valid": True,
                "target_frame_id": end, "selected_frame_fraction": indices[i],
                "backtrack_m": backtracks[i],
                "reason": exact_reason if i == 0 else "visible_path_backtrack"}
    return {**unknown_point("no_visible_ground_point", config.unknown_token_id), **target}


def fixed_world_point(world_point, world_from_camera, K, depth, depth_valid, config):
    """Visibility of a confirmed, fixed destination on the recorded ground plane."""
    uv, z = project_world([world_point], world_from_camera, K)
    height, width = depth.shape
    pixel = uv[0]
    invisible = {"state": "not_visible", "pixel": None, "world_point": None,
                 "token_id": 0, "loss_valid": True}
    if (z[0] <= 1e-6 or not np.isfinite(pixel).all() or not
            (0 <= pixel[0] < width and 0 <= pixel[1] < height)):
        return {**invisible, "reason": "stop_location_outside_view"}
    col, row = np.floor(pixel).astype(int)
    observed = float(depth[row, col])
    if not depth_valid[row, col] or not np.isfinite(observed) or observed <= 0:
        return unknown_point("stop_location_depth_unknown", config.unknown_token_id)
    tolerance = config.depth_tolerance_m + config.depth_relative_tolerance * observed
    if z[0] > observed + tolerance:
        return {**invisible, "reason": "stop_location_occluded"}
    return {"state": "point", "pixel": pixel.tolist(),
            "world_point": np.asarray(world_point).tolist(),
            "token_id": encode_point_pixel(*pixel, width, height), "loss_valid": True,
            "reason": "visible_terminal_ground_location"}


def terminal_ground_point(positions, world_from_camera, K, depth, depth_valid, config, *, frame_index):
    """opos is the final recorded destination, independent of visual recognition.

    Always retain its world coordinate and target frame, including when the
    destination is occluded, outside the view, or under the terminal camera.
    Before the configured window, supervise token 0 with an explicit temporal
    reason. This label restriction does not assert physical occlusion.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions must be a nonempty (N,3) array")
    if type(frame_index) is not int or not 0 <= frame_index < len(positions):
        raise ValueError("frame_index must be an integer in [0, N)")
    goal = positions[-1]
    if not np.isfinite(goal).all():
        raise ValueError("terminal destination must be finite")
    if frame_index < config.opos_visibility_start_frame(len(positions)):
        point = {"state": "not_visible", "pixel": None, "world_point": None,
                 "token_id": 0, "loss_valid": True, "reason": "before_opos_visibility_window"}
    else:
        point = fixed_world_point(goal, world_from_camera, K, depth, depth_valid, config)
    return {**point, "target_frame_id": len(positions) - 1, "target_world_point": goal.tolist()}
