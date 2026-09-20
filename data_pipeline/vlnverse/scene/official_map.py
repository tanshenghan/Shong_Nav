"""Validate and conservatively convert official coordinate-header freemaps.

This adapter requires three independent agreements: the embedded coordinate
axes match scene bounds, map classes match the scene PNG, and world room vertices
map to the supplied image vertices by nearest coordinate sample. It never treats
the nominal ``scale`` as the actual pixel spacing: the two native axes can have
slightly different spacings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid

# Official room vertices sometimes differ from bounds by float32 roundoff.
# This tolerance admits sub-micrometre calibration noise, never extra free space.
ROOM_BOUNDS_TOLERANCE_M = 1e-6


def _coordinate_axis(values, *, increasing, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain at least two finite coordinate samples")
    differences = np.diff(values)
    if not np.all(differences > 0 if increasing else differences < 0):
        raise ValueError(f"{name} has an unexpected direction")
    if not np.allclose(differences, differences[0], rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} coordinate samples must be evenly spaced")
    return values


def _cell_edges(centers):
    # Nearest-sample lookup defines Voronoi pixel cells. Clip exterior half
    # cells to the documented scene bounds rather than inventing extra coverage.
    return np.r_[centers[0], (centers[:-1] + centers[1:]) / 2, centers[-1]]


def _conservative_square_grid(blocked, x_centers, y_centers, resolution_m):
    if (isinstance(resolution_m, bool) or not np.isscalar(resolution_m)
            or not np.isfinite(resolution_m) or resolution_m <= 0):
        raise ValueError("resolution_m must be a finite positive scalar")
    native_x, native_y = _cell_edges(x_centers), _cell_edges(y_centers)
    width = int(np.ceil((native_x[-1] - native_x[0]) / resolution_m))
    height = int(np.ceil((native_y[-1] - native_y[0]) / resolution_m))
    if width * height > 50_000_000:
        raise ValueError("requested resolution would produce more than 50 million cells")
    target_x = native_x[0] + np.arange(width + 1) * resolution_m
    target_y = native_y[0] + np.arange(height + 1) * resolution_m

    def overlapping_cells(native_edges, target_edges):
        lower = np.searchsorted(native_edges, target_edges[:-1], side="right") - 1
        upper = np.searchsorted(native_edges, target_edges[1:], side="left")
        count = len(native_edges) - 1
        return np.clip(lower, 0, count), np.clip(upper, 0, count)

    left, right = overlapping_cells(native_x, target_x)
    bottom, top = overlapping_cells(native_y, target_y)
    integral = np.pad(blocked.astype(np.int64), ((1, 0), (1, 0)))
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    overlap_count = (integral[top[:, None], right[None, :]]
                     - integral[bottom[:, None], right[None, :]]
                     - integral[top[:, None], left[None, :]]
                     + integral[bottom[:, None], left[None, :]])
    occupied = overlap_count > 0
    occupied[:, target_x[1:] > native_x[-1]] = True
    occupied[target_y[1:] > native_y[-1], :] = True
    return OccupancyGrid(occupied, resolution_m, (native_x[0], native_y[0]))


def load_official_map(
    freemap_path: str | Path, scene_dir: str | Path, *, resolution_m: float = 0.05
) -> tuple[OccupancyGrid, dict]:
    """Return a canonical map and the calibration evidence for its provenance.

    Required official files are ``freemap.npy`` and the scene's
    ``occupancy.json``, ``occupancy.png``, ``rooms.json``, ``room_in_images.json``.
    The header format is X coordinates in row 0, Y coordinates in column 0;
    class 1 (PNG 255) is free, 0 (PNG 0) occupied, and 2 (PNG 127) unknown.
    Only verified free cells can create free space in the result.

    The output raster marks a square occupied if it has positive-area overlap
    with any native occupied/unknown rectangle or with the map exterior. Thus
    resampling can reduce free space, but never creates a free shortcut.
    """
    from PIL import Image

    freemap_path, scene_dir = Path(freemap_path), Path(scene_dir)
    paths = {name: scene_dir / name for name in (
        "occupancy.json", "occupancy.png", "rooms.json", "room_in_images.json"
    )}
    paths["freemap.npy"] = freemap_path
    metadata = json.loads(paths["occupancy.json"].read_text())
    native = np.load(freemap_path, allow_pickle=False)
    if native.ndim != 2 or min(native.shape) < 3 or not np.isfinite(native).all():
        raise ValueError("freemap must be a finite 2-D coordinate-header array")
    x_desc = _coordinate_axis(native[0, 1:], increasing=False, name="X header")
    y = _coordinate_axis(native[1:, 0], increasing=True, name="Y header")
    bounds = np.asarray([metadata["min"], metadata["max"]], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("occupancy bounds must be finite 3-D min/max values")
    if not np.all(bounds[1] > bounds[0]):
        raise ValueError("occupancy max bounds must exceed min bounds")
    if not np.allclose(
        [x_desc[-1], y[0], x_desc[0], y[-1]], bounds[:, :2].ravel(),
        rtol=0, atol=1e-8,
    ):
        raise ValueError("freemap coordinate headers do not match occupancy world bounds")
    classes = native[1:, 1:]
    if not np.isin(classes, [0, 1, 2]).all():
        raise ValueError("freemap contains unsupported classes; expected 0/1/2")
    with Image.open(paths["occupancy.png"]) as image:
        image_mode = image.mode
        pixels = np.asarray(image)
    if image_mode in {"RGB", "RGBA"}:
        if not np.array_equal(pixels[:, :, :3], np.repeat(pixels[:, :, :1], 3, axis=2)):
            raise ValueError("occupancy.png RGB channels must be identical grayscale values")
        if image_mode == "RGBA" and not np.all(pixels[:, :, 3] == 255):
            raise ValueError("occupancy.png alpha must be fully opaque")
        pixels = pixels[:, :, 0]
    elif image_mode != "L":
        raise ValueError("occupancy.png must use L, grayscale RGB, or opaque grayscale RGBA")
    expected_pixels = np.choose(classes.astype(np.int64), [0, 255, 127])
    if pixels.shape != classes.shape or not np.array_equal(pixels, expected_pixels):
        raise ValueError("freemap classes do not match occupancy.png (0/255/127)")

    rooms = json.loads(paths["rooms.json"].read_text())
    room_pixels = json.loads(paths["room_in_images.json"].read_text())
    if not rooms or len(rooms) != len(room_pixels):
        raise ValueError("world and image room polygons must be nonempty and paired")
    vertex_count, max_vertex_distance = 0, 0.0
    for world_room, image_room in zip(rooms, room_pixels, strict=True):
        world = np.asarray(world_room["polygon"], dtype=float)
        pixel = np.asarray(image_room["polygon"], dtype=float)
        if (world_room["room_type"] != image_room["room_type"] or world.ndim != 2
                or world.shape[1:] != (2,) or len(world) < 3 or pixel.shape != world.shape
                or not np.isfinite(world).all() or not np.isfinite(pixel).all()):
            raise ValueError("world and image room polygons do not have matching valid vertices")
        if (np.any(world < bounds[0, :2] - ROOM_BOUNDS_TOLERANCE_M)
                or np.any(world > bounds[1, :2] + ROOM_BOUNDS_TOLERANCE_M)):
            raise ValueError("world room vertices exceed the declared map bounds")
        columns = np.abs(world[:, 0, None] - x_desc).argmin(axis=1)
        rows = np.abs(world[:, 1, None] - y).argmin(axis=1)
        if not np.array_equal(np.column_stack([columns, rows]), pixel):
            raise ValueError("room vertices disagree with nearest coordinate-header calibration")
        difference = world - np.column_stack([x_desc[columns], y[rows]])
        max_vertex_distance = max(max_vertex_distance, float(np.linalg.norm(difference, axis=1).max()))
        vertex_count += len(world)

    grid = _conservative_square_grid(classes[:, ::-1] != 1, x_desc[::-1], y, resolution_m)
    evidence = {
        "adapter": "official_coordinate_header_freemap_v2",
        "native_shape": list(classes.shape),
        "native_spacing_xy_m": [float(x_desc[0] - x_desc[1]), float(y[1] - y[0])],
        "native_axes": "row increases world Y; column decreases world X",
        "native_classes": {"0": "occupied", "1": "free", "2": "unknown"},
        "occupancy_png_mode": image_mode,
        "occupancy_png_normalization": "exact equal RGB channels; RGBA requires alpha 255",
        "pixel_coordinate_model": "nearest embedded world coordinate; midpoint cell boundaries",
        "outside_bounds": "blocked; exterior pixel cells clipped to declared min/max",
        "verified_room_vertices": vertex_count,
        "room_bounds_tolerance_m": ROOM_BOUNDS_TOLERANCE_M,
        "max_room_vertex_to_pixel_center_distance_m": max_vertex_distance,
        "declared_map_height_range_m": bounds[:, 2].tolist(),
        "canonical_shape": list(grid.occupied.shape),
        "canonical_resolution_m": grid.resolution_m,
        "resampling": "any occupied/unknown rectangle overlap blocks the output square",
        "limitation": "2-D supplied occupancy only; not a new 3-D robot collision certificate",
        "inputs": {},
    }
    for name, path in paths.items():
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        evidence["inputs"][name] = {"path": str(path.resolve()), "sha256": digest}
    return grid, evidence
