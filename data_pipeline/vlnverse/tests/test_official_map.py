"""Calibrated headers, native class conventions and conservative resampling."""

import json

import numpy as np
import pytest
from PIL import Image

from data_pipeline.vlnverse.scene.official_map import load_official_map


@pytest.fixture
def official_files(tmp_path):
    x = np.array([5., 4., 3., 2., 1.])
    y = np.array([-2., 0., 2., 4.])
    classes = np.ones((4, 5))
    classes[1, 2] = 0
    classes[2, 1] = 2
    encoded = np.zeros((5, 6))
    encoded[0, 1:], encoded[1:, 0], encoded[1:, 1:] = x, y, classes
    np.save(tmp_path / "freemap.npy", encoded)
    Image.fromarray(np.choose(classes.astype(int), [0, 255, 127]).astype(np.uint8)).save(
        tmp_path / "occupancy.png"
    )
    (tmp_path / "occupancy.json").write_text(json.dumps({
        "min": [1, -2, .1], "max": [5, 4, 1.0], "scale": .05
    }))
    (tmp_path / "rooms.json").write_text(json.dumps([
        {"room_type": "room", "polygon": [[5, -2], [1, -2], [1, 4], [5, 4]]}
    ]))
    (tmp_path / "room_in_images.json").write_text(json.dumps([
        {"room_type": "room", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}
    ]))
    return tmp_path


def test_axes_unknown_and_anisotropic_cells_are_preserved_conservatively(official_files):
    grid, evidence = load_official_map(
        official_files / "freemap.npy", official_files, resolution_m=.25
    )
    assert grid.origin_xy_m == (1., -2.)
    assert grid.occupied.shape == (24, 16)
    assert evidence["native_spacing_xy_m"] == [1., 2.]
    assert evidence["verified_room_vertices"] == 4
    assert len(evidence["inputs"]["freemap.npy"]["sha256"]) == 64
    assert not grid.path_is_free([[3., 0.]], 0.)  # native obstacle
    assert not grid.path_is_free([[4., 2.]], 0.)  # native unknown
    assert grid.path_is_free([[1.3, 0.]], .1)
    assert not grid.is_segment_free([1.3, 0.], [4.7, 0.], .1)


def test_coarse_cell_that_overlaps_obstacle_is_blocked(official_files):
    grid, _ = load_official_map(
        official_files / "freemap.npy", official_files, resolution_m=1.25
    )
    # This location is outside the native obstacle, but its coarse output cell
    # intersects the obstacle. Any-overlap rasterization must reject it.
    assert not grid.path_is_free([[2.35, -0.6]], 0.)
    # Partial right/bottom cells cannot become additional free map coverage.
    assert not grid.path_is_free([[5.1, -1.5]], 0.)
    assert not grid.path_is_free([[1.3, 4.1]], 0.)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_equal_channel_opaque_pngs_preserve_the_exact_map(official_files, mode):
    reference, _ = load_official_map(official_files / "freemap.npy", official_files)
    path = official_files / "occupancy.png"
    with Image.open(path) as image:
        converted = image.convert(mode)
    converted.save(path)
    actual, evidence = load_official_map(official_files / "freemap.npy", official_files)
    np.testing.assert_array_equal(actual.occupied, reference.occupied)
    assert actual.origin_xy_m == reference.origin_xy_m
    assert evidence["occupancy_png_mode"] == mode


@pytest.mark.parametrize("corruption", ["color", "alpha", "palette", "class"])
def test_png_normalization_does_not_accept_color_alpha_or_class_ambiguity(
    official_files, corruption
):
    path = official_files / "occupancy.png"
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGBA"))
        palette = image.convert("P")
    if corruption == "color":
        pixels[0, 0, 1] = 254
    elif corruption == "alpha":
        pixels[0, 0, 3] = 254
    elif corruption == "class":
        pixels[0, 0, :3] = 127  # Valid grayscale value, but wrong freemap class.
    if corruption == "palette":
        palette.save(path)
    else:
        Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="occupancy.png"):
        load_official_map(official_files / "freemap.npy", official_files)


@pytest.mark.parametrize("offset", [4.4e-7, -4.4e-7])
def test_sub_micrometre_room_roundoff_does_not_expand_free_space(official_files, offset):
    reference, _ = load_official_map(official_files / "freemap.npy", official_files)
    path = official_files / "rooms.json"
    payload = json.loads(path.read_text())
    vertex = 0 if offset > 0 else 1  # Upper or lower X boundary.
    payload[0]["polygon"][vertex][0] += offset
    path.write_text(json.dumps(payload))
    actual, evidence = load_official_map(official_files / "freemap.npy", official_files)
    assert evidence["room_bounds_tolerance_m"] == 1e-6
    np.testing.assert_array_equal(actual.occupied, reference.occupied)
    assert actual.origin_xy_m == reference.origin_xy_m
    assert not actual.path_is_free([[5.0 + abs(offset), -1.0]], 0.0)


@pytest.mark.parametrize("offset", [2e-6, -2e-6, 0.107101472])
def test_actual_room_bound_overrun_remains_rejected(official_files, offset):
    path = official_files / "rooms.json"
    payload = json.loads(path.read_text())
    vertex = 0 if offset > 0 else 1
    payload[0]["polygon"][vertex][0] += offset
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exceed the declared map bounds"):
        load_official_map(official_files / "freemap.npy", official_files)


@pytest.mark.parametrize("corruption", ["axis", "bounds", "classes", "image", "rooms", "world_rooms"])
def test_inconsistent_calibration_is_rejected(official_files, corruption):
    if corruption in {"axis", "classes"}:
        array = np.load(official_files / "freemap.npy")
        if corruption == "axis":
            array[0, 2] += .1
        else:
            array[1, 1] = 3
        np.save(official_files / "freemap.npy", array)
    elif corruption == "image":
        Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(official_files / "occupancy.png")
    elif corruption == "bounds":
        path = official_files / "occupancy.json"
        payload = json.loads(path.read_text())
        payload["min"][0] += 1
        path.write_text(json.dumps(payload))
    elif corruption == "rooms":
        path = official_files / "room_in_images.json"
        payload = json.loads(path.read_text())
        payload[0]["polygon"][0][0] = 1
        path.write_text(json.dumps(payload))
    else:
        path = official_files / "rooms.json"
        payload = json.loads(path.read_text())
        # Its nearest coordinate would still be pixel zero; being outside the
        # map must fail instead of appearing to corroborate the calibration.
        payload[0]["polygon"][0][0] = 500
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_official_map(official_files / "freemap.npy", official_files)


@pytest.mark.parametrize("resolution", [0., -1., np.nan, np.inf, True])
def test_invalid_resolution_is_rejected(official_files, resolution):
    with pytest.raises(ValueError, match="resolution_m"):
        load_official_map(official_files / "freemap.npy", official_files, resolution_m=resolution)
