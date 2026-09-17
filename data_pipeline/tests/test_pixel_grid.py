"""Pixel/grid addressing must agree with model tokens and preserve missing labels."""

import numpy as np
import pytest

from data_pipeline.pointing import attach_pixel_grid, pixel_grid_arrays, unknown_point
from lightnav.vln_utils import encode_point_pixel


def point(pixel):
    return {"state": "point", "pixel": pixel, "loss_valid": True,
            "token_id": encode_point_pixel(*pixel, 470, 280)}


@pytest.mark.parametrize("pixel,rc,grid_rc,cell_id", [
    ([0., 0.], [0, 0], [0, 0], 1),
    ([235., 140.], [140, 235], [13, 24], 649),
    ([469.999, 279.999], [279, 469], [26, 47], 1296),
    ([np.nextafter(235., 0), 140.], [140, 234], [13, 23], 648),
])
def test_grid_uses_original_image_rows_columns_and_one_based_spatial_ids(pixel, rc, grid_rc, cell_id):
    original = point(pixel)
    result = attach_pixel_grid(original, 470, 280)
    assert "grid" not in original
    assert result["pixel_rc"] == rc
    assert result["grid"] == dict(row=grid_rc[0], col=grid_rc[1], cell_id=cell_id)
    arrays = pixel_grid_arrays([result], "apos")
    np.testing.assert_array_equal(arrays["apos_grid_rc"], [grid_rc])
    np.testing.assert_array_equal(arrays["apos_pixel_rc"], [rc])
    assert arrays["apos_grid_id"].dtype == np.int32 and arrays["apos_grid_id"][0] == cell_id


@pytest.mark.parametrize("pixel", [[470., 100.], [-.01, 100.], [100., 280.]])
def test_outside_image_points_are_not_clamped_to_a_grid_cell(pixel):
    with pytest.raises(ValueError, match="invalid/out-of-image"):
        attach_pixel_grid(point(pixel), 470, 280)


def test_grid_rejects_pixel_token_disagreement():
    with pytest.raises(ValueError, match="disagrees"):
        attach_pixel_grid({**point([235., 140.]), "token_id": 1}, 470, 280)


def test_zero_tokens_keep_unknown_and_not_visible_masks_without_a_spatial_cell():
    absent = unknown_point("no_visible_ground_point")
    absent["target_world_point"] = [1., 2., 3.]
    hidden = dict(state="not_visible", loss_valid=True, token_id=0, pixel=None)
    points = [attach_pixel_grid(p, 470, 280) for p in [absent, hidden]]
    assert all(p["grid"] is None and p["pixel_rc"] is None for p in points)
    assert points[0]["target_world_point"] == [1., 2., 3.]
    assert [p["token_id"] for p in points] == [0, 0]
    assert [p["loss_valid"] for p in points] == [False, True]
    assert [p["state"] for p in points] == ["unknown", "not_visible"]
    arrays = pixel_grid_arrays(points, "opos")
    assert all((a == -1).all() for a in arrays.values())
