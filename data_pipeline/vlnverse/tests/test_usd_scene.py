"""Pure array coverage for the static USD importer's per-corner primvars."""

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.usd_scene import loop_values


def test_vertex_uvs_follow_face_vertex_indices():
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    indices = np.array([0, 1, 2, 0, 2, 3])
    expanded = loop_values(uv, "vertex", indices, np.array([3, 3]), 4)
    np.testing.assert_array_equal(expanded, [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]])


def test_face_varying_uvs_preserve_seams():
    uv = np.array([[0, 0], [1, 0], [1, 1], [.2, .3], [1, 1], [0, 1]])
    expanded = loop_values(uv, "faceVarying", np.array([0, 1, 2, 0, 2, 3]), np.array([3, 3]), 4)
    np.testing.assert_array_equal(expanded, uv)
    assert not np.array_equal(expanded[0], expanded[3])


def test_uniform_primvars_expand_per_face():
    expanded = loop_values(np.array([[.1, .2], [.3, .4]]), "uniform",
                           np.array([0, 1, 2, 0, 2, 3, 4]), np.array([3, 4]), 5)
    np.testing.assert_array_equal(expanded, [[.1, .2]] * 3 + [[.3, .4]] * 4)


@pytest.mark.parametrize("interpolation", ["faceVarying", "vertex", "unknown"])
def test_inconsistent_uv_interpolation_is_rejected(interpolation):
    with pytest.raises(ValueError, match="interpolation"):
        loop_values(np.array([[0, 0], [1, 0]]), interpolation, np.array([0, 1, 2]), np.array([3]), 3)
