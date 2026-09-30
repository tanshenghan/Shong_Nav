"""UV transforms are testable without Blender, USD, GPU or scene assets."""

import math

import numpy as np

from data_pipeline.vlnverse.replay.usd_materials import uv_mapping_parameters


def apply_mapping(uv, mapping):
    theta = mapping["rotation"][2]
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    return rotation @ (np.asarray(uv) * mapping["scale"][:2]) + mapping["location"][:2]


def test_identity_mapping_preserves_imported_uv_coordinates():
    mapping = uv_mapping_parameters((1, 1), (0, 0), 0)
    for uv in ((0, 0), (.3, .8), (1, 1), (2, -1)):
        np.testing.assert_allclose(apply_mapping(uv, mapping), uv, atol=1e-12)


def test_v2_scale_offset_and_rotation_match_mdl_double_v_flip():
    uv = np.array([.2, .7])
    scale, offset, turns = np.array([2., .6]), np.array([.1, -.3]), .25
    theta = 2 * math.pi * turns
    mdl_uv = np.array([uv[0], 1 - uv[1]]) * scale
    mdl_uv[1] += 1 - scale[1]
    mdl_uv += [offset[0], -offset[1]]
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    mdl_uv = rotation @ (mdl_uv - .5) + .5
    mdl_uv[1] = 1 - mdl_uv[1]
    mapping = uv_mapping_parameters(scale, offset, -theta)
    np.testing.assert_allclose(apply_mapping(uv, mapping), mdl_uv, atol=1e-12)


def test_legacy_scale_and_angle_preserve_original_v_origin():
    uv, scale, angle = np.array([.4, .8]), np.array([.8, 2.]), .3
    mdl_uv = np.array([uv[0], 1 - uv[1]]) * scale - .5
    rotation = np.array([[math.cos(-angle), -math.sin(-angle)],
                         [math.sin(-angle), math.cos(-angle)]])
    mdl_uv = rotation @ mdl_uv + .5
    mdl_uv[1] = 1 - mdl_uv[1]
    mapping = uv_mapping_parameters(scale, (0, 1 - scale[1]), angle)
    np.testing.assert_allclose(apply_mapping(uv, mapping), mdl_uv, atol=1e-12)
