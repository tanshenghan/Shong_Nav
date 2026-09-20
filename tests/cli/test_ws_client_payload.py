"""The reference WebSocket client's tolerant parsing of the ``actions`` field."""

from __future__ import annotations

import math

import pytest

from lightnav.cli import ws_client

NEW_DATA = {
    "actions": {
        "step": 12,
        "actions": [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]],
    },
    "stop": True,
    "visible": None,
}

LEGACY_DATA = {
    "actions": [[[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]],
}

FLAT_DATA = {"actions": [1.0, 0.25, 0.1, 2.0, -0.5, -0.2]}


def test_ws_client_extracts_new_actions_dict():
    assert ws_client._extract_waypoints(NEW_DATA) == [
        [1.0, 0.25, 0.1],
        [2.0, -0.5, -0.2],
    ]


def test_ws_client_extracts_legacy_actions_list():
    assert ws_client._extract_waypoints(LEGACY_DATA) == [
        [1.0, 0.25, 0.1],
        [2.0, -0.5, -0.2],
    ]


def test_ws_client_extracts_flat_actions_list():
    assert ws_client._extract_waypoints(FLAT_DATA) == [
        [1.0, 0.25, 0.1],
        [2.0, -0.5, -0.2],
    ]


def test_ws_client_rejects_incomplete_flat_actions():
    with pytest.raises(ValueError):
        ws_client._extract_waypoints({"actions": [1.0, 2.0]})


def test_velocity_scaling_constants_match_the_evt_client():
    assert ws_client.WP_FWD_MAX == 0.375
    assert ws_client.WP_LAT_MAX == 0.25
    assert ws_client.WP_YAW_MAX == pytest.approx(math.pi / 20.0)


def test_jpeg_round_trip_keeps_the_frame_shape():
    import base64
    import io

    import numpy as np
    from PIL import Image

    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    b64 = ws_client._encode_jpeg_b64(rgb)
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    assert decoded.shape == (6, 8, 3)
