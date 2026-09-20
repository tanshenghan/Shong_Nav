"""The EVT-Bench client agent's pure helpers, imported straight from ``evt_bench/``.

The file targets the benchmark's Python 3.9 habitat env; its helpers must import and
work here without habitat, magnum or websocket-client being present.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIENT_PATH = ROOT / "evt_bench" / "trackvla_client_agent.py"


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("evt_trackvla_client_agent", CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_pull_in_the_simulator_or_the_socket_library(client):
    for name in ("habitat", "habitat_sim", "magnum", "websocket"):
        assert name not in sys.modules, f"{name} must only be imported lazily"


def test_source_is_python39_compatible():
    tree = ast.parse(CLIENT_PATH.read_text(), feature_version=(3, 9))
    future = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    assert future and any(a.name == "annotations" for a in future[0].names)
    assert not any(isinstance(n, ast.Match) for n in ast.walk(tree))


# -- parse_actions_payload ----------------------------------------------------------------


ROWS = [[1.0, 0.25, 0.1], [2.0, -0.5, -0.2]]


def test_parse_accepts_the_current_dict_shape(client):
    assert client.parse_actions_payload({"step": 12, "actions": ROWS}) == ROWS


def test_parse_accepts_the_legacy_wrapped_list(client):
    assert client.parse_actions_payload([ROWS]) == ROWS


def test_parse_accepts_a_flat_list(client):
    assert client.parse_actions_payload([1.0, 0.25, 0.1, 2.0, -0.5, -0.2]) == ROWS


def test_parse_accepts_a_plain_list_of_rows_and_tuples(client):
    assert client.parse_actions_payload(ROWS) == ROWS
    assert client.parse_actions_payload([(1, 2, 3)]) == [[1.0, 2.0, 3.0]]


@pytest.mark.parametrize("payload", [None, [], {}, {"actions": []}, {"step": 3}])
def test_parse_returns_an_empty_list_for_nothing(client, payload):
    assert client.parse_actions_payload(payload) == []


def test_parse_rejects_an_incomplete_flat_list(client):
    with pytest.raises(ValueError, match="divisible by 3"):
        client.parse_actions_payload([1.0, 2.0])


def test_parse_coerces_json_numbers_to_floats(client):
    rows = client.parse_actions_payload(json.loads('{"actions": [[1, 0, 0]]}'))
    assert rows == [[1.0, 0.0, 0.0]]
    assert all(isinstance(v, float) for v in rows[0])


# -- waypoint_to_base_velocity ------------------------------------------------------------


def test_velocity_constants(client):
    assert client.WP_FWD_MAX == 0.375
    assert client.WP_LAT_MAX == 0.25
    assert client.WP_YAW_MAX == pytest.approx(math.pi / 20.0)


def test_waypoint_scales_by_the_per_step_maxima(client):
    vx, vy, vyaw = client.waypoint_to_base_velocity([0.1875, 0.125, math.pi / 40.0])
    assert vx == pytest.approx(0.5)
    assert vy == pytest.approx(0.5)
    assert vyaw == pytest.approx(0.5)


def test_waypoint_clips_to_the_unit_cube(client):
    assert client.waypoint_to_base_velocity([10.0, -10.0, 10.0]) == [1.0, -1.0, 1.0]
    assert client.waypoint_to_base_velocity([-10.0, 10.0, -10.0]) == [-1.0, 1.0, -1.0]


def test_waypoint_keeps_signs_lateral_left_and_yaw_ccw(client):
    vx, vy, vyaw = client.waypoint_to_base_velocity(np.array([0.0, 0.05, -0.01], dtype=np.float32))
    assert vx == 0.0
    assert vy > 0.0  # +lateral = left
    assert vyaw < 0.0  # -yaw = clockwise
    assert all(isinstance(v, float) for v in (vx, vy, vyaw))


def test_zero_waypoint_is_a_zero_command(client):
    assert client.waypoint_to_base_velocity([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# -- act(): response handling without a socket ----------------------------------------------


class _FakeWS:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[dict] = []

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self) -> str:
        return json.dumps(self.replies.pop(0))


def _agent(client, replies):
    agent = object.__new__(client.TrackVLAClientAgent)
    agent.ws = _FakeWS(replies)
    agent.seq = 0
    agent.last_trajectory = None
    agent.last_action = [0.0, 0.0, 0.0]
    agent.last_stop = None
    agent.last_visible = None
    agent.last_raw_text = ""
    agent.last_latency_ms = 0.0
    return agent


def _obs(client, value=7):
    # EVT-Bench's jaw camera frame is RGBA; the client sends the RGB channels.
    return {client.JAW_RGB_KEY: np.full((6, 8, 4), value, dtype=np.uint8)}


def test_act_sends_a_next_request_and_scales_the_first_waypoint(client):
    ok = {
        "action": "next",
        "data": {
            "rc": 0,
            "seq": 0,
            "actions": {"step": 1, "actions": [[0.375, 0.0, 0.0], [0.75, 0.0, 0.0]]},
            "stop": False,
            "visible": True,
            "latency_ms": 12.5,
            "raw_text": "<tpos_3><traj_7>",
        },
    }
    agent = _agent(client, [ok])

    action = agent.act(_obs(client), "follow the person", episode_id="e1")

    assert action == [1.0, 0.0, 0.0]
    sent = agent.ws.sent[0]
    assert sent["action"] == "next"
    assert sent["data"]["seq"] == 0
    assert sent["data"]["instruction"] == "follow the person"
    assert isinstance(sent["data"]["image"], str) and sent["data"]["image"]
    assert agent.seq == 1
    assert agent.last_trajectory == [[0.375, 0.0, 0.0], [0.75, 0.0, 0.0]]
    assert agent.last_stop is False and agent.last_visible is True
    assert agent.last_raw_text == "<tpos_3><traj_7>"
    assert agent.last_latency_ms == 12.5


def test_act_reuses_the_last_action_on_a_server_error_or_a_buffer_only_ack(client):
    ok = {"data": {"rc": 0, "actions": {"step": 1, "actions": [[0.1875, 0.0, 0.0]]}}}
    error = {"data": {"rc": 500, "seq": 1, "msg": "decode failed"}}
    ack = {"data": {"rc": 0, "seq": 2, "msg": "image received"}}
    agent = _agent(client, [ok, error, ack])

    first = agent.act(_obs(client), "follow")
    assert first == [0.5, 0.0, 0.0]
    assert agent.last_trajectory == [[0.1875, 0.0, 0.0]]

    assert agent.act(_obs(client), "follow") == first  # rc != 0 -> keep driving
    assert agent.last_trajectory is None  # nothing decoded on an error

    assert agent.act(_obs(client), "follow") == first  # no actions -> keep driving
    assert agent.last_trajectory is None  # an ack carries no waypoints (matches the reference client)
    assert agent.seq == 3


def test_encode_jpeg_b64_round_trips_the_frame_shape(client):
    import base64
    import io

    from PIL import Image

    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    b64 = client.encode_jpeg_b64(rgb)
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    assert decoded.shape == (6, 8, 3)


def test_action_tuple_starts_with_the_humanoid_and_the_robot(client):
    assert client.ACTION_NAMES[0] == "agent_0_humanoid_navigate_action"
    assert client.ACTION_NAMES[1] == "agent_1_base_velocity"
    assert client.HIDE_ROBOT_MESH is False  # upstream behaviour by default
