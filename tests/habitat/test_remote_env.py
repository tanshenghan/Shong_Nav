"""RemoteEnvServer <-> RemoteEnvClient round trip over a loopback ZMQ socket.

The habitat server package is imported straight from ``habitat_server/`` (it needs only
pyzmq for this module); the env is a tiny fake, so no simulator is involved.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "habitat_server") not in sys.path:
    sys.path.insert(0, str(ROOT / "habitat_server"))

pytest.importorskip("zmq")

from lightnav_habitat.remote_server import RemoteEnvServer  # noqa: E402

from lightnav.habitat.remote_env import RemoteEnvClient  # noqa: E402


class FakeEnv:
    """Duck-typed env: reset/step/close plus the optional initialize hook."""

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.resets = 0
        self.actions: list = []

    def initialize(self) -> None:
        self.initialized = True

    def reset(self, seed=None, options=None):
        self.resets += 1
        self.actions = []
        obs = {
            "rgb": np.full((4, 6, 3), self.resets, dtype=np.uint8),
            "instruction": {"text": f"episode {self.resets}"},
        }
        info = {
            "episode_id": f"ep{self.resets}",
            "scene_id": "scene",
            "habitat_time_step": 1.0,
            "lin_vel_range": [0.0, 0.25],
            "ang_vel_range": [-30.0, 30.0],
            "seed": seed,
        }
        return obs, info

    def step(self, action):
        if action == "boom":
            raise ValueError("bad action")
        self.actions.append(action)
        n = len(self.actions)
        obs = {"rgb": np.full((4, 6, 3), n, dtype=np.uint8)}
        info = {"steps": n, "distance_to_goal": np.float32(1.5), "success": 0.0}
        return obs, 0.0, n >= 2, False, info

    def close(self) -> None:
        self.closed = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Served:
    def __init__(self, env: FakeEnv, port: int, ready_file: Path) -> None:
        self.env = env
        self.port = port
        self.address = f"tcp://127.0.0.1:{port}"
        self.ready_file = ready_file
        self.server: RemoteEnvServer | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        # Constructed inside the thread: the server only installs SIGINT/SIGTERM
        # handlers from the main thread, so pytest's own handlers stay untouched.
        self.server = RemoteEnvServer(self.env, address=self.address)
        self.server.start(ready_file=str(self.ready_file))

    def start(self) -> "_Served":
        self.thread.start()
        deadline = time.time() + 10.0
        while not self.ready_file.exists():
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("remote env server did not become ready")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
        self.thread.join(timeout=5.0)


@pytest.fixture
def served(tmp_path):
    env = FakeEnv()
    s = _Served(env, _free_port(), tmp_path / "server.ready").start()
    try:
        yield s
    finally:
        s.stop()


def _client(served: _Served, **kwargs) -> RemoteEnvClient:
    kwargs.setdefault("timeout_ms", 5000)
    kwargs.setdefault("recv_retries", 0)
    return RemoteEnvClient(served.address, **kwargs)


def test_ready_file_is_written_after_initialize(served):
    assert served.ready_file.exists()
    assert served.env.initialized is True


def test_reset_step_close_roundtrip(served):
    client = _client(served)
    try:
        obs, info = client.reset()
        assert isinstance(obs["rgb"], np.ndarray)
        assert obs["rgb"].shape == (4, 6, 3) and obs["rgb"].dtype == np.uint8
        assert obs["instruction"] == {"text": "episode 1"}
        assert info["episode_id"] == "ep1" and info["scene_id"] == "scene"
        assert info["lin_vel_range"] == [0.0, 0.25]
        assert info["seed"] is None

        action = {
            "action": "velocity_control",
            "action_args": {"linear_velocity": 0.5, "angular_velocity": -0.25},
        }
        obs, reward, terminated, truncated, info = client.step(action)
        assert served.env.actions == [action]  # passed through verbatim
        assert reward == 0.0 and isinstance(reward, float)
        assert terminated is False and truncated is False
        assert info["steps"] == 1
        assert float(info["distance_to_goal"]) == pytest.approx(1.5)
        assert int(obs["rgb"][0, 0, 0]) == 1

        _obs, _reward, terminated, truncated, info = client.step(action)
        assert terminated is True and truncated is False
        assert info["steps"] == 2
    finally:
        client.close()

    served.thread.join(timeout=5.0)
    assert served.env.closed is True
    assert not served.thread.is_alive()
    assert served.server.socket is None  # server released its socket


def test_close_is_idempotent(served):
    client = _client(served)
    client.reset()
    client.close()
    client.close()  # no socket anymore: must not raise
    with pytest.raises(ConnectionError):
        client.reset()


def test_server_error_becomes_runtime_error_and_the_server_survives(served):
    client = _client(served)
    try:
        client.reset()
        with pytest.raises(RuntimeError, match="bad action"):
            client.step("boom")
        # The REQ/REP pair is still in sync: the next command works.
        obs, info = client.reset()
        assert info["episode_id"] == "ep2"
        assert served.env.resets == 2
    finally:
        client.close()


def test_unknown_command_is_an_error_reply(served):
    client = _client(served)
    try:
        with pytest.raises(RuntimeError, match="Unknown command"):
            client._send_command("render")
    finally:
        client.close()


def test_recv_timeout_without_a_server_raises_connection_error(tmp_path):
    # Nothing listens on this port; a REQ send always succeeds, the recv times out.
    client = RemoteEnvClient(f"tcp://127.0.0.1:{_free_port()}", timeout_ms=200, recv_retries=1)
    try:
        t0 = time.monotonic()
        with pytest.raises(ConnectionError, match="Timeout"):
            client.reset()
        # Two attempts of 200 ms, without resending in between.
        assert time.monotonic() - t0 >= 0.35
    finally:
        client.close()
