"""Minimal WebSocket client agent for the EVT-Bench evaluation driver.

Copy this file next to EVT-Bench's ``run.py`` and apply ``run_py.patch`` so that
``run.py --model-name trackvla --model-path ws://host:port`` imports
``evaluate_agent`` from here.  Each Habitat process opens one WebSocket to a
running ``lightnav-serve`` instance and, per step, sends the Spot jaw camera
frame as a base64 JPEG and receives a short trajectory of ``[forward_m,
lateral_m, yaw_rad]`` waypoints; the first waypoint is scaled into the
``agent_1_base_velocity`` action.

The file targets Python 3.9 and depends only on ``numpy``, ``Pillow`` and
``websocket-client`` (``pip install websocket-client``) plus the EVT-Bench
habitat-lab fork, which is imported lazily inside ``evaluate_agent`` so that the
pure helpers (``parse_actions_payload``, ``waypoint_to_base_velocity``) can be
imported and tested without a simulator.

The evaluation loop in ``evaluate_agent`` is adapted from the EVT-Bench
(TrackVLA) evaluation driver, which is distributed under CC BY-NC-SA 4.0; see
THIRD_PARTY_NOTICES.md in the LightNav-0 repository.  EVT-Bench itself is
not redistributed here.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import os.path as osp
import uuid
from typing import Any, Optional

import numpy as np
from PIL import Image

DEFAULT_SERVER_URL = os.environ.get("TRACKVLA_WS_URL", "ws://localhost:8050")
DEFAULT_RECV_TIMEOUT = float(os.environ.get("TRACKVLA_WS_TIMEOUT", "60"))

# Per-step maxima used to normalise a waypoint [forward, lateral, yaw] into the
# base_velocity action space [-1, 1]^3.  Waypoints use +lateral = left and
# +yaw = counter-clockwise, the same convention as habitat's velocity control,
# so no sign flip is applied.
WP_FWD_MAX = 0.375
WP_LAT_MAX = 0.25
WP_YAW_MAX = math.pi / 20.0

# Frame key of the Spot jaw camera in EVT-Bench's observation dict.
JAW_RGB_KEY = "agent_1_articulated_agent_jaw_rgb"

# Multi-agent action tuple passed to ``env.step``.  The task configs define
# distractor humans agent_2..agent_8, but the upstream EVT-Bench drivers
# (``baseline_agent.py`` / ``agent_uninavid.py``) only step agent_2..agent_5 and
# the remaining distractors stand still.  The default reproduces that published
# protocol; ``EVT_NUM_DISTRACTORS=6`` additionally steps agent_6 and agent_7
# (a stricter variant with two more moving distractors).  Numbers obtained with
# different settings are not comparable.
NUM_DISTRACTORS = int(os.environ.get("EVT_NUM_DISTRACTORS", "4"))
ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
) + tuple(
    f"agent_{i}_oracle_nav_randcoord_action_obstacle" for i in range(2, 2 + NUM_DISTRACTORS)
)

# Opt-in, NOT upstream behaviour: zero-scale the robot's visual nodes so the jaw
# camera does not see the robot body.  Off by default to match EVT-Bench.
HIDE_ROBOT_MESH = os.environ.get("EVT_HIDE_ROBOT_MESH", "0") == "1"


def parse_actions_payload(actions: Any) -> list[list[float]]:
    """Return the trajectory as a list of ``[fwd, lat, yaw]`` rows.

    Accepts the current server shape ``{"step": S, "actions": [[f, l, y], ...]}``,
    the legacy wrapped list ``[[[f, l, y], ...]]`` and a flat list whose length
    is divisible by 3.  ``None`` or an empty payload yields ``[]``.
    """
    if actions is None:
        return []

    if isinstance(actions, dict):
        trajectories = actions.get("actions", [])
    else:
        trajectories = actions

    if not trajectories:
        return []

    if (
        isinstance(trajectories, list)
        and len(trajectories) == 1
        and isinstance(trajectories[0], list)
        and trajectories[0]
        and isinstance(trajectories[0][0], (list, tuple))
    ):
        trajectories = trajectories[0]

    if isinstance(trajectories, list) and not isinstance(trajectories[0], (list, tuple)):
        if len(trajectories) % 3 != 0:
            raise ValueError("flat actions length must be divisible by 3")
        trajectories = [trajectories[i : i + 3] for i in range(0, len(trajectories), 3)]

    return [[float(wp[0]), float(wp[1]), float(wp[2])] for wp in trajectories]


def waypoint_to_base_velocity(wp) -> list[float]:
    """Scale one ``[fwd_m, lat_m, yaw_rad]`` waypoint into ``[vx, vy, vyaw]`` in [-1, 1]."""
    vx = float(np.clip(float(wp[0]) / WP_FWD_MAX, -1.0, 1.0))
    vy = float(np.clip(float(wp[1]) / WP_LAT_MAX, -1.0, 1.0))
    vyaw = float(np.clip(float(wp[2]) / WP_YAW_MAX, -1.0, 1.0))
    return [vx, vy, vyaw]


def encode_jpeg_b64(rgb: np.ndarray) -> str:
    """Encode an HxWx3 uint8 RGB frame as a base64 JPEG string (PIL default quality)."""
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TrackVLAClientAgent:
    """One WebSocket session per Habitat process: login once, reset per episode, next per step."""

    def __init__(
        self,
        result_path: str,
        server_url: str = DEFAULT_SERVER_URL,
        recv_timeout: float = DEFAULT_RECV_TIMEOUT,
    ):
        print(f"Initialize TrackVLA websocket client. Server: {server_url}")
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)

        self.server_url = server_url
        self.recv_timeout = float(recv_timeout)
        self.client_id = f"evt_bench_{uuid.uuid4().hex[:8]}_{os.getpid()}"
        self.ws = None
        self.seq = 0
        self.last_trajectory: Optional[list[list[float]]] = None
        self.last_action = [0.0, 0.0, 0.0]
        self.last_stop = None
        self.last_visible = None
        self.last_raw_text = ""
        self.last_latency_ms = 0.0

        self._connect()
        self.reset()

    # -- transport -----------------------------------------------------------------
    def _connect(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ImportError(
                "trackvla_client_agent needs the 'websocket-client' package: "
                "pip install websocket-client"
            ) from exc
        self.ws = websocket.create_connection(self.server_url, timeout=self.recv_timeout)
        self._send({"action": "login", "data": {"clientId": self.client_id}})
        resp = self._recv()
        if resp.get("data", {}).get("rc") != 0:
            raise RuntimeError(f"Login failed: {resp}")
        print(f"Logged in as {self.client_id}")

    def _send(self, payload: dict) -> None:
        self.ws.send(json.dumps(payload))

    def _recv(self) -> dict:
        return json.loads(self.ws.recv())

    # -- episode lifecycle ---------------------------------------------------------
    def reset(self, episode=None) -> None:
        """Clear per-episode state and ask the server to clear its frame buffer.

        ``episode`` is accepted for driver compatibility and ignored.  A failed
        reset round-trip reconnects (new socket + login) once.
        """
        self.last_trajectory = None
        self.last_stop = None
        self.last_visible = None
        self.last_raw_text = ""
        self.last_latency_ms = 0.0
        self.seq = 0

        if self.ws is not None:
            try:
                self._send({"action": "reset", "data": {}})
                self._recv()
            except Exception as exc:
                print(f"Reset ws error, reconnecting: {exc}")
                self._connect()

    def act(self, observations, instruction: str, episode_id=None) -> list[float]:
        """Send the current jaw frame and return the ``[vx, vy, vyaw]`` base velocity."""
        rgb = observations[JAW_RGB_KEY][:, :, :3]
        rgb_uint8 = np.asarray(rgb, dtype=np.uint8)

        self._send(
            {
                "action": "next",
                "data": {
                    "seq": self.seq,
                    "image": encode_jpeg_b64(rgb_uint8),
                    "instruction": instruction,
                },
            }
        )
        resp = self._recv()
        self.seq += 1

        data = resp.get("data", {})
        trajectory = None
        raw_text = data.get("raw_text", "") or ""
        latency_ms = float(data.get("latency_ms", 0.0) or 0.0)
        if data.get("rc") != 0:
            # Any server-side error: keep driving with the previous action.
            print(f"Server error: {data}")
            action = list(self.last_action)
        else:
            trajectory = parse_actions_payload(data.get("actions")) or None
            if trajectory is None:
                action = list(self.last_action)
            else:
                action = waypoint_to_base_velocity(trajectory[0])

        ep_str = f"ep={episode_id}" if episode_id is not None else "ep=?"
        raw_clean = raw_text.replace("<|im_end|>", "").strip() if raw_text else "(empty)"
        print(
            f"[{ep_str} step {self.seq:>3}] {latency_ms:6.1f}ms  raw={raw_clean}  "
            f"stop={data.get('stop')} visible={data.get('visible')} action={action}",
            flush=True,
        )

        self.last_trajectory = trajectory
        self.last_stop = data.get("stop")
        self.last_visible = data.get("visible")
        self.last_raw_text = raw_text
        self.last_latency_ms = latency_ms
        self.last_action = action
        return action

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


def _hide_robot_mesh(robot_agent) -> None:
    """Zero-scale the robot URDF's visual nodes (opt-in via EVT_HIDE_ROBOT_MESH=1)."""
    import magnum as mn

    for node in robot_agent.sim_obj.visual_scene_nodes:
        node.scaling = mn.Vector3(0, 0, 0)


def evaluate_agent(config, dataset_split, save_path, server_url=DEFAULT_SERVER_URL) -> None:
    """Run the EVT-Bench evaluation loop for one dataset shard against one server.

    Adapted from the EVT-Bench (TrackVLA) evaluation driver (CC BY-NC-SA 4.0).
    Note the benchmark's own convention: the instruction is read from the FIRST
    episode of the shard and reused for every episode in it (``first_init``);
    EVT-Bench's humanoid detector sensor freezes its target id the same way, so
    results depend on the shard count (``--split-num``, keep 30).

    Per episode this writes ``<save_path>/<scene>/<episode_id>.json`` with keys
    ``finish, status, success, following_rate, following_step, total_step,
    collision`` and ``<episode_id>_info.json`` with a per-step trace.
    """
    import habitat
    from habitat_sim.gfx import LightInfo, LightPositionModel
    from tqdm import trange

    agent = TrackVLAClientAgent(save_path, server_url=server_url)

    first_init = True
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        agent.reset()

        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            if first_init:
                instruction = env.current_episode.info["instruction"]
                first_init = False

            finished = False
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent
            if HIDE_ROBOT_MESH:
                _hide_robot_mesh(robot_agent)

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = "Normal"
            info = env.get_metrics()

            while not env.episode_over:
                record_info = {}

                obs = sim.get_sensor_observations()
                action = agent.act(obs, instruction, env.current_episode.episode_id)

                action_dict = {
                    "action": ACTION_NAMES,
                    "action_args": {"agent_1_base_vel": action},
                }

                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info["human_following"] == 1.0:
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0
                else:
                    print("Lost")

                dist_to_human = float(
                    np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)
                )
                if dist_to_human > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = "Lost"
                        finished = False
                        break

                record_info["step"] = iter_step
                record_info["trajectory"] = agent.last_trajectory
                record_info["dis_to_human"] = dist_to_human
                record_info["facing"] = info["human_following"]
                record_infos.append(record_info)

                if info["human_collision"] == 1.0:
                    print("Collision detected!")
                    status = "Collision"
                    finished = False
                    break

                print(
                    f"========== ID: {env.current_episode.episode_id} Step now is: {iter_step} "
                    f"action is: {action} dis_to_main_human: {dist_to_human} ============"
                )

            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()
            agent.reset(env.current_episode)

            if env.episode_over:
                finished = True

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(
                os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w"
            ) as f:
                json.dump(record_infos, f, indent=2)
            result["finish"] = finished
            result["status"] = status
            if iter_step < 300:
                result["success"] = info["human_following_success"] and info["human_following"]
            else:
                result["success"] = info["human_following"]
            result["following_rate"] = followed_step / iter_step
            result["following_step"] = followed_step
            result["total_step"] = iter_step
            result["collision"] = info["human_collision"]
            with open(
                os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w"
            ) as f:
                json.dump(result, f, indent=2)

    agent.close()
