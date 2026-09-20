"""VLN-CE velocity policy: model trajectory tokens -> Habitat ``velocity_control`` actions.

On every ``act`` the latest RGB frame is pushed into the frame buffer, one
inference step is run with the VLN-CE trajectory prompt, the emitted token(s)
are decoded to ``(H, 3)`` robot-local waypoints ``[forward_m, lateral_m,
yaw_rad]`` and the first velocity-relevant waypoint is mapped to a normalized
``velocity_control`` action dict.

A decode failure (no trajectory token, out-of-range id, wrong RVQ level count)
falls back to a zero-velocity command. Habitat's ``velocity_control`` treats a
command below its minimum speeds as a stop, so that fallback ends the episode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lightnav.tracking import TrackingAgent
from lightnav.velocity import first_waypoint_to_velocity_cmd
from lightnav.vln_utils import parse_rvq_action_tokens, parse_traj_token

logger = logging.getLogger(__name__)


def extract_instruction(obs: dict[str, Any] | None) -> str:
    """Return the instruction text of an observation (``str`` or ``{"text": str}``), else ``""``."""
    if obs is None:
        return ""
    instruction_raw = obs.get("instruction", None)
    if instruction_raw is None:
        return ""
    if isinstance(instruction_raw, dict):
        return instruction_raw.get("text", "")
    return str(instruction_raw)


def select_action_waypoint(waypoints: np.ndarray, atol: float = 1e-6) -> tuple[int, np.ndarray]:
    """Return the first waypoint that can drive unicycle velocity, skipping leading origin rows."""
    for idx, waypoint in enumerate(waypoints):
        forward_or_yaw = np.asarray([waypoint[0], waypoint[2]], dtype=np.float32)
        if np.any(np.abs(forward_or_yaw) > atol):
            return idx, waypoint
    return 0, waypoints[0]


_select_action_waypoint = select_action_waypoint  # historical private name


class TrajVocabVLNCEPolicy:
    """Habitat VLN-CE policy composed over ``TrackingAgent``.

    Exactly one of ``centroids`` (flat ``<traj_k>`` vocabulary, ``(K, H, 3)``
    array or ``.npy`` path) or ``rvq_bundle`` (``<act_l*>`` RVQ decoding) must be
    given, and it must agree with the checkpoint's ``action_method`` when the
    engine exposes one.

    ``dt``, ``lin_vel_range`` (m/s) and ``ang_vel_range`` (deg/s) are the
    environment's ``velocity_control`` settings, reported by the Habitat server
    in ``info`` on reset.
    """

    def __init__(
        self,
        engine: Any,
        centroids: np.ndarray | str | Path | None = None,
        *,
        dt: float,
        lin_vel_range: tuple[float, float],
        ang_vel_range: tuple[float, float],
        num_history_frames: int = 64,
        rvq_bundle: Any = None,
    ) -> None:
        if (centroids is None) == (rvq_bundle is None):
            raise ValueError("pass exactly one of `centroids` (flat) or `rvq_bundle` (rvq).")
        self.method: str = "rvq" if rvq_bundle is not None else "flat"

        centroids_arr: np.ndarray | None = None
        if self.method == "flat":
            if isinstance(centroids, (str, Path)):
                arr = np.load(Path(centroids))
            else:
                arr = np.asarray(centroids)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"centroids must have shape (K, H, 3), got {arr.shape}")
            if arr.shape[0] < 2 or arr.shape[1] < 1:
                raise ValueError(
                    f"centroids must have K>=2 and H>=1, got K={arr.shape[0]}, H={arr.shape[1]}"
                )
            centroids_arr = np.ascontiguousarray(arr, dtype=np.float32)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # The sample builder swaps in the RVQ output sentence when the checkpoint's
        # action_method is "rvq", so the decoder must agree with it.
        bundle_method = getattr(getattr(engine, "bundle", None), "action_method", None)
        if bundle_method is not None and (bundle_method == "rvq") != (self.method == "rvq"):
            given = "an RVQ bundle" if self.method == "rvq" else "flat centroids"
            raise ValueError(
                f"checkpoint action_tokenizer method is {bundle_method!r} but the policy was "
                f"given {given}; pass --action_tokenizer_bundle for rvq checkpoints or "
                f"--traj_vocab_path for flat ones."
            )

        self.engine = engine
        self.dt: float = float(dt)
        self.lin_vel_range: tuple[float, float] = (float(lin_vel_range[0]), float(lin_vel_range[1]))
        self.ang_vel_range: tuple[float, float] = (float(ang_vel_range[0]), float(ang_vel_range[1]))
        self.num_history_frames: int = int(num_history_frames)

        self.agent = TrackingAgent(
            engine,
            centroids=centroids_arr,
            num_history_frames=self.num_history_frames,
            rvq_bundle=rvq_bundle,
        )
        self.centroids: np.ndarray | None = self.agent.centroids
        self.rvq = self.agent.rvq
        self.K: int | None = self.agent.K
        self.H: int = self.agent.H
        self.instruction: str = ""

        self._last_cluster_id: int | None = None
        self._last_rvq_codes: list[int] | None = None
        self._last_waypoints: np.ndarray | None = None
        self._last_action_waypoint_index: int | None = None
        self._last_action_waypoint: np.ndarray | None = None
        self._last_raw_text: str = ""

    def _clear_last(self) -> None:
        self._last_cluster_id = None
        self._last_rvq_codes = None
        self._last_waypoints = None
        self._last_action_waypoint_index = None
        self._last_action_waypoint = None
        self._last_raw_text = ""

    def reset(self, obs: dict[str, Any] | None = None) -> None:
        """Reset the frame buffer and seed the instruction from ``obs`` if given."""
        self.instruction = extract_instruction(obs) if obs is not None else ""
        self.agent.reset(self.instruction)
        self._clear_last()

    def observe(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        """Push the RGB frame into the history buffer without running inference."""
        self.agent.observe(obs["rgb"])

    def stop_action(self) -> dict:
        """The explicit STOP: a zero velocity_control command (sets ``is_stop_called``)."""
        return first_waypoint_to_velocity_cmd(
            np.zeros(3, dtype=np.float32), self.dt, self.lin_vel_range, self.ang_vel_range
        )

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict:
        """Run one inference step and return a Habitat ``velocity_control`` action dict."""
        self.agent.observe(obs["rgb"])
        text, _ = self.engine.generate_from_frames(
            self.agent._get_video_tensor(),
            self.agent.instruction,
            predict_horizon=1,
            frame_ids=list(self.agent._history_frame_ids),
            task_type="vlnce_traj",
        )
        self._last_raw_text = text

        try:
            waypoints, _ = self.agent.decode_waypoints(text)
        except ValueError as exc:
            logger.warning("%s decode failure (%s); zero-velocity fallback", self.method, exc)
            self._last_cluster_id = -1
            self._last_rvq_codes = None
            self._last_waypoints = None
            self._last_action_waypoint_index = None
            self._last_action_waypoint = None
            return first_waypoint_to_velocity_cmd(
                np.zeros(3, dtype=np.float32), self.dt, self.lin_vel_range, self.ang_vel_range
            )

        # decode_waypoints already validated the tokens; re-parse only to expose the ids.
        if self.method == "rvq":
            codes = parse_rvq_action_tokens(text)
            self._last_cluster_id = int(codes[0])
            self._last_rvq_codes = [int(c) for c in codes]
        else:
            self._last_cluster_id = int(parse_traj_token(text))
            self._last_rvq_codes = None

        self._last_waypoints = np.array(waypoints, dtype=np.float32, copy=True)
        waypoint_index, waypoint = select_action_waypoint(self._last_waypoints)
        self._last_action_waypoint_index = int(waypoint_index)
        self._last_action_waypoint = waypoint.copy()
        return first_waypoint_to_velocity_cmd(
            waypoint, self.dt, self.lin_vel_range, self.ang_vel_range
        )

    def get_info(self) -> dict[str, Any]:
        """Return the last predicted trajectory and ids for logging; minimal dict before any act.

        After a failed decode there is no ``predicted_traj``, but ``raw_text`` still
        carries what the model said so logs and rendered videos can show it.
        """
        if self._last_waypoints is None or self._last_cluster_id is None:
            return {"num_history_frames": self.num_history_frames, "raw_text": self._last_raw_text}
        return {
            "predicted_traj": self._last_waypoints,
            "cluster_id": self._last_cluster_id,
            "rvq_codes": self._last_rvq_codes,
            "raw_text": self._last_raw_text,
            "action_waypoint_index": self._last_action_waypoint_index,
            "action_waypoint": self._last_action_waypoint,
            "num_history_frames": self.num_history_frames,
        }
