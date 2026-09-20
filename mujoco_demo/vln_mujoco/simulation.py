from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image

from .mpc import TRACK_V_MAX, W_MAX
from .model import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    PHYSICS_DT,
    THIRD_PERSON_CAMERA_NAME,
    WHEEL_RADIUS,
    WHEEL_TRACK,
    build_model,
)


@dataclass(frozen=True)
class CameraFrame:
    stamp_ns: int
    rgb: bytes
    jpeg: bytes
    pose: tuple[float, float, float]


class Simulation:
    def __init__(self) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._left_joint = self.model.joint("left_wheel_joint").id
        self._right_joint = self.model.joint("right_wheel_joint").id
        self._base_joint = self.model.joint("base_joint").id
        self._base_qpos = self.model.jnt_qposadr[self._base_joint]
        self._base_dof = self.model.jnt_dofadr[self._base_joint]
        self._left_qpos = self.model.jnt_qposadr[self._left_joint]
        self._right_qpos = self.model.jnt_qposadr[self._right_joint]
        self._left_dof = self.model.jnt_dofadr[self._left_joint]
        self._right_dof = self.model.jnt_dofadr[self._right_joint]
        self._yaw = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._command = (0.0, 0.0)
        self._command_at = 0.0
        self._frame: CameraFrame | None = None
        self._third_person_frame: CameraFrame | None = None
        self._frame_count = 0
        self._started_at = time.monotonic()
        mujoco.mj_forward(self.model, self.data)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mujoco", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None

    def reset(self) -> None:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            self._command = (0.0, 0.0)
            self._command_at = 0.0
            self._yaw = 0.0
            mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, linear: float, angular: float) -> None:
        # Physical sanity bound only; per-mode VLN limits live in the MPC.
        linear = max(-TRACK_V_MAX, min(TRACK_V_MAX, float(linear)))
        angular = max(-W_MAX, min(W_MAX, float(angular)))
        with self._lock:
            self._command = (linear, angular)
            self._command_at = time.monotonic()

    def frame(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def third_person_frame(self) -> CameraFrame | None:
        with self._lock:
            return self._third_person_frame

    @staticmethod
    def _encode_frame(
        stamp_ns: int,
        rgb: np.ndarray,
        pose: tuple[float, float, float],
    ) -> CameraFrame:
        output = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(output, format="JPEG", quality=82)
        return CameraFrame(stamp_ns, rgb.tobytes(), output.getvalue(), pose)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            qpos = self.data.qpos
            x, y, z = (float(value) for value in qpos[self._base_qpos : self._base_qpos + 3])
            qw, qx, qy, qz = (
                float(value)
                for value in qpos[self._base_qpos + 3 : self._base_qpos + 7]
            )
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            left = float(self.data.qvel[self._left_dof])
            right = float(self.data.qvel[self._right_dof])
            linear = WHEEL_RADIUS * (left + right) / 2.0
            angular = WHEEL_RADIUS * (right - left) / WHEEL_TRACK
            command = self._command
            frame = self._frame
            elapsed = max(time.monotonic() - self._started_at, 1e-6)
            return {
                "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
                "velocity": {"linear": linear, "angular": angular},
                "command": {"linear": command[0], "angular": command[1]},
                "sim_time": float(self.data.time),
                "camera": {
                    "ready": frame is not None,
                    "stamp_ns": frame.stamp_ns if frame is not None else None,
                    "fps": self._frame_count / elapsed,
                },
            }

    def _advance_kinematics(self, now: float) -> None:
        linear, angular = self._command
        if self._command_at and now - self._command_at > 0.35:
            linear, angular = 0.0, 0.0
            self._command = (0.0, 0.0)

        previous_yaw = self._yaw
        self._yaw = math.atan2(
            math.sin(previous_yaw + angular * PHYSICS_DT),
            math.cos(previous_yaw + angular * PHYSICS_DT),
        )
        heading = previous_yaw + angular * PHYSICS_DT / 2.0
        self.data.qpos[self._base_qpos] += linear * math.cos(heading) * PHYSICS_DT
        self.data.qpos[self._base_qpos + 1] += linear * math.sin(heading) * PHYSICS_DT
        self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7] = (
            math.cos(self._yaw / 2.0),
            0.0,
            0.0,
            math.sin(self._yaw / 2.0),
        )

        left = (linear - angular * WHEEL_TRACK / 2.0) / WHEEL_RADIUS
        right = (linear + angular * WHEEL_TRACK / 2.0) / WHEEL_RADIUS
        self.data.qpos[self._left_qpos] += left * PHYSICS_DT
        self.data.qpos[self._right_qpos] += right * PHYSICS_DT
        self.data.qvel.fill(0.0)
        self.data.qvel[self._base_dof] = linear * math.cos(self._yaw)
        self.data.qvel[self._base_dof + 1] = linear * math.sin(self._yaw)
        self.data.qvel[self._base_dof + 5] = angular
        self.data.qvel[self._left_dof] = left
        self.data.qvel[self._right_dof] = right
        self.data.time += PHYSICS_DT
        mujoco.mj_forward(self.model, self.data)

    def _run(self) -> None:
        renderer = mujoco.Renderer(self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        next_step = time.monotonic()
        next_frame = next_step
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_step:
                    time.sleep(min(next_step - now, 0.002))
                    continue
                with self._lock:
                    self._advance_kinematics(now)
                next_step += PHYSICS_DT
                if now - next_step > 0.25:
                    next_step = now
                if now >= next_frame:
                    with self._lock:
                        renderer.update_scene(self.data, camera=CAMERA_NAME)
                        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        renderer.update_scene(self.data, camera=THIRD_PERSON_CAMERA_NAME)
                        third_person_rgb = np.asarray(
                            renderer.render(), dtype=np.uint8
                        ).copy()
                        capture_pose = (
                            float(self.data.qpos[self._base_qpos]),
                            float(self.data.qpos[self._base_qpos + 1]),
                            self._yaw,
                        )
                    stamp_ns = time.time_ns()
                    frame = self._encode_frame(stamp_ns, rgb, capture_pose)
                    third_person_frame = self._encode_frame(
                        stamp_ns, third_person_rgb, capture_pose
                    )
                    with self._lock:
                        self._frame = frame
                        self._third_person_frame = third_person_frame
                        self._frame_count += 1
                    next_frame = now + 0.05
        finally:
            renderer.close()
