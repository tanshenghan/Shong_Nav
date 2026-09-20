"""Capture-time-aligned pose tracking MPC ported from robot_deploy."""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import casadi as ca
import numpy as np

CONTROL_RATE_HZ = 10.0
HORIZON = 5
MPC_DT_S = 0.1
WAYPOINT_DT_S = 0.1
TRACK_V_MAX = 1.5
OBJNAV_V_MAX = 0.8
W_MAX = 3.0
A_MAX_V = 2.0
A_MAX_W = 5.0
Q_WEIGHTS = (10.0, 10.0, 1.0)
R_WEIGHTS = (0.1, 0.1)
ODOM_MATCH_MAX_GAP_S = 0.3
ODOM_TIMEOUT_S = 0.5


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def project_body_to_world(
    waypoints: Sequence[Sequence[float]],
    capture_pose: Sequence[float],
) -> np.ndarray:
    """Project x-forward/y-left/yaw-CCW waypoints into the world frame."""
    pose = np.asarray(capture_pose, dtype=np.float64)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("capture_pose must be a finite 3-vector")
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    projected = [
        (
            float(pose[0]) + cosine * float(forward) - sine * float(lateral),
            float(pose[1]) + sine * float(forward) + cosine * float(lateral),
            wrap_angle(float(pose[2]) + float(yaw)),
        )
        for forward, lateral, yaw in waypoints
    ]
    return np.asarray(projected, dtype=np.float64).reshape((-1, 3))


def project_world_to_local(
    trajectory: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Express a world trajectory in a fixed frame at the current pose."""
    points = np.asarray(trajectory, dtype=np.float64)
    pose = np.asarray(origin, dtype=np.float64)
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    delta_x = points[:, 0] - pose[0]
    delta_y = points[:, 1] - pose[1]
    return np.column_stack(
        (
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
            points[:, 2] - pose[2],
        )
    )


def project_local_to_world(
    trajectory: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Transform a local trajectory into the world frame."""
    points = np.asarray(trajectory, dtype=np.float64)
    pose = np.asarray(origin, dtype=np.float64)
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    return np.column_stack(
        (
            pose[0] + cosine * points[:, 0] - sine * points[:, 1],
            pose[1] + sine * points[:, 0] + cosine * points[:, 1],
            pose[2] + points[:, 2],
        )
    )


def build_pose_aligned_reference(
    trajectory: np.ndarray,
    robot_pose: np.ndarray,
    *,
    horizon: int,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """Select the waypoints following the nearest weighted pose."""
    points = np.asarray(trajectory, dtype=np.float64).copy()
    pose = np.asarray(robot_pose, dtype=np.float64)
    cost_weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("trajectory must be a non-empty Nx3 array")
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError("robot_pose must be a finite 3-vector")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory must contain only finite values")
    if (
        horizon <= 0
        or cost_weights.shape != (3,)
        or not np.all(np.isfinite(cost_weights))
        or np.any(cost_weights <= 0.0)
    ):
        raise ValueError("horizon and pose weights must be positive")

    errors = points - pose
    errors[:, 2] = [wrap_angle(value) for value in errors[:, 2]]
    nearest_index = int(np.argmin(np.sum(errors * errors * cost_weights, axis=1)))
    indices = np.minimum(
        nearest_index + 1 + np.arange(horizon, dtype=np.int64),
        len(points) - 1,
    )
    reference = points[indices].copy()
    previous_yaw = float(pose[2])
    for waypoint in reference:
        waypoint[2] = previous_yaw + wrap_angle(waypoint[2] - previous_yaw)
        previous_yaw = float(waypoint[2])
    return reference


def _rollout_unicycle(
    pose: np.ndarray,
    controls: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    states = np.empty((len(controls) + 1, 3), dtype=np.float64)
    states[0] = np.asarray(pose, dtype=np.float64)
    for index, (linear_velocity, angular_velocity) in enumerate(controls):
        x, y, yaw = states[index]
        states[index + 1] = (
            x + dt_s * linear_velocity * math.cos(yaw),
            y + dt_s * linear_velocity * math.sin(yaw),
            yaw + dt_s * angular_velocity,
        )
    return states


def _reference_velocity_guess(
    pose: np.ndarray,
    reference: np.ndarray,
    dt_s: float,
    v_max: float,
    w_max: float,
) -> np.ndarray:
    points = np.vstack((np.asarray(pose, dtype=np.float64), reference.copy()))
    previous_yaw = float(points[0, 2])
    for point in points[1:]:
        point[2] = previous_yaw + wrap_angle(point[2] - previous_yaw)
        previous_yaw = float(point[2])
    deltas = np.diff(points, axis=0)
    return np.column_stack(
        (
            np.clip(np.hypot(deltas[:, 0], deltas[:, 1]) / dt_s, 0.0, v_max),
            np.clip(deltas[:, 2] / dt_s, -w_max, w_max),
        )
    )


class MPCController:
    """CasADi/IPOPT trajectory tracker for a velocity-driven unicycle."""

    def __init__(
        self,
        *,
        horizon: int,
        dt_s: float,
        w_max: float,
        a_max_v: float,
        a_max_w: float,
        q_weights: tuple[float, float, float],
        r_weights: tuple[float, float],
    ) -> None:
        self.horizon = int(horizon)
        self.dt_s = float(dt_s)
        self.w_max = float(w_max)
        self.q_weights = tuple(float(value) for value in q_weights)
        self.r_weights = tuple(float(value) for value in r_weights)
        if (
            self.horizon <= 0
            or self.dt_s <= 0.0
            or min(self.w_max, a_max_v, a_max_w) <= 0.0
        ):
            raise ValueError("invalid MPC horizon, timing, or limit")

        opti = ca.Opti()
        controls = opti.variable(self.horizon, 2)
        states = opti.variable(self.horizon + 1, 3)
        initial_state = opti.parameter(3)
        reference = opti.parameter(self.horizon, 3)
        previous_control = opti.parameter(2)
        v_max = opti.parameter()

        opti.subject_to(states[0, :] == initial_state.T)
        for index in range(self.horizon):
            yaw = states[index, 2]
            derivative = ca.horzcat(
                controls[index, 0] * ca.cos(yaw),
                controls[index, 0] * ca.sin(yaw),
                controls[index, 1],
            )
            opti.subject_to(
                states[index + 1, :] == states[index, :] + self.dt_s * derivative
            )

        q_matrix = ca.diag(ca.DM(self.q_weights))
        r_matrix = ca.diag(ca.DM(self.r_weights))
        objective = 0
        for index in range(self.horizon):
            error = states[index + 1, :] - reference[index, :]
            objective += ca.mtimes([error, q_matrix, error.T])
            objective += ca.mtimes(
                [controls[index, :], r_matrix, controls[index, :].T]
            )
        opti.minimize(objective)
        opti.subject_to(controls[:, 0] >= 0.0)
        opti.subject_to(controls[:, 0] <= v_max)
        opti.subject_to(opti.bounded(-self.w_max, controls[:, 1], self.w_max))

        dv_max = float(a_max_v) * self.dt_s
        dw_max = float(a_max_w) * self.dt_s
        opti.subject_to(
            opti.bounded(-dv_max, controls[0, 0] - previous_control[0], dv_max)
        )
        opti.subject_to(
            opti.bounded(-dw_max, controls[0, 1] - previous_control[1], dw_max)
        )
        for index in range(self.horizon - 1):
            opti.subject_to(
                opti.bounded(
                    -dv_max,
                    controls[index + 1, 0] - controls[index, 0],
                    dv_max,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -dw_max,
                    controls[index + 1, 1] - controls[index, 1],
                    dw_max,
                )
            )

        opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": 100,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1e-8,
                "ipopt.acceptable_obj_change_tol": 1e-6,
            },
        )
        self.opti = opti
        self.controls = controls
        self.states = states
        self.initial_state = initial_state
        self.reference = reference
        self.previous_control = previous_control
        self.v_max = v_max

    def solve(
        self,
        pose: np.ndarray,
        reference: np.ndarray,
        previous: tuple[float, float],
        v_max: float,
    ) -> tuple[tuple[float, float], np.ndarray]:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (self.horizon, 3):
            raise ValueError(f"reference must have shape ({self.horizon}, 3)")
        if not math.isfinite(v_max) or v_max <= 0.0:
            raise ValueError("v_max must be positive")
        self.opti.set_value(self.initial_state, np.asarray(pose, dtype=np.float64))
        self.opti.set_value(self.reference, reference)
        self.opti.set_value(self.previous_control, previous)
        self.opti.set_value(self.v_max, float(v_max))
        controls_guess = _reference_velocity_guess(
            pose,
            reference,
            self.dt_s,
            v_max,
            self.w_max,
        )
        self.opti.set_initial(self.controls, controls_guess)
        self.opti.set_initial(
            self.states,
            _rollout_unicycle(pose, controls_guess, self.dt_s),
        )
        solution = self.opti.solve()
        solved_controls = np.asarray(solution.value(self.controls), dtype=np.float64)
        solved_states = np.asarray(solution.value(self.states), dtype=np.float64)
        command = (
            float(np.clip(solved_controls[0, 0], 0.0, v_max)),
            float(np.clip(solved_controls[0, 1], -self.w_max, self.w_max)),
        )
        return command, solved_states.copy()


@dataclass(frozen=True)
class MpcSolveResult:
    generation: int
    command: tuple[float, float]
    reference: np.ndarray
    prediction: np.ndarray
    solve_ms: float


class MpcTracker:
    """ROS-free wrapper retaining the timing and parameters of vln_mpc."""

    def __init__(self) -> None:
        self.controller = MPCController(
            horizon=HORIZON,
            dt_s=MPC_DT_S,
            w_max=W_MAX,
            a_max_v=A_MAX_V,
            a_max_w=A_MAX_W,
            q_weights=Q_WEIGHTS,
            r_weights=R_WEIGHTS,
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vln-mpc")
        self._future: Future[MpcSolveResult] | None = None
        self._generation = 0
        self._trajectory: np.ndarray | None = None
        self.command = (0.0, 0.0)
        self.previous_command = (0.0, 0.0)
        self.reference = np.empty((0, 3), dtype=np.float64)
        self.prediction = np.empty((0, 3), dtype=np.float64)
        self.solve_ms: float | None = None
        self.error = ""

    def reset(self) -> None:
        self._generation += 1
        self._trajectory = None
        self.command = (0.0, 0.0)
        self.previous_command = (0.0, 0.0)
        self.reference = np.empty((0, 3), dtype=np.float64)
        self.prediction = np.empty((0, 3), dtype=np.float64)
        self.solve_ms = None
        self.error = ""

    def close(self) -> None:
        self._generation += 1
        self._pool.shutdown(wait=False, cancel_futures=True)

    def set_body_path(
        self,
        waypoints: Sequence[Sequence[float]],
        capture_pose: Sequence[float],
    ) -> None:
        self._trajectory = project_body_to_world(waypoints, capture_pose)
        self._generation += 1
        self.error = ""
        if len(self._trajectory) == 0:
            self.command = (0.0, 0.0)
            self.previous_command = (0.0, 0.0)

    def submit(self, current_pose: Sequence[float]) -> None:
        if self._future is not None or self._trajectory is None:
            return
        if len(self._trajectory) == 0:
            self.command = (0.0, 0.0)
            self.previous_command = (0.0, 0.0)
            return
        pose = np.asarray(current_pose, dtype=np.float64)
        try:
            reference = build_pose_aligned_reference(
                self._trajectory,
                pose,
                horizon=HORIZON,
                weights=Q_WEIGHTS,
            )
        except ValueError as exc:
            self.error = str(exc)
            self.command = (0.0, 0.0)
            self.previous_command = (0.0, 0.0)
            return
        self._future = self._pool.submit(
            self._solve,
            self._generation,
            pose.copy(),
            reference,
            self.previous_command,
        )

    def poll(self) -> tuple[float, float] | None:
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        try:
            result = future.result()
        except Exception as exc:
            self.error = f"MPC solve failed: {exc}"
            self.command = (0.0, 0.0)
            self.previous_command = (0.0, 0.0)
            return self.command
        if result.generation != self._generation:
            return None
        self.command = result.command
        self.previous_command = result.command
        self.reference = result.reference
        self.prediction = result.prediction
        self.solve_ms = result.solve_ms
        self.error = ""
        return self.command

    def _solve(
        self,
        generation: int,
        pose: np.ndarray,
        reference: np.ndarray,
        previous: tuple[float, float],
    ) -> MpcSolveResult:
        started_s = time.perf_counter()
        local_reference = project_world_to_local(reference, pose)
        local_pose = np.zeros(3, dtype=np.float64)
        command, local_prediction = self.controller.solve(
            local_pose,
            local_reference,
            previous,
            OBJNAV_V_MAX,
        )
        prediction = project_local_to_world(local_prediction, pose)
        return MpcSolveResult(
            generation=generation,
            command=command,
            reference=reference,
            prediction=prediction,
            solve_ms=(time.perf_counter() - started_s) * 1000.0,
        )
