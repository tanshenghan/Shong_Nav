"""Small Unitree SDK2 transport used by the ROS adapter."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
        ObstaclesAvoidClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
        LowState_,
        SportModeState_,
    )
except ImportError as exc:  # pragma: no cover - depends on the robot runtime
    ChannelFactoryInitialize = None
    ChannelSubscriber = None
    ObstaclesAvoidClient = None
    SportClient = None
    LowState_ = None
    SportModeState_ = None
    SDK_IMPORT_ERROR: ImportError | None = exc
else:
    SDK_IMPORT_ERROR = None


@dataclass(frozen=True)
class Go2State:
    received_ns: int
    received_s: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    velocity: tuple[float, float, float]
    yaw_speed: float
    body_height: float
    mode: int
    gait_type: int
    error_code: int


@dataclass(frozen=True)
class Go2Battery:
    received_s: float
    percentage: float | None
    voltage: float
    motors_ok: bool | None


GO2_MODE_NAMES = {
    0: "idle",
    1: "balance_stand",
    2: "pose",
    3: "locomotion",
    4: "reserved",
    5: "lie_down",
    6: "joint_lock",
    7: "damping",
    8: "recovery_stand",
    9: "reserved",
    10: "sit",
    11: "front_flip",
    12: "front_jump",
    13: "front_pounce",
}


def normalized_mode(
    mode: int,
    *,
    error_code: int | None = None,
    body_height: float | None = None,
) -> str:
    """Map Unitree's numeric sport mode onto the shared web interface."""
    # On the tested Go2, mode remains 0 both while upright and while lying
    # down.  The latter is identified by error_code 1001 and a low body.
    if error_code == 1001 or (
        body_height is not None and body_height < 0.2
    ):
        return "DAMPING"
    # This Go2 keeps reporting mode 0 after the physical START button has
    # unlocked locomotion.  Upright mode 0 is therefore the robot's
    # walk-ready state in the shared interface.
    if mode == 0 and body_height is not None:
        return "WALK"
    if mode in {1, 3}:
        return "WALK"
    if mode in {0, 2, 8}:
        return "STAND"
    if mode in {5, 10}:
        return "SIT"
    if mode == 7:
        return "DAMPING"
    return "UNKNOWN"


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value[index]) for index in range(length))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Go2 {name}") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"non-finite Go2 {name}")
    return result


def state_from_message(
    message: Any,
    *,
    received_ns: int | None = None,
    received_s: float | None = None,
) -> Go2State:
    """Copy a DDS SportModeState into a transport-independent value."""
    position = _finite_vector(message.position, 3, "position")
    velocity = _finite_vector(message.velocity, 3, "velocity")
    quaternion_wxyz = _finite_vector(
        message.imu_state.quaternion, 4, "quaternion"
    )
    norm = math.sqrt(sum(value * value for value in quaternion_wxyz))
    if norm <= 1e-6:
        raise ValueError("invalid zero Go2 quaternion")
    qw, qx, qy, qz = (value / norm for value in quaternion_wxyz)
    yaw_speed = float(message.yaw_speed)
    body_height = float(message.body_height)
    if not math.isfinite(yaw_speed) or not math.isfinite(body_height):
        raise ValueError("non-finite Go2 motion state")
    return Go2State(
        received_ns=time.time_ns() if received_ns is None else int(received_ns),
        received_s=time.monotonic() if received_s is None else float(received_s),
        position=(position[0], position[1], position[2]),
        orientation_xyzw=(qx, qy, qz, qw),
        velocity=(velocity[0], velocity[1], velocity[2]),
        yaw_speed=yaw_speed,
        body_height=body_height,
        mode=int(message.mode),
        gait_type=int(message.gait_type),
        error_code=int(message.error_code),
    )


def battery_from_message(
    message: Any, *, received_s: float | None = None
) -> Go2Battery:
    bms_state = message.bms_state
    percentage = float(bms_state.soc)
    voltage = float(message.power_v)
    if not math.isfinite(percentage) or not math.isfinite(voltage):
        raise ValueError("non-finite Go2 battery state")
    cells = tuple(int(value) for value in bms_state.cell_vol)
    bms_populated = any(cells) or any(
        int(value)
        for value in (
            bms_state.status,
            bms_state.current,
            bms_state.cycle,
        )
    )
    motor_states = tuple(message.motor_state[:12])
    motors_ok = (
        all(
            int(state.mode) != 0 and int(state.temperature) > 0
            for state in motor_states
        )
        if len(motor_states) == 12
        else None
    )
    return Go2Battery(
        received_s=time.monotonic() if received_s is None else float(received_s),
        percentage=(
            max(0.0, min(100.0, percentage)) if bms_populated else None
        ),
        voltage=voltage,
        motors_ok=motors_ok,
    )


class UnitreeClient:
    """Serialize Unitree API calls and forward the latest DDS state."""

    ACTION_METHODS = {
        "stand": "StandUp",
        "walk": "BalanceStand",
        "sit": "StandDown",
        "stop": "StopMove",
        "damp": "Damp",
    }

    def __init__(
        self,
        *,
        network_interface: str,
        domain_id: int,
        timeout_s: float,
        state_callback: Callable[[Go2State], None],
        battery_callback: Callable[[Go2Battery], None],
        sent_callback: Callable[[tuple[float, float, float]], None],
        event_callback: Callable[[dict[str, Any]], None],
        error_callback: Callable[[str], None],
    ) -> None:
        if SDK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "unitree_sdk2py is required by go2_adapter"
            ) from SDK_IMPORT_ERROR
        if not network_interface:
            raise ValueError("network_interface must not be empty")
        self._state_callback = state_callback
        self._battery_callback = battery_callback
        self._sent_callback = sent_callback
        self._event_callback = event_callback
        self._error_callback = error_callback
        self._last_battery_s = float("-inf")
        self._condition = threading.Condition()
        self._actions: deque[str] = deque()
        self._latest_move: tuple[float, float, float] | None = None
        self._api_enabled = False
        self._closed = False
        self._timeout_s = float(timeout_s)

        ChannelFactoryInitialize(int(domain_id), network_interface)
        self._sport = SportClient()
        self._sport.SetTimeout(self._timeout_s)
        self._sport.Init()
        self._obstacles = ObstaclesAvoidClient()
        self._obstacles.SetTimeout(self._timeout_s)
        self._obstacles.Init()

        self._state_subscriber = ChannelSubscriber(
            "rt/lf/sportmodestate", SportModeState_
        )
        self._state_subscriber.Init(self._on_state, 10)
        self._battery_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._battery_subscriber.Init(self._on_battery, 1)
        self._worker = threading.Thread(
            target=self._command_loop,
            name="go2-sdk-command",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _result_code(result: Any) -> int:
        if result is None:
            return 0
        if isinstance(result, tuple):
            result = result[0]
        return int(result)

    def _on_state(self, message: Any) -> None:
        now_s = time.monotonic()
        try:
            self._state_callback(
                state_from_message(
                    message,
                    received_ns=time.time_ns(),
                    received_s=now_s,
                )
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._error_callback(str(exc))

    def _on_battery(self, message: Any) -> None:
        now_s = time.monotonic()
        if now_s - self._last_battery_s < 1.0:
            return
        self._last_battery_s = now_s
        try:
            self._battery_callback(
                battery_from_message(message, received_s=now_s)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._error_callback(str(exc))

    def move(self, linear: float, lateral: float, angular: float) -> bool:
        command = (float(linear), float(lateral), float(angular))
        with self._condition:
            if self._closed:
                return False
            self._latest_move = command
            self._condition.notify()
        return True

    def request_action(self, action: str) -> bool:
        if action not in self.ACTION_METHODS:
            raise ValueError(f"unknown Go2 action {action!r}")
        with self._condition:
            if self._closed:
                return False
            self._actions.append(action)
            self._condition.notify()
        return True

    def release_control(self) -> bool:
        """Queue a zero command followed by return to remote control."""
        with self._condition:
            if self._closed:
                return False
            self._latest_move = None
            self._actions.append("release_control")
            self._condition.notify()
        return True

    def _enable_api_control(self) -> bool:
        if self._api_enabled:
            return True
        result = self._obstacles.UseRemoteCommandFromApi(True)
        code = self._result_code(result)
        if code != 0:
            self._error_callback(f"Go2 API control acquisition failed: {code}")
            return False
        self._api_enabled = True
        self._event_callback(
            {"action": "acquire_control", "result": 0, "success": True}
        )
        return True

    def _release_api_control(self) -> None:
        if not self._api_enabled:
            return
        zero_code = self._result_code(self._obstacles.Move(0.0, 0.0, 0.0))
        result = self._obstacles.UseRemoteCommandFromApi(False)
        code = self._result_code(result)
        if zero_code == 0:
            self._sent_callback((0.0, 0.0, 0.0))
        self._event_callback(
            {
                "action": "release_control",
                "result": code,
                "success": zero_code == 0 and code == 0,
            }
        )
        if zero_code != 0:
            self._error_callback(f"Go2 zero command failed: {zero_code}")
        if code != 0:
            self._error_callback(f"Go2 API control release failed: {code}")
            return
        self._api_enabled = False

    def _command_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or bool(self._actions)
                    or self._latest_move is not None
                )
                if self._actions:
                    action = self._actions.popleft()
                    command = None
                elif self._latest_move is not None:
                    action = None
                    command = self._latest_move
                    self._latest_move = None
                else:
                    break
            try:
                if command is not None:
                    if not self._enable_api_control():
                        continue
                    result = self._obstacles.Move(*command)
                    code = self._result_code(result)
                    if code == 0:
                        self._sent_callback(command)
                    else:
                        self._error_callback(f"Go2 Move failed: {code}")
                    continue
                if action == "release_control":
                    self._release_api_control()
                    continue
                method = getattr(self._sport, self.ACTION_METHODS[action])
                result = method()
                code = self._result_code(result)
                self._event_callback(
                    {"action": action, "result": code, "success": code == 0}
                )
                if code != 0:
                    self._error_callback(
                        f"Go2 {self.ACTION_METHODS[action]} failed: {code}"
                    )
            except Exception as exc:  # pragma: no cover - hardware error path
                self._error_callback(f"Go2 SDK call failed: {exc}")

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._actions.clear()
            self._latest_move = None
            self._condition.notify_all()
        for subscriber in (
            self._state_subscriber,
            self._battery_subscriber,
        ):
            try:
                subscriber.Close()
            except Exception:
                pass
        self._worker.join(timeout=self._timeout_s + 1.0)
        if self._worker.is_alive():
            return
        try:
            self._release_api_control()
        except Exception:
            pass
