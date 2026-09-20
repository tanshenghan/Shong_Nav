"""Bridge the TRON high-level WebSocket API to a generic ROS 2 interface."""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlparse

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster

from .safety import select_command

try:
    import websocket
except ImportError as exc:  # pragma: no cover - reported on the robot
    raise RuntimeError("python3-websocket is required by tron_adapter") from exc


VALID_SOURCES = {"disabled", "manual", "auto"}
MODE_REQUESTS = {
    "stand": "request_stand_mode",
    "walk": "request_walk_mode",
    "sit": "request_sitdown",
}


class TronAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("tron_adapter")
        self.declare_parameter("robot_url", Parameter.Type.STRING)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("input_watchdog_s", 0.35)
        self.declare_parameter("zero_frames", 5)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self.robot_url = self._required_string_parameter("robot_url")
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.input_watchdog_s = float(
            self.get_parameter("input_watchdog_s").value
        )
        self.final_zero_frames = int(self.get_parameter("zero_frames").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        if (
            self.control_rate_hz <= 0.0
            or self.input_watchdog_s <= 0.0
            or self.final_zero_frames < 0
            or not self.odom_frame
            or not self.base_frame
        ):
            raise ValueError("invalid TRON adapter parameter")

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, "diagnostics", 10
        )
        self.odom_pub = self.create_publisher(Odometry, "odom", 20)
        self.sent_pub = self.create_publisher(
            TwistStamped, "cmd_vel", 20
        )
        self.source_pub = self.create_publisher(
            String, "control/source", latched
        )
        self.event_pub = self.create_publisher(String, "robot/events", 20)
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.publish_tf else None
        )

        self.create_subscription(
            TwistStamped, "web/cmd_vel", self._on_manual_command, 10
        )
        self.create_subscription(
            TwistStamped, "mpc/cmd_vel", self._on_auto_command, 10
        )
        self.create_service(
            SetBool, "control/set_manual", self._set_manual_enabled
        )
        self.create_service(
            SetBool, "control/set_auto", self._set_auto_enabled
        )
        self.create_service(Trigger, "control/stop", self._software_stop)
        self.create_service(Trigger, "robot/stand", self._request_stand)
        self.create_service(Trigger, "robot/walk", self._request_walk)
        self.create_service(Trigger, "robot/sit", self._request_sit)
        self.create_service(
            Trigger, "robot/emergency_stop", self._vendor_emergency_stop
        )

        self._lock = threading.Lock()
        self._connected = False
        self._robot_id = ""
        self._robot_info: dict[str, Any] = {}
        self._source = "disabled"
        self._manual_command = (0.0, 0.0)
        self._auto_command = (0.0, 0.0)
        self._manual_received_s = float("-inf")
        self._auto_received_s = float("-inf")
        self._zero_frames_remaining = 0
        self._odom_requested = False
        self._active_ws = None
        self._closed = False

        self._incoming: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2048)
        self._outgoing: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._websocket_loop,
            name="tron-adapter-websocket",
            daemon=True,
        )
        self._ws_thread.start()

        self.create_timer(0.02, self._process_incoming)
        self.create_timer(1.0 / self.control_rate_hz, self._command_tick)
        self.create_timer(1.0, self._publish_diagnostics)
        self._publish_source()
        self._publish_diagnostics()
        self.get_logger().info(f"connecting to TRON at {self.robot_url}")

    def _required_string_parameter(self, name: str) -> str:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise ValueError(f"required string parameter {name!r} is not set")
        value = str(parameter.value).strip()
        if not value:
            raise ValueError(f"required string parameter {name!r} is empty")
        return value

    def _on_manual_command(self, message: TwistStamped) -> None:
        command = (float(message.twist.linear.x), float(message.twist.angular.z))
        with self._lock:
            self._manual_command = command
            self._manual_received_s = time.monotonic()

    def _on_auto_command(self, message: TwistStamped) -> None:
        command = (float(message.twist.linear.x), float(message.twist.angular.z))
        with self._lock:
            self._auto_command = command
            self._auto_received_s = time.monotonic()

    def _set_manual_enabled(self, request, response):
        self._set_source("manual" if request.data else "disabled")
        response.success = True
        response.message = (
            "manual control enabled" if request.data else "control disabled"
        )
        return response

    def _set_auto_enabled(self, request, response):
        self._set_source("auto" if request.data else "disabled")
        response.success = True
        response.message = (
            "auto control enabled" if request.data else "control disabled"
        )
        return response

    def _set_source(self, source: str) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid control source {source!r}")
        now = time.monotonic()
        with self._lock:
            self._source = source
            self._manual_command = (0.0, 0.0)
            self._auto_command = (0.0, 0.0)
            self._manual_received_s = now
            self._auto_received_s = now
            self._zero_frames_remaining = max(
                self._zero_frames_remaining, self.final_zero_frames
            )
        self._publish_source()
        self._publish_diagnostics()

    def _request_stand(self, _request, response):
        return self._request_mode("stand", response)

    def _request_walk(self, _request, response):
        return self._request_mode("walk", response)

    def _request_sit(self, _request, response):
        return self._request_mode("sit", response)

    def _request_mode(self, mode: str, response):
        with self._lock:
            current_mode = str(self._robot_info.get("status", "UNKNOWN"))
        if mode == "walk" and current_mode not in {"STAND", "WALK"}:
            response.success = False
            response.message = (
                f"current mode is {current_mode}; request stand first"
            )
            return response
        guid = self._enqueue_vendor_request(MODE_REQUESTS[mode], {})
        response.success = guid is not None
        response.message = (
            f"{mode} request queued ({guid})"
            if guid
            else "robot is not connected"
        )
        return response

    def _software_stop(self, _request, response):
        self._disable_and_zero()
        response.success = True
        response.message = "control disabled and zero velocity requested"
        return response

    def _vendor_emergency_stop(self, _request, response):
        self._disable_and_zero()
        guid = self._enqueue_vendor_request("request_emgy_stop", {})
        response.success = guid is not None
        response.message = (
            "vendor emergency-stop request queued"
            if guid
            else "robot is not connected"
        )
        return response

    def _disable_and_zero(self) -> None:
        self._set_source("disabled")

    def _command_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            command = select_command(
                source=self._source,
                robot_mode=str(self._robot_info.get("status", "UNKNOWN")),
                manual_command=self._manual_command,
                manual_age_s=now - self._manual_received_s,
                auto_command=self._auto_command,
                auto_age_s=now - self._auto_received_s,
                watchdog_s=self.input_watchdog_s,
            )
            if command is not None:
                self._zero_frames_remaining = self.final_zero_frames
            elif self._zero_frames_remaining > 0:
                command = (0.0, 0.0)
                self._zero_frames_remaining -= 1
        if command is not None:
            self._enqueue_vendor_request(
                "request_twist", {"x": command[0], "y": 0.0, "z": command[1]}
            )

    def _enqueue_vendor_request(
        self, title: str, data: dict[str, Any]
    ) -> Optional[str]:
        with self._lock:
            connected = self._connected
            robot_id = self._robot_id
        if not connected or not robot_id:
            return None
        guid = uuid.uuid4().hex
        payload = {
            "accid": robot_id,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data,
        }
        self._put_queue(self._outgoing, payload)
        return guid

    @staticmethod
    def _put_queue(target: queue.Queue, value: Any) -> None:
        try:
            target.put_nowait(value)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(value)

    def _websocket_loop(self) -> None:
        peer = urlparse(self.robot_url).hostname
        if peer:
            for name in ("NO_PROXY", "no_proxy"):
                entries = [
                    item.strip()
                    for item in os.environ.get(name, "").split(",")
                    if item.strip()
                ]
                if peer not in entries:
                    entries.append(peer)
                os.environ[name] = ",".join(entries)

        backoff_s = 1.0
        while not self._stop_event.is_set():
            connection = None
            try:
                connection = websocket.create_connection(
                    self.robot_url, timeout=2.0, enable_multithread=True
                )
                connection.settimeout(0.05)
                with self._lock:
                    self._active_ws = connection
                    self._connected = True
                    self._odom_requested = False
                self._put_queue(
                    self._incoming,
                    {"_internal": "connection", "connected": True},
                )
                backoff_s = 1.0
                while not self._stop_event.is_set():
                    while True:
                        try:
                            payload = self._outgoing.get_nowait()
                        except queue.Empty:
                            break
                        connection.send(
                            json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        if payload.get("title") == "request_twist":
                            self._put_queue(
                                self._incoming,
                                {
                                    "_internal": "sent_twist",
                                    "data": payload.get("data", {}),
                                },
                            )
                    try:
                        raw = connection.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        message = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(message, dict):
                        self._put_queue(self._incoming, message)
            except Exception as exc:
                self._put_queue(
                    self._incoming,
                    {
                        "_internal": "connection",
                        "connected": False,
                        "error": str(exc),
                    },
                )
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                with self._lock:
                    self._active_ws = None
                    self._connected = False
                    self._robot_id = ""
                    self._robot_info = {}
                    self._odom_requested = False
                while True:
                    try:
                        self._outgoing.get_nowait()
                    except queue.Empty:
                        break
                self._put_queue(
                    self._incoming,
                    {"_internal": "connection", "connected": False},
                )
            if not self._stop_event.is_set():
                self._stop_event.wait(backoff_s)
                backoff_s = min(8.0, backoff_s * 2.0)

    def _process_incoming(self) -> None:
        for _ in range(200):
            try:
                message = self._incoming.get_nowait()
            except queue.Empty:
                return
            internal = message.get("_internal")
            if internal == "connection":
                if not bool(message.get("connected")):
                    self._disable_and_zero()
                error = message.get("error")
                if error:
                    self.get_logger().warning(f"TRON connection: {error}")
                self._publish_diagnostics()
                continue
            if internal == "sent_twist":
                self._publish_sent_twist(message.get("data") or {})
                continue
            self._handle_vendor_message(message)

    def _handle_vendor_message(self, message: dict[str, Any]) -> None:
        title = str(message.get("title", ""))
        raw_data = message.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        if title == "notify_robot_info":
            robot_id = str(message.get("accid") or data.get("accid") or "")
            request_odom = False
            with self._lock:
                self._robot_id = robot_id
                self._robot_info = dict(data)
                if robot_id and not self._odom_requested:
                    self._odom_requested = True
                    request_odom = True
            self._publish_diagnostics()
            if request_odom:
                self._enqueue_vendor_request(
                    "request_enable_odom", {"enable": True}
                )
            return
        if title == "notify_odom":
            self._publish_odom(data)
            return
        if title.startswith("response_") or title.startswith("notify_"):
            event = String()
            event.data = json.dumps(
                message, ensure_ascii=False, separators=(",", ":")
            )
            self.event_pub.publish(event)

    def _publish_diagnostics(self) -> None:
        if not rclpy.ok():
            return
        with self._lock:
            connected = self._connected
            robot_id = self._robot_id
            source = self._source
            data = dict(self._robot_info)
        mode = str(data.get("status", "UNKNOWN"))
        ready = connected and bool(robot_id)
        try:
            battery = float(data.get("battery", math.nan))
        except (TypeError, ValueError):
            battery = math.nan

        status = DiagnosticStatus()
        status.name = "robot_adapter"
        status.hardware_id = robot_id or "tron"
        if not connected:
            status.level = DiagnosticStatus.WARN
            status.message = "disconnected"
        elif not robot_id:
            status.level = DiagnosticStatus.WARN
            status.message = "waiting for robot info"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "online"
        values = {
            "adapter": "tron",
            "connected": ready,
            "robot_id": robot_id,
            "mode": mode,
            "battery": battery if math.isfinite(battery) else None,
            "imu": str(data.get("imu", "UNKNOWN")),
            "motor": str(data.get("motor", "UNKNOWN")),
            "control_source": source,
        }
        status.values = [
            KeyValue(key=key, value="" if value is None else str(value))
            for key, value in values.items()
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status.append(status)
        self.diagnostics_pub.publish(message)

    def _publish_source(self) -> None:
        if not rclpy.ok():
            return
        with self._lock:
            source = self._source
        message = String()
        message.data = source
        self.source_pub.publish(message)

    def _publish_sent_twist(self, data: dict[str, Any]) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.twist.linear.x = float(data.get("x", 0.0))
        message.twist.linear.y = float(data.get("y", 0.0))
        message.twist.angular.z = float(data.get("z", 0.0))
        self.sent_pub.publish(message)

    def _publish_odom(self, data: dict[str, Any]) -> None:
        position = data.get("pose_position") or []
        orientation = data.get("pose_orientation") or []
        linear = data.get("twist_linear") or []
        angular = data.get("twist_angular") or []
        if len(position) < 2 or len(orientation) < 4:
            return
        try:
            message = Odometry()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.odom_frame
            message.child_frame_id = self.base_frame
            message.pose.pose.position.x = float(position[0])
            message.pose.pose.position.y = float(position[1])
            message.pose.pose.position.z = (
                float(position[2]) if len(position) > 2 else 0.0
            )
            message.pose.pose.orientation.x = float(orientation[0])
            message.pose.pose.orientation.y = float(orientation[1])
            message.pose.pose.orientation.z = float(orientation[2])
            message.pose.pose.orientation.w = float(orientation[3])
            if linear:
                message.twist.twist.linear.x = float(linear[0])
                message.twist.twist.linear.y = (
                    float(linear[1]) if len(linear) > 1 else 0.0
                )
                message.twist.twist.linear.z = (
                    float(linear[2]) if len(linear) > 2 else 0.0
                )
            if angular:
                message.twist.twist.angular.x = float(angular[0])
                message.twist.twist.angular.y = (
                    float(angular[1]) if len(angular) > 1 else 0.0
                )
                message.twist.twist.angular.z = float(
                    angular[2] if len(angular) > 2 else angular[0]
                )
        except (TypeError, ValueError, OverflowError) as exc:
            self.get_logger().warning(f"invalid TRON odom: {exc}")
            return
        self.odom_pub.publish(message)
        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = message.pose.pose.position.x
            transform.transform.translation.y = message.pose.pose.position.y
            transform.transform.translation.z = message.pose.pose.position.z
            transform.transform.rotation = message.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._disable_and_zero()
        for _ in range(self.final_zero_frames):
            self._enqueue_vendor_request(
                "request_twist", {"x": 0.0, "y": 0.0, "z": 0.0}
            )
        deadline = time.monotonic() + 0.4
        while not self._outgoing.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        self._stop_event.set()
        with self._lock:
            connection = self._active_ws
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        self._ws_thread.join(timeout=3.0)

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TronAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
