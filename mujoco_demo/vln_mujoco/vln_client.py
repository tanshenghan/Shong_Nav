"""ROS-independent client for the VLN WebSocket protocol used by robot_deploy."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import aiohttp

Pixel = tuple[float, float]


class Logger(Protocol):
    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...


@dataclass(frozen=True)
class InferenceResult:
    sequence: int
    stamp_ns: int
    waypoints: tuple[tuple[float, float, float], ...]
    stop: bool | None
    visible: bool | None
    apos_state: str | None
    opos_state: str | None
    apos_px: Pixel | None
    opos_px: Pixel | None
    latency_ms: float


@dataclass(frozen=True)
class ClientSnapshot:
    state: str
    connected: bool
    server_url: str
    instruction: str
    sequence: int
    error: str


def normalize_server_url(value: str) -> str:
    url = str(value).strip()
    if not url:
        return ""
    if len(url) > 2048 or any(char.isspace() for char in url):
        raise ValueError("invalid VLN server URL")
    if "://" not in url:
        url = f"ws://{url}"
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid VLN server URL port") from exc
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("VLN server URL must use ws:// or wss://")
    return url


class VlnClient:
    def __init__(self, logger: Logger, server_url: str = "") -> None:
        self._logger = logger
        self._lock = threading.Lock()
        self._server_url = normalize_server_url(server_url)
        self._instruction = ""
        self._session = 0
        self._state = "IDLE"
        self._connected = False
        self._sequence = 0
        self._error = ""
        self._pending_frame: tuple[int, bytes] | None = None
        self._results: queue.Queue[InferenceResult] = queue.Queue()
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._thread_main, name="vln-client", daemon=True)
        self._thread.start()

    def set_server_url(self, value: str) -> str:
        url = normalize_server_url(value)
        with self._lock:
            if url != self._server_url:
                self._server_url = url
                self._session += 1
                self._connected = False
                self._sequence = 0
                self._error = ""
                self._state = "CONNECTING" if self._instruction else "IDLE"
                self._pending_frame = None
        return url

    def start(self, instruction: str) -> None:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        with self._lock:
            if not self._server_url:
                raise ValueError("configure the VLN server URL first")
            self._session += 1
            self._instruction = instruction
            self._state = "CONNECTING"
            self._connected = False
            self._sequence = 0
            self._error = ""
            self._pending_frame = None
            self._clear_results_locked()

    def stop(self) -> None:
        with self._lock:
            self._session += 1
            self._instruction = ""
            self._state = "IDLE"
            self._connected = False
            self._sequence = 0
            self._error = ""
            self._pending_frame = None
            self._clear_results_locked()

    def offer_frame(self, stamp_ns: int, jpeg: bytes) -> bool:
        with self._lock:
            if not self._instruction or not self._connected or self._pending_frame is not None:
                return False
            self._pending_frame = int(stamp_ns), bytes(jpeg)
            return True

    def take_results(self) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def snapshot(self) -> ClientSnapshot:
        with self._lock:
            return ClientSnapshot(
                self._state,
                self._connected,
                self._server_url,
                self._instruction,
                self._sequence,
                self._error,
            )

    def close(self) -> None:
        self._stop.set()
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)
        self._thread.join(timeout=4.0)

    def _thread_main(self) -> None:
        async def run() -> None:
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.current_task()
            try:
                await self._worker()
            finally:
                self._loop = None
                self._task = None

        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(run())

    async def _worker(self) -> None:
        while not self._stop.is_set():
            instruction, session = self._active()
            if not instruction:
                await asyncio.sleep(0.05)
                continue
            try:
                await self._connection(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_active(session):
                    self._set_runtime(session, state="ERROR", connected=False, error=str(exc))
                    self._logger.warning(f"VLN: {exc}")
                    await asyncio.sleep(1.0)

    async def _connection(self, session: int) -> None:
        with self._lock:
            url = self._server_url
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.ws_connect(url, heartbeat=10.0, max_msg_size=0) as websocket:
                await websocket.send_json({"action": "login", "data": {"clientId": "vln_mujoco"}})
                self._response_data(await self._receive(websocket, 3.0), "login")
                self._set_runtime(session, state="RUNNING", connected=True, error="")
                self._logger.info(f"VLN connected: {url}")
                while self._is_active(session) and not self._stop.is_set():
                    frame = self._take_frame(session)
                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue
                    stamp_ns, jpeg = frame
                    instruction, _ = self._active()
                    sequence = self._next_sequence(session)
                    sent_at = time.monotonic()
                    await websocket.send_json(
                        {
                            "action": "next",
                            "data": {
                                "seq": sequence,
                                "image": base64.b64encode(jpeg).decode("ascii"),
                                "instruction": instruction,
                            },
                        }
                    )
                    data = self._response_data(await self._receive(websocket, 3.0), "next")
                    if data.get("seq") != sequence:
                        raise RuntimeError("VLN response sequence mismatch")
                    pointing = data.get("pointing")
                    if not isinstance(pointing, dict):
                        raise RuntimeError("next response has no object data.pointing")
                    result = InferenceResult(
                        sequence=sequence,
                        stamp_ns=stamp_ns,
                        waypoints=self._parse_waypoints(data.get("actions")),
                        stop=self._optional_bool(data, "stop"),
                        visible=self._optional_bool(data, "visible"),
                        apos_state=self._optional_string(pointing, "apos_state"),
                        opos_state=self._optional_string(pointing, "opos_state"),
                        apos_px=self._optional_pixel(pointing, "apos_px"),
                        opos_px=self._optional_pixel(pointing, "opos_px"),
                        latency_ms=(time.monotonic() - sent_at) * 1000.0,
                    )
                    if self._is_active(session):
                        self._results.put_nowait(result)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"action": "reset"})

    def _active(self) -> tuple[str, int]:
        with self._lock:
            return self._instruction, self._session

    def _is_active(self, session: int) -> bool:
        with self._lock:
            return bool(self._instruction) and self._session == session

    def _set_runtime(
        self,
        session: int,
        *,
        state: str,
        connected: bool,
        error: str,
    ) -> None:
        with self._lock:
            if self._session == session:
                self._state = state
                self._connected = connected
                self._error = error

    def _take_frame(self, session: int) -> tuple[int, bytes] | None:
        with self._lock:
            if self._session != session:
                return None
            frame = self._pending_frame
            self._pending_frame = None
            return frame

    def _next_sequence(self, session: int) -> int:
        with self._lock:
            if self._session != session:
                raise RuntimeError("VLN session changed")
            self._sequence += 1
            return self._sequence

    def _clear_results_locked(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    async def _receive(websocket: aiohttp.ClientWebSocketResponse, timeout: float) -> str | bytes:
        message = await asyncio.wait_for(websocket.receive(), timeout=timeout)
        if message.type in {aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY}:
            return message.data
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"WebSocket error: {websocket.exception()}")
        raise RuntimeError("VLN WebSocket closed")

    @staticmethod
    def _decode(raw: str | bytes) -> dict:
        try:
            value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid VLN response JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("VLN response must be an object")
        return value

    @classmethod
    def _response_data(cls, raw: str | bytes, action: str) -> dict:
        data = cls._decode(raw).get("data")
        if not isinstance(data, dict) or isinstance(data.get("rc"), bool) or not isinstance(data.get("rc"), int):
            raise RuntimeError(f"{action} response has no integer data.rc")
        if data["rc"] != 0:
            raise RuntimeError(f"{action} failed: {data}")
        return data

    @classmethod
    def _parse_waypoints(cls, value: object) -> tuple[tuple[float, float, float], ...]:
        points: list[tuple[float, float, float]] = []

        def collect(item: object) -> None:
            if item is None:
                return
            if isinstance(item, dict):
                collect(item.get("actions"))
                return
            if not isinstance(item, list):
                raise RuntimeError("actions must be an array")
            if item and all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in item):
                if len(item) % 3:
                    raise RuntimeError("flat actions length is not divisible by 3")
                for index in range(0, len(item), 3):
                    point = tuple(float(number) for number in item[index : index + 3])
                    if not all(math.isfinite(number) for number in point):
                        raise RuntimeError("actions contain non-finite values")
                    points.append(point)
                return
            for nested in item:
                collect(nested)

        collect(value)
        return tuple(points)

    @staticmethod
    def _optional_bool(data: dict, key: str) -> bool | None:
        value = data.get(key)
        if value is None or isinstance(value, bool):
            return value
        raise RuntimeError(f"data.{key} must be boolean or null")

    @staticmethod
    def _optional_string(data: dict, key: str) -> str | None:
        value = data.get(key)
        if value is None or isinstance(value, str):
            return value
        raise RuntimeError(f"data.pointing.{key} must be a string or null")

    @staticmethod
    def _optional_pixel(data: dict, key: str) -> Pixel | None:
        value = data.get(key)
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in value
            )
        ):
            raise RuntimeError(
                f"data.pointing.{key} must be a finite [x, y] array or null"
            )
        return float(value[0]), float(value[1])
