"""ZeroMQ request/reply client for a remote Habitat environment server.

The server (see ``habitat_server/``) answers pickled ``{"command", "data"}``
requests with pickled reply dicts. Only ``reset``, ``step`` and ``close`` are
used, so no gym/gymnasium space objects ever cross the wire and the client
depends on ``pyzmq`` alone (pickle protocol 4).
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

try:
    import zmq
except ImportError:  # pragma: no cover - only hit without the ``habitat`` extra
    zmq = None

logger = logging.getLogger(__name__)

_PICKLE_PROTOCOL = 4


class RemoteEnvClient:
    """REQ-socket proxy for one remote environment.

    ``timeout_ms`` bounds a single ``send``/``recv``. Habitat scene loads on
    ``reset`` can take tens of seconds, hence the long default. On a receive
    timeout the client calls ``recv`` again up to ``recv_retries`` extra times
    WITHOUT resending: a REQ socket that has sent but not yet received is still
    waiting for that same reply, so this never re-executes a ``step``.

    ``connect`` never fails on its own; the first ``reset`` is the effective
    connectivity check.
    """

    def __init__(
        self,
        address: str = "tcp://localhost:5555",
        timeout_ms: int = 600000,
        recv_retries: int = 2,
    ) -> None:
        if zmq is None:
            raise ImportError(
                "RemoteEnvClient requires pyzmq. Install it with "
                "`pip install pyzmq` (or the `lightnav[habitat]` extra)."
            )
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.recv_retries = int(recv_retries)

        self.context: Any = zmq.Context()
        self.socket: Any = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)
        self._connected = True
        logger.info("RemoteEnvClient connected to %s", self.address)

    # -- serialization --------------------------------------------------------

    @staticmethod
    def _serialize(data: Any) -> bytes:
        return pickle.dumps(data, protocol=_PICKLE_PROTOCOL)

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        return pickle.loads(raw)

    # -- RPC ------------------------------------------------------------------

    def _send_command(self, command: str, data: Any = None) -> dict:
        if not self._connected or self.socket is None:
            raise ConnectionError("Not connected to server")
        try:
            self.socket.send(self._serialize({"command": command, "data": data}))
        except zmq.Again as exc:
            raise ConnectionError(
                f"Timeout sending '{command}' to server (timeout={self.timeout_ms}ms)"
            ) from exc
        except zmq.ZMQError as exc:
            raise ConnectionError(f"ZMQ send error: {exc}") from exc

        attempts = max(1, self.recv_retries + 1)
        for attempt in range(attempts):
            try:
                raw = self.socket.recv()
            except zmq.Again as exc:
                if attempt + 1 < attempts:
                    logger.warning(
                        "recv timeout for '%s' (attempt %d/%d, %dms each); "
                        "request still outstanding, waiting again",
                        command,
                        attempt + 1,
                        attempts,
                        self.timeout_ms,
                    )
                    continue
                raise ConnectionError(
                    f"Timeout waiting for response from server for '{command}' "
                    f"after {attempts}x{self.timeout_ms}ms"
                ) from exc
            except zmq.ZMQError as exc:
                raise ConnectionError(f"ZMQ communication error: {exc}") from exc
            response = self._deserialize(raw)
            if response.get("status") == "error":
                raise RuntimeError(f"Server error: {response.get('message')}")
            return response
        raise ConnectionError(f"No response from server for '{command}'")  # not reachable

    # -- environment interface ------------------------------------------------

    def reset(self) -> tuple[Any, dict]:
        response = self._send_command("reset", {"seed": None, "options": None})
        return response["obs"], response.get("info", {})

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        response = self._send_command("step", action)
        return (
            response["obs"],
            response["reward"],
            response["terminated"],
            response["truncated"],
            response.get("info", {}),
        )

    def close(self) -> None:
        if self._connected:
            try:
                self._send_command("close")
            except Exception:
                pass
        self._connected = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
