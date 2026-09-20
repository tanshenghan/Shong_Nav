"""ZeroMQ REP server that exposes a single environment instance to a remote client.

Only ``pyzmq`` is required; the environment object is duck-typed (``reset``, ``step``,
``close`` and an optional ``initialize``), so this module can be used without habitat.
"""

from __future__ import annotations

import logging
import pathlib
import pickle
import signal
import threading
import time
from typing import Any, Optional

import zmq

logger = logging.getLogger(__name__)


class RemoteEnvServer:
    """Serve ``reset`` / ``step`` / ``close`` of one environment over a ZMQ REP socket.

    Wire format (pickle, protocol 4 by default)::

        request  = {"command": "reset" | "step" | "close", "data": ...}
        reset    -> {"status": "success", "obs": dict, "info": dict}
        step     -> {"status": "success", "obs": dict, "reward": float,
                     "terminated": bool, "truncated": bool, "info": dict}
        close    -> {"status": "success"}            (the server loop then exits)
        failure  -> {"status": "error", "message": str}
    """

    def __init__(
        self,
        env: Any,
        address: str = "tcp://*:5555",
        pickle_protocol: int = 4,
    ) -> None:
        self.env = env
        self.address = address
        self.pickle_protocol = pickle_protocol

        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self._running = False
        self._step_count = 0
        self._start_time: Optional[float] = None

        # Signal handlers can only be installed from the main thread. A server
        # driven from a worker thread (e.g. tests) is stopped via ``close``/``stop``.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def stop(self) -> None:
        """Ask the serving loop to exit after the current poll interval."""
        self._running = False

    def _setup_socket(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.address)
        logger.info(f"Server bound to {self.address}")

    def _serialize(self, data: Any) -> bytes:
        return pickle.dumps(data, protocol=self.pickle_protocol)

    def _deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    # -- command handlers -----------------------------------------------------

    def _handle_reset(self, data: Any) -> dict:
        try:
            seed = data.get("seed") if data else None
            options = data.get("options") if data else None
            result = self.env.reset(seed=seed, options=options)
            obs, info = result if isinstance(result, tuple) else (result, {})
            return {"status": "success", "obs": obs, "info": info}
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_step(self, action: Any) -> dict:
        try:
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, done, info = result
                terminated, truncated = done, False
            self._step_count += 1
            return {
                "status": "success",
                "obs": obs,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        except Exception as e:
            logger.error(f"Step error: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_close(self, data: Any) -> dict:
        try:
            self.env.close()
            self._running = False
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Close error: {e}")
            return {"status": "error", "message": str(e)}

    _HANDLERS = {
        "reset": _handle_reset,
        "step": _handle_step,
        "close": _handle_close,
    }

    # -- main loop ------------------------------------------------------------

    def start(self, ready_file: Optional[str] = None) -> None:
        """Bind, initialize the environment, then serve until ``close`` or a signal.

        ``ready_file`` (if given) is touched only after ``env.initialize()`` returned,
        i.e. once the simulator is up and the first ``reset`` will not block on it.
        """
        self._setup_socket()
        self._running = True
        self._start_time = time.time()
        self._step_count = 0

        if hasattr(self.env, "initialize"):
            self.env.initialize()

        if ready_file is not None:
            pathlib.Path(ready_file).touch()
            logger.info(f"Ready file written: {ready_file}")

        logger.info("Server started, waiting for connections...")

        try:
            while self._running:
                try:
                    if not self.socket.poll(timeout=1000):
                        continue

                    message = self._deserialize(self.socket.recv())
                    command = message.get("command", "")
                    handler = self._HANDLERS.get(command)
                    if handler is None:
                        response = {"status": "error", "message": f"Unknown command: {command}"}
                    else:
                        response = handler(self, message.get("data"))

                    self.socket.send(self._serialize(response))
                except zmq.ZMQError as e:
                    if self._running:
                        logger.error(f"ZMQ error: {e}")
                    break
        finally:
            self._log_stats()
            self.close()

    def _log_stats(self) -> None:
        if self._start_time and self._step_count > 0:
            elapsed = time.time() - self._start_time
            logger.info(
                f"Stats: {self._step_count} steps in {elapsed:.2f}s "
                f"({self._step_count / elapsed:.2f} steps/sec)"
            )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
