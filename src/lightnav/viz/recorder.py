"""Local episode recording: per-step JPEG frames plus a JSON action log.

Layout::

    <root>/run_<YYYYmmdd_HHMMSS>/<conn_label>/episode_<NNN>/
        manifest.json     camera / timebase parameters used when rendering
        image_<step>.jpg  the frame the model acted on
        actions.jsonl     one record per step while the episode is open
        actions.json      the same records as an array, written on end_episode()

Recording is best-effort diagnostics: write failures are logged at WARNING and never
propagate to the caller. ``lightnav-render`` turns these directories into videos.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import numpy as np

from lightnav.viz.video import encode_jpeg_bytes

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
IMAGE_PREFIX = "image_"
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_datetime(value) -> datetime:
    """Coerce a datetime / ISO string / epoch seconds to an aware UTC datetime."""
    if value is None:
        return _now()
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="milliseconds")
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json_atomic(path: Path, obj, *, indent: int | None = 2) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _frame_size(image) -> list[int] | None:
    """``[w, h]`` of a frame given as encoded bytes or an HWC array, or None."""
    try:
        if isinstance(image, np.ndarray):
            if image.ndim < 2:
                return None
            return [int(image.shape[1]), int(image.shape[0])]
        if isinstance(image, (bytes, bytearray, memoryview)):
            from PIL import Image

            with Image.open(io.BytesIO(bytes(image))) as im:
                return [int(im.size[0]), int(im.size[1])]
    except Exception:  # noqa: BLE001
        return None
    return None


def _waypoints_list(waypoints) -> list[list[float]] | None:
    if waypoints is None:
        return None
    try:
        arr = np.asarray(waypoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return [[float(x) for x in row] for row in arr.tolist()]


class _Episode:
    """State of one open episode (created lazily on its first step)."""

    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        self.dir = directory
        self.manifest = manifest
        self.records: list[dict[str, Any]] = []
        self.jsonl: IO[str] | None = None
        self.prev_received: datetime | None = None
        self.manifest_dirty = True

    def open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(self.dir / "actions.jsonl", "a", encoding="utf-8")

    def flush_manifest(self) -> None:
        if self.manifest_dirty:
            _write_json_atomic(self.dir / "manifest.json", self.manifest)
            self.manifest_dirty = False

    def close_jsonl(self) -> None:
        if self.jsonl is not None:
            try:
                self.jsonl.close()
            finally:
                self.jsonl = None


class ConnectionRecorder:
    """Records the episodes of one client connection under ``<run_dir>/<label>/``."""

    def __init__(self, recorder: "EpisodeRecorder", label: str) -> None:
        self._rec = recorder
        self.label = label
        self.dir = recorder.run_dir / label
        self._episode_idx = 0
        self._pending = False
        self._pending_created_at: datetime | None = None
        self._ep: _Episode | None = None
        self._closed = False

    # ---- state ---------------------------------------------------------------
    @property
    def episode_dir(self) -> Path | None:
        """Directory of the open episode, or None before its first step."""
        return self._ep.dir if self._ep is not None else None

    @property
    def episode_open(self) -> bool:
        return self._pending or self._ep is not None

    @property
    def steps_recorded(self) -> int:
        return len(self._ep.records) if self._ep is not None else 0

    # ---- lifecycle -----------------------------------------------------------
    def begin_episode(self) -> None:
        """Start a new episode; an episode already open is ended first."""
        if self._closed:
            return
        try:
            if self._ep is not None:
                self.end_episode()
            self._pending = True
            self._pending_created_at = _now()
        except Exception as e:  # noqa: BLE001
            _log.warning("recorder[%s]: begin_episode failed: %s", self.label, e)

    def record_step(self, *, step, seq, image, instruction, waypoints, stop, visible,
                    raw_text, latency_ms, pointing=None, received_at=None, **extra) -> None:
        """Append one step: its frame (bytes verbatim, or an RGB array as JPEG) and record.

        ``step_dt_ms`` is the completion-to-completion duration of the previous
        interval derived from ``received_at`` (0.0 on the first step) and
        ``step_fps`` its inverse in Hz. Begins an episode when none is open.
        """
        if self._closed:
            return
        try:
            self._record_step(step=step, seq=seq, image=image, instruction=instruction,
                              waypoints=waypoints, stop=stop, visible=visible,
                              raw_text=raw_text, latency_ms=latency_ms, pointing=pointing,
                              received_at=received_at, extra=extra)
        except Exception as e:  # noqa: BLE001
            _log.warning("recorder[%s]: record_step %s failed: %s", self.label, step, e)

    def end_episode(self) -> None:
        """Finish the open episode: write ``actions.json`` and drop the jsonl. No-op without steps."""
        self._pending = False
        self._pending_created_at = None
        ep, self._ep = self._ep, None
        if ep is None:
            return
        try:
            ep.close_jsonl()
            ep.flush_manifest()
            _write_json_atomic(ep.dir / "actions.json", ep.records)
            (ep.dir / "actions.jsonl").unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            _log.warning("recorder[%s]: end_episode failed: %s", self.label, e)

    def close(self) -> None:
        """End the open episode and release the connection."""
        if self._closed:
            return
        try:
            self.end_episode()
        finally:
            self._closed = True
            self._rec._forget(self)

    def __enter__(self) -> "ConnectionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- internals -----------------------------------------------------------
    def _start_episode(self, received: datetime) -> _Episode:
        idx = self._episode_idx
        self._episode_idx += 1
        created = self._pending_created_at or received
        self._pending = False
        self._pending_created_at = None
        manifest = self._rec._manifest_base()
        manifest.update({
            "created_at": created.isoformat(timespec="milliseconds"),
            "conn": self.label,
            "episode": idx,
            "frame_size": None,
            "instruction": "",
        })
        ep = _Episode(self.dir / f"episode_{idx:03d}", manifest)
        ep.open()
        self._ep = ep
        return ep

    def _record_step(self, *, step, seq, image, instruction, waypoints, stop, visible,
                     raw_text, latency_ms, pointing, received_at, extra) -> None:
        received = _to_datetime(received_at)
        ep = self._ep if self._ep is not None else self._start_episode(received)

        if ep.prev_received is None:
            step_dt_ms = 0.0
        else:
            step_dt_ms = max(0.0, (received - ep.prev_received).total_seconds() * 1000.0)
        ep.prev_received = received
        step_fps = (1000.0 / step_dt_ms) if step_dt_ms > 0 else None

        frame_size = _frame_size(image)
        if ep.manifest.get("frame_size") is None and frame_size is not None:
            ep.manifest["frame_size"] = frame_size
            ep.manifest_dirty = True
        instr = str(instruction or "")
        if not ep.manifest.get("instruction") and instr.strip():
            ep.manifest["instruction"] = instr
            ep.manifest_dirty = True
        ep.flush_manifest()

        if self._rec.save_images and image is not None:
            self._write_image(ep, int(step), image)

        record: dict[str, Any] = dict(extra)
        record.update({
            "step": int(step),
            "seq": None if seq is None else int(seq),
            "received_at": received.isoformat(timespec="milliseconds"),
            "step_dt_ms": float(step_dt_ms),
            "step_fps": step_fps,
            "instruction": instr,
            "waypoints": _waypoints_list(waypoints),
            "stop": bool(stop),
            "visible": None if visible is None else bool(visible),
            "raw_text": str(raw_text or ""),
            "latency_ms": float(latency_ms) if latency_ms is not None else None,
            "pointing": pointing if isinstance(pointing, dict) else None,
            "frame_size": frame_size,
        })
        ep.records.append(record)
        if ep.jsonl is not None:
            ep.jsonl.write(json.dumps(record, default=_json_default) + "\n")
            ep.jsonl.flush()

    def _write_image(self, ep: _Episode, step: int, image) -> None:
        path = ep.dir / f"{IMAGE_PREFIX}{step:06d}.jpg"
        try:
            if isinstance(image, (bytes, bytearray, memoryview)):
                data = bytes(image)
            else:
                data = encode_jpeg_bytes(np.asarray(image))
            path.write_bytes(data)
        except Exception as e:  # noqa: BLE001
            _log.warning("recorder[%s]: image for step %d not written: %s", self.label, step, e)


def _exclusive_dir(base: Path) -> Path:
    """Create ``base`` (or ``base_2``, ``base_3``, ...) so that no other process owns it.

    Two recorders started in the same second would otherwise share a
    ``run_<timestamp>`` directory and silently overwrite each other's episodes.
    """
    candidate, k = base, 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{k}")
            k += 1


class EpisodeRecorder:
    """Owns one run directory and hands out a :class:`ConnectionRecorder` per client.

    The run directory is created eagerly and exclusively in the constructor, which
    raises ``OSError`` when the root is not writable: a misconfigured ``--record_dir``
    is a start-up error, not something to warn about on every step.
    """

    def __init__(self, root: str | Path, *, task: str, model_path: str, hfov_deg: float,
                 cam_height: float, forward_offset: float | None = None, video_fps: int = 10,
                 timeline: str = "realtime", waypoint_dt_s: float = 0.1,
                 save_images: bool = True, run_name: str | None = None,
                 extra: dict[str, Any] | None = None) -> None:
        if timeline not in ("realtime", "per_step"):
            raise ValueError(f"timeline must be 'realtime' or 'per_step', got {timeline!r}")
        self.root = Path(root)
        name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.task = str(task)
        self.model_path = str(model_path)
        self.hfov_deg = float(hfov_deg)
        self.cam_height = float(cam_height)
        self.forward_offset = None if forward_offset is None else float(forward_offset)
        self.video_fps = int(video_fps)
        self.timeline = timeline
        self.waypoint_dt_s = float(waypoint_dt_s)
        self.save_images = bool(save_images)
        self.extra = dict(extra or {})
        self._conns: dict[str, ConnectionRecorder] = {}
        self._n_conns = 0
        self._closed = False
        self.run_dir: Path = _exclusive_dir(self.root / name)

    def begin_connection(self, label: str | None = None) -> ConnectionRecorder:
        """Open a connection recorder; ``label`` defaults to ``conn001``, ``conn002``, ..."""
        self._n_conns += 1
        safe = _SAFE_LABEL.sub("_", str(label)).strip("._-") if label else ""
        if not safe:
            safe = f"conn{self._n_conns:03d}"
        # Claim the connection directory now (exclusively), so two server processes
        # sharing one run directory can never hand out the same label.
        try:
            safe = _exclusive_dir(self.run_dir / safe).name
        except OSError as e:
            _log.warning("recorder: cannot create connection dir for %r: %s", safe, e)
        conn = ConnectionRecorder(self, safe)
        self._conns[safe] = conn
        return conn

    def close(self) -> None:
        """Close every open connection (ending their episodes)."""
        if self._closed:
            return
        self._closed = True
        for conn in list(self._conns.values()):
            try:
                conn.close()
            except Exception as e:  # noqa: BLE001
                _log.warning("recorder: closing %s failed: %s", conn.label, e)

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- internals -----------------------------------------------------------
    def _manifest_base(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "created_at": None,
            "conn": None,
            "episode": None,
            "task": self.task,
            "model_path": self.model_path,
            "video_fps": self.video_fps,
            "video_timeline": self.timeline,
            "waypoint_dt_s": self.waypoint_dt_s,
            "overlay_hfov_deg": self.hfov_deg,
            "overlay_cam_height": self.cam_height,
            "overlay_forward_offset": self.forward_offset,
            "frame_size": None,
            "instruction": "",
            "extra": dict(self.extra),
        }

    def _forget(self, conn: ConnectionRecorder) -> None:
        self._conns.pop(conn.label, None)


__all__ = ["ConnectionRecorder", "EpisodeRecorder", "IMAGE_PREFIX", "SCHEMA_VERSION"]
