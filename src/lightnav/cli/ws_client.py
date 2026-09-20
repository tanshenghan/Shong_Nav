#!/usr/bin/env python3
"""Minimal WebSocket client for the inference server (no simulator needed).

Demonstrates the wire protocol that ``lightnav-serve`` speaks, so you can drive
the server from any frame source: this example replaces the simulator with an
mp4 / frame directory. Full shapes are in docs/PROTOCOL.md.

Protocol (JSON over websocket, client -> server then server -> client):

    -> {"action": "login", "data": {"clientId": "<id>"}}
    <- {"action": "login", "data": {"rc": 0, "msg": "ok"}}

    -> {"action": "reset", "data": {}}
    <- {"action": "reset", "data": {"rc": 0, "msg": "ok"}}

    -> {"action": "next",  "data": {"seq": N, "image": "<b64 JPEG>", "instruction": "..."}}
    <- {"action": "next",  "data": {"rc": 0, "seq": N,
                                     "actions": {"step": S,
                                                 "actions": [[fwd, lat, yaw], ... H waypoints]},
                                     "stop": false,
                                     "visible": null,
                                     "latency_ms": 12.3,
                                     "timings_ms": {...},
                                     "raw_text": "<tpos_12><traj_7>",
                                     "pointing": {...}}}      # only for pointing checkpoints

``actions["actions"]`` is the predicted (H, 3) chunk in robot-local frame
[forward_m, lateral_m(+=left), yaw_rad(+=ccw)]. ``stop`` is true when the model
predicts the stop action. ``visible`` is null for checkpoints that emit no
target-position tokens. A ``next`` with an empty instruction only buffers the
frame (``{"rc": 0, "seq": N, "msg": "image received"}``).

Usage:
    lightnav-ws-client \\
        --server ws://localhost:8050 \\
        --video clip.mp4 --fps 4 \\
        --instruction "follow the person in the red shirt"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from lightnav.serving.protocol import parse_actions_payload

# Per-step max deltas used to turn a waypoint [fwd, lat, yaw] into a normalized
# base_velocity command in [-1, 1]^3 (same constants as the EVT-Bench client).
WP_FWD_MAX = 0.375
WP_LAT_MAX = 0.25
WP_YAW_MAX = math.pi / 20.0


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _frames_from_video(path: str, target_fps: float | None) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Reading --video needs opencv: pip install opencv-python-headless"
        ) from e

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, round(src_fps / target_fps)) if (target_fps and src_fps > 0) else 1
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def _frames_from_dir(path: str) -> list[np.ndarray]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)
    return [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files]


def _extract_waypoints(data: dict) -> list[list[float]]:
    """Tolerant ``actions`` parser: dict ``{"step","actions"}``, legacy wrapped list, or flat."""
    return parse_actions_payload(data.get("actions"))


def main() -> int:
    # websockets>=12 ships a synchronous client; no extra dependency beyond what
    # the server already needs.
    from websockets.sync.client import connect

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--server",
        default=os.environ.get("TRACKVLA_WS_URL", "ws://localhost:8050"),
        help="Server URL (env: TRACKVLA_WS_URL).",
    )
    ap.add_argument("--instruction", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames")
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    frames = (
        _frames_from_video(args.video, args.fps) if args.video else _frames_from_dir(args.frames)
    )
    if not frames:
        raise SystemExit("no frames to send")
    print(f"[client] {len(frames)} frames -> {args.server}")

    with connect(args.server, max_size=64 * 1024 * 1024) as ws:
        client_id = f"example_{uuid.uuid4().hex[:8]}"
        ws.send(json.dumps({"action": "login", "data": {"clientId": client_id}}))
        assert json.loads(ws.recv())["data"]["rc"] == 0, "login failed"

        ws.send(json.dumps({"action": "reset", "data": {}}))
        assert json.loads(ws.recv())["data"]["rc"] == 0, "reset failed"

        last_action = [0.0, 0.0, 0.0]
        for seq, frame in enumerate(frames):
            ws.send(
                json.dumps(
                    {
                        "action": "next",
                        "data": {
                            "seq": seq,
                            "image": _encode_jpeg_b64(frame),
                            "instruction": args.instruction,
                        },
                    }
                )
            )
            data = json.loads(ws.recv()).get("data", {})
            if data.get("rc") != 0 or data.get("actions") is None:
                # Early frames before a prediction can be made just ack the image.
                print(f"[seq {seq:>3}] {data.get('msg', 'no action')}")
                continue

            traj = _extract_waypoints(data)
            wp0 = traj[0]
            vx = float(np.clip(wp0[0] / WP_FWD_MAX, -1.0, 1.0))
            vy = float(np.clip(wp0[1] / WP_LAT_MAX, -1.0, 1.0))
            vyaw = float(np.clip(wp0[2] / WP_YAW_MAX, -1.0, 1.0))
            last_action = [vx, vy, vyaw]
            print(
                f"[seq {seq:>3}] {data.get('latency_ms', 0):6.1f}ms  "
                f"stop={data.get('stop')} visible={data.get('visible')}  "
                f"wp0={wp0}  vel={[round(v, 3) for v in last_action]}  "
                f"raw={data.get('raw_text', '')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
