#!/usr/bin/env python3
"""Headless trajectory-token inference: a checkpoint + a trajectory vocabulary + a
video (or a directory of frames) + an instruction -> the predicted (H, 3)
waypoint chunk. The websocket-free equivalent of the inference server.

Examples:
    # vLLM in-process backend (fast), mp4 input
    lightnav-predict \\
        --model_path /path/to/hf_ckpt \\
        --traj_vocab_path /path/to/traj_vocab --K 256 --horizon 10 \\
        --backend vllm_local \\
        --video clip.mp4 --fps 4 \\
        --instruction "follow the person in the red shirt"

    # plain HuggingFace backend, a directory of frames
    lightnav-predict \\
        --model_path /path/to/hf_ckpt \\
        --traj_vocab_path /path/to/traj_vocab --K 256 --horizon 10 \\
        --backend hf \\
        --frames ./frames_dir \\
        --instruction "go to the kitchen"

    # RVQ action-tokenizer checkpoint (bundle dir with manifest.json + codebooks)
    lightnav-predict \\
        --model_path /path/to/hf_ckpt \\
        --action_tokenizer_bundle /path/to/action_tokenizer --horizon 10 \\
        --video clip.mp4 --instruction "follow the person"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_frames_from_video(path: str, target_fps: float | None) -> list[np.ndarray]:
    """Decode an mp4/avi into a list of HWC uint8 RGB frames (optionally subsampled)."""
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
    stride = 1
    if target_fps and src_fps > 0:
        stride = max(1, round(src_fps / target_fps))

    frames: list[np.ndarray] = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    print(f"[predict] decoded {len(frames)} frames (src_fps={src_fps:.1f}, stride={stride})")
    return frames


def _load_frames_from_dir(path: str) -> list[np.ndarray]:
    """Load a sorted directory of image files into HWC uint8 RGB frames."""
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no image files in {path}")
    frames = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files]
    print(f"[predict] loaded {len(frames)} frames from {path}")
    return frames


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model_path", required=True, help="HF checkpoint dir (or a dir containing hf_ckpt/)."
    )
    vocab = ap.add_mutually_exclusive_group(required=False)
    vocab.add_argument(
        "--traj_vocab_path",
        help="Dir with centroids_whole_chunk_K{K}_h{H}.npy, or a direct .npy path (flat vocab).",
    )
    vocab.add_argument(
        "--action_tokenizer_bundle",
        help="RVQ bundle dir (manifest.json + codebooks); --K is ignored.",
    )
    ap.add_argument(
        "--task",
        choices=["tracking", "vln"],
        default="tracking",
        help="Prompt family / eval_config task entry: tracking (trackvla) or vln (vlnce).",
    )
    ap.add_argument(
        "--K",
        type=int,
        default=None,
        help="Trajectory vocab size (flat vocab only); must match the centroids file name "
        "centroids_whole_chunk_K{K}_h{H}.npy (default: from eval_config.json, else 256).",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Waypoints per chunk H (centroid second dim / RVQ manifest horizon); default: from "
        "the checkpoint's eval_config.json / bundle manifest.",
    )
    ap.add_argument("--instruction", required=True, help="Natural-language task / target description.")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to an mp4/avi clip.")
    src.add_argument("--frames", help="Directory of frame images (sorted by filename).")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Subsample --video to this fps (default: keep all decoded frames).",
    )

    ap.add_argument("--backend", default="vllm_local", choices=["hf", "vllm_local"])
    ap.add_argument(
        "--num_history_frames",
        type=int,
        default=None,
        help="History window override (frames). Default: the checkpoint's eval_config.json "
        "value -- normally leave unset.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory fraction (vllm_local only).",
    )
    ap.add_argument(
        "--pool_spatial",
        type=int,
        default=None,
        help="Override post-ViT spatial pooling (None = use ckpt eval_config.json).",
    )
    ap.add_argument(
        "--aspect_mode",
        choices=["stretch", "keep"],
        default="stretch",
        help="stretch frames to the checkpoint's video_size (default) or keep their aspect "
        "ratio at the same pixel budget.",
    )
    ap.add_argument(
        "--predict_every",
        type=int,
        default=0,
        help="If >0, also predict every N observed frames (streaming trace); "
        "default 0 = predict once on the full buffer.",
    )
    return ap


def main() -> int:
    from lightnav.tracking import build_tracking_agent

    args = _build_parser().parse_args()

    frames = (
        _load_frames_from_video(args.video, args.fps)
        if args.video
        else _load_frames_from_dir(args.frames)
    )

    agent = build_tracking_agent(
        model_path=args.model_path,
        traj_vocab_path=args.traj_vocab_path,
        K=args.K,
        horizon=args.horizon,
        backend=args.backend,
        num_history_frames=args.num_history_frames,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        pool_spatial=args.pool_spatial,
        aspect_mode=args.aspect_mode,
        action_tokenizer_bundle=args.action_tokenizer_bundle,
        task_key="trackvla" if args.task == "tracking" else "vlnce",
    )
    agent.reset(instruction=args.instruction)
    task_type = "tracking" if args.task == "tracking" else "vlnce_traj"

    for i, frame in enumerate(frames):
        agent.observe(frame)
        if args.predict_every and (i + 1) % args.predict_every == 0:
            wp, raw, ms = agent.predict_waypoints(args.instruction, task_type=task_type)
            print(f"[step {i + 1:>4}] {ms:6.1f}ms  raw={raw.strip()}  wp0={wp[0].tolist()}")

    wp, raw, ms = agent.predict_waypoints(args.instruction, task_type=task_type)
    print("\n================ FINAL PREDICTION ================")
    print(f"raw model output : {raw.strip()}")
    print(f"latency          : {ms:.1f} ms")
    print("waypoints (H,3)  : forward_m, lateral_m(+=left), yaw_rad(+=ccw)")
    np.set_printoptions(precision=4, suppress=True)
    print(np.asarray(wp))

    # First-step velocity command (reference; mirrors the EVT-Bench client mapping).
    from lightnav.velocity import first_waypoint_to_velocity_cmd

    vel = first_waypoint_to_velocity_cmd(
        wp[0], dt=1.0 / 4.0, lin_vel_range=(0.0, 2.5), ang_vel_range=(-30.0, 30.0)
    )
    print(f"first-step velocity cmd (dt=0.25s): {vel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
