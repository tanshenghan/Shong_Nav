"""Render chosen preview-camera poses in Isaac Sim without a robot or physics rollout.

Isaac imports happen only after argument and input validation. Running --help or
importing this module on a CPU machine never starts Kit or imports Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

_LOAD_TIMEOUT_S = 300.0
_CAPTURE_TIMEOUT_S = 180.0
_CAMERA_PRIM = "/LightNavPreviewCamera"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--poses-npz", type=Path, required=True,
                        help="positions_xyz [N,3], yaw_rad [N]; positions are ground-reference poses")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrite")
    parser.add_argument("--camera-height-m", type=float, default=1.25,
                        help="Chosen preview height above input Z, not recovered official extrinsics")
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--subframes", type=int, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.camera_height_m) or args.camera_height_m <= 0:
        parser.error("--camera-height-m must be finite and positive")
    if not math.isfinite(args.hfov_deg) or not 0 < args.hfov_deg < 180:
        parser.error("--hfov-deg must be between 0 and 180")
    if min(args.width, args.height, args.subframes) <= 0:
        parser.error("--width, --height and --subframes must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    return args


def _load_poses(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as fields:
        missing = {"positions_xyz", "yaw_rad"} - set(fields.files)
        if missing:
            raise ValueError(f"pose NPZ is missing fields: {sorted(missing)}")
        positions = np.array(fields["positions_xyz"], copy=True)
        yaw = np.array(fields["yaw_rad"], copy=True)
        timestamps = np.array(fields["timestamp_s"], copy=True) if "timestamp_s" in fields else None
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
        raise ValueError("positions_xyz must be a nonempty [N,3] array")
    if yaw.shape != (len(positions),):
        raise ValueError("yaw_rad must have shape [N] matching positions_xyz")
    for name, values in (("positions_xyz", positions), ("yaw_rad", yaw)):
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite real numbers")
    if timestamps is not None:
        if (timestamps.shape != (len(positions),) or timestamps.dtype.kind not in "fiu"
                or not np.isfinite(timestamps).all() or (timestamps < 0).any()
                or (np.diff(timestamps) <= 0).any()):
            raise ValueError("timestamp_s must be finite, nonnegative and strictly increasing [N]")
    return positions.astype(np.float64), yaw.astype(np.float64), timestamps


def _validate_rgba(frame, width: int, height: int):
    """Fail on missing/corrupt/black buffers instead of manufacturing an observation."""
    import numpy as np

    rgba = np.asarray(frame)
    if rgba.shape != (height, width, 4):
        raise ValueError(f"expected RGBA {(height, width, 4)}, got {rgba.shape}")
    if rgba.dtype != np.uint8 or not np.isfinite(rgba).all():
        raise ValueError(f"RGBA must be finite uint8, got {rgba.dtype}")
    if not rgba[..., 3].any() or not rgba[..., :3].any():
        raise ValueError("empty/transparent or completely black render; inspect scene lighting and camera")
    return np.ascontiguousarray(rgba[..., :3])


def _wait_future(app, future, timeout_s: float, description: str):
    """Pump Kit's asynchronous task loop with a bounded wait."""
    deadline = time.monotonic() + timeout_s
    while not future.done():
        if not app.is_running():
            raise RuntimeError(f"Isaac stopped while {description}")
        if time.monotonic() >= deadline:
            future.cancel()
            raise TimeoutError(f"timed out after {timeout_s:g}s while {description}")
        app.update()
    return future.result()


def main(argv=None):
    args = _parse_args(argv)
    scene_path = args.scene_usd.expanduser().resolve(strict=True)
    pose_path = args.poses_npz.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if not scene_path.is_file() or not pose_path.is_file():
        raise ValueError("scene and pose inputs must be files")
    if output.exists():
        raise FileExistsError(output)
    positions, yaw, timestamps = _load_poses(pose_path)
    scene_hash, pose_hash = _sha256(scene_path), _sha256(pose_path)

    # Omniverse plugins must be imported only after SimulationApp has started.
    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": True, "hide_ui": True, "multi_gpu": False,
        "renderer": "PathTracing", "samples_per_pixel_per_frame": 8,
        "anti_aliasing": 0, "denoiser": True, "width": args.width, "height": args.height,
        "extra_args": ["--allow-root", "--/renderer/multiGpu/autoEnable=false"],
    })
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading, open_stage_async
        from isaacsim.sensors.camera import Camera
        from omni.kit.async_engine import run_coroutine
        from PIL import Image
        from pxr import UsdGeom

        loaded, error = _wait_future(
            app, run_coroutine(open_stage_async(str(scene_path))), _LOAD_TIMEOUT_S, "opening scene"
        )
        if not loaded:
            raise RuntimeError(f"USD scene failed to open: {error}")
        deadline = time.monotonic() + _LOAD_TIMEOUT_S
        while is_stage_loading():
            if time.monotonic() >= deadline:
                raise TimeoutError("scene assets were still loading after 300s")
            if not app.is_running():
                raise RuntimeError("Isaac stopped while loading scene assets")
            app.update()
        stage = omni.usd.get_context().get_stage()
        if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("preview requires a metre-scale, Z-up scene; calibrate other scene coordinates first")
        if stage.GetPrimAtPath(_CAMERA_PRIM).IsValid():
            raise ValueError(f"scene already contains reserved camera path {_CAMERA_PRIM}")
        # All authored camera/render properties remain in the anonymous session
        # layer. Never Save()/Export() the loaded scene or any referenced asset.
        stage.SetEditTarget(stage.GetSessionLayer())
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        camera = Camera(prim_path=_CAMERA_PRIM, resolution=(args.width, args.height))
        camera.initialize()
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(_CAMERA_PRIM))
        aperture = 20.955
        focal_length = aperture / (2 * math.tan(math.radians(args.hfov_deg) / 2))
        usd_camera.GetHorizontalApertureAttr().Set(aperture)
        usd_camera.GetVerticalApertureAttr().Set(aperture * args.height / args.width)
        usd_camera.GetFocalLengthAttr().Set(focal_length)
        usd_camera.GetClippingRangeAttr().Set((0.01, 1000.0))
        usd_camera.GetFStopAttr().Set(0.0)
        rep.orchestrator.set_capture_on_play(False)

        def set_pose(index):
            position = positions[index].copy()
            position[2] += args.camera_height_m
            angle = yaw[index] / 2
            quaternion = np.array([math.cos(angle), 0.0, 0.0, math.sin(angle)])
            camera.set_world_pose(position, quaternion, camera_axes="world")

        def capture():
            _wait_future(
                app,
                run_coroutine(rep.orchestrator.step_async(
                    rt_subframes=args.subframes, delta_time=0.0, pause_timeline=True,
                )),
                _CAPTURE_TIMEOUT_S,
                "capturing preview RGB",
            )

        set_pose(0)
        for _ in range(args.warmup_frames):
            capture()
        output.mkdir(parents=True, exist_ok=False)
        frames_dir = output / "frames"
        frames_dir.mkdir()
        frame_manifest = []
        for index in range(len(positions)):
            set_pose(index)
            capture()
            rgb = _validate_rgba(camera.get_rgba(), args.width, args.height)
            path = frames_dir / f"frame_{index:06d}.png"
            Image.fromarray(rgb).save(path, format="PNG")
            frame_manifest.append({
                "pose_index": index, "path": str(path.relative_to(output)),
                "sha256": _sha256(path),
                "timestamp_s": float(timestamps[index]) if timestamps is not None else None,
            })
        manifest = {
            "schema_version": 1, "format": "vlnverse_isaac_camera_preview_v1",
            "scene_path": str(scene_path), "scene_root_file_sha256": scene_hash,
            "referenced_asset_hashes_recorded": False,
            "input_pose_path": str(pose_path), "input_pose_sha256": pose_hash,
            "renderer_code_sha256": _sha256(Path(__file__)),
            "frame_count": len(frame_manifest), "frames": frame_manifest,
            "camera": {
                "kind": "chosen_preview_camera_not_recovered_official_extrinsics",
                "position_rule": "input positions_xyz + [0,0,camera_height_m]",
                "camera_height_m": args.camera_height_m, "hfov_deg": args.hfov_deg,
                "width": args.width, "height": args.height, "forward_offset_m": 0.0,
                "orientation_rule": "input yaw only; wxyz; camera_axes=world (+X forward, +Z up)",
                "horizontal_aperture_usd": aperture, "focal_length_usd": focal_length,
            },
            "rendering": {
                "api_target": "Isaac Sim 4.5", "renderer": "PathTracing", "spp": 8,
                "denoiser": True, "warmup_frames": args.warmup_frames,
                "rt_subframes": args.subframes, "timeline_delta_time_s": 0.0,
                "lighting": "scene_authored_lighting_preserved",
            },
            "physics_rollout": False, "robot_model_loaded": False,
            "scene_collision_validated": False, "official_camera_reproduction": False,
            "timestamps": "copied_from_input_timestamp_s_or_null; pose indices are not physical time",
            "limitations": [
                "camera teleport rendering only; no physical robot or tracking validation",
                "chosen height/HFOV; does not recover official camera calibration",
                "scene root hash excludes referenced assets; retain the original asset tree",
            ],
        }
        # Completion marker comes last; partial directories are never ready.
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output_dir": str(output), "frame_count": len(frame_manifest)}), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
