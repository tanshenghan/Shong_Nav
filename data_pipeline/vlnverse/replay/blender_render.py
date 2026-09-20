"""Render reference poses in a real USD scene with the optional CPU Blender env.

This is an offline visual preview, not Isaac-equivalent shading or physics replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data_pipeline.vlnverse.replay.future_path import (
    FuturePathConfig,
    camera_forward,
    create_future_path,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--poses-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-height-m", type=float, default=1.25)
    parser.add_argument("--camera-pitch-down-deg", type=float, default=0.,
                        help="Positive pitch looks downward; shared across comparison views")
    parser.add_argument("--future-steps", type=int, default=0,
                        help="Preview-only 3D ground markers for the next N pose indices; 0 disables")
    parser.add_argument("--hfov-deg", type=float, default=90.)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-frames", type=int, help="Optional short diagnostic render; recorded in manifest")
    parser.add_argument("--frame-start", type=int, default=0, help="First global pose index for a render shard")
    parser.add_argument("--frame-stop", type=int, help="Exclusive global pose index; default is the full input")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not 1 < args.hfov_deg < 175 or not np.isfinite(args.camera_height_m)
            or args.camera_height_m <= 0 or min(args.width, args.height, args.samples, args.threads) <= 0
            or args.future_steps < 0 or not -89 < args.camera_pitch_down_deg < 89
            or (args.max_frames is not None and args.max_frames <= 0)):
        raise ValueError("invalid preview camera or render settings")
    with np.load(args.poses_npz, allow_pickle=False) as values:
        positions, yaws = values["positions_xyz"].copy(), values["yaw_rad"].copy()
    if (positions.ndim != 2 or positions.shape[1] != 3 or yaws.shape != (len(positions),)
            or not len(positions) or not np.isfinite(positions).all() or not np.isfinite(yaws).all()):
        raise ValueError("poses must be finite positions_xyz[N,3] and yaw_rad[N]")
    stop = len(positions) if args.frame_stop is None else args.frame_stop
    if not 0 <= args.frame_start < stop <= len(positions):
        raise ValueError("frame range must be nonempty and inside the supplied pose array")
    if args.max_frames is not None:
        stop = min(stop, args.frame_start + args.max_frames)
    # Keep heavyweight optional dependencies out of CPU pipeline imports/help.
    import bpy
    from mathutils import Vector
    from PIL import Image

    started = time.monotonic()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print(f"Loading {args.scene_usd}", flush=True)
    from data_pipeline.vlnverse.replay.usd_scene import import_static_scene

    scene_audit = import_static_scene(args.scene_usd)
    material_audit = scene_audit["materials"]
    print(f"Imported {scene_audit['mesh_objects']} mesh objects in {time.monotonic()-started:.1f}s", flush=True)
    # All rays see the imported scene before any line or marker geometry exists.
    overlay = create_future_path(positions, FuturePathConfig(args.future_steps)) if args.future_steps else None
    future_path = overlay.manifest if overlay else dict(
        enabled=False, horizon_steps=0, preview_only=True, training_observations=False, frames=[])
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.8, .8, .8, 1.)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .8
    scene.world = world
    camera = bpy.data.objects.new("PreviewCamera", bpy.data.cameras.new("PreviewCamera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = np.deg2rad(args.hfov_deg)
    camera.data.clip_start, camera.data.clip_end = .03, 100.
    fill = bpy.data.objects.new("PreviewCameraFill", bpy.data.lights.new("PreviewCameraFill", "AREA"))
    scene.collection.objects.link(fill)
    fill.data.energy, fill.data.size = 60., 2.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total = stop - args.frame_start
    frame_checks = []
    for index in range(args.frame_start, stop):
        position, yaw = positions[index], yaws[index]
        camera.location = Vector((float(position[0]), float(position[1]), float(position[2] + args.camera_height_m)))
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = Vector(camera_forward(yaw, args.camera_pitch_down_deg)).to_track_quat("-Z", "Y")
        fill.location = camera.location.copy()
        fill.rotation_mode = "QUATERNION"
        fill.rotation_quaternion = camera.rotation_quaternion.copy()
        if overlay is not None:
            overlay.update(index)
        scene.render.filepath = str((args.output_dir / f"{index:06d}.png").resolve())
        bpy.ops.render.render(write_still=True)
        with Image.open(scene.render.filepath) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != (args.height, args.width, 3) or rgb.std() < 1:
            raise RuntimeError(f"invalid or nearly empty render at pose {index}")
        frame_checks.append(dict(index=index, mean=float(rgb.mean()), std=float(rgb.std())))
        print(f"FRAME {index-args.frame_start+1}/{total} global={index} elapsed={time.monotonic()-started:.1f}s", flush=True)
    manifest = dict(schema_version=1, renderer=f"Blender {bpy.app.version_string} Cycles CPU",
        scene_usd=str(args.scene_usd.resolve()), source_npz=str(args.poses_npz.resolve()),
        source_sha256=hashlib.sha256(args.poses_npz.read_bytes()).hexdigest(),
        input_frames=len(positions), rendered_frames=total, diagnostic_frame_limit=args.max_frames,
        frame_range=[args.frame_start, stop],
        camera=dict(height_above_source_z_m=args.camera_height_m, hfov_deg=args.hfov_deg,
                    pitch_down_deg=args.camera_pitch_down_deg,
                    width=args.width, height=args.height, convention="world +X forward at yaw zero, +Z up",
                    extrinsics_origin="chosen preview camera, not recovered official sensor calibration"),
        shading="approximate USD diffuse materials; shared world and camera fill lighting",
        materials=material_audit, scene_import=scene_audit, samples=args.samples, threads=args.threads,
        frames=frame_checks, future_path=future_path, elapsed_s=time.monotonic()-started,
        physics_replayed=False, isaac_equivalent_render=False)
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
