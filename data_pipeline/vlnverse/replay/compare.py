"""Encode Blender-rendered original/smoothed poses on a shared review timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def comparison_schedule(timestamps, source_progress, source_count, fps, playback_mode="reference_time"):
    """Hold each rendered pose until its next timestamp; do not invent RGB views.

    Original poses are placed at their matching source anchors on the retimed
    clock. This is a synthetic comparison, not recovered official action timing.
    """
    timestamps, progress = np.asarray(timestamps), np.asarray(source_progress)
    if (timestamps.ndim != 1 or not len(timestamps) or progress.shape != timestamps.shape
            or not np.isfinite(timestamps).all() or not np.isfinite(progress).all()
            or timestamps[0] != 0 or (np.diff(timestamps) <= 0).any()
            or (np.diff(progress) < 0).any() or source_count < 1
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("invalid comparison clock")
    if progress.min() < 0 or progress.max() > source_count - 1:
        raise ValueError("source progress is outside the source episode")
    source_times = np.interp(np.arange(source_count), progress, timestamps)
    if playback_mode == "pose_frames":
        # Stored poses have variable physical dt. This mode deliberately uses a
        # video clock, showing every smooth pose exactly once at the requested FPS.
        smooth = np.arange(len(timestamps))
        nearest = np.rint(progress)
        anchor_progress = np.where(np.abs(progress - nearest) <= 1e-9, nearest, progress)
        original = np.floor(anchor_progress).astype(int)
        return smooth / fps, original, smooth, source_times
    if playback_mode != "reference_time":
        raise ValueError("unknown playback mode")
    times = np.minimum(np.arange(int(np.ceil(timestamps[-1] * fps)) + 1) / fps, timestamps[-1])
    original = np.maximum(np.searchsorted(source_times, times, side="right") - 1, 0)
    smooth = np.maximum(np.searchsorted(timestamps, times, side="right") - 1, 0)
    return times, original, smooth, source_times


def _frames(directory, count):
    from PIL import Image

    result = []
    for index in range(count):
        path = directory / f"{index:06d}.png"
        with Image.open(path) as image:
            result.append(image.convert("RGB").copy())
    if len({image.size for image in result}) != 1:
        raise ValueError(f"inconsistent render resolution in {directory}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retimed-dir", type=Path, required=True)
    parser.add_argument("--original-images", type=Path, required=True)
    parser.add_argument("--smooth-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20, help="Video container FPS, independent of reference pose count")
    parser.add_argument("--playback-mode", choices=["reference_time", "pose_frames"], default="reference_time",
                        help="reference_time holds views on synthetic timestamps; pose_frames shows each smooth pose once")
    parser.add_argument("--overlay-style", choices=["scene", "lightnav"], default="scene",
                        help="lightnav reuses lightnav.viz ribbon/HUD on clean horizontal-camera RGB")
    parser.add_argument("--future-steps", type=int, default=30,
                        help="Number of future smooth poses for the lightnav ribbon")
    parser.add_argument("--official-rgb-npy", type=Path,
                        help="Use published RGB from the same official episode on the left; cameras may differ")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.future_steps < 1:
        raise ValueError("future-steps must be positive")
    from fractions import Fraction

    import av
    from PIL import Image, ImageDraw, ImageFont

    from data_pipeline.vlnverse.replay.video_layout import (
        fit_rgb_frame,
        instruction_banner,
        prepend_banner,
    )

    metadata = json.loads((args.retimed_dir / "manifest.json").read_text())
    if metadata["sampling"]["mode"] != "per_source_step":
        raise ValueError("this preview requires per_source_step samples with retained source anchors")
    with np.load(args.retimed_dir / "source_poses.npz", allow_pickle=False) as data:
        original_positions, original_yaw = data["positions_xyz"].copy(), data["yaw_rad"].copy()
    with np.load(args.retimed_dir / "trajectory.npz", allow_pickle=False) as data:
        positions, yaw = data["positions_xyz"].copy(), data["yaw_rad"].copy()
        reference_times = data["timestamp_s"].copy()
        reference_velocities = data["velocity_xyz"].copy()
        reference_yaw_rates = data["yaw_rate_rad_s"].copy()
        times, original_index, smooth_index, source_times = comparison_schedule(
            reference_times, data["source_progress"], len(original_yaw), args.fps, args.playback_mode)
    original_images = _frames(args.original_images, len(original_yaw))
    smooth_images = _frames(args.smooth_images, len(yaw))
    width, image_height = original_images[0].size
    if smooth_images[0].size != (width, image_height):
        raise ValueError("original/smoothed cameras must have matching resolution")
    original_render = json.loads((args.original_images / "render_manifest.json").read_text())
    smooth_render = json.loads((args.smooth_images / "render_manifest.json").read_text())
    if original_render["camera"] != smooth_render["camera"] or original_render["renderer"] != smooth_render["renderer"]:
        raise ValueError("original and smoothed render settings must match")
    if original_render["scene_usd"] != smooth_render["scene_usd"]:
        raise ValueError("original and smoothed renders must use the same scene")
    for rendered, filename in [(original_render, "source_poses.npz"), (smooth_render, "trajectory.npz")]:
        if rendered["source_sha256"] != hashlib.sha256((args.retimed_dir / filename).read_bytes()).hexdigest():
            raise ValueError("rendered images do not match the supplied pose file")
    official_rgb_metadata = None
    if args.official_rgb_npy is not None:
        from data_pipeline.vlnverse.source.rgb import load_episode_rgb

        rgb, official_rgb_metadata = load_episode_rgb(
            args.official_rgb_npy, metadata["source"]["source_path"], len(original_yaw))
        original_images = [fit_rgb_frame(frame, (width, image_height)) for frame in rgb]
    viz = None
    viz_metadata = None
    local_ribbons = None
    if args.overlay_style == "lightnav":
        from data_pipeline.vlnverse.replay import viz_style as viz

        viz.validate_camera(original_render["camera"], [original_render, smooth_render])
        viz_metadata = viz.style_provenance(original_render["camera"], args.future_steps)
        local_ribbons = [viz.future_local_ribbon_points(positions, yaw, k, args.future_steps)
                         for k in range(len(yaw))]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    height = image_height + 104
    if width % 2 or height % 2:
        raise ValueError("H.264 output dimensions must be even")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 20), ImageFont.truetype(font_path, 13)
    instruction = metadata.get("instruction")
    single_banner, single_layout = instruction_banner(instruction, width, font_path)
    combined_banner, combined_layout = instruction_banner(instruction, width * 2, font_path)
    future_path = smooth_render.get("future_path", {})
    future_horizon = future_path.get("horizon_steps", 0) if future_path.get("enabled") else 0
    names = ["original_gt.mp4", f"smoothed_{metadata['config']['poses_per_step']}poses.mp4", "side_by_side.mp4"]
    if official_rgb_metadata is not None:
        names[0], names[2] = "official_rgb.mp4", "official_vs_smoothed.mp4"
    writers = []
    try:
        for name, w, banner in zip(names, [width, width, width * 2],
                                   [single_banner, single_banner, combined_banner]):
            container = av.open(str(args.output_dir / name), "w", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=args.fps)
            stream.width, stream.height, stream.pix_fmt = w, height + banner.height, "yuv420p"
            stream.options = {"crf": "19", "preset": "fast"}
            writers.append((container, stream))
        for frame_number, (timestamp, i, j) in enumerate(zip(times, original_index, smooth_index)):
            panels = []
            for side, (label, color, image, index, count, position, angle) in enumerate([
                ("OFFICIAL RGB" if official_rgb_metadata else "GT POSE RENDER", (96, 172, 255),
                 original_images[i], i, len(original_yaw), original_positions[i], original_yaw[i]),
                (f"SMOOTHED | {metadata['config']['poses_per_step']} POSES / STEP", (79, 213, 160),
                 smooth_images[j], j, len(yaw), positions[j], yaw[j]),
            ]):
                if viz is not None:
                    velocity = (viz.reference_body_velocity(reference_velocities[j], yaw[j], reference_yaw_rates[j])
                                if side == 1 else None)
                    image = Image.fromarray(viz.draw_review(
                        image, label=("SMOOTH PREVIEW" if official_rgb_metadata else "SMOOTHED GT") if side else label,
                        video_frame=frame_number, fps=args.fps, stop=int(j) == len(yaw)-1,
                        camera=original_render["camera"],
                        waypoints=local_ribbons[j] if side else None, velocity=velocity))
                panel = Image.new("RGB", (width, height), (18, 24, 34))
                draw = ImageDraw.Draw(panel)
                if viz is None:
                    draw.text((14, 8), label, font=font, fill=color)
                else:
                    draw.text((14, 8), label if side == 0 else f"FUTURE {len(local_ribbons[j])} / {args.future_steps} POSES",
                              font=font, fill=color)
                clock_label = "play" if args.playback_mode == "pose_frames" else "ref"
                draw.text((14, 34), f"{metadata['episode_id']}  |  {clock_label} = {timestamp:.2f} s",
                          font=small, fill=(220, 225, 231))
                panel.paste(image, (0, 56))
                draw.text((14, height-43), f"pose {index+1}/{count}   XY {position[0]:.2f}, {position[1]:.2f} m   yaw {np.rad2deg(angle):.1f} deg",
                          font=small, fill=(220, 225, 231))
                footer = (f"{args.fps} FPS | pose-order playback | reference dt not preserved"
                          if args.playback_mode == "pose_frames"
                          else "Same preview camera | synthetic shared time | no physics")
                if official_rgb_metadata is not None and args.playback_mode == "reference_time":
                    footer = "Blender preview | synthetic time | camera differs"
                if side == 1 and future_horizon:
                    future_count = min(future_horizon, len(yaw) - int(j) - 1)
                    footer = f"{args.fps} FPS | cyan: next {future_count} / {future_horizon} smooth poses"
                if viz is not None:
                    footer = (f"{args.fps} FPS | Blender preview | reference velocities"
                              if side else f"{args.fps} FPS preview | GT poses held at source anchors")
                if side == 0 and official_rgb_metadata is not None:
                    footer = "Official RGB held | camera / renderer differ from right"
                draw.text((14, height-22), footer, font=small, fill=(158, 169, 182))
                panels.append(panel)
            combined = Image.new("RGB", (2 * width, height))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (width, 0))
            outputs = [prepend_banner(panel, single_banner) for panel in panels]
            combined = prepend_banner(combined, combined_banner)
            for (container, stream), image in zip(writers, [*outputs, combined]):
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                frame.pts, frame.time_base = frame_number, Fraction(1, args.fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
            if frame_number in {0, len(times)//2, len(times)-1}:
                combined.save(args.output_dir / f"comparison_{frame_number:06d}.jpg", quality=92)
        for container, stream in writers:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        for container, _ in writers:
            container.close()
    report = dict(episode_id=metadata["episode_id"], original_poses=len(original_yaw), smooth_poses=len(yaw),
        instruction=instruction, instruction_source="retimed manifest linked official episode metadata",
        instruction_layout=dict(single=single_layout, side_by_side=combined_layout),
        smooth_future_path=future_path,
        overlay_style=args.overlay_style, shared_viz=viz_metadata,
        official_rgb=official_rgb_metadata,
        left_view_source="official published RGB" if official_rgb_metadata else "GT poses re-rendered in Blender",
        matched_left_right_camera=official_rgb_metadata is None,
        future_pose_indices_per_smooth_frame=(
            [list(range(k+1, min(k+args.future_steps+1, len(yaw)))) for k in range(len(yaw))]
            if viz is not None else None),
        reference_duration_s=float(reference_times[-1]), video_frames=len(times), video_fps=args.fps,
        video_duration_s=len(times)/args.fps, camera=original_render["camera"], renderer=original_render["renderer"],
        playback_mode=args.playback_mode,
        preserves_reference_timing=args.playback_mode == "reference_time",
        playback=("each smooth pose appears exactly once at video FPS; reference dt is not preserved"
                  if args.playback_mode == "pose_frames"
                  else "hold each freshly rendered pose until its next synthetic timestamp; no RGB interpolation"),
        original_timing=("hold the preceding corresponding source pose until the next source anchor"
                         if args.playback_mode == "pose_frames"
                         else "source anchors aligned to the smoothed reference clock; not recorded physical timing"),
        original_pose_index_per_video_frame=original_index.tolist(),
        smooth_pose_index_per_video_frame=smooth_index.tolist(),
        reference_timestamp_s_per_video_frame=reference_times[smooth_index].tolist(),
        source_anchor_timestamp_s=source_times.tolist(), physics_replayed=False,
        retimed_manifest=str((args.retimed_dir / "manifest.json").resolve()),
        original_render_manifest=str((args.original_images / "render_manifest.json").resolve()),
        smooth_render_manifest=str((args.smooth_images / "render_manifest.json").resolve()),
        videos={name: hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest() for name in names})
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episode_id", "original_poses", "smooth_poses", "video_frames", "video_duration_s")}, indent=2))


if __name__ == "__main__":
    main()
