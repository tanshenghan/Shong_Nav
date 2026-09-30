"""Compare a previous stop/turn reference with a curved-plan camera preview.

The planned view uses every rendered pose once at 15 FPS. Default reference_time
requires 15 Hz poses; explicit pose_frames changes playback timing while retaining
the saved reference timestamps. The previous reference clock is normalized to the
planned reference duration. Panels need not share spatial progress at a frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FPS = 15
FUTURE_STEPS = 30


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _clock(values, *, planned=False, playback_mode="reference_time"):
    if playback_mode not in {"reference_time", "pose_frames"}:
        raise ValueError("playback_mode must be reference_time or pose_frames")
    times = np.asarray(values, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("reference timestamps must start at zero and strictly increase")
    if (planned and playback_mode == "reference_time"
            and not np.allclose(times, np.arange(len(times)) / FPS, atol=1e-8, rtol=0)):
        raise ValueError("planned reference must be uniformly sampled at exactly 15 Hz")
    return times


def normalized_reference_schedule(previous_times, planned_times, *, playback_mode="reference_time"):
    """Map the new clock to the old duration, then hold the preceding old view.

    Unlike arc-length matching, in-place turns retain a nonzero interval. Sparse
    video sampling can still omit old poses; callers must report omitted indices.
    """
    previous = _clock(previous_times)
    planned = _clock(planned_times, planned=True, playback_mode=playback_mode)
    mapped = planned / planned[-1] * previous[-1]
    previous_index = np.searchsorted(previous, mapped, side="right") - 1
    previous_index[0], previous_index[-1] = 0, len(previous) - 1
    return mapped, previous_index


def load_trajectory(path, *, planned=False, playback_mode="reference_time"):
    with np.load(path, allow_pickle=False) as data:
        required = {"positions_xyz", "yaw_rad", "timestamp_s", "velocity_xyz", "yaw_rate_rad_s"}
        if required - set(data.files):
            raise ValueError(f"trajectory is missing fields: {sorted(required - set(data.files))}")
        result = {key: np.asarray(data[key], dtype=float).copy() for key in required}
    times = _clock(result["timestamp_s"], planned=planned, playback_mode=playback_mode)
    for key, shape in (("positions_xyz", (len(times), 3)), ("velocity_xyz", (len(times), 3)),
                       ("yaw_rad", (len(times),)), ("yaw_rate_rad_s", (len(times),))):
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"trajectory {key} must be finite with shape {shape}")
    return result


def validate_comparison_identity(previous, planned):
    """Both paths must retain the episode identity and the full instruction."""
    for key in ("episode_id", "instruction"):
        if (not isinstance(previous.get(key), str) or not previous[key].strip()
                or planned.get(key) != previous[key]):
            raise ValueError(f"previous and planned {key} must match")
    for manifest in (previous, planned):
        for key in ("source", "quality", "config"):
            if not isinstance(manifest.get(key), dict):
                raise ValueError(f"trajectory manifest requires {key}")
        if manifest["quality"].get("valid") is False:
            raise ValueError("a failed trajectory quality report cannot be rendered as a completed plan")
    old_digest = previous["source"].get("source_sha256")
    new_digest = planned["source"].get("source_sha256")
    if old_digest is not None and new_digest is not None and old_digest != new_digest:
        raise ValueError("previous and planned source Parquet hashes differ")


def validate_render_manifest(manifest, trajectory_path, count):
    """Require complete, clean observations bound to this exact trajectory file."""
    if manifest.get("source_sha256") != _sha256(trajectory_path):
        raise ValueError("rendered images do not match the supplied trajectory hash")
    if manifest.get("input_frames") != count or manifest.get("rendered_frames") != count:
        raise ValueError("render manifest does not contain the complete trajectory")
    if manifest.get("future_path", {}).get("enabled", False):
        raise ValueError("comparison requires clean RGB without the 3D future-path overlay")
    camera = manifest.get("camera", {})
    if (not np.isfinite(float(camera.get("pitch_down_deg", 0.)))
            or abs(float(camera.get("pitch_down_deg", 0.))) > 1e-8):
        raise ValueError("shared ribbon projection requires a horizontal camera")
    if (not isinstance(camera.get("width"), int) or not isinstance(camera.get("height"), int)
            or min(camera["width"], camera["height"]) <= 0
            or not np.isfinite([camera.get("hfov_deg", np.nan),
                                camera.get("height_above_source_z_m", np.nan)]).all()
            or not 1 < camera["hfov_deg"] < 175 or camera["height_above_source_z_m"] <= 0):
        raise ValueError("render manifest has invalid camera geometry")
    if not manifest.get("scene_usd") or not manifest.get("renderer"):
        raise ValueError("render manifest must identify its scene and renderer")


def _read_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-retimed-dir", type=Path, required=True,
                        help="Previous source_poses.npz, trajectory.npz and manifest.json")
    parser.add_argument("--previous-images", type=Path, required=True, help="Clean previous smooth RGB and render manifest")
    parser.add_argument("--planned-dir", type=Path, required=True, help="New trajectory and manifest")
    parser.add_argument("--planned-images", type=Path, required=True, help="Fresh clean planned RGB and render manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; refuses overwrite")
    parser.add_argument("--playback-mode", choices=("reference_time", "pose_frames"), default="reference_time",
                        help="reference_time requires 15 Hz poses; pose_frames plays each saved pose once at 15 FPS")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    previous_path, planned_path = (args.previous_retimed_dir / "trajectory.npz",
                                   args.planned_dir / "trajectory.npz")
    previous_meta, planned_meta = (_read_json(args.previous_retimed_dir / "manifest.json"),
                                   _read_json(args.planned_dir / "manifest.json"))
    validate_comparison_identity(previous_meta, planned_meta)
    previous = load_trajectory(previous_path)
    planned = load_trajectory(planned_path, planned=True, playback_mode=args.playback_mode)
    source_path = args.previous_retimed_dir / "source_poses.npz"
    source_digest = _sha256(source_path)
    if (previous_meta.get("source_poses_sha256") is not None
            and previous_meta["source_poses_sha256"] != source_digest):
        raise ValueError("previous source_poses.npz hash disagrees with its manifest")
    render_paths = [args.previous_images / "render_manifest.json", args.planned_images / "render_manifest.json"]
    renders = [_read_json(path) for path in render_paths]
    for manifest, path, series in zip(renders, (previous_path, planned_path), (previous, planned)):
        validate_render_manifest(manifest, path, len(series["timestamp_s"]))
    for key in ("camera", "scene_usd", "renderer"):
        if renders[0][key] != renders[1][key]:
            raise ValueError(f"previous and planned render {key} must match")
    if not np.allclose(previous["positions_xyz"][[0, -1]], planned["positions_xyz"][[0, -1]], atol=1e-4, rtol=0):
        raise ValueError("previous and planned trajectories must have matching start and goal positions")
    mapped_times, previous_indices = normalized_reference_schedule(
        previous["timestamp_s"], planned["timestamp_s"], playback_mode=args.playback_mode)
    for directory, series in ((args.previous_images, previous), (args.planned_images, planned)):
        for index in range(len(series["timestamp_s"])):
            if not (directory / f"{index:06d}.png").is_file():
                raise FileNotFoundError(directory / f"{index:06d}.png")

    from fractions import Fraction

    import av
    from PIL import Image, ImageDraw, ImageFont

    from data_pipeline.vlnverse.replay import viz_style as viz
    from data_pipeline.vlnverse.replay.video_layout import instruction_banner, prepend_banner

    camera = renders[1]["camera"]
    width, image_height = camera["width"], camera["height"]
    panel_height = image_height + 124
    if width % 2 or panel_height % 2:
        raise ValueError("H.264 output dimensions must be even")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 20), ImageFont.truetype(font_path, 13)
    single_banner, single_layout = instruction_banner(planned_meta["instruction"], width, font_path)
    combined_banner, combined_layout = instruction_banner(planned_meta["instruction"], width * 2, font_path)
    style = viz.style_provenance(camera, FUTURE_STEPS)
    reference_time = args.playback_mode == "reference_time"
    style["hud_velocity"] = (
        "saved synthetic reference derivatives; previous velocities do not describe scaled video motion"
        if reference_time else "saved synthetic reference derivatives; neither panel's velocities describe playback motion")
    ribbons = [[viz.future_local_ribbon_points(series["positions_xyz"], series["yaw_rad"], i, FUTURE_STEPS)
                for i in range(len(series["timestamp_s"]))] for series in (previous, planned)]

    @lru_cache(maxsize=8)
    def frame_rgb(directory, index):
        with Image.open(directory / f"{index:06d}.png") as image:
            if image.size != (width, image_height):
                raise ValueError(f"rendered frame {index} has inconsistent camera dimensions")
            return np.asarray(image.convert("RGB")).copy()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    names = ["planned.mp4", "before_after.mp4"]
    writers, frame_audit = [], []
    try:
        for name, video_width, banner in zip(names, (width, width * 2), (single_banner, combined_banner)):
            container = av.open(str(args.output_dir / name.replace(".mp4", ".partial.mp4")), "w",
                                options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=FPS)
            stream.width, stream.height, stream.pix_fmt = video_width, panel_height + banner.height, "yuv420p"
            stream.options = {"crf": "19", "preset": "fast"}
            writers.append((container, stream))
        for frame_number, previous_index in enumerate(previous_indices):
            panels, future_ids, future_seconds = [], [], []
            for side, (series, directory, index, label) in enumerate((
                (previous, args.previous_images, int(previous_index), "PREVIOUS STOP/TURN"),
                (planned, args.planned_images, frame_number, "A* CURVED PREVIEW"),
            )):
                future = list(range(index + 1, min(index + FUTURE_STEPS + 1, len(series["timestamp_s"]))))
                future_ids.append(future)
                future_seconds.append(float(series["timestamp_s"][future[-1]] - series["timestamp_s"][index])
                                      if future else 0.)
                reference_velocity = viz.reference_body_velocity(
                    series["velocity_xyz"][index], series["yaw_rad"][index], series["yaw_rate_rad_s"][index])
                rgb = viz.draw_review(frame_rgb(directory, index), label=label,
                    video_frame=frame_number, fps=FPS, stop=index == len(series["timestamp_s"]) - 1,
                    camera=camera, waypoints=ribbons[side][index], velocity=reference_velocity)
                panel = Image.new("RGB", (width, panel_height), (18, 24, 34))
                draw = ImageDraw.Draw(panel)
                draw.text((14, 8), label, font=font, fill=(96, 172, 255) if side == 0 else (79, 213, 160))
                draw.text((14, 34), f"{planned_meta['episode_id']} | play {frame_number/FPS:.2f} s",
                          font=small, fill=(220, 225, 231))
                panel.paste(Image.fromarray(rgb), (0, 56))
                draw.text((14, panel_height-61),
                          f"pose {index+1}/{len(series['timestamp_s'])} | ref {series['timestamp_s'][index]:.2f} s | future {len(future)}/30",
                          font=small, fill=(220, 225, 231))
                draw.text((14, panel_height-41),
                          "normalized reference time | timing changed" if side == 0 else (
                              "15 FPS = 15 Hz reference | synthetic seconds" if reference_time
                              else "15 FPS | sampled poses | accelerated preview"),
                          font=small, fill=(158, 169, 182))
                draw.text((14, panel_height-21),
                          "HUD: old reference velocities, not playback velocities" if side == 0 else (
                              "A* + curved path | camera preview, no physics replay" if reference_time
                              else "HUD: reference velocities, not playback velocities"),
                          font=small, fill=(158, 169, 182))
                panels.append(panel)
            combined = Image.new("RGB", (2*width, panel_height))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (width, 0))
            outputs = [prepend_banner(panels[1], single_banner), prepend_banner(combined, combined_banner)]
            for (container, stream), image in zip(writers, outputs):
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                frame.pts, frame.time_base = frame_number, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    container.mux(packet)
            if frame_number in {0, len(previous_indices)//2, len(previous_indices)-1}:
                outputs[1].save(args.output_dir / f"comparison_{frame_number:06d}.jpg", quality=92)
            frame_audit.append(dict(video_frame_index=frame_number, playback_timestamp_s=frame_number/FPS,
                planned_pose_index=frame_number, planned_reference_timestamp_s=float(planned["timestamp_s"][frame_number]),
                previous_pose_index=int(previous_index), previous_target_reference_timestamp_s=float(mapped_times[frame_number]),
                previous_held_reference_timestamp_s=float(previous["timestamp_s"][previous_index]),
                previous_future_indices=future_ids[0], planned_future_indices=future_ids[1],
                previous_future_horizon_seconds=future_seconds[0],
                planned_future_horizon_seconds=future_seconds[1], future_horizon_seconds=future_seconds[1]))
        for container, stream in writers:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        for container, _ in writers:
            container.close()
    for name in names:
        (args.output_dir / name.replace(".mp4", ".partial.mp4")).rename(args.output_dir / name)
    old_counts = np.bincount(previous_indices, minlength=len(previous["timestamp_s"]))
    report = dict(schema_version=1, format="vlnverse_planned_comparison_v1",
        episode_id=planned_meta["episode_id"], instruction=planned_meta["instruction"],
        instruction_layout=dict(planned=single_layout, before_after=combined_layout),
        video_fps=FPS, video_frames=len(previous_indices), video_duration_s=len(previous_indices)/FPS,
        playback_mode=args.playback_mode,
        planned_reference_duration_s=float(planned["timestamp_s"][-1]),
        planned_clock=("one pose per frame; 15 Hz synthetic reference timestamps match video frame timestamps"
                       if reference_time else "one saved pose per frame at 15 FPS; saved reference timestamps are retained separately; accelerated preview"),
        terminal_frame_hold_s=1/FPS, planned_reference_timing_preserved=reference_time,
        previous_reference_duration_s=float(previous["timestamp_s"][-1]), previous_reference_timing_preserved=False,
        alignment="old reference time normalized onto planned duration; hold preceding old view; no spatial-progress synchronization",
        previous_clock_scale=float(planned["timestamp_s"][-1]/previous["timestamp_s"][-1]),
        previous_clock_scale_basis="planned reference seconds / previous reference seconds; separate from playback timing",
        previous_pose_display_counts=old_counts.tolist(), previous_omitted_pose_indices=np.flatnonzero(old_counts == 0).tolist(),
        future_horizon_poses=FUTURE_STEPS,
        future_horizon_basis="next saved poses; reference seconds recorded for each frame",
        planned_full_future_horizon_s=FUTURE_STEPS/FPS if reference_time else None,
        camera=camera, scene_usd=renders[1]["scene_usd"], renderer=renders[1]["renderer"], shared_viz=style,
        matched_left_right_camera=True, physics_replayed=False, training_observations=False,
        inputs={str(path.resolve()): _sha256(path) for path in (
            previous_path, planned_path, source_path, args.previous_retimed_dir / "manifest.json",
            args.planned_dir / "manifest.json", *render_paths)},
        frames=frame_audit, videos={name: _sha256(args.output_dir/name) for name in names})
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episode_id", "video_frames", "video_duration_s",
                                                 "previous_omitted_pose_indices")}, indent=2))


if __name__ == "__main__":
    main()
