#!/usr/bin/env python3
"""Render one beta_nav episode: instruction, future GT line, and saved apos/opos.

Standalone implementation; requires numpy, Pillow, imageio and imageio-ffmpeg.
Accept either an original episode or an annotation overlay. --annotations adds
labels to an original episode explicitly. No Habitat, model, or API is needed.
--episode-id reads self-contained BetaNav LeRobot v3 Parquet with pyarrow/ijson.
FPS controls playback only; the recorded navigation steps have no physical dt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (40, 200, 255)
HIDDEN_COLOR = (255, 184, 70)
APOS_COLOR = (90, 255, 145)
OPOS_COLOR = (255, 90, 185)
BACKGROUND = (17, 24, 39)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_episode(directory, annotations=None):
    directory = Path(directory).resolve()
    metadata = read_json(directory / "episode.json")
    source = Path(metadata.get("source_episode", directory))
    if not source.is_absolute():
        source = directory / source
    source = source.resolve()
    labels = Path(annotations).resolve() if annotations else directory
    source_metadata = read_json(source / "episode.json")
    metadata = read_json(labels / "episode.json")
    if not metadata.get("complete") or not source_metadata.get("complete"):
        raise ValueError("only complete episodes can be rendered")
    for key in ("episode_id", "frames", "instruction"):
        if metadata[key] != source_metadata[key]:
            raise ValueError(f"annotation/source mismatch: {key}")
    if "source_episode" in metadata:
        claimed = Path(metadata["source_episode"])
        if not claimed.is_absolute():
            claimed = labels / claimed
        if claimed.resolve() != source:
            raise ValueError("annotations refer to a different source episode")
    for name, expected in metadata.get("source_hashes", {}).items():
        if digest(source / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    if metadata["camera"] != source_metadata["camera"]:
        raise ValueError("annotation/source camera mismatch")
    frames = [json.loads(s) for s in (source / "frames.jsonl").read_text().splitlines()]
    samples = [json.loads(s) for s in (labels / "samples.jsonl").read_text().splitlines()]
    n = len(frames)
    if not n or n != len(samples) or n != metadata["frames"]:
        raise ValueError("frame/sample count mismatch")
    for i, (frame, sample) in enumerate(zip(frames, samples)):
        if frame["frame_id"] != i or sample["frame_id"] != i or sample.get("array_row", i) != i:
            raise ValueError("frames and annotations must have matching contiguous IDs")
    poses = np.asarray([f["world_from_body"] for f in frames], dtype=np.float64)
    cameras = np.asarray([f["world_from_camera_cv"] for f in frames], dtype=np.float64)
    K = np.asarray(metadata["camera"]["K"], dtype=np.float64)
    if poses.shape != (n, 4, 4) or cameras.shape != (n, 4, 4) or K.shape != (3, 3):
        raise ValueError("invalid pose or camera matrix shape")
    if not all(np.isfinite(a).all() for a in (poses, cameras, K)):
        raise ValueError("non-finite pose or camera matrix")
    return dict(source=source, labels=labels, metadata=metadata, frames=frames, samples=samples,
                positions=poses[:, :3, 3], cameras=cameras, K=K)


def load_lerobot_episode(directory, episode_id, *, scene=None, metadata=None):
    """Read one self-contained BetaNav v3 episode; never use deleted scratch paths.

    Stream the large provenance manifest, select compact episode-index columns,
    then read only this episode from its Parquet files. RGB/depth stay in Arrow
    until the renderer asks for a frame. No LeRobot, Torch or Habitat import.
    """
    import ijson
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(directory).resolve()
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v3.0" or info.get("video_path"):
        raise ValueError("requires image-backed BetaNav LeRobotDataset v3.0")
    if metadata is None:
        with (root / "meta/beta_nav.json").open("rb") as handle:
            metadata = next((row for row in ijson.items(handle, "episodes.item", use_float=True)
                             if str(row["episode_id"]) == str(episode_id)
                             and (scene is None or Path(row["scene_id"]).stem == scene)), None)
    if metadata is None:
        raise ValueError(f"source episode {episode_id} / scene {scene} not found")
    if str(metadata["episode_id"]) != str(episode_id) or (scene is not None and Path(metadata["scene_id"]).stem != scene):
        raise ValueError("cached metadata episode/scene mismatch")
    metadata = dict(metadata)
    episode_index = int(metadata["episode_index"])
    index_columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                     "dataset_from_index", "dataset_to_index"]
    index = None
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        rows = pq.read_table(path, columns=index_columns,
                             filters=[("episode_index", "=", episode_index)]).to_pylist()
        if rows:
            if len(rows) != 1:
                raise ValueError("duplicate episode index metadata")
            index = rows[0]
            break
    if index is None or index["length"] != metadata["frames"]:
        raise ValueError("episode index/provenance length mismatch")
    if metadata["instruction"] not in index["tasks"]:
        raise ValueError("episode index/provenance instruction mismatch")
    features = info["features"]
    columns = ["frame_index", "index", "episode_index", "source.episode_id", "source.scene_id",
               "observation.images.rgb", "observation.depth_m", "observation.depth_valid",
               "observation.world_from_body", "observation.world_from_camera_cv",
               "action.fm.waypoints_ego", "action.fm.valid_mask", "action.fm.future_frame_ids",
               "action.fm.vertical_offset_m"]
    columns += [name for name in features if name.startswith(("action.apos.", "action.opos."))]
    first = root / info["data_path"].format(chunk_index=index["data/chunk_index"],
                                          file_index=index["data/file_index"])
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if first not in files:
        raise ValueError("episode's first Parquet file is missing")
    tables, used_files, count = [], [], 0
    for path in files[files.index(first):]:
        table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode_index)])
        if not len(table):
            raise ValueError("episode rows are missing or file range is not contiguous")
        tables.append(table)
        used_files.append(path)
        count += len(table)
        if count >= metadata["frames"]:
            break
    table = pa.concat_tables(tables).sort_by("frame_index")
    n = metadata["frames"]
    if len(table) != n or table["frame_index"].to_pylist() != list(range(n)):
        raise ValueError("LeRobot frame indices must be complete and contiguous")
    if table["index"].to_pylist() != list(range(index["dataset_from_index"], index["dataset_to_index"])):
        raise ValueError("episode rows do not match the global index range")
    for name, value in [("source.episode_id", str(episode_id)), ("source.scene_id", metadata["scene_id"])]:
        if any(item != value for item in table[name].to_pylist()):
            raise ValueError(f"Parquet/provenance mismatch: {name}")
    poses = np.asarray(table["observation.world_from_body"].to_pylist(), dtype=np.float64)
    cameras = np.asarray(table["observation.world_from_camera_cv"].to_pylist(), dtype=np.float64)
    K = np.asarray(metadata["camera"]["K"], dtype=np.float64)
    if poses.shape != (n, 4, 4) or cameras.shape != (n, 4, 4) or K.shape != (3, 3):
        raise ValueError("invalid saved transforms/calibration")
    if not all(np.isfinite(value).all() for value in (poses, cameras, K)):
        raise ValueError("non-finite saved transforms/calibration")
    positions = poses[:, :3, 3]
    # Verify the actual future poses against the saved clean FM supervision.
    # This also checks the forward/left signs, masks, tail and vertical offsets.
    for t in range(n):
        mask = np.asarray(table["action.fm.valid_mask"][t].as_py(), dtype=bool)
        ids = np.asarray(table["action.fm.future_frame_ids"][t].as_py(), dtype=int)
        expected = t + np.arange(1, len(mask) + 1)
        if not np.array_equal(mask, expected < n) or not np.array_equal(ids[mask], expected[mask]):
            raise ValueError("FM future mask/IDs disagree with recorded action steps")
        ego = np.asarray(table["action.fm.waypoints_ego"][t].as_py())[mask]
        vertical = np.asarray(table["action.fm.vertical_offset_m"][t].as_py())[mask]
        local = np.c_[-ego[:, 1], np.zeros(len(ego)), -ego[:, 0]]
        world = local @ poses[t, :3, :3].T + positions[t]
        world[:, 1] = positions[t, 1] + vertical
        if not np.allclose(world, positions[ids[mask]], atol=2e-5, rtol=0):
            raise ValueError("clean FM waypoints disagree with recorded body positions")

    samples = []
    for t in range(n):
        sample = {"frame_id": t}
        for channel in ("apos", "opos"):
            values = {name.split(".", 2)[2]: table[name][t].as_py()
                      for name in columns if name.startswith(f"action.{channel}.")}
            # Scalar features are stored as size-one lists in v3.
            def scalar(name):
                return values[name][0] if isinstance(values[name], list) else values[name]
            point = {"state": scalar("state"), "reason": scalar("reason"),
                     "loss_valid": bool(scalar("valid_mask")), "token_id": int(scalar("token_id")),
                     "pixel": None, "world_point": None, "pixel_rc": None, "grid": None,
                     "target_world_point": values["target_world"],
                     "target_frame_id": int(scalar("target_frame_id"))}
            if point["state"] == "point" and point["loss_valid"]:
                pixel, world = np.asarray(values["pixel"]), np.asarray(values["world"])
                xyz = np.linalg.solve(cameras[t], np.r_[world, 1])[:3]
                projected = K @ xyz
                if xyz[2] <= 0 or not np.allclose(projected[:2] / projected[2], pixel, atol=.01, rtol=0):
                    raise ValueError(f"saved {channel} pixel/world projection mismatch at {t}")
                point.update(pixel=pixel.tolist(), world_point=world.tolist(), pixel_rc=values["pixel_rc"],
                             grid={"row": values["grid_rc"][0], "col": values["grid_rc"][1],
                                   "cell_id": int(scalar("grid_id"))})
            sample[channel] = point
        samples.append(sample)

    hashes = metadata.pop("observation_hashes", [])
    for key in ("array_hashes", "source_hashes"):
        metadata.pop(key, None)

    def observation(t):
        encoded = table["observation.images.rgb"][t].as_py()["bytes"]
        if not encoded:
            raise ValueError("RGB must be embedded in Parquet, not an external image path")
        with Image.open(io.BytesIO(encoded)) as image:
            rgb = image.convert("RGB")
        if hashes and hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest() != hashes[t]["observation.images.rgb"]:
            raise ValueError(f"RGB provenance hash mismatch at frame {t}")
        # Flatten Arrow directly: avoid constructing hundreds of thousands of Python
        # floats/bools per frame when rendering the full collection.
        shape = (metadata["camera"]["height"], metadata["camera"]["width"])
        depth = np.asarray(table["observation.depth_m"][t].values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(shape)
        valid = np.asarray(table["observation.depth_valid"][t].values.flatten().to_numpy(zero_copy_only=False), dtype=bool).reshape(shape)
        if hashes:
            for name, array in [("observation.depth_m", depth), ("observation.depth_valid", valid)]:
                if hashlib.sha256(array.tobytes()).hexdigest() != hashes[t][name]:
                    raise ValueError(f"{name} provenance hash mismatch at frame {t}")
        return rgb, depth, valid

    provenance = {"format": "LeRobotDataset v3.0", "dataset_root": str(root),
                  "episode_index": episode_index, "source_episode_id": str(episode_id),
                  "scene_id": metadata["scene_id"], "episode_range": index,
                  "info_sha256": digest(root / "meta/info.json"),
                  "parquet_files": [{"path": str(p), "bytes": p.stat().st_size,
                                     "mtime_ns": p.stat().st_mtime_ns} for p in used_files],
                  "observation_hashes_verified": bool(hashes),
                  "clean_fm_pose_alignment_verified": True}
    return dict(source=root, labels=root, metadata=metadata,
                frames=[{"frame_id": t, "world_from_body": pose} for t, pose in enumerate(poses)],
                samples=samples, positions=positions, cameras=cameras, K=K,
                read_observation=observation, provenance=provenance)


def read_observation(episode, t):
    if "read_observation" in episode:
        return episode["read_observation"](t)
    frame = episode["frames"][t]
    with Image.open(episode["source"] / frame["rgb"]) as image:
        rgb = image.convert("RGB")
    with np.load(episode["source"] / frame["depth"], allow_pickle=False) as data:
        depth, valid = data["depth_m"], data["valid"]
    return rgb, depth, valid


def future_ground_path(positions, frame_id, horizon):
    """Current feet anchor + up to horizon real future feet positions; no padding."""
    return np.asarray(positions[frame_id:min(frame_id + horizon + 1, len(positions))])


def clip_segment(camera_a, camera_b, K, width, height, near=.03):
    """Clip in camera Z then image space; return pixels and camera-Z endpoints."""
    a, b = np.asarray(camera_a, dtype=float).copy(), np.asarray(camera_b, dtype=float).copy()
    if max(a[2], b[2]) < near:
        return None
    if a[2] < near:
        a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
    if b[2] < near:
        b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
    uv = np.stack([K @ a, K @ b])
    uv = uv[:, :2] / uv[:, 2:]
    delta = uv[1] - uv[0]
    lo, hi = 0., 1.
    for axis, maximum in ((0, width - 1e-4), (1, height - 1e-4)):
        if abs(delta[axis]) < 1e-12:
            if not 0 <= uv[0, axis] <= maximum:
                return None
            continue
        crossings = sorted([(0 - uv[0, axis]) / delta[axis],
                            (maximum - uv[0, axis]) / delta[axis]])
        lo, hi = max(lo, crossings[0]), min(hi, crossings[1])
        if lo > hi:
            return None
    fraction = np.array([lo, hi])
    pixels = uv[0] + fraction[:, None] * delta
    pixels = np.clip(pixels, [0, 0], [width - 1e-4, height - 1e-4])
    # Reciprocal depth is linear in projected segment coordinates.
    z = 1 / ((1 - fraction) / a[2] + fraction / b[2])
    return pixels, z


def project_path(path, world_from_camera, K, depth, valid, tolerance=.05, relative=.01):
    """Return sampled screen segments with depth visibility, preserving all breaks."""
    transform = np.linalg.inv(world_from_camera)
    xyz = np.asarray(path) @ transform[:3, :3].T + transform[:3, 3]
    height, width = depth.shape
    segments = []
    for a, b in zip(xyz[:-1], xyz[1:]):
        clipped = clip_segment(a, b, K, width, height)
        if clipped is None:
            continue
        uv, z = clipped
        length = float(np.linalg.norm(uv[1] - uv[0]))
        if length < .01:  # In-place rotation has no ground displacement.
            continue
        alpha = np.linspace(0, 1, max(2, int(np.ceil(length)) + 1))
        pixels = uv[0] + alpha[:, None] * (uv[1] - uv[0])
        depths = 1 / ((1 - alpha) / z[0] + alpha / z[1])
        col, row = np.floor(pixels).astype(int).T
        observed = depth[row, col]
        visible = (valid[row, col] & np.isfinite(observed) & (observed > 0)
                   & (depths <= observed + tolerance + relative * observed))
        segments.append((pixels, visible))
    return segments


def font_at(size, path=None):
    candidates = [path] if path else ["DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise ValueError("TrueType font unavailable; pass --font with a font file")


def wrap_text(text, font, width):
    """Wrap the complete instruction, including exceptionally long words."""
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if font.getlength(candidate) <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = ""
        for char in word:
            if line and font.getlength(line + char) > width:
                lines.append(line)
                line = ""
            line += char
    if line:
        lines.append(line)
    return lines or [""]


def mark_point(image, label, name, color, scale, font):
    if label is None:
        return "unavailable"
    if label.get("state") != "point":
        return label.get("state", "unknown")
    if not label.get("loss_valid"):
        return "masked"
    pixel = np.asarray(label.get("pixel"), dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError(f"invalid {name} pixel")
    x, y = pixel * scale
    if not (0 <= x < image.width and 0 <= y < image.height):
        # A float32 projection can round a valid near-edge coordinate to width
        # or height. Preserve the saved label, but do not invent an in-frame dot.
        epsilon = np.spacing(np.float32(max(image.size)))
        if (-epsilon <= x <= image.width + epsilon
                and -epsilon <= y <= image.height + epsilon):
            return "boundary_not_drawn"
        raise ValueError(f"{name} pixel is outside the image")
    draw = ImageDraw.Draw(image)
    # Nested rings keep both labels visible when apos reaches the opos endpoint.
    # Both centers remain at the exact saved pixel; only marker sizes differ.
    r = (8 if name == "apos" else 4) * scale
    draw.ellipse((x-r, y-r, x+r, y+r), fill=BACKGROUND, outline="white", width=scale)
    draw.ellipse((x-r+scale, y-r+scale, x+r-scale, y+r-scale), outline=color, width=2*scale)
    draw.ellipse((x-scale, y-scale, x+scale, y+scale), fill=color)
    text_width = font.getlength(name)
    text_x = x - r - 3 * scale - text_width if name == "apos" else x + r + 3 * scale
    text_y = y - 12 * scale if name == "apos" else y + 3 * scale
    tx = max(2, min(text_x, image.width - text_width - 3 * scale))
    ty = max(2, min(text_y, image.height - 14 * scale))
    draw.text((tx, ty), name, font=font, fill=color, stroke_width=scale, stroke_fill=BACKGROUND)
    return "point"


def top_view_points(path, world_from_body, center, radius, extent_m):
    """Native body +X right / -Z forward; draw forward upward in the side panel."""
    transform = np.linalg.inv(world_from_body)
    local = np.asarray(path) @ transform[:3, :3].T + transform[:3, 3]
    return np.asarray(center) + local[:, [0, 2]] * (radius / extent_m)


def mark_future_steps(image, points, world_from_camera, K, depth, valid, scale, font,
                      tolerance=.05, relative=.01):
    """Number visible real future positions; repeated positions share a range."""
    camera_from_world = np.linalg.inv(world_from_camera)
    xyz = np.asarray(points) @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    groups = []
    for step, point in enumerate(xyz, 1):
        if point[2] <= .03:
            continue
        pixel = K @ point
        u, v = pixel[:2] / pixel[2]
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            continue
        col, row = int(u), int(v)
        observed = float(depth[row, col])
        if not valid[row, col] or not math.isfinite(observed) or observed <= 0:
            continue
        if point[2] > observed + tolerance + relative * observed:
            continue
        if groups and np.linalg.norm(np.asarray(groups[-1][0]) - [u, v]) < 1.5:
            groups[-1][1].append(step)
        else:
            groups.append(([u, v], [step]))
    draw = ImageDraw.Draw(image)
    for (u, v), steps in groups:
        x, y = u*scale, v*scale
        draw.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale),
                     fill=GT_COLOR, outline=BACKGROUND, width=scale)
        label = str(steps[0]) if len(steps) == 1 else f"{steps[0]}-{steps[-1]}"
        tx = max(0, min(x+5*scale, image.width-font.getlength(label)-2))
        ty = max(0, min(y-12*scale, image.height-14*scale))
        draw.text((tx, ty), label, font=font, fill=GT_COLOR, stroke_width=scale, stroke_fill=BACKGROUND)


class FrameRenderer:
    def __init__(self, episode, horizon=10, scale=2, font=None, hide_occluded=False, number_steps=False,
                 rgb_only=False):
        self.episode, self.horizon, self.scale = episode, horizon, scale
        self.hide_occluded, self.number_steps = hide_occluded, number_steps
        self.rgb_only = rgb_only
        camera = episode["metadata"]["camera"]
        self.width, self.height = camera["width"] * scale, camera["height"] * scale
        self.text_font, self.small_font = font_at(11 * scale, font), font_at(9 * scale, font)
        self.instruction = wrap_text(episode["metadata"]["instruction"], self.text_font,
                                     self.width - 20 * scale)
        self.header = (28 + 15 * len(self.instruction)) * scale
        self.footer = 43 * scale
        self.sidebar = 140 * scale
        max_distance = max(np.linalg.norm(future_ground_path(episode["positions"], t, horizon)
                                         - episode["positions"][t], axis=1).max()
                           for t in range(len(episode["frames"])))
        self.extent_m = max(.5, math.ceil(max_distance * 2) / 2)
        # H.264 yuv420p requires even dimensions; pad instead of stretching RGB.
        self.size = (self.width + self.sidebar + (self.width + self.sidebar) % 2,
                     (self.header + self.height + self.footer + 1) // 2 * 2)
        if rgb_only:
            if self.width % 2 or self.height % 2:
                raise ValueError("RGB-only H.264 requires even source/display dimensions")
            self.size = (self.width, self.height)

    def render(self, t):
        ep, s = self.episode, self.scale
        frame = ep["frames"][t]
        original, depth, valid = read_observation(ep, t)
        if original.size != (self.width // s, self.height // s):
            raise ValueError("RGB dimensions disagree with camera intrinsics")
        rgb = original.resize((self.width, self.height), Image.Resampling.LANCZOS)
        if depth.shape != (self.height // s, self.width // s) or valid.shape != depth.shape:
            raise ValueError("depth dimensions disagree with RGB")
        path = future_ground_path(ep["positions"], t, self.horizon)
        cfg = ep["metadata"].get("pointing_config", {})
        segments = project_path(path, ep["cameras"][t], ep["K"], depth, valid,
                                cfg.get("depth_tolerance_m", .05),
                                cfg.get("depth_relative_tolerance", .01))
        draw = ImageDraw.Draw(rgb)
        drawn = {"visible_line_parts": 0, "hidden_line_parts": 0}
        distance = 0.
        for pixels, visible in segments:
            for i in range(len(pixels) - 1):
                length = np.linalg.norm(pixels[i+1] - pixels[i])
                clear = bool(visible[i] and visible[i+1])
                if clear or (not self.hide_occluded and int(distance // 5) % 2 == 0):
                    points = [tuple(pixels[i] * s), tuple(pixels[i+1] * s)]
                    draw.line(points, fill=BACKGROUND, width=4*s)
                    draw.line(points, fill=GT_COLOR if clear else HIDDEN_COLOR, width=2*s)
                    drawn["visible_line_parts" if clear else "hidden_line_parts"] += 1
                distance += length
        if self.number_steps:
            mark_future_steps(rgb, path[1:], ep["cameras"][t], ep["K"], depth, valid,
                              s, self.small_font, cfg.get("depth_tolerance_m", .05),
                              cfg.get("depth_relative_tolerance", .01))
        sample = ep["samples"][t]
        states = {name: mark_point(rgb, sample.get(name), name, color, s, self.small_font)
                  for name, color in (("apos", APOS_COLOR), ("opos", OPOS_COLOR))}
        stats = dict(frame_id=t, future_steps=len(path)-1, **states, **drawn,
                     apos_reason=sample.get("apos", {}).get("reason"),
                     opos_reason=sample.get("opos", {}).get("reason"))
        if self.rgb_only:
            return rgb, stats
        output = Image.new("RGB", self.size, BACKGROUND)
        output.paste(rgb, (0, self.header))
        draw = ImageDraw.Draw(output)
        left, top = self.width + 10*s, self.header + 10*s
        draw.text((left, top), "FUTURE GT / TOP VIEW", fill=GT_COLOR, font=self.small_font)
        center = (self.width + 70*s, self.header + 97*s)
        radius = 54*s
        draw.rectangle((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius),
                       outline=(60, 74, 95), width=s)
        draw.line((center[0]-radius, center[1], center[0]+radius, center[1]), fill=(42, 55, 74), width=s)
        draw.line((center[0], center[1]-radius, center[0], center[1]+radius), fill=(42, 55, 74), width=s)
        draw.text((left, self.header+29*s), "Forward = up", fill=(170, 186, 205), font=self.small_font)
        points = top_view_points(path, np.asarray(frame["world_from_body"]), center, radius, self.extent_m)
        if len(points) > 1:
            draw.line([tuple(p) for p in points], fill=GT_COLOR, width=2*s)
            x, y = points[-1]
            draw.ellipse((x-2*s, y-2*s, x+2*s, y+2*s), fill=GT_COLOR)
            if self.number_steps:
                for i, (x, y) in enumerate(points[1:], 1):
                    draw.ellipse((x-1.5*s, y-1.5*s, x+1.5*s, y+1.5*s), fill=GT_COLOR)
                    if i == len(points)-1:
                        draw.text((x+3*s, y-12*s), str(i), fill=GT_COLOR, font=self.small_font)
        x, y = center
        draw.polygon([(x, y-4*s), (x-3*s, y+3*s), (x+3*s, y+3*s)], fill="white")
        for i, (text, color) in enumerate([
                (f"Range +/-{self.extent_m:g} m", (170, 186, 205)),
                ("White: current pose", (235, 241, 250)),
                ("Blue: full future path", GT_COLOR),
                (f"Actual steps: {len(path)-1}/{self.horizon}", (235, 241, 250))]):
            draw.text((left, self.header+(163+17*i)*s), text, fill=color, font=self.small_font)
        draw.text((10*s, 7*s), "INSTRUCTION", fill=GT_COLOR, font=self.small_font)
        for i, line in enumerate(self.instruction):
            draw.text((10*s, (23+15*i)*s), line, fill="white", font=self.text_font)
        y = self.header + self.height
        legend = [("GT visible", GT_COLOR),
                  ("Occluded: hidden" if self.hide_occluded else "GT occluded / depth invalid", HIDDEN_COLOR),
                  ("apos", APOS_COLOR), ("opos", OPOS_COLOR)]
        x = 10*s
        for text, color in legend:
            draw.text((x, y+5*s), text, fill=color, font=self.small_font)
            x += self.small_font.getlength(text) + 14*s
        status = (f"Episode {ep['metadata']['episode_id']} | Frame {t}/{len(ep['frames'])-1}"
                  f" | Future {len(path)-1}/{self.horizon} | apos: {states['apos']} | opos: {states['opos']}")
        draw.text((10*s, y+23*s), status, fill=(215, 223, 237), font=self.small_font)
        return output, stats


def render_video(episode, output, *, fps=5., horizon=10, scale=2, font=None,
                 hide_occluded=False, number_steps=False, preview_frame=0, rgb_only=False):
    import imageio.v2 as imageio
    import imageio_ffmpeg

    if not math.isfinite(fps) or not 0 < fps <= 60 or not 1 <= horizon <= 30 or not 1 <= scale <= 4:
        raise ValueError("require 0 < fps <= 60, horizon 1..30, scale 1..4")
    if not 0 <= preview_frame < len(episode["frames"]):
        raise ValueError("preview frame must be inside the episode")
    output = Path(output).resolve()
    preview, report_path = output.with_suffix(".png"), output.with_suffix(".json")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must end in .mp4")
    for path in (output, preview, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    renderer = FrameRenderer(episode, horizon=horizon, scale=scale, font=font,
                             hide_occluded=hide_occluded, number_steps=number_steps, rgb_only=rgb_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as stream:
        temporary = Path(stream.name)
    records, first_image = [], None
    try:
        writer = imageio.get_writer(str(temporary), fps=fps, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=2, quality=None,
                                    ffmpeg_params=["-crf", "18", "-movflags", "+faststart", "-threads", "1"])
        try:
            for t in range(len(episode["frames"])):
                image, stats = renderer.render(t)
                writer.append_data(np.asarray(image))
                if t == preview_frame:
                    first_image = image
                if "provenance" in episode:
                    stop = min(t+horizon+1, len(episode["frames"]))
                    stats.update(future_frame_ids=list(range(t+1, stop)),
                                 future_world=episode["positions"][t+1:stop].tolist(),
                                 pointing=episode["samples"][t])
                records.append(stats)
        finally:
            # imageio's exceptional context-manager exit does not always close
            # ffmpeg; finish its writer before removing temporary output.
            writer.close()
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(temporary))
        if count != len(records):
            raise ValueError(f"encoded frame count mismatch: {count} != {len(records)}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    first_image.save(preview)
    report = dict(video=str(output), preview=str(preview), episode_id=episode["metadata"]["episode_id"],
                  source_episode=str(episode["source"]), annotations=str(episode["labels"]),
                  instruction=episode["metadata"]["instruction"], frames=count, fps=fps,
                  duration_seconds=seconds, size=list(renderer.size), horizon=horizon, scale=scale,
                  preview_frame=preview_frame, number_steps=number_steps,
                  layout="rgb_only" if rgb_only else "instruction_rgb_topview",
                  physical_dt_s=None, trajectory_source="recorded world_from_body ground positions",
                  top_view=None if rgb_only else "body XZ plane, forward up, fixed scale across the episode",
                  occlusion_style=("solid cyan visible; occluded/depth-invalid segments hidden"
                                   if hide_occluded else "solid cyan visible; dashed amber occluded or depth invalid"),
                  code_sha256=digest(__file__),
                  frame_stats=records)
    if "provenance" in episode:
        report.update(input_provenance=episode["provenance"], camera=episode["metadata"]["camera"],
                      pointing_config=episode["metadata"].get("pointing_config", {}),
                      opos_visibility_start_frame=episode["metadata"].get("opos_visibility_start_frame"))
    else:
        report.update(source_frames_sha256=digest(episode["source"] / "frames.jsonl"),
                      annotation_samples_sha256=digest(episode["labels"] / "samples.jsonl"))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {key: report[key] for key in ("video", "preview", "frames", "fps", "duration_seconds", "size")}


def render_grid_image(episode, output, *, frame_id=0, channel="apos", scale=2, font=None):
    """Draw a precise image-grid figure and a magnified view of the selected cell."""
    if not 0 <= frame_id < len(episode["frames"]) or channel not in ("apos", "opos"):
        raise ValueError("invalid frame ID or point channel")
    if not 1 <= scale <= 4:
        raise ValueError("scale must be 1..4")
    output = Path(output).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("grid image output must end in .png")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("grid image and report must be new files")
    frame = episode["frames"][frame_id]
    original, _, _ = read_observation(episode, frame_id)
    w, h = original.size
    if [w, h] != [episode["metadata"]["camera"][k] for k in ("width", "height")]:
        raise ValueError("RGB dimensions disagree with the camera")
    grid_rows, grid_cols = episode["metadata"].get("pointing_grid", {}).get("grid_shape_rc", [27, 48])
    cell_w, cell_h = w / grid_cols, h / grid_rows
    point = episode["samples"][frame_id].get(channel)
    visible = point is not None and point.get("state") == "point" and point.get("loss_valid")
    selected, pixel_rc = None, None
    if visible:
        u, v = point["pixel"]
        if not (np.isfinite([u, v]).all() and 0 <= u < w and 0 <= v < h):
            raise ValueError("selected pixel is outside the original image")
        row, col = int(v / h * grid_rows), int(u / w * grid_cols)
        selected = dict(row=row, col=col, cell_id=1 + row * grid_cols + col)
        pixel_rc = [int(v), int(u)]
        if point.get("grid") is not None and point["grid"] != selected:
            raise ValueError("stored grid cell does not match the pixel")
        if "token_id" in point and point["token_id"] != selected["cell_id"]:
            raise ValueError("point token does not match the grid cell")
    color = APOS_COLOR if channel == "apos" else OPOS_COLOR
    small, text, title = font_at(15, font), font_at(20, font), font_at(28, font)
    grid_font = font_at(max(10, 6 * scale), font)
    margin, image_x, gap, panel_width = 36, 64, 36, 360
    width = image_x + w * scale + gap + panel_width + margin
    instruction = wrap_text(episode["metadata"]["instruction"], text, width - margin * 2)
    image_y = 102 + 25 * len(instruction) + 30
    body_height = max(h * scale, 560)
    canvas = Image.new("RGB", (width, image_y + body_height + 76), BACKGROUND)
    rgb = original.resize((w * scale, h * scale), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", rgb.size)
    grid_draw = ImageDraw.Draw(layer)
    if visible:
        grid_draw.rectangle((col*cell_w*scale, 0, (col+1)*cell_w*scale, h*scale), fill=(*color, 24))
        grid_draw.rectangle((0, row*cell_h*scale, w*scale, (row+1)*cell_h*scale), fill=(*color, 24))
    for c in range(grid_cols + 1):
        x = min(rgb.width-1, c * cell_w * scale)
        grid_draw.line((x, 0, x, rgb.height-1), fill=(230, 238, 248, 95), width=1)
    for r in range(grid_rows + 1):
        y = min(rgb.height-1, r * cell_h * scale)
        grid_draw.line((0, y, rgb.width-1, y), fill=(230, 238, 248, 95), width=1)
    if visible:
        grid_draw.rectangle((col*cell_w*scale, row*cell_h*scale,
                             (col+1)*cell_w*scale, (row+1)*cell_h*scale), fill=(*color, 75),
                            outline=(*color, 255), width=max(2, scale))
    rgb = Image.alpha_composite(rgb, layer).convert("RGB")
    if visible:
        d = ImageDraw.Draw(rgb)
        x, y = u*scale, v*scale
        d.ellipse((x-3, y-3, x+3, y+3), fill="white", outline=BACKGROUND, width=1)
    canvas.paste(rgb, (image_x, image_y))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), f"EPISODE {episode['metadata']['episode_id']} / FRAME {frame_id}",
              font=title, fill="white")
    draw.text((margin, 61), f"{channel.upper()} PIXEL TO GRID CELL  |  {grid_cols} columns x {grid_rows} rows",
              font=text, fill=color)
    for i, line in enumerate(instruction):
        draw.text((margin, 102+i*25), line, font=text, fill=(220, 229, 240))
    for c in range(grid_cols):
        draw.text((image_x+(c+.5)*cell_w*scale, image_y-14), str(c), anchor="mm", font=grid_font,
                  fill=color if visible and c == col else (190, 203, 220))
    for r in range(grid_rows):
        draw.text((image_x-14, image_y+(r+.5)*cell_h*scale), str(r), anchor="mm", font=grid_font,
                  fill=color if visible and r == row else (190, 203, 220))
    panel_x = image_x + w * scale + gap
    draw.text((panel_x, image_y), channel.upper(), font=title, fill=color)
    if visible:
        lines = [(40, "Exact pixel (x, y)", small, (180, 195, 215)),
                 (63, f"({u:.4f}, {v:.4f})", text, "white"),
                 (99, f"Pixel [row, col] = {pixel_rc}", small, "white"),
                 (133, f"Grid [row, col] = [{row}, {col}]", text, color),
                 (169, f"Cell ID: {selected['cell_id']}", title, color),
                 (209, f"<{channel}_{selected['cell_id']}>", text, "white"),
                 (241, "Rows and columns start at 0", small, (180, 195, 215)),
                 (272, "DETAIL / EXACT CELL AND PIXEL", small, (180, 195, 215))]
        for offset, label, typeface, fill in lines:
            draw.text((panel_x, image_y+offset), label, font=typeface, fill=fill)
        c0, c1 = max(0, col-1), min(grid_cols, col+2)
        r0, r1 = max(0, row-1), min(grid_rows, row+2)
        bounds = (c0*cell_w, r0*cell_h, c1*cell_w, r1*cell_h)
        zoom_scale = 250 / max(bounds[2]-bounds[0], bounds[3]-bounds[1])
        zoom_size = (round((bounds[2]-bounds[0])*zoom_scale), round((bounds[3]-bounds[1])*zoom_scale))
        zoom = original.transform(zoom_size, Image.Transform.EXTENT, bounds, Image.Resampling.BICUBIC)
        zd = ImageDraw.Draw(zoom)
        sx, sy = zoom.width / (bounds[2]-bounds[0]), zoom.height / (bounds[3]-bounds[1])
        for c in range(c0, c1+1):
            x = min(zoom.width-1, (c-c0)*cell_w*sx)
            zd.line((x, 0, x, zoom.height-1), fill="white", width=1)
        for r in range(r0, r1+1):
            y = min(zoom.height-1, (r-r0)*cell_h*sy)
            zd.line((0, y, zoom.width-1, y), fill="white", width=1)
        zd.rectangle(((col-c0)*cell_w*sx, (row-r0)*cell_h*sy,
                      (col+1-c0)*cell_w*sx, (row+1-r0)*cell_h*sy), outline=color, width=3)
        x, y = (u-bounds[0])*sx, (v-bounds[1])*sy
        zd.ellipse((x-5, y-5, x+5, y+5), fill="white", outline=BACKGROUND, width=2)
        zd.line((x-11, y, x+11, y), fill=color, width=1)
        zd.line((x, y-11, x, y+11), fill=color, width=1)
        canvas.paste(zoom, (panel_x, image_y+298))
    else:
        state = point.get("state", "unknown") if point else "unavailable"
        draw.text((panel_x, image_y+45), f"State: {state}", font=text, fill="white")
        draw.text((panel_x, image_y+80), "No valid spatial cell", font=text, fill=(180, 195, 215))
    baseline = image_y + body_height
    draw.text((margin, baseline+17),
              f"Original RGB: {w} x {h}  |  White dot: exact pixel  |  Colored box: grid cell  |  Indices: zero-based",
              font=small, fill=(215, 225, 238))
    draw.text((margin, baseline+42), "Coordinates refer to the original image, before display scaling and margins.",
              font=small, fill=(170, 185, 205))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    report = dict(image=str(output), episode_id=episode["metadata"]["episode_id"], frame_id=frame_id,
                  channel=channel, original_size_wh=[w, h], grid_shape_rc=[grid_rows, grid_cols],
                  pixel=point["pixel"] if visible else None, pixel_rc=pixel_rc, grid=selected,
                  code_sha256=digest(__file__))
    if "provenance" in episode:
        report["input_provenance"] = episode["provenance"]
    else:
        report.update(source_rgb=str(episode["source"] / frame["rgb"]),
                      annotation_samples_sha256=digest(episode["labels"] / "samples.jsonl"))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Episode directory, or LeRobot dataset root with --episode-id")
    parser.add_argument("--episode-id", help="Read this original episode ID from an image-backed LeRobot root")
    parser.add_argument("--scene", help="Scan ID to disambiguate a LeRobot source episode")
    parser.add_argument("--annotations", type=Path, help="Optional annotation episode for an original input")
    parser.add_argument("--output", type=Path, required=True, help="NEW .mp4 video or .png grid figure; writes metadata too")
    parser.add_argument("--fps", type=float, default=5., help="Playback frames/second; not physical action timing")
    parser.add_argument("--horizon", type=int, default=10, help="Real future steps to project (1..30)")
    parser.add_argument("--scale", type=int, default=2, help="Integer RGB display scale (1..4)")
    parser.add_argument("--font", type=Path, help="TrueType font supporting the instruction's language")
    parser.add_argument("--frame-id", type=int, help="Frame for a PNG grid figure; default 0, video uses all frames")
    parser.add_argument("--channel", choices=("apos", "opos"), default="apos", help="Selected point in a PNG grid figure")
    parser.add_argument("--hide-occluded", action="store_true", help="Hide depth-occluded/unknown trajectory segments in RGB")
    parser.add_argument("--number-steps", action="store_true", help="Number visible future ground positions from 1")
    parser.add_argument("--preview-frame", type=int, default=0, help="Frame used for the MP4 poster PNG; video still includes every frame")
    parser.add_argument("--rgb-only", action="store_true", help="Only the RGB overlay; with --scale 1 preserves native 470x280 without panels")
    args = parser.parse_args()
    if args.episode_id is not None and args.annotations:
        parser.error("LeRobot labels are embedded; --episode-id cannot be combined with --annotations")
    if args.scene and args.episode_id is None:
        parser.error("--scene requires --episode-id")
    episode = (load_lerobot_episode(args.episode, args.episode_id, scene=args.scene)
               if args.episode_id is not None else load_episode(args.episode, args.annotations))
    if args.output.suffix.lower() == ".png":
        result = render_grid_image(episode, args.output, frame_id=args.frame_id or 0,
                                   channel=args.channel, scale=args.scale, font=args.font)
    else:
        if args.frame_id is not None:
            parser.error("--frame-id requires PNG output; videos render all frames")
        result = render_video(episode, args.output, fps=args.fps, horizon=args.horizon,
                              scale=args.scale, font=args.font, hide_occluded=args.hide_occluded,
                              number_steps=args.number_steps, preview_frame=args.preview_frame,
                              rgb_only=args.rgb_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
