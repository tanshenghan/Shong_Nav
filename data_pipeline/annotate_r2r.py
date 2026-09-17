#!/usr/bin/env python3
"""Rebuild apos-10 and final-destination opos on CPU in a NEW dataset overlay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.collect_r2r import sha256, write_json  # noqa: E402
from data_pipeline.pointing import (  # noqa: E402
    PointingConfig,
    attach_pixel_grid,
    ground_point,
    ground_target_arrays,
    pixel_grid_arrays,
    pixel_grid_metadata,
    pointing_token,
    terminal_ground_point,
)


def provenance_code_paths(directory):
    """Snapshot current pipeline code only; config.py contains local secrets."""
    names = ("__init__.py", "annotate_r2r.py", "collect_r2r.py", "export_pointing_grid.py",
             "geometry.py", "pointing.py", "validate_pointing.py", "validate_r2r.py")
    return [directory / name for name in names if (directory / name).is_file()]


def read_episode(folder):
    episode = json.loads((folder / "episode.json").read_text())
    if not episode.get("complete"):
        raise ValueError(f"incomplete source episode: {folder.name}")
    frames = [json.loads(s) for s in (folder / "frames.jsonl").read_text().splitlines()]
    samples = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
    if not frames or len(frames) != len(samples) or len(frames) != episode["frames"]:
        raise ValueError("source frame/sample counts disagree")
    if [f["frame_id"] for f in frames] != list(range(len(frames))):
        raise ValueError("source frame IDs must be contiguous")
    return episode, frames, samples


def draw_preview(image_path, output, apos, opos):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    # Different radii preserve both labels when the two targets coincide.
    for point, color, radius in [(apos, (80, 255, 180), 7), (opos, (255, 80, 160), 3)]:
        if point["state"] == "point":
            x, y = point["pixel"]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=2)
    image.save(output)


def annotate_episode(source, destination, config):
    """Keep all motion GT, recompute both pointing channels from recorded geometry."""
    config.require_current()
    episode, frames, samples = read_episode(source)
    destination.mkdir(parents=True, exist_ok=False)
    camera = episode["camera"]
    poses = np.asarray([f["world_from_body"] for f in frames])
    K = np.asarray(camera["K"])
    binding = dict(bound=True, target_type="stop_location", model_kind="recorded_terminal_ground_point",
                   world_point=poses[-1, :3, 3].tolist(), target_frame_id=len(frames) - 1,
                   anchor_frame_id=None, label="final destination",
                   reason="recorded_terminal_destination", confirmation="recorded_expert_endpoint")
    apos, opos, rows = [], [], []
    with np.load(source / "targets.npz", allow_pickle=False) as original:
        targets = {k: original[k] for k in original.files if not k.startswith(("apos_", "opos_"))}
        point_dtypes = {k: original[k].dtype for k in original.files
                        if k in ("apos_pixel", "apos_world", "opos_pixel", "opos_world")}
    for t, frame in enumerate(frames):
        with np.load(source / frame["depth"], allow_pickle=False) as depth:
            d, valid = depth["depth_m"], depth["valid"]
        a = ground_point(poses[:, :3, 3], t, frame["world_from_camera_cv"], K, d, valid, config)
        o = terminal_ground_point(poses[:, :3, 3], frame["world_from_camera_cv"], K, d, valid, config,
                                  frame_index=t)
        a = attach_pixel_grid(a, d.shape[1], d.shape[0])
        o = attach_pixel_grid(o, d.shape[1], d.shape[0])
        apos.append(a)
        opos.append(o)
        # Whitelist the training interface; historical replay-action fields stay out.
        row = {k: samples[t][k] for k in ("frame_id", "history_start", "history_end_inclusive",
               "rvq_codes", "rvq_tokens", "rvq_loss_valid", "fm_valid_steps", "explicit_stop",
               "stop_loss_valid") if k in samples[t]}
        row.update(current_rgb=str((source / frame["rgb"]).resolve()),
                   current_depth=str((source / frame["depth"]).resolve()),
                   array_file="targets.npz", array_row=t, apos=a, opos=o,
                   apos_tokens=pointing_token("apos", a),
                   opos_tokens=pointing_token("opos", o))
        rows.append(row)
        if t in {0, len(frames) // 2, len(frames) - 1}:
            draw_preview(source / frame["rgb"], destination / f"preview_{t:06d}.png", a, o)
    for name, points in [("apos", apos), ("opos", opos)]:
        # Preserve source precision when only the other channel's policy changes.
        targets[name + "_pixel"] = np.asarray([p["pixel"] if p["state"] == "point"
                                               else [np.nan] * 2 for p in points],
                                               dtype=point_dtypes.get(name + "_pixel", np.float32))
        targets[name + "_world"] = np.asarray([p["world_point"] if p["state"] == "point"
                                               else [np.nan] * 3 for p in points],
                                               dtype=point_dtypes.get(name + "_world", np.float32))
        targets[name + "_token_id"] = np.asarray([p["token_id"] for p in points], dtype=np.int32)
        targets[name + "_valid_mask"] = np.asarray([p["loss_valid"] for p in points], dtype=bool)
        targets.update(pixel_grid_arrays(points, name))
        targets.update(ground_target_arrays(points, name))
    np.savez_compressed(destination / "targets.npz", **targets)
    with (destination / "samples.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(destination / "object_binding.json", binding)
    result = {"complete": True, "source_episode": str(source.resolve()),
              "episode_id": episode["episode_id"], "scene_id": episode["scene_id"],
              "instruction": episode["instruction"], "frames": len(frames), "camera": camera,
              "object_bound": False, "opos_bound": True, "opos_target_type": "stop_location",
              "opos_search": None, "pointing_config": config.metadata(),
              "pointing_grid": pixel_grid_metadata(d.shape[1], d.shape[0]),
              "apos_valid": sum(p["loss_valid"] for p in apos),
              "opos_valid": sum(p["loss_valid"] for p in opos),
              "opos_visible": sum(p["state"] == "point" for p in opos),
              "opos_visibility_start_frame": config.opos_visibility_start_frame(len(frames)),
              "min_semantic_depth_agreement": None,
              "source_hashes": {name: sha256(source / name) for name in
                                ["episode.json", "frames.jsonl", "targets.npz", "samples.jsonl"]}}
    write_json(destination / "episode.json", result)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Existing complete beta_nav collection")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW annotation overlay directory")
    parser.add_argument("--config", type=Path, default=REPO / "data_pipeline/configs/pointing.json")
    parser.add_argument("--episodes", type=int, default=10, help="At most this many episodes; -1 for all")
    parser.add_argument("--episode-id", help="Annotate exactly one source episode")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs/config only; no writes")
    args = parser.parse_args()
    if args.episodes == 0 or args.episodes < -1:
        parser.error("episodes must be positive or -1")
    return args


def main():
    args = arguments()
    source = args.dataset.resolve()
    if args.output_dir.exists():
        raise FileExistsError("annotation output must be a new directory")
    config = PointingConfig(**json.loads(args.config.read_text()))
    config.require_current()
    folders = sorted((source / "episodes").iterdir())
    folders = [p for p in folders if p.is_dir() and (p / "episode.json").is_file()]
    if args.episode_id:
        folders = [p for p in folders if p.name == args.episode_id]
    if args.episodes > 0:
        folders = folders[:args.episodes]
    if not folders:
        raise ValueError("no matching complete source episodes")
    records = [(p, read_episode(p)[0]) for p in folders]
    records.sort(key=lambda pair: (pair[1]["scene_id"], pair[0].name))
    code_paths = provenance_code_paths(REPO / "data_pipeline")
    report = {"format": "beta_nav_pointing_v1", "source_dataset": str(source),
              "source_manifest_sha256": sha256(source / "manifest.json"),
              "pointing_config": config.metadata(), "episodes": [p.name for p, _ in records],
              "pointing_grid": pixel_grid_metadata(records[0][1]["camera"]["width"],
                                                    records[0][1]["camera"]["height"]),
              "opos_requested": True, "opos_mode": config.opos_mode, "api_requested": False,
              "discrete_action_labels": False, "rgb_depth_storage": "referenced from source, unmodified",
              "code_sha256": {p.name: sha256(p) for p in code_paths}}
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = args.output_dir / "provenance"
    provenance.mkdir()
    for path in code_paths:
        shutil.copyfile(path, provenance / path.name)
    write_json(provenance / "pointing_config.json", config.metadata())
    write_json(args.output_dir / "manifest.json", report)
    results = []
    try:
        for folder, _ in records:
            result = annotate_episode(folder, args.output_dir / "episodes" / folder.name, config)
            results.append(result)
            print(json.dumps({k: result[k] for k in
                              ["episode_id", "frames", "apos_valid", "opos_visible"]}), flush=True)
        write_json(args.output_dir / "summary.json", {
            "complete": True, "episodes": len(results), "opos_requested": True,
            "frames": sum(r["frames"] for r in results),
            "bound_objects": 0, "bound_destinations": len(results),
            "apos_valid": sum(r["apos_valid"] for r in results),
            "opos_visible": sum(r["opos_visible"] for r in results)})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"error": str(exc), "completed_episodes": len(results)})
        raise


if __name__ == "__main__":
    main()
