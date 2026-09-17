#!/usr/bin/env python3
"""Export complete beta_nav episodes to a self-contained official LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.collect_r2r import sha256, write_json  # noqa: E402
from data_pipeline.lerobot_stats import masked_writer_statistics  # noqa: E402
from data_pipeline.pointing import PointingConfig  # noqa: E402

LEROBOT_VERSION = "0.4.4"
RGB_KEY = "observation.images.rgb"
ARRAY_FEATURES = {
    **{k: "observation." + k for k in ("world_from_body", "world_from_camera_cv", "world_from_depth_camera_cv")},
    "fm_waypoints_ego": "action.fm.waypoints_ego", "fm_clean_se2_diff": "action.fm.clean_se2_diff",
    "fm_valid_mask": "action.fm.valid_mask", "future_frame_ids": "action.fm.future_frame_ids",
    "future_vertical_offset_m": "action.fm.vertical_offset_m",
    **{"rvq_" + k: "action.rvq." + k for k in ("codes", "valid_mask", "reconstruction", "ade_m")},
    **{p + "_" + k: "action." + p + "." + k for p in ("apos", "opos") for k in (
        "pixel", "world", "token_id", "valid_mask", "pixel_rc", "grid_rc", "grid_id",
        "target_world", "target_frame_id")},
}
STRING_FEATURES = ["source.episode_id", "source.scene_id", "action.rvq.tokens"] + [
    "action." + p + "." + k for p in ("apos", "opos") for k in ("token", "state", "reason")]


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_source(folder):
    metadata = json.loads((folder / "episode.json").read_text())
    source = Path(metadata.get("source_episode", folder))
    if not metadata.get("complete"):
        raise ValueError(f"incomplete episode: {folder}")
    frames = [json.loads(s) for s in (source / "frames.jsonl").read_text().splitlines()]
    rows = [json.loads(s) for s in (folder / "samples.jsonl").read_text().splitlines()]
    with np.load(folder / "targets.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in ARRAY_FEATURES}
    if not frames or len(frames) != len(rows) or len(frames) != metadata["frames"]:
        raise ValueError("frame/sample count mismatch")
    if any(len(v) != len(frames) for v in arrays.values()):
        raise ValueError("GT/frame count mismatch")
    for name, digest in metadata.get("source_hashes", {}).items():
        if sha256(source / name) != digest:
            raise ValueError(f"source provenance changed: {name}")
    for i, (frame, row) in enumerate(zip(frames, rows, strict=True)):
        if frame["frame_id"] != i or row["frame_id"] != i or row["array_row"] != i:
            raise ValueError("frames must be contiguous and aligned")
        for key in ("rgb", "depth"):
            if not (source / frame[key]).is_file():
                raise FileNotFoundError(source / frame[key])
    config = PointingConfig.from_metadata(metadata["pointing_config"])
    config.require_current()
    return dict(folder=folder, source=source, metadata=metadata, frames=frames, rows=rows, arrays=arrays)


def build_features(episode):
    features = {}
    for source, key in ARRAY_FEATURES.items():
        values = episode["arrays"][source]
        features[key] = {"dtype": str(values.dtype), "shape": values.shape[1:] or (1,), "names": None}
    camera = episode["metadata"]["camera"]
    h, w = camera["height"], camera["width"]
    features[RGB_KEY] = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features["observation.depth_m"] = {"dtype": "float32", "shape": (h, w), "names": None}
    features["observation.depth_valid"] = {"dtype": "bool", "shape": (h, w), "names": None}
    for key in STRING_FEATURES:
        features[key] = {"dtype": "string", "shape": (1,), "names": None}
    for key in ("action.explicit_stop", "action.stop_loss_valid"):
        features[key] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def frame_payload(episode, t):
    frame, row, meta = episode["frames"][t], episode["rows"][t], episode["metadata"]
    result = {key: np.asarray(episode["arrays"][source][t]).reshape(
        episode["arrays"][source].shape[1:] or (1,)) for source, key in ARRAY_FEATURES.items()}
    with Image.open(episode["source"] / frame["rgb"]) as image:
        if image.mode != "RGB" or image.size != (meta["camera"]["width"], meta["camera"]["height"]):
            raise ValueError("RGB dimensions/mode disagree with camera metadata")
        result[RGB_KEY] = np.asarray(image).copy()
    with np.load(episode["source"] / frame["depth"], allow_pickle=False) as depth:
        result["observation.depth_m"] = depth["depth_m"]
        result["observation.depth_valid"] = depth["valid"]
    result.update(task=meta["instruction"], **{"source.episode_id": str(meta["episode_id"]),
                  "source.scene_id": meta["scene_id"], "action.rvq.tokens": row["rvq_tokens"] or ""})
    for channel in ("apos", "opos"):
        result["action." + channel + ".token"] = row[channel + "_tokens"]
        for key in ("state", "reason"):
            result["action." + channel + "." + key] = row[channel][key]
    for key in ("explicit_stop", "stop_loss_valid"):
        result["action." + key] = np.asarray([row[key]], dtype=bool)
    return result


def export_dataset(source, output, repo_id="beta-nav/r2r", episode_ids=None):
    """Local-only export. Read all observations from capture, never from GT videos."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    manifest = json.loads((source / "manifest.json").read_text())
    folders = sorted((source / "episodes").glob("*/episode.json"))
    folders = [p.parent for p in folders if episode_ids is None or p.parent.name in episode_ids]
    if not folders or (episode_ids is not None and set(episode_ids) != {p.name for p in folders}):
        raise ValueError("no matching episodes or a requested ID is missing")
    # Preflight every source before creating output; only retain one episode's arrays at a time.
    features = build_features(read_source(folders[0]))
    for folder in folders:
        if build_features(read_source(folder)) != features:
            raise ValueError("episodes must share feature shapes and dtypes")
    raw_manifest = manifest
    if manifest.get("source_dataset"):
        raw_manifest = json.loads((Path(manifest["source_dataset"]) / "manifest.json").read_text())
    bundle = Path(raw_manifest["rvq_bundle"])
    for name, digest in raw_manifest["rvq_sha256"].items():
        if Path(name).name != name or sha256(bundle / name) != digest:
            raise ValueError(f"RVQ asset hash mismatch: {name}")
    if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"exporter requires lerobot=={LEROBOT_VERSION}")
    import lerobot.datasets.lerobot_dataset as official

    dataset = official.LeRobotDataset.create(repo_id=repo_id, fps=1, features=features, root=output,
        robot_type="lightnav_habitat", use_videos=False, image_writer_threads=2, video_backend="pyav")
    records = []
    try:
        with masked_writer_statistics(official):
            for index, folder in enumerate(folders):
                episode = read_source(folder)
                observations = []
                for t in range(len(episode["frames"])):
                    payload = frame_payload(episode, t)
                    observations.append({key: array_digest(payload[key]) for key in
                                         (RGB_KEY, "observation.depth_m", "observation.depth_valid")})
                    dataset.add_frame(payload)
                dataset.save_episode()
                meta = episode["metadata"]
                record = dict(episode_index=index, episode_id=str(meta["episode_id"]), scene_id=meta["scene_id"],
                    frames=meta["frames"], instruction=meta["instruction"], camera=meta["camera"],
                    pointing_config=meta["pointing_config"],
                    opos_visibility_start_frame=PointingConfig.from_metadata(meta["pointing_config"]).opos_visibility_start_frame(meta["frames"]),
                    source_episode=str(folder), observation_hashes=observations,
                    array_hashes={key: array_digest(episode["arrays"][source_key]) for source_key, key in ARRAY_FEATURES.items()},
                    source_hashes={name: sha256(folder / name) for name in ("episode.json", "samples.jsonl", "targets.npz")})
                if "source_metadata" in meta:
                    record["source_metadata"] = meta["source_metadata"]
                records.append(record)
                print(json.dumps({"exported_episode": record["episode_id"], "frames": record["frames"]}), flush=True)
    finally:
        dataset.finalize()
        dataset.stop_image_writer()
    provenance = output / "meta/provenance"
    provenance.mkdir(parents=True)
    shutil.copyfile(source / "manifest.json", provenance / "source_manifest.json")
    for i, folder in enumerate(folders):
        dest = provenance / "episodes" / f"{i:06d}"
        dest.mkdir(parents=True)
        for name in ("episode.json", "samples.jsonl"):
            shutil.copyfile(folder / name, dest / name)
    code_files = ("export_lerobot.py", "lerobot_stats.py", "validate_lerobot.py", "pointing.py", "geometry.py",
                  "collect_r2r.py", "vlnce_source.py", "run_lerobot.sh", "setup_lerobot.sh", "requirements-lerobot.txt")
    for name in code_files:
        shutil.copyfile(REPO / "data_pipeline" / name, provenance / name)
    assets = output / "assets/action_tokenizer"
    assets.mkdir(parents=True)
    for name in raw_manifest["rvq_sha256"]:
        shutil.copyfile(bundle / name, assets / name)
    metadata = dict(format="beta_nav_lerobot_v1", repo_id=repo_id, lerobot_version=LEROBOT_VERSION,
        codebase_version="v3.0", time_basis="frame_index", timestamp_kind="synthetic", fps=1, physical_dt_s=None,
        rgb_storage="lossless PNG bytes embedded in Parquet", depth_storage="float32 camera-Z metres in Parquet",
        source_dataset=str(source), source_manifest_sha256=sha256(source / "manifest.json"),
        array_features=ARRAY_FEATURES, features=features, episodes=records, rvq_assets=raw_manifest["rvq_sha256"],
        stats_policy="exclude masked/padded/nonfinite values; FM and RVQ waypoints pool valid horizon rows; depth pools valid pixels",
        normalization="custom BetaModel adapter required; tokens, grid/frame IDs and masks are categorical/identity",
        self_contained=True, pushed_to_hub=False,
        code_sha256={name: sha256(provenance / name) for name in code_files})
    if "source_selection" in manifest:
        metadata["source_selection"] = manifest["source_selection"]
    write_json(output / "meta/beta_nav.json", metadata)
    (output / "README.md").write_text(
        "# BetaNav LeRobotDataset v3.0\n\n"
        "Load with lerobot.datasets.lerobot_dataset.LeRobotDataset. RGB PNG bytes, float32 metre depth, "
        "poses and all apos/opos/RVQ/FM GT are embedded in Parquet; source capture files are not needed.\n\n"
        "Logical fps=1 and timestamp=frame_index; physical_dt_s=null. Each frame carries the complete FM-30 "
        "waypoints and increments plus RVQ-10, pointing tokens/coordinates and independent masks. "
        "See meta/beta_nav.json for camera calibration, window policy, source IDs, hashes and units.\n\n"
        "This is an image-backed LeRobot dataset; no videos/ directory is required. "
        "GT review videos are separate from training RGB. The action.* schema requires a custom training adapter.\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Complete raw beta_nav run or annotation overlay with available source observations")
    parser.add_argument("--output-dir", required=True, type=Path, help="NEW self-contained LeRobotDataset directory")
    parser.add_argument("--repo-id", default="beta-nav/r2r", help="Local dataset identifier; nothing is uploaded")
    parser.add_argument("--episode-id", action="append", help="Select IDs; repeat for multiple episodes")
    args = parser.parse_args()
    export_dataset(args.dataset, args.output_dir, args.repo_id, args.episode_id)
    from data_pipeline.validate_lerobot import validate_lerobot

    report = validate_lerobot(args.output_dir, source=args.dataset)
    write_json(args.output_dir / "meta/validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
