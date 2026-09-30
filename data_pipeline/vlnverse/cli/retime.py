"""Convert an official pose episode to validated, synthetic smooth motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from data_pipeline.vlnverse.motion.retime import MotionConfig, retime_episode
from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid
from data_pipeline.vlnverse.source.official import load_official_episode
from data_pipeline.vlnverse.validation.motion import validate_motion


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-parquet", type=Path, required=True)
    parser.add_argument("--occupancy-npz", type=Path, required=True,
                        help="Calibrated canonical grid; never pass an uncalibrated native occupancy PNG")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory, never overwrite")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/retime.json")
    parser.add_argument("--episode-id")
    parser.add_argument("--metadata-jsonl", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = MotionConfig(**json.loads(args.config.read_text()))
    source = load_official_episode(args.episode_parquet, episode_id=args.episode_id, metadata_path=args.metadata_jsonl)
    grid = OccupancyGrid.from_npz(args.occupancy_npz)
    record = retime_episode(source, grid, config)
    quality = validate_motion(record, source, grid, config)
    package = Path(__file__).resolve().parents[1]
    source_code = {str(p.relative_to(package)): _sha256(p) for p in sorted(package.rglob("*.py"))
                   if "tests" not in p.relative_to(package).parts}
    manifest = dict(schema_version=2, format="vlnverse_smooth_motion_v2", episode_id=source.episode_id,
        instruction=source.instruction, source=source.provenance,
        map_path=str(args.occupancy_npz.resolve()), map_sha256=_sha256(args.occupancy_npz),
        config=asdict(config), config_sha256=_sha256(args.config), code_sha256=source_code,
        coordinates="scene XYZ metres, Z up; unwrapped yaw radians about +Z, positive counterclockwise",
        time_basis="synthetic_retimed_seconds", method="route_preserving_septic_stop_turn",
        sampling=dict(mode=config.sample_mode, poses_per_source_step=quality["poses_per_source_step"],
                      sample_hz=quality["sample_hz"], uniform_timestamps=quality["uniform_timestamps"],
                      shared_endpoints_stored_once=True),
        source_frames=len(source.positions_xyz), output_frames=len(record.timestamp_s),
        source_timestamp_semantics="preserved_as_unverified_source_metadata_not_used_for_retiming",
        source_progress_semantics="fractional source frame index; correspondence, not physical time",
        quality=quality, primitives=record.primitives,
        limitations=["2D static circular footprint only; scene/grid calibration must be verified",
                     "not a physics rollout; validate the actual robot in the simulator before training",
                     "new poses require freshly rendered observations; source images are not retimed RGB GT"])
    # Finish all validation before publishing any result. The manifest is the
    # completion marker; a failed write cannot present a partial dataset as ready.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_arrays = dict(positions_xyz=source.positions_xyz, yaw_rad=source.yaw_rad)
    if source.source_timestamps is not None:
        source_arrays["source_timestamps"] = source.source_timestamps
    np.savez_compressed(args.output_dir / "source_poses.npz", **source_arrays)
    np.savez_compressed(args.output_dir / "trajectory.npz", **record.arrays())
    manifest["source_poses_sha256"] = _sha256(args.output_dir / "source_poses.npz")
    manifest["trajectory_sha256"] = _sha256(args.output_dir / "trajectory.npz")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
