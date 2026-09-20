"""Plan and time a curved route while retaining reviewed episode checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-parquet", type=Path, required=True)
    parser.add_argument("--occupancy-npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Explicit ordered checkpoints and robot limits")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; refuse overwrite")
    parser.add_argument("--metadata-jsonl", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    from data_pipeline.vlnverse.motion.planned import PlannedMotionConfig, retime_planned_curve
    from data_pipeline.vlnverse.planning import PlanningConfig, plan_route
    from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid
    from data_pipeline.vlnverse.source.official import load_official_episode

    settings = json.loads(args.config.read_text())
    source = load_official_episode(args.episode_parquet, episode_id=settings.get("episode_id"),
                                   metadata_path=args.metadata_jsonl)
    if settings.get("source_sha256", source.provenance["source_sha256"]) != source.provenance["source_sha256"]:
        raise ValueError("checkpoint configuration belongs to a different source episode")
    if np.ptp(source.positions_xyz[:, 2]) > 1e-5:
        raise ValueError("planned motion currently requires a planar source")
    waypoints = settings["waypoint_indices"]
    notes = settings.get("checkpoint_notes", [""] * len(waypoints))
    if len(notes) != len(waypoints):
        raise ValueError("checkpoint notes must match the ordered waypoint indices")
    planning = PlanningConfig(**settings.get("planning", {}))
    motion = PlannedMotionConfig(**settings.get("motion", {}))
    grid = OccupancyGrid.from_npz(args.occupancy_npz)
    route = plan_route(grid, source.positions_xyz[:, :2], source.yaw_rad[0], source.yaw_rad[-1],
                       waypoint_indices=waypoints, config=planning)
    arrays, quality = retime_planned_curve(route.bezier_xy, route.parameter_knots,
                                           float(source.positions_xyz[0, 2]), motion)
    # Cross-stage checks catch swapped frames, lost endpoints or mismatched clocks.
    np.testing.assert_allclose(arrays["positions_xyz"][[0, -1]], source.positions_xyz[[0, -1]], atol=1e-8, rtol=0)
    angle_error = np.angle(np.exp(1j * (arrays["yaw_rad"][[0, -1]] - source.yaw_rad[[0, -1]])))
    if np.max(np.abs(angle_error)) > 1e-7:
        raise ValueError("planned endpoint orientations changed")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite motion output")
    np.testing.assert_allclose(np.diff(arrays["timestamp_s"]), 1 / motion.sample_hz, atol=1e-10, rtol=0)
    source_length = float(np.linalg.norm(np.diff(source.positions_xyz, axis=0), axis=1).sum())
    sample_length = float(np.linalg.norm(np.diff(arrays["positions_xyz"], axis=0), axis=1).sum())
    package = Path(__file__).resolve().parents[1]
    code_hashes = {str(p.relative_to(package)): _sha256(p) for p in sorted(package.rglob("*.py"))
                   if "tests" not in p.relative_to(package).parts}
    manifest = dict(
        schema_version=1, format="vlnverse_planned_motion_v1", episode_id=source.episode_id,
        instruction=source.instruction, source=source.provenance,
        map_path=str(args.occupancy_npz.resolve()), map_sha256=_sha256(args.occupancy_npz),
        config=dict(planning=asdict(planning), motion=asdict(motion)),
        config_path=str(args.config.resolve()), config_sha256=_sha256(args.config), code_sha256=code_hashes,
        method="ordered_checkpoint_astar_quintic_curve", time_basis="synthetic_planned_seconds",
        coordinates="scene XYZ metres, Z up; tangent yaw radians, positive counterclockwise",
        sampling=dict(mode="uniform_time", sample_hz=motion.sample_hz, uniform_timestamps=True),
        source_frames=len(source.yaw_rad), output_frames=len(arrays["timestamp_s"]),
        source_length_m=source_length, sampled_path_length_m=sample_length,
        quality=quality, planning=route.metadata,
        route_alignment=dict(
            waypoint_indices=waypoints, checkpoint_positions_xyz=source.positions_xyz[waypoints].tolist(),
            checkpoint_notes=notes, selection=settings.get("checkpoint_selection", "explicit configuration"),
            preserves_start_goal_position=True, preserves_start_goal_yaw=True,
            semantic_validation="ordered route constraints; not an automated proof of every instruction clause",
            official_step_correspondence=False),
        limitations=["Synthetic planner reference, not official GT or a physically executed rollout",
                     "Static calibrated 2D map and circular envelope; no full-height robot contact validation",
                     "Fresh rendered observations needed; CPU preview does not reproduce official materials and camera",
                     "Instruction retained under reviewed route constraints; semantic fidelity requires visual review"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_arrays = dict(positions_xyz=source.positions_xyz, yaw_rad=source.yaw_rad)
    if source.source_timestamps is not None:
        source_arrays["source_timestamps"] = source.source_timestamps
    np.savez_compressed(args.output_dir / "source_poses.npz", **source_arrays)
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    np.savez_compressed(args.output_dir / "planned_curve.npz", bezier_xy=route.bezier_xy,
                        parameter_knots=route.parameter_knots, astar_xy=route.astar_xy, guide_xy=route.guide_xy)
    for name in ("source_poses", "trajectory", "planned_curve"):
        manifest[f"{name}_sha256"] = _sha256(args.output_dir / f"{name}.npz")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(episode_id=source.episode_id, frames=manifest["output_frames"],
                         length_m=sample_length, duration_s=quality["duration_s"],
                         continuous_bounds=quality["continuous_maxima_upper_bounds"],
                         manifest=str((args.output_dir / "manifest.json").resolve())), indent=2))


if __name__ == "__main__":
    main()
