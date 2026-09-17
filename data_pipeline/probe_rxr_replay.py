"""Compare RxR replay motion/collision settings without producing training labels."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_pipeline.collect_r2r import Renderer, simulator_config, write_json  # noqa: E402
from data_pipeline.geometry import route_deviation  # noqa: E402
from data_pipeline.vlnce_source import DATA_ROOTS, rxr_replay_actions, source_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", nargs="+", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOTS["rxr"])
    parser.add_argument("--scenes-dir", type=Path, default=Path("/mnt/cfs/data/habitat/scene_datasets"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--role", default="guide")
    parser.add_argument("--rxr-navmesh", action="store_true", help="Probe an in-memory navmesh for RxR height=0.88/radius=0.18")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic report, not a dataset")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    episode_file, gt_file = source_paths(args.data_root, "rxr", args.split, args.role)
    with gzip.open(episode_file, "rt") as stream:
        episodes = {str(e["episode_id"]): e for e in json.load(stream)["episodes"]
                    if str(e["episode_id"]) in args.episode_id}
    with gzip.open(gt_file, "rt") as stream:
        gt = {k: v for k, v in json.load(stream).items() if k in args.episode_id}
    if set(episodes) != set(args.episode_id) or set(gt) != set(episodes):
        raise ValueError("requested episodes must have both source metadata and GT")
    args.config = REPO / "habitat_server/configs/vlnce_r2r.yaml"
    args.width, args.height, args.seed = 470, 280, 20260911
    base = simulator_config(args)
    reports = []
    for episode_id in args.episode_id:
        episode, expert = episodes[episode_id], gt[episode_id]
        scene = args.scenes_dir / episode["scene_id"].removeprefix("data/scene_datasets/")
        for turn, sliding in [(15, True), (15, False), (30, True), (30, False)]:
            cfg = copy.deepcopy(base)
            cfg["turn_angle"] = turn
            cfg["habitat_sim_v0"]["allow_sliding"] = sliding
            if args.rxr_navmesh:
                cfg["agents"]["main_agent"].update(height=.88, radius=.18)
            sim = Renderer(cfg, scene)
            try:
                if args.rxr_navmesh:
                    settings = sim.hsim.NavMeshSettings()
                    settings.set_defaults()
                    settings.agent_height, settings.agent_radius = .88, .18
                    if not sim.sim.recompute_navmesh(sim.sim.pathfinder, settings):
                        raise RuntimeError("RxR in-memory navmesh recomputation failed")
                sim.reset()
                sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                actions = rxr_replay_actions(expert["actions"]) if turn == 15 else expert["actions"]
                positions, collisions = [], 0
                for action in actions:
                    positions.append(np.array(sim.get_agent_state().position, dtype=np.float64))
                    if action == 0:
                        break
                    sim.step(action)
                    collisions += sim.previous_step_collided
                positions = np.asarray(positions)
                report = dict(episode_id=episode_id, scene_id=episode["scene_id"], turn_angle_deg=turn,
                    navmesh_mode="rxr_height_0.88_radius_0.18_in_memory" if args.rxr_navmesh else "existing_file",
                    allow_sliding=sliding, frames=len(positions), collisions=collisions,
                    max_route_deviation_m=float(route_deviation(positions, expert["locations"]).max()),
                    goal_distance_m=float(sim.geodesic_distance(positions[-1], episode["goals"][0]["position"])),
                    source_gt_goal_distance_m=float(sim.geodesic_distance(expert["locations"][-1], episode["goals"][0]["position"])),
                    terminal_world=positions[-1].tolist(), source_gt_terminal_world=expert["locations"][-1],
                    source_goal=episode["goals"][0]["position"])
                reports.append(report)
                print(json.dumps(report), flush=True)
            finally:
                sim.close()
    write_json(args.output, dict(dataset="rxr", split=args.split, role=args.role, results=reports))


if __name__ == "__main__":
    main()
