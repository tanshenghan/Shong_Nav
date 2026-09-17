#!/usr/bin/env python3
"""Re-render R2R/RxR experts with LightNav's Habitat camera; export RVQ-10/FM-30.

Run with the separate Habitat Python 3.9 interpreter. Geometry lives in this
package; only the dependency-light traj_vocab decoder is shared with inference.
R2R supplies the route, start state and instruction; all saved sensors and pose
labels come from this fresh simulator rollout. No neural model is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data_pipeline.geometry import (  # noqa: E402
    encode_rvq,
    future_targets,
    pinhole_intrinsics,
    route_deviation,
)
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
from data_pipeline.vlnce_source import LANGUAGES, SPLITS, load_source  # noqa: E402
from lightnav.traj_vocab import load_rvq_bundle  # noqa: E402


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_episodes(episodes, count, seed):
    """One instruction per physical route; deterministic, without success filtering."""
    pool = list(episodes)
    random.Random(seed).shuffle(pool)
    selected, seen = [], set()
    for episode in pool:
        key = (episode["scene_id"], episode.get("trajectory_id", episode["episode_id"]))
        if key in seen:
            continue
        selected.append(episode)
        seen.add(key)
        if count > 0 and len(selected) >= count:
            break
    return sorted(selected, key=lambda e: (e["scene_id"], str(e["episode_id"])))


def simulator_config(args):
    """Read LightNav's simulator parameters without requiring Habitat-Lab tasks."""
    import yaml

    cfg = yaml.safe_load(args.config.read_text())["habitat"]["simulator"]
    cfg["seed"] = args.seed
    cfg["habitat_sim_v0"]["gpu_device_id"] = args.gpu_id
    agent = cfg["agents"]["main_agent"]
    # Habitat-Lab AgentConfig and depth-sensor defaults used by LightNav.
    agent.setdefault("height", 1.5)
    agent.setdefault("radius", .1)
    if getattr(args, "dataset", "r2r") == "rxr":
        # RxR experts were generated with a different walkable mesh. Reusing
        # the R2R mesh blocks valid stair routes even with correct turn angles.
        agent.update(height=.88, radius=.18)
        cfg["habitat_sim_v0"]["allow_sliding"] = False
        cfg["navmesh_policy"] = "rxr_in_memory"
    sensors = agent["sim_sensors"]
    for sensor in sensors.values():
        sensor["width"], sensor["height"] = args.width, args.height
        sensor.setdefault("orientation", [0., 0., 0.])
    sensors["depth_sensor"]["normalize_depth"] = False
    sensors["depth_sensor"].setdefault("min_depth", 0.)
    sensors["depth_sensor"].setdefault("max_depth", 10.)
    for key in ("width", "height", "hfov", "position", "orientation"):
        if sensors["rgb_sensor"][key] != sensors["depth_sensor"][key]:
            raise ValueError("RGB and depth must have matching calibration")
    return cfg


def configure_navigation_mesh(sim, habitat_sim, cfg, scene):
    """Load R2R's mesh or rebuild RxR's in memory; never write source assets."""
    if cfg.get("navmesh_policy") == "rxr_in_memory":
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = cfg["agents"]["main_agent"]["height"]
        settings.agent_radius = cfg["agents"]["main_agent"]["radius"]
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError(f"RxR navigation mesh recomputation failed: {scene}")
    else:
        navmesh = Path(scene).with_suffix(".navmesh")
        if not navmesh.is_file() or not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise FileNotFoundError(f"existing MP3D navmesh required: {navmesh}")


class Renderer:
    """Native Habitat-Sim using LightNav camera, action and collision settings.

    Avoids loading unrelated task/sensor registries from a second Habitat-Lab
    installation. R2R loads the existing mesh; RxR rebuilds its mesh in memory
    using the source agent dimensions. Original scene assets are never modified.
    """

    def __init__(self, cfg, scene, *, semantic=False, scene_dataset_config=None):
        import habitat_sim
        import magnum as mn

        self.hsim = habitat_sim
        self.scene = str(scene)
        self.cfg = cfg
        self.previous_step_collided = False
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene)
        if scene_dataset_config is not None:
            backend.scene_dataset_config_file = str(scene_dataset_config)
        backend.gpu_device_id = cfg["habitat_sim_v0"]["gpu_device_id"]
        backend.allow_sliding = cfg["habitat_sim_v0"].get("allow_sliding", True)
        backend.enable_physics = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent_cfg = cfg["agents"]["main_agent"]
        agent.height, agent.radius = agent_cfg["height"], agent_cfg["radius"]
        specifications = []
        for name, sensor_cfg in agent_cfg["sim_sensors"].items():
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = "rgb" if name == "rgb_sensor" else "depth"
            spec.sensor_type = habitat_sim.SensorType.COLOR if spec.uuid == "rgb" else habitat_sim.SensorType.DEPTH
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [sensor_cfg["height"], sensor_cfg["width"]]
            spec.position = mn.Vector3(*sensor_cfg["position"])
            spec.orientation = mn.Vector3(*sensor_cfg["orientation"])
            spec.hfov = mn.Deg(float(sensor_cfg["hfov"]))
            specifications.append(spec)
        if semantic:
            rgb_cfg = agent_cfg["sim_sensors"]["rgb_sensor"]
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = "semantic"
            spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [rgb_cfg["height"], rgb_cfg["width"]]
            spec.position = mn.Vector3(*rgb_cfg["position"])
            spec.orientation = mn.Vector3(*rgb_cfg["orientation"])
            spec.hfov = mn.Deg(float(rgb_cfg["hfov"]))
            specifications.append(spec)
        agent.sensor_specifications = specifications
        agent.action_space = {
            1: habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=cfg["forward_step_size"])),
            2: habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=cfg["turn_angle"])),
            3: habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=cfg["turn_angle"])),
        }
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
        try:
            configure_navigation_mesh(self.sim, habitat_sim, cfg, scene)
        except Exception:
            self.sim.close()
            raise
        self.sim.seed(cfg["seed"])

    def reset(self):
        self.sim.reset()
        self.previous_step_collided = False

    def set_agent_state(self, position, rotation):
        from habitat_sim.utils.common import quat_from_coeffs

        state = self.hsim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_coeffs(rotation)
        self.sim.get_agent(0).set_state(state, reset_sensors=True)

    def get_agent_state(self):
        return self.sim.get_agent(0).get_state()

    def _sensors(self, raw):
        depth_cfg = self.cfg["agents"]["main_agent"]["sim_sensors"]["depth_sensor"]
        result = {"rgb": np.asarray(raw["rgb"])[..., :3],
                  "depth": np.clip(raw["depth"], depth_cfg["min_depth"], depth_cfg["max_depth"])}
        if "semantic" in raw:
            result["semantic"] = raw["semantic"]
        return result

    def get_observations_at(self):
        return self._sensors(self.sim.get_sensor_observations())

    def step(self, action):
        raw = self.sim.step(action)
        self.previous_step_collided = bool(raw.get("collided", False))
        return self._sensors(raw)

    def geodesic_distance(self, start, end):
        path = self.hsim.ShortestPath()
        path.requested_start, path.requested_end = start, end
        self.sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def close(self):
        self.sim.close()


def pose_matrix(state):
    import quaternion

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
    matrix[:3, 3] = state.position
    return matrix


def capture(sim, observation, destination, index, camera):
    """Write the current observation before stepping; export no discrete action label."""
    rgb = np.asarray(observation["rgb"])
    depth = np.asarray(observation["depth"], dtype=np.float32).squeeze()
    h, w = camera["height"], camera["width"]
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8 or depth.shape != (h, w):
        raise ValueError(f"unexpected rendered sensors: RGB {rgb.shape}/{rgb.dtype}, depth {depth.shape}")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("invalid depth values")
    depth_valid = (depth > camera["min_depth_m"]) & (depth < camera["max_depth_m"])
    state = sim.get_agent_state()
    body = pose_matrix(state)
    # Native OpenGL camera: +X right, +Y up, -Z forward.
    # Stored OpenCV camera: +X right, +Y down, +Z forward.
    cv_from_gl = np.diag([1., -1., -1., 1.])
    rgb_pose = pose_matrix(state.sensor_states["rgb"]) @ cv_from_gl
    depth_pose = pose_matrix(state.sensor_states["depth"]) @ cv_from_gl
    np.testing.assert_allclose(rgb_pose, depth_pose, atol=1e-6)
    offset = (np.linalg.inv(body) @ rgb_pose)[:3, 3]
    np.testing.assert_allclose(offset, camera["position_body_native"], atol=2e-5)
    rgb_name, depth_name = f"rgb/{index:06d}.png", f"depth/{index:06d}.npz"
    Image.fromarray(rgb).save(destination / rgb_name)
    np.savez_compressed(destination / depth_name, depth_m=depth, valid=depth_valid)
    return {
        "frame_id": index, "rgb": rgb_name, "depth": depth_name,
        "world_from_body": body.tolist(),
        "world_from_camera_cv": rgb_pose.tolist(),
        "world_from_depth_camera_cv": depth_pose.tolist(),
        "collision_from_previous_action": bool(sim.previous_step_collided),
        "valid_depth_fraction": float(depth_valid.mean()),
    }


def export_targets(directory, frames, bundle, terminal_success, camera, pointing_config):
    pointing_config.require_current()
    poses = np.asarray([f["world_from_body"] for f in frames])
    target = future_targets(poses, 30)
    n = len(frames)
    codes = np.full((n, len(bundle.levels)), -1, dtype=np.int32)
    reconstruction = np.zeros((n, bundle.horizon, 3), dtype=np.float32)
    rvq_valid = target["valid"][:, :bundle.horizon].all(axis=1)
    errors = np.full(n, np.nan, dtype=np.float32)
    apos, opos = [], []
    for t, frame in enumerate(frames):
        with np.load(directory / frame["depth"], allow_pickle=False) as depth:
            a = ground_point(poses[:, :3, 3], t, frame["world_from_camera_cv"],
                             camera["K"], depth["depth_m"], depth["valid"], pointing_config)
            height, width = depth["depth_m"].shape
            apos.append(attach_pixel_grid(a, width, height))
            o = terminal_ground_point(poses[:, :3, 3], frame["world_from_camera_cv"],
                                      camera["K"], depth["depth_m"], depth["valid"], pointing_config,
                                      frame_index=t)
            opos.append(attach_pixel_grid(o, width, height))
    for t in np.flatnonzero(rvq_valid):
        codes[t], reconstruction[t], errors[t] = encode_rvq(
            target["waypoints"][t, :bundle.horizon], bundle
        )
    if terminal_success:
        codes[-1] = bundle.manifest["stop"]["codes"]
        reconstruction[-1] = bundle.decode_waypoints(codes[-1])
        rvq_valid[-1] = True
    opos_arrays = {
        "opos_pixel": np.asarray([p["pixel"] if p["state"] == "point" else [np.nan] * 2
                                   for p in opos], dtype=np.float32),
        "opos_world": np.asarray([p["world_point"] if p["state"] == "point" else [np.nan] * 3
                                   for p in opos], dtype=np.float32),
        "opos_token_id": np.asarray([p["token_id"] for p in opos], dtype=np.int32),
        "opos_valid_mask": np.asarray([p["loss_valid"] for p in opos], dtype=bool),
        **ground_target_arrays(opos, "opos"), **pixel_grid_arrays(opos, "opos"),
    }
    np.savez_compressed(
        directory / "targets.npz",
        world_from_body=poses,
        world_from_camera_cv=np.asarray([f["world_from_camera_cv"] for f in frames]),
        world_from_depth_camera_cv=np.asarray([f["world_from_depth_camera_cv"] for f in frames]),
        fm_waypoints_ego=target["waypoints"], fm_clean_se2_diff=target["se2_diff"],
        fm_valid_mask=target["valid"], future_frame_ids=target["frame_ids"],
        future_vertical_offset_m=target["vertical_offset_m"],
        rvq_codes=codes, rvq_valid_mask=rvq_valid,
        rvq_reconstruction=reconstruction, rvq_ade_m=errors,
        apos_pixel=np.asarray([p["pixel"] if p["loss_valid"] else [np.nan, np.nan] for p in apos]),
        apos_world=np.asarray([p["world_point"] if p["loss_valid"] else [np.nan] * 3 for p in apos]),
        apos_token_id=np.asarray([p["token_id"] for p in apos], dtype=np.int32),
        apos_valid_mask=np.asarray([p["loss_valid"] for p in apos]),
        **ground_target_arrays(apos),
        **pixel_grid_arrays(apos, "apos"),
        **opos_arrays,
    )
    with (directory / "samples.jsonl").open("w") as stream:
        for t, frame in enumerate(frames):
            sample = {
                "frame_id": t, "current_rgb": frame["rgb"],
                "history_start": 0, "history_end_inclusive": t,
                "array_file": "targets.npz", "array_row": t,
                "rvq_codes": codes[t].tolist() if rvq_valid[t] else None,
                "rvq_tokens": "".join(f"<act_l{level}_{int(code)}>" for level, code in enumerate(codes[t]))
                if rvq_valid[t] else None,
                "rvq_loss_valid": bool(rvq_valid[t]),
                "fm_valid_steps": int(target["valid"][t].sum()),
                "explicit_stop": bool(t == n - 1 and terminal_success),
                "stop_loss_valid": bool(t == n - 1 and terminal_success),
                "apos": apos[t],
                "apos_tokens": pointing_token("apos", apos[t]),
            }
            sample.update(opos=opos[t],
                          opos_tokens=pointing_token("opos", opos[t]))
            stream.write(json.dumps(sample) + "\n")
    finite = errors[np.isfinite(errors)]
    return {
        "frames": n, "full_rvq_10_chunks": int(target["valid"][:, :10].all(axis=1).sum()),
        "full_fm_30_chunks": int(target["valid"].all(axis=1).sum()),
        "rvq_supervised_samples": int(rvq_valid.sum()),
        "terminal_stop_supervised": bool(terminal_success),
        "apos_supervised_samples": sum(p["loss_valid"] for p in apos),
        "opos_supervised_samples": sum(p["loss_valid"] for p in opos),
        "opos_visible": sum(p["state"] == "point" for p in opos),
        "opos_visibility_start_frame": pointing_config.opos_visibility_start_frame(n),
        "pointing_grid": pixel_grid_metadata(width, height),
        "rvq_mean_ade_m": float(finite.mean()) if len(finite) else None,
        "rvq_max_ade_m": float(finite.max()) if len(finite) else None,
    }


def collect_episode(sim, episode, gt, directory, camera, bundle, pointing_config=None):
    actions = [int(a) for a in gt["actions"]]
    if not actions or actions[-1] != 0 or 0 in actions[:-1]:
        raise ValueError("expected an expert route with exactly one final STOP")
    if any(a not in (0, 1, 2, 3) for a in actions):
        raise ValueError("unsupported expert action")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "rgb").mkdir()
    (directory / "depth").mkdir()
    sim.reset()
    sim.set_agent_state(episode["start_position"], episode["start_rotation"])
    observation = sim.get_observations_at()
    frames = []
    with (directory / "frames.jsonl").open("w") as stream:
        for t, action in enumerate(actions):
            frame = capture(sim, observation, directory, t, camera)
            frames.append(frame)
            stream.write(json.dumps(frame) + "\n")
            if action == 0:
                break
            observation = sim.step(action)
    poses = np.asarray([f["world_from_body"] for f in frames])
    positions = poses[:, :3, 3]
    deviation = route_deviation(positions, np.asarray(gt["locations"]))
    goal_distance = float(sim.geodesic_distance(positions[-1], episode["goals"][0]["position"]))
    if not np.isfinite(goal_distance):
        raise ValueError("replayed route ends disconnected from the goal")
    success = goal_distance < 3.0
    # A mismatched start heading / action convention otherwise creates bad labels silently.
    if float(deviation.max()) > 0.5 or not success:
        raise ValueError(f"expert replay failed: deviation={deviation.max():.3f}m, goal={goal_distance:.3f}m")
    pointing_config = pointing_config or PointingConfig()
    result = export_targets(directory, frames, bundle, success, camera, pointing_config)
    result.update(
        episode_id=str(episode["episode_id"]), trajectory_id=episode.get("trajectory_id"),
        scene_id=episode["scene_id"], instruction=episode["instruction"]["instruction_text"],
        goal_distance_m=goal_distance, max_gt_route_deviation_m=float(deviation.max()),
        source_reference_path=episode.get("reference_path"),
        goal_position=episode["goals"][0]["position"],
        camera=camera, pointing_config=pointing_config.metadata(),
        opos_status="terminal_ground_complete", complete=True,
    )
    if "source_metadata" in episode:
        result["source_metadata"] = episode["source_metadata"]
    write_json(directory / "episode.json", result)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New collection directory; never overwrite")
    parser.add_argument("--data-root", type=Path, required=True, help="R2R or RxR VLN-CE source root")
    parser.add_argument("--scenes-dir", type=Path, required=True, help="Root containing mp3d/<scene>/<scene>.glb")
    parser.add_argument("--rvq-bundle", type=Path, required=True, help="Released action_tokenizer directory")
    parser.add_argument("--config", type=Path, default=REPO / "habitat_server/configs/vlnce_r2r.yaml")
    parser.add_argument("--dataset", choices=["r2r", "rxr"], default="r2r")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--role", choices=["guide", "follower"], default="guide", help="RxR source role")
    parser.add_argument("--languages", nargs="+", choices=["all", *LANGUAGES], default=["all"])
    parser.add_argument("--episodes", type=int, default=10, help="Unique physical routes; -1 for all")
    parser.add_argument("--episode-ids-file", type=Path,
                        help="JSON list of exact source IDs; overrides route sampling, preserves instructions")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=470)
    parser.add_argument("--height", type=int, default=280)
    parser.add_argument("--pointing-config", type=Path, default=REPO / "data_pipeline/configs/pointing.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print manifest without rendering/writing")
    args = parser.parse_args()
    if args.episodes == 0 or args.episodes < -1 or min(args.width, args.height) < 1:
        parser.error("episodes must be -1 or positive; dimensions must be positive")
    return args


def main():
    args = arguments()
    import habitat_sim

    pointing_config = PointingConfig(**json.loads(args.pointing_config.read_text()))
    pointing_config.require_current()
    all_episodes, gt, source_selection = load_source(
        args.data_root, args.dataset, args.split, args.role, args.languages)
    episode_path, gt_path = (Path(source_selection[k]) for k in ("episode_file", "gt_file"))
    if args.episode_ids_file is not None:
        requested = [str(x) for x in json.loads(args.episode_ids_file.read_text())]
        by_id = {str(e["episode_id"]): e for e in all_episodes}
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= by_id.keys():
            raise ValueError("episode ID list must be nonempty, unique and present in the source split")
        episodes = sorted((by_id[x] for x in requested), key=lambda e: (e["scene_id"], str(e["episode_id"])))
    else:
        episodes = select_episodes(all_episodes, args.episodes, args.seed)
    bundle = load_rvq_bundle(args.rvq_bundle, 10)
    if bundle.levels != [256, 256, 256] or bundle.manifest.get("alpha_mode") != "none":
        raise ValueError("expected the released 3x256 RVQ bundle with alpha_mode=none")
    cfg = simulator_config(args)
    if args.dataset == "rxr" and (not np.isclose(cfg["turn_angle"], 15.)
                                 or not np.isclose(cfg["forward_step_size"], .25)):
        raise ValueError("RxR source conversion requires LightNav 15-degree turns and 0.25-metre forward steps")
    sensor = cfg["agents"]["main_agent"]["sim_sensors"]["rgb_sensor"]
    depth_cfg = cfg["agents"]["main_agent"]["sim_sensors"]["depth_sensor"]
    camera = {
        "width": args.width, "height": args.height, "hfov_deg": float(sensor["hfov"]),
        "position_body_native": sensor["position"], "orientation_euler": sensor["orientation"],
        "K": pinhole_intrinsics(args.width, args.height, float(sensor["hfov"])).tolist(),
        "min_depth_m": float(depth_cfg["min_depth"]), "max_depth_m": float(depth_cfg["max_depth"]),
        "depth_unit": "metre", "depth_type": "camera_z", "depth_normalized": False,
        "depth_clipped_to_range": True, "invalid_depth": "use each depth NPZ valid mask",
    }
    scene_files = {}
    for ep in episodes:
        relative = ep["scene_id"].removeprefix("data/scene_datasets/")
        scene = args.scenes_dir / relative
        if not scene.is_file() or str(ep["episode_id"]) not in gt:
            raise FileNotFoundError(f"missing scene or expert route: {scene} / {ep['episode_id']}")
        scene_files[ep["scene_id"]] = scene
    manifest = {
        "format": "beta_nav_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": f"{args.dataset.upper()} expert actions replayed in LightNav-configured Habitat; no model inference",
        "source_selection": source_selection,
        "split": args.split, "episodes": len(episodes), "seed": args.seed,
        "selection": "seeded shuffle, one instruction per unique scene/trajectory, then scene sorted",
        "episode_ids": [str(e["episode_id"]) for e in episodes],
        "data_source": str(episode_path), "gt_source": str(gt_path),
        "source_sha256": {str(p): sha256(p) for p in [episode_path, gt_path, args.config, args.pointing_config]},
        "collector_sha256": sha256(__file__),
        "geometry_code_sha256": sha256(REPO / "data_pipeline/geometry.py"),
        "pointing_code_sha256": sha256(REPO / "data_pipeline/pointing.py"),
        "source_adapter_sha256": sha256(REPO / "data_pipeline/vlnce_source.py"),
        "simulator_config": cfg,
        "renderer": "native Habitat-Sim with LightNav camera/action sampling; "
                    + ("RxR source collision settings and in-memory navmesh" if args.dataset == "rxr"
                       else "existing MP3D navmeshes"),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "python": sys.executable, "camera": camera,
        "world_axes": "Habitat: +Y up; native body +X right, +Y up, -Z forward",
        "camera_axes": "OpenCV: +X right, +Y down, +Z forward",
        "trajectory_axes": "forward metres, left metres, CCW yaw radians; local to current body",
        "sampling_mode": "consecutive_expert_action_transitions", "physical_dt_s": None,
        "rvq_horizon": 10, "fm_horizon": 30, "fm_representation": "se2_diff",
        "fm_normalization": "none; choose scales from training set before training",
        "rvq_encoder": "greedy residual ADE + 0.3 endpoint yaw; not the unpublished training encoder",
        "rvq_bundle": str(args.rvq_bundle),
        "rvq_sha256": {p.name: sha256(p) for p in args.rvq_bundle.iterdir() if p.is_file()},
        "pointing_labels": {"apos": "t+10 ground point; nearest visible point backwards along path",
                            "opos": "fixed recorded terminal ground position; configured frame window, "
                                    "then projection and depth visibility"},
        "pointing_config": pointing_config.metadata(),
        "discrete_action_labels": False,
        "tail_policy": "unknown future is masked; terminal STOP separately verified, no synthetic future padding labels",
    }
    if args.episode_ids_file is not None:
        manifest["selection"] = "explicit episode ID list; no instruction deduplication"
        manifest["episode_ids_sha256"] = sha256(args.episode_ids_file)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "resolved_simulator.json", cfg)
    results, sim = [], None
    try:
        for index, episode in enumerate(episodes):
            scene = str(scene_files[episode["scene_id"]])
            if sim is None or sim.scene != scene:
                if sim is not None:
                    sim.close()
                    sim = None
                sim = Renderer(cfg, scene)
            directory = args.output_dir / "episodes" / str(episode["episode_id"])
            result = collect_episode(sim, episode, gt[str(episode["episode_id"])], directory,
                                     camera, bundle, pointing_config)
            results.append(result)
            print(json.dumps({"completed": index + 1, "total": len(episodes),
                              "episode_id": result["episode_id"], "frames": result["frames"],
                              "rvq_ade_m": result["rvq_mean_ade_m"]}), flush=True)
            write_json(args.output_dir / "progress.json", {
                "completed": len(results), "requested": len(episodes),
                "frames": sum(r["frames"] for r in results),
            })
        summary = {
            "complete": True, "episodes": len(results),
            "frames": sum(r["frames"] for r in results),
            "full_rvq_10_chunks": sum(r["full_rvq_10_chunks"] for r in results),
            "full_fm_30_chunks": sum(r["full_fm_30_chunks"] for r in results),
            "camera": camera, "output_dir": str(args.output_dir),
        }
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {
            "error": repr(exc), "completed_episodes": len(results),
            "note": "Only episode directories with complete=true in episode.json are usable.",
        })
        raise
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
