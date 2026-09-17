#!/usr/bin/env python3
"""One closed-loop LightNav rollout, using a separate native Habitat-Sim worker.

The WebSocket server owns model inference. This client executes the reference
Habitat velocity mapping, records fresh sensors, and renders model predictions.
No expert actions, future observations, or Doubao labels enter model inference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_command(action, control):
    """Invert the reference normalized velocity command, including STOP thresholds."""
    values = []
    for key, limits in (("linear_velocity", "lin_vel_range"),
                        ("angular_velocity", "ang_vel_range")):
        raw = float(action["action_args"][key])
        if not math.isfinite(raw) or not -1 <= raw <= 1:
            raise ValueError("normalized velocity must be finite and in [-1,1]")
        lo, hi = control[limits]
        values.append(lo + (raw + 1) * .5 * (hi - lo))
    linear, angular = values
    stop = (abs(linear) < control["min_abs_lin_speed"]
            and abs(angular) < control["min_abs_ang_speed"])
    return dict(linear_m_s=linear, angular_deg_s=angular, stop=stop,
                time_step_s=control["time_step"])


def predicted_points(raw_text, width, height):
    """Keep model sentinels and exact grid centers; these are predictions, not GT."""
    from lightnav.serving.protocol import pointing_payload
    from lightnav.vln_utils import (
        decode_point_pixel_center,
        safe_parse_apos_token,
        safe_parse_opos_token,
    )

    payload = pointing_payload(raw_text, width=width, height=height) or {}
    result = {}
    for channel, parser in (("apos", safe_parse_apos_token), ("opos", safe_parse_opos_token)):
        token = parser(raw_text)
        pixel = payload.get(channel + "_px")
        grid = None
        if token is not None and 1 <= token <= 1296:
            pixel = list(decode_point_pixel_center(token, width, height))
            row, col = divmod(token - 1, 48)
            grid = dict(row=row, col=col, cell_id=token)
        result[channel] = dict(state=payload.get(channel + "_state", "none"),
                               token_id=token, pixel=pixel, grid=grid,
                               pixel_rc=None if pixel is None else
                               [math.floor(pixel[1]), math.floor(pixel[0])],
                               clamped=payload.get(channel + "_clamped", False))
    return result


def worker(config_path):
    """Native simulator only, executed with the separate Python 3.9 interpreter."""
    # Native C++ logging also uses stdout. Preserve a dedicated protocol descriptor.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    cfg = read_json(config_path)
    directory = Path(cfg["output"])
    with open(directory / "habitat.log", "a") as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
    import magnum as mn
    import quaternion

    from data_pipeline.collect_r2r import Renderer, capture

    sim = Renderer(cfg["simulator"], cfg["scene"])
    try:
        body = np.asarray(cfg["start_world_from_body"])
        quat = quaternion.from_rotation_matrix(body[:3, :3])
        sim.reset()
        sim.set_agent_state(body[:3, 3], [*quat.imag, quat.real])
        velocity = sim.hsim.physics.VelocityControl()
        velocity.controlling_lin_vel = velocity.controlling_ang_vel = True
        velocity.lin_vel_is_local = velocity.ang_vel_is_local = True
        frame_id = 0

        def observe(stopped=False):
            frame = capture(sim, sim.get_observations_at(), directory, frame_id, cfg["camera"])
            distance = sim.geodesic_distance(sim.get_agent_state().position, cfg["goal_position"])
            frame["goal_distance_m"] = float(distance) if np.isfinite(distance) else None
            frame["simulation_time_s"] = frame_id * cfg["control"]["time_step"]
            with (directory / "frames.jsonl").open("a") as stream:
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
            protocol.write(json.dumps(dict(frame=frame, stopped=stopped), allow_nan=False) + "\n")

        observe()
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("close"):
                break
            command = physical_command(request["action"], cfg["control"])
            sim.previous_step_collided = False
            if not command["stop"]:
                state = sim.get_agent_state()
                velocity.linear_velocity = np.array([0., 0., -command["linear_m_s"]])
                velocity.angular_velocity = np.array([0., math.radians(command["angular_deg_s"]), 0.])
                rigid = sim.hsim.RigidState(mn.Quaternion(state.rotation.imag, state.rotation.real),
                                           state.position)
                target = velocity.integrate_transform(command["time_step_s"], rigid)
                filter_step = (sim.sim.pathfinder.try_step if cfg["simulator"]["habitat_sim_v0"]
                               .get("allow_sliding", True) else sim.sim.pathfinder.try_step_no_sliding)
                end = filter_step(state.position, target.translation)
                requested = np.asarray(target.translation) - state.position
                actual = np.asarray(end) - state.position
                sim.previous_step_collided = bool(actual @ actual + 1e-5 < requested @ requested)
                sim.set_agent_state(end, [*target.rotation.vector, target.rotation.scalar])
            frame_id += 1
            observe(command["stop"])
            if command["stop"]:
                break
    finally:
        sim.close()
        protocol.close()


def prepare(args):
    import gzip

    import yaml

    from data_pipeline.collect_r2r import simulator_config

    source = args.source_episode.resolve()
    metadata = read_json(source / "episode.json")
    with (source / "frames.jsonl").open() as stream:
        first = json.loads(stream.readline())
    with gzip.open(args.r2r_data, "rt") as stream:
        original = next(ep for ep in json.load(stream)["episodes"]
                        if str(ep["episode_id"]) == str(metadata["episode_id"]))
    if original["instruction"]["instruction_text"] != metadata["instruction"]:
        raise ValueError("source instruction differs from original R2R episode")
    camera = metadata["camera"]
    args.width, args.height = camera["width"], camera["height"]
    simulator = simulator_config(args)
    rgb = simulator["agents"]["main_agent"]["sim_sensors"]["rgb_sensor"]
    if (rgb["hfov"] != camera["hfov_deg"] or rgb["position"] != camera["position_body_native"]
            or rgb["orientation"] != camera["orientation_euler"]):
        raise ValueError("source camera and simulator configuration disagree")
    control = yaml.safe_load(args.config.read_text())["habitat"]["task"]["actions"]["velocity_control"]
    scene = args.scenes_dir / metadata["scene_id"]
    if not scene.is_file() or not scene.with_suffix(".navmesh").is_file():
        raise FileNotFoundError(scene)
    return dict(format="beta_nav_lightnav_rollout_v1", complete=False,
                episode_id=str(metadata["episode_id"]), instruction=metadata["instruction"],
                source_episode=str(source), source_episode_sha256=digest(source / "episode.json"),
                source_r2r=str(args.r2r_data), source_r2r_sha256=digest(args.r2r_data),
                output=str(args.output.resolve()), scene=str(scene), camera=camera,
                simulator=simulator, control=control, seed=args.seed,
                start_world_from_body=first["world_from_body"], goal_position=original["goals"][0]["position"],
                server=args.server, model_path=str(args.model_path), backend="vllm_local",
                checkpoint_eval_config=read_json(args.model_path / "eval_config.json"),
                checkpoint_config_sha256=digest(args.model_path / "config.json"),
                checkpoint_eval_sha256=digest(args.model_path / "eval_config.json"),
                max_steps=args.max_steps, physical_dt_s=control["time_step"], video_fps=args.fps,
                model_input="current and past RGB PNG observations + instruction; no depth/pose/GT",
                trajectory_source="LightNav RVQ decoded robot-local [forward_m,left_m,yaw_rad]",
                controller="reference first velocity-relevant waypoint; unicycle; lateral dropped",
                pointing_source="LightNav model output; grid centers; native sentinels preserved",
                code_sha256=digest(__file__))


def render_prediction_frame(directory, cfg, frame, prediction):
    """Full instruction above RGB; prediction overlays and a local plan view."""
    from lightnav.viz.render import draw_traj_ribbon

    scale = 2
    width, height = cfg["camera"]["width"], cfg["camera"]["height"]
    small = ImageFont.truetype("DejaVuSans.ttf", 17)
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    from visual.render_episode_gt import wrap_text

    instruction = wrap_text(cfg["instruction"], font, width * scale + 270)
    header = 58 + 28 * len(instruction)
    size = (width * scale + 310, header + height * scale + 138)
    canvas = Image.new("RGB", (size[0], size[1] + size[1] % 2), (17, 24, 39))
    image = np.asarray(Image.open(directory / frame["rgb"]).convert("RGB"))
    waypoints = None if prediction is None else np.asarray(prediction["waypoints_ego"])
    if waypoints is not None:
        image = draw_traj_ribbon(image, waypoints, hfov_deg=cfg["camera"]["hfov_deg"],
                                 cam_height=cfg["camera"]["position_body_native"][1], forward_offset=0.)
    canvas.paste(Image.fromarray(image).resize((width * scale, height * scale)), (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"LightNav | episode {cfg['episode_id']} | closed-loop model prediction",
              font=font, fill=(90, 210, 255))
    for i, line in enumerate(instruction):
        draw.text((18, 48 + i * 28), line, font=font, fill="white")
    if prediction is not None:
        for channel, color in (("apos", (90, 255, 145)), ("opos", (255, 90, 185))):
            point = prediction[channel]
            if point["state"] == "point" and point["pixel"] is not None:
                u, v = point["pixel"]
                x, y = u * scale, header + v * scale
                draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="black", width=6)
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=4)
                label = channel + (" (edge)" if point["clamped"] else "")
                draw.text((min(x + 12, width * scale - 155), max(header, y - 26)),
                          label, font=font, fill=color, stroke_width=2, stroke_fill="black")
    side_x, origin_y = width * scale + 155, header + 370
    draw.text((width * scale + 18, header + 20), "Predicted local path", font=font, fill="white")
    draw.text((width * scale + 18, header + 51), "Forward up | left to left", font=small, fill="gray")
    if waypoints is not None:
        extent = max(1., float(np.max(np.abs(waypoints[:, :2]))) * 1.15)
        pixels_per_m = 115 / extent
        draw.line((side_x, origin_y - 255, side_x, origin_y + 60), fill=(80, 90, 110), width=2)
        draw.line((side_x - 130, origin_y, side_x + 130, origin_y), fill=(80, 90, 110), width=2)
        points = [(side_x, origin_y)] + [(side_x - left * pixels_per_m, origin_y - fwd * pixels_per_m)
                                       for fwd, left, _ in waypoints]
        draw.line(points, fill=(40, 200, 255), width=4)
        for i, (x, y) in enumerate(points[1:]):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="white")
            if i in (0, len(waypoints) - 1):
                draw.text((x + 6, y - 12), str(i + 1), font=small, fill="white")
        draw.text((width * scale + 18, header + 450), f"10 waypoints | scale {extent:.1f} m", font=small,
                  fill="white")
    draw.ellipse((side_x - 6, origin_y - 6, side_x + 6, origin_y + 6), fill=(255, 200, 70))
    footer_y = header + height * scale + 10
    distance = frame["goal_distance_m"]
    distance_text = "disconnected" if distance is None else f"{distance:.2f} m"
    status = "terminal observation (no new prediction)" if prediction is None else (
        "STOP" if prediction["executed_command"]["stop"] else "MOVE")
    draw.text((18, footer_y), f"Frame {frame['frame_id']} | {status} | goal distance {distance_text}",
              font=font, fill="white")
    if prediction is not None:
        draw.text((18, footer_y + 30),
                  f"apos: {prediction['apos']['state']} | opos: {prediction['opos']['state']} | "
                  f"RVQ: {prediction['rvq_codes']} | {prediction['response']['latency_ms']:.0f} ms",
                  font=small, fill="white")
        draw.text((18, footer_y + 56), prediction["raw_text"], font=small, fill=(180, 200, 215))
    draw.text((18, footer_y + 90), "RGB projection assumes a level ground plane. Points are model predictions.",
              font=small, fill=(160, 175, 190))
    return canvas


def render_video(directory, cfg, predictions, frames):
    import imageio.v2 as imageio
    import imageio_ffmpeg

    path = directory / "lightnav_navigation.mp4"
    partial = directory / "lightnav_navigation.partial.mp4"
    with imageio.get_writer(str(partial), fps=cfg["video_fps"], codec="libx264", pixelformat="yuv420p",
                            macro_block_size=2, ffmpeg_params=["-crf", "18", "-movflags", "+faststart"]) as writer:
        for i, frame in enumerate(frames):
            prediction = predictions[i] if i < len(predictions) else None
            image = render_prediction_frame(directory, cfg, frame, prediction)
            writer.append_data(np.asarray(image))
            if i == 0:
                image.save(directory / "preview.png")
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(partial))
    if count != len(frames):
        raise ValueError("video frame count mismatch")
    partial.replace(path)
    return dict(path=str(path), frames=count, duration_seconds=duration, size=list(image.size))


def run(args):
    from websockets.sync.client import connect

    from lightnav.habitat.policy import select_action_waypoint
    from lightnav.serving.protocol import parse_actions_payload
    from lightnav.velocity import first_waypoint_to_velocity_cmd
    from lightnav.vln_utils import parse_rvq_action_tokens

    cfg = prepare(args)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("rgb", "depth"):
        (directory / name).mkdir()
    write_json(directory / "episode.json", cfg)
    (directory / "rollout_lightnav.snapshot.py").write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONPATH=str(REPO / "data_pipeline/.venv/deps"), PYTHONDONTWRITEBYTECODE="1",
               HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet",
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
               LD_PRELOAD="/lib/x86_64-linux-gnu/libGLdispatch.so.0")
    process = subprocess.Popen([str(args.habitat_python), str(Path(__file__).resolve()),
                                "--worker", str(directory / "episode.json")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    predictions, frames = [], []
    reason = "max_steps"
    try:
        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"Habitat worker exited; inspect {directory / 'habitat.log'}")
            return json.loads(line)

        observation = receive()
        frames.append(observation["frame"])
        control = cfg["control"]
        with connect(args.server, max_size=64 * 1024 * 1024, open_timeout=30) as ws:
            def request(action, data):
                ws.send(json.dumps(dict(action=action, data=data)))
                result = json.loads(ws.recv(timeout=300))["data"]
                if result.get("rc") != 0:
                    raise RuntimeError(f"LightNav {action} failed: {result}")
                return result

            request("login", {"clientId": "beta-nav-" + cfg["episode_id"]})
            request("reset", {})
            with (directory / "predictions.jsonl").open("w") as stream:
                for step in range(args.max_steps):
                    frame = observation["frame"]
                    # Lossless PNG input: the server decodes supported image formats with PIL.
                    encoded = base64.b64encode((directory / frame["rgb"]).read_bytes()).decode()
                    response = request("next", dict(seq=step, image=encoded, instruction=cfg["instruction"]))
                    waypoints = np.asarray(parse_actions_payload(response.get("actions")), dtype=np.float32)
                    if waypoints.shape != (10, 3) or not np.isfinite(waypoints).all():
                        raise ValueError("LightNav must return ten finite waypoints")
                    index, chosen = select_action_waypoint(waypoints)
                    action = first_waypoint_to_velocity_cmd(chosen, control["time_step"],
                                                           control["lin_vel_range"], control["ang_vel_range"])
                    raw = response["raw_text"]
                    record = dict(frame_id=step, raw_text=raw, rvq_codes=parse_rvq_action_tokens(raw),
                                  waypoints_ego=waypoints.tolist(), action_waypoint_index=index,
                                  action=action, executed_command=physical_command(action, control),
                                  response=response,
                                  **predicted_points(raw, cfg["camera"]["width"], cfg["camera"]["height"]))
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    predictions.append(record)
                    process.stdin.write(json.dumps(dict(action=action)) + "\n")
                    process.stdin.flush()
                    observation = receive()
                    frames.append(observation["frame"])
                    print(f"step={step} raw={raw} distance={observation['frame']['goal_distance_m']} "
                          f"stop={observation['stopped']}", flush=True)
                    if observation["stopped"]:
                        reason = "model_stop" if response["stop"] else "controller_below_stop_threshold"
                        break
        if reason == "max_steps":
            process.stdin.write('{"close":true}\n')
            process.stdin.flush()
        process.wait(timeout=30)
        if process.returncode:
            raise RuntimeError(f"Habitat worker failed: exit {process.returncode}")
        distance = frames[-1]["goal_distance_m"]
        cfg.update(steps=len(predictions), frames=len(frames), termination_reason=reason,
                   goal_distance_m=distance, model_stopped=reason == "model_stop",
                   reached_within_3m=distance is not None and distance < 3,
                   success_3m=reason in ("model_stop", "controller_below_stop_threshold")
                   and distance is not None and distance < 3,
                   apos_states=dict(Counter(p["apos"]["state"] for p in predictions)),
                   opos_states=dict(Counter(p["opos"]["state"] for p in predictions)))
        write_json(directory / "episode.json", cfg)
        cfg["video"] = render_video(directory, cfg, predictions, frames)
        cfg["complete"] = True
        cfg["predictions_sha256"] = digest(directory / "predictions.jsonl")
        cfg["frames_sha256"] = digest(directory / "frames.jsonl")
        write_json(directory / "episode.json", cfg)
        print(json.dumps({k: cfg[k] for k in ("steps", "frames", "termination_reason", "goal_distance_m",
                                              "apos_states", "opos_states", "video")}, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:8067")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/cfs/Models/Light"))
    parser.add_argument("--habitat-python", type=Path,
                        default=Path("/mnt/pfs/jiayilin/Image2Sim-Robot/envs/habitat-image2sim/bin/python"))
    parser.add_argument("--config", type=Path, default=REPO / "habitat_server/configs/vlnce_r2r.yaml")
    parser.add_argument("--r2r-data", type=Path,
                        default=Path("/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"))
    parser.add_argument("--scenes-dir", type=Path, default=Path("/mnt/cfs/data/habitat/scene_datasets"))
    parser.add_argument("--gpu-id", type=int, default=3, help="Habitat render GPU; server GPU is separate.")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=5, help="Video playback rate, independent of control dt.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or not math.isfinite(args.fps) or not 0 < args.fps <= 60:
        parser.error("require max-steps >= 1 and 0 < fps <= 60")
    run(args)


if __name__ == "__main__":
    main()
