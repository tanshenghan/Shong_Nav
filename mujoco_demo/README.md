# mujoco_demo

A small, self-contained TurtleBot vision-language-navigation simulation
(Python package `vln_mujoco`): it runs
in the fixed MolmoSpaces ProcTHOR 10K validation `val_2` multi-room scene, using
the ceiling MJCF variant, TurtleBot kinematics, and MuJoCo RGB camera rendering,
with a single web page for VLN instructions, live view, WASD driving, emergency
stop, reset, and state readout.

## Highlights

- One scene: only the MolmoSpaces resources actually referenced by
  `val_2_ceiling` are bundled — no full dataset download required;
- One robot: a differential-drive TurtleBot geometry defined in-project, with no
  external meshes;
- One process: Python runs MuJoCo, the web page, and the VLN WebSocket client
  together;
- Pure kinematics: velocity commands are integrated directly into the TurtleBot
  pose — no motors, gravity, or contact dynamics;
- Two views: the web page switches live between the robot's first-person RGB
  and a rear-elevated third-person follow camera;
- On-change state sync: web state is pushed only when the robot, VLN, control
  ownership, or configuration changes;
- VLN semantic feedback: the first-person view shows APOS/OPOS markers, and
  `stop=true` ends the task and releases control automatically;
- MPC control: the same CasADi/IPOPT unicycle MPC, parameters, and
  capture-time pose alignment as `robot_deploy`'s `vln_mpc`;
- No ROS 2, no Node.js, no locomotion policy;
- The web interaction and VLN server protocol are compatible with
  `robot_deploy`'s `vln_web` and `vln_client`.

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+:

```bash
./run.sh
```

Open <http://127.0.0.1:8088>. Without a VLN server configured you can still
click **Take control** on the page and drive with WASD. A default server
address can also be set at startup:

```bash
./run.sh --vln-server ws://127.0.0.1:8050
./run.sh --host 0.0.0.0 --port 8088
```

Pressing space or clicking `STOP` on the page zeroes the velocity immediately.
Manual commands also auto-zero when not refreshed for 350 ms.

## Driving it with your trained Beta model

From the repository root:

```bash
bash mujoco_demo/run_beta.sh --host 0.0.0.0
```

This starts the HF Beta server, waits for a successful warm-up, and starts the
demo at port 8088. The default checkpoint is the latest committed checkpoint in
`/mnt/cfs/Models/beta_nav/r2r_fm_only_h30_prefix_kv_all36_beta1_1p5`
(step 3699 when configured). The server loads the trained expert and uses the
checkpoint's 30-waypoint horizon and continuous training prompt. It logs the
resolved checkpoint to `logs/beta-demo-server.log`.
For the base VLM, the launcher prefers the readable project-local copy at
`checkpoints/Light-demo`, then `/mnt/cfs/Models/Light`; `MODEL_PATH` overrides it.

Open the page, enter an instruction and press **Start VLN**. Check that LATENCY
is nonzero, WAYPOINTS increases, and the blue trajectory updates.
The FM-only model has no trained semantic STOP head. The Beta launcher now uses
the frozen original LightNav RVQ branch to decide STOP, while Beta FM supplies
the motion trajectory. Set `BETA_STOP_MODE=manual` to disable this extra inference.
The STOP decision is a model prediction, not a guarantee that the goal is reached.
You can always press **STOP** to end navigation. The client accepts replies without pointing and allows up to
30 seconds for each inference while the control loop runs independently.
Motion in this new scene is a deployment check, not proof of
navigation accuracy.

Overrides (environment variables): `BETA_CHECKPOINT`, `MODEL_PATH`,
`BETA_STOP_MODE` (`rvq` by default, or `manual`), `BETA_GPU`
(default `0`), `BETA_PORT` (default `8050`), `BETA_PYTHON` (default
`../.venv-beta-demo/bin/python`) and `DEMO_PYTHON` (default `BETA_PYTHON`). The
server environment needs the LightNav dependencies; the demo interpreter needs
the dependencies in this directory's `pyproject.toml`. Exiting the demo stops
the server launched by this script. For example:

```bash
BETA_GPU=1 BETA_CHECKPOINT=/path/to/step-00003699 bash mujoco_demo/run_beta.sh
```

To run the processes on different machines, start the server separately:

```bash
python -m lightnav.serving.ws_server --task vln --backend hf \
  --model_path /mnt/cfs/Models/Light --beta_checkpoint /path/to/beta-run
./run.sh --vln-server ws://GPU_HOST:8050
```

GPU monitoring in the local demo environment:

```bash
source .venv-beta-demo/bin/activate  # from repository root
nvitop
```

## Driving it with LightNav-0

For the configured local runtime and readable original checkpoint, start both
processes from the repository root:

```bash
bash mujoco_demo/run_lightnav.sh --host 0.0.0.0 --port 8088
```

This uses the original LightNav HF backend, RVQ actions and STOP prediction.
It does not load a Beta checkpoint. The environment directory is shared with
the Beta demo only for installed dependencies. Logs are in
`logs/lightnav-demo-server.log`. Overrides: `MODEL_PATH` (default
`checkpoints/Light-demo`), `LIGHTNAV_PYTHON` (default
`.venv-beta-demo/bin/python`), `LIGHTNAV_GPU` (default `0`), `LIGHTNAV_PORT`
(default `8050`) and `DEMO_PYTHON` (default `LIGHTNAV_PYTHON`).
Exiting the launcher stops both processes.

Start `lightnav-serve` on a GPU host (see
[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)):

```bash
PORT=8050 CUDA_VISIBLE_DEVICES=0 lightnav-serve \
    --task vln --model_path checkpoints/LightNav-0 --backend vllm_local
```

Then paste `ws://<gpu-host>:8050` into the console's **VLN Server WebSocket**
field, type an instruction, and press **Start VLN**. The simulated TurtleBot
streams its first-person frames to the server and tracks the returned waypoints
with the MPC — the same client protocol and controller the real robots use in
[`robot_deploy/`](../robot_deploy/README.md).

## Layout

The browser receives a continuous 20 FPS JPEG WebSocket stream on `/camera/ws`.
At most 12 frames are in flight; slow viewers skip to the newest frame instead of
accumulating an unbounded video delay. CAMERA shows the browser's received FPS
(hover for server render FPS). Runtime telemetry is limited to 10 Hz and uses a
separate control connection. `/api/camera.jpg` and `/api/third-person.jpg` remain
available as single annotated frames.

The trajectory is rendered as small ground markers and thin connectors in the
MuJoCo scene, with real perspective and object occlusion in both camera views.
Coordinates are in meters and anchored at image-capture pose; 30 predictions at
0.25 m each can legitimately span 7.5 m. Display rendering never feeds annotated
frames back into Beta. STOP clears both the path and controller state, rejects
late inference/MPC results, and has a POST `/api/stop` fallback. If predictions
stop arriving for five seconds, the robot pauses until a fresh path arrives.

```text
mujoco_demo/
├── vln_mujoco/
│   ├── assets/        # a single MolmoSpaces scene, nothing more
│   ├── web/           # static single page, no build step
│   ├── model.py       # scene + TurtleBot MJCF
│   ├── simulation.py  # kinematics, cameras, wheel visualisation
│   ├── mpc.py         # CasADi/IPOPT kinematic MPC
│   ├── vln_client.py  # VLN WebSocket client
│   └── server.py      # HTTP/WebSocket and control ownership
├── scripts/
├── tests/
└── run.sh
```

Body-frame waypoints returned by VLN are first transformed into the world frame
using the robot pose at image-capture time, then tracked by the MPC in the
current robot-local frame. The control rate, horizon, model, cost, constraints,
IPOPT settings, and parameters all match `robot_deploy`'s `vln_mpc`:

| Parameter | Value |
| --- | --- |
| control rate / `horizon` | `10 Hz` / `5` |
| MPC / waypoint `dt` | `0.1 s` / `0.1 s` |
| `track_v_max` / `objnav_v_max` | `1.5 m/s` / `0.8 m/s` |
| `w_max` | `3.0 rad/s` |
| `a_max_v` / `a_max_w` | `2.0 m/s²` / `5.0 rad/s²` |
| `q_x` / `q_y` / `q_yaw` | `10.0` / `10.0` / `1.0` |
| `r_v` / `r_w` | `0.1` / `0.1` |
| IPOPT | `max_iter=100`, `acceptable_tol=1e-8`, `acceptable_obj_change_tol=1e-6` |

## Why a single MolmoSpaces scene is enough

The official MolmoSpaces resource manager installs per scene archive, but
ProcTHOR scene XMLs also reference shared THOR meshes/textures. With the default
MolmoSpaces initialisation flow, the shared THOR object source may be unpacked
in full, so "selecting one scene" does not necessarily mean only that scene's
files end up on disk.

This project runs a pruning script once before release: it parses every `file=`
reference in the fixed scene XML and copies only the meshes/textures inside that
closure, preserving the original directory layout. The current asset closure is
62.4 MiB, so after cloning there is no need for the MolmoSpaces repo or its
downloader.

Developers can regenerate the assets from an existing MolmoSpaces checkout:

```bash
python3 scripts/vendor_molmospaces_scene.py \
  ~/Desktop/molmospaces/assets/scenes/procthor-10k-val/val_2_ceiling.xml
```

The upstream revision, file list, and checksums are recorded in
`vln_mujoco/assets/manifest.json`.

## Attribution and licensing

Developed and maintained by [Light Origins](https://www.lightorigins.com/en).
Copyright 2026 Light Origins.

The project source code is licensed under the
[Apache License 2.0](../LICENSE). The bundled MolmoSpaces ProcTHOR `val_2` scene
and THOR assets are licensed under CC BY 4.0; attribution, modification notes,
and license links for dependencies and assets are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

TurtleBot is a trademark of Open Robotics (Open Source Robotics Foundation).
The robot in this demo is an original simplified differential-drive geometry
created for this project, inspired by the TurtleBot form factor; it uses no
official meshes or design files, and this project is not affiliated with or
endorsed by Open Robotics or ROBOTIS.

## Known limits

- The TurtleBot is a lightweight geometry drawn for this project, not the
  official high-fidelity TurtleBot3 mesh;
- The spawn pose is fixed at `(x=6.5, y=13.8, yaw=0)` and environment objects
  are frozen;
- RGB only for now, at a camera resolution of `480 × 270`;
- The MPC only tracks the local waypoints provided by VLN — no navigation map,
  global planning, collision response, or ROS interface.
