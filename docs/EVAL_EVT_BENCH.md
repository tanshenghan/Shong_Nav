# Evaluating on EVT-Bench (embodied visual tracking)

[EVT-Bench](https://github.com/wsakobe/TrackVLA) (from the TrackVLA paper) scores a
robot that has to follow a target person through HM3D / MP3D scenes populated with
distractor humans.  Its driver (`run.py`) runs Habitat in a Python 3.9 conda env,
while `lightnav` needs Python 3.11, so the two sides talk over the WebSocket
protocol in [PROTOCOL.md](PROTOCOL.md):

```
EVT-Bench run.py (py3.9, habitat)  --ws://localhost:PORT-->  lightnav-serve (py3.11, GPU)
   trackvla_client_agent.py                                     tracking checkpoint
   jaw RGB frame -> base64 JPEG                                 waypoints [fwd, lat, yaw] x H
   first waypoint -> agent_1_base_velocity
```

Everything EVT-Bench-side (habitat-lab fork, task configs, datasets,
`analyze_results.py`) stays in your EVT-Bench checkout.  This repository only ships
the client agent, a patch for `run.py`, a config patcher and an orchestration
script (see [`evt_bench/`](../evt_bench/README.md)).

## 1. Install EVT-Bench

Follow the upstream README; in short:

```bash
conda create -n evt_bench python=3.9 cmake=3.14.0
conda activate evt_bench
conda install habitat-sim==0.3.1 withbullet -c conda-forge -c aihabitat
git clone https://github.com/wsakobe/TrackVLA ~/EVT-Bench
cd ~/EVT-Bench
pip install -e habitat-lab
pip install websocket-client          # needed by trackvla_client_agent.py
```

Data, all under `~/EVT-Bench/data/`:

- `scene_datasets/hm3d/{train,val,minival}/...` (HM3D, including the `*.basis.glb`
  files and `hm3d_annotated_basis.scene_dataset_config.json`) and
  `scene_datasets/mp3d/<scene>/<scene>.glb` (MP3D); the val episodes reference
  101 scenes across both.
- humanoid avatars: `python download_humanoid_data.py` (falls back to the Google
  Drive link in the upstream README) -> `data/humanoids/humanoid_data/...`.
- the tracking episodes ship with the repo: `data/datasets/track/{DT,STT,AT}/val/val.json.gz`
  (1,405 episodes per variant; DT = distractor tracking, STT = single-target
  tracking, AT = ambiguous tracking).

Headless rendering needs working EGL / GL libraries and one GPU per Habitat
process; each process also uses a few GB of RAM.

## 2. Add the client to EVT-Bench

```bash
cp /path/to/LightNav-0/evt_bench/trackvla_client_agent.py ~/EVT-Bench/
cd ~/EVT-Bench && git apply /path/to/LightNav-0/evt_bench/run_py.patch
```

The patch adds an `elif model_name == 'trackvla':` branch to `run.py` that imports
`evaluate_agent` from the copied file and passes `--model-path` through as the
server URL.  If `git apply` complains because upstream `run.py` moved, add the branch
by hand: it is the `baseline` branch with `from trackvla_client_agent import
evaluate_agent` and `evaluate_agent(config, dataset_split, save_path,
server_url=model_path)`.

What the client does per step: reads
`observations["agent_1_articulated_agent_jaw_rgb"][:, :, :3]` (Spot jaw camera,
480x270), JPEG-encodes it, sends `{"action": "next", "data": {"seq", "image",
"instruction"}}`, and maps the first waypoint of the reply to the base-velocity action
`[clip(fwd / 0.375), clip(lat / 0.25), clip(yaw / (pi/20))]` in `[-1, 1]^3`
(+lateral = left, +yaw = counter-clockwise).  On a non-zero `rc` or a missing
`actions` field it repeats its previous action.  It never special-cases `stop`: a
zero trajectory simply becomes zero velocity, and the episode ends through
EVT-Bench's own rules (300 steps, "too far for >20 steps" -> `Lost`, collision ->
`Collision`).  Timeouts: `TRACKVLA_WS_TIMEOUT` (seconds, default 60) is the receive
timeout per step.

The evaluation loop itself mirrors EVT-Bench's own drivers step for step, including
their quirks: the multi-agent action tuple steps `agent_0` (target), `agent_1`
(robot) and the distractors `agent_2..agent_5` only (the task configs define
`agent_2..agent_8`; the extra distractors stand still, exactly as upstream), and the
per-episode result JSON and its `success` rule are unchanged. `EVT_NUM_DISTRACTORS=6`
additionally steps `agent_6` and `agent_7` (two more moving distractors, a stricter
setting); numbers obtained with different values are not comparable, so always state
which one you used.

## 3. Run

Start servers, wait until they are ready, run all shards, aggregate:

```bash
cd /path/to/LightNav-0
# the checkpoint ships its decoder; set ACTION_TOKENIZER_BUNDLE=/path/to/rvq_bundle only to override it
MODEL_PATH=/path/to/checkpoint \
NUM_GPUS=2 SERVERS_PER_GPU=4 \
EVT_BENCH_REPO=$HOME/EVT-Bench EVT_CONDA_ENV=evt_bench \
TASK_VARIANTS="dt" \
bash scripts/eval_evt_bench.sh
```

The script calls [`scripts/start_servers.sh`](../scripts/start_servers.sh) (one
`lightnav-serve --task tracking` process per port, `NUM_GPUS x SERVERS_PER_GPU` of
them, `gpu_memory_utilization = 0.85 / SERVERS_PER_GPU` for the vLLM backend), polls
the per-port `.ready` files, then runs `run.py` for shard `0..CHUNKS-1` in waves of
`NUM_GPUS x SERVERS_PER_GPU x CLIENTS_PER_SERVER` processes.  Shard `i` talks to
server `i % TOTAL_SERVERS` and renders on that server's GPU
(`CUDA_VISIBLE_DEVICES`).  Servers are killed when the script exits.

All knobs are environment variables; the header of `scripts/eval_evt_bench.sh`
lists them.  The important ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_PATH` | required | checkpoint directory |
| `ACTION_TOKENIZER_BUNDLE` | optional | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `NUM_GPUS`, `SERVERS_PER_GPU`, `BASE_PORT` | 1, 4, 8050 | server topology; reduce `SERVERS_PER_GPU` on small GPUs |
| `CLIENTS_PER_SERVER` | 1 | Habitat processes per server (the server micro-batches concurrent sessions) |
| `EVT_BENCH_REPO` | `$HOME/EVT-Bench` | your checkout |
| `EVT_CONDA_ENV` or `EVT_PYTHON` | `evt_bench` | conda env name, or an explicit interpreter |
| `TASK_VARIANTS` | `dt` | any of `dt stt at`, run sequentially against the same servers |
| `CHUNKS` | 30 | dataset shards (`--split-num`); keep 30, see below |
| `EVT_JAW_HFOV`, `EVT_JAW_HEIGHT` | empty | jaw camera override, see below |
| `OUTPUT_ROOT` | `$EVT_BENCH_REPO/exp_results/lightnav_<timestamp>` | results; per-variant subdir |
| `READY_TIMEOUT_S` | 1800 | how long to wait for the servers (cold vLLM start can take minutes) |

Run one shard by hand (servers already running):

```bash
cd ~/EVT-Bench && conda activate evt_bench
PYTHONPATH=habitat-lab python run.py --run-type eval --model-name trackvla \
  --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml \
  --split-num 30 --split-id 0 --model-path ws://localhost:8050 \
  --save-path exp_results/manual/dt
```

### Why `CHUNKS=30`

`run.py` shards the 1,405 episodes with `dataset.get_splits(split_num)[split_id]`.
EVT-Bench's own driver reads the language instruction from the **first episode of a
shard** and reuses it for every episode of that shard, and its
`MainHumanoidDetectorSensor` freezes the target's semantic id on the first episode the
same way.  Later episodes in a shard are therefore prompted and scored with the first
episode's target.  Changing the shard count changes which episodes share a prompt and
hence the numbers; EVT-Bench's published results use 30 shards (~47 episodes each),
so the script defaults to 30 and warns otherwise.  Parallelism (`NUM_GPUS`,
`SERVERS_PER_GPU`, `CLIENTS_PER_SERVER`) only changes wall-clock time, not results.

### Jaw camera field of view (`EVT_JAW_HFOV`)

EVT-Bench mounts the Spot jaw RGB camera at hfov 86 (480x270) and the panoptic
camera the detector scores visibility on at hfov 90.  Our tracking checkpoints were
trained on frames rendered at ~115 degrees median horizontal FOV, so 86 is outside
their training distribution and biases the comparison; we evaluate our checkpoints
with `EVT_JAW_HFOV=120` and leave the mount height alone.  Leave the variable empty
(or pass `EVT_JAW_HFOV=86`) to evaluate exactly as upstream.

The override cannot go through `run.py`'s Hydra `opts` (the `trackvla` branch does
not forward them), so `evt_bench/patch_task_config.py` writes a patched copy of the
task yaml into `OUTPUT_ROOT` and the script passes its absolute path as
`--exp-config`.  `jaw_rgb_sensor` and `jaw_panoptic_sensor` are always patched
together so the model and the visibility detector look through the same lens; `hfov`
is written as an integer (habitat rejects floats) and the `# @package _global_` first
line is preserved.  Manual use:

```bash
python evt_bench/patch_task_config.py \
  ~/EVT-Bench/habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml \
  /abs/path/track_infer_dt_hfov120.yaml --hfov 120           # [--height 0.7]
```

Another opt-in that is **not** upstream behaviour: `EVT_HIDE_ROBOT_MESH=1` makes the
client zero-scale the robot's visual mesh so the jaw camera does not see the robot
body.  It is off by default.

## 4. Results

Per episode the driver writes `<save_path>/<scene>/<episode_id>.json`:

```json
{"finish": true, "status": "Normal", "success": 1.0, "following_rate": 0.93,
 "following_step": 279, "total_step": 300, "collision": 0.0}
```

- `status`: `Normal`, `Lost` (farther than 4 m from the target for more than 20
  consecutive steps) or `Collision` (came within 0.5 m of a human).
- `success`: `human_following_success and human_following` when the episode ended
  early, `human_following` at the 300-step limit.  `human_following` = within 3 m and
  the target is visible (3,000 < target pixels < 30 % of the panoptic image);
  `human_following_success` additionally requires 1 m <= distance and the stop action.
- `<episode_id>_info.json` holds a per-step trace (`step`, `trajectory`,
  `dis_to_human`, `facing`).

`analyze_results.py` (upstream, run from the EVT-Bench root) aggregates a results
directory:

```bash
cd ~/EVT-Bench && python analyze_results.py --path exp_results/lightnav_<ts>/dt [--n N]
# {'episode count:': 1405, 'success rate': SR, 'following rate:': FR, 'collision rate:': CR}
```

- **SR** (success rate) = fraction of episodes with `success` truthy.
- **FR / TR** (following / tracking rate) = `sum(following_step) / sum(total_step)`,
  where `total_step` is raised to the reference length shipped in
  `track_episode_step/<scene>/<episode>.json` when the episode ended early.
- **CR** (collision rate) = mean of `collision`.

It must run with the EVT-Bench root as the working directory (it reads
`track_episode_step/` relatively and writes `following_info.json` into the cwd), and
it raises `ZeroDivisionError` before the first episode has finished.  `--n` limits the
count to the first N result files by modification time, which is handy for a quick
look while an evaluation is still running (`watch -n 60 python analyze_results.py ...`).
The orchestration script saves the final line per variant to
`OUTPUT_ROOT/metrics_<variant>.txt`.

## 5. Troubleshooting

- `ModuleNotFoundError: magnum` / `websocket`: the driver is not running in the
  `evt_bench` env or `websocket-client` is missing.  The script checks both up front.
- Chunk logs (`OUTPUT_ROOT/logs/eval_<variant>_chunk_<i>.log`) show one line per step
  with the server latency, the raw model text and the decoded `stop` / `visible`
  flags.  A `Server error: {...}` line means the server answered `rc != 0` and the
  client repeated its previous action.
- A receive timeout inside `act()` is not retried and ends that shard.  The server
  binds its port only after a warm-up inference, so the first step should not time
  out; raise `TRACKVLA_WS_TIMEOUT` if the `hf` backend is slow on your GPU.
- `ValueError ... could not be converted to Integer` when starting a shard: a
  non-integer `hfov` reached Habitat; use `patch_task_config.py` rather than editing
  the yaml by hand.

## Licence

EVT-Bench / TrackVLA is released under CC BY-NC-SA 4.0 (non-commercial).  None of it is
redistributed here.  The evaluation loop in `evt_bench/trackvla_client_agent.py` is
adapted from the EVT-Bench driver and is therefore covered by that licence, not by
this repository's licence; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
